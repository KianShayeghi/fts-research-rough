"""Endpoint estimation for truncated FLR as in Hall & Hooker, 2016.

Response depends only on the curve on an informative window [u,v]:

    Y_i = ∫_u^v b(t) X_i(t) dt + ε_i, ε_i ~ N(0, σ²)

This module implements 'Method A: simultaneous inference' (3.1.1) and 'Algorithms
for selecting tuning parameters' for the corresponding method (3.2), adapted to the
data-generating process in `HH_dgp.py`. Its companion `HH.py` implements method B.

In method A, unlike method B, the coefficient function is *not* frozen at a full-interval
pilot estimate. Instead, for every candidate endpoint theta the pair (a, b) is re-estimated
by least squares in the basis of empirical FPCs of the *truncated* curves X_i(t) I(t < theta)
(the paper's psi_j = phi_j^theta), minimising

    S(a, beta_1..beta_m, theta | m, lambda) = sum_i (Y_i - a - ∫_0^theta sum_j beta_j psi_j X_i)²
                                              + n lambda theta²                             (3.4)

This "separate principal components analysis for each theta" is the significant extra
computational cost the paper notes for method A relative to method B.

In the paper, they estimate the right endpoint of the window. We are
interested in estimating the left endpoint, so we must mirror their
setup, i.e.

    - Reflect X(t) -> X(1 - t)
    - Run method A from paper
    - Map endpoint back, u_hat = 1 - theta_hat

The number of retained components m is selected by BIC on the final fit (the paper's step 7):
`fit`/`fit_mirror` refit at each candidate m in `m_range` and keep the m minimising

    BIC(m) = n log{ (1/n) sum_i (Y_i - a_hat - ∫_0^theta b_hat X_i)² } + (m + 1) log(n).

The quantities method A shares with method B -- the pilot, b_simp, the truncated-domain
components, and the step 6-7 machinery -- live in `_common.py`.
"""

import numpy as np

from ._common import (
    WEIGHT_FUNCTIONS, DegenerateSelection, check_mirrorable, fat_tailed_weight,
    fit_b_simp, floor_eigvals, normal_weight, pilot_fpca, theta_lam_variance,
    thin_tailed_weight, trunc_eigvals, trunc_fpca, uniform_weight, weighted_S_b,
)

__all__ = [
    "fit", "fit_mirror", "lambda_select", "theta_methodA", "theta_methodA_multi",
    "ab_methodA", "methodA_bic", "trunc_predict", "truncation_profile",
    "pilot_fpca", "fit_b_simp",
    "trunc_fpca", "trunc_eigvals", "check_mirrorable", "DegenerateSelection",
    "WEIGHT_FUNCTIONS",
    "normal_weight", "uniform_weight", "fat_tailed_weight", "thin_tailed_weight",
]


def _final_fit_select_m(X, Y, grid, lam_hat, m_range):
    """For a fixed lam_hat, run the final fit (theta scan -> `ab_methodA`) at each m in `m_range`
    and keep the m minimising the step-7 BIC.

    The theta scan is shared across all m by `theta_methodA_multi`, so only the coefficient
    refit at the winning theta is repeated per m.

    Returns a dict with the winning m, theta_hat, a_hat, b_hat, and the per-m BIC map.
    """
    m_values = list(m_range)
    scans = theta_methodA_multi(X, Y, grid, lam_hat, m_values)
    bics, best = {}, None
    for m in m_values:
        theta_hat = scans[m][0]
        a_hat, b_hat = ab_methodA(theta_hat, grid, X, Y, n_components=m)
        bic = methodA_bic(a_hat, b_hat, X, Y, grid, m, theta_hat)
        bics[m] = bic
        if best is None or bic < best["bic"]:
            best = {"m": m, "bic": bic, "theta_hat": theta_hat,
                    "a_hat": a_hat, "b_hat": b_hat}
    best["bics"] = bics
    return best


def trunc_predict(a_hat, b_hat, X, grid, theta):
    """Fitted values a_hat + ∫_0^theta b_hat X_i of a truncated model.

    The integral is taken over the sub-grid [0, theta], the same quadrature the fit itself
    used, so these are the fitted values of that least-squares fit and their residuals stay
    orthogonal to its design. Integrating over the whole grid with b_hat set to zero past
    theta is not equivalent: the trapezium rule straddling the jump at theta adds a spurious
    half panel, b_hat(theta) X_i(theta) h / 2.

    Returns (n,) predictions.

    PARAMETERS
    b_hat : (N,) coefficient on the full grid, zero past theta;
    theta : float, the truncation point b_hat was fitted at.
    """
    interval = np.asarray(grid) <= theta
    return a_hat + np.trapezoid(b_hat[interval] * X[:, interval], grid[interval], axis=1)


def methodA_bic(a_hat, b_hat, X, Y, grid, m, theta):
    """Step-7 BIC of a fitted method-A model (paper section 3.2):

        BIC(m) = n log{ (1/n) sum_i (Y_i - a_hat - ∫_0^theta b_hat X_i)² } + (m + 1) log(n).

    Selects the number of retained components m for the final fit. Predictions come from
    `trunc_predict`, so the residuals scored here are the residuals of the fit at theta.

    PARAMETERS
    m     : int, number of retained components being charged for;
    theta : float, the truncation point (a_hat, b_hat) were fitted at.
    """
    n = len(Y)
    rss = np.sum((Y - trunc_predict(a_hat, b_hat, X, grid, theta)) ** 2)
    return n * np.log(rss / n) + (m + 1) * np.log(n)


def _core(X, Y, grid, lambdas, theta_bar, n_basis, n_components, m_range,
          n_fourier, m_tuning, w, strict=False):
    """Shared method-A pipeline, in the paper's orientation. Returns theta_hat, a_hat, b_hat,
    lam_hat, m_hat and diagnostics.

    m enters both the lambda criterion (S_Y^A carries a sigma²(m + 1) term, and S_b^A sums
    (tau_j^theta)^-1 to m) and the final fit, so step 7 admits two readings:

    - `strict=False` (default): run `lambda_select` once, then BIC-select m over the final fit
      alone;
    - `strict=True`: re-run `lambda_select` for each candidate m, so each m carries its own
      lambda_hat_m, then BIC-select over the pairs (m, lambda_hat_m). Costs len(m_range)
      lambda selections, and ignores `m_tuning`, the criterion's m being the loop's m here.

    `m_range` defaults (when None) to range(2, n_components + 1).
    """
    if m_range is None:
        m_range = range(2, n_components + 1)
    m_range = [int(m) for m in m_range]
    if not m_range:
        raise ValueError("m_range is empty; give at least one candidate component count.")
    if min(m_range) < 1:
        raise ValueError(f"m_range must contain positive component counts, got {m_range}.")

    pilot_budget = max(m_range + [n_components])
    a, b, fpc = pilot_fpca(X, Y, grid, n_basis=n_basis, n_components=pilot_budget)

    if not strict:
        # One lambda selection, then BIC over the final-fit m (shared theta scan).
        lam_hat, diag = lambda_select(
            X, Y, grid, a, b, fpc, theta_bar, lambdas,
            n_components=pilot_budget, n_fourier=n_fourier, m_tuning=m_tuning, w=w
        )
        best = _final_fit_select_m(X, Y, grid, lam_hat, m_range)
        diag["bics"] = best["bics"]
        diag["lam_hats"] = {m: lam_hat for m in m_range}
        return (best["theta_hat"], best["a_hat"], best["b_hat"], lam_hat, best["m"], diag)

    # Strict: re-run the whole process (lambda_select included) for each m.
    bics, lam_hats, best, best_diag = {}, {}, None, None
    for m in m_range:
        # Surrogate component count tied to this m (literal "repeat for each m"); the pilot -- a
        # nuisance supplying sigma²/b_simp -- keeps its stable budget, so only the parts of
        # the surrogate that depend on m (the (m+1) variance term, the sum to m, the
        # m-component refit) change.
        lam_m, diag_m = lambda_select(
            X, Y, grid, a, b, fpc, theta_bar, lambdas,
            n_components=m, n_fourier=n_fourier, m_tuning=None, w=w
        )
        theta_m, _, _ = theta_methodA(X, Y, grid, lam_m, n_components=m)
        a_m, b_m = ab_methodA(theta_m, grid, X, Y, n_components=m)
        bic_m = methodA_bic(a_m, b_m, X, Y, grid, m, theta_m)
        bics[m] = bic_m
        lam_hats[m] = lam_m
        if best is None or bic_m < best["bic"]:
            best = {"m": m, "bic": bic_m, "theta_hat": theta_m, "a_hat": a_m,
                    "b_hat": b_m, "lam_hat": lam_m}
            best_diag = diag_m
    best_diag["bics"] = bics
    best_diag["lam_hats"] = lam_hats
    return (best["theta_hat"], best["a_hat"], best["b_hat"], best["lam_hat"],
            best["m"], best_diag)


def fit(X, Y, grid, lambdas, theta_bar=None, n_basis=20, n_components=10, m_range=None,
        n_fourier=1, m_tuning=None, w=None, strict=False):
    """Performs method A as in the paper: pilot, select lambda, then estimate the endpoint and
    coefficient, choosing the number of retained components m by BIC (step 7).

    The paper's window is [0, theta], so this estimates the *right* endpoint theta directly
    and returns it as "theta_hat". Use `fit_mirror` when the informative window is [u, 1]
    and it is the *left* endpoint that is unknown.

    Returns parameters (dict). "m_hat" is the selected component count; the diagnostics carry
    the per-m BIC values under "bics" and the per-m selected lambda under "lam_hats".

    PARAMETERS
    theta_bar : float or None, forces the step-3 b_simp's cutoff. If None (default), theta_bar
                is instead estimated jointly with b_simp's coefficients by unpenalised least
                squares;
    n_components : int, max FPCs; sets the default BIC search range (2..n_components) and the
                pilot / lambda_select budget;
    m_range   : iterable of int or None, candidate component counts for the step-7 BIC
                selection of the *final* fit; defaults to range(2, n_components + 1);
    m_tuning  : int, caps the number of FPCs used only inside lambda_select's S_Y/S_b/P_b
                criterion (see lambda_select), independently of the final fit's m;
    strict    : bool, chooses between the two readings of step 7 (see `_core`). False (default)
                selects lambda once; True re-runs lambda_select for each m and ignores m_tuning;
    w         : callable, passed through to `lambda_select`; defaults to `normal_weight`.
    """
    theta_hat, a_hat, b_hat, lam_hat, m_hat, diag = _core(
        X, Y, grid, lambdas, theta_bar, n_basis, n_components, m_range, n_fourier, m_tuning, w,
        strict=strict,
    )
    return {"a_hat": a_hat, "b_hat": b_hat, "lam_hat": lam_hat,
            "theta_hat": theta_hat, "m_hat": m_hat, "diag": diag}


def fit_mirror(X, Y, grid, lambdas, u_bar=None, n_basis=20, n_components=10, m_range=None,
               n_fourier=1, m_tuning=None, w=None, strict=False):
    """Performs method A for left-endpoint mirror, using reflection of curves. Selects m by BIC
    over `m_range` (step 7).

    The reflection is X(t) -> X(1 - t), applied by reversing the columns of X, so `grid` must
    be symmetric about its midpoint.

    Returns parameters (dict).

    PARAMETERS
    u_bar    : float or None, forces the step-3 b_simp's cutoff at 1 - u_bar
               (mirrored theta_bar). If None (default), theta_bar is instead estimated
               jointly with b_simp's coefficients by unpenalised least squares;
    n_components : int, max FPCs; sets the default BIC search range (2..n_components) and the
               pilot / lambda_select budget;
    m_range  : iterable of int or None, candidate component counts for the step-7 BIC
               selection of the final fit; defaults to range(2, n_components + 1);
    m_tuning : int, caps the number of FPCs used only inside lambda_select's S_Y/S_b/P_b
               criterion (see lambda_select), independently of the final fit's m;
    strict   : bool, see `fit` and `_core`; defaults to False;
    w        : callable, passed through to `lambda_select`; defaults to `normal_weight`.
    """
    check_mirrorable(grid)
    X_r = X[:, ::-1]
    theta_bar = None if u_bar is None else 1.0 - u_bar

    theta_hat, a_hat, b_hat_r, lam_hat, m_hat, diag = _core(
        X_r, Y, grid, lambdas, theta_bar, n_basis, n_components, m_range, n_fourier, m_tuning, w,
        strict=strict,
    )

    b_hat = b_hat_r[::-1]
    diag["u_lam"] = 1.0 - diag["theta_lams"]
    diag["u_bar"] = 1.0 - diag["theta_bar"]
    return {"u_hat": 1.0 - theta_hat, "a_hat": a_hat, "b_hat": b_hat,
            "lam_hat": lam_hat, "theta_hat": theta_hat, "m_hat": m_hat,
            "diag": diag}


def _trunc_scores(phi_theta, X, grid_i, interval):
    """Trapezoid FPC scores ∫_0^theta phi_j^theta X_i of each curve onto the truncated FPCs.

    Returns (n, k) array.
    """
    return np.trapezoid(
        phi_theta[:, None, :] * X[None, :, interval], grid_i, axis=2
    ).T


def _refit_at_theta(X, Y, grid, interval, m):
    """Least-squares fit of (a, b) on [0, theta] in the basis of the m truncated FPCs -- the
    inner solve of method A's S1 (3.3) at a single theta.

    Regresses Y on [1 | truncated FPC scores] and reconstructs b on the sub-grid from the slope
    coefficients. Returns a (float), b_sub (k,), phi_theta (k, N_i), tau_theta (k,) and the
    fitted (n,) values.
    """
    grid_i = grid[interval]
    phi_theta, tau_theta = trunc_fpca(X, grid, interval, m)
    scores = _trunc_scores(phi_theta, X, grid_i, interval)  # (n, k)
    design = np.column_stack([np.ones(len(Y)), scores])
    coef, *_ = np.linalg.lstsq(design, Y, rcond=None)
    a = coef[0]
    b_sub = coef[1:] @ phi_theta
    pred = design @ coef
    return a, b_sub, phi_theta, tau_theta, pred


def theta_methodA(X, Y, grid, lam, n_components=10, theta_grid=None):
    """Estimate the endpoint via method A: minimise the penalised objective
    S(a, b, theta | m, lambda) of (3.4), re-estimating (a, b) by a fresh truncated FPCA at
    every candidate theta.

    This takes the *global* minimum. Method A's step 6 is theta_lambda = argmin_theta
    {S_Y^A + lambda theta²}, with no analogue of method B's "first minimum" rule -- that rule
    arises for B only because its b̌ is a fixed full-interval estimate whose S_Y^B drops
    sharply near the right end of the domain.

    Returns theta_hat (float), theta candidates (array), and the objective S(theta) (array).

    PARAMETERS
    lam          : float, penalty parameter (choose via `lambda_select`);
    n_components : int, cap on the number of truncated FPCs used in each per-theta refit;
    theta_grid   : (N,) array, endpoints scanned over, defaults to grid's interior points.
    """
    n = len(Y)
    if theta_grid is None:
        theta_grid = grid[(grid > grid[0]) & (grid < grid[-1])]
    S = np.empty(len(theta_grid))
    for q, theta in enumerate(theta_grid):
        interval = grid <= theta
        # Cap components at what the sub-grid and sample can support (mirrors ab_methodB's guard).
        m = max(1, min(n_components, int(interval.sum()) - 1, n - 1))
        _, _, _, _, pred = _refit_at_theta(X, Y, grid, interval, m)
        S[q] = np.sum((Y - pred) ** 2) + n * lam * theta ** 2
    idx = int(np.argmin(S))
    return theta_grid[idx], theta_grid, S


def theta_methodA_multi(X, Y, grid, lam, m_values, theta_grid=None):
    """Multi-m version of `theta_methodA`: for a single lambda, scan theta once and return the
    penalised objective S_m(theta) for every m in `m_values`.

    The truncated FPCA at a given theta does not depend on m -- a larger m keeps more leading
    components of the same decomposition -- so one decomposition per theta serves every m. At
    each theta the design [1 | score_1 .. score_M], M = max(m_values), is factorised once by QR;
    regressing on orthonormal columns makes each m's residual sum of squares a partial sum over
    the leading (m + 1) of them.

    Returns a dict {m: (theta_hat_m, S_m_array)} matching `theta_methodA` for each m.
    """
    n = len(Y)
    if theta_grid is None:
        theta_grid = grid[(grid > grid[0]) & (grid < grid[-1])]
    M = max(m_values)
    S = {m: np.empty(len(theta_grid)) for m in m_values}
    pen = n * lam * theta_grid ** 2

    for q, theta in enumerate(theta_grid):
        interval = grid <= theta
        grid_i = grid[interval]
        m_cap = max(1, min(M, int(interval.sum()) - 1, n - 1))
        phi_theta, _ = trunc_fpca(X, grid, interval, m_cap)
        scores = _trunc_scores(phi_theta, X, grid_i, interval)  # (n, k), k <= m_cap
        k = scores.shape[1]
        design = np.column_stack([np.ones(n), scores])          # (n, k+1)

        # One thin QR; RSS using the first (m+1) columns is ||Y||² - ||Q_{:, :m+1}^T Y||².
        Q, _ = np.linalg.qr(design)                             # (n, k+1), orthonormal columns
        proj = Q.T @ Y                                          # (k+1,)
        yy = Y @ Y
        cum = np.cumsum(proj ** 2)                              # cum[j] = ||first j+1 cols proj||²
        for m in m_values:
            # Use min(m, k) components (short sub-grids cap k < m); +1 for the intercept column.
            cols = min(m, k) + 1
            rss = yy - cum[cols - 1]
            S[m][q] = rss + pen[q]

    out = {}
    for m in m_values:
        idx = int(np.argmin(S[m]))
        out[m] = (theta_grid[idx], S[m])
    return out


def ab_methodA(theta_hat, grid, X, Y, n_components=10):
    """Computes the final coefficient function and intercept once theta_hat is fixed.

    This is the per-theta refit of (3.3) evaluated at theta_hat: the truncated model
    E(Y|X) = a + ∫_0^theta_hat bX fitted in the truncated FPC basis (paper 3.1.1). Method A has
    no "truncate the pilot" branch as in method B, having never frozen a full-interval estimate.

    Returns a_hat (float) and b_hat (array, zero for t > theta_hat).
    """
    n = len(Y)
    interval = grid <= theta_hat
    m = max(1, min(n_components, int(interval.sum()) - 1, n - 1))
    a_hat, b_sub, _, _, _ = _refit_at_theta(X, Y, grid, interval, m)
    b_hat = np.zeros_like(grid)
    b_hat[interval] = b_sub
    return a_hat, b_hat


def lambda_select(X, Y, grid, a_pilot, b_pilot, fpc, theta_bar, lambdas,
                  sigma2=None, n_components=10, n_fourier=1, theta_grid=None,
                  m_tuning=None, w=normal_weight):
    """Select lambda using the method A branch of section 3.2 of the paper (steps 3-7).

    The scheme mirrors `HH.py`'s method-B `lambda_select`, but S_Y and S_b take their *method A*
    forms (steps 4-5, not the primed 4'-5'):

      - S_Y^A(theta) = sum_i (Ybar*_i - Yhat*_i(theta))² + sigma² (m + 1)
      - S_b^A(theta) = ∫_I (b_simp - b*_theta)²          + (sigma²/n) sum_j (tau_j^theta)^-1

    where at each theta the target (a*_theta, b*_theta) is refit onto the *truncated* FPCs
    phi_j^theta (the paper's psi_j = phi_j^theta). S_b^A carries a 1/n the paper's step-5
    display omits: the refit gives beta*_j variance sigma²/(n tau_j^theta), and the phi_j^theta
    are orthonormal on [0, theta], so that variance passes into ∫(b_simp - bhat*_theta)²
    unchanged. S_Y^A needs no such correction -- sigma²(m + 1) is already the summed
    prediction variance of an (m + 1)-parameter fit. The package README derives both, and
    `tests/test_scaling.py` pins them against the conditional expectations they equal.

    Two further things follow from refitting at each theta:

      1. S_Y's variance term is a constant in theta, where method B's
         sum_j (tau_j^1)^-1 sum_i (∫_0^theta phi_j^1 X_i)² grows with the window. It varies
         only through m being capped on short sub-grids.
      2. Step 6 takes theta_lambda as the *global* argmin of S_Y^A + n lambda theta², where
         method B takes its first interior minimum. The paper's method-A display of V carries
         a leading minus, which pairs with an oppositely oriented curvature: S_Y^A is locally
         convex at the penalised minimum, so s_Y'' > 0 and the plus form of
         `_common.theta_lam_variance` is the one giving V >= 0. Non-positive curvature is
         degenerate and returns NaN, as method B does.

    Returns lambda (float) and diagnostics (dict).

    PARAMETERS
    theta_bar : float or None, the cutoff b_simp (the step-3 parametric estimate) decays to
                zero at. If None, it is estimated jointly with b_simp's coefficients by
                unpenalised least squares;
    lambdas  : array, candidates for optimal lambda;
    sigma2   : float, pilot residual variance; estimated from the noisy pilot fit if None;
    n_components : int, cap on the truncated FPCs (the paper's m) used to refit (a*, b*) at each
                theta and to size the S_Y/S_b variance terms;
    m_tuning : int, overrules `n_components` inside this criterion only (see `HH.py`);
    w        : callable, the weighting density used in step 7 to average S_b(theta) around
               theta_lam.
    """
    if w is None:
        w = normal_weight
    if theta_grid is None:
        theta_grid = grid[(grid > grid[0]) & (grid < grid[-1])]

    n = len(Y)
    m_cap = n_components if m_tuning is None else m_tuning

    # If pilot residual variance is not supplied, we estimate it from the real (noisy) fit
    if sigma2 is None:
        Y_pilot = a_pilot + np.trapezoid(b_pilot[None, :] * X, grid, axis=1)
        sigma2 = np.mean((Y - Y_pilot) ** 2)

    # Step 3 (paper, section 3.2): b_simp is fit jointly with its own cutoff theta_bar by
    # minimising unpenalised squared error against (X, Y). Common to methods A and B.
    b_simp, theta_bar = fit_b_simp(X, Y, grid, a_pilot, n_fourier=n_fourier, theta_bar=theta_bar)

    # Noiseless parametric target Ybar_star = a_pilot + integral(b_simp * X) (paper's Y¯*_i).
    Ybar_star = a_pilot + np.trapezoid(b_simp[None, :] * X, grid, axis=1)

    bias_Y = np.empty(len(theta_grid))
    var_Y = np.empty(len(theta_grid))
    bias_b = np.empty(len(theta_grid))
    var_b = np.empty(len(theta_grid))
    ms = np.empty(len(theta_grid), dtype=int)
    for q, theta in enumerate(theta_grid):
        interval = grid <= theta
        grid_i = grid[interval]
        m = max(1, min(m_cap, int(interval.sum()) - 1, n - 1))
        ms[q] = m

        # Step 4 (method A): refit (a*_theta, b*_theta) onto the *truncated* FPCs phi_j^theta,
        # predicting the noiseless target Ybar_star. phi_theta / tau_theta come from one SVD.
        phi_theta, tau_theta = trunc_fpca(X, grid, interval, m)
        scores = _trunc_scores(phi_theta, X, grid_i, interval)  # (n, k)
        design = np.column_stack([np.ones(n), scores])
        coef, *_ = np.linalg.lstsq(design, Ybar_star, rcond=None)
        b_star_sub = coef[1:] @ phi_theta               # b*_theta on the sub-grid
        Y_hat_star = design @ coef                       # y-hat*_i(theta)

        # S_Y^A: squared reconstruction error + sigma²(m+1) prediction-variance of the (m+1)-param
        # fit. (design.shape[1] = k + 1 = actual parameter count, which equals m+1 unless the
        # sub-grid could not support m components, in which case it correctly counts fewer.)
        bias_Y[q] = np.sum((Ybar_star - Y_hat_star) ** 2)
        var_Y[q] = sigma2 * design.shape[1]

        # S_b^A: ∫(b_simp - b*_theta)² over the *full* interval (b*_theta is zero past theta)
        # + (sigma²/n) sum_j (tau_j^theta)^-1, the refit coefficients' variance carried onto
        # [0, theta], where the phi_j^theta are orthonormal.
        b_star_full = np.zeros_like(grid)
        b_star_full[interval] = b_star_sub
        bias_b[q] = np.trapezoid((b_simp - b_star_full) ** 2, grid)
        # tau_theta is the decomposition already computed above, so no second SVD is needed.
        var_b[q] = sigma2 / n * np.sum(1.0 / floor_eigvals(tau_theta, m))

    S_Y = bias_Y + var_Y
    S_b = bias_b + var_b

    pen = theta_grid ** 2
    P_b = np.empty(len(lambdas))
    theta_lams = np.empty(len(lambdas))
    Vs = np.empty(len(lambdas))
    for jx, lam in enumerate(lambdas):
        obj = S_Y + n * lam * pen
        # Step 6 (method A): global argmin, not method B's first interior minimum.
        idx = int(np.argmin(obj))
        theta_lams[jx] = theta_grid[idx]

        V = theta_lam_variance(S_Y, theta_grid, idx, lam, sigma2, n)
        Vs[jx] = V
        if not np.isfinite(V):
            P_b[jx] = np.nan
            continue
        # Step 7: average S_b^A around theta_lam under the weight density.
        P_b[jx] = weighted_S_b(S_b, theta_grid, theta_grid[idx], V, w)

    if np.all(np.isnan(P_b)):
        raise DegenerateSelection(
            "step 7 is undefined at every candidate lambda: theta_lambda lands where the "
            "curvature of S_Y is non-positive, so V(lambda) does not exist. S_Y has no "
            "interior minimum on this theta grid, which no choice of lambda can create."
        )
    lam_hat = lambdas[int(np.nanargmin(P_b))]
    return lam_hat, {
        "lambdas": np.asarray(lambdas), "theta_lams": theta_lams, "Vs": Vs, "P_b": P_b,
        "S_Y": S_Y, "S_b": S_b, "bias_Y": bias_Y, "var_Y": var_Y, "bias_b": bias_b,
        "var_b": var_b, "theta_grid": theta_grid, "b_simp": b_simp, "ms": ms,
        "sigma2": sigma2, "theta_bar": theta_bar,
    }


def truncation_profile(X, Y, grid, thetas=None, folds=None, n_folds=10, n_components=10,
                       mirror=False, rng=0):
    """Held-out prediction error as a function of where the curves are truncated.

    Section 3.2 selects `theta` through a penalised criterion whose `lambda` is itself chosen
    by the step 6-7 machinery, and on real data there is no true endpoint to check that
    against. This is the model-free alternative: fit method A's truncated model at each
    candidate `theta` and score it by cross-validated prediction error, so the question
    "where does the informative region end" is answered by what the retained region can
    actually predict.

    Read the profile at its elbow, not at its minimum. Past the true endpoint the retained
    curve carries no signal, so extending it costs only estimation variance and the error
    flattens rather than rising: on simulated data with a true `v` of 0.5 the minimum can sit
    anywhere in [0.5, 1] while the error falls sharply below 0.5 and is level above it. The
    minimiser is therefore not a consistent endpoint estimate, and `theta_1se` -- the
    smallest candidate scoring within one standard error of the best -- is the one to read:
    it asks where the error *stops falling*, which is the question the endpoint answers.
    This is the same distinction between prediction and support recovery that decides which
    lambda the group lasso in `fdaSP.SBC` needs, arriving from the other direction.

    A profile that falls all the way to the right-hand end is the data saying there is
    nothing to truncate.

    Returns a dict with thetas, mse (one per candidate, averaged over folds), se (its
    standard error across folds), theta_hat (the minimiser), theta_1se (the elbow),
    mse_full (the score with no truncation) and folds.

    PARAMETERS
    X            : (n, N) array of curves;
    Y            : (n,) array of responses;
    grid         : (N,) array, the domain;
    thetas       : (K,) array of candidate endpoints, or None for 20 evenly spaced interior
                   points of `grid`;
    folds        : (n,) array of fold labels, or None to assign them at random. Pass explicit
                   folds wherever the rows are not exchangeable;
    n_folds      : int, folds to assign when `folds` is None;
    n_components : int, cap on the truncated FPCs used in each per-theta refit;
    mirror       : bool, profile the window [u, 1] instead of [0, theta], by reflecting the
                   curves as `fit_mirror` does. The returned `thetas` are then values of `u`
                   in the original coordinate system, ascending;
    rng          : seed or Generator used only when `folds` is None.
    """
    X = np.atleast_2d(np.asarray(X, float))
    Y = np.asarray(Y, float).ravel()
    grid = np.asarray(grid, float)
    n = Y.size

    if mirror:
        check_mirrorable(grid)
        X = X[:, ::-1]

    if thetas is None:
        interior = grid[(grid > grid[0]) & (grid < grid[-1])]
        thetas = interior[np.linspace(0, interior.size - 1, 20).round().astype(int)]
    thetas = np.asarray(thetas, float)
    # In the reflected system a left endpoint u becomes the right endpoint 1 - u, so the
    # candidates are mapped over before the scan and mapped back before returning.
    scan = 1.0 - thetas[::-1] if mirror else thetas

    if folds is None:
        folds = np.random.default_rng(rng).permutation(n) % n_folds
    folds = np.asarray(folds)
    labels = np.unique(folds)

    # A candidate retaining fewer than three grid points cannot support a truncated FPCA at
    # all. Such candidates score infinity, so the minimum and the elbow skip them without an
    # empty column reaching the fold average.
    usable = np.array([int((grid <= theta).sum()) >= 3 for theta in scan])
    err = np.empty((labels.size, scan.size))
    err[:] = np.inf

    for f_i, f in enumerate(labels):
        te = folds == f
        tr = ~te
        for q, theta in enumerate(scan):
            if not usable[q]:
                continue
            interval = grid <= theta
            m = max(1, min(n_components, int(interval.sum()) - 1, int(tr.sum()) - 1))
            a, b_sub, _, _, _ = _refit_at_theta(X[tr], Y[tr], grid, interval, m)
            b_full = np.zeros_like(grid)
            b_full[interval] = b_sub
            pred = trunc_predict(a, b_full, X[te], grid, theta)
            err[f_i, q] = np.mean((Y[te] - pred) ** 2)

    mse = np.full(scan.size, np.inf)
    se = np.full(scan.size, np.inf)
    mse[usable] = err[:, usable].mean(axis=0)
    se[usable] = err[:, usable].std(axis=0, ddof=1) / np.sqrt(labels.size)

    # Everything above is indexed by the reflected scan; from here on it is indexed by the
    # caller's own `thetas`, so the mirrored case is flipped back once and for all.
    if mirror:
        mse, se, usable = mse[::-1], se[::-1], usable[::-1]

    best = int(np.argmin(mse))
    # The elbow: the shortest retained window scoring within one standard error of the best.
    # "Shortest" runs from the left for [0, theta] and from the right for [u, 1], so the
    # search starts at the truncating end and walks in towards the minimum.
    within = usable & (mse <= mse[best] + se[best])
    order = range(len(mse) - 1, best - 1, -1) if mirror else range(best + 1)
    elbow = next((q for q in order if within[q]), best)

    return {"thetas": thetas, "mse": mse, "se": se,
            "theta_hat": float(thetas[best]), "theta_1se": float(thetas[elbow]),
            "mse_full": float(mse[0] if mirror else mse[-1]), "folds": folds}
