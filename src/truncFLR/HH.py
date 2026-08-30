"""Endpoint estimation for truncated FLR as in Hall & Hooker, 2016.

Response depends only on the curve on an informative window [u,v]:

    Y_i = ∫_u^v b(t) X_i(t) dt + ε_i, ε_i ~ N(0, σ²)

This module implements 'Method B: iterative inference' (3.1.2) and 'Algorithms
for selecting tuning parameters' for the corresponding method (3.2), adapted to the
data-generating process in `HH_dgp.py`. Its companion `HH_A.py` implements method A.

In the paper, they estimate the right endpoint of the window. We are
interested in estimating the left endpoint, so we must mirror their
setup, i.e.

    - Reflect X(t) -> X(1 - t)
    - Run method B from paper
    - Map endpoint back, u_hat = 1 - theta_hat

The quantities method B shares with method A -- the pilot, b_simp, the truncated-domain
components, and the step 6-7 machinery -- live in `_common.py`.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from ._common import (
    WEIGHT_FUNCTIONS, DegenerateSelection, check_mirrorable, fat_tailed_weight,
    fit_b_simp, normal_weight, pilot_fpca, theta_lam_variance, thin_tailed_weight,
    trunc_eigvals, uniform_weight, weighted_S_b,
)

__all__ = [
    "fit", "fit_mirror", "lambda_select", "theta_methodB", "ab_methodB",
    "pilot_fpca", "fit_b_simp", "trunc_eigvals", "check_mirrorable", "PILOT_ORDER",
    "DegenerateSelection", "WEIGHT_FUNCTIONS",
    "normal_weight", "uniform_weight", "fat_tailed_weight", "thin_tailed_weight",
]

# Order of the B-spline basis `_common.pilot_fpca` represents the curves in, i.e. skfda's
# default: cubic, so four functions is the smallest basis it will build.
PILOT_ORDER = 4


def fit(X, Y, grid, lambdas, theta_bar=None, n_basis=20, n_components=10, n_fourier=1,
        m_tuning=None, refit=True, w=None):
    """Performs method B as in the paper, in the paper's own orientation.

    The paper's window is [0, theta], so this estimates the *right* endpoint theta directly
    and returns it as "theta_hat". Use `fit_mirror` when the informative window is [u, 1]
    and it is the *left* endpoint that is unknown -- it reflects the curves, runs this
    procedure, and maps back with u_hat = 1 - theta_hat.

    Returns parameters (dict)

    PARAMETERS
    theta_bar : float or None, forces the step-3 b_simp's cutoff. If None (default),
               theta_bar is instead estimated jointly with b_simp's coefficients by
               unpenalised least squares;
    m_tuning : int, caps the number of FPCs used only inside lambda_select's S_Y/S_b/P_b
               criterion (see lambda_select);
    refit    : bool, passed through to `ab_methodB`, chooses between re-running the pilot
               method restricted to [0, theta_hat] (refit=True, the default) and truncating
               the untruncated pilot (refit=False);
    w        : callable, passed through to `lambda_select` (see there for details); defaults
               to `normal_weight`, matching the paper.
    """
    a, b, fpc = pilot_fpca(X, Y, grid, n_basis=n_basis, n_components=n_components)

    lam_hat, diag = lambda_select(
        X, Y, grid, a, b, fpc, theta_bar, lambdas,
        n_fourier=n_fourier, m_tuning=m_tuning, w=w
    )

    theta_hat, _, _ = theta_methodB(X, Y, grid, a, b, lam_hat)

    a_hat, b_hat = ab_methodB(theta_hat, grid, a, b, X, Y=Y, refit=refit,
                              n_basis=n_basis, n_components=n_components)

    return {"a_hat": a_hat, "b_hat": b_hat, "lam_hat": lam_hat,
            "theta_hat": theta_hat, "diag": diag}


def fit_mirror(X, Y, grid, lambdas, u_bar=None, n_basis=20, n_components=10, n_fourier=1,
               m_tuning=None, refit=True, w=None):
    """Performs method B for left-endpoint mirror, using reflection of curves.

    The reflection is X(t) -> X(1 - t), applied by reversing the columns of X, so `grid` must
    be symmetric about its midpoint.

    Returns parameters (dict)

    PARAMETERS
    u_bar    : float or None, forces the step-3 b_simp's cutoff at 1 - u_bar
               (mirrored theta_bar). If None (default), theta_bar is instead estimated
               jointly with b_simp's coefficients by unpenalised least squares;
    m_tuning : int, caps the number of FPCs used only inside lambda_select's S_Y/S_b/P_b
               criterion (see lambda_select);
    refit    : bool, passed through to `ab_methodB`, chooses between re-running the pilot
               method restricted to [0, theta_hat] (refit=True, the default) and truncating
               the untruncated pilot (refit=False);
    w        : callable, passed through to `lambda_select` (see there for details); defaults
               to `normal_weight`, matching the paper.
    """
    check_mirrorable(grid)
    X_r = X[:, ::-1]
    a_r, b_r, fpc_r = pilot_fpca(X_r, Y, grid, n_basis=n_basis, n_components=n_components)

    theta_bar = None if u_bar is None else 1.0 - u_bar

    lam_hat, diag = lambda_select(
        X_r, Y, grid, a_r, b_r, fpc_r, theta_bar, lambdas,
        n_fourier=n_fourier, m_tuning=m_tuning, w=w
    )

    theta_hat, _, _ = theta_methodB(X_r, Y, grid, a_r, b_r, lam_hat)
    u_hat = 1.0 - theta_hat

    a_hat, b_hat_r = ab_methodB(theta_hat, grid, a_r, b_r, X_r, Y=Y, refit=refit,
                                n_basis=n_basis, n_components=n_components)
    b_hat = b_hat_r[::-1]

    diag["u_lam"] = 1.0 - diag["theta_lams"]
    diag["u_bar"] = 1.0 - diag["theta_bar"]
    return {"u_hat": u_hat, "a_hat": a_hat, "b_hat": b_hat, "lam_hat": lam_hat,
            "theta_hat": theta_hat, "diag": diag}


def lambda_select(X, Y, grid, a_pilot, b_pilot, fpc, theta_bar, lambdas,
                  sigma2=None, n_fourier=1, theta_grid=None, m_tuning=None, w=normal_weight):
    """Select lambda by the method B branch of section 3.2 (steps 3, 4', 5', 6', 7).

    At each candidate theta the target (a*, b*) is refit onto the *untruncated* FPCs phi_j^1
    and then truncated, giving

      - S_Y^B(theta) = sum_i (Ybar*_i - Ycheck*_i(theta))²
                       + sigma² sum_j (tau_j^1)^-1 sum_i (int_0^theta phi_j^1 X_i)²
      - S_b^B(theta) = int_I (b_simp - b* I(t < theta))² + sigma² sum_j (tau_j^theta)^-1
                       int_0^theta (phi_j^1)²

    Step 6' takes theta_lambda to be the *first* minimum of S_Y^B(theta) + n lambda theta²,
    since b_check is already a squared-error minimiser and S_Y^B drops sharply near the right
    end of the interval. V(lambda) and P_b(lambda) follow `_common.theta_lam_variance` and
    `_common.weighted_S_b`.

    Returns lambda (float) and diagnostics (dict)

    PARAMETERS
    theta_bar : float or None, the cutoff b_simp (the step-3 parametric estimate) decays to
                zero at. If None, it is estimated jointly with b_simp's coefficients by
                unpenalised least squares;
    lambdas  : array, candidates for optimal lambda;
    sigma2   : float, pilot residual variance; estimated from the noisy pilot fit if None;
    m_tuning : int, caps the number of FPCs used inside this criterion only, leaving the pilot
               and the final fit alone (see the package README);
    w        : callable, the weighting density used in step 7 to average S_b(theta) around
               theta_lam.
    """
    if w is None:
        w = normal_weight
    if theta_grid is None:
        theta_grid = grid[(grid > grid[0]) & (grid < grid[-1])]

    n = len(Y)
    phi, tau = fpc["phi"], fpc["tau"]
    if m_tuning is not None:
        phi, tau = phi[:m_tuning], tau[:m_tuning]
    X_bar = fpc["mean"]

    # If pilot residual variance is not supplied, we estimate it from the real (noisy) fit
    if sigma2 is None:
        Y_pilot = a_pilot + np.trapezoid(b_pilot[None, :] * X, grid, axis=1)
        sigma2 = np.mean((Y - Y_pilot) ** 2)

    # Step 3 (paper, section 3.2): b_simp is fit jointly with its own cutoff theta_bar by
    # minimising unpenalised squared error against (X, Y)
    b_simp, theta_bar = fit_b_simp(X, Y, grid, a_pilot, n_fourier=n_fourier, theta_bar=theta_bar)

    # Step 4' (paper, section 3.2): refit (a*, b*) to the noiseless parametric estimate
    # target Ybar_star = a_pilot + integral(b_simp * X), using the same full-interval FPCA
    # basis (phi) as the pilot
    Ybar_star = a_pilot + np.trapezoid(b_simp[None, :] * X, grid, axis=1)
    scores_full = np.trapezoid(phi[:, None, :] * X[None, :, :], grid, axis=2)  # (m, n)
    star_fit = LinearRegression().fit(scores_full.T, Ybar_star)
    b_star = star_fit.coef_ @ phi
    a_star = Ybar_star.mean() - np.trapezoid(b_star * X_bar, grid)

    S_Y = np.empty(len(theta_grid))
    S_b = np.empty(len(theta_grid))
    # Select the part of the grid where the coefficient function is active, i.e. [0, theta]
    for q, theta in enumerate(theta_grid):
        interval = grid <= theta
        grid_i = grid[interval]

        # Truncated reconstruction of the parametric target, y-check*_i(theta)
        Y_hat_star = a_star + np.trapezoid(
            b_star[interval][None, :] * X[:, interval], grid_i, axis=1
        )
        bias_Y = np.sum((Ybar_star - Y_hat_star) ** 2)
        # Truncated FPC scores
        sc = np.trapezoid(phi[:, interval][:, None, :] * X[None, :, interval], grid_i, axis=2)
        var_Y = sigma2 * np.sum((1.0 / tau)[:, None] * sc ** 2)
        S_Y[q] = bias_Y + var_Y

        b_trunc = np.where(interval, b_star, 0.0)
        bias_b = np.trapezoid((b_simp - b_trunc) ** 2, grid)
        phi2 = np.trapezoid(phi[:, interval] ** 2, grid_i, axis=1)
        tau_theta = trunc_eigvals(X, grid, interval, phi.shape[0])  # tau_j^theta, not tau_j^1
        S_b[q] = bias_b + sigma2 * np.sum((1.0 / tau_theta) * phi2)

    pen = theta_grid ** 2
    P_b = np.empty(len(lambdas))
    theta_lams = np.empty(len(lambdas))
    Vs = np.empty(len(lambdas))
    for jx, lam in enumerate(lambdas):
        obj = S_Y + n * lam * pen
        # Step 6': the first interior minimum, not the global one.
        idx = None
        for k in range(1, len(obj) - 1):
            if obj[k] <= obj[k - 1] and obj[k] <= obj[k + 1]:
                idx = k
                break
        if idx is None:
            idx = int(np.argmin(obj))
        theta_lams[jx] = theta_grid[idx]

        V = theta_lam_variance(S_Y, theta_grid, idx, lam, sigma2, n)
        Vs[jx] = V
        if not np.isfinite(V):
            P_b[jx] = np.nan
            continue
        # Step 7
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
        "S_Y": S_Y, "S_b": S_b, "theta_grid": theta_grid, "b_simp": b_simp, "b_star": b_star,
        "a_star": a_star, "sigma2": sigma2, "theta_bar": theta_bar,
    }


def theta_methodB(X, Y, grid, a_pilot, b_pilot, lam, theta_grid=None):
    """Estimate theta via method B, i.e. minimise

        T(theta) = sum_i {Y_i - a_check - int_0^theta b_check X_i}² + n lambda theta²    (3.5)

    taking its first interior minimum, or the global one if no interior minimum is found.
    Returns theta_hat (float), theta candidates (array), and T evaluated at each (array).

    PARAMETERS
    lam        : float, penalty parameter (choose via `lambda_select`);
    theta_grid : (N, ) array, endpoints scanned over, defaults to grid's interior points.
    """
    n = len(Y)
    if theta_grid is None:
        theta_grid = grid[(grid > grid[0]) & (grid < grid[-1])]
    T = np.empty(len(theta_grid))
    for q, theta in enumerate(theta_grid):
        interval = grid <= theta
        pred = a_pilot + np.trapezoid(
            b_pilot[interval][None, :] * X[:, interval], grid[interval], axis=1
        )
        T[q] = np.sum((Y - pred) ** 2) + n * lam * theta ** 2
    idx = None
    for k in range(1, len(T) - 1):
        if T[k] <= T[k - 1] and T[k] <= T[k + 1]:
            idx = k
            break
    if idx is None:
        idx = int(np.argmin(T))
    return theta_grid[idx], theta_grid, T


def ab_methodB(theta_hat, grid, a_pilot, b_pilot, X, Y=None, refit=True,
               n_basis=20, n_components=10):
    """Computes the final coefficient function and intercept once theta_hat is fixed.

    Two options from the paper (section 3.1.2, "Having computed theta_hat..."):

    - refit=True (default), "Alternatively...": rerun the standard pilot method (BIC-selected
      FPCA regression, see `pilot_fpca`) restricted to [0, theta_hat], in the linear model
      E(Y|X) = a + int_0^theta_hat bX. Requires Y; a_pilot/b_pilot are ignored.
    - refit=False, "most simply...": truncate the untruncated pilot b_pilot at theta_hat and
      correct a_pilot for the mass the truncation removes.

    Returns a_hat (float) and b_hat (array, zero for t > theta_hat).
    """
    if refit:
        interval = grid <= theta_hat
        grid_i = grid[interval]
        X_i = X[:, interval]

        # Restricted domain has fewer points near theta_hat ~ grid[0]; keep basis/component
        # counts from exceeding what the sub-grid can support. pilot_fpca's BIC search needs
        # m_range = range(2, n_components_i + 1) to be non-empty, hence the floor of 2, and
        # its cubic B-spline basis needs at least `PILOT_ORDER` functions however short the
        # sub-grid is, hence the floor there: without it a theta_hat leaving four or fewer
        # grid points asked skfda for a 3-function basis of order 4 and raised.
        n_basis_i = min(n_basis, max(len(grid_i) - 1, PILOT_ORDER))
        n_components_i = max(2, min(n_components, len(grid_i) - 1, len(Y) - 1))

        a_hat, b_sub, _ = pilot_fpca(X_i, Y, grid_i, n_basis=n_basis_i,
                                     n_components=n_components_i)

        b_hat = np.zeros_like(grid)
        b_hat[interval] = b_sub
        return a_hat, b_hat

    # b_hat(t) = b_pilot(t) for t <= theta_hat, else 0
    b_hat = np.where(grid <= theta_hat, b_pilot, 0.0)
    # Updating a using the integral of b_pilot * mean(X) over (theta_hat, 1]
    correction = np.trapezoid(np.where(grid > theta_hat, b_pilot, 0.0) * X.mean(axis=0), grid)
    a_hat = a_pilot - correction
    return a_hat, b_hat
