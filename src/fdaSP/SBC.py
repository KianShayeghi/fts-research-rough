"""Locally sparse scalar-on-function regression, as in Stefanucci, Bernardi & Canale, 2025.

Designs a method to estimate a coefficient function ψ(t) which is exactly zero on a (+ve measure)
subset of its domain. In null regions, the response is independent of the regressor.

Rather than assuming that the informative set is a single interval and scanning for its boundary
as in Hall & Hooker, 2016, this method allows it to be any union of intervals.

This is a Python port of ``f2sSP`` from the R package ``fdaSP``

The model is

    y_i = ∫_T x_i(t) ψ(t) dt + z_i ^ T γ + ε_i,    i = 1, ..., n

"""

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve

__all__ = [
    "bspline_basis",
    "bspline_design",
    "pen_matrices",
    "group_soft_th",
    "admm",
    "fit",
    "find_regions",
    "lambda_max",
    "lambda_path",
    "fit_path",
    "cv_fit",
    "blocked_folds",
    "embargo_mask",
    "predict",
    "zero_coefficients",
    "support_mask",
    "f2ssp_design",
    "f2ssp_penalty",
    "fit_like_f2ssp",
]


def bspline_basis(grid, n_basis=15, order=4):
    """Evaluate a B-spline on a grid. Returns Phi (array)

    PARAMETERS
    grid: array, observation grid spanning domain [0, 1];
    n_basis: int, number of basis functions;
    order: int, order of spline, i.e. degree + 1.
    """
    grid = np.asarray(grid, dtype=float)
    if n_basis < order:
        raise ValueError(f"n_basis ({n_basis}) must be >= order ({order}).")
    a, b = grid[0], grid[-1]
    n_interior = n_basis - order  # Number of interior knots
    interior = np.linspace(a, b, n_interior + 2)[1:-1]  # Position of interior knots
    # Relaxes smoothness at endpoints
    knots = np.concatenate([np.full(order, a), interior, np.full(order, b)])

    Phi = np.empty((grid.size, n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        Phi[:, i] = BSpline(knots, c, order - 1)(grid)
    return Phi


def bspline_design(X, grid, n_basis=15, order=4):
    """Build the functional design matrix W .

    Discretises integral via trapezium rule, so the model's functional term
    becomes an ordinary linear term. Returns W (array) and Phi (array).

    PARAMETERS
    X: array, curves evaluated on grid
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    grid = np.asarray(grid, dtype=float)
    Phi = bspline_basis(grid, n_basis=n_basis, order=order)

    # Trapezium weights
    w = np.zeros(grid.size)
    w[1:] += np.diff(grid)  # w = [0, ..., 1 - ...]
    w[:-1] += np.diff(grid)  # w= [0 + ..., ..., 1 - ...]
    w *= 0.5

    W = (X * w) @ Phi
    return W, Phi


def pen_matrices(n_basis, order=4, group_weights=None, var_weights=None):
    """Construct the penalty matrix F from (62.3) of the paper.

    Returns F (array), and groups (list)

    PARAMETERS
    group_weights: array, group-specific weight matrix Υ
    var_weights: array, diagonal of weighting matrix T
    """
    k, d = int(n_basis), int(order)
    if k < d:
        raise ValueError(f"n_basis ({k}) must be >= order ({d}).")
    B = k - d + 1

    groups = [np.arange(b, b + d) for b in range(B)]  # Number of groups

    # Count how many groups contain coefficient j
    counts = np.zeros(k)
    for g in groups:
        counts[g] += 1.0
    S_star = np.diag(1.0 / np.sqrt(counts))  # Normalise weighting of boundary and interior coefs

    upsilon = np.ones(B) if group_weights is None else np.asarray(group_weights, float)
    if upsilon.shape != (B,):
        raise ValueError(f"group_weights must have length B = {B}, got {upsilon.shape}.")
    if np.any(upsilon <= 0):
        raise ValueError("group_weights must be positive.")

    t = np.ones(k) if var_weights is None else np.asarray(var_weights, float)
    if t.shape != (k,):
        raise ValueError(f"var_weights must have length k = {k}, got {t.shape}.")
    if np.any(t < 0):
        raise ValueError("var_weights must be non-negative.")
    T = np.diag(t)

    F = np.zeros((B * d, k))
    for b, g in enumerate(groups):
        S_b = np.zeros((d, k))
        S_b[np.arange(d), g] = 1.0  # Selection matrices
        F[b * d:(b + 1) * d, :] = upsilon[b] * (S_b @ S_star @ T)  # Alternative F = ΥHT

    return F, groups


def group_soft_th(x, kappa):
    """Group soft-thresholding operator S_κ, used for ζ-step in ADMM algorithm (62.4).

    Shrinks entire block towards zero and sets it to exactly zero once ||x||_2 <= κ,
    giving the estimator its 'locally sparse' property.

    PARAMETERS
    x: array, the block to threshold;
    kappa: threshold λ/ρ in ADMM algorithm.
    """
    x_norm = np.linalg.norm(x)
    if x_norm <= kappa:
        return np.zeros_like(x)
    return (1.0 - kappa / x_norm) * x


def admm(
    W,
    y,
    F,
    lam,
    Z=None,
    order=4,
    rho=1.0,
    rho_update=True,
    tau=2.0,
    mu=10.0,
    abstol=1e-6,
    reltol=1e-4,
    max_iter=1000,
    init=None,
):
    """Solve equation (62.3) by alternating direction method of multipliers (ADMM) method.

    Returns a dictionary of relevant variables.

    PARAMETERS
    y: array, response variable;
    lam: float, penalty parameter;
    Z: array or None, scalar covariates, must add ones column before passing;
    rho: float, ADMM step size;
    rho_update: bool, optional iteration of rho at each step outline in paper;
    tau: float, multiplicative factor for rho update;
    mu: Residual multiplier, triggers a rho update if exceeded;
    abstol, reltol: float, absolute and relative tolerances for stopping ADMM;
    max_iter: int, cap on iterations;
    init: dict or None, starting values.
    """
    W = np.asarray(W, float)
    y = np.asarray(y, float).ravel()
    F = np.asarray(F, float)
    n, k = W.shape
    d = int(order)
    B = F.shape[0] // d

    if Z is not None:
        Z = np.atleast_2d(np.asarray(Z, float))
        if Z.shape[0] != n:
            raise ValueError(f"Z has {Z.shape[0]} rows but y has {n}.")
        ZtZ_inv_Zt = np.linalg.solve(Z.T @ Z, Z.T)  # Pre-compute once
        q = Z.shape[1]

    WtW = W.T @ W
    Wty = W.T @ y
    FtF = F.T @ F

    def factorise(r):
        return cho_factor(WtW + r * FtF, lower=True)

    chol = factorise(rho)

    if init is None:
        psi = np.zeros(k)
        gamma = np.zeros(q) if Z is not None else None
        zeta = np.zeros(B * d)
        u = np.zeros(B * d)
    else:
        psi = np.array(init["psi"], float)
        gamma = np.array(init["gamma"], float) if Z is not None else None
        zeta = np.array(init["zeta"], float)
        u = np.array(init["u"], float)

    converged = False
    r_norm = s_norm = eps_pr = eps_du = np.nan

    for iter in range(1, max_iter + 1):
        # Psi update: ridge-type solve with cholesky decomp
        rhs = Wty.copy()
        if Z is not None:
            rhs -= W.T @ (Z @ gamma)
        rhs += rho * (F.T @ (zeta - u))
        psi = cho_solve(chol, rhs)

        # Gamma update: OLS of partial residual on Z, only considered if covariates are included
        if Z is not None:
            gamma = ZtZ_inv_Zt @ (y - W @ psi)

        # Zeta update: blockwise soft threshold
        zeta_old = zeta
        Fpsi = F @ psi
        v = Fpsi + u
        zeta = np.concatenate([group_soft_th(v[b * d:(b + 1) * d], lam / rho) for b in range(B)])

        # Dual update
        r = Fpsi - zeta  # Primal residual
        u = u + r

        # Stopping rule (Boyd 3.3.1)
        s = rho * (F.T @ (zeta - zeta_old))  # Dual residual
        r_norm, s_norm = np.linalg.norm(r), np.linalg.norm(s)
        eps_pr = np.sqrt(B * d) * abstol + reltol * max(np.linalg.norm(Fpsi), np.linalg.norm(zeta))
        eps_du = np.sqrt(k) * abstol + reltol * np.linalg.norm(rho * (F.T @ u))
        if r_norm <= eps_pr and s_norm <= eps_du:
            converged = True
            break

        # If rho is to be updated, this step is included
        if rho_update:
            if r_norm > mu * s_norm:
                rho *= tau
                u /= tau
                chol = factorise(rho)
            elif s_norm > mu * r_norm:
                rho /= tau
                u *= tau
                chol = factorise(rho)

    return {
        "psi": psi,
        "gamma": gamma,
        "zeta": zeta,
        "u": u,
        "rho": rho,
        "n_iter": iter,
        "converged": converged,
        "r_norm": r_norm,
        "s_norm": s_norm,
        "eps_primal": eps_pr,
        "eps_dual": eps_du,
    }


def find_regions(mask, grid):
    """Convert a boolean mask over grid points into intervals.
    Returns a list of tuples.

    PARAMETERS
    mask: array, boolean, aligned with grid;
    grid: array, observation grid.
    """
    mask = np.asarray(mask, dtype=bool)
    grid = np.asarray(grid, dtype=float)
    if not mask.any():
        return []
    changes = np.flatnonzero(np.diff(mask.astype(int))) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [mask.size]])
    return [(grid[s], grid[e - 1]) for s, e in zip(starts, ends) if mask[s]]


def zero_coefficients(zeta, groups, order):
    """Which basis coefficients the overlap group-lasso has set exactly to zero.

    The ADMM constraint is F psi = zeta, so a group whose zeta block is exactly zero forces
    *every* coefficient in that group to zero: psi_j = 0 as soon as **any** group containing
    j is switched off. (Requiring all of them to be off would under-report the null set by
    up to order - 1 coefficients at each edge of a null region -- those coefficients are
    numerically zero in psi, but would be reported as support.)

    Returns a boolean array of length n_basis, True where the coefficient is null.

    PARAMETERS
    zeta   : (B * order,) array, the sparse ADMM variable returned by `admm`;
    groups : list of index arrays, as returned by `pen_matrices`;
    order  : int, spline order, i.e. the group size.
    """
    group_is_zero = np.all(np.asarray(zeta, float).reshape(len(groups), int(order)) == 0.0,
                           axis=1)
    n_basis = max(int(g[-1]) for g in groups) + 1
    is_zero = np.zeros(n_basis, dtype=bool)
    for b, g in enumerate(groups):
        if group_is_zero[b]:
            is_zero[g] = True
    return is_zero


def support_mask(Phi, coef_is_zero):
    """Grid points where the fitted coefficient function is not identically zero.

    A grid point is in the support if at least one basis function that is non-zero there has
    a non-zero coefficient. Returns a boolean array over the grid.
    """
    return (np.asarray(Phi, float) != 0) @ (~np.asarray(coef_is_zero, bool)) > 0


def fit(
        X,
        y,
        grid,
        lam,
        Z=None,
        n_basis=15,
        order=4,
        group_weights=None,
        var_weights=None,
        init=None,
        **admm_kwargs,
):
    """Fit the locally sparse estimator for a fixed lambda. Returns `admm` output and more.

    PARAMETERS
    **admm_kwargs: rho, rho_update, abstol, etc.
    """
    W, Phi = bspline_design(X, grid, n_basis=n_basis, order=order)
    F, groups = pen_matrices(
        n_basis, order=order, group_weights=group_weights, var_weights=var_weights
    )
    out = admm(W, y, F, lam, Z=Z, order=order, init=init, **admm_kwargs)

    # Snap the coefficients the penalty switched off to exact zeros. ADMM leaves them at
    # solver tolerance (~1e-11) in psi while zeta holds the exact zero, so without this the
    # reported `support` and the reported `beta` disagree and callers need a magic threshold.
    coef_is_zero = zero_coefficients(out["zeta"], groups, order)
    psi = np.where(coef_is_zero, 0.0, out["psi"])
    out["psi"] = psi

    beta = Phi @ psi
    fitted = W @ psi
    if Z is not None and out["gamma"] is not None:
        fitted = fitted + np.asarray(Z, float) @ out["gamma"]

    support = support_mask(Phi, coef_is_zero)

    out.update({
        "beta": beta,
        "coef_is_zero": coef_is_zero,
        "Phi": Phi,
        "W": W,
        "F": F,
        "groups": groups,
        "lam": lam,
        "fitted": fitted,
        "support": support,
        "null_regions": find_regions(~support, grid),
        "support_regions": find_regions(support, grid),
        "n_basis": n_basis,
        "order": order,
    })
    return out


def predict(res, X, grid, Z=None):
    """Predicts responses on new curves using fitted result dict. Returns fitted values (array)
    Remember to add intercept for Z.

    PARAMETERS
    res: dict, results of `fit`
    """
    W, _ = bspline_design(X, grid, n_basis=res["n_basis"], order=res["order"])
    pred = W @ res["psi"]
    if Z is not None and res["gamma"] is not None:
        pred = pred + np.atleast_2d(np.asarray(Z, float)) @ res["gamma"]
    return pred


def lambda_max(W, y, F, Z=None, order=4, n_bisect=30, **admm_kwargs):
    """Returns smallest value of lambda for which the fitted curve is all zero,
    i.e. upper bound of lambda. Calculated via bisecting on solver.

    PARAMETERS
    n_bisect : int, bisection steps"""
    W = np.asarray(W, float)
    y = np.asarray(y, float).ravel()

    # y is residualised on Z, and Z is passed on to `admm` as well, so the switch-off point is
    # the one for the functional term after Z has been projected out of both y and W.
    r0 = y.copy()
    if Z is not None:
        Z = np.atleast_2d(np.asarray(Z, float))
        r0 = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]

    # We test ζ as this is set exactly to zero in the optimisation step
    def is_zero(lam):
        out = admm(W, r0, F, lam, Z=Z, order=order, **admm_kwargs)
        return np.all(out["zeta"] == 0.0)

    # Crude upper bound, double until all zeroes
    upp = float(np.linalg.norm(W.T @ r0))
    if upp <= 0.0:
        return 1.0
    for _ in range(60):
        if is_zero(upp):
            break
        upp *= 2.0
    else:
        return upp

    # Binary search
    low = 0.0
    for _ in range(n_bisect):
        mid = 0.5 * (low + upp)
        if is_zero(mid):
            upp = mid
        else:
            low = mid
    return float(upp)


def lambda_path(W, y, F, Z=None, order=4, n_lambda=50, eps=1e-3):
    """Grid (geometric) from lambda_max down to eps * lambda max

    Descending, as in `glmnet` convention from R
    """
    lmax = lambda_max(W, y, F, Z=Z, order=order)
    if lmax <= 0:
        lmax = 1.0
    return np.geomspace(lmax, eps * lmax, n_lambda)


def fit_path(X, y, grid, lambdas=None, Z=None, n_basis=15, order=4, group_weights=None,
             var_weights=None, n_lambda=50, eps=1e-3, **admm_kwargs):
    """Fits the whole lambda path, warm-starting each solve from the previous one.

    Lambdas are solved in descending order, so each fit begins from a slightly less sparse
    solution and the path is cheaper than the same fits computed independently. Returns a
    dict holding, for every lambda: the spline coefficients "psis", the coefficient
    functions "betas" on the grid, the exactly-zero coefficients "coef_is_zero" and the
    grid-point "supports" they imply, the scalar coefficients "gammas", and the ADMM
    "iterations" and "converged" flags. "lambdas" comes back sorted descending, whatever
    order it was given in.

    PARAMETERS
    lambdas : array or None, the path to solve; built by `lambda_path` when None.
    """
    W, Phi = bspline_design(X, grid, n_basis=n_basis, order=order)
    F, groups = pen_matrices(n_basis, order=order, group_weights=group_weights,
                             var_weights=var_weights)
    if lambdas is None:
        lambdas = lambda_path(W, y, F, Z=Z, order=order, n_lambda=n_lambda, eps=eps)
    lambdas = np.asarray(lambdas, float)
    lambdas = np.sort(lambdas)[::-1]

    L, k = lambdas.size, n_basis
    psis = np.zeros((L, k))
    zeroed = np.zeros((L, k), dtype=bool)
    gammas = None if Z is None else np.zeros((L, np.atleast_2d(Z).shape[1]))
    n_iter = np.zeros(L, dtype=int)
    conv = np.zeros(L, dtype=bool)

    state = None
    for j, lam in enumerate(lambdas):
        kwargs = dict(admm_kwargs)
        if state is not None and "rho" not in kwargs:
            # Carry forward rho from previous solve to keep rho * u consistent in dual problem.
            # u is the scaled dual variable, with scale rho.
            kwargs["rho"] = state["rho"]
        out = admm(W, y, F, lam, Z=Z, order=order, init=state, **kwargs)
        state = out  # Warm start for next lambda -- seeded from the *unsnapped* iterate
        zeroed[j] = zero_coefficients(out["zeta"], groups, order)
        psis[j] = np.where(zeroed[j], 0.0, out["psi"])  # exact zeros, as in `fit`
        if gammas is not None:
            gammas[j] = out["gamma"]
        n_iter[j] = out["n_iter"]
        conv[j] = out["converged"]

    return {
        "lambdas": lambdas,
        "psis": psis,
        "betas": psis @ Phi.T,
        "coef_is_zero": zeroed,
        "supports": np.array([support_mask(Phi, z) for z in zeroed]),
        "gammas": gammas,
        "iterations": n_iter,
        "converged": conv,
        "W": W,
        "Phi": Phi,
        "F": F,
        "n_basis": n_basis,
        "order": order,
    }


def blocked_folds(n, n_folds):
    """Contiguous cross-validation fold labels for `n` observations in time order.

    Fold f is the block of consecutive indices [f*n/n_folds, (f+1)*n/n_folds), so serially
    dependent observations stay in the same fold instead of being scattered across all of
    them. Returns an (n,) integer array of fold labels.

    PARAMETERS
    n_folds : int, number of folds.
    """
    n, n_folds = int(n), int(n_folds)
    if n_folds < 2 or n_folds > n:
        raise ValueError(f"n_folds must satisfy 2 <= n_folds <= n = {n}, got {n_folds}.")
    return np.minimum(np.arange(n) * n_folds // n, n_folds - 1)


def embargo_mask(folds, f, embargo):
    """Training-set mask for fold `f`, dropping observations within `embargo` positions of it.

    Where predictor i is a window of the series and the response summarises the observations
    after it, neighbouring i share raw data, so an observation just outside a test block still
    carries information about it. Setting `embargo` to the width of that overlap removes the
    shared observations from the training set. Returns an (n,) boolean array, True where the
    observation is used for training.

    PARAMETERS
    folds   : (n,) array of fold labels, as from `blocked_folds`;
    f       : int, the held-out fold;
    embargo : int, number of positions either side of the test block to drop.
    """
    folds = np.asarray(folds)
    test = folds == f
    train = ~test
    if embargo > 0:
        near = np.convolve(test.astype(int), np.ones(2 * int(embargo) + 1), mode="same") > 0
        train &= ~near
    return train


def cv_fit(X, y, grid, lambdas=None, Z=None, n_basis=15, order=4, group_weights=None,
           var_weights=None, n_folds=10, one_se=False, n_lambda=50, eps=1e-3, rng=123,
           folds=None, embargo=0, **admm_kwargs):
    """n_fold cross-validation over lambda_path.

    Find lambda with the smallest MSE mean (one_se=False), add its SE, and take the largest
    lambda whose MSE mean is <= smallest MSE mean + its SE (one_se=True). Returns `fit`'s
    output at the selected lambda, plus the CV curve under "cv_mean"/"cv_se", the selected
    "lam_hat", the unrestricted minimiser "lam_min", and the fold labels used under "folds".

    Folds are assigned at random by default, which assumes the rows of X are exchangeable.
    Where they are not, as for the overlapping windows `empirical.rolling_windows` builds,
    pass `folds=blocked_folds(n, n_folds)` and an `embargo` at least as wide as the overlap.

    PARAMETERS
    folds   : (n,) array of fold labels or None, overriding the random assignment;
    embargo : int, observations within this many positions of the test block are dropped from
              the training set (see `embargo_mask`).
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = y.size
    Z = None if Z is None else np.atleast_2d(np.asarray(Z, float))

    if lambdas is None:
        W_all, _ = bspline_design(X, grid, n_basis=n_basis, order=order)
        F_all, _ = pen_matrices(
            n_basis, order=order, group_weights=group_weights, var_weights=var_weights
        )
        lambdas = lambda_path(W_all, y, F_all, Z=Z, order=order, n_lambda=n_lambda, eps=eps)
    lambdas = np.sort(np.asarray(lambdas, float))[::-1]
    L = lambdas.size

    if folds is None:
        folds = rng.permutation(n) % n_folds  # Randomises data and separates into folds
    else:
        folds = np.asarray(folds)
        if folds.shape != (n,):
            raise ValueError(f"folds must have length n = {n}, got {folds.shape}.")
        n_folds = len(np.unique(folds))
    fold_mse = np.full((n_folds, L), np.nan)

    for f in np.unique(folds):
        te = folds == f
        tr = embargo_mask(folds, f, embargo)
        Ztr = None if Z is None else Z[tr]
        Zte = None if Z is None else Z[te]
        # Training
        path = fit_path(
            X[tr], y[tr], grid, lambdas=lambdas, Z=Ztr, n_basis=n_basis, order=order,
            group_weights=group_weights, var_weights=var_weights, **admm_kwargs
        )
        # Test
        W_te, _ = bspline_design(X[te], grid, n_basis=n_basis, order=order)
        for j in range(L):
            pred = W_te @ path["psis"][j]
            if Zte is not None:
                pred = pred + Zte @ path["gammas"][j]
            fold_mse[int(f), j] = np.mean((y[te] - pred) ** 2)

    cv_mean = np.nanmean(fold_mse, axis=0)
    cv_se = np.nanstd(fold_mse, axis=0, ddof=1) / np.sqrt(n_folds)

    j_min = int(np.argmin(cv_mean))
    lam_min = lambdas[j_min]
    if one_se:
        # Since lambdas are descending, the first index within one SE of the minimum is the
        # largest admissible lambda
        bound = cv_mean[j_min] + cv_se[j_min]
        j_opt = int(np.argmax(cv_mean <= bound))
    else:
        j_opt = j_min
    lam_hat = lambdas[j_opt]

    out = fit(X, y, grid, lam_hat, Z=Z, n_basis=n_basis, order=order, group_weights=group_weights,
              var_weights=var_weights, **admm_kwargs)
    out.update({
        "lam_hat": lam_hat,
        "lam_min": lam_min,
        "lambdas": lambdas,
        "cv_mean": cv_mean,
        "cv_se": cv_se,
        "one_se": one_se,
        "folds": folds,
        "embargo": embargo,
    })
    return out


# ---------------------------------------------------------------------------------------
# Matching CRAN fdaSP::f2sSP.
#
# f2sSP solves the same problem (62.3) under four different conventions, so the two
# parametrise one estimator rather than computing two. The helpers below apply f2sSP's
# choice of each:
#
#   (a) Loss scaling. Its objective is (1/2n)||y - W psi||² + lambda*pen, where `admm`
#       uses (1/2)||.||². The 1/n only rescales lambda, and scaling the data by 1/sqrt(n)
#       absorbs it exactly.
#   (b) Penalty weights. It builds F_g = S_g diag(var_weights) with the default
#       var_weights = 1/counts, where counts_j is the number of groups containing j;
#       `pen_matrices` composes F_g = upsilon_g S_g S* T with the extra normalisation
#       S* = diag(1/sqrt(counts)). Both are valid weightings of the same overlap penalty.
#   (c) Design weighting. It forms W = X Phi as a plain matrix product, where
#       `bspline_design` folds trapezium quadrature weights in; the constant between them
#       is absorbed into lambda.
#   (d) Standardisation. Under standardize.data = TRUE it centres and scales the
#       basis-projected design and the response, solves, then maps back.
# ---------------------------------------------------------------------------------------


def f2ssp_design(X, n_basis=15, order=4):
    """Basis-projected design under convention (c), W = X Phi.

    A plain matrix product: unlike `bspline_design`, no trapezium quadrature weights are
    folded in, so W is the unweighted sum over grid points rather than an approximation to
    ∫ x_i(t) phi_j(t) dt. On an equally spaced grid the two agree up to the constant factor
    1/h, apart from the half weights the trapezium rule gives the two endpoints; that
    constant is absorbed into lambda.

    Returns W (n, n_basis) and Phi (p, n_basis).

    PARAMETERS
    X : (n, p) array, curves sampled on an equally spaced grid of [0, 1].
    """
    X = np.atleast_2d(np.asarray(X, float))
    grid = np.linspace(0.0, 1.0, X.shape[1])
    Phi = bspline_basis(grid, n_basis=n_basis, order=order)
    return X @ Phi, Phi


def f2ssp_penalty(n_basis=15, order=4):
    """Penalty matrix F and its groups under convention (b), F_g = S_g diag(1/counts).

    counts_j is the number of groups containing coefficient j, so each coefficient is
    weighted by how often the overlapping groups charge for it. This is `pen_matrices` with
    its extra S* = diag(1/sqrt(counts)) normalisation undone, so it is obtained by passing
    T = diag(1/sqrt(counts)):

        S_g S* T = S_g diag(1/sqrt(counts)) diag(1/sqrt(counts)) = S_g diag(1/counts).

    Returns F (B*order, n_basis) and groups (list of index arrays).
    """
    k, d = int(n_basis), int(order)
    if k < d:
        raise ValueError(f"n_basis ({k}) must be >= order ({d}).")
    counts = np.zeros(k)
    for b in range(k - d + 1):
        counts[np.arange(b, b + d)] += 1.0
    return pen_matrices(k, order=d, var_weights=1.0 / np.sqrt(counts))


def fit_like_f2ssp(X, y, lam, n_basis=15, order=4, **admm_kwargs):
    """Fit the estimator under conventions (a) to (d), from raw curves and a raw response.

    Reproduces `f2sSP(vY = y, mX = X, M = n_basis, lambda1 = lam, standardize.data = TRUE,
    intercept = FALSE, splOrd = order)`.

    Returns a dict with `beta` (p,) the coefficient function on the sampling grid, `psi`
    (n_basis,) its spline coefficients on the original scale, `Phi`, `F`, and the `admm`
    output.

    PARAMETERS
    X   : (n, p) array of curves;
    y   : (n,) response, on its original scale;
    lam : float, the penalty, in f2sSP's parametrisation;
    **admm_kwargs : forwarded to `admm` (tolerances, max_iter, ...).
    """
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float).ravel()
    W, Phi = f2ssp_design(X, n_basis=n_basis, order=order)
    F, _ = f2ssp_penalty(n_basis=n_basis, order=order)
    n = W.shape[0]

    # (d) centre and scale the basis-projected design and the response
    mU = np.sqrt(np.var(W, axis=0, ddof=1))
    mV = np.sqrt(np.var(y, ddof=1))
    if np.any(mU <= 0) or mV <= 0:
        raise ValueError("f2sSP standardisation needs non-constant design columns and y.")
    W_std = (W - W.mean(axis=0)) / mU
    y_std = (y - y.mean()) / mV

    # (a) fold the 1/n loss scaling into the data, so lambda means the same thing
    scale = 1.0 / np.sqrt(n)
    kwargs = {"abstol": 1e-8, "reltol": 1e-6, "max_iter": 20000}
    kwargs.update(admm_kwargs)
    out = admm(W_std * scale, y_std * scale, F, lam, order=order, **kwargs)

    psi = (out["psi"] / mU) * mV  # back to the original scale, undoing (d)
    return {"beta": Phi @ psi, "psi": psi, "Phi": Phi, "F": F, "admm": out}
