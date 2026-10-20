"""Two-region estimation with continuity at the truncation point, extending Hall & Hooker.

`HH.py` and `HH_A.py` both assume the coefficient function is *exactly zero* on one side of
the truncation point:

    Y_i = int_0^theta b(t) X_i(t) dt + eps_i,    b == 0 on (theta, 1]

That is the right model when there is a true null region, but it is a hard constraint, not an
estimated one -- there is no candidate in either method's search space with a small, non-zero
tail. This module drops it. The coefficient function is instead

    b(t) = b1(t)  for t <= theta
           b2(t)  for t >  theta,   with b1(theta) = b2(theta),

estimated as separate empirical-FPC expansions of X restricted to each side (`_common.trunc_fpca`,
the same decomposition method A refits at every candidate theta), joined by one linear equality
constraint at theta. Hall & Hooker's zero-tail model is the special case b2 == 0; this model can
represent that, but does not assume it, so it is also the right model for a coefficient function
with no null region at all (`HH_dgp.sim_dataset(beta_fn=...)` with something like a sine wave on
one side and a ramp on the other -- see the module README).

Because neither side is presumed uninformative, method B's `n lambda theta^2` penalty has no
analogue here: that term exists to pull theta toward a shorter *informative* window, which
presumes the excluded window is free. Nor is there a section-3.2-style bias/variance criterion
for theta -- deriving one is a real piece of new asymptotic theory (see the corrections
`truncFLR/README.md` had to make to the paper's own), and inventing one here rather than doing
that derivation properly would be worse than not having it. Instead theta is chosen the same way
`HH_A.truncation_profile` chooses it for real data with no true endpoint to check a criterion
against: by cross-validated prediction error. Unlike that profile, there is no elbow to read --
with no null tail there is no flat stretch past the truth for the minimiser to wander over -- so
the minimiser of the CV curve is the estimate directly, not `theta_1se`.

Continuity also removes a piece of complexity the zero-tail model needed: `HH_A.trunc_predict`
integrates over the sub-grid a fit was made on because integrating the whole grid with b zeroed
past theta straddles the jump with a spurious half trapezium panel. There is no jump here, so
`continuous_predict` integrates the reconstructed b_hat over the whole grid directly, and splitting
the trapezium rule at theta to fit each side is exact for the same reason -- the two sub-integrals
sum to the whole-grid integral with no half-panel correction needed.
"""

import numpy as np
from scipy.linalg import null_space

from ._common import trunc_fpca

__all__ = [
    "fit", "fit_at_theta", "fit_components", "theta_scan", "truncation_profile",
    "continuous_predict",
]


def continuous_predict(a, b_hat, X, grid):
    """Fitted values a + int b_hat X of a continuous two-region model.

    b_hat carries no jump at theta, so this is one whole-grid trapezium integral -- unlike
    `HH_A.trunc_predict`, no split quadrature is needed to avoid double-counting a panel at the
    truncation point. Returns (n,) predictions.
    """
    return a + np.trapezoid(np.asarray(b_hat)[None, :] * np.asarray(X), grid, axis=1)


def _constrained_lstsq(design, Y, constraint_row):
    """Least squares fit of `design @ [a, coef] ~ Y` subject to `constraint_row @ coef == 0`.

    `design`'s first column is the intercept, left unconstrained. Solved by eliminating the one
    constraint through a basis of its null space (`scipy.linalg.null_space`) and fitting in the
    reduced coordinates, so the result is an exact least-squares solution under the constraint,
    not a penalised approximation to one. Returns a (float) and coef ((p,) array, p = the number
    of non-intercept columns of `design`).
    """
    Ns = null_space(constraint_row.reshape(1, -1))  # (p, p - 1) basis of {c : constraint @ c = 0}
    reduced = np.column_stack([design[:, 0], design[:, 1:] @ Ns])
    coef_r, *_ = np.linalg.lstsq(reduced, Y, rcond=None)
    return float(coef_r[0]), Ns @ coef_r[1:]


def fit_components(X, Y, grid, theta, m1, m2):
    """Fit the continuous two-region model at one candidate `theta`, returning the pieces
    `fit_at_theta` assembles into `b_hat` (and what the continuity constraint should be checked
    against directly, rather than by reading a single array entry -- see
    `tests/test_hh_continuous.py`).

    Each side is expanded in its own empirical FPCs of X restricted to that side (`trunc_fpca`),
    so the two bases need not agree in shape or complexity; they are joined only through the one
    scalar constraint `b1(theta) == b2(theta)`, evaluated from each side's basis functions at the
    grid point the two intervals share. Component counts are capped by what each sub-grid and the
    sample can support, mirroring the guards `HH_A.theta_methodA` uses.

    Returns a_hat (float), c1 (k1,), phi1 (k1, N_lo), c2 (k2,), phi2 (k2, N_hi), where
    `b1 = c1 @ phi1` on `grid[grid <= theta]` and `b2 = c2 @ phi2` on `grid[grid >= theta]`.

    PARAMETERS
    m1, m2 : int, requested number of empirical FPCs on [0, theta] and [theta, 1] respectively.
    """
    grid = np.asarray(grid, dtype=float)
    n = len(Y)
    lo = grid <= theta
    hi = grid >= theta
    m1 = max(1, min(m1, int(lo.sum()) - 1, n - 1))
    m2 = max(1, min(m2, int(hi.sum()) - 1, n - 1))

    grid_lo, grid_hi = grid[lo], grid[hi]
    phi1, _ = trunc_fpca(X, grid, lo, m1)
    phi2, _ = trunc_fpca(X, grid, hi, m2)
    k1 = phi1.shape[0]

    scores1 = np.trapezoid(phi1[:, None, :] * X[None, :, lo], grid_lo, axis=2).T  # (n, k1)
    scores2 = np.trapezoid(phi2[:, None, :] * X[None, :, hi], grid_hi, axis=2).T  # (n, k2)
    design = np.column_stack([np.ones(n), scores1, scores2])

    # phi1's last column and phi2's first column are both evaluated at theta, the grid point
    # the two intervals share.
    constraint = np.concatenate([phi1[:, -1], -phi2[:, 0]])
    a_hat, coef = _constrained_lstsq(design, Y, constraint)
    return a_hat, coef[:k1], phi1, coef[k1:], phi2


def fit_at_theta(X, Y, grid, theta, m1, m2):
    """Fit the continuous two-region model at one candidate `theta` and assemble it into a
    coefficient function on the full grid (see `fit_components` for the underlying pieces).

    Returns a_hat (float) and b_hat ((N,) array on the full grid, continuous at theta).
    """
    grid = np.asarray(grid, dtype=float)
    a_hat, c1, phi1, c2, phi2 = fit_components(X, Y, grid, theta, m1, m2)

    lo = grid <= theta
    hi_strict = grid > theta
    b_hat = np.zeros_like(grid)
    b_hat[lo] = c1 @ phi1
    b_hat[hi_strict] = c2 @ phi2[:, 1:]  # theta itself already set from the lo side
    return a_hat, b_hat


def theta_scan(X, Y, grid, m1=5, m2=5, theta_grid=None):
    """In-sample RSS(theta) of the continuous two-region fit, for diagnostics/plotting -- not
    the tuning rule (see `truncation_profile`), since nothing here penalises overfitting.

    Returns theta_grid (array) and RSS at each candidate (array).
    """
    grid = np.asarray(grid, dtype=float)
    if theta_grid is None:
        theta_grid = grid[(grid > grid[0]) & (grid < grid[-1])]
    rss = np.empty(len(theta_grid))
    for q, theta in enumerate(theta_grid):
        a_hat, b_hat = fit_at_theta(X, Y, grid, theta, m1, m2)
        pred = continuous_predict(a_hat, b_hat, X, grid)
        rss[q] = np.sum((Y - pred) ** 2)
    return theta_grid, rss


def truncation_profile(X, Y, grid, thetas=None, folds=None, n_folds=10, m1=5, m2=5, rng=0):
    """Held-out prediction error as a function of the candidate truncation point.

    The model-free counterpart of `HH_A.truncation_profile`, and read differently: that profile
    is read at its elbow because past the true endpoint the retained window carries no signal, so
    the curve flattens rather than rising and the minimiser wanders over the flat stretch. Here
    neither side is presumed uninformative, so there is no such flat stretch to protect against,
    and `theta_hat` is the minimiser of `mse` directly.

    Returns a dict with thetas, mse (averaged over folds), se (its standard error across folds),
    theta_hat (the minimiser) and folds.

    PARAMETERS
    thetas  : (K,) array of candidate truncation points, or None for 20 evenly spaced interior
              points of `grid`;
    folds   : (n,) array of fold labels, or None to assign them at random;
    n_folds : int, folds to assign when `folds` is None;
    m1, m2  : int, requested empirical-FPC counts on each side, passed to `fit_at_theta`.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.asarray(Y, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float)
    n = Y.size

    if thetas is None:
        interior = grid[(grid > grid[0]) & (grid < grid[-1])]
        thetas = interior[np.linspace(0, interior.size - 1, 20).round().astype(int)]
    thetas = np.asarray(thetas, dtype=float)

    if folds is None:
        folds = np.random.default_rng(rng).permutation(n) % n_folds
    folds = np.asarray(folds)
    labels = np.unique(folds)

    # A candidate retaining fewer than three grid points on either side cannot support an FPCA
    # there at all; such candidates score infinity and are skipped by the fold average.
    usable = np.array([
        int((grid <= th).sum()) >= 3 and int((grid >= th).sum()) >= 3 for th in thetas
    ])
    err = np.full((labels.size, thetas.size), np.inf)

    for f_i, f in enumerate(labels):
        te, tr = folds == f, folds != f
        for q, theta in enumerate(thetas):
            if not usable[q]:
                continue
            a_hat, b_hat = fit_at_theta(X[tr], Y[tr], grid, theta, m1, m2)
            err[f_i, q] = np.mean((Y[te] - continuous_predict(a_hat, b_hat, X[te], grid)) ** 2)

    mse = np.full(thetas.size, np.inf)
    se = np.full(thetas.size, np.inf)
    mse[usable] = err[:, usable].mean(axis=0)
    se[usable] = err[:, usable].std(axis=0, ddof=1) / np.sqrt(labels.size)

    best = int(np.argmin(mse))
    return {"thetas": thetas, "mse": mse, "se": se, "theta_hat": float(thetas[best]),
            "folds": folds}


def fit(X, Y, grid, m1=5, m2=5, thetas=None, folds=None, n_folds=10, rng=0):
    """End to end: pick theta by cross-validated prediction error, then fit the continuous
    two-region model at the selected theta on the full data.

    Returns a dict with a_hat, b_hat, theta_hat and the profile diagnostics (under "profile").
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.asarray(Y, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float)

    profile = truncation_profile(X, Y, grid, thetas=thetas, folds=folds, n_folds=n_folds,
                                 m1=m1, m2=m2, rng=rng)
    theta_hat = profile["theta_hat"]
    a_hat, b_hat = fit_at_theta(X, Y, grid, theta_hat, m1, m2)
    return {"a_hat": a_hat, "b_hat": b_hat, "theta_hat": theta_hat, "profile": profile}
