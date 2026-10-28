"""Quantities shared by methods A and B of Hall & Hooker, 2016.

Both methods run the same section 3.2 tuning scheme, differing only in steps 4-6. The pieces
that do not differ live here: the pilot estimator (step 1), the parametric target b_simp
(step 3), the truncated-domain principal components phi_j^theta and eigenvalues tau_j^theta,
the step-6 variance V(lambda), and the step-7 weighted average of S_b.

Two of the three corrections the package makes to section 3.2 are made here -- the scale of
lambda in step 6 and the quadrature of step 7 -- and `theta_lam_variance` and `weighted_S_b`
carry the reasoning. The third, the scale of the S_Y and S_b variance terms, belongs to the
criteria themselves and is made in `HH.lambda_select` and `HH_A.lambda_select`. The package
README derives all three.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.special import gamma
from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.linear_model import LinearRegression

from ..core.quadrature import quadrature_weights

__all__ = [
    "normal_weight", "uniform_weight", "fat_tailed_weight", "thin_tailed_weight",
    "WEIGHT_FUNCTIONS", "pilot_fpca", "fit_b_simp", "trunc_fpca", "trunc_eigvals",
    "theta_lam_variance", "weighted_S_b", "floor_eigvals",
    "check_mirrorable", "DegenerateSelection",
]


class DegenerateSelection(ValueError):
    """Section 3.2 cannot rank the candidate lambdas on this data.

    Raised by both `HH.lambda_select` and `HH_A.lambda_select` when P_b(lambda) is undefined
    at every candidate, which happens when theta_lambda lands on an end of the theta grid for
    all of them: S_Y + n lambda theta^2 then has no interior minimum, the step-6 variance is a
    Taylor expansion about a minimum that does not exist, and there is nothing for step 7 to
    average over. The criterion has no opinion, rather than a weak one.
    """


def check_mirrorable(grid):
    """Raise unless `grid` is symmetric about its midpoint.

    Reversing the columns of X realises the reflection X(t) -> X(1 - t) only on such a grid,
    which is what `HH.fit_mirror` and `HH_A.fit_mirror` reflect with.
    """
    grid = np.asarray(grid, dtype=float)
    reflected = grid[0] + grid[-1] - grid[::-1]
    if not np.allclose(grid, reflected, atol=1e-12, rtol=0):
        raise ValueError(
            "fit_mirror needs a grid symmetric about its midpoint, so that reversing the "
            "columns of X realises the reflection X(t) -> X(1 - t)."
        )


# ------------------------------------------------------------------- step 7 weights ------


def normal_weight(theta_grid, theta_lam, V):
    """N(theta_lam, V) density -- the step 7 weight used in the paper, and the default for
    `lambda_select`'s `w` argument."""
    return np.exp(-0.5 * (theta_grid - theta_lam) ** 2 / V) / np.sqrt(2 * np.pi * V)


def uniform_weight(theta_grid, theta_lam, V):
    """Uniform density on an interval centred at theta_lam, matched to variance V (compact
    support, i.e. the thinnest tails possible)."""
    half_width = np.sqrt(3.0 * V)
    return np.where(np.abs(theta_grid - theta_lam) <= half_width, 1.0 / (2 * half_width), 0.0)


def fat_tailed_weight(theta_grid, theta_lam, V):
    """Laplace (double-exponential) density matched to variance V -- fatter tails than
    normal_weight."""
    b = np.sqrt(V / 2.0)
    return np.exp(-np.abs(theta_grid - theta_lam) / b) / (2 * b)


def thin_tailed_weight(theta_grid, theta_lam, V, beta=4.0):
    """Generalised normal density matched to variance V -- thinner tails than normal_weight.

    The shape parameter `beta` recovers normal_weight at beta=2, and tends to uniform_weight as
    beta -> infinity; the default beta=4 sits between the two.
    """
    unit_var = gamma(3.0 / beta) / gamma(1.0 / beta)  # variance of this density at scale=1
    scale = np.sqrt(V / unit_var)
    norm_const = beta / (2 * scale * gamma(1.0 / beta))
    return norm_const * np.exp(-(np.abs(theta_grid - theta_lam) / scale) ** beta)


WEIGHT_FUNCTIONS = {
    "normal": normal_weight,
    "uniform": uniform_weight,
    "fat_tailed": fat_tailed_weight,
    "thin_tailed": thin_tailed_weight,
}


# ------------------------------------------------------------------------ steps 6-7 ------


def theta_lam_variance(S_Y, theta_grid, idx, lam, sigma2, n):
    """Variance V(lambda) of theta_lambda (paper section 3.2, steps 6 and 6'),

        V(lambda) = (sigma^2 / n) s_Y''(theta_lambda) / {s_Y''(theta_lambda) + lambda}^2,

    where s_Y = S_Y / n is the *per-observation* prediction error. Returns V (float), or NaN
    where the curvature is non-positive and the expression is degenerate.

    Steps 5 and 5' display S_Y as a sum over i, but equation (3.4) writes the penalty as
    n*lambda*theta^2, "since both S and S_1 are of order n". Step 6 then pairs S_Y with a bare
    lambda in both `S_Y + lambda theta^2` and `(S_Y'' + lambda)^2`, which is the same criterion
    only when S_Y there means S_Y / n. What lambda can do to V is the ratio lambda / S_Y'';
    taken as a sum, that ratio is n times smaller, V collapses to order sigma^2 / n^2 -- not
    the order of a variance -- and the step-7 density is narrower than the theta grid by
    sqrt(n).

    PARAMETERS
    S_Y   : (Q,) array, the summed prediction error over `theta_grid`;
    idx   : int, index of theta_lambda in `theta_grid`;
    n     : int, sample size, which converts S_Y to its per-observation form.
    """
    j = min(max(int(idx), 1), len(theta_grid) - 2)  # force evaluation at interior points
    h = theta_grid[1] - theta_grid[0]
    s_Y2 = (S_Y[j + 1] - 2 * S_Y[j] + S_Y[j - 1]) / (n * h ** 2)
    if s_Y2 <= 0:
        return np.nan
    return (sigma2 / n) * s_Y2 / (s_Y2 + lam) ** 2


_STEP7_PER_SD = 8      # integration points per standard deviation of the weight
_STEP7_N_SD = 6.0      # half-width of the integration range, in standard deviations


def weighted_S_b(S_b, theta_grid, theta_lam, V, w):
    """Step-7 criterion P_b(lambda) = int S_b(theta) w(theta; theta_lambda, V) dtheta.

    The weight has standard deviation sqrt(V), which is finer than the spacing of
    `theta_grid`, so the integral is taken on a grid refined to `_STEP7_PER_SD` points per
    standard deviation with S_b interpolated linearly onto it. Returns P_b (float).

    Integrating on `theta_grid` itself is exact up to aliasing, and by Poisson summation
    overstates the integral by sum_m exp(-2 pi^2 m^2 V / h^2) for h the grid spacing: harmless
    at 1.4% while sqrt(V) is half a grid step, but tending to h / sqrt(2 pi V) once
    sqrt(V) << h, where the rule returns the single sample h S_b(theta_lambda) / sqrt(2 pi V)
    and P_b stops being an average of S_b at all. Refining removes the dependence on V being
    resolvable by a grid chosen for something else.

    PARAMETERS
    w : callable, the weighting density, called as `w(theta, theta_lam, V)`.
    """
    sd = np.sqrt(V)
    lo = max(theta_grid[0], theta_lam - _STEP7_N_SD * sd)
    hi = min(theta_grid[-1], theta_lam + _STEP7_N_SD * sd)
    n_points = int(np.ceil((hi - lo) * _STEP7_PER_SD / sd)) + 1
    fine = np.linspace(lo, hi, n_points)
    return float(np.trapezoid(np.interp(fine, theta_grid, S_b) * w(fine, theta_lam, V), fine))


# -------------------------------------------------------------------------- steps 1-3 ----


def pilot_fpca(X, Y, grid, n_basis=20, n_components=10, m_range=None):
    """Untruncated estimate of (a, b) with a BIC-selected component count (paper step 1).

    The curves are represented in a B-spline basis, Y is regressed on the leading m FPC
    scores, and b is mapped back onto `grid`. Returns a_pilot (float), b_pilot (array) and the
    retained components (dict of phi, tau, mean, m).

    PARAMETERS
    n_basis      : int, number of components in B-spline;
    n_components : int, max number of FPCs;
    m_range      : array, candidates for m, defaults to [2, ..., n_components].
    """
    basis = BSplineBasis(n_basis=n_basis, domain_range=(grid[0], grid[-1]))  # B spline of order 4
    fd = FDataGrid(X, grid).to_basis(basis)  # Represent discretised functional data as basis

    fpca = FPCA(n_components=n_components)
    scores_all = fpca.fit_transform(fd)  # (n, n_components) matrix of each curve's FPC scores
    comps_all = fpca.components_(grid).squeeze()  # Discretised FPCs on grid
    tau_all = fpca.explained_variance_  # FPC eigenvalues
    X_bar = X.mean(axis=0)
    n = len(Y)

    if m_range is None:
        m_range = range(2, n_components + 1)

    # BIC over number of retained components
    best_m, best_bic = None, np.inf
    for m in m_range:
        lin = LinearRegression().fit(scores_all[:, :m], Y)
        b = lin.coef_ @ comps_all[:m]
        a = Y.mean() - np.trapezoid(b * X_bar, grid)
        pred = a + np.trapezoid(b[None, :] * X, grid, axis=1)
        rss = np.sum((Y - pred) ** 2)
        bic = n * np.log(rss / n) + (m + 1) * np.log(n)
        if bic < best_bic:
            best_bic, best_m = bic, m

    lin = LinearRegression().fit(scores_all[:, :best_m], Y)
    b_pilot = lin.coef_ @ comps_all[:best_m]
    a_pilot = Y.mean() - np.trapezoid(b_pilot * X_bar, grid)

    fpc = {"phi": comps_all[:best_m],
           "tau": tau_all[:best_m],
           "mean": X_bar,
           "m": best_m}
    return a_pilot, b_pilot, fpc


def fit_b_simp(X, Y, grid, a_pilot, n_fourier=1, theta_bar=None, theta_grid=None):
    """Parametric target b_simp(t) = {c0 + c1 sin(2k pi t) + c2 cos(2k pi t)} I(t < theta_bar)
    of paper step 3, with (c0, c1, c2) fitted by unpenalised least squares.

    Returns b_simp (array) and theta_bar (float).

    PARAMETERS
    n_fourier  : int, number of sin/cos pairs added to the constant term. The paper's k=1,2,3
                 (section 3.2) are alternative single-frequency specifications it tried, not a
                 sum over k=1..3; n_fourier=1 matches that (one sin/cos pair, k=1);
    theta_bar  : float, if given, held fixed and only the coefficients are estimated. If None
                 (default), estimated jointly with the coefficients by grid search;
    theta_grid : array, candidates for theta_bar when it is being estimated; defaults to
                 grid's interior points.
    """
    cols = [np.ones_like(grid)]
    for k in range(1, n_fourier + 1):
        cols += [np.sin(2 * k * np.pi * grid), np.cos(2 * k * np.pi * grid)]
    Psi = np.column_stack(cols).T  # (p, N)
    Y_centred = Y - a_pilot

    def rss_and_coef(theta):
        interval = grid <= theta
        grid_i = grid[interval]
        scores = np.trapezoid(
            Psi[:, interval][:, None, :] * X[None, :, interval], grid_i, axis=2
        )  # (p, n)
        coef, *_ = np.linalg.lstsq(scores.T, Y_centred, rcond=None)
        rss = np.sum((Y_centred - scores.T @ coef) ** 2)
        return rss, coef

    if theta_bar is None:
        if theta_grid is None:
            theta_grid = grid[(grid > grid[0]) & (grid < grid[-1])]
        rss_curve = np.array([rss_and_coef(theta)[0] for theta in theta_grid])
        theta_bar = theta_grid[np.argmin(rss_curve)]

    _, coef = rss_and_coef(theta_bar)
    interval = grid <= theta_bar
    b_simp = np.zeros_like(grid)
    b_simp[interval] = Psi[:, interval].T @ coef
    return b_simp, theta_bar


# ------------------------------------------------------------- truncated-domain FPCA -----


def _top_components(A, k, vectors=True):
    """The `k` largest squared singular values of `A` (n, p), descending, and the matching
    right singular vectors as rows. Returns s2 (k',) and Vt (k', p) or None, k' = min(k, n, p).

    Both truncFLR theta scans decompose the curves once per candidate theta, about 800 times per
    fit, so this is where method A spends its time. With more curves than grid points the
    decomposition is taken from the p x p Gram matrix `A'A`, whose eigenpairs are the squared
    singular values and the right singular vectors, at a cost of n p^2 for the product and p^3
    for the eigensolver rather than an n p^2 SVD with a large constant, and only the `k` leading
    pairs are formed. Otherwise, or where a wanted eigenvalue is too small for the Gram matrix
    to resolve, it is the SVD. The two agree to rounding, which
    `tests/test_hh_a.py` checks along with every fit built on them; a singular vector is only
    defined up to sign, and nothing downstream depends on the sign.

    PARAMETERS
    A       : (n, p) the centred, square-root-weighted curves;
    k       : int, the number of leading components wanted;
    vectors : bool, also return the right singular vectors.
    """
    n, p = A.shape
    k = min(int(k), n, p)
    if n > p:
        G = A.T @ A
        if vectors:
            vals, vecs = eigh(G, subset_by_index=[p - k, p - 1])
        else:
            vals, vecs = eigh(G, eigvals_only=True, subset_by_index=[p - k, p - 1]), None
        # Below p eps lambda_max an eigenvalue of the Gram matrix is rounding; the SVD resolves
        # such components (a sub-grid holding X(0) = 0 has one) and the Gram matrix cannot.
        if vals[0] > p * np.finfo(float).eps * vals[-1]:
            return vals[::-1], (vecs[:, ::-1].T if vectors else None)
    if not vectors:
        return np.linalg.svd(A, compute_uv=False)[:k] ** 2, None
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    return s[:k] ** 2, Vt[:k]


def trunc_fpca(X, grid, interval, m):
    """Empirical FPCs phi_j^theta and eigenvalues tau_j^theta of the covariance operator of X
    restricted to `interval` (the truncated-domain components of section 3.1.1, step 4).

    Centred, trapezoid-weighted curves are decomposed by one SVD, so the right singular vectors
    are orthonormal in the L2([0, theta]) inner product and the squared singular values over n
    are the eigenvalues.

    Returns phi_theta (k, N_i) and tau_theta (k,), where N_i = interval.sum() and k = min(m, rank).

    PARAMETERS
    interval : (N,) boolean array, True on the truncated domain [0, theta];
    m        : int, number of leading components requested.
    """
    grid_i = grid[interval]
    w = quadrature_weights(grid_i)

    Xc = X[:, interval] - X[:, interval].mean(axis=0)
    n = Xc.shape[0]
    sqrt_w = np.sqrt(w)
    s2, Vt = _top_components(Xc * sqrt_w[None, :], m)

    phi_theta = Vt / sqrt_w[None, :]  # unweight so phi are orthonormal under trapezoid L2
    tau_theta = s2 / n
    return phi_theta, tau_theta


def floor_eigvals(tau, m):
    """Pad `tau` to length m at machine epsilon, so 1/tau blows up on the components a short
    sub-grid cannot support and those theta are penalised. Returns a (m,) array."""
    if len(tau) < m:
        tau = np.concatenate([tau, np.full(m - len(tau), np.finfo(float).eps)])
    return tau[:m]


def trunc_eigvals(X, grid, interval, m):
    """Top-m eigenvalues tau_j^theta of the empirical covariance operator of X restricted to
    `interval`, needed for S_b's variance term (paper section 3.2, steps 5 and 5').

    The same decomposition as `trunc_fpca`, but the singular vectors are not formed, since
    only the eigenvalues are wanted here and both lambda_select criteria call this once per
    candidate theta. Components the sub-grid cannot support are floored at machine epsilon,
    so 1/tau blows up and penalises such theta.

    PARAMETERS
    interval : (N,) boolean array, True on the truncated domain [0, theta];
    m        : int, number of leading eigenvalues to return.
    """
    grid_i = grid[interval]
    Xc = X[:, interval] - X[:, interval].mean(axis=0)
    s2, _ = _top_components(Xc * np.sqrt(quadrature_weights(grid_i))[None, :], m,
                            vectors=False)
    return floor_eigvals(s2 / Xc.shape[0], m)
