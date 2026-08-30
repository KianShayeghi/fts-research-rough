"""Matched-spectrum designs for comparing the three estimators across bases and targets.

The curves are drawn as a random expansion in an orthonormal basis,

    X_i(t) = sum_{k=1}^{K} c_ik e_k(t),    c_ik ~ N(0, v_k) independently,

so the covariance operator of X is sum_k v_k e_k (x) e_k and its eigenvalues are exactly the
prescribed v_k, whatever the basis. Fixing `variance_profile` and varying only the `e_k`
therefore changes the *directions* the predictor varies in without touching how fast that
variation decays, which is what makes an integrated squared error comparable across the three
bases. `raw_basis` returns each family unmodified -- the trigonometric one is
`truncFLR.HH_dgp`'s, the B-spline one is `fdaSP.SBC`'s -- and `basis` puts it on that common
footing, orthonormalising it and then ordering it from smoothest to roughest so that the index
the variance profile is attached to means the same thing in all three.

Three coefficient functions are supplied: a jump (`psi_step`), a smooth rise off zero
(`psi_cdf`), and two signed bumps separated by a null gap (`psi_multi`). The first two are
supported on a single interval [u, 1], which is the model `truncFLR` assumes; the third is
not, and no orientation of that model describes it.

Only the projection of psi onto the span of the basis is identified, since X lives in that
span and the part of psi orthogonal to it never reaches the response. `projection_gap`
reports how much of psi that is.
"""

import numpy as np
from numpy.polynomial import legendre
from scipy.linalg import cholesky, solve_triangular

from fdaSP.SBC import bspline_basis
from truncFLR.HH import trunc_eigvals
from truncFLR.HH_dgp import trig_basis

__all__ = [
    "BASIS_KINDS", "TARGETS", "raw_basis", "gram", "roughness_gram", "orthonormalise",
    "basis", "variance_profile", "sample_curves", "project", "projection_gap", "spectrum",
    "psi_step", "psi_cdf", "psi_multi", "simulate", "quad_weights", "pointwise_variance",
]

BASIS_KINDS = ("trig", "legendre", "bspline")


def quad_weights(grid):
    """Trapezium weights w on `grid`, so `w @ f` approximates `int f`. Returns (N,).

    The same rule `truncFLR._common.quadrature_weights` and `fdaSP.SBC.bspline_design` use, so
    every inner product in this comparison is taken under one quadrature.
    """
    grid = np.asarray(grid, dtype=float)
    w = np.zeros_like(grid)
    w[1:] += np.diff(grid)
    w[:-1] += np.diff(grid)
    return 0.5 * w


# ------------------------------------------------------------------------------- bases ---


def raw_basis(grid, kind="trig", n_basis=25):
    """The `n_basis` leading functions of one family, evaluated on `grid` and not yet
    orthonormalised. Returns (n_basis, N).

    The three families are deliberately borrowed rather than reinvented: `trig` is the
    expansion Hall & Hooker (2016) generate their curves from (`truncFLR.HH_dgp.trig_basis`),
    `bspline` is the one `fdaSP.SBC` expands the coefficient function in, and `legendre` --
    the shifted Legendre polynomials P_k(2t - 1) -- belongs to neither method, so it is the
    design on which neither is at home.

    PARAMETERS
    kind : str, one of `BASIS_KINDS`.
    """
    grid = np.asarray(grid, dtype=float)
    if kind == "trig":
        return trig_basis(grid, n_basis=n_basis)
    if kind == "legendre":
        x = 2.0 * grid - 1.0
        Psi = np.empty((n_basis, grid.size))
        for k in range(n_basis):
            c = np.zeros(k + 1)
            c[k] = 1.0
            Psi[k] = legendre.legval(x, c)
        return Psi
    if kind == "bspline":
        return bspline_basis(grid, n_basis=n_basis, order=4).T
    raise ValueError(f"unknown basis kind {kind!r}; known: {list(BASIS_KINDS)}.")


def gram(Psi, grid):
    """Gram matrix G_jk = int e_j e_k under the trapezium rule. Returns (K, K)."""
    Psi = np.atleast_2d(np.asarray(Psi, dtype=float))
    return (Psi * quad_weights(grid)[None, :]) @ Psi.T


def orthonormalise(Psi, grid):
    """Orthonormalise the rows of `Psi` in L2(`grid`), by Cholesky of their Gram matrix.

    With G = L L^T, the rows of L^-1 Psi satisfy (L^-1 Psi) W (L^-1 Psi)^T = I for W the
    trapezium weights, so the transformed rows span exactly what the originals did and differ
    from them by an upper-triangular change of coordinates. Where the family is already
    orthogonal -- the trigonometric one is, exactly, since the trapezium rule is exact on
    trigonometric polynomials sampled over whole periods -- L is diagonal and this only
    rescales. Returns (K, N).
    """
    Psi = np.atleast_2d(np.asarray(Psi, dtype=float))
    L = cholesky(gram(Psi, grid), lower=True)
    return solve_triangular(L, Psi, lower=True)


def roughness_gram(Psi, grid):
    """Roughness matrix R_jk = int e_j' e_k', by central differences on `grid`. Returns (K, K).

    Symmetric and positive semi-definite, with a null direction for every constant in the
    span. It is what `basis` diagonalises to order a family from smoothest to roughest.
    """
    Psi = np.atleast_2d(np.asarray(Psi, dtype=float))
    grid = np.asarray(grid, dtype=float)
    D = np.gradient(Psi, grid, axis=1)
    R = (D * quad_weights(grid)[None, :]) @ D.T
    return 0.5 * (R + R.T)


def basis(grid, kind="trig", n_basis=25):
    """The orthonormal system spanning one family, ordered from smoothest to roughest.
    Returns (n_basis, N).

    `orthonormalise` fixes the inner product but not the order, and the order is what the
    variance profile is attached to, so it has to mean the same thing in all three families.
    Index order does not: the shifted Legendre polynomials run from constant to degree K - 1
    and the trigonometric functions from constant to the highest frequency, both orderings by
    oscillation, but a B-spline basis runs left to right across the domain, so attaching a
    decaying variance to it would pile the curves' variation at one end and quietly hand the
    truncation model an orientation.

    Diagonalising `roughness_gram` inside the span removes that: the rows returned are the
    Demmler-Reinsch system of the family, orthonormal in L2 and ordered by int e' e. For the
    trigonometric and Legendre families this is the ordering they already had, since both are
    ordered by oscillation to begin with; for the B-spline family it replaces position by
    smoothness and leaves the design spatially even, which `pointwise_variance` shows.

    PARAMETERS
    kind : str, one of `BASIS_KINDS`.
    """
    E = orthonormalise(raw_basis(grid, kind=kind, n_basis=n_basis), grid)
    _, vecs = np.linalg.eigh(roughness_gram(E, grid))
    out = vecs.T @ E
    lead = np.argmax(np.abs(out), axis=1)
    signs = np.sign(out[np.arange(out.shape[0]), lead])
    return out * np.where(signs == 0, 1.0, signs)[:, None]


def pointwise_variance(Psi, var):
    """Var{X(t)} = sum_k v_k e_k(t)^2 for the curves `sample_curves` draws. Returns (N,).

    Flat-ish and symmetric where the design treats the domain evenly, and lopsided where it
    does not, which is what the ordering in `basis` is there to prevent.

    PARAMETERS
    Psi : (K, N) array, the orthonormal basis;
    var : (K,) array of coefficient variances, as from `variance_profile`.
    """
    Psi = np.atleast_2d(np.asarray(Psi, dtype=float))
    return np.asarray(var, dtype=float) @ (Psi ** 2)


def variance_profile(n_basis=25, decay=6.0):
    """Coefficient variances v_k = exp{-(k - 1) / decay}, the profile Hall & Hooker (2016)
    Section 4 generate their curves with. Returns (n_basis,).

    Held fixed across the three bases, these are the eigenvalues of the covariance operator of
    the curves `sample_curves` draws, so the designs differ only in their eigenfunctions.

    PARAMETERS
    decay : float, rate of exponential decay across the basis; larger keeps more directions
            informative.
    """
    return np.exp(-np.arange(int(n_basis)) / float(decay))


def sample_curves(n, Psi, var, rng=None):
    """Draw `n` independent curves sum_k c_k e_k with c_k ~ N(0, var_k). Returns (n, N).

    PARAMETERS
    Psi : (K, N) array, the orthonormalised basis;
    var : (K,) array of coefficient variances, as from `variance_profile`;
    rng : int or Generator, seed for reproducibility.
    """
    rng = np.random.default_rng(rng)
    Psi = np.atleast_2d(np.asarray(Psi, dtype=float))
    var = np.asarray(var, dtype=float)
    coef = rng.standard_normal((int(n), Psi.shape[0])) * np.sqrt(var)[None, :]
    return coef @ Psi


def project(f, Psi, grid):
    """Orthogonal projection of `f` onto the span of the rows of `Psi`, under the trapezium
    inner product. Returns the coefficients (K,) and the projection on `grid` (N,).

    `Psi` is assumed orthonormal, as `basis` returns it, so the coefficients are the plain
    inner products int f e_k.
    """
    f = np.asarray(f, dtype=float)
    Psi = np.atleast_2d(np.asarray(Psi, dtype=float))
    coef = (Psi * quad_weights(grid)[None, :]) @ f
    return coef, coef @ Psi


def projection_gap(f, Psi, grid):
    """Squared distance int (f - P f)^2 from `f` to the span of `Psi`. Returns a float.

    X lives in that span, so the response depends on psi only through P psi and the part
    measured here reaches no estimator at any sample size. It is not a floor on the realised
    error: an estimate is written in the *estimator's* basis, not the design's, and where that
    basis happens to suit psi it can supply the missing part for free. What the gap measures
    is how much of psi is prior information rather than evidence on this design, which is why
    the error against psi and the error against P psi are both worth reporting.
    """
    _, proj = project(f, Psi, grid)
    return float(quad_weights(grid) @ (np.asarray(f, dtype=float) - proj) ** 2)


def spectrum(X, grid, k=None):
    """Leading `k` eigenvalues of the empirical covariance operator of `X`, under the trapezium
    inner product. Returns (k,).

    `truncFLR.HH.trunc_eigvals` on the whole domain, so the design check and the estimators
    read the spectrum the same way.

    PARAMETERS
    k : int or None, how many eigenvalues to return; defaults to the sample size.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    grid = np.asarray(grid, dtype=float)
    k = X.shape[0] if k is None else int(k)
    return trunc_eigvals(X, grid, np.ones(grid.size, dtype=bool), k)


# --------------------------------------------------------------------- coefficient functions


def psi_step(t, u=0.6):
    """Indicator 1{t >= u}: zero on [0, u), one above it. Returns an array shaped like `t`.

    A jump at u, so the boundary the truncation model looks for is as sharp as it gets, and
    the least kind case for a smooth basis to represent.
    """
    return (np.asarray(t, dtype=float) >= u).astype(float)


def psi_cdf(t, u=0.8):
    """Zero on [0, u), then the Beta(2, 2) distribution function rescaled to [u, 1], rising
    smoothly from 0 at u to 1 at t = 1. Returns an array shaped like `t`.

    With s = (t - u) / (1 - u) the rise is 3s^2 - 2s^3, which meets zero with zero derivative
    at u, so unlike `psi_step` the coefficient leaves the axis smoothly and the endpoint is
    correspondingly harder to locate.
    """
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t >= u
    s = (t[m] - u) / (1.0 - u)
    out[m] = s ** 2 * (3.0 - 2.0 * s)
    return out


def psi_multi(t):
    """Raised-cosine bumps of opposite sign on [0.1, 0.3] and [0.55, 0.8], zero elsewhere.
    Returns an array shaped like `t`.

    The null set is the three disjoint intervals [0, 0.1), (0.3, 0.55) and (0.8, 1], so no
    single interval carries the signal and neither orientation of the truncation model is
    correctly specified. The signs are opposite so that an estimator which spans the gap
    rather than zeroing it cancels signal against signal rather than merely adding a little
    bias.
    """
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for lo, hi, sign in [(0.1, 0.3, 1.0), (0.55, 0.8, -1.0)]:
        m = (t >= lo) & (t <= hi)
        out[m] = sign * 0.5 * (1.0 - np.cos(2 * np.pi * (t[m] - lo) / (hi - lo)))
    return out


# `window` is the single interval the truncation model would have to find, and None where the
# target has no such interval. `null` marks the true null set T0 on a grid.
# `null` is the closed set {t : psi(t) = 0}, so its length is exactly 0.6, 0.8 and 0.55. Both
# smooth targets meet the axis *at* their boundary, so the boundary point belongs to the null
# set; only the step, which jumps, does not.
TARGETS = {
    "step": {"fn": psi_step,
             "null": lambda g: np.asarray(g) < 0.6,
             "window": (0.6, 1.0),
             "label": "step at 0.6"},
    "cdf": {"fn": psi_cdf,
            "null": lambda g: np.asarray(g) <= 0.8,
            "window": (0.8, 1.0),
            "label": "smooth rise from 0.8"},
    "multi": {"fn": psi_multi,
              "null": lambda g: (np.asarray(g) <= 0.1)
              | ((np.asarray(g) >= 0.3) & (np.asarray(g) <= 0.55))
              | (np.asarray(g) >= 0.8),
              "window": None,
              "label": "two signed bumps"},
}


# ------------------------------------------------------------------------------- the DGP ---


def simulate(kind="trig", target="step", n=200, n_grid=201, n_basis=25, decay=6.0,
             snr=20.0, n_test=500, rng=None):
    """One dataset from the matched-spectrum design: curves in basis `kind`, response driven by
    coefficient function `target`.

    The response is Y_i = int psi X_i + eps_i with eps_i ~ N(0, sigma^2) and sigma set from
    `snr`, exactly as `truncFLR.HH_dgp.sim_dataset` and `fdaSP.SBC_sim.simulate` set theirs, so
    the noise level is matched across every cell of the comparison. A further `n_test` curves
    are drawn from the same design and given responses at the same sigma, for out-of-sample
    error.

    Returns a dict carrying `grid`, the orthonormal basis `Psi` and its variances `var`, the
    training `X`/`Y` and test `X_test`/`Y_test`, the true `psi` and its null mask `null`, the
    noise level `sigma`, the identified part `psi_proj` of psi with the `gap` left over, and
    the scales `scale` = int psi^2 and `scale_proj` = int (P psi)^2 that relative errors
    against each are divided by.

    PARAMETERS
    kind    : str, one of `BASIS_KINDS`;
    target  : str, one of `TARGETS`;
    n_basis : int, number of basis functions the curves are expanded in;
    decay   : float, passed to `variance_profile`;
    snr     : float, signal-to-noise ratio var(signal) / sigma^2;
    n_test  : int, size of the held-out sample drawn from the same design;
    rng     : int or Generator, seed for reproducibility.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; known: {sorted(TARGETS)}.")
    rng = np.random.default_rng(rng)
    grid = np.linspace(0.0, 1.0, int(n_grid))
    Psi = basis(grid, kind=kind, n_basis=n_basis)
    var = variance_profile(n_basis=n_basis, decay=decay)

    X = sample_curves(n, Psi, var, rng=rng)
    X_test = sample_curves(n_test, Psi, var, rng=rng)

    spec = TARGETS[target]
    psi = spec["fn"](grid)
    w = quad_weights(grid)
    signal = (X * w[None, :]) @ psi
    sigma = float(np.sqrt(np.var(signal) / snr))
    Y = signal + sigma * rng.standard_normal(X.shape[0])
    Y_test = (X_test * w[None, :]) @ psi + sigma * rng.standard_normal(X_test.shape[0])

    coef, proj = project(psi, Psi, grid)
    return {"grid": grid, "Psi": Psi, "var": var, "X": X, "Y": Y,
            "X_test": X_test, "Y_test": Y_test, "psi": psi,
            "null": spec["null"](grid), "window": spec["window"], "sigma": sigma,
            "psi_proj": proj, "psi_coef": coef,
            "gap": float(w @ (psi - proj) ** 2), "scale": float(w @ psi ** 2),
            "scale_proj": float(w @ proj ** 2), "kind": kind, "target": target}
