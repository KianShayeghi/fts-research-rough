"""Data-generating process for a functional autoregressive process of order 1 (FAR(1)),
used to relax the i.i.d.-curves assumption made by both `truncFLR.HH_dgp` and `fdaSP.SBC_sim`.

    X_t(s) = ∫_0^1 psi(s, r) X_{t-1}(r) dr + eps_t(s),   t = 1, ..., n

eps_t are i.i.d. Gaussian-process innovations. Two kernels are supplied, both rescaled to a
target Hilbert-Schmidt norm `rho < 1`, which is sufficient for the process to admit a strictly
stationary, causal solution:

    gaussian_kernel     psi(s, r) = exp{-(s^2 + r^2) / 2}    (Bosq, 2000, Ch. 3)
    exponential_kernel  psi(s, r) = exp(-|s - r| / scale)

The first factorises as exp(-s^2/2) exp(-r^2/2), so the operator has rank one: it
autocorrelates a single direction in L2 and leaves every orthogonal direction i.i.d. across t.
The second has full rank, with eigenvalues decaying like j^-2, so the dependence is spread
across the whole space. Both estimators (`truncFLR.HH`, `fdaSP.SBC`) can be applied unmodified.
"""

import numpy as np

from ..core.gp import sample_gp
from ..core.quadrature import trunc_int
from ..fdaSP.SBC_sim import psi1, psi2, psi3  # noqa: F401 (re-exported for convenience)

__all__ = [
    "gaussian_kernel", "exponential_kernel", "hs_norm", "operator_norm", "kernel_rank",
    "sample_far1", "sim_dataset_window", "sim_dataset_psi",
    "lag1_autocorr", "psi1", "psi2", "psi3",
]


def _trapezoid_weights(grid):
    """Trapezium quadrature weights on `grid`, so `Psi @ x` approximates `int psi(s,r) x(r) dr`.
    """
    w = np.zeros_like(grid)
    w[1:] += np.diff(grid)
    w[:-1] += np.diff(grid)
    w *= 0.5
    return w


def _rescale(K, grid, rho):
    """Rescale kernel values `K` to Hilbert-Schmidt norm `rho` and fold in the quadrature
    weights, so `Psi @ x` approximates `int psi(s, r) x(r) dr`. Returns Psi (n_grid, n_grid)."""
    w = _trapezoid_weights(grid)
    norm = np.sqrt(np.sum((K ** 2) * w[None, :] * w[:, None]))
    return K * (rho / norm) * w[None, :]


def gaussian_kernel(grid, rho=0.7):
    """Discretised, quadrature-weighted FAR(1) operator, from the kernel
    psi(s, r) = exp{-(s^2 + r^2) / 2} (Bosq, 2000, Ch. 3's simulation example), rescaled so its
    Hilbert-Schmidt norm equals `rho`. Returns Psi (n_grid, n_grid) array such that
    `Psi @ x` approximates `int_0^1 psi(s, r) x(r) dr` for `x` sampled on `grid`.

    The kernel factorises as exp(-s^2/2) exp(-r^2/2), so the operator has rank one and its
    Hilbert-Schmidt and operator norms coincide. Only the component of X_{t-1} along
    exp(-r^2/2) carries forward, so the serial dependence lives in a one-dimensional subspace;
    `exponential_kernel` spreads it across the whole space.

    PARAMETERS
    rho : float, target Hilbert-Schmidt norm. rho < 1 is sufficient for a strictly stationary,
          causal FAR(1) solution to exist.
    """
    S, R = np.meshgrid(grid, grid, indexing="ij")
    return _rescale(np.exp(-(S ** 2 + R ** 2) / 2.0), grid, rho)


def exponential_kernel(grid, rho=0.7, scale=0.25):
    """Discretised, quadrature-weighted FAR(1) operator from the kernel
    psi(s, r) = exp(-|s - r| / scale), rescaled so its Hilbert-Schmidt norm equals `rho`.
    Returns Psi (n_grid, n_grid), as `gaussian_kernel` does.

    This kernel does not factorise: it has full rank, with eigenvalues decaying like j^-2, so
    every direction in L2 is autocorrelated and the operator norm is strictly below the
    Hilbert-Schmidt norm. `scale` sets how far along the domain the dependence reaches, with
    small values approaching a pointwise autoregression.

    PARAMETERS
    rho   : float, target Hilbert-Schmidt norm;
    scale : float, correlation length of the kernel in units of the domain.
    """
    S, R = np.meshgrid(grid, grid, indexing="ij")
    return _rescale(np.exp(-np.abs(S - R) / scale), grid, rho)


def _symmetrised(Psi, grid):
    """sqrt(w) K sqrt(w) for a quadrature-weighted `Psi`, whose singular values are those of
    the integral operator. Returns an (n_grid, n_grid) array."""
    w = _trapezoid_weights(grid)
    sw = np.sqrt(w)
    return sw[:, None] * (Psi / w[None, :]) * sw[None, :]


def hs_norm(Psi, grid):
    """Hilbert-Schmidt norm of the integral operator a quadrature-weighted `Psi` discretises.
    Returns a float."""
    return float(np.linalg.norm(_symmetrised(Psi, grid)))


def operator_norm(Psi, grid):
    """L2 operator norm of the integral operator a quadrature-weighted `Psi` discretises.

    The largest singular value of `Psi` itself is not this norm: the symmetric weighting
    sqrt(w) K sqrt(w) is the one whose singular values are those of the operator. Returns a
    float, never above `hs_norm`, with equality exactly when the kernel has rank one.
    """
    return float(np.linalg.svd(_symmetrised(Psi, grid), compute_uv=False)[0])


def kernel_rank(Psi, grid, tol=1e-10):
    """Numerical rank of the integral operator a quadrature-weighted `Psi` discretises.
    Returns an int: 1 for `gaussian_kernel`, full for `exponential_kernel`."""
    return int(np.linalg.matrix_rank(_symmetrised(Psi, grid), tol=tol))


def sample_far1(n, grid, Psi, burn_in=200, n_basis=25, decay=4.0, variance=1.0, rng=None):
    """Simulate n consecutive curves from a FAR(1) process with operator `Psi` (see
    `gaussian_kernel`), discretised as X_t = Psi @ X_{t-1} + eps_t.

    Returns (n, n_grid) array of curves, in time order (row 0 is the earliest).

    PARAMETERS
    Psi     : (n_grid, n_grid) array, discretised FAR(1) operator, see `gaussian_kernel`;
    burn_in : int, number of initial (discarded) steps run before the returned curves;
    n_basis, decay, variance : passed through to `sample_gp` for the innovations eps_t.
    """
    rng = np.random.default_rng(rng)
    eps = sample_gp(n + burn_in, grid, n_basis=n_basis, decay=decay, variance=variance, rng=rng)

    X = np.empty_like(eps)
    prev = np.zeros(len(grid))
    for t in range(n + burn_in):
        prev = Psi @ prev + eps[t]
        X[t] = prev
    return X[burn_in:]


def sim_dataset_window(
    n=200, grid=None, u=0.8, v=1.0, snr=26.25, rho=0.7, burn_in=200,
    n_basis=25, decay=4.0, variance=1.0, a=0.0, beta_fn=None, kernel=None, rng=None,
):
    """FAR(1) analogue of `truncFLR.HH_dgp.sim_dataset`. Returns the same dictionary shape as
    before, with the operator added under "Psi".

    PARAMETERS
    grid   : array or None, defaults to 501 points on [0, 1];
    rho    : float, FAR(1) dependence strength, see `gaussian_kernel`;
    kernel : callable or None, builds the operator as `kernel(grid, rho=rho)`; defaults to
             `gaussian_kernel`.
    """
    rng = np.random.default_rng(rng)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 501)

    Psi = (gaussian_kernel if kernel is None else kernel)(grid, rho=rho)
    X = sample_far1(
        n, grid, Psi, burn_in=burn_in, n_basis=n_basis, decay=decay, variance=variance, rng=rng
    )

    if beta_fn is None:
        beta_true = ((grid >= u) & (grid <= v)).astype(float)
        signal = trunc_int(X, grid, u=u, v=v)
        sigma = np.sqrt(np.var(signal) / snr)
    else:
        beta_true = np.asarray(beta_fn(grid), dtype=float)
        signal = trunc_int(X, grid, beta=beta_true)
        sigma = np.sqrt(np.var(signal) / snr)

    eps = rng.normal(0.0, sigma, size=n)
    Y = a + signal + eps
    return {"grid": grid, "X": X, "Y": Y, "u": u, "v": v,
            "sigma": sigma, "a": a, "b": beta_true, "rho": rho, "Psi": Psi}


def sim_dataset_psi(
    n=50, grid=None, psi_fn=psi1, rho=0.7, snr=4.0, burn_in=200,
    n_basis=25, decay=4.0, variance=1.0, kernel=None, rng=None,
):
    """FAR(1) analogue of `fdaSP.SBC_sim.simulate`.

    PARAMETERS
    grid   : array or None, defaults to 200 points on [0, 1] (matches `SBC_sim.simulate`'s
             default resolution);
    psi_fn : callable, true coefficient function;
    rho    : float, FAR(1) dependence strength;
    snr    : float, signal-to-noise ratio;
    kernel : callable or None, builds the operator as `kernel(grid, rho=rho)`; defaults to
             `gaussian_kernel`.
    """
    rng = np.random.default_rng(rng)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 200)

    Psi = (gaussian_kernel if kernel is None else kernel)(grid, rho=rho)
    X = sample_far1(
        n, grid, Psi, burn_in=burn_in, n_basis=n_basis, decay=decay, variance=variance, rng=rng
    )

    beta_true = psi_fn(grid)
    signal = np.trapezoid(X * beta_true, grid, axis=1)
    sigma = np.sqrt(np.var(signal) / snr)
    y = signal + sigma * rng.standard_normal(n)
    return X, y, grid, beta_true


def lag1_autocorr(x):
    """Lag-1 sample autocorrelation of a scalar time series"""
    x = np.asarray(x, dtype=float)
    x0, x1 = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    return float(np.sum(x0 * x1) / np.sqrt(np.sum(x0 ** 2) * np.sum(x1 ** 2)))
