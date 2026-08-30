"""Data-generating process for truncated functional linear models, introduced in
Hall & Hooker, 2016."""

import numpy as np


def trig_basis(grid, n_basis=25):
    """Trigonometric basis from Hall & Hooker (2016), Section 4: psi_1 = 1, psi_2k = sin(2k*pi*t),
    psi_{2k+1} = cos(2k*pi*t). Returns (n_basis, N) array of basis functions evaluated on `grid`.

    PARAMETERS
    n_basis : int, number of basis functions to return (paper uses the first 25).
    """
    Psi = np.empty((n_basis, len(grid)))
    for idx in range(n_basis):
        paper_idx = idx + 1
        if paper_idx == 1:
            Psi[idx] = 1.0
        elif paper_idx % 2 == 0:
            Psi[idx] = np.sin(paper_idx * np.pi * grid)
        else:
            Psi[idx] = np.cos((paper_idx - 1) * np.pi * grid)
    return Psi


def sample_gp(n, grid, mean=None, n_basis=25, decay=4.0, variance=1.0, rng=None):
    """Draw n iid curves as a random trigonometric-basis expansion, as in Hall & Hooker (2016),
    Section 4: X(t) = sum_{k=1}^{n_basis} c_k psi_k(t), with
    c_k ~ N(0, variance * exp{-(k-1)/decay}) drawn independently. Returns each realisation
    at each grid point (array).

    PARAMETERS
    mean    : R -> R function describing average value of curve at each grid point;
    n_basis : int, number of trigonometric basis functions (paper uses 25);
    decay   : float, rate of exponential decay of coefficient variance across basis functions;
    rng     : int, seed for reproducibility.
    """
    rng = np.random.default_rng(rng)
    Psi = trig_basis(grid, n_basis)
    var_k = variance * np.exp(-np.arange(n_basis) / decay)
    coef = rng.standard_normal((n, n_basis)) * np.sqrt(var_k)[None, :]
    mu = np.zeros_like(grid) if mean is None else mean(grid)
    return mu[None, :] + coef @ Psi


def trunc_int(X, grid, u=0.0, v=1.0, beta=None):
    """Integral via trapezium rule. Returns an array.

    PARAMETERS
    X    : (n, N) array, each row represents one curve;
    u    : lower bound of integral;
    v    : upper bound of integral;
    beta : (N, ) coefficient function on grid, sets beta ≡ 1 by default.
    """
    interval = (grid >= u) & (grid <= v)    # N-dimensional Boolean array, only true in window

    if beta is None:
        return np.trapezoid(X[:, interval], grid[interval], axis=1)
    return np.trapezoid(beta[interval] * X[:, interval], grid[interval], axis=1)


def sim_dataset(
    n=200, n_grid=301, u=0.1, v=1.0, snr=26.25, n_basis=25, decay=4.0,
    variance=1.0, mean=None, a=0.0, beta_fn=None, rng=None,
):
    """Full DGP.

    Returns a dictionary bundling grid, curves, responses, and the true coefficient function

    PARAMETERS
    n_grid  : int, number of time points. HH.theta_methodB/lambda_select search for theta only
              among this grid's points, so a coarse n_grid quantises theta_hat and can make the
              first-local-minimum used to pick theta stick at one value across a wide range of
              lambda (see notebook discussion) -- 301 is fine enough to avoid that in practice
              while staying fast;
    snr     : float, signal-to-noise ratio, var(signal) / sigma²;
    n_basis : int, number of trigonometric basis functions used to build X (paper uses 25);
    decay   : float, rate of exponential decay of coefficient variance across basis functions;
    beta_fn : callable, evaluates function on grid. Sets true beta to 1 on [u, v] and 0 outside.
    """
    rng = np.random.default_rng(rng)
    grid = np.linspace(0.0, 1.0, n_grid)
    X = sample_gp(n, grid, mean, n_basis, decay, variance, rng=rng)

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
            "sigma": sigma, "a": a, "b": beta_true}
