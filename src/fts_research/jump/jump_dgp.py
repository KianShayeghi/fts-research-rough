"""Simulating from the jump equation, for the estimator in `jump_eq` to be checked against.

Two generators, and the difference between them is the whole point.

`simulate_jump` draws the score vectors as plain Gaussian noise and returns them. The
estimator then sees the regressors the model is written in terms of, so the only thing being
checked is whether it solves the algebra it claims to solve. Recovery here should be exact up
to sampling error, and a failure is a bug rather than a finding.

`simulate_curves_and_jumps` goes the long way round, which is the way the real problem goes.
Curves are drawn on an intraday grid; the jump recursion is driven by `<delta, X_{t-1}>`
integrated **on the grid**, so the data-generating process does not secretly live in the
`M`-dimensional score space the estimator will work in; and the scores are then recovered by
running `far.far_est.fpca` on the simulated curves. What the estimator receives is then
generated regressors, exactly as on real data, and the loadings it can hope to recover are
`d_k = <delta, phi_hat_k>` against the *estimated* eigenfunctions -- which is why `d_true` is
computed after the decomposition rather than before it. Anything else would compare an estimate
against a target in a different basis and call the difference bias.

Curves come from `core.gp.sample_gp`, the i.i.d.-curve process the rest of the repository
simulates from, or under `rho > 0` from `far.far_dgp.sample_far1`, which drives the same
innovation process through a FAR(1) recursion. Real intraday curves are serially dependent --
`notebooks/overnight_jumps.ipynb` measures it -- and the jump equation treats the curve as a
predetermined regressor rather than as an i.i.d. draw, so it should survive that and the
option is here to check that it does.

Every curve is anchored at zero on the left, which is the cumulative intraday return's own
boundary condition `X_t(0) = 0` and not a modelling choice.
"""

import numpy as np

from ..core.gp import sample_gp
from ..core.quadrature import quadrature_weights
from ..far import far_dgp
from ..far.far_est import fpca

__all__ = [
    "simulate_jump",
    "simulate_curves_and_jumps",
]


def _sigma_from_snr(signal, snr):
    """Innovation standard deviation giving a signal-to-noise ratio of `snr`. Returns a float.

    `snr` is `var(signal) / sigma_J^2`, the definition `truncFLR.HH_dgp.sim_dataset` and
    `fdaSP.SBC_sim.simulate` use, so a design here is comparable with a design there. The
    signal it is measured against is the *exogenous* part of the conditional mean --
    `<delta, X_{t-1}>` plus `z_t' k` -- which is fixed before the recursion runs; the
    autoregressive term then adds to what is explained, so the realised R-squared of the fitted
    equation is a little above `snr / (1 + snr)`.
    """
    variance = float(np.var(signal))
    if variance <= 0:
        raise ValueError("the signal has no variance, so no signal-to-noise ratio is defined.")
    return float(np.sqrt(variance / float(snr)))


def _run_recursion(a, gamma, signal, k, Z, sigma_J, rng):
    """Iterate `J_t = a + gamma J_{t-1} + signal_{t-1} + z_t' k + eta_t` from its stationary
    mean. Returns the (n,) jump series."""
    n = signal.size
    J = np.empty(n)
    J[0] = a / (1.0 - gamma) if abs(gamma) < 1.0 else 0.0
    eta = rng.normal(0.0, sigma_J, size=n)
    for t in range(1, n):
        mean = a + gamma * J[t - 1] + signal[t - 1]
        if k is not None:
            mean = mean + float(k @ Z[t])
        J[t] = mean + eta[t]
    return J


def simulate_jump(n, a, gamma, d, snr=4.0, k=None, psi=None, Z=None, rng=None):
    """Draw jumps from the equation with the score vectors supplied rather than estimated.

    Returns a dict with J (n,), psi (n, M), Z (n, p) or None -- the three `jump_eq.fit_mle`
    takes, whose estimates should then recover `(a, gamma, d, k)` -- and sigma_J, the
    innovation standard deviation `snr` implied.

    Nothing here is meant to resemble a market: the scores are standard Gaussian noise unless
    they are passed in. This is the generator for checking that the estimator is right, and
    `simulate_curves_and_jumps` is the one for checking that it is usable.

    PARAMETERS
    n       : int, sessions to draw;
    a       : float, intercept;
    gamma   : float, persistence of the jump in its own lag;
    d       : (M,) array of loadings on the score vector;
    snr     : float, signal-to-noise ratio, `var(psi d + Z k) / sigma_J^2`;
    k       : (p,) array or None, loadings on the covariates;
    psi     : (n, M) array or None, score vectors; standard Gaussian if not given;
    Z       : (n, p) array or None, covariates; standard Gaussian if `k` is given and this is
              not;
    rng     : int, Generator or None, seed.
    """
    rng = np.random.default_rng(rng)
    d = np.asarray(d, float)
    if psi is None:
        psi = rng.standard_normal((n, d.size))
    psi = np.atleast_2d(np.asarray(psi, float))
    if k is not None:
        k = np.asarray(k, float)
        if Z is None:
            Z = rng.standard_normal((n, k.size))
        Z = np.atleast_2d(np.asarray(Z, float))

    signal = psi @ d + (Z @ k if k is not None else 0.0)
    sigma_J = _sigma_from_snr(signal, snr)
    J = _run_recursion(a, gamma, psi @ d, k, Z, sigma_J, rng)
    return {"J": J, "psi": psi, "Z": Z, "sigma_J": sigma_J}


def simulate_curves_and_jumps(n, a, gamma, delta, snr=4.0, k=None, grid=None, n_grid=79,
                              n_components=5, Z=None, rho=0.0, n_basis=25, decay=4.0,
                              variance=0.02, rng=None):
    """Draw intraday curves and the overnight jumps that follow them, then decompose.

    Returns a dict with J (n,), X (n, n_grid) the curves, grid, psi (n, M) the estimated
    scores, phi (M, n_grid) the estimated principal components, tau (M,) their eigenvalues,
    mean (n_grid,), d_true (M,) the loadings `<delta, phi_hat_k>` the estimator can recover,
    delta (n_grid,) the function that actually drove the recursion, signal (n,) the exogenous
    part of the conditional mean, sigma_J the innovation standard deviation `snr` implied, and
    Z and k as given.

    The signal is `<delta, X_{t-1}>` by the trapezium rule on the grid, so it is not confined
    to the span of the retained components; whatever `delta` has outside that span goes into
    the equation and is unavailable to the estimator, which is the honest version of the
    problem.

    PARAMETERS
    n            : int, sessions to draw;
    a            : float, intercept;
    gamma        : float, persistence of the jump in its own lag;
    delta        : (n_grid,) array, the true functional coefficient on the grid;
    snr          : float, signal-to-noise ratio, `var(<delta, X> + Z k) / sigma_J^2`;
    k            : (p,) array or None, loadings on the covariates;
    grid         : (n_grid,) array or None, the intraday domain; equally spaced on [0, 1] if
                   not given;
    n_grid       : int, points on the grid where `grid` is not given;
    n_components : int, principal components retained from the simulated curves;
    Z            : (n, p) array or None, covariates; standard Gaussian if `k` is given and
                   this is not;
    rho          : float, Hilbert-Schmidt norm of a FAR(1) operator to draw the curves
                   through. Zero draws them i.i.d. from `core.gp.sample_gp`;
    n_basis      : int, basis functions the curve process is expanded in;
    decay        : float, decay rate of the coefficient variances of that expansion;
    variance     : float, leading coefficient variance, which sets the scale of the curves;
    rng          : int, Generator or None, seed.
    """
    rng = np.random.default_rng(rng)
    grid = np.linspace(0.0, 1.0, int(n_grid)) if grid is None else np.asarray(grid, float)
    delta = np.asarray(delta, float)
    if delta.size != grid.size:
        raise ValueError(f"delta has {delta.size} points, the grid has {grid.size}.")

    if rho > 0:
        Psi = far_dgp.exponential_kernel(grid, rho=float(rho))
        X = far_dgp.sample_far1(n, grid, Psi, n_basis=n_basis, decay=decay,
                                variance=variance, rng=rng)
    else:
        X = sample_gp(n, grid, n_basis=n_basis, decay=decay, variance=variance, rng=rng)
    X = X - X[:, [0]]                       # the cumulative intraday return starts at zero

    w = quadrature_weights(grid)
    signal = (X * delta[None, :]) @ w        # <delta, X_t>, on the grid

    if k is not None:
        k = np.asarray(k, float)
        if Z is None:
            Z = rng.standard_normal((n, k.size))
        Z = np.atleast_2d(np.asarray(Z, float))

    sigma_J = _sigma_from_snr(signal + (Z @ k if k is not None else 0.0), snr)
    J = _run_recursion(a, gamma, signal, k, Z, sigma_J, rng)

    dec = fpca(X, grid, n_components=n_components)
    d_true = (dec["phi"] * delta[None, :]) @ w      # <delta, phi_hat_k>

    return {"J": J, "X": X, "grid": grid, "psi": dec["scores"], "phi": dec["phi"],
            "tau": dec["tau"], "mean": dec["mean"], "d_true": d_true, "delta": delta,
            "Z": Z, "k": k, "signal": signal, "sigma_J": sigma_J}
