"""Data-generating process for truncated functional linear models, introduced in
Hall & Hooker, 2016.

The curves themselves are the random trigonometric expansion of `core.gp`, which
`far.far_dgp` also draws its innovations from; what is specific to this model, and so
lives here, is pairing them with a coefficient function supported on one interval.
"""

import numpy as np

from ..core.gp import sample_gp
from ..core.quadrature import trunc_int

__all__ = ["sim_dataset"]


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
