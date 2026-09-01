"""Gaussian curves drawn as a random basis expansion,

    X(t) = sum_{k=1}^{K} c_k psi_k(t),    c_k ~ N(0, v_k) independently,

whose covariance operator is sum_k v_k psi_k (x) psi_k when the psi_k are orthonormal, so its
eigenvalues are the prescribed v_k and its eigenfunctions the basis itself. This is the
i.i.d.-curve process `truncFLR.HH_dgp` simulates from, and the innovation process
`far.far_dgp` drives its FAR(1) recursion with.
"""

import numpy as np

from .bases import trig_basis

__all__ = ["sample_gp"]


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
