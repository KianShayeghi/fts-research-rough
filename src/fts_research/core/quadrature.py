"""Integration on a sampling grid, by the trapezium rule.

Curves are held as values on a common grid rather than as coefficients, so every integral in
this repository is a quadrature. All of them are taken under this one rule: the L2 inner
products the principal components of `truncFLR` and `far` are orthonormal in, the design
integrals of `fdaSP.SBC`, and the integrated squared errors `bench` scores with. An estimate
made in one package is therefore comparable with an estimate made in another, and a
discrepancy between two of them is a difference of method rather than of quadrature.
"""

import numpy as np

__all__ = ["quadrature_weights", "trunc_int"]


def quadrature_weights(grid):
    """Trapezium weights w on `grid`, so `sum_k w_k f_k` approximates `int f`. Returns (N,)."""
    grid = np.asarray(grid, dtype=float)
    w = np.zeros_like(grid)
    w[1:] += np.diff(grid)
    w[:-1] += np.diff(grid)
    return 0.5 * w


def trunc_int(X, grid, u=0.0, v=1.0, beta=None):
    """Integral of `beta` X over [u, v] by the trapezium rule. Returns an array.

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
