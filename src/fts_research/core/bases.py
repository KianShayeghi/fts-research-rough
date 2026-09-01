"""Basis systems the packages here expand curves and coefficient functions in.

Both are evaluated on a grid and returned as an array, rather than as an object carrying its
own arithmetic, since everything downstream works pointwise under `quadrature`. The
trigonometric system is the one Hall & Hooker (2016) generate their curves from; the B-spline
system is the one Stefanucci, Bernardi & Canale (2025) expand the coefficient function in, and
`fdaSP.SBC` reproduces `splines::bs(s, df = n_basis, intercept = TRUE)` exactly, which
`tests/test_f2ssp_match.py` checks against R.
"""

import numpy as np
from scipy.interpolate import BSpline

__all__ = ["trig_basis", "bspline_basis"]


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
