# core

What more than one of the other four packages needs. Nothing here belongs to a method or to a
paper: a quadrature rule, two basis systems, the Gaussian process the simulations draw curves
from, and the three places on disk this repository reads and writes.

The point of collecting them is that there is then exactly one of each. Curves are held as
values on a grid rather than as coefficients, so every inner product in the repository is a
quadrature, and an integrated squared error computed in `bench` is comparable with one
computed in `truncFLR` only because both are taken under the same weights. Two copies of the
trapezium rule can agree today and disagree after one edit; one copy cannot.

`tests/test_layout.py` holds the arrangement in place. `core` imports none of the other
packages, so it can be read on its own, and no package may reach into another's private
module for something that belongs here — which is what `far` was doing with
`truncFLR._common.quadrature_weights`, and what `bench` had worked around by keeping a second
copy of the same six lines.

## Modules

**`quadrature.py`**: the trapezium weights `w` on a grid, so that `w @ f` approximates
`∫ f`, and `trunc_int`, the same rule over a sub-interval `[u, v]` with an optional
coefficient function. Every integral in the repository goes through one of the two.

**`bases.py`**: the two basis systems the methods expand in. `trig_basis` is the
trigonometric system of Hall & Hooker (2016, Section 4), `ψ_1 = 1`, `ψ_2k = sin(2kπt)`,
`ψ_{2k+1} = cos(2kπt)`, which `truncFLR.HH_dgp` generates curves from and `bench.design`
uses as one of its three designs. `bspline_basis` evaluates an order-`p` B-spline basis with
an open knot vector, reproducing `splines::bs(s, df = n_basis, intercept = TRUE)` exactly;
`fdaSP.SBC` expands the coefficient function in it, and `tests/test_f2ssp_match.py` checks the
agreement with R to machine precision.

**`gp.py`**: curves drawn as `X(t) = Σ_k c_k ψ_k(t)` with `c_k ~ N(0, v_k)` independent, so
the covariance operator is `Σ_k v_k ψ_k ⊗ ψ_k`, its eigenvalues are the prescribed `v_k` and
its eigenfunctions the basis itself. This is the i.i.d.-curve process `truncFLR.HH_dgp`
simulates from and the innovation process `far.far_dgp` drives its recursion with, so an
estimator run on either sees curves with the same spectrum.

**`paths.py`**: the four directories, and the only module allowed to name one. `data/` holds
inputs, `artifacts/reference/` what R produced for this implementation to be checked against,
`artifacts/cache/` what this implementation produced and would take minutes to recompute, and
`figures/` what is exported for writing up. They are told apart by how each comes back if it
is deleted, which is the distinction that decides what is committed.

**`provenance.py`**: what each committed file under `data/` actually is — its source, the day
it was taken, its SHA-256, and for the intraday panels the window of sessions it covers.
Written by the fetch itself, so the record is made where it is known; read back by
`verify`, which `tests/test_provenance.py` asserts against the working tree.

**`style.py`**: one visual system for every figure. Colour carries a single meaning wherever
the three estimators appear together — which method — and `save_figure` writes a figure into
`figures/` as PDF and PNG.

## Usage

```python
import numpy as np

from fts_research.core import paths, provenance
from fts_research.core.bases import bspline_basis, trig_basis
from fts_research.core.gp import sample_gp
from fts_research.core.quadrature import quadrature_weights, trunc_int

grid = np.linspace(0, 1, 301)
w = quadrature_weights(grid)                 # ∫ f ≈ w @ f, on any grid

X = sample_gp(200, grid, n_basis=25, decay=4.0, rng=0)
scores = (X * w[None, :]) @ trig_basis(grid, n_basis=25).T
signal = trunc_int(X, grid, u=0.8, v=1.0)    # ∫_0.8^1 x_i(t) dt

Phi = bspline_basis(grid, n_basis=15, order=4)
```

Where things are, and whether the data is still what it says it is:

```python
print(paths.DATA_DIR, paths.REFERENCE_DIR, paths.CACHE_DIR, paths.FIGURE_DIR)
print(provenance.verify())                   # [] when the working tree matches the manifest
```

```bash
python -m fts_research.core.provenance       # the manifest, and a non-zero exit on a mismatch
```

## Tests

`pytest tests/test_core.py tests/test_layout.py tests/test_provenance.py` — the quadrature
against exact integrals of polynomials and on an unevenly spaced grid, the trigonometric basis
against the display in the paper and for orthogonality under the shared weights, the B-spline
basis for partition of unity, local support and exact reproduction of cubics, and the process
for the coefficient variances it is defined by. `test_layout.py` reads the import statements
of every module back against the declared dependency graph, and `test_provenance.py` reads
`data/manifest.json` back against the files on disk.

## References

Hall, P. and Hooker, G. (2016). Truncated linear models for functional data.
*Journal of the Royal Statistical Society: Series B* 78(3), 637–653.
[doi:10.1111/rssb.12125](https://doi.org/10.1111/rssb.12125)

Stefanucci, M., Bernardi, M. and Canale, A. (2025). Locally sparse estimation for functional
linear models with scalar response. In G. Aneiros, E. G. Bongiorno, A. Goia and M. Hušková
(eds.), *New Trends in Functional Statistics and Related Fields*, Contributions to Statistics,
chapter 62, pp. 521–527. Springer.
[doi:10.1007/978-3-031-92383-8_62](https://doi.org/10.1007/978-3-031-92383-8_62)
