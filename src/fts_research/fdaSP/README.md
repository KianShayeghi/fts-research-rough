# fdaSP

Locally sparse scalar-on-function regression, following Stefanucci, Bernardi & Canale (2025):

    y_i = ∫_T x_i(t) ψ(t) dt + z_i ^ T γ + ε_i,    i = 1, ..., n

Estimates a coefficient function ψ(t) that is exactly zero on a (positive-measure) subset of
its domain. Unlike `truncFLR`, the informative set is not assumed to be a single interval: it
can be any union of intervals, so there is no boundary to scan for.

## Modules

**`SBC.py`**: the estimator. ψ is expanded in a B-spline basis and the coefficients penalised
by an overlap group lasso, one group per window of `order` consecutive coefficients, solved
by ADMM. A group set to zero switches off the region its basis functions support, which is
what makes the sparsity local. A Python port of `f2sSP` from the R package `fdaSP`. Includes
`lambda_max` by bisection, the descending lambda path, warm-started path fitting, and k-fold CV
with an optional one-SE rule; `fit` reports the null and support regions directly. The
`f2ssp_*`/`fit_like_f2ssp` helpers reproduce the R package's own parametrisation; see
*Matching the R package* below.

**`SBC_sim.py`**: the paper's three test functions (`psi1` piecewise sine with a null region,
`psi2` Epanechnikov kernel, `psi3` cubic control with no null region) plus the Monte Carlo
study comparing the group lasso against Lasso, Ridge, Elastic Net, unpenalised splines and
FPCA regression, reported as ISE overall and split by null/supported region.

**`empirical.py`**: the already-functional real data (Tecator NIR spectra, the application
dataset of the paper) plus a daily equity price panel and `rolling_windows`, which turns a
scalar time series into functional predictors by slicing it into overlapping windows. That
slicing is the worked example of rows that are *not* exchangeable, which is what
`SBC.blocked_folds` exists for.

**`intraday.py`**: intraday bars for nine instruments -- five single stocks, two ETFs and two
index funds -- and the reshaping that turns them into one curve and one scalar per trading
day. The curve is the cumulative intraday return `X_t(s) = log P_t(s) - log P_t(0)`; the
scalar is the overnight jump `J_t = log(Open_t) - log(Close_{t-1})`. The two partition the
open-to-open log return exactly, so the curve is precisely what the jump omits and regressing
one on the other is a scalar-on-function problem on the only two pieces there are.
`session_curves` also builds the running realised volatility and the return *rate*, which is
the predictor to use when the coefficient function has to be readable; `session_summary`
supplies per-session scalar covariates; `date_blocked_folds` blocks cross-validation by
calendar date, which on a pooled panel matters more than the usual window-overlap argument
suggests -- nine instruments contribute a row each on every date and share that date's
market-wide move. See *Snapshots, not downloads* below.

**`rbridge.py`** and **`r_reference.py`**: the bridge to R used to check the port. `rbridge`
locates and calls `Rscript`; `r_reference` holds the R snippets that produce the reference
outputs and caches each as a CSV under `artifacts/reference/`. Both are only needed for the
comparison below, and nothing in the estimator depends on them.

## Conventions worth knowing

**Grid integration is trapezoidal throughout**, via `np.trapezoid`. `SBC.bspline_design` folds
the trapezium weights into the design matrix so the functional term becomes an ordinary linear
one. `f2sSP` does *not*; see below.

**Cross-validation assumes exchangeable rows.** `cv_fit` assigns folds at random, which is
right for the simulated curves and for Tecator. It is not right for the overlapping windows
`empirical.rolling_windows` builds: consecutive predictors share `width - 1` observations and
consecutive responses share `horizon - 1`, so a random split trains on near-duplicates of
what it then scores. Pass `folds=blocked_folds(n, n_folds)` and `embargo=width + horizon`
there, which makes the folds contiguous stretches of time and drops every training window
sharing a raw observation with the test block.

**Null regions are exact.** The ADMM constraint is `F psi = zeta`, and `zeta` is what the
soft-threshold sets to exactly zero, so a group whose `zeta` block vanishes forces **every**
coefficient in that group to zero. `SBC.fit` and `SBC.fit_path` snap those coefficients to
exact zeros, so `beta` and the reported `support` always agree and no magnitude threshold is
needed downstream.

## Data

To fetch everything up front (e.g. before working offline):

```bash
python -m fts_research.fdaSP.empirical   # add --force to re-download
python -m fts_research.fdaSP.intraday    # add --force to take a fresh snapshot
```

See `empirical.SOURCES`: Tecator from OpenML dataset 505 (originally StatLib; public domain,
redistributable; first 215 of 240 samples, the standard interpolation set) and a daily price
series from the public `plotly/datasets` repository. `intraday.PANELS` records the two bar
panels, both from the public Yahoo Finance chart endpoint.

### Snapshots, not downloads

The two datasets in `empirical` are static files at fixed URLs, so fetching them again returns
what it returned before. **The intraday panels are not.** The chart endpoint serves a window
ending today -- 730 days for hourly bars, 60 for five-minute bars -- so re-fetching on a later
date returns different bars and every number computed from them moves. The committed CSVs are
therefore dated snapshots: `ensure_panel` prefers the cache and downloads only when a file is
missing, `--force` is what deliberately replaces a snapshot, and `snapshot_span` reads the
window off the cache so a notebook can state which data its numbers came from. Each fetch
also records the day it ran, the window it took and the file's SHA-256 in
`data/manifest.json`, which `tests/test_provenance.py` reads back against the working tree.

Bars arrive unadjusted, so `fetch_bars` divides every bar before a split by that split's ratio
-- an unadjusted 4-for-1 split otherwise reads as a 75% overnight fall -- and carries cash
dividends, which `overnight_jumps` adds back so that a stock opening lower by its dividend on
the ex-date does not register as a price move.

## Matching the R package

`fdaSP.SBC` and CRAN `fdaSP::f2sSP` compute the same estimator but parametrise it differently,
so a default side-by-side run looks like a discrepancy when it is only a difference of
convention. Four things differ:

| | `SBC` default | `f2sSP` |
|---|---|---|
| (a) loss | ½‖y − Wψ‖² | ½ₙ‖y − Wψ‖² |
| (b) penalty weights | `F_g = υ_g S_g S* T` with `S* = diag(1/√counts)` | `F_g = S_g diag(1/counts)` |
| (c) design | `X Φ` with trapezium weights folded in | plain `X Φ` |
| (d) standardisation | none | centres and scales `W` and `y`, then maps back |

The `1/n` loss and the design weighting only rescale λ, and (b) is a different but equally
valid weighting of the same overlap group lasso.
`SBC.fit_like_f2ssp(X, y, lam, n_basis, order)` applies all four and reproduces the R fit to
~1e-3 at fixed penalties, and to within 1e-2 across the regularised part of the λ path
(λ ≥ 0.01). Agreement degrades in the near-unpenalised tail, where both solvers stop at
different points of a near-flat objective. `notebooks/LocSpaFLR_validation.ipynb` shows
this; `tests/test_f2ssp_match.py` asserts it against the reference CSVs committed in
`artifacts/`.

Regenerate those reference CSVs with:

```bash
python scripts/make_r_artifacts.py   # needs R with fdaSP; set FTS_RSCRIPT if not on PATH
```

## Usage

```python
import numpy as np
from fts_research.fdaSP import SBC, SBC_sim

X, y, grid, beta_true = SBC_sim.simulate(SBC_sim.psi1, n=50, m=100)
cv = SBC.cv_fit(X, y, grid, n_basis=15, n_folds=5)
beta_hat, support = cv["beta"], cv["support"]
print(cv["null_regions"])                     # beta_hat is exactly 0 on these intervals
```

On serially dependent rows, cross-validate over blocks instead:

```python
from fts_research.fdaSP import SBC, empirical as emp

X, y, grid = emp.rolling_windows(returns, width=20, horizon=5)
cv = SBC.cv_fit(X, y - y.mean(), grid, n_basis=10, n_folds=10,
                folds=SBC.blocked_folds(len(y), 10), embargo=25)
```

Reproduce the simulation study (slow, 100 replicates × 3 functions, the OGL and 5 competitors):

```python
results = SBC_sim.run_sim(n_rep=100)
rows = SBC_sim.summarise(results)
```

## Tests

`pytest tests/test_sbc.py tests/test_f2ssp_match.py tests/test_empirical.py
tests/test_rbridge.py` — the ADMM output is checked to be a genuine minimiser of (62.3), the
group soft-threshold to be the prox of the ℓ₂ norm, the null regions to be exact and monotone
in λ, the blocked folds to leave no window overlapping its test block, and the whole
estimator to agree with the R reference. `-m "not slow"` skips the CV
pipelines; R-dependent tests skip themselves when R is absent.

## References

Stefanucci, M., Bernardi, M. and Canale, A. (2025). Locally sparse estimation for functional
linear models with scalar response. In G. Aneiros, E. G. Bongiorno, A. Goia and M. Hušková
(eds.), *New Trends in Functional Statistics and Related Fields*, Contributions to Statistics,
chapter 62, pp. 521-527. Springer.
[doi:10.1007/978-3-031-92383-8_62](https://doi.org/10.1007/978-3-031-92383-8_62)

Bernardi, M. and Stefanucci, M. (2023). fdaSP: sparse functional data analysis methods. R
package. [doi:10.32614/CRAN.package.fdaSP](https://doi.org/10.32614/CRAN.package.fdaSP) — the
implementation `SBC` is checked against, whose outputs are committed under
`artifacts/reference/`.

Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of
return. *The Journal of Business* 53(1), 61-65.
[doi:10.1086/296071](https://doi.org/10.1086/296071) — the range `session_summary` reports,
which is the summary of a session that best predicts the size of the jump that follows it.
