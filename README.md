# fts-research

[![tests](https://github.com/KianShayeghi/fts-research/actions/workflows/ci.yml/badge.svg)](https://github.com/KianShayeghi/fts-research/actions/workflows/ci.yml)

Advanced Modelling and Forecasting of Functional Time Series.

Research code for scalar-on-function regression,

    y_i = ∫_T x_i(t) psi(t) dt + eps_i,

where the coefficient function `psi` is zero on part of its domain, so the estimation problem
includes finding *where*. Two approaches are implemented: `truncFLR`, which assumes the
informative set is a single interval and estimates its endpoint (Hall & Hooker, 2016), and
`fdaSP`, which lets it be any union of intervals and induces the sparsity by penalisation
(Stefanucci, Bernardi & Canale, 2025). `far` supplies serially dependent curves, so both can
be run without the i.i.d. assumption, and also estimates and tests a FAR(1) from observed
curves; `bench` runs all three methods over a matched grid of designs.

## What is here

**A correction to Hall & Hooker's tuning scheme.** Their §3.2 selects the truncation penalty
`lambda` by a surrogate criterion, and run as the displays are written it returns the smallest
candidate on the grid regardless of the data. The cause is a scale mismatch: equation (3.4)
writes the penalty as `n*lambda*theta^2`, "since both `S` and `S_1` are of order `n`", while
step 6 pairs `S_Y` with a bare `lambda`. Read as the sum step 5 displays, `lambda` is `n` times
too small to register: `(S_Y'' + lambda)/S_Y''` never reaches `1.005` anywhere on the candidate
grid, the variance `V(lambda)` collapses to order `sigma^2/n^2`, and the step-7 density ends up
some twenty-five times narrower than the grid it is integrated on. `P_b` then tracks
`S_b(theta_lambda)`, which is monotone, so `lambda_hat` lands on the floor. Taking step 6's
`S_Y` per observation and resolving the step-7 quadrature fixes both.
`notebooks/truncFLR_lambdadegen.ipynb`.

**Method A is essentially unbiased once that is fixed.** Over 100 paired replications
(n = 200, snr = 26.25, true `u` = 0.8), method A recovers the endpoint with bias `-0.0027`
against method B's `-0.0298`, reproducing the ordering of the paper's Table 1. Paired on the
same datasets the gap is `-0.023` in absolute endpoint error (`t = -23.9`) and `-0.0038` in
integrated squared error (`t = -7.0`), at mean ISEs of `0.0070` and `0.0109`. Both beat the
untruncated FPCA pilot, which over 25 replications of the same design averages `0.0219`
against method A's `0.0091` and method B's `0.0120`.
`notebooks/truncFLR_methodA.ipynb`, `notebooks/truncFLR_lambdadegen.ipynb`.

**The truncation model does not beat the group lasso, even on its own ground.** Over 50 paired
replications the locally sparse estimator has the lower median ISE on both a single-window
coefficient function, which is exactly Hall & Hooker's model, and a two-window one, which is
not. On the single window it is ahead of method B on 62% of draws, which a signed-rank test
does not separate from a tie (`p = 0.088`); on the pair it wins outright. What the truncation
model has that the penalty does not is an explicit endpoint estimate with a sampling
distribution, rather than a null set read off whichever basis groups the penalty switched off.
`notebooks/HH_vs_SBC.ipynb`.

**Method A fails loudly rather than quietly.** In 1 replication of 50 it truncated to ten grid
points and BIC then fitted the 9 components that sub-grid can support, giving an ISE of
`1.9e16` against a median of `0.0019`. That is the behaviour the paper warns about and the
reason its Table 2 reports medians; both symptoms, a large `m_hat` on a short retained
interval, are visible in the fitted object. Reported here with medians, quartiles and
signed-rank tests rather than means and `t`-tests. `notebooks/HH_vs_SBC.ipynb`.

**Which of the two models wins depends on the shape of the coefficient function, not on the
basis the curves are built from.** Over nine designs -- three curve-generating bases crossed
with three coefficient functions, matched so that the covariance spectrum is identical and only
the eigenfunctions differ -- method A has the lowest median relative ISE on all three
step-function cells, and the group lasso on all three smooth-boundary cells and on two of the
three multi-window ones. That ordering is the same on Hall & Hooker's own trigonometric design
as on the group lasso's own B-spline one. Matching the designs is not free: taken in index
order a B-spline basis runs left to right across the domain, so a decaying variance profile on
it piles the curves' variation at one end and hands the truncation model an orientation before
it has seen any data. `bench` orders every basis by roughness instead.
`notebooks/basis_and_coef_comp.ipynb`.

**Most of the gap between the two models is the tuning rule rather than the estimator.** Scored
at the best `lambda` on its own grid, method B is the most accurate estimator in that study on
the step function, at less than half the group lasso's error. Under section 3.2 it is not: the
criterion costs it a factor of 240 in one cell, and costs method A more than eleven orders of
magnitude in two others, by choosing the largest `lambda` on the grid, which drives `theta_hat`
to a retained interval of some eight grid points on which BIC then buys every component
available. Cross-validation costs the group lasso under 30% by comparison, and held-out error
hides the failure completely -- method A predicts within 30% of the noise floor on the cells
where its coefficient function is wrong by ten orders of magnitude.
`notebooks/basis_and_coef_comp.ipynb`.

**The Python port matches CRAN `fdaSP::f2sSP`.** The two parametrise the same estimator under
four different conventions; matched, they agree to `1e-3` at fixed penalties and to `1e-2`
across the regularised part of the path. The comparison runs without R, against reference
outputs committed under `artifacts/`. `notebooks/LocSpaFLR_validation.ipynb`.

**Cross-validation never delivers local sparsity.** On all three of the paper's test functions
and on Tecator, the CV-selected `lambda` zeroes nothing at all. Cross-validation targets
prediction risk, while an exact zero is worth more for support recovery than for prediction, so
the two optima do not coincide. Along one path the three optima sit in that order, prediction
first and estimation just above it, with support recovery an order of magnitude further out. On
Tecator 17 of 60 penalties leave the estimate partially sparse and neither CV rule selects from
among them.
The same holds across all nine designs of `notebooks/basis_and_coef_comp.ipynb`, where the
CV-selected fit returns a null set of length zero on every replication while the one-standard-
error rule recovers one for a small cost in error.
`notebooks/LocSpaFLR_f2s.ipynb`, `notebooks/tecator_application.ipynb`.

**Fold leakage is about where the duplication comes from, not about overlap as such.**
Slicing a return series into length-`W` windows gives rows sharing `W - 1` observations, and a
penalised linear estimator barely notices: it cannot memorise an individual window, so a
near-duplicate in its training set buys it little. Duplication across a *panel* is another
matter. Nine instruments contribute a row each on every trading date and share that date's
market-wide move, so a random split puts eight near-copies of every test row into the training
set, and random folds then report more than twice the out-of-sample R-squared that date-blocked
folds do. The cross-validated error used to *select* the penalty differs by under two per cent,
so the leak is invisible in the selected model and shows up only in the reported one.
`notebooks/overnight_jumps.ipynb`.

**The paper's Tecator null region is real, and the paper's tuning rule does not select it.**
Section 62.5 reports the coefficient function exactly zero above 940 nm, under cross-validated
error plus the one-standard-error rule. Run as written at `M = 15`, neither the port nor the
authors' own CRAN `f2sSP_cv` zeroes any part of the domain, and `f2sSP_cv`'s minimum is the
last penalty on its own path, so no fold assignment would move it. The reported null set does
appear along the port's path, at a penalty `7.75` standard errors above the cross-validated
minimum where the one-standard-error rule moves one. Whether it is reachable at all is decided
by a convention: standardising the design, as `f2sSP` does, makes these highly collinear
spectra switch off together, so its path jumps from all-zero to dense with nothing partially
sparse in between, against seventeen of sixty for the port's trapezium weighting.
`notebooks/tecator_application.ipynb`.

**On Tecator the truncation model buys the same support recovery without paying for it.**
Held-out prediction error falls steeply up to about 915 nm and is flat from there to 965 nm, so
the paper's 940 nm lies inside the interval the data cannot separate from the best, and keeping
the long-wavelength end instead is clearly worse. Truncating cuts held-out error by 37% against
using the whole spectrum. The group lasso reaches the same null set only by shrinking
everything it keeps at the same time, since one `lambda` controls both, and that more than
doubles its held-out error. Section 3.2's own criterion finds neither point: method A lands at
919 nm, method B collapses to 868 nm on a penalty near the top of its grid.
`notebooks/tecator_application.ipynb`.

**A truncation profile has to be read at its elbow, not at its minimum.** Past the true
endpoint the retained curve carries no signal, so extending it costs only estimation variance
and the held-out error flattens rather than rising; the minimiser then wanders over that flat
stretch and is not a consistent endpoint estimate. `HH_A.truncation_profile` reports both, and
on simulated data the one-standard-error elbow recovers true endpoints of `0.3`, `0.5` and
`0.7` where the minimiser returns `0.84`, `0.84` and `0.68`. This is the same
prediction-versus-support-recovery split that decides `lambda` for the group lasso, arriving
from the other direction. `tests/test_hh_a.py`, `notebooks/tecator_application.ipynb`.

**Everything predictable about an overnight jump is a magnitude, not a direction.** Over some
500 sessions each for nine instruments, a Ljung-Box test rejects independence of the signed
overnight jump for none of them and of its absolute value for all nine. The intraday curve
process looks more promising and is not: both a chi-squared portmanteau test and a permutation
reference reject independence, yet a fitted FAR(1) beats the flat forecast out of sample for
only two of nine, by margins under a percentage point. Dividing each curve by its own norm,
which keeps the shape of the session and discards how far it moved, removes the dependence in
seven of the eight instruments where it was present, while the norms themselves are
autocorrelated in all nine. The dependence is volatility clustering, which lives in the
conditional variance, and a FAR(1) models the conditional mean.
`notebooks/overnight_jumps.ipynb`.

**When a curve's informative summary is one of its values, the functional term is the wrong
slot.** The previous session's running realised volatility predicts the size of the coming
overnight jump, but handing the estimator the whole curve does *worse* out of sample than a
regression on that curve's endpoint alone. A bounded `psi` integrated against `X` cannot
evaluate `X` at a point, so the signal belongs in the model's unpenalised scalar term, usually
treated as somewhere to park nuisance covariates, and the functional term is what adds the
noise. `notebooks/overnight_jumps.ipynb`.

**The direction does become predictable once the search widens past the market's own history.**
London trades through the New York night. Its session truncated at 13:00 local, which is before
the New York open under either daylight-saving offset, predicts the US overnight jump for seven
of eight instruments: correlation `0.272` for SPY over 482 nights, above `0.23` for the two
other broad-market instruments, and an out-of-sample R-squared of `0.042` for the group lasso.
This is news propagating between markets rather than an inefficiency. Neither estimator finds
structure inside it -- the group lasso zeroes none of the morning and the truncation profile
varies by four per cent across every candidate endpoint with no elbow -- so the information
accumulates steadily and the useful summary is again close to the total move.
`notebooks/overnight_jumps.ipynb`.

**The standard FAR(1) test kernel is rank one.** Bosq's `exp{-(s^2 + r^2)/2}` factorises, so
it autocorrelates a single direction in L² and leaves every orthogonal direction i.i.d., a much
narrower departure from independence than it appears. `far` also supplies a full-rank kernel.
Matching the two on Hilbert-Schmidt norm does not match them on difficulty: spreading the same
budget across the spectrum drops the operator norm from `0.70` to `0.58`, and the resulting
ISEs come out mixed rather than ordered. `notebooks/FAR1_nonIID.ipynb`.

## Layout

```
src/
  truncFLR/     Truncated FLR: endpoint estimation, methods A and B (truncFLR/README.md)
  fdaSP/        Locally sparse FLR, plus the real datasets and the R bridge (fdaSP/README.md)
  far/          FAR(1): the data-generating process, and estimation and testing (far/README.md)
  bench/        Matched-spectrum three-way comparison across bases and targets (bench/README.md)
notebooks/
  truncFLR.ipynb              Truncated FLR: endpoint estimation (method B)
  truncFLR_methodA.ipynb      Method A, and the paired comparison against method B
  truncFLR_lambdadegen.ipynb  The lambda degeneracy in section 3.2, and its correction
  LocSpaFLR_f2s.ipynb         Locally sparse FLR, function-to-scalar, on simulated data
  LocSpaFLR_validation.ipynb  Python port checked against CRAN fdaSP::f2sSP
  tecator_application.ipynb   The paper's own application, reproduced, and the same
                              boundary estimated as a parameter by truncFLR
  overnight_jumps.ipynb       All three methods, plus FAR(1), on intraday price curves
                              and the overnight jumps that follow them
  HH_vs_SBC.ipynb             The two approaches head to head
  basis_and_coef_comp.ipynb   All three methods, over three bases and three coefficients
  FAR1_nonIID.ipynb           Both methods re-run on FAR(1)-generated (non-i.i.d.) curves
tests/          pytest suite covering all four packages
scripts/        make_r_artifacts.py -- regenerate the cached R reference outputs
data/           cached real datasets, including the two dated intraday snapshots
artifacts/      cached fdaSP::f2sSP reference outputs and the cached bench study, so the
                validation and comparison notebooks run without R and without re-fitting
```

Read `truncFLR.ipynb` and `LocSpaFLR_f2s.ipynb` first: they set up the two models on
simulated data, where the truth is known.

## Setup

Managed with [uv](https://docs.astral.sh/uv/), `src/` layout.

```bash
uv sync --extra notebooks
```

Installs the project in editable mode plus the `dev` group, so `pytest`, `flake8` and Jupyter
are all available. Note that `uv sync` alone omits the `notebooks` extra and so removes
Jupyter.

## Data

The empirical notebooks use two sources, both cached as CSVs under `data/`. `fdaSP.empirical`
holds the Tecator spectra and a daily price panel, fetched from the public URLs recorded in
`empirical.SOURCES`; `fdaSP.intraday` holds two panels of intraday bars for nine instruments,
from the public Yahoo Finance chart endpoint. To fetch everything up front (before working
offline):

```bash
python -m fdaSP.empirical            # --force to re-download
python -m fdaSP.intraday             # --force to take a fresh snapshot
```

Set `FTS_DATA_DIR` to keep the cache somewhere else.

**The intraday panels are dated snapshots, not reproducible downloads.** The endpoint serves a
window ending today -- 730 days of hourly bars, 60 days of five-minute bars -- so a re-fetch
returns different data and every number in `notebooks/overnight_jumps.ipynb` moves with it.
The committed CSVs are what those numbers were computed on; the loader prefers them and
downloads only when a file is missing. `intraday.snapshot_span` reports the window.

## R (optional)

`notebooks/LocSpaFLR_validation.ipynb` compares `fdaSP.SBC` against CRAN `fdaSP::f2sSP`. It
runs **without R**, using the reference outputs committed in `artifacts/`. Where R with the
`fdaSP` package is installed, those cells regenerate the reference instead of loading it. Set
`FTS_RSCRIPT` if `Rscript` is not found automatically. To regenerate the cache headlessly:

```bash
python scripts/make_r_artifacts.py
```

## Tests

```bash
pytest                  # everything (~2 min)
pytest -m "not slow"    # skip the full estimator pipelines (~10 s)
```

Run on every push by `.github/workflows/ci.yml`, against Python 3.12 and 3.13.

## Dependencies

`numpy`, `scipy`, `scikit-learn`, `scikit-fda`, `pandas`, `matplotlib`. Note that
`scikit-fda` constrains the usable Python version via `numba`; see `pyproject.toml` for the
pinned range.

## References

Hall, P. and Hooker, G. (2016). Truncated linear models for functional data.
*Journal of the Royal Statistical Society: Series B* 78(3), 637–653.
[doi:10.1111/rssb.12125](https://doi.org/10.1111/rssb.12125)

Stefanucci, M., Bernardi, M. and Canale, A. (2025). Locally sparse estimation for functional
linear models with scalar response. In G. Aneiros, E. G. Bongiorno, A. Goia and M. Hušková
(eds.), *New Trends in Functional Statistics and Related Fields*, Contributions to Statistics,
chapter 62, pp. 521–527. Springer.
[doi:10.1007/978-3-031-92383-8_62](https://doi.org/10.1007/978-3-031-92383-8_62)

Bosq, D. (2000). *Linear Processes in Function Spaces*. Springer.
[doi:10.1007/978-1-4612-1154-9](https://doi.org/10.1007/978-1-4612-1154-9)

## Licence

MIT, see `LICENSE`.
