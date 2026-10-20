# ftsa

Forecasting a functional time series through its principal component scores, after Hyndman & Shang
[[47]][hyndman-shang-2009] and the intraday application of Shang [[73]][shang-2017a]. The sample of
curves is decomposed once,

    X_t(s) = mu(s) + sum_{k=1}^{M} psi_{t,k} phi_k(s) + e_t(s),

which turns one dependent sequence of functions into `M` dependent sequences of scalars; each
score series is forecast on its own, and the curve is reassembled from the forecast scores. All
the temporal modelling happens in `R^M` and none of it in function space, which is what makes
the method as cheap as it is.

The package is named for the R package it ports, as `fdaSP` is: Shang's `ftsa::forecast.ftsm`
(Shang [[72]][shang-2013]) is the routine reproduced here. It shares its decomposition with `far`
rather than keeping a second one — `far_est.fpca` is what both call — so the scores forecast here
are the scores `far_est.far1_fit` regresses, and the two forecasters below differ in what they do
with those scores and in nothing else.

## Modules

**`forecast.py`**: the pipeline. `decompose` is `far_est.fpca` truncated by variance explained,
Shang's 90% rule, and it also returns the reconstruction residuals `e_t` the truncation leaves,
which the bootstrap needs. `forecast_scores` takes the two routes Shang compares:

- `method="arima"` — one univariate ARIMA per score series, the order chosen by corrected AIC over
  the `(p, d, q)` grid up to order three. The scores are uncorrelated across `k` by construction,
  so univariate models are the principled default, and this is what R's `auto.arima` is called
  score-by-score to do. R searches the grid stepwise from a starting model where this searches all
  of it, so on a near-tie the two can pick different orders; that is the one place a replication of
  Shang cannot be made exact, and it is not a Python limitation.
- `method="far"` — one vector autoregression on the whole score vector, at order one. That is Aue,
  Norinho & Hörmann's [[3]][aue-norinho-hormann-2015] route, and at order one it *is*
  `far.far_est`'s operator: `forecast_scores` calls `far_est.score_operator`, so the comparison
  between the two methods is a comparison of Shang's forecaster against this repository's FAR(1) on
  identical components rather than against a second implementation of one.

`prediction_bands` is Shang's nonparametric bootstrap (2017a, §5.1), which resamples both error
sources a reconstructed curve has: the score forecast errors, drawn from each component's own
in-sample one-step residuals, and the residual functions the truncation at `M` left behind.
Neither is given a distribution, which is the point of doing it this way on returns. Past one
step the residual pool is still the one-step pool, so each draw is scaled by the ratio of the
model's `h`-step to its one-step forecast standard deviation; at `h = 1` that factor is one and
the scheme is Shang's exactly.

**`updating.py`**: the second half of both 2017 papers, after Shang & Hyndman
[[71]][shang-hyndman-2011]. Once the first `m` points of a day are observed, the forecast of the
rest of it is revised. Four methods differ only in which score vector they rebuild the unobserved
block from: `ols` regresses the observed block on the same block of the components, `ridge` shrinks
that toward the mean curve, `pls` shrinks it toward the day-ahead forecast instead, and `block`
moving rotates the day boundary to the update point so the observed morning enters as the tail of a
complete curve rather than as a design matrix. `select_lambda` chooses the penalty on the training
days, by the error the update is actually scored on.

Two benchmarks come with them, and on a cumulative curve they are not optional. `hold` carries
the last observed value forward, which for a cumulative intraday return is the martingale, and
`mean` shifts the mean curve to meet it. Most of what is being predicted at any update point has
already been observed, so a method that merely reproduces the level scores well without having
forecast anything; the out-of-sample R-squared against `hold` is the R-squared for predicting
the increment, which is the part that is not already known. `anchor` goes one step further and
separates the two ways a method can fail, shifting the prediction to meet the last observed
value so that only the shape of the remainder is judged.

**`evaluation.py`**: the protocol the accuracy tables come from — expanding window, one step ahead,
re-estimated at every split, so nothing is fitted on a curve it is later scored against. Shang's
three measures, `interval_score` being Gneiting & Raftery's [[34]][gneiting-raftery-2007], each
reported both averaged over the holdout and profiled along the intraday grid, since *when* in the
session a forecaster goes wrong is what a grand mean throws away. `replication_curves` builds the
design itself, from the committed 2017 S&P 500 panel through `fdaSP.intraday.session_curves`, so
the notebook and the cache cannot drift apart. `cached_rolling` keeps the forecasts — not the
scores — under `artifacts/cache/`, so a measure this repository has not thought of yet can be
computed without refitting.

## Usage

```python
import numpy as np

from fts_research.ftsa import evaluation as ev
from fts_research.ftsa import forecast as fc

X, grid, dates = ev.replication_curves()          # 125 sessions of CIDR curves, in per cent

out = fc.forecast_curves(X, grid, h=1, method="arima", n_boot=1000, rng=0)
print(out["n_components"], out["explained"], out["orders"])
lower, upper = out["forecast"]["bands"][0.8]
```

Scoring the two routes against each other, off the committed cache:

```python
frame = ev.cached_rolling()                       # or refresh=True to recompute
print(ev.summarise(frame))
```

```bash
python scripts/make_ftsa_cache.py --refresh
```

```python
from fts_research.ftsa import updating as up

m = up.split_index(grid, 0.3)                     # observed to 30% of the session
out = up.update_curve(X[:-1], grid, X[-1, :m], method="pls")
print(out["lam"], out["prediction"].shape)

print(up.summarise_updates(up.cached_updating()).xs(False, level="anchored").round(4))
```

`notebooks/shang_replication.ipynb` is the day-ahead replication and
`notebooks/shang_updating.ipynb` the updating half.

## Tests

`pytest tests/test_ftsa.py` — that `decompose` is `far_est.fpca` and not a second
decomposition, that the variance rule keeps the fewest components reaching its threshold, that
`method="far"` returns exactly the curve `far_est.far1_predict` returns, that the ARIMA route
recovers a known AR(1), that the bands nest and cover and that dropping Shang's second error
source narrows them, that the interval score reduces to the band width when nothing is missed
and charges `2/alpha` per unit outside when something is, and — the property the whole protocol
exists for — that the rolling-origin loop never fits on a curve before scoring it, checked by
shifting the level from the first holdout session onwards.

`pytest tests/test_updating.py` — that least squares recovers the scores exactly from any block
of a curve that really is a combination of the components, that `ridge` and `pls` reduce to it
at zero penalty and to the mean curve and the day-ahead forecast respectively at infinite
penalty, that the rotation block moving performs is a rotation, that anchoring moves a
prediction by exactly the boundary error and no more, and that the penalty is chosen against a
criterion that is not identically zero — which it is if the in-sample errors are measured on
the reconstructed curves rather than the observed ones, since a truncated curve lies in the
span of its own components. The promise the module rests on gets its own test for every method:
the unobserved part of the new day is replaced with nonsense and each forecast must come back
unchanged.

## References

The works this package rests on. Full entries, with journals and DOIs, are in
[the repository's one bibliography](../../../docs/references.md); the numbers below
link into it.

<!-- references:
  aue-norinho-hormann-2015, gneiting-raftery-2007, hyndman-khandakar-2008, hyndman-shang-2009,
  shang-hyndman-2011, shang-2013, shang-2017a, shang-2017b
-->
- **[3]** Aue, Norinho & Hörmann (2015), [On the prediction of stationary functional time
  series][aue-norinho-hormann-2015].
- **[34]** Gneiting & Raftery (2007), [Strictly proper scoring rules, prediction, and
  estimation][gneiting-raftery-2007].
- **[46]** Hyndman & Khandakar (2008), [Automatic time series forecasting: the forecast package for
  R][hyndman-khandakar-2008].
- **[47]** Hyndman & Shang (2009), [Forecasting functional time series][hyndman-shang-2009].
- **[71]** Shang & Hyndman (2011), [Nonparametric time series forecasting with dynamic
  updating][shang-hyndman-2011].
- **[72]** Shang (2013), [ftsa: an R package for analyzing functional time series][shang-2013].
- **[73]** Shang (2017a), [Forecasting intraday S&P 500 index returns: a functional time series
  approach][shang-2017a].
- **[74]** Shang (2017b), [Functional time series forecasting with dynamic updating: an application
  to intraday particulate matter concentration][shang-2017b].
<!-- /references -->

<!-- reference-links -->
[aue-norinho-hormann-2015]: ../../../docs/references.md#ref-aue-norinho-hormann-2015
[gneiting-raftery-2007]: ../../../docs/references.md#ref-gneiting-raftery-2007
[hyndman-khandakar-2008]: ../../../docs/references.md#ref-hyndman-khandakar-2008
[hyndman-shang-2009]: ../../../docs/references.md#ref-hyndman-shang-2009
[shang-hyndman-2011]: ../../../docs/references.md#ref-shang-hyndman-2011
[shang-2013]: ../../../docs/references.md#ref-shang-2013
[shang-2017a]: ../../../docs/references.md#ref-shang-2017a
[shang-2017b]: ../../../docs/references.md#ref-shang-2017b
<!-- /reference-links -->
