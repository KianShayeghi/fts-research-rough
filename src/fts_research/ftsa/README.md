# ftsa

Forecasting a functional time series through its principal component scores, after Hyndman &
Shang (2009) and the intraday application of Shang (2017a). The sample of curves is decomposed
once,

    X_t(s) = mu(s) + sum_{k=1}^{M} psi_{t,k} phi_k(s) + e_t(s),

which turns one dependent sequence of functions into `M` dependent sequences of scalars; each
score series is forecast on its own, and the curve is reassembled from the forecast scores. All
the temporal modelling happens in `R^M` and none of it in function space, which is what makes
the method as cheap as it is.

The package is named for the R package it ports, as `fdaSP` is: Shang's `ftsa::forecast.ftsm`
(Shang, 2013) is the routine reproduced here. It shares its decomposition with `far` rather
than keeping a second one — `far_est.fpca` is what both call — so the scores forecast here are
the scores
`far_est.far1_fit` regresses, and the two forecasters below differ in what they do with those
scores and in nothing else.

## Modules

**`forecast.py`**: the pipeline. `decompose` is `far_est.fpca` truncated by variance explained,
Shang's 90% rule, and it also returns the reconstruction residuals `e_t` the truncation leaves,
which the bootstrap needs. `forecast_scores` takes the two routes Shang compares:

- `method="arima"` — one univariate ARIMA per score series, the order chosen by corrected AIC
  over the `(p, d, q)` grid up to order three. The scores are uncorrelated across `k` by
  construction, so univariate models are the principled default, and this is what R's
  `auto.arima` is called score-by-score to do. R searches the grid stepwise from a starting
  model where this searches all of it, so on a near-tie the two can pick different orders; that
  is the one place a replication of Shang cannot be made exact, and it is not a Python
  limitation.
- `method="far"` — one vector autoregression on the whole score vector, at order one. That is
  Aue, Norinho & Hörmann's (2015) route, and at order one it *is* `far.far_est`'s operator:
  `forecast_scores` calls `far_est.score_operator`, so the comparison between the two methods
  is a comparison of Shang's forecaster against this repository's FAR(1) on identical
  components rather than against a second implementation of one.

`prediction_bands` is Shang's nonparametric bootstrap (2017a, §5.1), which resamples both error
sources a reconstructed curve has: the score forecast errors, drawn from each component's own
in-sample one-step residuals, and the residual functions the truncation at `M` left behind.
Neither is given a distribution, which is the point of doing it this way on returns. Past one
step the residual pool is still the one-step pool, so each draw is scaled by the ratio of the
model's `h`-step to its one-step forecast standard deviation; at `h = 1` that factor is one and
the scheme is Shang's exactly.

**`updating.py`**: the second half of both 2017 papers, after Shang & Hyndman (2011). Once the
first `m` points of a day are observed, the forecast of the rest of it is revised. Four methods
differ only in which score vector they rebuild the unobserved block from: `ols` regresses the
observed block on the same block of the components, `ridge` shrinks that toward the mean curve,
`pls` shrinks it toward the day-ahead forecast instead, and `block` moving rotates the day
boundary to the update point so the observed morning enters as the tail of a complete curve
rather than as a design matrix. `select_lambda` chooses the penalty on the training days, by
the error the update is actually scored on.

Two benchmarks come with them, and on a cumulative curve they are not optional. `hold` carries
the last observed value forward, which for a cumulative intraday return is the martingale, and
`mean` shifts the mean curve to meet it. Most of what is being predicted at any update point has
already been observed, so a method that merely reproduces the level scores well without having
forecast anything; the out-of-sample R-squared against `hold` is the R-squared for predicting
the increment, which is the part that is not already known. `anchor` goes one step further and
separates the two ways a method can fail, shifting the prediction to meet the last observed
value so that only the shape of the remainder is judged.

**`evaluation.py`**: the protocol the accuracy tables come from — expanding window, one step
ahead, re-estimated at every split, so nothing is fitted on a curve it is later scored against.
Shang's three measures, `interval_score` being Gneiting & Raftery's (2007), each reported both
averaged over the holdout and profiled along the intraday grid, since *when* in the session a
forecaster goes wrong is what a grand mean throws away. `replication_curves` builds the design
itself, from the committed 2017 S&P 500 panel through `fdaSP.intraday.session_curves`, so the
notebook and the cache cannot drift apart. `cached_rolling` keeps the forecasts — not the
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

Aue, A., Norinho, D. D. and Hörmann, S. (2015). On the prediction of stationary functional time
series. *Journal of the American Statistical Association* 110(509), 378-392.
[doi:10.1080/01621459.2014.909317](https://doi.org/10.1080/01621459.2014.909317)

Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
estimation. *Journal of the American Statistical Association* 102(477), 359-378.
[doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)

Hyndman, R. J. and Khandakar, Y. (2008). Automatic time series forecasting: the forecast package
for R. *Journal of Statistical Software* 27(3), 1-22.
[doi:10.18637/jss.v027.i03](https://doi.org/10.18637/jss.v027.i03)

Hyndman, R. J. and Shang, H. L. (2009). Forecasting functional time series. *Journal of the
Korean Statistical Society* 38(3), 199-211.
[doi:10.1016/j.jkss.2009.06.002](https://doi.org/10.1016/j.jkss.2009.06.002)

Shang, H. L. and Hyndman, R. J. (2011). Nonparametric time series forecasting with dynamic
updating. *Mathematics and Computers in Simulation* 81(7), 1310-1324.
[doi:10.1016/j.matcom.2010.04.027](https://doi.org/10.1016/j.matcom.2010.04.027)

Shang, H. L. (2013). ftsa: an R package for analyzing functional time series. *The R Journal*
5(1), 64-72. [doi:10.32614/RJ-2013-006](https://doi.org/10.32614/RJ-2013-006)

Shang, H. L. (2017a). Forecasting intraday S&P 500 index returns: a functional time series
approach. *Journal of Forecasting* 36(7), 741-755.
[doi:10.1002/for.2467](https://doi.org/10.1002/for.2467) — the paper replicated here.

Shang, H. L. (2017b). Functional time series forecasting with dynamic updating: an application
to intraday particulate matter concentration. *Econometrics and Statistics* 1, 184-200.
[doi:10.1016/j.ecosta.2016.08.004](https://doi.org/10.1016/j.ecosta.2016.08.004) — the
companion paper, which puts the same decomposition and the same two score forecasters on
half-hourly particulate matter and develops the dynamic updating at greater length.
