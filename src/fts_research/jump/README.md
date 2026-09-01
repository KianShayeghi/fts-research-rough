# jump

The overnight jump equation: a scalar-on-function regression that carries its own lag.

    J_t = a + gamma J_{t-1} + ∫ delta(u) X_{t-1}(u) du + z_t' k + eta_t,
    eta_t ~ iid N(0, sigma_J²)

`J_t` is the overnight log jump and `X_{t-1}` the previous session's intraday curve. Those two
are the whole of an open-to-open return and they do not overlap — the curve is exactly what the
jump omits — so this is a regression of one half of the move on the other, with the scalar's
own persistence and any predetermined covariates alongside.

Expanding `delta` in the empirical principal components of the curves turns the functional term
into `d' psi_{t-1}`, and the model into a linear regression on `(1, J_{t-1}, psi_{t-1}', z_t')`.
Every regressor there is measurable at the close of `t - 1`: `psi` enters lagged and `z`
contemporaneously, because `z_t` summarises the session that closed as the jump was forming.
With predetermined regressors and Gaussian homoskedastic errors the conditional likelihood is
that of a linear model, so the conditional MLE is least squares and closed form. There is no
sampler to diagnose and no optimiser to fail, which is why this is the first estimate of the
equation to report and the one a Bayesian fit is checked against.

This is the model that `notebooks/overnight_jumps.ipynb` motivates and does not write down. That
notebook establishes what is in the data — the direction of the jump is unforecastable from its
own past and from the curve, its size is not, and the informative summary of a session is
usually one of the curve's values rather than a functional of it. The equation here is where
those findings become parameters: `gamma` is the persistence, `delta` the shape of the
dependence on the session, and `k` the slot the scalar summaries belong in.

The decomposition is `far.far_est.fpca`, the one the rest of the repository uses, so `psi` here
is `psi` in `far` and in `ftsa`. The relationship to `ftsa` is worth stating plainly, since the
two packages read the same data: `ftsa` forecasts the next *curve* from the sequence of curves,
and this predicts the *scalar* the cumulative-intraday-return transform deletes. Anchoring every
curve at the opening price is what erases the overnight move, so a functional time series model
of CIDR curves cannot see the jump at all, however well it forecasts.

## Modules

**`jump_eq.py`**: the estimator. `build_design` stacks the regressors with the lags aligned and
returns the block structure, `fit_mle` solves the normal equations, and `reconstruct_delta` puts
the fitted loadings back on the domain, which is the object to plot — the vector `d` is an
artefact of whichever basis the curves happened to produce.

Two things travel with the point estimate and are why `fit_mle` reports more than one.

- The standard errors come in both flavours. Under i.i.d. innovations the homoskedastic and
  Newey-West errors agree; on real jumps the volatility clusters even where the AR(1) mean is
  correct, and the sandwich is then the honest one. `tests/test_jump.py` measures both cases.
- The scores are **generated regressors** — estimated from the same curves — so errors that
  condition on the decomposition understate the uncertainty in `d` (Pagan, 1984). Nothing here
  corrects for that, and both sets are conditional on it.

`fit_lasso` is the overfitting diagnostic. The MLE shrinks nothing, and the `d` block is as wide
as the number of components retained rather than as the number carrying signal, so the fitted
`delta(u)` inherits whatever the high-order directions picked up. The lasso penalises the `d`
and `k` blocks and partials the intercept and `gamma` out first, so the penalty falls only
where the overfitting is; the gap between the two reconstructions of `delta` is a direct
reading of what unpenalised estimation costs. Cross-validation selects it for prediction, and
so — exactly as `fdaSP.SBC` does on its own designs — need not return a single exact zero: what
it reliably does is order the shrinkage.

**`jump_dgp.py`**: two simulators, and the difference between them is the point.
`simulate_jump` hands the estimator the score vectors directly, so the only thing under test is
whether the algebra is right and a failure is a bug. `simulate_curves_and_jumps` goes the way
the real problem goes: curves on an intraday grid, a signal `∫ delta X_{t-1}` integrated **on
the grid** under `core.quadrature` so the process does not secretly live in the `M`-dimensional
score space, and then `far_est.fpca` to recover the scores the estimator will actually see. The
loadings it can hope to recover are `d_k = <delta, phi_hat_k>` against the *estimated*
eigenfunctions, computed after the decomposition rather than before it — comparing an estimate
with a target in a different basis and calling the difference bias is the mistake this avoids.

Curves come from `core.gp.sample_gp`, or under `rho > 0` from `far.far_dgp.sample_far1`, so the
estimator can be exercised on serially dependent curves. It should survive them: the curve is a
predetermined regressor here, not an i.i.d. draw, and `tests/test_jump.py` checks that it does.
`snr` is the signal-to-noise convention `truncFLR.HH_dgp` and `fdaSP.SBC_sim` use, measured
against the exogenous part of the conditional mean.

## Usage

```python
import numpy as np

from fts_research.jump import jump_dgp as dgp
from fts_research.jump import jump_eq as je

grid = np.linspace(0, 1, 79)
delta = 0.8 * np.sin(np.pi * grid) * grid          # loads on the late session

sim = dgp.simulate_curves_and_jumps(n=260, a=0.05, gamma=0.15, delta=delta, snr=4.0,
                                    k=np.array([0.3, 0.0, -0.2]), grid=grid, rng=7)

fit = je.fit_mle(sim["J"], sim["psi"], sim["Z"])
print(fit.summary(hac=True))
print(fit.block("d"), sim["d_true"])               # what it recovered, and what it could

delta_hat = je.reconstruct_delta(fit.block("d"), sim["phi"])
```

The same equation on real sessions, where the curves and the jumps both come out of one price
series:

```python
from fts_research.far.far_est import fpca
from fts_research.fdaSP import intraday as itd

bars = itd.load_bars("1h", tickers=["SPY"])
X, grid, dates = itd.session_curves(bars)
jumps = itd.overnight_jumps(bars)
Xa, ya, when = itd.align_curves_to_jumps(X, dates, jumps["jump"])

scores = fpca(Xa, grid, n_components=4)["scores"]
print(je.fit_mle(ya, scores).summary(hac=True))
```

`notebooks/jump_equation.ipynb` runs both, simulated first and then real.

## Tests

`pytest tests/test_jump.py` — the design matrix against a hand-built lag, since putting `psi_t`
where `psi_{t-1}` belongs would fit a model that reads the session it is meant to predict and
would fit it *better*, so nothing downstream would complain; the estimator against a direct
solve of the normal equations and against a design whose parameters are known; the reported
log-likelihood against the Gaussian one; the two sets of standard errors under i.i.d.
innovations and under a variance that follows a regressor; the lasso for the ordering it puts
on signal against noise and for beating the MLE on `delta` where most directions are noise;
and the generated-regressor step — that `d_true` is taken against the estimated components
under the shared quadrature, and that the signal really was integrated on the grid rather than
through the scores.

## References

Pagan, A. (1984). Econometric issues in the analysis of regressions with generated regressors.
*International Economic Review* 25(1), 221-247.
[doi:10.2307/2648877](https://doi.org/10.2307/2648877)

Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and
autocorrelation consistent covariance matrix. *Econometrica* 55(3), 703-708.
[doi:10.2307/1913610](https://doi.org/10.2307/1913610)
