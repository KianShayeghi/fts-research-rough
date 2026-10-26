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
  condition on the decomposition understate the uncertainty in `d` (Pagan [[64]][pagan-1984]).
  Nothing here corrects for that, and both sets are conditional on it.

`fit_lasso` is the overfitting diagnostic. The MLE shrinks nothing, and the `d` block is as wide
as the number of components retained rather than as the number carrying signal, so the fitted
`delta(u)` inherits whatever the high-order directions picked up. The lasso penalises the `d`
and `k` blocks and partials the intercept and `gamma` out first, so the penalty falls only
where the overfitting is; the gap between the two reconstructions of `delta` is a direct
reading of what unpenalised estimation costs. Cross-validation selects it for prediction, and
so — exactly as `fdaSP.SBC` does on its own designs — need not return a single exact zero: what
it reliably does is order the shrinkage.

**`covariates.py`**: the `z_t` block, one column per published claim about overnight moves.
`overnight_jumps.ipynb` finds that the block matters and the functional term does not, which
makes what goes in it the empirical question the equation turns on, so the candidates are
written down here with the paper each one comes from rather than assembled in a notebook cell.
`session_covariates` extends `fdaSP.intraday.session_summary` with the closing and opening
fractions of the session, the two realised semivariances and their signed difference, and
volume against its own trailing mean; `foreign_covariates` takes a session from a market that
trades through the domestic night and cuts it at a wall-clock time; `jump_design` does the
lagging; `panel_design` stacks it across instruments.

Two things in there are load-bearing. The **timing**: every column describes a session that
had closed, or a foreign session truncated before the domestic open, so the whole design
existed at the moment being forecast. An off-by-one the other way builds a model that reads
the session it is meant to predict, fits better for doing so, and fails nothing downstream --
which is why `tests/test_covariates.py` checks the alignment against prices whose answer is
arithmetic. The **cutoff**: the overnight window is not simultaneous across exchanges, and a
European close is four and a half hours after a New York open, so a same-day foreign return
taken whole is a look-ahead into most of the session it is supposed to forecast.
`foreign_covariates` therefore takes `until` rather than a session, and will take None only so
that the size of that look-ahead can be measured rather than assumed small.

`benjamini_hochberg` is the last piece. A screen over nine instruments and a dozen covariates
is a hundred-odd tests, five of which clear the five per cent level by chance, so an
uncorrected table of stars says nothing at all.

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

**`joint_dgp.py`** and **`joint_est.py`**: the jump and the session after it, as one model.
`jump_eq` takes the curve as given and asks what it says about tonight's jump; the joint model
also writes down where the curve comes from, so that the two can be simulated together and
every identity between them holds on the simulated path:

    J_t    = alpha + rho J_{t-1} + ∫ beta(u) X_{t-1}(u) du + eta_t,
    X_t(s) = mu(s) - gamma_t G(s) J_t + ∫ Psi(s, r) X_{t-1}(r) dr + eps_t(s),
    eps_t(s) = ∫_0^s sigma(v) dW_t(v),

with `P_t(s) = P_{t-1}(1) + J_t + X_t(s)` putting the log-price curve back together.
`gamma_t G(s)` is the share of the jump the session has given back by `s` -- the gap filling
-- and the curve equation is a FAR(1) with the jump as an exogenous regressor
([[25]][damon-guillas-2002]). `X_t(0) = 0` needs `mu(0) = G(0) = Psi(0, .) = 0` and
`eps_t(0) = 0`; `JointModel` refuses a parameter set that breaks the first three, and the
fourth holds because the curve noise is an integral of Brownian motion, drawn exactly on the
grid from independent increments whose variances follow a U-shaped `sigma^2`
([[1]][andersen-bollerslev-1997]).

`joint_dgp` holds the theory as well as the simulator, so a simulation is checked against what
it should produce rather than against itself. Substituting the jump into the curve equation
writes the pair as one autoregression on `(J_t, X_t)`, and stationarity is a property of that,
not of the two equations separately: with `Psi = 0` the jump's effective persistence is
`rho - gamma <beta, G>`, and a random `gamma_t` makes it a random-coefficient autoregression
whose second moments need the mean-square operator inside the unit circle
([[63]][nicholls-quinn-1982]). `stationarity` reports both radii, `moments` solves for the
stationary mean and covariance on the grid, `calibrate` uses those to match four target
moments exactly, and `scale_to_r2` sets the signal strength of `beta` by the population
R-squared it implies, which is the fair way to compare two shapes of it. `population_fit` is
the regression an infinite sample would converge to, for any set of regressors built from
yesterday's curve, so a misspecified model is costed without sampling error in the way;
`step_functional` builds the two regressors a step coefficient can mean, on the curve's level
or on its returns; and `forecast` is the model's conditional expectation of tomorrow given
today, which is the plainest way to see what a parameter does.

`joint_est` fits the two equations one at a time, which loses nothing: the system is recursive
and its errors independent, so the conditional likelihood factorises and `J_t` is exogenous in
the curve equation. The curve equation is regressed at every grid point on the jump and the
scores of yesterday's curve ([[82]][yao-muller-wang-2005]), and the slope on the jump, `b(s)`,
is the unrestricted gap-fill profile. `gamma` is read off it by L2 projection onto `G`, and
by generalised least squares on its increments -- the Brownian noise has independent
increments, so that is the Gaussian MLE given the rest, and in the notebook's Monte Carlo it
has about a fifth less variance. `gamma_variance` reads `Var(gamma_t)` off the
heteroskedasticity a random `gamma_t` leaves. `shape_test` asks whether `b` is a multiple of
`G` at all, by the L2 norm of its orthogonal part, a weighted sum of chi-squares under the
null ([[76]][shen-faraway-2004]) whose exact tail `imhof_sf` computes ([[48]][imhof-1961]).
`onset_profile` locates the onset of a step `beta`, on the curve's level or on its returns,
and `nesting_test` asks which of the two the jump follows, by regressing it on both: each
t-test there is a J test of one model against the other ([[26]][davidson-mackinnon-1981]).

Two findings belong with the code, because they decide how it should be used.
`notes_beta(kind="return")` exists because a coefficient on the *level* of a cumulative curve
and one on its *returns* are different models: a step on the level weights every return before
the onset equally and is a kink in the weight on returns, while a step on the returns is a
switch. `docs/jump-model-choices.md` explains the difference and
`notebooks/jump_equation_v2.ipynb` measures it. And when the model does meet real curves, they
only satisfy its identities when built by `fdaSP.intraday.session_curves(...,
anchor="open")`; under the default anchor the first bar's return is in neither the curve nor
the jump.

**`basic_dgp.py`** and **`basic_est.py`**: the GBM of *Basic Jump Equation*, and the null
model it is a special case of. These go with `notebooks/basic_jump_equation.ipynb`, which
builds the basic model up one step at a time. So far it covers the null and nothing past it:
there is no regression of a jump on a curve here yet.

`basic_dgp` simulates a geometric Brownian motion observed on the bars of each session. One
session is the unit of time, and the night has length `T`, so session `t` runs over
`[a_t, a_t + 1]` with `a_t = t (1 + T) - 1`. The prices are drawn exactly at the observation
times, so they carry no discretisation error. `split_sessions` cuts them into the notes'
two pieces, both in percent: the jump `J_t = P_t(0) - P_{t-1}(1)` and the CIDR curve
`X_t(s) = P_t(s) - P_t(0)`. `gbm_null_parameters` maps the GBM onto the null model:

- `alpha = 100 m T`;
- `sigma_eta = 100 sigma sqrt(T)`;
- `mu(s) = 100 m s`;
- `sigma_eps = 100 sigma`.

That mapping is the truth every estimate in the notebook is held against. `T` is variance
time, not clock time. With the same `sigma` running through the night, the clock-time ratio
of 17.5 to 6.5 hours would give the night almost three quarters of each day's variance.

`basic_est` fits the null: `J_t = alpha + eta_t`, and `X_t(s) = mu(s) + sigma_eps W_t(s)` with
`mu` free apart from `mu(0) = 0`. The fit goes through the curve's increments. Those are
independent Gaussians with variance `sigma_eps^2 ds`, so the likelihood is exact on the grid
and every estimate has a closed form:

- `alpha_hat` is the mean jump;
- `mu_hat` is the mean curve;
- `s_eps^2` is realised variance pooled over days.

Every interval is exact (t, chi-squared or F), and `tests/test_basic_jump.py` measures the
coverage of each.

`null_diagnostics` tests each assumption of the null on its own, so a failure says which
assumption failed. The checks are:

- the jumps are normal: Shapiro-Wilk ([[75]][shapiro-wilk-1965]) and Jarque-Bera
  ([[50]][jarque-bera-1987]);
- the jumps and their squares are independent over days: Ljung-Box ([[57]][ljung-box-1978]);
- the jump is independent of the same day's session;
- the volatility is constant through the session: Bartlett ([[9]][bartlett-1937]), across
  half-hour buckets. This is where the U shape of [[1]][andersen-bollerslev-1997] would show;
- the increments within a session are independent and Gaussian;
- the covariance is `sigma_eps^2 min(s, s')`: an exact Monte Carlo test, since the distance
  it uses is invariant to `mu` and `sigma_eps`;
- the drift is linear, as the GBM requires and the null does not: an exact F test;
- the curves are independent over days: the portmanteau test of [[29]][gabrys-kokoszka-2007].

With thirteen checks at 5%, a dataset drawn from the null fails at least one of them about two
times in five, so each row also carries the Holm ([[41]][holm-1979]) adjusted p-value, which is the
column to read a verdict from. `goodness_of_fit` reports the numbers the basic model will be
compared against: both maximised log-likelihoods with their AIC and BIC, the jump's in-sample
root mean squared error about its mean, and the coverage of the fitted 95% predictive band.
The basic model regresses `J_t` on `X_{t-1}` and so can use only days `2, ..., n`, and the
null has to be scored on those same days. Out of sample, the benchmark is
`expanding_mean_forecast`, the null's own forecast of each jump from the jumps before it.
Its error runs above the in-sample one by about `sqrt(1 + 1/m)`. And because the jump
equation, given the curves, is a Gaussian linear regression, the test of `beta = 0` is an
exact F test. `lr_test_size` gives the true size of the chi-squared likelihood ratio in its
place, which at a quarter's worth of days rises from 5.5% with one component to 8.2% with
eight.

The main finding sits in the fit rather than in the code. The two halves of the model are
estimated at very different rates. `sigma_eps` has `(n - 1) K` degrees of freedom and is
fixed to a fraction of a percent within one quarter. `alpha` and `mu(1)` have standard errors
of `sigma / sqrt(n)` and cannot be told from zero in the same quarter. Whatever the basic model
finds in the jump equation, it has to find it against a mean that the data barely pin down.

**`window_dgp.py`**: data for the basic model, with a known answer. It keeps the GBM's
sessions from `basic_dgp` and replaces its nights with jumps that read the end of the session
before:

    J_t = 100 m T + kappa T (bhat_{t-1} - 100 m) + 100 sigma sqrt(T) Z_t,

where `bhat_{t-1}` is the least-squares slope of `X_{t-1}` over the last 20% of the session.
`bhat` is a statistic of the observed curve, not the drift. It is exactly
`N(100 m, sigma_eps^2 sum g^2 ds)`, with sd 2.43% per session against a drift of 0.02%, so it
is almost entirely the window's realised trend. That trend is what the jump responds to: the
DGP is late-session momentum. `window_return_weights` writes `bhat` as a parabola-weighted
average of the window's returns. `window_slope_sd` and `window_slope_ci` give its exact
standard deviation and a valid interval for the drift. The textbook OLS standard error of the
slope treats a Brownian path's deviations from the line as independent, and it is about eleven
times too small.

`kappa = 0` is the null, and `kappa = 1` is a night that carries on the window's slope in full
(`R^2 = 0.60`). The slope is a linear functional of the curve, so this is exactly the basic
model, with `beta(u) = kappa T 12 (u - 0.9) / 0.2^3` on `[0.8, 1]` and zero before it
(`window_beta`). Three properties hold:

- the overnight path is never simulated, because its Gaussian end point is all the jump is;
- the curves are unchanged, and with the same seed they are `gbm_sessions`' own;
- a burn-in curve, drawn after the sample, gives the first jump its previous session.

`window_jump_truth` gives the moments of `bhat` and of the jump exactly on the grid.
`vol="realised"` sets the night's variance from the window's realised variance instead. Under
a constant `sigma` that carries almost nothing and makes the noise heteroskedastic, so it is
off by default.
`notebooks/window_jump_dgp.ipynb` presents the model, plots the data and fits the basic
model to them by OLS on principal component scores, then compares that fit with
`truncFLR` (methods A and B) and `fdaSP`'s group lasso over 100 simulated quarters.

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

The scalar block, which on real jumps is the part that carries the signal:

```python
from fts_research.fdaSP import intraday as itd
from fts_research.jump import covariates as cov
from fts_research.jump import jump_eq as je

bars = itd.load_bars("1h")
london = cov.foreign_covariates(bars[bars["ticker"] == "^FTSE"], until="13:00")
design = cov.jump_design(bars[bars["ticker"] == "SPY"], foreign=london)

names = ["late", "rvol", "foreign_ret"]
fit = je.fit_mle(design["jump"].to_numpy(), Z=design[names].to_numpy())
print(fit.summary(hac=True))
print(fit.wald("k", hac=True))                     # are the three of them jointly zero
```

`notebooks/jump_covariates.ipynb` runs that screen across the panel, with
`benjamini_hochberg` over the whole table of tests rather than a five per cent level over
each of them.

The null model of *Basic Jump Equation*, on the three months of GBM prices it is first
studied on:

```python
import numpy as np

from fts_research.jump import basic_dgp as bd
from fts_research.jump import basic_est as be

path = bd.gbm_sessions(n_days=63, n_bars=390, S0=100.0, sigma=0.01, T=0.25, nu=2.5e-4,
                       rng=np.random.default_rng(0))
J, X = bd.split_sessions(path["S"], S0=100.0)

fit = be.fit_null(J, X, path["s"])
truth = bd.gbm_null_parameters(sigma=0.01, T=0.25, nu=2.5e-4)
print(be.parameter_table(fit, truth))              # estimates, exact intervals, truth
print(be.null_diagnostics(fit, J, X, rng=np.random.default_rng(1)))
print(be.goodness_of_fit(fit, J, X))               # the baseline for the basic model
```

`notebooks/basic_jump_equation.ipynb` runs it with every figure and reads every table.

The same GBM, with nights that carry on half of the late-session drift:

```python
from fts_research.jump import window_dgp as wd

sim = wd.window_jump_sessions(n_days=63, kappa=0.5, rng=np.random.default_rng(0))
truth = wd.window_jump_truth(sim["s"], kappa=0.5)
print(truth["r2"], truth["kappa_T"])               # 0.27 of the jump is predictable
J, X = bd.split_sessions(sim["S"], sim["S0"])      # the same J and X as sim["J"], sim["X"]
```

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

`pytest tests/test_covariates.py` — the alignment first, against five sessions whose prices
make the answer arithmetic: the row for a jump must carry the *previous* session's summaries
and not that session's own, and must carry the previous *jump* as its lag. Then the
measurements, each against a hand computation on the same bars — that the two realised
semivariances add back to the realised variance and that their difference has the sign of the
session, that the closing and opening returns are the bars they claim to be, that relative
volume excludes the session it describes, and that a cutoff really drops the bars after it
rather than down-weighting them. Then the multiple-testing step, against a four-test example
worked by hand and against uniform p-values, where it must reject almost nothing.

`pytest tests/test_joint.py` -- the model before the estimators. The three identities on a
simulated path, and every curve and jump recomputed from its own equation with the simulator's
draws; a parameter set that would move the open refused; the curve noise's covariance against
Brownian motion's, and its increments independent; the mean-square radius against a dense
Kronecker eigendecomposition, and a pair of equations each stationary on its own that explodes
together; the stationary mean against both expectation equations, and the solved moments
against 100,000 simulated days; calibration hitting its targets exactly. Then the estimators:
Frisch-Waugh consistency, exact recovery with the curve noise switched off, every scalar in a
large sample, `Var(gamma_t)` found where it exists and not invented where it does not, the
onset found on returns, Imhof against the chi-squared and against simulation, and -- marked
slow -- the shape test holding its size and the gamma intervals holding their coverage.

`pytest tests/test_basic_jump.py` -- in four groups, one per thing the notebook leans on.

- **The simulator.** It reproduces, bit for bit, the path the notebook was first run on. Its
  jumps and curves have the moments `gbm_null_parameters` claims, which is what makes the
  truth columns true.
- **The slicing.** The curves start at zero, and the jumps and session returns rebuild every
  close. The same prices written as one-minute bars and passed through
  `fdaSP.intraday.session_curves(anchor="open")` and `overnight_jumps` give the same curves
  and jumps, so moving to market data will change the data and nothing else.
- **The fit.** Its closed forms are checked on three hand-written curves over an uneven grid,
  and its log-likelihoods against `scipy.stats.norm`. It recovers a nonlinear `mu` in a large
  sample, and each of its five exact intervals covers 95% over 500 replications.
- **The diagnostics.**
  - The Ljung-Box, Bartlett and Holm helpers agree with statsmodels and scipy.
  - The exact checks reject 5% of the time on null data, and the portmanteau test no more
    than that.
  - The fitted band covers 95% of null data.
  - Each check rejects, after the Holm adjustment, an alternative that breaks only its own
    assumption: a U-shaped volatility, AR and GARCH jumps, t-distributed jumps and
    increments, a jump tied to its session, MA increments, Ornstein-Uhlenbeck curves, a
    sinusoidal drift and AR curves.

`pytest tests/test_window_jump.py` -- seven tests, one per claim `window_dgp` makes:

- the slope weights give the least-squares slope, and the return weights give the same number;
- the curves are the GBM's and the restitched prices slice back exactly;
- each jump reads the previous curve, and `beta` integrates to the same signal;
- `kappa = 0` is the null;
- the variance of the jump and its slope on `bhat` match `window_jump_truth`;
- `bhat` has the stated sd, both intervals cover 95%, the plug-in one uses the window's 77
  degrees of freedom, and the textbook OLS standard error is far too small;
- the realised-variance switch is unbiased and bad settings are refused.

## References

The works this package rests on. Full entries, with journals and DOIs, are in
[the repository's one bibliography](../../../docs/references.md); the numbers below
link into it.

<!-- references:
  pagan-1984, newey-west-1987, damon-guillas-2002, andersen-bollerslev-1997, nicholls-quinn-1982,
  yao-muller-wang-2005, shen-faraway-2004, imhof-1961, davidson-mackinnon-1981, shapiro-wilk-1965,
  jarque-bera-1987, ljung-box-1978, bartlett-1937, gabrys-kokoszka-2007, holm-1979
-->
- **[1]** Andersen & Bollerslev (1997), [Intraday periodicity and volatility persistence in
  financial markets][andersen-bollerslev-1997].
- **[9]** Bartlett (1937), [Properties of sufficiency and statistical tests][bartlett-1937].
- **[25]** Damon & Guillas (2002), [The inclusion of exogenous variables in functional
  autoregressive ozone forecasting][damon-guillas-2002].
- **[26]** Davidson & MacKinnon (1981), [Several tests for model specification in the presence of
  alternative hypotheses][davidson-mackinnon-1981].
- **[29]** Gabrys & Kokoszka (2007), [Portmanteau test of independence for functional
  observations][gabrys-kokoszka-2007].
- **[41]** Holm (1979), [A simple sequentially rejective multiple test procedure][holm-1979].
- **[48]** Imhof (1961), [Computing the distribution of quadratic forms in normal
  variables][imhof-1961].
- **[50]** Jarque & Bera (1987), [A test for normality of observations and regression
  residuals][jarque-bera-1987].
- **[57]** Ljung & Box (1978), [On a measure of lack of fit in time series models][ljung-box-1978].
- **[62]** Newey & West (1987), [A simple, positive semi-definite, heteroskedasticity and
  autocorrelation consistent covariance matrix][newey-west-1987].
- **[63]** Nicholls & Quinn (1982), [*Random Coefficient Autoregressive Models: An
  Introduction*][nicholls-quinn-1982].
- **[64]** Pagan (1984), [Econometric issues in the analysis of regressions with generated
  regressors][pagan-1984].
- **[75]** Shapiro & Wilk (1965), [An analysis of variance test for normality (complete
  samples)][shapiro-wilk-1965].
- **[76]** Shen & Faraway (2004), [An F test for linear models with functional
  responses][shen-faraway-2004].
- **[82]** Yao, Müller & Wang (2005), [Functional linear regression analysis for longitudinal
  data][yao-muller-wang-2005].
<!-- /references -->

<!-- reference-links -->
[andersen-bollerslev-1997]: ../../../docs/references.md#ref-andersen-bollerslev-1997
[bartlett-1937]: ../../../docs/references.md#ref-bartlett-1937
[damon-guillas-2002]: ../../../docs/references.md#ref-damon-guillas-2002
[davidson-mackinnon-1981]: ../../../docs/references.md#ref-davidson-mackinnon-1981
[gabrys-kokoszka-2007]: ../../../docs/references.md#ref-gabrys-kokoszka-2007
[holm-1979]: ../../../docs/references.md#ref-holm-1979
[imhof-1961]: ../../../docs/references.md#ref-imhof-1961
[jarque-bera-1987]: ../../../docs/references.md#ref-jarque-bera-1987
[ljung-box-1978]: ../../../docs/references.md#ref-ljung-box-1978
[newey-west-1987]: ../../../docs/references.md#ref-newey-west-1987
[nicholls-quinn-1982]: ../../../docs/references.md#ref-nicholls-quinn-1982
[pagan-1984]: ../../../docs/references.md#ref-pagan-1984
[shapiro-wilk-1965]: ../../../docs/references.md#ref-shapiro-wilk-1965
[shen-faraway-2004]: ../../../docs/references.md#ref-shen-faraway-2004
[yao-muller-wang-2005]: ../../../docs/references.md#ref-yao-muller-wang-2005
<!-- /reference-links -->
