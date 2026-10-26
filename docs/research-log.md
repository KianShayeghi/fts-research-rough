# Research log

Open questions, dead ends and things to try, newest last. This is tracked because the reasoning
behind a discarded approach is the part that is hardest to reconstruct later and the part a
chapter has to explain; the code records what was kept, and this records what was not.

## 2026-07-30 -- the lambda degeneracy in section 3.2

Explore the variance-bias ratio in S_Y: why does variance blow up? m_tuning has been a makeshift way to deal with this, but maybe a 1/n factor needs to be introduced? We want the unpenalised S_y to overestimate theta, so the penalty drags it down. As it stands, the faithful implementation isn't doing that. Not sure if there is a mistake.

Maybe try method A? Play around with S_Y and S_B and see what is going wrong and causing the lambda estimator to degenerate. 

I need to understand better what is actually happening between the point where we're getting an endpoint estimate and degenerating to minimum lambda. Because even when we overestimate, it still happens...

The problem lies in steps 3 and 4. The 3-component basis they propose for b_simp gives us a parametric estimate of the curve. Then, we refit using m >> 3 components... Of course, this makes the bias collapse to nearly zero, the variance dominates, and drives lambda to zero!

Maybe can try plotting penalised S_Y/S_B/P_B with sliding variance and see where the minimum moves to. The idea is that the unpenalised estimator undertruncates, and the penalisation pushes down theta. So is that actually what is happening?? 

Add a naive method to HH.py, seeing what happens if we choose to circumvent the step 7 methodology.

## 2026-08-31 -- the overnight-jump project, folded in

Brought the old `overnight-jump` repository into this one. Two things came out of it: the Shang
[[73]][shang-2017a] forecaster, now `ftsa`, and the jump equation, now `jump`. Notes on what
changed and what is still open.

The FPCA front end was the duplication worth killing. Both of the old modules ran their own
skfda decomposition, so the scores driving the jump equation were not the scores `far_est.fpca`
produces and nothing would have caught the divergence. Both now call `far_est.fpca`, which also
means the "multivariate" route in Shang's comparison is literally `far_est`'s operator rather
than a second implementation of it; pulled `score_operator` out of `far1_fit` so the two share
it. That is what makes the ARIMA-vs-FAR(1) comparison in the replication mean anything.

Dropped `dgp.py` and `data_generating_process.ipynb` entirely -- the squared-exponential GP and
the truncated response were superseded by `core.gp`, `core.quadrature.trunc_int` and
`HH_dgp.sim_dataset`, and the notebook is `truncFLR.ipynb` from an earlier stage. `data_prep.py`
went too: `intraday.session_curves` already builds CIDR curves, so the 2017 file only needed a
loader, not a second pipeline.

The replication result is not what the paper's tables suggest. Shang reports the univariate and
multivariate score forecasters as close; over nine instruments the vector route is better on
eight (sign test p = 0.0391), and neither beats the flat forecast on the 2017 window. The second
half is not surprising given `overnight_jumps.ipynb` -- the curve sequence has no forecastable
conditional mean -- but the first half is a real difference from the paper and I should check
whether it survives on a longer panel than sixty sessions before leaning on it. The obvious
suspicion is that re-selecting an ARIMA order at every split is itself the problem, in which
case fixing the order across splits would close most of the gap. Worth running.

Open on `jump`:

- generated-regressor standard errors. Both `se_ols` and `se_hac` condition on the
  decomposition, so the reported uncertainty in d is too small by an amount neither measures.
  Bootstrapping over the FPCA step is the fix. It cannot overturn the current conclusions, which
  are failures to reject, but it is needed before any positive claim about delta.
- the shrinkage prior. The M = 5 against M = 12 comparison is the argument for it stated as a
  measurement; the lasso is only a diagnostic and cross-validation will not deliver sparsity
  here any more than it does for the group lasso.
- the residual magnitude is autocorrelated in both notebooks (+0.2078 for the FPCA residuals of
  the 2017 curves). Conditional mean models everywhere, conditional variance nowhere. A
  GARCH-in-jumps term is the obvious next equation, and would also explain why the interval is
  the only part of the Shang pipeline that survives contact with the data.

`data/spx_1m_2017H1.csv` has no recorded upstream source. It is a fixed historical window so it
cannot drift, but it also cannot be re-fetched, and the manifest entry says so with
`fetchable: false`. Find out where it came from and record it properly.

## 2026-08-31 -- dynamic updating, and what it actually buys

Implemented the second half of Shang (2017a/b): `ftsa.updating`, with the four methods
(`ols`, `ridge`, `pls`, block moving) and a new notebook. Notes on two things that were not
obvious going in.

The first is a bug I had for an hour and should not have had. `select_lambda` picks the penalty
by in-sample update error over the training days, and I first measured it on the reconstructed
curves `mu + Phi' psi`. Those lie exactly in the span of the components, so least squares on any
block of one recovers its scores exactly and the error is identically zero at every lambda; the
criterion selected whichever end of the grid it was handed. It has to be measured on the curves
as observed, residual included, because the part the components do not span is the whole reason
shrinkage is worth anything. There is a regression test for it now. Worth remembering as a
general trap: any criterion evaluated on a model's own fitted values instead of the data is
going to look flat.

The second is the finding. Updating works against the un-updated forecast, exactly as the papers
say, and does nothing at all against carrying the last observed value forward. What made it
interpretable was anchoring -- shifting each method so its fitted value at the last observed
point equals the value seen there. Unanchored, `pls` sits 1 to 2.4 units of variance below the
martingale; anchored, it is within 0.06 of it everywhere and the paired tests cannot separate
them. So the entire deficit was level error, on a level the method had been handed. A truncated
principal component expansion minimises L2 error across the whole domain and is under no
obligation to pass through the last observed point, which on a cumulative curve is most of the
answer.

That anchoring split is not in Shang as far as I can tell, and it is the thing I would keep.
It generalises: any time a functional method is given a partially observed curve, the level at
the boundary and the shape of the remainder are different questions, and reporting only the
total error confounds a failure to interpolate with a failure to forecast.

Open:

- the anchored `ts` row is the one to think about. Yesterday's forecast of the shape, slid to
  today's observed level, scores as well as the full penalised update over the first half of the
  session. If that survives on other instruments then the entire updating apparatus reduces to
  "take the day-ahead shape and re-anchor it", which is one line and no estimation.
- block moving on CIDR curves rotates the day boundary into the middle of every curve, and the
  transform makes that a discontinuity -- every rotated curve drops to zero at the same interior
  point. It still does respectably, better than the regression updates at 20%. I do not
  understand why yet, and the rotated-curve decomposition is worth looking at directly.
- everything here is the 2017 window. The panel sweep exists for the day-ahead forecaster and
  not for the updates; running it would cost about ten minutes and would say whether the
  ordering holds across instruments.

Fix citations and layout of repo. Bibliography section, merge packages, merge readmes, clean up notebooks and tests.

## 2026-10-15 -- the basic jump equation, step 2: the null model

`notebooks/basic_jump_equation.ipynb` now slices the simulated GBM quarter into jumps and CIDR
curves, fits the null model and tests it. The simulator and the slicing moved into
`jump.basic_dgp` without changing a draw, and the fit and diagnostics are in `jump.basic_est`.

The curve's likelihood is exact through its increments, so every estimate is closed-form and
every interval exact, with nothing asymptotic about them. Every parameter is covered.

The useful finding is the identification asymmetry. Three months fix `sigma_eps` to under half
a percent, while `alpha` and `mu(1)` have t-statistics under 1 against zero and against five
times their truth alike. The true session drift would need about forty years of days to sit
two standard errors from zero.

Thirteen diagnostics, one per assumption, all pass after Holm. Two raw p-values came in under
5% (linear drift and the portmanteau test). The notebook's own Monte Carlo shows that two in
five null datasets do that at this design, so it is expected rather than a warning.

One thing I got wrong on the way and that is worth remembering: calibrating the diagnostics by
simulating from the *fitted* null fails every dataset. The fitted mean is the free `mu_hat`,
which carries the sample's noise, so data drawn from it fail the linear-drift test by
construction. Calibration has to simulate from the data-generating process.
`basic_est.simulate_null` now says so.

Open:

- the power of the test of `beta = 0` against sample length and effect size. Over 63 days only
  a large `beta` can be visible, because the jump's mean is not identified and `beta` has to
  show up in the jump's variance. That is the next step's first question.
- the free `mu` costs 390 parameters to describe noise, and the F test does not reject
  linearity. The basic model could take `mu(s) = mu_1 s` instead. It changes nothing in the
  jump equation, but it would matter once the jump enters the curve (v2).

Corrections, made the same day after checking the notebook's numbers:

- "Two in five null datasets do that" was wrong for *two* raw failures. Two in five is the
  rate of at least one. The same Monte Carlo puts two or more at 14%: rarer, still not a
  warning.
- The baseline for the basic model was on the wrong days. The basic model regresses `J_t` on
  `X_{t-1}` and can only use days 2-63, so the null's jump log-likelihood has to be computed on
  those too: -38.2903, not -38.4060 on all 63.
- The out-of-sample benchmark is the expanding mean's forecast error (0.4630 over days 22-63,
  `basic_est.expanding_mean_forecast`), not the in-sample RMSE.
- The test of `beta = 0` is exact as an F test, because the jump equation is a Gaussian
  regression given the curves. The chi-squared likelihood ratio is oversized at 62 days, from
  5.5% with one component to 8.2% with eight (`basic_est.lr_test_size`).
- Wording: `alpha` and `mu` are identified, just imprecisely estimated, and `nu` is the drift
  of the price (8% a year), not of its log (6.3%).

## 2026-10-16 -- jumps that read the end of the session

New DGP for the basic model: `jump.window_dgp`, with `notebooks/window_jump_dgp.ipynb`. The
GBM's sessions are kept and its nights replaced, so that each jump depends on the
least-squares slope of the last 20% of the session before. A dial `kappa` runs from the null
(0) to a night that carries on that slope in full (1).

Decisions, and why:

- **Least-squares slope rather than the endpoint return `(X(1) - X(0.8)) / 0.2`.** The endpoint
  return is the GBM drift MLE, but its `beta` is two point masses. The slope has an L2
  `beta(u) = kappa T 12 (u - 0.9) / 0.2^3` on the window, so the DGP is exactly the basic
  model with a target a fit can be held against.
- **Free `kappa`, noise at the GBM's size.** Taken literally (`kappa = 1`), the proposal gives
  `R^2 = 0.60` and `sd(J) = 0.79%` against 0.50%. That is far stronger than any market will
  show, so the notebook runs `kappa` in {0, 0.25, 0.5, 1}, with `R^2` = 0, 0.08, 0.27, 0.60.
- **Realised-variance switch, off by default.** Under a constant `sigma` the window's realised
  variance is `chi^2_77 / 77` noise and makes `eta` heteroskedastic. A real variance channel
  needs stochastic session volatility, which is not built yet.

Open:

- fit `beta(u)` by PC regression and see how badly the discontinuity at 0.8 is smeared;
- power of the F test of `beta = 0` against `kappa` and `n`;
- a stochastic-volatility version, so that late-session realised variance informs the night.

## 2026-10-19 -- what `bhat` actually is

Correction to the 2026-10-16 entry and the first version of the notebook, which called `bhat`
"the drift of the last 20%". It is a statistic, the least-squares slope of the observed curve,
and almost none of it is the drift: `bhat = 100 m + sigma_eps xi`, with `xi` a
parabola-weighted average of the window's Brownian increments, sd 2.43% per session against a
drift of 0.02%. The drift is constant across days and predicts nothing. All the signal in the
DGP is `xi_{t-1}`, so what is built in is late-session momentum in the realised path.

Inference on `bhat` (`window_dgp.window_slope_sd`, `window_slope_ci`; section 2 of the
notebook):

- it is exactly Gaussian and i.i.d. over days, with variance `sigma_eps^2 sum g^2 ds`
  (about `1.2 sigma_eps^2 / w`);
- the textbook OLS interval for the slope treats the 79 points as independent. It is 11.2
  times too narrow and covers the drift 13.6% of the time. Never use it on a Brownian curve;
- the Brownian interval covers 95%, whether `sigma_eps` is known or taken from the window's
  realised variance on 77 df (95.03% over 500,000 sessions). Either way it is about +-4.8% per
  session wide;
- none of this enters the jump equation. The jump is defined by `bhat` itself, which is
  observed exactly, so the regression of `J_t` on `bhat_{t-1}` has no errors-in-variables
  problem.

A bug found on the way: `window_slope_ci` first counted the degrees of freedom with
`count_nonzero` on the return weights, whose pre-window entries are 1e-14 rather than zero. It
used t(389) instead of t(77), and its coverage was 94.7%. Fixed, and the test now checks the
quantile directly, since coverage at n = 20,000 could not see a 0.3-point shortfall.

## 2026-10-21 -- `dhat` renamed `bhat`

The window slope is now `bhat` (`b_mu` for its mean) in the notebook, `window_dgp` and the
docs, and the entries above use the new name. "d" was short for "drift", which it is not.
`beta` was not an option, because `beta(u)` is the basic model's coefficient function, so it is
`b`, as in the slope of a fitted line `gamma + b s`.

## 2026-10-22 -- the basic model fitted to the window DGP

Section 5 of `notebooks/window_jump_dgp.ipynb` fits `J_t` on the first `M` FPC scores of
`X_{t-1}` by OLS (the conditional MLE) and tests `beta = 0` with the exact F test.

- The signal is not in the leading components. For Brownian curves the components are
  `sqrt(2) sin((k - 1/2) pi u)`, nearly flat near the close, so a late-window slope loads on
  components 3 to 8. The first component holds 81% of the curve variance and 0.8% of the
  signal. The 95%-of-variance rule gives `M = 4`, which captures 43% of the signal; `M = 8`
  captures 98%. On real data `M` has to be chosen by a criterion, not by variance explained.
- With `M = 8` and 62 days, the F test rejects at `kappa = 0.5` and `1`, not at `kappa = 0.25`
  (true `R^2` 0.08), and not at `kappa = 0`.
- `beta_hat` estimates the projection of `beta` onto the sample components, which rounds off the
  discontinuity at 0.8 and wiggles before it. The fit finds the late effect and its sign, but
  not the window's start.

Open: power of the F test against `kappa`, `n` and `M`; choosing `M` without knowing `beta`.

## 2026-10-22 -- truncFLR and fdaSP on the window DGP

Section 6 of `notebooks/window_jump_dgp.ipynb` compares the untruncated fit (8 components),
truncFLR methods A and B (told the window is at the end of the session) and the fdaSP group lasso
(10-fold CV) over 100 quarters of 62 days, at `kappa = 0.5` and `1`.

- Method A is the most accurate (median relative ISE 0.152 at `kappa = 1`) and the only method that
  finds the window: median `u_hat` 0.795, within 0.02 of 0.8 in 77% of quarters. Its tail is
  long: the worst quarter at `kappa = 0.5` has relative ISE 485.5.
- Method B fails in almost every quarter (median `u_hat` 0.99). This `beta` changes sign inside
  the window, so truncating the whole-session pilot to part of the window predicts worse than
  nothing. Scanning from `u = 1`, the RSS dips to a shallow local minimum in the last minutes
  (`u = 0.990` on the section 6 quarter) and then climbs far above the total sum of squares,
  peaking at `u = 0.9`, so the first-interior-minimum rule stops in the dip. This is a failure of
  the rule on a sign-changing coefficient, not of the section 3.2 penalty, since every `lambda`
  gives the same answer.
- The group lasso is second on median error (0.231) but zeroes any part of the session in only
  5% of quarters: CV tunes for prediction.
- The untruncated fit loses on median error but has the best worst case.

## References

<!-- references: cited -->
- **[73]** Shang (2017a), [Forecasting intraday S&P 500 index returns: a functional time series
  approach][shang-2017a].
<!-- /references -->

<!-- reference-links -->
[shang-2017a]: references.md#ref-shang-2017a
<!-- /reference-links -->
