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
[[46]][shang-2017a] forecaster, now `ftsa`, and the jump equation, now `jump`. Notes on what
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

## References

<!-- references: cited -->
- **[46]** Shang (2017a), [Forecasting intraday S&P 500 index returns: a functional time series
  approach][shang-2017a].
<!-- /references -->

<!-- reference-links -->
[shang-2017a]: references.md#ref-shang-2017a
<!-- /reference-links -->
