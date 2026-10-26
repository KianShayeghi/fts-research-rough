# fts-research

[![tests](https://github.com/KianShayeghi/fts-research/actions/workflows/ci.yml/badge.svg)](https://github.com/KianShayeghi/fts-research/actions/workflows/ci.yml)

Advanced Modelling and Forecasting of Functional Time Series.

Research code for scalar-on-function regression,

    y_i = ∫_T x_i(t) psi(t) dt + eps_i,

where the coefficient function `psi` is zero on part of its domain, so the estimation problem
includes finding *where*. Two approaches are implemented: `truncFLR`, which assumes the informative
set is a single interval and estimates its endpoint (Hall & Hooker [[37]][hall-hooker-2016]), and
`fdaSP`, which lets it be any union of intervals and induces the sparsity by penalisation
(Stefanucci, Bernardi & Canale [[77]][stefanucci-bernardi-canale-2025]). `far` supplies serially
dependent curves, so both can be run without the i.i.d. assumption, and also estimates and tests a
FAR(1) from observed curves; `bench` runs all three methods over a matched grid of designs.

Two further packages take the same curves in the other direction, forward in time rather than onto
a scalar. `ftsa` forecasts the next curve through its principal component scores, which is Shang's
[[73]][shang-2017a] method, scores it against the FAR(1) on identical components, and revises it
through the day as the session is observed; `jump` is the model for the quantity that forecaster
cannot see, the overnight move between one session's close and the next one's open.

## What is here

**Three corrections to Hall & Hooker's tuning scheme.** Their §3.2 selects the truncation
penalty `lambda` by a surrogate criterion, and run as the displays are written it returns the
smallest candidate on the grid regardless of the data. Steps 5 and 5' define that criterion's
two surrogates as conditional expectations, so what they should equal is checkable rather than
arguable, and three of the four displays evaluating them are out by a factor. The variance
terms of `S_Y^B`, `S_b^B` and `S_b^A` are `n` times too large, a coefficient fitted to `n`
curves having variance `sigma^2/(n tau_j)` where `tau_j` is an eigenvalue of the covariance
operator; `S_b^B`'s is also governed by the full-interval `tau_j^1` that `b_check*` was
estimated from rather than the truncated `tau_j^theta`, which is what makes it diverge as
`theta -> 0` and `S_b` monotone. Separately, equation (3.4) writes the penalty as
`n*lambda*theta^2` while step 6 pairs `S_Y` with a bare `lambda`, so `lambda` is `n` times too
small to move `V(lambda)`, which collapses to order `sigma^2/n^2`; and step 7's Gaussian is
then twenty-five times narrower than the grid it is integrated on, where by Poisson summation
the trapezium rule returns `h*S_b(theta_lambda)/sqrt(2 pi V)` instead of an average of `S_b`.
Corrected, `lambda_hat` is interior on 23 of 25 seeds and method B's endpoint loses its bias
and two-thirds of its RMSE. `notebooks/truncFLR_lambdadegen.ipynb`.

**With the tuning corrected, methods A and B are indistinguishable.** The paper's Table 1
reports that method A carries less bias in the endpoint than method B, and that is what this
repository found while §3.2 ran as displayed. It is the tuning rather than the method: over 100
paired replications (n = 200, snr = 26.25, true `u` = 0.8), method A now recovers the endpoint
with bias `-0.0023` and method B with `-0.0017`, and the paired differences are within noise
(`t = -0.39` in absolute endpoint error, `t = +0.80` in integrated squared error). Method B's
under-truncation was the cost of a penalty pinned at the floor of its candidate grid, and it
buys back method A's accuracy for one principal components analysis instead of a fresh one at
every candidate endpoint. Both beat the untruncated FPCA pilot, which over 25 replications of
the same design averages `0.0219` against method A's `0.0094` and method B's `0.0084`.
`notebooks/truncFLR_methodA.ipynb`, `notebooks/truncFLR_lambdadegen.ipynb`.

**The truncation model does not beat the group lasso, even on its own ground.** Over 50 paired
replications the locally sparse estimator has the lower median ISE on both a single-window
coefficient function, which is exactly Hall & Hooker's model, and a two-window one, which is
not. On the single window it is ahead of method B on 60% of draws, which a signed-rank test
does not separate from a tie (`p = 0.099`); on the pair it wins outright. What the truncation
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
outputs committed under `artifacts/reference/`. `notebooks/LocSpaFLR_validation.ipynb`.

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

**Shang's two score forecasters are not close, and the multivariate one wins.** The paper
reports that forecasting each principal component score with its own ARIMA and forecasting the
whole score vector with one autoregression differ little. Rebuilt on 125 sessions of five-minute
S&P 500 curves and on nine instruments of the five-minute panel, they do not: the vector route
is the more accurate on eight of the nine, which a sign test puts at `p = 0.0391`, and it is
ahead on the 2017 window too. The margins are not small. Measured as out-of-sample R-squared
against the flat forecast, the univariate route runs from `-0.5812` to `+0.0729` and the vector
route from `-0.1050` to `+0.1351`. Fitting `M` separate models throws away every cross-lag
relation between the scores and re-selects an order at every split; at order one the vector
route is exactly this repository's FAR(1), so the comparison is between two uses of one
decomposition rather than between two implementations. `notebooks/shang_replication.ipynb`.

**Neither forecaster beats predicting nothing.** The paper's accuracy tables have no floor in
them, and with one added neither route clears it: on the 2017 window both are beaten by the
flat forecast, which predicts each session by the mean of those before it, and by the zero
curve, which predicts no move at all. Paired over the 50 held-out sessions none of the gaps is
significant -- `t = +1.33` for the vector route against the flat forecast, `t = +0.77` for the
flat forecast against the zero curve -- which is the point rather than a caveat: there is
nothing in the conditional mean of the curve sequence for any of them to find, so all four land
in the same place. What does survive is the interval, which covers close to its nominal rate;
the flat forecast's empirical quantiles cover as well and score slightly better, so the
bootstrap is buying the shape of the uncertainty and not the level of the forecast.
`notebooks/shang_replication.ipynb`.

**Dynamic updating buys the observed level, not a forecast.** Both of Shang's 2017 papers pair
the day-ahead forecaster with dynamic updating: once the first part of a session has been seen,
the forecast of the rest of it is revised. It works, in the paper's own terms. From 40% of the
session onwards every update beats the un-updated forecast, and by the last fifth it has removed
above 82% of that error. It does not work against the right benchmark. Carrying the last
observed value forward beats every method at every one of nine update points, significantly, and
the out-of-sample R-squared for predicting the remaining *increment* is negative throughout.
Anchoring each method at the last observed value closes almost the whole gap and leaves them
indistinguishable from that martingale, so what they were losing was level, on a number they had
been handed, rather than shape. The strongest form of it is that the un-updated forecast slid to
meet today's observed level scores within a few hundredths of the martingale at every point in
the session: the morning contributes nothing yesterday did not already have.
`notebooks/shang_updating.ipynb`.

**A short observed block breaks least squares, and the penalty repairs it.** With a fifth of the
session observed and two components to fit, ordinary least squares on the observed block is
worse than the martingale by a factor of nearly twenty two; ridge at the same update point by a
factor of two and a third, so the penalty removes about nine tenths of the excess, significantly
at the two earliest update points and not at the two latest. The penalty has to be chosen
against the *observed* curves: a truncated curve lies exactly in the span of its own components,
so least squares on any block of a reconstruction recovers its scores exactly, and a criterion
measured there is identically zero at every penalty and selects whichever end of the grid it is
handed. `notebooks/shang_updating.ipynb`.

**The jump equation's functional term is empty, and its nuisance term is not.** The overnight
move is what the cumulative intraday return transform deletes, and `jump` is the equation for
it: `J_t` on its own lag, the previous session's curve, and predetermined covariates. Fitted
across nine instruments and some 500 sessions each, the joint test that the coefficient
function is zero rejects for none of the nine, on the signed jump or on its size, while the
test that the two scalar covariates are zero rejects for all nine on the size. Out of sample
the ordering is the same: adding the covariates lifts the median held-out R-squared from
`-0.016` to `+0.0268`, and adding the curve on top of them returns it to `-0.026`, helping two
of nine. A bounded coefficient function integrated against `X` cannot evaluate `X` at a point,
so where a curve's informative summary is one of its values the functional slot cannot express
it and an unpenalised block simply adds variance. `notebooks/jump_equation.ipynb`.

**Retaining more components makes the unpenalised estimate of the coefficient function worse.**
Conditional maximum likelihood of the jump equation is closed form and recovers its parameters
cleanly, but it shrinks nothing, and the width of the functional block is set by how many
principal components were retained rather than by how many carry signal. On one design, going
from 5 components to 12 improves the best reconstruction available in the retained span from
`0.0242` to `0.0065` and degrades the maximum likelihood reconstruction from `0.1673` to
`1.2968` -- more information, a worse estimate, by a factor of eight. That is the case for
shrinkage stated as a measurement. `notebooks/jump_equation.ipynb`.

**Written as one system, the jump and the session after it are recoverable bar one function.**
`jump.joint_dgp` adds the curve's own equation to the jump's -- a FAR(1) with the jump as an
exogenous regressor and a gap-fill term `-gamma_t G(s) J_t` -- and solves for the stationary
moments, so a simulation is checked against what it should produce rather than against
itself. Stationarity is joint, not per equation: the gap-fill feeds yesterday's jump back into
tonight's through the coefficient function, so with `Psi = 0` the jump's persistence is
`rho - gamma <beta, G>`. On a design built from first principles, with no market data in it,
every scalar is unbiased at the parametric rate with intervals that cover, and generalised
least squares on the increments of the gap-fill profile -- the MLE under the Brownian curve
noise -- has less variance than the projection estimate of `gamma`. The coefficient function's
effect on the jump is recovered; the function itself is ill-posed, as it should be.
`notebooks/jump_equation_v2.ipynb`.

**A coefficient on the level of a cumulative curve and one on its returns are different
models, and the data can tell them apart.** Integrating by parts, `beta = 1{u >= 0.85}` on the
CIDR curve weights every return made before 0.85 by `0.15`: it says the jump responds to the
whole day's move, while the same step on the returns says only the last stretch matters. The
two regressors are nearly uncorrelated, because the level carries the morning's accumulated
move; each model, fitted to the other's data, recovers a fraction of the signal even in
population; a comprehensive regression on both, whose t-tests are Davidson-MacKinnon J tests,
names the right one from a year of simulated data; and an onset on returns is located sharply
where an onset on levels barely is. `notebooks/jump_equation_v2.ipynb`,
`docs/jump-model-choices.md`.

**The intraday curves did not start at the open.** `fdaSP.intraday.session_curves` put each
bar's close at the bar's start time, so a curve began at the first bar's close and the first
bar's return was in neither the curve nor the overnight jump -- on the hourly panel, an hour
whose return has half the session's standard deviation. `anchor="open"` builds the curve the
jump identities need; the default is unchanged so that the earlier notebooks still reproduce.
`tests/test_intraday.py`.

**The standard FAR(1) test kernel is rank one.** Bosq's `exp{-(s^2 + r^2)/2}` factorises, so
it autocorrelates a single direction in L² and leaves every orthogonal direction i.i.d., a much
narrower departure from independence than it appears. `far` also supplies a full-rank kernel.
Matching the two on Hilbert-Schmidt norm does not match them on difficulty: spreading the same
budget across the spectrum drops the operator norm from `0.70` to `0.58`, and the resulting
ISEs come out mixed rather than ordered. `notebooks/FAR1_nonIID.ipynb`.

**Cross-validation's failure to deliver a null set is a property of the rule, not of the
estimator.** Four rules read one penalty path on one dataset and differ in nothing else. Over
25 replications of each of four designs, cross-validation returns a null set of length exactly
zero on every replication of `psi1` and of `psi3`, on 84% of `psi2` and on 68% of the
right-anchored design; complementary-pairs stability selection returns a non-empty one on every
replication of all four, and where both find one it is the larger and the more accurate. The
estimator could always represent a null set. `region`, `notebooks/region_selection.ipynb`.

**A locally sparse estimate can carry the distribution theory a truncation estimate has.**
`notebooks/HH_vs_SBC.ipynb` records what the penalty lacks: an endpoint with a sampling
distribution, rather than a null set read off whichever basis groups happened to switch off.
The missing piece is supplied without assuming one. A B-spline of order `d` covers each point
with `d` basis functions, so the estimate is zero on knot interval `b` exactly when group `b`
of the overlap penalty vanishes, which makes "group `b` is selected" and "the estimate is
non-zero on interval `b`" the same event and a selection probability a statement about the
domain. Subsampling then gives a three-way partition -- informative, null, and neither -- with
a Shah-Samworth bound on each of the first two. The third piece is the honest one: an estimator
that names a null set and stops has asserted something about every point of the domain,
including the ones its data cannot separate. `region/README.md`.

**Both halves of that bound come from the same theorem, read through the complement.**
Shah and Samworth's Theorem 1(ii) bounds what is wrongly declared null directly, but scales it
by `E|N ∩ H_theta|`, for which the number of candidates is vacuous rather than conservative --
the bound is infinite at every threshold including zero, where nothing is declared null and the
truth is zero. Switching a group off is a selection like any other, though: the procedure that
keeps `q` groups discards `n_groups - q` of them and has selection probabilities `1 - p_b`, and
Theorem 1(*i*) applied to *that* procedure is the same statement with the rigorous scaling by
the number of candidates. `region/bounds.py`, `region/stability.py`.

**Section A.4's algorithm is stated for one case and is used for both.** The tightest of the
three bounds needs `D(eta, t, B, r)`, the largest tail an `r`-concave distribution on the
lattice with mean at most `eta` can have, and the paper computes it for a *decreasing*
maximiser -- which holds when `t > 2 eta`, and whose complement it sets aside as "less
important for our application". It is not less important for a functional penalty, whose groups
are few and often selected, so `eta` is routinely above a half and the condition routinely
fails. Run as written the algorithm returns zero there, which reports the error control as
*smaller* than it is, and that is the one direction an error control may not fail in.
`region.bounds.r_concave_tail` falls back on Markov's inequality outside the stated range, and
the tests hold it against Markov, against explicitly constructed members of the family it
maximises over, and against its own monotonicity. `tests/test_region.py`.

**The error control holds and the answer is still wrong, on the design that has no answer.**
`psi3` has no null region, and stability selection declares part of the domain null on every
one of 25 replications -- a quarter of it on average, more than a third under adaptive
weights -- while
the reported bound is never violated. It is not violated because it controls the number of
declared-null groups whose selection probability *under the base procedure* is high, and near
`t = 0.5`, where `10(t - 0.5)^3` is under four per cent of its peak, the base procedure
genuinely almost never selects them. A bound stated relative to a base procedure cannot rescue
a signal that base procedure never finds. Cross-validation and EBIC, which invent nothing here,
are the right comparison and they pass this design where the certified method does not.
`notebooks/region_selection.ipynb`.

**Improving the base procedure is worth more than reading it harder.** On `psi1`, whose
informative region is two pieces at opposite ends of the domain, plain stability selection
recovers a median Jaccard overlap of 0.40 with the true null set and leaves half the domain
undecided; weighting the penalty by a pilot fit -- the group analogue of the adaptive lasso --
raises that to 0.83 and cuts the undecided band to an eighth, at no change to the error budget.
`region.criteria.adaptive_weights`.

**The sign of an overnight jump survives a corrected screen for exactly two predictors, and
neither is the market's own return.** Thirteen covariates, each from a published claim about
overnight returns, are run against the signed jump and against its size for eight instruments
over 477 nights, with Newey-West errors and Benjamini-Hochberg across all 208 tests at once. On
the size, realised volatility and the Parkinson range survive for all eight instruments and the
length of the closure for six. On the sign, the London session truncated before the New York
open survives for five, positively, and signed jump variation for five, negatively. The
previous session's return survives for none, its closing fraction for none, its opening
fraction for none, and the jump's own lag for none. `jump/covariates.py`,
`notebooks/jump_covariates.ipynb`, `docs/overnight-jumps-literature.md`.

**What reverses overnight is where the previous session's variation came from, not its
return.** Signed jump variation is the difference of the upside and downside realised
semivariances, and it enters the sign equation where the return does not -- yet the two are
summaries of the same session. A day that rose hard and gave it back has the return of one that
fell hard and recovered, and the opposite signed variation. The closing-imbalance channel that
the drift literature is stated in does not appear at all: neither the closing return nor the
close's position in the session's range predicts the sign for any instrument, which is either
because an hourly bar is a poor proxy for a signed order imbalance or because the channel
itself closed over exactly this window. `notebooks/jump_covariates.ipynb`.

**Truncating a foreign session before the domestic open makes the forecast better, not merely
legal.** The London market trades through the New York night and closes two hours after New
York opens, so a whole same-day London return is a look-ahead into most of the session it is
being used to forecast the start of. Cut at 13:00 London it correlates 0.27 with the SPY
overnight jump; taken whole, 0.15, and the date-blocked out-of-sample R-squared is positive for
seven of eight instruments against one of eight. The afternoon responds to the move that
follows the open instead of predicting it, so removing it removes variance and no signal. The
truncation is doing the work of an estimator, which is the argument for `truncFLR` and `fdaSP`
arriving from the data. `notebooks/jump_covariates.ipynb`.

**On the London curve the method certifies the informative half and not the null half.** Put
the whole London return-rate curve in as the regressor, against 482 New York overnight jumps,
and let `region` choose. At 12 knot intervals nothing is certifiable in either direction -- not
because of the data but because of the arithmetic: the base procedure keeps four fifths of what
it is offered at every size, and the bound has no purchase above `theta = 1/2`. At 32 it
certifies `[0.344, 0.438]`, which is 10:45 to 11:30 London, and certifies nothing null; a sweep
over knot resolution finds the same interval at 22 and 42 as well, while the null half is
either empty or one interval of the coarsest available width whose position moves with the
basis, which is noise rather than a finding. What is stable underneath is where the proportions
fall away: they hold above a half to `s = 0.562`, which is 12:30 London, and collapse there --
just before the 13:00 cutoff `jump_covariates` chose by hand, and well before the New York open
at `s = 0.688` or `0.812`. So the London session stops carrying information about the New York
open *before* New York opens, and a per-family error rate of one will not certify the null side
of that. On the same nights and the same path the prediction rules go the other way entirely:
cross-validation returns no null set, and both the one-standard-error rule and EBIC return the
*whole domain*, zeroing the coefficient function outright. A rule chosen for prediction risk
will discard a signal too small to pay for itself in held-out error, which is the wrong trade
when the question is which part of the domain matters. The sweep and what it would take are in
`docs/research-log.md`. `notebooks/region_selection.ipynb`.

**Three months pin down the volatility of the session and not its drift.** Under the null of
the basic jump equation -- a jump that is i.i.d. normal and a curve that is a Brownian motion
with a free mean -- the likelihood is exact on the grid through the curve's increments. Every
estimator is closed-form and every interval exact. On a quarter of one-minute GBM sessions the
curve volatility has 24,180 degrees of freedom and a relative standard error under half a
percent. The mean jump and the session drift, sharpened only by the number of days, cannot be
told from zero or from five times their true values. Thirteen diagnostics, one per assumption,
pass after a Holm adjustment. Without the adjustment two in five datasets drawn from the null
fail at least one of them. The baseline the basic model is tested against is scored on the
62 days it can use, out of sample against the expanding mean, and through the exact F test
rather than the chi-squared likelihood ratio. `jump.basic_est`,
`notebooks/basic_jump_equation.ipynb`, `docs/basic-jump-equation.md`.

## Layout

```
src/fts_research/
  core/         Quadrature, bases, the curve process, the paths and the visual system
                -- what more than one of the four below needs (core/README.md)
  truncFLR/     Truncated FLR: endpoint estimation, methods A and B (truncFLR/README.md)
  fdaSP/        Locally sparse FLR, plus the real datasets and the R bridge (fdaSP/README.md)
  far/          FAR(1): the data-generating process, and estimation and testing (far/README.md)
  ftsa/         Forecasting a curve sequence through its principal component scores, revising
                that forecast as the session is observed, and the rolling-origin protocol every
                route is scored under (ftsa/README.md)
  jump/         The overnight jump equation: closed-form MLE of a scalar-on-function
                autoregression, its shrinkage diagnostic, the covariates the scalar block
                is built from, the joint model of the jump with the session after it, and
                the GBM and null model of the basic jump equation (jump/README.md)
  region/       Where the coefficient function is zero, selected for that rather than for
                prediction, with an error bound on the answer (region/README.md)
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
  shang_replication.ipynb     Shang (2017a) rebuilt on 2017 S&P 500 curves, its two score
                              forecasters scored against each other and against doing nothing
  shang_updating.ipynb        The same forecast revised through the session, against the
                              martingale it has to beat
  jump_equation.ipynb         The overnight jump equation, on simulated curves and then on
                              nine instruments of real sessions
  jump_equation_v2.ipynb      The jump and the session after it as one recursive model,
                              studied on simulated data only: its consistency, what of it
                              can be recovered from how many days, and the three modelling
                              choices that decide what it means
  jump_covariates.ipynb       What predicts the sign of an overnight jump, and what only
                              predicts its size: thirteen covariates from the literature,
                              screened with a false-discovery correction
  region_selection.ipynb      Choosing the null set for support recovery rather than for
                              prediction, with the error bound that comes with it
  basic_jump_equation.ipynb   The basic jump equation built up one step at a time on a
                              simulated GBM: the slicing into jumps and curves, the null
                              model fitted exactly, and every assumption of it tested
  window_jump_dgp.ipynb       Simulated data in which the overnight jump depends on the
                              least-squares slope of the last 20% of the session before,
                              so the basic model has a known, non-zero beta(u); the
                              model fitted, and against truncFLR and fdaSP
tests/          pytest suite covering all five packages, the documentation, the notebooks
                and the layout itself
scripts/        bibliography.py      -- generate and check the bibliography (see below)
                make_r_artifacts.py  -- rebuild the R reference outputs (needs R)
                make_study_cache.py  -- rebuild the bench study cache
                make_ftsa_cache.py   -- rebuild the rolling-origin and updating caches
                make_region_cache.py -- rebuild the null-set selection study
data/           the real datasets, with `manifest.json` recording what each one is: its
                source, the day it was taken, its SHA-256, and where it is a window of
                sessions, the window it covers
artifacts/      generated files, in two halves (artifacts/README.md): `reference/` is what
                CRAN fdaSP::f2sSP produced, so the validation notebook and its test run
                without R; `cache/` is what this repository produced and would take minutes
                to recompute, so the comparison and replication notebooks open without
                re-fitting
figures/        figures exported for writing up; regenerated, so not committed
docs/           references.toml -- the bibliography, and the only place a journal, a volume
                or a DOI is written down
                references.md, references.bib -- generated from it, numbered and as BibTeX
                research-log.md -- the open questions and dead ends, in date order
                overnight-jumps-literature.md -- what is known about the close-to-open
                return, by mechanism, and which of it these panels can test
                jump-model-choices.md -- levels or returns, the sign of the FAR kernel and
                the shape of the gap-fill in the joint jump model, explained
                basic-jump-equation.md -- the null jump model on simulated GBM data: the
                fit, the diagnostics and the baseline the basic model is tested against
```

The eight packages are layered: `core` depends on nothing, `truncFLR` and `fdaSP` on `core`,
`far` on `core` and `fdaSP`, `ftsa` and `jump` on `far` and on `fdaSP` for the panel they are
defined on, `region` on the estimator whose null set it selects, and `bench` on the two
estimators it compares. That is not a description -- `tests/test_layout.py` reads the import
statements back against it, so a shared piece cannot drift into a method package unnoticed, and
it holds that the packages nothing imports are exactly the four the repository ends in.

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
python -m fts_research.fdaSP.empirical    # --force to re-download
python -m fts_research.fdaSP.intraday     # --force to take a fresh snapshot
```

Set `FTS_DATA_DIR` to keep the cache somewhere else.

**The intraday panels are dated snapshots, not reproducible downloads.** The endpoint serves a
window ending today -- 730 days of hourly bars, 60 days of five-minute bars -- so a re-fetch
returns different data and every number in `notebooks/overnight_jumps.ipynb` moves with it.
The committed CSVs are what those numbers were computed on; the loader prefers them and
downloads only when a file is missing. `intraday.snapshot_span` reports the window.

One dataset is not fetched at all. `data/spx_1m_2017H1.csv` holds one-minute
regular-session closes of the S&P 500 cash index for 2017-01-03 to 2017-06-30, which is the
design `notebooks/shang_replication.ipynb` needs and which no public endpoint serves; it is
loaded by `intraday.load_spx_2017` and, being a fixed historical window, has the opposite
failure mode to the snapshots -- it cannot be silently replaced, and it cannot be recovered
either. Its manifest entry carries `fetchable: false` to say so, and `tests/test_provenance.py`
permits that declaration for this file and no other.

Each fetch records what it took in `data/manifest.json` -- source, date, SHA-256, and the
window for the two panels -- so the file a number was computed on can be identified later, and
a snapshot that has been silently replaced can be told from one that has not:

```bash
python -m fts_research.core.provenance     # the manifest, and a non-zero exit on a mismatch
```

## R (optional)

`notebooks/LocSpaFLR_validation.ipynb` compares `fdaSP.SBC` against CRAN `fdaSP::f2sSP`. It
runs **without R**, using the reference outputs committed in `artifacts/reference/`. Where
R with the
`fdaSP` package is installed, those cells regenerate the reference instead of loading it. Set
`FTS_RSCRIPT` if `Rscript` is not found automatically. To regenerate the cache headlessly:

```bash
python scripts/make_r_artifacts.py
```

The `bench` study cache is the other half of `artifacts/`, and is rebuilt without R, as are
the rolling-origin forecasts `notebooks/shang_replication.ipynb` reads:

```bash
python scripts/make_study_cache.py --refresh
python scripts/make_ftsa_cache.py --refresh
python scripts/make_region_cache.py --refresh
```

## Tests

```bash
pytest                  # everything (~2 min)
pytest -m "not slow"    # skip the full estimator pipelines (~10 s)
```

Besides the estimators, the suite checks the things that rot silently: `test_docs.py` holds
the documentation against the code, `test_notebooks.py` the notebooks against the repository,
`test_layout.py` the package layering against its declared graph, and `test_provenance.py`
the committed data against `data/manifest.json`.

Run on every push by `.github/workflows/ci.yml`, against Python 3.12 and 3.13.

## Dependencies

`numpy`, `scipy`, `scikit-learn`, `scikit-fda`, `pandas`, `matplotlib`, `statsmodels`. Note
that `scikit-fda` constrains the usable Python version via `numba`; see `pyproject.toml` for
the pinned range. `statsmodels` is there for one thing only -- the ARIMA fits behind
`ftsa.forecast`, which replicate a published forecaster built on R's `forecast::auto.arima` and
so want a real maximum likelihood ARIMA rather than a reimplementation of one.

## References

Every work cited anywhere in this repository is described once in `docs/references.toml` and
numbered in [`docs/references.md`](docs/references.md), which is the only place a journal, a
volume or a DOI is written down; the numbers below link into it. `docs/references.bib` is the
same list as BibTeX.

To cite a work in prose, write `[[0]][key]` where `key` is its key in the TOML and let
`python scripts/bibliography.py --write` fill in the number; a notebook writes the same marker
with an inline link, and a docstring cites by author and year, because nothing renders markdown
for a docstring. To add a work, put it in the TOML and run the same command: the numbering is
alphabetical, so a new entry renumbers the ones after it in every file that cites them, and
that is done here rather than by hand. `--check`, which the test suite runs, fails if any
citation anywhere has drifted from what the bibliography now says.

The finance side of `jump` reads a literature of its own, and
`docs/overnight-jumps-literature.md` is that reading -- what is known about the close-to-open
return, organised by the mechanism each strand proposes, with what these panels can and cannot
test about each one.

<!-- references:
  hall-hooker-2016, stefanucci-bernardi-canale-2025, bosq-2000, shang-2017a, shang-2017b,
  shang-hyndman-2011, hyndman-shang-2009, shah-samworth-2013, meinshausen-buhlmann-2010,
  chen-chen-2008, zou-2006, benjamini-hochberg-1995, aue-norinho-hormann-2015
-->
- **[3]** Aue, Norinho & Hörmann (2015), [On the prediction of stationary functional time
  series][aue-norinho-hormann-2015].
- **[10]** Benjamini & Hochberg (1995), [Controlling the false discovery rate: a practical and
  powerful approach to multiple testing][benjamini-hochberg-1995].
- **[16]** Bosq (2000), [*Linear Processes in Function Spaces*][bosq-2000].
- **[21]** Chen & Chen (2008), [Extended Bayesian information criteria for model selection with
  large model spaces][chen-chen-2008].
- **[37]** Hall & Hooker (2016), [Truncated linear models for functional data][hall-hooker-2016].
- **[47]** Hyndman & Shang (2009), [Forecasting functional time series][hyndman-shang-2009].
- **[61]** Meinshausen & Bühlmann (2010), [Stability selection][meinshausen-buhlmann-2010].
- **[70]** Shah & Samworth (2013), [Variable selection with error control: another look at
  stability selection][shah-samworth-2013].
- **[71]** Shang & Hyndman (2011), [Nonparametric time series forecasting with dynamic
  updating][shang-hyndman-2011].
- **[73]** Shang (2017a), [Forecasting intraday S&P 500 index returns: a functional time series
  approach][shang-2017a].
- **[74]** Shang (2017b), [Functional time series forecasting with dynamic updating: an application
  to intraday particulate matter concentration][shang-2017b].
- **[77]** Stefanucci, Bernardi & Canale (2025), [Locally sparse estimation for functional linear
  models with scalar response][stefanucci-bernardi-canale-2025].
- **[83]** Zou (2006), [The adaptive lasso and its oracle properties][zou-2006].
<!-- /references -->

## Licence

MIT, see `LICENSE`.

<!-- reference-links -->
[aue-norinho-hormann-2015]: docs/references.md#ref-aue-norinho-hormann-2015
[benjamini-hochberg-1995]: docs/references.md#ref-benjamini-hochberg-1995
[bosq-2000]: docs/references.md#ref-bosq-2000
[chen-chen-2008]: docs/references.md#ref-chen-chen-2008
[hall-hooker-2016]: docs/references.md#ref-hall-hooker-2016
[hyndman-shang-2009]: docs/references.md#ref-hyndman-shang-2009
[meinshausen-buhlmann-2010]: docs/references.md#ref-meinshausen-buhlmann-2010
[shah-samworth-2013]: docs/references.md#ref-shah-samworth-2013
[shang-hyndman-2011]: docs/references.md#ref-shang-hyndman-2011
[shang-2017a]: docs/references.md#ref-shang-2017a
[shang-2017b]: docs/references.md#ref-shang-2017b
[stefanucci-bernardi-canale-2025]: docs/references.md#ref-stefanucci-bernardi-canale-2025
[zou-2006]: docs/references.md#ref-zou-2006
<!-- /reference-links -->
