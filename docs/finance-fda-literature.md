# How finance models intraday price movement as functional data

`docs/overnight-jumps-literature.md` is about *mechanisms*: what is known about the close-to-open
return, and which of it these panels can test. This is its methodological counterpart. It asks a
different question of an adjacent literature -- when a paper treats a trading day as a curve, what
is the curve, what does it assume about the sequence of them, and which of those assumptions do the
estimators in this repository inherit.

It is organised around three questions rather than by strand, because those are the questions the
work here is stuck on and the literature answers them fairly directly:

1. What do we need that Stefanucci, Bernardi and Canale [[77]][stefanucci-bernardi-canale-2025]
   cannot do? (§4)
2. Is a FAR(1) a good data-generating process for this data? (§2)
3. What happens when the noise is not i.i.d.? (§3)

The single sentence the rest of this file elaborates: **the finance literature that uses functional
data puts its structure in the conditional variance and regresses on variation curves, while the
regression literature this repository implements puts its structure in the conditional mean and
regresses on price curves.** Every one of the three questions turns out to be a consequence of
that.

---

## 1. The transform decides everything downstream

A trading day is not a curve until a transform makes it one, and there are three in circulation.
Writing `P_i(t)` for the price on day `i` at rescaled intraday time `t` in `[0, 1]`:

| | Definition | Used by | Value at `t = 0` |
|---|---|---|---|
| **OCIDR** | `100[log P_i(t) - log P_{i-1}(1)]` | Rice, Wirjanto & Zhao [[68]][rice-wirjanto-zhao-2020] [[69]][rice-wirjanto-zhao-2021] | **the overnight jump** |
| **CIDR** | `log P_i(t) - log P_i(0)` | Gabrys, Horváth & Kokoszka [[30]][gabrys-horvath-kokoszka-2010]; Kokoszka & Reimherr [[52]][kokoszka-reimherr-2013]; Shang [[73]][shang-2017a] | `0`, by construction |
| **IDR** | `log P_i(t) - log P_i(t - w)` | Aue, Horváth & Pellatt [[4]][aue-horvath-pellatt-2017] | a single `w`-minute return |

`fdaSP.intraday.session_curves` builds the middle one under `kind="cidr"`, and two relatives of the
third under `kind="ret"` and `kind="rvol"`. It does not build the first.

Each of the three is a *candidate* stationary transform of a price path that is not itself
stationary, and the volatility papers treat that as something to check rather than assert: both
Rice, Wirjanto and Zhao papers run the KPSS-type functional stationarity test of Horváth, Kokoszka
and Rice [[45]][horvath-kokoszka-rice-2014] on every transform before modelling it, and report that
all three pass. This repository asserts the stationarity of its own transforms. Horváth and
Kokoszka's monograph [[44]][horvath-kokoszka-2012] is the standard reference for this material and
the one the whole volatility strand points back at.

**The overnight cumulative intraday return is the one this repository is missing.** It is anchored
at the *previous day's close* rather than at today's open, so
`X_i(0) = log P_i(0) - log P_{i-1}(1)` is exactly the overnight jump. The whole of `jump` exists because the CIDR transform imposes
`X_t(0) = 0` and so deletes the overnight move, which then has to be modelled by a separate scalar
equation outside the functional framework. The volatility literature does not have that problem
because it picked a different transform. Rice, Wirjanto and Zhao [[69]][rice-wirjanto-zhao-2021]
prefer it explicitly, as "more suitable for modeling volatility, given the important role of the
overnight effect", citing Hansen and Lunde [[39]][hansen-lunde-2006].

Under OCIDR the jump is a *point evaluation of the curve*. That is worth holding next to two things
this repository already concluded: that a bounded coefficient function integrated against `X`
cannot evaluate `X` at a point (`notebooks/jump_equation.ipynb`), and that Kneip, Poß and Sarda
[[51]][kneip-poss-sarda-2016] wrote down the model that adds exactly such point evaluations to the
integral term and estimates where they are.

**The transform also decides how ill-posed the estimation problem is.** Aue, Horváth and Pellatt
fit 5-minute IDR curves of SPY over 999 trading days and report that the leading eigenvalue
explains **56%** of the variation, adding that increasing the number of eigenfunctions "did not
qualitatively improve the estimators". This repository's Shang replication, on CIDR curves built
from five-minute S&P 500 data, finds two components clearing ninety per cent
(`notebooks/shang_replication.ipynb`). The two are not matched studies -- SPY over 2003-2006
against the index over 2017 -- so the gap is not a controlled comparison, but it is far too large
to be the period: a cumulative curve is dominated by a level factor and a curve of increments is
not, and that alone reorders the spectrum. Since the achievable rate in a functional linear
regression is governed by how fast the eigenvalues decay [[36]][hall-horowitz-2007], the
difference is not cosmetic. Measuring it properly on one panel is the first item in §5.

One more consequence, which §2 returns to: a CIDR curve is a cumulative sum and so behaves like a
Brownian path, whose eigenvalues decay **polynomially**. Every simulated curve process here --
`core.gp.sample_gp`, `bench.design.variance_profile`, `far.far_dgp` -- prescribes **exponential**
decay, which is the forgiving regime.

## 2. Is a FAR(1) a good data-generating process?

No. The literature is unusually clear both about why, and about what to use instead.

### 2.1 The conditional mean is empty, and that is the published consensus

Kokoszka and Reimherr [[52]][kokoszka-reimherr-2013] set up precisely this test. They build a
martingale-style framework for the predictability of the *shapes* of intraday price curves, derive
test statistics from the **signs of the functional principal component scores**, give the
asymptotics under null and alternative, and conclude that "the shapes of the intraday price curves
of large US corporations are not predictable."

This is worth stating plainly because `docs/research-log.md` currently treats the gap between this
repository's Shang replication and Shang's own tables as a discrepancy needing defence. It is not.
The finding that neither score forecaster beats the flat forecast, and that a FAR(1) beats it for
two instruments of nine by under a percentage point, is what the canonical test paper on this exact
question already reports. The repository's contribution there is the *benchmark*, not the negative
result: Shang's accuracy tables have no floor in them, and with one added the conclusion matches
Kokoszka and Reimherr.

### 2.2 What the literature uses instead

Hörmann, Horváth and Reeder [[43]][hormann-horvath-reeder-2013] proposed the functional ARCH, and
Aue, Horváth and Pellatt [[4]][aue-horvath-pellatt-2017] the fGARCH(1,1):

    y_i = sigma_i eps_i,    sigma_i^2 = delta + alpha y_{i-1}^2 + beta sigma_{i-1}^2,

with `delta` a non-negative function, `alpha` and `beta` integral operators mapping non-negative
functions to non-negative ones, and `(eps_i)` i.i.d. random functions. Cerovecki et al.
[[20]][cerovecki-etal-2019] generalise to fGARCH(p, q) and supply the quasi-likelihood estimator
that both Rice, Wirjanto and Zhao papers use.

Four things about this transfer directly to `far`:

**Stationarity is controlled the same way `far_dgp` already controls it.** Existence of a strictly
stationary solution is characterised through the random integral kernel
`gamma_0(t,s) = alpha(t,s) eps_0^2(s) + beta(t,s)` and its Hilbert-Schmidt norm. `far_dgp.hs_norm`,
`operator_norm` and `kernel_rank` are written against exactly that object and need no changes.

**Estimation is a score-space recursion.** Project `delta`, `alpha` and `beta` onto an
`M`-dimensional orthonormal basis and the volatility equation becomes
`s_i = d + A y_{i-1} + B s_{i-1}`, a matrix recursion in the projected squared curves. That is
structurally the same object as `far_est.score_operator`, which is what makes an fARCH or fGARCH
data-generating process a small addition here rather than a new package.

**Their test kernel is rank one, exactly as Bosq's is.** Aue, Horváth and Pellatt simulate from
`alpha(t,s) = beta(t,s) = 12t(1-t)s(1-s)`, which factorises as `0.4 phi_1(t) phi_1(s)` with
`phi_1(t) = sqrt(30) t(1-t)`; they fit it with `M = 1`. `notebooks/FAR1_nonIID.ipynb` has already
made this criticism of Bosq's `exp{-(s^2 + r^2)/2}` and supplied a full-rank alternative. The same
argument and the same remedy apply here, which means the observation is available without new work.

**Their innovations are rescaled Brownian motions**, `eps_i(t) = sqrt(log 2 / (2200 t)) B_i(2400 t
/ log 2)`, on a 285-point grid with 1000 curves of burn-in. So the standard simulation in this
literature already has the polynomially-decaying spectrum §1 argues for, where `sample_gp` does
not.

### 2.3 Lag one is not enough either

Rice, Wirjanto and Zhao [[68]][rice-wirjanto-zhao-2020] run goodness-of-fit tests on fitted
residuals and find FARCH(1) "generally deemed to be inadequate for each curve type", while
FGARCH(1,1) is acceptable for all three. A plain lag-one ARCH would be as unrepresentative a
straw man as the rank-one FAR(1) is, for the same kind of reason.

## 3. What happens when the noise is not i.i.d.?

In this data it is not noise. It is the signal, and there is a test that says so.

### 3.1 Two statistics, and why the difference between them is our question

Rice, Wirjanto and Zhao [[68]][rice-wirjanto-zhao-2020] frame conditional heteroscedasticity as a
hypothesis test -- `H_0`: the curves are i.i.d.; `H_A`: the curves are conditionally
heteroscedastic -- and propose two portmanteau statistics, deliberately testing serial correlation
in the *squares* so that a rejection is not merely first-moment dependence:

- `V_{N,K} = N sum_{h=1}^{K} rho_hat_h^2`, the Box-Ljung-Pierce statistic on the scalar series of
  squared norms `||X_1||^2, ..., ||X_N||^2`. Limiting distribution `chi^2_K`. This asks whether
  the *level* of volatility is autocorrelated: to what extent large-in-magnitude curves make
  subsequent curves large in magnitude.
- `M_{N,K} = N sum_{h=1}^{K} ||gamma_hat_h||^2`, on the estimated autocovariance kernel of the
  squared curves `X_i^2(t)` -- the statistic of Kokoszka, Rice and Shang
  [[53]][kokoszka-rice-shang-2017] applied to squares. Its limit is a weighted sum of independent
  `chi^2_1` variables with weights from the eigenvalues of `cov(X^2(t), X^2(s))`. This asks whether
  the intraday *shape* of volatility is autocorrelated.

The gap between those two statistics is the whole question this repository is circling. `V` is a
scalar summary; `M` is functional. If the second carries information the first does not, then
"which part of the session" is a real question about volatility.

### 3.2 The result that matters

On 5-minute S&P 500 curves, `N = 502` days from 31 December 2015 to 2 January 2018, all three
transforms reject strongly on both statistics at every lag `K` tried.

Then the decisive experiment. The natural hypothesis is that the intraday volatility is just
yesterday's daily move scaled up -- "a large magnitude of the return on the previous day spells
high volatility for the entire intraday price on the following day". So they fit a scalar
GARCH(1,1) to the daily returns, check by the Li-Mak test that it is adequate *for the daily
series* (p = 0.79, 0.77, 0.36, 0.13 at K = 1, 5, 10, 20), divide every intraday curve by that
**ex-ante** conditional standard deviation, and re-test the de-volatised curves.

They still reject: for OCIDR, `V` gives p = 0.03, 0.00, 0.00, 0.00 across the same four lags, and
`M` gives 0.01, 0.00, 0.00, 0.00. The authors' conclusion is worth quoting in full, because it is
the strongest external support the direction here has:

> the magnitude of the return cannot fully explain the volatility of intraday prices observed on
> subsequent days; instead we should consider the entire path of the price curve on previous days
> in order to adequately model future intra-day conditional heteroscedasticity

**This corrects an inference drawn in `notebooks/overnight_jumps.ipynb`.** That notebook divides
each curve by *its own norm*, finds the dependence removed in seven instruments of eight, and
reads it as showing the useful summary is close to the total move. But the norm is a
contemporaneous realised quantity, so dividing by it removes level dependence close to
tautologically -- it is not a forecast and nothing was predicted. Rice, Wirjanto and Zhao normalise
by a quantity built only from information up to day `i - 1`, and the dependence survives. The
diagnosis that the dependence is conditional heteroscedasticity rather than a conditional mean
stands, and is correct; the further claim that a scalar magnitude captures it does not.

### 3.3 The four departures, which are not the same thing

The estimators here assume `eps_i ~ N(0, sigma^2)`, i.i.d. and independent of `X`, with `sigma`
fixed from a signal-to-noise ratio -- a convention shared across `HH_dgp`, `SBC_sim`, `far_dgp`,
`bench.design` and `jump_dgp`. Four distinct things can go wrong, and lumping them together is the
reason "non-i.i.d. noise" is hard to reason about:

| Departure | What it breaks | Is it really *noise*? |
|---|---|---|
| serially dependent **curves** | the effective sample size in every selection rule | no, it is the process |
| serially dependent **errors** | the same, plus reported standard errors | no, it is the process |
| **heteroscedastic** errors, `Var(eps_i \| X_i) = g(X_i)` | efficiency; the scalar `sigma^2` in §3.2's criterion; unweighted CV | yes |
| **heavy-tailed** errors | robustness of least squares; one outlier can set `lambda` | yes |

Weak dependence in the sense of Hörmann and Kokoszka [[42]][hormann-kokoszka-2010] buys consistency
for the first two and no more; it does not license the variance calculations that every tuning rule
here is made of. Chen, Guo and Qiao [[22]][chen-guo-qiao-2022] is the nearest existing functional
linear regression that drops both assumptions, and Gabrys, Horváth and Kokoszka
[[30]][gabrys-horvath-kokoszka-2010] -- already cited here as the CIDR source -- is also a test for
whether the errors of a fitted functional linear model are correlated at all, which is a diagnostic
this repository has never run.

## 4. What do we need that Stefanucci, Bernardi and Canale cannot do?

### 4.1 The locally sparse family, placed

Seven papers, one question: where is the coefficient function zero, and how is that imposed.

| | How local sparsity arises | What it returns | Tuned by |
|---|---|---|---|
| Hall & Hooker [[37]][hall-hooker-2016] | assumed: `psi = 0` past an endpoint | the endpoint, with a sampling distribution | §3.2's criterion |
| James, Wang & Zhu [[49]][james-wang-zhu-2009] | L1 on *derivatives* of `psi` | piecewise constant/linear structure | CV |
| Lin, Cao, Wang & Wang [[55]][lin-cao-wang-wang-2017] | functional SCAD | a null set, with the oracle property | CV |
| Guan, Lin & Cao [[35]][guan-lin-cao-2020] | nested group **bridge**, shrinks the tail | an endpoint, by penalisation | BIC |
| Liu, Divani & Petersen [[56]][liu-divani-petersen-2022] | nested group **lasso**, in a GLM | an endpoint, convexly | BIC |
| Stefanucci, Bernardi & Canale [[77]][stefanucci-bernardi-canale-2025] | overlap group lasso | any union of intervals | CV |
| Kneip, Poß & Sarda [[51]][kneip-poss-sarda-2016] | points of impact beside the integral | impact locations `tau_k` | — |

Two things this table settles. **The truncation/penalisation split is a false dichotomy**: Guan,
Lin and Cao and then Liu, Divani and Petersen estimate the endpoint *by* penalising, which is
Hall and Hooker's model fitted with Stefanucci, Bernardi and Canale's machinery, published between
the two. And **Liu, Divani and Petersen already separate the smoothing penalty from the truncation
penalty**, where `fdaSP.SBC` has one `lambda` doing both -- which is the cost
`notebooks/tecator_application.ipynb` measured, the right null set bought only by over-shrinking
everything kept.

### 4.2 The regressor is wrong, not just the tuning

`notebooks/jump_equation.ipynb` put the previous session's price curve into the functional slot,
found the joint test that the coefficient function is zero rejects for none of nine instruments,
and concluded that where a curve's informative summary is one of its values, a bounded `psi`
integrated against `X` cannot express it.

The volatility literature never regresses on a price curve. Rice, Wirjanto and Zhao's
[[69]][rice-wirjanto-zhao-2021] functional GARCH-X puts a non-negative covariate **curve**
`x_i(t)` into the conditional variance, and their four choices are all *variation* curves:

| Covariate curve | Definition | The scalar it generalises |
|---|---|---|
| HMV, historical mean variation | `(1/m) sum_{j=1}^{m} \|r_{i-j}(t)\|`, `m = 5` | Corsi's HAR average [[24]][corsi-2009]; `rv` in `jump.covariates` |
| IRV, intraday range variation | `100 \|log P_i^high(t) - log P_i^low(t)\|` | **the Parkinson range** [[65]][parkinson-1980] |
| IBV, intraday bipower variation | `\|r_i(t)\| \|r_i(t - w)\|` | Barndorff-Nielsen & Shephard's jump-robust measure [[7]][barndorff-nielsen-shephard-2006] |
| IVOL, intraday volume | the volume curve | trading volume |

The first two are the curve analogues of precisely the two scalar covariates
`notebooks/jump_covariates.ipynb` found survive a Benjamini-Hochberg correction across 208 tests
for **eight instruments of eight**, on the jump's size. That is not a coincidence; it is the same
signal, measured once as a number and once as a function.

And a variation curve does not have the property the repository's objection depends on. Its
information is spread along the domain -- it says *when* during the session the variation happened
-- rather than concentrated in the endpoint the way a cumulative price curve's is. So the
functional slot is the right slot for it, and `intraday.session_curves` already builds one under
`kind="rvol"`.

### 4.3 The gap

Putting §3 and §4.2 together. The conditional scale of tomorrow's move depends on the whole path of
today's variation curve and not on its total. Nobody has asked which part of that path. Three
things none of the seven papers in §4.1 does, and none of the volatility papers does either:

1. **A locally sparse coefficient function on a variation curve predicting a conditional scale**,
   rather than on a price curve predicting a conditional mean. Liu, Divani and Petersen come
   closest -- a truncated functional GLM -- but truncate a tail rather than selecting a union of
   intervals, and assume i.i.d. curves.
2. **An error bound for support recovery that holds under serial dependence.** Every selection rule
   in play carries an effective sample size that dependence invalidates. `region.stability` already
   blocks its complementary-pairs draw so that dependent rows are never split across a pair, but
   Shah and Samworth's theorem is stated for independent observations and nothing establishes what
   the bound is worth once the draw is blocked -- `region/README.md` says so outright. The gap is
   theoretical, on a procedure already implemented and tested.
3. **Any shrinkage at all on the volatility side.** Rice, Wirjanto and Zhao fit the covariate curve
   with an unpenalised operator and report that the FGARCH-X "provides a significant correction ...
   in terms of an in-sample fitting, while its out-of-sample performances do not appear to be more
   superior than those of the existing functional GARCH models." An in-sample gain that does not
   survive out of sample is the textbook symptom of too many effective parameters and no
   regularisation.

## 5. What these panels can test

Following §8 of `docs/overnight-jumps-literature.md`, in the same spirit.

**Testable now, with what is committed.** Build OCIDR curves from `data/intraday_5m.csv` -- a small
addition to `intraday.session_curves`, which already has the previous close it needs -- and run
both portmanteau statistics of §3.1 on OCIDR, CIDR and `rvol` curves for the nine instruments.
Compare the empirical eigenvalue decay of the three transforms with `bench.design.spectrum`, which
answers §1's polynomial-versus-exponential question directly. Run Gabrys, Horváth and Kokoszka's
error-correlation test on the fitted jump equation, and the KPSS-type stationarity test
[[45]][horvath-kokoszka-rice-2014] on each transform, so that the assumption is checked here the
way it is checked in the papers.

**Testable with modest new code.** An fARCH(1) or fGARCH(1,1) curve process, via the score-space
recursion of §2.2, injected through `bench.study.run_cell`'s existing `data=` parameter. Then the
question that separates estimator failure from tuning failure: does the gap between each method's
prescribed tuning and its oracle widen under that process?

**Not testable here.** Anything needing the intraday path of the overnight session itself, which is
the futures data `overnight-jumps-literature.md` §8 already asks for; the quasi-likelihood
estimator of Cerovecki et al., which needs a non-negativity constraint none of these estimators
carries; and anything needing a panel spanning the 2021 structural break from both sides.

<!-- references: cited -->
- **[4]** Aue, Horváth & Pellatt (2017), [Functional generalized autoregressive conditional
  heteroskedasticity][aue-horvath-pellatt-2017].
- **[7]** Barndorff-Nielsen & Shephard (2006), [Econometrics of testing for jumps in financial
  economics using bipower variation][barndorff-nielsen-shephard-2006].
- **[20]** Cerovecki et al. (2019), [Functional GARCH models: the quasi-likelihood approach and its
  applications][cerovecki-etal-2019].
- **[22]** Chen, Guo & Qiao (2022), [Functional linear regression: dependence and error
  contamination][chen-guo-qiao-2022].
- **[24]** Corsi (2009), [A simple approximate long-memory model of realized
  volatility][corsi-2009].
- **[30]** Gabrys, Horváth & Kokoszka (2010), [Tests for error correlation in the functional linear
  model][gabrys-horvath-kokoszka-2010].
- **[35]** Guan, Lin & Cao (2020), [Estimating truncated functional linear models with a nested
  group bridge approach][guan-lin-cao-2020].
- **[36]** Hall & Horowitz (2007), [Methodology and convergence rates for functional linear
  regression][hall-horowitz-2007].
- **[37]** Hall & Hooker (2016), [Truncated linear models for functional data][hall-hooker-2016].
- **[39]** Hansen & Lunde (2006), [Realized variance and market microstructure
  noise][hansen-lunde-2006].
- **[42]** Hörmann & Kokoszka (2010), [Weakly dependent functional data][hormann-kokoszka-2010].
- **[43]** Hörmann, Horváth & Reeder (2013), [A functional version of the ARCH
  model][hormann-horvath-reeder-2013].
- **[44]** Horváth & Kokoszka (2012), [*Inference for Functional Data with
  Applications*][horvath-kokoszka-2012].
- **[45]** Horváth, Kokoszka & Rice (2014), [Testing stationarity of functional time
  series][horvath-kokoszka-rice-2014].
- **[49]** James, Wang & Zhu (2009), [Functional linear regression that's
  interpretable][james-wang-zhu-2009].
- **[51]** Kneip, Poß & Sarda (2016), [Functional linear regression with points of
  impact][kneip-poss-sarda-2016].
- **[52]** Kokoszka & Reimherr (2013), [Predictability of shapes of intraday price
  curves][kokoszka-reimherr-2013].
- **[53]** Kokoszka, Rice & Shang (2017), [Inference for the autocovariance of a functional time
  series under conditional heteroscedasticity][kokoszka-rice-shang-2017].
- **[55]** Lin et al. (2017), [Locally sparse estimator for functional linear regression
  models][lin-cao-wang-wang-2017].
- **[56]** Liu, Divani & Petersen (2022), [Truncated estimation in functional generalized linear
  regression models][liu-divani-petersen-2022].
- **[65]** Parkinson (1980), [The extreme value method for estimating the variance of the rate of
  return][parkinson-1980].
- **[68]** Rice, Wirjanto & Zhao (2020), [Tests for conditional heteroscedasticity of functional
  data][rice-wirjanto-zhao-2020].
- **[69]** Rice, Wirjanto & Zhao (2021), [*Exploring volatility of crude oil intra-day return
  curves: a functional GARCH-X model*][rice-wirjanto-zhao-2021].
- **[73]** Shang (2017a), [Forecasting intraday S&P 500 index returns: a functional time series
  approach][shang-2017a].
- **[77]** Stefanucci, Bernardi & Canale (2025), [Locally sparse estimation for functional linear
  models with scalar response][stefanucci-bernardi-canale-2025].
<!-- /references -->

<!-- reference-links -->
[aue-horvath-pellatt-2017]: references.md#ref-aue-horvath-pellatt-2017
[barndorff-nielsen-shephard-2006]: references.md#ref-barndorff-nielsen-shephard-2006
[cerovecki-etal-2019]: references.md#ref-cerovecki-etal-2019
[chen-guo-qiao-2022]: references.md#ref-chen-guo-qiao-2022
[corsi-2009]: references.md#ref-corsi-2009
[gabrys-horvath-kokoszka-2010]: references.md#ref-gabrys-horvath-kokoszka-2010
[guan-lin-cao-2020]: references.md#ref-guan-lin-cao-2020
[hall-horowitz-2007]: references.md#ref-hall-horowitz-2007
[hall-hooker-2016]: references.md#ref-hall-hooker-2016
[hansen-lunde-2006]: references.md#ref-hansen-lunde-2006
[hormann-kokoszka-2010]: references.md#ref-hormann-kokoszka-2010
[hormann-horvath-reeder-2013]: references.md#ref-hormann-horvath-reeder-2013
[horvath-kokoszka-2012]: references.md#ref-horvath-kokoszka-2012
[horvath-kokoszka-rice-2014]: references.md#ref-horvath-kokoszka-rice-2014
[james-wang-zhu-2009]: references.md#ref-james-wang-zhu-2009
[kneip-poss-sarda-2016]: references.md#ref-kneip-poss-sarda-2016
[kokoszka-reimherr-2013]: references.md#ref-kokoszka-reimherr-2013
[kokoszka-rice-shang-2017]: references.md#ref-kokoszka-rice-shang-2017
[lin-cao-wang-wang-2017]: references.md#ref-lin-cao-wang-wang-2017
[liu-divani-petersen-2022]: references.md#ref-liu-divani-petersen-2022
[parkinson-1980]: references.md#ref-parkinson-1980
[rice-wirjanto-zhao-2020]: references.md#ref-rice-wirjanto-zhao-2020
[rice-wirjanto-zhao-2021]: references.md#ref-rice-wirjanto-zhao-2021
[shang-2017a]: references.md#ref-shang-2017a
[stefanucci-bernardi-canale-2025]: references.md#ref-stefanucci-bernardi-canale-2025
<!-- /reference-links -->
