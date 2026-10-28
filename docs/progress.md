# Progress to date

Everything this repository has done, as of 2026-10-27, arranged as Section 4 of the first-year
review report. 4.1–4.4 follow the report skeleton. The work outside it is listed afterwards,
and a timeline at the end puts it all in date order. Every number here comes from a committed
notebook output or cache. Where a number is still waiting on a pre-writing item, the text says
which one.

## 4.1 The Shang replication

**Shang's [[74]][shang-2017a] descriptive claims hold up. The vector score forecaster beats per-score ARIMA on 8 of 9
instruments (two-sided sign test p = 0.0391), and on the 2017 window neither beats the flat
forecast.** `notebooks/shang_replication.ipynb`, `src/fts_research/ftsa/README.md`.

- **Data.** 125 regular sessions, 2017-01-03 to 2017-06-30, of Oanda's SPX500_USD CFD on the
  S&P 500, as one-minute mid quotes from the FutureSharks `financial-data` archive. Each is built
  into a cumulative intraday return curve on 78 five-minute marks (`data/spx_1m_2017H1.csv`,
  which `fdaSP.intraday.ensure_spx_2017` rebuilds byte for byte from a pinned commit). The CFD
  tracks the cash index to within about a point at the close, but it is not the index. The
  second dataset is the nine-instrument five-minute panel: about 60 sessions each, 24 of them
  held out.
- **Decomposition.** Shang's 90% rule keeps M = 2 components, holding 85.14% and 7.24% of the
  variance. They have the shapes the paper describes: a level factor, and a component that
  changes sign mid-session.
- **Score forecasters.** The FAR(1) on the score vector, which is `far_est`'s operator itself,
  is more accurate than M separate ARIMA models on 8 of 9 instruments. The paper reports the two
  as close. The likely mechanism is that the vector route keeps the cross-lag terms, while ARIMA
  re-selects an order at every split from at most 124 points.
- **Against the floor.** On the 2017 window (75 sessions to train, 50 to test), the
  out-of-sample R² against the flat forecast is −0.1192 for ARIMA and −0.0249 for the FAR(1). No
  paired gap is significant: FAR(1) against flat gives t = +1.33 (p = 0.189), and flat against
  the zero curve gives t = +0.77 (p = 0.447). On the panel, ARIMA beats the flat forecast on 1 of
  9 instruments and the FAR(1) on 5 of 9. Of those 5, three are near-copies of one series (QQQ,
  SPY, ^GSPC), and each instrument has only 24 test sessions.
- **Intervals.** The bootstrap's nominal 80% band covers 0.7905 (ARIMA) and 0.8313 (FAR(1)).
  The training sessions' empirical quantiles cover 0.8264 and get a slightly better interval
  score.
- **Dynamic updating** (Shang [[75]][shang-2017b], `notebooks/shang_updating.ipynb`) beats the un-updated forecast. It loses
  to carrying the last observed value forward at all nine update points. Anchoring each method at
  that value leaves it indistinguishable from the martingale, so the whole deficit was level
  error.

**Table 1.** Out-of-sample R² against the flat forecast, one-step-ahead rolling origin
(`artifacts/cache/shang_rolling.csv`, `artifacts/cache/shang_rolling_panel.csv`).

| Instrument | Univariate ARIMA | FAR(1) on the scores |
| --- | --- | --- |
| S&P 500, 2017 window | −0.1192 | −0.0249 |
| AAPL | −0.0296 | −0.0410 |
| AMZN | −0.1312 | −0.0391 |
| MSFT | −0.1009 | −0.0273 |
| NVDA | +0.0729 | +0.1351 |
| QQQ | −0.2826 | +0.1326 |
| SPY | −0.4697 | +0.1145 |
| TSLA | −0.0329 | +0.0188 |
| ^FTSE | −0.2043 | −0.1050 |
| ^GSPC | −0.5812 | +0.1178 |
| Panel range (median) | −0.5812 to +0.0729 (−0.1312) | −0.1050 to +0.1351 (+0.0188) |

## 4.2 Estimators and validation

**Both estimators are implemented and validated, and the paper's §3.2 tuning has been
corrected.** Run as written, §3.2 returns the smallest penalty on the grid whatever the data.

- **truncFLR.** Hall & Hooker [[37]][hall-hooker-2016] method A (`HH_A.py`, the coefficient
  refitted on the truncated-curve components at every candidate endpoint) and method B (`HH.py`,
  a frozen pilot, scanning only the endpoint). BIC chooses the number of components in both.
  `fit_mirror` reflects the domain, so the window can be anchored at the close, which is what
  the overnight-jump problem needs.
- **fdaSP.** A Python port of the Stefanucci, Bernardi & Canale
  [[78]][stefanucci-bernardi-canale-2025] overlap group lasso. With four parametrisation
  conventions matched, it agrees with CRAN `fdaSP::f2sSP` to 1e-3 at fixed penalties and to 1e-2
  along the regularised path. The B-spline bases agree to 9.99e-16. The comparison runs without
  R, against `artifacts/reference/` (`notebooks/LocSpaFLR_validation.ipynb`).
- **The §3.2 finding** (draft wording; item 5, the supervisor check, has not happened yet). The
  variance terms of three of the four displays that evaluate the step 5 and 5′ surrogates
  (`S_Y^B`, `S_b^B`, `S_b^A`) are n times too large. Equation (3.4) and step 6 differ by a factor
  of n in how they scale λ. Step 7's Gaussian weight ends up 25 times narrower than the grid it
  is integrated on. Once corrected, λ̂ is interior on 23 of 25 seeds and method B's endpoint
  loses its bias and two-thirds of its RMSE. Methods A and B then cannot be separated: over 100
  paired replications at n = 200 the endpoint bias is −0.0023 against −0.0017, and the paired t
  is −0.39 on endpoint error and +0.80 on ISE (`notebooks/truncFLR_lambdadegen.ipynb`,
  `notebooks/truncFLR_methodA.ipynb`).
- **Local sparsity.** Cross-validation almost never zeroes part of the domain. On the paper's
  three test functions and on Tecator it zeroes nothing. Over 25 replications of four designs it
  returns an empty null set on 100%, 100%, 84% and 68% of them. Complementary-pairs stability
  selection with a Shah–Samworth [[71]][shah-samworth-2013] bound returns a non-empty null set
  on every replication (`notebooks/region_selection.ipynb`). The caveat belongs in the future-work
  sentence: on `psi3`, which has no null region, it still declares about a quarter of the domain
  null, and the bound is never violated.

## 4.3 Simulation study on the window DGP

**Under realistic signal strengths, a quarter of data detects nothing.** At R² = 0.02 the
8-component F test rejects in 9.0% of 62-day samples, and it needs about 765 days for 80%
power. On one quarter at R² ≤ 0.05 no estimator beats β̂ ≡ 0. Method A is the only estimator
that locates the window, and it does so reliably only with years of data.

**The null.** The GBM is simulated as 63 one-minute sessions, with the overnight jump i.i.d.
normal and the curve a Brownian motion with a free mean. The likelihood is exact through the
increments, so every estimate is closed-form and every interval exact. σ_ε has 24,180 degrees
of freedom and a relative standard error under 0.5%. α and μ(1) have t-statistics under 1. All
13 diagnostics, one per assumption, pass after Holm [[41]][holm-1979]. Without the adjustment,
two in five datasets drawn from the null fail at least one of them (`notebooks/basic_jump_equation.ipynb`,
`docs/basic-jump-equation.md`).

**The window DGP.** The jump carries on a share κ of the least-squares slope b̂ of the
previous session's last 20%:

    J_t = 100 m T + κ T (b̂_{t-1} − 100 m) + 100 σ √T Z_t,
    β(u) = κ T · 12 (u − 0.9) / 0.2³  on [0.8, 1], zero before.

κ ∈ {0, 0.25, 0.5, 1} gives R² of 0, 0.08, 0.27 and 0.60. b̂ is a statistic of the realised
path, not the drift: its sd is 2.43% per session against a drift of 0.02%. The textbook OLS
interval for it is 11.2 times too narrow (`src/fts_research/jump/window_dgp.py`,
`notebooks/window_jump_dgp.ipynb`).

**The leading components hold little of the signal.** For Brownian curves the components are
nearly flat near the close, so a late-window slope loads on components 3 to 8.

| M | Share of curve variance | Share of the signal |
| --- | --- | --- |
| 1 | 81.1% | 0.8% |
| 4 (the 95%-of-variance rule) | 95.0% | 43.1% |
| 8 | 97.5% | 97.6% |

**Power at realistic signal** (item 1, done: `src/fts_research/jump/window_power.py`,
`artifacts/cache/window_power.csv`). There are 500 replications per cell and 1,000 under the
null. All four tests below are exact, at level 5%.

| R² | n = 62 | 125 | 250 | 390 | 500 | 765 | 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| *F test on 8 components* | | | | | | | |
| 0.01 | 0.072 | 0.118 | 0.134 | 0.204 | 0.256 | 0.460 | 0.566 |
| 0.02 | 0.090 | 0.168 | 0.278 | 0.456 | 0.560 | 0.788 | 0.910 |
| 0.05 | 0.178 | 0.344 | 0.650 | 0.906 | 0.954 | 0.996 | 1.000 |
| *1-df oracle on b̂* | | | | | | | |
| 0.02 | 0.202 | 0.334 | 0.568 | 0.806 | 0.868 | 0.974 | 0.996 |
| *95%-of-variance M* | | | | | | | |
| 0.02 | 0.080 | 0.120 | 0.212 | 0.322 | 0.352 | 0.608 | 0.762 |
| *BIC-chosen M, permutation-calibrated* | | | | | | | |
| 0.02 | 0.058 | 0.080 | 0.088 | 0.116 | 0.096 | 0.218 | 0.238 |

- The analytic guide puts 80% power at R² = 0.02 at 389 pairs for the oracle and 765 for the
  8-component test. That is 765, not the 740 the plan has, because 8 components hold 97.6% of
  the signal. At R² = 0.05 the figures are 154 and 303. The simulated 8-component power at 765
  pairs is 0.788.
- BIC loses the signal. Pooled over n, it picks M = 1 in 89.5% of datasets at R² = 0.02,
  because the signal sits in components 3 to 8. Referred to the F table after selection, the BIC
  test has size 9.5% at n = 62. Permutation calibration restores 4.7% (Phipson & Smyth
  [[67]][phipson-smyth-2010]) but keeps no power.
- **Figure 2** is the power figure of §3 of `notebooks/window_jump_power.ipynb`: one panel per
  exact test, a line per R² against n, with the analytic guide dashed.

**Table 2** (item 2, done: `artifacts/cache/window_methods.csv`, §4 of
`notebooks/window_jump_power.ipynb`). There are 100 replications per cell, on the same datasets
across cells. Methods A and B are anchored at the close, and the group lasso is told nothing.
Relative ISE is ∫(β̂ − β)² / ∫β², so β̂ ≡ 0 scores 1. Held-out R² is pooled over the next 63
jumps of every replication, against the expanding mean. truncFLR zeroes [0, û) by construction.

| n | R² | Method | Relative ISE, median [IQR] | \|û − 0.8\| ≤ 0.02 | Zeroes any of [0, 1] | Held-out R² |
| --- | --- | --- | --- | --- | --- | --- |
| 62 | 0.02 | Untruncated, M = 8 | 4.56 [2.87, 7.43] | — | 0.00 | −0.109 |
| | | truncFLR method A | 177.66 [11.59, 852.44] | 0.04 | 1.00 | −0.068 |
| | | truncFLR method B | 66.53 [7.28, 2777.66] | 0.00 | 1.00 | −0.131 |
| | | Group lasso, CV | 1.00 [1.00, 1.05] | 0.00 | 0.76 | −0.038 |
| | 0.05 | Untruncated, M = 8 | 1.96 [1.26, 3.09] | — | 0.00 | −0.079 |
| | | truncFLR method A | 19.66 [1.11, 142.38] | 0.10 | 1.00 | −0.040 |
| | | truncFLR method B | 17.47 [2.53, 514.74] | 0.00 | 1.00 | −0.095 |
| | | Group lasso, CV | 1.00 [1.00, 1.04] | 0.00 | 0.64 | −0.039 |
| | 0.27 | Untruncated, M = 8 | 0.54 [0.41, 0.74] | — | 0.00 | 0.152 |
| | | truncFLR method A | 0.38 [0.18, 0.76] | 0.38 | 1.00 | 0.187 |
| | | truncFLR method B | 2.77 [1.11, 10.77] | 0.01 | 1.00 | 0.023 |
| | | Group lasso, CV | 0.45 [0.31, 0.69] | 0.00 | 0.12 | 0.146 |
| | 0.60 | Untruncated, M = 8 | 0.35 [0.29, 0.43] | — | 0.00 | 0.501 |
| | | truncFLR method A | 0.16 [0.09, 0.27] | 0.74 | 1.00 | 0.557 |
| | | truncFLR method B | 6.24 [1.52, 15.84] | 0.00 | 1.00 | 0.071 |
| | | Group lasso, CV | 0.24 [0.20, 0.34] | 0.00 | 0.05 | 0.519 |
| 1000 | 0.02 | Untruncated, M = 8 | 0.44 [0.35, 0.63] | — | 0.00 | 0.011 |
| | | truncFLR method A | 0.30 [0.16, 0.74] | 0.48 | 1.00 | 0.016 |
| | | truncFLR method B | 7.71 [2.28, 98.72] | 0.00 | 1.00 | −0.001 |
| | | Group lasso, CV | 0.43 [0.30, 0.62] | 0.00 | 0.13 | 0.013 |
| | 0.05 | Untruncated, M = 8 | 0.30 [0.26, 0.36] | — | 0.00 | 0.040 |
| | | truncFLR method A | 0.16 [0.10, 0.29] | 0.71 | 1.00 | 0.047 |
| | | truncFLR method B | 3.08 [1.20, 8.10] | 0.02 | 1.00 | 0.007 |
| | | Group lasso, CV | 0.28 [0.22, 0.36] | 0.00 | 0.09 | 0.042 |
| | 0.27 | Untruncated, M = 8 | 0.21 [0.20, 0.23] | — | 0.00 | 0.255 |
| | | truncFLR method A | 0.09 [0.06, 0.11] | 0.96 | 1.00 | 0.268 |
| | | truncFLR method B | 1.68 [1.07, 4.06] | 0.00 | 1.00 | 0.042 |
| | | Group lasso, CV | 0.17 [0.15, 0.18] | 0.00 | 0.09 | 0.261 |
| | 0.60 | Untruncated, M = 8 | 0.20 [0.19, 0.21] | — | 0.00 | 0.574 |
| | | truncFLR method A | 0.06 [0.04, 0.07] | 0.99 | 1.00 | 0.593 |
| | | truncFLR method B | 1.17 [0.94, 1.75] | 0.00 | 1.00 | 0.098 |
| | | Group lasso, CV | 0.13 [0.12, 0.14] | 0.00 | 0.06 | 0.587 |

**Paired tests.** Wilcoxon signed-rank tests on per-replication relative ISE, Holm-adjusted
across the 12 tests at each n. Each cell names the more accurate method where the adjusted
p is below 0.05. These were computed from the cache with `scipy.stats.wilcoxon` and are not yet a
cell of the notebook.

| n | R² | Method A vs untruncated | Group lasso vs untruncated | Method A vs group lasso |
| --- | --- | --- | --- | --- |
| 62 | 0.02 | untruncated, p < 0.001 | group lasso, p < 0.001 | group lasso, p < 0.001 |
| | 0.05 | untruncated, p < 0.001 | group lasso, p < 0.001 | group lasso, p < 0.001 |
| | 0.27 | p = 0.32 | p = 0.23 | p = 0.95 |
| | 0.60 | A, p < 0.001 | group lasso, p < 0.001 | A, p < 0.001 |
| 1000 | 0.02 | p = 0.47 | p = 0.47 | p = 0.47 |
| | 0.05 | A, p < 0.001 | p = 0.35 | A, p < 0.001 |
| | 0.27 | A, p < 0.001 | group lasso, p < 0.001 | A, p < 0.001 |
| | 0.60 | A, p < 0.001 | group lasso, p < 0.001 | A, p < 0.001 |

**Interpretation.**

- **One quarter.** At R² ≤ 0.05 no estimator beats returning zero. Every median relative ISE is
  at least 1, and every held-out R² is negative. The group lasso "wins" those cells because CV
  returns β̂ ≡ 0 in 64% of quarters at R² = 0.02.
- **Method A** is the only method that locates the window. At a thousand days û is within 0.02
  of 0.8 in 48% of datasets at R² = 0.02 and 71% at 0.05, against about 3% by chance under the
  null. It has the lowest median error from R² = 0.02 on, but at 0.02 the paired tests cannot
  separate it from the oracle or the group lasso. Its lead is significant from R² = 0.05. §3.2
  cannot rank its λ grid in up to 39% of datasets (the null at 500 days), and the fit then
  falls back to the smallest λ.
- **Method B** fails at every strength and every n. β changes sign inside the window, so
  scanning from u = 1 its RSS dips to a shallow minimum in the last minutes and then climbs
  above the total sum of squares. The first-interior-minimum rule stops in that dip (û = 0.990
  at all 119 λ on the `notebooks/window_jump_dgp.ipynb` quarter). This is a failure of the
  rule, not of the penalty.
- **The group lasso** forecasts nearly as well as method A (held-out R² 0.013 against 0.016 at a
  thousand days and R² = 0.02) but never locates the window. At realistic strength its sparsity
  switches the whole function off, which makes it a test of β = 0 rather than an estimate of
  where β is non-zero.

## 4.4 First empirical results

- **Own-market curves.** The joint test that β = 0 rejects for none of nine instruments, on the
  signed jump or on its size. Adding the scalar covariates lifts the median held-out R² from
  −0.016 to +0.0268, and adding the curve on top returns it to −0.026
  (`notebooks/jump_equation.ipynb`).
- **Magnitude is predictable.** Thirteen covariates are screened against 477 nights for eight
  instruments, 208 tests under Benjamini–Hochberg. Realised volatility and the Parkinson range
  survive for all eight instruments (`notebooks/jump_covariates.ipynb`).
- **Sign.** The London session truncated at 13:00 predicts the New York overnight jump for seven
  of eight instruments. For SPY over 482 nights the correlation is 0.272 and the group lasso's
  out-of-sample R² is 0.042. The whole London session gives 0.15, because its afternoon reacts
  to the New York open instead of predicting it.
- **Caveat.** These come from about 500 hourly sessions per instrument, taken from a dated Yahoo
  snapshot.

## Done, but outside Section 4

- **Nine-design comparison** (`notebooks/basis_and_coef_comp.ipynb`). Three bases are crossed
  with three coefficient functions, on a matched spectrum. Method A wins every step-function
  cell. The group lasso wins every smooth-boundary cell and two of three multi-window cells.
  Most of the gap between them is the tuning rule.
- **Head to head** (`notebooks/HH_vs_SBC.ipynb`). Over 50 paired replications the group lasso
  has the lower median ISE even on a single window. Method A failed once, with an ISE of 1.9e16.
- **Tecator** (`notebooks/tecator_application.ipynb`). The paper's null region above 940 nm
  exists along the path, but its own CV rule never selects it. Truncation cuts held-out error by
  37%. Under §3.2, method A lands at 919 nm and method B at 868 nm.
- **Serially dependent curves** (`notebooks/FAR1_nonIID.ipynb`). Bosq's standard FAR(1) test
  kernel is rank one, so `far` adds a full-rank kernel.
- **Stability-selection theory** (`src/fts_research/region/README.md`). Both halves of the
  null/informative bound come from Shah–Samworth Theorem 1(i), read through the complement.
  Section A.4's algorithm undercounts the error when η > t/2, and a Markov fallback now covers
  that case.
- **Joint jump–session model** (`notebooks/jump_equation_v2.ipynb`, `docs/jump-model-choices.md`).
  A FAR(1) with the jump as an exogenous regressor and a gap-fill term. All scalars are recovered
  at the parametric rate. Levels and returns coefficients are different models, and J tests tell
  them apart from a year of data.
- **truncFLR extensions** (`src/fts_research/truncFLR/README.md`, `docs/session-2026-09-hh-multi.md`).
  `HH_continuous` drops the exact-zero assumption, and `HH_multi` handles several breaks by
  binary segmentation.
- **Data hygiene** (`notebooks/overnight_jumps.ipynb`, `tests/test_intraday.py`). Random folds
  across a panel report more than twice the date-blocked R². The intraday curves now start at
  the open.

## Open

- Item 2: put the paired Wilcoxon tests behind Table 2 into `notebooks/window_jump_power.ipynb`,
  so that every number in it is printed by a cell.
- Item 3: the notation fixes, and a one-table notation key for the report.
- Item 6: freeze the results and export the figures.
- Item 5: the supervisor's check of the factor-of-n derivation and of the Section A.4 claim, which
  fixes the 4.2 wording.
- A test that searches the components for the signal (an adaptive Neyman-type max-F, calibrated
  by permutation). BIC-chosen M cannot find a signal that lives in components 3 to 8.
- Generated-regressor standard errors for the jump equation, a conditional-variance (GARCH)
  term, and a stochastic-volatility version of the window DGP.

## Timeline

| When | What | Where |
| --- | --- | --- |
| By 2026-07-30 | truncFLR methods A and B, and the §3.2 λ degeneracy traced to steps 3–4 | `docs/research-log.md` |
| By 2026-08-30 (first commit) | §3.2 corrections; fdaSP port checked against CRAN; Tecator; head-to-head and nine-design studies; FAR(1) curves; overnight-jump panels | `notebooks/truncFLR_lambdadegen.ipynb`, `notebooks/LocSpaFLR_validation.ipynb`, `notebooks/basis_and_coef_comp.ipynb` |
| 2026-08-31 | Overnight-jump repository folded in; Shang replication and dynamic updating; stability selection with its bound | `notebooks/shang_replication.ipynb`, `notebooks/shang_updating.ipynb`, `notebooks/region_selection.ipynb` |
| 2026-09-01 | Jump equation on nine instruments; covariate screen; literature reviews | `notebooks/jump_equation.ipynb`, `notebooks/jump_covariates.ipynb`, `docs/overnight-jumps-literature.md` |
| 2026-09 (cache 09-15) | truncFLR extensions: continuous and multi-break | `docs/session-2026-09-hh-multi.md` |
| 2026-09-29 to 10-02 | Joint jump–session model on simulated data | `notebooks/jump_equation_v2.ipynb` |
| 2026-10-15 | Basic jump equation: GBM null fitted exactly, 13 diagnostics | `notebooks/basic_jump_equation.ipynb` |
| 2026-10-16 to 10-22 | Window DGP, exact inference on b̂, PC-regression fit, comparison with truncFLR and fdaSP | `notebooks/window_jump_dgp.ipynb` |
| 2026-10-26 | First-year review skeleton and pre-writing plan | Claude Doc, not in the repository |
| 2026-10-26 to 10-27 | Item 1: power at realistic R² | `src/fts_research/jump/window_power.py` |
| 2026-10-27 | Item 2: estimator comparison at realistic R², and the notebook that reads both studies | `notebooks/window_jump_power.ipynb` |
| 2026-10-28 | Item 4: the 2017 file traced to Oanda's SPX500_USD CFD and made rebuildable from a pinned commit | `src/fts_research/fdaSP/intraday.py` |

## References

<!-- references: cited -->
- **[37]** Hall & Hooker (2016), [Truncated linear models for functional data][hall-hooker-2016].
- **[41]** Holm (1979), [A simple sequentially rejective multiple test procedure][holm-1979].
- **[67]** Phipson & Smyth (2010), [Permutation p-values should never be zero: calculating exact
  p-values when permutations are randomly drawn][phipson-smyth-2010].
- **[71]** Shah & Samworth (2013), [Variable selection with error control: another look at
  stability selection][shah-samworth-2013].
- **[74]** Shang (2017a), [Forecasting intraday S&P 500 index returns: a functional time series
  approach][shang-2017a].
- **[75]** Shang (2017b), [Functional time series forecasting with dynamic updating: an application
  to intraday particulate matter concentration][shang-2017b].
- **[78]** Stefanucci, Bernardi & Canale (2025), [Locally sparse estimation for functional linear
  models with scalar response][stefanucci-bernardi-canale-2025].
<!-- /references -->

<!-- reference-links -->
[hall-hooker-2016]: references.md#ref-hall-hooker-2016
[holm-1979]: references.md#ref-holm-1979
[phipson-smyth-2010]: references.md#ref-phipson-smyth-2010
[shah-samworth-2013]: references.md#ref-shah-samworth-2013
[shang-2017a]: references.md#ref-shang-2017a
[shang-2017b]: references.md#ref-shang-2017b
[stefanucci-bernardi-canale-2025]: references.md#ref-stefanucci-bernardi-canale-2025
<!-- /reference-links -->
