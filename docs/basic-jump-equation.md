# The null model on simulated data

This note summarises `notebooks/basic_jump_equation.ipynb`. It shows that the **null model**
from the *Basic Jump Equation* notes is implemented and works. The test data are simulated from
geometric Brownian motion (GBM), so every true parameter is known and every estimate can be
checked against it. No market data are used yet.

**Summary.**

1. Slicing the price path into overnight jumps and intraday curves is exact.
2. The fitted null model recovers the truth. Every 95% interval covers its true value.
3. All 13 assumption checks pass, after correcting for multiple testing.
4. The volatilities are estimated precisely. The means (the mean jump and the intraday drift)
   cannot be told apart from zero in three months of data. This drives most of what follows for
   the basic model.

Every number below is printed by the notebook with its seeds fixed. Running the notebook
writes the figures to `figures/basic_jump_*.png` (and `.pdf`).

---

## 1. Set-up

### The objects

On the percent log price $P_t(s) = 100\log S_t(s)$, with $s \in [0, 1]$ the time within
session $t$, each day gives a pair:

$$
\underbrace{J_t = P_t(0) - P_{t-1}(1)}_{\text{overnight jump}}, \qquad
\underbrace{X_t(s) = P_t(s) - P_t(0)}_{\text{CIDR curve, } X_t(0) = 0}.
$$

### The null model

$$
J_t = \alpha + \eta_t, \quad \eta_t \overset{\text{iid}}{\sim} N(0, \sigma_\eta^2), \qquad
X_t(s) = \mu(s) + \sigma_\varepsilon W_t(s), \quad \mu(0) = 0,
$$

with $W_t$ i.i.d. standard Brownian motions, independent of the $\eta_t$. In words: overnight
jumps cannot be predicted from intraday returns.

### Why GBM is the right test case

As derived in the notes, GBM with an overnight closure of (variance) length $T$ gives:

- **raw prices** $S_t(s)$: mean and variance grow with $t$, so they are not stationary;
- **price differences** $S_t(s) - S_t(0)$: the variance still depends on the price level;
- **log price differences** $X_t(s)$: i.i.d. across days.

Taking logs *and* differencing is what makes the days i.i.d. GBM is then an exact special case
of the null, with known true values:

| parameter | meaning | GBM value |
|---|---|---|
| $\alpha$ | mean overnight jump | $100mT$ |
| $\sigma_\eta$ | overnight volatility | $100\sigma\sqrt{T}$ |
| $\mu(s)$ | mean intraday return | $100ms$ (a straight line) |
| $\sigma_\varepsilon$ | intraday volatility | $100\sigma$ |

where $m = \nu - \sigma^2/2$.

### Simulation design

| | value | interpretation |
|---|---|---|
| days | 63 | one quarter |
| grid | 391 points | one-minute bars over a 6.5-hour session |
| $S_0$ | 100 | starting price |
| $\sigma$ | 0.01 | 1% intraday standard deviation per session |
| $T$ | 0.25 | 0.5% overnight standard deviation; the night carries 20% of daily variance |
| $\nu$ | $2.5 \times 10^{-4}$ | about 8% expected price growth a year |

So the true values are $\alpha = 0.005$, $\sigma_\eta = 0.5$, $\mu(1) = 0.02$ and
$\sigma_\varepsilon = 1$, all in percent.

![Three months of one-minute GBM prices](../figures/basic_jump_price_path.png)

*Figure 1. The simulated GBM price path. Overnight gaps are dotted.*

---

## 2. Slicing the data into jumps and curves

The path gives 63 jumps and 63 curves. As a check, adding the pieces back up rebuilds every
closing price with **zero error**.

| | mean | standard deviation |
|---|---|---|
| jumps $J_t$ | +0.039% | 0.449% |
| session returns $X_t(1)$ | +0.101% | 0.979% |

![The 63 CIDR curves, tail to head](../figures/basic_jump_curves.png)

*Figure 2. The 63 intraday curves laid end to end. Each starts at zero at the open and wanders
like a Brownian motion.*

![Overnight jumps and their distribution](../figures/basic_jump_jumps.png)

*Figure 3. The overnight jumps (blue up, orange down) and their histogram. There are no
visible runs.*

---

## 3. Fitting the null model

Under the null, each curve's 390 one-minute increments are independent normals, and the jumps
are one normal sample. So every estimator is in closed form (sample means and variances), and
every interval is **exact** (Student $t$, $\chi^2$ or $F$), not a large-sample approximation.
$\mu(s)$ is estimated freely as the mean curve; the GBM's straight-line drift is tested in
section 5, not imposed.

| parameter | estimate | 95% interval | truth | covered |
|---|---|---|---|---|
| $\alpha$ (mean jump) | 0.039 | [−0.074, 0.152] | 0.005 | yes |
| $\sigma_\eta$ (jump sd) | 0.449 | [0.382, 0.544] | 0.500 | yes |
| $\mu(1)$ (session drift) | 0.101 | [−0.145, 0.347] | 0.020 | yes |
| $\sigma_\varepsilon$ (intraday vol) | 0.996 | [0.987, 1.005] | 1.000 | yes |
| $\sigma_\eta^2/\sigma_\varepsilon^2$ (GBM: $T$) | 0.203 | [0.147, 0.299] | 0.250 | yes |

![Parameter estimates against the truth](../figures/basic_jump_parameters.png)

*Figure 4. Each parameter's estimate and exact 95% interval (green) against its true value
(dashed).*

**Two takeaways.**

- **Volatilities are precise.** $\hat\sigma_\varepsilon$ is within 0.45% of the truth, because
  every one-minute increment of every day is information about it.
- **Means are not.** The intervals for $\alpha$ and $\mu(1)$ cover zero as easily as the
  truth. More intraday data does not help, since only more days sharpen a mean. For the true
  drift to be two standard errors from zero would take about 10,000 days (forty years).

---

## 4. The fitted model against the data

![The curves against the fitted null](../figures/basic_jump_fitted_curves.png)

*Figure 5. Left: the 63 curves, the fitted mean (green) and the fitted 95% band for a new curve.
The true mean and band are dashed. Right: the fitted mean alone with its 95% interval.*

The curves fill the band as a Brownian motion should, with the spread growing like
$\sqrt s$. The fitted and true bands almost coincide, because $\sigma_\varepsilon$ is well
estimated. The fitted mean wanders (it is an average of only 63 noisy paths), but both the
true drift and zero stay inside its interval.

![Jump density, QQ plot and prediction interval](../figures/basic_jump_jump_fit.png)

*Figure 6. Left: the jump histogram against the fitted (green) and true (dashed) normal
densities. Centre: QQ plot with a 95% envelope. Right: the jumps against the fitted 95%
prediction interval.*

The fitted normal matches the jumps, and the QQ plot stays essentially within its envelope.

![The data beside a draw from the fitted null](../figures/basic_jump_data_vs_draw.png)

*Figure 7. Top: the data. Bottom: one simulated quarter from the fitted model.*

The data and a simulation from the fitted model are hard to tell apart, which is the visual
check that the model describes the data.

---

## 5. Testing the assumptions

Each assumption of the null gets its own test, so that a failure says *which* assumption broke.
With 13 tests at 5%, a correct model will often fail one by chance, so p-values are adjusted
with Holm's method ([Holm 1979](references.md#ref-holm-1979)), which keeps the chance of *any*
false rejection at 5%.

| assumption | test | p-value | Holm p | verdict |
|---|---|---|---|---|
| **jumps** normal | Shapiro-Wilk | 0.116 | 1.000 | pass |
| jumps normal | Jarque-Bera | 0.254 | 1.000 | pass |
| jumps independent | Ljung-Box, 5 lags | 0.645 | 1.000 | pass |
| jumps independent | Ljung-Box, 10 lags | 0.697 | 1.000 | pass |
| no volatility clustering | Ljung-Box on $J_t^2$, 5 lags | 0.855 | 1.000 | pass |
| no volatility clustering | Ljung-Box on $J_t^2$, 10 lags | 0.513 | 1.000 | pass |
| jump independent of its session | correlation $t$-test | 0.952 | 1.000 | pass |
| **curves**: constant intraday volatility | Bartlett, 13 half-hour buckets | 0.722 | 1.000 | pass |
| independent increments | lag-1 autocorrelation | 0.516 | 1.000 | pass |
| Gaussian increments | Jarque-Bera | 0.346 | 1.000 | pass |
| Brownian covariance $\sigma^2\min(s,s')$ | Monte Carlo distance test | 0.851 | 1.000 | pass |
| straight-line drift (GBM only) | $F$ test | 0.036 | 0.428 | pass |
| curves independent across days | Gabrys-Kokoszka portmanteau | 0.024 | 0.310 | pass |

**All 13 checks pass.** Two raw p-values fall below 5%, but the data really are GBM, so these
are chance. This is expected. In 200 fresh datasets simulated from the true GBM:

| | share of datasets |
|---|---|
| at least one raw test fails at 5% | 42% |
| two or more raw tests fail at 5% | 15% |
| at least one Holm-adjusted test fails at 5% | 5% |

The tests also have power. `tests/test_basic_jump.py` checks that each one rejects the
violation it targets: U-shaped volatility, GARCH or autocorrelated jumps, fat tails, a curved
drift, dependent increments, dependent curves, and a jump tied to its session.

![Curve and jump diagnostics](../figures/basic_jump_curve_diagnostics.png)

*Figure 8. The diagnostics as pictures, each with its 95% band under the null. Top: volatility
through the session (flat), the variance of the curve (linear in $s$, the Brownian signature),
QQ plot of the increments (straight). Bottom: autocorrelations of the jumps, of the curves, and
of the increments within a session (all inside or near their bands).*

---

## 6. Things to consider for the basic model

The basic model adds the research hypothesis

$$
J_t = \alpha + \int_0^1 \beta(u)\, X_{t-1}(u)\,\mathrm{d}u + \eta_t .
$$

Points to settle before implementing it:

1. **Estimating $\beta(u)$.** A function can't be estimated from 62 days directly. The plan is
   to summarise yesterday's curve by a few principal component scores and regress $J_t$ on them.
   How many components to use is a choice to make carefully.
2. **Compare on the same days.** $J_t$ needs $X_{t-1}$, so day 1 is lost. The null must be
   refitted on days 2–63 for a fair comparison (log-likelihood −38.29 there).
3. **Use an exact test.** Testing $\beta = 0$ is then an ordinary regression $F$ test, which is
   exact. The usual $\chi^2$ likelihood-ratio test over-rejects at 62 days (8% instead of 5%
   with 8 components).
4. **Out-of-sample benchmark.** The null forecasts each jump by the mean of past jumps. Its
   out-of-sample error over days 22–63 is **RMSE 0.463**. A useful $\beta$ has to beat this.
5. **Power.** Three months cannot even pin down the mean jump, so $\beta$ will only be visible
   if it explains a large share of the jump's *variance*. A power study (detection rate against
   the size of $\beta$ and the number of days) will tell us how much data the question needs.
6. **Real data will break the null.** Market data will likely show U-shaped intraday
   volatility, volatility clustering and fat tails. The diagnostics above are ready to detect
   each of these, and they will tell us which extension to make first.

**Next step:** simulate the basic model with a known $\beta \ne 0$, fit it, test it against
the null, and measure the power.
