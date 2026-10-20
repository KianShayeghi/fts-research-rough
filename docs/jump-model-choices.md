# Three choices in the joint jump model, and what each one means

The joint model of `jump.joint_dgp` writes the overnight jump and the session after it as a
pair of equations, in percent:

    J_t    = alpha + rho J_{t-1} + ∫ beta(u) X_{t-1}(u) du + eta_t,
    X_t(s) = mu(s) - gamma_t G(s) J_t + ∫ Psi(s, r) X_{t-1}(r) dr + eps_t(s).

`J_t` is the overnight jump and `X_t(s)` the cumulative intraday return, the curve that
starts at zero at the open. Most of the model is settled. Three choices in it are not, and
each changes what the model says about markets rather than how it is estimated. This page
explains them without the code; `notebooks/jump_equation_v2.ipynb` simulates each one, on data
the model generated itself, and shows a test that settles it. Section numbers below refer to
that notebook.

## 1. Does the jump respond to the level of yesterday's curve, or to its returns?

### Two readings of one term

`∫ beta(u) X_{t-1}(u) du` puts the coefficient on the *level* of yesterday's curve: on how far
the price had moved since the open by each time `u`. The other reading puts it on the
*returns*, the move made in each moment of the day: `∫ b(v) dX_{t-1}(v)`. Because every curve
starts at zero, integration by parts turns one into the other:

    ∫ beta(u) X(u) du  =  ∫ B(v) dX(v),        B(v) = ∫_v^1 beta(u) du.

So both readings say the same *kind* of thing: the jump responds to a weighted sum of
yesterday's returns. What differs is which weights look simple. Take the notes' step at 0.85
in each reading.

| | step on the level, `beta = 1{u >= 0.85}` | step on the returns, `b = 1{v >= 0.85}` |
|---|---|---|
| regressor | `∫_{0.85}^1 X(u) du` | `X(1) - X(0.85)` |
| weight on a return made at `v` | `0.15` before 0.85, falling to 0 at the close | 0 before 0.85, 1 after |
| what it says | the jump responds to the whole day's move, late moves discounted | only the move in the last 15% of the session matters |
| the onset at 0.85 is | a kink in the weights | a switch |

### "Surely there is a random part on the level?"

Yes, and that is the whole difference. The level at any late time carries every return made
before it. The level-step regressor is `0.15 X(0.85)`, the morning's entire accumulated move,
plus a small piece from the last stretch. In the notebook's design the morning is 94% of its
variance, and the two regressors correlate only 0.18 (section 7.1). They are close to being
different variables.

- If the truth is the **returns** model, that morning component is pure noise to the jump,
  and a regression on the level is mostly a regression on noise.
- If the truth is the **level** model, the morning *is* the signal, and a regression on the
  late returns throws it away.

What the level is not is "more random" in the time-series sense. Every day's curve restarts at
zero, so `X_t(u)` is stationary from one day to the next, and regressing on it carries no
spurious-regression problem.

### Which one is the jump predictable on?

Whichever one generated it. That is the point of studying it in simulation, where the truth is
a choice. The notebook builds both truths, and an even mix of the two, all at the same
population R-squared, and computes what each model would capture with infinite data
(section 7.2):

- **Level truth.** The level step captures all of it, and the return step at 0.85 about a
  tenth. A return step spanning the whole day, which is just the day's return `X(1)`, still
  gets over 90%, because a level step weights every morning return equally.
- **Return truth.** The level step captures about an eighth, and no level step at any onset
  reaches a quarter. Smooth functions of the level need many principal components to get
  close, because `X(1) - X(0.85)` evaluates the curve at two points.
- **Mixed truth.** Each step alone captures about 60%, and the two together capture all of it.

The costs are asymmetric. Wrongly assuming returns, when the truth is on levels, costs little,
provided the window is wide. Wrongly assuming levels, when the truth is on returns, costs most of
the signal.

### How to decide, from data

Regress the jump on both regressors at once, at the hypothesised onset, and test each slope.
With one non-shared regressor on each side, each t-test is exactly the J test for non-nested
models [[26]][davidson-mackinnon-1981]: it asks whether one model adds anything once the other
is in. In the notebook's simulations it names the right model from a single year of data, and
under the mixed truth it finds both terms (section 7.2, `joint_est.nesting_test`).

The onset itself is a separate question. With the onset unknown, a step on the returns is
located on exactly the right five-minute bar in almost every ten-year sample, while a step on
the level is located only loosely. Hall & Hooker's estimator [[37]][hall-hooker-2016] does no
better on the level, because neighbouring levels of a cumulative curve are nearly collinear
(section 7.2, `joint_est.onset_profile`).

**Recommendation.** If the hypothesis is "the last part of the session predicts the overnight
move", write the term on returns, `kappa (X_{t-1}(1) - X_{t-1}(theta))`. If it is genuinely open,
fit both and let the nesting test decide.

## 2. The sign of the FAR kernel: continuation or reversal

`Psi(s, r)` says how yesterday's curve moves today's. The term `∫ Psi(s, r) X_{t-1}(r) dr` adds
to today's expected curve a weighted sum of yesterday's.

- **`Psi >= 0`**, as the notes' `s exp(-(s - r)^2 / c)` is: after a day that rose, today is
  pushed up. Good days follow good days. This is continuation, or momentum.
- **`-Psi`**: after a day that rose, today is pushed down. This is reversal, or mean reversion.

The notes' kernel is never negative, so it can only produce the first, and which of the two
holds is exactly the kind of thing the model should be able to find out. In the simulation,
flipping the sign turns a lag-one autocorrelation of the day's return of about +0.18 into about
-0.26. Both versions are stationary, and the estimator recovers either (section 8,
`joint_dgp.forecast` for what each one expects of tomorrow).

**Recommendation.** Write the kernel as `kappa s exp(-(s - r)^2 / c)` with `kappa` free in sign;
`notes_design(psi_scale=...)` is that scale.

## 3. The shape of the gap-fill: when during the day the jump is given back

After a jump `J_t`, the curve equation subtracts `gamma_t G(s) J_t`: by time `s`, the session has
given back a fraction `gamma_t G(s)` of the overnight gap. `G` runs from 0 at the open to 1 at
the close, so `gamma_t` is how much of the gap is given back by the close, and `G` is *when* it
happens.

- **`G(s) = s`**, the notes' choice: the gap is given back at a steady rate all day.
- **Front-loaded**, `G(s)` proportional to `1 - exp(-s / tau)`: most of it goes in the first
  `tau` of the session, the way a gap is often said to be "filled in the first hour", and the
  price then wanders as usual.

The two agree about the close and about `gamma`. They disagree about every moment in between,
which is what an intraday model is for (section 9.1 draws the expected price path under
each).

The model does not have to assume either. The curve equation's slope on the jump, `b(s)`, is
the whole gap-fill profile, estimated pointwise with no shape imposed. `joint_est.shape_test`
then asks whether it is a multiple of `s`. Its null distribution is a weighted sum of
chi-squares, as in the functional F test of [[76]][shen-faraway-2004], and the p-value is the exact
tail by Imhof's method [[48]][imhof-1961]. It holds its size, and it detects a gap filled in the
first tenth of the session from about a year of data (section 9.2).

**Recommendation.** Estimate `b(s)` unrestricted first and test `G(s) = s` against it. Use a
parametric `G` only where the test does not reject.

## A note on `gamma_t`

`gamma_t ~ U(0, 1)` restricts the gap to being *filled*. A negative `gamma` would mean the
session extends the overnight move, and estimation should allow it. A random `gamma_t` does not
bias the estimate of its mean, but its variance is identified only through how the residual
spread grows with `J_t^2`. That takes decades of data in the notebook's design (section 9.3), and
on real data it is confounded with sessions simply being more volatile after big jumps.

<!-- references: davidson-mackinnon-1981, hall-hooker-2016, shen-faraway-2004, imhof-1961 -->
- **[26]** Davidson & MacKinnon (1981), [Several tests for model specification in the presence of
  alternative hypotheses][davidson-mackinnon-1981].
- **[37]** Hall & Hooker (2016), [Truncated linear models for functional data][hall-hooker-2016].
- **[48]** Imhof (1961), [Computing the distribution of quadratic forms in normal
  variables][imhof-1961].
- **[76]** Shen & Faraway (2004), [An F test for linear models with functional
  responses][shen-faraway-2004].
<!-- /references -->

<!-- reference-links -->
[davidson-mackinnon-1981]: references.md#ref-davidson-mackinnon-1981
[hall-hooker-2016]: references.md#ref-hall-hooker-2016
[imhof-1961]: references.md#ref-imhof-1961
[shen-faraway-2004]: references.md#ref-shen-faraway-2004
<!-- /reference-links -->
