# Foundations: what the methods here are doing, and what they assume

The background a chapter has to state before any of the results in `README.md` mean anything.
It is deliberately short and it is not a survey -- it covers only what this repository actually
uses, in the order the assumptions start to matter, and it says where each one is relied on in
the code.

---

## 1. A curve is a point in a Hilbert space, and FPCA is its expansion

Every method here treats a session as an element of `L^2([0, 1])` with the usual inner product
`<f, g> = int f(t) g(t) dt`. Quadrature is trapezoidal throughout (`core.quadrature`), which is
what makes "the estimate is zero on a set of measure 0.6" a statement the code can check rather
than a count of grid points.

For a mean-zero process `X` with covariance operator `C`, the Karhunen-Loève expansion is

    X(t) = sum_{k>=1} xi_k phi_k(t),    xi_k = <X, phi_k>,    E[xi_j xi_k] = tau_k 1{j = k},

with `(tau_k, phi_k)` the eigenpairs of `C`. Functional principal component analysis is the
empirical version of this and nothing more: `far_est.fpca` is the single decomposition every
package in this repository routes through, which is the point -- `ftsa` and `jump` used to run
their own and nothing would have caught a divergence.

Two things about the eigenvalues are load-bearing later:

- They are a **sequence**, not a number. How fast `tau_k -> 0` is the single quantity that decides
  how hard a design is, and §2 says why.
- `core.gp.sample_gp` and `bench.design.variance_profile` **prescribe** them, as
  `tau_k ∝ exp{-(k-1)/decay}`. That is a modelling choice made for convenience, and
  `docs/finance-fda-literature.md` §1 argues it is the wrong one for intraday curves.

## 2. Scalar-on-function regression is an ill-posed inverse problem

The model is

    y_i = a + int psi(t) X_i(t) dt + eps_i.

Writing both sides in the eigenbasis, `psi = sum_k b_k phi_k` and the normal equations give
`b_k = E[y xi_k] / tau_k`. The numerator is estimable at the usual rate. The denominator goes to
zero. So recovering `psi` requires dividing by increasingly small numbers estimated with error,
which is the definition of an ill-posed inverse problem and the reason every estimator here
regularises somehow -- by truncating the expansion at `m` components (`_common.pilot_fpca`, BIC
over `m`), by penalising roughness, or by penalising towards zero.

Hall and Horowitz [[36]][hall-horowitz-2007] give the minimax rates, and the shape of the answer is
that **the achievable rate is governed by how fast `tau_k` decays relative to how fast `b_k` does**.
Polynomially decaying eigenvalues give polynomial rates; the faster the decay, the more ill-posed
the inversion. This is why `bench.design` goes to the trouble of matching the *spectrum* across its
three bases and varying only the eigenfunctions: without that, a comparison between bases is a
comparison between difficulties.

Cai and Hall [[19]][cai-hall-2006] then separate two things that are easy to conflate. Predicting a
new `y` is a question about `int psi X`, which lives in the well-conditioned directions; recovering
`psi` itself is a question about every direction including the badly-conditioned ones. They
converge at different rates under different conditions, and a procedure can be excellent at one
while being arbitrarily bad at the other.

That theorem is the formal version of four separate findings in this repository, which is worth
stating explicitly because they look like four accidents and are one fact:

- cross-validation never returns a null set, on any of the paper's designs or on Tecator;
- held-out error goes *flat* past the true endpoint rather than rising, so the minimiser of a
  truncation profile is not a consistent endpoint estimate and `HH_A.truncation_profile` has to be
  read at its elbow;
- on the London curve, the one-standard-error rule and EBIC zero the **whole** domain;
- method A can predict within 30% of the noise floor while its coefficient function is wrong by
  ten orders of magnitude (`notebooks/basis_and_coef_comp.ipynb`).

A rule chosen for prediction risk will discard a signal too small to pay for itself in held-out
error. When the question is *which part of the domain matters*, that is the wrong trade, and
`region` exists because of it.

## 3. Three ways to say "`psi` is simple"

The methods here differ in what structure they impose, not in the model.

**Smoothness.** `psi` has small `int (psi'')^2`. The classical choice; it never produces an exact
zero, so it cannot answer "where".

**Truncation.** `psi(t) = 0` for `t` past an unknown endpoint. One parameter, and it is the
quantity of interest. Hall and Hooker [[37]][hall-hooker-2016] estimate it by penalised least
squares with an `n lambda theta^2` penalty, which `truncFLR.HH` and `truncFLR.HH_A` implement --
and whose §3.2 tuning scheme `notebooks/truncFLR_lambdadegen.ipynb` corrects in three places.

**Local sparsity.** `psi = 0` on some unknown union of intervals. Stefanucci, Bernardi and Canale
[[78]][stefanucci-bernardi-canale-2025] get it from an overlap group lasso on B-spline
coefficients, which `fdaSP.SBC` implements: a B-spline of order `d` covers each point with `d`
basis functions, so requiring a whole *overlapping group* to vanish is what makes the estimate
exactly zero on a knot interval rather than merely small. James, Wang and Zhu
[[49]][james-wang-zhu-2009] get it instead by penalising *derivatives* of `psi`, and Lin, Cao, Wang
and Wang [[55]][lin-cao-wang-wang-2017] by a non-convex functional SCAD penalty that buys the
oracle property at the cost of convexity.

Truncation is a special case of local sparsity, which is why the two are comparable at all, and why
`docs/finance-fda-literature.md` §4.1 can put all seven papers in one table.

The identity that `region` turns on is worth isolating, because it is what makes a selection
probability a statement about the *domain* rather than about a basis: with an overlap penalty,
"group `b` is selected" and "the estimate is non-zero on knot interval `b`" are the **same event**.

## 4. Dependence: what it costs and what it does not

Both source papers assume i.i.d. curves. These panels are consecutive trading sessions.

Hörmann and Kokoszka [[42]][hormann-kokoszka-2010] give the standard weakening:
`L^p-m-approximability`, under which `X_t` is well approximated by an `m`-dependent sequence and
the usual laws of large numbers and central limit theorems survive. Bosq [[16]][bosq-2000] supplies
the FAR(1) that `far.far_dgp` simulates as the concrete example.

The distinction that matters here, and it is the whole argument of the current research direction:

> Weak dependence buys **consistency**. It does not buy the **variance calculations**, and every
> selection rule in this repository is a variance calculation.

Concretely, each of these carries an effective sample size that serial dependence invalidates:

| Rule | Where `n` enters | Used by |
|---|---|---|
| §3.2's `V(lambda) = (sigma^2/n) S_Y'' / (S_Y'' + lambda)^2` | explicitly, and the repo already corrected it once by a factor of `n` | `HH.lambda_select`, `HH_A.lambda_select` |
| `k`-fold cross-validation | folds assumed exchangeable | `SBC.cv_fit`, `HH_A.truncation_profile` |
| BIC, `n log(rss/n) + (m+1) log n` | the penalty term | `_common.pilot_fpca`, `HH_A.methodA_bic` |
| complementary-pairs subsampling | the theorem assumes independence; the code blocks the draw, so the gap is in the calibration rather than the sampling | `region.stability` |

`notebooks/overnight_jumps.ipynb` measured a version of this on real data: random folds report more
than twice the out-of-sample R-squared that date-blocked folds do, while the cross-validated error
used to *select* the penalty differs by under two per cent. So dependence shows up in the reported
number well before it shows up in the selected model -- which means checking only the selection is
not enough to conclude a rule is safe.

Chen, Guo and Qiao [[22]][chen-guo-qiao-2022] is the nearest existing treatment of FLR that drops
both the independent-curves and the i.i.d.-error assumptions, and Gabrys, Horváth and Kokoszka
[[30]][gabrys-horvath-kokoszka-2010] supply the test for whether the errors of a fitted functional
linear model are correlated at all.

## 5. Selecting for support recovery, and certifying the answer

Given §2, a rule that targets prediction will not return a null set. The alternative this
repository uses is stability selection: Meinshausen and Bühlmann
[[61]][meinshausen-buhlmann-2010] fit the same penalised estimator on many subsamples and keep what
is selected often; Shah and Samworth [[71]][shah-samworth-2013] sharpen the error bound and add
complementary pairs.

Applied to a functional penalty through the identity in §3, this gives a **three-way** partition of
the domain -- informative, null, and neither -- with a bound on each of the first two. The third
piece is the honest one: an estimator that names a null set and stops has asserted something about
every point of the domain, including the ones its data cannot separate.

Two caveats the repository has already established, and one it has not.

*Established.* A bound stated relative to a base procedure cannot rescue a signal that base
procedure never finds -- on `psi3`, which has no null region, stability selection declares part of
the domain null on every replication while never violating its own bound. And improving the base
procedure is worth more than reading it harder: adaptive weights [[84]][zou-2006] raise median
Jaccard overlap from 0.40 to 0.83 at no change to the error budget.

*Not established.* Shah and Samworth's Theorem 1 assumes independent observations, and consecutive
trading sessions are not. `region.stability.complementary_pairs` already answers that in the draw:
with `block` set to a width or to explicit labels it splits blocks rather than observations, so a
block is whole in one half of a pair and absent from the other, which is the same repair
`fdaSP.SBC.blocked_folds` makes to cross-validation. What does not exist is any result saying the
*bound* survives the change. `region/README.md` is explicit about this -- blocking "is not a proof;
the bounds are used as the calibration of a threshold" -- so the reported per-family error rate on
dependent curves is a calibration rather than a guarantee. The gap is theoretical, on a procedure
that is already implemented and tested, and it is the open question the current direction turns on;
see `docs/finance-fda-literature.md` §4.3.

<!-- references: cited -->
- **[16]** Bosq (2000), [*Linear Processes in Function Spaces*][bosq-2000].
- **[19]** Cai & Hall (2006), [Prediction in functional linear regression][cai-hall-2006].
- **[22]** Chen, Guo & Qiao (2022), [Functional linear regression: dependence and error
  contamination][chen-guo-qiao-2022].
- **[30]** Gabrys, Horváth & Kokoszka (2010), [Tests for error correlation in the functional linear
  model][gabrys-horvath-kokoszka-2010].
- **[36]** Hall & Horowitz (2007), [Methodology and convergence rates for functional linear
  regression][hall-horowitz-2007].
- **[37]** Hall & Hooker (2016), [Truncated linear models for functional data][hall-hooker-2016].
- **[42]** Hörmann & Kokoszka (2010), [Weakly dependent functional data][hormann-kokoszka-2010].
- **[49]** James, Wang & Zhu (2009), [Functional linear regression that's
  interpretable][james-wang-zhu-2009].
- **[55]** Lin et al. (2017), [Locally sparse estimator for functional linear regression
  models][lin-cao-wang-wang-2017].
- **[61]** Meinshausen & Bühlmann (2010), [Stability selection][meinshausen-buhlmann-2010].
- **[71]** Shah & Samworth (2013), [Variable selection with error control: another look at
  stability selection][shah-samworth-2013].
- **[78]** Stefanucci, Bernardi & Canale (2025), [Locally sparse estimation for functional linear
  models with scalar response][stefanucci-bernardi-canale-2025].
- **[84]** Zou (2006), [The adaptive lasso and its oracle properties][zou-2006].
<!-- /references -->

<!-- reference-links -->
[bosq-2000]: references.md#ref-bosq-2000
[cai-hall-2006]: references.md#ref-cai-hall-2006
[chen-guo-qiao-2022]: references.md#ref-chen-guo-qiao-2022
[gabrys-horvath-kokoszka-2010]: references.md#ref-gabrys-horvath-kokoszka-2010
[hall-horowitz-2007]: references.md#ref-hall-horowitz-2007
[hall-hooker-2016]: references.md#ref-hall-hooker-2016
[hormann-kokoszka-2010]: references.md#ref-hormann-kokoszka-2010
[james-wang-zhu-2009]: references.md#ref-james-wang-zhu-2009
[lin-cao-wang-wang-2017]: references.md#ref-lin-cao-wang-wang-2017
[meinshausen-buhlmann-2010]: references.md#ref-meinshausen-buhlmann-2010
[shah-samworth-2013]: references.md#ref-shah-samworth-2013
[stefanucci-bernardi-canale-2025]: references.md#ref-stefanucci-bernardi-canale-2025
[zou-2006]: references.md#ref-zou-2006
<!-- /reference-links -->
