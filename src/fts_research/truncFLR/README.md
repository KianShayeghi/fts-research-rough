# truncFLR

Scalar-on-function regression where the coefficient function is zero on part of its domain,

    y_i = ∫_T x_i(t) psi(t) dt + eps_i,

with the informative region assumed to be a single interval, so the problem is to estimate its
endpoint. Follows Hall & Hooker [[37]][hall-hooker-2016].

## Modules

**`HH.py`**: "Method B: iterative inference" (§3.1.2) with tuning-parameter selection (§3.2).
Two stages: an untruncated FPCA pilot estimate of `(a, b)` with BIC-selected component count,
then a scan over candidate endpoints minimising a penalised residual sum of squares. The
pilot is *frozen*, and only the endpoint is searched over.

**`HH_A.py`**: the same problem via "Method A: simultaneous inference" (§3.1.1). The
coefficient function is **not** frozen: at every candidate endpoint `theta`, `(a, b)` is
re-estimated by least squares in the basis of empirical FPCs of the *truncated* curves
`X_i(t) 1{t < theta}`. That separate principal components analysis at each `theta` costs
more than method B's single one. The number of retained components `m` is selected by BIC on
the final fit, as in step 7.

**`HH_dgp.py`**: the model, i.e. curves from the paper's own Section 4 recipe paired with a
coefficient function supported on one window `[u, v]`, the response being the integral over
it. The curves themselves are `core.gp.sample_gp`, a random trigonometric-basis expansion
(`psi_1 = 1`, `psi_2k = sin(2k*pi*t)`, `psi_{2k+1} = cos(2k*pi*t)`, coefficient variance
decaying as `exp{-(k-1)/decay}`), which `far.far_dgp` draws its innovations from as well, so
an estimator run on either sees curves with the same spectrum.

**`_common.py`**: what the two methods share, since §3.2 differs only in steps 4-6. The pilot
(step 1), the parametric target `b_simp` (step 3), the truncated-domain components
`phi_j^theta` and eigenvalues `tau_j^theta`, the step-7 weight densities, and the step 6-7
quantities `theta_lam_variance` and `weighted_S_b`. The truncated components are recomputed at
every candidate theta, about 800 times per method A fit, so with more curves than grid points
they come from the leading eigenpairs of the `N_theta x N_theta` Gram matrix rather than an
`n x N_theta` SVD, falling back to the SVD wherever a wanted eigenvalue is below the Gram
matrix's rounding level. At a thousand curves on a 391-point grid that, with the scores taken
as one product against the trapezium weights, brings a method A fit from 81 s to 17 s on one
thread, and the fit is the same: the same lambda, endpoint and component count, and a
coefficient function equal to 3e-13.

**`HH_continuous.py`**: drops the assumption both methods share, that the coefficient function
is exactly zero on one side of the truncation point. Instead `b(t) = b1(t)` for `t <= theta`,
`b2(t)` for `t > theta`, each its own empirical-FPC expansion of `X` restricted to that side,
joined by one linear constraint `b1(theta) = b2(theta)`. Hall & Hooker's model is the special
case `b2 == 0`, reachable but never assumed, so this is the one to reach for on a coefficient
function with no true null region at all -- a sine wave before the kink and a ramp after it,
say. There is no `n lambda theta^2` penalty and no section-3.2 analogue here (deriving one
would be new asymptotic theory, not an implementation detail); `theta` is chosen by
cross-validated prediction error instead, the same tool `HH_A.truncation_profile` reaches for
where §3.2 cannot be trusted, and read at the minimiser directly rather than at an elbow, since
with no null tail there is no flat stretch for the minimiser to wander over.

**`HH_multi.py`**: drops the other assumption both methods share, that there is only *one*
break. `binary_segment` recurses Hall & Hooker's single-split estimator over the domain --
standard binary segmentation, adapted to a functional-coefficient break instead of a mean shift.
At each segment it fits three candidates (no split, truncated from the left, truncated from the
right -- the same orientation contest `bench.study.run_cell` runs once globally, run here per
segment) and splits only if the better orientation beats the no-split fit by more than one
standard error of their *paired*, fold-by-fold CV difference; an unpaired comparison of the two
candidates' marginal CV errors does not have the power to detect a real but modest break, since
a BIC-selected FPCA pilot is itself flexible enough to track a step-like coefficient function
fairly closely. Once the recursion settles on a set of breakpoints, `select_joint_refit` does
one BIC-selected joint refit across every leaf, sharing a single intercept -- the many-break
analogue of `ab_methodB`'s refit at a single `theta_hat`, and needed because a fixed component
count fits a leaf with no real signal in it almost exactly in sample.

This recovers a clean single break reliably (it is, after all, one call to the same estimator
`HH.py` already validates), but it misses an informative region with null on both sides of it.
That is not because a single-edge window cannot represent one -- inside the kept window `b` is
unrestricted, so `[0, 0.5]` contains a bump on `[0.2, 0.5]` exactly -- but because of how method
B *chooses* the window. Its theta scan takes the first interior minimum of
`T(theta) = RSS + n lambda theta^2` (step 6'), and a null stretch at the window's anchored end
makes `T` nearly flat there, so the pilot's ringing over that stretch produces a shallow early
dip. On the bump design of `tests/test_hh_multi.py` (SNR 200) `T` has interior minima at 0.06
and 0.50; the global minimum, 0.50, is the bump's right edge, and the first-minimum rule takes
0.06. `HH_multi` inherits the rule by calling `HH.fit`/`fit_mirror` per segment, and the test pins
the resulting behaviour. Method A takes the global minimum and is the obvious per-segment detector
to try instead (untested); `fdaSP.SBC`'s overlap group lasso already handles "any union of
intervals" in one fit (`region/README.md`). `scripts/make_truncflr_extensions_figures.py` draws
the objective (`figures/ext_t2_bump_mechanism`).

**`HH_A.truncation_profile`**: the model-free alternative to §3.2, for real data where there
is no true endpoint to check a criterion against. It fits the truncated model at each candidate
`theta` and scores it by cross-validated prediction error, so "where does the informative region
end" is answered by what the retained region can actually predict.

Read the profile at its elbow, not at its minimum. Past the true endpoint the retained curve
carries no signal, so extending it costs only estimation variance and the error flattens rather
than rising; the minimiser then wanders over that flat stretch and is not a consistent endpoint
estimate. `theta_1se`, the shortest window scoring within one standard error of the best, asks
where the error *stopped falling*, which is the question the endpoint answers, and on simulated
data it recovers true endpoints of 0.3, 0.5 and 0.7 where the minimiser returns 0.84, 0.84 and
0.68. `mirror=True` profiles `[u, 1]` instead. This is the same split between prediction and
support recovery that decides `lambda` for `fdaSP.SBC`, reached from the other direction.

## Conventions worth knowing

**`fit` estimates the right endpoint; `fit_mirror` estimates the left one.** The paper's window
is `[0, theta]`, so `fit` returns the estimated right endpoint as `theta_hat`. When the
informative window is `[u, 1]` and it is `u` that is unknown, `fit_mirror` reflects the
curves `X(t) → X(1-t)`, runs the paper's method, then maps back with `u_hat = 1 - theta_hat`.
The reflection is applied by reversing the columns of `X`, so the grid must be symmetric about
its midpoint; `fit_mirror` raises otherwise. Internally every variable named `theta` lives in
the reflected coordinate system and every `u` lives in the original one; `b_hat` is reflected
back before being returned, and the diagnostics carry both (`theta_lams` and `u_lam`,
`theta_bar` and `u_bar`).

**Grid integration is trapezoidal throughout**, via `np.trapezoid`.

## Reading section 3.2

Steps 5 and 5' define `S_Y` and `S_b` as conditional expectations given the curves, over the
noise of the hypothetical data `Y*_i = a_check + int b_simp X_i + eps*_i` of (3.6). Both are
therefore quantities that can be checked rather than read, and three of the four displays
that evaluate them are out by a factor. This module implements the corrected ones;
`notebooks/truncFLR_lambdadegen.ipynb` works through the consequences and
`tests/test_scaling.py` pins every term against the expectation it is meant to equal.

**The variance terms are per-observation.** The step-4' fit gives coefficients
`beta_j` variance `sigma^2 / sum_i (score_ij - mean)^2 = sigma^2/(n tau_j)`, since `tau_j` is
an eigenvalue of the covariance operator -- a variance per curve, which is what any FPCA
routine returns. The displayed `sigma^2 sum_j (tau_j)^-1` therefore overstates the variance
by `n` in `S_Y^B`, `S_b^B` and `S_b^A` alike. Method A's `S_Y^A` is the exception and needs no
correction: `sigma^2 (m + 1)` is already the summed prediction variance of an
`(m + 1)`-parameter fit, and it is the one variance term written without a `tau`.

**`S_b^B` is governed by the full-interval eigenvalues.** Method B estimates `b_check*` once,
on all of `I`, and truncation only restricts where that one estimate is read. Its variance is
set by `tau_j^1` throughout, so

    S_b^B(theta) = int_I (b_simp - b_check* I(t < theta))^2
                   + (sigma^2/n) sum_j (tau_j^1)^-1 int_0^theta (phi_j^1)^2,

where step 5' writes `tau_j^theta`. That substitution is not a matter of scale: `tau_j^theta`
vanishes as `theta -> 0`, so the displayed term diverges precisely where the estimator's
variance is smallest, less of `b_check*` being retained. Method A's `tau_j^theta` is correct
for method A, which does refit in the truncated components at every `theta`.

**`lambda` is on the scale of (3.4).** Equation (3.4) writes the penalty `n*lambda*theta^2`,
the `n` inserted because "both `S` and `S_1` are of order `n`". Step 6 then pairs `S_Y` with a
bare `lambda`, in `S_Y + lambda*theta^2` and again in

    V(lambda) = (sigma^2 / n) * S_Y'' / (S_Y'' + lambda)^2.

One `lambda` serves both only if the `S_Y` of step 6 is the per-observation error `S_Y/n`, and
that is also the only reading under which `V` is of order `sigma^2/n` rather than
`sigma^2/n^2`. What `lambda` can do to `V` is the ratio `lambda/S_Y''`; reading `S_Y` as a sum
divides it by `n`, and over the whole candidate grid `lambda` then moves the denominator by a
few percent where per observation it moves it by more than a factor of fourteen.

**Step 7 is integrated on a grid that resolves its weight.** With `theta_lambda` on a grid
point, the trapezium rule applied to the step-7 Gaussian is exact up to aliasing, and Poisson
summation gives the factor: `sum_m exp(-2 pi^2 m^2 V / h^2)`, for `h` the grid spacing. That is
`1.014` when `sqrt(V)` is half a grid step and `h/sqrt(2 pi V)` once `sqrt(V) << h`, where the
integral degenerates into the single sample `h*S_b(theta_lambda)/sqrt(2 pi V)` and stops being
an average of `S_b`. `weighted_S_b` integrates on a grid refined to the width of the weight, so
the criterion does not depend on `V` happening to be resolvable.

Taken together the three restore section 3.2 to what its text describes. `S_Y^B` drops away
near the right-hand end of the interval, which is why step 6' takes its *first* interior
minimum rather than its global one; `S_b^B` turns down where the truncation matches the
step-3 cutoff instead of diverging at zero; and `P_b` is an average of `S_b` with an interior
minimiser. Left as displayed, `lambda_hat` is the smallest candidate offered on every seed
tried, and method B's endpoint comes back about nine grid steps short.

**Both methods run the paper's procedure otherwise**, at the BIC-selected `m`.
`lambda_select`'s `m_tuning` argument caps the component count inside the section 3.2 criterion
only, leaving the pilot and the final fit alone. It is a handle on how many components the
criterion charges for, useful for keeping the tuning cheap on large grids; neither method needs
it.

## Usage

```python
import numpy as np
from fts_research.truncFLR import HH, HH_A, HH_dgp

# Hall & Hooker: recover the left endpoint of the informative window
data = HH_dgp.sim_dataset(n=200, n_grid=301, u=0.8, v=1.0, snr=26.25, rng=0)
lambdas = np.linspace(0.0, 1.0, 250)[1:]

res_b = HH.fit_mirror(data["X"], data["Y"], data["grid"], lambdas)
res_a = HH_A.fit_mirror(data["X"], data["Y"], data["grid"], lambdas)

print(res_b["u_hat"], res_a["u_hat"], "vs true", data["u"])
print("method A BIC-selected components:", res_a["m_hat"])
```

Where §3.2 cannot be trusted -- on real data, where there is no truth to check it against --
read the endpoint off held-out prediction instead:

```python
prof = HH_A.truncation_profile(data["X"], data["Y"], data["grid"], mirror=True)
print(prof["theta_1se"], "elbow, against", prof["theta_hat"], "at the minimum")
```

For the "any union of intervals" locally sparse approach (Stefanucci, Bernardi & Canale
[[78]][stefanucci-bernardi-canale-2025]), see `fdaSP` (`fdaSP/README.md`).

Where the assumption behind both methods -- a coefficient function exactly zero on one side --
does not hold at all, `HH_continuous` drops it instead of working around it:

```python
from fts_research.truncFLR import HH_continuous

# A coefficient function with no null region on either side of its kink
def beta_fn(t, theta=0.5):
    return np.where(t < theta, np.sin(6 * np.pi * t), 1.0 + 2 * (t - theta))

data = HH_dgp.sim_dataset(n=300, n_grid=151, beta_fn=beta_fn, snr=50.0, rng=3)
res = HH_continuous.fit(data["X"], data["Y"], data["grid"], m1=5, m2=5)
print(res["theta_hat"], "vs true", 0.5)
```

Where there might be more than one break, `HH_multi.binary_segment` recurses `HH.fit`/
`fit_mirror` over the domain instead of calling either once:

```python
from fts_research.truncFLR import HH_multi

data = HH_dgp.sim_dataset(n=300, n_grid=101, u=0.3, v=1.0, snr=40.0, rng=1)
res = HH_multi.binary_segment(data["X"], data["Y"], data["grid"], lambdas,
                              n_folds=5, max_depth=3)
print(res["breakpoints"], "vs true", [data["u"]])
```

## Tests

`pytest tests/test_hh_a.py tests/test_hh_b.py tests/test_common.py tests/test_scaling.py` —
the truncated FPCA's orthonormality in the L2([0, theta]) inner product, the agreement of the
Gram-matrix decomposition with the SVD on its own and through both methods end to end, the
equivalence of the vectorised and plain theta scans, the mirroring algebra, the scale and quadrature of steps 6
and 7, end-to-end endpoint recovery, and the profile's elbow recovering an endpoint in both
orientations where its minimum does not. `test_scaling.py` is the one that holds §3.2 to its
own definitions: it re-derives every S_Y and S_b term from the least-squares algebra,
independently of `lambda_select`, and confirms that derivation by simulating the step-4' noise.
`pytest tests/test_hh_continuous.py` checks the continuity constraint directly against the
fitted coefficients (not just that the array has one entry at theta), the whole-grid/split
quadrature identity `continuous_predict` relies on, and endpoint recovery on both a design with
a true null region and one with none at all. `pytest tests/test_hh_multi.py` checks
`joint_refit`'s partition has no gap or overlap, that BIC selection actually controls the
overfitting a fixed component count produces on a short null leaf, that `segment_contest` finds
a clean break and declines to invent one on pure noise, that `binary_segment` reproduces both
end to end and stops once the true structure is found, and pins the informative-flanked-by-null
(or reverse) limitation as known rather than accidental. `-m "not slow"` skips the pipelines
that call `HH.fit`/`fit_mirror` inside cross-validation.
`-m "not slow"` skips the full pipelines.

## References

The works this package rests on. Full entries, with journals and DOIs, are in
[the repository's one bibliography](../../../docs/references.md); the numbers below
link into it.

<!-- references: hall-hooker-2016, stefanucci-bernardi-canale-2025 -->
- **[37]** Hall & Hooker (2016), [Truncated linear models for functional data][hall-hooker-2016].
- **[78]** Stefanucci, Bernardi & Canale (2025), [Locally sparse estimation for functional linear
  models with scalar response][stefanucci-bernardi-canale-2025].
<!-- /references -->

<!-- reference-links -->
[hall-hooker-2016]: ../../../docs/references.md#ref-hall-hooker-2016
[stefanucci-bernardi-canale-2025]: ../../../docs/references.md#ref-stefanucci-bernardi-canale-2025
<!-- /reference-links -->
