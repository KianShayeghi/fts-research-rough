# Session summary: continuity and multiple breakpoints in truncFLR

2026-09-10/13. Two new estimators added to `truncFLR`, both extending Hall & Hooker
[[37]][hall-hooker-2016] beyond the one assumption method A and method B share -- a coefficient
function exactly zero on one side of a single truncation point -- plus the diagnostic that
motivated the first of them.
No new notebooks; the existing `truncFLR.ipynb` diff in this working tree is unrelated (see
below).

## Starting point

Three directions, as posed at the start of the session:

1. Separate estimation on each side of the truncation point, constrained for continuity, instead
   of zeroing one side -- for coefficient functions with no true null region.
2. Iterate Hall & Hooker's method to estimate multiple breakpoints.
3. Try coefficient functions with no true null region at all (e.g. a sine wave before the
   truncation point, increasing after it), to see how the existing estimators behave.

Item 3 needed no new code -- `HH_dgp.sim_dataset(beta_fn=...)` already accepted an arbitrary
callable -- so it became the test case for item 1 rather than a separate deliverable.

## Packages

### `src/fts_research/truncFLR/HH_continuous.py` (new)

Continuity-constrained two-region estimation. The coefficient function is

    b(t) = b1(t)  for t <= theta
           b2(t)  for t >  theta,     b1(theta) = b2(theta)

with `b1` and `b2` each their own empirical-FPC expansion of `X` restricted to that side
(`_common.trunc_fpca`), joined by one linear equality constraint solved by null-space
elimination (`scipy.linalg.null_space`) -- design "B" from the two options considered, chosen
over a single spline with a knot at theta because it lets each side carry a genuinely different
local basis rather than sharing one.

Hall & Hooker's zero-tail model is the special case `b2 == 0`, reachable but never assumed. There
is no `n*lambda*theta^2` penalty and no section-3.2 analogue -- deriving one would be new
asymptotic theory for a model the paper never poses, not an implementation detail -- so `theta`
is chosen by cross-validated prediction error (`truncation_profile`, `fit`), read at the
minimiser directly rather than at an elbow, since with no null tail there is no flat stretch for
the minimiser to wander over. A side effect of continuity: prediction (`continuous_predict`)
needs no split-quadrature correction for a jump, since there is none -- splitting the trapezium
rule at theta and summing is exact.

**API**: `fit_components`, `fit_at_theta`, `theta_scan`, `truncation_profile`, `fit`,
`continuous_predict`.

**Tests** (`tests/test_hh_continuous.py`, 5, all passing): the continuity constraint checked
directly against the fitted coefficients (`b1(theta) == b2(theta)` to `1e-8`, not inferred from
one array entry); the whole-grid/split-quadrature identity; endpoint recovery on Hall & Hooker's
own indicator design (`u=0.8`); and recovery of a sine-then-ramp design with no null region on
either side.

Two corrections to how that last test should be read, both found by running the estimators and
plotting the result rather than by reading the assertions:

- **its truth is discontinuous.** `sin(6*pi*t)` reaches 0 at `t=0.5` while the ramp `1+2(t-0.5)`
  starts at 1, so the design jumps by one unit at `theta` -- outside the continuity-constrained
  model's own class. It locates `theta` at 0.473 anyway (true 0.5, relative ISE 1.9%), so the
  test is a robustness check, not a within-class recovery check. A genuinely continuous variant
  (ramp `2(t-0.5)`, meeting the sine at zero) gives `theta_hat = 0.527`, relative ISE 2.6%.
- **"cannot represent at all" was too strong.** On the continuous variant, method B returns
  `u_hat = 0.587` at 73.5% relative ISE -- it zeroes the whole oscillating half -- but method A
  returns 3.3% relative ISE. It does that by setting `u_hat = 0.04`, i.e. by declining to
  truncate at all. The accurate claim is about the *endpoint*, not the fit: neither existing
  method returns a usable truncation point on a design with no null region, one by being wrong
  and the other by abstaining.

### `src/fts_research/truncFLR/HH_multi.py` (new)

Multiple-breakpoint estimation by binary segmentation: recursively apply `HH.fit`/`fit_mirror`
to sub-segments of the domain, the standard way to turn a single-changepoint detector into a
multiple-changepoint one.

At each segment, `segment_contest` decides whether to split at all -- `HH.fit`/`fit_mirror`
always return *some* theta, even on a segment with no real break -- by k-fold cross-validation:
fit the segment three ways (no split, truncated left, truncated right -- the same orientation
contest `bench.study.run_cell` already runs once globally, reused here per segment) and split
only if the better orientation beats the no-split fit by more than one standard error of their
**paired**, fold-by-fold CV difference. This paired form mattered in practice: an unpaired
comparison of the two candidates' marginal CV means did not have the power to detect a real but
modest break, because a BIC-selected FPCA pilot is itself flexible enough to track a step-like
coefficient function fairly closely on its own; the shared noise in fold-by-fold differences
cancels where two independent standard errors do not.

Recursion continues into *both* pieces a split produces (the "null" side can turn informative
again further along), stopping on a piece when it has too few grid points, `max_depth` is
reached, or the contest declines to split. Once the tree of breakpoints is fixed, one joint
least-squares refit (`joint_refit`) assembles the final coefficient function across all leaves
sharing a single intercept -- the many-break analogue of `HH.ab_methodB`'s refit at one
`theta_hat`. A fixed component count overfits a leaf with no real signal in it almost exactly
in-sample, so `binary_segment` uses `select_joint_refit`, which BIC-selects one component count
shared across every leaf (the many-leaf analogue of `HH_A.methodA_bic`), rather than
`joint_refit` directly.

**API**: `segment_contest`, `joint_refit`, `select_joint_refit`, `binary_segment`.

**Tests** (`tests/test_hh_multi.py`, 7, all passing): `joint_refit` partitions the domain with no
gap or overlap; a fixed component count measurably overfits a short null leaf and BIC selection
controls it; `segment_contest` finds a clean single break and declines to invent one on pure
noise; `binary_segment` reproduces both end to end, recovering a single break within 0.1 of the
truth (integrated squared error < 0.1 against the true coefficient) and correctly stopping once
found (both children offered the contest, both decline). One test pins a known **limitation**:
an informative region flanked by null on both sides is missed. *Corrected 2026-09-14*: this was
first explained as the single-edge family being unable to represent a bump, which is wrong --
the window `[0, 0.5]` contains the bump exactly. The cause is method B's first-interior-minimum
rule: on the bump design `T(theta)` has interior minima at 0.06 and 0.50, the global one is the
bump's right edge, and the rule takes 0.06. See the full write-up (`figures/ext_t2_bump_mechanism`
from `scripts/make_truncflr_extensions_figures.py`).

### `src/fts_research/truncFLR/_common.py` (modified)

`weighted_S_b` (the step-7 criterion shared by `HH.lambda_select` and `HH_A.lambda_select`) no
longer has its 33-point minimum on the integration grid, and `per_sd`/`n_sd` became module
constants. *Corrected 2026-09-14*: this was first described as a fix made while debugging
`HH_multi`; the session record does not support that. Patching the original function back in
leaves every reported result identical, and the change coarsens the quadrature where the window
is clamped at an edge of the theta grid -- worth reverting unless it was made deliberately.

### `src/fts_research/truncFLR/README.md` (modified)

Documents both new modules in the "Modules" section, adds a usage example for each, and extends
the "Tests" section with what `test_hh_continuous.py`/`test_hh_multi.py` each pin down.

## Notebooks

No new notebooks this session. `notebooks/truncFLR.ipynb` shows a diff in this working tree
(kernel display-name metadata, unicode em-dash normalisation, a re-run execution count, one added
`np.shape(X)` output cell) -- unrelated to this session's work, most likely from opening/running
the notebook separately. Worth a follow-up cell demonstrating `HH_continuous` and
`HH_multi.binary_segment` there once you've had a chance to run them on your own designs.

## Findings worth keeping

- On a coefficient function with no null region, `HH_continuous` is the only one of the three
  that returns a usable truncation point: 0.527 against a true 0.5, where method B returns 0.587
  at 73.5% relative ISE and method A returns 0.04, i.e. abstains from truncating. That is the
  empirical case for the continuous model, and it is about the endpoint rather than the fit --
  method A's *fit* is fine (3.3%) precisely because it truncated nothing.
- Dropping a correct assumption costs something. On the paper's own indicator design, where a
  true null region does exist, methods A and B both recover `u_hat = 0.797` against a true 0.8 at
  2.0% relative ISE, and `HH_continuous` gets 0.737 at 3.6%. The continuous model is the tool for
  when the null region is in doubt, not a replacement for when it is not.
- Binary segmentation over method B finds one clean break (0.29 against 0.3) but misses a bump
  flanked by null on both sides, at SNR 200. The cause is method B's first-interior-minimum
  rule, not the model family (corrected above); method A as the per-segment detector is the
  untested fix.
- The paired, fold-by-fold CV comparison in `segment_contest` was necessary, not a stylistic
  choice -- the unpaired version of the same comparison lacked the power to detect a real single
  break on the design it was tested against.

## Verification

Full suite: `pytest -q -m "not slow"` -- 1343 passed, 1 unrelated skip, 0 failures (existing
suite unaffected by the `_common.py` change). Targeted: `pytest tests/test_hh_continuous.py
tests/test_hh_multi.py tests/test_common.py tests/test_scaling.py` -- 57 passed.

## Working-tree state at end of session

```
modified:   notebooks/truncFLR.ipynb        (unrelated, see Notebooks above)
modified:   src/fts_research/truncFLR/README.md
modified:   src/fts_research/truncFLR/_common.py
modified:   uv.lock
new file:   src/fts_research/truncFLR/HH_continuous.py
new file:   src/fts_research/truncFLR/HH_multi.py
new file:   tests/test_hh_continuous.py
new file:   tests/test_hh_multi.py
```

Nothing has been committed; all of the above is unstaged in the working tree.

<!-- references: cited -->
- **[37]** Hall & Hooker (2016), [Truncated linear models for functional data][hall-hooker-2016].
<!-- /references -->

<!-- reference-links -->
[hall-hooker-2016]: references.md#ref-hall-hooker-2016
<!-- /reference-links -->
