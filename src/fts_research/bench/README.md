# bench

A controlled three-way comparison of the estimators in `truncFLR` and `fdaSP`, over nine
designs: three bases the curves are generated from, crossed with three coefficient functions
they have to recover.

    y_i = ∫_T x_i(t) psi(t) dt + eps_i,    i = 1, ..., n

The methods are method A and method B of Hall & Hooker (2016), which assume the informative
set is a single interval and estimate its endpoint, and the overlap group lasso of Stefanucci,
Bernardi & Canale (2025), which lets it be any union of intervals. The untruncated FPCA pilot
is carried alongside as the no-sparsity baseline, so what each assumption buys is visible
rather than inferred.

## Modules

**`design.py`**: the nine designs. Curves are drawn as `X_i = sum_k c_ik e_k` with
`c_ik ~ N(0, v_k)` independent, so the eigenvalues of the covariance operator of `X` are
exactly the prescribed `v_k` whatever the basis, and the three designs differ only in their
eigenfunctions. The three families are borrowed rather than reinvented: `trig` is the
expansion Hall & Hooker generate their own curves from (`core.bases.trig_basis`),
`bspline` is the one `fdaSP.SBC` expands the coefficient function in, and `legendre` belongs
to neither, so it is the design on which neither method is at home. Also here are the three
targets `psi_step`, `psi_cdf` and `psi_multi`, and `projection_gap`, which measures how much
of `psi` a design leaves outside the span of its own curves.

**`study.py`**: the runner. One replication of one cell fits every method on the same dataset,
scores each one against the truth, and returns a row per (method, tuning rule, orientation).
`run_study` spreads the replications over worker processes and `cached_study` reads the result
back from `artifacts/cache/`. One run writes two caches: the per-replication records, and the
per-grid-point frequency with which each method left each point in the support, which
`cached_support` reads and which is what turns the null-set comparison from a table into a
picture. `cached_curves` does the same for the fitted coefficient functions of a single
replication, so `notebooks/basis_and_coef_comp.ipynb` opens without re-running anything.

The figures are drawn under the shared visual system in `core/style.py`, which the two
empirical notebooks use as well: colour carries exactly one meaning -- which method -- so the
basis and the coefficient function stay facets labelled in text, `SEQUENTIAL` is the
single-hue ramp the support-frequency figure needs, and `clip_to` is what makes a panel
readable when one method's fit reaches `1e6` on a scale where the truth is order one.

## What "fairly" means here

**Orientation is estimated, not given.** The truncation model has to be told which end of the
domain carries the signal: `HH.fit` looks for a window `[0, theta]` and `HH.fit_mirror` for a
window `[u, 1]`. Handing it the orientation matching the truth would be information the group
lasso is never given, so both are fitted and the one with the smaller step-7 BIC is kept. The
`bic_pick` column marks it, and how often it agrees with the truth is a result in itself.

**Tuning is separated from estimation.** Every method is run under the rule its own paper
prescribes -- section 3.2 for methods A and B, ten-fold cross-validation for the group lasso --
and again at the `lambda` on its own grid minimising the realised error. The second is not an
estimator; it is an upper bound on what any rule could have delivered from that path, and the
gap between the two is the cost of the rule rather than of the estimator. The group lasso's
one-standard-error rule sits between them. All three of its rules read one solver run, since
`cv_fit` refits cold and `fit_path` warm-starts, and the two agree only to ADMM tolerance --
enough to put an oracle behind a rule it should not be able to lose to.

**The ordering of a basis is not free.** `orthonormalise` fixes the inner product but not the
order, and the order is what the variance profile is attached to. Index order means
"increasing oscillation" for the trigonometric and Legendre families but "left to right across
the domain" for a B-spline family, so a decaying variance on the latter piles the curves'
variation at one end of the domain and hands the truncation model an orientation before it has
seen any data. `design.basis` diagonalises `roughness_gram` inside the span instead, ordering
every family from smoothest to roughest; `pointwise_variance` shows the result is spatially
even, and `tests/test_bench.py` keeps a copy of the old design to compare against. The
argument for the ordering is that even variance alone. It used to have a second one -- method
B's section 3.2 criterion had nothing to say on the old design, `V(lambda)` undefined at every
candidate -- but that was a symptom of `S_Y^B`'s variance term being `n` times too large, as
step 5' displays it; on the corrected scale (see `truncFLR/README.md`) the criterion is
defined on the old design too, and the test says so.

**Errors are relative.** `relative_ise` divides by `int psi^2`, so 1.0 is the error of the
estimate that gives up and returns zero and the three targets are on one scale. The split over
the true null set and its complement adds back to the whole, `fdaSP.SBC_sim.ise` being careful
about a null set that is a union of intervals.

**Prediction respects each fit's own quadrature.** For the truncated fits the held-out integral
is taken over the sub-grid the coefficient was fitted on, since integrating over the whole grid
with `beta` zeroed past `theta` straddles the jump with a trapezium panel and adds a spurious
half of `beta(theta) X(theta) h` -- the correction `truncFLR.HH_A.trunc_predict` documents.

## Cost

Both `truncFLR` objectives are `RSS(theta) + n*lambda*theta^2`, and only the second term moves
with `lambda`. `hh_b_state` and `hh_a_state` therefore run the expensive `theta` scan once at
zero penalty, and `hh_b_at` and `hh_a_at` re-minimise the cached curve at each candidate,
caching the refit by the index it selects. That is what makes the oracle sweep over 119
penalties affordable. It is also the one place this package could silently diverge from the
estimators it is comparing, so the two are held against `HH.fit` and `HH_A.fit` and required to
agree exactly, not approximately.

## Usage

```python
from fts_research.bench import design, study

# One design: B-spline curves, coefficient supported on two disjoint bumps
data = design.simulate(kind="bspline", target="multi", n=200, n_grid=201, rng=0)
print(data["gap"] / data["scale"])           # fraction of psi the curves cannot see

# One replication of one cell: every method, every tuning rule, both orientations
records, curves = study.run_cell("trig", "cdf", rep=0)

# The whole study, cached under artifacts/cache/
df = study.cached_study()
print(study.summarise(df, metric="rise"))

# The fitted curves behind the nine-panel figure, cached the same way
curves = study.cached_curves()
```

The three caches are rebuilt by `python scripts/make_study_cache.py --refresh`, or by
`study.cached_study(refresh=True)` in a session, which takes a few minutes across ten worker
processes. Windows spawns rather than forks, so a *script* doing that needs the call under
`if __name__ == "__main__":`; a notebook does not, and `n_jobs=1` avoids the question.

## Tests

`pytest tests/test_bench.py` -- that every basis is orthonormal under the shared quadrature and
that the covariance spectrum comes out equal to the prescribed variance profile, that the
roughness ordering leaves no design lopsided, that each target is zero exactly on the null set
it declares, that the relative error splits additively over a union of intervals, and that
`hh_b_at` and `hh_a_at` reproduce `HH.fit` and `HH_A.fit` bit for bit. The two caches are tied
together by an identity -- integrating the averaged null mask must give the averaged null
length -- so a figure drawn from one and a table drawn from the other cannot silently describe
different runs. The palette is checked against `study.METHOD_ORDER` here, so a method added to
the study cannot go uncoloured; the rest of the visual system is checked in
`tests/test_core.py`, where it now lives. `-m "not slow"` skips the pipelines.

## References

Hall, P. and Hooker, G. (2016). Truncated linear models for functional data.
*Journal of the Royal Statistical Society: Series B* 78(3), 637-653.
[doi:10.1111/rssb.12125](https://doi.org/10.1111/rssb.12125)

Stefanucci, M., Bernardi, M. and Canale, A. (2025). Locally sparse estimation for functional
linear models with scalar response. In G. Aneiros, E. G. Bongiorno, A. Goia and M. Hušková
(eds.), *New Trends in Functional Statistics and Related Fields*, Contributions to Statistics,
chapter 62, pp. 521-527. Springer.
[doi:10.1007/978-3-031-92383-8_62](https://doi.org/10.1007/978-3-031-92383-8_62)
