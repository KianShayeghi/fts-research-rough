# region

Where the coefficient function is zero, chosen for that and not for prediction, and returned
with a bound on how wrong it can be.

`fdaSP.SBC` fits the overlap group lasso, whose whole purpose is a coefficient function that
is exactly zero on part of its domain. Which part depends on `lambda`, and `lambda` is chosen
by cross-validation, which minimises prediction error. Those are different targets and they
have different optima: an exact zero is worth more for describing a coefficient function than
for predicting a response, so the prediction optimum sits at a smaller penalty. This
repository has measured the gap three times and it is not marginal --
`notebooks/LocSpaFLR_f2s.ipynb` finds the cross-validated fit returning a null set of length
zero on all three of the paper's own designs, `notebooks/basis_and_coef_comp.ipynb` on all
nine of its own, and `notebooks/tecator_application.ipynb` finds the null region the paper
reports sitting `7.75` standard errors from the cross-validated minimum. The estimator can
represent a null set; the rule selecting it cannot find one.

This package selects it directly. The claim it returns is not a point estimate of a boundary
but a three-way partition of the domain -- **informative**, **null**, and **neither** -- with
an error bound on each of the first two. The third piece is the honest one. An estimator that
names a null set and stops has asserted something about every point of the domain, including
the points its data cannot separate, and `notebooks/HH_vs_SBC.ipynb` records the cost of that
in the other direction: what the truncation model has and the penalty does not is "an explicit
endpoint estimate with a sampling distribution, rather than a null set read off whichever
basis groups the penalty switched off". This is the missing distribution theory, supplied
without assuming one.

## How

**The variables are knot intervals.** A B-spline of order `d` covers each point of the domain
with `d` basis functions, so the estimate is identically zero on knot interval `b` exactly
when coefficients `b, ..., b + d - 1` all vanish -- which is exactly group `b` of the overlap
penalty. "Group `b` is selected" and "the estimate is not zero on knot interval `b`" are the
same event, so a selection probability here is a probability about the domain and not about a
basis. `tests/test_region.py` checks that correspondence against the basis itself rather than
taking it on trust, because everything else rests on it.

**The probabilities come from complementary pairs.** Refit the estimator on disjoint halves of the
data, `n_pairs` times, and record how often each group survives. Shah and Samworth's
[[43]][shah-samworth-2013] Theorem 1 then bounds the expected number of rarely-selected groups a
threshold keeps, under no assumption about the model, the estimator, or the number of pairs.

**Both halves of the claim use the same theorem.** Theorem 1 has a second part bounding what
is wrongly excluded, but it scales by `E|N ∩ H_theta|`, for which the number of candidates is
not conservative but vacuous, so it can only be used with a plug-in. Switching a group off is
itself a selection, though: the base procedure keeping `q` groups simultaneously discards the
other `n_groups - q`, its selection probabilities are `1 - p_b`, and Theorem 1's *first* part
applied to that procedure bounds the expected number of groups declared null whose true
selection probability is high -- the same statement, with the rigorous scaling. That is the
package's own step and it is what makes the null half of the partition carry a real bound.

**The base procedure is indexed by size, not by penalty.** Equation (8)'s bound is driven by
`theta = q / n_groups`, through quantities no larger than Markov's `theta / tau`; once `theta`
passes a half no threshold certifies anything, because a procedure keeping most of what it is
offered keeps the noise too. A functional penalty has few groups and a coefficient function
usually non-zero over much of the domain, so a fixed `lambda` near the cross-validated one
puts `theta` well above a half and nothing is provable. Lowering the penalty on each subsample
until `q` groups have entered -- which the path gives for nothing -- makes `theta` a choice
rather than an accident. The two halves then want opposite sizes, and get them: an inclusion
is certified where the procedure is sparse, an exclusion where it is dense, and the
proportions being monotone in size the two sets do not collide.

**The subsampling is in blocks.** The theorem assumes independent observations. Curves indexed
by trading day are not, and a panel is worse -- `notebooks/overnight_jumps.ipynb` measures a
random split reporting more than twice the out-of-sample R-squared that date-blocked folds do,
because nine instruments share each date's market-wide move. `complementary_pairs` splits
blocks, so a block is whole in one half of a pair and absent from the other. That is the same
repair `fdaSP.SBC.blocked_folds` makes to cross-validation and it is not a proof; the bounds
are used as the calibration of a threshold, as they are everywhere.

## Modules

**`bounds.py`**: the error control, ported from the paper rather than approximated. Three
bounds on one quantity, in increasing order of what they assume: Theorem 1(i), which assumes
nothing; Theorem 2, which assumes the paired proportions are unimodal; and equation (8), which
assumes `r`-concavity and is the one to use. That last needs `D(eta, t, n_pairs, r)`, the
largest tail an `r`-concave distribution on the lattice with mean at most `eta` can have,
which has no closed form; `r_concave_tail` computes it by the algorithm of the paper's Section
A.4, parametrising the extremal mass functions by their support and one scale.

Section A.4 is stated for a decreasing maximiser, which holds when `t > 2 eta`, and sets the
other case aside as "less important for our application". It is not unimportant here -- a
functional penalty's groups are few and often selected, so `eta` is routinely above a half and
the condition routinely fails. Outside it `r_concave_tail` returns Markov's `eta / t`, which
holds for every distribution. Getting that wrong makes the reported error control too small
rather than too large, which is the one direction it may not fail in, so the tests check the
bound against Markov's inequality, against explicitly constructed members of the family it is
a maximum over, and against its own monotonicity.

**`stability.py`**: the estimator. `selection_curves` refits on the complementary halves and
returns the selection proportions as a surface in the size, `resolve` turns them into a
decision at every size with the two bounds attached, and `select` reads off the partition.
`regions` reports it as intervals, whose ends are knots rather than grid points, because the
knots are the only boundaries a group-level decision can name.

**`criteria.py`**: what it is measured against, and the measurements. `ebic` is Chen and
Chen's criterion, the competitor that prices a coefficient rather than a prediction -- with
the caveat, documented on the function and checked in the tests, that its extra term counts
models of a given size and so is *not* monotone at the sizes a small basis works at.
`adaptive_weights` reweights the penalty by a pilot fit, the group analogue of Zou's adaptive
lasso, which is the standard route to selection consistency and the one thing here that
improves the base procedure rather than reading it better. `relaxed_fit` removes the shrinkage
from whatever was kept, so that rules selecting different sets at different penalties produce
comparable coefficient functions. `support_error` scores a null set as measures on the domain,
split into the half that asserts something false and the half that only fails to assert
something true.

**`study.py`**: five rules over one path on four designs, scored on recovery rather than on
error. The designs are Stefanucci, Bernardi and Canale's three -- the third of which has no
null region and is the control against inventing one -- and a fourth whose null set is
`[0, 0.7)`, which is Hall and Hooker's own configuration and the shape `jump` asks about.
`scripts/make_region_cache.py` runs it; `notebooks/region_selection.ipynb` reads the result.

## Usage

```python
import numpy as np

from fts_research.fdaSP import SBC_sim
from fts_research.region import criteria, stability

X, y, grid, beta = SBC_sim.simulate(SBC_sim.psi2, n=100, m=100, snr=4.0, rng=1)

curves = stability.selection_curves(X, y, grid, n_basis=15, order=4, n_pairs=50, rng=0)
chosen = stability.select(curves, pfer=1.0, pner=1.0)

print(stability.regions(chosen["informative"], curves))    # certified non-zero
print(stability.regions(chosen["null"], curves))           # certified zero
print(stability.regions(chosen["undecided"], curves))      # neither, and said so
print(chosen["bound_false"], chosen["bound_missed"])       # what each half is worth
```

On serially dependent curves, or on a panel, pass the blocks the subsampling must not split:

```python
from fts_research.region import stability

curves = stability.selection_curves(X, y, grid, n_pairs=50, block=5)          # runs of five
curves = stability.selection_curves(X, y, grid, n_pairs=50, block=session)    # or by date
```

## Tests

`pytest tests/test_region.py` -- the bound against Markov's inequality and against members of
the family it maximises over, since an error there is silent and in the unsafe direction; the
three bounds against each other, and the worst case against Meinshausen and Buhlmann's own
statement of it; the group-to-knot-interval correspondence against the basis; the
complementary pairs for disjointness, for never splitting a block, and for honouring explicit
labels; the proportions for monotonicity in the size and for living on the lattice they must;
`resolve` for deciding nothing it cannot certify and for deciding less under a tighter budget;
`select` for partitioning the domain; and the whole procedure on a design whose null set is
known, where what is certified null must be null.

## References

The works this package rests on. Full entries, with journals and DOIs, are in
[the repository's one bibliography](../../../docs/references.md); the numbers below
link into it.

<!-- references:
  chen-chen-2008, meinshausen-2007, meinshausen-buhlmann-2010, shah-samworth-2013, wang-leng-2008,
  zou-2006
-->
- **[16]** Chen & Chen (2008), [Extended Bayesian information criteria for model selection with
  large model spaces][chen-chen-2008].
- **[36]** Meinshausen (2007), [Relaxed lasso][meinshausen-2007].
- **[37]** Meinshausen & Bühlmann (2010), [Stability selection][meinshausen-buhlmann-2010].
- **[43]** Shah & Samworth (2013), [Variable selection with error control: another look at
  stability selection][shah-samworth-2013].
- **[49]** Wang & Leng (2008), [A note on adaptive group lasso][wang-leng-2008].
- **[53]** Zou (2006), [The adaptive lasso and its oracle properties][zou-2006].
<!-- /references -->

<!-- reference-links -->
[chen-chen-2008]: ../../../docs/references.md#ref-chen-chen-2008
[meinshausen-2007]: ../../../docs/references.md#ref-meinshausen-2007
[meinshausen-buhlmann-2010]: ../../../docs/references.md#ref-meinshausen-buhlmann-2010
[shah-samworth-2013]: ../../../docs/references.md#ref-shah-samworth-2013
[wang-leng-2008]: ../../../docs/references.md#ref-wang-leng-2008
[zou-2006]: ../../../docs/references.md#ref-zou-2006
<!-- /reference-links -->
