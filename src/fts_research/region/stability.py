"""Which part of the domain is informative, decided by how often the penalty keeps it.

`fdaSP.SBC` fits the overlap group lasso, whose null set is whatever the penalty happened to
switch off at the selected `lambda`, and `lambda` is selected by cross-validation, which
targets prediction risk. Those are not the same target: an exact zero buys more for
describing the coefficient function than it does for predicting the response, so the
prediction optimum sits at a smaller penalty than the support optimum, and on every design
in this repository the cross-validated fit returns a null set of length zero.

This module chooses the null set directly instead, and reports what the choice can be
trusted to. The variables of the selection problem are the **knot intervals**. A B-spline of
order `d` covers each point of the domain with `d` basis functions, so the fitted function is
identically zero on knot interval `b` exactly when coefficients `b, ..., b + d - 1` all
vanish -- which is exactly group `b` of the overlap penalty. "Group `b` is selected" and "the
estimate is not identically zero on knot interval `b`" are therefore the same event, and the
selection probability of group `b` is a probability about the domain rather than about a
basis.

Complementary pairs stability selection (Shah and Samworth, 2013) estimates those
probabilities by refitting on disjoint halves of the data, and `bounds` turns them into the
two error bounds a locally sparse estimate needs: on what is wrongly called informative, and
on what is wrongly called null. The result is a three-way partition of the domain --
informative, null, and neither -- rather than a boundary asserted without a standard error.
That the third piece is usually not empty is the honest part: an estimate that names a null
set and nothing else has hidden the domain it cannot resolve.

**The base procedure is indexed by how many groups it keeps, not by the penalty.** Section
3.4 of the paper offers both, and for this problem only one of them works. Equation (8)'s
bound is driven by `theta = q / n_groups`, the average selection rate, through quantities no
larger than Markov's `theta / tau`; once `theta` exceeds a half no threshold can certify an
inclusion, because a procedure that keeps most of what it is offered keeps the noise too. A
functional penalty has few groups and a coefficient function that is usually non-zero over
much of the domain, so a fixed `lambda` anywhere near the cross-validated one puts `theta`
above a half and nothing is provable. Fixing `q` instead -- lowering the penalty on each
subsample until `q` groups have entered, which the path gives for nothing -- makes `theta`
the statistician's choice rather than the data's, and `selection_curves` returns the whole
profile in `q` so that the choice is visible.

The two claims are then read at different `q`, which costs nothing and is not a fudge: the
proportions are monotone in `q` by construction, so an inclusion certified at a small `q` and
an exclusion certified at a larger one can never contradict each other, and `select` enforces
that ordering. Sparse fits are where a group can be shown to matter, dense fits are where one
can be shown not to, and both are statements about the same coefficient function.

Two departures from the paper are deliberate and neither is free.

**Subsampling in blocks.** Shah and Samworth assume the observations are independent. Curves
indexed by trading day are not, and a panel is worse -- every instrument contributes a row on
each date and shares that date's market-wide move, which is the leak
`notebooks/overnight_jumps.ipynb` measures. `complementary_pairs` therefore splits blocks
rather than observations, so a block is whole in one half of a pair and absent from the
other. That preserves the complementary-pairs geometry the theorem is stated for and the
dependence within a block, and it is the same repair `fdaSP.SBC.blocked_folds` makes to
cross-validation. It is not a proof: the bounds are proved under independence and are used
here as they are used everywhere, as the calibration of a threshold rather than as a
guarantee about serially dependent data.

**A plug-in on the exclusion side.** The inclusion bound is scaled by the number of
candidates, which is conservative and needs nothing estimated. Its mirror is scaled by what
the base procedure itself drops, for which the number of candidates is not conservative but
vacuous, so `resolve` substitutes the unbiased proportions. `bounds.missed_exclusion_bound`
says which term that is and why there is no alternative.

References
Meinshausen, N. and Buhlmann, P. (2010). Stability selection. Journal of the Royal
Statistical Society, Series B 72(4), 417-473. doi:10.1111/j.1467-9868.2010.00740.x
Shah, R. D. and Samworth, R. J. (2013). Variable selection with error control: another look
at stability selection. Journal of the Royal Statistical Society, Series B 75(1), 55-80.
doi:10.1111/j.1467-9868.2011.01034.x
"""

import numpy as np

from ..fdaSP import SBC
from . import bounds

__all__ = [
    "knot_edges",
    "knot_interval_of",
    "group_activity",
    "entry_step",
    "complementary_pairs",
    "selection_curves",
    "resolve",
    "select",
    "regions",
]


def knot_edges(n_basis=15, order=4, lo=0.0, hi=1.0):
    """The distinct knots of the B-spline basis `core.bases.bspline_basis` builds.

    There are `n_basis - order + 1` intervals between them, one per group of the overlap
    penalty, and the fitted coefficient function is identically zero on interval `b` exactly
    when group `b` is switched off. Returns an (n_basis - order + 2,) array.

    PARAMETERS
    n_basis : int, basis functions;
    order   : int, spline order, i.e. degree plus one;
    lo, hi  : float, the ends of the domain.
    """
    if n_basis < order:
        raise ValueError(f"n_basis ({n_basis}) must be >= order ({order}).")
    return np.linspace(float(lo), float(hi), int(n_basis) - int(order) + 2)


def knot_interval_of(grid, n_basis=15, order=4):
    """Which knot interval each grid point falls in. Returns an (N,) integer array.

    The right-hand endpoint of the domain belongs to the last interval rather than to a new
    one, so the labels run from 0 to `n_basis - order`.

    PARAMETERS
    grid    : (N,) array, the sampling grid;
    n_basis : int, basis functions;
    order   : int, spline order.
    """
    grid = np.asarray(grid, float)
    edges = knot_edges(n_basis, order, grid[0], grid[-1])
    index = np.searchsorted(edges, grid, side="right") - 1
    return np.clip(index, 0, edges.size - 2)


def group_activity(psis, n_basis, order):
    """Which groups the fitted coefficients leave non-zero, along a whole path.

    Group `b` holds coefficients `b, ..., b + order - 1`, so it is active when any of them
    survives, and inactive when the estimate is identically zero on knot interval `b`.
    Returns a boolean array of shape `psis.shape[:-1] + (n_basis - order + 1,)`.

    PARAMETERS
    psis    : (..., n_basis) array of spline coefficients, exact zeros where switched off,
              as `fdaSP.SBC.fit_path` returns them;
    n_basis : int, basis functions;
    order   : int, spline order, i.e. the group size.
    """
    psis = np.asarray(psis, float)
    if psis.shape[-1] != int(n_basis):
        raise ValueError(f"psis has {psis.shape[-1]} coefficients, n_basis is {n_basis}.")
    non_zero = psis != 0.0
    windows = np.lib.stride_tricks.sliding_window_view(non_zero, int(order), axis=-1)
    return windows.any(axis=-1)


def entry_step(active):
    """Where along a descending penalty path each group first becomes active.

    The path is solved from `lambda_max` downwards, so this ranks the groups by how much
    penalty each survives: an early entry is a strong one. A group never active is given the
    path length, which sorts it behind every group that does enter. Returns an integer array
    of shape `active.shape[:-2] + (n_groups,)`.

    PARAMETERS
    active : (..., n_lambda, n_groups) boolean array, as `group_activity` returns.
    """
    active = np.asarray(active, bool)
    n_lambda = active.shape[-2]
    ever = active.any(axis=-2)
    return np.where(ever, active.argmax(axis=-2), n_lambda)


def _block_labels(n, block):
    """One label per observation, grouping those that must stay together. Returns an (n,)
    array."""
    if np.ndim(block) > 0:
        labels = np.asarray(block)
        if labels.shape != (n,):
            raise ValueError(f"block labels must have length n = {n}, got {labels.shape}.")
        return labels
    width = int(block)
    if width < 1:
        raise ValueError(f"block must be a positive integer or an array, got {block!r}.")
    return np.arange(n) // width


def complementary_pairs(n, n_pairs=50, block=1, rng=0):
    """Disjoint halves of the sample, drawn `n_pairs` times. Returns a list of `2 * n_pairs`
    index arrays, consecutive ones forming a pair.

    With `block=1` this is the paper's scheme: a random half of the observations and its
    complement, each of size `floor(n / 2)`. With a wider block, or with explicit labels, the
    split is made over blocks rather than observations, so serially dependent rows -- or the
    rows a panel contributes on one date -- are never divided between the two halves of a
    pair. Each half then takes `floor(m / 2)` of the `m` blocks, which is `floor(n / 2)`
    observations when the blocks are of one length and there is an even number of them, and
    a little fewer otherwise.

    PARAMETERS
    n       : int, observations;
    n_pairs : int, complementary pairs to draw;
    block   : int or (n,) array, the width of a contiguous block, or a label per observation
              naming the group it belongs to;
    rng     : int or Generator, seed for the draw.
    """
    n, n_pairs = int(n), int(n_pairs)
    if n < 4:
        raise ValueError(f"need at least four observations to split into halves, got {n}.")
    rng = np.random.default_rng(rng)
    labels = _block_labels(n, block)
    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError("the blocks must number at least two to form a complementary pair.")
    members = [np.flatnonzero(labels == value) for value in unique]

    half = unique.size // 2
    subsets = []
    for _ in range(n_pairs):
        order = rng.permutation(unique.size)
        for take in (order[:half], order[half:2 * half]):
            subsets.append(np.sort(np.concatenate([members[b] for b in take])))
    return subsets


def selection_curves(X, y, grid, lambdas=None, Z=None, n_basis=15, order=4, n_pairs=50,
                     block=1, group_weights=None, var_weights=None, n_lambda=50, eps=1e-3,
                     rng=0, **admm_kwargs):
    """Refit the overlap group lasso on complementary halves and record what each keeps.

    Every subsample is solved over the *same* penalty grid, taken from the full data, so the
    entry order of the groups is comparable across halves. Returns a dict holding `pi`
    (n_groups, n_groups), the proportion of subsamples on which each group survives when the
    base procedure keeps `q` groups, row `q - 1`; `q`, its row sums, the paper's estimate of
    `E|S|`; `pi_lambda` (n_lambda, n_groups), the same proportions read at fixed penalties,
    which is what a picture of the path wants; `entry` (2 * n_pairs, n_groups); the `subsets`
    drawn; and `lambdas`, `grid`, `interval`, `edges`, `n_basis`, `order`, `n_groups` and
    `n_pairs`, which the readers below need.

    PARAMETERS
    X, y, grid    : as `fdaSP.SBC.fit_path` takes them;
    lambdas       : array or None, the common penalty grid; built from the full data by
                    `fdaSP.SBC.lambda_path` when None;
    Z             : (n, p) array or None, unpenalised covariates;
    n_basis       : int, basis functions;
    order         : int, spline order;
    n_pairs       : int, complementary pairs of subsamples;
    block         : int or (n,) array, passed to `complementary_pairs`;
    group_weights : (n_groups,) array or None, adaptive weights, from
                    `criteria.adaptive_weights`;
    var_weights   : (n_basis,) array or None, passed to `fdaSP.SBC.pen_matrices`;
    n_lambda, eps : int and float, the penalty grid when `lambdas` is None;
    rng           : int or Generator, seed for the subsampling;
    **admm_kwargs : forwarded to `fdaSP.SBC.admm`.
    """
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float).ravel()
    grid = np.asarray(grid, float)
    n = y.size
    Z = None if Z is None else np.atleast_2d(np.asarray(Z, float))

    if lambdas is None:
        W, _ = SBC.bspline_design(X, grid, n_basis=n_basis, order=order)
        F, _ = SBC.pen_matrices(n_basis, order=order, group_weights=group_weights,
                                var_weights=var_weights)
        lambdas = SBC.lambda_path(W, y, F, Z=Z, order=order, n_lambda=n_lambda, eps=eps)
    lambdas = np.sort(np.asarray(lambdas, float))[::-1]

    n_groups = int(n_basis) - int(order) + 1
    subsets = complementary_pairs(n, n_pairs=n_pairs, block=block, rng=rng)
    active = np.zeros((len(subsets), lambdas.size, n_groups), dtype=bool)
    for j, take in enumerate(subsets):
        path = SBC.fit_path(X[take], y[take], grid, lambdas=lambdas,
                            Z=None if Z is None else Z[take], n_basis=n_basis, order=order,
                            group_weights=group_weights, var_weights=var_weights,
                            **admm_kwargs)
        active[j] = group_activity(path["psis"], n_basis, order)

    entry = entry_step(active)
    ranked = np.sort(entry, axis=1)                     # entry of the qth earliest group
    pi = np.stack([(entry <= ranked[:, q][:, None]).mean(axis=0) for q in range(n_groups)])
    return {
        "pi": pi,
        "q": pi.sum(axis=1),
        "pi_lambda": active.mean(axis=0),
        "entry": entry,
        "active": active,
        "subsets": subsets,
        "lambdas": lambdas,
        "grid": grid,
        "interval": knot_interval_of(grid, n_basis, order),
        "edges": knot_edges(n_basis, order, grid[0], grid[-1]),
        "n_basis": int(n_basis),
        "order": int(order),
        "n_groups": n_groups,
        "n_pairs": int(n_pairs),
        "block": block,
    }


def resolve(curves, pfer=1.0, pner=1.0, shape="r-concave"):
    """Turn the selection proportions into a decision at every size, with its error bounds.

    The informative half is Theorem 1(i) as the paper states it: at each size the average
    selection proportion `theta = q / n_groups` fixes what counts as a rarely-selected group,
    and `bounds.threshold_for_inclusion` gives the least threshold above one half whose
    expected number of false inclusions is within `pfer`.

    The null half is the same theorem applied to the **complementary** procedure. Switching a
    group off is a selection like any other: the base procedure that keeps `q` groups
    simultaneously discards `n_groups - q` of them, its selection probabilities are `1 - p_b`
    and its average rate is `1 - theta`, and Theorem 1(i) then bounds the expected number of
    groups CPSS declares null whose true selection probability is at least `theta` -- which
    is what a null region needs bounded. Theorem 1(ii) states the same thing directly but
    scales it by `E|N ∩ H_theta|`, for which the number of candidates is vacuous rather than
    conservative, so it can only be used with a plug-in; read through the complement, the
    rigorous scaling by the number of candidates applies to both halves.

    The two halves therefore need opposite sizes. Certifying an inclusion needs `theta` below
    a half, so a sparse base procedure; certifying an exclusion needs `1 - theta` below a
    half, so a dense one. Neither threshold need exist, and `tau_hi` and `tau_lo` are NaN
    where the budget cannot be met at any threshold.

    Returns a dict of arrays over the sizes: `theta`, `tau_hi`, `tau_lo`, `informative` and
    `null` (n_groups, n_groups) boolean, `n_informative`, `n_null` and `n_decided`
    (n_groups,), and `bound_false` and `bound_missed`, the two bounds actually attained.

    PARAMETERS
    curves : dict from `selection_curves`;
    pfer   : float, the expected number of falsely informative groups allowed;
    pner   : float, the expected number of falsely null groups allowed;
    shape  : one of "r-concave", "unimodal" and "worst-case", which theorem to use.
    """
    pi, q = curves["pi"], curves["q"]
    n_pairs, n_groups = curves["n_pairs"], curves["n_groups"]
    sizes = pi.shape[0]

    theta = q / n_groups
    tau_hi = np.full(sizes, np.nan)
    tau_lo = np.full(sizes, np.nan)
    bound_false = np.full(sizes, np.inf)
    bound_missed = np.full(sizes, np.inf)
    informative = np.zeros_like(pi, dtype=bool)
    null = np.zeros_like(pi, dtype=bool)

    for size in range(sizes):
        tau_hi[size] = bounds.threshold_for_inclusion(
            theta[size], pfer, n_pairs, n_groups, q[size], shape)
        if np.isfinite(tau_hi[size]):
            informative[size] = pi[size] >= tau_hi[size]
            bound_false[size] = bounds.false_selection_bound(
                theta[size], tau_hi[size], n_pairs, n_groups, q[size], shape)

        off, dropped = 1.0 - theta[size], n_groups - q[size]
        tau_off = bounds.threshold_for_inclusion(off, pner, n_pairs, n_groups, dropped, shape)
        if np.isfinite(tau_off):
            tau_lo[size] = 1.0 - tau_off                # in terms of pi rather than 1 - pi
            null[size] = 1.0 - pi[size] >= tau_off
            bound_missed[size] = bounds.false_selection_bound(
                off, tau_off, n_pairs, n_groups, dropped, shape)

    return {
        "theta": theta,
        "tau_hi": tau_hi,
        "tau_lo": tau_lo,
        "informative": informative,
        "null": null,
        "n_informative": informative.sum(axis=1),
        "n_null": null.sum(axis=1),
        "n_decided": informative.sum(axis=1) + null.sum(axis=1),
        "bound_false": bound_false,
        "bound_missed": bound_missed,
        "pfer": float(pfer),
        "pner": float(pner),
        "shape": shape,
    }


def select(curves, pfer=1.0, pner=1.0, shape="r-concave"):
    """The three-way partition of the domain, and the error bound each half carries.

    Nothing here is a risk to trade off, so nothing is cross-validated and no prediction
    error is computed: every size on the profile carries the same error budget, and what
    differs is how much of the domain the budget buys. The informative set is read at the
    size certifying the most groups informative and the null set at the size certifying the
    most null.

    Ties are broken in opposite directions, and towards the stronger statement in each case.
    Surviving a base procedure allowed to keep only a few groups is stronger evidence of
    signal than surviving one allowed to keep many, so an inclusion is read at the smallest
    tying size; being dropped by a procedure allowed to keep many is stronger evidence of a
    zero than being dropped by one allowed to keep few, so an exclusion is read at the
    largest. Reading both at the smallest is what makes a nearly-detected region look null:
    a procedure keeping four groups drops a weak fifth almost always, which says more about
    the four than about the fifth.

    Neither choice is degenerate, because both counts fall away at both ends: a base
    procedure keeping almost nothing has no proportion high enough to certify an inclusion,
    one keeping almost everything has none low enough to certify an exclusion, and each count
    is largest somewhere in between. The two sizes are chosen independently, so nothing
    forbids a group from being certified informative at one and null at the other; where that
    happens the two thresholds are contradicting each other on the same estimand and neither
    can be trusted, so `select` moves the group to undecided and reports it in `conflict`
    rather than letting an ordering convention hide it.

    Returns a dict with `size_in` and `size_out`, the two sizes; `pi_in` and `pi_out`, the
    proportions there; `tau_hi` and `tau_lo`; boolean `informative`, `null`, `undecided` and
    `conflict` over the groups and `informative_grid`, `null_grid` and `undecided_grid` over
    the grid; the bounds `bound_false` and `bound_missed`; and `profile`, the whole `resolve`
    output.

    PARAMETERS
    curves : dict from `selection_curves`;
    pfer   : float, the expected number of falsely informative groups allowed;
    pner   : float, the expected number of falsely null groups allowed;
    shape  : one of "r-concave", "unimodal" and "worst-case", which theorem to use.
    """
    profile = resolve(curves, pfer=pfer, pner=pner, shape=shape)
    size_in = int(np.argmax(profile["n_informative"]))
    n_null = profile["n_null"]
    size_out = int(n_null.size - 1 - np.argmax(n_null[::-1]))

    at = curves["interval"]
    informative = profile["informative"][size_in]
    null = profile["null"][size_out]
    conflict = informative & null
    informative = informative & ~conflict
    null = null & ~conflict
    undecided = ~(informative | null)
    return {
        "size_in": size_in + 1,
        "size_out": size_out + 1,
        "pi_in": curves["pi"][size_in],
        "pi_out": curves["pi"][size_out],
        "theta_in": float(profile["theta"][size_in]),
        "theta_out": float(profile["theta"][size_out]),
        "tau_hi": float(profile["tau_hi"][size_in]),
        "tau_lo": float(profile["tau_lo"][size_out]),
        "informative": informative,
        "null": null,
        "undecided": undecided,
        "conflict": conflict,
        "informative_grid": informative[at],
        "null_grid": null[at],
        "undecided_grid": undecided[at],
        "bound_false": float(profile["bound_false"][size_in]),
        "bound_missed": float(profile["bound_missed"][size_out]),
        "profile": profile,
    }


def regions(mask, curves):
    """The intervals of the domain a group-level mask covers. Returns a list of (lo, hi).

    The knots are the only boundaries a group-level decision can name, so the intervals are
    exact rather than read off a grid: adjacent groups are merged and the ends are knots.

    PARAMETERS
    mask   : (n_groups,) boolean array, as `select` returns;
    curves : dict from `selection_curves`.
    """
    mask = np.asarray(mask, bool)
    edges = curves["edges"]
    if mask.size != edges.size - 1:
        raise ValueError(f"mask has {mask.size} entries, there are {edges.size - 1} groups.")
    if not mask.any():
        return []
    changes = np.flatnonzero(np.diff(mask.astype(int))) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [mask.size]])
    return [(float(edges[s]), float(edges[e])) for s, e in zip(starts, ends) if mask[s]]
