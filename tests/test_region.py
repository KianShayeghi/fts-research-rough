"""Tests for `region`: the error bounds, the stability selection built on them, and the study.

The bounds are the part that has to be right. `bounds.r_concave_tail` computes a maximum over
a family of distributions with no closed form, by an algorithm the paper states for one case
and leaves the other aside, and an error there is silent -- it makes the reported error
control too small rather than too large, which is the one direction it may not fail in. It is
therefore checked three ways: against Markov's inequality, which no distribution can beat;
against explicitly constructed members of the family it is supposed to be a maximum over,
which it must not fall below; and against its own monotonicity in both arguments.

The rest is checked against constructions whose answer is known: the knot intervals against
the basis they are supposed to index, the complementary pairs against the disjointness the
theorem needs, and the whole procedure against designs whose null set is known by
construction.
"""

import numpy as np
import pytest

from fts_research.core.bases import bspline_basis
from fts_research.fdaSP import SBC, SBC_sim
from fts_research.region import bounds, criteria, stability, study


# --------------------------------------------------------------------------- bounds ------

GRID = [(0.02, 0.3), (0.05, 0.4), (0.05, 0.9), (0.1, 0.5), (0.2, 0.7), (0.3, 0.95),
        (0.45, 0.8), (0.6, 0.9), (0.75, 0.99)]


@pytest.mark.parametrize("eta,t", GRID)
@pytest.mark.parametrize("n_pairs,r", [(15, -0.5), (50, -0.5), (100, -0.25)])
def test_r_concave_tail_never_beats_markov(eta, t, n_pairs, r):
    """`D` is a maximum over distributions with mean at most `eta`, and every one of them
    obeys `P(X >= t) <= eta / t`."""
    value = bounds.r_concave_tail(eta, t, n_pairs, r)
    assert 0.0 <= value <= min(1.0, eta / t) + 1e-12


@pytest.mark.parametrize("eta,t", GRID)
@pytest.mark.parametrize("n_pairs,r", [(15, -0.5), (40, -0.5), (40, -0.25)])
def test_r_concave_tail_dominates_members_of_the_family_it_maximises_over(eta, t, n_pairs, r):
    """Build mass functions that really are `r`-concave with mean at most `eta`, and check
    none of them has a larger tail than the bound claims is the largest.

    The family used is the paper's own parametrisation, `f_i` proportional to
    `(a + i) ** (1 / r)` on `0, ..., k`, whose `r`th power is linear and therefore convex, so
    every member is `r`-concave by Definition 4. Its scale and support are swept, and the
    ones whose mean is within the constraint are the feasible points the bound must dominate.
    """
    value = bounds.r_concave_tail(eta, t, n_pairs, r)
    cut = int(np.ceil(n_pairs * t - 1e-9))
    lattice = np.arange(n_pairs + 1) / n_pairs
    worst = 0.0
    for k in range(1, n_pairs + 1):
        index = np.arange(k + 1, dtype=float)
        for a in np.geomspace(1e-6, 1e6, 60):
            mass = (a + index) ** (1.0 / r)
            mass = mass / mass.sum()
            if float(lattice[:k + 1] @ mass) > eta + 1e-12:
                continue
            worst = max(worst, float(mass[cut:].sum()) if cut <= k else 0.0)
    assert value >= worst - 1e-9


@pytest.mark.parametrize("n_pairs,r", [(25, -0.5), (50, -0.25)])
def test_r_concave_tail_is_monotone_in_both_arguments(n_pairs, r):
    """More mean allowed can only allow a fatter tail; a tail taken further out can only be
    thinner."""
    for eta in (0.05, 0.2, 0.4):
        tails = [bounds.r_concave_tail(eta, t, n_pairs, r) for t in np.linspace(0.5, 1.0, 11)]
        assert np.all(np.diff(tails) <= 1e-9), tails
    for t in (0.6, 0.85):
        tails = [bounds.r_concave_tail(eta, t, n_pairs, r)
                 for eta in np.linspace(0.01, 0.5, 11)]
        assert np.all(np.diff(tails) >= -1e-9), tails


def test_r_concave_tail_conventions():
    assert bounds.r_concave_tail(0.3, 0.3, 50, -0.5) == 1.0      # t <= eta
    assert bounds.r_concave_tail(0.3, 0.2, 50, -0.5) == 1.0
    assert bounds.r_concave_tail(0.3, 1.5, 50, -0.5) == 0.0      # off the lattice
    with pytest.raises(ValueError):
        bounds.r_concave_tail(0.1, 0.6, 50, 0.5)


def test_the_three_bounds_are_ordered_by_what_they_assume():
    """Assuming more can only tighten: worst case, then unimodal, then r-concave."""
    theta, n_low, n_selected = 0.05, 1000, 50
    for tau in (0.6, 0.7, 0.8, 0.9, 1.0):
        worst = bounds.false_selection_bound(theta, tau, 50, n_low, n_selected, "worst-case")
        uni = bounds.false_selection_bound(theta, tau, 50, n_low, n_selected, "unimodal")
        tight = bounds.false_selection_bound(theta, tau, 50, n_low, n_selected)
        assert tight <= uni <= worst, (tau, tight, uni, worst)


def test_the_worst_case_bound_is_meinshausen_and_buhlmanns():
    """With `theta = q / p` and the two substitutions of Section 3.4, Theorem 1(i) collapses
    to `q ** 2 / (p (2 tau - 1))`, which is Meinshausen and Buhlmann's Theorem 1."""
    p, q, tau = 1000, 50, 0.6
    got = bounds.false_selection_bound(q / p, tau, 50, p, q, "worst-case")
    assert got == pytest.approx(q ** 2 / (p * (2 * tau - 1)))


def test_unimodal_bound_is_out_of_range_where_the_theorem_is():
    assert bounds.unimodal_multiplier(0.9, 0.8, 50) == np.inf    # theta above 3 ** -0.5
    assert bounds.unimodal_multiplier(0.05, 0.5, 50) == np.inf   # tau not above a half
    assert np.isfinite(bounds.unimodal_multiplier(0.05, 0.8, 50))


@pytest.mark.parametrize("shape", ["r-concave", "unimodal", "worst-case"])
def test_threshold_for_inclusion_is_the_least_one_meeting_the_budget(shape):
    # Loose enough that the worst-case bound, which is 2.5 even at a threshold of one, can
    # be met at all; the two tighter shapes meet it much sooner and the property is the same.
    theta, budget, n_pairs, p, q = 0.05, 3.0, 50, 1000, 50
    tau = bounds.threshold_for_inclusion(theta, budget, n_pairs, p, q, shape)
    assert bounds.false_selection_bound(theta, tau, n_pairs, p, q, shape) <= budget
    achievable = bounds.achievable_thresholds(n_pairs)
    below = achievable[(achievable > 0.5) & (achievable < tau - 1e-12)]
    for smaller in below:
        assert bounds.false_selection_bound(theta, smaller, n_pairs, p, q, shape) > budget


def test_threshold_for_inclusion_is_nan_where_nothing_can_be_certified():
    """A base procedure keeping four fifths of what it is offered certifies nothing: no
    threshold can push the expected number of false inclusions below one."""
    assert np.isnan(bounds.threshold_for_inclusion(0.8, 1.0, 50, 12, 9.6))


def test_missed_exclusion_bound_matches_theorem_one_part_two():
    assert bounds.missed_exclusion_bound(0.3, 0.25, 4.0) == pytest.approx(0.7 / 0.5 * 4.0)
    assert bounds.missed_exclusion_bound(0.3, 0.5, 4.0) == np.inf
    assert bounds.threshold_for_exclusion(0.3, 1.0, 50, 0.0) == pytest.approx(0.49)


def test_achievable_thresholds_are_the_values_pi_can_take():
    got = bounds.achievable_thresholds(3)
    np.testing.assert_allclose(got, [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1])


# ------------------------------------------------------------------------ the groups ------


@pytest.mark.parametrize("n_basis,order", [(15, 4), (25, 4), (12, 3)])
def test_a_group_is_exactly_the_basis_functions_alive_on_its_knot_interval(n_basis, order):
    """The whole method rests on this: group `b` of the overlap penalty holds coefficients
    `b, ..., b + order - 1`, and those are exactly the basis functions that are non-zero
    somewhere inside knot interval `b`. If they were not, switching a group off would not
    zero the function on an interval and the selection probabilities would not be about the
    domain at all.
    """
    grid = np.linspace(0, 1, 2001)
    Phi = bspline_basis(grid, n_basis=n_basis, order=order)
    edges = stability.knot_edges(n_basis, order)
    _, groups = SBC.pen_matrices(n_basis, order=order)
    assert edges.size - 1 == len(groups)

    for b, group in enumerate(groups):
        inside = (grid > edges[b] + 1e-9) & (grid < edges[b + 1] - 1e-9)
        alive = np.flatnonzero(np.abs(Phi[inside]).max(axis=0) > 1e-10)
        np.testing.assert_array_equal(alive, group)


def test_knot_interval_of_labels_every_grid_point_once():
    grid = np.linspace(0, 1, 101)
    label = stability.knot_interval_of(grid, 15, 4)
    edges = stability.knot_edges(15, 4)
    assert label.min() == 0 and label.max() == edges.size - 2
    assert np.all(np.diff(label) >= 0)
    for point, b in zip(grid, label):
        assert edges[b] - 1e-12 <= point <= edges[b + 1] + 1e-12


def test_knot_edges_rejects_a_basis_smaller_than_its_order():
    with pytest.raises(ValueError):
        stability.knot_edges(3, 4)


def test_group_activity_is_any_surviving_coefficient_in_the_group():
    psis = np.zeros((3, 7))
    psis[0, 2] = 1.0                                   # coefficient 2: groups 0, 1, 2
    psis[1] = 1.0
    got = stability.group_activity(psis, n_basis=7, order=3)
    assert got.shape == (3, 5)
    np.testing.assert_array_equal(got[0], [True, True, True, False, False])
    assert got[1].all() and not got[2].any()


def test_group_activity_rejects_a_mismatched_basis():
    with pytest.raises(ValueError):
        stability.group_activity(np.zeros((2, 7)), n_basis=8, order=3)


def test_entry_step_is_the_first_penalty_at_which_a_group_survives():
    active = np.array([[[False, False, False],
                        [True, False, False],
                        [True, True, False]]])
    np.testing.assert_array_equal(stability.entry_step(active), [[1, 2, 3]])


# ------------------------------------------------------------- complementary pairs ------


def test_complementary_pairs_are_disjoint_halves():
    subsets = stability.complementary_pairs(41, n_pairs=8, rng=0)
    assert len(subsets) == 16
    for first, second in zip(subsets[::2], subsets[1::2]):
        assert len(first) == len(second) == 41 // 2
        assert not set(first) & set(second)
        assert set(first) | set(second) <= set(range(41))


def test_complementary_pairs_never_split_a_block():
    """A block is whole in one half of a pair or absent from it, which is what keeps the
    dependence inside a block out of the comparison between the halves."""
    subsets = stability.complementary_pairs(40, n_pairs=5, block=4, rng=1)
    for take in subsets:
        for start in range(0, 40, 4):
            inside = set(range(start, start + 4)) & set(take)
            assert inside in ({}, set(), set(range(start, start + 4)))


def test_complementary_pairs_honour_explicit_labels():
    """A panel contributes one row per instrument on each date, and a date must not be split
    between the halves of a pair."""
    labels = np.repeat(np.arange(10), 3)               # ten dates, three instruments each
    subsets = stability.complementary_pairs(30, n_pairs=4, block=labels, rng=2)
    for take in subsets:
        for date in range(10):
            rows = set(np.flatnonzero(labels == date))
            assert rows <= set(take) or not (rows & set(take))


@pytest.mark.parametrize("bad", [dict(n=3), dict(block=0), dict(block=np.zeros(20))])
def test_complementary_pairs_validates_its_inputs(bad):
    kwargs = dict(n=20, n_pairs=2)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        stability.complementary_pairs(**kwargs)


# ------------------------------------------------------------------ selection curves ------


@pytest.fixture(scope="module")
def curves(sbc_data):
    """Selection curves on the shared locally sparse dataset, small enough to be quick."""
    return stability.selection_curves(sbc_data["X"], sbc_data["y"], sbc_data["grid"],
                                      n_basis=15, order=4, n_pairs=8, n_lambda=15, rng=0)


def test_selection_curves_are_proportions_on_the_right_lattice(curves):
    pi = curves["pi"]
    assert pi.shape == (curves["n_groups"], curves["n_groups"])
    assert np.all((pi >= 0) & (pi <= 1))
    np.testing.assert_allclose(pi * 2 * curves["n_pairs"],
                               np.round(pi * 2 * curves["n_pairs"]), atol=1e-9)
    np.testing.assert_allclose(curves["q"], pi.sum(axis=1))


def test_selection_proportions_never_fall_as_the_base_procedure_keeps_more(curves):
    """Monotonicity in the size is what makes an inclusion certified at one size and an
    exclusion certified at another comparable at all."""
    assert np.all(np.diff(curves["pi"], axis=0) >= -1e-12)


def test_selection_curves_fill_the_lattice_at_the_largest_size(curves):
    """Allowed every group, the base procedure keeps every group."""
    np.testing.assert_allclose(curves["pi"][-1], 1.0)


def test_resolve_thresholds_straddle_a_half(curves):
    profile = stability.resolve(curves, pfer=1.0, pner=1.0)
    finite_hi = profile["tau_hi"][np.isfinite(profile["tau_hi"])]
    finite_lo = profile["tau_lo"][np.isfinite(profile["tau_lo"])]
    assert np.all(finite_hi > 0.5)
    assert np.all(finite_lo < 0.5)
    assert np.all(profile["bound_false"][np.isfinite(profile["tau_hi"])] <= 1.0 + 1e-12)
    assert np.all(profile["bound_missed"][np.isfinite(profile["tau_lo"])] <= 1.0 + 1e-12)


def test_resolve_decides_nothing_it_cannot_certify(curves):
    """Where a threshold does not exist the corresponding decision must be empty, rather
    than falling back on a threshold the bound does not support."""
    profile = stability.resolve(curves, pfer=1.0, pner=1.0)
    assert not profile["informative"][~np.isfinite(profile["tau_hi"])].any()
    assert not profile["null"][~np.isfinite(profile["tau_lo"])].any()


def test_a_tighter_budget_never_decides_more(curves):
    loose = stability.resolve(curves, pfer=2.0, pner=2.0)
    tight = stability.resolve(curves, pfer=0.2, pner=0.2)
    assert np.all(tight["n_informative"] <= loose["n_informative"])
    assert np.all(tight["n_null"] <= loose["n_null"])


def test_select_partitions_the_domain(curves):
    chosen = stability.select(curves, pfer=1.0, pner=1.0)
    parts = chosen["informative"] + chosen["null"] + chosen["undecided"]
    np.testing.assert_array_equal(parts, np.ones(curves["n_groups"], dtype=int))
    assert not (chosen["informative"] & chosen["null"]).any()
    on_grid = (chosen["informative_grid"].astype(int) + chosen["null_grid"]
               + chosen["undecided_grid"])
    np.testing.assert_array_equal(on_grid, np.ones(curves["grid"].size, dtype=int))


def test_regions_merge_adjacent_groups_and_end_at_knots(curves):
    mask = np.zeros(curves["n_groups"], dtype=bool)
    mask[[1, 2, 3, 7]] = True
    got = stability.regions(mask, curves)
    edges = curves["edges"]
    assert got == [(edges[1], edges[4]), (edges[7], edges[8])]
    assert stability.regions(np.zeros(curves["n_groups"], bool), curves) == []
    with pytest.raises(ValueError):
        stability.regions(np.zeros(3, bool), curves)


@pytest.mark.slow
def test_the_certified_null_set_lands_inside_the_true_one():
    """`psi2` is zero outside [0.25, 0.75] and large in the middle, which is the case a
    support estimator should find. What is certified null must be genuinely null."""
    X, y, grid, _ = SBC_sim.simulate(SBC_sim.psi2, n=100, m=100, snr=4.0, rng=1)
    curves = stability.selection_curves(X, y, grid, n_basis=15, order=4, n_pairs=25,
                                        n_lambda=30, rng=0)
    chosen = stability.select(curves, pfer=1.0, pner=1.0)
    true_null = (grid < 0.25) | (grid > 0.75)
    assert chosen["null_grid"].any(), "nothing was certified null at all"
    assert not (chosen["null_grid"] & ~true_null).any()


# ------------------------------------------------------------------------ criteria ------


def test_measure_is_the_length_of_the_set():
    grid = np.linspace(0, 1, 401)
    assert criteria.measure(np.ones(grid.size, bool), grid) == pytest.approx(1.0)
    assert criteria.measure(grid <= 0.25, grid) == pytest.approx(0.25, abs=2e-3)
    half = grid > 0.5
    assert (criteria.measure(half, grid) + criteria.measure(~half, grid)
            == pytest.approx(1.0))


def test_support_error_splits_the_symmetric_difference():
    grid = np.linspace(0, 1, 201)
    truth, guess = grid < 0.4, grid < 0.6
    got = criteria.support_error(guess, truth, grid)
    assert got["false_null"] + got["missed_null"] == pytest.approx(got["symmetric"])
    assert got["false_null"] == pytest.approx(0.2, abs=1e-2)
    assert got["missed_null"] == pytest.approx(0.0, abs=1e-9)
    assert criteria.support_error(truth, truth, grid)["jaccard"] == pytest.approx(1.0)
    empty = np.zeros(grid.size, bool)
    assert np.isnan(criteria.support_error(empty, empty, grid)["jaccard"])


def test_adaptive_weights_charge_least_where_the_pilot_is_largest(sbc_data):
    weights = criteria.adaptive_weights(sbc_data["X"], sbc_data["y"], sbc_data["grid"])
    assert weights.mean() == pytest.approx(1.0)
    assert np.all(weights > 0)

    grid, n_basis, order = sbc_data["grid"], 15, 4
    W, _ = SBC.bspline_design(sbc_data["X"], grid, n_basis=n_basis, order=order)
    pilot = np.linalg.lstsq(W, sbc_data["y"], rcond=None)[0]
    _, groups = SBC.pen_matrices(n_basis, order=order)
    norms = np.array([np.linalg.norm(pilot[g]) for g in groups])
    assert np.corrcoef(norms, weights)[0, 1] < -0.5

    flat = criteria.adaptive_weights(sbc_data["X"], sbc_data["y"], grid, gamma=0.0)
    np.testing.assert_allclose(flat, 1.0)


def test_ebic_is_the_criterion_it_says_it_is(sbc_data):
    from scipy import special
    path = SBC.fit_path(sbc_data["X"], sbc_data["y"], sbc_data["grid"], n_lambda=12)
    got = criteria.ebic(path, sbc_data["y"], gamma=0.5)
    n, n_basis = sbc_data["y"].size, path["n_basis"]
    k = np.count_nonzero(path["psis"], axis=1)
    rss = np.sum((sbc_data["y"][None, :] - path["psis"] @ path["W"].T) ** 2, axis=1)
    expected = (n * np.log(rss / n) + k * np.log(n)
                + 2 * 0.5 * (special.gammaln(n_basis + 1) - special.gammaln(k + 1)
                             - special.gammaln(n_basis - k + 1)))
    np.testing.assert_allclose(got["values"], expected)
    np.testing.assert_array_equal(got["k"], k)
    assert got["step"] == int(np.argmin(expected))


def test_ebic_at_zero_gamma_is_the_ordinary_bic(sbc_data):
    path = SBC.fit_path(sbc_data["X"], sbc_data["y"], sbc_data["grid"], n_lambda=12)
    plain = criteria.ebic(path, sbc_data["y"], gamma=0.0)
    n = sbc_data["y"].size
    expected = n * np.log(plain["rss"] / n) + plain["k"] * np.log(n)
    np.testing.assert_allclose(plain["values"], expected)


def test_ebics_extra_term_vanishes_at_both_ends_of_the_basis(sbc_data):
    """The term prices the number of models of a given size, and there is one model of size
    zero and one of size `n_basis`. It is therefore not monotone, and raising `gamma` can
    move the choice to the full model rather than away from it -- the behaviour documented
    on `ebic`, and the reason it is measured here rather than assumed away.
    """
    path = SBC.fit_path(sbc_data["X"], sbc_data["y"], sbc_data["grid"], n_lambda=12)
    plain = criteria.ebic(path, sbc_data["y"], gamma=0.0)
    extended = criteria.ebic(path, sbc_data["y"], gamma=1.0)
    term = extended["values"] - plain["values"]
    assert np.all(term >= -1e-9)
    full = extended["k"] == path["n_basis"]
    if full.any():
        np.testing.assert_allclose(term[full], 0.0, atol=1e-9)
    assert term.max() > 0


def test_relaxed_fit_is_zero_on_the_null_groups_and_least_squares_elsewhere(sbc_data):
    grid, n_basis, order = sbc_data["grid"], 15, 4
    null = np.zeros(n_basis - order + 1, dtype=bool)
    null[:3] = True
    got = criteria.relaxed_fit(sbc_data["X"], sbc_data["y"], grid, null,
                               n_basis=n_basis, order=order)
    np.testing.assert_array_equal(got["psi"][:order + 2], 0.0)
    edges = stability.knot_edges(n_basis, order)
    np.testing.assert_allclose(got["beta"][grid < edges[3]], 0.0, atol=1e-12)

    none = criteria.relaxed_fit(sbc_data["X"], sbc_data["y"], grid,
                                np.zeros(n_basis - order + 1, dtype=bool))
    W, _ = SBC.bspline_design(sbc_data["X"], grid, n_basis=n_basis, order=order)
    expected, *_ = np.linalg.lstsq(W, sbc_data["y"], rcond=None)
    np.testing.assert_allclose(none["psi"], expected, atol=1e-8)


def test_relaxed_fit_validates_the_group_mask(sbc_data):
    with pytest.raises(ValueError):
        criteria.relaxed_fit(sbc_data["X"], sbc_data["y"], sbc_data["grid"],
                             np.zeros(5, dtype=bool))


# --------------------------------------------------------------------------- study ------


def test_psi_late_is_a_right_anchored_window():
    grid = np.linspace(0, 1, 601)
    values = study.psi_late(grid)
    assert np.all(values[grid < 0.7] == 0.0)
    assert values[grid > 0.75].max() > 1.5
    assert abs(values[np.searchsorted(grid, 0.7)]) < 1e-2      # meets zero smoothly
    for name, (fn, null_fn) in study.DESIGNS.items():
        mask = null_fn(grid)
        assert np.allclose(fn(grid)[mask], 0.0, atol=1e-12), name


@pytest.mark.slow
def test_run_once_scores_every_rule_on_one_dataset():
    rows = study.run_once("psi2", n=60, m=60, seed=0, n_pairs=4, n_lambda=10, n_folds=3)
    assert [row["rule"] for row in rows] == list(study.RULES)
    for row in rows:
        assert row["design"] == "psi2"
        assert row["false_null"] + row["missed_null"] == pytest.approx(row["symmetric"])
        assert row["ise"] >= 0 and row["ise_relaxed"] >= 0
    stable = [row for row in rows if row["rule"].startswith("stability")]
    assert all(row["undecided"] >= 0 for row in stable)


@pytest.mark.slow
def test_summarise_gives_one_row_per_design_and_rule():
    frame = study.run_study(designs=["psi2"], n_reps=2, n=60, m=60, n_pairs=4, n_lambda=10,
                            n_folds=3)
    table = study.summarise(frame)
    assert list(table.index.names) == ["design", "rule"]
    assert len(table) == len(study.RULES)
    assert (table["n_reps"] == 2).all()
    assert set(["false_null", "missed_null", "ise", "false_null_rate"]) <= set(table.columns)
