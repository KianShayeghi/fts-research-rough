"""Tests for `truncFLR.HH_multi`, multiple-breakpoint estimation by binary segmentation.

The pieces worth pinning down: `joint_refit` partitions the domain without gap or overlap (the
half-open leaf convention it borrows from `HH.py`'s own `grid <= theta`), `select_joint_refit`'s
BIC step actually does what it exists for -- a fixed component count badly overfits a leaf with
no real signal in it, which is exactly what a first version of this module did before the BIC
step was added -- `segment_contest` finds a clean single break and correctly declines to invent
one on pure noise, and `binary_segment` reproduces both end to end. The last test pins a known
limitation inherited from method B: an informative region with null on both sides of it is
missed, because method B's theta scan takes the *first* interior minimum of its objective and a
null stretch at the window's anchored end produces a spurious early one (see the last test).
"""

import numpy as np
import pytest

from fts_research.core.gp import sample_gp
from fts_research.truncFLR import HH_dgp, HH_multi

LAMBDAS = np.linspace(0.0, 1.0, 60)[1:]


def _null_data(n=200, n_grid=81, rng=2):
    """Curves with a real covariance structure and a response with none of it in it -- the
    honest way to build a "no informative region anywhere" design. `HH_dgp.sim_dataset` derives
    its noise scale from the *variance of the signal*, so a beta_fn of all zeros collapses that
    variance to zero and makes Y identically zero -- trivially reconstructed by any fit, and not
    actually a test of anything.
    """
    rng = np.random.default_rng(rng)
    grid = np.linspace(0.0, 1.0, n_grid)
    X = sample_gp(n, grid, rng=rng)
    Y = rng.normal(0.0, 1.0, size=n)
    return grid, X, Y


# ------------------------------------------------------------------------- joint_refit -----

def test_joint_refit_partitions_the_domain_without_gap_or_overlap(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    breakpoints = [grid[20], grid[55]]

    _, _, leaf_masks, _ = HH_multi.joint_refit(X, Y, grid, breakpoints, n_components=3)

    assert len(leaf_masks) == 3
    stacked = np.vstack(leaf_masks)
    np.testing.assert_array_equal(stacked.sum(axis=0), np.ones(grid.size, dtype=int))
    for mask in leaf_masks:
        assert mask.any()


def test_select_joint_refit_avoids_overfitting_a_null_leaf():
    """A fixed, generous component count fits a leaf with no real signal in it almost exactly
    in sample -- the failure this module's BIC step exists to prevent. Overfitting shows up
    where the leaf is short relative to the component count (few grid points, few effective
    degrees of freedom to spare): a single break near 0.3 on a 101-point grid leaves the null
    leaf about 29 points wide, short enough that ten unpenalised components chase the noise
    there and the coefficient swings into the tens. BIC selection should keep it near zero
    instead. (Hall & Hooker's own [0.8, 1] design does not reproduce this -- its null leaf is
    long enough on a 301-point grid that ten components are not actually that many relative to
    it, which is itself the point: overfitting here is about the leaf's own size, not whether it
    is null.)
    """
    def beta_fn(t):
        return (t >= 0.3).astype(float)

    data = HH_dgp.sim_dataset(n=300, n_grid=101, beta_fn=beta_fn, snr=40.0, rng=1)
    grid, X, Y = data["grid"], data["X"], data["Y"]
    breakpoint_ = float(grid[np.argmin(np.abs(grid - 0.3))])

    _, b_fixed, leaf_masks, _ = HH_multi.joint_refit(X, Y, grid, [breakpoint_], n_components=10)
    null_leaf = leaf_masks[0]  # [0, breakpoint], the true null side
    assert np.abs(b_fixed[null_leaf]).max() > 5, \
        "expected the fixed, unselected component count to overfit the null leaf"

    _, b_hat, leaf_masks_sel, _, m_hat, _ = HH_multi.select_joint_refit(
        X, Y, grid, [breakpoint_], n_components=10
    )
    null_leaf_sel = leaf_masks_sel[0]
    assert np.abs(b_hat[null_leaf_sel]).max() < 1.0


# --------------------------------------------------------------------- segment_contest -----

@pytest.mark.slow
def test_mirrored_split_on_a_subsegment_reports_a_breakpoint_inside_it():
    """A mirrored fit on a segment [lo, hi] reflects about that segment's midpoint, not 0.5, so
    the breakpoint is lo + hi - theta_hat. `HH.fit_mirror`'s own u_hat = 1 - theta_hat is right
    only on the whole domain: on [0.3, 1] with a true break at 0.6 it reads 0.3, the segment's
    edge, while the fitted b_hat is correctly zero below 0.6.
    """
    def beta_fn(t):
        return (t >= 0.6).astype(float)

    data = HH_dgp.sim_dataset(n=300, n_grid=101, beta_fn=beta_fn, snr=40.0, rng=1)
    grid, X, Y = data["grid"], data["X"], data["Y"]
    seg = grid >= 0.3 - 1e-12

    _, breakpoint_ = HH_multi._predict_split(X[:, seg], Y, X[:5, seg], grid[seg], LAMBDAS,
                                             True, 20, 10, 1)
    assert grid[seg][0] < breakpoint_ < grid[seg][-1]
    assert abs(breakpoint_ - 0.6) < 0.1


@pytest.mark.slow
def test_segment_contest_finds_a_clean_single_break():
    def beta_fn(t):
        return (t >= 0.3).astype(float)

    data = HH_dgp.sim_dataset(n=300, n_grid=101, beta_fn=beta_fn, snr=40.0, rng=1)
    grid, X, Y = data["grid"], data["X"], data["Y"]

    breakpoint_ = HH_multi.segment_contest(X, Y, grid, LAMBDAS, n_folds=4, rng=0)
    assert breakpoint_ is not None
    assert abs(breakpoint_ - 0.3) < 0.1


@pytest.mark.slow
def test_segment_contest_declines_to_split_pure_noise():
    grid, X, Y = _null_data()
    assert HH_multi.segment_contest(X, Y, grid, LAMBDAS, n_folds=4, rng=0) is None


# ---------------------------------------------------------------------- binary_segment -----

@pytest.mark.slow
def test_binary_segment_recovers_a_single_clean_break_and_stops():
    def beta_fn(t):
        return (t >= 0.3).astype(float)

    data = HH_dgp.sim_dataset(n=300, n_grid=101, beta_fn=beta_fn, snr=40.0, rng=1)
    grid, X, Y = data["grid"], data["X"], data["Y"]

    res = HH_multi.binary_segment(X, Y, grid, LAMBDAS, n_folds=4, max_depth=3,
                                  min_points=10, rng=0)

    assert len(res["breakpoints"]) == 1
    assert abs(res["breakpoints"][0] - 0.3) < 0.1
    assert len(res["leaf_masks"]) == 2
    # Both children must have been offered the contest and declined it -- no spurious further
    # break invented once the true structure has been found.
    child_contests = [c for c in res["contests"] if c["depth"] == 1]
    assert len(child_contests) == 2
    assert all(c["breakpoint"] is None for c in child_contests)

    err = np.trapezoid((res["b_hat"] - data["b"]) ** 2, grid)
    assert err < 0.1


@pytest.mark.slow
def test_binary_segment_does_not_invent_a_break_on_pure_noise():
    grid, X, Y = _null_data()
    res = HH_multi.binary_segment(X, Y, grid, LAMBDAS, n_folds=4, max_depth=3,
                                  min_points=10, rng=0)
    assert res["breakpoints"] == []
    assert np.abs(res["b_hat"]).max() < 1.0


@pytest.mark.slow
def test_binary_segment_does_not_split_an_interior_region_flanked_by_null_on_both_sides():
    """A known limitation inherited from method B, pinned so a change to it is deliberate.

    The window [0, 0.5] contains this bump exactly -- inside the kept window b is unrestricted --
    so the model family is not the problem. Method B's theta scan (`HH.theta_methodB`, the
    paper's step 6') takes the *first* interior minimum of T(theta) = RSS + n lambda theta^2, and
    the null stretch [0, 0.2) at the anchored end leaves T nearly flat there, so the pilot's
    ringing gives it a shallow dip at 0.06 ahead of the global minimum at 0.50. Both truncated
    candidates therefore land on nonsense windows and lose to the no-split pilot. Swapping the
    per-segment detector for method A, which takes the global minimum, is the obvious fix to try.
    """
    def beta_fn(t):
        return ((t >= 0.2) & (t <= 0.5)).astype(float)

    data = HH_dgp.sim_dataset(n=300, n_grid=101, beta_fn=beta_fn, snr=200.0, rng=1)
    grid, X, Y = data["grid"], data["X"], data["Y"]

    res = HH_multi.binary_segment(X, Y, grid, LAMBDAS, n_folds=4, max_depth=3,
                                  min_points=10, rng=0)
    assert res["breakpoints"] == []
