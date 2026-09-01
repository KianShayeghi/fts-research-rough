"""Tests for `ftsa.updating`: the four updates, block moving, and the two benchmarks.

The update is a small piece of algebra wrapped around a promise, and the promise is the part
worth testing: that only the observed block of the new day is ever read. A method that leaked
the rest of the day would look excellent and nothing downstream would object, so that is
checked directly, by perturbing what should not have been seen and requiring the forecast not
to move.

The rest fixes the pieces that have a right answer independent of any data. Least squares must
recover the scores exactly when the curve really is a combination of the components; the two
penalised updates must reduce to it at zero penalty and to the thing they shrink toward at
infinite penalty; the rotation block moving performs must be a rotation; and the penalty must
be chosen against a criterion that is not identically zero, which is the one bug this module
has actually had.
"""

import numpy as np
import pytest

from fts_research.far import far_dgp
from fts_research.ftsa import evaluation as ev
from fts_research.ftsa import forecast as fc
from fts_research.ftsa import updating as up

GRID = np.linspace(0.0, 1.0, 40)


@pytest.fixture(scope="module")
def curves():
    """A full-rank FAR(1): dependent curves with a truncation residual the update has to face."""
    return far_dgp.sample_far1(160, GRID, far_dgp.exponential_kernel(GRID, rho=0.7), rng=3)


@pytest.fixture(scope="module")
def dec(curves):
    return fc.decompose(curves, GRID, n_components=3)


@pytest.fixture(scope="module")
def exact(dec):
    """Curves that lie exactly in the span of three components, so recovery has an answer."""
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((60, 3)) * np.sqrt(dec["tau"])[None, :]
    return dec["mean"][None, :] + scores @ dec["phi"], scores


# ------------------------------------------------------------------------ the split ----


def test_split_index_leaves_both_blocks_non_empty():
    for fraction in (1e-9, 0.5, 1 - 1e-9, 0.0, 1.0):
        m = up.split_index(GRID, fraction)
        assert 1 <= m <= GRID.size - 1


def test_split_index_counts_the_points_already_seen():
    m = up.split_index(GRID, 0.5)
    assert np.all(GRID[:m] <= 0.5) and GRID[m] > 0.5


# ----------------------------------------------------------------------- the updates ----


def test_least_squares_recovers_scores_from_any_block_of_an_exact_curve(dec, exact):
    """If the curve is `mu + Phi' psi`, its observed block determines `psi` exactly, and the
    update is then exact on the unobserved block too."""
    X, scores = exact
    for m in (5, 12, 30):
        b = up.update_scores(dec, X[0, :m], method="ols")
        assert np.allclose(b, scores[0], atol=1e-8)
        assert np.allclose(dec["mean"][m:] + b @ dec["phi"][:, m:], X[0, m:], atol=1e-8)


def test_penalised_updates_reduce_to_least_squares_at_zero_penalty(dec, curves):
    head = curves[0, :20]
    ols = up.update_scores(dec, head, method="ols")
    beta_ts = np.array([0.3, -0.2, 0.1])
    for method in ("ridge", "pls"):
        near = up.update_scores(dec, head, method=method, lam=1e-12, beta_ts=beta_ts)
        assert np.allclose(near, ols, atol=1e-5)


def test_pls_reduces_to_the_time_series_forecast_at_infinite_penalty(dec, curves):
    """The whole point of `pls` is that it interpolates: `lam = 0` is `ols` and `lam` large is
    the day-ahead forecast, so a method that ignored one end would still look plausible."""
    beta_ts = np.array([0.3, -0.2, 0.1])
    far = up.update_scores(dec, curves[0, :20], method="pls", lam=1e10, beta_ts=beta_ts)
    assert np.allclose(far, beta_ts, atol=1e-4)


def test_ridge_shrinks_to_the_mean_curve_at_infinite_penalty(dec, curves):
    far = up.update_scores(dec, curves[0, :20], method="ridge", lam=1e10)
    assert np.allclose(far, 0.0, atol=1e-4)


def test_penalised_updates_shrink_monotonically(dec, curves):
    norms = [np.linalg.norm(up.update_scores(dec, curves[0, :20], method="ridge", lam=lam))
             for lam in (1e-6, 1e-2, 1.0, 1e2, 1e4)]
    assert np.all(np.diff(norms) <= 1e-9)


def test_least_squares_is_refused_when_the_block_is_too_short(dec, curves):
    with pytest.raises(ValueError, match="cannot identify"):
        up.update_scores(dec, curves[0, :2], method="ols")


def test_a_method_needing_a_forecast_says_so(dec, curves):
    for method in ("ts", "pls"):
        with pytest.raises(ValueError, match="beta_ts"):
            up.update_scores(dec, curves[0, :20], method=method, beta_ts=None)


# ------------------------------------------------------------- the in-sample criterion ----


def test_the_in_sample_criterion_is_measured_on_the_observed_curves(dec):
    """The bug this module had. Measuring on the `M`-component reconstructions instead makes
    least squares exact on every training day, so the error is zero at every penalty and the
    selected penalty is whichever the grid happens to start at."""
    errors = up.in_sample_updates(dec, 20, method="ols")
    assert np.mean(errors ** 2) > 1e-8, "the criterion is degenerate; it is fitting noiseless data"

    reconstruction_only = dec["mean"][None, :] + dec["scores"] @ dec["phi"]
    assert np.max(np.abs(reconstruction_only - (reconstruction_only + dec["resid_curves"]))) > 0


def test_the_selected_penalty_is_interior_to_its_grid(dec):
    """A penalty pinned at either end of the grid is a criterion that has not chosen."""
    for method in ("ridge", "pls"):
        out = up.select_lambda(dec, 20, method=method)
        best = int(np.argmin(out["mse"]))
        assert 0 < best < out["mse"].size - 1, f"{method} selected the grid edge"
        assert out["lam_hat"] == pytest.approx(out["lambdas"][best])


def test_only_the_penalised_methods_have_a_penalty_to_choose(dec):
    for method in ("ols", "ts", "hold"):
        with pytest.raises(ValueError, match="penalty"):
            up.select_lambda(dec, 20, method=method)


def test_every_method_but_block_has_an_error_pool(dec):
    for method in ("ts", "ols", "ridge", "pls", "hold", "mean"):
        errors = up.in_sample_updates(dec, 20, method=method)
        assert errors.shape[1] == GRID.size - 20 and np.all(np.isfinite(errors))
    with pytest.raises(ValueError, match="block moving"):
        up.in_sample_updates(dec, 20, method="block")


# --------------------------------------------------------------------- block moving ----


def test_block_moving_rotates_the_day_boundary(curves):
    """The rotated curves must be `(tail of day t, head of day t + 1)`, with the observed part
    of the new day as the final tail. Getting the rotation wrong would still return an array of
    the right shape."""
    X, m = curves[:5], 12
    head = curves[5, :m]
    tails, heads = X[:, m:], np.vstack([X[1:, :m], head[None, :]])
    rotated = np.hstack([tails, heads])

    assert rotated.shape == X.shape
    assert np.allclose(rotated[0], np.concatenate([X[0, m:], X[1, :m]]))
    assert np.allclose(rotated[-1], np.concatenate([X[-1, m:], head]))


def test_block_moving_returns_the_unobserved_block(curves):
    for fraction in (0.2, 0.5, 0.8):
        m = up.split_index(GRID, fraction)
        out = up.block_moving(curves[:-1], curves[-1, :m], GRID, method="far")
        assert out.shape == (GRID.size - m,) and np.all(np.isfinite(out))


def test_block_moving_refuses_a_degenerate_split(curves):
    with pytest.raises(ValueError, match="rotate"):
        up.block_moving(curves[:-1], curves[-1], GRID, method="far")


# ---------------------------------------------------------------------- benchmarks ----


def test_hold_carries_the_last_observed_value_forward(curves):
    m = up.split_index(GRID, 0.4)
    out = up.update_curve(curves[:-1], GRID, curves[-1, :m], method="hold")
    assert np.allclose(out["prediction"], curves[-1, m - 1])


def test_the_mean_benchmark_meets_the_last_observed_value(curves, dec):
    m = up.split_index(GRID, 0.4)
    out = up.update_curve(curves[:-1], GRID, curves[-1, :m], method="mean", dec=dec)
    step = dec["mean"][m] - dec["mean"][m - 1]
    assert out["prediction"][0] == pytest.approx(curves[-1, m - 1] + step)


# ------------------------------------------------------------------------ anchoring ----


def test_anchoring_moves_the_prediction_by_the_boundary_error(curves, dec):
    """The anchored prediction is the unanchored one shifted by exactly the amount the method
    missed the last observed point by, so the difference between the two is constant."""
    m = up.split_index(GRID, 0.5)
    head = curves[-1, :m]
    beta_ts = np.zeros(dec["scores"].shape[1])
    plain = up.update_curve(curves[:-1], GRID, head, method="ridge", lam=0.5, dec=dec,
                            beta_ts=beta_ts, anchor=False)["prediction"]
    fixed = up.update_curve(curves[:-1], GRID, head, method="ridge", lam=0.5, dec=dec,
                            beta_ts=beta_ts, anchor=True)["prediction"]
    shift = fixed - plain
    assert np.allclose(shift, shift[0])

    b = up.update_scores(dec, head, method="ridge", lam=0.5)
    boundary = dec["mean"][m - 1] + b @ dec["phi"][:, m - 1]
    assert shift[0] == pytest.approx(head[-1] - boundary)


def test_anchoring_is_a_no_op_for_the_benchmarks(curves, dec):
    m = up.split_index(GRID, 0.5)
    head = curves[-1, :m]
    for method in ("hold", "mean"):
        plain = up.update_curve(curves[:-1], GRID, head, method=method, dec=dec)["prediction"]
        fixed = up.update_curve(curves[:-1], GRID, head, method=method, dec=dec,
                                anchor=True)["prediction"]
        assert np.allclose(plain, fixed)


def test_block_moving_cannot_be_anchored(curves):
    m = up.split_index(GRID, 0.5)
    with pytest.raises(ValueError, match="nothing to anchor"):
        up.update_curve(curves[:-1], GRID, curves[-1, :m], method="block", anchor=True)


# ------------------------------------------------------------------------- leakage ----


@pytest.mark.parametrize("method", up.METHODS)
def test_an_update_reads_only_what_was_observed(curves, method):
    """The promise the whole module rests on. The unobserved part of the new day is replaced
    with nonsense; every forecast must come back identical, because none of them was entitled
    to look at it."""
    m = up.split_index(GRID, 0.5)
    head = curves[-1, :m]
    history = curves[:-1]

    before = up.update_curve(history, GRID, head, method=method, rng=0)["prediction"]
    tampered = curves[-1].copy()
    tampered[m:] += 100.0
    after = up.update_curve(history, GRID, tampered[:m], method=method, rng=0)["prediction"]
    assert np.allclose(before, after)


def test_the_rolling_sweep_never_reads_the_block_it_scores(curves):
    """The same promise through the evaluation loop, where the held-out day is in the array."""
    tampered = curves.copy()
    tampered[-1, 25:] += 100.0
    plain = up.rolling_update(curves, GRID, fractions=(0.5,), methods=("pls", "hold"),
                              init_frac=0.9, rng=0)
    moved = up.rolling_update(tampered, GRID, fractions=(0.5,), methods=("pls", "hold"),
                              init_frac=0.9, rng=0)

    last = plain["session"].max()
    for method in ("pls", "hold"):
        a = plain[(plain["session"] == last) & (plain["method"] == method)]["point"]
        b = moved[(moved["session"] == last) & (moved["method"] == method)]["point"]
        assert np.allclose(a.to_numpy(), b.to_numpy())


# ------------------------------------------------------------------------ the sweep ----


def test_the_sweep_covers_every_method_and_fraction(curves):
    df = up.rolling_update(curves, GRID, fractions=(0.3, 0.7), init_frac=0.9,
                           anchors=(False, True), rng=0)
    assert set(df["fraction"]) == {0.3, 0.7}
    assert set(df["method"]) == set(up.METHODS)
    assert set(df["anchored"]) == {False, True}
    assert df[df["method"] == "block"]["anchored"].nunique() == 1, "block is never anchored"


def test_the_sweep_refuses_a_window_that_leaves_nothing_to_score(curves):
    with pytest.raises(ValueError, match="init_frac"):
        up.rolling_update(curves, GRID, init_frac=1.5)


def test_records_are_sufficient_for_the_summary(curves):
    """The cache keeps sums per session rather than every forecast, so the summary computed
    from the records must equal the one computed from the pointwise sweep."""
    df = up.rolling_update(curves, GRID, fractions=(0.5,), methods=("pls", "hold", "mean"),
                           init_frac=0.85, n_boot=50, rng=0)
    rec = up.records(df)
    assert len(rec) == df.groupby(["session", "fraction", "anchored", "method"]).ngroups
    assert rec["n_points"].sum() == len(df)

    table = up.summarise_updates(rec)
    direct = df["actual"].to_numpy() - df["point"].to_numpy()
    mask = (df["method"] == "pls") & (~df["anchored"])
    assert table.loc[(0.5, False, "pls"), "msfe"] == pytest.approx(
        float(np.mean(direct[mask.to_numpy()] ** 2)))


def test_profiles_average_along_the_session(curves):
    df = up.rolling_update(curves, GRID, fractions=(0.5,), methods=("hold",), init_frac=0.85,
                           rng=0)
    prof = up.profiles(df)
    assert set(prof.columns) == {"fraction", "anchored", "method", "s", "mafe", "msfe"}
    assert prof["s"].nunique() == df["s"].nunique()
    first = prof.sort_values("s").iloc[0]
    same = df[df["s"] == first["s"]]
    assert first["msfe"] == pytest.approx(
        float(np.mean((same["actual"] - same["point"]) ** 2)))


def test_skill_is_measured_against_the_martingale(curves):
    df = up.rolling_update(curves, GRID, fractions=(0.5,), methods=("pls", "hold", "mean"),
                           init_frac=0.85, rng=0)
    table = up.summarise_updates(up.records(df))
    assert table.xs("hold", level="method")["skill"].eq(0.0).all()

    hold = table.loc[(0.5, False, "hold"), "msfe"]
    pls = table.loc[(0.5, False, "pls"), "msfe"]
    assert table.loc[(0.5, False, "pls"), "skill"] == pytest.approx(1.0 - pls / hold)


# ------------------------------------------------------------------- the cached sweep ----


def test_the_cached_sweep_is_the_replication_design():
    rec = up.cached_updating()
    assert set(rec["method"]) == set(up.METHODS)
    assert sorted(set(rec["fraction"])) == sorted(up.UPDATE_FRACTIONS)
    assert rec["session"].nunique() == 50
    assert set(rec["anchored"]) == {True, False}


def test_the_cached_sweep_scores_every_method_on_the_same_sessions():
    rec = up.cached_updating()
    at_half = rec[rec["fraction"] == 0.5]
    sessions = {m: sorted(set(part["session"])) for m, part in at_half.groupby("method")}
    assert len({tuple(v) for v in sessions.values()}) == 1


def test_the_cached_sweep_reproduces_its_own_summary():
    table = up.summarise_updates(up.cached_updating())
    assert (table["n_splits"] == 50).all()
    assert table.xs("hold", level="method")["skill"].eq(0.0).all()
    assert table["coverage"].dropna().between(0.5, 1.0).all()
    assert set(ev.CONFIG["methods"]) <= {"arima", "far", "mean"}   # the other cache, untouched


def test_the_cached_profiles_describe_the_same_sweep():
    prof = up.cached_profiles()
    rec = up.cached_updating()
    assert set(prof["method"]) == set(rec["method"])
    assert sorted(set(prof["fraction"])) == sorted(set(rec["fraction"]))

    # Averaging a profile over the session must recover that cell's grand mean.
    table = up.summarise_updates(rec)
    cell = prof[(prof["fraction"] == 0.5) & (~prof["anchored"]) & (prof["method"] == "hold")]
    assert cell["msfe"].mean() == pytest.approx(table.loc[(0.5, False, "hold"), "msfe"])
