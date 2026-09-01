"""Tests for `jump.covariates`: the scalar block of the jump equation, and its timing.

The timing is the whole risk here. Every column has to be measurable before the jump it is
paired with, and an off-by-one in the other direction produces a design that reads the session
it is forecasting -- which fits better, looks better, and is worthless. Nothing downstream can
detect that, so it is checked here against sessions whose prices are chosen so that the right
answer is arithmetic.

The rest of the module is measurement, and each measurement is checked against a hand
computation on the same bars rather than against another implementation of itself.
"""

import numpy as np
import pandas as pd
import pytest

from fts_research.fdaSP import intraday as itd
from fts_research.jump import covariates as cov
from test_intraday import make_bars


@pytest.fixture
def five_days():
    """Five sessions of five bars, with a gap between each close and the next open."""
    return make_bars({
        "2024-01-02": [100.0, 101.0, 102.0, 101.5, 103.0],
        "2024-01-03": [104.0, 104.5, 103.0, 105.0, 106.0],
        "2024-01-04": [105.0, 105.5, 107.0, 106.0, 108.0],
        "2024-01-05": [110.0, 109.0, 109.5, 111.0, 112.0],
        "2024-01-08": [111.0, 112.5, 112.0, 113.0, 114.0],
    })


# ------------------------------------------------------------------ session summaries ------


def test_session_covariates_keep_what_session_summary_defines(five_days):
    """The shared columns must be `fdaSP.intraday`'s, not a second definition of them."""
    summary = itd.session_summary(five_days)
    got = cov.session_covariates(five_days, volume_window=2)
    for column in summary.columns:
        pd.testing.assert_series_equal(got[column], summary[column])


def test_realised_semivariances_split_the_realised_variance(five_days):
    """A session's up and down variation must add back to its realised variance, and their
    difference is the signed jump variation."""
    got = cov.session_covariates(five_days, volume_window=2)
    np.testing.assert_allclose(got["rsv_up"] + got["rsv_down"], got["rvol"] ** 2, atol=1e-14)
    np.testing.assert_allclose(got["sjv"], got["rsv_up"] - got["rsv_down"], atol=1e-14)
    assert (got["rsv_up"] >= 0).all() and (got["rsv_down"] >= 0).all()


def test_semivariances_pick_out_the_sign_of_the_move():
    """A session that only rises has no downside variation, and one that only falls has no
    upside."""
    bars = make_bars({"2024-01-02": [100.0, 101.0, 102.0, 103.0, 104.0],
                      "2024-01-03": [104.0, 103.0, 102.0, 101.0, 100.0]})
    got = cov.session_covariates(bars, volume_window=2)
    assert got["rsv_down"].iloc[0] == 0.0 and got["rsv_up"].iloc[0] > 0
    assert got["rsv_up"].iloc[1] == 0.0 and got["rsv_down"].iloc[1] > 0
    assert got["sjv"].iloc[0] > 0 > got["sjv"].iloc[1]


def test_late_and_early_are_the_returns_over_the_ends_of_the_session(five_days):
    """With five bars and a fifth of the session asked for, each is exactly one bar."""
    got = cov.session_covariates(five_days, fraction=0.25, volume_window=2)
    closes = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
    assert got["late"].iloc[0] == pytest.approx(np.log(103.0) - np.log(101.5))
    assert got["early"].iloc[0] == pytest.approx(np.log(101.0) - np.log(100.0))
    assert got["ret"].iloc[0] == pytest.approx(np.log(closes[-1]) - np.log(closes[0]))


def test_late_and_early_cover_the_whole_session_when_asked_for_all_of_it(five_days):
    got = cov.session_covariates(five_days, fraction=1.0, volume_window=2)
    open_to_close = itd.session_summary(five_days)["ret"]
    np.testing.assert_allclose(got["late"], open_to_close, atol=1e-14)
    np.testing.assert_allclose(got["early"], open_to_close, atol=1e-14)


def test_a_fraction_too_small_to_reach_a_bar_still_takes_one(five_days):
    tiny = cov.session_covariates(five_days, fraction=1e-6, volume_window=2)
    one_bar = cov.session_covariates(five_days, fraction=0.25, volume_window=2)
    np.testing.assert_allclose(tiny["late"], one_bar["late"])


def test_relative_volume_is_taken_against_earlier_sessions_only(five_days):
    """A session's own volume must not enter the mean it is measured against, or a spike
    normalises itself away."""
    bars = five_days.copy()
    bars.loc[bars["session"] == pd.Timestamp("2024-01-08").date(), "volume"] = 1e6
    got = cov.session_covariates(bars, volume_window=3)
    assert got["vol_rel"].iloc[-1] > 100
    assert np.isnan(got["vol_rel"].iloc[0])


# ---------------------------------------------------------------------- foreign session ----


def test_foreign_covariates_stop_at_the_cutoff():
    """Bars after the cutoff must be gone, not down-weighted: this is what separates a
    forecast from a look-ahead into the session being forecast."""
    bars = make_bars({"2024-01-02": [100.0, 101.0, 102.0, 108.0, 109.0]},
                     start="08:00", freq_min=60)
    early = cov.foreign_covariates(bars, until="10:00", prefix="ldn")
    whole = cov.foreign_covariates(bars, until=None, prefix="ldn")
    assert early["ldn_ret"].iloc[0] == pytest.approx(np.log(102.0) - np.log(100.0))
    assert whole["ldn_ret"].iloc[0] == pytest.approx(np.log(109.0) - np.log(100.0))
    assert early["ldn_rvol"].iloc[0] < whole["ldn_rvol"].iloc[0]


def test_foreign_covariates_are_indexed_by_the_night_they_belong_to(five_days):
    got = cov.foreign_covariates(five_days, until=None)
    assert list(got.index) == list(itd.session_summary(five_days).index)
    assert list(got.columns) == ["foreign_ret", "foreign_rvol"]


def test_foreign_covariates_reject_a_cutoff_before_the_open(five_days):
    with pytest.raises(ValueError):
        cov.foreign_covariates(five_days, until="00:30")


# ------------------------------------------------------------------------ the alignment ----


def test_jump_design_pairs_a_jump_with_the_session_before_it(five_days):
    """The row for the jump opening session `t` must carry session `t - 1`'s summaries. If it
    carried session `t`'s the model would read the session it is forecasting."""
    design = cov.jump_design(five_days, volume_window=2)
    covariates = cov.session_covariates(five_days, volume_window=2)
    jumps = itd.overnight_jumps(five_days)

    for date in design.index:
        position = list(covariates.index).index(date)
        previous = covariates.index[position - 1]
        assert design.loc[date, "jump"] == pytest.approx(jumps.loc[date, "jump"])
        for column in ("ret", "late", "early", "close_loc", "rvol", "sjv", "rng"):
            assert design.loc[date, column] == pytest.approx(
                covariates.loc[previous, column]), (date, column)
        assert design.loc[date, column] != pytest.approx(covariates.loc[date, column])


def test_jump_design_lags_the_jump_by_one_night(five_days):
    design = cov.jump_design(five_days, volume_window=2)
    jumps = itd.overnight_jumps(five_days)
    for i, date in enumerate(design.index):
        position = list(jumps.index).index(date)
        assert design.loc[date, "jump_lag"] == pytest.approx(jumps["jump"].iloc[position - 1])
    np.testing.assert_allclose(design["abs_jump_lag"], np.abs(design["jump_lag"]))
    np.testing.assert_allclose(design["abs_jump"], np.abs(design["jump"]))


def test_jump_design_carries_the_weekend(five_days):
    """The jump into Monday 8 January spans three calendar days from Friday 5 January."""
    design = cov.jump_design(five_days, volume_window=2)
    assert design.loc[pd.Timestamp("2024-01-08").date(), "gap_days"] == 3.0


def test_jump_design_joins_the_foreign_columns_on_the_same_night(five_days):
    foreign = cov.foreign_covariates(five_days, until=None, prefix="foreign")
    design = cov.jump_design(five_days, foreign=foreign, volume_window=2)
    for date in design.index:
        assert design.loc[date, "foreign_ret"] == pytest.approx(
            foreign.loc[date, "foreign_ret"])


def test_every_predictor_is_named_in_the_source_table(five_days):
    """A column nobody can say the provenance of has no business in the design."""
    design = cov.jump_design(five_days, volume_window=2)
    predictors = [c for c in design.columns if c not in ("jump", "abs_jump")]
    missing = [c for c in predictors if c not in cov.SOURCE]
    assert not missing, f"predictors with no entry in SOURCE: {missing}"


# ------------------------------------------------------------------- multiple testing ------


def test_benjamini_hochberg_against_a_hand_computation():
    """Four tests at 5%: the sorted p-values are compared with 0.0125, 0.025, 0.0375, 0.05,
    and the largest index that clears its own line takes everything below it with it."""
    got = cov.benjamini_hochberg([0.001, 0.04, 0.3, 0.02], alpha=0.05)
    np.testing.assert_array_equal(got["reject"], [True, False, False, True])
    assert got["n_reject"] == 2
    assert got["threshold"] == pytest.approx(0.02)
    np.testing.assert_allclose(got["qvalues"], [0.004, 0.05333333, 0.3, 0.04], rtol=1e-6)


def test_benjamini_hochberg_rejects_nothing_when_nothing_is_there():
    rng = np.random.default_rng(0)
    got = cov.benjamini_hochberg(rng.uniform(size=200), alpha=0.05)
    assert got["n_reject"] <= 5                      # a false discovery rate, not a count
    assert cov.benjamini_hochberg([0.6, 0.7, 0.8])["n_reject"] == 0
    assert cov.benjamini_hochberg([])["n_reject"] == 0


def test_benjamini_hochberg_is_never_stricter_than_bonferroni():
    p = np.array([0.001, 0.006, 0.02, 0.2, 0.5])
    got = cov.benjamini_hochberg(p, alpha=0.05)
    assert got["n_reject"] >= int(np.sum(p <= 0.05 / p.size))
    assert np.all(got["qvalues"] >= p - 1e-12)
    assert np.all(got["qvalues"] <= 1.0)


def test_benjamini_hochberg_treats_a_missing_test_as_no_evidence():
    got = cov.benjamini_hochberg([0.001, np.nan, np.nan, np.nan], alpha=0.05)
    np.testing.assert_array_equal(got["reject"], [True, False, False, False])


# ------------------------------------------------------------------- the committed panel ---


needs_panel = pytest.mark.skipif(
    not __import__("os").path.exists(itd.panel_path("1h")),
    reason="cached intraday panel is absent; run `python -m fts_research.fdaSP.intraday`",
)


@needs_panel
@pytest.mark.slow
def test_panel_design_stacks_every_ticker_over_the_same_nights():
    panel = cov.panel_design("1h")
    assert set(panel["ticker"]) == set(itd.TICKERS) - {"^FTSE"}
    assert {"jump", "abs_jump", "session", "foreign_ret"} <= set(panel.columns)
    assert panel.notna().all().all()

    # One night's foreign session is one number, whatever instrument it is paired with. That
    # is the shared row a random cross-validation split would leak across.
    by_date = panel.groupby("session")["foreign_ret"].nunique()
    assert (by_date == 1).all()
    assert panel.groupby("ticker").size().min() > 100
