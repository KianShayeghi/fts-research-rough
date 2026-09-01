"""Tests for `fdaSP.intraday`: the reshaping of raw bars into curves, jumps and covariates.

Most of these run on synthetic bars built in the test, because a data layer is exactly the
place where a bug hides behind plausible-looking output -- an off-by-one in the curve/jump
alignment produces numbers of the right size, the right sign and the right autocorrelation,
and is only visible against a series whose answer is known by construction.

The handful that read the committed panels skip themselves when those are absent. Nothing
here touches the network.
"""

import os

import numpy as np
import pandas as pd
import pytest

from fts_research.fdaSP import intraday as itd


def make_bars(prices_by_day, ticker="TEST", start="09:30", freq_min=60, div=None):
    """Bars in the shape `load_bars` returns, from {date string: [prices]}.

    Open is the previous listed price where one exists, so a gap between one day's last
    price and the next day's first is exactly the overnight jump the tests then look for.

    PARAMETERS
    prices_by_day : dict of {"YYYY-MM-DD": sequence of close prices};
    ticker        : str, the symbol to label them with;
    start         : str, wall-clock time of the first bar of each session;
    freq_min      : int, minutes between bars;
    div           : dict of {"YYYY-MM-DD": amount} or None, cash dividends.
    """
    rows = []
    for day, closes in prices_by_day.items():
        base = pd.Timestamp(f"{day} {start}")
        for i, c in enumerate(closes):
            rows.append({
                "ticker": ticker,
                "ts": base + pd.Timedelta(minutes=freq_min * i),
                "open": closes[i - 1] if i else c,
                "high": c * 1.001,
                "low": c * 0.999,
                "close": c,
                "volume": 1000.0,
                "session": pd.Timestamp(day).date(),
                "div": (div or {}).get(day, 0.0),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def three_days():
    """Three full sessions of five bars, with a known gap between each close and next open."""
    return make_bars({
        "2024-01-02": [100.0, 101.0, 102.0, 101.5, 103.0],
        "2024-01-03": [104.0, 104.5, 103.0, 105.0, 106.0],
        "2024-01-04": [105.0, 105.5, 107.0, 106.0, 108.0],
    })


# ------------------------------------------------------------------------------ sessions


def test_sessions_splits_on_the_local_trading_date(three_days):
    days = itd.sessions(three_days)
    assert len(days) == 3
    assert all(len(g) == 5 for g in days)


def test_sessions_drops_a_short_day_by_default(three_days):
    """An early close is a shorter trading day, not a faster one, so its curve does not live
    on the same domain as the rest and the default filter removes it."""
    half = make_bars({"2024-01-05": [109.0, 110.0]})
    days = itd.sessions(pd.concat([three_days, half], ignore_index=True))
    assert len(days) == 3
    assert all(g["session"].iloc[0] != pd.Timestamp("2024-01-05").date() for g in days)


def test_min_bars_can_be_lowered_to_keep_a_short_day(three_days):
    half = make_bars({"2024-01-05": [109.0, 110.0]})
    both = pd.concat([three_days, half], ignore_index=True)
    assert len(itd.sessions(both, min_bars=2)) == 4


def test_sessions_refuses_a_frame_holding_more_than_one_ticker(three_days):
    other = make_bars({"2024-01-02": [50.0] * 5}, ticker="OTHER")
    with pytest.raises(ValueError, match="one ticker at a time"):
        itd.sessions(pd.concat([three_days, other], ignore_index=True))


# ------------------------------------------------------------------------ session_curves


def test_every_curve_starts_at_exactly_zero(three_days):
    """The CIDR construction subtracts the session's own opening log price, so X_t(0) = 0 is
    an identity and not an approximation."""
    X, grid, _ = itd.session_curves(three_days)
    assert np.all(X[:, 0] == 0.0)
    assert grid[0] == 0.0 and grid[-1] == 1.0


def test_curve_endpoint_is_the_first_to_last_bar_log_return(three_days):
    X, _, _ = itd.session_curves(three_days)
    day = itd.sessions(three_days)[0]
    expected = np.log(day["close"].iloc[-1]) - np.log(day["close"].iloc[0])
    assert X[0, -1] == pytest.approx(expected)


def test_curves_come_back_in_date_order(three_days):
    _, _, dates = itd.session_curves(three_days)
    assert list(dates) == sorted(dates)


def test_a_flat_session_gives_a_flat_curve():
    flat = make_bars({"2024-01-02": [100.0] * 6})
    X, _, _ = itd.session_curves(flat)
    assert np.allclose(X, 0.0)


def test_uneven_bar_spacing_places_points_by_clock_time():
    """`s` is the fraction of the session *elapsed*, not the fraction of bars seen, so a gap
    in the middle of a day must stretch the curve rather than compress it."""
    rows = make_bars({"2024-01-02": [100.0, 110.0, 121.0]})
    rows.loc[2, "ts"] = pd.Timestamp("2024-01-02 12:30")     # last bar 3h after the first
    X, grid, _ = itd.session_curves(rows, n_points=7)
    # The middle bar sits one hour into a three-hour session, i.e. at s = 1/3.
    assert X[0, np.argmin(np.abs(grid - 1 / 3))] == pytest.approx(np.log(1.10), rel=1e-9)


def test_n_points_interpolates_onto_the_requested_grid(three_days):
    X, grid, _ = itd.session_curves(three_days, n_points=21)
    assert X.shape == (3, 21) and grid.size == 21


# ----------------------------------------------------------------------- overnight_jumps


def test_jump_is_the_close_to_open_log_change(three_days):
    jumps = itd.overnight_jumps(three_days)
    assert len(jumps) == 2
    assert jumps["jump"].iloc[0] == pytest.approx(np.log(104.0) - np.log(103.0))
    assert jumps["jump"].iloc[1] == pytest.approx(np.log(105.0) - np.log(106.0))


def test_jump_is_indexed_by_the_opening_session(three_days):
    """The jump into day t belongs to day t, because that is the day it is a forecast of."""
    jumps = itd.overnight_jumps(three_days)
    assert list(jumps.index) == [pd.Timestamp("2024-01-03").date(),
                                 pd.Timestamp("2024-01-04").date()]


def test_jump_and_curve_reconstruct_the_open_to_open_return(three_days):
    """The whole premise of the application: the curve is exactly what the jump omits. Their
    sum over consecutive days must be the open-to-open log return, to machine precision."""
    X, _, dates = itd.session_curves(three_days)
    jumps = itd.overnight_jumps(three_days)
    days = itd.sessions(three_days)
    opens = np.array([g["open"].iloc[0] for g in days], float)

    for i in range(len(days) - 1):
        open_to_open = np.log(opens[i + 1]) - np.log(opens[i])
        assert X[i, -1] + jumps["jump"].iloc[i] == pytest.approx(open_to_open, abs=1e-12)


def test_gap_days_records_how_far_a_jump_reaches(three_days):
    assert list(itd.overnight_jumps(three_days)["gap_days"]) == [1, 1]


def test_a_weekend_jump_is_marked_as_three_days():
    friday_monday = make_bars({"2024-01-05": [100.0] * 4, "2024-01-08": [102.0] * 4})
    assert itd.overnight_jumps(friday_monday)["gap_days"].iloc[0] == 3


def test_dividends_are_added_back_before_differencing():
    """A stock opens lower by roughly its dividend on the ex-date. That is a payment to the
    holder, not a price move, and leaving it in puts a spurious negative jump on every
    ex-date."""
    bars = make_bars({"2024-01-02": [100.0] * 4, "2024-01-03": [99.0] * 4},
                     div={"2024-01-03": 1.0})
    net = itd.overnight_jumps(bars)["jump"].iloc[0]
    gross = itd.overnight_jumps(bars, net_of_dividends=False)["jump"].iloc[0]
    assert net == pytest.approx(0.0, abs=1e-12)
    assert gross == pytest.approx(np.log(99.0) - np.log(100.0))


def test_overnight_jumps_needs_two_sessions():
    with pytest.raises(ValueError, match="at least two sessions"):
        itd.overnight_jumps(make_bars({"2024-01-02": [100.0] * 4}))


# ------------------------------------------------------------------ align_curves_to_jumps


def test_alignment_pairs_a_curve_with_the_following_night(three_days):
    X, _, dates = itd.session_curves(three_days)
    jumps = itd.overnight_jumps(three_days)
    Xp, y, when = itd.align_curves_to_jumps(X, dates, jumps)

    assert len(y) == 2
    assert list(when) == list(dates[:2])                       # predictor days, not response
    assert np.allclose(Xp[0], X[0])
    assert y[0] == pytest.approx(jumps["jump"].iloc[0])


def test_alignment_never_pairs_across_a_dropped_session(three_days):
    """If the jump into day t+1 is missing, day t's curve has nothing to predict and must be
    dropped rather than silently matched to the night after next."""
    X, _, dates = itd.session_curves(three_days)
    jumps = itd.overnight_jumps(three_days).iloc[1:]           # drop the first night
    Xp, y, when = itd.align_curves_to_jumps(X, dates, jumps)

    assert len(y) == 1
    assert when[0] == dates[1]
    assert np.allclose(Xp[0], X[1])


def test_alignment_raises_when_nothing_can_be_paired(three_days):
    X, _, dates = itd.session_curves(three_days)
    empty = itd.overnight_jumps(three_days).iloc[:0]
    with pytest.raises(ValueError, match="no curve could be paired"):
        itd.align_curves_to_jumps(X, dates, empty)


# ------------------------------------------------------------------------ session_summary


def test_summary_return_matches_the_curve_endpoint(three_days):
    X, _, _ = itd.session_curves(three_days)
    summary = itd.session_summary(three_days)
    assert summary["ret"].to_numpy() == pytest.approx(X[:, -1], abs=1e-12)


def test_realised_volatility_is_zero_on_a_flat_day_and_positive_otherwise():
    flat = make_bars({"2024-01-02": [100.0] * 6})
    assert itd.session_summary(flat)["rvol"].iloc[0] == pytest.approx(0.0)
    moving = make_bars({"2024-01-02": [100.0, 102.0, 99.0, 103.0, 98.0, 101.0]})
    assert itd.session_summary(moving)["rvol"].iloc[0] > 0.05


def test_close_location_is_one_when_the_day_closes_on_its_high():
    rising = make_bars({"2024-01-02": [100.0, 101.0, 102.0, 103.0]})
    # `make_bars` puts the high 0.1% above each close, so the last bar's high is the day's.
    assert itd.session_summary(rising)["close_loc"].iloc[0] > 0.9


def test_missing_volume_reports_as_nan():
    """The index symbols are computed rather than traded and report no volume; a zero there
    would read as a quiet day rather than as a quantity that does not exist."""
    bars = make_bars({"2024-01-02": [100.0] * 5})
    bars["volume"] = np.nan
    assert np.isnan(itd.session_summary(bars)["volume"].iloc[0])


# -------------------------------------------------------------------------- split events


def test_split_factor_rescales_only_the_bars_before_the_split():
    """An unadjusted 4-for-1 split reads as a 75% overnight fall unless the earlier bars are
    put on today's share count."""
    ts = np.array([100, 200, 300, 400], dtype="int64")
    events = {"splits": {"1": {"date": 250, "numerator": 4, "denominator": 1}}}
    factor = itd._split_factors(events, ts)
    assert list(factor) == [4.0, 4.0, 1.0, 1.0]


def test_split_factors_compose_across_two_splits():
    ts = np.array([100, 200, 300], dtype="int64")
    events = {"splits": {"a": {"date": 150, "numerator": 2, "denominator": 1},
                         "b": {"date": 250, "numerator": 3, "denominator": 1}}}
    assert list(itd._split_factors(events, ts)) == [6.0, 3.0, 1.0]


def test_no_events_leaves_every_bar_alone():
    ts = np.array([1, 2, 3], dtype="int64")
    assert list(itd._split_factors(None, ts)) == [1.0, 1.0, 1.0]
    assert list(itd._split_factors({}, ts)) == [1.0, 1.0, 1.0]


# ------------------------------------------------------------------------------- lookups


def test_panel_path_rejects_an_unknown_panel():
    with pytest.raises(KeyError, match="unknown panel"):
        itd.panel_path("1s")


def test_every_panel_declares_what_it_is_for():
    for name, spec in itd.PANELS.items():
        assert {"interval", "range", "file", "note"} <= set(spec)
        assert len(spec["note"]) > 60, f"panel {name} has no usable note"


# -------------------------------------------------------------- the committed snapshots


def cached(name):
    return os.path.exists(itd.panel_path(name))


needs_panel = pytest.mark.skipif(
    not (cached("1h") and cached("5m")),
    reason="cached intraday panels are absent; run `python -m fts_research.fdaSP.intraday`",
)


@needs_panel
@pytest.mark.parametrize("name", ["1h", "5m"])
def test_cached_panel_loads_with_the_columns_downstream_expects(name):
    bars = itd.load_bars(name)
    assert {"ticker", "ts", "open", "high", "low", "close", "volume", "div", "session"} \
        <= set(bars.columns)
    assert len(bars) > 1000
    assert bars["close"].gt(0).all()


@needs_panel
@pytest.mark.parametrize("name", ["1h", "5m"])
def test_cached_panel_holds_every_declared_ticker(name):
    assert set(itd.load_bars(name)["ticker"]) == set(itd.TICKERS)


@needs_panel
def test_the_two_panels_agree_on_the_jumps_they_share():
    """The panels are independent downloads at different resolutions, so a jump both cover
    is computed twice from different bars. They must agree: if they do not, the session
    boundaries are being found differently in the two and one of them is wrong."""
    coarse = itd.overnight_jumps(itd.load_bars("1h", tickers=["AAPL"]))
    fine = itd.overnight_jumps(itd.load_bars("5m", tickers=["AAPL"]))
    shared = coarse.index.intersection(fine.index)
    assert len(shared) > 30
    assert np.allclose(coarse.loc[shared, "jump"], fine.loc[shared, "jump"], atol=2e-3)


@needs_panel
def test_no_jump_in_the_snapshot_is_absurd():
    """A missed split shows up as a jump of tens of per cent. Earnings gaps reach the teens,
    so the bound is set where only a corporate action could sit."""
    for name in ("1h", "5m"):
        bars = itd.load_bars(name)
        for sym in itd.TICKERS:
            jumps = itd.overnight_jumps(bars[bars["ticker"] == sym])["jump"]
            assert jumps.abs().max() < 0.25, f"{sym} in {name} jumps {jumps.abs().max():.2f}"


@needs_panel
def test_stack_panel_preserves_curve_shape_while_equalising_scale():
    """Pooling divides each ticker by its own jump standard deviation. That is one constant
    per ticker, so every curve's shape survives it exactly and only the scale moves."""
    pooled = itd.stack_panel("5m", tickers=["AAPL", "SPY"])
    raw = itd.stack_panel("5m", tickers=["AAPL", "SPY"], standardise=False)

    assert pooled["X"].shape == raw["X"].shape
    for sym in ("AAPL", "SPY"):
        take = pooled["ticker"] == sym
        ratio = raw["X"][take][:, 1:] / pooled["X"][take][:, 1:]
        assert np.nanstd(ratio) < 1e-8, f"{sym} was not rescaled by a single constant"
    assert abs(np.std(pooled["y"][pooled["ticker"] == "AAPL"])
               - np.std(pooled["y"][pooled["ticker"] == "SPY"])) < 0.05


@needs_panel
def test_snapshot_span_reports_the_window_the_numbers_come_from():
    start, end = itd.snapshot_span("1h")
    assert start < end


@needs_panel
def test_the_two_standardisations_differ_where_the_jump_scale_is_unrepresentative():
    """`^FTSE`'s opening print is a stale composite, so its jump standard deviation is far
    below every other ticker's while its intraday range is ordinary. Dividing its curves by
    that number -- which `standardise="jump"` does -- inflates them past the rest of the
    panel, and `standardise="each"` exists precisely to avoid it."""
    each = itd.stack_panel("1h", kind="rvol", standardise="each")
    jump = itd.stack_panel("1h", kind="rvol", standardise="jump")

    def endpoint_mean(d, sym):
        return float(np.mean(d["X"][d["ticker"] == sym][:, -1]))

    others = [endpoint_mean(jump, s) for s in ("AAPL", "SPY", "QQQ")]
    assert endpoint_mean(jump, "^FTSE") > 3 * max(others)
    assert endpoint_mean(each, "^FTSE") < 3 * max(endpoint_mean(each, s)
                                                  for s in ("AAPL", "SPY", "QQQ"))


@needs_panel
def test_target_absolute_returns_the_size_of_the_signed_jump():
    signed = itd.stack_panel("1h", target="signed", standardise="jump")
    absolute = itd.stack_panel("1h", target="absolute", standardise="jump")
    assert np.allclose(np.abs(signed["y"]), absolute["y"])
    assert np.allclose(signed["X"], absolute["X"])


def test_stack_panel_rejects_an_unknown_target_or_standardisation():
    with pytest.raises(ValueError, match="target must be"):
        itd.stack_panel("1h", target="squared")
    with pytest.raises(ValueError, match="standardise must be"):
        itd.stack_panel("1h", standardise=True)


def test_volatility_curves_start_at_zero_and_increase(three_days):
    """The running realised volatility accumulates squared increments, so it can only go up,
    and its endpoint is the session's realised volatility."""
    V, grid, _ = itd.session_curves(three_days, kind="rvol")
    assert np.all(V[:, 0] == 0.0)
    assert np.all(np.diff(V, axis=1) >= -1e-12)
    assert V[0, -1] == pytest.approx(itd.session_summary(three_days)["rvol"].iloc[0])


def test_session_curves_rejects_an_unknown_kind(three_days):
    with pytest.raises(ValueError, match="kind must be"):
        itd.session_curves(three_days, kind="price")


def test_date_blocked_folds_keep_a_date_whole_and_contiguous():
    """Rows from the same day but different tickers share that day's market-wide move, so a
    split that separates them tests a model on a near-copy of what it was fitted on."""
    dates = np.array(["2024-01-0%d" % d for d in (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)])
    folds = itd.date_blocked_folds(dates, 3)
    for d in set(dates.tolist()):
        assert len(set(folds[dates == d])) == 1, "a date was split across folds"
    assert np.all(np.diff(folds) >= 0), "folds are not contiguous in time"


def test_date_blocked_folds_refuse_more_folds_than_dates():
    with pytest.raises(ValueError, match="cannot fill"):
        itd.date_blocked_folds(np.array(["a", "a", "b"]), 5)
