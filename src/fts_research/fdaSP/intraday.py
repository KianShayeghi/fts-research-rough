"""Intraday price bars, reshaped into the daily curves and overnight jumps that
`notebooks/overnight_jumps.ipynb` needs.

A trading day supplies one curve and one scalar. The curve is the cumulative intraday
return (CIDR),

    X_t(s) = log P_t(s) - log P_t(0),   s in [0, 1] the fraction of the session elapsed,

which starts at zero by construction and so carries the shape of the day rather than its
level. The scalar is the overnight jump

    J_t = log(Open_t) - log(Close_{t-1}),

the move made while the market is shut. The two partition the open-to-open log return: the
CIDR curve is exactly what the jump omits, and the jump is exactly what the curve omits.
Regressing J_t on X_{t-1} is therefore a scalar-on-function regression on the only two pieces
there are.

Bars come from the public Yahoo Finance chart endpoint, cached as CSVs under the repository's
`data/` directory (override with `FTS_DATA_DIR`). Two panels are kept, because the endpoint
trades resolution against history:

    PANELS["1h"]   hourly bars, ~2 years, 7 points a session -- the long panel, for anything
                   whose power comes from the number of days;
    PANELS["5m"]   5-minute bars, 60 days, ~78 points a session -- the fine panel, for
                   anything whose power comes from the resolution of the curve.

**The cached CSVs are dated snapshots, not reproducible downloads.** The endpoint serves a
window ending today, so re-fetching on a later date returns different bars and every number
downstream moves with them. `ensure_panel` therefore prefers the cache and downloads only
when a file is missing, and records the day it took the snapshot and the window it covers
in `data/manifest.json`. `snapshot_span` reads that window off the cache itself.

Run `python -m fts_research.fdaSP.intraday` to fetch everything up front, or with `--force`
to take a fresh snapshot.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from ..core import paths, provenance

__all__ = [
    "TICKERS",
    "PANELS",
    "CHART_URL",
    "panel_path",
    "fetch_bars",
    "ensure_panel",
    "ensure_all",
    "load_bars",
    "snapshot_span",
    "SPX_2017",
    "load_spx_2017",
    "sessions",
    "session_curves",
    "overnight_jumps",
    "session_summary",
    "align_curves_to_jumps",
    "align_curves_to_series",
    "stack_panel",
    "date_blocked_folds",
]

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

# A "ret" curve is a step function, so its grid resolves the steps rather than matching the
# bars. The trapezium error falls as 1/n_points; the cap keeps a 5-minute panel's design from
# growing to thousands of columns for no gain, since the information is still one value a bar.
_RATE_REFINEMENT = 20
_RATE_MAX_POINTS = 401

_SPX_TZ = "America/New_York"

_USER_AGENT = "Mozilla/5.0 (compatible; fts-research/0.1)"
_TIMEOUT = 120
_PAUSE = 1.5        # seconds between requests; the endpoint resets the connection if hammered
_RETRIES = 4

# Five single stocks, two ETFs and two index funds, which is the spread the application asks
# for. The two indices are quoted rather than traded, so they report no volume and their
# opening print is a stale composite of constituent opens; `session_summary` returns NaN
# volume for them and the notebook reads their jumps with that caveat attached.
TICKERS = ("AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "SPY", "QQQ", "^GSPC", "^FTSE")

PANELS = {
    "1h": {
        "interval": "1h",
        "range": "2y",
        "file": "intraday_1h.csv",
        "note": (
            "Hourly bars, the longest history the endpoint serves below daily (730 days). "
            "About 7 points a US session, so the curve is coarse but the panel is some 500 "
            "days deep: this is the panel for serial-dependence questions."
        ),
    },
    "5m": {
        "interval": "5m",
        "range": "60d",
        "file": "intraday_5m.csv",
        "note": (
            "Five-minute bars, the finest resolution the endpoint serves over a useful "
            "window (60 days). About 78 points a US session, so the curve is well resolved "
            "but the panel is only some 60 days deep: this is the panel for curve-shape "
            "questions, pooled across tickers to recover the sample size."
        ),
    },
}


def panel_path(name):
    """Absolute path of a cached panel CSV, whether or not it exists yet.

    PARAMETERS
    name : str, a key of `PANELS`.
    """
    if name not in PANELS:
        raise KeyError(f"unknown panel {name!r}; known: {sorted(PANELS)}.")
    return paths.data_path(PANELS[name]["file"])


def _get(url):
    """One GET against the chart endpoint, retrying the transient failures it throws when
    several symbols are requested in a row. Returns the response body as bytes."""
    last = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            time.sleep(_PAUSE * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last}") from last


def _split_factors(events, ts):
    """Cumulative split divisor for each timestamp in `ts`.

    Yahoo's quote block is unadjusted, so a 4-for-1 split reads as a 75% overnight fall.
    Dividing every bar strictly before a split by that split's ratio puts the whole series on
    today's share count. Returns a (len(ts),) array, all 1.0 where there were no splits.

    PARAMETERS
    events : dict or None, the chart response's `events` block;
    ts     : (T,) array of epoch seconds, ascending.
    """
    factor = np.ones(len(ts), dtype=float)
    ts = np.asarray(ts, dtype=float)
    for ev in (events or {}).get("splits", {}).values():
        num = float(ev.get("numerator", 1.0))
        den = float(ev.get("denominator", 1.0))
        if den == 0 or num == 0:
            continue
        factor[ts < float(ev["date"])] *= num / den
    return factor


def fetch_bars(symbol, interval, range_):
    """Download one symbol's bars from the Yahoo chart endpoint. Returns a DataFrame with
    columns ticker, ts (exchange-local wall clock), session, open, high, low, close, volume
    and div.

    Prices are put on today's share count by `_split_factors`, so an unadjusted split does
    not read as an overnight collapse. Cash dividends are carried in `div`, keyed to the
    ex-dividend session, because a stock opens lower by roughly the dividend and that is a
    payment to the holder rather than a price move. Pre- and post-market bars are excluded
    at the source.

    PARAMETERS
    symbol   : str, a Yahoo symbol; a `^`-prefixed index symbol is URL-encoded here;
    interval : str, bar width, e.g. "1h" or "5m";
    range_   : str, history to request, e.g. "2y" or "60d".
    """
    url = (f"{CHART_URL}{urllib.parse.quote(symbol)}?interval={interval}&range={range_}"
           f"&includePrePost=false&events=div%2Csplit")
    payload = json.loads(_get(url))
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"{symbol}: the endpoint returned no data for {interval}/{range_}.")
    res = result[0]

    ts = np.asarray(res["timestamp"], dtype="int64")
    quote = res["indicators"]["quote"][0]
    frame = pd.DataFrame({k: pd.to_numeric(pd.Series(quote.get(k)), errors="coerce")
                          for k in ("open", "high", "low", "close", "volume")})

    factor = _split_factors(res.get("events"), ts)
    for col in ("open", "high", "low", "close"):
        frame[col] = frame[col].to_numpy(float) / factor
    frame["volume"] = frame["volume"].to_numpy(float) * factor

    # Stored as exchange-local wall clock with the offset dropped. Absolute instants are
    # never compared across exchanges here, whereas time of day is read constantly -- the
    # shape of a session is indexed by how far through it we are -- so local time is the
    # useful column and it is also the shorter one to cache.
    tz = res["meta"].get("exchangeTimezoneName", "UTC")
    local = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None)
    frame.insert(0, "ts", local)
    frame.insert(0, "ticker", symbol)

    # The session label is the *local* trading date, so a London bar at 08:00 and a New York
    # bar at 09:30 each land on the day their own exchange calls it. It is computed here,
    # where the exchange timezone is known, and cached alongside the bars.
    frame["session"] = frame["ts"].dt.date

    frame["div"] = 0.0
    for ev in (res.get("events") or {}).get("dividends", {}).values():
        when = pd.Timestamp(int(ev["date"]), unit="s", tz="UTC").tz_convert(tz).date()
        frame.loc[frame["session"] == when, "div"] = float(ev["amount"])

    frame = frame.dropna(subset=["open", "high", "low", "close"])
    return frame.sort_values("ts").reset_index(drop=True)


def ensure_panel(name, force=False, tickers=None):
    """Return the path to a cached panel CSV, downloading it first if necessary.

    The cache is preferred over the network unless `force` is set: the endpoint serves a
    window ending today, so a re-fetch silently replaces the snapshot on which every
    downstream number was computed.

    PARAMETERS
    name    : str, a key of `PANELS`;
    force   : bool, take a fresh snapshot even though the file is already cached;
    tickers : sequence of str or None, symbols to fetch; defaults to `TICKERS`.
    """
    path = panel_path(name)
    if os.path.exists(path) and not force:
        return path

    spec = PANELS[name]
    os.makedirs(paths.DATA_DIR, exist_ok=True)
    parts, failed = [], []
    for i, sym in enumerate(TICKERS if tickers is None else tickers):
        if i:
            time.sleep(_PAUSE)
        try:
            parts.append(fetch_bars(sym, spec["interval"], spec["range"]))
        except Exception as exc:                 # one dead symbol must not cost us the rest
            failed.append(f"{sym} ({exc})")
    if not parts:
        raise RuntimeError(
            f"panel {name!r} is not cached at {path} and nothing could be downloaded: "
            f"{'; '.join(failed)}. Fetch it manually, or set FTS_DATA_DIR to a directory "
            f"that already holds it."
        )
    if failed:
        print(f"warning: {name} panel is missing {', '.join(failed)}", file=sys.stderr)

    out = pd.concat(parts, ignore_index=True)
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].round(4)
    out["volume"] = out["volume"].round().astype("Int64")
    tmp = path + ".part"
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)      # atomic, so an interrupted fetch never leaves a half file
    sessions_seen = out["session"].astype(str)
    provenance.record(
        spec["file"], path,
        f"{CHART_URL}{{symbol}}?interval={spec['interval']}&range={spec['range']}",
        span=[sessions_seen.min(), sessions_seen.max()])
    return path


def ensure_all(force=False):
    """Download and cache every panel. Returns {name: path}."""
    return {name: ensure_panel(name, force=force) for name in PANELS}


def load_bars(name="1h", tickers=None):
    """Load a cached bar panel. Returns a long DataFrame, one row per bar, with columns
    ticker, ts (exchange-local wall clock), open, high, low, close, volume, div and session
    (the local trading date).

    PARAMETERS
    name    : str, a key of `PANELS`;
    tickers : sequence of str or None, restrict to these symbols; None keeps all of them.
    """
    df = pd.read_csv(ensure_panel(name))
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    df["session"] = pd.to_datetime(df["session"]).dt.date
    if tickers is not None:
        df = df[df["ticker"].isin(list(tickers))]
    return df.sort_values(["ticker", "ts"]).reset_index(drop=True)


def snapshot_span(name="1h"):
    """First and last session in a cached panel, as a (start, end) pair of dates.

    The panels are dated snapshots of a rolling window, so this is what pins the numbers a
    notebook reports to the data they were computed on.

    PARAMETERS
    name : str, a key of `PANELS`.
    """
    ses = load_bars(name)["session"]
    return ses.min(), ses.max()


# The one panel here that is not a snapshot. Shang (2017a) forecasts five-minute S&P 500
# cumulative intraday return curves over the first half of a year, and the chart endpoint above
# serves five-minute bars for sixty days only, so no fetch reproduces that design. This file is
# a fixed historical window instead: one-minute regular-session closes of the cash index for
# 2017-01-03 to 2017-06-30, which resample to Shang's own grid and, being historical rather
# than rolling, do not move under anyone's feet the way the two panels above do.
SPX_2017 = {
    "file": "spx_1m_2017H1.csv",
    "ticker": "^GSPC",
    "note": (
        "One-minute regular-session closes of the S&P 500 cash index, 2017-01-03 to "
        "2017-06-30. 125 full sessions of 09:30 to 15:59 and three early closes, which "
        "`load_spx_2017` drops. Closes only: `overnight_jumps` and `session_summary` need "
        "the open, high and low this file does not carry, and do not apply to it."
    ),
}


def load_spx_2017(min_span=0.95):
    """Load the fixed 2017 S&P 500 panel, in the long format `load_bars` returns.

    Returns a DataFrame with ticker, ts (exchange-local wall clock), close and session, one row
    per one-minute bar, in time order; only the sessions running the full trading day survive.

    An early close is a shorter trading day rather than a faster one, so the three US holiday
    half-days in this window are not curves on the same domain as the rest and are dropped
    here. They are told apart by the span from a session's first bar to its last, not by how
    many bars it holds: this feed has gaps inside otherwise complete days, and the bar-count
    rule `sessions` applies to the fetched panels would throw nine full sessions away with
    them. With the early closes already gone, pass `min_bars=1` to `session_curves` and let
    this filter stand.

    PARAMETERS
    min_span : float, fraction of the modal first-bar-to-last-bar span a session must reach.
    """
    path = paths.data_path(SPX_2017["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{SPX_2017['file']} is not at {path}. It is committed under data/ rather than "
            f"fetched, so a missing file means the checkout is incomplete or FTS_DATA_DIR "
            f"points elsewhere."
        )
    raw = pd.read_csv(path)
    ts = pd.to_datetime(raw["time"], utc=True, format="ISO8601")
    local = ts.dt.tz_convert(_SPX_TZ).dt.tz_localize(None)

    frame = pd.DataFrame({"ticker": SPX_2017["ticker"], "ts": local,
                          "close": raw["close"].astype(float)})
    frame["session"] = frame["ts"].dt.date
    frame = frame.sort_values("ts").reset_index(drop=True)

    span = frame.groupby("session")["ts"].agg(lambda g: (g.max() - g.min()).total_seconds())
    keep = span.index[span >= float(min_span) * span.mode().iloc[0]]
    return frame[frame["session"].isin(set(keep))].reset_index(drop=True)


def sessions(bars, min_bars=None):
    """Split one ticker's bars into trading sessions. Returns a list of DataFrames, one per
    session, in date order.

    A session is kept only when it carries at least `min_bars` bars, which drops the
    half-days and the partial session at either end of the sample. Those are not curves on
    the same domain as the rest: an early close is a shorter trading day, not a faster one.

    PARAMETERS
    bars     : DataFrame from `load_bars`, restricted to a single ticker;
    min_bars : int or None, fewest bars a session may have; defaults to 80% of the modal
               session length, which keeps full days and drops early closes.
    """
    if bars["ticker"].nunique() > 1:
        raise ValueError("`sessions` takes one ticker at a time; filter `bars` first.")
    groups = [g for _, g in bars.groupby("session", sort=True)]
    if not groups:
        return []
    if min_bars is None:
        modal = int(pd.Series([len(g) for g in groups]).mode().iloc[0])
        min_bars = int(np.ceil(0.8 * modal))
    return [g.reset_index(drop=True) for g in groups if len(g) >= min_bars]


def session_curves(bars, n_points=None, min_bars=None, price="close", kind="cidr",
                   until=None):
    """Build one curve per session, on a shared grid.

    Two curves can be built from the same bars, and the application needs both:

        kind="cidr"   `X_t(s) = log P_t(s) - log P_t(0)`, the cumulative intraday return.
                      Starts at exactly zero, so what is left is the *direction* of the day.
        kind="rvol"   `V_t(s) = sqrt(sum_{u <= s} (dlog P_u)^2)`, the running realised
                      volatility. Starts at zero and increases, and its endpoint is the
                      session's realised volatility, so what it carries is the *magnitude*
                      of the day and not its sign.
        kind="ret"    `R_t(s) = dX_t/ds`, the return *rate*, so that `int_0^1 R_t` is the
                      session total. This is the one to regress on when the coefficient
                      function has to be readable: `psi` constant then says the total move
                      matters however it arrived, and `psi` concentrated near `s = 1` says
                      only the late part of the session does. On the cumulative curve those
                      two hypotheses are separated only by a boundary term, which is why
                      `"cidr"` is the wrong predictor for that question even though it holds
                      the same information.

                      Nothing is known about the price inside a bar, so the rate is a step
                      function, and the trapezium rule recovers the session total from it
                      only to O(1/n_points) -- the error is made at the steps. The default
                      grid is therefore much finer than the bar count. That error rescales
                      the whole design and is absorbed into `psi`, so it shifts no
                      conclusion; it is a reason not to read `int R_t` as the exact total.

    In both cases `s` is the fraction of the retained span elapsed, linearly interpolated
    onto `n_points` equally spaced values.

    `until` cuts every session off at a wall-clock time and renormalises `s` over what is
    left, so `s = 1` is that cutoff rather than the close. This is what makes a curve from
    one exchange usable as a predictor of another's open: the London session runs while New
    York is shut, and truncating it before New York opens is the difference between a
    forecast and a look-ahead.

    Returns X (n_sessions, n_points), grid (n_points,) and dates (n_sessions,), in date order.

    PARAMETERS
    bars     : DataFrame from `load_bars`, restricted to a single ticker;
    n_points : int or None, points on the shared grid. Defaults to the modal session
               length, so a "cidr" or "rvol" curve interpolates nothing; a "ret" curve
               instead defaults to `_RATE_REFINEMENT` times that, capped at
               `_RATE_MAX_POINTS`, to resolve its steps;
    min_bars : int or None, passed to `sessions`;
    price    : str, the bar column read as the price, "close" by default;
    kind     : {"cidr", "rvol", "ret"}, which of the three curves above to build;
    until    : str or None, a local wall-clock time such as "13:00"; bars after it are
               dropped from every session before the curve is built.
    """
    if kind not in ("cidr", "rvol", "ret"):
        raise ValueError(f"kind must be 'cidr', 'rvol' or 'ret', got {kind!r}.")
    days = sessions(bars, min_bars=min_bars)
    if until is not None:
        days = [g[g["ts"].dt.strftime("%H:%M") <= until] for g in days]
        days = [g.reset_index(drop=True) for g in days if len(g) >= 2]
    if not days:
        raise ValueError("no session survived the `min_bars` filter.")
    if n_points is None:
        n_points = int(pd.Series([len(g) for g in days]).mode().iloc[0])
        if kind == "ret":
            n_points = min(_RATE_MAX_POINTS, _RATE_REFINEMENT * n_points + 1)
    grid = np.linspace(0.0, 1.0, int(n_points))

    X = np.empty((len(days), grid.size))
    for i, g in enumerate(days):
        secs = g["ts"].astype("int64").to_numpy() / 1e9
        span = secs[-1] - secs[0]
        s = (secs - secs[0]) / span if span > 0 else np.linspace(0, 1, len(secs))
        logp = np.log(g[price].to_numpy(float))
        if kind == "ret":
            # Differences divided by their own step in s, then held constant across the step
            # they were measured over, so the trapezium integral of the result returns the
            # session total whatever the bar spacing.
            step = np.diff(s)
            rate = np.where(step > 0, np.diff(logp) / np.where(step > 0, step, 1.0), 0.0)
            X[i] = rate[np.clip(np.searchsorted(s, grid, side="left") - 1, 0, len(rate) - 1)]
            continue
        value = (logp - logp[0] if kind == "cidr"
                 else np.sqrt(np.concatenate([[0.0], np.cumsum(np.diff(logp) ** 2)])))
        X[i] = np.interp(grid, s, value)
    return X, grid, np.array([g["session"].iloc[0] for g in days])


def overnight_jumps(bars, min_bars=None, net_of_dividends=True):
    """Overnight log jump for each session, `J_t = log(Open_t) - log(Close_{t-1})`.

    Open and close are the first and last bar of the session, so the jump and the curve come
    from one price series and together reconstruct the open-to-open return exactly. Only
    sessions adjacent in the surviving series are differenced, so a jump spanning a dropped
    half-day is excluded rather than silently stretched over it; `gap_days` says how far
    each one actually reaches.

    Returns a DataFrame indexed by the *opening* session date, with columns jump, prev_close,
    open, div and gap_days (calendar days since the previous session, 3 across a weekend).

    PARAMETERS
    bars             : DataFrame from `load_bars`, restricted to a single ticker;
    min_bars         : int or None, passed to `sessions`;
    net_of_dividends : bool, add the dividend back before differencing. A stock opens lower
                       by roughly the dividend on its ex-date, which is a payment to the
                       holder rather than a price move, so leaving it in puts a spurious
                       negative jump on every ex-date (default True).
    """
    days = sessions(bars, min_bars=min_bars)
    if len(days) < 2:
        raise ValueError("need at least two sessions to form a jump.")

    opens = np.array([g["open"].iloc[0] for g in days], float)
    closes = np.array([g["close"].iloc[-1] for g in days], float)
    divs = np.array([float(g["div"].iloc[0]) if "div" in g else 0.0 for g in days])
    dates = [g["session"].iloc[0] for g in days]

    adj = opens + divs if net_of_dividends else opens
    jump = np.log(adj[1:]) - np.log(closes[:-1])
    gap = np.array([(pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days
                    for i in range(1, len(dates))])
    return pd.DataFrame(
        {"jump": jump, "prev_close": closes[:-1], "open": opens[1:], "div": divs[1:],
         "gap_days": gap},
        index=pd.Index(dates[1:], name="session"),
    )


def session_summary(bars, min_bars=None):
    """Per-session scalars describing the trading day, as candidate covariates.

    Returns a DataFrame indexed by session date, with columns

        ret       open-to-close log return over the session, i.e. `X_t(1)`;
        rvol      realised volatility, the root sum of squared bar-to-bar log returns;
        rng       Parkinson (1980) range, `log(high) - log(low)` over the session;
        volume    total volume, NaN for the index symbols, which do not report one;
        close_loc where the close sat in the session's range, 0 at the low and 1 at the high;
        n_bars    bars the session was built from.

    Realised volatility and the range measure how far the day moved without regard to
    direction, which is the natural thing to reach for once direction itself turns out to be
    unpredictable.

    PARAMETERS
    bars     : DataFrame from `load_bars`, restricted to a single ticker;
    min_bars : int or None, passed to `sessions`.
    """
    rows, dates = [], []
    for g in sessions(bars, min_bars=min_bars):
        logc = np.log(g["close"].to_numpy(float))
        hi, lo = g["high"].to_numpy(float).max(), g["low"].to_numpy(float).min()
        vol = g["volume"].to_numpy(float)
        span = hi - lo
        rows.append({
            "ret": logc[-1] - np.log(float(g["open"].iloc[0])),
            "rvol": float(np.sqrt(np.sum(np.diff(logc) ** 2))),
            "rng": float(np.log(hi) - np.log(lo)),
            "volume": float(np.nansum(vol)) if np.isfinite(vol).any() else np.nan,
            "close_loc": float((g["close"].iloc[-1] - lo) / span) if span > 0 else np.nan,
            "n_bars": len(g),
        })
        dates.append(g["session"].iloc[0])
    return pd.DataFrame(rows, index=pd.Index(dates, name="session"))


def align_curves_to_series(X, dates, series, offset=0):
    """Pair each session's curve with a scalar drawn from `series`.

    With `offset=0` the scalar is the one recorded on the *same* date, which is what a
    cross-market predictor needs: the London session of day t and the New York open of day t
    are the same night. With `offset=1` it is the one recorded on the next session present
    in `dates`, which is what a same-market predictor needs, since a session can only
    forecast the night that follows it.

    A curve with no partner is dropped rather than matched to the nearest available one, so
    no pair ever straddles a missing day.

    Returns X_kept (m, n_points), y (m,) and the (m,) array of predictor session dates.

    PARAMETERS
    X      : (n, n_points) array from `session_curves`;
    dates  : (n,) array of session dates from `session_curves`;
    series : Series or dict keyed by session date, holding the scalar to pair;
    offset : int, 0 to take the scalar from the same date, 1 from the following session.
    """
    if offset not in (0, 1):
        raise ValueError(f"offset must be 0 or 1, got {offset!r}.")
    lookup = dict(series) if not hasattr(series, "index") else dict(zip(series.index, series))

    rows, y, when = [], [], []
    for i, d in enumerate(dates[:len(dates) - offset]):
        key = dates[i + offset]
        if key in lookup and np.isfinite(float(lookup[key])):
            rows.append(i)
            y.append(float(lookup[key]))
            when.append(d)
    if not rows:
        raise ValueError("no curve could be paired with a scalar.")
    return X[np.asarray(rows)], np.asarray(y, float), np.asarray(when)


def align_curves_to_jumps(X, dates, jumps):
    """Pair each session's curve with the jump that follows it.

    The regression is `J_t ~ X_{t-1}`, so row i of the returned design is session i's curve
    and row i of the response is the jump into the *next* session. A session whose successor
    was dropped is dropped too, so no pair straddles a missing day.

    Returns X_prev (m, n_points), y (m,) and the (m,) array of predictor session dates.

    PARAMETERS
    X     : (n, n_points) array from `session_curves`;
    dates : (n,) array of session dates from `session_curves`;
    jumps : DataFrame from `overnight_jumps`, indexed by the opening session.
    """
    return align_curves_to_series(X, dates, jumps["jump"], offset=1)


def _positive(scale):
    """A standard deviation safe to divide by: falls back to 1.0 where the series is constant
    or the value is not finite."""
    scale = float(scale)
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def stack_panel(name="5m", tickers=None, n_points=None, standardise="each", min_bars=None,
                kind="cidr", target="signed"):
    """Build the pooled curve-and-jump design across every ticker in a panel.

    Each ticker is reshaped by `session_curves`, `overnight_jumps` and
    `align_curves_to_jumps`, and the results stacked. Row i is one ticker's curve for one
    session, paired with that ticker's jump into the following session.

    Pooling needs the tickers put on a common scale first -- NVDA moves several times as far
    as SPY in a night, so raw stacking would fit the volatile names and ignore the rest --
    and *which* common scale matters more than it looks:

        "each"   divide the curves and the jumps by their own per-ticker standard deviations.
                 The fitted coefficient is then a standardised one, comparable across
                 tickers whatever the relation between a ticker's intraday range and its
                 overnight gap.
        "jump"   divide both by the jump standard deviation, so the curve and the jump keep
                 their relative size and continue to partition one open-to-open return.
                 This is the honest choice when that partition is the point, and a trap
                 otherwise: `^FTSE`'s opening print is a stale composite that barely moves,
                 so its jump standard deviation is several times too small and dividing its
                 curves by that number inflates them well past every other ticker's.

    Either divisor is one constant per series, so the shape of every curve, and the sign and
    relative size of every jump, survive it exactly.

    Returns a dict with X (m, n_points), y (m,), grid (n_points,), ticker (m,) and date (m,).

    PARAMETERS
    name        : str, a key of `PANELS`;
    tickers     : sequence of str or None, symbols to pool; defaults to those present;
    n_points    : int or None, shared grid size; defaults to the first ticker's modal length;
    standardise : {"each", "jump"} or False, how the tickers are put on a common scale
                  before stacking; see above. Default "each";
    min_bars    : int or None, passed to `sessions`;
    kind        : {"cidr", "rvol"}, which curve to build, passed to `session_curves`;
    target      : {"signed", "absolute"}, whether the response is the jump or its size. The
                  two ask different questions -- which way the market will move overnight,
                  and how far -- and the answers here are not the same.
    """
    if target not in ("signed", "absolute"):
        raise ValueError(f"target must be 'signed' or 'absolute', got {target!r}.")
    if standardise not in ("each", "jump", False, None):
        raise ValueError(f"standardise must be 'each', 'jump' or False, got {standardise!r}.")
    bars = load_bars(name, tickers=tickers)
    present = [t for t in (TICKERS if tickers is None else tickers)
               if t in set(bars["ticker"])]
    if not present:
        raise ValueError("no requested ticker is present in this panel.")

    Xs, ys, tk, dt, grid = [], [], [], [], None
    for sym in present:
        one = bars[bars["ticker"] == sym]
        Xc, g, dates = session_curves(one, n_points=n_points, min_bars=min_bars, kind=kind)
        if grid is None:
            grid, n_points = g, g.size
        jumps = overnight_jumps(one, min_bars=min_bars)
        Xp, y, when = align_curves_to_jumps(Xc, dates, jumps)
        # The response divisor is the *signed* jump's spread either way, so switching the
        # target does not switch the scale each ticker is measured on.
        y_scale = _positive(np.std(y)) if standardise else 1.0
        x_scale = _positive(np.std(Xp)) if standardise == "each" else y_scale
        Xs.append(Xp / x_scale)
        ys.append((np.abs(y) if target == "absolute" else y) / y_scale)
        tk += [sym] * len(y)
        dt += list(when)
    return {"X": np.vstack(Xs), "y": np.concatenate(ys), "grid": grid,
            "ticker": np.array(tk), "date": np.array(dt)}


def date_blocked_folds(dates, n_folds=10):
    """Cross-validation folds for a pooled panel, blocked by calendar date.

    Two things rule out a random split here. Rows from the same day but different tickers
    share that day's market-wide move, so they are near-copies of each other; and rows from
    adjacent days overlap in the volatility regime they were drawn from. Blocking by date
    puts every ticker's row for a given day in the same fold, and makes each fold a
    contiguous stretch of the calendar, so a model is always tested on a period it was not
    fitted on.

    Returns an (n,) integer array of fold labels aligned to `dates`.

    PARAMETERS
    dates   : (n,) array of session dates, as returned by `stack_panel` under "date";
    n_folds : int, number of folds.
    """
    dates = np.asarray(dates)
    unique = np.array(sorted(set(dates.tolist())))
    if len(unique) < n_folds:
        raise ValueError(f"{len(unique)} distinct dates cannot fill {n_folds} folds.")
    block = np.array_split(np.arange(len(unique)), n_folds)
    label = {}
    for f, idx in enumerate(block):
        for i in idx:
            label[unique[i]] = f
    return np.array([label[d] for d in dates.tolist()], dtype=int)


def _main(argv):
    force = "--force" in argv
    print(f"data directory: {paths.DATA_DIR}")
    for name, path in ensure_all(force=force).items():
        df = pd.read_csv(path)
        size = os.path.getsize(path) / 1024
        print(f"  {name:<4} {PANELS[name]['file']:<20} {size:9.1f} KB  "
              f"{len(df):7d} bars  {df['ticker'].nunique()} tickers")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
