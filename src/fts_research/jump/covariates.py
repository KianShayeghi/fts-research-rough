"""The `z_t` block of the jump equation, one column per published claim about overnight moves.

`jump_eq` writes the overnight move as `J_t = a + gamma J_{t-1} + <delta, X_{t-1}> + z_t' k`
and leaves `z_t` to the caller. `notebooks/overnight_jumps.ipynb` finds that the block
matters and the functional term does not, so what goes in it is the empirical question the
equation turns on, and this module is where the candidates are written down.

Every column is a scalar summary of a session that had already closed when the jump formed,
or of a foreign session truncated before the domestic market opened, so the whole design is
measurable at the moment being forecast. That is not a formality. The overnight window is
not simultaneous across exchanges, and a European close is four and a half hours after a New
York open, so a "same-day" foreign return is a look-ahead of most of the session it is
supposed to be forecasting -- the flaw that separates the accuracies reported for overnight
direction in the machine learning literature from those in the finance literature.
`foreign_covariates` takes a wall-clock cutoff rather than a session, and passes it to
`fdaSP.intraday.session_curves`, which renormalises the domain over what is left.

What each column is, and why it is here.

`late`     The return over the closing fraction of the session. Boyarchenko, Larsen and
           Whelan (2023) trace the overnight drift to the order imbalance intermediaries
           absorb at the close, measured over 15:15 to 16:15, and Boyarchenko, Larsen and
           Whelan (2026) attribute the drift's disappearance since 2021 to the compression
           of that imbalance's dispersion; the closing return is what a bar panel can see of
           it. The same window is the predictor in Gao, Han, Li and Zhou's (2018) intraday
           momentum and in the cross-market overnight momentum of Xu et al. (2025).
           A price-pressure account predicts a negative coefficient, a momentum account a
           positive one, and the sign is the test.
`early`    The return over the opening fraction, the other half of Gao et al.'s pair.
`ret`      The whole session, which is `X_{t-1}(1)`. Lou, Polk and Skouras (2019) find
           overnight and intraday returns continue within their own period and reverse
           across periods, which makes this the reversal term.
`close_loc` Where the close sat in the session's range, on [0, 1]. A bounded, scale-free
           reading of the same closing pressure `late` measures in returns, so the two
           disagree exactly where a large closing move was not a large move relative to the
           day.
`rvol`     Realised volatility, the size of the session. `notebooks/overnight_jumps.ipynb`
           finds the size of the jump forecastable where its sign is not, and this is the
           regressor that finding points at.
`sjv`      Signed jump variation, the difference of the two realised semivariances of
           Barndorff-Nielsen, Kinnebrock and Shephard (2010). Patton and Sheppard (2015)
           show that future volatility responds to the sign of past variation and not only
           to its size, so this separates a session that fell hard from one that rose hard
           at equal realised volatility.
`rng`      The high-low range, Parkinson's (1980) volatility estimator, which is robust to
           the bar spacing in a way a sum of squared returns is not.
`vol_rel`  Volume against its own trailing mean. Berkman, Koch, Tuttle and Zhang (2012) find
           the overnight-return pattern concentrated in stocks that have just drawn retail
           attention, and abnormal volume is the attention proxy that a price panel carries.
`gap_days` Calendar days the jump spans, three across a weekend. A longer close is more
           hours in which news can arrive.
`jump_lag`, `abs_jump_lag`
           The AR(1) term of the equation and its size. Kept here as well so that a screen
           of the covariates prices them against the persistence they have to beat.

References
`docs/overnight-jumps-literature.md` reads these together, with what each one does and does not
say about the sign of the overnight move, and with the two that turn out to survive.

Barndorff-Nielsen, O. E., Kinnebrock, S. and Shephard, N. (2010). Measuring downside risk:
realised semivariance. In T. Bollerslev, J. Russell and M. Watson (eds.), Volatility and Time
Series Econometrics, 117-136. Oxford University Press.
Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and
powerful approach to multiple testing. Journal of the Royal Statistical Society, Series B
57(1), 289-300.
Berkman, H., Koch, P. D., Tuttle, L. and Zhang, Y. J. (2012). Paying attention: overnight
returns and the hidden cost of buying at the open. Journal of Financial and Quantitative
Analysis 47(4), 715-741.
Boyarchenko, N., Larsen, L. C. and Whelan, P. (2023). The overnight drift. Review of
Financial Studies 36(9), 3502-3547.
Boyarchenko, N., Larsen, L. C. and Whelan, P. (2026). The disappearing overnight drift.
Liberty Street Economics, Federal Reserve Bank of New York, 1 July 2026.
Gao, L., Han, Y., Li, S. Z. and Zhou, G. (2018). Market intraday momentum. Journal of
Financial Economics 129(2), 394-414.
Hamao, Y., Masulis, R. W. and Ng, V. (1990). Correlations in price changes and volatility
across international stock markets. Review of Financial Studies 3(2), 281-307.
Lou, D., Polk, C. and Skouras, S. (2019). A tug of war: overnight versus intraday expected
returns. Journal of Financial Economics 134(1), 192-213.
Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of
return. Journal of Business 53(1), 61-65.
Patton, A. J. and Sheppard, K. (2015). Good volatility, bad volatility: signed jumps and the
persistence of volatility. Review of Economics and Statistics 97(3), 683-697.
Xu, D., Li, B., Singh, T., Chen, X. and Li, J. (2025). Cross-market overnight time-series
momentum. Journal of International Financial Markets, Institutions and Money.
"""

import numpy as np
import pandas as pd

from ..fdaSP import intraday

__all__ = [
    "SOURCE",
    "session_covariates",
    "foreign_covariates",
    "jump_design",
    "panel_design",
    "benjamini_hochberg",
]

# The claim each column is testing, and where it comes from. Read by
# `notebooks/jump_covariates.ipynb` to label its tables, and by `tests/test_covariates.py`,
# which holds it against the columns `session_covariates` and `jump_design` actually build.
SOURCE = {
    "ret": "Lou, Polk & Skouras (2019): cross-period reversal",
    "late": "Boyarchenko, Larsen & Whelan (2023): closing imbalance, absorbed overnight",
    "early": "Gao, Han, Li & Zhou (2018): intraday momentum",
    "close_loc": "closing pressure, as a position in the session's range",
    "rvol": "realised volatility: the size of the session",
    "sjv": "Patton & Sheppard (2015): signed jump variation",
    "rng": "Parkinson (1980): the high-low range",
    "vol_rel": "Berkman, Koch, Tuttle & Zhang (2012): attention, as abnormal volume",
    "gap_days": "hours the market was shut, and so hours for news to arrive",
    "jump_lag": "the equation's own AR(1) term",
    "abs_jump_lag": "its size, which `overnight_jumps.ipynb` finds is the forecastable part",
    "foreign_ret": "Hamao, Masulis & Ng (1990): a foreign session that trades through the "
                   "domestic night, truncated before the domestic open",
    "foreign_rvol": "the same session's realised volatility",
}


def _tail_head(logp, fraction):
    """Return over the last and the first `fraction` of a session's bars. Returns two
    floats.

    At least one bar goes into each, so on a seven-bar panel a fifth of the session is the
    final bar rather than nothing.
    """
    n_bars = logp.size
    width = max(1, int(round(float(fraction) * (n_bars - 1))))
    return float(logp[-1] - logp[-1 - width]), float(logp[width] - logp[0])


def session_covariates(bars, fraction=1 / 6, volume_window=20, min_bars=None):
    """Scalar summaries of each session, as candidate predictors of the jump that follows.

    Extends `fdaSP.intraday.session_summary` -- which supplies `ret`, `rvol`, `rng`,
    `volume`, `close_loc` and `n_bars` under the definitions the rest of the repository uses
    -- with the columns the overnight literature asks for: `late` and `early`, the returns
    over the closing and opening fractions of the session; `rsv_up` and `rsv_down`, the two
    realised semivariances, and their difference `sjv`; `logvol`; and `vol_rel`, volume over
    its own trailing mean. Returns a DataFrame indexed by session date.

    Every column describes a session that has closed, so every one of them is known before
    the jump that opens the next session. `jump_design` is what does the lagging.

    PARAMETERS
    bars          : DataFrame from `fdaSP.intraday.load_bars`, restricted to one ticker;
    fraction      : float in (0, 1], the share of the session `late` and `early` cover; a
                    sixth is close to the final hour of a six-and-a-half-hour session, which
                    is the window the closing-imbalance evidence is measured over;
    volume_window : int, sessions in the trailing mean `vol_rel` is taken against, which
                    excludes the session itself so that a volume spike does not normalise
                    itself away; the first five sessions have no mean to be taken against,
                    or the first `volume_window` of them where that is fewer, and `vol_rel`
                    is missing there rather than taken against one or two days;
    min_bars      : int or None, passed to `fdaSP.intraday.sessions`.
    """
    summary = intraday.session_summary(bars, min_bars=min_bars)
    days = intraday.sessions(bars, min_bars=min_bars)

    rows = []
    for day in days:
        logp = np.log(day["close"].to_numpy(float))
        late, early = _tail_head(logp, fraction)
        step = np.diff(logp)
        rows.append({
            "late": late,
            "early": early,
            "rsv_up": float(np.sum(step[step > 0] ** 2)),
            "rsv_down": float(np.sum(step[step < 0] ** 2)),
        })
    extra = pd.DataFrame(rows, index=summary.index)
    extra["sjv"] = extra["rsv_up"] - extra["rsv_down"]

    volume = summary["volume"]
    window = int(volume_window)
    trailing = volume.shift(1).rolling(window, min_periods=min(5, window)).mean()
    extra["logvol"] = np.log(volume.where(volume > 0))
    extra["vol_rel"] = volume / trailing
    return summary.join(extra)


def foreign_covariates(bars, until, min_bars=None, prefix="foreign"):
    """A foreign session, truncated at a wall-clock time, as a predictor of the domestic open.

    A market that trades through the domestic night sees the news the domestic open will
    price, which is Hamao, Masulis and Ng's (1990) transmission and the reason
    `notebooks/overnight_jumps.ipynb` finds the London morning forecasts the New York open
    where nothing in New York's own history does. `until` is what makes it a forecast: bars
    after it are dropped before the return is taken, so a cutoff before the domestic open
    leaves an information set that existed at the moment being forecast. Passing None keeps
    the whole session, which for a European market is a look-ahead into the domestic
    afternoon and is offered only so that the size of that look-ahead can be measured.

    Returns a DataFrame indexed by the foreign session's date, with `{prefix}_ret` the
    truncated log return and `{prefix}_rvol` its realised volatility. The index is the
    calendar date the domestic session of the same date opens on, so it pairs with
    `jump_design` under `fdaSP.intraday.align_curves_to_series`' same-date convention.

    PARAMETERS
    bars     : DataFrame from `fdaSP.intraday.load_bars`, restricted to the foreign ticker;
    until    : str or None, a local wall-clock time such as "13:00";
    min_bars : int or None, passed to `fdaSP.intraday.sessions`;
    prefix   : str, the stem of the two column names.
    """
    days = intraday.sessions(bars, min_bars=min_bars)
    if until is not None:
        days = [g[g["ts"].dt.strftime("%H:%M") <= until] for g in days]
        days = [g.reset_index(drop=True) for g in days if len(g) >= 2]
    if not days:
        raise ValueError(f"no session survived the cutoff {until!r}.")

    rows, index = [], []
    for day in days:
        logp = np.log(day["close"].to_numpy(float))
        rows.append({f"{prefix}_ret": float(logp[-1] - logp[0]),
                     f"{prefix}_rvol": float(np.sqrt(np.sum(np.diff(logp) ** 2)))})
        index.append(day["session"].iloc[0])
    return pd.DataFrame(rows, index=pd.Index(index, name="session"))


def jump_design(bars, foreign=None, fraction=1 / 6, volume_window=20, min_bars=None,
                net_of_dividends=True):
    """The jump and everything known before it, one row per overnight window.

    The row for the jump that opens session `t` carries the covariates of session `t - 1`,
    the jump that opened session `t - 1`, and any foreign columns recorded on date `t`
    itself, which is the same night. Getting that wrong is not a small error: putting
    session `t`'s own summaries in the row would fit a model reading the session it is meant
    to forecast, and would fit it better. Rows with a missing value are dropped, which costs
    the first `volume_window` sessions to the trailing volume mean.

    Returns a DataFrame indexed by the opening session's date, with `jump` the target,
    `abs_jump` its size, and the predictors named in `SOURCE`.

    PARAMETERS
    bars             : DataFrame from `fdaSP.intraday.load_bars`, one ticker;
    foreign          : DataFrame or None, as `foreign_covariates` returns, joined on date;
    fraction         : float, passed to `session_covariates`;
    volume_window    : int, passed to `session_covariates`;
    min_bars         : int or None, passed to `fdaSP.intraday.sessions`;
    net_of_dividends : bool, passed to `fdaSP.intraday.overnight_jumps`.
    """
    covariates = session_covariates(bars, fraction=fraction, volume_window=volume_window,
                                    min_bars=min_bars)
    jumps = intraday.overnight_jumps(bars, min_bars=min_bars,
                                     net_of_dividends=net_of_dividends)

    dates = list(covariates.index)
    if list(jumps.index) != dates[1:]:
        raise ValueError("the jump index does not follow the session index; the two were "
                         "built from different `min_bars`.")

    columns = ["ret", "late", "early", "close_loc", "rvol", "sjv", "rng", "vol_rel"]
    lagged = covariates.iloc[:-1][columns].copy()
    lagged.index = pd.Index(dates[1:], name="session")

    design = lagged
    design.insert(0, "jump", jumps["jump"].to_numpy())
    design.insert(1, "abs_jump", np.abs(jumps["jump"].to_numpy()))
    design["gap_days"] = jumps["gap_days"].to_numpy(float)
    design["jump_lag"] = jumps["jump"].shift(1).to_numpy()
    design["abs_jump_lag"] = np.abs(design["jump_lag"])
    if foreign is not None:
        design = design.join(foreign, how="left")
    return design.dropna()


def panel_design(name="1h", tickers=None, foreign_ticker="^FTSE", until="13:00",
                 fraction=1 / 6, volume_window=20, min_bars=None):
    """`jump_design` for every ticker of a panel, stacked, with the same foreign columns.

    Returns a DataFrame with `ticker` and `session` columns alongside the design, in
    (ticker, date) order. The foreign columns are identical across tickers by construction,
    which is the point -- one night's news reaches every instrument -- and is also why a
    random cross-validation split of this frame leaks: every ticker contributes a row on each
    date and shares that date's market-wide move. Split on `session` instead, with
    `fdaSP.intraday.date_blocked_folds`.

    PARAMETERS
    name           : str, which panel `fdaSP.intraday.load_bars` should serve;
    tickers        : sequence or None, the domestic tickers; every one but the foreign
                     ticker when None;
    foreign_ticker : str or None, the market trading through the domestic night;
    until          : str or None, the wall-clock cutoff passed to `foreign_covariates`;
    fraction       : float, passed to `session_covariates`;
    volume_window  : int, passed to `session_covariates`;
    min_bars       : int or None, passed to `fdaSP.intraday.sessions`.
    """
    bars = intraday.load_bars(name)
    available = list(dict.fromkeys(bars["ticker"]))
    if tickers is None:
        tickers = [t for t in available if t != foreign_ticker]

    foreign = None
    if foreign_ticker is not None:
        foreign = foreign_covariates(bars[bars["ticker"] == foreign_ticker], until=until,
                                     min_bars=min_bars)

    frames = []
    for ticker in tickers:
        one = jump_design(bars[bars["ticker"] == ticker], foreign=foreign,
                          fraction=fraction, volume_window=volume_window, min_bars=min_bars)
        one = one.reset_index()
        one.insert(0, "ticker", ticker)
        frames.append(one)
    return pd.concat(frames, ignore_index=True)


def benjamini_hochberg(pvalues, alpha=0.05):
    """Which of many tests survive at a false discovery rate of `alpha`.

    A screen over nine instruments and a dozen covariates is a hundred-odd tests, of which
    five would clear the five per cent level by chance, so an uncorrected table of stars says
    nothing. Benjamini and Hochberg's step-up procedure rejects the `k` smallest p-values
    where `k` is the largest index with `p_(k) <= k alpha / m`, which controls the expected
    share of rejections that are false rather than the chance of any false rejection at all
    -- the right target for a screen, whose purpose is to produce candidates rather than to
    settle one hypothesis.

    Returns a dict with `reject`, a boolean array in the order given; `qvalues`, the
    monotone adjusted p-values; `threshold`, the largest p-value rejected, or 0 where none
    is; and `n_reject`.

    PARAMETERS
    pvalues : array of p-values, in any order, NaNs treated as ones;
    alpha   : float, the false discovery rate to control.
    """
    p = np.asarray(pvalues, float).ravel()
    p = np.where(np.isnan(p), 1.0, p)
    m = p.size
    if m == 0:
        return {"reject": np.zeros(0, bool), "qvalues": np.zeros(0), "threshold": 0.0,
                "n_reject": 0}

    order = np.argsort(p)
    ranked = p[order]
    scaled = ranked * m / np.arange(1, m + 1)
    qsorted = np.minimum.accumulate(scaled[::-1])[::-1]      # monotone from the top down
    qvalues = np.empty(m)
    qvalues[order] = np.minimum(qsorted, 1.0)

    below = np.flatnonzero(ranked <= np.arange(1, m + 1) * float(alpha) / m)
    threshold = float(ranked[below[-1]]) if below.size else 0.0
    reject = p <= threshold
    return {"reject": reject, "qvalues": qvalues, "threshold": threshold,
            "n_reject": int(reject.sum())}
