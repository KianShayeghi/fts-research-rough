"""Scoring a curve forecaster one step ahead, out of sample, on an expanding window.

This is the protocol behind Shang's (2017a) accuracy tables and the only one that settles
anything on real data: at each split the model is re-estimated on the sessions up to that
point and used to predict the next one, whose realised curve is then the holdout. Nothing is
fitted on a curve it is later scored against, and the number of sessions available grows the
way it does in use.

Three measures, all of them Shang's, all reported both averaged over the holdout and profiled
along the intraday grid, because *when* in the session a forecaster goes wrong is the part a
grand mean throws away:

    MAFE_j = (1/q) sum_i |X_i(t_j) - Xhat_i(t_j)|
    MSFE_j = (1/q) sum_i [X_i(t_j) - Xhat_i(t_j)]^2
    S_a[l, u; x] = (u - l) + (2/a)(l - x) 1{x < l} + (2/a)(x - u) 1{x > u}

The third is Gneiting & Raftery's (2007) interval score, which is the one that scores a
prediction band rather than a point: it pays for the width of the interval and is charged for
each miss in proportion to how far outside the realised curve fell, so a band cannot buy a
better score by being wide or by being narrow. Empirical coverage is reported alongside it,
since the two answer different questions and a band can be well covered and badly scored.

Lower is better for all three. `rolling_origin` returns the forecasts themselves as well as
the scores, so a measure this module does not compute can still be computed from its output.

It also scores a fourth thing, which is not a forecaster: `method="mean"` predicts every session
by the mean of the sessions before it, and bands it by their empirical quantiles. That is the
benchmark `far.far_est.rolling_forecast` insists on, and it is here for the same reason -- an
accuracy table with no floor in it cannot say whether a method has found anything. A forecaster
that does not beat the flat forecast has not.

The whole evaluation takes minutes, so `cached_rolling` keeps its forecasts under
`artifacts/cache/` in the same way `bench.study.cached_study` keeps the comparison study, and
`scripts/make_ftsa_cache.py` rebuilds them headlessly. `CONFIG` is the design both the cache
and `notebooks/shang_replication.ipynb` read, so the notebook cannot drift from the cache it
opens.
"""

import os

import numpy as np
import pandas as pd

from ..core.paths import cache_path
from ..fdaSP import intraday as itd
from . import forecast as fc

__all__ = [
    "CONFIG",
    "ARTIFACT",
    "interval_score",
    "replication_curves",
    "panel_curves",
    "rolling_origin",
    "summarise",
    "cached_rolling",
    "cached_panel",
]

ARTIFACT = "shang_rolling.csv"
PANEL_ARTIFACT = "shang_rolling_panel.csv"

# The replication design. `n_points` is the 79-mark five-minute grid Shang samples the regular
# session on, one of whose marks is the opening anchor the transform sets to zero; `init_frac`
# leaves 50 sessions to score on, which is the holdout the accuracy tables are averaged over.
CONFIG = {
    "n_points": 79,
    "scale": 100.0,
    "var_threshold": 0.90,
    "alpha": 0.20,
    "n_boot": 400,
    "init_frac": 0.60,
    "methods": ("arima", "far", "mean"),
    "seed": 20250616,
    "panel": "5m",
}


def interval_score(lower, upper, actual, alpha):
    """Gneiting & Raftery's (2007) interval score of a nominal 100(1 - `alpha`)% band, at each
    point. Returns an array shaped like `actual`, lower being better.

    PARAMETERS
    lower, upper : arrays, the band;
    actual       : array, what was realised;
    alpha        : float, one minus the nominal coverage, so 0.2 for an 80% interval.
    """
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    actual = np.asarray(actual, float)
    below = (actual < lower).astype(float)
    above = (actual > upper).astype(float)
    return ((upper - lower)
            + (2.0 / alpha) * (lower - actual) * below
            + (2.0 / alpha) * (actual - upper) * above)


def replication_curves(n_points=None, scale=None):
    """The curves Shang's design asks for, built from the committed 2017 S&P 500 panel.

    Returns X (n_sessions, n_points - 1) the cumulative intraday return curves in per cent,
    grid (n_points - 1,) the fraction of the session elapsed, and dates (n_sessions,).

    The transform is `fdaSP.intraday.session_curves` under `kind="cidr"`, which is Shang's
    equation (1): every curve is anchored at the session's opening log price. That anchor is
    the first grid mark and is identically zero for every session, carrying no information and
    no variance, so it is dropped here as the paper drops it -- and it is worth being explicit
    that dropping it is not a smoothing step. Anchoring is what erases the overnight move from
    the data altogether.

    PARAMETERS
    n_points : int or None, marks on the intraday grid before the anchor is dropped; defaults
               to `CONFIG["n_points"]`;
    scale    : float or None, multiplier on the log returns; defaults to `CONFIG["scale"]`,
               which is Shang's, and puts the curves in per cent.
    """
    n_points = CONFIG["n_points"] if n_points is None else int(n_points)
    scale = CONFIG["scale"] if scale is None else float(scale)
    bars = itd.load_spx_2017()
    X, grid, dates = itd.session_curves(bars, n_points=n_points, min_bars=1, kind="cidr")
    return scale * X[:, 1:], grid[1:], dates


def panel_curves(ticker, name=None, n_points=None, scale=None):
    """The same curves as `replication_curves`, for one instrument of a cached snapshot panel.

    Returns X (n_sessions, n_points - 1), grid (n_points - 1,) and dates (n_sessions,).

    The 2017 window is one window of one index, so a conclusion drawn on it is a conclusion
    about that window until it has been asked of something else. These panels are the something
    else: nine instruments over a recent quarter, at the same five-minute resolution, built by
    the same transform. They are shorter -- sixty sessions against a hundred and twenty-five --
    so each one on its own settles less, and it is the agreement across the nine that carries
    the weight.

    PARAMETERS
    ticker   : str, one of `fdaSP.intraday.TICKERS`;
    name     : str or None, a key of `fdaSP.intraday.PANELS`; `CONFIG` by default;
    n_points : int or None, marks on the intraday grid before the anchor is dropped; the modal
               session length of that instrument by default, which interpolates nothing;
    scale    : float or None, multiplier on the log returns; `CONFIG` by default.
    """
    name = CONFIG["panel"] if name is None else str(name)
    scale = CONFIG["scale"] if scale is None else float(scale)
    bars = itd.load_bars(name, tickers=[ticker])
    if bars.empty:
        raise KeyError(f"{ticker!r} is not in the {name!r} panel.")
    X, grid, dates = itd.session_curves(bars, n_points=n_points, kind="cidr")
    return scale * X[:, 1:], grid[1:], dates


def rolling_origin(X, grid, method="arima", var_threshold=None, alpha=None, n_boot=None,
                   init_frac=None, ridge=0.0, rng=None, verbose=False):
    """Expanding-window, one-step-ahead evaluation of `ftsa.forecast` on curves in time order.

    Returns a dict with actual, point, lower and upper each (q, N) over the `q` holdout
    sessions, the per-point profiles mafe_by_t, msfe_by_t, iscore_by_t and coverage_by_t each
    (N,), their grand means mafe, msfe, iscore and coverage, the index of the first holdout
    session under `start`, the components retained at each split under `n_components`, and
    `method`.

    PARAMETERS
    X             : (n, N) array of curves in time order, row 0 earliest;
    grid          : (N,) array, the domain;
    method        : {"arima", "far", "mean"}, the first two passed to
                    `ftsa.forecast.forecast_scores`, the third the flat-forecast benchmark;
    var_threshold : float or None, passed to `ftsa.forecast.decompose`; `CONFIG` by default;
    alpha         : float or None, one minus the nominal coverage of the scored band;
    n_boot        : int or None, bootstrap replicates at each split;
    init_frac     : float or None, fraction of the sample the first training window holds;
    ridge         : float, passed through under `method="far"`;
    rng           : int, Generator or None, seed for the bootstrap;
    verbose       : bool, report progress as the splits go by.
    """
    X = np.atleast_2d(np.asarray(X, float))
    grid = np.asarray(grid, float)
    var_threshold = CONFIG["var_threshold"] if var_threshold is None else float(var_threshold)
    alpha = CONFIG["alpha"] if alpha is None else float(alpha)
    n_boot = CONFIG["n_boot"] if n_boot is None else int(n_boot)
    init_frac = CONFIG["init_frac"] if init_frac is None else float(init_frac)
    rng = np.random.default_rng(rng)

    n = X.shape[0]
    start = max(int(np.floor(init_frac * n)), 10)
    if start >= n:
        raise ValueError(f"init_frac leaves no session to score on (n = {n}).")

    actual, point, lower, upper, retained = [], [], [], [], []
    for s in range(start, n):
        if method == "mean":
            # The benchmark: predict the mean of what has been seen, and band it by the
            # empirical quantiles of the same sessions. No model, and nothing to fit.
            centre = X[:s].mean(axis=0)
            lo = np.quantile(X[:s], alpha / 2.0, axis=0)
            hi = np.quantile(X[:s], 1.0 - alpha / 2.0, axis=0)
            kept = 0
        else:
            dec = fc.decompose(X[:s], grid, var_threshold=var_threshold)
            band = fc.prediction_bands(dec, h=1, levels=(1.0 - alpha,), n_boot=n_boot,
                                       method=method, ridge=ridge, rng=rng)
            lo, hi = (side[0] for side in band["bands"][1.0 - alpha])
            centre = band["center"][0]
            kept = dec["n_components"]
        actual.append(X[s])
        point.append(centre)
        lower.append(lo)
        upper.append(hi)
        retained.append(kept)
        if verbose and (s - start) % 10 == 0:
            print(f"  {method}: split {s - start + 1} of {n - start} (train {s} sessions)")

    actual = np.vstack(actual)
    point = np.vstack(point)
    lower = np.vstack(lower)
    upper = np.vstack(upper)

    error = actual - point
    mafe_by_t = np.mean(np.abs(error), axis=0)
    msfe_by_t = np.mean(error ** 2, axis=0)
    iscore_by_t = np.mean(interval_score(lower, upper, actual, alpha), axis=0)
    coverage_by_t = np.mean((actual >= lower) & (actual <= upper), axis=0)

    return {"actual": actual, "point": point, "lower": lower, "upper": upper,
            "mafe_by_t": mafe_by_t, "msfe_by_t": msfe_by_t, "iscore_by_t": iscore_by_t,
            "coverage_by_t": coverage_by_t, "mafe": float(mafe_by_t.mean()),
            "msfe": float(msfe_by_t.mean()), "iscore": float(iscore_by_t.mean()),
            "coverage": float(coverage_by_t.mean()), "start": start,
            "n_components": np.array(retained), "method": method, "alpha": alpha}


def summarise(df, alpha=None, by="method"):
    """Reduce cached forecasts to one row per group, with mafe, msfe, iscore, coverage and
    n_splits. Returns a DataFrame indexed by `by`.

    The cache stores the forecasts rather than the scores, so every measure is recomputed here
    from what was actually predicted; a measure this repository has not thought of yet can be
    added without re-running anything.

    PARAMETERS
    df    : DataFrame from `cached_rolling` or `cached_panel`;
    alpha : float or None, one minus the nominal coverage the bands were built at; `CONFIG` by
            default;
    by    : str or list of str, the columns to group on -- "method" for the replication,
            ["ticker", "method"] for the panel.
    """
    alpha = CONFIG["alpha"] if alpha is None else float(alpha)
    rows = []
    keys = [by] if isinstance(by, str) else list(by)
    for key, part in df.groupby(keys if len(keys) > 1 else keys[0], sort=True):
        error = part["actual"].to_numpy() - part["point"].to_numpy()
        score = interval_score(part["lower"].to_numpy(), part["upper"].to_numpy(),
                               part["actual"].to_numpy(), alpha)
        covered = (part["actual"] >= part["lower"]) & (part["actual"] <= part["upper"])
        label = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(keys, label))
                    | {"mafe": float(np.mean(np.abs(error))),
                       "msfe": float(np.mean(error ** 2)),
                       "iscore": float(np.mean(score)),
                       "coverage": float(np.mean(covered)),
                       "n_splits": int(part["session"].nunique())})
    return pd.DataFrame(rows).set_index(keys)


# Curves are in per cent and the prices they come from are quantised at about 4e-3 of one, so
# six decimals is already two orders finer than the data and the rest is noise on disk.
_DECIMALS = 6
_GRID_DECIMALS = 9


def _records(result, dates, grid):
    """One row per (holdout session, grid point) of a `rolling_origin` result."""
    sessions = np.asarray(dates)[result["start"]:]
    q, N = result["actual"].shape
    return pd.DataFrame({
        "method": result["method"],
        "session": np.repeat([str(d) for d in sessions], N),
        "t": np.round(np.tile(grid, q), _GRID_DECIMALS),
    } | {name: np.round(result[name].ravel(), _DECIMALS)
         for name in ("actual", "point", "lower", "upper")})


def cached_panel(name=PANEL_ARTIFACT, refresh=False, tickers=None, methods=None, **kwargs):
    """The same evaluation across every instrument of a cached snapshot panel, read from
    `artifacts/cache/` if it is there and computed and cached if not. Returns a long DataFrame
    with ticker, method, session, t, actual, point, lower and upper.

    This is the robustness half of the replication. The 2017 window settles what happens on
    that window; nine instruments over a recent quarter settle whether it was the window.

    PARAMETERS
    name     : str, file name under `artifacts/cache/`;
    refresh  : bool, recompute and overwrite even where the cache exists;
    tickers  : sequence of str or None, instruments to score; `fdaSP.intraday.TICKERS` by
               default;
    methods  : sequence of str or None, forecasters to score; `CONFIG` by default;
    **kwargs : forwarded to `rolling_origin`.
    """
    path = cache_path(name)
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)

    methods = CONFIG["methods"] if methods is None else tuple(methods)
    parts = []
    for ticker in (itd.TICKERS if tickers is None else tuple(tickers)):
        X, grid, dates = panel_curves(ticker)
        for i, method in enumerate(methods):
            result = rolling_origin(X, grid, method=method, rng=CONFIG["seed"] + i, **kwargs)
            record = _records(result, dates, grid)
            record.insert(0, "ticker", ticker)
            parts.append(record)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def cached_rolling(name=ARTIFACT, refresh=False, methods=None, **kwargs):
    """The rolling-origin forecasts, read from `artifacts/cache/` if they are there and
    computed and cached if not. Returns a long DataFrame with method, session, t, actual,
    point, lower and upper.

    Every score in `notebooks/shang_replication.ipynb` is derived from this one file, so the
    two forecasters are always compared on the same splits. The design is `CONFIG` and the
    bootstrap is seeded from it, so a rebuild on the same configuration reproduces the cache.

    PARAMETERS
    name     : str, file name under `artifacts/cache/`;
    refresh  : bool, recompute and overwrite even where the cache exists;
    methods  : sequence of str or None, forecasters to score; `CONFIG` by default;
    **kwargs : forwarded to `rolling_origin`.
    """
    path = cache_path(name)
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)

    X, grid, dates = replication_curves()
    methods = CONFIG["methods"] if methods is None else tuple(methods)
    parts = []
    for i, method in enumerate(methods):
        result = rolling_origin(X, grid, method=method, rng=CONFIG["seed"] + i, **kwargs)
        parts.append(_records(result, dates, grid))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(path, index=False)
    return out
