"""Revising a day's forecast once part of that day has been observed.

The forecaster in `forecast.py` speaks at the close of day `n` and says nothing afterwards. By
mid-session on day `n + 1` the first `m` of its `N` points have been seen, and the question is
what the rest of the day looks like given them. Shang & Hyndman (2011) call this dynamic
updating, and both of Shang's 2017 papers pair it with the forecaster this package implements.

Write the grid as an observed block `o = (s_1, ..., s_m)` and an unobserved block
`u = (s_{m+1}, ..., s_N)`, and the decomposition of the historical curves as
`X_t = mu + sum_k psi_{t,k} phi_k`. Every method below is a different answer to the same
question -- which score vector `b` to reconstruct the unobserved block from:

`ts`     the day-ahead forecast, unchanged. `b` is the time series forecast of the scores and
         the observed block is thrown away. This is the thing updating has to beat.
`ols`    regress what was observed onto the same block of the components,
         `b = argmin ||X_o - mu_o - Phi_o' b||^2`, and reconstruct `u` from it. Uses the new
         information and nothing else; needs `m >= M` and is noisy when `m` is close to it.
`ridge`  the same, with `+ lam ||b||^2`, so `b` is pulled toward zero, i.e. toward the mean
         curve.
`pls`    the same, with `+ lam ||b - b_ts||^2`, so `b` is pulled toward the time series
         forecast rather than toward the mean. This is Shang & Hyndman's penalised least
         squares update and it is the one that interpolates between the two sources: `lam = 0`
         is `ols`, `lam -> inf` is `ts`.
`block`  block moving, which uses no regression at all. The day boundary is rotated to the
         update point, so the `n` historical days and the observed part of day `n + 1` become
         `n` *complete* curves on the rotated domain; forecasting the next one and reading off
         its leading `N - m` points gives the remainder of day `n + 1`. The observed block
         enters as the tail of the last rotated curve rather than as a regression design.

And two that fit nothing, which are what the others have to be worth more than:

`hold`   carry the last observed value forward, `X(s) = X(s_m)` for `s` in `u`. On a curve
         that is already a cumulative sum this is the martingale, and it is the benchmark that
         matters: a cumulative return whose best forecast is its current value is what an
         efficient market looks like.
`mean`   the mean curve, shifted to meet the last observed value:
         `X(s) = X(s_m) + mu(s) - mu(s_m)`.

`hold` is the reason a cumulative curve needs care here. Most of `X_{n+1}(s)` for `s` in `u` is
`X_{n+1}(s_m)`, which is *observed*, so any method that reproduces the level scores well
whether or not it has forecast anything. Comparing against `hold` is what separates the two,
and `rolling_update` reports it alongside everything else for that reason: the out-of-sample
R-squared against `hold` is exactly the R-squared for predicting the *increment*
`X(s) - X(s_m)`, which is the part not already known.

A method can then fail in two ways, and `anchor` tells them apart. It can put the curve at the
wrong level, missing `X(s_m)` even though `X(s_m)` was observed, or it can get the level right
and the shape of the remainder wrong. Anchoring subtracts the first: the prediction is shifted
so that the method's own fitted value at the last observed point equals what was seen there,
leaving only the shape. `block` cannot be anchored, having no fitted value at that point, and
`hold` and `mean` are anchored already by construction.

Bands come from the same place as the point forecasts: the in-sample update errors on the
training days, resampled. `in_sample_updates` computes them for every training day at once,
which is also what `select_lambda` minimises over.

`rolling_update` sweeps the whole cross of update points, methods and anchorings; `records` and
`profiles` reduce it to the per-session sums a paired test needs and the per-point averages a
figure needs, and those are what `cached_updating` commits.
"""

import os

import numpy as np
import pandas as pd

from ..core.paths import cache_path
from . import evaluation as ev
from . import forecast as fc

__all__ = [
    "METHODS",
    "FITTED",
    "LAMBDAS",
    "UPDATE_FRACTIONS",
    "ARTIFACT",
    "split_index",
    "update_scores",
    "in_sample_updates",
    "select_lambda",
    "block_moving",
    "update_curve",
    "rolling_update",
    "records",
    "profiles",
    "summarise_updates",
    "cached_updating",
    "cached_profiles",
]

# Every method `update_curve` accepts, in the order they are reported.
METHODS = ("ts", "ols", "ridge", "pls", "block", "hold", "mean")

# The ones that estimate something, as against the two benchmarks and the un-updated forecast.
FITTED = ("ols", "ridge", "pls", "block")

# Penalties are quoted relative to the mean diagonal of the observed-block Gram matrix, so the
# grid means the same thing at every update point and on every instrument.
LAMBDAS = tuple(np.logspace(-4.0, 3.0, 29))

# Where in the session to update, as a fraction elapsed. Shang tabulates accuracy against this.
UPDATE_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

ARTIFACT = "shang_updating.csv"
PROFILE_ARTIFACT = "shang_updating_profiles.csv"


def split_index(grid, fraction):
    """Number of grid points observed by `fraction` of the way through the session.

    Returns an int `m` with `1 <= m <= len(grid) - 1`, so both blocks are non-empty however
    extreme the fraction.

    PARAMETERS
    grid     : (N,) array, the domain;
    fraction : float in (0, 1), the point of the session the update is made at.
    """
    grid = np.asarray(grid, float)
    m = int(np.searchsorted(grid, float(fraction), side="right"))
    return int(np.clip(m, 1, grid.size - 1))


def _gram(phi_o):
    """`Phi_o Phi_o'` and the scale penalties are quoted against, its mean diagonal."""
    G = phi_o @ phi_o.T
    return G, float(np.mean(np.diag(G)))


def update_scores(dec, head, method="pls", lam=1.0, beta_ts=None):
    """Score vector to reconstruct the unobserved block from. Returns an (M,) array.

    PARAMETERS
    dec     : dict from `ftsa.forecast.decompose`, fitted on the historical curves;
    head    : (m,) array, the observed part of the new curve;
    method  : {"ts", "ols", "ridge", "pls"}, as described in the module docstring;
    lam     : float, penalty for "ridge" and "pls", as a multiple of the mean diagonal of the
              observed-block Gram matrix;
    beta_ts : (M,) array or None, the time series forecast of the scores. Required by "ts" and
              "pls", ignored by the other two.
    """
    head = np.asarray(head, float).ravel()
    m = head.size
    phi_o = dec["phi"][:, :m]
    centred = head - dec["mean"][:m]

    if method == "ts":
        if beta_ts is None:
            raise ValueError("method 'ts' needs the time series score forecast in beta_ts.")
        return np.asarray(beta_ts, float)

    if method == "ols":
        if m < dec["phi"].shape[0]:
            raise ValueError(
                f"{m} observed points cannot identify {dec['phi'].shape[0]} scores by least "
                f"squares; use 'ridge' or 'pls', which can.")
        return np.linalg.lstsq(phi_o.T, centred, rcond=None)[0]

    if method not in ("ridge", "pls"):
        raise ValueError(f"method must be one of {('ts', 'ols', 'ridge', 'pls')}, got {method!r}.")

    G, scale = _gram(phi_o)
    penalty = float(lam) * scale
    rhs = phi_o @ centred
    if method == "pls":
        if beta_ts is None:
            raise ValueError("method 'pls' needs the time series score forecast in beta_ts.")
        rhs = rhs + penalty * np.asarray(beta_ts, float)
    return np.linalg.solve(G + penalty * np.eye(G.shape[0]), rhs)


def _anchored(prediction, boundary, observed):
    """Shift a prediction so its value at the last observed point is the value seen there."""
    return prediction - boundary + observed


def in_sample_updates(dec, m, method="pls", lam=1.0, beta_ts=None, anchor=False):
    """Update every training day as though it had been observed only to `m`, and return the
    errors left on its unobserved block. Returns an (n', N - m) array.

    These are what `select_lambda` minimises and what the bands resample, and they are the
    honest in-sample analogue of the update: each training curve is reconstructed from its own
    observed block under exactly the rule being scored, then compared with the rest of itself.

    The curves used are the observed ones and not their `M`-component reconstructions. That
    distinction is the whole of the criterion: a truncated curve lies exactly in the span of
    the components, so least squares on any block of it recovers its scores exactly and the
    error is zero at every penalty, which would select no shrinkage whatever the data. What
    makes shrinkage worth choosing is precisely the part of a real curve the components do not
    span, and it has to be left in.

    Every method but `block` is covered, including the two benchmarks, so a band built from
    this pool means the same thing for all of them. Block moving is the exception: its
    in-sample analogue would need the rotation rebuilt for every training day, and it is here
    to be scored on its point forecast rather than on an interval.

    Under "pls" and "ts" each day needs its own score forecast. The one used is the FAR(1)
    one-step forecast in score space, `Psi psi_{t-1}`, because it is available for every
    training day at once from one operator rather than from `n` separate model fits; day zero
    has no predecessor and is dropped.

    PARAMETERS
    dec     : dict from `ftsa.forecast.decompose`;
    m       : int, points treated as observed;
    method  : any of `METHODS` except "block";
    lam     : float, penalty for "ridge" and "pls";
    beta_ts : (n', M) array or None, per-day score forecasts; computed as above when not given;
    anchor  : bool, shift each day's update to meet its own last observed value.
    """
    from ..far.far_est import score_operator

    if method == "block":
        raise ValueError("block moving has no cheap in-sample analogue; see the docstring.")
    scores = dec["scores"]
    curves = dec["mean"][None, :] + scores @ dec["phi"] + dec["resid_curves"]   # as observed

    if method == "hold":
        return curves[:, m:] - curves[:, m - 1][:, None]
    if method == "mean":
        shape = dec["mean"][m:] - dec["mean"][m - 1]
        return curves[:, m:] - (curves[:, m - 1][:, None] + shape[None, :])

    needs_forecast = method in ("ts", "pls")

    if needs_forecast and beta_ts is None:
        Psi = score_operator(scores)
        beta_ts = scores[:-1] @ Psi.T                        # (n - 1, M)
    if needs_forecast:
        beta_ts = np.atleast_2d(np.asarray(beta_ts, float))
        days = np.arange(curves.shape[0] - beta_ts.shape[0], curves.shape[0])
    else:
        days = np.arange(curves.shape[0])

    errors = np.empty((days.size, dec["phi"].shape[1] - m))
    for i, t in enumerate(days):
        b = update_scores(dec, curves[t, :m], method=method, lam=lam,
                          beta_ts=beta_ts[i] if needs_forecast else None)
        fitted = dec["mean"][m:] + b @ dec["phi"][:, m:]
        if anchor:
            fitted = _anchored(fitted, dec["mean"][m - 1] + b @ dec["phi"][:, m - 1],
                               curves[t, m - 1])
        errors[i] = curves[t, m:] - fitted
    return errors


def select_lambda(dec, m, method="pls", lambdas=LAMBDAS, beta_ts=None, anchor=False):
    """Choose the penalty by in-sample update error on the training days.

    Returns a dict with lam_hat, lambdas, and mse, the mean squared error left on the
    unobserved block at each candidate.

    The criterion is the error the update is actually scored on, measured on days whose
    remainder is known, so it needs no separate holdout and no refitting of the decomposition.

    PARAMETERS
    dec     : dict from `ftsa.forecast.decompose`;
    m       : int, points treated as observed;
    method  : {"ridge", "pls"}; the others have no penalty to choose;
    lambdas : sequence of float, candidates;
    beta_ts : (n', M) array or None, passed to `in_sample_updates`;
    anchor  : bool, passed to `in_sample_updates`, so the penalty is chosen for the criterion
              the update will be scored on.
    """
    if method not in ("ridge", "pls"):
        raise ValueError(f"only 'ridge' and 'pls' have a penalty, got {method!r}.")
    lambdas = np.asarray(list(lambdas), float)
    mse = np.array([float(np.mean(in_sample_updates(dec, m, method=method, lam=lam,
                                                    beta_ts=beta_ts, anchor=anchor) ** 2))
                    for lam in lambdas])
    return {"lam_hat": float(lambdas[int(np.argmin(mse))]), "lambdas": lambdas, "mse": mse}


def block_moving(X, head, grid, method="far", var_threshold=None, n_components=None):
    """Block moving: rotate the day boundary to the update point and forecast the next day.

    Returns the (N - m,) forecast of the unobserved block.

    With `m` points of day `n + 1` observed, the rotated curves
    `Z_t = (X_t(s_{m+1}), ..., X_t(s_N), X_{t+1}(s_1), ..., X_{t+1}(s_m))` are complete for
    `t = 1, ..., n`, the last of them ending in what has just been seen. The next rotated
    curve opens with `X_{n+1}(s_{m+1}), ..., X_{n+1}(s_N)`, which is exactly the block wanted,
    so forecasting it one step ahead and taking its leading `N - m` points answers the question
    without regressing anything on anything.

    The rotation is what makes this different in kind from the three regression updates: the
    observed block enters as data at the end of a curve rather than as a design matrix, so the
    method never has to invert a Gram matrix that a short observed block makes ill-conditioned.
    What it pays instead is that the rotated curves straddle a day boundary, and on a
    cumulative intraday return that boundary is a discontinuity -- every rotated curve drops
    back to zero at the same interior point, by construction of the transform.

    PARAMETERS
    X             : (n, N) array of complete historical curves in time order;
    head          : (m,) array, the observed part of the new curve;
    grid          : (N,) array, the domain;
    method        : {"arima", "far"}, passed to `ftsa.forecast.forecast_scores`;
    var_threshold : float or None, passed to `ftsa.forecast.decompose`;
    n_components  : int or None, passed to `ftsa.forecast.decompose`.
    """
    X = np.atleast_2d(np.asarray(X, float))
    head = np.asarray(head, float).ravel()
    grid = np.asarray(grid, float)
    n, N = X.shape
    m = head.size
    if not 1 <= m < N:
        raise ValueError(f"{m} observed points on a grid of {N} leaves nothing to rotate.")

    tails = X[:, m:]                                          # (n, N - m)
    heads = np.vstack([X[1:, :m], head[None, :]])             # (n, m), last row the new data
    rotated = np.hstack([tails, heads])                       # (n, N)

    kwargs = {} if var_threshold is None else {"var_threshold": var_threshold}
    dec = fc.decompose(rotated, grid, n_components=n_components, **kwargs)
    step = fc.forecast_scores(dec["scores"], h=1, method=method)
    return fc.reconstruct(dec["mean"], dec["phi"], step["point"])[0, :N - m]


def _band(dec, m, method, lam, anchor, prediction, alpha, n_boot, rng):
    """Pointwise band around `prediction`, from the centred in-sample update errors.

    The same construction for every method that has a pool, so the interval scores in
    `summarise_updates` compare like with like: what is resampled is how wrong this rule was on
    the training days at this update point, added to what it says about the new one.
    """
    pool = in_sample_updates(dec, m, method=method, lam=lam, anchor=anchor)
    pool = pool - pool.mean(axis=0, keepdims=True)
    draws = prediction[None, :] + pool[rng.integers(0, pool.shape[0], int(n_boot))]
    return {"lower": np.quantile(draws, alpha / 2.0, axis=0),
            "upper": np.quantile(draws, 1.0 - alpha / 2.0, axis=0)}


def update_curve(X, grid, head, method="pls", lam=None, beta_ts=None, dec=None,
                 score_method="far", var_threshold=None, n_components=None, alpha=None,
                 n_boot=0, anchor=False, rng=None):
    """Forecast the rest of a day from the part of it already observed.

    Returns a dict with prediction (N - m,), m, method, lam the penalty used, and, when
    `n_boot` is positive, lower and upper (N - m,) from resampling the in-sample update errors.

    PARAMETERS
    X             : (n, N) array of complete historical curves in time order;
    grid          : (N,) array, the domain;
    head          : (m,) array, the observed part of the new curve;
    method        : one of `METHODS`;
    lam           : float or None, the penalty; chosen by `select_lambda` when None;
    beta_ts       : (M,) array or None, the score forecast for the new day; computed when None;
    dec           : dict or None, a decomposition already fitted on `X`, to avoid refitting;
    score_method  : {"arima", "far"}, the forecaster the "ts" and "pls" updates start from;
    var_threshold : float or None, passed to `ftsa.forecast.decompose`;
    n_components  : int or None, passed to `ftsa.forecast.decompose`;
    alpha         : float or None, one minus the nominal coverage of the band;
    n_boot        : int, bootstrap replicates; zero returns no band;
    anchor        : bool, shift the prediction to meet the last observed value, leaving only
                    the shape of the remainder to be judged. Refused for "block", and a no-op
                    for "hold" and "mean", which meet it already;
    rng           : int, Generator or None, seed for the resampling.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    X = np.atleast_2d(np.asarray(X, float))
    head = np.asarray(head, float).ravel()
    grid = np.asarray(grid, float)
    m = head.size
    alpha = ev.CONFIG["alpha"] if alpha is None else float(alpha)
    rng = np.random.default_rng(rng)

    if method in ("hold", "mean"):
        if dec is None:
            kwargs = {} if var_threshold is None else {"var_threshold": var_threshold}
            dec = fc.decompose(X, grid, n_components=n_components, **kwargs)
        prediction = (np.full(grid.size - m, head[-1]) if method == "hold"
                      else head[-1] + dec["mean"][m:] - dec["mean"][m - 1])
        out = {"prediction": prediction, "m": m, "method": method, "lam": np.nan}
        if n_boot:
            out.update(_band(dec, m, method, 0.0, anchor, prediction, alpha, n_boot, rng))
        return out
    if method == "block":
        if anchor:
            raise ValueError(
                "block moving has no fitted value at the last observed point, so there is "
                "nothing to anchor it by; score it unanchored.")
        prediction = block_moving(X, head, grid, method=score_method,
                                  var_threshold=var_threshold, n_components=n_components)
        return {"prediction": prediction, "m": m, "method": method, "lam": np.nan}

    if dec is None:
        kwargs = {} if var_threshold is None else {"var_threshold": var_threshold}
        dec = fc.decompose(X, grid, n_components=n_components, **kwargs)

    if beta_ts is None and method in ("ts", "pls"):
        beta_ts = fc.forecast_scores(dec["scores"], h=1, method=score_method)["point"][0]
    if lam is None and method in ("ridge", "pls"):
        lam = select_lambda(dec, m, method=method, anchor=anchor)["lam_hat"]

    b = update_scores(dec, head, method=method, lam=lam or 0.0, beta_ts=beta_ts)
    prediction = dec["mean"][m:] + b @ dec["phi"][:, m:]
    if anchor:
        prediction = _anchored(prediction, dec["mean"][m - 1] + b @ dec["phi"][:, m - 1],
                               head[-1])
    out = {"prediction": prediction, "m": m, "method": method,
           "lam": float(lam) if lam is not None else np.nan}

    if n_boot:
        out.update(_band(dec, m, method, lam or 0.0, anchor, prediction, alpha, n_boot, rng))
    return out


def rolling_update(X, grid, fractions=UPDATE_FRACTIONS, methods=METHODS, score_method="far",
                   var_threshold=None, init_frac=None, alpha=None, n_boot=0, anchors=(False,),
                   rng=None, verbose=False):
    """Expanding-window evaluation of every update method at every update point.

    Returns a long DataFrame with one row per held-out session, update fraction, method,
    anchoring and grid point, holding s, actual, point and, where `n_boot` is positive, lower
    and upper.

    The decomposition and the score forecast are computed once per held-out session and shared
    across the update points, which is what makes a nine-point sweep affordable: the expensive
    part of the loop does not depend on where in the session the update is made.

    PARAMETERS
    X             : (n, N) array of curves in time order;
    grid          : (N,) array, the domain;
    fractions     : sequence of float, points of the session to update at;
    methods       : sequence of str, methods to score;
    score_method  : {"arima", "far"}, the forecaster the "ts" and "pls" updates start from;
    var_threshold : float or None, passed to `ftsa.forecast.decompose`;
    init_frac     : float or None, fraction of the sample the first training window holds;
    alpha         : float or None, one minus the nominal coverage of the bands;
    n_boot        : int, bootstrap replicates per update; zero returns no bands;
    anchors       : sequence of bool, whether to shift each update to meet the last observed
                    value. The cache scores both, so the level error and the shape error can be
                    told apart;
    rng           : int, Generator or None, seed;
    verbose       : bool, report progress.
    """
    X = np.atleast_2d(np.asarray(X, float))
    grid = np.asarray(grid, float)
    init_frac = ev.CONFIG["init_frac"] if init_frac is None else float(init_frac)
    alpha = ev.CONFIG["alpha"] if alpha is None else float(alpha)
    rng = np.random.default_rng(rng)

    n = X.shape[0]
    start = max(int(np.floor(init_frac * n)), 10)
    if start >= n:
        raise ValueError(f"init_frac leaves no session to score on (n = {n}).")

    rows = []
    for s in range(start, n):
        history = X[:s]
        kwargs = {} if var_threshold is None else {"var_threshold": var_threshold}
        dec = fc.decompose(history, grid, **kwargs)
        beta_ts = fc.forecast_scores(dec["scores"], h=1, method=score_method)["point"][0]

        for fraction in fractions:
            m = split_index(grid, fraction)
            head = X[s, :m]
            for method in methods:
                if method == "ols" and m < dec["n_components"]:
                    continue                      # not identified; the table says so by absence
                for anchor in anchors:
                    if anchor and method == "block":
                        continue                  # nothing to anchor it by
                    out = update_curve(history, grid, head, method=method, beta_ts=beta_ts,
                                       dec=dec, score_method=score_method, alpha=alpha,
                                       n_boot=n_boot, anchor=bool(anchor), rng=rng)
                    block = {"session": s, "fraction": float(fraction), "method": method,
                             "anchored": bool(anchor), "m": m, "s": grid[m:],
                             "actual": X[s, m:], "point": out["prediction"]}
                    if n_boot:
                        block["lower"] = out.get("lower", np.nan)
                        block["upper"] = out.get("upper", np.nan)
                    rows.append(pd.DataFrame(block))
        if verbose and (s - start) % 10 == 0:
            print(f"  split {s - start + 1} of {n - start} (train {s} sessions)")

    return pd.concat(rows, ignore_index=True)


def records(df, alpha=None):
    """Reduce the pointwise sweep to one row per session, update point, anchoring and method.

    Returns a DataFrame with session, fraction, anchored, method, m, n_points and the sums the
    accuracy measures are built from: sae and sse over the unobserved block, and, where the
    frame carries a band, siscore and covered.

    Sums rather than means, and one row per *session* rather than per cell, because that is
    what a paired test needs: every method forecasts the same fifty sessions, so the comparison
    between two of them is a comparison of two columns of fifty numbers. Averaging first would
    throw that away, and keeping every grid point would multiply the same realised curve across
    thirteen method-anchoring pairs for no gain.

    PARAMETERS
    df    : DataFrame from `rolling_update`;
    alpha : float or None, one minus the nominal coverage of the bands; `ftsa.evaluation.CONFIG`
            by default.
    """
    alpha = ev.CONFIG["alpha"] if alpha is None else float(alpha)
    frame = df if "anchored" in df.columns else df.assign(anchored=False)
    error = frame["actual"].to_numpy() - frame["point"].to_numpy()
    work = frame[["session", "fraction", "anchored", "method", "m"]].copy()
    work["n_points"] = 1
    work["sae"] = np.abs(error)
    work["sse"] = error ** 2
    how = {"m": "first", "n_points": "sum", "sae": "sum", "sse": "sum"}
    if "lower" in frame.columns:
        # A method with no band -- block moving -- must come back as missing rather than as a
        # coverage of zero, which would read as a band that never covered anything.
        banded = frame["lower"].notna().to_numpy() & frame["upper"].notna().to_numpy()
        work["siscore"] = np.where(banded, ev.interval_score(
            frame["lower"].to_numpy(), frame["upper"].to_numpy(),
            frame["actual"].to_numpy(), alpha), np.nan)
        work["covered"] = np.where(banded, ((frame["actual"] >= frame["lower"])
                                            & (frame["actual"] <= frame["upper"])), np.nan)
        how["siscore"] = how["covered"] = lambda column: column.sum(min_count=1)

    keys = ["session", "fraction", "anchored", "method"]
    return work.groupby(keys, sort=True).agg(how).reset_index()


def profiles(df):
    """Mean absolute and squared error at each intraday point, by update point and method.

    Returns a DataFrame with fraction, anchored, method, s, mafe and msfe. This is the shape of
    the error along the session, which the per-session sums cannot recover and which is the one
    picture a grand mean always hides.

    PARAMETERS
    df : DataFrame from `rolling_update`.
    """
    frame = df if "anchored" in df.columns else df.assign(anchored=False)
    error = frame["actual"].to_numpy() - frame["point"].to_numpy()
    work = frame[["fraction", "anchored", "method", "s"]].copy()
    work["mafe"] = np.abs(error)
    work["msfe"] = error ** 2
    return (work.groupby(["fraction", "anchored", "method", "s"], sort=True)
            .mean().reset_index())


def summarise_updates(rec, alpha=None):
    """Reduce the per-session records to one row per update point, anchoring and method.

    Returns a DataFrame indexed by (fraction, anchored, method) with mafe, msfe, skill and
    n_splits, and with iscore and coverage for every method that carries a band.

    `skill` is the out-of-sample R-squared against `hold`, the martingale update, at the same
    update point. That is the comparison the table exists for: on a cumulative curve most of
    what is being predicted has already been observed, so a raw error can look small while the
    method has forecast nothing. It is also exactly the R-squared for predicting the increment
    `X(s) - X(s_m)`, `hold` being the prediction that the increment is zero.

    PARAMETERS
    rec   : DataFrame from `records` or `cached_updating`;
    alpha : float or None, kept for symmetry with `records`, which is where the bands were
            scored.
    """
    keys = ["fraction", "anchored", "method"]
    grouped = rec.groupby(keys, sort=True)
    out = pd.DataFrame({
        "mafe": grouped["sae"].sum() / grouped["n_points"].sum(),
        "msfe": grouped["sse"].sum() / grouped["n_points"].sum(),
        "n_splits": grouped["session"].nunique(),
    })
    if "siscore" in rec.columns:
        # `min_count=1` so a method with no band stays missing instead of summing to zero.
        points = grouped["n_points"].sum()
        out["iscore"] = grouped["siscore"].sum(min_count=1) / points
        out["coverage"] = grouped["covered"].sum(min_count=1) / points

    floor = out.xs("hold", level="method")["msfe"].groupby("fraction").first()
    out["skill"] = [1.0 - out.loc[key, "msfe"] / floor.loc[key[0]] for key in out.index]
    return out


def cached_updating(name=ARTIFACT, profile_name=PROFILE_ARTIFACT, refresh=False, **kwargs):
    """The per-session update records, read from `artifacts/cache/` if they are there and
    computed and cached if not. Returns the records DataFrame.

    One run writes both this and the profiles, so the two always describe the same sweep. What
    is kept is the per-session sums and the per-point averages rather than every forecast, in
    the way `bench.study.cached_study` keeps per-replication records: the sweep crosses nine
    update points with thirteen method-anchoring pairs, so a pointwise cache would repeat each
    realised curve thirteen times and reach fifteen megabytes for nothing the analysis uses.

    The design is `ftsa.evaluation.CONFIG` and `UPDATE_FRACTIONS`, and the bootstrap is seeded
    from it, so a rebuild on the same configuration reproduces the committed files.

    PARAMETERS
    name         : str, file name under `artifacts/cache/` for the records;
    profile_name : str, file name under `artifacts/cache/` for the intraday profiles;
    refresh      : bool, recompute and overwrite even where the cache exists;
    **kwargs     : forwarded to `rolling_update`.
    """
    path, profile_path = cache_path(name), cache_path(profile_name)
    if os.path.exists(path) and os.path.exists(profile_path) and not refresh:
        return pd.read_csv(path)

    X, grid, _dates = ev.replication_curves()
    kwargs.setdefault("anchors", (False, True))
    frame = rolling_update(X, grid, n_boot=ev.CONFIG["n_boot"], rng=ev.CONFIG["seed"],
                           **kwargs)
    rec, prof = records(frame), profiles(frame)
    for column in ("sae", "sse", "siscore", "covered"):
        if column in rec:
            rec[column] = rec[column].round(8)
    for column in ("s", "mafe", "msfe"):
        prof[column] = prof[column].round(8)
    rec.to_csv(path, index=False)
    prof.to_csv(profile_path, index=False)
    return rec


def cached_profiles(name=PROFILE_ARTIFACT, refresh=False, **kwargs):
    """The intraday error profiles of the same sweep. Returns a DataFrame with fraction,
    anchored, method, s, mafe and msfe.

    Written by the same run as `cached_updating`, so the two cannot describe different sweeps.

    PARAMETERS
    name     : str, file name under `artifacts/cache/`;
    refresh  : bool, recompute the sweep and overwrite;
    **kwargs : forwarded to `cached_updating`.
    """
    path = cache_path(name)
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    cached_updating(refresh=True, **kwargs)
    return pd.read_csv(path)
