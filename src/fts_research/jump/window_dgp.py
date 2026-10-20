"""A GBM whose overnight jump reads the last part of the previous session.

`basic_dgp` simulates a geometric Brownian motion, in which every night is a Brownian
increment of its own and the jump is independent of everything before it: the null model.
This module keeps that GBM's sessions and replaces its nights, so that the jump depends on how
the previous session ended. On the percent log price `P = 100 log S`, with the window
`[a, 1]`, `a = start`, of width `w = 1 - a` and midpoint `u_bar = (1 + a) / 2`,

    X_t(s)    = 100 m s + 100 sigma W_t(s)                  the GBM's sessions, unchanged;
    dhat_{t-1} = least-squares slope of X_{t-1}(s) on s over [a, 1];
    J_t       = 100 m T + kappa T (dhat_{t-1} - 100 m) + sqrt(T v_{t-1}) Z_t,  Z_t iid N(0, 1);
    P_t(s)    = P_{t-1}(1) + J_t + X_t(s).

**What `dhat` is.** It is the slope of the straight line fitted by least squares to the
previous session's curve over the window,

    dhat = int_a^1 (u - u_bar) X(u) du / int_a^1 (u - u_bar)^2 du
         = (12 / w^3) int_a^1 (u - u_bar) X(u) du,

computed on the grid as `sum_k c_k X(s_k)` with `c_k = (s_k - s_bar) / sum_j (s_j - s_bar)^2`
(`window_slope_weights`). It is a statistic of the observed curve, with nothing unknown in it.
Summing by parts writes it as a weighted average of the window's returns per unit time,

    dhat = sum_k g_k dX_k,    g_k = sum_{j >= k} c_j,    sum_k g_k ds_k = 1,

with `g` zero outside the window and, in the continuum, the parabola
`g(v) = (6 / w^3) ((w / 2)^2 - (v - u_bar)^2)` (`window_return_weights`). So the slope weights
the middle of the window most and its two ends least. (On the grid the weights before the
window are zero up to rounding, about 1e-14, not exactly.)

**`dhat` is not the drift.** Under the model it splits into the drift and the path,

    dhat_{t-1} = d_mu + sigma_eps xi_{t-1},    xi_{t-1} = sum_k g_k dW_{t-1, k},

where `d_mu = 100 m` is the window slope of the mean curve, the same every day, and `xi` is
Gaussian with variance `sum_k g_k^2 ds_k`, about `1.2 / w`. The increments of a Brownian
motion are independent, so `dhat` is exactly `N(d_mu, sigma_eps^2 sum_k g_k^2 ds_k)` and i.i.d.
over days (`window_slope_sd`). With `sigma_eps = 1%` and `w = 0.2` its standard deviation is
2.43% per session, about a hundred times the drift. Almost all of `dhat` is therefore the
direction the price happened to move in over the window, and that is what the jump responds
to: the model is momentum in yesterday's realised late-session trend, not in a drift. The
true drift is a constant and could predict nothing.

**Inference on `dhat` needs the Brownian covariance, not the textbook OLS formula.** The usual
standard error of a regression slope assumes the deviations of the 79 points from the line
are independent. Those of a Brownian path are not: neighbouring points share almost all of
their history. The OLS formula is about eleven times too small on the one-minute grid, and its
95% interval for `d_mu` covers about 14% of the time. The correct interval is
`dhat +- z sigma_eps sqrt(sum g^2 ds)`, exact with `sigma_eps` known (it is pinned down to a
fraction of a percent by pooling days, as `basic_est.fit_null` does). With `sigma_eps` taken
from the window's own realised variance it is a t interval on `K_w - 1` degrees of freedom.
`dhat` and that realised variance come from the same returns, so they are not exactly
independent and the t interval is not exact in theory, but the effect is too small to
measure: its coverage is 95.03% over 500,000 simulated sessions. `window_slope_ci` gives both. The interval is about +-4.8% per
session, which is what "78 minutes say nothing about the drift" means in numbers.

None of that uncertainty enters the jump equation, because the jump is defined by `dhat`
itself, which is observed without error. Regressing `J_t` on `dhat_{t-1}` is an ordinary
regression on an exactly measured regressor, so inference on `kappa T` is the usual
regression inference.

**`kappa` is a dial.** 0 gives back the GBM's own jump law exactly, 1 is a night that carries
on the window's slope in full, and a negative `kappa` is a reversal. `v_{t-1}` is the night's
variance rate, either the GBM's `(100 sigma)^2` (`vol="fixed"`) or the window's realised
variance (`vol="realised"`).

**No overnight path is simulated.** A GBM run over a night of length `T` with log drift `d`
and variance rate `v` has a Gaussian log endpoint, `N(d T, v T)`, and the jump is that
endpoint and nothing else. One normal draw per night is therefore exactly the same thing as
simulating the night's path and keeping its end. The rule is written for the log price
directly, so the Ito term `sigma^2 / 2` never enters.

**This is the basic model with a known coefficient.** `kappa T dhat = int beta(u) X(u) du`
with

    beta(u) = kappa T 12 (u - u_bar) / w^3    on the window, and 0 before it,

which lies in L2, so the jump equation is exactly the notes'
`J_t = alpha + int beta(u) X_{t-1}(u) du + eta_t`, with `alpha = 100 m T (1 - kappa)`.
`window_beta` gives `beta` on the grid, under the trapezium weights of `core.quadrature`, so
that its integral against a curve is `kappa T dhat` to rounding.

**Replacing the nights leaves the sessions alone.** The GBM's log increments over disjoint
intervals are independent, so the sessions do not depend on the nights between them. Throwing
the nights away and drawing new ones changes the gaps between sessions and nothing within
them: the curves are still i.i.d. Brownian, and with the same seed they are the curves
`basic_dgp.gbm_sessions` gives. `joint_dgp.price_curves` lays the curves end to end with the
new jumps in the gaps, so `basic_dgp.split_sessions` gives back exactly the jumps and curves
that went in.

**The first jump reads a burn-in curve.** `J_1` needs a session before day 1, so a day-0
curve `X_0` is drawn from the same GBM and used for that alone. Every one of the `n` jumps
then follows the same law. `X_0` is not part of the returned sample, and its level plays no
part: only its shape over the window enters `J_1`.

**The realised variance carries almost nothing here.** Under a constant `sigma` the window's
realised variance is `(100 sigma)^2 chi^2_df / df`, with `df` one less than the number of
increments in the window: 77 for the last 20% of 390 bars, so its standard deviation is about
16% of the truth (8% on the volatility) and it knows nothing about the night. It makes the
jump's noise heteroskedastic, which the basic model does not allow, so `vol="fixed"` is the
default and the one under which the basic model holds exactly.

`window_jump_truth` gives the moments of the process on its grid, so a simulation is checked
against what it should produce up to Monte Carlo error alone.
"""

import numpy as np
from scipy import stats

from ..core.quadrature import quadrature_weights
from . import basic_dgp as bd
from .joint_dgp import price_curves

__all__ = ["window_slope_weights", "window_return_weights", "window_slope",
           "window_slope_sd", "window_slope_ci", "window_realised_variance", "window_beta",
           "window_jump_sessions", "window_jump_truth"]

VOL = ("fixed", "realised")


def _window(grid, start):
    """Boolean mask of the grid points in `[start, 1]`, with the checks every function needs."""
    grid = np.asarray(grid, float).ravel()
    if not 0.0 < start < 1.0:
        raise ValueError(f"start must lie in (0, 1), got {start}.")
    # A small tolerance, so that a start of 0.8 includes the grid point at 312 / 390.
    on = grid >= start - 1e-12
    if on.sum() < 3:
        raise ValueError(f"the window [{start}, 1] holds {on.sum()} grid points; need 3.")
    return grid, on


def window_slope_weights(grid, start=0.8):
    """Weights `c` such that `X @ c` is the least-squares slope of the curve `X` on `s` over
    the grid points in `[start, 1]`. Returns (N,), zero before the window.

    `c_k = (s_k - s_bar) / sum_j (s_j - s_bar)^2` on the window, so `sum c = 0` and
    `sum c s = 1`: the slope ignores the curve's level and returns `b` for a line `a + b s`.

    PARAMETERS
    grid  : (N,) the within-session grid, increasing;
    start : float, where the window begins; 0.8 is the last 20% of the session.
    """
    grid, on = _window(grid, start)
    dev = grid[on] - grid[on].mean()
    c = np.zeros(grid.size)
    c[on] = dev / (dev @ dev)
    return c


def window_return_weights(grid, start=0.8):
    """Weights `g` such that the slope is `sum_k g_k dX_k`, a weighted sum of the curve's
    increments `dX_k = X(s_k) - X(s_{k-1})`. Returns (N - 1,), one per increment.

    `g_k = sum_{j >= k} c_j`, by summing the slope weights by parts. It is zero for every
    increment before the window, because `sum c = 0`, and `sum_k g_k ds_k = 1`, so the slope is
    a weighted average of the returns per unit time `dX_k / ds_k` with weights `g_k ds_k`. On a
    fine grid `g` is the parabola `(6 / w^3) ((w / 2)^2 - (v - u_bar)^2)` over the window.

    PARAMETERS
    grid  : (N,) the grid;
    start : float, where the window begins.
    """
    c = window_slope_weights(grid, start)
    return np.cumsum(c[::-1])[::-1][1:]


def window_slope(X, grid, start=0.8):
    """`dhat`: the least-squares slope of each curve over `[start, 1]`, in the curves' units
    per session. Returns (n,).

    PARAMETERS
    X     : (n, N) curves on `grid`;
    grid  : (N,) the grid;
    start : float, where the window begins.
    """
    return np.atleast_2d(np.asarray(X, float)) @ window_slope_weights(grid, start)


def window_slope_sd(grid, start=0.8, sigma_eps=1.0):
    """The exact standard deviation of `dhat` when the curve is `mu(s) + sigma_eps W(s)`.
    Returns a float.

    `dhat - d_mu = sigma_eps sum_k g_k dW_k`, and Brownian increments are independent with
    variance `ds_k`, so the variance is `sigma_eps^2 sum_k g_k^2 ds_k`. That equals
    `sigma_eps^2 c' min(s_i, s_j) c`, and is about `1.2 sigma_eps^2 / w` on a fine grid.

    PARAMETERS
    grid      : (N,) the grid;
    start     : float, where the window begins;
    sigma_eps : float, the curve's volatility per unit session time, in the curves' units.
    """
    g = window_return_weights(grid, start)
    return float(sigma_eps * np.sqrt(g**2 @ np.diff(np.asarray(grid, float))))


def window_slope_ci(X, grid, start=0.8, level=0.95, sigma_eps=None):
    """Confidence intervals for `d_mu`, the window slope of the mean curve, from each curve's
    `dhat`. Returns (n, 2), lower and upper.

    With `sigma_eps` given the interval is `dhat +- z sigma_eps sqrt(sum g^2 ds)`, exact under
    a Brownian curve. Without it, `sigma_eps` is taken from each curve's own window realised
    variance and the normal quantile becomes a t quantile on `K_w - 1` degrees of freedom; that
    interval is not exact in theory, because `dhat` and the realised variance come from the
    same returns, but its coverage is 95% to Monte Carlo error (95.03% over 500,000 sessions).

    This is not the textbook OLS interval. That one treats the curve's deviations from the
    fitted line as independent, which a Brownian path's are not, and on the one-minute grid it
    is about eleven times too narrow.

    PARAMETERS
    X         : (n, N) curves on `grid`;
    grid      : (N,) the grid;
    start     : float, where the window begins;
    level     : float, coverage;
    sigma_eps : float or None, the curve's volatility if known, e.g. from `basic_est.fit_null`.
    """
    d = window_slope(X, grid, start)
    unit = window_slope_sd(grid, start, 1.0)
    if sigma_eps is None:
        rv = window_realised_variance(X, grid, start)
        df = int(_window(grid, start)[1].sum()) - 2        # K_w increments, less one
        half = stats.t.ppf((1 + level) / 2, df) * np.sqrt(rv) * unit
    else:
        half = np.full(d.size, stats.norm.ppf((1 + level) / 2) * sigma_eps * unit)
    return np.column_stack([d - half, d + half])


def window_realised_variance(X, grid, start=0.8):
    """The realised variance rate of each curve over `[start, 1]`, with its own drift
    removed. Returns (n,), in the curves' units squared per session.

    Each increment `dX_k` over a cell of length `ds_k` is `d ds_k + sigma sqrt(ds_k) e_k` for a
    Brownian motion with drift, so after dividing by `sqrt(ds_k)` this is a regression through
    the origin, and its residual variance on `K_w - 1` degrees of freedom is unbiased for
    `sigma^2` on any grid. `K_w` is the number of increments in the window.

    PARAMETERS
    X     : (n, N) curves on `grid`;
    grid  : (N,) the grid;
    start : float, where the window begins.
    """
    grid, on = _window(grid, start)
    X = np.atleast_2d(np.asarray(X, float))
    cells = on[:-1]                                    # cells whose left end is in the window
    ds = np.diff(grid)[cells]
    dX = np.diff(X, axis=1)[:, cells]
    d = dX.sum(axis=1, keepdims=True) / ds.sum()
    return ((dX - d * ds) ** 2 / ds).sum(axis=1) / (ds.size - 1)


def window_beta(grid, start=0.8, kappa=1.0, T=0.25):
    """The jump's coefficient function in the basic model, on the grid. Returns (N,).

    `beta = kappa T c / w`, with `c` the slope weights and `w` the trapezium weights, so that
    `sum_k w_k beta_k X(s_k) = kappa T dhat` exactly. On a fine grid it is the line
    `kappa T 12 (u - u_bar) / w^3` on the window and zero before it: the slope weighs the late
    points of the window up and the early ones down.

    PARAMETERS
    grid  : (N,) the grid;
    start : float, where the window begins;
    kappa : float, the share of the window's slope the night carries on;
    T     : float, the night's length in variance time.
    """
    grid = np.asarray(grid, float)
    return kappa * T * window_slope_weights(grid, start) / quadrature_weights(grid)


def window_jump_sessions(n_days=63, n_bars=390, S0=100.0, sigma=0.01, T=0.25, nu=2.5e-4,
                         start=0.8, kappa=0.0, vol="fixed", rng=None):
    """Simulate the GBM's sessions with nights that read the window of the session before.
    Returns a dict with

        S      (n_days, n_bars + 1) the restitched prices, a row per session;
        s      (n_bars + 1,) the within-session grid;
        S0     float, the close before day 1, as given;
        J      (n_days,) the new jumps, `J[t]` opening session `t + 1`, in percent;
        X      (n_days, n_bars + 1) the curves, in percent -- the GBM's, unchanged;
        X0     (n_bars + 1,) the burn-in curve, used only for `J[0]`;
        slope  (n_days,) `dhat_{t-1}`, the window slope of the curve before each jump;
        rv     (n_days,) the window realised variance of the same curves;
        signal (n_days,) `kappa T (slope - 100 m)`, the predictable part of the jump;
        eta    (n_days,) `sqrt(T v) Z`, the rest of it;
        Z      (n_days,) the standard normals behind `eta`;
        S_gbm  (n_days, n_bars + 1) the GBM's prices before the nights were replaced;
        J_gbm  (n_days,) the GBM's own jumps, discarded;
        m, kappa, start, vol, T    as used.

    The draws are made in a fixed order -- the GBM path as `basic_dgp.gbm_sessions` draws it,
    then the burn-in session, then `Z` -- so with the same seed the curves are the ones
    `gbm_sessions` gives, and `kappa` and `vol` change the jumps without changing a single
    draw. Runs at several values of `kappa` from one seed therefore differ only in the signal.

    PARAMETERS
    n_days, n_bars, S0, sigma, T, nu : the GBM, as in `basic_dgp.gbm_sessions`;
    start : float, where the window begins; 0.8 is the last 20% of the session;
    kappa : float, the share of the window's slope the night carries on; 0 is the null;
    vol   : {"fixed", "realised"}, the night's variance rate: the GBM's `(100 sigma)^2`, or
            the window's realised variance;
    rng   : numpy Generator or None.
    """
    if vol not in VOL:
        raise ValueError(f"vol must be one of {VOL}, got {vol!r}.")
    rng = np.random.default_rng() if rng is None else rng
    gbm = dict(n_bars=n_bars, S0=S0, sigma=sigma, T=T, nu=nu)

    path = bd.gbm_sessions(n_days=n_days, rng=rng, **gbm)
    J_gbm, X = bd.split_sessions(path["S"], S0)
    _, X0 = bd.split_sessions(bd.gbm_sessions(n_days=1, rng=rng, **gbm)["S"], S0)
    Z = rng.standard_normal(n_days)

    s, m = path["s"], path["m"]
    before = np.vstack([X0, X[:-1]])                   # the curve before each jump
    slope = window_slope(before, s, start)
    rv = window_realised_variance(before, s, start)
    v = rv if vol == "realised" else np.full(n_days, (100.0 * sigma) ** 2)

    signal = kappa * T * (slope - 100.0 * m)
    eta = np.sqrt(T * v) * Z
    J = 100.0 * m * T + signal + eta
    S = np.exp(price_curves(J, X, p0=100.0 * np.log(S0)) / 100.0)
    return {"S": S, "s": s, "S0": S0, "J": J, "X": X, "X0": X0[0], "slope": slope, "rv": rv,
            "signal": signal, "eta": eta, "Z": Z, "S_gbm": path["S"], "J_gbm": J_gbm,
            "m": m, "kappa": kappa, "start": start, "vol": vol, "T": T}


def window_jump_truth(grid, sigma=0.01, T=0.25, nu=2.5e-4, start=0.8, kappa=0.0):
    """The moments of `dhat` and of the jump under `window_jump_sessions`, exact on `grid`.
    Returns a dict with

        d_mu        100 m, the window slope of the mean curve, which `dhat` is centred on;
        sd_slope    the standard deviation of `dhat`, from `window_slope_sd`;
        var_slope   its square;
        alpha       100 m T (1 - kappa), the intercept of the basic model;
        mean_J      100 m T, the jump's mean, which `kappa` does not move;
        kappa_T     kappa T, the jump's slope on `dhat`;
        beta (N,)   the coefficient function, from `window_beta`;
        var_signal  kappa^2 T^2 var_slope;
        var_noise   (100 sigma)^2 T, the noise variance, exact for `vol="fixed"` and its
                    expectation for `vol="realised"`;
        var_J, sd_J the jump's variance and standard deviation;
        r2          var_signal / var_J, the share of the jump the previous curve predicts.

    The jumps are i.i.d., because each reads a different session and the sessions are i.i.d.,
    so these are the moments of every jump, the first included.

    PARAMETERS
    grid                     : (N,) the within-session grid;
    sigma, T, nu             : the GBM;
    start, kappa             : the window and the share of its slope carried on.
    """
    grid = np.asarray(grid, float)
    m = nu - sigma**2 / 2
    sd_slope = window_slope_sd(grid, start, 100.0 * sigma)
    var_signal = kappa**2 * T**2 * sd_slope**2
    var_noise = (100.0 * sigma) ** 2 * T
    var_J = var_signal + var_noise
    return {"d_mu": 100.0 * m, "sd_slope": sd_slope, "var_slope": sd_slope**2,
            "alpha": 100.0 * m * T * (1 - kappa), "mean_J": 100.0 * m * T,
            "kappa_T": kappa * T, "beta": window_beta(grid, start, kappa, T),
            "var_signal": var_signal, "var_noise": var_noise,
            "var_J": var_J, "sd_J": float(np.sqrt(var_J)), "r2": var_signal / var_J}
