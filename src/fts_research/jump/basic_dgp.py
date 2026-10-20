"""The geometric Brownian motion of *Basic Jump Equation*, and how a day splits into a jump
and a curve.

The notes start from a price in continuous time,

    dS = nu S dtau + sigma S dB_tau,    so    S(tau) = S_0 exp(m tau + sigma B_tau),
    m  = nu - sigma^2 / 2,

with one trading session as the unit of time and a night of length `T` between sessions.
Session `t` runs over `[a_t, a_t + 1]`, `a_t = t (1 + T) - 1`, and its price curve is
`S_t(s) = S(a_t + s)` for `s` in `[0, 1]`. On the percent log price `P = 100 log S` the notes
then define the two pieces each day contributes,

    J_t    = P_t(0) - P_{t-1}(1)    the overnight jump,
    X_t(s) = P_t(s) - P_t(0)        the cumulative intraday return (CIDR) curve,

and show that under the GBM every piece is a Brownian increment over its own interval, so the
pairs `(J_t, X_t)` are i.i.d. and the GBM is a special case of the null model

    J_t    = alpha + eta_t,                  eta_t ~ iid N(0, sigma_eta^2),
    X_t(s) = mu(s) + sigma_eps W_t(s),       W_t iid standard Brownian motions,

with `alpha = 100 m T`, `sigma_eta^2 = 10^4 sigma^2 T`, `mu(s) = 100 m s` and
`sigma_eps = 100 sigma`. `gbm_null_parameters` is that mapping, and is where every "truth"
column in `notebooks/basic_jump_equation.ipynb` comes from.

**`T` is variance time.** The GBM runs the same `sigma` through the night, so `T` sets the
night's share `T / (1 + T)` of each day's variance. Read as clock time (17.5 / 6.5 hours) it
would put almost three quarters of the variance in the hours nothing trades; it is therefore
chosen for the overnight variance it should produce rather than for the length of the night.
Weekends are left out, as in the notes, so every night has the same `T` and `J_t` is i.i.d.

**The path is exact on the grid.** `gbm_sessions` draws the Gaussian increments of `log S`
between consecutive observations -- the night into each open, one bar into every other
point -- so the prices carry no discretisation error. Whatever error a later estimate has is
the estimator's, not the simulator's.

`split_sessions` is the slicing. It is two lines, and lives here rather than in
`fdaSP.intraday` because those functions read bar DataFrames with timestamps and interpolate
onto a grid, while a simulated path already sits on one. `tests/test_basic_jump.py` checks the
two agree, so the route the market data will take gives the same `(J, X)` as this one.
"""

import numpy as np

__all__ = ["gbm_sessions", "split_sessions", "gbm_null_parameters"]


def gbm_sessions(n_days=63, n_bars=390, S0=100.0, sigma=0.01, T=0.25, nu=2.5e-4, rng=None):
    """Simulate a GBM price observed on `n_bars` equal bars of each of `n_days` sessions,
    exactly at the observation times. Returns a dict with

        S   (n_days, n_bars + 1) raw prices, `S[t, k] = S_{t+1}(s_k)`, a row per session;
        s   (n_bars + 1,) the within-session grid, `s_k = k / n_bars`;
        a   (n_days,) the session start times `a_t = t (1 + T) - 1`, `t = 1, ..., n_days`;
        tau (n_days, n_bars + 1) the time `a_t + s_k` of every observation;
        m   float, the drift `nu - sigma^2 / 2` of `log S`.

    The draws are made in a fixed order -- one standard normal per observation, row by row,
    the open of each row first -- so a seed reproduces the same path whatever the caller does
    with it afterwards.

    PARAMETERS
    n_days : int, sessions simulated; 63 is a quarter of a 252-day trading year;
    n_bars : int, bars per session; 390 is one-minute bars over a 6.5-hour session, and every
             fifth point of the resulting grid is the five-minute grid;
    S0     : float, the price at `tau = 0`, i.e. the close before day 1, so that `J_1` exists;
    sigma  : float, volatility per unit of session time; 0.01 is a 1% session standard
             deviation;
    T      : float, length of the night in variance time; 0.25 gives an overnight standard
             deviation half a session's, the night carrying 20% of the day's variance;
    nu     : float, drift of `S` per unit of session time; 2.5e-4 is expected price growth of
             about 8% a year, and an expected log return of about 6.3%;
    rng    : numpy Generator or None, the source of randomness; None draws a fresh one.
    """
    if n_days < 1 or n_bars < 1:
        raise ValueError("need at least one session of at least one bar.")
    if sigma <= 0 or T <= 0 or S0 <= 0:
        raise ValueError("sigma, T and S0 must be positive.")
    rng = np.random.default_rng() if rng is None else rng
    m = nu - sigma**2 / 2

    s = np.linspace(0.0, 1.0, n_bars + 1)
    day = np.arange(1, n_days + 1)
    a = day * (1 + T) - 1
    tau = a[:, None] + s[None, :]

    # The time step into each observation: the night into the open, one bar into every other.
    dtau = np.diff(np.column_stack([a - T, tau]), axis=1)
    log_step = m * dtau + sigma * np.sqrt(dtau) * rng.standard_normal(dtau.shape)
    S = S0 * np.exp(np.cumsum(log_step.ravel()).reshape(n_days, n_bars + 1))
    return {"S": S, "s": s, "a": a, "tau": tau, "m": m}


def split_sessions(S, S0):
    """Cut a matrix of session prices into overnight jumps and CIDR curves, in percent.
    Returns `(J, X)`: J (n,) with `J[t] = P_t(0) - P_{t-1}(1)`, and X (n, N) with
    `X[t] = P_t(s) - P_t(0)`, where `P = 100 log S`.

    `S0` is the close before the first session; it makes the first jump `J_1` defined, so
    every session contributes a pair and the identity `P_t(1) = 100 log S0 + sum_{u <= t}
    (J_u + X_u(1))` holds from the first day.

    PARAMETERS
    S  : (n, N) array of positive prices, a row per session in time order, the open in the
         first column and the close in the last;
    S0 : float, the price at the close before the first session.
    """
    S = np.atleast_2d(np.asarray(S, float))
    if np.any(S <= 0) or S0 <= 0:
        raise ValueError("prices must be positive to take logs.")
    P = 100.0 * np.log(S)
    prev_close = np.concatenate([[100.0 * np.log(S0)], P[:-1, -1]])
    return P[:, 0] - prev_close, P - P[:, [0]]


def gbm_null_parameters(sigma, T, nu):
    """The null-model parameters a GBM implies, in percent. Returns a dict with

        alpha     100 m T, the mean overnight jump;
        sigma_eta 100 sigma sqrt(T), the jump standard deviation;
        drift     100 m, so that `mu(s) = drift * s` and `mu(1) = drift`;
        sigma_eps 100 sigma, the volatility of the curve noise `sigma_eps W_t(s)`;
        T         the night length, which the GBM ties in as `sigma_eta^2 / sigma_eps^2`;
        m         nu - sigma^2 / 2, the drift of `log S`.

    These are pages 8 and 9 of the notes. The null model leaves all four free; the GBM fixes
    them through three numbers, so it also imposes two restrictions the null does not: `mu`
    linear in `s`, and `sigma_eta^2 = T sigma_eps^2`.

    PARAMETERS
    sigma : float, GBM volatility per unit of session time;
    T     : float, night length in the same units;
    nu    : float, GBM drift of `S` per unit of session time.
    """
    m = nu - sigma**2 / 2
    return {"alpha": 100.0 * m * T, "sigma_eta": 100.0 * sigma * np.sqrt(T),
            "drift": 100.0 * m, "sigma_eps": 100.0 * sigma, "T": float(T), "m": m}
