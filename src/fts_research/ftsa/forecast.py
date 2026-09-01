"""Forecasting a functional time series through its principal component scores.

The method is Hyndman & Shang's (2009), as Shang (2017a) applies it to intraday index returns
in the Journal of Forecasting and, in the same year and with the same machinery, to intraday
particulate matter in Econometrics and Statistics (Shang, 2017b).
A sample of curves in time order is decomposed once,

    X_t(s) = mu(s) + sum_{k=1}^{M} psi_{t,k} phi_k(s) + e_t(s),

which turns one dependent sequence of functions into `M` dependent sequences of scalars. Each
score series is forecast on its own and the predicted curve is reassembled,

    X_hat_{n+h}(s) = mu_hat(s) + sum_{k=1}^{M} psi_hat_{n+h,k} phi_hat_k(s),

so all of the temporal modelling happens in `R^M` and none of it in function space. The
decomposition is `far.far_est.fpca`, which is the one this repository uses everywhere else, so
the scores here are the scores `far.far_est.far1_fit` regresses and the components are
orthonormal under `core.quadrature`'s weights rather than under some second convention.

Two routes forecast the scores, and they are the two Shang compares:

`method="arima"`  one univariate ARIMA per score series, order chosen by corrected AIC over a
                  small (p, d, q) grid. The scores are uncorrelated across `k` by construction,
                  so univariate models are the principled default; this is Shang's primary
                  "TS method" and R's `ftsa::forecast.ftsm` calls `auto.arima` the same way.
`method="far"`    one vector autoregression on the whole score vector, at order one. That is
                  Aue, Norinho & Hormann's (2015) route, and at order one it is exactly the
                  operator `far.far_est.far1_fit` estimates -- so the comparison between the
                  two methods here is a comparison of Shang's forecaster against this
                  repository's FAR(1), fitted on identical components.

The univariate route can differ from R's on the same series, because `auto.arima` searches
stepwise from a starting model and this searches the whole grid; on ties the two can land on
different orders and the forecasts move with them. The grid is used because it is the same
criterion, is reproducible across machines, and on score series this short usually agrees.

Prediction intervals are the nonparametric bootstrap of Shang (2017a, Section 5.1), which
resamples both of the error sources the reconstruction has: the score forecast errors, and the
residual functions `e_t` the truncation at `M` components leaves behind.

Both of Shang's papers pair this forecaster with *dynamic updating*, revising the forecast of
the rest of a day once its first hours are observed. That is `updating.py`, which starts from
the one-day-ahead forecast this module produces. See `ftsa/README.md` for the references.

Nothing here builds curves from prices. `fdaSP.intraday.session_curves` does that, and the
cumulative intraday return it returns under `kind="cidr"` is Shang's own transform: anchoring
every curve at the opening price is what makes the curve carry the shape of the session, and
it is also what erases the overnight move, which `jump.jump_eq` is the model for.
"""

import warnings

import numpy as np

from ..far.far_est import fpca, score_operator

__all__ = [
    "decompose",
    "forecast_scores",
    "reconstruct",
    "prediction_bands",
    "forecast_curves",
]

# Widest (p, d, q) the automatic order search considers, matching Hyndman & Khandakar's (2008)
# defaults closely enough for score series of a few hundred points.
MAX_P = 3
MAX_Q = 3
MAX_D = 1


def decompose(X, grid, var_threshold=0.90, n_components=None, max_components=10):
    """Principal component decomposition of curves in time order, truncated by variance
    explained.

    Returns the dict `far.far_est.fpca` returns -- mean (N,), phi (M, N), tau (M,),
    scores (n, M), total, grid -- truncated to the retained components, and with `explained`
    the cumulative fraction of total variance the retained components carry and `resid_curves`
    the (n, N) reconstruction residuals `e_t` the truncation leaves.

    Shang keeps enough components to explain at least 90% of the total variation, which is the
    default here. It is a rule about the *marginal* spread of the curves and knows nothing
    about which components carry the dynamics -- `far.far_est.select_p` is the alternative that
    does, choosing by held-out forecast error instead, and the two need not agree.

    PARAMETERS
    X              : (n, N) array of curves, one per row, in time order;
    grid           : (N,) array, the domain the curves are sampled on;
    var_threshold  : float, smallest cumulative variance fraction the retained components must
                     reach;
    n_components   : int or None, fix the number of components and ignore `var_threshold`;
    max_components : int, most components the threshold rule may reach for.
    """
    X = np.atleast_2d(np.asarray(X, float))
    grid = np.asarray(grid, float)
    n = X.shape[0]
    if not 0.0 < float(var_threshold) <= 1.0:
        raise ValueError(
            f"var_threshold is a fraction of total variance, got {var_threshold}.")

    reach = int(min(max_components, n - 1, grid.size))
    if n_components is not None:
        reach = max(reach, int(n_components))
    dec = fpca(X, grid, n_components=reach)

    cum = np.cumsum(dec["tau"]) / dec["total"]
    if n_components is None:
        M = int(np.searchsorted(cum, float(var_threshold)) + 1)
        M = int(np.clip(M, 1, cum.size))
    else:
        M = int(np.clip(int(n_components), 1, cum.size))

    out = {"mean": dec["mean"], "phi": dec["phi"][:M], "tau": dec["tau"][:M],
           "scores": dec["scores"][:, :M], "total": dec["total"], "grid": dec["grid"],
           "explained": cum[:M], "n_components": M}
    out["resid_curves"] = X - reconstruct(out["mean"], out["phi"], out["scores"])
    return out


def _aicc(llf, n, n_params):
    """Corrected AIC, the criterion Hyndman & Khandakar (2008) select an order by. Returns a
    float, infinite where the sample is too short to correct."""
    k = n_params + 1                      # the innovation variance is a parameter too
    denominator = n - k - 1
    if denominator <= 0:
        return np.inf
    return -2.0 * llf + 2.0 * k + (2.0 * k * (k + 1.0)) / denominator


def _auto_arima(y, max_p=MAX_P, max_q=MAX_Q, max_d=MAX_D):
    """Lowest-AICc ARIMA over the (p, d, q) grid. Returns the fit and its order.

    R's `auto.arima` steps through the grid from a starting model and stops at a local
    minimum; this evaluates the whole grid, which costs a few dozen short fits and returns the
    same answer wherever the stepwise search does not get stuck. The two can still disagree on
    a near-tie, which is the one place a replication of Shang cannot be made exact.
    """
    from statsmodels.tsa.arima.model import ARIMA

    best_crit, best_fit, best_order = np.inf, None, None
    n = len(y)
    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0 and d == 0:
                    continue                  # white noise about a constant has no dynamics
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = ARIMA(y, order=(p, d, q),
                                    trend="c" if d == 0 else "n").fit(
                            method_kwargs={"warn_convergence": False})
                except Exception:             # a non-invertible or unidentified order
                    continue
                crit = _aicc(fit.llf, n, p + q)
                if np.isfinite(crit) and crit < best_crit:
                    best_crit, best_fit, best_order = crit, fit, (p, d, q)

    if best_fit is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ARIMA(y, order=(1, 0, 0), trend="c").fit(), (1, 0, 0)
    return best_fit, best_order


def _far_forecast(scores, h, ridge):
    """Forecast the whole score vector by the order-one vector autoregression.

    The operator is `far.far_est.score_operator`, so this is the FAR(1) of `far.far_est`
    read in score space. The h-step forecast is `Psi^h` applied to the last score vector and
    its error covariance accumulates as `sum_{j<h} Psi^j Sigma Psi^j'`, with `Sigma` the
    empirical covariance of the one-step residuals.
    """
    Psi = score_operator(scores, ridge=ridge)
    resid = scores[1:] - scores[:-1] @ Psi.T             # (n - 1, M)
    Sigma = np.cov(resid, rowvar=False, ddof=1)
    Sigma = np.atleast_2d(Sigma)

    M = scores.shape[1]
    point = np.empty((h, M))
    variance = np.empty((h, M))
    step = np.eye(M)
    accumulated = np.zeros((M, M))
    state = scores[-1]
    for j in range(h):
        accumulated = accumulated + step @ Sigma @ step.T
        state = Psi @ state
        point[j] = state
        variance[j] = np.diag(accumulated)
        step = Psi @ step
    return point, variance, resid, Psi


def forecast_scores(scores, h=1, method="arima", ridge=0.0):
    """Forecast each principal component score series `h` steps ahead.

    Returns a dict with point (h, M) the forecast scores, variance (h, M) their forecast-error
    variances, orders a list of the model chosen per component, and resid (n', M) the in-sample
    one-step residuals the bootstrap resamples.

    PARAMETERS
    scores : (n, M) array of scores in time order, row 0 earliest;
    h      : int, horizons to forecast;
    method : {"arima", "far"}, univariate ARIMA per component, or the order-one vector
             autoregression on the whole score vector;
    ridge  : float, passed to `far.far_est.score_operator` under `method="far"`.
    """
    scores = np.atleast_2d(np.asarray(scores, float))
    n, M = scores.shape
    h = int(h)
    if h < 1:
        raise ValueError(f"h must be at least 1, got {h}.")

    if method == "far":
        point, variance, resid, Psi = _far_forecast(scores, h, ridge)
        return {"point": point, "variance": variance, "resid": resid,
                "orders": [("VAR", 1)] * M, "Psi": Psi, "method": method}

    if method != "arima":
        raise ValueError(f"method must be 'arima' or 'far', got {method!r}.")

    point = np.empty((h, M))
    variance = np.empty((h, M))
    orders, residuals = [], []
    for k in range(M):
        fit, order = _auto_arima(scores[:, k])
        forecast = fit.get_forecast(steps=h)
        point[:, k] = np.asarray(forecast.predicted_mean, float)
        variance[:, k] = np.asarray(forecast.var_pred_mean, float)
        orders.append(order)
        residuals.append(np.asarray(fit.resid, float))

    return {"point": point, "variance": variance,
            "resid": np.column_stack(residuals), "orders": orders, "method": method}


def reconstruct(mean, phi, scores):
    """Assemble curves from scores, `X(s) = mu(s) + sum_k psi_k phi_k(s)`. Returns an
    (h, N) array.

    PARAMETERS
    mean   : (N,) array, the mean function;
    phi    : (M, N) array of principal components;
    scores : (h, M) or (M,) array of scores, one row per curve to build.
    """
    mean = np.asarray(mean, float)
    phi = np.atleast_2d(np.asarray(phi, float))
    scores = np.atleast_2d(np.asarray(scores, float))
    return mean[None, :] + scores @ phi


def prediction_bands(dec, h=1, levels=(0.80, 0.95), n_boot=1000, method="arima",
                     add_residual_function=True, ridge=0.0, rng=None):
    """Pointwise prediction bands by the nonparametric bootstrap of Shang (2017a, Section 5.1).

    Returns a dict with center (h, N) the point forecast, bands {level: (lower, upper)} each
    (h, N), orders as `forecast_scores` reports them, and draws the number of replicates.

    A predicted curve is wrong for two reasons and the bootstrap resamples both. The scores are
    forecast with error, so each replicate perturbs the point forecast by an innovation drawn
    from that component's own in-sample one-step residuals; and the curve is only its first `M`
    components, so each replicate also adds a residual function `e_t` drawn from the sample the
    decomposition left over. Neither error source is given a distribution: what is resampled is
    what was observed, which is the point of doing it this way on returns.

    Past one step the residual pool is still the one-step pool, so each draw is scaled by the
    ratio of the model's h-step to its one-step forecast standard deviation. At `h = 1` that
    factor is one and this is Shang's scheme exactly; beyond it, the shape of the innovation
    distribution is empirical and only its spread comes from the model.

    PARAMETERS
    dec                   : dict from `decompose`;
    h                     : int, horizons to forecast;
    levels                : sequence of float, nominal coverages to return bands for;
    n_boot                : int, bootstrap replicates;
    method                : {"arima", "far"}, passed to `forecast_scores`;
    add_residual_function : bool, include the second error source. False gives the
                            scores-only band, which is what the difference between the two
                            measures;
    ridge                 : float, passed to `forecast_scores`;
    rng                   : int, Generator or None, seed for the resampling.
    """
    rng = np.random.default_rng(rng)
    scores = dec["scores"]
    M = scores.shape[1]

    base = forecast_scores(scores, h=h, method=method, ridge=ridge)
    center = reconstruct(dec["mean"], dec["phi"], base["point"])

    # One-step innovation pools, centred, one per component; and the h-step inflation factor.
    pools = []
    for k in range(M):
        r = base["resid"][:, k]
        r = r[np.isfinite(r)]
        pools.append(r - r.mean() if r.size else np.zeros(1))
    spread = np.sqrt(base["variance"] / np.maximum(base["variance"][0][None, :], 1e-300))

    residual_curves = dec["resid_curves"] if add_residual_function else None
    N = dec["phi"].shape[1]
    draws = np.empty((int(n_boot), h, N))
    for b in range(int(n_boot)):
        perturbed = base["point"].copy()
        for k in range(M):
            perturbed[:, k] += rng.choice(pools[k], size=h, replace=True) * spread[:, k]
        curve = reconstruct(dec["mean"], dec["phi"], perturbed)
        if residual_curves is not None and residual_curves.shape[0]:
            curve = curve + residual_curves[rng.integers(0, residual_curves.shape[0], size=h)]
        draws[b] = curve

    bands = {}
    for level in levels:
        tail = (1.0 - float(level)) / 2.0
        bands[float(level)] = (np.quantile(draws, tail, axis=0),
                               np.quantile(draws, 1.0 - tail, axis=0))
    return {"center": center, "bands": bands, "orders": base["orders"], "draws": int(n_boot)}


def forecast_curves(X, grid, h=1, var_threshold=0.90, n_components=None, method="arima",
                    levels=(0.80, 0.95), n_boot=1000, ridge=0.0, rng=None):
    """The whole pipeline on one sample of curves: decompose, forecast the scores, reassemble,
    and bootstrap the bands.

    Returns a dict with `fpca` the decomposition, `forecast` the output of `prediction_bands`,
    and `explained`, `n_components` and `orders` repeated at the top level, those being the
    three diagnostics Shang reports and the three a reader checks first.

    PARAMETERS
    X             : (n, N) array of curves in time order;
    grid          : (N,) array, the domain;
    h             : int, horizons to forecast;
    var_threshold : float, passed to `decompose`;
    n_components  : int or None, passed to `decompose`;
    method        : {"arima", "far"}, passed to `forecast_scores`;
    levels        : sequence of float, nominal coverages;
    n_boot        : int, bootstrap replicates;
    ridge         : float, passed to `forecast_scores`;
    rng           : int, Generator or None, seed for the resampling.
    """
    dec = decompose(X, grid, var_threshold=var_threshold, n_components=n_components)
    out = prediction_bands(dec, h=h, levels=levels, n_boot=n_boot, method=method,
                           ridge=ridge, rng=rng)
    return {"fpca": dec, "forecast": out, "explained": dec["explained"],
            "n_components": dec["n_components"], "orders": out["orders"]}
