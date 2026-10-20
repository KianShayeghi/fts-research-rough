"""Fitting the null model of *Basic Jump Equation*, and checking it.

    J_t    = alpha + eta_t,                  eta_t ~ iid N(0, sigma_eta^2),
    X_t(s) = mu(s) + sigma_eps W_t(s),       W_t iid standard Brownian motions,  mu(0) = 0,

with `{J_t}` independent of `{X_t}`. This is the model that says the overnight jump cannot be
predicted from intraday returns; the basic model adds `int beta(u) X_{t-1}(u) du` to the jump
equation, and this module supplies the baseline it is tested against. Nothing here regresses
a jump on a curve.

**The likelihood is exact on the grid.** A curve observed at `0 = s_0 < ... < s_K = 1` is the
same information as its `K` increments, and under the null those are independent,

    dX_{t,k} = X_t(s_k) - X_t(s_{k-1}) ~ N(dmu_k, sigma_eps^2 ds_k),

because a Brownian motion has independent Gaussian increments with variance equal to the
elapsed time. The curve equation is therefore `K` normal means sharing one variance, and the
jump equation one normal mean with its own. Both are fitted in closed form:

- `alpha_hat` is the mean jump and `s_eta^2` its sample variance (divisor `n - 1`);
- `dmu_hat_k` is the mean of the `k`-th increment over days, so `mu_hat(s)` is the mean curve.
  `mu` is left free, as the notes leave it, and only `mu(0) = 0` is imposed -- by the data;
- `s_eps^2 = sum_{t,k} (dX_{t,k} - dmu_hat_k)^2 / ds_k / ((n - 1) K)` is realised variance
  pooled over days, each increment scaled by its own length.

Every interval reported is exact rather than asymptotic, because each estimator is a linear or
quadratic form in Gaussians: `alpha_hat` and `mu_hat(s)` are t-distributed about the truth,
`(n - 1) s_eta^2 / sigma_eta^2` is chi-squared on `n - 1` degrees of freedom and
`(n - 1) K s_eps^2 / sigma_eps^2` on `(n - 1) K`, and the residual sums of squares are
independent of the means. The ratio `s_eta^2 / s_eps^2` is F-distributed about
`sigma_eta^2 / sigma_eps^2`, which the GBM sets to the night length `T`.

The two halves of the model are estimated at very different rates, and that asymmetry is the
main thing the fit shows. `sigma_eps` is pinned down by in-fill -- it has `(n - 1) K`
degrees of freedom, and the quadratic variation of a single path would recover it exactly --
whereas `mu(1)` and `alpha` are sharpened only by the number of days, with standard errors
`sigma_eps / sqrt(n)` and `sigma_eta / sqrt(n)`. Both are identified, and consistently
estimated as days accumulate, but three months fix the volatility to a fraction of a percent
and leave the drift indistinguishable from zero.

**`null_diagnostics` checks each assumption of the null separately.** Each row names the
assumption, and each test is chosen to be exact or well calibrated at this sample size:

- jumps are Gaussian -- Shapiro-Wilk, which is the most powerful of the standard omnibus
  tests at small `n`, and Jarque-Bera, which reads the same departure as skewness and
  kurtosis;
- jumps are serially independent -- Ljung-Box on the jumps and on their squares, the second
  being where volatility clustering would show;
- jumps are independent of the same day's session -- the t-test of `corr(J_t, X_t(1))`. The
  lagged correlation is the basic model's question and is deliberately not asked here;
- the curve's volatility is constant through the session -- Bartlett's test of equal variance
  across buckets of the session. On market data this is where the U shape of Andersen &
  Bollerslev (1997) would appear;
- increments within a session are independent -- the lag-1 autocorrelation of the residual
  increments, against its standard error `1 / sqrt((n - 1)(K - 1))`;
- increments are Gaussian -- Jarque-Bera on the standardised residual increments;
- the covariance is Brownian, `sigma_eps^2 min(s, s')` -- the Hilbert-Schmidt distance from
  the empirical covariance, calibrated by simulation. The statistic is invariant to `mu` and
  `sigma_eps`, so simulating standard Brownian motion gives an exact Monte Carlo test;
- the drift is linear, which the GBM implies and the null does not require -- the F test of
  `dmu_k = c ds_k` against free `dmu_k`, exact on `(K - 1, (n - 1) K)` degrees of freedom;
- the curves are independent across days -- the portmanteau test of Gabrys & Kokoszka (2007),
  through `far.far_est.portmanteau_test`.

Because the residuals in each increment column sum to zero across days, two buckets'
residuals are independent but those within one column are not quite. The bucket and
autocorrelation statistics are therefore built from the column residual sums of squares and
their degrees of freedom, not from the standardised increments treated as one i.i.d. sample.

With thirteen checks at 5%, a dataset drawn from the null fails at least one of them about two
times in five -- not one in twenty -- so each row also carries Holm's (1979) adjusted
p-value, and that is the one to read a verdict from. `tests/test_basic_jump.py` checks that
the exact tests reject at their nominal rate under the null, that the family-wise rate after
the adjustment is the nominal one, and that every check rejects the violation it is aimed at.

`goodness_of_fit` turns the fit into the numbers the basic model will be compared against:
the two maximised log-likelihoods with their AIC and BIC, the in-sample root mean squared
error of the jump about its mean, and the share of the data inside the fitted 95% predictive
band. Two things about that comparison are easy to get wrong, and each has a helper.

- **Out of sample, the benchmark is a forecast, not a fit.** `expanding_mean_forecast`
  forecasts each jump by the mean of the jumps before it, which is what the null does out of
  sample. Its error exceeds the in-sample root mean squared error by about `sqrt(1 + 1/m)`
  with `m` jumps in the window, and it is the number a predictor of the jump has to beat.
- **At a quarter's worth of days the likelihood-ratio test is not chi-squared.** In the basic
  model the jump equation, given the curves, is a Gaussian linear regression on the
  principal component scores of yesterday's curve, so `beta = 0` has an exact F test.
  `lr_test_size` gives the true size of the asymptotic chi-squared version, which runs above
  its nominal level by more as the number of components grows.

The basic model regresses `J_t` on `X_{t-1}`, so it can use only `t = 2, ..., n`. Any
likelihood, information criterion or error it is compared with has to be computed on those
same days: `fit_null(J[1:], X[1:], grid)` gives the null's.
"""

import numpy as np
import pandas as pd
from scipy import stats

from ..core.quadrature import quadrature_weights
from ..far.far_est import portmanteau_test

__all__ = [
    "fit_null",
    "parameter_table",
    "predictive_band",
    "standardised_increments",
    "bucket_variances",
    "ljung_box",
    "bartlett_from_variances",
    "simulate_null",
    "covariance_distance",
    "null_diagnostics",
    "holm",
    "goodness_of_fit",
    "expanding_mean_forecast",
    "lr_test_size",
]


def _check_data(J, X, grid):
    J = np.asarray(J, float).ravel()
    X = np.atleast_2d(np.asarray(X, float))
    grid = np.asarray(grid, float).ravel()
    if X.shape != (J.size, grid.size):
        raise ValueError(f"X is {X.shape}; expected ({J.size}, {grid.size}) from J and grid.")
    if J.size < 3:
        raise ValueError("need at least three days.")
    if grid[0] != 0.0 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must start at 0 and increase.")
    if np.max(np.abs(X[:, 0])) > 1e-9:
        raise ValueError("CIDR curves start at zero; X[:, 0] is not zero.")
    return J, X, grid


def fit_null(J, X, grid, level=0.95):
    """Fit the null model by maximum likelihood, with exact intervals. Returns a dict with

        n, K                  days, and increments per curve;
        grid (N,), level      as given;
        alpha, alpha_se, alpha_ci (2,)
                              mean jump, its standard error and t interval;
        sigma_eta, sigma_eta_ci (2,), sigma_eta_mle
                              jump standard deviation (divisor n - 1), its chi-square
                              interval, and the maximum likelihood value (divisor n);
        mu (N,), mu_se (N,), mu_ci (2, N), dmu (K,)
                              the mean curve, its pointwise standard error and t interval,
                              and the mean increments it is the running sum of;
        sigma_eps, sigma_eps_ci (2,), sigma_eps_mle
                              curve volatility (divisor (n - 1) K), its chi-square interval,
                              and the maximum likelihood value (divisor n K);
        df_eta, df_eps        the degrees of freedom n - 1 and (n - 1) K;
        rss_columns (K,)      sum over days of the squared residual increments of each column,
                              each scaled by its length -- the material the bucket test uses;
        loglik_jump, loglik_curve
                              the maximised log-likelihoods of the two equations;
        n_params_jump, n_params_curve
                              2 (alpha, sigma_eta) and K + 1 (the mean increments, sigma_eps).

    The curve log-likelihood is that of the increments, which is the density of the curve
    itself because `X_t(0) = 0` is fixed and the map to increments has unit Jacobian. Both
    log-likelihoods are what the basic model will be compared against.

    PARAMETERS
    J     : (n,) overnight jumps in percent, in time order;
    X     : (n, N) CIDR curves in percent, `X[t]` the session that follows `J[t]`;
    grid  : (N,) the within-session grid, starting at 0;
    level : float, coverage of every interval; 0.95 by convention.
    """
    J, X, grid = _check_data(J, X, grid)
    n, K = J.size, grid.size - 1
    q_lo, q_hi = (1 - level) / 2, (1 + level) / 2

    # The jump equation: one normal mean and its variance.
    alpha = float(J.mean())
    s2_eta = float(J.var(ddof=1))
    df_eta = n - 1
    alpha_se = np.sqrt(s2_eta / n)
    t_eta = stats.t.ppf(q_hi, df_eta)
    sigma_eta_ci = np.sqrt(df_eta * s2_eta / stats.chi2.ppf([q_hi, q_lo], df_eta))

    # The curve equation, through its increments: K normal means sharing one variance.
    ds = np.diff(grid)
    dX = np.diff(X, axis=1)
    dmu = dX.mean(axis=0)
    rss_columns = ((dX - dmu) ** 2).sum(axis=0) / ds
    df_eps = (n - 1) * K
    s2_eps = float(rss_columns.sum() / df_eps)
    mu = np.concatenate([[0.0], np.cumsum(dmu)])
    mu_se = np.sqrt(s2_eps * grid / n)
    t_eps = stats.t.ppf(q_hi, df_eps)
    sigma_eps_ci = np.sqrt(df_eps * s2_eps / stats.chi2.ppf([q_hi, q_lo], df_eps))

    # Maximised log-likelihoods, at the divisor-n variances.
    v_eta = s2_eta * df_eta / n
    v_eps = s2_eps * df_eps / (n * K)
    loglik_jump = -0.5 * n * (np.log(2 * np.pi * v_eta) + 1.0)
    loglik_curve = (-0.5 * n * K * (np.log(2 * np.pi * v_eps) + 1.0)
                    - 0.5 * n * np.log(ds).sum())

    return {
        "n": n, "K": K, "grid": grid, "level": level,
        "alpha": alpha, "alpha_se": float(alpha_se),
        "alpha_ci": alpha + np.array([-1.0, 1.0]) * t_eta * alpha_se,
        "sigma_eta": np.sqrt(s2_eta), "sigma_eta_ci": sigma_eta_ci,
        "sigma_eta_mle": float(np.sqrt(v_eta)),
        "mu": mu, "mu_se": mu_se, "mu_ci": mu + np.outer([-1.0, 1.0], t_eps * mu_se),
        "dmu": dmu,
        "sigma_eps": np.sqrt(s2_eps), "sigma_eps_ci": sigma_eps_ci,
        "sigma_eps_mle": float(np.sqrt(v_eps)),
        "df_eta": df_eta, "df_eps": df_eps, "rss_columns": rss_columns,
        "loglik_jump": float(loglik_jump), "loglik_curve": float(loglik_curve),
        "n_params_jump": 2, "n_params_curve": K + 1,
    }


def parameter_table(fit, truth=None):
    """The fitted parameters as a table, against the truth where it is known. Returns a
    DataFrame indexed by parameter, with columns estimate, se, lower, upper and, given
    `truth`, truth and covered.

    Five rows: `alpha`, `sigma_eta`, `mu(1)` (the session's mean return, the one value of the
    free `mu` the GBM pins to a scalar), `sigma_eps`, and the variance ratio
    `sigma_eta^2 / sigma_eps^2`, which the GBM sets to the night length `T` and the null
    leaves free. The ratio's interval is exact, from the F distribution of
    `(s_eta^2 / sigma_eta^2) / (s_eps^2 / sigma_eps^2)`. The standard errors of the two
    standard deviations are the large-sample `s / sqrt(2 df)`; their intervals are the exact
    chi-square ones.

    PARAMETERS
    fit   : dict from `fit_null`;
    truth : dict or None, as `basic_dgp.gbm_null_parameters` returns it, with keys alpha,
            sigma_eta, drift, sigma_eps and T.
    """
    lvl = fit["level"]
    ratio = fit["sigma_eta"] ** 2 / fit["sigma_eps"] ** 2
    ratio_se = ratio * np.sqrt(2 / fit["df_eta"] + 2 / fit["df_eps"])
    f_lo, f_hi = stats.f.ppf([(1 + lvl) / 2, (1 - lvl) / 2], fit["df_eta"], fit["df_eps"])
    rows = {
        "alpha": (fit["alpha"], fit["alpha_se"], *fit["alpha_ci"]),
        "sigma_eta": (fit["sigma_eta"], fit["sigma_eta"] / np.sqrt(2 * fit["df_eta"]),
                      *fit["sigma_eta_ci"]),
        "mu(1)": (fit["mu"][-1], fit["mu_se"][-1], *fit["mu_ci"][:, -1]),
        "sigma_eps": (fit["sigma_eps"], fit["sigma_eps"] / np.sqrt(2 * fit["df_eps"]),
                      *fit["sigma_eps_ci"]),
        "sigma_eta^2/sigma_eps^2": (ratio, ratio_se, ratio / f_lo, ratio / f_hi),
    }
    table = pd.DataFrame.from_dict(rows, orient="index",
                                   columns=["estimate", "se", "lower", "upper"])
    if truth is not None:
        table["truth"] = [truth["alpha"], truth["sigma_eta"], truth["drift"],
                          truth["sigma_eps"], truth["T"]]
        table["covered"] = (table["lower"] <= table["truth"]) & (table["truth"] <= table["upper"])
    return table


def predictive_band(fit, level=None):
    """Prediction intervals for a new day under the fitted null. Returns a dict with lower
    (N,) and upper (N,), the pointwise band for a new curve, and jump (2,), the interval for
    a new jump.

    The band is exact: a new curve's deviation from the fitted mean, `X_new(s) - mu_hat(s)`,
    is normal with variance `sigma_eps^2 s (1 + 1/n)` and independent of `s_eps`, so the
    band is `mu_hat(s) +- t s_eps sqrt(s (1 + 1/n))` on `(n - 1) K` degrees of freedom, and
    likewise for the jump on `n - 1`. It is pointwise: at 95%, about one point in twenty of
    a new curve falls outside it, not one curve in twenty.

    PARAMETERS
    fit   : dict from `fit_null`;
    level : float or None, coverage; None uses the fit's own level.
    """
    level = fit["level"] if level is None else level
    q = (1 + level) / 2
    infl = np.sqrt(1 + 1 / fit["n"])
    half = stats.t.ppf(q, fit["df_eps"]) * fit["sigma_eps"] * np.sqrt(fit["grid"]) * infl
    half_j = stats.t.ppf(q, fit["df_eta"]) * fit["sigma_eta"] * infl
    return {"lower": fit["mu"] - half, "upper": fit["mu"] + half,
            "jump": fit["alpha"] + np.array([-half_j, half_j])}


def standardised_increments(fit, X):
    """Residual increments of the curves, each divided by its standard deviation. Returns an
    (n, K) array, which under the null has standard normal entries.

    The residual `dX_{t,k} - dmu_hat_k` has variance `sigma_eps^2 ds_k (n - 1) / n`, the
    factor coming from the mean being estimated from the same column; dividing by exactly
    that makes the entries unit-variance. Entries in different columns are independent; those
    within a column have correlation `-1 / (n - 1)`, which is negligible for a QQ plot or a
    kurtosis but is why the exact tests are built from the column sums of squares instead.

    PARAMETERS
    fit : dict from `fit_null`;
    X   : (n, N) the curves the fit was made on.
    """
    X = np.atleast_2d(np.asarray(X, float))
    ds = np.diff(fit["grid"])
    resid = np.diff(X, axis=1) - fit["dmu"]
    n = X.shape[0]
    return resid / (fit["sigma_eps"] * np.sqrt(ds * (n - 1) / n))


def bucket_variances(fit, n_buckets=13, level=0.95):
    """The curve's variance rate in contiguous buckets of the session, relative to the pooled
    rate. Returns a DataFrame with one row per bucket and columns start, end, df, ratio,
    lower and upper.

    `ratio` is the bucket's own variance estimate over the pooled `s_eps^2`, so it is 1 under
    constant volatility. Each bucket's estimate is `sigma_eps^2 chi^2_df / df` with
    `df = (n - 1) * (increments in the bucket)`, and [lower, upper] is the central interval
    of that distribution about 1 -- the band a bucket should fall in when the null holds,
    ignoring the small sampling error of the pooled rate itself.

    PARAMETERS
    fit       : dict from `fit_null`;
    n_buckets : int, buckets the session is split into; 13 is half an hour each over a
                6.5-hour session;
    level     : float, coverage of the per-bucket band.
    """
    grid = fit["grid"]
    cols = np.array_split(np.arange(fit["K"]), n_buckets)
    df = np.array([(fit["n"] - 1) * len(c) for c in cols], float)
    s2_b = np.array([fit["rss_columns"][c].sum() for c in cols]) / df
    lo = stats.chi2.ppf((1 - level) / 2, df) / df
    hi = stats.chi2.ppf((1 + level) / 2, df) / df
    return pd.DataFrame({"start": [grid[c[0]] for c in cols],
                         "end": [grid[c[-1] + 1] for c in cols],
                         "df": df.astype(int), "ratio": s2_b / fit["sigma_eps"] ** 2,
                         "lower": lo, "upper": hi})


def ljung_box(x, lag):
    """The Ljung-Box portmanteau statistic for serial correlation up to `lag`. Returns
    `(Q, p_value)`, the p-value from chi-squared on `lag` degrees of freedom.

    `Q = n (n + 2) sum_{h=1}^{lag} r_h^2 / (n - h)`, with `r_h` the lag-h sample
    autocorrelation; the `n + 2` and `n - h` are Ljung & Box's (1978) small-sample correction
    of Box and Pierce's sum, which is what makes the chi-squared reference usable at a few
    dozen observations.

    PARAMETERS
    x   : (n,) series in time order;
    lag : int, largest lag included.
    """
    x = np.asarray(x, float).ravel()
    n = x.size
    if not 1 <= lag < n:
        raise ValueError(f"lag must be between 1 and {n - 1}, got {lag}.")
    xc = x - x.mean()
    c0 = xc @ xc
    r = np.array([xc[:-h] @ xc[h:] for h in range(1, lag + 1)]) / c0
    Q = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, lag + 1)))
    return float(Q), float(stats.chi2.sf(Q, lag))


def bartlett_from_variances(s2, df):
    """Bartlett's test that several normal variances are equal, from the variance estimates
    and their degrees of freedom. Returns `(statistic, p_value)`, the p-value from
    chi-squared on `len(s2) - 1` degrees of freedom.

    Written from the estimates rather than from raw groups, as `scipy.stats.bartlett` is,
    because the buckets here are pooled over many increment columns, each with its mean
    removed, and the degrees of freedom are what Bartlett's correction needs. Given ordinary
    groups it agrees with `scipy.stats.bartlett`, which `tests/test_basic_jump.py` checks.

    PARAMETERS
    s2 : (g,) array of independent unbiased variance estimates;
    df : (g,) array of their degrees of freedom.
    """
    s2, df = np.asarray(s2, float), np.asarray(df, float)
    g, total = s2.size, df.sum()
    pooled = (df * s2).sum() / total
    stat = total * np.log(pooled) - (df * np.log(s2)).sum()
    corr = 1 + ((1 / df).sum() - 1 / total) / (3 * (g - 1))
    stat /= corr
    return float(stat), float(stats.chi2.sf(stat, g - 1))


def simulate_null(fit, n, rng=None):
    """Draw new days from the fitted null model. Returns `(J, X)`, J (n,) and X (n, N).

    The draw is exact on the fit's grid, as `basic_dgp.gbm_sessions` is: the increments of
    each curve are drawn independently with means `dmu_hat` and variances
    `s_eps^2 ds`. It is the model's own picture of the data, for setting beside the data.

    It is not a draw from the truth. The fitted mean is the free `mu_hat`, which carries the
    sampling noise of the data it was fitted to, so curves drawn from it have a mean that is
    not linear in `s`, and the linear-drift check of `null_diagnostics` rejects them by
    construction. To calibrate the diagnostics, simulate from the data-generating process.

    PARAMETERS
    fit : dict from `fit_null`;
    n   : int, days drawn;
    rng : numpy Generator or None.
    """
    rng = np.random.default_rng() if rng is None else rng
    ds = np.diff(fit["grid"])
    J = fit["alpha"] + fit["sigma_eta"] * rng.standard_normal(n)
    dX = fit["dmu"] + fit["sigma_eps"] * np.sqrt(ds) * rng.standard_normal((n, ds.size))
    return J, np.column_stack([np.zeros(n), np.cumsum(dX, axis=1)])


def covariance_distance(X, grid):
    """The relative Hilbert-Schmidt distance between the curves' sample covariance and the
    Brownian covariance `s_eps^2 min(s, s')` fitted to them. Returns a float, 0 for a perfect
    match.

    Both kernels are integrated with the trapezium weights of `core.quadrature`, so the
    value approximates `||C_hat - s^2 K|| / ||s^2 K||` in `L^2([0, 1]^2)`. `s_eps^2` is the
    increment estimate of `fit_null`, so the distance compares two estimates of the same
    covariance: one assuming the Brownian form, one not.

    PARAMETERS
    X    : (n, N) curves starting at zero;
    grid : (N,) their grid.
    """
    X = np.atleast_2d(np.asarray(X, float))
    n = X.shape[0]
    ds = np.diff(grid)
    dX = np.diff(X, axis=1)
    s2 = (((dX - dX.mean(axis=0)) ** 2).sum(axis=0) / ds).sum() / ((n - 1) * ds.size)
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc / (n - 1)
    B = s2 * np.minimum.outer(grid, grid)
    w = quadrature_weights(grid)
    return float(np.sqrt(w @ (C - B) ** 2 @ w) / np.sqrt(w @ B**2 @ w))


def _covariance_test(X, grid, n_boot, rng):
    """Monte Carlo p-value for `covariance_distance` under the null. Returns (D, p_value)."""
    D = covariance_distance(X, grid)
    n = X.shape[0]
    sd = np.sqrt(np.diff(grid))
    # The distance is invariant to mu and to sigma_eps, so standard Brownian motion is the
    # whole null distribution and the test is exact up to Monte Carlo error.
    sims = np.array([covariance_distance(
        np.column_stack([np.zeros(n), np.cumsum(sd * rng.standard_normal((n, sd.size)),
                                                axis=1)]), grid) for _ in range(n_boot)])
    return D, float((1 + np.sum(sims >= D)) / (n_boot + 1))


def null_diagnostics(fit, J, X, n_buckets=13, lags=(5, 10), n_components=4,
                     portmanteau_lag=5, n_boot=999, rng=None):
    """Test each assumption of the null model on the data it was fitted to. Returns a
    DataFrame with one row per check and columns check, assumption, test, statistic,
    reference, p_value and p_holm.

    The rows, in order: jump normality (Shapiro-Wilk, Jarque-Bera); jump serial independence
    (Ljung-Box on the jumps and on their squares at each of `lags`); independence of the jump
    from the same day's session return (t-test of the correlation); constant volatility
    through the session (Bartlett across `n_buckets` buckets); independent increments within
    a session (lag-1 autocorrelation z-test); Gaussian increments (Jarque-Bera); Brownian
    covariance (Monte Carlo test of `covariance_distance`); linear drift (exact F test); and
    serial independence of the curves (Gabrys-Kokoszka portmanteau). The module docstring
    gives the reasoning for each.

    Thirteen tests at 5% reject about 0.6 times on average even when every assumption holds,
    and at least once in about two datasets out of five, so a single p-value under 0.05 is not
    evidence against the model. `p_holm` is Holm's
    (1979) step-down adjustment across all the rows, which controls the chance of any false
    rejection at the nominal level whatever the dependence between the tests; it is the
    column to read a verdict from. The portmanteau test is the one asymptotic test here, and
    at a quarter's worth of days it is conservative: `tests/test_basic_jump.py` measures its
    size below the nominal 5%.

    PARAMETERS
    fit             : dict from `fit_null`, fitted to `J` and `X`;
    J               : (n,) the jumps;
    X               : (n, N) the curves;
    n_buckets       : int, session buckets for the volatility test; 13 is half-hourly;
    lags            : tuple of int, Ljung-Box lags; 5 and 10 sessions are a week and a
                      fortnight;
    n_components    : int, principal components the portmanteau test projects onto; 4
                      carry 95% of a Brownian curve's variance, the first alone 81%;
    portmanteau_lag : int, lags summed in the portmanteau statistic;
    n_boot          : int, Brownian draws for the covariance test; 999 gives p-values on a
                      grid of 0.001;
    rng             : numpy Generator or None, for the covariance test.
    """
    J, X, grid = _check_data(J, X, fit["grid"])
    rng = np.random.default_rng() if rng is None else rng
    n, K = fit["n"], fit["K"]
    rows = []

    def add(check, assumption, test, statistic, reference, p):
        rows.append({"check": check, "assumption": assumption, "test": test,
                     "statistic": float(statistic), "reference": reference,
                     "p_value": float(p)})

    sw = stats.shapiro(J)
    add("jumps normal", "eta_t is Gaussian", "Shapiro-Wilk W", sw.statistic,
        "W under normality", sw.pvalue)
    jb = stats.jarque_bera(J)
    add("jumps normal", "eta_t is Gaussian", "Jarque-Bera", jb.statistic, "chi2(2)", jb.pvalue)
    for name, series in (("jumps", J), ("squared jumps", J**2)):
        for lag in lags:
            Q, p = ljung_box(series, lag)
            add(f"{name} independent", "eta_t is independent over days",
                f"Ljung-Box, {lag} lags", Q, f"chi2({lag})", p)

    r = np.corrcoef(J, X[:, -1])[0, 1]
    t_r = r * np.sqrt((n - 2) / (1 - r**2))
    add("jump vs session", "J_t is independent of X_t", "t-statistic of corr(J_t, X_t(1))", t_r,
        f"t({n - 2})", 2 * stats.t.sf(abs(t_r), n - 2))

    buckets = bucket_variances(fit, n_buckets=n_buckets)
    B, p = bartlett_from_variances(buckets["ratio"] * fit["sigma_eps"] ** 2, buckets["df"])
    add("constant volatility", "sigma_eps does not vary with s",
        f"Bartlett, {n_buckets} buckets", B, f"chi2({n_buckets - 1})", p)

    ds = np.diff(grid)
    u = (np.diff(X, axis=1) - fit["dmu"]) / np.sqrt(ds)
    rho1 = (u[:, :-1] * u[:, 1:]).sum() / ((n - 1) * (K - 1)) / fit["sigma_eps"] ** 2
    z = rho1 * np.sqrt((n - 1) * (K - 1))
    add("independent increments", "W_t has independent increments",
        "lag-1 autocorrelation, z-test", z, "N(0, 1)", 2 * stats.norm.sf(abs(z)))

    jb_inc = stats.jarque_bera(standardised_increments(fit, X).ravel())
    add("Gaussian increments", "W_t has Gaussian increments", "Jarque-Bera", jb_inc.statistic,
        "chi2(2)", jb_inc.pvalue)

    D, p = _covariance_test(X, grid, n_boot, rng)
    add("Brownian covariance", "Cov X_t(s), X_t(s') = sigma_eps^2 min(s, s')",
        "relative HS distance", D, f"Monte Carlo, {n_boot} draws", p)

    lack = n * np.sum((fit["dmu"] - fit["mu"][-1] * ds) ** 2 / ds)
    F = lack / (K - 1) / fit["sigma_eps"] ** 2
    add("linear drift", "mu(s) = mu(1) s, as the GBM implies", "F test", F,
        f"F({K - 1}, {fit['df_eps']})", stats.f.sf(F, K - 1, fit["df_eps"]))

    pm = portmanteau_test(X, grid, max_lag=portmanteau_lag, n_components=n_components)
    add("curves independent", "X_t is independent over days",
        f"Gabrys-Kokoszka, {pm['max_lag']} lags, {pm['n_components']} PCs", pm["stat"],
        f"chi2({pm['df']})", pm["p_value"])
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p_value"].to_numpy())
    return table


def holm(p):
    """Holm's step-down adjustment of a family of p-values. Returns an array of adjusted
    p-values in the input order.

    The `i`-th smallest p-value is multiplied by `m - i + 1`, the running maximum is taken so
    the adjusted values keep the order of the raw ones, and the result is capped at 1.
    Rejecting every hypothesis whose adjusted p-value is below `a` controls the
    family-wise error rate at `a` under any dependence between the tests, and is uniformly
    more powerful than Bonferroni.

    PARAMETERS
    p : (m,) array of p-values.
    """
    p = np.asarray(p, float)
    m = p.size
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adj
    return out


def goodness_of_fit(fit, J, X, level=None):
    """How well the fitted null describes the data, as numbers the basic model will be
    compared against. Returns a DataFrame indexed by equation ("jump", "curve") with columns
    loglik, n_params, n_obs, aic, bic, rmse and band_coverage.

    `n_obs` is `n` jumps and `n K` increments. `rmse` is the in-sample root mean squared error
    of the fitted mean: of `J_t` about `alpha_hat` for the jumps, and of the session return
    `X_t(1)` about `mu_hat(1)` for the curves. The jump's is the maximum likelihood
    `sigma_eta`; it has seen the jumps it is scored on, so the out-of-sample benchmark is
    `expanding_mean_forecast` instead. `band_coverage` is the share of observations inside the
    `predictive_band`: of the
    jumps, and of the curve points with `s > 0` (at `s = 0` the band is a point). It should
    be close to `level` when the null holds; in-sample, slightly above it.

    PARAMETERS
    fit   : dict from `fit_null`;
    J     : (n,) the jumps;
    X     : (n, N) the curves;
    level : float or None, coverage of the band; None uses the fit's own level.
    """
    J, X, _ = _check_data(J, X, fit["grid"])
    band = predictive_band(fit, level)
    inside = (X[:, 1:] >= band["lower"][1:]) & (X[:, 1:] <= band["upper"][1:])
    jump_in = (J >= band["jump"][0]) & (J <= band["jump"][1])
    out = {}
    for eq, ll, k, nobs, resid, cover in (
            ("jump", fit["loglik_jump"], fit["n_params_jump"], fit["n"],
             J - fit["alpha"], jump_in.mean()),
            ("curve", fit["loglik_curve"], fit["n_params_curve"], fit["n"] * fit["K"],
             X[:, -1] - fit["mu"][-1], inside.mean())):
        out[eq] = {"loglik": ll, "n_params": k, "n_obs": nobs, "aic": 2 * k - 2 * ll,
                   "bic": k * np.log(nobs) - 2 * ll, "rmse": float(np.sqrt(np.mean(resid**2))),
                   "band_coverage": float(cover)}
    return pd.DataFrame.from_dict(out, orient="index")


def expanding_mean_forecast(J, first, start=0):
    """Forecast each jump from index `first` on by the mean of the jumps before it, as the
    null model forecasts out of sample. Returns a dict with t (the indices forecast),
    forecast, error, rmse and inflation.

    This is the benchmark a predictor of the jump has to beat out of sample. Under the null
    the forecast of `J_t` from `m` earlier jumps has mean squared error
    `sigma_eta^2 (1 + 1/m)`, so it is larger than the in-sample `rmse` of `goodness_of_fit`,
    which has seen the jump it is scored on. `inflation` is the square root of the average of
    `1 + 1/m` over the forecasts: the expected `rmse` here is about `sigma_eta * inflation`.

    PARAMETERS
    J     : (n,) the jumps;
    first : int, index of the first jump forecast; `J[t]` is forecast by the mean of
            `J[start:t]`;
    start : int, index of the first jump any window uses. A model of `J_t` on `X_{t-1}` has
            no curve before day 1, so `start=1` trains this benchmark on the same jumps.
    """
    J = np.asarray(J, float)
    if J.ndim != 1:
        raise ValueError(f"J must be one-dimensional, got shape {J.shape}.")
    if not 0 <= start < first < J.size:
        raise ValueError(f"need 0 <= start < first < len(J), got start={start}, "
                         f"first={first}, len(J)={J.size}.")
    t = np.arange(first, J.size)
    m = t - start
    csum = np.concatenate([[0.0], np.cumsum(J[start:])])
    forecast = csum[m] / m
    error = J[t] - forecast
    return {"t": t, "forecast": forecast, "error": error,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "inflation": float(np.sqrt(np.mean(1 + 1 / m)))}


def lr_test_size(n, p, nominal=0.05):
    """The true size of the likelihood-ratio test of `p` zero coefficients in a Gaussian
    linear regression with an intercept, `p` regressors and `n` observations, when it is
    referred to chi-squared on `p` degrees of freedom. Returns a float.

    With both models fitted by maximum likelihood the statistic is `n log(RSS_0 / RSS_1)`, and
    `RSS_0 / RSS_1 = 1 + p F / (n - p - 1)`, where `F` is the usual F statistic, exactly
    `F(p, n - p - 1)` under the null when the regressors are independent of the errors. The
    test therefore rejects when `F > (n - p - 1) (exp(c / n) - 1) / p`, with `c` the
    chi-squared critical value, and its size is the F tail beyond that point. It exceeds
    `nominal` at every finite `n`, by more as `p` grows relative to `n`; the F test is exact.

    PARAMETERS
    n       : int, observations in the regression;
    p       : int, coefficients set to zero under the null, besides the intercept;
    nominal : float, the level the chi-squared critical value is taken at.
    """
    if not 1 <= p < n - 1:
        raise ValueError(f"need 1 <= p < n - 1, got n={n}, p={p}.")
    c = stats.chi2.isf(nominal, p)
    cut = (n - p - 1) * np.expm1(c / n) / p
    return float(stats.f.sf(cut, p, n - p - 1))
