"""Tests for the GBM of *Basic Jump Equation* and its null model (`jump.basic_dgp`,
`jump.basic_est`).

The tests fall into four groups, one per thing `notebooks/basic_jump_equation.ipynb` relies on.

- **The simulator** must reproduce the path the notebook was first run on, and that path must
  have the moments `gbm_null_parameters` says it has, because that mapping supplies every
  "truth" column the notebook prints.
- **The slicing** must partition the log price exactly into jumps and curves, and must agree
  with the `fdaSP.intraday` pipeline that market data will go through, so that switching to
  real data later changes the data and nothing else.
- **The fit** must have its closed forms, its log-likelihoods must be the Gaussian ones the
  basic model will be compared against, and every interval it reports as exact must cover the
  truth at its nominal rate.
- **The diagnostics** must hold their size when the null is true -- otherwise a rejection in
  the notebook means nothing -- and each must reject the violation it is aimed at, otherwise
  a pass means nothing either.

The Monte Carlo tests are seeded, so they are deterministic; their tolerances are about three
binomial standard errors, so a correct implementation passes them with room to spare and an
off-by-one in a degree of freedom does not.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fts_research.fdaSP import intraday
from fts_research.jump import basic_dgp as bd
from fts_research.jump import basic_est as be
from fts_research.jump.joint_dgp import intraday_variance

NB = dict(n_days=63, n_bars=390, S0=100.0, sigma=0.01, T=0.25, nu=2.5e-4)


def _brownian(n, grid, rng, sigma=1.0, mu=None, cell_var=None):
    """`n` curves `mu(s) + sigma W(s)` on `grid`, exact at the grid points; `cell_var`, if
    given, replaces `sigma^2 ds` as the variance of each increment."""
    ds = np.diff(grid)
    var = sigma**2 * ds if cell_var is None else cell_var
    dX = np.sqrt(var) * rng.standard_normal((n, ds.size))
    if mu is not None:
        dX += np.diff(mu)
    return np.column_stack([np.zeros(n), np.cumsum(dX, axis=1)])


def _row(table, check, test_prefix=""):
    rows = table[(table["check"] == check) & table["test"].str.startswith(test_prefix)]
    assert len(rows) == 1, f"expected one {check!r} row, found {len(rows)}"
    return rows.iloc[0]


@pytest.fixture(scope="module")
def notebook_path():
    return bd.gbm_sessions(**NB, rng=np.random.default_rng(0))


# --------------------------------------------------------------------------- simulator


def test_the_simulator_reproduces_the_notebook_path(notebook_path):
    """Moving the generator from the notebook into the package must not change a single
    price: the notebook's section 1 output, and everything fitted to it, would silently
    change otherwise. This is the notebook's original cell, verbatim."""
    rng = np.random.default_rng(0)
    n_days, n_bars, S0, sigma, T, nu = 63, 390, 100.0, 0.01, 0.25, 2.5e-4
    m = nu - sigma**2 / 2
    s = np.linspace(0.0, 1.0, n_bars + 1)
    day = np.arange(1, n_days + 1)
    a = day * (1 + T) - 1
    tau = a[:, None] + s[None, :]
    dtau = np.diff(np.column_stack([a - T, tau]), axis=1)
    log_step = m * dtau + sigma * np.sqrt(dtau) * rng.standard_normal(dtau.shape)
    S = S0 * np.exp(np.cumsum(log_step.ravel()).reshape(n_days, n_bars + 1))

    np.testing.assert_array_equal(notebook_path["S"], S)
    np.testing.assert_array_equal(notebook_path["tau"], tau)
    assert notebook_path["a"][0] == T                # day 1 opens after one night


def test_the_simulated_path_has_the_gbm_null_moments():
    """Every truth column in the notebook comes from `gbm_null_parameters`. If the simulator
    and that mapping disagreed -- a night of the wrong length, a missing factor of 100, the
    drift of `S` used where the drift of `log S` belongs -- the notebook would report
    "covered: False" against a truth that was never true."""
    rng = np.random.default_rng(1)
    pars = dict(n_days=250, n_bars=20, S0=50.0, sigma=0.02, T=0.4, nu=0.01)
    J, X1, Xh = [], [], []
    for _ in range(80):
        g = bd.gbm_sessions(**pars, rng=rng)
        j, x = bd.split_sessions(g["S"], pars["S0"])
        J.append(j)
        X1.append(x[:, -1])
        Xh.append(x[:, 10])
    J, X1, Xh = map(np.concatenate, (J, X1, Xh))
    truth = bd.gbm_null_parameters(pars["sigma"], pars["T"], pars["nu"])
    N = J.size

    assert abs(J.mean() - truth["alpha"]) < 4 * truth["sigma_eta"] / np.sqrt(N)
    assert abs(X1.mean() - truth["drift"]) < 4 * truth["sigma_eps"] / np.sqrt(N)
    assert J.var() == pytest.approx(truth["sigma_eta"] ** 2, rel=4 * np.sqrt(2 / N))
    assert X1.var() == pytest.approx(truth["sigma_eps"] ** 2, rel=4 * np.sqrt(2 / N))
    assert np.cov(Xh, X1)[0, 1] == pytest.approx(truth["sigma_eps"] ** 2 / 2, rel=0.03)
    assert abs(np.corrcoef(J, X1)[0, 1]) < 4 / np.sqrt(N)


# --------------------------------------------------------------------------- slicing


def test_curves_start_at_zero_and_rebuild_the_prices(notebook_path):
    """The definitions on page 1 of the notes partition the log price: every curve starts at
    zero, and the first close plus the running sum of jumps and session returns gives back
    every close. A slicing that dropped or double-counted the first bar, or measured the jump
    from the wrong close, would break the second identity."""
    S = notebook_path["S"]
    J, X = bd.split_sessions(S, NB["S0"])
    P = 100 * np.log(S)
    np.testing.assert_array_equal(X[:, 0], 0.0)
    rebuilt = 100 * np.log(NB["S0"]) + np.cumsum(J + X[:, -1])
    np.testing.assert_allclose(rebuilt, P[:, -1], rtol=0, atol=1e-10)
    np.testing.assert_allclose(P[:, 0], P[:, -1] - X[:, -1], rtol=0, atol=1e-10)


def test_slicing_matches_the_real_data_pipeline(notebook_path):
    """Written as one-minute bars, the simulated path goes through the functions market data
    will: `session_curves(anchor="open")` and `overnight_jumps`. They must give the same
    curves and jumps, in log units rather than percent, so that the simulation is a faithful
    stand-in for the data the method will eventually meet."""
    S = notebook_path["S"]
    n_days, n_bars = S.shape[0], S.shape[1] - 1
    dates = pd.bdate_range("2026-01-05", periods=n_days)
    frames = []
    for t, day in enumerate(dates):
        start = day + pd.Timedelta(hours=9, minutes=30)
        frames.append(pd.DataFrame({
            "ticker": "SIM", "session": day.date(),
            "ts": start + pd.to_timedelta(np.arange(n_bars), unit="min"),
            "open": S[t, :-1], "close": S[t, 1:]}))
    bars = pd.concat(frames, ignore_index=True)

    J, X = bd.split_sessions(S, NB["S0"])
    curves, grid, _ = intraday.session_curves(bars, anchor="open")
    jumps = intraday.overnight_jumps(bars)
    np.testing.assert_allclose(grid, notebook_path["s"], atol=1e-12)
    np.testing.assert_allclose(100 * curves, X, atol=1e-9)
    np.testing.assert_allclose(100 * jumps["jump"].to_numpy(), J[1:], atol=1e-9)


# --------------------------------------------------------------------------- fit


def test_the_fit_has_its_closed_forms():
    """On three hand-written curves over an uneven grid, every estimate equals its formula.
    The uneven grid is what catches an increment scaled by the wrong cell length, and the
    three days are what catch a divisor of n where n - 1 belongs."""
    grid = np.array([0.0, 0.25, 1.0])
    X = np.array([[0.0, 0.5, 1.0], [0.0, -0.5, 0.0], [0.0, 0.3, 1.1]])
    J = np.array([0.2, -0.1, 0.5])
    fit = be.fit_null(J, X, grid)

    assert fit["alpha"] == pytest.approx(J.mean())
    assert fit["sigma_eta"] == pytest.approx(J.std(ddof=1))
    np.testing.assert_allclose(fit["mu"], X.mean(axis=0))
    assert fit["mu"][0] == 0.0
    dX = np.diff(X, axis=1)
    by_hand = sum(((dX[:, k] - dX[:, k].mean()) ** 2).sum() / d
                  for k, d in enumerate([0.25, 0.75])) / ((3 - 1) * 2)
    assert fit["sigma_eps"] ** 2 == pytest.approx(by_hand)
    np.testing.assert_allclose(fit["mu_se"], np.sqrt(by_hand * grid / 3))
    assert (fit["df_eta"], fit["df_eps"], fit["n_params_curve"]) == (2, 4, 3)


def test_the_log_likelihood_is_the_gaussian_one():
    """The stored log-likelihoods are what the basic model's likelihood ratio and information
    criteria will be computed against, so they must be exactly the Gaussian log density of
    the data at the maximum likelihood estimates: of the jumps, and of the curve increments,
    each with variance `sigma_eps^2 ds`."""
    rng = np.random.default_rng(2)
    grid = np.sort(np.concatenate([[0.0, 1.0], rng.uniform(size=8)]))
    X = _brownian(25, grid, rng, sigma=0.7)
    J = 0.1 + 0.4 * rng.standard_normal(25)
    fit = be.fit_null(J, X, grid)

    ll_jump = stats.norm.logpdf(J, fit["alpha"], fit["sigma_eta_mle"]).sum()
    ds = np.diff(grid)
    ll_curve = stats.norm.logpdf(np.diff(X, axis=1), fit["dmu"],
                                 fit["sigma_eps_mle"] * np.sqrt(ds)).sum()
    assert fit["loglik_jump"] == pytest.approx(ll_jump, rel=1e-12)
    assert fit["loglik_curve"] == pytest.approx(ll_curve, rel=1e-12)


def test_the_fit_recovers_the_truth_in_large_samples():
    """Consistency: with two thousand days every estimate sits within a few standard errors
    of the parameter that generated the data, including a mean curve that is far from linear
    -- the null leaves `mu` free, and so must the fit."""
    rng = np.random.default_rng(3)
    grid = np.linspace(0, 1, 51)
    mu = 0.4 * np.sin(np.pi * grid)
    X = _brownian(2000, grid, rng, sigma=1.3, mu=mu)
    J = -0.2 + 0.6 * rng.standard_normal(2000)
    fit = be.fit_null(J, X, grid)

    assert abs(fit["alpha"] + 0.2) < 4 * fit["alpha_se"]
    assert abs(fit["sigma_eps"] - 1.3) < 4 * 1.3 / np.sqrt(2 * fit["df_eps"])
    assert abs(fit["sigma_eta"] - 0.6) < 4 * 0.6 / np.sqrt(2 * fit["df_eta"])
    assert np.all(np.abs(fit["mu"] - mu)[1:] < 4.5 * fit["mu_se"][1:])


def test_every_interval_has_its_nominal_coverage():
    """`fit_null` and `parameter_table` call their intervals exact. Over 500 replications each
    must cover the truth 95% of the time, within three binomial standard errors (0.93 to
    0.97); an asymptotic interval passed off as exact at n = 30 would fall short, and a wrong
    degree of freedom would miss in one direction or the other."""
    rng = np.random.default_rng(4)
    n, grid = 30, np.linspace(0, 1, 21)
    mu = 0.3 * np.sin(np.pi * grid)                   # mu(1) = 0, and nowhere near linear
    truth = {"alpha": 0.05, "sigma_eta": 0.5, "drift": 0.0, "sigma_eps": 1.2,
             "T": 0.25 / 1.44}
    hits, mid = [], []
    for _ in range(500):
        X = _brownian(n, grid, rng, sigma=1.2, mu=mu)
        J = 0.05 + 0.5 * rng.standard_normal(n)
        fit = be.fit_null(J, X, grid)
        hits.append(be.parameter_table(fit, truth)["covered"].to_numpy())
        lo, hi = fit["mu_ci"][:, 10]
        mid.append(lo <= mu[10] <= hi)
    coverage = np.mean(hits, axis=0)
    se = np.sqrt(0.95 * 0.05 / 500)
    assert np.all(np.abs(coverage - 0.95) < 3 * se), dict(
        zip(["alpha", "sigma_eta", "mu(1)", "sigma_eps", "ratio"], coverage))
    assert abs(np.mean(mid) - 0.95) < 3 * se


# --------------------------------------------------------------------------- diagnostics


def test_the_helpers_agree_with_the_reference_implementations():
    """`ljung_box`, `bartlett_from_variances` and `holm` are written here so that the bucket
    test can take pooled variances and so that `jump` does not import statsmodels; each must
    agree with the standard implementation where both apply."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(5)
    x = rng.standard_normal(80)
    ref = acorr_ljungbox(x, lags=[7])
    Q, p = be.ljung_box(x, 7)
    assert Q == pytest.approx(ref["lb_stat"].iloc[0], rel=1e-10)
    assert p == pytest.approx(ref["lb_pvalue"].iloc[0], rel=1e-8)

    groups = [rng.normal(scale=s, size=k) for s, k in ((1, 10), (1.5, 25), (0.8, 17))]
    B, pB = be.bartlett_from_variances([g.var(ddof=1) for g in groups],
                                       [g.size - 1 for g in groups])
    assert (B, pB) == pytest.approx(tuple(stats.bartlett(*groups)), rel=1e-10)

    pv = rng.uniform(size=12) ** 3
    np.testing.assert_allclose(be.holm(pv), multipletests(pv, method="holm")[1])


def test_the_exact_checks_hold_their_size_under_the_null():
    """On data that satisfy the null, the exact checks must reject at their nominal rate --
    about 5%, within three binomial standard errors over 400 replications -- or the
    diagnostics table would flag a correctly specified model. The portmanteau test is
    asymptotic and is only required not to over-reject: at a few dozen days it is
    conservative, which is the safe direction."""
    rng = np.random.default_rng(6)
    n, grid = 40, np.linspace(0, 1, 61)
    checks = ["linear drift", "constant volatility", "jump vs session",
              "independent increments", "curves independent"]
    rejected = {c: 0 for c in checks}
    reps = 400
    for _ in range(reps):
        X = _brownian(n, grid, rng, mu=0.2 * grid)
        J = 0.1 + 0.5 * rng.standard_normal(n)
        table = be.null_diagnostics(be.fit_null(J, X, grid), J, X, n_boot=1, rng=rng)
        for c in checks:
            rejected[c] += _row(table, c)["p_value"] < 0.05
    rate = {c: v / reps for c, v in rejected.items()}
    band = 3 * np.sqrt(0.05 * 0.95 / reps)
    for c in checks[:-1]:
        assert abs(rate[c] - 0.05) < band, rate
    assert rate["curves independent"] < 0.05 + band, rate


def test_holm_controls_the_family_wise_rate_under_the_null():
    """The notebook reads its verdicts from the Holm-adjusted column because thirteen checks
    at 5% flag a correct model far more often than one time in twenty. This measures both
    halves of that claim on null data at the notebook's sample size: without the adjustment
    some check fails in well over a quarter of datasets, and with it no more than 5% (within
    three binomial standard errors over 300 datasets)."""
    rng = np.random.default_rng(9)
    n, grid = 63, np.linspace(0, 1, 79)
    any_raw = any_holm = 0
    reps = 300
    for _ in range(reps):
        X = _brownian(n, grid, rng)
        J = rng.standard_normal(n)
        table = be.null_diagnostics(be.fit_null(J, X, grid), J, X, n_boot=99, rng=rng)
        any_raw += (table["p_value"] < 0.05).any()
        any_holm += (table["p_holm"] < 0.05).any()
    assert any_raw / reps > 0.25
    assert any_holm / reps < 0.05 + 3 * np.sqrt(0.05 * 0.95 / reps)


def _cases():
    """One alternative per check: (name, check, test prefix, generator of (J, X, grid))."""
    def null_curves(n, rng, grid):
        return _brownian(n, grid, rng)

    def u_shape(rng):
        grid = np.linspace(0, 1, 391)
        cell = np.diff(grid) * intraday_variance(0.5 * (grid[1:] + grid[:-1]), ratio=2.5)
        return rng.standard_normal(63), _brownian(63, grid, rng, cell_var=cell), grid

    def ar_jumps(rng):
        grid = np.linspace(0, 1, 41)
        e = rng.standard_normal(200)
        J = np.empty(200)
        J[0] = e[0]
        for t in range(1, 200):
            J[t] = 0.6 * J[t - 1] + e[t]
        return J, null_curves(200, rng, grid), grid

    def heavy_jumps(rng):
        grid = np.linspace(0, 1, 41)
        return rng.standard_t(3, size=200), null_curves(200, rng, grid), grid

    def clustered_jumps(rng):
        grid = np.linspace(0, 1, 41)
        h, J = np.empty(400), np.empty(400)
        h[0], J[0] = 1.0, rng.standard_normal()
        for t in range(1, 400):                      # GARCH(1, 1) with strong persistence
            h[t] = 0.05 + 0.25 * J[t - 1] ** 2 + 0.7 * h[t - 1]
            J[t] = np.sqrt(h[t]) * rng.standard_normal()
        return J, null_curves(400, rng, grid), grid

    def correlated(rng):
        grid = np.linspace(0, 1, 41)
        X = null_curves(63, rng, grid)
        return 0.5 * X[:, -1] + 0.3 * rng.standard_normal(63), X, grid

    def ma_increments(rng):
        grid = np.linspace(0, 1, 391)
        e = rng.standard_normal((63, 391))
        dX = (e[:, 1:] + 0.3 * e[:, :-1]) * np.sqrt(np.diff(grid))
        return rng.standard_normal(63), np.column_stack([np.zeros(63), dX.cumsum(1)]), grid

    def t_increments(rng):
        grid = np.linspace(0, 1, 391)
        dX = rng.standard_t(4, size=(63, 390)) * np.sqrt(np.diff(grid) / 2)
        return rng.standard_normal(63), np.column_stack([np.zeros(63), dX.cumsum(1)]), grid

    def ou_curves(rng):
        grid = np.linspace(0, 1, 101)
        dt, theta = 0.01, 6.0
        X = np.zeros((63, 101))
        for k in range(1, 101):
            X[:, k] = X[:, k - 1] * np.exp(-theta * dt) + np.sqrt(
                (1 - np.exp(-2 * theta * dt)) / (2 * theta)) * rng.standard_normal(63)
        return rng.standard_normal(63), X, grid

    def curved_drift(rng):
        grid = np.linspace(0, 1, 391)
        mu = 0.5 * np.sin(2 * np.pi * grid)
        return rng.standard_normal(63), _brownian(63, grid, rng, mu=mu), grid

    def ar_curves(rng):
        grid = np.linspace(0, 1, 51)
        W = _brownian(150, grid, rng)
        X = W.copy()
        for t in range(1, 150):
            X[t] = 0.7 * X[t - 1] + W[t]
        return rng.standard_normal(150), X, grid

    return [
        ("U-shaped volatility", "constant volatility", "", u_shape),
        ("AR(1) jumps", "jumps independent", "Ljung-Box, 5", ar_jumps),
        ("t(3) jumps", "jumps normal", "Shapiro", heavy_jumps),
        ("t(3) jumps", "jumps normal", "Jarque", heavy_jumps),
        ("GARCH jumps", "squared jumps independent", "Ljung-Box, 5", clustered_jumps),
        ("jump tied to session", "jump vs session", "", correlated),
        ("MA(1) increments", "independent increments", "", ma_increments),
        ("t(4) increments", "Gaussian increments", "", t_increments),
        ("Ornstein-Uhlenbeck curves", "Brownian covariance", "", ou_curves),
        ("sinusoidal drift", "linear drift", "", curved_drift),
        ("AR(1) curves", "curves independent", "", ar_curves),
    ]


@pytest.mark.parametrize("name, check, prefix, make", _cases(),
                         ids=[f"{c[0]} -> {c[1]} {c[2]}".strip() for c in _cases()])
def test_each_check_detects_the_violation_it_targets(name, check, prefix, make):
    """A check that cannot fail tells the notebook's reader nothing when it passes. Each
    alternative here breaks exactly one assumption of the null, plainly, and the check aimed
    at that assumption must reject it at 1% -- even after the Holm adjustment across all
    thirteen checks, which is the column the notebook reads its verdicts from.

    A Monte Carlo p-value cannot fall below `1 / (n_boot + 1)`, and Holm multiplies the
    smallest by thirteen, so the covariance check gets 1999 draws here; the other checks
    do not use the draws and get a token 99."""
    rng = np.random.default_rng(7)
    J, X, grid = make(rng)
    n_boot = 1999 if check == "Brownian covariance" else 99
    table = be.null_diagnostics(be.fit_null(J, X, grid), J, X, n_boot=n_boot, rng=rng)
    row = _row(table, check, prefix)
    assert row["p_holm"] < 0.01, f"{name}: {check} p = {row['p_value']:.3g}"


def test_the_band_coverage_is_nominal_under_the_null():
    """The share of the data inside the fitted 95% predictive band is reported as a measure
    of fit, so on data from the null it must come out at 95% -- for the curve points, pooled
    over many, to within a percentage point, and for the jumps within binomial error. A band
    built with `s` where `sqrt(s)` belongs, or without the `1 + 1/n` for the estimated mean,
    would miss."""
    rng = np.random.default_rng(8)
    grid = np.linspace(0, 1, 101)
    n = 400
    X = _brownian(n, grid, rng, sigma=0.8, mu=0.1 * grid)
    J = 0.3 * rng.standard_normal(n)
    gof = be.goodness_of_fit(be.fit_null(J, X, grid), J, X)
    assert gof.loc["curve", "band_coverage"] == pytest.approx(0.95, abs=0.01)
    assert gof.loc["jump", "band_coverage"] == pytest.approx(0.95, abs=3 * np.sqrt(0.0475 / n))


def test_bad_input_is_refused():
    """Curves that do not start at zero are not CIDR curves, and a grid that does not start
    at zero is not the session; fitting either would produce numbers without meaning."""
    grid = np.linspace(0, 1, 5)
    X = np.zeros((4, 5))
    with pytest.raises(ValueError, match="start at zero"):
        be.fit_null(np.zeros(4), X + 1.0, grid)
    with pytest.raises(ValueError, match="grid must start at 0"):
        be.fit_null(np.zeros(4), X, grid + 0.1)
    with pytest.raises(ValueError, match="expected"):
        be.fit_null(np.zeros(3), X, grid)
    with pytest.raises(ValueError, match="positive"):
        bd.split_sessions(np.array([[1.0, -1.0]]), 1.0)


# --------------------------------------------------------------------- the basic model's baseline


def test_the_expanding_mean_forecast_uses_only_the_past():
    """Each forecast must be the mean of the jumps strictly before it, from `start` on. A
    window that included the jump being forecast would score the null in sample, and a
    window that began at day 1 when the basic model starts at day 2 would train the two on
    different data."""
    J = np.array([4.0, 1.0, 2.0, 3.0, 6.0])
    out = be.expanding_mean_forecast(J, first=3, start=1)
    np.testing.assert_array_equal(out["t"], [3, 4])
    np.testing.assert_allclose(out["forecast"], [1.5, 2.0])
    np.testing.assert_allclose(out["error"], [1.5, 4.0])
    assert out["rmse"] == pytest.approx(np.sqrt((1.5**2 + 4.0**2) / 2))
    assert out["inflation"] == pytest.approx(np.sqrt(np.mean([1 + 1 / 2, 1 + 1 / 3])))
    for first, start in ((0, 0), (2, 2), (5, 0)):
        with pytest.raises(ValueError, match="start < first"):
            be.expanding_mean_forecast(J, first=first, start=start)


def test_the_expanding_mean_forecast_error_is_inflated_as_stated():
    """Under the null the out-of-sample mean squared error is `sigma^2 (1 + 1/m)`, which is
    what `inflation` reports and what the notebook compares the in-sample error with. Over
    four thousand quarters the average must match it to well under a percent."""
    rng = np.random.default_rng(11)
    sigma, reps = 0.5, 4000
    mse = np.mean([be.expanding_mean_forecast(0.1 + sigma * rng.standard_normal(63),
                                              first=21, start=1)["rmse"] ** 2
                   for _ in range(reps)])
    inflation = be.expanding_mean_forecast(np.zeros(63), first=21, start=1)["inflation"]
    assert mse == pytest.approx(sigma**2 * inflation**2, rel=0.01)


def test_the_lr_test_size_matches_simulation():
    """`lr_test_size` is a closed form for how often the chi-squared likelihood-ratio test of
    `beta = 0` rejects a true null. Simulated regressions of the notebook's size, with the
    statistic computed the long way from the two residual sums of squares, must reject at
    that rate within binomial error, and the rate must sit above the nominal 5%."""
    rng = np.random.default_rng(12)
    n, p, reps = 62, 4, 20000
    crit = stats.chi2.isf(0.05, p)
    Z = np.column_stack([np.ones(n), rng.standard_normal((n, p))])
    y = rng.standard_normal((reps, n))
    rss1 = ((y - y @ (Z @ np.linalg.pinv(Z))) ** 2).sum(axis=1)     # the hat matrix is symmetric
    rss0 = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    rate = np.mean(n * np.log(rss0 / rss1) > crit)
    size = be.lr_test_size(n, p)
    assert size > 0.05
    assert rate == pytest.approx(size, abs=3 * np.sqrt(size * (1 - size) / reps))


def test_the_lr_test_size_is_nominal_in_large_samples_and_refuses_too_many_regressors():
    """The chi-squared reference is the large-sample limit, so the size must approach the
    nominal level as `n` grows and grow with `p` at fixed `n`."""
    assert be.lr_test_size(10**6, 4) == pytest.approx(0.05, abs=1e-4)
    sizes = [be.lr_test_size(62, p) for p in (1, 2, 4, 8)]
    assert all(a < b for a, b in zip(sizes, sizes[1:]))
    with pytest.raises(ValueError, match="1 <= p < n - 1"):
        be.lr_test_size(10, 9)
