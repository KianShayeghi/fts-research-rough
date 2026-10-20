"""Tests for the window-jump GBM (`jump.window_dgp`), which `notebooks/window_jump_dgp.ipynb`
uses to make overnight jumps that depend on the end of the session before.

Seven tests, one per claim the notebook makes about the data. The Monte Carlo is one long seeded
run, and its tolerances are about three standard errors.
"""

import numpy as np
import pytest
from scipy import stats

from fts_research.core.quadrature import quadrature_weights
from fts_research.jump import basic_dgp as bd
from fts_research.jump import window_dgp as wd

GBM = dict(n_bars=390, S0=100.0, sigma=0.01, T=0.25, nu=2.5e-4)
GRID = np.linspace(0.0, 1.0, 391)


def test_slope_weights_are_the_least_squares_slope():
    """`dhat` is what the notebook says it is: the LS slope over the window, blind before it,
    and the same number as the weighted average of the window's returns."""
    X = bd.split_sessions(bd.gbm_sessions(n_days=5, rng=np.random.default_rng(1), **GBM)["S"],
                          100.0)[1]
    on = GRID >= 0.8 - 1e-12
    slopes = [np.polyfit(GRID[on], x[on], 1)[0] for x in X]
    assert np.allclose(wd.window_slope(X, GRID, 0.8), slopes)
    assert np.all(wd.window_slope_weights(GRID, 0.8)[~on] == 0)
    g = wd.window_return_weights(GRID, 0.8)
    assert np.allclose(np.diff(X, axis=1) @ g, slopes)
    assert np.isclose(g @ np.diff(GRID), 1.0)


def test_curves_untouched_and_restitching_exact():
    """Replacing the nights changes only the gaps: the curves are the GBM's, and slicing the
    restitched prices gives back the jumps and curves that went in."""
    gbm_X = bd.split_sessions(bd.gbm_sessions(n_days=63, rng=np.random.default_rng(0), **GBM)["S"],
                              100.0)[1]
    sims = [wd.window_jump_sessions(kappa=k, rng=np.random.default_rng(0), **GBM)
            for k in (0.0, 1.0)]
    for sim in sims:
        assert np.array_equal(sim["X"], gbm_X)
        J, X = bd.split_sessions(sim["S"], sim["S0"])
        assert np.allclose(J, sim["J"], atol=1e-10) and np.allclose(X, sim["X"], atol=1e-10)
    assert np.array_equal(sims[0]["Z"], sims[1]["Z"])            # common random numbers


def test_jump_reads_the_previous_curve():
    """`J_t` is built from `X_{t-1}`, and `J_1` from the burn-in curve, never from `X_t`."""
    sim = wd.window_jump_sessions(n_days=10, kappa=1.0, rng=np.random.default_rng(2), **GBM)
    before = np.vstack([sim["X0"], sim["X"][:-1]])
    assert np.allclose(sim["slope"], wd.window_slope(before, sim["s"]))
    # The coefficient function integrates against that curve to the same signal.
    beta = wd.window_beta(sim["s"], kappa=1.0, T=GBM["T"])
    assert np.allclose(before @ (quadrature_weights(sim["s"]) * beta), GBM["T"] * sim["slope"])


def test_kappa_zero_is_the_null():
    """At `kappa = 0` the jumps follow the GBM's own law, `N(100 m T, 10^4 sigma^2 T)`."""
    sim = wd.window_jump_sessions(kappa=0.0, rng=np.random.default_rng(3), **GBM)
    null = bd.gbm_null_parameters(GBM["sigma"], GBM["T"], GBM["nu"])
    assert np.allclose(sim["J"], null["alpha"] + null["sigma_eta"] * sim["Z"])


def test_moments_match_the_truth():
    """The truth table in the notebook is right: Var(J) and the slope of J_t on d_{t-1}."""
    kappa = 0.5
    sim = wd.window_jump_sessions(n_days=20_000, kappa=kappa, rng=np.random.default_rng(4),
                                  **GBM)
    truth = wd.window_jump_truth(sim["s"], GBM["sigma"], GBM["T"], GBM["nu"], kappa=kappa)
    n = sim["J"].size
    assert abs(sim["slope"].var() / truth["var_slope"] - 1) < 3 * np.sqrt(2 / n)
    assert abs(sim["J"].var() / truth["var_J"] - 1) < 3 * np.sqrt(2 / n)
    slope = np.polyfit(sim["slope"], sim["J"], 1)[0]
    se = np.sqrt(truth["var_noise"] / (n * truth["var_slope"]))
    assert abs(slope - truth["kappa_T"]) < 3 * se


def test_slope_distribution_and_intervals():
    """`dhat` has the sd `window_slope_sd` says, both intervals cover 95%, and the textbook
    OLS interval does not -- the claim section 2 of the notebook rests on."""
    n = 20_000
    X = bd.split_sessions(bd.gbm_sessions(n_days=n, rng=np.random.default_rng(6), **GBM)["S"],
                          100.0)[1]
    truth = wd.window_jump_truth(GRID, GBM["sigma"], GBM["T"], GBM["nu"])
    d = wd.window_slope(X, GRID)
    assert abs(d.std() / truth["sd_slope"] - 1) < 3 * np.sqrt(1 / (2 * n))
    tol = 3 * np.sqrt(0.95 * 0.05 / n)
    for sigma_eps in (1.0, None):
        ci = wd.window_slope_ci(X, GRID, sigma_eps=sigma_eps)
        covered = (ci[:, 0] <= truth["d_mu"]) & (truth["d_mu"] <= ci[:, 1])
        assert abs(covered.mean() - 0.95) < tol
    # The plug-in interval is a t interval on the window's 77 degrees of freedom. Coverage
    # alone cannot see the difference from t(389) at this n, so check the quantile directly.
    ci = wd.window_slope_ci(X[:5], GRID)
    half = 0.5 * (ci[:, 1] - ci[:, 0])
    unit = np.sqrt(wd.window_realised_variance(X[:5], GRID)) * wd.window_slope_sd(GRID)
    assert np.allclose(half / unit, stats.t.ppf(0.975, 77))
    on = GRID >= 0.8 - 1e-12
    s, Xw = GRID[on], X[:, on]
    resid = Xw - Xw.mean(1, keepdims=True) - np.outer(d, s - s.mean())
    se_ols = np.sqrt((resid**2).sum(1) / (s.size - 2) / ((s - s.mean()) ** 2).sum())
    assert truth["sd_slope"] / se_ols.mean() > 5


def test_realised_variance_switch():
    """The realised variance is unbiased for the GBM's, it sets the noise when switched on,
    and bad settings are refused."""
    sim = wd.window_jump_sessions(n_days=5_000, kappa=0.5, vol="realised",
                                  rng=np.random.default_rng(5), **GBM)
    df = 77                                                   # 78 increments in the window
    assert abs(sim["rv"].mean() - 1.0) < 3 * np.sqrt(2 / df / sim["rv"].size)
    assert np.allclose(sim["eta"], np.sqrt(GBM["T"] * sim["rv"]) * sim["Z"])
    with pytest.raises(ValueError):
        wd.window_jump_sessions(n_days=5, vol="garch", rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        wd.window_slope_weights(GRID, start=1.0)
