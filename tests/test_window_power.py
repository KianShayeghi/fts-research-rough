"""Tests for the realistic-signal study of the window DGP (`jump.window_power`), which
`notebooks/window_jump_power.ipynb` reads back from `artifacts/cache/`.

The first group checks the machinery on its own: the common random numbers, the nested
regressions, the tests' p-values, the exactness of the permutation calibration, and the analytic
power guide against a direct simulation. The second holds the committed caches to the item's
"done when": every exact test has size 5% to Monte Carlo error, the 1-df oracle's power is what
the analytic guide says it is, and the naive F test at a BIC-chosen M is too liberal.
"""

import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fts_research.bench import study
from fts_research.core.paths import cache_path
from fts_research.core.quadrature import quadrature_weights
from fts_research.far.far_est import fpca
from fts_research.jump import window_dgp as wd
from fts_research.jump import window_power as wp
from fts_research.truncFLR import HH, HH_A
from fts_research.truncFLR.HH import DegenerateSelection

SMALL = {"n_bars": 78, "n_power": (40,), "n_rep_size": 4, "n_rep_power": 2, "n_perm": 99}


def test_cells_and_common_random_numbers():
    """The cells carry the kappas `kappa_for_r2` gives, and moving a dataset to another kappa
    reproduces `window_jump_sessions` from the same seed: the same curves, the same jumps."""
    cells = wp.design_cells()
    r2 = cells[cells["kind"] == "r2"]
    assert np.allclose(r2["kappa"], wd.kappa_for_r2(r2["r2"].to_numpy()))
    assert np.allclose(cells.set_index("cell").loc["kappa=0.5", "r2"], 0.2703, atol=1e-4)

    data = wp.simulate(62, wp.rng_of("power", 62, 3))
    for kappa in (0.0, 0.1174, 1.0):
        sim = wd.window_jump_sessions(n_days=63, kappa=kappa, rng=wp.rng_of("power", 62, 3))
        assert np.array_equal(data["X_prev"], sim["X"][:-1])
        assert np.allclose(wp.jumps_at(data, kappa), sim["J"][1:], rtol=0, atol=1e-13)
    assert not np.array_equal(wp.simulate(62, wp.rng_of("power", 62, 4))["J0"], data["J0"])


def test_true_beta_is_the_slope_functional():
    """`true_beta` integrates against a curve to `kappa T bhat` up to the trapezium's error at
    the window's two ends, which falls tenfold on a grid ten times finer; `window_beta`, the
    grid representer, differs from it only at the close."""
    errors = []
    for n_bars in (390, 3900):
        data = wp.simulate(30, wp.rng_of("power", 30, 0), cfg={"n_bars": n_bars})
        grid = data["s"]
        integral = np.trapezoid(wp.true_beta(grid, kappa=0.5) * data["X_prev"], grid, axis=1)
        errors.append(np.abs(integral - 0.5 * 0.25 * data["slope"]).max())
    assert errors[1] < errors[0] / 5 and errors[1] < 0.01
    # The representer's least-squares weights divide by the discrete sum of squares over the
    # window's K = 79 points, K (K^2 - 1) h^2 / 12, where the line divides by (K - 1)^3 h^2 / 12.
    K = 79
    ratio = K * (K**2 - 1) / (K - 1) ** 3
    grid = np.linspace(0.0, 1.0, 391)
    rep, beta = wd.window_beta(grid, kappa=0.5), wp.true_beta(grid, kappa=0.5)
    on = grid >= 0.8 - 1e-12
    assert np.all(rep[~on] == 0) and np.all(beta[~on] == 0)
    assert np.allclose(rep[on][:-1] * ratio, beta[on][:-1], rtol=1e-12, atol=1e-12)
    assert np.isclose(rep[-1] * ratio, 2 * beta[-1], rtol=1e-12)


def test_nested_fits_are_least_squares():
    """The running sum over orthogonal scores is the residual sum of squares of every nested
    regression, for every response at once."""
    data = wp.simulate(62, wp.rng_of("power", 62, 0))
    scores = fpca(data["X_prev"], data["s"])["scores"][:, :10]
    Y = np.column_stack([wp.jumps_at(data, k) for k in (0.0, 0.3, 1.0)])
    rss = wp.nested_fits(scores, Y)
    for M in range(11):
        D = np.column_stack([np.ones(62), scores[:, :M]])
        resid = Y - D @ np.linalg.lstsq(D, Y, rcond=None)[0]
        assert np.allclose(rss[M], (resid**2).sum(axis=0), rtol=1e-10)


def test_power_tests_are_the_textbook_f_tests():
    """Each p-value is the F test of its regression computed directly, and the BIC choice is
    the BIC minimiser over M = 1..10."""
    data = wp.simulate(62, wp.rng_of("power", 62, 1))
    y = wp.jumps_at(data, 0.5)
    rows = {r["test"]: r for r in wp.power_tests(y, data["X_prev"], data["s"], data["slope"],
                                                 rng=np.random.default_rng(0))}
    scores = fpca(data["X_prev"], data["s"])["scores"]

    def f_p(D):
        D1 = np.column_stack([np.ones(62), D])
        rss1 = np.sum((y - D1 @ np.linalg.lstsq(D1, y, rcond=None)[0]) ** 2)
        rss0 = np.sum((y - y.mean()) ** 2)
        q = D1.shape[1] - 1
        return stats.f.sf((rss0 - rss1) / q / (rss1 / (62 - q - 1)), q, 62 - q - 1), rss1

    assert np.isclose(rows["slope"]["p"], f_p(data["slope"][:, None])[0], rtol=1e-8)
    assert np.isclose(rows["M8"]["p"], f_p(scores[:, :8])[0], rtol=1e-8)
    m_var = rows["var95"]["M"]
    assert np.isclose(rows["var95"]["p"], f_p(scores[:, :m_var])[0], rtol=1e-8)
    bic = [62 * np.log(f_p(scores[:, :M])[1] / 62) + (M + 1) * np.log(62) for M in range(1, 11)]
    assert rows["bic_naive"]["M"] == int(np.argmin(bic)) + 1
    assert np.isclose(rows["bic_naive"]["p"], f_p(scores[:, :rows["bic_naive"]["M"]])[0])
    assert rows["bic_perm"]["M"] == rows["bic_naive"]["M"]
    with pytest.raises(ValueError):
        wp.power_tests(y[:10], data["X_prev"][:10], data["s"], data["slope"][:10])


@pytest.mark.slow
def test_permutation_calibration_is_exact_and_the_naive_test_is_not():
    """Under the null, the permutation p-value of select-then-test lies on the grid
    {1, ..., B + 1} / (B + 1) and rejects at its nominal rate, while the same statistic read
    off the F table rejects too often: BIC chose M by looking at the jumps."""
    rng = np.random.default_rng(11)
    grid = np.linspace(0.0, 1.0, 40)
    n, n_rep, B = 40, 600, 99
    p_perm, p_naive = [], []
    for _ in range(n_rep):
        X = np.cumsum(rng.standard_normal((n, grid.size)), axis=1) / np.sqrt(grid.size)
        X[:, 0] = 0.0
        y = rng.standard_normal(n)
        tests = wp.power_tests(y, X, grid, X[:, -1], n_perm=B, rng=rng)
        rows = {r["test"]: r["p"] for r in tests}
        p_perm.append(rows["bic_perm"])
        p_naive.append(rows["bic_naive"])
    p_perm, p_naive = np.array(p_perm), np.array(p_naive)
    assert np.allclose(p_perm * (B + 1), np.round(p_perm * (B + 1)))
    assert p_perm.min() >= 1 / (B + 1)
    for level in (0.05, 0.10):
        assert abs((p_perm <= level).mean() - level) < 3 * np.sqrt(level * (1 - level) / n_rep)
    assert (p_naive <= 0.05).mean() > 0.05 + 3 * np.sqrt(0.05 * 0.95 / n_rep)


def test_analytic_power_matches_simulation():
    """`f_test_power` is the power of the F test with Gaussian regressors -- one of them, and
    three -- it is the level at no signal, and `n_for_power` inverts it. On the item's design
    it puts 80% power at 389 days for the 1-df oracle and 765 for eight components."""
    rng = np.random.default_rng(5)
    n_rep = 4000
    for n, q, f2 in ((60, 1, 0.08), (120, 3, 0.06)):
        X = rng.standard_normal((n_rep, n, q))
        gamma = np.full(q, np.sqrt(f2 / q))
        y = X @ gamma + rng.standard_normal((n_rep, n))
        D = np.concatenate([np.ones((n_rep, n, 1)), X], axis=2)
        coef = np.linalg.solve(np.einsum("rni,rnj->rij", D, D),
                               np.einsum("rni,rn->ri", D, y)[..., None])[..., 0]
        rss1 = ((y - np.einsum("rni,ri->rn", D, coef)) ** 2).sum(axis=1)
        rss0 = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        F = (rss0 - rss1) / q / (rss1 / (n - q - 1))
        rate = (F > stats.f.isf(0.05, q, n - q - 1)).mean()
        power = wp.f_test_power(n, q, f2)
        assert abs(rate - power) < 3 * wp.mc_se(power, n_rep)
    assert wp.f_test_power(100, 4, 0.0, level=0.1) == 0.1
    assert np.isclose(wp.f_test_power(wp.n_for_power(3, 0.05), 3, 0.05), 0.8)
    f2 = wp.effective_f2(0.02)
    assert np.isclose(f2, 0.02 / 0.98)
    assert round(wp.n_for_power(1, f2)) == 389
    assert round(wp.n_for_power(8, wp.effective_f2(0.02, wp.signal_share(8)))) == 765


def test_signal_share():
    """Four components hold 43% of the signal and eight 98%, the numbers section 5 of
    `window_jump_dgp.ipynb` reports, and the share rises to one."""
    share = wp.signal_share(np.arange(1, 201))
    assert np.isclose(share[3], 0.4313, atol=5e-5) and np.isclose(share[7], 0.9760, atol=5e-5)
    assert np.all(np.diff(share) >= -1e-15) and share[-1] > 0.999
    assert isinstance(wp.signal_share(8), float)


def test_power_replication_is_reproducible_and_covers_its_cells(monkeypatch):
    """A replication is a pure function of (n, rep): the same rows twice, every cell below
    `n_rep_power`, only the null above it, and every p-value in (0, 1]."""
    first = pd.DataFrame(wp.power_replication(40, 0, SMALL))
    assert first.equals(pd.DataFrame(wp.power_replication(40, 0, SMALL)))
    assert list(first["cell"]) == list(wp.design_cells(SMALL)["cell"])
    p = first[[f"p_{t}" for t in wp.TESTS]].to_numpy()
    assert np.all((p > 0) & (p <= 1))
    late = wp.power_replication(40, SMALL["n_rep_power"], SMALL)
    assert [r["cell"] for r in late] == ["R2=0"]

    df = wp.run_power(cfg=SMALL, n_jobs=1, verbose=False)
    rates = wp.rejection_rates(df, cfg=SMALL)
    assert set(rates["test"]) == set(wp.TESTS)
    null = rates[rates["cell"] == "R2=0"]
    assert (null["n_rep"] == SMALL["n_rep_size"]).all()
    assert (rates[rates["cell"] != "R2=0"]["n_rep"] == SMALL["n_rep_power"]).all()


@pytest.mark.slow
def test_methods_replication_scores_every_method():
    """One replication of the estimator comparison: a record per method, a relative error only
    where beta is not zero, and the zero-setting flags each method can actually reach."""
    cfg = {"n_bars": 78, "n_test": 20}
    null, supports = wp.methods_replication(62, 0, "R2=0", cfg)
    signal, _ = wp.methods_replication(62, 0, "kappa=1", cfg)
    assert [r["method"] for r in null] == list(wp.METHODS) == [r["method"] for r in signal]
    assert all(np.isnan(r["rise"]) for r in null)
    assert all(np.isfinite(r["rise"]) and r["ise"] >= 0 for r in signal)
    by = {r["method"]: r for r in signal}
    assert not by["FPCA-oracle"]["zeroes_any"] and by["FPCA-oracle"]["m"] == 8
    assert by["HH-A"]["zeroes_any"] and not by["HH-A"]["zeroes_all"]
    assert all(np.isclose(r["r2_test"], 1 - r["mse_test"] / r["mse_bench"]) for r in signal)
    assert set(supports) == set(wp.METHODS) and supports["FPCA-BIC"].all()


def _bench_fits(X, y, grid, rep):
    """The four sparse fits as `bench.study` makes them, for `fit_methods` to be held to."""
    lambdas = np.linspace(0.0, 1.0, wp.CONFIG["n_lambda_hh"])[1:]
    out = {}
    for name, state_fn, at_fn in (("HH-A", study.hh_a_state, study.hh_a_at),
                                  ("HH-B", study.hh_b_state, study.hh_b_at)):
        state = state_fn(X, y, grid, lambdas, mirror=True)
        fit = at_fn(state, state["lam_hat"])
        out[name] = (fit["a_hat"], fit["b_hat"], fit["support"], fit["m"], state["lam_hat"],
                     state["lam_failed"])
    ogl = study.ogl_run(X - X.mean(axis=0), y, grid, n_basis=15, n_folds=10, n_lambda=30,
                        rng=rep)
    for name, j in (("OGL-CV", ogl["j_cv"]), ("OGL-1SE", ogl["j_1se"])):
        out[name] = (None, ogl["path"]["betas"][j], ogl["path"]["supports"][j], None,
                     float(ogl["lambdas"][j]), False)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("degenerate", [False, True])
def test_fits_are_the_bench_fits(degenerate, monkeypatch):
    """`jump` may not import `bench`, so `fit_methods` calls `truncFLR` and `fdaSP` directly.
    Its fits must be `bench.study`'s to the last bit -- the cached comparison was built either
    way -- including the fall back to the smallest lambda where section 3.2 is degenerate."""
    if degenerate:
        def refuses(*args, **kwargs):
            raise DegenerateSelection("forced")
        monkeypatch.setattr(HH, "lambda_select", refuses)
        monkeypatch.setattr(HH_A, "lambda_select", refuses)
    data = wp.simulate(62, wp.rng_of("methods", 62, 0), cfg={"n_bars": 78})
    X, grid = data["X_prev"], data["s"]
    y = wp.jumps_at(data, 1.0)
    ours = wp.fit_methods(X, y, grid, rep=3)
    for name, (a_hat, beta, support, m, lam, failed) in _bench_fits(X, y, grid, 3).items():
        assert np.array_equal(ours[name]["beta"], beta), name
        assert np.array_equal(ours[name]["support"], support), name
        assert ours[name]["lam"] == lam and ours[name]["lam_failed"] == failed, name
        if a_hat is not None:
            assert ours[name]["a_hat"] == a_hat and ours[name]["m"] == m, name
    assert ours["HH-A"]["lam_failed"] == degenerate
    for name in ("OGL-CV", "OGL-1SE"):
        assert np.isclose(np.mean(y - ours[name]["a_hat"]
                                  - X @ (quadrature_weights(grid) * ours[name]["beta"])), 0)
    assert wp.implied_start(ours["HH-A"]["support"], grid) == study.implied_endpoint(
        ours["HH-A"]["support"], grid)


@pytest.mark.slow
def test_a_stopped_methods_run_resumes_to_the_same_result(tmp_path):
    """The comparison checkpoints each finished task. A run stopped part-way -- even mid-line
    -- resumes to exactly the records and supports of an uninterrupted one, and a checkpoint
    from another configuration is refused rather than mixed in."""
    cfg = {"n_bars": 78, "n_methods": (40,), "n_rep_methods": 2, "n_test": 10,
           "r2": (0.0, 0.05), "kappa_ref": ()}
    whole = tmp_path / "whole.jsonl"
    df, freq = wp.run_methods(cfg, n_jobs=1, verbose=False, checkpoint=str(whole))
    assert len(df) == 4 * len(wp.METHODS)
    assert list(df["method"][:len(wp.METHODS)]) == list(wp.METHODS)

    lines = whole.read_text(encoding="utf-8").splitlines()
    stopped = tmp_path / "stopped.jsonl"
    stopped.write_text("\n".join(lines[:3] + [lines[3][:50]]), encoding="utf-8")
    df2, freq2 = wp.run_methods(cfg, n_jobs=1, verbose=False, checkpoint=str(stopped))
    pd.testing.assert_frame_equal(df.drop(columns="seconds"), df2.drop(columns="seconds"))
    pd.testing.assert_frame_equal(freq, freq2)
    with pytest.raises(ValueError):
        wp.run_methods({**cfg, "n_test": 11}, n_jobs=1, verbose=False, checkpoint=str(whole))


# ------------------------------------------------------------------ the committed caches ---

POWER_CSV = cache_path(wp.POWER_ARTIFACT, create=False)
needs_cache = pytest.mark.skipif(not os.path.exists(POWER_CSV),
                                 reason="run scripts/make_window_power_cache.py first")


@pytest.fixture(scope="module")
def rates():
    return wp.rejection_rates(wp.cached_power())


@needs_cache
def test_cache_is_the_design(rates):
    """The cache holds every cell, n and test, with 1,000 replications at the null and 500
    elsewhere."""
    cfg = wp.CONFIG
    assert set(rates["n"]) == set(cfg["n_power"])
    assert set(rates["cell"]) == set(wp.design_cells()["cell"])
    null = rates["cell"] == "R2=0"
    assert (rates.loc[null, "n_rep"] == cfg["n_rep_size"]).all()
    assert (rates.loc[~null, "n_rep"] == cfg["n_rep_power"]).all()


@needs_cache
def test_cache_exact_tests_hold_their_size(rates):
    """Done when: size at R^2 = 0 is 5% to Monte Carlo error. With 28 (test, n) cells a 2-SE
    band is missed by chance about 1.4 times, so each cell is held to 3 SE, the count outside
    2 SE to what a binomial allows, and each test's size pooled over n to 2 SE."""
    null = rates[(rates["cell"] == "R2=0") & rates["test"].isin(wp.EXACT_TESTS)]
    se = wp.mc_se(0.05, null["n_rep"])
    assert (np.abs(null["rate"] - 0.05) <= 3 * se).all()
    outside = int((np.abs(null["rate"] - 0.05) > 2 * se).sum())
    assert stats.binom.sf(outside - 1, len(null), 0.0455) > 0.01
    pooled = null.groupby("test").apply(lambda d: np.average(d["rate"], weights=d["n_rep"]),
                                        include_groups=False)
    assert (np.abs(pooled - 0.05) <= 2 * wp.mc_se(0.05, null.groupby("test")["n_rep"].sum())).all()


@needs_cache
def test_cache_naive_bic_test_is_liberal(rates):
    """The F table at a BIC-chosen M over-rejects under the null, most at the smallest n."""
    naive = rates[(rates["cell"] == "R2=0") & (rates["test"] == "bic_naive")]
    pooled = np.average(naive["rate"], weights=naive["n_rep"])
    assert pooled > 0.05 + 3 * wp.mc_se(0.05, naive["n_rep"].sum())
    small = naive.set_index("n").loc[min(wp.CONFIG["n_power"])]
    assert small["rate"] > 0.05 + 3 * small["se"]


@needs_cache
def test_cache_oracle_power_is_the_analytic_power(rates):
    """Done when: the 1-df oracle's power at R^2 = 0.02 is the analytic guide's -- about 80% at
    390 days -- and the eight-component test's is about 80% near 765. The first is exact, so
    every realistic cell is held to it at 3.5 SE (about 21 cells); the second uses estimated
    components, so it is held to the guide at the two guide points only."""
    slope = rates[(rates["test"] == "slope") & rates["cell"].str.startswith("R2=")]
    guide = np.array([wp.f_test_power(n, 1, wp.effective_f2(r2))
                      for n, r2 in zip(slope["n"], slope["r2"])])
    se = wp.mc_se(guide, slope["n_rep"])
    assert (np.abs(slope["rate"] - guide) <= 3.5 * np.maximum(se, 1e-3)).all()

    at = rates.set_index(["cell", "test", "n"])["rate"]
    assert abs(at[("R2=0.02", "slope", 390)] - 0.80) < 3 * wp.mc_se(0.8, 500)
    share = wp.signal_share(8)
    for n in (390, 765):
        guide = wp.f_test_power(n, 8, wp.effective_f2(0.02, share))
        assert abs(at[("R2=0.02", "M8", n)] - guide) < 3 * wp.mc_se(guide, 500)
