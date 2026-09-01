"""Tests for `fdaSP.SBC`, the locally sparse scalar-on-function estimator.

The interesting properties here are mathematical rather than structural: the basis must be
a genuine B-spline basis, the ADMM output must actually minimise (62.3), the penalty must
produce contiguous null regions, and the CV machinery must respect both its own selection
rule and its fold structure.
"""

import numpy as np
import pytest

from conftest import ogl_objective
from fts_research.fdaSP import SBC


# --------------------------------------------------------------------------- basis ------

def test_bspline_design_is_the_trapezoid_integral():
    """W[i, j] must equal int x_i(t) phi_j(t) dt by the trapezium rule."""
    rng = np.random.default_rng(0)
    grid = np.linspace(0, 1, 121)
    X = rng.standard_normal((5, grid.size))
    W, Phi = SBC.bspline_design(X, grid, n_basis=9, order=4)
    expected = np.trapezoid(X[:, None, :] * Phi.T[None, :, :], grid, axis=2)
    np.testing.assert_allclose(W, expected, atol=1e-12)


# ------------------------------------------------------------------------- penalty ------

def test_pen_matrices_group_structure():
    F, groups = SBC.pen_matrices(15, order=4)
    assert len(groups) == 15 - 4 + 1
    assert F.shape == (len(groups) * 4, 15)
    for b, g in enumerate(groups):
        np.testing.assert_array_equal(g, np.arange(b, b + 4))
        block = F[b * 4:(b + 1) * 4]
        # Each row of a block selects exactly one coefficient, the one in its group.
        assert np.count_nonzero(block) == 4
        np.testing.assert_array_equal(np.flatnonzero(block.any(axis=0)), g)


def test_pen_matrices_every_coefficient_is_penalised():
    """No basis coefficient may escape the penalty, or it could never be zeroed."""
    F, _ = SBC.pen_matrices(15, order=4)
    assert np.all(np.abs(F).sum(axis=0) > 0)


@pytest.mark.parametrize("bad", [dict(group_weights=np.ones(3)),
                                 dict(group_weights=-np.ones(12)),
                                 dict(var_weights=np.ones(3)),
                                 dict(var_weights=-np.ones(15))])
def test_pen_matrices_validates_weights(bad):
    with pytest.raises(ValueError):
        SBC.pen_matrices(15, order=4, **bad)


def test_group_soft_th_thresholds_and_shrinks():
    x = np.array([3.0, 4.0])                     # norm 5
    np.testing.assert_array_equal(SBC.group_soft_th(x, 5.0), np.zeros(2))
    np.testing.assert_array_equal(SBC.group_soft_th(x, 6.0), np.zeros(2))
    np.testing.assert_allclose(SBC.group_soft_th(x, 1.0), x * (1 - 1 / 5))
    np.testing.assert_allclose(SBC.group_soft_th(x, 0.0), x)


def test_group_soft_th_is_the_prox_of_the_l2_norm():
    """S_k(v) must minimise 1/2||z - v||^2 + k||z||, checked against a fine 2-D search."""
    rng = np.random.default_rng(3)
    v = rng.standard_normal(2) * 2
    kappa = 0.7
    z_star = SBC.group_soft_th(v, kappa)

    def f(z):
        return 0.5 * np.sum((z - v) ** 2) + kappa * np.linalg.norm(z)

    perturb = rng.standard_normal((500, 2)) * 0.3
    assert f(z_star) <= np.min([f(z_star + p) for p in perturb]) + 1e-12


# ---------------------------------------------------------------------------- ADMM ------

def test_admm_solves_the_penalised_problem(sbc_data):
    """The returned psi must be a genuine minimiser of the objective, not just a fixed point."""
    W, _ = SBC.bspline_design(sbc_data["X"], sbc_data["grid"], n_basis=12, order=4)
    F, _ = SBC.pen_matrices(12, order=4)
    lam = 0.05
    out = SBC.admm(W, sbc_data["y"], F, lam, order=4, abstol=1e-10, reltol=1e-8, max_iter=20000)
    assert out["converged"]

    f0 = ogl_objective(W, sbc_data["y"], F, lam, out["psi"])
    rng = np.random.default_rng(5)
    for scale in (1e-3, 1e-2, 1e-1):
        for _ in range(20):
            step = rng.standard_normal(out["psi"].size) * scale
            assert ogl_objective(W, sbc_data["y"], F, lam, out["psi"] + step) >= f0 - 1e-9


def test_admm_zero_lambda_is_least_squares(sbc_data):
    W, _ = SBC.bspline_design(sbc_data["X"], sbc_data["grid"], n_basis=10, order=4)
    F, _ = SBC.pen_matrices(10, order=4)
    out = SBC.admm(W, sbc_data["y"], F, 0.0, order=4, abstol=1e-12, reltol=1e-10, max_iter=20000)
    ols, *_ = np.linalg.lstsq(W, sbc_data["y"], rcond=None)
    np.testing.assert_allclose(out["psi"], ols, atol=1e-6)


def test_admm_large_lambda_kills_everything(sbc_data):
    W, _ = SBC.bspline_design(sbc_data["X"], sbc_data["grid"], n_basis=10, order=4)
    F, _ = SBC.pen_matrices(10, order=4)
    out = SBC.admm(W, sbc_data["y"], F, 1e6, order=4, max_iter=5000)
    # zeta is the exactly-sparse variable; psi only reaches zero to solver tolerance.
    np.testing.assert_array_equal(out["zeta"], np.zeros_like(out["zeta"]))
    np.testing.assert_allclose(out["psi"], 0.0, atol=1e-5)


def test_admm_scalar_covariates_are_projected_out(sbc_data):
    """With Z present, gamma must be the OLS of the partial residual on Z."""
    n = sbc_data["y"].size
    rng = np.random.default_rng(2)
    Z = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = sbc_data["y"] + Z @ np.array([2.0, -1.5])

    W, _ = SBC.bspline_design(sbc_data["X"], sbc_data["grid"], n_basis=10, order=4)
    F, _ = SBC.pen_matrices(10, order=4)
    out = SBC.admm(W, y, F, 0.02, Z=Z, order=4, abstol=1e-10, reltol=1e-8, max_iter=20000)

    expected, *_ = np.linalg.lstsq(Z, y - W @ out["psi"], rcond=None)
    np.testing.assert_allclose(out["gamma"], expected, atol=1e-6)


def test_admm_rejects_mismatched_Z(sbc_data):
    W, _ = SBC.bspline_design(sbc_data["X"], sbc_data["grid"], n_basis=10, order=4)
    F, _ = SBC.pen_matrices(10, order=4)
    with pytest.raises(ValueError, match="rows but y has"):
        SBC.admm(W, sbc_data["y"], F, 0.1, Z=np.ones((3, 1)), order=4)


def test_admm_warm_start_reaches_the_same_solution(sbc_data):
    W, _ = SBC.bspline_design(sbc_data["X"], sbc_data["grid"], n_basis=12, order=4)
    F, _ = SBC.pen_matrices(12, order=4)
    kw = dict(order=4, abstol=1e-10, reltol=1e-8, max_iter=20000)
    cold = SBC.admm(W, sbc_data["y"], F, 0.05, **kw)
    seed = SBC.admm(W, sbc_data["y"], F, 0.08, **kw)
    warm = SBC.admm(W, sbc_data["y"], F, 0.05, init=seed, rho=seed["rho"], **kw)
    np.testing.assert_allclose(warm["psi"], cold["psi"], atol=1e-6)
    assert warm["converged"]


# --------------------------------------------------------------------- support/regions ---

def test_find_regions_on_a_known_mask():
    grid = np.linspace(0, 1, 11)
    mask = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    assert SBC.find_regions(mask, grid) == [(grid[2], grid[4]), (grid[7], grid[8])]


def test_find_regions_edges_and_empty():
    grid = np.linspace(0, 1, 5)
    assert SBC.find_regions(np.zeros(5, bool), grid) == []
    assert SBC.find_regions(np.ones(5, bool), grid) == [(0.0, 1.0)]
    assert SBC.find_regions(np.array([1, 0, 0, 0, 1], bool), grid) == [(0.0, 0.0), (1.0, 1.0)]


def test_find_regions_partitions_the_mask():
    """Null and support regions must together cover the grid and never overlap."""
    rng = np.random.default_rng(1)
    grid = np.linspace(0, 1, 60)
    mask = rng.random(60) < 0.4
    covered = np.zeros(60, bool)
    for lo, hi in SBC.find_regions(mask, grid) + SBC.find_regions(~mask, grid):
        covered |= (grid >= lo) & (grid <= hi)
    assert covered.all()


# ----------------------------------------------------------------------------- fit ------

def test_fit_support_is_consistent_with_beta(sbc_data):
    res = SBC.fit(sbc_data["X"], sbc_data["y"] - sbc_data["y"].mean(), sbc_data["grid"],
                  lam=0.5, n_basis=12, order=4, abstol=1e-9, reltol=1e-7, max_iter=20000)
    # Wherever the support flag is False the coefficient function must be exactly zero.
    np.testing.assert_array_equal(res["beta"][~res["support"]], 0.0)
    # Null and support regions partition the domain.
    assert len(res["null_regions"]) + len(res["support_regions"]) >= 1
    for lo, hi in res["null_regions"]:
        seg = (sbc_data["grid"] >= lo) & (sbc_data["grid"] <= hi)
        np.testing.assert_allclose(res["beta"][seg], 0.0, atol=0)


def test_zero_coefficients_uses_the_any_rule():
    """A zeroed group kills every coefficient it contains, not only those it alone contains."""
    _, groups = SBC.pen_matrices(12, order=4)
    zeta = np.ones(len(groups) * 4)
    zeta[4 * 4:5 * 4] = 0.0                       # switch off group 4 = {4, 5, 6, 7}
    is_zero = SBC.zero_coefficients(zeta, groups, 4)
    np.testing.assert_array_equal(np.flatnonzero(is_zero), np.arange(4, 8))


def test_zero_coefficients_matches_the_solver(sbc_data):
    """Every coefficient the rule calls zero must be numerically zero in the ADMM psi, and
    every one it calls non-zero must be genuinely non-zero."""
    W, _ = SBC.bspline_design(sbc_data["X"], sbc_data["grid"], n_basis=12, order=4)
    F, groups = SBC.pen_matrices(12, order=4)
    y = sbc_data["y"] - sbc_data["y"].mean()
    out = SBC.admm(W, y, F, 0.2, order=4, abstol=1e-11, reltol=1e-9, max_iter=50000)
    assert out["converged"]

    is_zero = SBC.zero_coefficients(out["zeta"], groups, 4)
    assert is_zero.any() and not is_zero.all(), "need a partially sparse solution to test"
    assert np.max(np.abs(out["psi"][is_zero])) < 1e-8
    assert np.min(np.abs(out["psi"][~is_zero])) > 1e-4


def test_fit_makes_the_null_region_exactly_zero(sbc_data):
    """The estimator's selling point: beta is *identically* zero on the null region."""
    grid = sbc_data["grid"]
    res = SBC.fit(sbc_data["X"], sbc_data["y"] - sbc_data["y"].mean(), grid,
                  lam=0.2, n_basis=12, order=4, abstol=1e-11, reltol=1e-9, max_iter=50000)
    assert res["null_regions"], "expected a null region at this lambda"
    for lo, hi in res["null_regions"]:
        seg = (grid >= lo) & (grid <= hi)
        np.testing.assert_array_equal(res["beta"][seg], 0.0)   # exact, not approximate
    assert res["support_regions"], "expected some support to survive"


def test_penalty_reduces_error_on_the_true_null_region(sbc_data):
    """Against an unpenalised spline fit, the OGL must do better where psi1 is truly zero."""
    grid = sbc_data["grid"]
    y = sbc_data["y"] - sbc_data["y"].mean()
    true_null = (grid >= 0.3) & (grid <= 0.7)
    kw = dict(n_basis=12, order=4, abstol=1e-10, reltol=1e-8, max_iter=30000)

    unpen = SBC.fit(sbc_data["X"], y, grid, lam=0.0, **kw)
    pen = SBC.fit(sbc_data["X"], y, grid, lam=0.2, **kw)

    def err(b):
        return np.trapezoid((b[true_null] - sbc_data["beta"][true_null]) ** 2, grid[true_null])

    assert err(pen["beta"]) < err(unpen["beta"])


def test_predict_matches_fitted_values(sbc_data):
    res = SBC.fit(sbc_data["X"], sbc_data["y"], sbc_data["grid"], lam=0.1, n_basis=10, order=4)
    np.testing.assert_allclose(SBC.predict(res, sbc_data["X"], sbc_data["grid"]),
                               res["fitted"], atol=1e-12)


def test_predict_with_covariates(sbc_data):
    n = sbc_data["y"].size
    Z = np.ones((n, 1))
    res = SBC.fit(sbc_data["X"], sbc_data["y"], sbc_data["grid"], lam=0.1,
                  Z=Z, n_basis=10, order=4)
    np.testing.assert_allclose(SBC.predict(res, sbc_data["X"], sbc_data["grid"], Z=Z),
                               res["fitted"], atol=1e-12)


# ---------------------------------------------------------------------------- path ------

def test_lambda_max_is_the_switch_off_point(sbc_data):
    W, _ = SBC.bspline_design(sbc_data["X"], sbc_data["grid"], n_basis=10, order=4)
    F, _ = SBC.pen_matrices(10, order=4)
    lmax = SBC.lambda_max(W, sbc_data["y"], F, order=4)
    assert lmax > 0

    at = SBC.admm(W, sbc_data["y"], F, lmax, order=4, max_iter=5000)
    below = SBC.admm(W, sbc_data["y"], F, lmax * 0.5, order=4, max_iter=5000)
    assert np.all(at["zeta"] == 0.0), "solution must be null at lambda_max"
    assert np.any(below["zeta"] != 0.0), "solution must be non-null below lambda_max"


def test_lambda_path_is_descending_and_geometric(sbc_data):
    W, _ = SBC.bspline_design(sbc_data["X"], sbc_data["grid"], n_basis=10, order=4)
    F, _ = SBC.pen_matrices(10, order=4)
    lams = SBC.lambda_path(W, sbc_data["y"], F, order=4, n_lambda=25, eps=1e-3)
    assert lams.size == 25
    assert np.all(np.diff(lams) < 0)
    np.testing.assert_allclose(lams[-1] / lams[0], 1e-3, rtol=1e-9)
    ratios = lams[1:] / lams[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)


def test_fit_path_sparsity_increases_with_lambda(sbc_data):
    path = SBC.fit_path(sbc_data["X"], sbc_data["y"] - sbc_data["y"].mean(), sbc_data["grid"],
                        n_basis=12, order=4, n_lambda=20,
                        abstol=1e-9, reltol=1e-7, max_iter=20000)
    assert np.all(np.diff(path["lambdas"]) < 0), "path must be sorted descending"
    # betas are exactly sparse, so this needs no threshold.
    zero_frac = np.mean(path["betas"] == 0.0, axis=1)
    assert zero_frac[0] == 1.0, "the largest lambda on the path must zero everything"
    # Sparsity must decrease monotonically as the penalty is relaxed.
    assert np.all(np.diff(zero_frac) <= 0)
    # The reported supports must agree with the betas they came from.
    for j in range(path["lambdas"].size):
        np.testing.assert_array_equal(path["betas"][j][~path["supports"][j]], 0.0)


def test_fit_path_matches_individual_fits(sbc_data):
    """Warm starting must not change the answer, only the iteration count."""
    lams = np.array([1.0, 0.3, 0.1])
    kw = dict(n_basis=10, order=4, abstol=1e-10, reltol=1e-8, max_iter=20000)
    path = SBC.fit_path(sbc_data["X"], sbc_data["y"], sbc_data["grid"], lambdas=lams, **kw)
    for j, lam in enumerate(lams):
        single = SBC.fit(sbc_data["X"], sbc_data["y"], sbc_data["grid"], lam=lam, **kw)
        np.testing.assert_allclose(path["psis"][j], single["psi"], atol=1e-5)


def test_fit_path_accepts_unsorted_lambdas(sbc_data):
    lams = np.array([0.1, 1.0, 0.3])
    path = SBC.fit_path(sbc_data["X"], sbc_data["y"], sbc_data["grid"],
                        lambdas=lams, n_basis=10, order=4)
    np.testing.assert_allclose(path["lambdas"], np.array([1.0, 0.3, 0.1]))


# ------------------------------------------------------------------------------ CV ------

@pytest.mark.slow
def test_cv_fit_selection_rules(sbc_data):
    kw = dict(n_basis=12, order=4, n_folds=5, n_lambda=20, rng=4)
    cv = SBC.cv_fit(sbc_data["X"], sbc_data["y"] - sbc_data["y"].mean(), sbc_data["grid"],
                    one_se=False, **kw)
    cv1 = SBC.cv_fit(sbc_data["X"], sbc_data["y"] - sbc_data["y"].mean(), sbc_data["grid"],
                     one_se=True, **kw)

    j_min = int(np.argmin(cv["cv_mean"]))
    assert cv["lam_hat"] == cv["lambdas"][j_min] == cv["lam_min"]
    assert np.all(cv["cv_se"] >= 0)

    # The 1-SE rule must pick a *larger* penalty that is still within one SE of the minimum.
    assert cv1["lam_hat"] >= cv["lam_hat"]
    j_opt = int(np.flatnonzero(cv1["lambdas"] == cv1["lam_hat"])[0])
    assert cv1["cv_mean"][j_opt] <= cv1["cv_mean"][j_min] + cv1["cv_se"][j_min] + 1e-12


@pytest.mark.slow
def test_cv_fit_is_reproducible(sbc_data):
    kw = dict(n_basis=10, order=4, n_folds=5, n_lambda=12, rng=99)
    a = SBC.cv_fit(sbc_data["X"], sbc_data["y"], sbc_data["grid"], **kw)
    b = SBC.cv_fit(sbc_data["X"], sbc_data["y"], sbc_data["grid"], **kw)
    np.testing.assert_allclose(a["cv_mean"], b["cv_mean"])
    assert a["lam_hat"] == b["lam_hat"]


# --------------------------------------------------------------- scalar covariates -------

@pytest.fixture(scope="module")
def sbc_data_with_Z(sbc_data):
    """The same curves, plus an intercept and one scalar covariate with known coefficients."""
    n = sbc_data["y"].size
    rng = np.random.default_rng(0)
    Z = np.column_stack([np.ones(n), rng.standard_normal(n)])
    gamma = np.array([3.0, -2.0])
    return {**sbc_data, "Z": Z, "gamma": gamma, "y": sbc_data["y"] + Z @ gamma}


def test_lambda_max_with_covariates_residualises_first(sbc_data_with_Z):
    """Z must be projected out before the switch-off point is found, or an intercept alone
    would keep the solution non-null."""
    d = sbc_data_with_Z
    W, _ = SBC.bspline_design(d["X"], d["grid"], n_basis=10, order=4)
    F, _ = SBC.pen_matrices(10, order=4)
    lmax = SBC.lambda_max(W, d["y"], F, Z=d["Z"], order=4)

    resid = d["y"] - d["Z"] @ np.linalg.lstsq(d["Z"], d["y"], rcond=None)[0]
    lmax_direct = SBC.lambda_max(W, resid, F, order=4)
    # The two routes differ only by the bisection's own tolerance.
    assert lmax == pytest.approx(lmax_direct, rel=1e-2)


def test_fit_path_with_covariates_tracks_gamma(sbc_data_with_Z):
    d = sbc_data_with_Z
    path = SBC.fit_path(d["X"], d["y"], d["grid"], Z=d["Z"], n_basis=10, order=4, n_lambda=10)
    assert path["gammas"].shape == (path["lambdas"].size, 2)
    assert np.all(np.isfinite(path["gammas"]))


@pytest.mark.slow
def test_cv_fit_with_covariates_recovers_their_coefficients(sbc_data_with_Z):
    d = sbc_data_with_Z
    cv = SBC.cv_fit(d["X"], d["y"], d["grid"], Z=d["Z"], n_basis=10, order=4,
                    n_folds=5, n_lambda=12, rng=0)
    np.testing.assert_allclose(cv["gamma"], d["gamma"], atol=0.15)
    # The fitted values must include the Z contribution, not just the functional term.
    assert np.sqrt(np.mean((d["y"] - cv["fitted"]) ** 2)) < 0.5


# ------------------------------------------------------------------------- CV folds ------


def test_blocked_folds_are_contiguous_and_balanced():
    folds = SBC.blocked_folds(97, 5)
    assert folds.shape == (97,)
    assert sorted(np.unique(folds)) == [0, 1, 2, 3, 4]
    for f in np.unique(folds):
        idx = np.flatnonzero(folds == f)
        assert np.all(np.diff(idx) == 1), "fold is not one contiguous block"
    sizes = np.bincount(folds)
    assert sizes.max() - sizes.min() <= 1


def test_blocked_folds_are_in_time_order():
    """Fold labels increase along the series, so every fold is a stretch of consecutive time."""
    folds = SBC.blocked_folds(50, 5)
    assert np.all(np.diff(folds) >= 0)


@pytest.mark.parametrize("bad", [1, 0, 51])
def test_blocked_folds_rejects_impossible_fold_counts(bad):
    with pytest.raises(ValueError):
        SBC.blocked_folds(50, bad)


def test_embargo_mask_drops_neighbours_of_the_test_block():
    folds = SBC.blocked_folds(30, 3)              # fold 1 is indices 10..19
    train = SBC.embargo_mask(folds, 1, embargo=3)
    assert not train[10:20].any(), "test observations must never be trained on"
    assert not train[7:10].any() and not train[20:23].any(), "embargo not applied"
    assert train[:7].all() and train[23:].all(), "embargo reaches too far"


def test_embargo_mask_of_zero_is_the_plain_complement():
    folds = SBC.blocked_folds(30, 3)
    np.testing.assert_array_equal(SBC.embargo_mask(folds, 1, 0), folds != 1)


def test_cv_fit_accepts_explicit_folds(sbc_data):
    X, y, grid = sbc_data["X"], sbc_data["y"], sbc_data["grid"]
    folds = SBC.blocked_folds(len(y), 4)
    out = SBC.cv_fit(X, y, grid, n_basis=10, n_folds=4, n_lambda=6, folds=folds, embargo=2)
    np.testing.assert_array_equal(out["folds"], folds)
    assert out["embargo"] == 2
    assert out["cv_mean"].shape == out["lambdas"].shape
    assert np.all(np.isfinite(out["cv_mean"]))


def test_cv_fit_rejects_folds_of_the_wrong_length(sbc_data):
    X, y, grid = sbc_data["X"], sbc_data["y"], sbc_data["grid"]
    with pytest.raises(ValueError):
        SBC.cv_fit(X, y, grid, n_basis=10, n_lambda=4, folds=np.zeros(len(y) + 1, int))


def test_random_folds_split_overlapping_windows_across_the_fold_boundary():
    """Random folds put near-duplicate windows on both sides of the split; blocked ones do not.

    This is the structural property, and it holds by construction. Whether it costs anything
    is a separate question that depends on the estimator and on where the duplication comes
    from. A penalised linear fit cannot memorise an individual window, so overlapping windows
    cost it little. Duplication across a *panel* is another matter: in
    `notebooks/overnight_jumps.ipynb` nine instruments contribute a row each on every date
    and share that date's market-wide move, and random folds there report more than twice the
    out-of-sample R-squared that date-blocked folds do.
    """
    n, W, H, n_folds = 300, 20, 5, 10
    rng = np.random.default_rng(0)
    random_folds = rng.permutation(n) % n_folds
    blocked = SBC.blocked_folds(n, n_folds)

    def shared(folds, embargo):
        """Number of (train, test) pairs whose raw observation windows overlap."""
        total = 0
        for f in np.unique(folds):
            test = np.flatnonzero(folds == f)
            train = np.flatnonzero(SBC.embargo_mask(folds, f, embargo))
            total += int(sum(np.any(np.abs(test - i) < W + H) for i in train))
        return total

    assert shared(random_folds, 0) > 1000, "random folds should overlap heavily"
    assert shared(blocked, W + H) == 0, "blocked folds with an embargo must overlap not at all"
    assert shared(blocked, 0) > 0, "without the embargo, block edges still overlap"


@pytest.mark.slow
def test_blocked_folds_never_train_on_a_window_overlapping_the_test_block():
    """With an embargo of width + horizon, no training window shares a raw observation with
    any test window, which is the property random folds break."""
    W, H, n_folds = 20, 5, 5
    n = 200
    folds = SBC.blocked_folds(n, n_folds)
    for f in range(n_folds):
        test = np.flatnonzero(folds == f)
        train = np.flatnonzero(SBC.embargo_mask(folds, f, W + H))
        # window i covers raw observations [i, i + W + H)
        for i in train:
            assert not np.any(np.abs(test - i) < W + H)
