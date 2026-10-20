"""Tests for the joint jump-and-curve model: `jump.joint_dgp` and `jump.joint_est`.

The model is checked against its own definitions before any estimator is checked against it.
The identities the notes build on must hold to machine precision on every simulated path; the
boundary conditions must be enforced rather than hoped for; the stationary mean must solve the
notes' expectation equation *and* the jump-mean equation it leaves implicit; and the
population moments `moments` solves for must be the ones a long simulation produces, which is
what makes them usable as the truth in everything after. Only then are the estimators held to
known parameters, to their own standard errors, and -- for the shape test -- to its nominal
size.
"""

import numpy as np
import pytest
from scipy import stats

from fts_research.core.quadrature import quadrature_weights
from fts_research.jump import joint_dgp as jd
from fts_research.jump import joint_est as est


@pytest.fixture(scope="module")
def model():
    return jd.notes_design()


@pytest.fixture(scope="module")
def random_gamma_model():
    return jd.notes_design(gamma=0.5, gamma_dist="uniform")


@pytest.fixture(scope="module")
def long_path(model):
    return jd.simulate(model, 100_000, rng=11)


# ------------------------------------------------------------------- identities and checks


def test_simulated_curves_start_at_zero_and_rebuild_the_prices(model):
    sim = jd.simulate(model, 300, rng=0)
    J, X = sim["J"], sim["X"]
    assert np.all(X[:, 0] == 0.0), "the CIDR curve must start at exactly zero"

    P = jd.price_curves(J, X, p0=4.6)
    np.testing.assert_allclose(P[0, 0] - 4.6, J[0], atol=1e-12)
    np.testing.assert_allclose(P[1:, 0] - P[:-1, -1], J[1:], atol=1e-12)    # J_t
    np.testing.assert_allclose(P - P[:, [0]], X, atol=1e-12)                # X_t(s)


def test_the_curve_equation_holds_on_the_simulated_path(model):
    """Recompute every curve from its own equation, with the draws the simulator made."""
    sim = jd.simulate(model, 50, rng=1)
    J, X, gamma, eps = sim["J"], sim["X"], sim["gamma"], sim["eps"]
    rebuilt = model.mu - gamma[1:, None] * model.G[None, :] * J[1:, None] \
        + X[:-1] @ model.Psi_op.T + eps[1:]
    np.testing.assert_allclose(rebuilt, X[1:], atol=1e-12)
    jump = model.alpha + model.rho * J[:-1] + X[:-1] @ (model.beta * model.w) + sim["eta"][1:]
    np.testing.assert_allclose(jump, J[1:], atol=1e-12)


@pytest.mark.parametrize("change,match", [
    ({"mu": lambda g: 0.1 + g}, "mu"),
    ({"G": lambda g: 1.0 + g}, "G"),
    ({"kernel": lambda g: np.ones((g.size, g.size))}, "Psi"),
])
def test_a_model_that_would_move_the_open_is_refused(model, change, match):
    (name, build), = change.items()
    with pytest.raises(ValueError, match=match):
        model.with_params(**{name: build(model.grid)})


def test_a_fixed_gamma_cannot_carry_a_variance(model):
    with pytest.raises(ValueError, match="fixed"):
        model.with_params(gamma_var=0.1)


@pytest.mark.parametrize("c", [0.01, 0.1, 1.0, 100.0])
def test_the_notes_kernel_is_inside_the_bound_at_every_bandwidth(c):
    grid = np.linspace(0, 1, 101)
    K = jd.notes_kernel(grid, c=c)
    w = quadrature_weights(grid)
    hs = np.sqrt(np.sum(K ** 2 * w[:, None] * w[None, :]))
    assert hs <= np.sqrt(1 / 3) + 1e-3
    assert np.all(K[0] == 0.0)


def test_the_return_onset_is_the_late_session_return():
    grid = np.linspace(0, 1, 79)
    beta = jd.notes_beta(grid, "return", onset=0.85, amplitude=2.0)
    X = np.random.default_rng(0).standard_normal((20, grid.size)).cumsum(axis=1)
    start = int(np.argmax(grid >= 0.85))
    np.testing.assert_allclose(X @ (beta * quadrature_weights(grid)),
                               2.0 * (X[:, -1] - X[:, start]), atol=1e-10)


def test_the_front_loaded_profile_is_normalised():
    grid = np.linspace(0, 1, 79)
    G = jd.gapfill_profile(grid, "front", tau=0.1)
    assert G[0] == 0.0 and np.isclose(G[-1], 1.0)
    assert G[int(0.3 * 78)] > 0.9, "most of the gap should be filled early"


# ------------------------------------------------------------------------------ the noise


def test_the_u_shape_has_the_stated_ratio_and_integral():
    v = np.linspace(0, 1, 2001)
    s2 = jd.intraday_variance(v, level=0.7, ratio=2.5)
    assert np.isclose(s2[0] / s2[1000], 2.5) and np.isclose(s2[-1] / s2[1000], 2.5)
    assert np.isclose(np.trapezoid(s2, v), 0.7, rtol=1e-6)


def test_cell_variances_integrate_the_spot_variance():
    grid = np.linspace(0, 1, 79)
    np.testing.assert_allclose(jd.cell_variances(grid, 0.4), 0.4 * np.diff(grid))
    cells = jd.cell_variances(grid, lambda v: jd.intraday_variance(v, 0.7, 3.0))
    assert np.isclose(cells.sum(), 0.7, rtol=1e-12)
    fine = np.linspace(grid[10], grid[11], 10_001)
    assert np.isclose(cells[10], np.trapezoid(jd.intraday_variance(fine, 0.7, 3.0), fine))


def test_the_curve_noise_has_brownian_covariance(model, long_path):
    eps = long_path["eps"]
    assert np.all(eps[:, 0] == 0.0)
    target = jd.noise_covariance(model.cell_var)
    sample = np.cov(eps[:, ::6], rowvar=False)
    np.testing.assert_allclose(sample, target[::6, ::6], atol=0.02 * target.max())
    increments = np.diff(eps, axis=1)
    corr = np.corrcoef(increments[:, ::7], rowvar=False)
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert np.max(np.abs(off)) < 5 / np.sqrt(eps.shape[0]), "increments must be independent"


def test_t_innovations_have_the_stated_variance_and_tails():
    m = jd.notes_design(eta_df=6.0)
    eta = jd.simulate(m, 200_000, rng=3)["eta"]
    assert np.isclose(eta.std(), m.sigma_eta, rtol=0.02)
    assert np.isclose(stats.kurtosis(eta), 6.0 / (6.0 - 4.0), rtol=0.3)


def test_uniform_gamma_has_the_stated_moments(random_gamma_model):
    gamma = jd.simulate(random_gamma_model, 50_000, rng=4)["gamma"]
    assert gamma.min() >= 0.0 and gamma.max() <= 1.0
    assert np.isclose(gamma.mean(), 0.5, atol=0.01)
    assert np.isclose(gamma.var(), 1 / 12, rtol=0.03)


# ------------------------------------------------------------------------------ the theory


def test_without_psi_the_radius_is_the_effective_persistence(model):
    """With `Psi = 0` the jump's persistence is `rho - gamma <beta, G>`, in closed form."""
    flat = model.with_params(kernel=np.zeros_like(model.kernel))
    effective = model.rho - model.gamma_mean * np.sum(model.beta * model.G * model.w)
    assert np.isclose(jd.stationarity(flat)["radius"], abs(effective), rtol=1e-10)


def test_stationarity_is_a_joint_condition(model):
    """Each equation stationary on its own, the pair explosive: momentum feeding back."""
    coupled = model.with_params(rho=0.9, gamma_mean=-1.0,
                                kernel=np.zeros_like(model.kernel), beta=2.0 * model.beta)
    check = jd.stationarity(coupled)
    assert check["decoupled"] < 1.0
    assert check["radius"] > 1.0 and not check["stationary"]
    with pytest.raises(ValueError, match="stationary"):
        jd.moments(coupled)


def test_a_fixed_gamma_has_mean_square_radius_the_square_of_the_radius(model):
    check = jd.stationarity(model)
    assert np.isclose(check["ms_radius"], check["radius"] ** 2, rtol=1e-6)


def test_the_mean_square_radius_is_that_of_the_dense_kronecker_operator():
    """On a coarse grid `E[A (x) A]` is small enough to build and decompose directly."""
    m = jd.notes_design(n_grid=21, gamma=0.5, gamma_dist="uniform")
    ops = m.operators()
    A, EB = ops["L_bar"] @ ops["B"], ops["E"] @ ops["B"]
    dense = np.kron(A, A) + m.gamma_var * np.kron(EB, EB)
    assert jd.stationarity(m)["ms_radius"] == pytest.approx(
        np.max(np.abs(np.linalg.eigvals(dense))), rel=1e-9)


def test_a_random_gamma_raises_the_mean_square_radius(model, random_gamma_model):
    fixed = random_gamma_model.with_params(gamma_var=0.0, gamma_dist="fixed")
    assert jd.stationarity(random_gamma_model)["ms_radius"] > \
        jd.stationarity(fixed)["ms_radius"]


@pytest.mark.parametrize("which", ["model", "random_gamma_model"])
def test_the_stationary_mean_solves_both_expectation_equations(which, request):
    m = request.getfixturevalue(which)
    mom = jd.moments(m)
    EJ, mX = mom["mean_J"], mom["mean_X"]
    curve = m.mu - m.gamma_mean * m.G * EJ + m.Psi_op @ mX       # the notes' equation
    jump = m.alpha + m.rho * EJ + np.sum(m.beta * m.w * mX)      # the one left implicit
    np.testing.assert_allclose(curve, mX, atol=1e-12)
    assert np.isclose(jump, EJ, atol=1e-12)
    assert mX[0] == pytest.approx(0.0, abs=1e-14)


def test_population_moments_match_a_long_simulation(model, long_path):
    mom = jd.moments(model)
    J, X = long_path["J"], long_path["X"]
    n = J.size
    assert abs(J.mean() - mom["mean_J"]) < 4 * np.sqrt(mom["var_J"] / n)
    assert np.isclose(J.var(), mom["var_J"], rtol=0.02)
    assert np.isclose(X[:, -1].var(), mom["cov_X"][-1, -1], rtol=0.02)
    assert np.isclose(np.cov(J[1:], X[:-1, -1])[0, 1], mom["lag1"][0, -1], atol=0.01)
    assert abs(np.corrcoef(J[:-1], J[1:])[0, 1] - mom["acf1_J"]) < 4 / np.sqrt(n)
    np.testing.assert_allclose(np.cov(X[:, ::13], rowvar=False),
                               mom["cov_X"][::13, ::13], atol=0.02)


def test_population_moments_match_a_long_simulation_with_random_gamma(random_gamma_model):
    mom = jd.moments(random_gamma_model)
    sim = jd.simulate(random_gamma_model, 100_000, rng=12)
    assert np.isclose(sim["J"].var(), mom["var_J"], rtol=0.02)
    assert np.isclose(sim["X"][:, -1].var(), mom["cov_X"][-1, -1], rtol=0.02)
    assert np.isclose(np.var(sim["X"][:, 39]), mom["cov_X"][39, 39], rtol=0.02)


def test_calibration_hits_its_four_targets(model, random_gamma_model):
    for m in (model, random_gamma_model):
        fitted = jd.calibrate(m, mean_jump=0.06, mean_close=0.01, var_jump=0.41,
                              var_close=0.56)
        mom = jd.moments(fitted)
        assert np.allclose([mom["mean_J"], mom["mean_X"][-1], mom["var_J"],
                            mom["cov_X"][-1, -1]], [0.06, 0.01, 0.41, 0.56], atol=1e-9)


def test_calibration_refuses_a_variance_the_model_already_exceeds(model):
    with pytest.raises(ValueError, match="negative"):
        jd.calibrate(model, 0.06, 0.01, var_jump=0.41, var_close=0.01)


def test_scaling_to_an_r2_hits_it(model):
    for r2 in (0.05, 0.2):
        assert np.isclose(jd.moments(jd.scale_to_r2(model, r2))["r2_jump"], r2, atol=1e-9)
    with pytest.raises(ValueError, match="floor"):
        jd.scale_to_r2(model, 1e-6)


# ---------------------------------------------------------------------------- estimation


def test_the_curve_equation_is_frisch_waugh_consistent(model):
    sim = jd.simulate(model, 400, rng=5)
    fit = est.fit_curve_equation(sim["J"], sim["X"], model.grid, n_components=4)
    Jt = fit["J_tilde"]
    np.testing.assert_allclose(fit["b_hat"], Jt @ sim["X"][1:] / (Jt @ Jt), atol=1e-10)
    assert fit["b_hat"][0] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(fit["resid"][:, 0], 0.0, atol=1e-12)


def test_with_almost_no_curve_noise_gamma_and_mu_are_recovered_exactly(model):
    quiet = model.with_params(cell_var=1e-14 * np.ones_like(model.cell_var))
    sim = jd.simulate(quiet, 600, rng=6)
    fit = est.fit_curve_equation(sim["J"], sim["X"], model.grid, n_components=12)
    assert fit["gamma_ols"] == pytest.approx(0.3, abs=1e-5)
    assert fit["gamma_gls"] == pytest.approx(0.3, abs=1e-5)
    np.testing.assert_allclose(fit["mu_hat"], quiet.mu, atol=1e-5)


def test_large_sample_recovery_of_every_scalar(model):
    sim = jd.simulate(model, 20_000, rng=7)
    jump = est.fit_jump_equation(sim["J"], sim["X"], model.grid, n_components=5)
    curve = est.fit_curve_equation(sim["J"], sim["X"], model.grid, n_components=5)
    se = jump["fit"].se_ols
    assert abs(jump["rho"] - model.rho) < 4 * se[1]
    assert abs(jump["alpha"] - model.alpha) < 4 * se[0] + 0.01
    assert jump["sigma_eta"] == pytest.approx(model.sigma_eta, rel=0.02)
    assert abs(curve["gamma_ols"] - 0.3) < 4 * curve["se_ols"]
    assert abs(curve["gamma_gls"] - 0.3) < 4 * curve["se_gls"]

    # beta itself is ill-posed, its effect on the jump is not: the fitted signal tracks the
    # true one closely even where the reconstructed function does not.
    w = model.w
    true_signal = sim["X"] @ (model.beta * w)
    fitted_signal = sim["X"] @ (jump["beta_hat"] * w)
    assert np.corrcoef(true_signal, fitted_signal)[0, 1] > 0.95

    # the spot variance of the noise, read off the residual increments
    mids = 0.5 * (model.grid[1:] + model.grid[:-1])
    truth = model.cell_var / np.diff(model.grid)
    assert np.median(np.abs(curve["sigma2_hat"] / truth - 1)) < 0.05
    assert mids.size == curve["sigma2_hat"].size


def test_the_random_gamma_variance_is_recovered_and_a_fixed_one_is_not_invented(model):
    """Identification runs through jumps that are large next to the curve noise, so the check
    of the estimator uses a design where they are: the default design identifies the
    variance too, but needs far more days to do it (`notebooks/jump_equation_v2.ipynb`)."""
    loud = jd.notes_design(gamma=0.5, gamma_dist="uniform", sigma_eta=1.0, noise_level=0.5)
    sim = jd.simulate(loud, 20_000, rng=8)
    fit = est.fit_curve_equation(sim["J"], sim["X"], model.grid, n_components=6)
    assert abs(fit["gamma_ols"] - 0.5) < 4 * fit["se_ols"]
    var = est.gamma_variance(fit)
    assert abs(var["var"] - 1 / 12) < 3 * var["se"] and var["t"] > 3

    fixed = jd.simulate(model, 20_000, rng=9)
    none = est.gamma_variance(est.fit_curve_equation(fixed["J"], fixed["X"], model.grid, 6))
    assert abs(none["t"]) < 3.5


def test_the_onset_on_returns_is_found_exactly():
    m = jd.scale_to_r2(jd.notes_design(beta="return"), 0.05)
    sim = jd.simulate(m, 5_000, rng=10)
    prof = est.onset_profile(sim["J"], sim["X"], m.grid, on="return")
    start = m.grid[np.argmax(m.grid >= 0.85)]
    assert prof["theta_hat"] == start
    assert prof["rho_hat"] == pytest.approx(m.rho, abs=0.05)


def test_the_onset_on_levels_is_found_when_the_signal_is_strong():
    m = jd.scale_to_r2(jd.notes_design(), 0.5)
    sim = jd.simulate(m, 5_000, rng=10)
    prof = est.onset_profile(sim["J"], sim["X"], m.grid, on="level")
    assert abs(prof["theta_hat"] - 0.85) < 0.03


def test_hs_distance_of_a_unit_kernel_is_one():
    grid = np.linspace(0, 1, 41)
    assert est.hs_distance(np.ones((41, 41)), np.zeros((41, 41)), grid) == pytest.approx(1.0)
    assert est.hs_distance(np.eye(41), np.eye(41), grid) == 0.0


# ---------------------------------------------------------------------------- the test


@pytest.mark.parametrize("k", [1, 2, 4, 10])
@pytest.mark.parametrize("x", [0.3, 2.0, 7.0, 25.0])
def test_imhof_reproduces_the_chi_squared(k, x):
    assert est.imhof_sf(x, np.ones(k)) == pytest.approx(stats.chi2.sf(x, k), abs=1e-7)


def test_imhof_is_scale_free_and_matches_simulation_for_unequal_weights():
    lam = 1e-5 * np.exp(-np.arange(30) / 3.0)
    draws = (np.random.default_rng(0).standard_normal((200_000, lam.size)) ** 2) @ lam
    for q in (0.5, 0.9, 0.99):
        x = np.quantile(draws, q)
        assert est.imhof_sf(x, lam) == pytest.approx(1 - q, abs=4e-3)
    assert est.imhof_sf(-1.0, lam) == 1.0
    assert est.imhof_sf(1.0, np.zeros(3)) == 0.0


def test_the_shape_test_rejects_a_front_loaded_gap_fill():
    m = jd.notes_design(gap="front", tau=0.1)
    sim = jd.simulate(m, 2_000, rng=13)
    fit = est.fit_curve_equation(sim["J"], sim["X"], m.grid, n_components=5)
    assert est.shape_test(fit)["p_value"] < 1e-3
    # ...and does not reject when told the right profile
    right = est.fit_curve_equation(sim["J"], sim["X"], m.grid, n_components=5, G=m.G)
    assert est.shape_test(right)["p_value"] > 0.01


@pytest.mark.slow
def test_the_shape_test_holds_its_size(model):
    p = []
    for seed in range(150):
        sim = jd.simulate(model, 500, rng=1000 + seed)
        p.append(est.shape_test(est.fit_curve_equation(sim["J"], sim["X"], model.grid,
                                                       5))["p_value"])
    p = np.array(p)
    assert 0.01 <= np.mean(p < 0.05) <= 0.11
    assert stats.kstest(p, "uniform").pvalue > 0.001


@pytest.mark.slow
def test_gamma_intervals_cover_and_gls_is_the_more_efficient(random_gamma_model):
    rows = []
    for seed in range(150):
        sim = jd.simulate(random_gamma_model, 600, rng=2000 + seed)
        fit = est.fit_curve_equation(sim["J"], sim["X"], random_gamma_model.grid, 5)
        rows.append([fit["gamma_ols"], fit["se_ols"], fit["gamma_gls"], fit["se_gls"]])
    r = np.array(rows)
    for est_col, se_col in ((0, 1), (2, 3)):
        cover = np.mean(np.abs(r[:, est_col] - 0.5) < 1.96 * r[:, se_col])
        assert 0.88 <= cover <= 0.99
    assert r[:, 2].var() < r[:, 0].var()


# ------------------------------------------------------------ levels, returns, forecasts


@pytest.mark.parametrize("theta", [0.3, 0.85, 0.9])
def test_the_step_functionals_are_the_notes_coefficients(theta):
    grid = np.linspace(0, 1, 79)
    w = quadrature_weights(grid)
    np.testing.assert_allclose(jd.step_functional(grid, theta, "level"),
                               jd.notes_beta(grid, "indicator", onset=theta) * w)
    np.testing.assert_allclose(jd.step_functional(grid, theta, "return"),
                               jd.notes_beta(grid, "return", onset=theta) * w, atol=1e-12)
    with pytest.raises(ValueError, match="level"):
        jd.step_functional(grid, theta, "rate")


def test_integration_by_parts_turns_a_level_coefficient_into_a_return_weight():
    """`int beta X = int B dX`, `B(v) = int_v^1 beta`, which is the identity the whole
    levels-against-returns comparison rests on. Checked on the grid, on random paths."""
    grid = np.linspace(0, 1, 401)
    beta = jd.notes_beta(grid, "smooth", onset=0.6)
    X = np.random.default_rng(1).standard_normal((5, grid.size)).cumsum(axis=1) / 20
    X -= X[:, [0]]
    on_levels = X @ (beta * quadrature_weights(grid))
    tail = np.concatenate([np.cumsum((0.5 * (beta[1:] + beta[:-1]) * np.diff(grid))[::-1])[::-1],
                           [0.0]])                                # B(v) at each grid point
    B_mid = 0.5 * (tail[1:] + tail[:-1])
    on_returns = np.diff(X, axis=1) @ B_mid
    np.testing.assert_allclose(on_levels, on_returns, atol=2e-4)


def test_the_population_fit_with_the_true_functional_is_the_model(model):
    mom = jd.moments(model)
    fit = jd.population_fit(model, model.beta * model.w, mom)
    assert fit["r2"] == pytest.approx(mom["r2_jump"], abs=1e-10)
    np.testing.assert_allclose(fit["coef"], [model.rho, 1.0], atol=1e-8)
    assert fit["resid_var"] == pytest.approx(model.sigma_eta ** 2, rel=1e-8)


def test_the_population_fit_orders_models_as_least_squares_must(model):
    mom = jd.moments(model)
    floor = jd.population_fit(model, None, mom)["r2"]
    one = jd.population_fit(model, jd.step_functional(model.grid, 0.5, "return"), mom)["r2"]
    two = jd.population_fit(model, np.vstack([
        jd.step_functional(model.grid, 0.5, "return"),
        jd.step_functional(model.grid, 0.5, "level")]), mom)["r2"]
    assert 0 < floor < one < two <= mom["r2_jump"] + 1e-12


def test_the_population_fit_is_what_least_squares_converges_to(model, long_path):
    c = jd.step_functional(model.grid, 0.6, "return")
    J, X = long_path["J"], long_path["X"]
    design = np.column_stack([np.ones(J.size - 1), J[:-1], X[:-1] @ c])
    coef, *_ = np.linalg.lstsq(design, J[1:], rcond=None)
    pop = jd.population_fit(model, c)
    np.testing.assert_allclose(coef[1:], pop["coef"], atol=0.02)
    r2 = 1 - np.var(J[1:] - design @ coef) / np.var(J[1:])
    assert r2 == pytest.approx(pop["r2"], abs=0.01)


def test_a_level_truth_is_best_fitted_by_a_level_step_and_a_return_truth_by_a_return_step(
        model):
    returns = jd.scale_to_r2(jd.notes_design(beta="return"), jd.moments(model)["r2_jump"])
    for truth, right, wrong in ((model, "level", "return"), (returns, "return", "level")):
        mom = jd.moments(truth)
        good = jd.population_fit(truth, jd.step_functional(truth.grid, 0.85, right), mom)
        bad = jd.population_fit(truth, jd.step_functional(truth.grid, 0.85, wrong), mom)
        assert good["r2"] == pytest.approx(mom["r2_jump"], abs=1e-10)
        assert bad["r2"] < 0.3 * good["r2"]


def test_the_stationary_mean_is_a_fixed_point_of_the_forecast(model, random_gamma_model):
    for m in (model, random_gamma_model):
        mom = jd.moments(m)
        EJ, EX = jd.forecast(m, mom["mean_J"], mom["mean_X"])
        assert EJ == pytest.approx(mom["mean_J"], abs=1e-12)
        np.testing.assert_allclose(EX, mom["mean_X"], atol=1e-12)


def test_forecast_errors_are_unpredictable_from_the_day_before(model, long_path):
    """A conditional mean leaves errors orthogonal to what it conditions on."""
    J, X = long_path["J"], long_path["X"]
    pairs = [jd.forecast(model, J[t - 1], X[t - 1]) for t in range(1, 20_001)]
    jump_error = J[1:20_001] - np.array([p[0] for p in pairs])
    close_error = X[1:20_001, -1] - np.array([p[1][-1] for p in pairs])
    for regressor in (J[:20_000], X[:20_000, -1], X[:20_000, 40]):
        for error in (jump_error, close_error):
            assert abs(np.corrcoef(regressor, error)[0, 1]) < 4 / np.sqrt(20_000)


def test_the_sign_of_psi_sets_continuation_or_reversal_and_is_recovered():
    for scale, sign in ((1.0, 1), (-1.0, -1)):
        m = jd.notes_design(psi_scale=scale)
        mom = jd.moments(m)
        assert np.sign(mom["lag1"][-1, -1]) == sign
        sim = jd.simulate(m, 2520, rng=21)
        K = est.fit_curve_equation(sim["J"], sim["X"], m.grid, 2)["kernel"]
        assert np.corrcoef(K.ravel(), m.kernel.ravel())[0, 1] > 0.85


@pytest.mark.parametrize("truth", ["level", "return"])
def test_the_nesting_test_names_the_right_model(truth):
    base = jd.notes_design()
    m = base if truth == "level" else jd.scale_to_r2(jd.notes_design(beta="return"),
                                                     jd.moments(base)["r2_jump"])
    sim = jd.simulate(m, 2520, rng=22)
    test = est.nesting_test(sim["J"], sim["X"], m.grid, 0.85)
    other = "return" if truth == "level" else "level"
    assert test["p_value"][truth] < 1e-4
    assert test["p_value"][other] > 0.01
    assert test["coef"][truth] == pytest.approx(1.0 if truth == "level" else
                                                m.beta.max() * m.w[-1], rel=0.3)


@pytest.mark.slow
def test_the_nesting_test_holds_its_size_in_both_directions():
    base = jd.notes_design()
    returns = jd.scale_to_r2(jd.notes_design(beta="return"), jd.moments(base)["r2_jump"])
    for m, null in ((base, "return"), (returns, "level")):
        p = [est.nesting_test(*(lambda s: (s["J"], s["X"]))(jd.simulate(m, 500, rng=3000 + r)),
                              m.grid, 0.85)["p_value"][null] for r in range(200)]
        assert 0.02 <= np.mean(np.array(p) < 0.05) <= 0.09
