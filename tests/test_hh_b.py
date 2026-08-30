"""Tests for `truncFLR.HH`, method B of Hall & Hooker (2016).

Method B freezes an untruncated pilot and only searches over the endpoint, so the checks
here concern the pilot, the first-local-minimum endpoint rule that distinguishes B from A,
the mirroring algebra, and the tuning-parameter diagnostics.
"""

import numpy as np
import pytest

from truncFLR import HH, HH_dgp

# The candidate grid the notebooks use: lambda is order 1 on these problems.
LAMBDAS = np.linspace(0.0, 1.0, 250)[1:]


# ---------------------------------------------------------------------------- pilot ------

def test_pilot_fpca_selects_m_by_bic(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, fpc = HH.pilot_fpca(X, Y, grid, n_basis=15, n_components=7)
    assert 2 <= fpc["m"] <= 7
    assert b.shape == grid.shape
    np.testing.assert_allclose(fpc["mean"], X.mean(axis=0), rtol=1e-12)
    assert np.all(np.diff(fpc["tau"]) <= 1e-12), "eigenvalues must be descending"


def test_pilot_fpca_matches_method_A(hh_data):
    """Both modules ship the same pilot; a divergence would break every comparison."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    from truncFLR import HH_A
    a1, b1, f1 = HH.pilot_fpca(X, Y, grid, n_basis=15, n_components=6)
    a2, b2, f2 = HH_A.pilot_fpca(X, Y, grid, n_basis=15, n_components=6)
    assert a1 == pytest.approx(a2)
    np.testing.assert_allclose(b1, b2, rtol=1e-10)
    assert f1["m"] == f2["m"]


def test_pilot_leaks_into_the_null_region(hh_data_big):
    """The premise of the whole method: an untruncated pilot is non-zero where b is zero."""
    d = hh_data_big
    _, b_pilot, _ = HH.pilot_fpca(d["X"], d["Y"], d["grid"])
    null = d["grid"] < d["u"]
    assert np.trapezoid(b_pilot[null] ** 2, d["grid"][null]) > 1e-3


# -------------------------------------------------------------------- endpoint search ----

def test_theta_methodB_objective_is_rss_plus_penalty(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, _ = HH.pilot_fpca(X, Y, grid, n_basis=15, n_components=5)
    n, lam = len(Y), 2e-3
    _, tg, S = HH.theta_methodB(X, Y, grid, a, b, lam)
    _, tg0, S0 = HH.theta_methodB(X, Y, grid, a, b, 0.0)
    np.testing.assert_allclose(tg, tg0)
    np.testing.assert_allclose(S, S0 + n * lam * tg ** 2, rtol=1e-9)


def test_theta_methodB_endpoint_is_on_the_grid(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, _ = HH.pilot_fpca(X, Y, grid, n_basis=15, n_components=5)
    theta, tg, S = HH.theta_methodB(X, Y, grid, a, b, 1e-3)
    assert theta in tg
    assert S.shape == tg.shape
    assert np.all(np.isfinite(S))


def test_ab_methodB_truncates_the_pilot_when_not_refitting(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, _ = HH.pilot_fpca(X, Y, grid, n_basis=15, n_components=5)
    theta = 0.6
    a_hat, b_hat = HH.ab_methodB(theta, grid, a, b, X, Y=Y, refit=False)
    np.testing.assert_array_equal(b_hat[grid > theta], 0.0)
    np.testing.assert_allclose(b_hat[grid <= theta], b[grid <= theta])


def test_ab_methodB_refit_is_zero_past_theta(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, _ = HH.pilot_fpca(X, Y, grid, n_basis=15, n_components=5)
    theta = 0.6
    a_hat, b_hat = HH.ab_methodB(theta, grid, a, b, X, Y=Y, refit=True)
    np.testing.assert_array_equal(b_hat[grid > theta], 0.0)
    assert np.any(b_hat[grid <= theta] != 0.0)
    # A refit is a genuinely different estimate, not the truncated pilot.
    assert not np.allclose(b_hat[grid <= theta], b[grid <= theta])


# ------------------------------------------------------------------------ lambda_select --

def test_lambda_select_returns_a_candidate_and_diagnostics(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, fpc = HH.pilot_fpca(X, Y, grid, n_basis=15, n_components=6)
    lambdas = np.logspace(-6, 0, 25)
    lam_hat, diag = HH.lambda_select(X, Y, grid, a, b, fpc, None, lambdas, m_tuning=2)

    assert lam_hat in lambdas
    for key in ("lambdas", "theta_lams", "Vs", "P_b", "S_Y", "S_b",
                "theta_grid", "b_simp", "sigma2", "theta_bar"):
        assert key in diag, f"missing diagnostic {key!r}"
    assert diag["sigma2"] > 0
    assert lam_hat == lambdas[int(np.nanargmin(diag["P_b"]))]


def test_lambda_select_honours_a_fixed_theta_bar(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, fpc = HH.pilot_fpca(X, Y, grid, n_basis=15, n_components=6)
    _, diag = HH.lambda_select(X, Y, grid, a, b, fpc, 0.45, np.logspace(-6, 0, 10), m_tuning=2)
    assert diag["theta_bar"] == 0.45
    np.testing.assert_array_equal(diag["b_simp"][grid > 0.45], 0.0)


def test_lambda_select_variance_is_non_negative(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, fpc = HH.pilot_fpca(X, Y, grid, n_basis=15, n_components=6)
    _, diag = HH.lambda_select(X, Y, grid, a, b, fpc, None, np.logspace(-6, 0, 15), m_tuning=2)
    finite = np.isfinite(diag["Vs"])
    assert finite.any()
    assert np.all(diag["Vs"][finite] >= 0)
    # Where curvature is degenerate both V and P_b must be NaN together, never one alone.
    np.testing.assert_array_equal(np.isnan(diag["Vs"]), np.isnan(diag["P_b"]))


def test_fit_b_simp_uses_one_fourier_pair_by_default(hh_data):
    """n_fourier = 1 means constant + one sin/cos pair -- the paper's k = 1 specification."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    b1, t1 = HH.fit_b_simp(X, Y, grid, 0.0, n_fourier=1, theta_bar=0.5)
    b3, t3 = HH.fit_b_simp(X, Y, grid, 0.0, n_fourier=3, theta_bar=0.5)
    assert t1 == t3 == 0.5
    # More basis functions can only fit the same data better (or equally well).
    assert not np.allclose(b1, b3)


# ------------------------------------------------------------------------ end to end -----

@pytest.mark.slow
def test_fit_mirror_recovers_the_left_endpoint(hh_data_big):
    d = hh_data_big
    res = HH.fit_mirror(d["X"], d["Y"], d["grid"], LAMBDAS)
    assert abs(res["u_hat"] - d["u"]) < 0.1
    assert res["u_hat"] == pytest.approx(1.0 - res["theta_hat"])
    np.testing.assert_array_equal(res["b_hat"][d["grid"] < res["u_hat"] - 1e-12], 0.0)


@pytest.mark.slow
def test_fit_mirror_equals_fit_on_reflected_curves(hh_data):
    d = hh_data
    lambdas = np.logspace(-6, 0, 20)
    mirror = HH.fit_mirror(d["X"], d["Y"], d["grid"], lambdas, m_tuning=2)
    direct = HH.fit(d["X"][:, ::-1], d["Y"], d["grid"], lambdas, m_tuning=2)

    assert mirror["theta_hat"] == direct["theta_hat"]
    assert mirror["lam_hat"] == direct["lam_hat"]
    np.testing.assert_allclose(mirror["b_hat"], direct["b_hat"][::-1])
    assert "u_hat" not in direct, "the unmirrored fit estimates the right endpoint, not u"


@pytest.mark.slow
def test_fit_mirror_beats_the_untruncated_pilot(hh_data_big):
    d = hh_data_big
    _, b_pilot, _ = HH.pilot_fpca(d["X"], d["Y"], d["grid"])
    res = HH.fit_mirror(d["X"], d["Y"], d["grid"], LAMBDAS)

    def ise(b):
        return np.trapezoid((b - d["b"]) ** 2, d["grid"])

    assert ise(res["b_hat"]) < ise(b_pilot)


@pytest.mark.slow
def test_lambda_hat_is_interior_at_the_bic_selected_m(hh_data_big):
    """The paper's own procedure must pick a lambda from inside the candidate grid.

    Reading S_Y as a sum in step 6 puts lambda n times below the curvature it is added to, and
    lambda_hat then lands on the smallest candidate whatever the data.
    """
    d = hh_data_big
    res = HH.fit_mirror(d["X"], d["Y"], d["grid"], LAMBDAS)
    assert LAMBDAS[0] < res["lam_hat"] < LAMBDAS[-1]


@pytest.mark.slow
def test_step_seven_density_is_resolved_by_the_refined_quadrature(hh_data_big):
    """P_b must be an average of S_b, so it stays inside the range of S_b.

    Integrating the step-7 weight on the theta grid instead returns roughly
    S_b(theta_lambda) * h / sqrt(2 pi V), which leaves that range as soon as V is small.
    """
    d = hh_data_big
    grid, X, Y = d["grid"], d["X"][:, ::-1], d["Y"]
    a, b, fpc = HH.pilot_fpca(X, Y, grid)
    _, diag = HH.lambda_select(X, Y, grid, a, b, fpc, None, LAMBDAS)

    P_b, S_b = diag["P_b"], diag["S_b"]
    finite = np.isfinite(P_b)
    assert finite.any()
    assert np.all(P_b[finite] >= S_b.min())
    assert np.all(P_b[finite] <= S_b.max())


@pytest.mark.slow
@pytest.mark.parametrize("weight", sorted(HH.WEIGHT_FUNCTIONS))
def test_every_weight_function_runs_end_to_end(hh_data, weight):
    d = hh_data
    res = HH.fit_mirror(d["X"], d["Y"], d["grid"], np.linspace(0, 1, 40)[1:],
                        w=HH.WEIGHT_FUNCTIONS[weight])
    assert 0.0 <= res["u_hat"] <= 1.0
    assert np.all(np.isfinite(res["b_hat"]))


# ----------------------------------------------------------------------------- DGP -------

def test_sim_dataset_shapes_and_truth():
    d = HH_dgp.sim_dataset(n=40, n_grid=51, u=0.3, v=0.9, snr=10.0, rng=1)
    assert d["X"].shape == (40, 51)
    assert d["Y"].shape == (40,)
    assert d["grid"][0] == 0.0 and d["grid"][-1] == 1.0
    np.testing.assert_array_equal(d["b"], ((d["grid"] >= 0.3) & (d["grid"] <= 0.9)).astype(float))
    assert d["sigma"] > 0


def test_sim_dataset_hits_the_requested_snr():
    d = HH_dgp.sim_dataset(n=4000, n_grid=101, u=0.8, v=1.0, snr=26.25, rng=3)
    signal = HH_dgp.trunc_int(d["X"], d["grid"], u=0.8, v=1.0)
    assert np.var(signal) / d["sigma"] ** 2 == pytest.approx(26.25, rel=1e-6)


def test_sim_dataset_is_reproducible():
    a = HH_dgp.sim_dataset(n=20, n_grid=41, rng=17)
    b = HH_dgp.sim_dataset(n=20, n_grid=41, rng=17)
    np.testing.assert_array_equal(a["X"], b["X"])
    np.testing.assert_array_equal(a["Y"], b["Y"])


def test_sim_dataset_accepts_a_custom_beta():

    def beta(t):
        return np.sin(np.pi * t)

    d = HH_dgp.sim_dataset(n=30, n_grid=51, beta_fn=beta, rng=2)
    np.testing.assert_allclose(d["b"], beta(d["grid"]))


def test_trig_basis_matches_the_paper():
    grid = np.linspace(0, 1, 61)
    Psi = HH_dgp.trig_basis(grid, n_basis=5)
    np.testing.assert_allclose(Psi[0], 1.0)
    np.testing.assert_allclose(Psi[1], np.sin(2 * np.pi * grid))
    np.testing.assert_allclose(Psi[2], np.cos(2 * np.pi * grid))
    np.testing.assert_allclose(Psi[3], np.sin(4 * np.pi * grid))
    np.testing.assert_allclose(Psi[4], np.cos(4 * np.pi * grid))


def test_trunc_int_matches_a_direct_integral():
    grid = np.linspace(0, 1, 201)
    X = np.vstack([np.ones_like(grid), grid])
    got = HH_dgp.trunc_int(X, grid, u=0.25, v=0.75)
    np.testing.assert_allclose(got, [0.5, 0.25], atol=1e-6)


def test_sample_gp_variance_decays_across_the_basis():
    grid = np.linspace(0, 1, 101)
    X = HH_dgp.sample_gp(20000, grid, n_basis=6, decay=2.0, variance=1.0, rng=0)
    Psi = HH_dgp.trig_basis(grid, n_basis=6)
    # Recover the coefficients by projecting onto the (orthogonal) trig basis.
    coef = np.linalg.lstsq(Psi.T, X.T, rcond=None)[0]
    emp_var = coef.var(axis=1)
    expected = np.exp(-np.arange(6) / 2.0)
    np.testing.assert_allclose(emp_var, expected, rtol=0.06)
