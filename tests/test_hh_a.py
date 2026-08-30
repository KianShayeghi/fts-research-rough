"""Tests for `truncFLR.HH_A`, method A of Hall & Hooker (2016).

Method A re-runs a truncated FPCA at every candidate endpoint, so the pieces worth pinning
down are: the truncated decomposition itself (orthonormal in the right inner product), the
vectorised multi-m theta scan (it must agree with the plain one, since it is purely an
optimisation), the mirroring algebra, and end-to-end endpoint recovery.
"""

import numpy as np
import pytest

from truncFLR import HH, HH_A, HH_dgp

# The candidate grid the notebooks use: lambda is order 1 on these problems.
LAMBDAS = np.linspace(0.0, 1.0, 250)[1:]


def trapz_weights(grid):
    w = np.zeros_like(grid)
    w[1:-1] = (grid[2:] - grid[:-2]) / 2.0
    w[0] = (grid[1] - grid[0]) / 2.0
    w[-1] = (grid[-1] - grid[-2]) / 2.0
    return w


# ------------------------------------------------------------------- truncated FPCA -----

def test_trunc_fpca_components_are_orthonormal_in_L2(hh_data):
    """phi_j^theta must be orthonormal under the trapezium L2 inner product on [0, theta]."""
    grid, X = hh_data["grid"], hh_data["X"]
    interval = grid <= 0.6
    phi, tau = HH_A.trunc_fpca(X, grid, interval, m=5)

    w = trapz_weights(grid[interval])
    gram = (phi * w[None, :]) @ phi.T
    np.testing.assert_allclose(gram, np.eye(5), atol=1e-8)
    assert phi.shape == (5, interval.sum())
    assert tau.shape == (5,)


def test_trunc_fpca_eigenvalues_are_positive_and_descending(hh_data):
    grid, X = hh_data["grid"], hh_data["X"]
    _, tau = HH_A.trunc_fpca(X, grid, grid <= 0.7, m=6)
    assert np.all(tau > 0)
    assert np.all(np.diff(tau) <= 0)


def test_trunc_fpca_reconstructs_the_covariance_trace(hh_data):
    """sum_j tau_j over all components must equal the total variance int Var(X(t)) dt."""
    grid, X = hh_data["grid"], hh_data["X"]
    interval = grid <= 0.8
    n = X.shape[0]
    _, tau = HH_A.trunc_fpca(X, grid, interval, m=n)

    Xc = X[:, interval] - X[:, interval].mean(axis=0)
    total = np.trapezoid(np.sum(Xc ** 2, axis=0) / n, grid[interval])
    np.testing.assert_allclose(tau.sum(), total, rtol=1e-6)


def test_trunc_fpca_agrees_with_method_B_eigenvalues(hh_data):
    """HH_A.trunc_fpca and HH.trunc_eigvals must read the same spectrum off the same data."""
    grid, X = hh_data["grid"], hh_data["X"]
    interval = grid <= 0.55
    _, tau_a = HH_A.trunc_fpca(X, grid, interval, m=4)
    tau_b = HH.trunc_eigvals(X, grid, interval, m=4)
    np.testing.assert_allclose(tau_a, tau_b, rtol=1e-10)


def test_trunc_eigvals_floors_short_subgrids(hh_data):
    """Asking for more components than the sub-grid supports must floor, not raise."""
    grid, X = hh_data["grid"], hh_data["X"]
    interval = grid <= grid[2]                    # only three grid points
    tau = HH_A.trunc_eigvals(X, grid, interval, m=8)
    assert tau.shape == (8,)
    assert np.all(tau[3:] == np.finfo(float).eps), "missing components must be floored"


# ------------------------------------------------------------------- the theta scan -----

@pytest.mark.parametrize("lam", [0.0, 1e-4, 1e-2])
def test_theta_methodA_multi_matches_the_plain_scan(hh_data, lam):
    """The vectorised multi-m scan is an optimisation, so it must be numerically identical."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    m_values = [2, 3, 5, 8]
    multi = HH_A.theta_methodA_multi(X, Y, grid, lam, m_values)
    for m in m_values:
        theta, _, S = HH_A.theta_methodA(X, Y, grid, lam, n_components=m)
        np.testing.assert_allclose(multi[m][1], S, rtol=1e-8, atol=1e-8)
        assert multi[m][0] == theta


def test_theta_methodA_penalty_pushes_the_endpoint_left(hh_data):
    """Larger lambda must never move the selected endpoint to the right."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    thetas = [HH_A.theta_methodA(X, Y, grid, lam, n_components=3)[0]
              for lam in (0.0, 1e-3, 1e-2, 1e-1)]
    assert np.all(np.diff(thetas) <= 0), f"theta must be non-increasing in lambda: {thetas}"


def test_theta_methodA_objective_is_rss_plus_penalty(hh_data):
    """S(theta) must be exactly RSS(theta) + n * lambda * theta^2."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    lam, n = 5e-3, len(Y)
    _, tg, S = HH_A.theta_methodA(X, Y, grid, lam, n_components=3)
    _, tg0, S0 = HH_A.theta_methodA(X, Y, grid, 0.0, n_components=3)
    np.testing.assert_allclose(tg, tg0)
    np.testing.assert_allclose(S, S0 + n * lam * tg ** 2, rtol=1e-9)


def test_theta_methodA_multi_rejects_nothing_and_caps_short_subgrids(hh_data):
    """Near theta = 0 there are too few grid points for large m; the scan must still run."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    out = HH_A.theta_methodA_multi(X, Y, grid, 0.0, [2, 10])
    for m in (2, 10):
        assert np.all(np.isfinite(out[m][1]))


# ------------------------------------------------------------------- final estimates -----

def test_ab_methodA_is_zero_past_theta(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    theta = 0.62
    a_hat, b_hat = HH_A.ab_methodA(theta, grid, X, Y, n_components=4)
    assert b_hat.shape == grid.shape
    np.testing.assert_array_equal(b_hat[grid > theta], 0.0)
    assert np.any(b_hat[grid <= theta] != 0.0)
    assert np.isfinite(a_hat)


def test_ab_methodA_is_the_least_squares_fit(hh_data):
    """At theta the fitted model must be the LS fit of Y on [1 | truncated FPC scores]."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    theta, m = 0.7, 3
    interval = grid <= theta
    a_hat, b_hat = HH_A.ab_methodA(theta, grid, X, Y, n_components=m)

    pred = HH_A.trunc_predict(a_hat, b_hat, X, grid, theta)
    phi, _ = HH_A.trunc_fpca(X, grid, interval, m)
    scores = np.trapezoid(phi[:, None, :] * X[None, :, interval], grid[interval], axis=2).T
    design = np.column_stack([np.ones(len(Y)), scores])
    # Residuals of a least-squares fit are orthogonal to the design.
    np.testing.assert_allclose(design.T @ (Y - pred), 0.0, atol=1e-8)


def test_trunc_predict_uses_the_sub_grid_quadrature(hh_data):
    """Integrating over the whole grid with b zeroed past theta adds a spurious half panel."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    theta = 0.7
    a_hat, b_hat = HH_A.ab_methodA(theta, grid, X, Y, n_components=3)
    interval = grid <= theta

    pred = HH_A.trunc_predict(a_hat, b_hat, X, grid, theta)
    expected = a_hat + np.trapezoid(b_hat[interval] * X[:, interval], grid[interval], axis=1)
    np.testing.assert_allclose(pred, expected, rtol=1e-12)

    whole_grid = a_hat + np.trapezoid(b_hat[None, :] * X, grid, axis=1)
    h = grid[1] - grid[0]
    spurious = b_hat[interval][-1] * X[:, interval][:, -1] * h / 2.0
    np.testing.assert_allclose(whole_grid - pred, spurious, rtol=1e-8, atol=1e-12)


def test_methodA_bic_matches_its_formula(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    theta = 0.7
    a_hat, b_hat = HH_A.ab_methodA(theta, grid, X, Y, n_components=3)
    n = len(Y)
    rss = np.sum((Y - HH_A.trunc_predict(a_hat, b_hat, X, grid, theta)) ** 2)
    expected = n * np.log(rss / n) + 4 * np.log(n)
    assert HH_A.methodA_bic(a_hat, b_hat, X, Y, grid, 3, theta) == pytest.approx(expected)


def test_methodA_bic_penalises_extra_components(hh_data):
    """At a fixed fit, BIC must grow by exactly log(n) for each extra component charged."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a_hat, b_hat = HH_A.ab_methodA(0.7, grid, X, Y, n_components=3)
    b3 = HH_A.methodA_bic(a_hat, b_hat, X, Y, grid, 3, 0.7)
    b4 = HH_A.methodA_bic(a_hat, b_hat, X, Y, grid, 4, 0.7)
    assert b4 - b3 == pytest.approx(np.log(len(Y)))


def test_methodA_bic_scores_the_actual_least_squares_residuals(hh_data):
    """The RSS in BIC must be the minimum achievable at that theta and m."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    theta, m, n = 0.7, 3, len(Y)
    a_hat, b_hat = HH_A.ab_methodA(theta, grid, X, Y, n_components=m)
    bic = HH_A.methodA_bic(a_hat, b_hat, X, Y, grid, m, theta)
    rss = np.exp((bic - (m + 1) * np.log(n)) / n) * n

    interval = grid <= theta
    phi, _ = HH_A.trunc_fpca(X, grid, interval, m)
    scores = np.trapezoid(phi[:, None, :] * X[None, :, interval], grid[interval], axis=2).T
    design = np.column_stack([np.ones(n), scores])
    _, residuals, *_ = np.linalg.lstsq(design, Y, rcond=None)
    assert rss == pytest.approx(float(residuals[0]), rel=1e-8)


# ---------------------------------------------------------------------- step 3 target ----

def test_fit_b_simp_respects_a_fixed_cutoff(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    b_simp, theta_bar = HH_A.fit_b_simp(X, Y, grid, a_pilot=0.0, theta_bar=0.5)
    assert theta_bar == 0.5
    np.testing.assert_array_equal(b_simp[grid > 0.5], 0.0)
    assert np.any(b_simp[grid <= 0.5] != 0.0)


def test_fit_b_simp_estimates_its_own_cutoff(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    b_simp, theta_bar = HH_A.fit_b_simp(X, Y, grid, a_pilot=0.0)
    assert grid[0] < theta_bar < grid[-1]
    np.testing.assert_array_equal(b_simp[grid > theta_bar], 0.0)


def test_fit_b_simp_matches_method_B(hh_data):
    """b_simp is common to both methods, so the two modules must produce the same target."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a_pilot = float(Y.mean())
    ba, ta = HH_A.fit_b_simp(X, Y, grid, a_pilot)
    bb, tb = HH.fit_b_simp(X, Y, grid, a_pilot)
    assert ta == tb
    np.testing.assert_allclose(ba, bb, rtol=1e-10, atol=1e-12)


# ------------------------------------------------------------------------ lambda_select --

def test_lambda_select_returns_a_candidate_and_full_diagnostics(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, fpc = HH_A.pilot_fpca(X, Y, grid, n_basis=15, n_components=6)
    lambdas = np.logspace(-6, 0, 25)
    lam_hat, diag = HH_A.lambda_select(X, Y, grid, a, b, fpc, None, lambdas,
                                       n_components=6, m_tuning=2)

    assert lam_hat in lambdas
    for key in ("lambdas", "theta_lams", "Vs", "P_b", "S_Y", "S_b",
                "theta_grid", "b_simp", "ms", "sigma2", "theta_bar"):
        assert key in diag, f"missing diagnostic {key!r}"
    assert diag["sigma2"] > 0
    assert np.all(np.isfinite(diag["S_Y"])) and np.all(np.isfinite(diag["S_b"]))
    assert lam_hat == lambdas[int(np.nanargmin(diag["P_b"]))]
    # m_tuning must cap the component count actually used in the criterion.
    assert diag["ms"].max() <= 2


def test_lambda_select_theta_lam_minimises_the_penalised_surrogate(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, fpc = HH_A.pilot_fpca(X, Y, grid, n_basis=15, n_components=6)
    lambdas = np.logspace(-6, 0, 15)
    _, diag = HH_A.lambda_select(X, Y, grid, a, b, fpc, None, lambdas,
                                 n_components=6, m_tuning=2)
    n, tg, S_Y = len(Y), diag["theta_grid"], diag["S_Y"]
    for j, lam in enumerate(lambdas):
        obj = S_Y + n * lam * tg ** 2
        assert diag["theta_lams"][j] == tg[int(np.argmin(obj))]


def test_lambda_select_curvature_is_positive(hh_data):
    """S_Y^A is locally convex at the penalised minimum, so V(lambda) takes the plus form.

    The paper's method-A step 6 displays V with a leading minus, which pairs with an
    oppositely oriented curvature. `lambda_select` instead computes
    V = (sigma^2/n) S_Y'' / (S_Y'' + lambda)^2 and treats S_Y'' <= 0 as degenerate. That is
    only right if the penalised argmin really does sit in a convex stretch of S_Y^A, which
    is what this pins: if the sign ever flips, V goes NaN and lambda selection silently
    loses candidates, so the choice of form has to be revisited rather than rediscovered.
    """
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, fpc = HH_A.pilot_fpca(X, Y, grid, n_basis=15, n_components=6)
    lambdas = np.logspace(-6, 0, 15)
    _, diag = HH_A.lambda_select(X, Y, grid, a, b, fpc, None, lambdas,
                                 n_components=6, m_tuning=2)

    S_Y, theta_grid = diag["S_Y"], diag["theta_grid"]
    h = theta_grid[1] - theta_grid[0]
    n = len(Y)
    curvatures = []
    for lam, theta_lam in zip(lambdas, diag["theta_lams"]):
        # Step 6 as documented: theta_lambda is the *global* argmin, not the first local one.
        idx = int(np.argmin(S_Y + n * lam * theta_grid ** 2))
        assert theta_grid[idx] == theta_lam, "theta_lambda is no longer the global argmin"
        j = min(max(idx, 1), len(theta_grid) - 2)
        curvatures.append((S_Y[j + 1] - 2 * S_Y[j] + S_Y[j - 1]) / h ** 2)

    curvatures = np.asarray(curvatures)
    assert np.all(curvatures > 0), (
        "S_Y'' is non-positive at the penalised argmin for some lambda, so the plus form of "
        f"V documented in lambda_select no longer holds: {curvatures}"
    )
    # ...and the variance the module actually reports must then be finite everywhere.
    assert np.all(np.isfinite(diag["Vs"]))


def test_lambda_select_variance_is_non_negative(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, fpc = HH_A.pilot_fpca(X, Y, grid, n_basis=15, n_components=6)
    _, diag = HH_A.lambda_select(X, Y, grid, a, b, fpc, None, np.logspace(-6, 0, 15),
                                 n_components=6, m_tuning=2)
    finite = np.isfinite(diag["Vs"])
    assert finite.any(), "degenerate curvature everywhere -- lambda selection is meaningless"
    assert np.all(diag["Vs"][finite] >= 0)


# ---------------------------------------------------------------------------- pilot ------

def test_pilot_fpca_intercept_centres_the_fit(hh_data):
    """a_pilot is defined so the fitted mean response equals the sample mean."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    a, b, fpc = HH_A.pilot_fpca(X, Y, grid, n_basis=15, n_components=6)
    pred = a + np.trapezoid(b[None, :] * X, grid, axis=1)
    assert pred.mean() == pytest.approx(Y.mean(), abs=1e-8)
    assert 2 <= fpc["m"] <= 6
    assert fpc["phi"].shape == (fpc["m"], grid.size)
    assert fpc["tau"].shape == (fpc["m"],)


# ------------------------------------------------------------------------ end to end -----

def test_fit_rejects_an_empty_m_range(hh_data):
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    with pytest.raises(ValueError, match="m_range is empty"):
        HH_A.fit(X, Y, grid, np.logspace(-6, 0, 5), m_range=[])


@pytest.mark.slow
def test_fit_mirror_recovers_the_left_endpoint(hh_data_big):
    """The headline claim: method A finds u = 0.8 on the notebook's own configuration."""
    d = hh_data_big
    res = HH_A.fit_mirror(d["X"], d["Y"], d["grid"], LAMBDAS)

    assert abs(res["u_hat"] - d["u"]) < 0.05
    assert res["u_hat"] == pytest.approx(1.0 - res["theta_hat"])
    assert res["lam_hat"] > 0
    assert res["m_hat"] >= 2
    # b_hat lives in the original orientation: zero to the *left* of u_hat.
    np.testing.assert_array_equal(res["b_hat"][d["grid"] < res["u_hat"] - 1e-12], 0.0)
    assert np.any(res["b_hat"] != 0.0)


@pytest.mark.slow
def test_fit_mirror_beats_the_untruncated_pilot(hh_data_big):
    """Truncating must reduce integrated squared error against the true indicator."""
    d = hh_data_big
    a_p, b_p, _ = HH_A.pilot_fpca(d["X"], d["Y"], d["grid"])
    res = HH_A.fit_mirror(d["X"], d["Y"], d["grid"], LAMBDAS)

    def ise(b):
        return np.trapezoid((b - d["b"]) ** 2, d["grid"])

    assert ise(res["b_hat"]) < ise(b_p)


@pytest.mark.slow
def test_fit_mirror_diagnostics_are_mapped_back(hh_data):
    """fit_mirror must report u-coordinates, i.e. 1 - the reflected theta quantities."""
    d = hh_data
    res = HH_A.fit_mirror(d["X"], d["Y"], d["grid"], np.linspace(0, 1, 40)[1:])
    diag = res["diag"]
    np.testing.assert_allclose(diag["u_lam"], 1.0 - diag["theta_lams"])
    assert diag["u_bar"] == pytest.approx(1.0 - diag["theta_bar"])


@pytest.mark.slow
def test_fit_mirror_equals_fit_on_reflected_curves(hh_data):
    """fit_mirror is exactly `fit` on X(1-t) with the endpoint mapped back."""
    d = hh_data
    lambdas = np.logspace(-6, 0, 20)
    mirror = HH_A.fit_mirror(d["X"], d["Y"], d["grid"], lambdas)
    direct = HH_A.fit(d["X"][:, ::-1], d["Y"], d["grid"], lambdas)

    assert mirror["theta_hat"] == direct["theta_hat"]
    assert mirror["u_hat"] == pytest.approx(1.0 - direct["theta_hat"])
    assert mirror["lam_hat"] == direct["lam_hat"]
    assert mirror["m_hat"] == direct["m_hat"]
    np.testing.assert_allclose(mirror["b_hat"], direct["b_hat"][::-1])
    assert "u_hat" not in direct, "the unmirrored fit estimates the right endpoint, not u"


@pytest.mark.slow
def test_strict_step7_carries_a_separate_lambda_for_each_m(hh_data):
    """Step 7 admits two readings, and they part company once lambda responds to m.

    The strict reading re-runs the whole lambda selection for every candidate m, so each m
    carries its own lambda_hat_m; the fast reading commits to one lambda and then BIC-selects
    m over the final fit alone. Both must return a usable fit, and the per-m lambdas must
    genuinely vary, which is what the fast reading is trading away.
    """
    d = hh_data
    lambdas = np.linspace(0.0, 1.0, 120)[1:]
    fast = HH_A.fit_mirror(d["X"], d["Y"], d["grid"], lambdas)
    strict = HH_A.fit_mirror(d["X"], d["Y"], d["grid"], lambdas, strict=True)

    for res in (fast, strict):
        assert 0.0 < res["u_hat"] < 1.0
        assert res["m_hat"] in res["diag"]["bics"]

    lam_hats = strict["diag"]["lam_hats"]
    assert set(lam_hats) == set(strict["diag"]["bics"])
    assert len(lam_hats) > 1
    assert len(set(lam_hats.values())) > 1, (
        "lambda_hat is the same for every m, so the lambda criterion is not seeing m"
    )
    # The fast reading reuses one lambda across every candidate m, by construction.
    assert len(set(fast["diag"]["lam_hats"].values())) == 1


@pytest.mark.slow
def test_fit_mirror_handles_a_smooth_boundary(hh_data):
    """Model 2 of the paper: a half-sine bump, whose boundary is harder to detect."""
    def bump(t):
        w = (t >= 0.8) & (t <= 1.0)
        b = np.zeros_like(t)
        b[w] = np.sin(np.pi * (t[w] - 0.8) / 0.2)
        return b

    d = HH_dgp.sim_dataset(n=150, n_grid=201, u=0.8, v=1.0, snr=26.25, beta_fn=bump, rng=5)
    res = HH_A.fit_mirror(d["X"], d["Y"], d["grid"], LAMBDAS)
    assert 0.6 < res["u_hat"] < 1.0
    assert np.all(np.isfinite(res["b_hat"]))


# ------------------------------------------------------------------- truncation_profile


@pytest.mark.slow
@pytest.mark.parametrize("v, expected", [(0.3, 0.22), (0.5, 0.53), (0.7, 0.68)])
def test_the_elbow_recovers_the_endpoint_where_the_minimum_does_not(v, expected):
    """Past the true endpoint the retained curve carries no signal, so extending it costs
    only estimation variance and the profile flattens instead of rising. The minimiser then
    wanders over that flat stretch, while the one-standard-error elbow asks where the error
    stopped falling, which is the question the endpoint answers."""
    d = HH_dgp.sim_dataset(n=200, n_grid=101, u=0.0, v=v, snr=26.25, rng=1)
    prof = HH_A.truncation_profile(d["X"], d["Y"], d["grid"])
    assert prof["theta_1se"] == pytest.approx(expected, abs=0.12)
    assert prof["theta_hat"] >= prof["theta_1se"]


@pytest.mark.slow
def test_the_profile_declines_to_truncate_when_the_whole_domain_matters():
    """The control: with `v = 1` there is nothing to cut, and the elbow must sit near the
    right-hand end rather than inventing a boundary."""
    d = HH_dgp.sim_dataset(n=200, n_grid=101, u=0.0, v=1.0, snr=26.25, rng=1)
    assert HH_A.truncation_profile(d["X"], d["Y"], d["grid"])["theta_1se"] > 0.85


@pytest.mark.slow
@pytest.mark.parametrize("u, expected", [(0.3, 0.32), (0.5, 0.47), (0.8, 0.89)])
def test_the_mirrored_profile_recovers_a_left_endpoint(u, expected):
    d = HH_dgp.sim_dataset(n=200, n_grid=101, u=u, v=1.0, snr=26.25, rng=1)
    prof = HH_A.truncation_profile(d["X"], d["Y"], d["grid"], mirror=True)
    assert prof["theta_1se"] == pytest.approx(expected, abs=0.12)


def test_the_profile_error_falls_sharply_before_the_endpoint(hh_data):
    """Truncating *inside* the informative window throws away signal, so the error there
    must be far worse than at the full domain. This is the half of the profile that is
    genuinely informative."""
    d = HH_dgp.sim_dataset(n=150, n_grid=101, u=0.0, v=0.6, snr=26.25, rng=2)
    prof = HH_A.truncation_profile(d["X"], d["Y"], d["grid"], n_folds=5)
    early = prof["mse"][prof["thetas"] < 0.3].min()
    assert early > 3 * np.nanmin(prof["mse"])


def test_the_profile_reports_one_score_per_candidate(hh_data):
    thetas = np.linspace(0.2, 0.9, 8)
    prof = HH_A.truncation_profile(hh_data["X"], hh_data["Y"], hh_data["grid"],
                                   thetas=thetas, n_folds=5)
    assert np.allclose(prof["thetas"], thetas)
    assert prof["mse"].shape == thetas.shape == prof["se"].shape
    assert np.all(np.isfinite(prof["mse"]))
    assert prof["theta_hat"] in set(thetas)
    assert prof["theta_1se"] in set(thetas)


def test_the_profile_honours_folds_it_is_given(hh_data):
    """Where the rows are not exchangeable the caller supplies the split, so it must be the
    one actually used rather than a random one silently substituted for it."""
    n = len(hh_data["Y"])
    folds = np.arange(n) % 4
    prof = HH_A.truncation_profile(hh_data["X"], hh_data["Y"], hh_data["grid"],
                                   thetas=np.linspace(0.3, 0.9, 5), folds=folds)
    assert np.array_equal(prof["folds"], folds)


def test_the_mirrored_profile_needs_a_symmetric_grid(hh_data):
    grid = np.sort(np.random.default_rng(0).uniform(0, 1, hh_data["X"].shape[1]))
    with pytest.raises(ValueError):
        HH_A.truncation_profile(hh_data["X"], hh_data["Y"], grid, mirror=True)
