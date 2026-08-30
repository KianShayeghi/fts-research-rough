"""Tests for `far.far_est`: the FAR(1) estimator, the two dependence tests, and the
out-of-sample forecast loop.

The estimator is checked against `far.far_dgp`, which supplies curves whose operator is
known exactly, so "did it recover the kernel" is a question with an answer rather than a
plausibility argument. The two tests are checked for *size* under the i.i.d. null as well as
for power, because a dependence test that over-rejects would manufacture the finance
notebook's entire conclusion.
"""

import numpy as np
import pytest

from far import far_dgp as dgp
from far import far_est as est
from truncFLR._common import quadrature_weights

GRID = np.linspace(0.0, 1.0, 40)


def weights():
    return quadrature_weights(GRID)


def true_kernel(Psi):
    """`far_dgp`'s operators carry the quadrature weights folded in, so `Psi @ x` integrates.
    Dividing them out recovers the kernel values `far1_fit` estimates."""
    return Psi / weights()[None, :]


def hs(kernel):
    w = weights()
    return float(np.sqrt(np.sum((kernel ** 2) * w[None, :] * w[:, None])))


@pytest.fixture(scope="module")
def indep():
    """Curves with no serial dependence at all: the null both tests must respect."""
    return dgp.sample_far1(300, GRID, np.zeros((GRID.size, GRID.size)), rng=1)


@pytest.fixture(scope="module")
def dependent():
    """A full-rank FAR(1) at rho = 0.7, the alternative both tests must detect."""
    Psi = dgp.exponential_kernel(GRID, rho=0.7)
    return {"Psi": Psi, "X": dgp.sample_far1(800, GRID, Psi, rng=2)}


# --------------------------------------------------------------------------------- fpca


def test_components_are_orthonormal_in_the_l2_inner_product(indep):
    """`phi` are unweighted by sqrt(w) so that `int phi_j phi_k = delta_jk`, not so that the
    raw vectors are Euclidean-orthonormal. Checking the wrong one would pass on a uniform
    grid and hide the bug everywhere else."""
    phi = est.fpca(indep, GRID, n_components=6)["phi"]
    gram = np.trapezoid(phi[:, None, :] * phi[None, :, :], GRID, axis=2)
    assert np.allclose(gram, np.eye(6), atol=1e-8)


def test_eigenvalues_descend_and_sum_to_the_total_variance(indep):
    dec = est.fpca(indep, GRID)
    assert np.all(np.diff(dec["tau"]) <= 1e-12)
    assert dec["tau"].sum() == pytest.approx(dec["total"], rel=1e-8)


def test_scores_reconstruct_the_curves(indep):
    """With every component kept, mean + scores @ phi must be the original curves back."""
    dec = est.fpca(indep, GRID)
    assert np.allclose(dec["mean"] + dec["scores"] @ dec["phi"], indep, atol=1e-8)


def test_n_components_truncates_rather_than_changing_the_decomposition(indep):
    full = est.fpca(indep, GRID)
    few = est.fpca(indep, GRID, n_components=3)
    assert np.allclose(np.abs(few["phi"]), np.abs(full["phi"][:3]), atol=1e-8)
    assert np.allclose(few["tau"], full["tau"][:3])


# ---------------------------------------------------------------------------- far1_fit


def test_psi_solves_the_score_space_normal_equations(dependent):
    """`Psi` is claimed to be the least-squares regression of each score vector on its
    predecessor, so the residuals must be orthogonal to the regressors."""
    fit = est.far1_fit(dependent["X"], GRID, p=4)
    xi = fit["fpca"]["scores"]
    resid = xi[1:] - xi[:-1] @ fit["Psi"].T
    assert np.allclose(xi[:-1].T @ resid, 0.0, atol=1e-6)


def test_kernel_is_recovered_from_a_long_series(dependent):
    """The estimate must approach the operator that generated the data, not merely be
    stable. Scored relative to the true kernel's own Hilbert-Schmidt norm."""
    fit = est.far1_fit(dependent["X"], GRID, p=4)
    truth = true_kernel(dependent["Psi"])
    assert hs(fit["kernel"] - truth) / hs(truth) < 0.35
    assert fit["hs_norm"] == pytest.approx(dgp.hs_norm(dependent["Psi"], GRID), abs=0.08)


def test_independent_curves_yield_a_much_smaller_operator(indep, dependent):
    """Under independence the fitted norm is estimation noise, not zero; what must hold is
    that it is far below the dependent case at the same p and sample size."""
    null = est.far1_fit(indep, GRID, p=4)
    alt = est.far1_fit(dependent["X"][:300], GRID, p=4)
    assert null["hs_norm"] < 0.4 * alt["hs_norm"]
    assert null["r2"] < 0.05


def test_operator_norm_never_exceeds_the_hilbert_schmidt_norm(dependent):
    fit = est.far1_fit(dependent["X"], GRID, p=5)
    assert fit["op_norm"] <= fit["hs_norm"] + 1e-9


def test_more_components_inflate_the_fitted_norm(dependent):
    """The covariance operator's inverse is unbounded, so each extra component is divided by
    a smaller eigenvalue and the estimate drifts upward. This is the reason `p` regularises,
    and it is asserted rather than described."""
    norms = [est.far1_fit(dependent["X"][:200], GRID, p=p)["hs_norm"] for p in (2, 6, 12)]
    assert norms[0] < norms[1] < norms[2]


def test_ridge_shrinks_the_operator(dependent):
    plain = est.far1_fit(dependent["X"][:150], GRID, p=8, ridge=0.0)
    shrunk = est.far1_fit(dependent["X"][:150], GRID, p=8, ridge=0.5)
    assert shrunk["hs_norm"] < plain["hs_norm"]


def test_fit_rejects_a_series_too_short_to_fit(dependent):
    with pytest.raises(ValueError, match="at least three curves"):
        est.far1_fit(dependent["X"][:2], GRID, p=1)


# ------------------------------------------------------------------------ far1_predict


def test_predict_on_the_training_lags_reproduces_the_fitted_values(dependent):
    fit = est.far1_fit(dependent["X"], GRID, p=4)
    assert np.allclose(est.far1_predict(fit, dependent["X"][:-1]), fit["fitted"], atol=1e-9)


def test_predict_accepts_a_single_curve(dependent):
    fit = est.far1_fit(dependent["X"], GRID, p=4)
    one = est.far1_predict(fit, dependent["X"][0])
    assert one.shape == (1, GRID.size)
    assert np.allclose(one[0], fit["fitted"][0])


def test_predicting_from_the_mean_curve_returns_the_mean_curve(dependent):
    """A curve with no deviation from the mean has zero scores, so the operator has nothing
    to act on and the forecast is the mean itself."""
    fit = est.far1_fit(dependent["X"], GRID, p=4)
    mean = fit["fpca"]["mean"]
    assert np.allclose(est.far1_predict(fit, mean)[0], mean, atol=1e-9)


# ----------------------------------------------------------------------------- acf_norms


def test_dependence_shows_up_at_lag_one(indep, dependent):
    assert est.acf_norms(dependent["X"], GRID, max_lag=5)[0] > 0.4
    assert est.acf_norms(indep, GRID, max_lag=5)[0] < 0.15


def test_acf_norms_decay_with_the_lag_for_a_far1(dependent):
    """A FAR(1)'s lag-h autocovariance is Psi^h applied to the lag-0 one, so its norm falls
    geometrically; the first lag must dominate the fifth."""
    acf = est.acf_norms(dependent["X"], GRID, max_lag=5)
    assert acf[0] > 3 * acf[4]


def test_acf_norms_are_invariant_to_reversing_time(dependent):
    """The lag-h norm is a norm of a cross-product matrix; reversing the series transposes
    each one, which leaves the Frobenius norm alone."""
    forward = est.acf_norms(dependent["X"], GRID, max_lag=4)
    backward = est.acf_norms(dependent["X"][::-1], GRID, max_lag=4)
    assert np.allclose(forward, backward, atol=1e-10)


# ----------------------------------------------------------------------- the two tests


def test_portmanteau_degrees_of_freedom_are_p_squared_times_the_lags(indep):
    out = est.portmanteau_test(indep, GRID, max_lag=5, n_components=3)
    assert out["df"] == 3 * 3 * 5


def test_both_tests_accept_independent_curves(indep):
    assert est.portmanteau_test(indep, GRID, max_lag=5, n_components=3)["p_value"] > 0.1
    assert est.permutation_test(indep, GRID, max_lag=5, n_perm=199, rng=4)["p_value"] > 0.1


def test_both_tests_reject_a_far1(dependent):
    assert est.portmanteau_test(dependent["X"], GRID, max_lag=5, n_components=3)["p_value"] < 1e-6
    assert est.permutation_test(dependent["X"], GRID, max_lag=5, n_perm=199, rng=4)["p_value"] \
        <= 0.005


def test_permutation_p_value_can_never_be_zero(dependent):
    """The observed statistic joins its own null, which is what makes the test exact; a
    p-value of 0 would mean it had not."""
    out = est.permutation_test(dependent["X"], GRID, max_lag=3, n_perm=99, rng=0)
    assert out["p_value"] >= 1.0 / 100
    assert out["null"].size == 99


def test_permutation_test_is_reproducible(indep):
    a = est.permutation_test(indep, GRID, max_lag=3, n_perm=99, rng=7)
    b = est.permutation_test(indep, GRID, max_lag=3, n_perm=99, rng=7)
    assert a["p_value"] == b["p_value"]
    assert np.allclose(a["null"], b["null"])


def test_permutation_reference_does_not_depend_on_the_observed_ordering(dependent):
    """Reshuffling destroys the ordering, so the null distribution is a property of the
    *set* of curves and not of the order they arrived in.

    A fixed seed draws the same *index* permutations either way, which land on different
    rows once the input is reordered, so the two runs are independent samples from one
    distribution rather than the same sample. What must match is therefore the distribution:
    the two null means are compared against the Monte Carlo error of the difference.
    """
    X = dependent["X"][:200]
    a = est.permutation_test(X, GRID, max_lag=3, n_perm=499, rng=3)
    b = est.permutation_test(X[::-1], GRID, max_lag=3, n_perm=499, rng=3)

    se = np.sqrt(a["null"].var(ddof=1) / a["null"].size + b["null"].var(ddof=1) / b["null"].size)
    assert abs(a["null"].mean() - b["null"].mean()) < 4 * se
    assert a["observed"] == pytest.approx(b["observed"], rel=1e-9)


@pytest.mark.slow
def test_both_tests_hold_their_size_under_the_null():
    """A dependence test that over-rejects would invent serial dependence in the finance
    notebook wherever there is none, so the null rejection rate is measured rather than
    assumed. 120 replications put a 5% rate's standard error near 2 points, so the bound is
    set at 12%."""
    rng = np.random.default_rng(11)
    zero = np.zeros((GRID.size, GRID.size))
    chi, perm = [], []
    for _ in range(120):
        X = dgp.sample_far1(150, GRID, zero, rng=rng)
        chi.append(est.portmanteau_test(X, GRID, max_lag=5, n_components=3)["p_value"])
        perm.append(est.permutation_test(X, GRID, max_lag=5, n_perm=199, rng=rng)["p_value"])
    assert np.mean(np.array(chi) <= 0.05) <= 0.12
    assert np.mean(np.array(perm) <= 0.05) <= 0.12


# ------------------------------------------------------------------- selection, forecast


def test_select_p_returns_a_candidate_and_beats_the_flat_forecast(dependent):
    out = est.select_p(dependent["X"][:300], GRID, p_range=range(1, 8))
    assert out["p_hat"] in set(out["p_values"])
    assert out["mse"].min() < out["mse_flat"]


def test_select_p_rejects_a_test_block_that_leaves_nothing_to_fit(dependent):
    with pytest.raises(ValueError, match="too little to fit"):
        est.select_p(dependent["X"][:10], GRID, n_test=8)


def test_rolling_forecast_beats_doing_nothing_when_there_is_dependence(dependent):
    out = est.rolling_forecast(dependent["X"][:250], GRID, p=4, min_train=120)
    assert out["r2_oos"] > 0.05
    assert out["pred"].shape == out["truth"].shape


def test_rolling_forecast_does_not_beat_doing_nothing_on_independent_curves(indep):
    """The honest null result: with no dependence to exploit, refitting an operator costs
    estimation variance and must not look like skill."""
    out = est.rolling_forecast(indep[:250], GRID, p=4, min_train=120)
    assert out["r2_oos"] < 0.05


def test_rolling_forecast_never_looks_ahead(dependent):
    """Every forecast is made from an expanding window ending at t - 1, so perturbing the
    *last* curve in the series must leave every earlier forecast untouched. This is the
    property that makes `r2_oos` an out-of-sample number at all."""
    X = dependent["X"][:200].copy()
    base = est.rolling_forecast(X, GRID, p=3, min_train=150)

    tampered = X.copy()
    tampered[-1] += 50.0
    after = est.rolling_forecast(tampered, GRID, p=3, min_train=150)

    assert np.allclose(base["pred"][:-1], after["pred"][:-1], atol=1e-10)


def test_rolling_forecast_rejects_a_window_leaving_nothing_to_forecast(dependent):
    with pytest.raises(ValueError, match="nothing to forecast"):
        est.rolling_forecast(dependent["X"][:50], GRID, p=3, min_train=50)
