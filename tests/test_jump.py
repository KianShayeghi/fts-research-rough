"""Tests for `jump`: the design matrix's timing, the closed-form MLE, and the lasso diagnostic.

The estimator is closed form, so the interesting failures are not numerical. They are the
timing -- putting `psi_t` where `psi_{t-1}` belongs would fit a model that reads the session it
is meant to predict, and would fit it *better*, so nothing downstream would complain -- and the
generated-regressor step, where the loadings the estimator can recover are defined against
principal components it did not know in advance.

So the arrangement is checked directly against a hand-built lag, then the algebra against a
design where the truth is known exactly, then the whole route through simulated curves, where
the honest target is `<delta, phi_hat_k>` and not the coefficients the curves were drawn with.
"""

import numpy as np
import pytest

from fts_research.core.quadrature import quadrature_weights
from fts_research.jump import jump_dgp as dgp
from fts_research.jump import jump_eq as je

GRID = np.linspace(0.0, 1.0, 79)
DELTA = 0.8 * np.sin(np.pi * GRID) * GRID          # loads on the late session
K_TRUE = np.array([0.30, 0.0, -0.20])
D_TRUE = np.array([0.40, -0.25, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def clean():
    """Scores drawn as noise: the design the estimator's algebra is checked on."""
    return dgp.simulate_jump(n=600, a=0.05, gamma=0.15, d=D_TRUE, snr=4.0, k=K_TRUE, rng=42)


@pytest.fixture(scope="module")
def through_curves():
    """The long route: curves, an integral against them on the grid, then a decomposition."""
    return dgp.simulate_curves_and_jumps(
        n=400, a=0.05, gamma=0.15, delta=DELTA, snr=4.0, k=K_TRUE, grid=GRID,
        n_components=5, rng=7)


# ---------------------------------------------------------------------------- design ----


def test_the_design_lags_psi_and_does_not_lag_z():
    """`psi` describes the session before the jump and `z` the session that closed as it
    formed. Getting this backwards fits a better model of a different question."""
    J = np.arange(6.0)
    psi = np.arange(6.0).reshape(6, 1) * 10.0
    Z = np.arange(6.0).reshape(6, 1) * 100.0
    y, X, names, blocks = je.build_design(J, psi, Z)

    assert np.allclose(y, J[1:])
    assert np.allclose(X[:, blocks["gamma"]].ravel(), J[:-1])
    assert np.allclose(X[:, blocks["d"]].ravel(), psi[:-1].ravel())
    assert np.allclose(X[:, blocks["k"]].ravel(), Z[1:].ravel())
    assert names == ["const", "gamma(J_lag)", "d[0]", "k[0]"]


def test_the_blocks_partition_the_columns():
    y, X, names, blocks = je.build_design(np.arange(9.0), np.zeros((9, 3)), np.zeros((9, 2)))
    covered = sorted(i for s in blocks.values() for i in range(*s.indices(X.shape[1])))
    assert covered == list(range(X.shape[1])) == list(range(len(names)))
    assert X.shape == (8, 7)


def test_the_blocks_are_optional():
    J = np.arange(20.0) % 3
    _y, X, names, blocks = je.build_design(J)
    assert X.shape == (19, 2) and set(blocks) == {"const", "gamma"}
    assert names == ["const", "gamma(J_lag)"]


def test_a_series_too_short_to_lag_is_refused():
    with pytest.raises(ValueError, match="three sessions"):
        je.build_design(np.array([1.0, 2.0]))


def test_a_mismatched_block_is_refused():
    with pytest.raises(ValueError, match="rows"):
        je.build_design(np.arange(10.0), np.zeros((9, 2)))


# ------------------------------------------------------------------------------ mle ----


def test_the_mle_recovers_the_parameters_it_was_drawn_with(clean):
    fit = je.fit_mle(clean["J"], clean["psi"], clean["Z"])
    truth = np.concatenate([[0.05, 0.15], D_TRUE, K_TRUE])
    assert np.all(np.abs(fit.theta - truth) <= 3.0 * fit.se_ols)
    assert fit.sigma2 == pytest.approx(clean["sigma_J"] ** 2, rel=0.15)


def test_the_mle_is_the_least_squares_solution(clean):
    """It is claimed to be closed form, so it must equal a direct solve of the normal
    equations and not merely come close to one."""
    fit = je.fit_mle(clean["J"], clean["psi"], clean["Z"])
    y, X, _names, _blocks = je.build_design(clean["J"], clean["psi"], clean["Z"])
    assert np.allclose(fit.theta, np.linalg.solve(X.T @ X, X.T @ y), atol=1e-10)
    assert np.allclose(fit.resid, y - X @ fit.theta)


def test_the_reported_log_likelihood_is_the_gaussian_one(clean):
    fit = je.fit_mle(clean["J"], clean["psi"], clean["Z"])
    expected = np.sum(-0.5 * np.log(2 * np.pi * fit.sigma2)
                      - 0.5 * fit.resid ** 2 / fit.sigma2)
    assert fit.loglik == pytest.approx(expected)


def test_hac_and_homoskedastic_errors_agree_under_iid_innovations(clean):
    """The innovations here really are i.i.d., so the sandwich must not be paying much for
    robustness; a large gap would mean the lag weighting is wrong rather than that the data
    are heteroskedastic."""
    fit = je.fit_mle(clean["J"], clean["psi"], clean["Z"])
    ratio = fit.se_hac / fit.se_ols
    assert np.all(ratio > 0.6) and np.all(ratio < 1.6)


def test_hac_errors_widen_where_the_innovation_variance_follows_a_regressor():
    """The conditional mean is right and the conditional variance is not constant: the
    innovation is scaled by the size of the second score, so the homoskedastic errors are the
    optimistic ones. The sandwich must notice, and must notice most on the coefficient of the
    regressor the variance follows.

    This is the failure the model document's own remark anticipates on real jumps -- the
    volatility clusters even where the AR(1) mean is correct -- which is why `fit_mle` reports
    both sets and why the notebook quotes the second."""
    rng = np.random.default_rng(3)
    n = 3000
    psi = rng.standard_normal((n, 2))
    J = np.empty(n)
    J[0] = 0.0
    for t in range(1, n):
        scale = 0.2 + 1.5 * abs(psi[t - 1, 1])
        J[t] = 0.2 * J[t - 1] + 0.4 * psi[t - 1, 0] + scale * rng.normal()
    fit = je.fit_mle(J, psi)
    ratio = fit.se_hac / fit.se_ols
    assert ratio[3] > 1.3, "the coefficient on the noisiest direction should widen most"
    assert np.mean(ratio) > 1.05


def test_a_degenerate_fit_is_refused():
    with pytest.raises(ValueError, match="cannot identify"):
        je.fit_mle(np.arange(6.0), np.zeros((6, 9)))


def test_the_summary_names_every_coefficient(clean):
    fit = je.fit_mle(clean["J"], clean["psi"], clean["Z"])
    text = fit.summary(hac=True)
    assert all(name in text for name in fit.names)
    assert "Newey-West" in text and "homoskedastic" in fit.summary(hac=False)
    assert np.all(fit.pvalues() >= 0) and np.all(fit.pvalues() <= 1)
    assert np.allclose(fit.block("k"), fit.theta[-3:])
    with pytest.raises(KeyError, match="block"):
        je.fit_mle(np.arange(30.0) % 4).block("d")


# ----------------------------------------------------------------------------- lasso ----


def test_the_lasso_leaves_the_intercept_and_gamma_unpenalised(clean):
    out = je.fit_lasso(clean["J"], clean["psi"], clean["Z"], random_state=0)
    assert set(out["names_penalised"]) == {f"d[{i}]" for i in range(5)} | {
        f"k[{j}]" for j in range(3)}
    assert "const" not in out["coef"] and "gamma(J_lag)" not in out["coef"]


def test_the_lasso_shrinks_the_null_directions_hardest(clean):
    """`d[2:]` and `k[1]` are exactly zero in the design, and the four others are not.

    What is asserted is the *ordering* rather than an exact zero. Cross-validation selects for
    prediction, and an exact zero is worth more for support recovery than for prediction, so
    the penalty it picks need not zero anything -- which is the same split this repository
    measures for the group lasso in `notebooks/LocSpaFLR_f2s.ipynb`, arriving here from a
    different estimator."""
    out = je.fit_lasso(clean["J"], clean["psi"], clean["Z"], random_state=0)
    signal = [abs(out["coef"][name]) for name in ("d[0]", "d[1]", "k[0]", "k[2]")]
    null = [abs(out["coef"][name]) for name in ("d[2]", "d[3]", "d[4]", "k[1]")]
    assert max(null) < min(signal)
    assert max(null) < 0.1 * min(signal)
    assert set(out["selected"]) <= set(out["names_penalised"])


def test_the_lasso_beats_the_mle_on_delta_when_most_directions_are_noise():
    """The case the diagnostic exists for: many retained components, few carrying signal.
    Unpenalised estimation fits the rest, and the reconstructed delta inherits it."""
    sim = dgp.simulate_curves_and_jumps(n=200, a=0.05, gamma=0.15, delta=DELTA, snr=2.0,
                                        grid=GRID, n_components=12, rng=11)
    mle = je.fit_mle(sim["J"], sim["psi"]).block("d")
    lasso = je.fit_lasso(sim["J"], sim["psi"], random_state=0)
    fitted = np.array([lasso["coef"][f"d[{i}]"] for i in range(sim["psi"].shape[1])])

    error = lambda d: float(np.sqrt(np.mean(  # noqa: E731
        (je.reconstruct_delta(d, sim["phi"]) - DELTA) ** 2)))
    assert error(fitted) < error(mle)


def test_the_lasso_needs_something_to_penalise():
    with pytest.raises(ValueError, match="nothing to penalise"):
        je.fit_lasso(np.arange(40.0) % 5, np.zeros((40, 0)))


# ------------------------------------------------------------- the generated regressors ----


def test_the_true_loadings_are_taken_against_the_estimated_components(through_curves):
    """`d_true` must be `<delta, phi_hat_k>` under the shared quadrature. Computing it against
    the basis the curves were drawn from instead would put the estimate and its target in
    different bases and score the difference as bias."""
    sim = through_curves
    w = quadrature_weights(sim["grid"])
    assert np.allclose(sim["d_true"], (sim["phi"] * DELTA[None, :]) @ w)


def test_the_signal_is_integrated_on_the_grid_not_through_the_scores(through_curves):
    """If the recursion were driven by `d_true' psi`, the part of `delta` outside the retained
    span would be invisible and the problem would be easier than the real one."""
    sim = through_curves
    projected = sim["psi"] @ sim["d_true"]
    assert not np.allclose(sim["signal"], projected + sim["signal"].mean(), atol=1e-6)
    assert np.corrcoef(sim["signal"], projected)[0, 1] > 0.8


def test_the_leading_loadings_recover_through_the_decomposition(through_curves):
    sim = through_curves
    fit = je.fit_mle(sim["J"], sim["psi"], sim["Z"])
    assert fit.theta[1] == pytest.approx(0.15, abs=0.06)
    assert np.all(np.abs(fit.block("k") - K_TRUE) < 0.06)
    assert fit.block("d")[0] == pytest.approx(sim["d_true"][0], abs=3 * fit.se_hac[2])


def test_the_curves_start_at_zero_as_a_cumulative_return_must(through_curves):
    assert np.all(through_curves["X"][:, 0] == 0.0)


def test_reconstruct_delta_is_the_expansion_it_claims_to_be(through_curves):
    sim = through_curves
    d = np.array([1.0, -2.0, 0.5, 0.0, 0.0])
    assert np.allclose(je.reconstruct_delta(d, sim["phi"]), d @ sim["phi"])


@pytest.mark.slow
def test_the_estimator_survives_serially_dependent_curves():
    """The curves are a predetermined regressor rather than an i.i.d. draw, so a FAR(1) curve
    process should cost the estimator nothing. Paired over seeds, since a single draw would
    say more about the seed than about the process."""
    errors = {0.0: [], 0.6: []}
    for seed in range(12):
        for rho in errors:
            sim = dgp.simulate_curves_and_jumps(
                n=300, a=0.05, gamma=0.15, delta=DELTA, snr=4.0, grid=GRID,
                n_components=4, rho=rho, rng=seed)
            fit = je.fit_mle(sim["J"], sim["psi"])
            errors[rho].append(abs(fit.theta[1] - 0.15))
    assert np.median(errors[0.6]) < 3.0 * np.median(errors[0.0]) + 0.02
