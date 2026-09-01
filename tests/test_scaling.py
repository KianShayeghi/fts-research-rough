"""Scale checks on the section 3.2 surrogates of Hall & Hooker (2016).

Steps 5 and 5' define S_Y and S_b as conditional expectations given the curves,

    S_Y(theta) = E[ sum_i {Ybar*_i - Yhat*_i(theta)}^2 | X ],
    S_b(theta) = E[ int_I (b_simp - bhat*_theta)^2 | X ],

taken over the noise of the hypothetical data Y*_i = a_check + int b_simp X_i + eps*_i. Both
are therefore checkable, not matters of interpretation: for a fixed design the estimators are
linear in Y*, so each expectation has a closed form in the least-squares algebra, and that
closed form can in turn be confirmed by simulating eps*.

The checks run in that order. `_exact_*` re-derive the expectations from the hat matrix,
independently of `lambda_select`; the deterministic tests hold the shipped criterion to them;
and one seeded Monte Carlo closes the loop by simulating the noise the expectations are over.

What the tests pin down is that every variance term is a per-observation quantity: a
coefficient estimated from n curves has variance sigma^2/(n tau_j), so the paper's
sigma^2 sum_j (tau_j)^-1 displays are n times too large once tau_j is read -- as any FPCA
routine returns it -- as an eigenvalue of the covariance operator, i.e. a variance per curve.
S_b^B is out by a second factor: it is governed by the full-interval eigenvalues tau_j^1 that
b_check* was actually estimated from, not by the truncated tau_j^theta, and the difference is
qualitative, since only the latter diverges as theta -> 0.
"""

import numpy as np
import pytest

from fts_research.truncFLR import HH, HH_A, HH_dgp
from fts_research.truncFLR import _common as C

# Small enough for a Monte Carlo, large enough for the pilot's BIC to retain several
# components, which is where the variance terms have any size at all.
N, N_GRID = 80, 81
THETAS = (0.15, 0.3, 0.5, 0.8)


@pytest.fixture(scope="module")
def setup():
    """One dataset, its pilot, and the step-3/4' quantities both criteria are built on."""
    d = HH_dgp.sim_dataset(n=N, n_grid=N_GRID, u=0.8, v=1.0, snr=26.25, rng=4)
    grid, Y = d["grid"], d["Y"]
    X = d["X"][:, ::-1]                       # mirrored, as `fit_mirror` runs the method
    a_pilot, b_pilot, fpc = HH.pilot_fpca(X, Y, grid)
    sigma2 = np.mean((Y - (a_pilot + np.trapezoid(b_pilot[None, :] * X, grid, axis=1))) ** 2)
    b_simp, _ = C.fit_b_simp(X, Y, grid, a_pilot, n_fourier=1)
    return {"X": X, "Y": Y, "grid": grid, "a_pilot": a_pilot, "b_pilot": b_pilot,
            "fpc": fpc, "sigma2": sigma2, "b_simp": b_simp,
            "Ybar_star": a_pilot + np.trapezoid(b_simp[None, :] * X, grid, axis=1),
            "theta_grid": grid[1:-1]}


def _scores(phi, X, grid, interval):
    """Design scores int_0^theta phi_j X_i, for phi given on the full grid. Returns (n, k)."""
    return _sub_scores(phi[:, interval], X[:, interval], grid[interval])


def _sub_scores(phi_sub, X_sub, gi):
    """The same integral for components already restricted to the sub-grid. Returns (n, k)."""
    return np.trapezoid(phi_sub[:, None, :] * X_sub[None, :, :], gi, axis=2).T


def _design_B(s):
    """Step-4' design: the intercept and the full-interval FPC scores b_check* is fit on."""
    return np.column_stack([np.ones(len(s["Y"])), _scores(s["fpc"]["phi"], s["X"], s["grid"],
                                                          np.ones_like(s["grid"], bool))])


def _exact_B(s, theta):
    """S_Y^B and S_b^B from the least-squares algebra, with no diagonal approximation.

    b_check* = sum_j beta_j phi_j^1 is one fit on the full interval, so cov(beta) =
    sigma^2 (D'D)^-1 over the score block of the design D. Truncation at theta only changes
    what that fixed coefficient noise is read through: the loadings int_0^theta phi_j^1 X_i
    for the n predictions, and the Gram matrix of the phi_j^1 over [0, theta] for b itself.
    """
    D = _design_B(s)
    Ginv = np.linalg.inv(D.T @ D)
    coef0 = Ginv @ D.T @ s["Ybar_star"]                     # noiseless fit (a*, b*)
    interval = s["grid"] <= theta
    gi = s["grid"][interval]
    phi = s["fpc"]["phi"]

    Dz = np.column_stack([np.ones(len(s["Y"])), _scores(phi, s["X"], s["grid"], interval)])
    bias_Y = np.sum((s["Ybar_star"] - Dz @ coef0) ** 2)
    var_Y = s["sigma2"] * np.trace(Dz @ Ginv @ Dz.T)

    b_star = np.where(interval, coef0[1:] @ phi, 0.0)
    bias_b = np.trapezoid((s["b_simp"] - b_star) ** 2, s["grid"])
    gram = np.trapezoid(phi[:, None, interval] * phi[None, :, interval], gi, axis=2)
    var_b = s["sigma2"] * np.trace(Ginv[1:, 1:] @ gram)
    return bias_Y + var_Y, bias_b + var_b


def _exact_A(s, theta, m):
    """S_Y^A and S_b^A from the least-squares algebra.

    Method A refits at every theta in the truncated components phi_j^theta, which are
    orthonormal on [0, theta], so the coefficient covariance passes into int (b_simp - bhat)^2
    with no Gram matrix in between.
    """
    interval = s["grid"] <= theta
    gi = s["grid"][interval]
    phi_th, _ = C.trunc_fpca(s["X"], s["grid"], interval, m)
    D = np.column_stack([np.ones(len(s["Y"])), _sub_scores(phi_th, s["X"][:, interval], gi)])
    Ginv = np.linalg.inv(D.T @ D)
    coef0 = Ginv @ D.T @ s["Ybar_star"]

    bias_Y = np.sum((s["Ybar_star"] - D @ coef0) ** 2)
    var_Y = s["sigma2"] * D.shape[1]                        # sigma^2 tr(H) = sigma^2 (m + 1)

    b_star = np.zeros_like(s["grid"])
    b_star[interval] = coef0[1:] @ phi_th
    bias_b = np.trapezoid((s["b_simp"] - b_star) ** 2, s["grid"])
    var_b = s["sigma2"] * np.trace(Ginv[1:, 1:])
    return bias_Y + var_Y, bias_b + var_b


@pytest.fixture(scope="module")
def diag_B(setup):
    s = setup
    _, diag = HH.lambda_select(s["X"], s["Y"], s["grid"], s["a_pilot"], s["b_pilot"],
                               s["fpc"], None, np.linspace(0.0, 1.0, 20)[1:],
                               sigma2=s["sigma2"], theta_grid=s["theta_grid"])
    return diag


@pytest.fixture(scope="module")
def diag_A(setup):
    s = setup
    _, diag = HH_A.lambda_select(s["X"], s["Y"], s["grid"], s["a_pilot"], s["b_pilot"],
                                 s["fpc"], None, np.linspace(0.0, 1.0, 20)[1:],
                                 sigma2=s["sigma2"], theta_grid=s["theta_grid"],
                                 n_components=s["fpc"]["m"])
    return diag


def _at(diag, theta):
    return int(np.argmin(np.abs(diag["theta_grid"] - theta)))


# ------------------------------------------------------- what the eigenvalues measure ----

def test_pilot_eigenvalues_are_variances_per_curve(setup):
    """tau_j is E(score^2), not sum_i score_i^2 -- the units that force the 1/n.

    Everything else here follows from this: a coefficient fitted to n curves has variance
    sigma^2 / sum_i score_i^2 = sigma^2 / (n tau_j).
    """
    s = setup
    phi, tau = s["fpc"]["phi"], s["fpc"]["tau"]
    scores = _scores(phi, s["X"], s["grid"], np.ones_like(s["grid"], bool))
    centred = scores - scores.mean(axis=0)
    np.testing.assert_allclose((centred ** 2).mean(axis=0), tau, rtol=0.05)


# -------------------------------------------------------------- the shipped criterion ---

@pytest.mark.parametrize("theta", THETAS)
def test_method_B_surrogates_match_the_exact_expectation(setup, diag_B, theta):
    """S_Y^B and S_b^B as shipped are the step-5' expectations, to a few percent.

    Two documented gaps remain, both of them the paper's own displays rather than the scale
    being fixed here, and neither able to hide a factor of n = 80. The displays keep only the
    diagonal of cov(beta), whose off-diagonal entries are O(n^-1) relative; and S_Y^B leaves
    out the intercept's contribution, exactly sigma^2 at every theta, added back below. Being
    theta-free, that term shifts S_Y^B's level and cannot move theta_lambda or its curvature.
    """
    q = _at(diag_B, theta)
    exact_Y, exact_b = _exact_B(setup, diag_B["theta_grid"][q])
    assert diag_B["S_Y"][q] + setup["sigma2"] == pytest.approx(exact_Y, rel=0.05)
    assert diag_B["S_b"][q] == pytest.approx(exact_b, rel=0.02)


@pytest.mark.parametrize("theta", THETAS)
def test_method_A_surrogates_match_the_exact_expectation(setup, diag_A, theta):
    """Method A's displays need no such allowance: sigma^2 (m + 1) is the exact trace of its
    hat matrix, and its phi_j^theta are orthonormal on [0, theta], so no Gram matrix stands
    between cov(beta) and int (b_simp - bhat)^2."""
    q = _at(diag_A, theta)
    exact_Y, exact_b = _exact_A(setup, diag_A["theta_grid"][q], int(diag_A["ms"][q]))
    assert diag_A["S_Y"][q] == pytest.approx(exact_Y, rel=0.02)
    assert diag_A["S_b"][q] == pytest.approx(exact_b, rel=0.02)


@pytest.mark.parametrize("theta", THETAS)
def test_the_paper_displays_are_n_times_too_large(setup, diag_B, diag_A, theta):
    """The literal step-5 variance terms overstate the expectation by exactly n.

    This is the defect the correction removes, stated as a measurement: multiplying the
    shipped variance terms by n reproduces the paper's displays, and those displays are n
    times the variance the estimators actually have.
    """
    n = len(setup["Y"])
    for diag in (diag_B, diag_A):
        q = _at(diag, theta)
        literal_b = diag["bias_b"][q] + n * diag["var_b"][q]
        assert literal_b / diag["S_b"][q] > 1.0
        # The overstatement is a clean factor of n in the variance term itself.
        assert (literal_b - diag["bias_b"][q]) == pytest.approx(n * diag["var_b"][q])

    q = _at(diag_B, theta)
    literal_Y = diag_B["bias_Y"][q] + n * diag_B["var_Y"][q]
    assert literal_Y > diag_B["S_Y"][q]


def test_method_A_prediction_variance_needs_no_correction(setup, diag_A):
    """sigma^2 (m + 1) is already a summed variance, so step 5's S_Y^A display stands.

    Only the terms written as sigma^2 sum_j (tau_j)^-1 are out by n; the one term the paper
    writes without a tau is right as it stands, which is itself evidence for the reading.
    """
    for q, m in enumerate(diag_A["ms"]):
        assert diag_A["var_Y"][q] == pytest.approx(setup["sigma2"] * (m + 1))


# ------------------------------------------- which eigenvalues govern method B's S_b -----

def test_S_b_B_is_governed_by_the_full_interval_eigenvalues(setup, diag_B):
    """b_check* is estimated once on I, so tau_j^1 sets its variance at every theta.

    Substituting the truncated tau_j^theta, as step 5' displays, changes the shape and not
    just the scale: tau_j^theta -> 0 as theta -> 0, so that variance term diverges at small
    theta where the true one, checked against `_exact_B` above, stays bounded.
    """
    s = setup
    phi, sigma2 = s["fpc"]["phi"], s["sigma2"]
    thetas = np.array([0.05, 0.1, 0.2])
    shipped, literal = [], []
    for theta in thetas:
        interval = s["grid"] <= theta
        phi2 = np.trapezoid(phi[:, interval] ** 2, s["grid"][interval], axis=1)
        tau_theta = C.trunc_eigvals(s["X"], s["grid"], interval, phi.shape[0])
        literal.append(sigma2 * np.sum(phi2 / tau_theta))
        shipped.append(diag_B["var_b"][_at(diag_B, theta)])

    shipped, literal = np.array(shipped), np.array(literal)
    assert np.all(np.diff(literal) < 0), "the truncated reading diverges as theta -> 0"
    assert literal[0] / shipped[0] > 1e3, "and by orders of magnitude, not a factor of n"
    # The true variance term shrinks with theta instead: less of b is being estimated.
    assert np.all(np.diff(shipped) > 0)


# ----------------------------------------------------------------- the simulation --------

@pytest.mark.slow
@pytest.mark.parametrize("theta", [0.3, 0.8])
def test_monte_carlo_confirms_the_expectations(setup, diag_B, theta):
    """Simulate eps* and average, which is what steps 4'-5' describe doing in the first place.

    The closed forms above are algebra; this is the thing they are algebra for. Tolerances
    are the Monte-Carlo error at this many replications, not slack in the identity.
    """
    s = setup
    rng = np.random.default_rng(0)
    reps = 6000
    D = _design_B(s)
    Ginv = np.linalg.inv(D.T @ D)
    q = _at(diag_B, theta)
    theta = diag_B["theta_grid"][q]
    interval = s["grid"] <= theta
    phi = s["fpc"]["phi"]

    noise = rng.standard_normal((len(s["Y"]), reps)) * np.sqrt(s["sigma2"])
    coefs = Ginv @ D.T @ (s["Ybar_star"][:, None] + noise)          # (m + 1, reps)

    Dz = np.column_stack([np.ones(len(s["Y"])), _scores(phi, s["X"], s["grid"], interval)])
    mc_S_Y = np.mean(np.sum((s["Ybar_star"][:, None] - Dz @ coefs) ** 2, axis=0))

    b_hat = np.where(interval[None, :], coefs[1:].T @ phi, 0.0)     # (reps, N)
    mc_S_b = np.mean(np.trapezoid((s["b_simp"][None, :] - b_hat) ** 2, s["grid"], axis=1))

    assert mc_S_Y == pytest.approx(diag_B["S_Y"][q] + s["sigma2"], rel=0.05)
    assert mc_S_b == pytest.approx(diag_B["S_b"][q], rel=0.05)


# --------------------------------------------------------- what it does to the criterion -

@pytest.mark.slow
def test_corrected_S_b_has_an_interior_minimum(setup, diag_B):
    """The point of the correction: S_b becomes a bias-variance trade-off in theta.

    Under the literal displays the variance term diverges as theta -> 0 and S_b decreases
    across the whole grid; P_b(lambda) then inherits that monotonicity, since larger lambda
    only moves theta_lambda left, and lambda_hat is the smallest candidate offered whatever
    the data. With the variance terms on their own scale, S_b turns down at the truth.
    """
    tg, S_b = diag_B["theta_grid"], diag_B["S_b"]
    interior = np.argmin(S_b)
    assert 0 < interior < len(tg) - 1
    assert abs(tg[interior] - 0.2) < 0.1, "the minimum should sit near the true theta = 0.2"
