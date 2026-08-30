"""Tests for the two data-generating modules: `far.far_dgp` and `fdaSP.SBC_sim`.

A DGP that is silently wrong invalidates every simulation result built on it, so these
check the actual mathematical properties -- the FAR(1) operator's Hilbert-Schmidt norm, the
stationarity that follows from it, the serial dependence it induces, and the null regions of
the paper's three test functions.
"""

import numpy as np
import pytest

from far import far_dgp
from fdaSP import SBC, SBC_sim


# --------------------------------------------------------------------- FAR(1) operator ---

@pytest.mark.parametrize("rho", [0.3, 0.7, 0.9])
def test_gaussian_kernel_has_the_requested_hs_norm(rho):
    """The rescaling must actually land on ||psi||_HS = rho, since rho < 1 is what buys
    stationarity."""
    grid = np.linspace(0, 1, 201)
    Psi = far_dgp.gaussian_kernel(grid, rho=rho)
    w = np.zeros_like(grid)
    w[1:] += np.diff(grid)
    w[:-1] += np.diff(grid)
    w *= 0.5
    K = Psi / w[None, :]                              # undo the folded-in quadrature weights
    hs = np.sqrt(np.sum((K ** 2) * w[None, :] * w[:, None]))
    assert hs == pytest.approx(rho, rel=1e-8)


def test_gaussian_kernel_approximates_the_integral_operator():
    """Psi @ x must approximate int psi(s, r) x(r) dr."""
    grid = np.linspace(0, 1, 401)
    Psi = far_dgp.gaussian_kernel(grid, rho=1.0)
    S, R = np.meshgrid(grid, grid, indexing="ij")
    K = np.exp(-(S ** 2 + R ** 2) / 2.0)
    w = np.zeros_like(grid)
    w[1:] += np.diff(grid)
    w[:-1] += np.diff(grid)
    w *= 0.5
    hs = np.sqrt(np.sum((K ** 2) * w[None, :] * w[:, None]))
    x = np.sin(3 * grid)
    expected = np.trapezoid((K / hs) * x[None, :], grid, axis=1)
    np.testing.assert_allclose(Psi @ x, expected, atol=1e-6)


def test_gaussian_kernel_is_symmetric():
    grid = np.linspace(0, 1, 101)
    K = far_dgp.gaussian_kernel(grid, rho=0.7)
    w = np.zeros_like(grid)
    w[1:] += np.diff(grid)
    w[:-1] += np.diff(grid)
    w *= 0.5
    np.testing.assert_allclose(K / w[None, :], (K / w[None, :]).T, atol=1e-12)


# ----------------------------------------------------------------------- the process -----

def test_sample_far1_shapes_and_reproducibility():
    grid = np.linspace(0, 1, 51)
    Psi = far_dgp.gaussian_kernel(grid, rho=0.7)
    a = far_dgp.sample_far1(30, grid, Psi, burn_in=20, rng=4)
    b = far_dgp.sample_far1(30, grid, Psi, burn_in=20, rng=4)
    assert a.shape == (30, 51)
    np.testing.assert_array_equal(a, b)


def test_sample_far1_is_stationary_after_burn_in():
    """With rho < 1 the marginal variance must not drift between the start and end."""
    grid = np.linspace(0, 1, 61)
    Psi = far_dgp.gaussian_kernel(grid, rho=0.8)
    X = far_dgp.sample_far1(2000, grid, Psi, burn_in=300, rng=0)
    first, last = X[:500].var(axis=0), X[-500:].var(axis=0)
    assert np.max(np.abs(first - last)) / first.mean() < 0.35


def test_sample_far1_actually_induces_serial_dependence():
    grid = np.linspace(0, 1, 61)
    scores = {}
    for rho in (0.0, 0.5, 0.9):
        Psi = far_dgp.gaussian_kernel(grid, rho=rho)
        X = far_dgp.sample_far1(3000, grid, Psi, burn_in=300, rng=1)
        proj = np.trapezoid(X, grid, axis=1)       # a scalar summary of each curve
        scores[rho] = far_dgp.lag1_autocorr(proj)

    assert abs(scores[0.0]) < 0.06, "rho = 0 must give (near) independent curves"
    assert scores[0.5] > 0.2
    assert scores[0.9] > scores[0.5], "more dependence must show up as more autocorrelation"


def test_lag1_autocorr_on_a_known_series():
    rng = np.random.default_rng(0)
    n = 200000
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + e[t]
    assert far_dgp.lag1_autocorr(x) == pytest.approx(0.6, abs=0.02)


def test_lag1_autocorr_is_bounded():
    rng = np.random.default_rng(2)
    assert abs(far_dgp.lag1_autocorr(rng.standard_normal(500))) <= 1.0
    assert far_dgp.lag1_autocorr(np.arange(100.0)) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- drop-in DGP wrappers ---

def test_sim_dataset_window_matches_the_iid_interface():
    """It must be substitutable for HH_dgp.sim_dataset, keys and all."""
    from truncFLR import HH_dgp
    far = far_dgp.sim_dataset_window(n=40, grid=np.linspace(0, 1, 61), u=0.8, rng=0)
    iid = HH_dgp.sim_dataset(n=40, n_grid=61, u=0.8, rng=0)
    assert set(iid).issubset(set(far))
    assert far["X"].shape == iid["X"].shape
    np.testing.assert_array_equal(far["b"], iid["b"])
    assert far["rho"] == 0.7


def test_sim_dataset_psi_matches_the_iid_interface():
    X, y, grid, beta = far_dgp.sim_dataset_psi(n=30, psi_fn=far_dgp.psi1, rng=0)
    assert X.shape == (30, grid.size)
    assert y.shape == (30,)
    np.testing.assert_allclose(beta, SBC_sim.psi1(grid))


def test_sim_dataset_window_hits_the_requested_snr():
    d = far_dgp.sim_dataset_window(n=3000, grid=np.linspace(0, 1, 61), u=0.8, snr=10.0, rng=1)
    from truncFLR import HH_dgp
    signal = HH_dgp.trunc_int(d["X"], d["grid"], u=0.8, v=1.0)
    assert np.var(signal) / d["sigma"] ** 2 == pytest.approx(10.0, rel=1e-6)


# ------------------------------------------------------------------ the test functions ---

def test_psi1_vanishes_on_its_null_region():
    t = np.linspace(0, 1, 1001)
    v = SBC_sim.psi1(t)
    null = (t > 0.3) & (t < 0.7)
    np.testing.assert_array_equal(v[null], 0.0)
    assert np.any(np.abs(v[~null]) > 0.1)


def test_psi2_vanishes_outside_its_support():
    t = np.linspace(0, 1, 1001)
    v = SBC_sim.psi2(t)
    outside = (t < 0.25) | (t > 0.75)
    np.testing.assert_array_equal(v[outside], 0.0)
    assert v.max() == pytest.approx(1.0, abs=1e-3)   # Epanechnikov peak at t = 0.5


def test_psi3_has_no_null_region():
    t = np.linspace(0, 1, 1001)
    v = SBC_sim.psi3(t)
    assert np.count_nonzero(v == 0.0) <= 1           # only the single root at t = 0.5


def test_true_functions_registry_masks_match_the_functions():
    t = np.linspace(0, 1, 501)
    for name, (fn, null_mask) in SBC_sim.TRUE_FUNCTIONS.items():
        mask = null_mask(t)
        np.testing.assert_allclose(fn(t)[mask], 0.0, atol=1e-12,
                                   err_msg=f"{name}: declared null region is not null")


def test_simulate_shapes_and_snr():
    X, y, grid, beta = SBC_sim.simulate(SBC_sim.psi1, n=3000, m=80, snr=4.0, rng=0)
    assert X.shape == (3000, 80)
    signal = np.trapezoid(X * beta, grid, axis=1)
    assert np.var(signal) / np.var(y - signal) == pytest.approx(4.0, rel=0.1)


def test_simulate_is_reproducible():
    a = SBC_sim.simulate(SBC_sim.psi2, n=20, m=40, rng=5)
    b = SBC_sim.simulate(SBC_sim.psi2, n=20, m=40, rng=5)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_ise_matches_a_direct_integral():
    grid = np.linspace(0, 1, 401)
    a, b = np.sin(grid), np.zeros_like(grid)
    assert SBC_sim.ise(a, b, grid) == pytest.approx(np.trapezoid(np.sin(grid) ** 2, grid))
    assert SBC_sim.ise(a, a, grid) == pytest.approx(0.0)


def test_ise_on_an_empty_mask_is_nan():
    grid = np.linspace(0, 1, 11)
    assert np.isnan(SBC_sim.ise(grid, grid, grid, mask=np.zeros(11, bool)))


@pytest.mark.parametrize("mask_fn", [
    lambda g: g <= 0.5,                             # one interval
    lambda g: (g >= 0.3) & (g <= 0.7),              # psi1's null set
    lambda g: (g < 0.25) | (g > 0.75),              # psi2's null set, two intervals
    lambda g: (g < 0.3) | (g > 0.7),                # psi1's supported set, two intervals
    lambda g: (g < 0.1) | ((g > 0.4) & (g < 0.5)) | (g > 0.9),   # three intervals
])
def test_ise_splits_additively_over_a_partition(mask_fn):
    """ISE over a region and over its complement must sum to the ISE over the whole domain.

    The masks that matter are the disjoint ones: psi2 scores its null set over
    (t < 0.25) | (t > 0.75), and psi1 its supported set over (t < 0.3) | (t > 0.7).
    """
    grid = np.linspace(0, 1, 601)
    a, b = np.sin(3 * grid), np.cos(grid)
    mask = mask_fn(grid)
    total = SBC_sim.ise(a, b, grid)
    parts = SBC_sim.ise(a, b, grid, mask) + SBC_sim.ise(a, b, grid, ~mask)
    assert parts == pytest.approx(total, rel=1e-12)


def test_ise_over_a_region_is_bounded_by_the_total():
    """A sub-region can never score above the whole domain, whatever shape it is."""
    grid = np.linspace(0, 1, 601)
    a, b = np.sin(3 * grid), np.cos(grid)
    total = SBC_sim.ise(a, b, grid)
    for mask in [(grid < 0.25) | (grid > 0.75), (grid < 0.3) | (grid > 0.7)]:
        assert 0.0 <= SBC_sim.ise(a, b, grid, mask) <= total


def test_ise_ignores_the_gap_between_disjoint_intervals():
    """Error confined to the gap must not leak into the regions either side of it."""
    grid = np.linspace(0, 1, 601)
    beta_true = np.zeros_like(grid)
    beta_hat = np.where((grid > 0.25) & (grid < 0.75), 5.0, 0.0)
    outside = (grid <= 0.25) | (grid >= 0.75)
    assert SBC_sim.ise(beta_hat, beta_true, grid, outside) == pytest.approx(0.0, abs=1e-12)


def test_ise_matches_a_direct_integral_over_a_union_of_intervals():
    grid = np.linspace(0, 1, 20001)
    a, b = np.sin(3 * grid), np.zeros_like(grid)
    mask = (grid < 0.25) | (grid > 0.75)
    direct = (np.trapezoid(np.sin(3 * grid[grid < 0.25]) ** 2, grid[grid < 0.25])
              + np.trapezoid(np.sin(3 * grid[grid > 0.75]) ** 2, grid[grid > 0.75]))
    assert SBC_sim.ise(a, b, grid, mask) == pytest.approx(direct, rel=1e-3)


@pytest.mark.slow
def test_run_sim_reports_an_additive_null_supported_split():
    """The reported ISE_T0 and ISE_T1 must add to the reported ISE, for every function.

    psi1's supported set and psi2's null set are each a union of two intervals, so this is the
    end-to-end form of the check on `ise` above: a split that integrates the masked points in
    one pass instead of restricting the weights would break it on exactly those two.
    """
    results = SBC_sim.run_sim(which=("psi1", "psi2", "psi3"), n_rep=2, n=30, m=60,
                              n_basis=10, n_folds=3, n_lambda=6, verbose=False)
    for row in SBC_sim.summarise(results):
        parts = row["ise_T1_mean"]
        if not np.isnan(row["ise_T0_mean"]):
            parts += row["ise_T0_mean"]
        assert parts == pytest.approx(row["ise_mean"], rel=1e-10), (
            f"{row['function']}/{row['method']}: T0 + T1 = {parts} but ISE = {row['ise_mean']}"
        )


@pytest.mark.slow
def test_fpca_reg_recovers_a_smooth_coefficient():
    X, y, grid, beta = SBC_sim.simulate(SBC_sim.psi3, n=200, m=80, snr=20.0, rng=3)
    beta_hat, k = SBC_sim.fpca_reg(X, y, grid, max_components=8, cv=5, rng=0)
    assert 1 <= k <= 8
    assert SBC_sim.ise(beta_hat, beta, grid) < SBC_sim.ise(np.zeros_like(beta), beta, grid)


@pytest.mark.slow
def test_ogl_beats_the_zero_function_on_a_far1_dataset():
    """The point of the FAR(1) module: the estimators still work on non-i.i.d. curves."""
    X, y, grid, beta = far_dgp.sim_dataset_psi(n=120, psi_fn=far_dgp.psi1, rho=0.7,
                                               snr=8.0, rng=2)
    cv = SBC.cv_fit(X, y - y.mean(), grid, n_basis=12, order=4, n_folds=5, n_lambda=15, rng=0)
    assert SBC_sim.ise(cv["beta"], beta, grid) < SBC_sim.ise(np.zeros_like(beta), beta, grid)


# ------------------------------------------------------------------------ kernel rank ----


@pytest.mark.parametrize("rho", [0.3, 0.7, 0.9])
def test_exponential_kernel_has_the_requested_hs_norm(rho):
    grid = np.linspace(0, 1, 200)
    Psi = far_dgp.exponential_kernel(grid, rho=rho)
    assert far_dgp.hs_norm(Psi, grid) == pytest.approx(rho, rel=1e-10)


def test_gaussian_kernel_has_rank_one():
    """exp{-(s^2 + r^2)/2} factorises, so only one direction in L2 is autocorrelated."""
    grid = np.linspace(0, 1, 200)
    assert far_dgp.kernel_rank(far_dgp.gaussian_kernel(grid, rho=0.7), grid) == 1


def test_exponential_kernel_has_full_rank():
    grid = np.linspace(0, 1, 200)
    assert far_dgp.kernel_rank(far_dgp.exponential_kernel(grid, rho=0.7), grid) == len(grid)


def test_operator_norm_equals_hs_norm_only_at_rank_one():
    """||Psi||_op <= ||Psi||_HS always, with equality exactly for a rank-one kernel."""
    grid = np.linspace(0, 1, 200)
    gauss = far_dgp.gaussian_kernel(grid, rho=0.7)
    assert far_dgp.operator_norm(gauss, grid) == pytest.approx(0.7, rel=1e-9)

    expo = far_dgp.exponential_kernel(grid, rho=0.7)
    assert far_dgp.operator_norm(expo, grid) < far_dgp.hs_norm(expo, grid)
    assert far_dgp.operator_norm(expo, grid) < 1.0     # still contractive, so stationary


def test_exponential_kernel_is_symmetric():
    grid = np.linspace(0, 1, 120)
    w = far_dgp._trapezoid_weights(grid)
    K = far_dgp.exponential_kernel(grid, rho=0.7) / w[None, :]
    np.testing.assert_allclose(K, K.T, rtol=1e-12)


@pytest.mark.parametrize("k", [2, 4, 6])
def test_exponential_kernel_induces_serial_dependence_across_the_domain(k):
    """A full-rank operator autocorrelates every direction, not just one.

    Under `gaussian_kernel` only the score along exp(-r^2/2) carries forward, so a direction
    orthogonal to it is serially independent. Under `exponential_kernel` it is not.
    """
    grid = np.linspace(0, 1, 120)
    g0 = np.exp(-grid ** 2 / 2)
    probe = np.sin(k * np.pi * grid)
    probe = probe - np.trapezoid(probe * g0, grid) / np.trapezoid(g0 ** 2, grid) * g0

    scores = {}
    for name, kernel in [("gaussian", far_dgp.gaussian_kernel),
                         ("exponential", far_dgp.exponential_kernel)]:
        X = far_dgp.sample_far1(4000, grid, kernel(grid, rho=0.8), rng=3)
        scores[name] = abs(far_dgp.lag1_autocorr(np.trapezoid(X * probe, grid, axis=1)))

    assert scores["gaussian"] < 0.02
    assert scores["exponential"] > 0.05
    assert scores["exponential"] > 4 * scores["gaussian"]


def test_sim_datasets_accept_an_alternative_kernel():
    grid = np.linspace(0, 1, 120)
    d = far_dgp.sim_dataset_window(n=40, grid=grid, kernel=far_dgp.exponential_kernel, rng=0)
    assert far_dgp.kernel_rank(d["Psi"], grid) == len(grid)
    X, y, g, beta = far_dgp.sim_dataset_psi(
        n=30, grid=grid, kernel=far_dgp.exponential_kernel, rng=0
    )
    assert X.shape == (30, len(grid)) and y.shape == (30,)
