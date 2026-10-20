"""Tests for `truncFLR.HH_continuous`, the continuity-constrained two-region estimator.

Unlike method A/B, there is no zero tail to check for, so the pieces worth pinning down are:
the continuity constraint itself (both sides must agree at theta exactly, checked from the
fitted coefficients directly rather than by reading a single array entry), that
`continuous_predict`'s single whole-grid integral agrees with integrating each side separately
(the identity that lets it skip `HH_A.trunc_predict`'s split-quadrature correction), and that
the estimator recovers a truncation point on data with a genuine null region (where it should
behave like the paper's model) as well as on data with none at all.
"""

import numpy as np

from fts_research.truncFLR import HH_continuous, HH_dgp


def test_fit_components_agree_at_theta(hh_data):
    """b1(theta) = c1 @ phi1[:, -1] and b2(theta) = c2 @ phi2[:, 0] must be equal to machine
    precision -- the constraint `fit_components` solves for, not an approximation to it."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    theta = grid[np.argmin(np.abs(grid - 0.55))]

    a_hat, c1, phi1, c2, phi2 = HH_continuous.fit_components(X, Y, grid, theta, m1=5, m2=5)

    b1_theta = c1 @ phi1[:, -1]
    b2_theta = c2 @ phi2[:, 0]
    np.testing.assert_allclose(b1_theta, b2_theta, atol=1e-8)


def test_fit_at_theta_assembles_the_shared_value_once(hh_data):
    """b_hat at the theta grid point must equal the (shared) value from fit_components, and
    the array must have no duplicate or missing entry there."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    theta = grid[np.argmin(np.abs(grid - 0.55))]
    idx = np.argmin(np.abs(grid - theta))

    a_hat, c1, phi1, c2, phi2 = HH_continuous.fit_components(X, Y, grid, theta, m1=5, m2=5)
    _, b_hat = HH_continuous.fit_at_theta(X, Y, grid, theta, m1=5, m2=5)

    np.testing.assert_allclose(b_hat[idx], c1 @ phi1[:, -1], atol=1e-8)


def test_continuous_predict_matches_split_quadrature(hh_data):
    """Integrating b_hat over the whole grid must equal integrating each side and summing --
    the exact trapezium-rule identity that lets `continuous_predict` skip
    `HH_A.trunc_predict`'s split-quadrature correction for the jump (there is none here)."""
    grid, X, Y = hh_data["grid"], hh_data["X"], hh_data["Y"]
    theta = grid[len(grid) // 3]
    a_hat, b_hat = HH_continuous.fit_at_theta(X, Y, grid, theta, m1=4, m2=4)

    whole = HH_continuous.continuous_predict(a_hat, b_hat, X, grid)

    lo, hi = grid <= theta, grid >= theta
    split = a_hat \
        + np.trapezoid(b_hat[lo][None, :] * X[:, lo], grid[lo], axis=1) \
        + np.trapezoid(b_hat[hi][None, :] * X[:, hi], grid[hi], axis=1)

    np.testing.assert_allclose(whole, split, rtol=1e-10)


def test_recovers_theta_on_data_with_a_true_null_region(hh_data_big):
    """On Hall & Hooker's own indicator design (null on [0, 0.8)) the continuous model should
    still land its cross-validated theta_hat near the true endpoint, since b2 == 0 is inside
    its search space even though it is never assumed."""
    grid, X, Y, u = hh_data_big["grid"], hh_data_big["X"], hh_data_big["Y"], hh_data_big["u"]

    res = HH_continuous.fit(X, Y, grid, m1=4, m2=4, n_folds=5, rng=0)
    assert abs(res["theta_hat"] - u) < 0.15


def test_recovers_a_kink_with_no_null_region_on_either_side():
    """A coefficient function that is a sine wave before theta and a ramp after it (the
    motivating example for this module) has no null region at all; the continuous estimator
    should still find a truncation point near the true kink, which the zero-tail HH/HH_A
    models cannot represent."""
    true_theta = 0.5

    def beta_fn(t):
        return np.where(t < true_theta, np.sin(6 * np.pi * t), 1.0 + 2.0 * (t - true_theta))

    data = HH_dgp.sim_dataset(n=300, n_grid=151, beta_fn=beta_fn, snr=50.0, rng=3)
    grid, X, Y = data["grid"], data["X"], data["Y"]

    res = HH_continuous.fit(X, Y, grid, m1=5, m2=5, n_folds=5, rng=0)
    assert abs(res["theta_hat"] - true_theta) < 0.15
