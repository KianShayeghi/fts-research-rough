"""Checks on the pieces methods A and B share, in `truncFLR._common`.

The step 6-7 quantities are the ones with substance here: `theta_lam_variance` reads S_Y in
its per-observation form, so that lambda sits on the same scale in the step-6 displays as in
equation (3.4), and `weighted_S_b` integrates the step-7 density on a grid fine enough to
resolve it. Both are pinned below, along with the identity of the shared definitions.
"""

import numpy as np
import pytest

from fts_research.truncFLR import HH, HH_A
from fts_research.truncFLR import _common as C


# --------------------------------------------------------------------------- sharing ----


@pytest.mark.parametrize("name", ["normal_weight", "uniform_weight", "fat_tailed_weight",
                                  "thin_tailed_weight", "pilot_fpca", "fit_b_simp",
                                  "trunc_eigvals", "WEIGHT_FUNCTIONS"])
def test_both_modules_expose_the_same_object(name):
    """The two modules re-export one definition, so neither can drift from the other."""
    assert getattr(HH, name) is getattr(C, name)
    assert getattr(HH_A, name) is getattr(C, name)


def test_method_a_also_shares_the_truncated_fpca():
    assert HH_A.trunc_fpca is C.trunc_fpca
    assert HH_A.trunc_eigvals is C.trunc_eigvals


# ------------------------------------------------------------------------- step 6 -------


def _parabola(theta_grid, curvature, centre=0.5):
    """S_Y with a known second derivative, so V can be checked against its closed form."""
    return 0.5 * curvature * (theta_grid - centre) ** 2


def test_theta_lam_variance_matches_its_formula():
    """V = (sigma^2/n) s'' / (s'' + lambda)^2 with s'' the per-observation curvature."""
    tg = np.linspace(0.0, 1.0, 501)
    n, sigma2, lam, curvature = 200, 0.4, 0.03, 600.0
    S_Y = _parabola(tg, curvature)
    idx = int(np.argmin(S_Y))

    s2 = curvature / n
    expected = (sigma2 / n) * s2 / (s2 + lam) ** 2
    got = C.theta_lam_variance(S_Y, tg, idx, lam, sigma2, n)
    assert got == pytest.approx(expected, rel=1e-6)


def test_theta_lam_variance_uses_the_per_observation_curvature():
    """Doubling n at fixed per-observation error must not change s'', only the 1/n factor.

    Taking S_Y as a sum instead would divide V by a further n, which is what drives lambda
    out of the denominator and collapses the step-7 density.
    """
    tg = np.linspace(0.0, 1.0, 501)
    sigma2, lam, curvature = 0.4, 0.03, 600.0
    small = C.theta_lam_variance(_parabola(tg, curvature), tg, 250, lam, sigma2, 100)
    large = C.theta_lam_variance(_parabola(tg, 2 * curvature), tg, 250, lam, sigma2, 200)
    assert large == pytest.approx(small / 2, rel=1e-9)


def test_theta_lam_variance_responds_to_lambda():
    """lambda has to move V over the range the lambda grids actually cover.

    The per-observation curvature is order 1 on these problems, so a lambda of order 1 belongs
    in the denominator alongside it. Reading S_Y as a sum puts s'' at order n instead, and
    lambda then changes V by parts in a million.
    """
    tg = np.linspace(0.0, 1.0, 501)
    S_Y = _parabola(tg, 100.0)                     # per-observation curvature 0.5 at n = 200
    V = [C.theta_lam_variance(S_Y, tg, 250, lam, 0.4, 200) for lam in (0.0, 0.5, 2.0)]
    assert V[0] > V[1] > V[2]
    assert V[1] == pytest.approx(0.25 * V[0], rel=1e-9)
    assert V[2] == pytest.approx(V[0] / 25.0, rel=1e-9)


def test_theta_lam_variance_is_nan_on_non_positive_curvature():
    tg = np.linspace(0.0, 1.0, 501)
    concave = -_parabola(tg, 600.0)
    assert np.isnan(C.theta_lam_variance(concave, tg, 250, 0.01, 0.4, 200))


def test_theta_lam_variance_evaluates_at_interior_points():
    """An argmin on either edge is pulled inside, so the second difference is defined."""
    tg = np.linspace(0.0, 1.0, 51)
    S_Y = _parabola(tg, 600.0)
    for idx in (0, len(tg) - 1):
        assert np.isfinite(C.theta_lam_variance(S_Y, tg, idx, 0.01, 0.4, 200))


# ------------------------------------------------------------------------- step 7 -------


def test_weighted_S_b_integrates_a_constant_to_itself():
    """The weight is a density, so a flat S_b comes back unchanged."""
    tg = np.linspace(0.0, 1.0, 301)
    S_b = np.full_like(tg, 2.5)
    got = C.weighted_S_b(S_b, tg, 0.5, 1e-4, C.normal_weight)
    assert got == pytest.approx(2.5, rel=1e-6)


def test_weighted_S_b_recovers_S_b_at_theta_lam_as_V_shrinks():
    tg = np.linspace(0.0, 1.0, 301)
    S_b = np.exp(3 * tg)
    for V in (1e-4, 1e-6, 1e-8):
        got = C.weighted_S_b(S_b, tg, 0.5, V, C.normal_weight)
        assert got == pytest.approx(np.exp(1.5), rel=10 * np.sqrt(V) + 1e-6)


def test_weighted_S_b_resolves_a_density_narrower_than_the_grid():
    """A weight with sd below the grid spacing must still average, not sample.

    Integrating on `theta_grid` itself leaves the density unresolved and returns roughly
    S_b(theta_lam) * h / sqrt(2 pi V), which grows without bound as V shrinks.
    """
    tg = np.linspace(0.0, 1.0, 101)          # spacing 0.01
    S_b = np.ones_like(tg)
    V = 0.0004 ** 2                          # sd 25 times finer than the grid
    on_grid = np.trapezoid(S_b * C.normal_weight(tg, 0.5, V), tg)
    resolved = C.weighted_S_b(S_b, tg, 0.5, V, C.normal_weight)
    assert resolved == pytest.approx(1.0, rel=1e-6)
    assert abs(on_grid - 1.0) > 1.0, "the unresolved integral is meant to be badly wrong here"


@pytest.mark.parametrize("name", sorted(C.WEIGHT_FUNCTIONS))
def test_weighted_S_b_is_a_weighted_average_for_every_weight(name):
    """Whatever the density, P_b of a monotone S_b lies between its extremes."""
    tg = np.linspace(0.0, 1.0, 301)
    S_b = np.exp(3 * tg)
    got = C.weighted_S_b(S_b, tg, 0.5, 4e-4, C.WEIGHT_FUNCTIONS[name])
    assert S_b.min() < got < S_b.max()


def test_weighted_S_b_stays_inside_the_theta_grid():
    """theta_lam near an endpoint truncates the range rather than extrapolating S_b."""
    tg = np.linspace(0.2, 0.8, 301)
    S_b = np.ones_like(tg)
    assert np.isfinite(C.weighted_S_b(S_b, tg, 0.2, 1e-3, C.normal_weight))
    assert np.isfinite(C.weighted_S_b(S_b, tg, 0.8, 1e-3, C.normal_weight))


# ------------------------------------------------------------------------- weights ------


@pytest.mark.parametrize("name", sorted(C.WEIGHT_FUNCTIONS))
def test_weight_functions_are_densities_with_the_requested_variance(name):
    """Every weight must integrate to 1 and carry the variance V it was matched to."""
    theta_lam, V = 0.5, 0.01
    t = np.linspace(theta_lam - 12 * np.sqrt(V), theta_lam + 12 * np.sqrt(V), 200001)
    dens = C.WEIGHT_FUNCTIONS[name](t, theta_lam, V)

    assert np.all(dens >= 0)
    np.testing.assert_allclose(np.trapezoid(dens, t), 1.0, rtol=1e-4)
    # `uniform` has jump discontinuities, so its quadrature error is O(h), not O(h^2).
    np.testing.assert_allclose(np.trapezoid(t * dens, t), theta_lam, atol=1e-4)
    np.testing.assert_allclose(np.trapezoid((t - theta_lam) ** 2 * dens, t), V, rtol=1e-3)


def test_thin_tailed_weight_recovers_the_normal_at_beta_two():
    t = np.linspace(0.2, 0.8, 5001)
    np.testing.assert_allclose(C.thin_tailed_weight(t, 0.5, 0.01, beta=2.0),
                               C.normal_weight(t, 0.5, 0.01), rtol=1e-9)
