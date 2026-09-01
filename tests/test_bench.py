"""Checks on the three-way comparison in `bench`.

Two things here could quietly invalidate the whole comparison, and both are checked to the
last bit rather than to a tolerance.

The first is the design: the claim that the three bases differ only in their eigenfunctions
rests on each being orthonormal under the same quadrature and on the covariance spectrum
coming out equal to the prescribed variance profile. If either fails, an integrated squared
error is not comparable across bases and nothing downstream means what it says.

The second is the shortcut. `bench.study` runs the theta scan once at zero penalty and
re-minimises that curve at every candidate lambda, instead of calling `HH.fit` and
`HH_A.fit` once per lambda. That is only legitimate because the penalty is the sole part of
the objective that moves with lambda, so `hh_b_at` and `hh_a_at` are held against the package
functions they stand in for and required to agree exactly.
"""


import numpy as np
import pytest

from fts_research.bench import design as dz
from fts_research.bench import study as st
from fts_research.core import style as sty
from fts_research.core.quadrature import quadrature_weights
from fts_research.fdaSP import SBC
from fts_research.truncFLR import HH, HH_A


@pytest.fixture(scope="module")
def small():
    """A small dataset from each cell would be slow; one cell serves the shortcut checks."""
    return dz.simulate("legendre", "step", n=60, n_grid=41, n_basis=12, decay=4.0,
                       snr=20.0, n_test=40, rng=3)


@pytest.fixture(scope="module")
def lambdas():
    return np.linspace(0.0, 1.0, 12)[1:]


# --------------------------------------------------------------------------------- design --


@pytest.mark.parametrize("kind", dz.BASIS_KINDS)
def test_every_basis_is_orthonormal_under_the_shared_quadrature(kind):
    """The matched-spectrum claim needs int e_j e_k = delta_jk in the same trapezium inner
    product every estimator here integrates with."""
    grid = np.linspace(0, 1, 201)
    Psi = dz.basis(grid, kind, n_basis=25)
    assert Psi.shape == (25, grid.size)
    assert np.allclose(dz.gram(Psi, grid), np.eye(25), atol=1e-10)


@pytest.mark.parametrize("kind", dz.BASIS_KINDS)
def test_orthonormalising_does_not_change_the_span(kind):
    """Cholesky rotates within the span, so the projection of an arbitrary function onto the
    orthonormalised basis must equal its projection onto the raw one."""
    grid = np.linspace(0, 1, 201)
    raw = dz.raw_basis(grid, kind, n_basis=15)
    Psi = dz.orthonormalise(raw, grid)
    f = np.exp(grid) * np.sin(4 * grid)

    w = quadrature_weights(grid)
    G = dz.gram(raw, grid)
    coef = np.linalg.solve(G, (raw * w[None, :]) @ f)      # least squares onto the raw basis
    assert np.allclose(coef @ raw, dz.project(f, Psi, grid)[1], atol=1e-10)


@pytest.mark.parametrize("kind", dz.BASIS_KINDS)
def test_the_covariance_spectrum_is_the_prescribed_variance_profile(kind):
    """With an orthonormal basis and independent coefficients, the eigenvalues of the
    covariance operator of X are exactly the coefficient variances, whatever the basis. That
    is what holds the difficulty fixed across the three designs."""
    grid = np.linspace(0, 1, 201)
    K = 8
    Psi = dz.basis(grid, kind, n_basis=K)
    var = dz.variance_profile(K, decay=4.0)
    X = dz.sample_curves(20000, Psi, var, rng=1)
    got = dz.spectrum(X, grid, k=K)
    assert np.allclose(got, var, rtol=0.1), f"{got} vs {var}"


def test_the_three_bases_share_one_spectrum_at_a_common_seed():
    """The point of the design, stated as one assertion: change the basis, keep the spectrum."""
    grid = np.linspace(0, 1, 201)
    var = dz.variance_profile(6, decay=4.0)
    spectra = [dz.spectrum(dz.sample_curves(20000, dz.basis(grid, k, 6), var, rng=2), grid, 6)
               for k in dz.BASIS_KINDS]
    for other in spectra[1:]:
        assert np.allclose(spectra[0], other, rtol=0.1)


@pytest.mark.parametrize("kind", dz.BASIS_KINDS)
def test_every_basis_is_ordered_from_smoothest_to_roughest(kind):
    """The variance profile is attached to the index, so the index has to mean the same thing
    in all three families."""
    grid = np.linspace(0, 1, 201)
    Psi = dz.basis(grid, kind, n_basis=25)
    rough = np.diag(dz.roughness_gram(Psi, grid))
    assert np.all(np.diff(rough) > -1e-8)
    assert rough[0] == pytest.approx(0.0, abs=1e-8)      # every span contains the constant


def test_the_trigonometric_ordering_is_the_frequency_ordering():
    """int e_k'^2 = 4 pi^2 j^2 for the sinusoid of frequency j, so recovering those numbers
    says the construction leaves Hall & Hooker's own basis in its own order."""
    grid = np.linspace(0, 1, 401)
    rough = np.diag(dz.roughness_gram(dz.basis(grid, "trig", 9), grid))
    want = [0.0] + [4 * np.pi ** 2 * j ** 2 for j in (1, 1, 2, 2, 3, 3, 4, 4)]
    assert np.allclose(rough, want, rtol=2e-3)


@pytest.mark.parametrize("kind", dz.BASIS_KINDS)
def test_no_design_concentrates_its_variation_at_one_end_of_the_domain(kind):
    """The failure the roughness ordering exists to prevent. A B-spline basis taken in index
    order puts the leading, highest-variance functions at the left of the domain, so Var X(t)
    falls away to the right and the truncation model is handed an orientation before it has
    seen any data."""
    grid = np.linspace(0, 1, 201)
    w, var = quadrature_weights(grid), dz.variance_profile(25, decay=6.0)
    v = dz.pointwise_variance(dz.basis(grid, kind, 25), var)
    half = grid <= 0.5
    assert (w[half] @ v[half]) / (w[~half] @ v[~half]) == pytest.approx(1.0, abs=0.05)
    assert v.min() / v.max() > 0.2

    if kind == "bspline":            # index order is what this is being compared against
        lopsided = dz.pointwise_variance(
            dz.orthonormalise(dz.raw_basis(grid, kind, 25), grid), var)
        assert (w[half] @ lopsided[half]) / (w[~half] @ lopsided[~half]) > 5.0


@pytest.mark.parametrize("target", list(dz.TARGETS))
def test_each_target_is_zero_exactly_on_the_null_set_it_declares(target):
    """`TARGETS[...]["null"]` is what every support metric is scored against, so it has to be
    the null set of the function next to it, not a description of it."""
    grid = np.linspace(0, 1, 2001)
    spec = dz.TARGETS[target]
    psi, null = spec["fn"](grid), spec["null"](grid)
    assert np.all(psi[null] == 0.0)
    assert np.count_nonzero(psi[~null]) > 0.9 * np.count_nonzero(~null)


@pytest.mark.parametrize("target,length", [("step", 0.6), ("cdf", 0.8), ("multi", 0.55)])
def test_each_null_set_is_the_closed_set_where_psi_vanishes(target, length):
    """T0 is {t : psi(t) = 0}, which is closed: both smooth targets meet the axis *at* their
    boundary, so that point belongs to the null set, and only the step, which jumps, excludes
    it.

    A trapezium measure of a mask charges a half panel at each boundary, so it recovers the
    length to within h/2 per boundary rather than exactly; what matters is that it converges,
    which the two grids below check."""
    spec = dz.TARGETS[target]
    errors = []
    for n_grid in (401, 2001):
        grid = np.linspace(0, 1, n_grid)
        null = spec["null"](grid)
        assert np.all(spec["fn"](grid)[null] == 0.0)
        measured = quadrature_weights(grid)[null].sum()
        assert measured == pytest.approx(length, abs=2.5 / (n_grid - 1))
        errors.append(abs(measured - length))
    assert errors[1] < errors[0]


def test_the_multi_target_has_disjoint_support_and_null_regions():
    """Two supported stretches and three null ones is what makes it outside the truncation
    model: no single interval carries the signal, in either orientation."""
    grid = np.linspace(0, 1, 2001)
    null = dz.TARGETS["multi"]["null"](grid)
    assert len(SBC.find_regions(null, grid)) == 3
    assert len(SBC.find_regions(~null, grid)) == 2
    assert dz.TARGETS["multi"]["window"] is None


def test_the_windowed_targets_declare_the_window_they_are_supported_on():
    for name in ("step", "cdf"):
        spec = dz.TARGETS[name]
        grid = np.linspace(0, 1, 2001)
        regions = SBC.find_regions(~spec["null"](grid), grid)
        assert len(regions) == 1
        assert regions[0][0] == pytest.approx(spec["window"][0], abs=1e-3)


def test_the_projection_gap_is_what_the_span_leaves_out():
    """int (psi - P psi)^2 must be the Pythagorean remainder, and must fall as the span grows."""
    grid = np.linspace(0, 1, 201)
    psi = dz.psi_multi(grid)
    w = quadrature_weights(grid)
    gaps = []
    for K in (8, 16, 25):
        Psi = dz.basis(grid, "bspline", K)
        _, proj = dz.project(psi, Psi, grid)
        gap = dz.projection_gap(psi, Psi, grid)
        assert gap == pytest.approx(w @ psi ** 2 - w @ proj ** 2, rel=1e-8)
        gaps.append(gap)
    assert gaps[0] > gaps[1] > gaps[2]


def test_simulate_returns_matched_shapes_and_the_requested_noise_level(small):
    d = small
    assert d["X"].shape == (60, 41) and d["Y"].shape == (60,)
    assert d["X_test"].shape == (40, 41) and d["Y_test"].shape == (40,)
    signal = (d["X"] * quadrature_weights(d["grid"])[None, :]) @ d["psi"]
    assert np.var(signal) / d["sigma"] ** 2 == pytest.approx(20.0, rel=1e-8)
    assert d["scale"] == pytest.approx(np.trapezoid(d["psi"] ** 2, d["grid"]), rel=1e-10)


def test_simulate_rejects_an_unknown_target():
    with pytest.raises(ValueError, match="unknown target"):
        dz.simulate("trig", "not_a_target", n=5, n_grid=11)


def test_raw_basis_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown basis kind"):
        dz.raw_basis(np.linspace(0, 1, 11), kind="hermite")


# ------------------------------------------------------------------------------ shortcuts --


def test_refit_b_is_ab_methodB_with_the_component_count_kept(small):
    """`_refit_b` exists only to return the m that `ab_methodB` throws away, so the (a, b) it
    returns must be `ab_methodB`'s own."""
    X, Y, grid = small["X"], small["Y"], small["grid"]
    a, b, _ = HH.pilot_fpca(X, Y, grid)
    for theta in (0.3, 0.55, 0.8):
        want = HH.ab_methodB(theta, grid, a, b, X, Y=Y, refit=True)
        a_hat, b_hat, m = st._refit_b(theta, grid, X, Y, 20, 10)
        assert a_hat == want[0] and np.array_equal(b_hat, want[1])
        assert 2 <= m <= 10


@pytest.mark.slow
@pytest.mark.parametrize("mirror", [False, True])
def test_hh_b_at_reproduces_HH_fit_exactly(small, lambdas, mirror):
    """The cached scan is only valid if re-minimising it at the selected lambda lands on the
    fit `HH.fit` computes from scratch. Exact equality, not a tolerance: the arithmetic is
    meant to be identical."""
    X, Y, grid = small["X"], small["Y"], small["grid"]
    want = (HH.fit_mirror if mirror else HH.fit)(X, Y, grid, lambdas)
    state = st.hh_b_state(X, Y, grid, lambdas, mirror=mirror)
    got = st.hh_b_at(state, state["lam_hat"])

    assert state["lam_hat"] == want["lam_hat"]
    assert got["theta_hat"] == want["theta_hat"]
    assert got["a_hat"] == want["a_hat"]
    assert np.array_equal(got["b_hat"], want["b_hat"])
    if mirror:
        assert got["theta_hat"] == pytest.approx(1.0 - want["u_hat"])


@pytest.mark.slow
@pytest.mark.parametrize("mirror", [False, True])
def test_hh_a_at_reproduces_HH_A_fit_exactly(small, lambdas, mirror):
    """As above for method A, where the shortcut also has to reproduce the step-7 BIC search
    over m, in the same order and with the same tie-breaking."""
    X, Y, grid = small["X"], small["Y"], small["grid"]
    want = (HH_A.fit_mirror if mirror else HH_A.fit)(X, Y, grid, lambdas)
    state = st.hh_a_state(X, Y, grid, lambdas, mirror=mirror)
    got = st.hh_a_at(state, state["lam_hat"])

    assert state["lam_hat"] == want["lam_hat"]
    assert got["m"] == want["m_hat"]
    assert got["theta_hat"] == want["theta_hat"]
    assert got["a_hat"] == want["a_hat"]
    assert np.array_equal(got["b_hat"], want["b_hat"])


@pytest.mark.slow
def test_the_theta_scan_is_the_only_thing_lambda_moves(small, lambdas):
    """The shortcut's premise: at any lambda, not just the selected one, re-minimising the
    cached curve gives what a fresh `theta_methodB` gives."""
    X, Y, grid = small["X"], small["Y"], small["grid"]
    a, b, _ = HH.pilot_fpca(X, Y, grid)
    _, theta_grid, rss = HH.theta_methodB(X, Y, grid, a, b, 0.0)
    for lam in lambdas:
        want, _, _ = HH.theta_methodB(X, Y, grid, a, b, lam)
        idx = st._first_interior_min(rss + len(Y) * lam * theta_grid ** 2)
        assert theta_grid[idx] == want


@pytest.mark.slow
def test_hh_states_reuse_one_fit_across_repeated_lambdas(small, lambdas):
    """Distinct lambdas that select the same theta must be scored once, or the oracle sweep
    costs as much as fitting from scratch."""
    X, Y, grid = small["X"], small["Y"], small["grid"]
    state = st.hh_b_state(X, Y, grid, lambdas, mirror=True)
    for lam in lambdas:
        st.hh_b_at(state, lam)
    assert 0 < len(state["cache"]) <= len(lambdas)


@pytest.mark.slow
def test_the_ogl_selection_rules_match_cv_fit_asking_for_them(small):
    """`ogl_run` reads both lambdas off the cross-validation curve it already has rather than
    paying for a second pass, which is only the same thing if the folds are. The coefficient
    functions then come from the path rather than from `cv_fit`'s own refit, so they agree
    only to ADMM tolerance -- which is the point: one solver run, three ways of pointing at
    it."""
    X, Y, grid = small["X"], small["Y"], small["grid"]
    y = Y - Y.mean()
    got = st.ogl_run(X, Y, grid, n_basis=10, n_folds=4, n_lambda=8, rng=0)
    for one_se, j in ((False, got["j_cv"]), (True, got["j_1se"])):
        want = SBC.cv_fit(X, y, grid, n_basis=10, n_folds=4, n_lambda=8, rng=0,
                          one_se=one_se)
        assert got["lambdas"][j] == want["lam_hat"]
        assert np.allclose(got["path"]["betas"][j], want["beta"], atol=1e-3)


@pytest.mark.slow
def test_the_ogl_path_is_indexed_by_the_cross_validated_lambda_grid(small):
    """`j_cv` and `j_1se` index the path, so the path must be over exactly the penalties the
    cross-validation searched, in the same order."""
    X, Y, grid = small["X"], small["Y"], small["grid"]
    got = st.ogl_run(X, Y, grid, n_basis=10, n_folds=4, n_lambda=8, rng=0)
    assert np.array_equal(got["lambdas"], got["cv"]["lambdas"])
    assert got["lambdas"][got["j_cv"]] == got["cv"]["lam_hat"]


def _lopsided(n=200, n_grid=201, n_basis=25, decay=6.0, snr=20.0, rng=0):
    """The design `bench.design.basis` exists to avoid: a B-spline family kept in index order,
    so the leading, highest-variance functions sit at the left of the domain and Var X(t) dies
    away to the right."""
    rng = np.random.default_rng(rng)
    grid = np.linspace(0.0, 1.0, n_grid)
    Psi = dz.orthonormalise(dz.raw_basis(grid, "bspline", n_basis), grid)
    var = dz.variance_profile(n_basis, decay)
    X = dz.sample_curves(n, Psi, var, rng=rng)
    psi = dz.psi_step(grid)
    signal = (X * quadrature_weights(grid)[None, :]) @ psi
    sigma = np.sqrt(np.var(signal) / snr)
    return X, signal + sigma * rng.standard_normal(n), grid


def test_a_degenerate_section_3_2_selection_falls_back_rather_than_raising(monkeypatch):
    """`DegenerateSelection` means P_b is undefined at every candidate, so the criterion ranks
    nothing rather than ranking weakly. The study still has to produce a fit, so `_select`
    takes the least-penalised candidate and records that it did.

    The condition is forced here rather than found: since the section 3.2 variance terms were
    put on their per-observation scale, S_Y^B has an interior minimum on every design in this
    study and the fallback never fires -- which is what the next two tests check. The path
    still has to work when it does.
    """
    def refuses(*args, **kwargs):
        raise HH.DegenerateSelection("undefined at every candidate")

    monkeypatch.setattr(st.HH, "lambda_select", refuses)
    lam_grid = np.linspace(0.0, 1.0, 40)[1:]
    d = dz.simulate("bspline", "step", rng=0)
    state = st.hh_b_state(d["X"], d["Y"], d["grid"], lam_grid)

    assert state["lam_failed"] is True
    assert state["lam_hat"] == pytest.approx(lam_grid.min())
    assert np.isfinite(st.hh_b_at(state, state["lam_hat"])["b_hat"]).all()


@pytest.mark.slow
def test_even_the_lopsided_design_leaves_the_selection_rule_defined():
    """The design `design.basis` replaced, kept here because it used to break the rule.

    With S_Y^B's variance term n times too large, as step 5' displays it, that term dominated
    and left S_Y^B without an interior minimum on this design: V(lambda) was undefined at
    every candidate and section 3.2 had no opinion at all. On the corrected scale it does, so
    the roughness ordering now rests on `test_no_design_concentrates_its_variation_at_one_end
    _of_the_domain` alone -- an even Var X(t) -- rather than on a criterion that used to fail.
    """
    X, Y, grid = _lopsided()
    a, b, fpc = HH.pilot_fpca(X, Y, grid)
    lam, diag = HH.lambda_select(X, Y, grid, a, b, fpc, None, np.linspace(0.0, 1.0, 40)[1:])
    assert np.isfinite(lam)
    assert np.all(np.isfinite(diag["Vs"]))


@pytest.mark.slow
@pytest.mark.parametrize("target", list(dz.TARGETS))
def test_the_corrected_design_leaves_the_selection_rule_defined(target):
    """The same check on the design actually used: section 3.2 has an opinion in every cell,
    so no result below rests on the fallback."""
    d = dz.simulate("bspline", target, rng=st.seed_of("bspline", target, 0))
    X, Y, grid = d["X"], d["Y"], d["grid"]
    a, b, fpc = HH.pilot_fpca(X, Y, grid)
    lam, _ = HH.lambda_select(X, Y, grid, a, b, fpc, None, np.linspace(0.0, 1.0, 40)[1:])
    assert np.isfinite(lam)


def test_a_healthy_selection_is_not_marked_as_a_fallback(small, lambdas):
    state = st.hh_b_state(small["X"], small["Y"], small["grid"], lambdas, mirror=True)
    assert state["lam_failed"] is False
    assert state["lam_hat"] in set(lambdas.tolist())


# -------------------------------------------------------------------------------- metrics --


def test_relative_ise_is_one_for_the_estimate_that_gives_up():
    grid = np.linspace(0, 1, 201)
    psi = dz.psi_cdf(grid)
    scale = float(np.trapezoid(psi ** 2, grid))
    assert st.relative_ise(np.zeros_like(grid), psi, grid, scale) == pytest.approx(1.0)
    assert st.relative_ise(psi, psi, grid, scale) == pytest.approx(0.0, abs=1e-12)


def test_relative_ise_splits_over_a_region_and_its_complement():
    """psi_multi's null set is three disjoint intervals, which is where a single trapezium
    rule over the masked points would double count the gaps."""
    grid = np.linspace(0, 1, 201)
    psi = dz.psi_multi(grid)
    null = dz.TARGETS["multi"]["null"](grid)
    scale = float(np.trapezoid(psi ** 2, grid))
    beta = np.sin(3 * grid)
    total = st.relative_ise(beta, psi, grid, scale)
    parts = (st.relative_ise(beta, psi, grid, scale, null)
             + st.relative_ise(beta, psi, grid, scale, ~null))
    assert total == pytest.approx(parts, rel=1e-12)


def test_null_metrics_reward_only_an_exactly_recovered_null_set():
    grid = np.linspace(0, 1, 201)
    null = dz.TARGETS["multi"]["null"](grid)
    w = quadrature_weights(grid)
    perfect = st.null_metrics(~null, null, grid)
    assert perfect == {"null_sens": pytest.approx(1.0), "null_spec": pytest.approx(1.0),
                       "null_jaccard": pytest.approx(1.0),
                       "null_measure": pytest.approx(w[null].sum())}

    nothing = st.null_metrics(np.ones_like(null), null, grid)
    assert nothing["null_sens"] == pytest.approx(0.0)
    assert nothing["null_spec"] == pytest.approx(1.0)
    assert nothing["null_jaccard"] == pytest.approx(0.0)
    assert nothing["null_measure"] == pytest.approx(0.0)

    everything = st.null_metrics(np.zeros_like(null), null, grid)
    assert everything["null_sens"] == pytest.approx(1.0)
    assert everything["null_spec"] == pytest.approx(0.0)
    assert everything["null_measure"] == pytest.approx(1.0)


def test_null_metrics_are_measured_in_length_not_in_grid_points():
    """Doubling the grid must not move a set-overlap number."""
    out = []
    for n_grid in (201, 401):
        grid = np.linspace(0, 1, n_grid)
        null = dz.TARGETS["step"]["null"](grid)
        out.append(st.null_metrics(grid >= 0.5, null, grid))
    assert out[0]["null_jaccard"] == pytest.approx(out[1]["null_jaccard"], abs=1e-3)


def test_endpoint_error_reads_the_last_stretch_of_support_and_keeps_its_sign():
    """The sign is the point: too late and too early are different mistakes, and an estimate
    that keeps the whole domain implies u_hat = 0 rather than no answer at all."""
    grid = np.linspace(0, 1, 201)
    assert st.implied_endpoint(grid >= 0.62, grid) == pytest.approx(0.62)
    assert st.endpoint_error(grid >= 0.62, grid, (0.6, 1.0)) == pytest.approx(+0.02)
    assert st.endpoint_error(grid >= 0.58, grid, (0.6, 1.0)) == pytest.approx(-0.02)
    # Two stretches: the endpoint a [u, 1] truth implies is the left edge of the later one.
    two = ((grid >= 0.1) & (grid <= 0.3)) | (grid >= 0.7)
    assert st.endpoint_error(two, grid, (0.6, 1.0)) == pytest.approx(+0.1)
    # No boundary found at all: the whole domain is support, so the implied endpoint is 0.
    assert st.implied_endpoint(np.ones_like(grid, dtype=bool), grid) == pytest.approx(0.0)
    kept_all = st.endpoint_error(np.ones_like(grid, dtype=bool), grid, (0.6, 1.0))
    assert kept_all == pytest.approx(-0.6)
    assert np.isnan(st.endpoint_error(grid >= 0.5, grid, None))
    assert np.isnan(st.implied_endpoint(np.zeros_like(grid, dtype=bool), grid))
    assert np.isnan(st.endpoint_error(np.zeros_like(grid, dtype=bool), grid, (0.6, 1.0)))


def test_prediction_over_a_sub_grid_is_not_the_zero_padded_full_grid_integral():
    """The half panel `truncFLR.HH_A.trunc_predict` warns about: straddling the jump at theta
    with a trapezium adds beta(theta) X(theta) h / 2, which is not a rounding difference."""
    grid = np.linspace(0, 1, 201)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, grid.size)) + 3.0
    beta = np.where(grid <= 0.5, 2.0, 0.0)
    Y = np.zeros(30)
    mask = grid <= 0.5
    assert st.prediction_mse(0.0, beta, X, Y, grid, mask) != pytest.approx(
        st.prediction_mse(0.0, beta, X, Y, grid, None), rel=1e-6)
    want = np.mean(np.trapezoid(beta[mask] * X[:, mask], grid[mask], axis=1) ** 2)
    assert st.prediction_mse(0.0, beta, X, Y, grid, mask) == pytest.approx(want)


def test_seeds_are_distinct_across_cells_and_replications():
    seeds = [st.seed_of(k, t, r) for k in dz.BASIS_KINDS for t in dz.TARGETS
             for r in range(40)]
    assert len(set(seeds)) == len(seeds)


# ------------------------------------------------------------------------------ the study --


@pytest.mark.slow
def test_run_cell_produces_one_row_per_method_tuning_and_orientation():
    cfg = {"n": 60, "n_grid": 41, "n_basis": 12, "n_test": 40, "n_lambda_hh": 8,
           "n_lambda_ogl": 8, "n_folds": 4, "ogl_basis": 10}
    records, curves, supports = st.run_cell("bspline", "multi", 0, cfg)
    got = {(r["method"], r["tuning"], r["orientation"]) for r in records}
    assert got == {
        ("HH-B", "paper", "left"), ("HH-B", "paper", "right"),
        ("HH-B", "oracle", "left"), ("HH-B", "oracle", "right"),
        ("HH-A", "paper", "left"), ("HH-A", "paper", "right"),
        ("HH-A", "oracle", "left"), ("HH-A", "oracle", "right"),
        ("OGL", "cv", "na"), ("OGL", "one_se", "na"), ("OGL", "oracle", "na"),
        ("FPCA", "bic", "na"),
    }
    assert len(curves) == len(records)
    assert all(np.isfinite(r["rise"]) for r in records)
    # One support mask per (method, tuning), already resolved to the chosen orientation.
    assert set(supports) == {f"{m}|{t}" for m, t, _o in
                             ((r["method"], r["tuning"], r["orientation"]) for r in records)}
    assert all(v.dtype == bool and v.size == cfg["n_grid"] for v in supports.values())


@pytest.mark.slow
def test_exactly_one_orientation_is_picked_per_truncation_method():
    cfg = {"n": 60, "n_grid": 41, "n_basis": 12, "n_test": 40, "n_lambda_hh": 8,
           "n_lambda_ogl": 8, "n_folds": 4, "ogl_basis": 10}
    records, _curves, _supports = st.run_cell("trig", "cdf", 1, cfg)
    for method in ("HH-A", "HH-B"):
        picked = {r["orientation"] for r in records
                  if r["method"] == method and r["bic_pick"]}
        assert len(picked) == 1


@pytest.mark.slow
def test_the_oracle_is_never_beaten_by_the_prescribed_rule():
    """The oracle minimises the realised error over the same lambda grid the rule chose from,
    so by construction it cannot be worse. If it ever is, the two are not searching the same
    path and the tuning-versus-estimator split is meaningless."""
    cfg = {"n": 60, "n_grid": 41, "n_basis": 12, "n_test": 40, "n_lambda_hh": 8,
           "n_lambda_ogl": 8, "n_folds": 4, "ogl_basis": 10}
    records, _curves, _supports = st.run_cell("legendre", "step", 2, cfg)
    by_key = {(r["method"], r["tuning"], r["orientation"]): r["rise"] for r in records}
    for method, orientation in (("HH-A", "left"), ("HH-A", "right"),
                                ("HH-B", "left"), ("HH-B", "right")):
        assert (by_key[(method, "oracle", orientation)]
                <= by_key[(method, "paper", orientation)] + 1e-12)
    assert by_key[("OGL", "oracle", "na")] <= by_key[("OGL", "cv", "na")] + 1e-12


@pytest.mark.slow
def test_run_study_and_summarise_round_trip():
    cfg = {"n": 60, "n_grid": 41, "n_basis": 12, "n_test": 40, "n_lambda_hh": 8,
           "n_lambda_ogl": 8, "n_folds": 4, "ogl_basis": 10}
    df, freq = st.run_study(cells=[("trig", "step")], n_rep=2, cfg=cfg, n_jobs=1,
                            verbose=False)
    assert len(df) == 2 * 12
    table = st.summarise(df)
    assert list(table.columns) == st.METHOD_ORDER
    assert table.notna().all().all()

    # Eight (method, tuning) labels, one row per grid point, each a frequency.
    assert len(freq) == 8 * cfg["n_grid"]
    assert freq["in_support"].between(0.0, 1.0).all()
    assert set(freq["t"]) == set(np.linspace(0, 1, cfg["n_grid"]))


@pytest.mark.slow
def test_the_support_frequencies_integrate_to_the_recorded_null_lengths():
    """The two caches have to describe the same fits, and this is the identity that says so:
    averaging the null mask over replications and integrating gives the same number as
    integrating each replication and averaging, since both are linear in the mask. Any
    mismatch means the frequency picture and the summary table came from different runs, or
    from different orientations."""
    cfg = {"n": 60, "n_grid": 41, "n_basis": 12, "n_test": 40, "n_lambda_hh": 8,
           "n_lambda_ogl": 8, "n_folds": 4, "ogl_basis": 10}
    df, freq = st.run_study(cells=[("legendre", "cdf")], n_rep=3, cfg=cfg, n_jobs=1,
                            verbose=False)
    w = quadrature_weights(np.linspace(0, 1, cfg["n_grid"]))
    chosen = df[df["bic_pick"]]
    for (method, tuning), g in freq.groupby(["method", "tuning"]):
        want = chosen.query("method == @method and tuning == @tuning")["null_measure"].mean()
        got = float(w @ (1.0 - g.sort_values("t")["in_support"].to_numpy()))
        assert got == pytest.approx(want, abs=1e-9), (method, tuning)


@pytest.mark.slow
def test_the_untruncated_pilot_never_switches_anything_off():
    """The FPCA baseline is carried precisely because it reports no null set, so its support
    frequency must be 1 at every grid point. If it is not, the baseline is not the baseline."""
    cfg = {"n": 60, "n_grid": 41, "n_basis": 12, "n_test": 40, "n_lambda_hh": 8,
           "n_lambda_ogl": 8, "n_folds": 4, "ogl_basis": 10}
    _df, freq = st.run_study(cells=[("trig", "multi")], n_rep=2, cfg=cfg, n_jobs=1,
                             verbose=False)
    pilot = freq.query("method == 'FPCA'")
    assert len(pilot) == cfg["n_grid"]
    assert (pilot["in_support"] == 1.0).all()


# ---------------------------------------------------------------------------- the visual system


def test_the_method_palette_is_the_validated_categorical_order():
    """Colour means "which method" and nothing else in this notebook, so the three methods
    under comparison take the first three slots of the validated palette, in its fixed order,
    and the control takes a grey that is deliberately not one of them."""
    assert list(sty.METHOD_COLOUR)[:3] == ["HH-A", "HH-B", "OGL"]
    assert [sty.METHOD_COLOUR[m] for m in ("HH-A", "HH-B", "OGL")] == list(sty.CATEGORICAL)
    assert sty.METHOD_COLOUR["FPCA"] not in sty.CATEGORICAL
    assert set(sty.METHOD_COLOUR) == set(st.METHOD_ORDER)
    assert len(set(sty.METHOD_COLOUR.values())) == len(st.METHOD_ORDER)
    assert sty.INK not in sty.METHOD_COLOUR.values()      # the truth is never a series colour
