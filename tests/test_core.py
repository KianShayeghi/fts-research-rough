"""Tests for `fts_research.core`, the numerics every package here is built on.

These are the definitions that would otherwise be reimplemented once per method, and the
comparisons in `bench` and the agreement with R in `tests/test_f2ssp_match.py` only mean
anything because there is one of each. So they are pinned against closed forms rather than
against each other: the quadrature against exact integrals of polynomials, the trigonometric
basis against the display in Hall & Hooker (2016, Section 4), the B-spline basis against the
partition-of-unity and local-support properties that characterise it, and the Gaussian process
against the coefficient variances it is defined by.
"""

import colorsys
import os

import numpy as np
import pytest

from fts_research.core import paths, style
from fts_research.core.bases import bspline_basis, trig_basis
from fts_research.core.gp import sample_gp
from fts_research.core.quadrature import quadrature_weights, trunc_int


# ------------------------------------------------------------------------ quadrature ----


def test_quadrature_weights_integrate_polynomials():
    grid = np.linspace(0.0, 1.0, 401)
    w = quadrature_weights(grid)
    assert w.sum() == pytest.approx(1.0)
    assert w @ grid == pytest.approx(0.5, abs=1e-9)
    assert w @ (grid ** 2) == pytest.approx(1 / 3, abs=1e-5)


def test_quadrature_weights_match_trapezoid():
    grid = np.linspace(0.0, 1.0, 137)
    f = np.sin(3 * grid) + grid ** 3
    assert quadrature_weights(grid) @ f == pytest.approx(np.trapezoid(f, grid))


def test_quadrature_weights_handle_an_uneven_grid():
    """The weight at a point is half the length of the two intervals adjoining it, not a
    constant h, so a grid that is not equally spaced integrates correctly."""
    grid = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    w = quadrature_weights(grid)
    np.testing.assert_allclose(w, [0.05, 0.25, 0.4, 0.25, 0.05])
    assert w.sum() == pytest.approx(1.0)


def test_trunc_int_matches_a_direct_integral():
    grid = np.linspace(0, 1, 201)
    X = np.vstack([np.ones_like(grid), grid])
    got = trunc_int(X, grid, u=0.25, v=0.75)
    np.testing.assert_allclose(got, [0.5, 0.25], atol=1e-6)


def test_trunc_int_weights_by_a_coefficient_function():
    grid = np.linspace(0, 1, 401)
    X = np.ones((1, grid.size))
    np.testing.assert_allclose(trunc_int(X, grid, beta=grid ** 2), [1 / 3], atol=1e-5)


# ----------------------------------------------------------------------------- bases ----


def test_trig_basis_matches_the_paper():
    grid = np.linspace(0, 1, 61)
    Psi = trig_basis(grid, n_basis=5)
    np.testing.assert_allclose(Psi[0], 1.0)
    np.testing.assert_allclose(Psi[1], np.sin(2 * np.pi * grid))
    np.testing.assert_allclose(Psi[2], np.cos(2 * np.pi * grid))
    np.testing.assert_allclose(Psi[3], np.sin(4 * np.pi * grid))
    np.testing.assert_allclose(Psi[4], np.cos(4 * np.pi * grid))


def test_trig_basis_is_orthogonal_under_the_shared_quadrature():
    """`bench.design` and `truncFLR.HH_dgp` both rely on this, and the trapezium rule is
    exact on these functions once the grid resolves them."""
    grid = np.linspace(0, 1, 501)
    Psi = trig_basis(grid, n_basis=7)
    G = (Psi * quadrature_weights(grid)[None, :]) @ Psi.T
    np.testing.assert_allclose(G, np.diag(np.diag(G)), atol=1e-10)


def test_bspline_basis_is_a_partition_of_unity():
    """B-splines with an open knot vector sum to 1 everywhere on the domain."""
    grid = np.linspace(0, 1, 137)
    Phi = bspline_basis(grid, n_basis=15, order=4)
    assert Phi.shape == (137, 15)
    np.testing.assert_allclose(Phi.sum(axis=1), 1.0, atol=1e-12)


def test_bspline_basis_is_non_negative_and_locally_supported():
    grid = np.linspace(0, 1, 200)
    Phi = bspline_basis(grid, n_basis=12, order=4)
    assert (Phi >= -1e-14).all()
    # A cubic B-spline touches at most order + 1 knot spans, so each column's support is a
    # short contiguous run -- never the whole domain (once there are interior knots).
    for j in range(Phi.shape[1]):
        nz = np.flatnonzero(Phi[:, j] > 1e-12)
        assert nz.size < grid.size
        assert np.all(np.diff(nz) == 1), "support of a basis function must be contiguous"


def test_bspline_basis_reproduces_polynomials():
    """Order-4 B-splines span cubics, so a cubic is representable exactly."""
    grid = np.linspace(0, 1, 101)
    Phi = bspline_basis(grid, n_basis=10, order=4)
    target = 1 - 2 * grid + 3 * grid ** 2 - 0.5 * grid ** 3
    coef, *_ = np.linalg.lstsq(Phi, target, rcond=None)
    np.testing.assert_allclose(Phi @ coef, target, atol=1e-10)


def test_bspline_basis_rejects_too_few_functions():
    with pytest.raises(ValueError, match="must be >= order"):
        bspline_basis(np.linspace(0, 1, 10), n_basis=3, order=4)


# -------------------------------------------------------------------------------- gp ----


def test_sample_gp_variance_decays_across_the_basis():
    grid = np.linspace(0, 1, 101)
    X = sample_gp(20000, grid, n_basis=6, decay=2.0, variance=1.0, rng=0)
    Psi = trig_basis(grid, n_basis=6)
    # Recover the coefficients by projecting onto the (orthogonal) trig basis.
    coef = np.linalg.lstsq(Psi.T, X.T, rcond=None)[0]
    emp_var = coef.var(axis=1)
    expected = np.exp(-np.arange(6) / 2.0)
    np.testing.assert_allclose(emp_var, expected, rtol=0.06)


def test_sample_gp_is_reproducible_and_shifts_by_its_mean():
    grid = np.linspace(0, 1, 41)
    a = sample_gp(8, grid, rng=5)
    np.testing.assert_array_equal(a, sample_gp(8, grid, rng=5))
    shifted = sample_gp(8, grid, mean=lambda t: 3.0 + 0.0 * t, rng=5)
    np.testing.assert_allclose(shifted - a, 3.0)


# ----------------------------------------------------------------------------- paths ----


def test_repo_root_contains_the_project_marker():
    root = paths.repo_root()
    assert os.path.isabs(root)
    assert os.path.exists(os.path.join(root, "pyproject.toml"))
    assert os.path.isdir(os.path.join(root, "src", "fts_research"))


def test_the_four_directories_are_absolute_and_distinct():
    dirs = [paths.DATA_DIR, paths.REFERENCE_DIR, paths.CACHE_DIR, paths.FIGURE_DIR]
    assert all(os.path.isabs(d) for d in dirs)
    assert len(set(dirs)) == 4


def test_reference_and_cache_are_the_two_halves_of_artifacts():
    """The split is the point: R's outputs and this repository's own caches are rebuilt by
    different commands, and only the second is reproducible here."""
    assert paths.REFERENCE_DIR.endswith(os.path.join("artifacts", "reference"))
    assert paths.CACHE_DIR.endswith(os.path.join("artifacts", "cache"))
    assert os.path.dirname(paths.REFERENCE_DIR) == paths.ARTIFACT_DIR
    assert os.path.dirname(paths.CACHE_DIR) == paths.ARTIFACT_DIR


def test_reference_path_creates_the_directory():
    """R's write.csv fails outright if the target directory is missing -- the single most
    common way the validation notebook broke on a fresh checkout."""
    path = paths.reference_path("smoke_test_marker.csv")
    assert os.path.isdir(os.path.dirname(path))
    assert os.path.isabs(path)


@pytest.mark.parametrize("helper,attr", [("reference_path", "REFERENCE_DIR"),
                                         ("cache_path", "CACHE_DIR"),
                                         ("figure_path", "FIGURE_DIR")])
def test_write_paths_can_skip_creation(helper, attr, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, attr, str(tmp_path / "nope"))
    path = getattr(paths, helper)("x.csv", create=False)
    assert not os.path.exists(os.path.dirname(path))


def test_data_path_follows_the_data_dir_override(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", str(tmp_path))
    assert paths.data_path("tecator.csv") == os.path.join(str(tmp_path), "tecator.csv")


# --------------------------------------------------------------- style: the visual system


def _blank_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    return plt, fig


def test_save_figure_writes_one_file_per_format(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FIGURE_DIR", str(tmp_path))
    plt, fig = _blank_figure()
    try:
        written = style.save_figure(fig, "smoke", formats=("png", "pdf"))
    finally:
        plt.close(fig)
    assert [os.path.basename(p) for p in written] == ["smoke.png", "smoke.pdf"]
    assert all(os.path.getsize(p) > 0 for p in written)


@pytest.mark.parametrize("bad", ["smoke.png", "sub/smoke"])
def test_save_figure_rejects_a_name_that_is_a_path(bad):
    plt, fig = _blank_figure()
    try:
        with pytest.raises(ValueError, match="bare stem"):
            style.save_figure(fig, bad)
    finally:
        plt.close(fig)


def _luminance(hex_colour):
    """Relative luminance of a hex colour, by the WCAG definition."""
    srgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return float(lin @ (0.2126, 0.7152, 0.0722))


def test_the_sequential_ramp_is_one_hue_running_light_to_dark():
    """A magnitude ramp has to be monotone in lightness, or a reader cannot order two cells by
    eye; and it has to be one hue, or it is a rainbow. Both are checked rather than assumed."""
    lums = [_luminance(c) for c in style.SEQUENTIAL_STEPS]
    assert all(a > b for a, b in zip(lums, lums[1:])), lums

    hues = []
    for c in style.SEQUENTIAL_STEPS:
        r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        hues.append(colorsys.rgb_to_hsv(r, g, b)[0] * 360.0)
    assert max(hues) - min(hues) < 25.0, hues        # one hue, not a ramp across the wheel

    sampled = [style.SEQUENTIAL(x) for x in np.linspace(0, 1, 9)]
    assert all(a[:3] != b[:3] for a, b in zip(sampled, sampled[1:]))


def test_clip_to_blanks_the_excursion_and_reports_its_size():
    """The device that makes method A's failures readable: the in-range shape survives, the
    excursion becomes a gap rather than a vertical stroke, and its magnitude comes back as a
    number to label the panel with."""
    y = np.array([0.0, 0.5, 1e6, 0.5, -3.0, 0.25])
    shown, worst = style.clip_to(y, -1.0, 2.0)
    assert np.array_equal(np.isnan(shown), [False, False, True, False, True, False])
    assert np.array_equal(shown[~np.isnan(shown)], [0.0, 0.5, 0.5, 0.25])
    assert worst == pytest.approx(1e6)

    inside, none = style.clip_to(np.array([0.0, 1.0]), -1.0, 2.0)
    assert none is None and np.array_equal(inside, [0.0, 1.0])


def test_fmt_switches_to_scientific_only_where_fixed_point_stops_being_readable():
    assert style.fmt(0.02364) == "0.0236"
    assert style.fmt(1.0) == "1"
    assert style.fmt(239.65) == "240"
    assert style.fmt(0.0) == "0"
    assert style.fmt(2.7e11) == "3e11"
    assert style.fmt(1.5e-6) == "2e-6"


def test_use_sets_the_shared_defaults_without_importing_pyplot_itself():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context():
        style.use(plt)
        assert plt.rcParams["axes.grid"] is True
        assert plt.rcParams["grid.linestyle"] == "-"          # never a dashed grid
        assert plt.rcParams["axes.spines.top"] is False
        cycle = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
        assert cycle == list(style.CATEGORICAL)


def test_the_ordinal_ramp_is_ordered_dark_and_never_reaches_the_pale_end():
    """Four basis functions ordered by roughness are an ordered set, not four identities, so
    they take steps of the one ramp rather than four hues -- which past three slots would start
    cycling. The pale end is held back so the first step still reads against the surface."""
    steps = style.ordinal(4)
    assert len(steps) == len(set(steps)) == 4
    lums = [_luminance(c) for c in steps]
    assert all(a > b for a, b in zip(lums, lums[1:])), lums
    assert _luminance(steps[0]) <= _luminance(style.SEQUENTIAL_STEPS[3])
    assert steps[-1] == style.SEQUENTIAL_STEPS[-1]
    assert style.ordinal(1) == [style.SEQUENTIAL_STEPS[-1]]
    with pytest.raises(ValueError, match="at least one step"):
        style.ordinal(0)
