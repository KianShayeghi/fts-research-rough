"""The port check: `fdaSP.SBC` must compute the same estimator as CRAN `fdaSP::f2sSP`.

The machine-checkable half of `notebooks/LocSpaFLR_validation.ipynb`, run against the
reference CSVs committed in `artifacts/`, so it needs no R. The R-gated tests at the bottom
re-derive those CSVs where R with `fdaSP` is installed.
"""

import numpy as np
import pandas as pd
import pytest

from fdaSP import SBC, r_reference as rref, rbridge as rb


@pytest.fixture(scope="module")
def reference():
    """The committed f2sSP outputs: basis, fixed-lambda fit, design, and the full path."""
    return {
        "Phi": pd.read_csv(rref.cached("R_basis_bs.csv")).to_numpy(float),
        "beta": pd.read_csv(rref.cached("tecator_match_Rbeta.csv")).to_numpy(float),
        "XB": pd.read_csv(rref.cached("tecator_match_XB.csv")).to_numpy(float),
        "lam": pd.read_csv(rref.cached("tecator_match_lam.csv"))["lambda"].to_numpy(float),
        "path_beta": pd.read_csv(rref.cached("tecator_f2sSP_beta_path.csv")).to_numpy(float),
        "path_lam": pd.read_csv(rref.cached("tecator_f2sSP_lam_path.csv"))["lambda"]
                      .to_numpy(float),
    }


# ---------------------------------------------------------------- the three conventions --

def test_python_basis_matches_splines_bs(reference):
    """scipy's B-spline basis must equal R's bs(s, df = M, intercept = TRUE) exactly."""
    RPhi = reference["Phi"]
    grid = np.linspace(0, 1, RPhi.shape[0])
    Phi = SBC.bspline_basis(grid, n_basis=RPhi.shape[1], order=rref.ORDER)
    assert np.max(np.abs(Phi - RPhi)) < 1e-12


def test_f2ssp_penalty_is_S_g_times_inverse_counts():
    """Convention (b): f2sSP's F_g = S_g diag(1 / colSums(GRmat))."""
    F, groups = SBC.f2ssp_penalty(rref.M, order=rref.ORDER)
    counts = np.zeros(rref.M)
    for g in groups:
        counts[g] += 1.0
    expected = np.zeros_like(F)
    for b, g in enumerate(groups):
        S = np.zeros((rref.ORDER, rref.M))
        S[np.arange(rref.ORDER), g] = 1.0
        expected[b * rref.ORDER:(b + 1) * rref.ORDER] = S @ np.diag(1.0 / counts)
    np.testing.assert_allclose(F, expected, atol=1e-15)


def test_f2ssp_penalty_differs_from_the_package_default():
    """The two packages' default parametrisations are *not* the same."""
    default, _ = SBC.pen_matrices(rref.M, order=rref.ORDER)
    matched, _ = SBC.f2ssp_penalty(rref.M, order=rref.ORDER)
    assert not np.allclose(default, matched)


def test_f2ssp_design_matches_the_matrix_R_fitted_on(reference, tecator):
    """Convention (c): W = X Phi as a plain product, with no quadrature weights."""
    W, _ = SBC.f2ssp_design(tecator["X"], n_basis=rref.M, order=rref.ORDER)
    assert np.max(np.abs(W - reference["XB"])) < 1e-10


def test_f2ssp_design_is_not_the_quadrature_weighted_design(tecator):
    """The two designs really do differ, so matching them is not a no-op."""
    grid = np.linspace(0, 1, tecator["X"].shape[1])
    W_plain, _ = SBC.f2ssp_design(tecator["X"], n_basis=rref.M, order=rref.ORDER)
    W_quad, _ = SBC.bspline_design(tecator["X"], grid, n_basis=rref.M, order=rref.ORDER)
    assert not np.allclose(W_plain, W_quad)


# ------------------------------------------------------------------- the fits themselves --

def test_fixed_lambda_fits_reproduce_the_R_reference(reference, tecator):
    """The headline claim, at four penalties spanning the useful range."""
    for j, lam in enumerate(reference["lam"]):
        got = SBC.fit_like_f2ssp(tecator["X"], tecator["y"], lam,
                                 n_basis=rref.M, order=rref.ORDER)["beta"]
        ref = reference["beta"][j]
        assert np.corrcoef(ref, got)[0, 1] > 0.9999, f"shape mismatch at lambda = {lam}"
        assert np.max(np.abs(ref - got)) < 5e-3, f"level mismatch at lambda = {lam}"


def test_whole_path_agrees_in_the_regularised_regime(reference, tecator):
    """Agreement must hold across f2sSP's own lambda grid, not just at hand-picked values."""
    lam, ref = reference["path_lam"], reference["path_beta"]
    keep = lam >= 0.01
    assert keep.sum() > 20, "reference path no longer covers the regularised regime"

    got = np.array([SBC.fit_like_f2ssp(tecator["X"], tecator["y"], lm,
                                       n_basis=rref.M, order=rref.ORDER)["beta"]
                    for lm in lam[keep]])
    assert np.max(np.abs(got - ref[keep])) < 1e-2


def test_agreement_degrades_only_in_the_unpenalised_tail(reference, tecator):
    """Below lambda ~ 0.005 both solvers stop on a near-flat objective.

    Asserted rather than merely stated, so that the notebook's claim is revisited if the
    breakdown ever moves into the useful range.
    """
    lam, ref = reference["path_lam"], reference["path_beta"]
    tail = lam < 0.005
    assert tail.any()
    peak = np.max(np.abs(ref[tail]))
    assert peak > 5.0, "the tail should blow the coefficient function up by an order of magnitude"


def test_python_solution_is_a_better_minimiser_where_the_two_disagree(reference, tecator):
    """Where the tail disagreement appears, check it is a solver artefact, not a wrong model.

    Both solutions are scored on the *same* objective; ours must be at least as good, i.e.
    the discrepancy is R stopping early on a flat problem rather than a different estimator.
    """
    lam = reference["path_lam"]
    j = int(np.argmin(lam))                      # the most unpenalised solution on the path
    W, Phi = SBC.f2ssp_design(tecator["X"], n_basis=rref.M, order=rref.ORDER)
    F, _ = SBC.f2ssp_penalty(rref.M, order=rref.ORDER)
    y = tecator["y"]
    n = W.shape[0]
    mU = np.sqrt(np.var(W, axis=0, ddof=1))
    mV = np.sqrt(np.var(y, ddof=1))
    Ws = (W - W.mean(axis=0)) / mU / np.sqrt(n)
    ys = (y - y.mean()) / mV / np.sqrt(n)

    def obj(psi):
        r = ys - Ws @ psi
        blocks = (F @ psi).reshape(-1, rref.ORDER)
        return 0.5 * r @ r + lam[j] * np.sum(np.linalg.norm(blocks, axis=1))

    mine = SBC.fit_like_f2ssp(tecator["X"], y, lam[j], n_basis=rref.M, order=rref.ORDER)
    psi_mine = mine["psi"] * mU / mV
    # Recover R's spline coefficients from its coefficient function (Phi has full column rank).
    psi_R, *_ = np.linalg.lstsq(Phi, reference["path_beta"][j], rcond=None)
    assert obj(psi_mine) <= obj(psi_R * mU / mV) + 1e-9


# ------------------------------------------------------------------------- R-gated -------

@pytest.mark.needs_r
def test_rbridge_finds_R_and_the_fdaSP_package():
    assert rb.find_rscript() is not None
    out = rb.run_r('cat(as.character(packageVersion("fdaSP")))')
    assert out.stdout.strip()


@pytest.mark.needs_r
@pytest.mark.slow
def test_regenerating_the_basis_reproduces_the_committed_file(reference):
    """Guards against the committed artifacts drifting from what R actually produces."""
    fresh = pd.read_csv(rref.basis()).to_numpy(float)
    np.testing.assert_allclose(fresh, reference["Phi"], atol=1e-12)


@pytest.mark.needs_r
@pytest.mark.slow
def test_regenerating_the_fixed_lambda_fit_reproduces_the_committed_file(reference):
    beta_path, XB_path, lam_path = rref.fixed_lambda_fit()
    np.testing.assert_allclose(pd.read_csv(XB_path).to_numpy(float), reference["XB"],
                               atol=1e-10)
    np.testing.assert_allclose(pd.read_csv(lam_path)["lambda"].to_numpy(float),
                               reference["lam"], atol=1e-12)
    fresh = pd.read_csv(beta_path).to_numpy(float)
    # f2sSP's own ADMM is not bit-reproducible across runs, but must land in the same place.
    assert np.max(np.abs(fresh - reference["beta"])) < 1e-6
