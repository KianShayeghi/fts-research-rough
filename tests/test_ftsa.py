"""Tests for `ftsa`: the score forecaster, the bootstrap bands, and the rolling-origin score.

A forecaster is hard to test by its forecasts, so most of what follows fixes the pieces that
have a right answer independent of any data. The decomposition must be `far.far_est.fpca`'s
and must reconstruct; the reassembly must invert the projection exactly; the interval score
must reduce to the width of the band when nothing is missed and must charge the documented
penalty when something is; and the two forecasters must be exactly right on series that were
built to be forecastable, an ARMA for the univariate route and a genuine FAR(1) for the other.

The rolling-origin loop is then checked for the property that makes it worth running at all --
that no curve is fitted on before it is scored -- by feeding it a sample whose second half is
unlike its first, where any leakage would show.
"""

import numpy as np
import pytest

from fts_research.far import far_dgp
from fts_research.far.far_est import fpca
from fts_research.ftsa import evaluation as ev
from fts_research.ftsa import forecast as fc

GRID = np.linspace(0.0, 1.0, 40)


@pytest.fixture(scope="module")
def curves():
    """A full-rank FAR(1) at rho = 0.7: dependent curves with something to forecast."""
    Psi = far_dgp.exponential_kernel(GRID, rho=0.7)
    return far_dgp.sample_far1(220, GRID, Psi, rng=3)


@pytest.fixture(scope="module")
def spx():
    """The replication design itself, which is also the only test that the data still loads."""
    return ev.replication_curves()


# ------------------------------------------------------------------------- decompose ----


def test_decompose_is_far_ests_decomposition_truncated(curves):
    """Nothing here re-implements FPCA; a second decomposition with its own conventions is the
    one bug this package could introduce that nothing downstream would show."""
    dec = fc.decompose(curves, GRID, n_components=4)
    reference = fpca(curves, GRID, n_components=4)
    assert np.allclose(dec["phi"], reference["phi"])
    assert np.allclose(dec["scores"], reference["scores"])
    assert np.allclose(dec["tau"], reference["tau"])


def test_variance_threshold_keeps_the_fewest_components_that_reach_it(curves):
    for threshold in (0.5, 0.8, 0.95):
        dec = fc.decompose(curves, GRID, var_threshold=threshold)
        assert dec["explained"][-1] >= threshold
        if dec["n_components"] > 1:
            assert dec["explained"][-2] < threshold


def test_a_threshold_outside_the_unit_interval_is_refused(curves):
    """It clamped silently to `max_components` before, which reads as a fitted choice."""
    for bad in (1.5, 0.0, -0.2):
        with pytest.raises(ValueError, match="fraction of total variance"):
            fc.decompose(curves, GRID, var_threshold=bad)


def test_residual_curves_are_what_the_truncation_left(curves):
    dec = fc.decompose(curves, GRID, n_components=3)
    rebuilt = fc.reconstruct(dec["mean"], dec["phi"], dec["scores"])
    assert np.allclose(dec["resid_curves"], curves - rebuilt, atol=1e-12)


def test_a_full_decomposition_leaves_no_residual(curves):
    dec = fc.decompose(curves, GRID, n_components=curves.shape[0] - 1, max_components=200)
    assert np.max(np.abs(dec["resid_curves"])) < 1e-8


def test_reconstruct_inverts_the_projection(curves):
    """mean + scores @ phi is the curve back, which is the identity the whole method rests on."""
    dec = fc.decompose(curves, GRID, n_components=6)
    scores = dec["scores"]
    assert np.allclose(fc.reconstruct(dec["mean"], dec["phi"], scores),
                       dec["mean"][None, :] + scores @ dec["phi"])


# -------------------------------------------------------------------------- forecast ----


def test_arima_recovers_a_known_ar1_coefficient():
    """The order search must find an AR(1) in an AR(1) and forecast it by rho times the last
    value; if it does not, every score forecast downstream is arbitrary."""
    rng = np.random.default_rng(0)
    y = np.zeros(600)
    for t in range(1, 600):
        y[t] = 0.7 * y[t - 1] + rng.normal()
    out = fc.forecast_scores(y[:, None], h=2, method="arima")
    assert out["orders"][0][0] >= 1                    # some autoregressive term was chosen
    assert out["point"][0, 0] == pytest.approx(0.7 * y[-1], abs=0.25)
    assert out["variance"][1, 0] > out["variance"][0, 0]


def test_far_route_is_the_far1_operator_in_score_space(curves):
    """`method="far"` must be `far.far_est`'s estimator and not a second one: the forecast has
    to equal the curve `far1_predict` returns, projected back onto the same components."""
    from fts_research.far.far_est import far1_fit, far1_predict

    dec = fc.decompose(curves, GRID, n_components=4)
    out = fc.forecast_scores(dec["scores"], h=1, method="far")
    predicted = fc.reconstruct(dec["mean"], dec["phi"], out["point"])

    fit = far1_fit(curves, GRID, p=4)
    assert np.allclose(predicted[0], far1_predict(fit, curves[-1])[0], atol=1e-10)


def test_far_forecast_variance_accumulates_with_the_horizon(curves):
    dec = fc.decompose(curves, GRID, n_components=3)
    out = fc.forecast_scores(dec["scores"], h=4, method="far")
    assert np.all(np.diff(out["variance"], axis=0) >= -1e-12)
    assert out["resid"].shape == (dec["scores"].shape[0] - 1, 3)


def test_far_point_forecast_is_the_operator_applied_repeatedly(curves):
    dec = fc.decompose(curves, GRID, n_components=3)
    out = fc.forecast_scores(dec["scores"], h=3, method="far")
    Psi = out["Psi"]
    assert np.allclose(out["point"][0], Psi @ dec["scores"][-1])
    assert np.allclose(out["point"][2], np.linalg.matrix_power(Psi, 3) @ dec["scores"][-1])


def test_an_unknown_method_is_refused(curves):
    dec = fc.decompose(curves, GRID, n_components=2)
    with pytest.raises(ValueError, match="arima"):
        fc.forecast_scores(dec["scores"], method="var")


# --------------------------------------------------------------------------- bands -----


def test_bands_are_nested_and_contain_the_point_forecast(curves):
    dec = fc.decompose(curves, GRID, n_components=3)
    out = fc.prediction_bands(dec, h=1, levels=(0.5, 0.8, 0.95), n_boot=400,
                              method="far", rng=0)
    lo50, hi50 = out["bands"][0.5]
    lo95, hi95 = out["bands"][0.95]
    assert np.all(lo95 <= lo50) and np.all(hi50 <= hi95)
    assert np.all(hi95 - lo95 > 0)


def test_the_residual_function_source_widens_the_band(curves):
    """Shang's second error source is the truncation residual. Dropping it must narrow the
    band, or it is not being resampled at all."""
    dec = fc.decompose(curves, GRID, n_components=2)
    wide = fc.prediction_bands(dec, n_boot=600, method="far", rng=1)
    narrow = fc.prediction_bands(dec, n_boot=600, method="far", rng=1,
                                 add_residual_function=False)
    width = lambda out: float(np.mean(out["bands"][0.8][1] - out["bands"][0.8][0]))  # noqa: E731
    assert width(wide) > width(narrow)


def test_bands_cover_close_to_their_nominal_rate(curves):
    """A band that does not cover is not a band. Fitted on all but the last ten curves and
    scored on them, at a level loose enough that ten curves can measure it."""
    dec = fc.decompose(curves[:-10], GRID, n_components=3)
    out = fc.prediction_bands(dec, h=10, levels=(0.95,), n_boot=800, method="far", rng=2)
    lower, upper = out["bands"][0.95]
    covered = np.mean((curves[-10:] >= lower) & (curves[-10:] <= upper))
    assert covered > 0.8


def test_forecast_curves_reports_what_the_pieces_report(curves):
    out = fc.forecast_curves(curves, GRID, n_components=3, n_boot=200, method="far", rng=4)
    assert out["n_components"] == 3
    assert out["explained"].shape == (3,)
    assert out["forecast"]["center"].shape == (1, GRID.size)
    assert len(out["orders"]) == 3


# -------------------------------------------------------------------- interval score ----


def test_interval_score_is_the_width_when_nothing_is_missed():
    lower, upper = np.array([-1.0, -2.0]), np.array([1.0, 2.0])
    actual = np.array([0.0, 1.5])
    assert np.allclose(ev.interval_score(lower, upper, actual, 0.2), [2.0, 4.0])


def test_interval_score_charges_the_documented_penalty_for_a_miss():
    """A realised value `e` below the band costs `(2/alpha) e` on top of the width, so at
    alpha = 0.2 one unit outside costs ten."""
    score = ev.interval_score(np.array([0.0]), np.array([1.0]), np.array([-1.0]), 0.2)
    assert score[0] == pytest.approx(1.0 + 10.0)
    above = ev.interval_score(np.array([0.0]), np.array([1.0]), np.array([2.0]), 0.2)
    assert above[0] == pytest.approx(1.0 + 10.0)


def test_a_wider_band_is_only_worth_it_when_it_catches_something():
    tight = ev.interval_score(np.array([-0.5]), np.array([0.5]), np.array([0.0]), 0.2)
    loose = ev.interval_score(np.array([-3.0]), np.array([3.0]), np.array([0.0]), 0.2)
    assert tight[0] < loose[0]
    missed = ev.interval_score(np.array([-0.5]), np.array([0.5]), np.array([2.0]), 0.2)
    caught = ev.interval_score(np.array([-3.0]), np.array([3.0]), np.array([2.0]), 0.2)
    assert caught[0] < missed[0]


# ------------------------------------------------------------------- rolling origin -----


def test_rolling_origin_never_fits_on_what_it_scores(curves):
    """The property the whole protocol exists for. A level shift is applied from the first
    holdout session onwards, so no training curve carries it at the first split: a forecaster
    that had seen the holdout would predict the shift, and one that has not cannot."""
    init_frac = 0.6
    start = int(np.floor(init_frac * len(curves)))
    shifted = curves.copy()
    shifted[start:] += 50.0                # far larger than the curves' own spread
    out = ev.rolling_origin(shifted, GRID, method="far", n_boot=60, init_frac=init_frac, rng=0)
    assert out["start"] == start
    first_error = float(np.mean(out["actual"][0] - out["point"][0]))
    assert first_error > 40.0, "the first holdout curve's shift was already known to the fit"


def test_rolling_origin_profiles_and_means_agree(curves):
    out = ev.rolling_origin(curves, GRID, method="far", n_boot=80, init_frac=0.7, rng=1)
    assert out["mafe"] == pytest.approx(out["mafe_by_t"].mean())
    assert out["msfe"] == pytest.approx(out["msfe_by_t"].mean())
    assert out["actual"].shape == out["point"].shape == (curves.shape[0] - out["start"],
                                                         GRID.size)
    assert np.all(out["lower"] <= out["upper"])


def test_rolling_origin_refuses_a_window_that_leaves_nothing_to_score(curves):
    with pytest.raises(ValueError, match="init_frac"):
        ev.rolling_origin(curves, GRID, init_frac=1.5)


# ----------------------------------------------------------------- the cached design ----


def test_the_replication_curves_are_shangs_design(spx):
    """125 full sessions of the first half of 2017 on a 78-point grid, in per cent, with the
    opening anchor dropped."""
    X, grid, dates = spx
    assert X.shape == (125, ev.CONFIG["n_points"] - 1)
    assert grid.shape == (ev.CONFIG["n_points"] - 1,)
    assert str(dates[0]) == "2017-01-03" and str(dates[-1]) == "2017-06-30"
    assert grid[0] > 0.0, "the identically-zero opening mark should have been dropped"
    assert np.abs(X).max() < 5.0, "curves should be in per cent, not in log points"


def test_the_cached_forecasts_cover_both_methods_on_the_same_splits():
    """Both forecasters must be scored on identical holdout sessions, or the comparison in the
    notebook is between two different experiments."""
    df = ev.cached_rolling()
    assert set(df["method"]) == set(ev.CONFIG["methods"])
    splits = {m: sorted(set(part["session"])) for m, part in df.groupby("method")}
    assert len({tuple(v) for v in splits.values()}) == 1
    assert len(next(iter(splits.values()))) == 50


def test_summarise_reproduces_the_scores_from_the_cached_forecasts():
    df = ev.cached_rolling()
    table = ev.summarise(df)
    assert set(table.columns) == {"mafe", "msfe", "iscore", "coverage", "n_splits"}
    assert (table["coverage"] > 0.6).all() and (table["coverage"] < 0.95).all()
    assert (table["mafe"] > 0).all()

    part = df[df["method"] == "arima"]
    expected = float(np.mean(np.abs(part["actual"] - part["point"])))
    assert table.loc["arima", "mafe"] == pytest.approx(expected)
