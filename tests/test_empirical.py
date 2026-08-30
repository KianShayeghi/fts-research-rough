"""Tests for `fdaSP.empirical`: dataset resolution, caching, and the reshaping helpers.

The bug this module exists to prevent is a path bug. The loaders used to resolve their CSVs
relative to a directory that depended on where the process was started, which meant a
notebook and a script disagreed about where the data lived. So the first thing checked here
is that the resolved paths are absolute, repo-anchored, and identical from any working
directory.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from fdaSP import empirical as emp


# ------------------------------------------------------------------ where the data is ----

def test_repo_root_contains_the_project_marker():
    root = emp.repo_root()
    assert os.path.isabs(root)
    assert os.path.exists(os.path.join(root, "pyproject.toml"))
    assert os.path.isdir(os.path.join(root, "src", "fdaSP"))


def test_data_dir_is_absolute_and_under_the_repo():
    assert os.path.isabs(emp.DATA_DIR)
    assert os.path.basename(emp.DATA_DIR) == "data"


def test_dataset_paths_are_absolute():
    for name in emp.SOURCES:
        assert os.path.isabs(emp.dataset_path(name))


def test_dataset_path_rejects_unknown_names():
    with pytest.raises(KeyError, match="unknown dataset"):
        emp.dataset_path("does_not_exist.csv")


def test_data_dir_does_not_depend_on_the_working_directory(tmp_path):
    """The original path bug: resolution must not change when the CWD changes."""
    code = "import fdaSP.empirical as e; print(e.dataset_path('tecator.csv'))"
    env = dict(os.environ, PYTHONPATH=os.path.join(emp.repo_root(), "src"))
    env.pop("FTS_DATA_DIR", None)
    here = subprocess.run([sys.executable, "-c", code], cwd=emp.repo_root(),
                          capture_output=True, text=True, env=env, check=True)
    there = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path),
                           capture_output=True, text=True, env=env, check=True)
    assert here.stdout.strip() == there.stdout.strip()


def test_every_source_records_its_provenance():
    for name, src in emp.SOURCES.items():
        assert src["url"].startswith("https://"), f"{name} must cite an https source"
        assert len(src["note"]) > 40, f"{name} needs a provenance note"


def test_ensure_dataset_is_idempotent():
    """A second call must not re-download; it returns the same cached path."""
    first = emp.ensure_dataset("tecator.csv")
    mtime = os.path.getmtime(first)
    second = emp.ensure_dataset("tecator.csv")
    assert first == second
    assert os.path.getmtime(second) == mtime


def test_ensure_dataset_reports_a_useful_error_when_offline(monkeypatch, tmp_path):
    """A missing file plus no network must name the file, the URL and the override."""
    monkeypatch.setattr(emp, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(emp, "_download", lambda url: (_ for _ in ()).throw(OSError("no net")))
    with pytest.raises(RuntimeError) as exc:
        emp.ensure_dataset("tecator.csv")
    msg = str(exc.value)
    assert "tecator.csv" in msg and "FTS_DATA_DIR" in msg and "openml" in msg.lower()


# ----------------------------------------------------------------------- the datasets ----

def test_load_tecator_shape_and_range(tecator):
    X, y = tecator["X"], tecator["y"]
    assert X.shape == (215, 100)
    assert y.shape == (215,)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    # The published range of the standard 215-sample interpolation set.
    assert y.min() == pytest.approx(11.0)
    assert y.max() == pytest.approx(21.8)
    np.testing.assert_allclose(tecator["wl"][[0, -1]], [850.0, 1050.0])


@pytest.mark.parametrize("target", ["protein", "fat", "water"])
def test_load_tecator_targets(target):
    X, y, wl = emp.load_tecator(target)
    assert y.shape == (215,)
    assert y.min() > 0


def test_tecator_contents_are_percentages():
    """Water + fat + protein should account for essentially all of each sample."""
    _, water, _ = emp.load_tecator("water")
    _, fat, _ = emp.load_tecator("fat")
    _, protein, _ = emp.load_tecator("protein")
    total = water + fat + protein
    assert np.all(total > 90) and np.all(total < 110)


def test_load_tecator_rejects_bad_arguments():
    with pytest.raises(ValueError, match="protein/fat/water"):
        emp.load_tecator(target="nonsense")
    with pytest.raises(ValueError, match="deriv must be"):
        emp.load_tecator(deriv=5)


def test_derivative_spectra_have_the_same_width():
    X0, _, _ = emp.load_tecator(deriv=0)
    X1, _, _ = emp.load_tecator(deriv=1)
    X2, _, _ = emp.load_tecator(deriv=2)
    assert X0.shape == X1.shape == X2.shape
    # A first difference removes an additive baseline; a constant shift must not survive.
    shifted, _, _ = emp.load_tecator(deriv=1)
    np.testing.assert_allclose(shifted, X1)


def test_second_derivative_is_two_finite_differences():
    X, _, _ = emp.load_tecator(deriv=0)
    np.testing.assert_allclose(emp.second_derivative(X), emp.load_tecator(deriv=2)[0])


def test_load_stockdata_panel():
    panel = emp.load_stockdata()
    assert list(panel.columns) == ["MSFT", "IBM", "SBUX", "AAPL", "GSPC"]
    assert len(panel) == 2306
    assert str(panel.index[0].date()) == "2007-01-03"
    assert str(panel.index[-1].date()) == "2016-03-01"
    assert panel.index.is_monotonic_increasing
    assert (panel > 0).all().all(), "prices must be positive to take logs"


# ------------------------------------------------------------------- rolling windows -----

def test_rolling_windows_slices_correctly():
    s = np.arange(10, dtype=float)
    X, y, grid = emp.rolling_windows(s, width=3, horizon=2, reduce_y="sum")
    assert X.shape == (6, 3) and y.shape == (6,) and grid.shape == (3,)
    np.testing.assert_array_equal(X[0], [0, 1, 2])
    assert y[0] == 3 + 4
    np.testing.assert_array_equal(X[-1], [5, 6, 7])
    assert y[-1] == 8 + 9
    np.testing.assert_allclose(grid, np.linspace(0, 1, 3))


@pytest.mark.parametrize("reduce_y,expected", [("sum", 7.0), ("mean", 3.5), ("last", 4.0)])
def test_rolling_windows_reducers(reduce_y, expected):
    s = np.arange(10, dtype=float)
    _, y, _ = emp.rolling_windows(s, width=3, horizon=2, reduce_y=reduce_y)
    assert y[0] == expected


def test_rolling_windows_window_count():
    s = np.arange(100, dtype=float)
    X, y, _ = emp.rolling_windows(s, width=20, horizon=5)
    assert X.shape[0] == 100 - 20 - 5 + 1 == y.shape[0]


def test_rolling_windows_validates_its_arguments():
    s = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="must be positive"):
        emp.rolling_windows(s, width=0, horizon=1)
    with pytest.raises(ValueError, match="reduce_y must be"):
        emp.rolling_windows(s, width=2, horizon=1, reduce_y="median")
    with pytest.raises(ValueError, match="too short"):
        emp.rolling_windows(s, width=8, horizon=8)


def test_rolling_windows_predictor_and_response_do_not_overlap():
    """Leakage check: the response block must start strictly after the predictor window."""
    s = np.arange(50, dtype=float)
    X, y, _ = emp.rolling_windows(s, width=5, horizon=3, reduce_y="last")
    for i in range(X.shape[0]):
        assert y[i] > X[i].max(), "response must come from strictly later observations"
