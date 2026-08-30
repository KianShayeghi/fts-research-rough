"""Shared fixtures and markers for the test suite.

Every estimator here is deterministic given a seed, so the simulated-data fixtures are
session-scoped and shared across assertions. Two markers keep the fast suite fast:

- `slow`    -- full estimator pipelines (seconds each). Deselect with `-m "not slow"`.
- `needs_r` -- requires R with the `fdaSP` package; skipped automatically otherwise.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fdaSP import SBC_sim                      # noqa: E402
from truncFLR import HH_dgp                    # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full estimator pipelines, seconds each")
    config.addinivalue_line("markers", "needs_r: requires R with the fdaSP package installed")


def pytest_collection_modifyitems(config, items):
    """Skip `needs_r` tests when R or the fdaSP package is unavailable."""
    if not any(item.get_closest_marker("needs_r") for item in items):
        return
    from fdaSP import rbridge as rb
    if rb.r_available("fdaSP"):
        return
    skip = pytest.mark.skip(reason=f"R with fdaSP not found (Rscript: {rb.find_rscript()})")
    for item in items:
        if item.get_closest_marker("needs_r"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def hh_data():
    """Small Hall & Hooker dataset: window [0.8, 1], indicator coefficient."""
    return HH_dgp.sim_dataset(n=120, n_grid=101, u=0.8, v=1.0, snr=26.25, rng=7)


@pytest.fixture(scope="session")
def hh_data_big():
    """The notebook's own configuration, for the accuracy tests."""
    return HH_dgp.sim_dataset(n=200, n_grid=301, u=0.8, v=1.0, snr=26.25, rng=123)


@pytest.fixture(scope="session")
def sbc_data():
    """Locally sparse scalar-on-function data with a genuine null region on [0.3, 0.7]."""
    X, y, grid, beta = SBC_sim.simulate(SBC_sim.psi1, n=60, m=80, snr=4.0, rng=11)
    return {"X": X, "y": y, "grid": grid, "beta": beta}


@pytest.fixture(scope="session")
def tecator():
    """Tecator spectra and protein content. Downloads and caches on first use."""
    from fdaSP import empirical as emp
    X, y, wl = emp.load_tecator("protein")
    return {"X": X, "y": y, "wl": wl}


def ogl_objective(W, y, F, lam, psi, order=4):
    """Objective of problem (62.3): 1/2 ||y - W psi||^2 + lam sum_g ||F_g psi||_2.

    Used by several tests to check that a solver output really is a minimiser.
    """
    resid = y - W @ psi
    blocks = (F @ psi).reshape(-1, order)
    return 0.5 * float(resid @ resid) + lam * float(np.sum(np.linalg.norm(blocks, axis=1)))
