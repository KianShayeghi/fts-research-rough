"""Where this repository keeps its inputs and its outputs.

Four directories, told apart by how each one comes back if it is deleted:

    data/                 Inputs. Fetched from the public sources recorded in
                          `fdaSP.empirical` and `fdaSP.intraday`, and committed, because the
                          intraday panels are dated snapshots that a later fetch does not
                          return. `provenance.py` records what each committed file is.
    artifacts/reference/  What another implementation produced, for this one to be checked
                          against: CRAN `fdaSP::f2sSP`, run by `scripts/make_r_artifacts.py`.
                          Committed, so `tests/test_f2ssp_match.py` and
                          `notebooks/LocSpaFLR_validation.ipynb` run where R is not installed.
    artifacts/cache/      What this implementation produced, where recomputing it takes
                          minutes: the `bench` study. Committed, so the comparison notebook
                          opens without refitting, and rebuilt by `scripts/make_study_cache.py`.
    figures/              Figures exported for writing up, by `style.save_figure`. Every one
                          is redrawn by a notebook or a script, so none is committed.

Nothing outside this module joins one of those names onto a path; `tests/test_layout.py`
holds that. `FTS_DATA_DIR` moves `data/` elsewhere, it being the one of the four that is
large enough to want to keep off a small disk.
"""

import os

__all__ = [
    "repo_root", "DATA_DIR", "ARTIFACT_DIR", "REFERENCE_DIR", "CACHE_DIR", "FIGURE_DIR",
    "data_path", "reference_path", "cache_path", "figure_path",
]


def repo_root():
    """Directory containing `pyproject.toml`, found by walking up from this file. Falls back
    to three levels above this file if no marker is found."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = here
    while True:
        if os.path.exists(os.path.join(path, "pyproject.toml")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return os.path.abspath(os.path.join(here, "..", "..", ".."))
        path = parent


DATA_DIR = os.environ.get("FTS_DATA_DIR") or os.path.join(repo_root(), "data")
ARTIFACT_DIR = os.path.join(repo_root(), "artifacts")
REFERENCE_DIR = os.path.join(ARTIFACT_DIR, "reference")
CACHE_DIR = os.path.join(ARTIFACT_DIR, "cache")
FIGURE_DIR = os.path.join(repo_root(), "figures")


def _resolve(directory, name, create):
    if create:
        os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name)


def data_path(name, create=False):
    """Absolute path of a file under `data/`, whether or not it exists yet.

    PARAMETERS
    create : bool, create the directory if it is missing. False here, and True in the three
             below, because a dataset is fetched by `fdaSP.empirical.ensure_dataset` rather
             than written wherever this path is taken.
    """
    return _resolve(DATA_DIR, name, create)


def reference_path(name, create=True):
    """Absolute path of a file under `artifacts/reference/`, whether or not it exists yet.

    PARAMETERS
    create : bool, create the directory if it is missing; R's `write.csv` fails outright if
             it is not there.
    """
    return _resolve(REFERENCE_DIR, name, create)


def cache_path(name, create=True):
    """Absolute path of a file under `artifacts/cache/`, whether or not it exists yet.

    PARAMETERS
    create : bool, create the directory if it is missing.
    """
    return _resolve(CACHE_DIR, name, create)


def figure_path(name, create=True):
    """Absolute path of a file under `figures/`, whether or not it exists yet.

    PARAMETERS
    create : bool, create the directory if it is missing.
    """
    return _resolve(FIGURE_DIR, name, create)
