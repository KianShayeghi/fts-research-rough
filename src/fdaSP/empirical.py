"""Real data for the empirical notebooks, reshaped into the (X, y, grid) form a
scalar-on-function regression expects.

Two sources:

- Tecator: near-infrared absorbance spectra of meat samples with fat/water/protein content,
  the application dataset of Stefanucci, Bernardi & Canale (2025, Section 62.5). Already
  functional -- each spectrum is a curve over wavelength.
- Daily equity prices: a scalar series, made functional by `rolling_windows`, which slices it
  into overlapping length-W windows of returns paired with a scalar summary of the following
  H observations. This is the worked example of rows that are *not* exchangeable, which is
  what `SBC.blocked_folds` exists for.

Intraday bars, the trading-day curves built from them and the overnight jumps paired with
those curves live in `intraday.py` instead.

The DGP and simulation live in `SBC_sim`; the estimator in `SBC`.

CSVs are cached under the repository's `data/` directory (override with `FTS_DATA_DIR`), and
downloaded from the public source recorded in `SOURCES` on first use. Run
`python -m fdaSP.empirical` to fetch everything up front, or with `--force` to re-download.
"""

import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

__all__ = [
    "DATA_DIR",
    "SOURCES",
    "repo_root",
    "dataset_path",
    "ensure_dataset",
    "ensure_all",
    "load_tecator",
    "second_derivative",
    "load_stockdata",
    "rolling_windows",
]

_USER_AGENT = "fts-research/0.1 (+https://github.com/KianShayeghi/fts-research)"
_TIMEOUT = 120


def repo_root():
    """Directory containing `pyproject.toml`, found by walking up from this file. Falls back
    to two levels above this file if no marker is found."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = here
    while True:
        if os.path.exists(os.path.join(path, "pyproject.toml")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return os.path.abspath(os.path.join(here, "..", ".."))
        path = parent


DATA_DIR = os.environ.get("FTS_DATA_DIR") or os.path.join(repo_root(), "data")

# Provenance for every cached CSV. `builder` turns the raw download into the frame this
# module's loaders expect; `url` and `note` are what you cite.
SOURCES = {
    "tecator.csv": {
        "url": "https://openml.org/data/v1/download/52617/tecator.arff",
        "note": (
            "Tecator Infratec Food and Feed Analyzer, 850-1050 nm NIT absorbance spectra "
            "(OpenML dataset 505, originally StatLib). Public domain, redistributable. "
            "The first 215 of the 240 samples are kept -- the standard "
            "training+monitoring interpolation set, and the n used by the paper."
        ),
    },
    "stockdata_daily.csv": {
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv",
        "note": (
            "Daily closing prices for MSFT, IBM, SBUX, AAPL and the S&P 500 index (GSPC), "
            "2007-01-03 to 2016-03-01, from the public plotly/datasets repository."
        ),
    },
}


def dataset_path(name):
    """Absolute path of a cached dataset CSV, whether or not it exists yet."""
    if name not in SOURCES:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(SOURCES)}.")
    return os.path.join(DATA_DIR, name)


def _download(url):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def _build_tecator(raw):
    """Parse the OpenML Tecator ARFF into a tidy frame.

    Each row is 100 absorbances, then 22 principal components, then moisture, fat and
    protein. The components are dropped -- they are a PCA of the training subset only, and we
    work with the spectra themselves. Only the first 215 samples are kept.
    """
    text = raw.decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().lower().startswith("@data"))
    rows = [ln for ln in lines[start + 1:] if ln.strip()]
    A = np.array([[float(v) for v in ln.split(",")] for ln in rows])
    if A.shape[1] != 125:
        raise ValueError(f"expected 125 ARFF columns, got {A.shape[1]}.")
    A = A[:215]
    cols = {f"x_{j + 1}": A[:, j] for j in range(100)}
    cols["water"], cols["fat"], cols["protein"] = A[:, 122], A[:, 123], A[:, 124]
    return pd.DataFrame(cols)


def _build_stockdata(raw):
    """Plotly's stockdata.csv, with Date moved to the front and rows sorted by date."""
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    cols = ["Date"] + [c for c in df.columns if c != "Date"]
    return df[cols].sort_values("Date").reset_index(drop=True)


_BUILDERS = {
    "tecator.csv": _build_tecator,
    "stockdata_daily.csv": _build_stockdata,
}


def ensure_dataset(name, force=False):
    """Return the path to a cached dataset CSV, downloading it first if necessary.

    PARAMETERS
    name  : str, a key of ``SOURCES``;
    force : bool, re-download and overwrite even if the file is already cached.
    """
    path = dataset_path(name)
    if os.path.exists(path) and not force:
        return path

    src = SOURCES[name]
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        raw = _download(src["url"])
    except Exception as exc:  # network failure, proxy, offline...
        raise RuntimeError(
            f"{name} is not cached at {path} and could not be downloaded from "
            f"{src['url']}: {exc}. Fetch it manually, or set FTS_DATA_DIR to a directory "
            f"that already holds it."
        ) from exc

    df = _BUILDERS[name](raw)
    tmp = path + ".part"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic, so an interrupted fetch never leaves a half file
    return path


def ensure_all(force=False):
    """Download and cache every dataset. Returns {name: path}."""
    return {name: ensure_dataset(name, force=force) for name in SOURCES}


def load_tecator(target="protein", deriv=0):
    """Load the Tecator dataset.

    Returns X (n, 100) absorbance spectra, y (n,) response, and wl (100,) the
    wavelength grid in nanometres.

    PARAMETERS
    target : {"protein", "fat", "water"}, scalar response;
    deriv  : {0, 1, 2}, number of finite differences applied along the wavelength axis. The
             second difference is the standard preprocessing for these spectra: it removes the
             additive baseline and the linear tilt that scattering puts on every curve.
    """
    if target not in ("protein", "fat", "water"):
        raise ValueError(f"target must be protein/fat/water, got {target!r}.")
    if deriv not in (0, 1, 2):
        raise ValueError(f"deriv must be 0, 1 or 2, got {deriv}.")

    df = pd.read_csv(ensure_dataset("tecator.csv"))
    xcols = [c for c in df.columns if c.startswith("x_")]
    X = df[xcols].to_numpy(float)
    y = df[target].to_numpy(float)
    wl = np.linspace(850.0, 1050.0, len(xcols))

    for _ in range(deriv):
        X = _finite_diff(X)
    return X, y, wl


def _finite_diff(X):
    """First finite difference along the wavelength axis, edge-padded to preserve width.

    The difference is not divided by the channel spacing, so it is h times the derivative;
    the constant is absorbed into psi and leaves the fit unchanged.
    """
    d = np.diff(X, axis=1)
    return np.column_stack([d[:, :1], d])


def second_derivative(X):
    """Second finite difference of each spectrum, edge-padded to the original width."""
    X = np.atleast_2d(np.asarray(X, float))
    return _finite_diff(_finite_diff(X))


def load_stockdata():
    """Load the long daily price panel (MSFT/IBM/SBUX/AAPL/GSPC, 2007-2016).

    Returns a DataFrame indexed by Date with one column per instrument.
    """
    df = pd.read_csv(ensure_dataset("stockdata_daily.csv"))
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").set_index("Date")


def rolling_windows(series, width, horizon, reduce_y="sum"):
    """Slice a 1-D series into functional predictors and scalar responses.

    Predictor i is the window s[i : i + width], rescaled onto the grid [0, 1] so lag position
    within the window becomes the functional argument; the paired response summarises the
    following `horizon` observations, which are strictly after it.

    Consecutive predictors share `width - 1` observations and consecutive responses share
    `horizon - 1`, so the rows are far from exchangeable. Cross-validating over them needs
    `SBC.blocked_folds` and an embargo of at least `width + horizon`; random folds put
    near-duplicate windows on both sides of the split.

    Returns X (m, width), y (m,), grid (width,).

    PARAMETERS
    series   : array, the 1-D series to slice;
    width    : int, number of points in each predictor window;
    horizon  : int, number of following points summarised into the response;
    reduce_y : {"sum", "mean", "last"}, how the horizon block becomes a scalar.
    """
    s = np.asarray(series, float).ravel()
    if width < 1 or horizon < 1:
        raise ValueError("width and horizon must be positive.")
    reducers = {"sum": np.sum, "mean": np.mean, "last": lambda a: a[-1]}
    if reduce_y not in reducers:
        raise ValueError(f"reduce_y must be one of {list(reducers)}.")
    red = reducers[reduce_y]

    m = s.size - width - horizon + 1
    if m <= 0:
        raise ValueError("series too short for the requested width and horizon.")
    X = np.empty((m, width))
    y = np.empty(m)
    for i in range(m):
        X[i] = s[i:i + width]
        y[i] = red(s[i + width:i + width + horizon])
    grid = np.linspace(0.0, 1.0, width)
    return X, y, grid


def _main(argv):
    force = "--force" in argv
    print(f"data directory: {DATA_DIR}")
    for name, path in ensure_all(force=force).items():
        size = os.path.getsize(path)
        print(f"  {name:<22} {size / 1024:8.1f} KB  <- {SOURCES[name]['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
