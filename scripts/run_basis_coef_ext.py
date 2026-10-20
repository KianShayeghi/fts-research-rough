"""Extend `notebooks/basis_and_coef_comp.ipynb`'s study with `truncFLR.HH_continuous` and
`truncFLR.HH_multi`, on the notebook's nine designs and on the same designs with a small
perturbation added to the null region.

    python scripts/run_basis_coef_ext.py             # run, or read back from the cache
    python scripts/run_basis_coef_ext.py --refresh   # re-run and overwrite the cache
    python scripts/run_basis_coef_ext.py --quick     # 3 replications, for a smoke test

Writes `artifacts/cache/basis_coef_ext_records.csv` (one row per replication, method, tuning,
orientation and variant, the columns `bench.study.score` produces plus the new methods'
breakpoints) and `artifacts/cache/basis_coef_ext_arrays.npz` (every prescribed fit's
coefficient function and support mask, and each replication's true psi).
`scripts/make_basis_coef_ext_figures.py` draws everything from those two files.

The existing four methods are run by `bench.study.run_cell` unchanged -- HH-A and HH-B in both
orientations with section 3.2's lambda and the step-7 BIC choosing the orientation, their
oracle lambda alongside, the group lasso by ten-fold CV (and its 1-SE and oracle alternatives),
the BIC-selected FPCA pilot -- so the clean variant reproduces the notebook's own numbers.

**The two variants.** `clean` is `bench.design.simulate` exactly. `noisy` takes that same draw
-- the same curves, the same test curves, the same standard-normal errors and the same sigma --
and adds a perturbation eta to psi on the true null set T0 only, so the one thing that differs
between the two variants of a replication is that psi is no longer exactly zero on T0. Every
comparison between the variants is therefore paired. On each maximal interval [lo, hi] of T0,

    eta(t) = c * sum_{k=1}^{K} z_k sin(k pi (t - lo) / (hi - lo)) / k,    z_k ~ N(0, 1),

a random sine series that vanishes at both ends of the interval, so psi stays continuous
wherever it was continuous and no new jump is created at a boundary; c is set so the root mean
square of eta over T0 is exactly `NOISE["rms"]` (5% of max|psi| = 1) on every replication, and
only its shape is random. Each replication draws its own eta from a seed derived from its own.
Errors in the noisy variant are measured against the perturbed psi, which is the true
coefficient there; T0 keeps its meaning as the region the perturbation lives on.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

import numpy as np                                                              # noqa: E402
import pandas as pd                                                             # noqa: E402

from fts_research.bench import design, study                                    # noqa: E402
from fts_research.core.paths import cache_path                                  # noqa: E402
from fts_research.core.quadrature import quadrature_weights                     # noqa: E402
from fts_research.fdaSP import SBC                                              # noqa: E402
from fts_research.truncFLR import HH_continuous, HH_multi                       # noqa: E402

RECORDS = "basis_coef_ext_records.csv"
ARRAYS = "basis_coef_ext_arrays.npz"

VARIANTS = ("clean", "noisy")
NOISE = {"rms": 0.05, "n_terms": 10}

# The new estimators' settings. HH_continuous: the package defaults (m1 = m2 = 5 components per
# side, ten folds) on a candidate grid of every fourth grid point (spacing 0.02, so both true
# breaks 0.6 and 0.8 are candidates) rather than the default 20 points, which miss both.
# HH_multi: the package defaults, with the study's own lambda grid, pilot basis and FPC budget.
CONT = {"m1": 5, "m2": 5, "n_folds": 10, "theta_stride": 4}
MULTI = {"n_folds": 5, "max_depth": 4, "min_points": 8}

# Every coefficient function kept per replication: each existing method at the rule its own
# paper prescribes (at the orientation its step-7 BIC picked), the group lasso's 1-SE
# alternative, and the two new methods.
CURVE_KEYS = ("HH-A|paper", "HH-B|paper", "OGL|cv", "OGL|one_se", "FPCA|bic",
              "HH-cont|own", "HH-multi|own")
NEW_METHODS = ("HH-cont", "HH-multi")

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")


# ----------------------------------------------------------------------------- the designs --


def null_perturbation(grid, null, rng, rms=NOISE["rms"], n_terms=NOISE["n_terms"]):
    """A random sine series on each maximal interval of `null`, vanishing at both ends of it
    and zero off it, scaled to root mean square `rms` over `null`. Returns (N,)."""
    grid = np.asarray(grid, dtype=float)
    null = np.asarray(null, dtype=bool)
    eta = np.zeros_like(grid)
    k = np.arange(1, int(n_terms) + 1)
    for lo, hi in SBC.find_regions(null, grid):
        if hi <= lo:
            continue
        m = (grid >= lo - 1e-12) & (grid <= hi + 1e-12)
        s = (grid[m] - lo) / (hi - lo)
        z = rng.standard_normal(k.size)
        eta[m] = (z / k) @ np.sin(np.pi * np.outer(k, s))
    w = quadrature_weights(grid)
    current = np.sqrt((w[null] @ eta[null] ** 2) / w[null].sum())
    return eta * (rms / current)


def make_data(kind, target, rep, variant, cfg=None):
    """One replication of one cell, in one variant. Returns a dict shaped like
    `bench.design.simulate`'s, plus `eta` (the perturbation, zero in the clean variant) and
    `variant`."""
    cfg = {**study.CONFIG, **(cfg or {})}
    seed = study.seed_of(kind, target, rep)
    d = design.simulate(kind, target, n=cfg["n"], n_grid=cfg["n_grid"], n_basis=cfg["n_basis"],
                        decay=cfg["decay"], snr=cfg["snr"], n_test=cfg["n_test"], rng=seed)
    d["variant"] = variant
    grid = d["grid"]
    if variant == "clean":
        d["eta"] = np.zeros_like(grid)
        return d

    w = quadrature_weights(grid)
    # Recover the draw's own errors so the noisy variant reuses them exactly.
    eps = (d["Y"] - (d["X"] * w[None, :]) @ d["psi"]) / d["sigma"]
    eps_test = (d["Y_test"] - (d["X_test"] * w[None, :]) @ d["psi"]) / d["sigma"]
    eta = null_perturbation(grid, d["null"], np.random.default_rng([seed, 1]))
    psi = d["psi"] + eta
    coef, proj = design.project(psi, d["Psi"], grid)
    d.update({
        "psi": psi, "eta": eta,
        "Y": (d["X"] * w[None, :]) @ psi + d["sigma"] * eps,
        "Y_test": (d["X_test"] * w[None, :]) @ psi + d["sigma"] * eps_test,
        "psi_proj": proj, "psi_coef": coef, "gap": float(w @ (psi - proj) ** 2),
        "scale": float(w @ psi ** 2), "scale_proj": float(w @ proj ** 2),
    })
    return d


# ------------------------------------------------------------------------ the new methods --


def run_new(data, rep, cfg):
    """Fit `HH_continuous` and `HH_multi` on one dataset. Returns records (list of dicts shaped
    like `bench.study.run_cell`'s), curves and supports (dicts keyed "method|own")."""
    X, Y, grid = data["X"], data["Y"], data["grid"]
    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda_hh"])[1:]
    window = data["window"]
    full = np.ones(grid.size, dtype=bool)
    records, curves, supports = [], {}, {}

    def add(method, a_hat, b_hat, secs, **extra):
        rec = {"basis": data["kind"], "target": data["target"], "rep": rep, "method": method,
               "tuning": "own", "orientation": "na", "lam": np.nan, "bic": np.nan,
               "seconds": secs, "lam_failed": False, "bic_pick": True, **extra}
        # Neither returns exact zeros, so the support is the whole grid; both coefficient
        # functions are integrated over the whole grid, as each package does itself.
        rec.update(study.score(a_hat, b_hat, full, data))
        records.append(rec)
        curves[f"{method}|own"] = np.asarray(b_hat, dtype=float)
        supports[f"{method}|own"] = full

    # --- HH_continuous ------------------------------------------------------------------
    t0 = time.perf_counter()
    thetas = grid[CONT["theta_stride"]:-1:CONT["theta_stride"]]
    res = HH_continuous.fit(X, Y, grid, m1=CONT["m1"], m2=CONT["m2"], thetas=thetas,
                            n_folds=CONT["n_folds"], rng=rep)
    th = float(res["theta_hat"])
    add("HH-cont", res["a_hat"], res["b_hat"], time.perf_counter() - t0, m=CONT["m1"],
        theta_hat=th, n_bp=1, breakpoints=f"{th:.4f}",
        bp_err=(th - window[0]) if window is not None else np.nan)

    # --- HH_multi -----------------------------------------------------------------------
    t0 = time.perf_counter()
    r = HH_multi.binary_segment(X, Y, grid, lambdas, n_basis=cfg["pilot_basis"],
                                n_components=cfg["n_components"], n_fourier=cfg["n_fourier"],
                                n_folds=MULTI["n_folds"], max_depth=MULTI["max_depth"],
                                min_points=MULTI["min_points"], rng=rep)
    bps = [float(b) for b in r["breakpoints"]]
    near = (min(bps, key=lambda b: abs(b - window[0])) - window[0]
            if (window is not None and bps) else np.nan)
    add("HH-multi", r["a_hat"], r["b_hat"], time.perf_counter() - t0, m=int(r["m_hat"]),
        theta_hat=np.nan, n_bp=len(bps), breakpoints=";".join(f"{b:.4f}" for b in bps),
        bp_err=near, n_contests=len(r["contests"]))
    return records, curves, supports


# ------------------------------------------------------------------------ one replication --


def task(args):
    """Every method on one replication of one cell in one variant. Returns the records, the
    kept coefficient functions and support masks (arrays in `CURVE_KEYS` order), and psi."""
    kind, target, rep, variant, cfg = args
    cfg = {**study.CONFIG, **(cfg or {})}
    data = make_data(kind, target, rep, variant, cfg)

    # `run_cell`'s supports are already resolved to the orientation each truncation method's
    # own BIC picked, which is the fit the estimator returns.
    records, curves, supports = study.run_cell(kind, target, rep, cfg, data=data)
    picked = {f"{r['method']}|{r['tuning']}": r["orientation"] for r in records if r["bic_pick"]}
    new_records, new_curves, new_supports = run_new(data, rep, cfg)
    curves = {**{k: curves[f"{k}|{picked[k]}"] for k in picked}, **new_curves}
    supports = {**supports, **new_supports}

    for rec in records + new_records:
        rec["variant"] = variant
    C = np.stack([curves[k] for k in CURVE_KEYS]).astype(np.float32)
    S = np.stack([supports[k] for k in CURVE_KEYS])
    return records + new_records, C, S, data["psi"], data["eta"]


# ------------------------------------------------------------------------------ the study ---


def run(n_rep=None, n_jobs=None, verbose=True):
    """Every (variant, basis, target, replication). Returns the records DataFrame and a dict of
    arrays indexed [variant, basis, target, rep, (curve key,) grid point]."""
    cfg = dict(study.CONFIG)
    n_rep = int(n_rep or cfg["n_rep"])
    targets = list(design.TARGETS)
    tasks = [(k, t, r, v, None) for v in VARIANTS for k in design.BASIS_KINDS for t in targets
             for r in range(n_rep)]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 2)

    shape = (len(VARIANTS), len(design.BASIS_KINDS), len(targets), n_rep)
    n_grid = cfg["n_grid"]
    curves = np.full(shape + (len(CURVE_KEYS), n_grid), np.nan, dtype=np.float32)
    supports = np.zeros(shape + (len(CURVE_KEYS), n_grid), dtype=bool)
    psi = np.zeros(shape + (n_grid,))
    eta = np.zeros(shape + (n_grid,))
    rows = []

    env = {v: os.environ.get(v) for v in _THREAD_VARS}
    os.environ.update({v: "1" for v in _THREAD_VARS})
    started = time.time()
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for done, (tk, out) in enumerate(zip(tasks, pool.map(task, tasks, chunksize=1)), 1):
                kind, target, rep, variant, _ = tk
                idx = (VARIANTS.index(variant), design.BASIS_KINDS.index(kind),
                       targets.index(target), rep)
                recs, C, S, p, e = out
                rows.extend(recs)
                curves[idx], supports[idx], psi[idx], eta[idx] = C, S, p, e
                if verbose and (done % 20 == 0 or done == len(tasks)):
                    print(f"  {done}/{len(tasks)}  [{time.time() - started:6.0f}s]", flush=True)
    finally:
        for v, old in env.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old

    arrays = {"curves": curves, "supports": supports, "psi": psi, "eta": eta,
              "grid": np.linspace(0.0, 1.0, n_grid), "curve_keys": np.array(CURVE_KEYS),
              "variants": np.array(VARIANTS), "bases": np.array(design.BASIS_KINDS),
              "targets": np.array(targets)}
    return pd.DataFrame(rows), arrays


def cached(refresh=False, **kwargs):
    """The study, from `artifacts/cache/` if both files are there, run and cached if not.
    Returns (records DataFrame, arrays dict)."""
    rpath, apath = cache_path(RECORDS), cache_path(ARRAYS)
    if os.path.exists(rpath) and os.path.exists(apath) and not refresh:
        with np.load(apath) as z:
            return pd.read_csv(rpath), {k: z[k] for k in z.files}
    df, arrays = run(**kwargs)
    df.to_csv(rpath, index=False)
    np.savez_compressed(apath, **arrays)
    return df, arrays


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--quick", action="store_true", help="3 replications, not cached")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    if args.quick:
        df, arrays = run(n_rep=3, n_jobs=args.jobs)
        print(df.groupby(["variant", "target", "method", "tuning"])["rise"].median()
              .unstack(["variant"]).to_string())
    else:
        df, _ = cached(refresh=args.refresh, n_jobs=args.jobs)
        print(f"{len(df)} records")
