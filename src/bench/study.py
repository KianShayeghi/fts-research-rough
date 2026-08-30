"""Fits all three estimators on every (basis, target) cell and records what each recovered.

Nine cells, three bases by three coefficient functions, are laid out by `bench.design`. On
each replication of each cell every method sees the same dataset, so every comparison here is
paired and it is the distribution of the difference that decides whether a gap is real.

Three things are done to keep the comparison fair rather than merely simultaneous.

*Orientation is estimated, not given.* The truncation model has to be told which end of the
domain carries the signal: `truncFLR.HH.fit` looks for a window [0, theta] and `fit_mirror`
for a window [u, 1]. Handing it the orientation that matches the truth would be information
the group lasso is never given, so both orientations are fitted and the one with the smaller
step-7 BIC is kept. `bic_pick` marks it, and how often that agrees with the truth is itself a
result.

*Tuning is separated from estimation.* Every method is run twice: once under the rule its own
paper prescribes -- section 3.2 for methods A and B, ten-fold cross-validation for the group
lasso -- and once at the lambda on its own grid that minimises the realised error. The second
is not an estimator, it is an upper bound on what any selection rule could have delivered from
that path, and the gap between the two is the cost of the rule rather than of the estimator.

*The theta scan is shared across lambda.* Both truncFLR objectives are (residual sum of
squares)(theta) + n lambda theta^2, and only the second term moves with lambda, so the
expensive scan is run once with the penalty switched off and every candidate lambda then
re-minimises the cached curve. `hh_b_at` and `hh_a_at` reproduce `HH.fit` and `HH_A.fit`
exactly at the selected lambda, which `tests/test_bench.py` asserts to the last bit.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from fdaSP import SBC, SBC_sim
from fdaSP.empirical import repo_root
from truncFLR import HH, HH_A

from . import design

__all__ = [
    "CONFIG", "ARTIFACT", "METHOD_ORDER", "relative_ise", "null_metrics", "endpoint_error",
    "prediction_mse", "implied_endpoint", "interior", "hh_b_state", "hh_b_at",
    "hh_a_state", "hh_a_at",
    "ogl_run", "score", "run_cell", "run_study", "cached_study", "artifact_path",
    "summarise", "seed_of", "PRESCRIBED", "CURVES_ARTIFACT", "SHOWCASE", "cached_curves",
    "SUPPORT_ARTIFACT", "cached_support",
]

# The whole design in one place. `run_cell` takes overrides, so a notebook can shrink it for
# a quick pass without editing anything here.
CONFIG = {
    "n": 200,               # curves per replication
    "n_grid": 201,          # sampling grid, and so the resolution of the theta scan
    "n_basis": 25,          # basis functions the curves are expanded in
    "decay": 6.0,           # coefficient variance decay, matched across the three bases
    "snr": 20.0,            # var(signal) / sigma^2
    "n_test": 500,          # held-out curves for prediction error
    "n_rep": 40,            # replications per cell
    "n_lambda_hh": 120,     # truncFLR penalty grid, (0, 1]
    "pilot_basis": 20,      # B-spline basis for the truncFLR pilot
    "n_components": 10,     # FPC budget, and so the method-A BIC range 2..10
    "n_fourier": 1,         # sin/cos pairs in the step-3 parametric target b_simp
    "ogl_basis": 15,        # B-spline basis the group lasso expands psi in
    "order": 4,             # spline order, and so the group size
    "n_folds": 10,          # cross-validation folds
    "n_lambda_ogl": 30,     # penalties on the group lasso path
}

ARTIFACT = "basis_target_study.csv"
CURVES_ARTIFACT = "basis_target_curves.csv"
SUPPORT_ARTIFACT = "basis_target_support.csv"

# The fitted coefficient functions worth keeping for the single-replication figure: each
# method under the rule its own paper prescribes, the group lasso's one-standard-error
# alternative, and every oracle, so the picture of a fit and the cost of its tuning can be
# read off the same panel.
SHOWCASE = (("HH-A", "paper"), ("HH-B", "paper"), ("OGL", "cv"), ("OGL", "one_se"),
            ("FPCA", "bic"), ("HH-A", "oracle"), ("HH-B", "oracle"), ("OGL", "oracle"))

METHOD_ORDER = ["HH-A", "HH-B", "OGL", "FPCA"]

# The selection rule each method's own paper prescribes: section 3.2 for the truncation
# methods, cross-validation for the group lasso, BIC over components for the untruncated
# pilot. `oracle` and `one_se` are the alternatives held against these.
PRESCRIBED = ("paper", "cv", "bic")

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")


# ------------------------------------------------------------------------------ metrics ---


def relative_ise(beta_hat, psi, grid, scale, mask=None):
    """Integrated squared error of `beta_hat` against `psi`, as a fraction of `scale`.

    Dividing by scale = int psi^2 makes the number comparable across coefficient functions of
    different size: 1.0 is the error of the estimate that gives up and returns zero, so
    anything at or above 1 has learned nothing. `fdaSP.SBC_sim.ise` does the integration, so
    the split over a region and its complement adds back to the whole even when the region is
    a union of intervals. Returns a float, or NaN where `mask` is empty.

    PARAMETERS
    scale : float, the normalising int psi^2 (or int (P psi)^2 against the identified part);
    mask  : (N,) boolean array or None, region to restrict the integral to.
    """
    return SBC_sim.ise(beta_hat, psi, grid, mask) / scale


def null_metrics(support, null_true, grid):
    """How much of the true null set an estimate switched off, and how much of the support it
    kept. Returns a dict of `null_sens`, `null_spec` and `null_jaccard`.

    Every set is measured in Lebesgue measure by the trapezium weights rather than by counting
    grid points, so the numbers do not move with `n_grid`. `null_sens` is the fraction of the
    true null set the estimate also calls null, `null_spec` the fraction of the true support it
    keeps, `null_jaccard` the overlap of the two null sets over their union -- which is 1 only
    for an exactly recovered null set, and NaN where neither set has any -- and `null_measure`
    the total length of the estimated null set, which is 0 exactly when the estimate switched
    nothing off.

    PARAMETERS
    support   : (N,) boolean array, grid points the estimate does *not* set to zero;
    null_true : (N,) boolean array, the true null set T0.
    """
    w = design.quad_weights(grid)
    support = np.asarray(support, dtype=bool)
    t0 = np.asarray(null_true, dtype=bool)
    hat0 = ~support
    inter, union = w[hat0 & t0].sum(), w[hat0 | t0].sum()
    size0, size1 = w[t0].sum(), w[~t0].sum()
    return {
        "null_sens": float(inter / size0) if size0 > 0 else np.nan,
        "null_spec": float(w[support & ~t0].sum() / size1) if size1 > 0 else np.nan,
        "null_jaccard": float(inter / union) if union > 0 else np.nan,
        "null_measure": float(w[hat0].sum()),
    }


def implied_endpoint(support, grid):
    """The left endpoint an estimated support set implies, i.e. the left edge of the last
    stretch of support it leaves standing. Returns a float, NaN where there is no support.

    Where the truth is a single window [u, 1], this is the quantity the truncation model
    reports directly and the penalty only implies, and `fdaSP.SBC.find_regions` reads it off
    the mask the same way for both. An estimate that leaves the whole domain in the support
    therefore implies u_hat = 0, which is the honest reading of "it found no boundary".
    """
    regions = SBC.find_regions(np.asarray(support, dtype=bool), grid)
    return float(regions[-1][0]) if regions else np.nan


def endpoint_error(support, grid, window):
    """Signed error u_hat - u of the endpoint implied by an estimated support set.

    Signed rather than absolute, because the direction is the interesting part: a method that
    truncates systematically early is making a different mistake from one that is merely
    imprecise. Returns a float, NaN where `window` is None or the estimate has no support.

    PARAMETERS
    window : (u, v) tuple or None, the true informative window; None where the target has no
             single such window and the quantity is not defined.
    """
    if window is None:
        return np.nan
    u_hat = implied_endpoint(support, grid)
    return np.nan if np.isnan(u_hat) else float(u_hat - window[0])


def prediction_mse(a_hat, beta_hat, X, Y, grid, mask=None):
    """Mean squared error of the predictions a_hat + int beta_hat X on held-out curves.
    Returns a float.

    `mask` restricts the quadrature to the sub-grid the coefficient was fitted on. That
    matters for the truncated fits and not for the penalised one: integrating over the whole
    grid with beta_hat zeroed past the truncation point straddles the jump with a trapezium
    panel and adds a spurious half of beta_hat(theta) X(theta) h, the correction
    `truncFLR.HH_A.trunc_predict` makes.

    PARAMETERS
    mask : (N,) boolean array or None, the sub-grid to integrate over; the whole grid if None.
    """
    grid = np.asarray(grid, dtype=float)
    if mask is None:
        pred = a_hat + np.trapezoid(np.asarray(beta_hat) * X, grid, axis=1)
    else:
        mask = np.asarray(mask, dtype=bool)
        pred = a_hat + np.trapezoid(np.asarray(beta_hat)[mask] * X[:, mask], grid[mask], axis=1)
    return float(np.mean((np.asarray(Y) - pred) ** 2))


# ---------------------------------------------------------------------- truncFLR runners ---


def interior(grid):
    """The candidate endpoints both truncFLR theta scans use, i.e. `grid`'s interior points.
    Returns (Q,)."""
    grid = np.asarray(grid, dtype=float)
    return grid[(grid > grid[0]) & (grid < grid[-1])]


def _first_interior_min(v):
    """Index of the first interior local minimum of `v`, or of its global minimum where there
    is none -- the rule `HH.theta_methodB` and step 6' select theta_lambda by."""
    for k in range(1, len(v) - 1):
        if v[k] <= v[k - 1] and v[k] <= v[k + 1]:
            return k
    return int(np.argmin(v))


def _select(selector, X, Y, grid, a, b, fpc, lambdas, n_fourier, **kwargs):
    """Run one section 3.2 selection, falling back to the smallest candidate where it is
    degenerate. Returns lambda_hat (float) and whether the fallback fired (bool).

    `truncFLR.HH.DegenerateSelection` means P_b(lambda) is undefined at every candidate, so
    the criterion ranks nothing rather than ranking weakly. The study still has to produce a
    fit, so it takes the least-penalised candidate on the stated ground that the criterion
    supplied no reason to penalise at all -- a convention of this comparison, not of the
    paper, and one whose frequency is reported alongside the results it affects.
    """
    try:
        lam_hat, _ = selector(X, Y, grid, a, b, fpc, None, lambdas,
                              n_fourier=n_fourier, m_tuning=None, w=None, **kwargs)
        return float(lam_hat), False
    except HH.DegenerateSelection:
        return float(np.min(lambdas)), True


def _refit_b(theta, grid, X, Y, n_basis, n_components):
    """`HH.ab_methodB(refit=True)`, also returning the component count its pilot chose.

    Identical arithmetic to `ab_methodB`, which discards the count; the orientation choice
    needs it, since the step-7 BIC charges for it. Returns a_hat, b_hat and m.
    """
    interval = np.asarray(grid) <= theta
    grid_i = grid[interval]
    n_basis_i = min(n_basis, max(len(grid_i) - 1, HH.PILOT_ORDER))
    n_components_i = max(2, min(n_components, len(grid_i) - 1, len(Y) - 1))
    a_hat, b_sub, fpc = HH.pilot_fpca(X[:, interval], Y, grid_i, n_basis=n_basis_i,
                                      n_components=n_components_i)
    b_hat = np.zeros_like(np.asarray(grid, dtype=float))
    b_hat[interval] = b_sub
    return a_hat, b_hat, int(fpc["m"])


def hh_b_state(X, Y, grid, lambdas, mirror=False, n_basis=20, n_components=10, n_fourier=1):
    """Everything method B needs before a lambda is chosen: the pilot, the section 3.2
    selection, and the theta scan with the penalty switched off. Returns a dict `hh_b_at`
    consumes.

    The scan is the expensive part and does not depend on lambda, since `HH.theta_methodB`
    minimises RSS(theta) + n lambda theta^2 and only the penalty moves. Running it once at
    zero penalty leaves `hh_b_at` adding a parabola to a cached curve.

    PARAMETERS
    mirror : bool, fit the window [u, 1] by reflecting the curves, as `HH.fit_mirror` does,
             rather than the paper's [0, theta].
    """
    grid = np.asarray(grid, dtype=float)
    if mirror:
        HH.check_mirrorable(grid)
        X = X[:, ::-1]
    a, b, fpc = HH.pilot_fpca(X, Y, grid, n_basis=n_basis, n_components=n_components)
    lam_hat, failed = _select(HH.lambda_select, X, Y, grid, a, b, fpc, lambdas, n_fourier)
    _, theta_grid, rss = HH.theta_methodB(X, Y, grid, a, b, 0.0)
    return {"method": "HH-B", "X": X, "Y": Y, "grid": grid, "mirror": mirror,
            "a_pilot": a, "b_pilot": b, "fpc": fpc, "lam_hat": lam_hat,
            "lam_failed": failed, "theta_grid": theta_grid, "rss": rss,
            "n_basis": n_basis, "n_components": n_components, "cache": {}}


def hh_b_at(state, lam):
    """Method B's fit at one lambda, from the cached scan in `state`.

    Reproduces `HH.fit(..., refit=True)` bit for bit at `state["lam_hat"]`: the penalised
    objective is assembled with the same arithmetic and minimised by the same first-interior
    -minimum rule, and the refit is `ab_methodB`'s. Returns a dict with theta_hat, a_hat,
    b_hat (in the original orientation), m, the step-7 bic, the support mask and a `key`
    identifying the fit, so repeated lambdas cost nothing.
    """
    X, Y, grid = state["X"], state["Y"], state["grid"]
    n = len(Y)
    idx = _first_interior_min(state["rss"] + n * lam * state["theta_grid"] ** 2)
    if idx not in state["cache"]:
        theta = state["theta_grid"][idx]
        a_hat, b_hat, m = _refit_b(theta, grid, X, Y, state["n_basis"], state["n_components"])
        bic = HH_A.methodA_bic(a_hat, b_hat, X, Y, grid, m, theta)
        support = grid <= theta
        if state["mirror"]:
            b_hat, support = b_hat[::-1], support[::-1]
        state["cache"][idx] = {"theta_hat": float(theta), "a_hat": float(a_hat),
                               "b_hat": b_hat, "m": m, "bic": float(bic),
                               "support": support, "key": int(idx)}
    return state["cache"][idx]


def hh_a_state(X, Y, grid, lambdas, mirror=False, n_basis=20, n_components=10, n_fourier=1):
    """Everything method A needs before a lambda is chosen: the pilot, the section 3.2
    selection, and one theta scan per candidate component count at zero penalty. Returns a
    dict `hh_a_at` consumes.

    `HH_A.theta_methodA_multi` shares the truncated principal components analysis at each
    theta across every m, so the whole family of scans costs one pass.

    PARAMETERS
    mirror : bool, fit the window [u, 1] by reflecting the curves, as `HH_A.fit_mirror` does.
    """
    grid = np.asarray(grid, dtype=float)
    if mirror:
        HH_A.check_mirrorable(grid)
        X = X[:, ::-1]
    m_range = [int(m) for m in range(2, n_components + 1)]
    a, b, fpc = HH.pilot_fpca(X, Y, grid, n_basis=n_basis, n_components=n_components)
    lam_hat, failed = _select(HH_A.lambda_select, X, Y, grid, a, b, fpc, lambdas, n_fourier,
                              n_components=n_components)
    scans = HH_A.theta_methodA_multi(X, Y, grid, 0.0, m_range)
    return {"method": "HH-A", "X": X, "Y": Y, "grid": grid, "mirror": mirror,
            "a_pilot": a, "b_pilot": b, "fpc": fpc, "lam_hat": lam_hat,
            "lam_failed": failed, "theta_grid": interior(grid),
            "rss": {m: scans[m][1] for m in m_range}, "m_range": m_range, "cache": {}}


def hh_a_at(state, lam):
    """Method A's fit at one lambda, from the cached scans in `state`.

    Reproduces `HH_A.fit(..., strict=False)` bit for bit at `state["lam_hat"]`: each candidate
    m minimises its own cached scan plus the same penalty, the winning theta is refitted by
    `ab_methodA`, and the step-7 BIC picks m the same way and in the same order. Returns a
    dict shaped like `hh_b_at`'s, with `m` the selected component count.
    """
    X, Y, grid = state["X"], state["Y"], state["grid"]
    n = len(Y)
    pen = n * lam * state["theta_grid"] ** 2
    best = None
    for m in state["m_range"]:
        idx = int(np.argmin(state["rss"][m] + pen))
        key = (m, idx)
        if key not in state["cache"]:
            theta = state["theta_grid"][idx]
            a_hat, b_hat = HH_A.ab_methodA(theta, grid, X, Y, n_components=m)
            bic = HH_A.methodA_bic(a_hat, b_hat, X, Y, grid, m, theta)
            support = grid <= theta
            if state["mirror"]:
                b_hat, support = b_hat[::-1], support[::-1]
            state["cache"][key] = {"theta_hat": float(theta), "a_hat": float(a_hat),
                                   "b_hat": b_hat, "m": m, "bic": float(bic),
                                   "support": support, "key": key}
        fit = state["cache"][key]
        if best is None or fit["bic"] < best["bic"]:
            best = fit
    return best


# --------------------------------------------------------------------------- OGL runner ---


def ogl_run(X, Y, grid, n_basis=15, order=4, n_folds=10, n_lambda=30, rng=0):
    """The locally sparse estimator, with the three lambda rules held against one solver run.
    Returns a dict carrying the cross-validation output `cv`, the fitted `path`, and the path
    indices `j_cv` and `j_1se` the two selection rules choose.

    The response is centred before fitting, as `fdaSP.SBC` has no intercept, and the training
    mean is added back for prediction. One cross-validation serves both selection rules: the
    one-standard-error lambda is read off the same curve `cv_fit` returns, which is what
    `cv_fit(one_se=True)` would have done on the same folds. Every coefficient function is
    then taken from `fit_path` over those same penalties rather than refitted, so the rules
    differ only in which lambda they pick and not in which iterate the ADMM stopped at --
    `cv_fit` refits cold while the path warm-starts, and the two agree only to solver
    tolerance, which is enough to put an oracle behind a rule it cannot lose to.

    PARAMETERS
    rng : int, seed for the fold assignment.
    """
    y = np.asarray(Y, dtype=float) - float(np.mean(Y))
    cv = SBC.cv_fit(X, y, grid, n_basis=n_basis, order=order, n_folds=n_folds,
                    n_lambda=n_lambda, rng=rng)
    path = SBC.fit_path(X, y, grid, lambdas=cv["lambdas"], n_basis=n_basis, order=order)
    j_cv = int(np.argmin(cv["cv_mean"]))
    bound = cv["cv_mean"][j_cv] + cv["cv_se"][j_cv]
    j_1se = int(np.argmax(cv["cv_mean"] <= bound))
    return {"cv": cv, "path": path, "j_cv": j_cv, "j_1se": j_1se,
            "lambdas": path["lambdas"], "a_hat": float(np.mean(Y))}


# --------------------------------------------------------------------------- scoring -----


def score(a_hat, beta_hat, support, data, predict_mask=None):
    """Every metric one fitted coefficient function earns on one dataset. Returns a dict.

    `rise` is against the true psi and `rise_id` against its projection onto the span of the
    curves, which is the only part of psi the data identify; the two differ by how much of the
    answer the estimator's own basis supplied rather than learned. `u_err` is signed, so its
    sign says which way an endpoint was missed. `mse_test` is divided by sigma^2, so 1.0 is
    the noise floor no predictor can beat.

    PARAMETERS
    support      : (N,) boolean array, grid points the estimate does not set to zero;
    data         : dict from `bench.design.simulate`;
    predict_mask : (N,) boolean array or None, sub-grid to integrate over when predicting.
    """
    grid, psi = data["grid"], data["psi"]
    out = {
        "rise": relative_ise(beta_hat, psi, grid, data["scale"]),
        "rise_T0": relative_ise(beta_hat, psi, grid, data["scale"], data["null"]),
        "rise_T1": relative_ise(beta_hat, psi, grid, data["scale"], ~data["null"]),
        "rise_id": relative_ise(beta_hat, data["psi_proj"], grid, data["scale_proj"]),
        "u_hat": implied_endpoint(support, grid),
        "u_err": endpoint_error(support, grid, data["window"]),
        "mse_test": prediction_mse(a_hat, beta_hat, data["X_test"], data["Y_test"], grid,
                                   predict_mask) / data["sigma"] ** 2,
        "n_support_regions": len(SBC.find_regions(np.asarray(support, dtype=bool), grid)),
    }
    out.update(null_metrics(support, data["null"], grid))
    return out


def seed_of(kind, target, rep, base=20250829):
    """A distinct, reproducible seed for each (basis, target, replication). Returns an int.

    Every method sees the same dataset within a replication, which is what makes the
    comparison paired; different cells necessarily see different curves, since the basis is
    what the cell varies.

    PARAMETERS
    base : int, offset the whole study can be moved by to draw a fresh set of datasets.
    """
    i = design.BASIS_KINDS.index(kind)
    j = list(design.TARGETS).index(target)
    return int(base) + 1000 * (len(design.TARGETS) * i + j) + int(rep)


# ------------------------------------------------------------------------ one replication --


def _oracle(at_fn, state, lambdas, data):
    """The smallest relative ISE any lambda on `lambdas` could have produced, and the fit
    achieving it. Distinct lambdas often select the same theta, so each fit is scored once."""
    best, seen = None, {}
    for lam in lambdas:
        fit = at_fn(state, lam)
        key = fit["key"]
        if key not in seen:
            seen[key] = relative_ise(fit["b_hat"], data["psi"], data["grid"], data["scale"])
            if best is None or seen[key] < best[0]:
                best = (seen[key], float(lam), fit)
    return best[1], best[2]


def run_cell(kind, target, rep, cfg=None):
    """Fit every method on one replication of one cell.

    Returns a list of records, one per (method, tuning, orientation); a dict of the fitted
    coefficient functions keyed "method|tuning|orientation", which the driver discards and the
    single-replication figures keep; and a dict of the support masks keyed "method|tuning",
    already resolved to the orientation each truncation method's own BIC selected, since that
    is the fit the estimator actually returns.

    PARAMETERS
    kind   : str, one of `bench.design.BASIS_KINDS`;
    target : str, one of `bench.design.TARGETS`;
    rep    : int, replication index, which sets the seed through `seed_of`;
    cfg    : dict or None, overrides applied on top of `CONFIG`.
    """
    cfg = {**CONFIG, **(cfg or {})}
    data = design.simulate(kind, target, n=cfg["n"], n_grid=cfg["n_grid"],
                           n_basis=cfg["n_basis"], decay=cfg["decay"], snr=cfg["snr"],
                           n_test=cfg["n_test"], rng=seed_of(kind, target, rep))
    X, Y, grid = data["X"], data["Y"], data["grid"]
    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda_hh"])[1:]
    common = {"basis": kind, "target": target, "rep": rep}

    records, curves, masks = [], {}, {}

    def add(method, tuning, orientation, a_hat, beta, support, lam, m, bic, secs,
            predict_mask=None, lam_failed=False):
        rec = dict(common, method=method, tuning=tuning, orientation=orientation,
                   lam=lam, m=m, bic=bic, seconds=secs, lam_failed=lam_failed)
        rec.update(score(a_hat, beta, support, data, predict_mask))
        records.append(rec)
        key = f"{method}|{tuning}|{orientation}"
        curves[key] = np.asarray(beta, dtype=float)
        masks[key] = np.asarray(support, dtype=bool)
        return rec

    # --- the two truncation methods, in both orientations -------------------------------
    pilot = None
    for method, state_fn, at_fn in (("HH-B", hh_b_state, hh_b_at),
                                    ("HH-A", hh_a_state, hh_a_at)):
        paper_rows = {}
        for orientation, mirror in (("left", False), ("right", True)):
            t0 = time.perf_counter()
            state = state_fn(X, Y, grid, lambdas, mirror=mirror, n_basis=cfg["pilot_basis"],
                             n_components=cfg["n_components"], n_fourier=cfg["n_fourier"])
            fit = at_fn(state, state["lam_hat"])
            secs = time.perf_counter() - t0
            paper_rows[orientation] = add(method, "paper", orientation, fit["a_hat"],
                                          fit["b_hat"], fit["support"], state["lam_hat"],
                                          fit["m"], fit["bic"], secs, fit["support"],
                                          lam_failed=state["lam_failed"])
            t0 = time.perf_counter()
            lam_o, fit_o = _oracle(at_fn, state, lambdas, data)
            add(method, "oracle", orientation, fit_o["a_hat"], fit_o["b_hat"],
                fit_o["support"], lam_o, fit_o["m"], fit_o["bic"],
                time.perf_counter() - t0, fit_o["support"])
            if method == "HH-B" and not mirror:
                pilot = state           # the untruncated pilot, i.e. the no-sparsity baseline

        chosen = min(paper_rows, key=lambda o: paper_rows[o]["bic"])
        for rec in records:
            if rec["method"] == method:
                rec["bic_pick"] = rec["orientation"] == chosen

    # --- the locally sparse estimator ---------------------------------------------------
    t0 = time.perf_counter()
    ogl = ogl_run(X, Y, grid, n_basis=cfg["ogl_basis"], order=cfg["order"],
                  n_folds=cfg["n_folds"], n_lambda=cfg["n_lambda_ogl"], rng=rep)
    secs = time.perf_counter() - t0
    path, lams = ogl["path"], ogl["lambdas"]
    rises = [relative_ise(b, data["psi"], grid, data["scale"]) for b in path["betas"]]
    for tuning, j, spent in (("cv", ogl["j_cv"], secs),
                             ("one_se", ogl["j_1se"], 0.0),
                             ("oracle", int(np.argmin(rises)), 0.0)):
        add("OGL", tuning, "na", ogl["a_hat"], path["betas"][j], path["supports"][j],
            float(lams[j]), np.nan, np.nan, spent)

    # --- the untruncated pilot, i.e. what truncation and penalisation are both buying ----
    add("FPCA", "bic", "na", float(pilot["a_pilot"]), pilot["b_pilot"],
        np.ones(grid.size, dtype=bool), np.nan, int(pilot["fpc"]["m"]), np.nan, 0.0)

    for rec in records:
        rec.setdefault("bic_pick", True)

    picked = {(r["method"], r["tuning"]): r["orientation"] for r in records if r["bic_pick"]}
    supports = {f"{method}|{tuning}": masks[f"{method}|{tuning}|{orientation}"]
                for (method, tuning), orientation in picked.items()}
    return records, curves, supports


# ------------------------------------------------------------------------- the whole study --


def _task(args):
    """One `run_cell` call in a worker process, returning the records and the support masks.

    The fitted curves are dropped here: `cached_curves` refits the one replication it shows,
    which is cheaper than shipping 12 arrays back from every task in the study.
    """
    kind, target, rep, cfg = args
    records, _curves, supports = run_cell(kind, target, rep, cfg)
    return records, {k: np.packbits(v) for k, v in supports.items()}


def run_study(cells=None, n_rep=None, cfg=None, n_jobs=None, verbose=True):
    """Run every cell over every replication. Returns two tidy DataFrames: the records,
    one row per
    (basis, target, replication, method, tuning, orientation); and the support frequencies,
    one row per (basis, target, method, tuning, grid point), holding the fraction of
    replications in which that point was left in the support.

    Replications are independent, so they are spread over worker processes. Each worker is
    given a single BLAS thread: the linear algebra here is small enough that its own threading
    only contends with the parallelism at this level. Windows spawns rather than forks, so a
    *script* calling this needs its call under `if __name__ == "__main__":` or the workers
    re-import it and try to start a pool of their own; a notebook needs nothing, its `__main__`
    having no file to re-import. Passing `n_jobs=1` sidesteps the question entirely.

    PARAMETERS
    cells   : list of (basis, target) pairs, or None for all nine;
    n_rep   : int, replications per cell, overriding `CONFIG`;
    cfg     : dict or None, overrides applied on top of `CONFIG`;
    n_jobs  : int or None, worker processes; 1 runs in this process, None picks a default;
    verbose : bool, print progress as cells finish.
    """
    cfg = {**CONFIG, **(cfg or {})}
    if n_rep is not None:
        cfg["n_rep"] = int(n_rep)
    if cells is None:
        cells = [(k, t) for k in design.BASIS_KINDS for t in design.TARGETS]
    tasks = [(k, t, r, cfg) for k, t in cells for r in range(cfg["n_rep"])]

    if n_jobs is None:
        n_jobs = max(1, min(8, (os.cpu_count() or 2) - 1))

    n_grid = cfg["n_grid"]
    rows, done = [], 0
    tally = {}

    def collect(kind, target, records, packed):
        rows.extend(records)
        for label, bits in packed.items():
            key = (kind, target, label)
            mask = np.unpackbits(bits, count=n_grid).astype(float)
            if key in tally:
                tally[key] += mask
            else:
                tally[key] = mask

    if n_jobs == 1:
        for task in tasks:
            records, packed = _task(task)
            collect(task[0], task[1], records, packed)
            done += 1
            if verbose and done % 10 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
    else:
        env = {v: os.environ.get(v) for v in _THREAD_VARS}
        os.environ.update({v: "1" for v in _THREAD_VARS})   # inherited by spawned workers
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                for task, (recs, packed) in zip(tasks, pool.map(_task, tasks, chunksize=1)):
                    collect(task[0], task[1], recs, packed)
                    done += 1
                    if verbose and done % 10 == 0:
                        print(f"  {done}/{len(tasks)}", flush=True)
        finally:
            for v, old in env.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old

    grid = np.linspace(0.0, 1.0, n_grid)
    freq = pd.concat(
        [pd.DataFrame({"basis": kind, "target": target,
                       "method": label.split("|")[0], "tuning": label.split("|")[1],
                       "t": grid, "in_support": total / cfg["n_rep"]})
         for (kind, target, label), total in tally.items()],
        ignore_index=True)
    return pd.DataFrame(rows), freq


def artifact_path(name=ARTIFACT):
    """Absolute path of a cached study CSV under `artifacts/`, whether or not it exists yet."""
    return os.path.join(repo_root(), "artifacts", name)


def cached_study(name=ARTIFACT, support_name=SUPPORT_ARTIFACT, refresh=False, **kwargs):
    """The study, read from `artifacts/` if it is cached there and run and cached if not.
    Returns the records DataFrame.

    The full study takes minutes, so the notebook reads it back rather than re-running it, in
    the same way `fdaSP.r_reference` caches the R comparison. One run produces both caches, so
    the records and the support frequencies always describe the same fits.

    PARAMETERS
    name         : str, file name under `artifacts/` for the records;
    support_name : str, file name under `artifacts/` for the support frequencies;
    refresh      : bool, re-run and overwrite even where the cache exists;
    **kwargs     : forwarded to `run_study`.
    """
    path, support_path = artifact_path(name), artifact_path(support_name)
    if os.path.exists(path) and os.path.exists(support_path) and not refresh:
        return pd.read_csv(path)
    df, freq = run_study(**kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    freq.to_csv(support_path, index=False)
    return df


def cached_support(name=SUPPORT_ARTIFACT, refresh=False, **kwargs):
    """How often each grid point was left in the support, by cell and method. Returns a long
    DataFrame with `basis`, `target`, `method`, `tuning`, `t` and `in_support`.

    Written by the same run as `cached_study`, at the orientation each truncation method's own
    BIC selected. A value of 0 means the method called that point null on every replication and
    1 that it never did, so the whole null-set comparison can be drawn as a picture rather than
    summarised into one number.

    PARAMETERS
    name     : str, file name under `artifacts/`;
    refresh  : bool, re-run the study and overwrite;
    **kwargs : forwarded to `cached_study`.
    """
    path = artifact_path(name)
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    cached_study(refresh=True, **kwargs)
    return pd.read_csv(path)


def cached_curves(name=CURVES_ARTIFACT, refresh=False, rep=0, cfg=None):
    """The coefficient functions every method fitted on one replication of every cell, read
    from `artifacts/` if cached there and fitted and cached if not.

    Returns a long DataFrame with `basis`, `target`, `method`, `tuning`, `orientation`, `t`
    and `beta`, holding the `SHOWCASE` fits at the orientation each truncation method's own
    BIC selected. This is what the nine-panel figure draws, and caching it is what lets the
    notebook open without refitting anything.

    PARAMETERS
    name    : str, file name under `artifacts/`;
    refresh : bool, refit and overwrite even where the cache exists;
    rep     : int, which replication to show;
    cfg     : dict or None, overrides applied on top of `CONFIG`.
    """
    path = artifact_path(name)
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)

    merged = {**CONFIG, **(cfg or {})}
    grid = np.linspace(0.0, 1.0, merged["n_grid"])
    rows = []
    for kind in design.BASIS_KINDS:
        for target in design.TARGETS:
            records, curves, _supports = run_cell(kind, target, rep, cfg)
            picked = {r["method"]: r["orientation"] for r in records
                      if r["tuning"] in PRESCRIBED and r["bic_pick"]}
            for method, tuning in SHOWCASE:
                orientation = picked[method]
                beta = curves[f"{method}|{tuning}|{orientation}"]
                rows.append(pd.DataFrame({
                    "basis": kind, "target": target, "method": method, "tuning": tuning,
                    "orientation": orientation, "t": grid, "beta": beta}))
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return out


def summarise(df, metric="rise", tuning=PRESCRIBED):
    """Median of `metric` by cell and method, for the estimators as their own papers tune
    them. Returns a DataFrame indexed by (basis, target) with one column per method.

    The truncation methods are taken at the orientation their step-7 BIC selected, so each
    cell reports one number per method rather than one per orientation.

    PARAMETERS
    metric : str, column to summarise;
    tuning : str or tuple of str, which selection rules to keep; `PRESCRIBED` by default.
    """
    keep = (tuning,) if isinstance(tuning, str) else tuple(tuning)
    sub = df[df["tuning"].isin(keep) & df["bic_pick"]]
    out = sub.pivot_table(index=["basis", "target"], columns="method", values=metric,
                          aggfunc="median")
    cols = [c for c in METHOD_ORDER if c in out.columns]
    return out[cols]
