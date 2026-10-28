"""Power of the test of `beta = 0`, and the four estimators, at realistic signal strengths.

`window_dgp` makes jumps that carry on a share `kappa` of the previous session's late-window
slope. `notebooks/window_jump_dgp.ipynb` studies it at `kappa = 0.5` and `1`, where the
previous curve predicts 27% and 60% of the jump. The real-data notebooks find the own-curve
term empty, so the signal there is at most a few per cent. This module runs the same DGP at
`R^2` in {0, 0.01, 0.02, 0.05} (`window_dgp.kappa_for_r2`), and keeps `kappa` in
{0.25, 0.5, 1} as reference rows.

**Common random numbers.** Each (n, replication) is simulated once, at `kappa = 0`. The jump at
any other `kappa` is

    J_t(kappa) = J_t(0) + kappa T (bhat_{t-1} - 100 m),

which is exactly what `window_dgp.window_jump_sessions` returns from the same seed, because
`kappa` changes no draw (`jumps_at`). Every cell of a replication therefore shares its curves
and its noise, so differences between cells are paired.

**The tests** (`power_tests`), all of the regression of `J_t` on yesterday's curve with an
intercept, on `n` pairs:

    slope      J on bhat_{t-1}: the 1-df oracle, told the window and the shape of beta;
    M8         F test on the first 8 principal component scores: the notebook's oracle M;
    var95      F test on the first M scores, M the smallest holding 95% of the curves'
               variance, chosen from the curves alone;
    bic_naive  F test at the M in 1..10 that minimises BIC, referred to the F table;
    bic_perm   the same statistic, referred to its permutation distribution.

Under the null the jumps are i.i.d. and independent of the curves, and the curves are fixed
regressors, so the first three are exact F tests. `bic_naive` is not: it looks at the jumps to
choose M and then tests at the M that fitted them best, so it rejects too often. `bic_perm`
repairs it without giving up the data-driven M. The jumps are exchangeable under the null, so
permuting them against the curves and re-running *select-then-test* on each permutation gives
the procedure's exact null distribution, and `p = (1 + #{p*_b <= p}) / (B + 1)` is exact
(Phipson & Smyth, 2010).

The sample scores of centred curves are orthogonal, so the regression on the first M of them
is a running sum: `RSS_M = TSS - sum_{k <= M} (xi_k' J)^2 / (xi_k' xi_k)`. All ten nested fits,
for the data and for every permutation, cost one matrix product.

**The analytic guide** (`f_test_power`). With an intercept and Gaussian regressors, the F
statistic given the regressors is noncentral F with noncentrality
`lambda = gamma' X_c' X_c gamma / sigma^2`, and `X_c' X_c` is Wishart on `n - 1` degrees of
freedom, so `lambda = f^2 chi^2_{n-1}` with `f^2 = R^2 / (1 - R^2)`. The power is the
noncentral-F tail averaged over that chi-square, exact for the `slope` test. For a test on the
first M population components, the part of the signal outside them is independent Gaussian
noise, so `f^2 = s R^2 / (1 - s R^2)` with `s` the share of the signal inside
(`signal_share`, `effective_f2`). The `M8` test uses sample components, so this is a guide for
it, not its exact power.

**The estimators** (`fit_methods`) are those of section 6 of the notebook: the untruncated
principal component fit, with `M = 8` (the oracle) and with M by BIC; truncFLR methods A and
B, told the window ends at the close, with lambda by the paper's section 3.2; and the fdaSP
group lasso, told nothing, with lambda by ten-fold cross-validation and by its
one-standard-error rule. Each is scored against the continuous `beta` of `window_dgp`, and it
forecasts the next 63 jumps against the expanding-mean forecast of the null.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.paths import cache_path
from ..core.quadrature import quadrature_weights
from ..far.far_est import fpca
from ..fdaSP import SBC, SBC_sim
from ..truncFLR import HH, HH_A
from ..truncFLR.HH import DegenerateSelection
from . import basic_est as be
from . import window_dgp as wd

__all__ = ["CONFIG", "TESTS", "EXACT_TESTS", "METHODS", "POWER_ARTIFACT", "METHODS_ARTIFACT",
           "SUPPORT_ARTIFACT", "rng_of", "design_cells", "simulate", "jumps_at", "true_beta",
           "nested_fits", "power_tests", "f_test_power", "n_for_power", "signal_share",
           "effective_f2", "fit_methods", "score_methods", "power_replication",
           "methods_replication", "rejection_rates", "run_power", "run_methods", "cached_power",
           "cached_methods", "cached_methods_support", "mc_se"]

# The whole design in one place; every driver takes overrides on top of it.
CONFIG = {
    "n_bars": 390, "S0": 100.0, "sigma": 0.01, "T": 0.25, "nu": 2.5e-4,   # the GBM
    "start": 0.8,                          # the window [0.8, 1]
    "r2": (0.0, 0.01, 0.02, 0.05),         # the realistic cells, set by R^2
    "kappa_ref": (0.25, 0.5, 1.0),         # the reference cells, set by kappa
    "n_power": (62, 125, 250, 390, 500, 765, 1000),   # pairs (J_t, X_{t-1}) per dataset
    "n_rep_size": 1000,                    # replications at R^2 = 0
    "n_rep_power": 500,                    # replications in every other cell
    "m_oracle": 8,                         # the notebook's M, which holds 98% of the signal
    "m_max": 10,                           # BIC chooses M in 1..m_max
    "var_share": 0.95,                     # the curve-variance rule
    "n_perm": 999,                         # permutations behind bic_perm
    "level": 0.05,
    "n_methods": (62, 250, 500, 1000),     # the estimator comparison
    "n_rep_methods": 100,
    "n_test": 63,                          # held-out jumps, one quarter
    "n_lambda_hh": 120,                    # truncFLR penalty grid, as in bench and section 6
    "ogl_basis": 15, "n_folds": 10, "n_lambda_ogl": 30,
    "base": 20261026,                      # seeds every draw; move it for a fresh study
}

TESTS = ("slope", "M8", "var95", "bic_naive", "bic_perm")
EXACT_TESTS = ("slope", "M8", "var95", "bic_perm")
METHODS = ("FPCA-oracle", "FPCA-BIC", "HH-A", "HH-B", "OGL-CV", "OGL-1SE")

POWER_ARTIFACT = "window_power.csv"
METHODS_ARTIFACT = "window_methods.csv"
SUPPORT_ARTIFACT = "window_methods_support.csv"

_STREAMS = ("power", "methods", "perm")
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")


def _cfg(cfg):
    return {**CONFIG, **(cfg or {})}


def rng_of(stream, n, rep, base=CONFIG["base"]):
    """A generator for one (stream, n, replication), independent of every other. Returns a
    numpy Generator.

    The seed is the tuple itself, hashed by `SeedSequence`, so no two (stream, n, rep) share a
    stream and a cell can be rerun on its own.

    PARAMETERS
    stream : str, "power", "methods" or "perm";
    n      : int, pairs per dataset;
    rep    : int, replication index;
    base   : int, moves the whole study to a fresh set of draws.
    """
    return np.random.default_rng([int(base), _STREAMS.index(stream), int(n), int(rep)])


def design_cells(cfg=None):
    """The cells of the study, with `kappa` and the true `R^2` of each. Returns a DataFrame
    with columns `cell`, `kind` ("r2" or "kappa"), `kappa` and `r2`.

    The realistic cells are set by `R^2` through `window_dgp.kappa_for_r2`; the reference cells
    by `kappa`, with `R^2` from `window_dgp.window_jump_truth`.

    PARAMETERS
    cfg : dict or None, overrides applied on top of `CONFIG`.
    """
    cfg = _cfg(cfg)
    grid = np.linspace(0.0, 1.0, cfg["n_bars"] + 1)
    rows = [{"cell": f"R2={r2:g}", "kind": "r2", "r2": float(r2),
             "kappa": wd.kappa_for_r2(r2, cfg["T"], grid, cfg["start"])} for r2 in cfg["r2"]]
    for k in cfg["kappa_ref"]:
        tr = wd.window_jump_truth(grid, cfg["sigma"], cfg["T"], cfg["nu"], cfg["start"], k)
        rows.append({"cell": f"kappa={k:g}", "kind": "kappa", "r2": tr["r2"],
                     "kappa": float(k)})
    return pd.DataFrame(rows, columns=["cell", "kind", "kappa", "r2"])


def simulate(n, rng, n_extra=0, cfg=None):
    """One dataset of `n + n_extra` pairs `(J_t, X_{t-1})` at `kappa = 0`, with what
    `jumps_at` needs to move it to any other `kappa`. Returns a dict with

        s       (N,) the grid;
        X_prev  (n + n_extra, N) the curve before each jump;
        J0      (n + n_extra,) the jumps at kappa = 0;
        slope   (n + n_extra,) bhat of each curve in `X_prev`;
        m, T    the GBM's log drift and the night's length.

    `n + n_extra + 1` sessions are drawn and the first jump, which reads the burn-in curve, is
    dropped, as in the notebook: the pairs use only the sessions of the sample.

    PARAMETERS
    n       : int, pairs to fit on;
    rng     : numpy Generator;
    n_extra : int, further pairs after them, held out for forecasting;
    cfg     : dict or None, overrides applied on top of `CONFIG`.
    """
    cfg = _cfg(cfg)
    sim = wd.window_jump_sessions(n_days=n + n_extra + 1, n_bars=cfg["n_bars"], S0=cfg["S0"],
                                  sigma=cfg["sigma"], T=cfg["T"], nu=cfg["nu"],
                                  start=cfg["start"], kappa=0.0, rng=rng)
    return {"s": sim["s"], "X_prev": sim["X"][:-1], "J0": sim["J"][1:],
            "slope": sim["slope"][1:], "m": sim["m"], "T": sim["T"]}


def jumps_at(data, kappa):
    """The jumps of `data` at `kappa`: `J(0) + kappa T (bhat - 100 m)`. Returns (n,).

    PARAMETERS
    data  : dict from `simulate`;
    kappa : float.
    """
    return data["J0"] + kappa * data["T"] * (data["slope"] - 100.0 * data["m"])


def true_beta(grid, start=0.8, kappa=1.0, T=0.25):
    """The jump's coefficient function as a function, `kappa T 12 (u - u_bar) / w^3` on
    `[start, 1]` and 0 before it, evaluated on `grid`. Returns (N,).

    This is the continuum `beta` estimates are scored against. `window_dgp.window_beta` is its
    representer on the grid, which doubles the end point to absorb the trapezium's half
    weight there, and is the right object for integrating, not for scoring.

    PARAMETERS
    grid  : (N,) the grid;
    start : float, where the window begins;
    kappa : float, the share of the slope carried on;
    T     : float, the night's length in variance time.
    """
    grid = np.asarray(grid, float)
    w = 1.0 - start
    return np.where(grid >= start - 1e-12, kappa * T * 12 * (grid - start - w / 2) / w**3, 0.0)


# ------------------------------------------------------------------------------ the tests ---


def nested_fits(scores, Y):
    """Residual sums of squares of the regressions of each column of `Y` on an intercept and
    the first M columns of `scores`, for every M. Returns (M_max + 1, c), row M for M columns.

    `scores` must have orthogonal, mean-zero columns, as the principal component scores of
    centred curves do; then each column adds `(xi_k' y)^2 / (xi_k' xi_k)` to the explained sum
    of squares, whatever came before it.

    PARAMETERS
    scores : (n, M_max) orthogonal, mean-zero regressors;
    Y      : (n,) or (n, c) responses.
    """
    Y = np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    tss = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    gain = (scores.T @ Y) ** 2 / (scores**2).sum(axis=0)[:, None]
    return np.vstack([tss, tss - np.cumsum(gain, axis=0)])


def _f_pvalues(rss, n):
    """F-test p-values of M = 1..M_max against M = 0, from `nested_fits`. Returns (M_max, c)."""
    M = np.arange(1, rss.shape[0])[:, None]
    df2 = n - M - 1
    F = (rss[0] - rss[1:]) / M / (rss[1:] / df2)
    return stats.f.sf(F, M, df2)


def _bic_pick(rss, n):
    """Index into M = 1..M_max of the BIC minimiser, per column. Returns (c,) ints."""
    M = np.arange(1, rss.shape[0])[:, None]
    return np.argmin(n * np.log(rss[1:] / n) + (M + 1) * np.log(n), axis=0)


def power_tests(J, X_prev, grid, slope, m_oracle=8, m_max=10, var_share=0.95, n_perm=999,
                rng=None):
    """Every test of `beta = 0` on one set of curves, for one or several jump series. Returns a
    list of dicts, one per (column of `J`, test), with `col`, `test`, `M` and `p`.

    One principal component analysis of the curves serves every column, and one set of
    permutations serves every column, so the columns -- the cells of a replication -- are
    tested on the same footing.

    PARAMETERS
    J         : (n,) or (n, c) jumps, `J[t]` the jump after the curve `X_prev[t]`;
    X_prev    : (n, N) the curves;
    grid      : (N,) the grid;
    slope     : (n,) bhat of each curve, for the 1-df oracle;
    m_oracle  : int, the fixed M of the oracle F test;
    m_max     : int, the largest M BIC considers;
    var_share : float, the share of curve variance that sets M for `var95`;
    n_perm    : int, permutations for `bic_perm`;
    rng       : numpy Generator for the permutations.
    """
    rng = np.random.default_rng() if rng is None else rng
    J = np.asarray(J, float)
    J = J[:, None] if J.ndim == 1 else J
    n = J.shape[0]
    pc = fpca(X_prev, grid)
    m_var = int(np.searchsorted(np.cumsum(pc["tau"]) / pc["total"], var_share - 1e-12) + 1)
    k = max(m_max, m_oracle, m_var)
    if k > n - 2:
        raise ValueError(f"{k} components need at least {k + 2} pairs, got {n}.")
    scores = pc["scores"][:, :k]

    rss = nested_fits(scores, J)
    p_all = _f_pvalues(rss, n)                          # (k, c), row M - 1
    pick = _bic_pick(rss[:m_max + 1], n)               # (c,), index M - 1
    p_naive = p_all[pick, np.arange(J.shape[1])]

    # One permutation of the days per draw, shared by the columns.
    perms = np.argsort(rng.random((n_perm, n)), axis=1)
    rows = []
    sc = scores[:, :m_max]
    for c in range(J.shape[1]):
        Yp = J[perms, c].T                             # (n, n_perm)
        rss_p = nested_fits(sc, Yp)
        p_star = _f_pvalues(rss_p, n)[_bic_pick(rss_p, n), np.arange(n_perm)]
        p_perm = (1 + np.count_nonzero(p_star <= p_naive[c])) / (n_perm + 1)

        r = np.corrcoef(slope, J[:, c])[0, 1]
        F1 = (n - 2) * r**2 / (1 - r**2)
        rows += [{"col": c, "test": "slope", "M": 1, "p": float(stats.f.sf(F1, 1, n - 2))},
                 {"col": c, "test": "M8", "M": m_oracle, "p": float(p_all[m_oracle - 1, c])},
                 {"col": c, "test": "var95", "M": m_var, "p": float(p_all[m_var - 1, c])},
                 {"col": c, "test": "bic_naive", "M": int(pick[c] + 1),
                  "p": float(p_naive[c])},
                 {"col": c, "test": "bic_perm", "M": int(pick[c] + 1), "p": float(p_perm)}]
    return rows


# ------------------------------------------------------------------------- analytic guide ---


_GL_X, _GL_W = np.polynomial.legendre.leggauss(200)


def f_test_power(n, df1, f2, level=0.05):
    """The power of the F test of `df1` coefficients in a regression with an intercept on `n`
    observations, when the regressors are Gaussian and the signal has `f^2 = R^2 / (1 - R^2)`.
    Returns a float.

    Given the regressors the statistic is noncentral F with noncentrality `f^2 chi^2_{n-1}`,
    so the power is the noncentral tail averaged over that chi-square, by 200-point
    Gauss-Legendre quadrature on its quantile scale. At `f2 = 0` it is `level`.

    PARAMETERS
    n     : int, observations;
    df1   : int, coefficients tested;
    f2    : float, the signal's `R^2 / (1 - R^2)` among the regressors tested;
    level : float, the test's size.
    """
    df2 = n - df1 - 1
    if df2 < 1:
        raise ValueError(f"need n > df1 + 1, got n={n}, df1={df1}.")
    crit = stats.f.isf(level, df1, df2)
    if f2 == 0:
        return float(level)
    q = stats.chi2.ppf((_GL_X + 1) / 2, n - 1)
    return float(0.5 * _GL_W @ stats.ncf.sf(crit, df1, df2, f2 * q))


def n_for_power(df1, f2, power=0.8, level=0.05):
    """The (real) number of observations at which `f_test_power` reaches `power`. Returns a
    float.

    PARAMETERS
    df1   : int, coefficients tested;
    f2    : float, as in `f_test_power`;
    power : float, the target;
    level : float, the test's size.
    """
    return float(optimize.brentq(lambda n: f_test_power(n, df1, f2, level) - power,
                                 df1 + 3, 1e7))


def signal_share(M, grid=None, start=0.8):
    """The share of the signal `int beta X` that the first `M` population principal components
    of Brownian curves carry. Returns a float, or an array for an array `M`.

    The components of a Brownian motion are `sqrt(2) sin((k - 1/2) pi u)` with eigenvalues
    `1 / ((k - 1/2) pi)^2`, so component `k` carries `<beta, phi_k>^2 lambda_k` of the
    signal's variance. The total is the exact `T^2 Var(bhat)` of `window_dgp`, so the share
    does not depend on `kappa`, `T` or `sigma`.

    PARAMETERS
    M     : int or array of int, components kept;
    grid  : (N,) the grid, or None for the one-minute grid of 390 bars;
    start : float, where the window begins.
    """
    grid = np.linspace(0.0, 1.0, 391) if grid is None else np.asarray(grid, float)
    M = np.asarray(M, int)
    k = np.arange(1, int(M.max()) + 1)
    lam = 1 / ((k - 0.5) * np.pi) ** 2
    phi = np.sqrt(2) * np.sin(np.outer(k - 0.5, np.pi * grid))
    d = phi @ (quadrature_weights(grid) * wd.window_beta(grid, start, kappa=1.0, T=1.0))
    share = np.cumsum(d**2 * lam) / wd.window_slope_sd(grid, start, 1.0) ** 2
    out = share[M - 1]
    return float(out) if out.ndim == 0 else out


def effective_f2(r2, share=1.0):
    """The `f^2` of a test that sees a share `share` of a signal with `R^2 = r2`: the rest of
    the signal is noise to it. Returns `share r2 / (1 - share r2)`.

    PARAMETERS
    r2    : float, the share of the jump's variance the whole signal explains;
    share : float, the share of the signal inside the regressors tested.
    """
    return share * r2 / (1.0 - share * r2)


def mc_se(p, n_rep):
    """The Monte Carlo standard error of a rejection rate `p` over `n_rep` replications.
    Returns `sqrt(p (1 - p) / n_rep)`.

    PARAMETERS
    p     : float or array, the rate;
    n_rep : int or array, replications.
    """
    return np.sqrt(np.asarray(p) * (1 - np.asarray(p)) / np.asarray(n_rep))


# ------------------------------------------------------------------------- the estimators ---


def _method_a(X, y, grid, lambdas):
    """truncFLR method A on a window `[u, 1]`, as `truncFLR.HH_A.fit_mirror` fits it. Returns
    a_hat, b_hat, the support mask, m, lambda and whether section 3.2 was degenerate.

    Where section 3.2 cannot rank the lambdas it raises; the fit is then made at the smallest
    lambda on the grid, the convention of `bench.study`, and the step-7 BIC chooses m as
    `fit_mirror` would have.
    """
    try:
        fit = HH_A.fit_mirror(X, y, grid, lambdas)
        theta, a_hat, b_r, m = fit["theta_hat"], fit["a_hat"], fit["b_hat"][::-1], fit["m_hat"]
        lam, failed = fit["lam_hat"], False
    except DegenerateSelection:
        lam, failed = float(np.min(lambdas)), True
        X_r = X[:, ::-1]
        m_range = list(range(2, 11))
        scans = HH_A.theta_methodA_multi(X_r, y, grid, lam, m_range)
        best = None
        for m_try in m_range:
            th = scans[m_try][0]
            a_try, b_try = HH_A.ab_methodA(th, grid, X_r, y, n_components=m_try)
            bic = HH_A.methodA_bic(a_try, b_try, X_r, y, grid, m_try, th)
            if best is None or bic < best[0]:
                best = (bic, th, a_try, b_try, m_try)
        _, theta, a_hat, b_r, m = best
    support = (grid <= theta)[::-1]          # the window in the mirrored orientation, reflected
    return float(a_hat), b_r[::-1], support, int(m), float(lam), failed


def _method_b(X, y, grid, lambdas):
    """truncFLR method B on a window `[u, 1]`, as `truncFLR.HH.fit_mirror` fits it with the
    refit on the selected window. Returns what `_method_a` does.

    `fit_mirror` does not report how many components the refit kept, so the steps are run
    here: the pilot, section 3.2 (the smallest lambda where it is degenerate, as for method A),
    the theta scan, and the refit, whose component count is the one its own pilot chose.
    """
    HH.check_mirrorable(grid)
    X_r = X[:, ::-1]
    a, b, fpc = HH.pilot_fpca(X_r, y, grid, n_basis=20, n_components=10)
    try:
        lam, _ = HH.lambda_select(X_r, y, grid, a, b, fpc, None, lambdas, n_fourier=1,
                                  m_tuning=None, w=None)
        failed = False
    except DegenerateSelection:
        lam, failed = float(np.min(lambdas)), True
    theta, _, _ = HH.theta_methodB(X_r, y, grid, a, b, lam)
    interval = grid <= theta
    grid_i = grid[interval]
    n_basis_i = min(20, max(len(grid_i) - 1, HH.PILOT_ORDER))
    n_components_i = max(2, min(10, len(grid_i) - 1, len(y) - 1))
    a_hat, b_sub, fpc_i = HH.pilot_fpca(X_r[:, interval], y, grid_i, n_basis=n_basis_i,
                                        n_components=n_components_i)
    b_r = np.zeros(grid.size)
    b_r[interval] = b_sub
    return float(a_hat), b_r[::-1], interval[::-1], int(fpc_i["m"]), float(lam), failed


def _group_lasso(X, y, grid, n_basis, n_folds, n_lambda, rep):
    """The fdaSP group lasso, with lambda by ten-fold cross-validation and by its
    one-standard-error rule, read off one cross-validation and one warm-started path. Returns
    a dict keyed "OGL-CV" and "OGL-1SE" of (beta, support, lambda).

    `fdaSP.SBC` has no intercept, so `X` and `y` must be centred. Both coefficient functions
    come from the path rather than from refits, so the two rules differ only in the lambda
    they choose. This is `bench.study.ogl_run`.
    """
    cv = SBC.cv_fit(X, y, grid, n_basis=n_basis, order=4, n_folds=n_folds, n_lambda=n_lambda,
                    rng=rep)
    path = SBC.fit_path(X, y, grid, lambdas=cv["lambdas"], n_basis=n_basis, order=4)
    j_cv = int(np.argmin(cv["cv_mean"]))
    j_1se = int(np.argmax(cv["cv_mean"] <= cv["cv_mean"][j_cv] + cv["cv_se"][j_cv]))
    return {name: (path["betas"][j], np.asarray(path["supports"][j], bool),
                   float(path["lambdas"][j]))
            for name, j in (("OGL-CV", j_cv), ("OGL-1SE", j_1se))}


def implied_start(support, grid):
    """The left end of the last interval on which a fit is not zero: the window start it
    implies. Returns a float; 0 for a fit that zeroes nothing, NaN for one that zeroes all.

    PARAMETERS
    support : (N,) bool, where the fit is not zero;
    grid    : (N,) the grid.
    """
    regions = SBC.find_regions(np.asarray(support, bool), grid)
    return float(regions[-1][0]) if regions else np.nan


def fit_methods(X_prev, y, grid, rep=0, cfg=None):
    """Every estimator on one dataset. Returns a dict keyed by `METHODS`, each a dict with
    `a_hat`, `beta`, `support`, `mask` (the sub-grid to predict over, or None), `m`, `lam`,
    `lam_failed` and `seconds`.

    The untruncated fits regress `y` on the principal component scores, so
    `beta_hat = sum_k d_k phi_k` and `a_hat = ybar - int beta_hat Xbar`. The truncFLR fits are
    mirrored so that the window is `[u, 1]`, with lambda by section 3.2 and the smallest
    lambda where that is degenerate. The group lasso has no intercept, so it is fitted to
    centred curves and jumps and the intercept is restored as for the untruncated fit. These
    are the fits `bench.study` makes, and `tests/test_window_power.py` checks that they are,
    to the last bit.

    PARAMETERS
    X_prev : (n, N) the curves;
    y      : (n,) the jumps after them;
    grid   : (N,) the grid;
    rep    : int, seeds the group lasso's fold assignment;
    cfg    : dict or None, overrides applied on top of `CONFIG`.
    """
    cfg = _cfg(cfg)
    n = len(y)
    qw = quadrature_weights(grid)
    xbar = X_prev.mean(axis=0)
    full = np.ones(grid.size, bool)
    out = {}

    t0 = time.perf_counter()
    pc = fpca(X_prev, grid, n_components=max(cfg["m_oracle"], cfg["m_max"]))
    scores, phi = pc["scores"], pc["phi"]
    d = scores.T @ (y - y.mean()) / (scores**2).sum(axis=0)
    rss = nested_fits(scores[:, :cfg["m_max"]], y)
    m_bic = int(_bic_pick(rss, n)[0] + 1)
    secs = time.perf_counter() - t0
    for name, m in (("FPCA-oracle", cfg["m_oracle"]), ("FPCA-BIC", m_bic)):
        beta = d[:m] @ phi[:m]
        out[name] = {"a_hat": float(y.mean() - qw @ (beta * xbar)), "beta": beta,
                     "support": full, "mask": None, "m": m, "lam": np.nan,
                     "lam_failed": False, "seconds": secs}

    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda_hh"])[1:]
    for name, method in (("HH-A", _method_a), ("HH-B", _method_b)):
        t0 = time.perf_counter()
        a_hat, beta, support, m, lam, failed = method(X_prev, y, grid, lambdas)
        out[name] = {"a_hat": a_hat, "beta": beta, "support": support, "mask": support,
                     "m": m, "lam": lam, "lam_failed": failed,
                     "seconds": time.perf_counter() - t0}

    t0 = time.perf_counter()
    ogl = _group_lasso(X_prev - xbar, y - y.mean(), grid, cfg["ogl_basis"], cfg["n_folds"],
                       cfg["n_lambda_ogl"], rep)
    secs = time.perf_counter() - t0
    for name, (beta, support, lam) in ogl.items():
        out[name] = {"a_hat": float(y.mean() - qw @ (beta * xbar)), "beta": beta,
                     "support": support, "mask": None, "m": np.nan, "lam": lam,
                     "lam_failed": False, "seconds": secs}
    return out


def _forecast_mse(a_hat, beta, X, J, grid, mask=None):
    """Mean squared error of the forecasts `a_hat + int beta X` of `J`, integrating over
    `mask` only where a truncated fit was made on it, so the trapezium does not straddle the
    drop at its start (`bench.study.prediction_mse`)."""
    if mask is None:
        pred = a_hat + np.trapezoid(np.asarray(beta) * X, grid, axis=1)
    else:
        pred = a_hat + np.trapezoid(np.asarray(beta)[mask] * X[:, mask], grid[mask], axis=1)
    return float(np.mean((np.asarray(J) - pred) ** 2))


def score_methods(fits, beta, grid, start, X_test, J_test, mse_bench):
    """What each fit in `fits` earns. Returns a dict keyed like `fits`, each a dict of metrics:

        ise         int (beta_hat - beta)^2;
        rise        ise / int beta^2, NaN where beta = 0 (1 is the error of beta_hat = 0);
        u_hat       the left end of the last interval where beta_hat is not zero
                    (`implied_start`; 0 for a fit that zeroes nothing);
        u_hit       |u_hat - start| <= 0.02;
        zeroes_any  beta_hat is exactly zero somewhere;
        zeroes_all  beta_hat is exactly zero everywhere, the right answer when beta = 0;
        mse_test    mean squared error of its forecasts of the held-out jumps;
        r2_test     1 - mse_test / mse_bench, positive where it beats the null's forecast.

    PARAMETERS
    fits      : dict from `fit_methods`;
    beta      : (N,) the true coefficient function, from `true_beta`;
    grid      : (N,) the grid;
    start     : float, the true window start;
    X_test    : (m, N) the held-out curves;
    J_test    : (m,) the jumps after them;
    mse_bench : float, the expanding-mean forecast's error on the same jumps.
    """
    scale = np.trapezoid(beta**2, grid)
    out = {}
    for name, f in fits.items():
        ise = SBC_sim.ise(f["beta"], beta, grid)
        u_hat = implied_start(f["support"], grid)
        mse = _forecast_mse(f["a_hat"], f["beta"], X_test, J_test, grid, f["mask"])
        out[name] = {"ise": float(ise), "rise": float(ise / scale) if scale > 0 else np.nan,
                     "u_hat": u_hat, "u_hit": bool(abs(u_hat - start) <= 0.02),
                     "zeroes_any": bool(not f["support"].all()),
                     "zeroes_all": bool(not f["support"].any()),
                     "mse_test": mse, "r2_test": 1.0 - mse / mse_bench}
    return out


# ----------------------------------------------------------------------- one replication ---


def power_replication(n, rep, cfg=None):
    """Every test on one dataset of `n` pairs, at every cell the replication belongs to.
    Returns a list of records, one per cell, with `n`, `rep`, `cell`, the p-value `p_<test>`
    of each of `TESTS`, and the M that `var95` and BIC chose (`M_var95`, `M_bic`).

    Replications below `n_rep_power` cover every cell; the rest, up to `n_rep_size`, only
    `R^2 = 0`, which is where the size needs the extra precision.

    PARAMETERS
    n   : int, pairs;
    rep : int, replication index;
    cfg : dict or None, overrides applied on top of `CONFIG`.
    """
    cfg = _cfg(cfg)
    cells = design_cells(cfg)
    if rep >= cfg["n_rep_power"]:
        cells = cells[(cells["kind"] == "r2") & (cells["r2"] == 0)]
    data = simulate(n, rng_of("power", n, rep, cfg["base"]), cfg=cfg)
    J = np.column_stack([jumps_at(data, k) for k in cells["kappa"]])
    tests = power_tests(J, data["X_prev"], data["s"], data["slope"], m_oracle=cfg["m_oracle"],
                        m_max=cfg["m_max"], var_share=cfg["var_share"], n_perm=cfg["n_perm"],
                        rng=rng_of("perm", n, rep, cfg["base"]))
    rows = [{"n": n, "rep": rep, "cell": cell} for cell in cells["cell"]]
    for t in tests:
        rows[t["col"]][f"p_{t['test']}"] = t["p"]
        if t["test"] in ("var95", "bic_naive"):
            rows[t["col"]]["M_var95" if t["test"] == "var95" else "M_bic"] = t["M"]
    return rows


def rejection_rates(df, level=0.05, cfg=None):
    """The share of replications in which each test rejected, by cell and n. Returns a long
    DataFrame with `cell`, `kappa`, `r2`, `n`, `test`, `rate`, `n_rep` and `se`, the Monte
    Carlo standard error of `rate`.

    PARAMETERS
    df    : DataFrame from `run_power` or `cached_power`;
    level : float, the level the tests are run at;
    cfg   : dict or None, overrides applied on top of `CONFIG`, for the cells' kappa and R^2.
    """
    long = df.melt(id_vars=["n", "rep", "cell"], value_vars=[f"p_{t}" for t in TESTS],
                   var_name="test", value_name="p")
    long["test"] = long["test"].str[2:]
    long["reject"] = long["p"] <= level
    out = (long.groupby(["cell", "n", "test"], sort=False)["reject"]
           .agg(rate="mean", n_rep="size").reset_index())
    out["se"] = mc_se(out["rate"], out["n_rep"])
    return out.merge(design_cells(cfg)[["cell", "kappa", "r2"]], on="cell")[
        ["cell", "kappa", "r2", "n", "test", "rate", "n_rep", "se"]]


def methods_replication(n, rep, cell, cfg=None):
    """Every estimator on one dataset of `n` pairs in one cell, scored, with the next
    `n_test` pairs held out. Returns a list of records, one per method, and a dict of the
    support masks keyed by method.

    The dataset depends on (n, rep) and not on the cell, so the cells of a replication share
    their curves and noise.

    PARAMETERS
    n    : int, pairs to fit on;
    rep  : int, replication index;
    cell : str, a `cell` of `design_cells`;
    cfg  : dict or None, overrides applied on top of `CONFIG`.
    """
    cfg = _cfg(cfg)
    row = design_cells(cfg).set_index("cell").loc[cell]
    data = simulate(n, rng_of("methods", n, rep, cfg["base"]), n_extra=cfg["n_test"], cfg=cfg)
    J = jumps_at(data, row["kappa"])
    grid, X = data["s"], data["X_prev"]
    bench = be.expanding_mean_forecast(J, first=n)
    fits = fit_methods(X[:n], J[:n], grid, rep=rep, cfg=cfg)
    beta = true_beta(grid, cfg["start"], row["kappa"], cfg["T"])
    scores = score_methods(fits, beta, grid, cfg["start"], X[n:], J[n:],
                           float(np.mean(bench["error"] ** 2)))
    common = {"n": n, "rep": rep, "cell": cell, "kappa": float(row["kappa"]),
              "r2": float(row["r2"])}
    records = [{**common, "method": name, **scores[name],
                "mse_bench": float(np.mean(bench["error"] ** 2)), "m": fits[name]["m"],
                "lam": fits[name]["lam"], "lam_failed": fits[name]["lam_failed"],
                "seconds": fits[name]["seconds"]} for name in METHODS]
    return records, {name: np.asarray(fits[name]["support"], bool) for name in METHODS}


# ----------------------------------------------------------------------------- the drivers ---


def _power_task(args):
    n, rep, cfg = args
    return power_replication(n, rep, cfg)


def _methods_task(args):
    n, rep, cell, cfg = args
    records, supports = methods_replication(n, rep, cell, cfg)
    return records, {k: np.packbits(v).tobytes().hex() for k, v in supports.items()}


def _map(fn, tasks, n_jobs, verbose, every):
    """`fn` over `tasks` in worker processes with one BLAS thread each. Yields (task, result)
    in the order the tasks finish, so a caller that keeps what it has is never held up by one
    slow task."""
    if n_jobs is None:
        n_jobs = max(1, min(15, (os.cpu_count() or 2) - 1))
    started = time.time()

    def report(done):
        if verbose and (done % every == 0 or done == len(tasks)):
            print(f"  {done}/{len(tasks)} in {time.time() - started:.0f} s", flush=True)

    if n_jobs == 1:
        for i, task in enumerate(tasks, 1):
            yield task, fn(task)
            report(i)
        return
    env = {v: os.environ.get(v) for v in _THREAD_VARS}
    os.environ.update({v: "1" for v in _THREAD_VARS})
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(fn, task): task for task in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                yield futures[future], future.result()
                report(i)
    finally:
        for v, old in env.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old


def _jsonable(cfg):
    """`cfg` as it reads back from JSON, tuples as lists, so two configurations compare."""
    return json.loads(json.dumps(cfg))


def _load_checkpoint(path, cfg):
    """The finished tasks in a checkpoint file, keyed by (n, rep, cell). A checkpoint written
    under another configuration is refused rather than mixed in."""
    if path is None or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    if not lines or json.loads(lines[0]).get("config") != _jsonable(cfg):
        raise ValueError(f"{path} was written under another configuration; delete it to rerun.")
    done = {}
    for line in lines[1:]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:            # a line cut off when the run was stopped
            continue
        done[tuple(entry["task"])] = (entry["records"], entry["supports"])
    return done


def run_power(cfg=None, n_jobs=None, verbose=True):
    """The power study: every n of `n_power`, every replication. Returns a DataFrame of the
    records of `power_replication`, ordered by n, replication and cell.

    Windows spawns rather than forks, so a script calling this with `n_jobs != 1` needs the
    call under `if __name__ == "__main__":`.

    PARAMETERS
    cfg     : dict or None, overrides applied on top of `CONFIG`;
    n_jobs  : int or None, worker processes; 1 runs in this process;
    verbose : bool, print progress.
    """
    cfg = _cfg(cfg)
    tasks = [(n, rep, cfg) for n in cfg["n_power"] for rep in range(cfg["n_rep_size"])]
    found = {(task[0], task[1]): rows
             for task, rows in _map(_power_task, tasks, n_jobs, verbose, every=500)}
    return pd.DataFrame([row for n, rep, _ in tasks for row in found[(n, rep)]])


def run_methods(cfg=None, n_jobs=None, verbose=True, checkpoint=None):
    """The estimator comparison: every n of `n_methods`, cell and replication. Returns the
    records DataFrame, ordered by n, replication, cell and method, and a long DataFrame of
    support frequencies with `n`, `cell`, `method`, `s` and `in_support`, the share of
    replications that left `s` in the support.

    The largest n go first, so the slowest tasks do not start last. The whole comparison takes
    hours, so with `checkpoint` every finished task is appended to that file as it arrives,
    and a rerun picks up where the last one stopped. The results do not depend on how often
    it was stopped, since every task is seeded by `rng_of` alone.

    PARAMETERS
    cfg        : dict or None, overrides applied on top of `CONFIG`;
    n_jobs     : int or None, worker processes; 1 runs in this process;
    verbose    : bool, print progress;
    checkpoint : str or None, a JSON-lines file of finished tasks to resume from and add to.
    """
    cfg = _cfg(cfg)
    cells = list(design_cells(cfg)["cell"])
    tasks = [(n, rep, cell, cfg) for n in sorted(cfg["n_methods"], reverse=True)
             for rep in range(cfg["n_rep_methods"]) for cell in cells]
    done = _load_checkpoint(checkpoint, cfg)
    todo = [t for t in tasks if t[:3] not in done]
    if verbose and done:
        print(f"  resuming: {len(done)} of {len(tasks)} tasks already done", flush=True)
    fh = None
    if checkpoint is not None:
        fresh = not os.path.exists(checkpoint)
        fh = open(checkpoint, "a", encoding="utf-8")
        if fresh:
            fh.write(json.dumps({"config": _jsonable(cfg)}) + "\n")
    try:
        for task, (records, packed) in _map(_methods_task, todo, n_jobs, verbose, every=50):
            done[task[:3]] = (records, packed)
            if fh is not None:
                fh.write(json.dumps({"task": list(task[:3]), "records": records,
                                     "supports": packed},
                                    default=lambda o: o.item()) + "\n")
                fh.flush()
    finally:
        if fh is not None:
            fh.close()

    n_grid = cfg["n_bars"] + 1
    rows, tally = [], {}
    for n, rep, cell, _ in tasks:
        records, packed = done[(n, rep, cell)]
        rows.extend(records)
        for name, hexbits in packed.items():
            bits = np.unpackbits(np.frombuffer(bytes.fromhex(hexbits), np.uint8), count=n_grid)
            tally[(n, cell, name)] = tally.get((n, cell, name), 0) + bits.astype(float)
    grid = np.linspace(0.0, 1.0, n_grid)
    freq = pd.concat([pd.DataFrame({"n": n, "cell": cell, "method": name, "s": grid,
                                    "in_support": total / cfg["n_rep_methods"]})
                      for (n, cell, name), total in tally.items()], ignore_index=True)
    return pd.DataFrame(rows), freq


def cached_power(name=POWER_ARTIFACT, refresh=False, **kwargs):
    """The power study, read from `artifacts/cache/` if cached there and run and cached if
    not. Returns the DataFrame of `run_power`.

    PARAMETERS
    name     : str, file name under `artifacts/cache/`;
    refresh  : bool, re-run and overwrite;
    **kwargs : forwarded to `run_power`.
    """
    path = cache_path(name)
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    df = run_power(**kwargs)
    df.to_csv(path, index=False, float_format="%.6g")
    return pd.read_csv(path)


def cached_methods(name=METHODS_ARTIFACT, support_name=SUPPORT_ARTIFACT, refresh=False,
                   **kwargs):
    """The estimator comparison, read from `artifacts/cache/` if cached there and run and
    cached if not. Returns the records DataFrame of `run_methods`; one run writes both caches.

    The run checkpoints to `<name>.partial` beside the cache, so a run that is stopped resumes
    from there, and the checkpoint is removed once both caches are written. `refresh` reruns
    what is cached but keeps what a stopped run had finished.

    PARAMETERS
    name         : str, file name under `artifacts/cache/` for the records;
    support_name : str, file name under `artifacts/cache/` for the support frequencies;
    refresh      : bool, re-run and overwrite;
    **kwargs     : forwarded to `run_methods`.
    """
    path, support_path = cache_path(name), cache_path(support_name)
    if os.path.exists(path) and os.path.exists(support_path) and not refresh:
        return pd.read_csv(path)
    checkpoint = path + ".partial"
    df, freq = run_methods(checkpoint=checkpoint, **kwargs)
    df.to_csv(path, index=False, float_format="%.8g")
    freq.to_csv(support_path, index=False, float_format="%.6g")
    os.remove(checkpoint)
    return pd.read_csv(path)


def cached_methods_support(name=SUPPORT_ARTIFACT):
    """The support frequencies written with `cached_methods`. Returns a DataFrame with `n`,
    `cell`, `method`, `s` and `in_support`.

    PARAMETERS
    name : str, file name under `artifacts/cache/`.
    """
    path = cache_path(name)
    if not os.path.exists(path):
        cached_methods(refresh=True)
    return pd.read_csv(path)
