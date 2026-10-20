"""Figures, tables and numbers for the extended basis/coefficient comparison.

    python scripts/make_basis_coef_ext_figures.py

Reads the cache `scripts/run_basis_coef_ext.py` writes (running it first if absent) and writes
`figures/cmp_*.pdf` / `.png` through `core.style.save_figure`, `figures/cmp_numbers.json` (every
number the write-up quotes) and `figures/cmp_tables.json` (the tables, as HTML fragments).
The two mechanism figures refit one replication live, which takes about a minute.

Colour follows `core.style` for the notebook's four methods. HH_continuous and HH_multi take
slots 5 (magenta, darkened to #c2378e for line contrast on white) and 7 (violet, #4a3aa7) of the
reference categorical palette -- every hued pair clears a simulated-CVD OKLab distance of 9 --
and are redundantly encoded by dash pattern, as is the FPCA control.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import matplotlib                                                               # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                 # noqa: E402
import numpy as np                                                              # noqa: E402
import pandas as pd                                                             # noqa: E402
from scipy.stats import wilcoxon                                                # noqa: E402

import run_basis_coef_ext as ext                                                # noqa: E402
from fts_research.bench import design, study                                    # noqa: E402
from fts_research.core import paths, style                                      # noqa: E402
from fts_research.core.quadrature import quadrature_weights                     # noqa: E402
from fts_research.fdaSP import SBC                                              # noqa: E402
from fts_research.truncFLR import HH_continuous, HH_multi                       # noqa: E402
from fts_research.truncFLR._common import trunc_fpca                            # noqa: E402

METHODS = ["HH-A", "HH-B", "OGL", "FPCA", "HH-cont", "HH-multi"]
KEY = {"HH-A": "HH-A|paper", "HH-B": "HH-B|paper", "OGL": "OGL|cv", "FPCA": "FPCA|bic",
       "HH-cont": "HH-cont|own", "HH-multi": "HH-multi|own"}
COLOUR = {**style.METHOD_COLOUR, "HH-cont": "#c2378e", "HH-multi": "#4a3aa7"}
DASH = {"HH-A": "-", "HH-B": "-", "OGL": "-", "FPCA": (0, (1.2, 1.2)),
        "HH-cont": (0, (5, 1.6)), "HH-multi": (0, (6, 1.6, 1.4, 1.6))}
WIDTH = {"HH-A": 1.3, "HH-B": 1.3, "OGL": 1.3, "FPCA": 1.2, "HH-cont": 1.5, "HH-multi": 1.5}
TRUTH = style.INK
TARGETS = list(design.TARGETS)
BASES = list(design.BASIS_KINDS)
YLIM = {"step": (-0.9, 1.9), "cdf": (-0.9, 1.9), "multi": (-1.7, 1.7)}
BREAKS = {"step": [0.6], "cdf": [0.8], "multi": [0.1, 0.3, 0.55, 0.8]}
VLABEL = {"clean": "clean designs", "noisy": "null region perturbed"}
XLO, XHI = 2e-3, 3.0
DPI = 170
JITTER = np.random.default_rng(0)
N, TABLES = {}, {}


def save(fig, name):
    style.save_figure(fig, name, dpi=DPI)
    plt.close(fig)
    print("wrote", name, flush=True)


def method_legend(fig, methods=METHODS, truth=True, y=0.0, ncol=None, truth_label=None):
    handles = [plt.Line2D([], [], color=COLOUR[m], lw=1.8, ls=DASH[m], label=m) for m in methods]
    if truth:
        handles.append(plt.Line2D([], [], color=TRUTH, lw=1.7, ls="--",
                                  label=truth_label or r"true $\psi$"))
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles),
               bbox_to_anchor=(0.5, y), columnspacing=1.8, handlelength=2.6)


def shade(ax, target, grid, alpha=0.45):
    for lo, hi in SBC.find_regions(design.TARGETS[target]["null"](grid), grid):
        ax.axvspan(lo, hi, color=style.BAND, alpha=alpha, lw=0, zorder=0)


def prescribed(df):
    return df[df["tuning"].isin(["paper", "cv", "bic", "own"]) & df["bic_pick"]]


def fmt(x):
    if np.isinf(x):
        return "&infin;"
    if not np.isfinite(x):
        return "&ndash;"
    x = float(x)
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e3:
        return f"{x:.3g}"
    m, e = f"{x:.1e}".split("e")
    return f"{m}e{int(e)}"


def paired_table(frame, metric, methods=METHODS, better="low", formatter=fmt, agg="median"):
    """HTML table: rows (target, basis), one column per method, each cell the clean value on
    top and the noisy value below it; the best method per row and variant in bold."""
    pv = frame.pivot_table(index=["variant", "target", "basis"], columns="method",
                           values=metric, aggfunc=agg)
    pv = pv[[m for m in methods if m in pv.columns]]
    head = "".join(f'<th><span class="sw" style="--c:{COLOUR[m]}"></span>{m}</th>'
                   for m in pv.columns)
    rows = []
    for target in TARGETS:
        for i, basis in enumerate(BASES):
            cells = []
            for m in pv.columns:
                parts = []
                for v in ext.VARIANTS:
                    row = pv.loc[(v, target, basis)]
                    val = row[m]
                    finite = row[np.isfinite(row.astype(float))]
                    best = (finite.min() if better == "low" else finite.max()) \
                        if better and len(finite) else None
                    cls = "b" if (best is not None and np.isfinite(val) and val == best) else ""
                    parts.append(f'<span class="v {v} {cls}">{formatter(val)}</span>')
                cells.append(f"<td>{''.join(parts)}</td>")
            tlab = f'<th rowspan="3" class="tg">{target}</th>' if i == 0 else ""
            rows.append(f"<tr>{tlab}<th class='bs'>{basis}</th>{''.join(cells)}</tr>")
    return (f'<table class="pt"><thead><tr><th>target</th><th>basis</th>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table>")


# ------------------------------------------------------------------------------ helpers ----


def truncation_point(support, grid):
    """The fitted breakpoint of a single-window support mask: theta_hat for [0, theta],
    u_hat for [u, 1], NaN where the support is the whole grid."""
    if support.all() or not support.any():
        return np.nan
    if support[0]:
        return float(grid[np.flatnonzero(support)[-1]])
    return float(grid[np.flatnonzero(support)[0]])


def dot_panel(a, cell, methods, metric="rise", lo=XLO, hi=XHI):
    for row, method in enumerate(methods):
        y = len(methods) - 1 - row
        v = cell.loc[cell["method"] == method, metric].to_numpy(dtype=float)
        v = v[~np.isnan(v)]           # an infinite error is a real (off-scale) fit, kept
        if not v.size:
            continue
        inside, outside = np.clip(v, lo, hi), v > hi
        a.scatter(inside[~outside], y + JITTER.uniform(-0.18, 0.18, (~outside).sum()),
                  s=7, color=COLOUR[method], alpha=0.3, lw=0)
        if outside.any():
            a.scatter(np.full(outside.sum(), hi), y + JITTER.uniform(-0.18, 0.18, outside.sum()),
                      s=16, marker=">", color=COLOUR[method], alpha=0.55, lw=0)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        a.plot([np.clip(q1, lo, hi), np.clip(q3, lo, hi)], [y, y], color=COLOUR[method],
               lw=2.2, solid_capstyle="butt", zorder=3)
        a.plot([np.clip(med, lo, hi)], [y], marker="o", ms=6.5, color=COLOUR[method],
               mec="white", mew=1.2, zorder=4)
        if med > hi:
            a.annotate(style.fmt(med), (hi, y), xytext=(-3, 6), textcoords="offset points",
                       ha="right", fontsize=7, color=style.MUTED)
    a.set_xscale("log")
    a.set_xlim(lo, hi)
    a.set_ylim(-0.7, len(methods) - 0.3)
    a.set_yticks(range(len(methods)))
    a.set_yticklabels(list(reversed(methods)))
    a.axvline(1.0, color=style.MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    a.grid(axis="y", visible=False)


# ------------------------------------------------------------------ mechanism figures ------


def unconstrained_components(X, Y, grid, theta, m1, m2):
    """HH_continuous's two-region design at `theta` fitted *without* the continuity row, for
    the figure only -- what the constraint changes."""
    lo, hi = grid <= theta, grid >= theta
    phi1, _ = trunc_fpca(X, grid, lo, m1)
    phi2, _ = trunc_fpca(X, grid, hi, m2)
    s1 = np.trapezoid(phi1[:, None, :] * X[None, :, lo], grid[lo], axis=2).T
    s2 = np.trapezoid(phi2[:, None, :] * X[None, :, hi], grid[hi], axis=2).T
    D = np.column_stack([np.ones(len(Y)), s1, s2])
    coef, *_ = np.linalg.lstsq(D, Y, rcond=None)
    k1 = phi1.shape[0]
    return coef[1:1 + k1] @ phi1, coef[1 + k1:] @ phi2


def fig_cont_mechanism():
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    out = {}
    for row, target in enumerate(["step", "cdf"]):
        d = ext.make_data("trig", target, 0, "clean")
        X, Y, g = d["X"], d["Y"], d["grid"]
        thetas = g[ext.CONT["theta_stride"]:-1:ext.CONT["theta_stride"]]
        prof = HH_continuous.truncation_profile(X, Y, g, thetas=thetas, n_folds=10,
                                                m1=5, m2=5, rng=0)
        th = prof["theta_hat"]
        a_hat, c1, phi1, c2, phi2 = HH_continuous.fit_components(X, Y, g, th, 5, 5)
        b1u, b2u = unconstrained_components(X, Y, g, th, 5, 5)
        lo, hi = g <= th, g >= th
        true_bp = BREAKS[target][0]

        ax = axes[row, 0]
        ok = np.isfinite(prof["mse"])
        t, mse, se = prof["thetas"][ok], prof["mse"][ok], prof["se"][ok]
        ax.fill_between(t, mse - se, mse + se, color=COLOUR["HH-cont"], alpha=0.14, lw=0)
        ax.plot(t, mse, color=COLOUR["HH-cont"], lw=1.4, marker="o", ms=2.8)
        ax.axvline(true_bp, color=TRUTH, ls=":", lw=1.1)
        ax.axvline(th, color=COLOUR["HH-cont"], lw=1.1)
        ax.text(th, 0.97, rf"$\hat\theta$ = {th:.2f}  " if th > 0.5 else rf"  $\hat\theta$ = {th:.2f}",
                transform=ax.get_xaxis_transform(), fontsize=7.5, color=style.MUTED, va="top",
                ha="right" if th > 0.5 else "left")
        ax.text(true_bp, 0.06, f" true break {true_bp}", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=style.MUTED)
        ax.set_ylim(mse.min() - se.max(), np.percentile(mse + se, 90))
        ax.set_title(f"({'ad'[row]})  {target}: 10-fold CV error over " r"$\theta$", loc="left")
        ax.set_ylabel("held-out MSE  (band: ±1 SE)")

        ax = axes[row, 1]
        shades = style.ordinal(3)
        for j in range(3):
            ax.plot(g[lo], phi1[j], color=shades[j], lw=1.2,
                    label=rf"$\phi^{{(1)}}_{j + 1}$ / $\phi^{{(2)}}_{j + 1}$")
            ax.plot(g[hi], phi2[j], color=shades[j], lw=1.2, ls=(0, (4, 1.5)))
        ax.axvline(th, color=COLOUR["HH-cont"], lw=1.1)
        ax.set_ylim(-3.4, 3.6)
        ax.set_title(f"({'be'[row]})  local FPCs on each side of " r"$\hat\theta$", loc="left")
        ax.text(0.02, 0.04, r"solid: $X$ on $[0,\hat\theta]$   dashed: $X$ on $[\hat\theta,1]$",
                transform=ax.transAxes, fontsize=7.2, color=style.MUTED)
        if row == 0:
            ax.legend(loc="upper left", fontsize=7, ncol=3, handlelength=1.4, columnspacing=0.9)

        ax = axes[row, 2]
        shade(ax, target, g)
        ax.plot(g, d["psi"], color=TRUTH, ls="--", lw=1.6, label=r"true $\psi$")
        ax.plot(g[lo], b1u, color=style.CONTROL, lw=1.1, label="same bases, no constraint")
        ax.plot(g[hi], b2u, color=style.CONTROL, lw=1.1)
        b1, b2 = c1 @ phi1, c2 @ phi2
        ax.plot(g[lo], b1, color=COLOUR["HH-cont"], lw=1.7, label=r"HH-cont, $b_1(\hat\theta)=b_2(\hat\theta)$")
        ax.plot(g[hi], b2, color=COLOUR["HH-cont"], lw=1.7, ls=(0, (5, 1.6)))
        ax.plot([th], [b1[-1]], "o", color=COLOUR["HH-cont"], mec="white", ms=6, zorder=6)
        ax.set_ylim(-0.9, 1.7)
        ax.set_title(f"({'cf'[row]})  the fit at " r"$\hat\theta$", loc="left")
        if row == 0:
            ax.legend(loc="upper left", fontsize=7)
        out[target] = {"theta_hat": th, "jump_unconstrained": float(b2u[0] - b1u[-1]),
                       "b_at_theta": float(b1[-1]),
                       "rise": study.relative_ise(np.concatenate([b1, b2[1:]]), d["psi"], g,
                                                  d["scale"])}
    for ax in axes[1]:
        ax.set_xlabel("t")
    axes[1, 0].set_xlabel(r"candidate breakpoint $\theta$")
    fig.tight_layout()
    save(fig, "cmp_cont_mechanism")
    N["cont_mechanism"] = out


def fig_multi_mechanism():
    d = ext.make_data("trig", "cdf", 0, "clean")
    X, Y, g = d["X"], d["Y"], d["grid"]
    cfg = study.CONFIG
    lambdas = np.linspace(0.0, 1.0, cfg["n_lambda_hh"])[1:]
    r = HH_multi.binary_segment(X, Y, g, lambdas, n_basis=cfg["pilot_basis"],
                                n_components=cfg["n_components"], n_fourier=cfg["n_fourier"],
                                n_folds=5, max_depth=4, min_points=8, rng=0)

    # The root contest again, on the very folds binary_segment drew for it.
    seed = hash((round(g[0], 10), round(g[-1], 10), 0, 0)) % (2 ** 32)
    folds = np.random.default_rng(seed).permutation(len(Y)) % 5
    e_no, _ = HH_multi._cv_errors(lambda a, b, c: HH_multi._predict_nosplit(a, b, c, g, 20, 10),
                                  X, Y, folds)
    e_l, bp_l = HH_multi._cv_errors(lambda a, b, c: HH_multi._predict_split(
        a, b, c, g, lambdas, False, 20, 10, 1), X, Y, folds)
    e_r, bp_r = HH_multi._cv_errors(lambda a, b, c: HH_multi._predict_split(
        a, b, c, g, lambdas, True, 20, 10, 1), X, Y, folds)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6))
    ax = axes[0, 0]
    for c in r["contests"]:
        y = -c["depth"]
        split = c["breakpoint"] is not None
        ax.plot([c["lo"], c["hi"]], [y, y], color=COLOUR["HH-multi"] if split else style.CONTROL,
                lw=7, solid_capstyle="butt", alpha=0.85 if split else 0.55)
        mid = 0.5 * (c["lo"] + c["hi"])
        lab = f"split at {c['breakpoint']:.3f}" if split else "no split"
        ax.text(mid, y + 0.2, lab, ha="center", fontsize=7.5, color=style.MUTED)
        if split:
            ax.plot([c["breakpoint"]] * 2, [y - 0.22, y + 0.12], color=style.INK, lw=1.4)
    for b in BREAKS["cdf"]:
        ax.axvline(b, color=TRUTH, ls=":", lw=1.0)
    depth = max(c["depth"] for c in r["contests"])
    ax.set_ylim(-depth - 0.6, 0.6)
    ax.set_yticks([-k for k in range(depth + 1)])
    ax.set_yticklabels([f"depth {k}" for k in range(depth + 1)])
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="y", visible=False)
    ax.set_title("(a)  the segments contested, top to bottom", loc="left")
    ax.set_xlabel("t   (dotted: true break 0.8)")

    ax = axes[0, 1]
    f = np.arange(1, 6)
    ax.plot(f, e_no, "o-", color=style.CONTROL, lw=1.3, label="no split (FPCA pilot)")
    ax.plot(f, e_l, "^-", color=COLOUR["HH-B"], lw=1.1, alpha=0.6, ls=(0, (3, 1.5)),
            label=r"method B, window $[0,\theta]$")
    ax.plot(f, e_r, "s-", color=COLOUR["HH-B"], lw=1.3, label=r"method B, window $[u,1]$")
    best = e_r if e_r.mean() < e_l.mean() else e_l
    diff = e_no - best
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    ax.set_yscale("log")
    ax.set_xticks(f)
    ax.set_xlabel("fold")
    ax.set_ylabel("held-out MSE (log scale)")
    ax.set_title(f"(b)  root contest: paired gain {diff.mean():.2e} vs SE {se:.2e}", loc="left")
    ax.legend(loc="upper left", fontsize=7.2)

    ax = axes[1, 0]
    ms = sorted(r["bics"])
    # The same BIC with the RSS the fit itself minimises: per-leaf quadrature, which leaves the
    # one grid panel between adjacent leaves out. `select_joint_refit` integrates the whole grid.
    own = {}
    n = len(Y)
    for m in ms:
        a_m, b_m, masks, k_m = HH_multi.joint_refit(X, Y, g, r["breakpoints"], n_components=m)
        fv = a_m + sum(np.trapezoid(b_m[mk][None, :] * X[:, mk], g[mk], axis=1) for mk in masks)
        own[m] = n * np.log(np.sum((Y - fv) ** 2) / n) + (k_m + 1) * np.log(n)
    ax.plot(ms, [r["bics"][m] for m in ms], "o-", color=COLOUR["HH-multi"], lw=1.3,
            label="as select_joint_refit computes it (whole-grid RSS)")
    ax.plot(ms, [own[m] for m in ms], "s--", color=style.CONTROL, lw=1.2, ms=4.5,
            label="with the fit's own per-leaf quadrature")
    ax.plot([r["m_hat"]], [r["bics"][r["m_hat"]]], "o", ms=10, mfc="none", mec=style.INK, mew=1.2)
    ax.set_yscale("symlog", linthresh=1000)
    ax.set_xticks(ms)
    ax.set_xlabel("components per leaf, m  (shared by every leaf)")
    ax.set_ylabel("BIC  (symmetric log)")
    ax.set_title(f"(c)  joint refit: BIC picks m = {r['m_hat']}", loc="left")
    ax.legend(loc="upper left", fontsize=7.2)
    N["multi_bic_own_quadrature"] = {int(m): float(v) for m, v in own.items()}
    N["multi_bic_package"] = {int(m): float(v) for m, v in r["bics"].items()}

    ax = axes[1, 1]
    shade(ax, "cdf", g)
    ax.plot(g, d["psi"], color=TRUTH, ls="--", lw=1.6, label=r"true $\psi$")
    ax.plot(g, r["b_hat"], color=COLOUR["HH-multi"], lw=1.6, ls=DASH["HH-multi"],
            label="HH-multi")
    for b in r["breakpoints"]:
        ax.axvline(b, color=COLOUR["HH-multi"], lw=0.9)
    ax.set_ylim(-0.6, 1.4)
    ax.set_xlabel("t")
    ax.set_title("(d)  final fit: one FPC expansion per leaf, one intercept", loc="left")
    ax.legend(loc="upper left", fontsize=7.2)
    fig.tight_layout()
    save(fig, "cmp_multi_mechanism")
    N["multi_mechanism"] = {
        "breakpoints": r["breakpoints"], "m_hat": int(r["m_hat"]),
        "contests": r["contests"], "e_no": e_no.tolist(), "e_left": e_l.tolist(),
        "e_right": e_r.tolist(), "paired_mean": float(diff.mean()), "paired_se": float(se),
        "bp_left_lastfold": float(bp_l), "bp_right_lastfold": float(bp_r),
        "rise": study.relative_ise(r["b_hat"], d["psi"], g, d["scale"]),
        "max_abs_on_null": float(np.abs(r["b_hat"][design.TARGETS["cdf"]["null"](g)]).max())}


# ---------------------------------------------------------------------- design figure ------


def fig_targets(arrays):
    g = arrays["grid"]
    w = quadrature_weights(g)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1]})
    stats = {}
    for j, target in enumerate(TARGETS):
        psi0 = design.TARGETS[target]["fn"](g)
        ti = TARGETS.index(target)
        noisy = arrays["psi"][1, 0, ti, 0]
        ax = axes[0, j]
        shade(ax, target, g, 0.6)
        ax.axhline(0, color=style.MUTED, lw=0.6)
        ax.plot(g, noisy, color=style.SEQUENTIAL_STEPS[9], lw=1.1,
                label="perturbed (one replication)")
        ax.plot(g, psi0, color=TRUTH, lw=1.7, label=r"$\psi$ as in the notebook")
        ax.set_ylim(-1.35, 1.35)
        ax.set_title(f"{target}: {design.TARGETS[target]['label']}", loc="left")
        ax = axes[1, j]
        shade(ax, target, g, 0.6)
        eta = arrays["eta"][1, 0, ti]
        for k, shade_c in zip(range(4), style.ordinal(4)):
            ax.plot(g, eta[k], color=shade_c, lw=1.0)
        ax.axhline(0, color=style.MUTED, lw=0.6)
        ax.set_ylim(-0.19, 0.19)
        ax.set_xlabel("t")
        # Size of the perturbation, over every replication and basis.
        E = arrays["eta"][1, :, ti].reshape(-1, g.size)
        ratio = (E ** 2 @ w) / (w @ psi0 ** 2)
        stats[target] = {"int_eta2_over_int_psi2_median": float(np.median(ratio)),
                         "max_abs_eta_median": float(np.median(np.abs(E).max(axis=1))),
                         "max_abs_eta_max": float(np.abs(E).max())}
        ax.text(0.02, 0.05, rf"$\int\eta^2/\int\psi^2$ = {np.median(ratio):.1%} (median)",
                transform=ax.transAxes, fontsize=7.5, color=style.MUTED)
    axes[0, 0].set_ylabel(r"$\psi(t)$")
    axes[1, 0].set_ylabel(r"perturbation $\eta(t)$" "\n(four replications)")
    axes[0, 0].legend(loc="upper left", fontsize=7.2)
    fig.suptitle("The coefficient functions, and the perturbation added on the null set "
                 r"$T_0$ (shaded): RMS 0.05 on $T_0$, zero at its edges", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, "cmp_targets")
    N["perturbation"] = stats


def perturbation_signal_share():
    """Var(int eta X) / Var(int psi X) on the training curves: how much of the response the
    perturbation moves. Returns a nested dict target -> basis -> median over replications."""
    out = {}
    for target in TARGETS:
        out[target] = {}
        for kind in BASES:
            v = []
            for rep in range(10):
                d = ext.make_data(kind, target, rep, "noisy")
                w = quadrature_weights(d["grid"])
                s_eta = (d["X"] * w) @ d["eta"]
                s_all = (d["X"] * w) @ d["psi"]
                v.append(np.var(s_eta) / np.var(s_all - s_eta))
            out[target][kind] = float(np.median(v))
    N["perturbation_signal_share"] = out


# ------------------------------------------------------------------------ result figures ----


def fig_fits(arrays, variant):
    g = arrays["grid"]
    vi = ext.VARIANTS.index(variant)
    keys = list(arrays["curve_keys"])
    fig, ax = plt.subplots(3, 3, figsize=(11.5, 7.6), sharex=True)
    for i, kind in enumerate(BASES):
        for j, target in enumerate(TARGETS):
            a = ax[i, j]
            lo, hi = YLIM[target]
            shade(a, target, g)
            off = []
            for m in METHODS:
                beta = arrays["curves"][vi, i, j, 0, keys.index(KEY[m])]
                shown, worst = style.clip_to(beta, lo, hi)
                a.plot(g, shown, color=COLOUR[m], lw=WIDTH[m], ls=DASH[m])
                if worst is not None:
                    off.append((m, worst))
            a.plot(g, arrays["psi"][vi, i, j, 0], color=TRUTH, ls="--", lw=1.6, zorder=5)
            a.set_ylim(lo, hi)
            for k, (m, worst) in enumerate(off):
                a.annotate(f"{m} off scale, reaches {style.fmt(worst)}", (0.04, 0.05 + 0.1 * k),
                           xycoords="axes fraction", fontsize=7.2, color=style.MUTED)
            if i == 0:
                a.set_title(target)
            if j == 0:
                a.set_ylabel(f"{kind}\n" r"$\hat\psi(t)$")
            if i == 2:
                a.set_xlabel("t")
    fig.suptitle(f"One replication of each cell ({VLABEL[variant]}), every method at its own "
                 "prescribed tuning", y=0.995)
    method_legend(fig, y=-0.005)
    plt.tight_layout(rect=(0, 0.035, 1, 0.985))
    save(fig, f"cmp_fits_{variant}")


def fig_bands(arrays, variant):
    g = arrays["grid"]
    vi = ext.VARIANTS.index(variant)
    keys = list(arrays["curve_keys"])
    fig, ax = plt.subplots(len(METHODS), 3, figsize=(11.5, 12.5), sharex=True)
    for j, target in enumerate(TARGETS):
        psi0 = design.TARGETS[target]["fn"](g)
        lo, hi = YLIM[target]
        for i, m in enumerate(METHODS):
            a = ax[i, j]
            C = arrays["curves"][vi, :, j, :, keys.index(KEY[m])].reshape(-1, g.size)
            C = C.astype(float)
            q10, q50, q90 = np.nanpercentile(C, [10, 50, 90], axis=0)
            shade(a, target, g)
            a.fill_between(g, np.clip(q10, lo, hi), np.clip(q90, lo, hi), color=COLOUR[m],
                           alpha=0.2, lw=0)
            a.plot(g, style.clip_to(q50, lo, hi)[0], color=COLOUR[m], lw=1.5, ls=DASH[m])
            a.plot(g, psi0, color=TRUTH, ls="--", lw=1.2, zorder=5)
            a.set_ylim(lo, hi)
            if j == 0:
                a.set_ylabel(m, fontsize=9.5, color=style.INK)
            if i == 0:
                a.set_title(target)
            if i == len(METHODS) - 1:
                a.set_xlabel("t")
    fig.suptitle(f"Pointwise median (line) and 10–90% band of the fitted coefficient over all "
                 f"120 fits per target\n(40 replications × 3 bases, {VLABEL[variant]}); dashed "
                 r"ink is $\psi$ without the perturbation", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, f"cmp_bands_{variant}")


def fig_rise(pres, variant):
    fig, ax = plt.subplots(3, 3, figsize=(11.5, 7.8), sharex=True, sharey=True)
    for i, kind in enumerate(BASES):
        for j, target in enumerate(TARGETS):
            a = ax[i, j]
            dot_panel(a, pres.query("variant == @variant and basis == @kind and "
                                    "target == @target"), METHODS)
            if i == 0:
                a.set_title(target)
            if j == 0:
                a.set_ylabel(kind, fontsize=9.5)
            else:
                a.tick_params(labelleft=False)
            if i == 2:
                a.set_xlabel("relative ISE (log scale)")
    fig.suptitle(f"Every replication, every cell ({VLABEL[variant]}). Dot = median, bar = "
                 "interquartile range,\ntriangles on the right edge are off scale; the dashed "
                 "line is the zero estimate.", fontsize=9.5, y=1.03)
    plt.tight_layout()
    save(fig, f"cmp_rise_{variant}")


def fig_dumbbell(pres, metric="rise", name="cmp_clean_vs_noisy", xlabel="median relative ISE"):
    med = pres.groupby(["variant", "basis", "target", "method"])[metric].median()
    fig, ax = plt.subplots(3, 3, figsize=(11.5, 6.6), sharex=True, sharey=True)
    for i, kind in enumerate(BASES):
        for j, target in enumerate(TARGETS):
            a = ax[i, j]
            for row, m in enumerate(METHODS):
                y = len(METHODS) - 1 - row
                c, n_ = med[("clean", kind, target, m)], med[("noisy", kind, target, m)]
                cx, nx = np.clip(c, XLO, XHI), np.clip(n_, XLO, XHI)
                a.plot([cx, nx], [y, y], color=COLOUR[m], lw=2.0, solid_capstyle="butt")
                a.plot([cx], [y], "o", ms=6.5, color=style.SURFACE, mec=COLOUR[m], mew=1.6,
                       zorder=3)
                a.plot([nx], [y], "o", ms=6.5, color=COLOUR[m], mec="white", mew=1.2, zorder=4)
                if n_ > XHI or c > XHI:
                    a.annotate(f"{style.fmt(c)} → {style.fmt(n_)}", (XHI, y), xytext=(-3, 6),
                               textcoords="offset points", ha="right", fontsize=6.8,
                               color=style.MUTED)
            a.set_xscale("log")
            a.set_xlim(XLO, XHI)
            a.set_ylim(-0.6, len(METHODS) - 0.4)
            a.set_yticks(range(len(METHODS)))
            a.set_yticklabels(list(reversed(METHODS)))
            a.grid(axis="y", visible=False)
            if i == 0:
                a.set_title(target)
            if j == 0:
                a.set_ylabel(kind, fontsize=9.5)
            else:
                a.tick_params(labelleft=False)
            if i == 2:
                a.set_xlabel(f"{xlabel} (log scale)")
    hollow = plt.Line2D([], [], marker="o", ls="none", ms=6.5, color=style.SURFACE,
                        mec=style.MUTED, mew=1.6, label="clean design")
    filled = plt.Line2D([], [], marker="o", ls="none", ms=6.5, color=style.MUTED,
                        mec="white", mew=1.2, label="null region perturbed")
    fig.legend(handles=[hollow, filled], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"What the perturbation costs each method: {xlabel}, same replications, from "
                 "the clean design (hollow)\nto the perturbed one (filled)", fontsize=9.5, y=1.03)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, name)


def fig_support(arrays, variant):
    g = arrays["grid"]
    vi = ext.VARIANTS.index(variant)
    keys = list(arrays["curve_keys"])
    strip = [("truth", None), ("HH-A", "HH-A|paper"), ("HH-B", "HH-B|paper"),
             ("OGL (CV)", "OGL|cv"), ("OGL (1-SE)", "OGL|one_se"), ("FPCA", "FPCA|bic"),
             ("HH-cont", "HH-cont|own"), ("HH-multi", "HH-multi|own")]
    fig, ax = plt.subplots(3, 3, figsize=(11.5, 7.2), sharex=True)
    for i, kind in enumerate(BASES):
        for j, target in enumerate(TARGETS):
            a = ax[i, j]
            null = design.TARGETS[target]["null"](g)
            rows = [(~null).astype(float)]
            for _, k in strip[1:]:
                rows.append(arrays["supports"][vi, i, j, :, keys.index(k)].mean(axis=0))
            im = a.imshow(np.vstack(rows), aspect="auto", cmap=style.SEQUENTIAL, vmin=0, vmax=1,
                          extent=(0.0, 1.0, len(strip), 0.0), interpolation="nearest")
            a.axhline(1.0, color=style.SURFACE, lw=2.0)
            a.axhline(6.0, color=style.SURFACE, lw=2.0)
            for left, right in SBC.find_regions(~null, g):
                for edge in (left, right):
                    if 0.0 < edge < 1.0:
                        a.axvline(edge, color=style.INK, lw=0.7, ls=(0, (3, 2)), alpha=0.55)
            a.set_yticks(np.arange(len(strip)) + 0.5)
            a.grid(False)
            if i == 0:
                a.set_title(target)
            if j == 0:
                a.set_yticklabels([s[0] if s[0] != "truth" else
                                   ("true support" if variant == "clean" else "true signal region")
                                   for s in strip], fontsize=7.5)
                a.set_ylabel(kind, fontsize=9.5, labelpad=30)
            else:
                a.tick_params(labelleft=False)
            if i == 2:
                a.set_xlabel("t")
    bar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.08, aspect=55)
    bar.set_label("share of the 40 replications in which the point was kept in the support",
                  fontsize=8.5, color=style.MUTED)
    bar.outline.set_visible(False)
    fig.suptitle(f"Where each method says the coefficient is not exactly zero ({VLABEL[variant]})."
                 " Pale is a reported null set;\nthe top strip is the truth, and the two new "
                 "methods (below the white rule) never report one.", fontsize=9.5, y=1.02)
    save(fig, f"cmp_support_{variant}")


def breakpoint_table(pres, arrays):
    """Long frame: variant, basis, target, rep, method, breakpoint -- every breakpoint each
    method reports (one for HH-A/B/cont, any number for HH-multi)."""
    g = arrays["grid"]
    keys = list(arrays["curve_keys"])
    rows = []
    for vi, v in enumerate(ext.VARIANTS):
        for i, kind in enumerate(BASES):
            for j, target in enumerate(TARGETS):
                for rep in range(arrays["supports"].shape[3]):
                    for m in ("HH-A", "HH-B"):
                        s = arrays["supports"][vi, i, j, rep, keys.index(KEY[m])]
                        rows.append((v, kind, target, rep, m, truncation_point(s, g)))
    new = pres[pres["method"].isin(ext.NEW_METHODS)]
    for _, r in new.iterrows():
        bps = [] if pd.isna(r["breakpoints"]) or r["breakpoints"] == "" else \
            [float(x) for x in str(r["breakpoints"]).split(";")]
        if not bps:
            rows.append((r["variant"], r["basis"], r["target"], r["rep"], r["method"], np.nan))
        for b in bps:
            rows.append((r["variant"], r["basis"], r["target"], r["rep"], r["method"], b))
    return pd.DataFrame(rows, columns=["variant", "basis", "target", "rep", "method", "bp"])


def fig_breaks(bp, variant):
    meths = ["HH-A", "HH-B", "HH-cont", "HH-multi"]
    fig, ax = plt.subplots(3, 3, figsize=(11.5, 6.4), sharex=True, sharey=True)
    for i, kind in enumerate(BASES):
        for j, target in enumerate(TARGETS):
            a = ax[i, j]
            cell = bp.query("variant == @variant and basis == @kind and target == @target")
            for row, m in enumerate(meths):
                y = len(meths) - 1 - row
                sub = cell[cell["method"] == m]
                v = sub["bp"].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                a.scatter(v, y + JITTER.uniform(-0.2, 0.2, v.size), s=8, color=COLOUR[m],
                          alpha=0.35, lw=0)
                if m != "HH-multi" and v.size:
                    a.plot([np.median(v)], [y], "o", ms=6.5, color=COLOUR[m], mec="white",
                           mew=1.2, zorder=4)
                if m == "HH-multi":
                    none = sub.groupby("rep")["bp"].apply(lambda s: s.isna().all()).sum()
                    if none:
                        a.text(1.0, y + 0.28, f"{none}/40 no split", ha="right", fontsize=6.8,
                               color=style.MUTED)
            for b in BREAKS[target]:
                a.axvline(b, color=TRUTH, lw=1.0, ls="--", zorder=1)
            a.set_xlim(-0.02, 1.02)
            a.set_ylim(-0.6, len(meths) - 0.4)
            a.set_yticks(range(len(meths)))
            a.set_yticklabels(list(reversed(meths)))
            a.grid(axis="y", visible=False)
            if i == 0:
                a.set_title(target)
            if j == 0:
                a.set_ylabel(kind, fontsize=9.5)
            if i == 2:
                a.set_xlabel("reported breakpoint")
    fig.suptitle(f"Where each method puts the break ({VLABEL[variant]}). HH-A/B: the edge of the "
                 "kept window; HH-cont: " r"$\hat\theta$" "; HH-multi: every breakpoint.\n"
                 "Dot = median; dashed lines are the true breaks.", fontsize=9.5, y=1.03)
    plt.tight_layout()
    save(fig, f"cmp_breaks_{variant}")


# ---------------------------------------------------------------------------- tables -------


def holm(p):
    p = np.asarray(p, dtype=float)
    adj, running = np.empty(len(p)), 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def tests_table(pres):
    wide = pres.pivot_table(index=["variant", "basis", "target", "rep"], columns="method",
                            values="rise")
    contrasts = [("HH-cont", "HH-A"), ("HH-cont", "HH-B"), ("HH-cont", "OGL"),
                 ("HH-multi", "HH-A"), ("HH-multi", "HH-B"), ("HH-multi", "OGL"),
                 ("HH-cont", "HH-multi")]
    res = {}
    for a, b in contrasts:
        for v in ext.VARIANTS:
            cells, ps = [], []
            for target in TARGETS:
                for kind in BASES:
                    c = wide.loc[(v, kind, target)]
                    # Infinite errors are capped so the pair still ranks (inf - inf is NaN).
                    d = (np.minimum(c[a], 1e300) - np.minimum(c[b], 1e300)).to_numpy()
                    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                    cells.append((target, kind, float(np.mean(d < 0)), float(np.median(d))))
                    ps.append(p)
            for (target, kind, wins, md), padj in zip(cells, holm(ps)):
                res[(a, b, v, target, kind)] = (wins, md, padj)
    head = "".join(f"<th>{a}<br><span class='vs'>vs {b}</span></th>" for a, b in contrasts)
    rows = []
    for target in TARGETS:
        for i, kind in enumerate(BASES):
            tds = []
            for a, b in contrasts:
                parts = []
                for v in ext.VARIANTS:
                    wins, md, padj = res[(a, b, v, target, kind)]
                    star = "*" if padj < 0.05 else ""
                    cls = "win" if (padj < 0.05 and wins > 0.5) else (
                        "lose" if (padj < 0.05 and wins < 0.5) else "")
                    parts.append(f'<span class="v {v} {cls}">{wins:.2f}{star}</span>')
                tds.append(f"<td>{''.join(parts)}</td>")
            tlab = f'<th rowspan="3" class="tg">{target}</th>' if i == 0 else ""
            rows.append(f"<tr>{tlab}<th class='bs'>{kind}</th>{''.join(tds)}</tr>")
    TABLES["tests"] = (f'<table class="pt"><thead><tr><th>target</th><th>basis</th>{head}</tr>'
                       f"</thead><tbody>{''.join(rows)}</tbody></table>")
    N["tests"] = {f"{a} vs {b} | {v} | {t} | {k}": {"first_wins": w, "median_diff": md,
                                                     "p_holm": p}
                  for (a, b, v, t, k), (w, md, p) in res.items()}


def noise_cost_table(pres):
    """Per method and cell: median of the paired ratio rise_noisy / rise_clean, with a Holm-
    corrected Wilcoxon test of the paired difference (within method, across the nine cells)."""
    wide = pres.pivot_table(index=["method", "basis", "target", "rep"], columns="variant",
                            values="rise")
    out = {}
    for m in METHODS:
        ps, keys_ = [], []
        for target in TARGETS:
            for kind in BASES:
                c = wide.loc[(m, kind, target)]
                # A replication with an infinite error in either variant has no finite ratio;
                # it is dropped from the ratio and the test, and counted in the fail table.
                c = c[np.isfinite(c["noisy"]) & np.isfinite(c["clean"])]
                d = (c["noisy"] - c["clean"]).to_numpy()
                ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
                keys_.append((target, kind, float(np.median(c["noisy"] / c["clean"])),
                              float(np.mean(d > 0))))
        for (target, kind, ratio, worse), p in zip(keys_, holm(ps)):
            out[(m, target, kind)] = (ratio, worse, p)
    head = "".join(f'<th><span class="sw" style="--c:{COLOUR[m]}"></span>{m}</th>'
                   for m in METHODS)
    rows = []
    for target in TARGETS:
        for i, kind in enumerate(BASES):
            tds = []
            for m in METHODS:
                ratio, worse, p = out[(m, target, kind)]
                star = "*" if p < 0.05 else ""
                cls = "lose" if (p < 0.05 and ratio > 1) else ("win" if (p < 0.05 and ratio < 1)
                                                               else "")
                tds.append(f'<td><span class="v one {cls}">×{ratio:.2f}{star}</span>'
                           f'<span class="sub">{worse:.0%} worse</span></td>')
            tlab = f'<th rowspan="3" class="tg">{target}</th>' if i == 0 else ""
            rows.append(f"<tr>{tlab}<th class='bs'>{kind}</th>{''.join(tds)}</tr>")
    TABLES["noise_cost"] = (f'<table class="pt"><thead><tr><th>target</th><th>basis</th>{head}'
                            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>")
    N["noise_cost"] = {f"{m} | {t} | {k}": {"median_ratio": r, "share_worse": w, "p_holm": p}
                       for (m, t, k), (r, w, p) in out.items()}


def breakpoint_summary(pres, bp):
    """HTML: per cell and variant, the median (IQR) breakpoint each method reports; for
    HH-multi the distribution of the number of breakpoints and, on multi, how many of the four
    true breaks it finds within 0.05."""
    rows = []
    for target in TARGETS:
        for i, kind in enumerate(BASES):
            tds = []
            for m in ["HH-A", "HH-B", "HH-cont"]:
                parts = []
                for v in ext.VARIANTS:
                    x = bp.query("variant == @v and basis == @kind and target == @target and "
                                 "method == @m")["bp"].dropna().to_numpy()
                    if x.size:
                        q1, q2, q3 = np.percentile(x, [25, 50, 75])
                        parts.append(f'<span class="v {v}">{q2:.3f} <span class="iqr">'
                                     f'[{q1:.2f}, {q3:.2f}]</span></span>')
                    else:
                        parts.append(f'<span class="v {v}">&ndash;</span>')
                tds.append(f"<td>{''.join(parts)}</td>")
            parts, parts2 = [], []
            for v in ext.VARIANTS:
                sub = pres.query("variant == @v and basis == @kind and target == @target and "
                                 "method == 'HH-multi'")
                cnt = sub["n_bp"].astype(int).value_counts().reindex(range(0, 6), fill_value=0)
                dist = " / ".join(f"{cnt[k]}" for k in range(4)) + f" / {cnt[4:].sum()}"
                parts.append(f'<span class="v {v}">{dist}</span>')
                found = []
                for _, r in sub.iterrows():
                    b = [] if pd.isna(r["breakpoints"]) or r["breakpoints"] == "" else \
                        [float(x) for x in str(r["breakpoints"]).split(";")]
                    found.append(sum(any(abs(x - t) <= 0.05 for x in b) for t in BREAKS[target]))
                parts2.append(f'<span class="v {v}">{np.mean(found):.2f} of '
                              f'{len(BREAKS[target])}</span>')
            tds.append(f"<td>{''.join(parts)}</td><td>{''.join(parts2)}</td>")
            tlab = f'<th rowspan="3" class="tg">{target}</th>' if i == 0 else ""
            rows.append(f"<tr>{tlab}<th class='bs'>{kind}</th>{''.join(tds)}</tr>")
    head = ("<th>HH-A<br><span class='vs'>median [IQR]</span></th>"
            "<th>HH-B<br><span class='vs'>median [IQR]</span></th>"
            "<th>HH-cont θ̂<br><span class='vs'>median [IQR]</span></th>"
            "<th>HH-multi<br><span class='vs'>reps with 0/1/2/3/4+ breaks</span></th>"
            "<th>HH-multi<br><span class='vs'>true breaks found ±0.05</span></th>")
    TABLES["breaks"] = (f'<table class="pt"><thead><tr><th>target</th><th>basis</th>{head}'
                        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def oracle_table(df):
    tun = ["HH-A", "HH-B", "OGL"]
    pres = prescribed(df)
    best = (df[df["tuning"] == "oracle"]
            .groupby(["variant", "basis", "target", "rep", "method"])["rise"].min())
    rule = pres[pres["method"].isin(tun)].set_index(
        ["variant", "basis", "target", "rep", "method"])["rise"]
    pair = pd.concat({"rule": rule, "oracle": best}, axis=1).dropna()
    pair["ratio"] = pair["rule"] / pair["oracle"]
    TABLES["oracle"] = paired_table(pair.reset_index(), "ratio", methods=tun, better=None,
                                    formatter=lambda x: f"{x:.2f}" if x < 1e3 else fmt(x))


def orientation_table(df):
    paper = df[(df["tuning"] == "paper") & df["bic_pick"]]
    t = paper.assign(right=(paper["orientation"] == "right").astype(float))
    TABLES["orientation"] = paired_table(t, "right", methods=["HH-A", "HH-B"], better=None,
                                         formatter=lambda x: f"{x:.2f}", agg="mean")


def reproduction_check(pres):
    """The clean variant against the notebook's own cached study, method by method."""
    try:
        old = study.cached_study()
    except Exception as exc:  # the cache is optional
        N["reproduction"] = f"not checked: {exc}"
        return
    old_p = old[old["tuning"].isin(study.PRESCRIBED) & old["bic_pick"]]
    a = old_p.groupby(["basis", "target", "method"])["rise"].median()
    b = pres[pres["variant"] == "clean"].groupby(["basis", "target", "method"])["rise"].median()
    joined = pd.concat({"notebook": a, "here": b}, axis=1).dropna()
    rel = (joined["here"] / joined["notebook"] - 1).abs()
    N["reproduction"] = {"max_rel_diff_of_medians": float(rel.max()),
                         "n_cells_x_methods": int(len(rel)),
                         "n_identical": int((rel < 1e-9).sum())}
    # Row-by-row, on the fits themselves.
    keys = ["basis", "target", "rep", "method"]
    m = old_p[keys + ["rise"]].merge(pres[pres["variant"] == "clean"][keys + ["rise"]],
                                     on=keys, suffixes=("_nb", "_here"))
    N["reproduction"]["rows_compared"] = int(len(m))
    N["reproduction"]["rows_identical_1e-9"] = int(
        (np.abs(m["rise_here"] / m["rise_nb"] - 1) < 1e-9).sum())


def main():
    style.use(plt)
    df, arrays = ext.cached()
    df["bic_pick"] = df["bic_pick"].astype(bool)
    pres = prescribed(df)
    print(f"{len(df)} records, {len(pres)} prescribed")

    reproduction_check(pres)
    perturbation_signal_share()
    fig_targets(arrays)
    fig_cont_mechanism()
    fig_multi_mechanism()
    for v in ext.VARIANTS:
        fig_fits(arrays, v)
        fig_bands(arrays, v)
        fig_rise(pres, v)
        fig_support(arrays, v)
    fig_dumbbell(pres)
    fig_dumbbell(pres, metric="rise_T1", name="cmp_clean_vs_noisy_T1",
                 xlabel="median relative ISE on the signal region")
    bp = breakpoint_table(pres, arrays)
    for v in ext.VARIANTS:
        fig_breaks(bp, v)

    TABLES["rise"] = paired_table(pres, "rise")
    TABLES["rise_T0"] = paired_table(pres, "rise_T0")
    TABLES["rise_T1"] = paired_table(pres, "rise_T1")
    TABLES["mse"] = paired_table(pres, "mse_test", formatter=lambda x: f"{x:.3f}")
    TABLES["fail"] = paired_table(pres.assign(useless=(pres["rise"] >= 1).astype(float)),
                                  "useless", better=None, agg="sum",
                                  formatter=lambda x: f"{int(x)}")
    TABLES["jaccard"] = paired_table(pres, "null_jaccard", better="high",
                                     formatter=lambda x: f"{x:.3f}")
    TABLES["null_measure"] = paired_table(pres, "null_measure", better=None,
                                          formatter=lambda x: f"{x:.3f}")
    oracle_table(df)
    orientation_table(df)
    tests_table(pres)
    noise_cost_table(pres)
    breakpoint_summary(pres, bp)

    N["median_rise"] = {f"{v}|{b}|{t}|{m}": float(x) for (v, b, t, m), x in
                        pres.groupby(["variant", "basis", "target", "method"])["rise"]
                        .median().items()}
    # HH-multi leaves shorter than min_points: `binary_segment` stops contesting them but the
    # joint refit still fits them, and a one-point leaf has zero quadrature weight.
    g = arrays["grid"]
    mm = pres[pres["method"] == "HH-multi"].copy()

    def shortest_leaf(s):
        bps = [] if pd.isna(s) or s == "" else sorted(float(x) for x in str(s).split(";"))
        edges = [g[0] - 1] + bps + [g[-1]]
        return min(int(((g > lo + 1e-9) & (g <= hi + 1e-9)).sum())
                   for lo, hi in zip(edges[:-1], edges[1:]))
    mm["shortest"] = mm["breakpoints"].map(shortest_leaf)
    mm["tiny"] = (mm["shortest"] < ext.MULTI["min_points"]) & (mm["n_bp"] > 0)
    N["hh_multi_tiny_leaf"] = {
        f"{v}|{t}": {"share_with_leaf_lt_min_points": float(sub["tiny"].mean()),
                     "median_rise_tiny": float(sub.loc[sub["tiny"], "rise"].median())
                     if sub["tiny"].any() else None,
                     "median_rise_not_tiny": float(sub.loc[~sub["tiny"], "rise"].median()),
                     "n_infinite": int(np.isinf(sub["rise"]).sum())}
        for (v, t), sub in mm.groupby(["variant", "target"])}
    N["seconds"] = pres.groupby("method")["seconds"].median().round(2).to_dict()
    N["hh_multi_m_hat"] = pres[pres["method"] == "HH-multi"].groupby(
        ["variant", "target"])["m"].median().to_dict()
    N["hh_multi_m_hat"] = {f"{k[0]}|{k[1]}": float(v) for k, v in N["hh_multi_m_hat"].items()}
    with open(paths.figure_path("cmp_numbers.json"), "w") as fh:
        json.dump(N, fh, indent=1, default=float)
    with open(paths.figure_path("cmp_tables.json"), "w") as fh:
        json.dump(TABLES, fh, indent=1)
    print("done")


if __name__ == "__main__":
    main()
