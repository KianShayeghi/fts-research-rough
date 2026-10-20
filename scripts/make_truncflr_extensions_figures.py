"""Draw the figures for the write-up of `truncFLR.HH_continuous` and `truncFLR.HH_multi`.

    python scripts/make_truncflr_extensions_figures.py

Writes `figures/ext_*.pdf` and `.png` through `core.style.save_figure`, and
`figures/ext_numbers.json` holding every number the write-up quotes, so a figure or a figure
in the text is reproduced by re-running this rather than recovered from a notebook. Every design
is a single seed -- the seeds the tests use -- so these are existence checks, not a simulation
study. Takes a few minutes; nothing is cached.

Colour follows `core.style`: one hue per method, the truth in ink, the untruncated pilot in the
control grey, the true null set shaded. The two new estimators need hues the validated palette
does not supply (`style.CATEGORICAL` has three slots, taken by methods A, B and the group
lasso), so `HH_continuous` and `HH_multi` take the provisional hues below, redundantly
encoded by line style. They should move into `style.METHOD_COLOUR` once checked for colour-vision
deficiency against the other three.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

import matplotlib                                                               # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                 # noqa: E402
import numpy as np                                                              # noqa: E402

from fts_research.core import paths, style                                      # noqa: E402
from fts_research.core.gp import sample_gp                                      # noqa: E402
from fts_research.truncFLR import HH, HH_A, HH_continuous, HH_dgp, HH_multi    # noqa: E402
from fts_research.truncFLR._common import pilot_fpca                            # noqa: E402
from fts_research.truncFLR.HH_multi import (                                    # noqa: E402
    _cv_errors, _predict_nosplit, _predict_split,
)

LAMBDAS = np.linspace(0.0, 1.0, 250)[1:]      # the notebooks' grid, for methods A and B
LAM_MULTI = np.linspace(0.0, 1.0, 60)[1:]     # the grid `tests/test_hh_multi.py` uses

C_A = style.METHOD_COLOUR["HH-A"]
C_B = style.METHOD_COLOUR["HH-B"]
C_CONT = "#c2378e"    # provisional: HH_continuous
C_MULTI = "#6b4bc4"   # provisional: HH_multi


def beta_cont(t):
    """sin(6 pi t), then a ramp from the value it reaches, sin(3 pi) = 0: continuous at 0.5."""
    return np.where(t < 0.5, np.sin(6 * np.pi * t), 2.0 * (t - 0.5))


def beta_jump(t):
    """The design `tests/test_hh_continuous.py` uses: the ramp starts at 1, a unit jump at 0.5."""
    return np.where(t < 0.5, np.sin(6 * np.pi * t), 1.0 + 2.0 * (t - 0.5))


def beta_step(t):
    return (t >= 0.3).astype(float)


def beta_bump(t):
    return ((t >= 0.2) & (t <= 0.5)).astype(float)


def rel_ise(b_hat, b, grid):
    return float(np.trapezoid((b_hat - b) ** 2, grid) / np.trapezoid(b ** 2, grid))


def shade_null(ax, grid, b):
    """Shade every maximal run of grid points where the true coefficient is exactly zero."""
    zero = b == 0
    i = 0
    while i < len(grid):
        if zero[i]:
            j = i
            while j + 1 < len(grid) and zero[j + 1]:
                j += 1
            ax.axvspan(grid[i], grid[j], color=style.BAND, alpha=0.55, lw=0, zorder=0)
            i = j + 1
        else:
            i += 1


def truth_line(ax, grid, b, label="true $b(t)$"):
    ax.plot(grid, b, color=style.INK, lw=2.0, label=label, zorder=5)


def clipped(ax, grid, y, lo, hi, **kw):
    shown, worst = style.clip_to(y, lo, hi)
    if worst is not None:
        kw["label"] = kw.get("label", "") + f"  (clipped, reaches {style.fmt(worst)})"
    ax.plot(grid, shown, **kw)


def main():
    started = time.time()
    style.use(plt)
    N = {}

    def log(msg):
        print(f"[{time.time() - started:6.1f}s] {msg}", flush=True)

    D = {
        "paper": HH_dgp.sim_dataset(n=200, n_grid=301, u=0.8, v=1.0, snr=26.25, rng=123),
        "cont": HH_dgp.sim_dataset(n=300, n_grid=151, beta_fn=beta_cont, snr=50.0, rng=3),
        "jump": HH_dgp.sim_dataset(n=300, n_grid=151, beta_fn=beta_jump, snr=50.0, rng=3),
        "step": HH_dgp.sim_dataset(n=300, n_grid=101, beta_fn=beta_step, snr=40.0, rng=1),
        "bump": HH_dgp.sim_dataset(n=300, n_grid=101, beta_fn=beta_bump, snr=200.0, rng=1),
    }
    rng = np.random.default_rng(2)          # `tests/test_hh_multi.py::_null_data`
    g_noise = np.linspace(0.0, 1.0, 81)
    X_noise = sample_gp(200, g_noise, rng=rng)
    Y_noise = rng.normal(0.0, 1.0, size=200)

    # ------------------------------------------------------------ figure 1: designs ------
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    ax = axes[0, 0]
    g = D["cont"]["grid"]
    for x in D["cont"]["X"][:12]:
        ax.plot(g, x, color=style.CONTROL, lw=0.8, alpha=0.8)
    ax.set_title("(a)  twelve covariate curves $X_i(t)$", loc="left")
    panels = [("paper", "(b)  paper's design: $1\\{t \\geq 0.8\\}$"),
              ("cont", "(c)  sine $\\to$ ramp, continuous at 0.5"),
              ("jump", "(d)  sine $\\to$ ramp, jump at 0.5"),
              ("step", "(e)  single break: $1\\{t \\geq 0.3\\}$"),
              ("bump", "(f)  interior bump: $1\\{0.2 \\leq t \\leq 0.5\\}$")]
    for ax, (key, title) in zip(axes.flat[1:], panels):
        d = D[key]
        shade_null(ax, d["grid"], d["b"])
        ax.plot(d["grid"], d["b"], color=style.INK, lw=1.8)
        ax.set_title(title, loc="left")
    for ax in axes.flat:
        ax.set_xlabel("t")
    fig.tight_layout()
    style.save_figure(fig, "ext_designs", dpi=200)
    plt.close(fig)
    log("figure 1 (designs)")

    # --------------------------------------------- Task 1: HH_continuous on three designs -
    fits = {}
    for key, m in (("cont", 5), ("jump", 5), ("paper", 4)):
        d = D[key]
        X, Y, g = d["X"], d["Y"], d["grid"]
        interior = g[(g > g[0]) & (g < g[-1])]
        cont = HH_continuous.fit(X, Y, g, m1=m, m2=m, n_folds=5, rng=0)
        full = HH_continuous.truncation_profile(X, Y, g, thetas=interior, n_folds=5, m1=m,
                                                m2=m, rng=0)
        th_scan, rss = HH_continuous.theta_scan(X, Y, g, m1=m, m2=m)
        fB = HH.fit_mirror(X, Y, g, LAMBDAS)
        fA = HH_A.fit_mirror(X, Y, g, LAMBDAS)
        fits[key] = dict(cont=cont, full=full, scan=(th_scan, rss), B=fB, A=fA)
        truth = 0.8 if key == "paper" else 0.5
        a_cont, c1, phi1, c2, phi2 = HH_continuous.fit_components(
            X, Y, g, cont["theta_hat"], m, m)
        N[f"t1_{key}"] = {
            "true_theta": truth,
            "m1_m2": m,
            "candidates_default": [round(float(t), 4) for t in cont["profile"]["thetas"]],
            "theta_hat": float(cont["theta_hat"]),
            "theta_hat_full_grid": float(full["theta_hat"]),
            "cv_mse_at_theta_hat": float(np.min(cont["profile"]["mse"])),
            "rel_ise_cont": rel_ise(cont["b_hat"], d["b"], g),
            "b1_at_theta": float(c1 @ phi1[:, -1]),
            "b2_at_theta": float(c2 @ phi2[:, 0]),
            "theta_scan_argmin": float(th_scan[int(np.argmin(rss))]),
            "u_hat_B": float(fB["u_hat"]), "rel_ise_B": rel_ise(fB["b_hat"], d["b"], g),
            "u_hat_A": float(fA["u_hat"]), "rel_ise_A": rel_ise(fA["b_hat"], d["b"], g),
            "m_hat_A": int(fA["m_hat"]),
            "profile_default": [[round(float(t), 4), float(mm), float(s)] for t, mm, s in
                                zip(cont["profile"]["thetas"], cont["profile"]["mse"],
                                    cont["profile"]["se"])],
        }
        log(f"task 1 fits: {key}")

    # ------------------------------------------------- figure 2: the CV profile for theta -
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for ax, key, title in ((axes[0], "cont", "(a)  sine $\\to$ ramp, continuous at 0.5"),
                           (axes[1], "paper", "(b)  paper's design, jump at 0.8")):
        f, d = fits[key], D[key]
        full, prof = f["full"], f["cont"]["profile"]
        ok = np.isfinite(full["mse"])
        t, mse, se = full["thetas"][ok], full["mse"][ok], full["se"][ok]
        ax.fill_between(t, mse - se, mse + se, color=C_CONT, alpha=0.15, lw=0)
        ax.plot(t, mse, color=C_CONT, lw=1.4, label="CV error, every grid point (±1 SE)")
        okd = np.isfinite(prof["mse"])
        ax.plot(prof["thetas"][okd], prof["mse"][okd], "o", color=C_CONT, ms=4,
                label="CV error, the 20 default candidates")
        th, rss = f["scan"]
        ax.plot(th, rss / len(d["Y"]), color=style.CONTROL, ls="--", lw=1.1,
                label="in-sample RSS / n")
        truth = 0.8 if key == "paper" else 0.5
        ax.axvline(truth, color=style.INK, ls=":", lw=1.1, label="true $\\theta$")
        ax.axvline(f["cont"]["theta_hat"], color=C_CONT, lw=1.1,
                   label=f"$\\hat\\theta$ = {f['cont']['theta_hat']:.3f}")
        lo = min(np.nanmin(rss / len(d["Y"])), np.nanmin(mse - se))
        ax.set_ylim(lo * 0.9, np.nanpercentile(mse + se, 97) * 1.08)
        ax.set_xlabel("candidate truncation point $\\theta$")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper center", fontsize=7.2)
    axes[0].set_ylabel("mean squared prediction error")
    fig.tight_layout()
    style.save_figure(fig, "ext_t1_profile", dpi=200)
    plt.close(fig)
    log("figure 2 (task 1 profile)")

    # --------------------------------------------------- figure 3: task 1 fitted curves ---
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax, key, title in ((axes[0], "cont", "(a)  continuous at 0.5"),
                           (axes[1], "jump", "(b)  jump at 0.5"),
                           (axes[2], "paper", "(c)  paper's design, null on [0, 0.8)")):
        f, d = fits[key], D[key]
        g, b = d["grid"], d["b"]
        shade_null(ax, g, b)
        lo, hi = b.min() - 0.6, b.max() + 0.6
        truth_line(ax, g, b)
        clipped(ax, g, f["A"]["b_hat"], lo, hi, color=C_A, ls="--", lw=1.2,
                label="method A")
        clipped(ax, g, f["B"]["b_hat"], lo, hi, color=C_B, ls=":", lw=1.4,
                label="method B")
        clipped(ax, g, f["cont"]["b_hat"], lo, hi, color=C_CONT, lw=1.7,
                label="HH_continuous")
        truth = 0.8 if key == "paper" else 0.5
        ax.axvline(truth, color=style.INK, ls=":", lw=1.0)
        ax.axvline(f["cont"]["theta_hat"], color=C_CONT, lw=1.0)
        ax.set_ylim(lo, hi)
        ax.set_title(title, loc="left")
        ax.set_xlabel("t")
    axes[0].set_ylabel("$b(t)$")
    axes[0].legend(loc="upper left", fontsize=7.2)
    fig.tight_layout()
    style.save_figure(fig, "ext_t1_fits", dpi=200)
    plt.close(fig)
    log("figure 3 (task 1 fits)")

    # ------------------------------------------------- Task 2: the contest, dissected -----
    def contest_errors(X, Y, g, depth=0):
        seed = hash((round(g[0], 10), round(g[-1], 10), depth, 0)) % (2 ** 32)
        folds = np.random.default_rng(seed).permutation(len(Y)) % 4
        e_no, _ = _cv_errors(lambda a, b, c: _predict_nosplit(a, b, c, g, 20, 10),
                             X, Y, folds)
        e_f, bp_f = _cv_errors(lambda a, b, c: _predict_split(a, b, c, g, LAM_MULTI, False,
                                                              20, 10, 1), X, Y, folds)
        e_m, bp_m = _cv_errors(lambda a, b, c: _predict_split(a, b, c, g, LAM_MULTI, True,
                                                              20, 10, 1), X, Y, folds)
        return e_no, (e_f, bp_f), (e_m, bp_m)

    def decide(e_no, e_split):
        diff = e_no - e_split
        se_diff = diff.std(ddof=1) / np.sqrt(len(diff))
        se_no = e_no.std(ddof=1) / np.sqrt(len(e_no))
        return {"mean_no": float(e_no.mean()), "mean_split": float(e_split.mean()),
                "se_no": float(se_no), "mean_diff": float(diff.mean()),
                "se_diff": float(se_diff), "diffs": [float(x) for x in diff],
                "paired_splits": bool(diff.mean() > se_diff),
                "unpaired_splits": bool(e_split.mean() < e_no.mean() - se_no)}

    d = D["step"]
    e_no, (e_f, bp_f), (e_m, bp_m) = contest_errors(d["X"], d["Y"], d["grid"])
    best, best_bp, best_name = ((e_m, bp_m, "right-truncated") if e_m.mean() < e_f.mean()
                                else (e_f, bp_f, "left-truncated"))
    N["t2_root_contest_step"] = dict(
        e_no=[float(x) for x in e_no], e_left=[float(x) for x in e_f],
        e_right=[float(x) for x in e_m], bp_left_last_fold=float(bp_f),
        bp_right_last_fold=float(bp_m), winner=best_name, **decide(e_no, best))

    db = D["bump"]
    b_no, (b_f, bb_f), (b_m, bb_m) = contest_errors(db["X"], db["Y"], db["grid"])
    bbest = b_m if b_m.mean() < b_f.mean() else b_f
    N["t2_root_contest_bump"] = dict(
        mean_left=float(b_f.mean()), mean_right=float(b_m.mean()),
        bp_left_last_fold=float(bb_f), bp_right_last_fold=float(bb_m), **decide(b_no, bbest))
    log("task 2 contests")

    # ------------------------------------------------ figure 4: paired vs unpaired --------
    rc = N["t2_root_contest_step"]
    folds = np.arange(1, 5)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5),
                             gridspec_kw={"width_ratios": [1.25, 1, 0.8]})
    ax = axes[0]
    ax.plot(folds, e_no, "o-", color=style.CONTROL, lw=1.3, label="no split (pilot)")
    ax.plot(folds, best, "s-", color=C_B, lw=1.3, label="right-truncated (method B)")
    ax.set_xticks(folds)
    ax.set_xlabel("fold")
    ax.set_ylabel("held-out MSE")
    ax.set_title("(a)  per-fold error, one shared partition", loc="left")
    ax.legend(loc="upper left", fontsize=7.2)
    ax.text(0.98, 0.04, f"left-truncated candidate: {e_f.min():.2f}–{e_f.max():.2f}, off scale",
            transform=ax.transAxes, ha="right", fontsize=7, color=style.MUTED)
    ax = axes[1]
    diffs = np.array(rc["diffs"])
    ax.bar(folds, diffs, color=C_B, alpha=0.75, width=0.55)
    ax.axhline(0, color=style.INK, lw=0.8)
    ax.axhspan(rc["mean_diff"] - rc["se_diff"], rc["mean_diff"] + rc["se_diff"],
               color=style.INK, alpha=0.08, lw=0)
    ax.axhline(rc["mean_diff"], color=style.INK, lw=1.2, ls="--")
    ax.set_xticks(folds)
    ax.set_xlabel("fold")
    ax.set_ylabel("no-split error $-$ split error")
    ax.set_title(f"(b)  paired: mean {rc['mean_diff']:.1e} > SE {rc['se_diff']:.1e}",
                 loc="left")
    ax = axes[2]
    ax.errorbar([0], [rc["mean_no"]], yerr=[rc["se_no"]], fmt="o", color=style.CONTROL,
                capsize=4, lw=1.3)
    se_split = best.std(ddof=1) / np.sqrt(4)
    ax.errorbar([1], [rc["mean_split"]], yerr=[se_split], fmt="s", color=C_B, capsize=4,
                lw=1.3)
    ax.axhline(rc["mean_no"] - rc["se_no"], color=style.MUTED, ls=":", lw=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no split", "split"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_title("(c)  unpaired: overlap", loc="left")
    ax.set_ylabel("mean held-out MSE ± 1 SE")
    fig.tight_layout()
    style.save_figure(fig, "ext_t2_contest", dpi=200)
    plt.close(fig)
    log("figure 4 (contest)")

    # ------------------------------------------------- Task 2: the three end-to-end runs --
    runs = {}
    for key, X, Y, g in (("step", D["step"]["X"], D["step"]["Y"], D["step"]["grid"]),
                         ("noise", X_noise, Y_noise, g_noise),
                         ("bump", D["bump"]["X"], D["bump"]["Y"], D["bump"]["grid"])):
        r = HH_multi.binary_segment(X, Y, g, LAM_MULTI, n_folds=4, max_depth=3,
                                    min_points=10, rng=0)
        runs[key] = r
        entry = {"breakpoints": [float(b) for b in r["breakpoints"]], "m_hat": int(r["m_hat"]),
                 "n_leaves": len(r["leaf_masks"]),
                 "contests": [{"lo": c["lo"], "hi": c["hi"], "depth": c["depth"],
                               "breakpoint": c["breakpoint"]} for c in r["contests"]],
                 "max_abs_b_hat": float(np.abs(r["b_hat"]).max())}
        if key != "noise":
            entry["rel_ise"] = rel_ise(r["b_hat"], D[key]["b"], g)
        N[f"t2_run_{key}"] = entry
    log("task 2 runs")

    d, db = D["step"], D["bump"]
    a_p, b_p, fpc_p = pilot_fpca(d["X"], d["Y"], d["grid"])
    a_pb, b_pb, fpc_pb = pilot_fpca(db["X"], db["Y"], db["grid"])
    fwd = HH.fit(db["X"], db["Y"], db["grid"], LAM_MULTI)
    mir = HH.fit_mirror(db["X"], db["Y"], db["grid"], LAM_MULTI)
    N["t2_bump_whole_domain"] = {"theta_hat_left": float(fwd["theta_hat"]),
                                 "u_hat_right": float(mir["u_hat"]),
                                 "rel_ise_left": rel_ise(fwd["b_hat"], db["b"], db["grid"]),
                                 "rel_ise_right": rel_ise(mir["b_hat"], db["b"], db["grid"]),
                                 "rel_ise_pilot": rel_ise(b_pb, db["b"], db["grid"]),
                                 "pilot_m": int(fpc_pb["m"])}
    N["t2_step_pilot"] = {"rel_ise_pilot": rel_ise(b_p, d["b"], d["grid"]),
                          "pilot_m": int(fpc_p["m"])}

    # --------------------------------------------------- figure 5: task 2 fitted curves ----
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    ax = axes[0]
    shade_null(ax, d["grid"], d["b"])
    truth_line(ax, d["grid"], d["b"])
    ax.plot(d["grid"], b_p, color=style.CONTROL, lw=1.2, label="untruncated pilot")
    ax.plot(d["grid"], runs["step"]["b_hat"], color=C_MULTI, lw=1.7, ls="-.",
            label="HH_multi")
    for bp in runs["step"]["breakpoints"]:
        ax.axvline(bp, color=C_MULTI, lw=1.0)
    ax.axvline(0.3, color=style.INK, ls=":", lw=1.0)
    ax.set_title("(a)  single break at 0.3", loc="left")
    ax.set_ylabel("$b(t)$")
    ax.legend(loc="lower right", fontsize=7.2)
    ax = axes[1]
    ax.axhline(0, color=style.INK, lw=2.0, label="true $b(t) = 0$")
    ax.plot(g_noise, runs["noise"]["b_hat"], color=C_MULTI, lw=1.7, ls="-.",
            label="HH_multi")
    ax.set_ylim(-0.5, 0.5)
    ax.set_title("(b)  $Y$ independent of $X$", loc="left")
    ax.legend(loc="upper right", fontsize=7.2)
    ax = axes[2]
    shade_null(ax, db["grid"], db["b"])
    truth_line(ax, db["grid"], db["b"])
    lo, hi = -0.6, 1.6
    clipped(ax, db["grid"], fwd["b_hat"], lo, hi, color=C_B, ls="--", lw=1.1,
            label="left-truncated (B)")
    clipped(ax, db["grid"], mir["b_hat"], lo, hi, color=C_B, ls=":", lw=1.4,
            label="right-truncated (B)")
    ax.plot(db["grid"], runs["bump"]["b_hat"], color=C_MULTI, lw=1.7, ls="-.",
            label="HH_multi (no split)")
    ax.set_ylim(lo, hi)
    ax.set_title("(c)  interior bump, SNR 200", loc="left")
    ax.legend(loc="upper right", fontsize=6.8)
    for ax in axes:
        ax.set_xlabel("t")
    fig.tight_layout()
    style.save_figure(fig, "ext_t2_fits", dpi=200)
    plt.close(fig)
    log("figure 5 (task 2 fits)")

    # ----------------------------------------------- figure 6: the BIC-selected refit -----
    bp = runs["step"]["breakpoints"][0]
    _, b_fix, masks, n_fix = HH_multi.joint_refit(d["X"], d["Y"], d["grid"], [bp],
                                                  n_components=10)
    _, b_sel, _, n_sel, m_sel, bics = HH_multi.select_joint_refit(d["X"], d["Y"], d["grid"],
                                                                  [bp], n_components=10)
    N["t2_bic"] = {"breakpoint": float(bp), "m_hat": int(m_sel),
                   "bics": {int(k): float(v) for k, v in bics.items()},
                   "max_abs_null_leaf_fixed10": float(np.abs(b_fix[masks[0]]).max()),
                   "max_abs_null_leaf_selected": float(np.abs(b_sel[masks[0]]).max()),
                   "rel_ise_fixed10": rel_ise(b_fix, d["b"], d["grid"]),
                   "rel_ise_selected": rel_ise(b_sel, d["b"], d["grid"])}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    shade_null(ax, d["grid"], d["b"])
    truth_line(ax, d["grid"], d["b"])
    clipped(ax, d["grid"], b_fix, -3, 4, color=style.CONTROL, lw=1.2,
            label="joint refit, m = 10 per leaf")
    ax.plot(d["grid"], b_sel, color=C_MULTI, lw=1.7, ls="-.",
            label=f"joint refit, BIC-selected m = {m_sel}")
    ax.axvline(bp, color=C_MULTI, lw=1.0)
    ax.set_ylim(-3, 4)
    ax.set_xlabel("t")
    ax.set_ylabel("$b(t)$")
    ax.set_title(f"(a)  refit at the recovered break {bp:.2f}", loc="left")
    ax.legend(loc="upper left", fontsize=7.2)
    ax = axes[1]
    ms = sorted(bics)
    ax.plot(ms, [bics[m] for m in ms], "o-", color=C_MULTI, lw=1.3)
    ax.plot([m_sel], [bics[m_sel]], "o", ms=9, mfc="none", mec=style.INK, mew=1.2)
    ax.set_xticks(ms)
    ax.set_xlabel("components per leaf, m")
    ax.set_ylabel("BIC")
    ax.set_title("(b)  BIC over the shared component count", loc="left")
    fig.tight_layout()
    style.save_figure(fig, "ext_t2_bic", dpi=200)
    plt.close(fig)
    log("figure 6 (BIC)")

    # ------------------- figure 8: why method B misplaces theta on the interior bump ------
    # Method B's theta scan takes the *first* interior minimum of its penalised objective T
    # (HH.theta_methodB, the paper's step 6'). On the bump design T has a spurious early dip,
    # from the pilot's wiggles over the leading null stretch, ahead of the true minimum.
    a_b, b_b, f_b = pilot_fpca(db["X"], db["Y"], db["grid"])
    lam_b, diag_b = HH.lambda_select(db["X"], db["Y"], db["grid"], a_b, b_b, f_b, None,
                                     LAM_MULTI)
    th_first, th_grid, T = HH.theta_methodB(db["X"], db["Y"], db["grid"], a_b, b_b, lam_b)
    minima = [float(th_grid[k]) for k in range(1, len(T) - 1)
              if T[k] <= T[k - 1] and T[k] <= T[k + 1]]
    th_global = float(th_grid[int(np.argmin(T))])
    N["t2_bump_mechanism"] = {
        "lam_hat": float(lam_b), "b_simp_cutoff": float(diag_b["theta_bar"]),
        "theta_first_interior_min": float(th_first), "theta_global_min": th_global,
        "interior_minima": minima,
        "T_at_first": float(T[int(np.argmin(np.abs(th_grid - th_first)))]),
        "T_at_global": float(T.min()),
        "max_abs_pilot_on_leading_null": float(np.abs(b_b[db["grid"] < 0.2]).max())}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    shade_null(ax, db["grid"], db["b"])
    truth_line(ax, db["grid"], db["b"])
    ax.plot(db["grid"], b_b, color=style.CONTROL, lw=1.3, label="untruncated pilot")
    ax.set_xlabel("t")
    ax.set_ylabel("$b(t)$")
    ax.set_title("(a)  the pilot method B truncates", loc="left")
    ax.legend(loc="upper right", fontsize=7.2)
    ax = axes[1]
    ax.plot(th_grid, T, color=C_B, lw=1.4)
    ax.axvline(th_first, color=C_B, ls=":", lw=1.2)
    ax.axvline(th_global, color=style.INK, ls=":", lw=1.2)
    ax.text(th_first + 0.015, T.max() * 0.62, f"first interior minimum\n$\\theta$ = {th_first:.2f}"
            "  (chosen)", fontsize=7.2, color=style.MUTED)
    ax.text(th_global + 0.015, T.max() * 0.30, f"global minimum\n$\\theta$ = {th_global:.2f}",
            fontsize=7.2, color=style.MUTED)
    ax.set_xlabel("candidate truncation point $\\theta$  (window $[0, \\theta]$ kept)")
    ax.set_ylabel("$T(\\theta)$ = RSS + $n\\lambda\\theta^2$")
    ax.set_title("(b)  method B's objective on the whole domain", loc="left")
    ins = ax.inset_axes([0.60, 0.50, 0.36, 0.38])
    near = th_grid <= 0.18
    ins.plot(th_grid[near], T[near], color=C_B, lw=1.2, marker="o", ms=2.5)
    ins.axvline(th_first, color=C_B, ls=":", lw=1.0)
    ins.set_title("zoom: $\\theta \\leq 0.18$", fontsize=7, color=style.MUTED)
    ins.tick_params(labelsize=6.5)
    fig.tight_layout()
    style.save_figure(fig, "ext_t2_bump_mechanism", dpi=200)
    plt.close(fig)
    log("figure 8 (bump mechanism)")

    # ------------------------------ Task 3: the existing estimators on no-null designs ----
    for key in ("cont", "jump"):
        dd = D[key]
        X, Y, g = dd["X"], dd["Y"], dd["grid"]
        out = {}
        for name, fn, mirror in (("B_left", HH.fit, False), ("B_right", HH.fit_mirror, True),
                                 ("A_left", HH_A.fit, False),
                                 ("A_right", HH_A.fit_mirror, True)):
            r = fn(X, Y, g, LAMBDAS)
            out[name] = {"endpoint": float(r["u_hat"] if mirror else r["theta_hat"]),
                         "rel_ise": rel_ise(r["b_hat"], dd["b"], g)}
        profs = {}
        for mirror in (False, True):
            p = HH_A.truncation_profile(X, Y, g, mirror=mirror)
            profs[mirror] = p
            out["profile_" + ("right" if mirror else "left")] = {
                "theta_hat": p["theta_hat"], "theta_1se": p["theta_1se"],
                "mse_full": p["mse_full"],
                "thetas": [round(float(t), 4) for t in p["thetas"]],
                "mse": [float(x) for x in p["mse"]]}
        out["continuous_theta_hat"] = N[f"t1_{key}"]["theta_hat"]
        out["continuous_rel_ise"] = N[f"t1_{key}"]["rel_ise_cont"]
        N[f"t3_{key}"] = out
        fits[key]["profiles"] = profs
        log(f"task 3 existing estimators: {key}")

    # -------------------------------------------- figure 7: the existing profile, no null -
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, key, title in ((axes[0], "cont", "(a)  sine $\\to$ ramp, continuous at 0.5"),
                           (axes[1], "jump", "(b)  sine $\\to$ ramp, jump at 0.5")):
        for mirror, ls, lab in ((False, "--", "window $[0, \\theta]$ kept"),
                                (True, "-", "window $[u, 1]$ kept")):
            p = fits[key]["profiles"][mirror]
            ok = np.isfinite(p["mse"])
            ax.plot(p["thetas"][ok], p["mse"][ok], ls, color=C_A, lw=1.4, marker="o", ms=3,
                    label=lab)
        ax.axhline(fits[key]["profiles"][True]["mse_full"], color=style.CONTROL, lw=1.0,
                   ls=":", label="no truncation")
        ax.axvline(0.5, color=style.INK, ls=":", lw=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("truncation point (the window's free end)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("held-out MSE (log scale)")
    axes[0].legend(loc="upper center", fontsize=7.2)
    fig.tight_layout()
    style.save_figure(fig, "ext_t3_profiles", dpi=200)
    plt.close(fig)
    log("figure 7 (task 3 profiles)")

    with open(paths.figure_path("ext_numbers.json"), "w") as fh:
        json.dump(N, fh, indent=1, default=float)
    log(f"wrote {paths.figure_path('ext_numbers.json')}")


if __name__ == "__main__":
    main()
