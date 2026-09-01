"""Four ways of choosing a null set, scored on whether they find the one that is there.

`fdaSP.SBC_sim` scores the overlap group lasso against its competitors on integrated squared
error, which is the paper's own table and answers the estimation question. It does not answer
the selection one, and the two come apart: `notebooks/LocSpaFLR_f2s.ipynb` finds the
cross-validated fit with the lowest prediction error returning a null set of length zero on
every design there, so a rule can win the reported comparison outright while recovering none
of the structure the estimator exists to recover.

This module scores the recovery instead. Every rule is run on one dataset, over one penalty
path, and differs from the others in nothing but how it reads that path; what is recorded is
the measure of the domain each one calls null where the coefficient function is not zero, and
the measure of true null set each one leaves in the support. Integrated squared error is
recorded alongside, twice -- as fitted, and after `criteria.relaxed_fit` removes the
shrinkage -- so that the estimation question can be read off the same run and the two answers
compared rather than assumed to agree.

The four designs are the three of Stefanucci, Bernardi and Canale (2025), whose third has no
null region at all and is the control against inventing one, and a fourth with the null set
at one end. That last is not decoration: it is the shape the application has, where the
overnight jump is regressed on the session before it and the hypothesis is that only the
close of that session matters, and it is also Hall and Hooker's (2016) own configuration, so
it is where a locally sparse estimator and a truncation estimator are asking the same
question.
"""

import numpy as np
import pandas as pd

from ..fdaSP import SBC, SBC_sim
from . import criteria, stability

__all__ = ["psi_late", "DESIGNS", "RULES", "run_once", "run_study", "summarise"]


def psi_late(t):
    """A coefficient function supported on [0.7, 1] and zero before it. Returns an array.

    The right-anchored window, which is Hall and Hooker's (2016) configuration and the one
    `jump` asks about: the response depends on the end of the curve and on nothing earlier.
    It meets zero smoothly at 0.7, so the boundary is not a jump and cannot be found by
    looking for one.

    PARAMETERS
    t : array, the domain.
    """
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    late = t >= 0.7
    out[late] = 2.0 * np.sin(np.pi * (t[late] - 0.7) / 0.3)
    return out


# Each design is a coefficient function and the null set it has, as a mask on the grid.
DESIGNS = {
    "psi1": (SBC_sim.psi1, lambda g: (g >= 0.3) & (g <= 0.7)),
    "psi2": (SBC_sim.psi2, lambda g: (g < 0.25) | (g > 0.75)),
    "psi3": (SBC_sim.psi3, lambda g: np.zeros_like(g, dtype=bool)),
    "late": (psi_late, lambda g: g < 0.7),
}

RULES = ("cv", "cv-1se", "ebic", "stability", "stability-adaptive")


def _one_sided_step(cv_mean, cv_se):
    """Index of the largest penalty whose cross-validated error is within one standard error
    of the smallest. The path descends, so that is the first such index."""
    best = int(np.argmin(cv_mean))
    return int(np.argmax(cv_mean <= cv_mean[best] + cv_se[best]))


def _score(name, null_hat, beta_hat, X, y, grid, beta_true, null_true, n_basis, order,
           extra=None):
    """One row of the study: where a rule put the null set, and what it cost."""
    groups = stability.knot_interval_of(grid, n_basis, order)
    null_groups = np.array([bool(np.all(null_hat[groups == b]))
                            for b in range(n_basis - order + 1)])
    relaxed = criteria.relaxed_fit(X, y, grid, null_groups, n_basis=n_basis, order=order)
    row = {"rule": name}
    row.update(criteria.support_error(null_hat, null_true, grid))
    row["ise"] = SBC_sim.ise(beta_hat, beta_true, grid)
    row["ise_relaxed"] = SBC_sim.ise(relaxed["beta"], beta_true, grid)
    row.update(extra or {})
    return row


def run_once(design="psi1", n=100, m=100, snr=4.0, seed=0, n_basis=15, order=4,
             n_lambda=30, n_pairs=25, n_folds=5, pfer=1.0, pner=1.0, gamma=1.0):
    """Every rule on one simulated dataset. Returns a list of dicts, one per rule.

    All five read the same penalty path, so nothing separates them but the reading. The two
    cross-validated rules share one call, since the one-standard-error index is a function of
    the cross-validation curve the other already computed; the adaptive rule is the only one
    to refit the path, having changed the penalty it is a path of.

    PARAMETERS
    design            : str, a key of `DESIGNS`;
    n, m, snr, seed   : as `fdaSP.SBC_sim.simulate` takes them;
    n_basis, order    : int, the estimation basis;
    n_lambda          : int, penalties on the path;
    n_pairs           : int, complementary pairs for the two stability rules;
    n_folds           : int, cross-validation folds;
    pfer, pner        : float, the two error budgets, passed to `stability.select`;
    gamma             : float, passed to `criteria.adaptive_weights`.
    """
    psi_fn, null_fn = DESIGNS[design]
    X, y, grid, beta = SBC_sim.simulate(psi_fn, n=n, m=m, snr=snr, rng=seed)
    null_true = null_fn(grid)
    shared = dict(n_basis=n_basis, order=order)

    path = SBC.fit_path(X, y, grid, n_lambda=n_lambda, **shared)
    lambdas = path["lambdas"]
    cv = SBC.cv_fit(X, y, grid, lambdas=lambdas, n_folds=n_folds, rng=seed, **shared)
    steps = {"cv": int(np.argmin(cv["cv_mean"])),
             "cv-1se": _one_sided_step(cv["cv_mean"], cv["cv_se"])}
    steps["ebic"] = criteria.ebic(path, y)["step"]

    rows = []
    for rule, step in steps.items():
        rows.append(_score(rule, ~path["supports"][step], path["betas"][step], X, y, grid,
                           beta, null_true, n_basis, order,
                           extra={"lam": float(lambdas[step]), "undecided": 0.0,
                                  "bound_false": np.nan, "bound_missed": np.nan}))

    for rule in ("stability", "stability-adaptive"):
        weights = None
        these = lambdas
        if rule.endswith("adaptive"):
            weights = criteria.adaptive_weights(X, y, grid, gamma=gamma, **shared)
            these = None                                # the weights move lambda_max
        curves = stability.selection_curves(X, y, grid, lambdas=these, n_pairs=n_pairs,
                                            group_weights=weights, n_lambda=n_lambda,
                                            rng=seed, **shared)
        chosen = stability.select(curves, pfer=pfer, pner=pner)
        relaxed = criteria.relaxed_fit(X, y, grid, chosen["null"], **shared)
        rows.append(_score(rule, chosen["null_grid"], relaxed["beta"], X, y, grid, beta,
                           null_true, n_basis, order,
                           extra={"lam": np.nan,
                                  "undecided": criteria.measure(chosen["undecided_grid"],
                                                                grid),
                                  "bound_false": chosen["bound_false"],
                                  "bound_missed": chosen["bound_missed"]}))

    for row in rows:
        row.update({"design": design, "seed": seed, "n": n, "snr": snr})
    return rows


def run_study(designs=None, n_reps=25, n=100, m=100, snr=4.0, n_basis=15, order=4,
              n_lambda=30, n_pairs=25, n_folds=5, pfer=1.0, pner=1.0, gamma=1.0,
              progress=None):
    """`run_once` over every design and seed. Returns a long DataFrame, one row per rule.

    PARAMETERS
    designs  : sequence or None, keys of `DESIGNS`; all of them when None;
    n_reps   : int, seeds per design, run as 0, 1, ..., n_reps - 1;
    progress : callable or None, called with (design, seed) before each replication, so a
               script can report where it is without this module printing anything;
    n, m, snr, n_basis, order, n_lambda, n_pairs, n_folds, pfer, pner, gamma :
               as `run_once` takes them.
    """
    designs = tuple(DESIGNS) if designs is None else tuple(designs)
    rows = []
    for design in designs:
        for seed in range(int(n_reps)):
            if progress is not None:
                progress(design, seed)
            rows += run_once(design=design, n=n, m=m, snr=snr, seed=seed, n_basis=n_basis,
                             order=order, n_lambda=n_lambda, n_pairs=n_pairs,
                             n_folds=n_folds, pfer=pfer, pner=pner, gamma=gamma)
    return pd.DataFrame(rows)


def summarise(frame, statistic="median"):
    """The study as a table, one row per design and rule. Returns a DataFrame.

    Medians by default rather than means, because a locally sparse fit can miss a support
    entirely on one replication and take its integrated squared error with it, which is the
    behaviour `notebooks/HH_vs_SBC.ipynb` reports for method A and the reason its tables are
    medians too. `false_null_rate` is the share of replications on which a rule called any
    informative domain null, which a median over a measure hides.

    PARAMETERS
    frame     : DataFrame from `run_study`;
    statistic : str, an aggregation `pandas` understands, applied to the measures.
    """
    columns = ["false_null", "missed_null", "symmetric", "null_len", "jaccard", "ise",
               "ise_relaxed", "undecided", "bound_false", "bound_missed"]
    grouped = frame.groupby(["design", "rule"], sort=False)
    out = grouped[columns].agg(statistic)
    out["false_null_rate"] = grouped["false_null"].apply(lambda s: float((s > 1e-12).mean()))
    out["n_reps"] = grouped.size()
    order = [r for r in RULES if r in set(frame["rule"])]
    return out.reset_index().set_index(["design", "rule"]).reindex(
        pd.MultiIndex.from_product([list(dict.fromkeys(frame["design"])), order],
                                   names=["design", "rule"])).dropna(how="all")
