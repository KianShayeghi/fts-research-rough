"""Multiple-breakpoint estimation by recursively applying Hall & Hooker's method.

`HH.py` and `HH_A.py` each estimate a single split of the domain into one informative window
and one null region. Where the true coefficient function alternates informative and null more
than once, iterating that single-split estimator recovers each break in turn: binary
segmentation, the standard way to turn a single-changepoint detector into a multiple-changepoint
one. `fdaSP.SBC`'s overlap group lasso already handles "any union of intervals" directly in one
fit (`region/README.md`); this module is the cheaper, more assumption-laden route to the same
place from the estimator already validated in `truncFLR`, and it is the one that inherits
method B's own endpoint at each break rather than a group lasso's selection set.

At every active segment, "should this segment be split at all" has to be answered before
recursing, since `HH.fit`/`fit_mirror` return *some* theta even on a segment with no real break
in it. As in `HH_continuous`, that question is answered by cross-validated prediction error
rather than a fabricated extension of section 3.2 to a model the paper never poses: fit the
segment three ways -- no split at all (`_common.pilot_fpca`), truncated from the left
(`HH.fit`), truncated from the right (`HH.fit_mirror`) -- score each by k-fold CV, and only
split if the better orientation beats the no-split fit by more than the no-split fit's own
standard error (the same one-standard-error caution `HH_A.truncation_profile`'s elbow and
`HH_continuous`'s CV selection already use to avoid chasing noise). This is also the orientation
contest `bench.study.run_cell` already runs to decide which end of a segment carries the null
region, reused here per segment instead of once globally.

Recursion continues into *both* pieces a split produces, not only the one the winning fit called
informative -- the "null" piece can itself turn informative further along, which is exactly the
alternation this module exists to find. It stops on a piece when: the piece has too few grid
points to support a pilot fit at all, `max_depth` is reached, or the split-vs-no-split contest
does not clear its threshold.

The search for breakpoints uses whichever local intercept each candidate fit happens to have,
since only the *location* of a break is being tested there; the final coefficient estimate needs
one intercept shared across the whole domain, so once the breakpoints are fixed, `binary_segment`
finishes with one joint refit (`joint_refit`) of a local empirical-FPC basis per leaf segment
against a single design matrix -- the many-breakpoint analogue of `ab_methodB`'s refit at a
single theta_hat. That refit also needs its own component-count selection
(`select_joint_refit`, by BIC): a fixed cap fits every leaf with the same number of components
regardless of how much signal it carries, which reproduces noise almost exactly in a leaf that
is truly null.
"""

import numpy as np

from . import HH
from ._common import DegenerateSelection, pilot_fpca, trunc_fpca

__all__ = ["binary_segment", "segment_contest", "joint_refit", "select_joint_refit"]


def _cap_budget(n_basis, n_components, n_pts, n_obs):
    """Same floor `HH.ab_methodB` applies to a short retained sub-grid (HH.py:317-324),
    needed here because a segment's own pilot call is not capped by `HH.fit`/`fit_mirror`."""
    n_basis_c = min(n_basis, max(n_pts - 1, HH.PILOT_ORDER))
    n_components_c = max(2, min(n_components, n_pts - 1, n_obs - 1))
    return n_basis_c, n_components_c


def _predict_nosplit(X_tr, Y_tr, X_te, grid, n_basis, n_components):
    n_basis_c, n_components_c = _cap_budget(n_basis, n_components, grid.size, len(Y_tr))
    a, b, _ = pilot_fpca(X_tr, Y_tr, grid, n_basis=n_basis_c, n_components=n_components_c)
    return a + np.trapezoid(b[None, :] * X_te, grid, axis=1)


def _predict_split(X_tr, Y_tr, X_te, grid, lambdas, mirror, n_basis, n_components, n_fourier):
    n_basis_c, n_components_c = _cap_budget(n_basis, n_components, grid.size, len(Y_tr))
    fit_fn = HH.fit_mirror if mirror else HH.fit
    res = fit_fn(X_tr, Y_tr, grid, lambdas, n_basis=n_basis_c, n_components=n_components_c,
                 n_fourier=n_fourier)
    a_hat, b_hat = res["a_hat"], res["b_hat"]
    # `fit_mirror` maps back with u_hat = 1 - theta_hat, which is the reflection about the
    # midpoint of [0, 1] only. On a segment [lo, hi] the reversed columns reflect about the
    # segment's own midpoint, so the left endpoint of the kept window is lo + hi - theta_hat;
    # reading `u_hat` there puts the breakpoint off the segment's own break by lo + hi - 1.
    breakpoint_ = grid[0] + grid[-1] - res["theta_hat"] if mirror else res["theta_hat"]
    # The fit's own quadrature: the retained window is [lo, theta_hat] going forward, [u, hi]
    # mirrored -- integrating the other side too would straddle the jump with a spurious half
    # panel, the same correction `HH_A.trunc_predict` documents.
    interval = grid >= breakpoint_ - 1e-12 if mirror else grid <= res["theta_hat"]
    pred = a_hat + np.trapezoid(b_hat[interval] * X_te[:, interval], grid[interval], axis=1)
    return pred, breakpoint_


def _cv_errors(predict, X, Y, folds):
    """Per-fold CV errors, over the fold labels in `folds`, of
    `predict(X_tr, Y_tr, X_te) -> (n_te,) predictions` (or `(pred, aux)`, `aux` taken from the
    last fold only, for candidates that also report a fitted breakpoint).

    `folds` is taken as given rather than drawn here, so that the no-split and both truncated
    candidates in `segment_contest` are scored fold-for-fold on the *same* partition of the
    segment's rows: `segment_contest` needs the errors paired, not just their marginal means, to
    have any power to detect a real but modest split (see there).

    Returns (errs, aux), `errs` a `(len(unique(folds)),)` array, or None if any fold raises
    `DegenerateSelection`.
    """
    errs, aux = [], None
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        try:
            out = predict(X[tr], Y[tr], X[te])
        except DegenerateSelection:
            return None
        pred, aux = out if isinstance(out, tuple) else (out, None)
        errs.append(np.mean((Y[te] - pred) ** 2))
    return np.asarray(errs), aux


def segment_contest(X, Y, grid, lambdas, n_basis=20, n_components=10, n_fourier=1,
                    n_folds=5, rng=0):
    """Decide whether one segment should be split, and where.

    Scores the no-split pilot fit and both truncation orientations by k-fold CV MSE on one
    shared fold partition, and splits only if the better orientation beats the no-split fit by
    more than one standard error of their *paired, fold-by-fold* difference.

    Pairing matters here and a plain comparison of the two candidates' marginal CV means does
    not have the power to use: a BIC-selected FPCA pilot with up to `n_components` components is
    itself flexible enough to track a step-like true coefficient function fairly closely, so its
    CV error sits close to the noise floor next to a genuine single break, and the *difference*
    between the two candidates' errors on the same folds is far less noisy than either error on
    its own -- both are riding the same draws of test curves and the same noise realisation, so
    that shared variance cancels in the difference and does not cancel in two separate standard
    errors compared against each other. On a clean single-break design (indicator coefficient,
    snr=40, five-ish folds) the paired one-standard-error test finds the break comfortably; the
    unpaired version of the same comparison does not clear its own threshold at all.

    Returns the winning breakpoint (float) if the segment should split, or None if it should not
    (either because no candidate clears the threshold, or because every truncated candidate was
    degenerate on this segment).

    PARAMETERS
    X, Y, grid : the segment's own curves, responses and sub-grid (already restricted to it);
    lambdas    : array, candidate penalties passed to `HH.fit`/`fit_mirror`;
    n_folds    : int, CV folds;
    rng        : seed for the one fold partition all three candidates are scored on.
    """
    grid = np.asarray(grid, dtype=float)
    folds = np.random.default_rng(rng).permutation(len(Y)) % n_folds

    nosplit = _cv_errors(
        lambda Xtr, Ytr, Xte: _predict_nosplit(Xtr, Ytr, Xte, grid, n_basis, n_components),
        X, Y, folds,
    )
    if nosplit is None:
        return None
    err_no, _ = nosplit

    best = None
    for mirror in (False, True):
        scored = _cv_errors(
            lambda Xtr, Ytr, Xte, mirror=mirror: _predict_split(
                Xtr, Ytr, Xte, grid, lambdas, mirror, n_basis, n_components, n_fourier
            ),
            X, Y, folds,
        )
        if scored is None:
            continue
        err_split, breakpoint_ = scored
        if best is None or err_split.mean() < best[0].mean():
            best = (err_split, breakpoint_)

    if best is None:
        return None
    err_split, breakpoint_ = best
    diff = err_no - err_split  # positive on a fold means the split predicted it better
    se_diff = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
    if diff.mean() > se_diff:
        return float(breakpoint_)
    return None


def joint_refit(X, Y, grid, breakpoints, n_components=10):
    """One joint least-squares refit given a fixed set of breakpoints: a local empirical-FPC
    basis per leaf segment (`_common.trunc_fpca`, no continuity constraint between leaves --
    these are genuine jumps, not the kink `HH_continuous` models), sharing one intercept across
    the whole domain, the many-breakpoint analogue of `HH.ab_methodB`'s refit at a single
    theta_hat.

    Leaf boundaries are half-open on the left (t > previous breakpoint) except the first leaf,
    matching the `grid <= theta` convention `HH.py`/`HH_A.py` use for a single split, so every
    grid point belongs to exactly one leaf.

    `n_components` is a *cap*, not a target -- every leaf gets up to that many components
    regardless of how much signal it actually carries, so calling this directly with a generous
    cap on a leaf that is truly null overfits noise with as many free coefficients as the cap
    allows (ten components fit to twenty-odd points of pure noise reproduces that noise almost
    exactly, in sample). `binary_segment` does not call this directly for that reason; it uses
    `select_joint_refit`, which BIC-selects one component count shared across every leaf, the
    many-leaf analogue of `HH_A.methodA_bic`'s own per-split BIC.

    Returns a_hat (float), b_hat ((N,) array), leaf_masks (list of (N,) boolean arrays, left to
    right) and n_params (int, the total number of coefficients fitted across all leaves,
    excluding the intercept -- what a BIC computed over this fit should charge for).
    """
    grid = np.asarray(grid, dtype=float)
    n = len(Y)
    edges = [grid[0]] + sorted(float(b) for b in breakpoints) + [grid[-1]]

    leaf_masks, blocks = [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (grid >= lo) & (grid <= hi) if i == 0 else (grid > lo) & (grid <= hi)
        leaf_masks.append(mask)
        m = max(1, min(n_components, int(mask.sum()) - 1, n - 1))
        phi, _ = trunc_fpca(X, grid, mask, m)
        grid_i = grid[mask]
        scores = np.trapezoid(phi[:, None, :] * X[None, :, mask], grid_i, axis=2).T
        blocks.append((mask, phi, scores))

    design = np.column_stack([np.ones(n)] + [b[2] for b in blocks])
    coef, *_ = np.linalg.lstsq(design, Y, rcond=None)

    a_hat = float(coef[0])
    b_hat = np.zeros_like(grid)
    j = 1
    n_params = 0
    for mask, phi, _ in blocks:
        k = phi.shape[0]
        b_hat[mask] = coef[j:j + k] @ phi
        j += k
        n_params += k
    return a_hat, b_hat, leaf_masks, n_params


def select_joint_refit(X, Y, grid, breakpoints, n_components=10, m_range=None):
    """`joint_refit`, with the shared per-leaf component count chosen by BIC instead of fixed.

    Refits at every candidate `m` in `m_range` (default `1..n_components`) and keeps the one
    minimising `n log(rss/n) + (n_params + 1) log(n)` -- `HH_A.methodA_bic`'s own criterion,
    generalised from one split's component count to one shared across every leaf of a many-break
    partition. This is the step that keeps a truly null leaf from being fit with as many
    components as a strongly informative one just because both share the same cap; a null leaf's
    RSS stops improving past m=1 or so, so BIC's `(m+1) log(n)` penalty turns it back down while
    an informative leaf's RSS keeps falling enough to be worth the same extra parameter.

    Returns the same four values as `joint_refit`, plus m_hat (int, the selected count) and bics
    (dict m -> BIC).
    """
    if m_range is None:
        m_range = range(1, n_components + 1)
    n = len(Y)

    best, bics = None, {}
    for m in m_range:
        a_hat, b_hat, leaf_masks, n_params = joint_refit(X, Y, grid, breakpoints, n_components=m)
        pred = a_hat + np.trapezoid(b_hat[None, :] * X, grid, axis=1)
        rss = np.sum((Y - pred) ** 2)
        bic = n * np.log(rss / n) + (n_params + 1) * np.log(n)
        bics[m] = float(bic)
        if best is None or bic < best[0]:
            best = (bic, m, a_hat, b_hat, leaf_masks, n_params)

    _, m_hat, a_hat, b_hat, leaf_masks, n_params = best
    return a_hat, b_hat, leaf_masks, n_params, m_hat, bics


def binary_segment(X, Y, grid, lambdas, n_basis=20, n_components=10, n_fourier=1,
                   n_folds=5, max_depth=4, min_points=8, rng=0):
    """Binary segmentation: recursively split the domain with `segment_contest`, then refit the
    whole coefficient function jointly at the resulting breakpoints (`joint_refit`).

    Returns a dict with a_hat, b_hat, breakpoints (sorted list of floats), leaf_masks (left to
    right, matching `breakpoints`) and contests (one dict per segment actually tested, in the
    order visited, recording its bounds, depth and the breakpoint found or None).

    PARAMETERS
    max_depth  : int, longest chain of splits from the root; stops runaway recursion on data
                 with no real structure left rather than relying on the contest alone;
    min_points : int, a segment shorter than this is a leaf without running the contest --
                 too few points to support a pilot fit meaningfully;
    rng        : seed for every segment's CV fold assignment. A fixed integer draws the same
                 fold pattern (scaled by grid position) at every segment so the whole tree is
                 reproducible from one seed.
    """
    grid = np.asarray(grid, dtype=float)
    breakpoints, contests = [], []
    queue = [(np.ones(grid.size, dtype=bool), 0)]

    while queue:
        mask, depth = queue.pop()
        n_pts = int(mask.sum())
        lo, hi = grid[mask][0], grid[mask][-1]

        if n_pts < min_points or depth >= max_depth:
            continue

        # A distinct sub-seed per segment (by position and depth) avoids every segment at a
        # given depth reusing the identical fold assignment relative to its own rows.
        seg_rng = hash((round(lo, 10), round(hi, 10), depth, rng)) % (2 ** 32)
        breakpoint_ = segment_contest(X[:, mask], Y, grid[mask], lambdas, n_basis=n_basis,
                                      n_components=n_components, n_fourier=n_fourier,
                                      n_folds=n_folds, rng=seg_rng)
        contests.append({"lo": float(lo), "hi": float(hi), "depth": depth,
                         "breakpoint": breakpoint_})
        if breakpoint_ is None:
            continue

        breakpoints.append(breakpoint_)
        left = mask & (grid <= breakpoint_)
        right = mask & (grid > breakpoint_)
        queue.append((left, depth + 1))
        queue.append((right, depth + 1))

    breakpoints = sorted(breakpoints)
    a_hat, b_hat, leaf_masks, _, m_hat, bics = select_joint_refit(
        X, Y, grid, breakpoints, n_components=n_components
    )
    return {"a_hat": a_hat, "b_hat": b_hat, "breakpoints": breakpoints,
            "leaf_masks": leaf_masks, "contests": contests, "m_hat": m_hat, "bics": bics}
