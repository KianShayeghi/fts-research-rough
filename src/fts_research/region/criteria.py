"""What the null set is chosen by, and how a choice of it is scored.

`stability` is one answer to "which penalty, and which null set". This module holds the
others it has to be measured against, and the measurements themselves, so that a comparison
runs every rule over one path and one dataset and differs in nothing else.

The rules divide by what they are trying to do. Cross-validation and the one-standard-error
rule minimise prediction error, which is not the same target: `notebooks/LocSpaFLR_f2s.ipynb`
and `notebooks/tecator_application.ipynb` find the cross-validated penalty returning a null
set of length zero on every design in this repository, because an exact zero is worth more
for describing a coefficient function than for predicting a response. An information
criterion is closer to the right target -- it prices a coefficient rather than a prediction
-- and `ebic` is the extended Bayesian criterion of Chen and Chen (2008), whose extra term
prices the *search* as well and is what makes it consistent when the number of candidates
grows with the sample. Stability selection is a third target again, and the only one that
returns an error bound with the set.

Two utilities travel with them. `adaptive_weights` reweights the penalty by a pilot fit, so
that a group the data already says is large is charged less for staying: the group analogue
of Zou's adaptive lasso, and the standard route to selection consistency when the
irrepresentable condition fails. `relaxed_fit` removes the shrinkage the penalty applied to
whatever it kept, which is what makes a coefficient function comparable across rules that
selected different sets at different penalties.

The scores are measures on the domain, not counts of coefficients. A null set is a subset of
`[0, 1]` and two estimates of it are compared by the Lebesgue measure of where they differ,
split into the half that matters -- domain called null where the coefficient function is not
zero -- and the half that is only conservative.

References
Chen, J. and Chen, Z. (2008). Extended Bayesian information criteria for model selection with
large model spaces. Biometrika 95(3), 759-771. doi:10.1093/biomet/asn034
Meinshausen, N. (2007). Relaxed lasso. Computational Statistics and Data Analysis 52(1),
374-393. doi:10.1016/j.csda.2006.12.019
Wang, H. and Leng, C. (2008). A note on adaptive group lasso. Computational Statistics and
Data Analysis 52(12), 5277-5286. doi:10.1016/j.csda.2008.05.006
Zou, H. (2006). The adaptive lasso and its oracle properties. Journal of the American
Statistical Association 101(476), 1418-1429. doi:10.1198/016214506000000735
"""

import numpy as np
from scipy import special

from ..core.quadrature import quadrature_weights
from ..fdaSP import SBC

__all__ = [
    "measure",
    "support_error",
    "adaptive_weights",
    "ebic",
    "relaxed_fit",
]


def measure(mask, grid):
    """Lebesgue measure of the part of the domain a grid mask covers. Returns a float.

    Taken under `core.quadrature`, the rule every integral in this repository is taken under,
    so a measure here and an integrated squared error in `fdaSP.SBC_sim` are on one scale and
    the measures of a set and its complement sum to the length of the domain.

    PARAMETERS
    mask : (N,) boolean array over the grid;
    grid : (N,) array, the sampling grid.
    """
    return float(quadrature_weights(np.asarray(grid, float)) @ np.asarray(mask, bool))


def support_error(null_hat, null_true, grid):
    """How far an estimated null set is from the true one, as measures on the domain.

    Returns a dict with `false_null`, the measure called null where the coefficient function
    is not zero; `missed_null`, the measure of true null set left in the support;
    `symmetric`, their sum, which is the measure of the symmetric difference; `null_len` and
    `true_len`, the measures of the two sets; and `jaccard`, their intersection over their
    union, which is 1 for an exact recovery and NaN where both sets are empty.

    The two halves are not interchangeable. Calling informative domain null asserts something
    false about the coefficient function; leaving true null set in the support only fails to
    assert something true, which is what every estimator without an exact zero does anyway.

    PARAMETERS
    null_hat  : (N,) boolean array, the estimated null set on the grid;
    null_true : (N,) boolean array, the true one;
    grid      : (N,) array, the sampling grid.
    """
    null_hat = np.asarray(null_hat, bool)
    null_true = np.asarray(null_true, bool)
    union = measure(null_hat | null_true, grid)
    intersection = measure(null_hat & null_true, grid)
    return {
        "false_null": measure(null_hat & ~null_true, grid),
        "missed_null": measure(~null_hat & null_true, grid),
        "symmetric": measure(null_hat ^ null_true, grid),
        "null_len": measure(null_hat, grid),
        "true_len": measure(null_true, grid),
        "jaccard": intersection / union if union > 0 else float("nan"),
    }


def adaptive_weights(X, y, grid, Z=None, n_basis=15, order=4, gamma=1.0, ridge=1e-6):
    """Group weights from a pilot fit, so the penalty charges least where the signal is.

    The overlap group lasso applies one penalty to every group, which is what makes it shrink
    the coefficients it keeps as hard as the ones it is deciding about, and what makes one
    `lambda` unable to serve both estimation and selection. Weighting group `b` by
    `||pilot_b|| ** -gamma` breaks that tie: a group the pilot already says is large is
    nearly unpenalised and one it says is small is heavily penalised, which is Zou's argument
    for the adaptive lasso and Wang and Leng's for its group form. The pilot is ridge on the
    basis-projected design, which is defined whatever `n_basis` and `n` are.

    Returns an (n_basis - order + 1,) array of weights with mean one, so that `lambda` keeps
    roughly the scale it had unweighted.

    PARAMETERS
    X, y, grid : as `fdaSP.SBC.fit_path` takes them;
    Z          : (n, p) array or None, unpenalised covariates, projected out of the pilot;
    n_basis    : int, basis functions;
    order      : int, spline order;
    gamma      : float, the exponent; 1 is Zou's default and 0 recovers the unweighted
                 penalty;
    ridge      : float, the pilot's penalty, relative to the mean squared column norm of the
                 design, so the same number means the same thing whatever the data's scale.
    """
    W, _ = SBC.bspline_design(X, grid, n_basis=n_basis, order=order)
    y = np.asarray(y, float).ravel()
    if Z is not None:
        Z = np.atleast_2d(np.asarray(Z, float))
        residualise = np.eye(y.size) - Z @ np.linalg.pinv(Z)
        W, y = residualise @ W, residualise @ y

    scale = float(np.mean(np.sum(W ** 2, axis=0)))
    pilot = np.linalg.solve(W.T @ W + ridge * scale * np.eye(W.shape[1]), W.T @ y)

    _, groups = SBC.pen_matrices(n_basis, order=order)
    norms = np.array([np.linalg.norm(pilot[g]) for g in groups])
    floor = 1e-12 * max(float(norms.max()), 1.0)
    weights = np.maximum(norms, floor) ** -float(gamma)
    return weights / weights.mean()


def ebic(path, y, Z=None, gamma=None):
    """The extended Bayesian information criterion along a fitted path.

    `n log(RSS / n) + k log n + 2 gamma log C(n_basis, k)`, with `k` the number of spline
    coefficients the penalty left non-zero. The first two terms are the ordinary BIC; the
    third is Chen and Chen's, and prices the number of models of that size the penalty could
    have chosen, which is what the ordinary criterion's flat prior over models ignores and
    what makes it over-select once the candidates outnumber what the sample can separate.

    `k` is the count of surviving coefficients, which is the group lasso's active-set
    analogue of the lasso's unbiased degrees of freedom rather than an exact degrees of
    freedom for an overlapping penalty. It undercounts the shrinkage the penalty applies
    within a surviving group, so the criterion is optimistic about a dense fit, in the same
    direction for every penalty on the path.

    The extra term is **not** monotone in `k`, and at the sizes a functional basis works at
    that matters. `log C(n_basis, k)` counts models of size `k` and so vanishes at both ends
    and peaks in the middle: with `n_basis = 15` it is largest near `k = 7` and exactly zero
    at `k = 15`, so raising `gamma` can move the choice from a sparse fit to the full one
    rather than towards sparsity. Chen and Chen state the criterion where the candidates far
    outnumber what the sample can support, and there `k` never approaches `n_basis` and the
    term is increasing over the whole reachable range. Here it can, and the criterion should
    be read knowing that the search it prices is a search over subsets of a small basis.

    Returns a dict with `values` over the path, `step` the minimising index, `lam`, `rss`
    and `k`.

    PARAMETERS
    path  : dict from `fdaSP.SBC.fit_path`;
    y     : (n,) array, the response the path was fitted to;
    Z     : (n, p) array or None, unpenalised covariates, as the path was given;
    gamma : float or None, the extra term's weight; Chen and Chen's
            `1 - log n / (2 log n_basis)`, floored at zero, when None.
    """
    y = np.asarray(y, float).ravel()
    n, n_basis = y.size, path["n_basis"]
    fitted = path["psis"] @ path["W"].T
    if Z is not None and path["gammas"] is not None:
        fitted = fitted + path["gammas"] @ np.atleast_2d(np.asarray(Z, float)).T
    rss = np.sum((y[None, :] - fitted) ** 2, axis=1)
    k = np.count_nonzero(path["psis"], axis=1)

    if gamma is None:
        gamma = max(0.0, 1.0 - np.log(n) / (2.0 * np.log(n_basis)))
    penalty = k * np.log(n) + 2.0 * gamma * special.gammaln(n_basis + 1) \
        - 2.0 * gamma * (special.gammaln(k + 1) + special.gammaln(n_basis - k + 1))
    values = n * np.log(np.maximum(rss, np.finfo(float).tiny) / n) + penalty

    step = int(np.argmin(values))
    return {"values": values, "step": step, "lam": float(path["lambdas"][step]),
            "rss": rss, "k": k, "gamma": float(gamma)}


def relaxed_fit(X, y, grid, null_groups, Z=None, n_basis=15, order=4):
    """Least squares on whatever the penalty kept, with the penalty then removed.

    One `lambda` decides both which groups survive and how hard the survivors are shrunk, so
    a coefficient function read straight off a penalised fit carries a bias whose size
    depends on the penalty that produced it -- and rules that select different sets select
    them at different penalties, which makes their estimates incomparable. Refitting without
    the penalty on the retained coefficients removes that, and is Meinshausen's relaxed
    lasso taken to its limit. Coefficients belonging only to null groups are held at zero, so
    the fit is exactly zero on the null set by construction.

    Returns a dict with `beta` the coefficient function on the grid, `psi` its spline
    coefficients, `gamma` the scalar coefficients or None, and `kept`, the boolean mask of
    free coefficients.

    PARAMETERS
    X, y, grid  : as `fdaSP.SBC.fit_path` takes them;
    null_groups : (n_basis - order + 1,) boolean array, the groups declared null;
    Z           : (n, p) array or None, unpenalised covariates, fitted alongside;
    n_basis     : int, basis functions;
    order       : int, spline order.
    """
    W, Phi = SBC.bspline_design(X, grid, n_basis=n_basis, order=order)
    y = np.asarray(y, float).ravel()
    _, groups = SBC.pen_matrices(n_basis, order=order)
    null_groups = np.asarray(null_groups, bool)
    if null_groups.size != len(groups):
        raise ValueError(f"null_groups has {null_groups.size} entries, there are "
                         f"{len(groups)} groups.")

    kept = np.ones(n_basis, dtype=bool)
    for b, g in enumerate(groups):
        if null_groups[b]:
            kept[g] = False

    design = W[:, kept]
    n_scalar = 0
    if Z is not None:
        Z = np.atleast_2d(np.asarray(Z, float))
        n_scalar = Z.shape[1]
        design = np.column_stack([design, Z])

    psi = np.zeros(n_basis)
    scalar = None
    if design.shape[1]:
        solved, *_ = np.linalg.lstsq(design, y, rcond=None)
        if n_scalar:
            psi[kept], scalar = solved[:-n_scalar], solved[-n_scalar:]
        else:
            psi[kept] = solved
    elif n_scalar:
        scalar = np.zeros(n_scalar)

    return {"beta": Phi @ psi, "psi": psi, "gamma": scalar, "kept": kept}
