"""Data-generating process and replication of simulation study from Stefanucci,
Bernardi & Canale (2025)"""

import numpy as np
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold

from ..core.bases import bspline_basis
from .SBC import cv_fit


def psi1(t):
    """Piecewise sine vanishing on [0.3, 0.7]
    """
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    low = t <= 0.3
    upp = t >= 0.7
    out[low] = 2 * (1 - t[low]) * np.sin(2 * np.pi * (t[low] + 0.2))
    out[upp] = 2 * t[upp] * np.sin(2 * np.pi * (t[upp] - 0.2))
    return out


def psi2(t):
    """Rescaled Epanechnikov kernel, support on [0.25, 0.75]
    """
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    m = (t >= 0.25) & (t <= 0.75)
    out[m] = 1 - (4 * (t[m] - 0.5)) ** 2
    return out


def psi3(t):
    """No null region, control case
    """
    t = np.asarray(t, float)
    return 10 * (t - 0.5) ** 3


TRUE_FUNCTIONS = {
    "psi1": (psi1, lambda g: (g >= 0.3) & (g <= 0.7)),
    "psi2": (psi2, lambda g: (g < 0.25) | (g > 0.75)),
    "psi3": (psi3, lambda g: np.zeros_like(g, dtype=bool)),
}


def simulate(psi_fn, n=50, m=100, n_knots_x=25, order=4, snr=4.0, rng=123):
    """Generates a dataset as in the paper.

    Returns curves, responses, and true coefficient function

    PARAMETERS
    psi_fn    : callable, true coeff function;
    n         : int, sample size;
    m         : int, grid size;
    n_knots_x : int, internal knots for basis used to generate curves;
    snr       : float, signal-to-noise ratio.
    """
    rng = np.random.default_rng(rng)
    grid = np.linspace(0.0, 1.0, m)

    k_x = n_knots_x + order  # Basis size for generating basis
    Phi_x = bspline_basis(grid, n_basis=k_x, order=order)  # Generate basis
    coef = rng.standard_normal((n, k_x))  # Generate basis coefficients
    X = coef @ Phi_x.T  # Generate curves

    beta_true = psi_fn(grid)  # True coefficient function
    signal = np.trapezoid(X * beta_true, grid, axis=1)  # Generate noiseless response
    sigma = np.sqrt(np.var(signal) / snr)  # Signal-to-noise ratio
    y = signal + sigma * rng.standard_normal(n)  # Generate noisy response
    return X, y, grid, beta_true


def ise(beta_hat, beta_true, grid, mask=None):
    """Integrated squared error int (beta_hat - beta_true)^2, optionally over a sub-region.

    The trapezium weights are formed on the whole grid and the region restriction applied to
    them, so the ISE over T0 and over its complement sum to the ISE over the whole domain for
    any `mask`. Passing only the masked points to a single trapezium rule does not: where the
    region is a union of intervals, as psi2's null set (t < 0.25) | (t > 0.75) is, that joins
    the last point of one interval to the first of the next and adds a panel spanning the gap
    between them. Returns the ISE (float), or NaN where the mask is empty.

    PARAMETERS
    mask : (N,) boolean array or None, region to restrict the integral to.
    """
    sq = (np.asarray(beta_hat, float) - np.asarray(beta_true, float)) ** 2
    grid = np.asarray(grid, float)
    w = np.zeros_like(grid)
    w[1:] += np.diff(grid)
    w[:-1] += np.diff(grid)
    w *= 0.5
    if mask is None:
        return float(w @ sq)
    mask = np.asarray(mask, bool)
    if not mask.any():  # If every element is False... i.e. no support
        return np.nan
    return float(w[mask] @ sq[mask])


def fpca_reg(X, y, grid, max_components=10, cv=5, rng=0):
    """FPCA regression, choosing the number of components by K-fold CV.
    Returns beta_hat and the chosen number of components.

    PARAMETERS
    X             : array, curves evaluated on grid;
    max_components: int, largest number of FPCs considered;
    cv            : int, number of CV folds.
    """
    X = np.atleast_2d(np.asarray(X, float))
    y = np.asarray(y, float).ravel()
    n = X.shape[0]
    K = min(max_components, n - 1)

    fd = FDataGrid(data_matrix=X, grid_points=grid)
    fpca = FPCA(n_components=K)
    scores = fpca.fit_transform(fd)  # One fit; component count is chosen by CV below
    eigfun = np.squeeze(fpca.components_.data_matrix, axis=-1)

    # Choose number of components by K-fold CV on held-out MSE
    kf = KFold(n_splits=cv, shuffle=True, random_state=rng)
    cv_mse = np.zeros(K)
    for k in range(1, K + 1):
        errs = [
            np.mean((y[te] - LinearRegression().fit(scores[tr, :k], y[tr])
                     .predict(scores[te, :k])) ** 2)
            for tr, te in kf.split(scores)
        ]
        cv_mse[k - 1] = np.mean(errs)
    k_opt = int(np.argmin(cv_mse)) + 1

    reg = LinearRegression().fit(scores[:, :k_opt], y)
    beta_hat = reg.coef_ @ eigfun[:k_opt]
    return beta_hat, k_opt


def fit_comps(X, y, grid, W, Phi, alphas=None):
    """Lasso / Ridge / Elastic Net / regression splines on the basis coefficients, plus
    FPCA regression on the raw curves.

    PARAMETERS
    alphas : array, regularisation coefficient candidates
    """
    if alphas is None:
        alphas = np.geomspace(1e-4, 1e1, 30)
    cv = 5  # Number of folds
    out = {}

    lasso = LassoCV(alphas=alphas, fit_intercept=False, cv=cv, max_iter=10000).fit(W, y)
    out["Lasso"] = Phi @ lasso.coef_

    ridge = RidgeCV(alphas=alphas, fit_intercept=False, cv=cv).fit(W, y)
    out["Ridge"] = Phi @ ridge.coef_

    enet = ElasticNetCV(
        alphas=alphas, l1_ratio=[0.25, 0.5, 0.75], fit_intercept=False, cv=cv, max_iter=10000
    ).fit(W, y)
    out["Enet"] = Phi @ enet.coef_

    spl = LinearRegression(fit_intercept=False).fit(W, y)
    out["Splines"] = Phi @ spl.coef_

    out["FPCA"], _ = fpca_reg(X, y, grid, cv=cv)

    return out


def run_sim(which=("psi1", "psi2", "psi3"),
            n_rep=100, n=50, m=100, n_basis=15, order=4, snr=4.0,
            n_folds=5, n_lambda=30, seed=123, verbose=True):
    """Run the full simulation study: for each true coefficient function, fit the OGL and its
    five competitors on `n_rep` independently generated datasets and record every ISE.

    Returns a nested dictionary, function -> method -> metric -> (n_rep,) array, where the
    metrics are the ISE over the whole domain ("ise"), over the true null region ("ise_T0"),
    and over its complement ("ise_T1"). `summarise` reduces it to the paper's Figure 62.2.

    PARAMETERS
    which  : tuple of str, which of `TRUE_FUNCTIONS` to run;
    n_rep  : int, independent datasets per function;
    n_folds, n_lambda : int, passed to `cv_fit` for the OGL's own lambda selection.
    """
    results = {}
    for name in which:
        psi_fn, null_fn = TRUE_FUNCTIONS[name]
        # Ogl = overlapping group lasso
        methods = ["Ogl", "Lasso", "Ridge", "Enet", "Splines", "FPCA"]
        # Full ISE, null region ISE, supported region ISE
        acc = {mname: {"ise": [], "ise_T0": [], "ise_T1": []} for mname in methods}

        for r in range(n_rep):
            X, y, grid, beta_true = simulate(
                psi_fn, n=n, m=m, order=order, snr=snr, rng=seed * 10000 + r
            )

            T0 = null_fn(grid)
            T1 = ~T0

            ogl = cv_fit(
                X, y, grid, n_basis=n_basis, order=order, n_folds=n_folds,
                n_lambda=n_lambda, rng=r
            )

            # Gather all coefficient functions...
            betas = {"Ogl": ogl["beta"]}
            betas.update(fit_comps(X, y, grid, ogl["W"], ogl["Phi"]))

            # Append all ISEs for each method and their beta_hat to acc.
            for mname, bh in betas.items():
                acc[mname]["ise"].append(ise(bh, beta_true, grid))
                acc[mname]["ise_T0"].append(ise(bh, beta_true, grid, T0))
                acc[mname]["ise_T1"].append(ise(bh, beta_true, grid, T1))

            # Progress, flush=True makes it print immediately
            if verbose and (r + 1) % 10 == 0:
                print(f"{name}: {r + 1}/{n_rep}", flush=True)

        # Freeze the accumulated lists into array, prepared for summary statistics
        # Function -> Method -> Metric
        results[name] = {
            mname: {k: np.asarray(v) for k, v in d.items()}
            for mname, d in acc.items()
        }
    return results


def summarise(results):
    """Mean and sd of ISE by function and method (Fig 62.2 from paper)
    """
    rows = []
    for fname, per_method in results.items():
        for mname, d in per_method.items():
            rows.append({
                "function": fname,
                "method": mname,
                # Monte Carlo estimate of E[ISE]
                "ise_mean": float(np.mean(d["ise"])),
                # Square root of the unbiased variance estimator
                "ise_sd": float(np.std(d["ise"], ddof=1)),
                # Same for other metrics, skip NaN
                "ise_T0_mean": float(np.nanmean(d["ise_T0"])),
                "ise_T1_mean": float(np.nanmean(d["ise_T1"])),
            })
    return rows
