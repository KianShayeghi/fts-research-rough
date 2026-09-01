"""The overnight jump equation: a scalar-on-function regression with its own lag.

    J_t = a + gamma J_{t-1} + <delta, X_{t-1}> + z_t' k + eta_t,   eta_t ~ iid N(0, sigma_J^2)

`J_t` is the overnight log jump and `X_{t-1}` the previous session's intraday curve, which is
exactly the split `fdaSP.intraday` makes of an open-to-open return: the curve is what the jump
omits and the jump is what the curve omits. Expanding `delta` in the empirical principal
components of the curves, `<delta, X_{t-1}> = d' psi_{t-1}` with `psi_{t-1}` the score vector,
and the model becomes a linear regression on the stacked regressor

    x_t = (1, J_{t-1}, psi_{t-1}', z_t')'.

Every one of those is measurable at the close of `t - 1`: the timing is what makes the problem
easy. `psi` enters lagged and `z` contemporaneously because `z_t` describes the session that
has already finished when the jump forms. With predetermined regressors and Gaussian
homoskedastic errors the conditional likelihood is that of a linear model, so the conditional
MLE is ordinary least squares and closed form -- there is no sampler to check and no
optimiser to fail, which is why this is the first estimate of the equation to report.

Two caveats travel with it, and both are why the module reports more than a point estimate.

The scores are **generated regressors**: `psi_{t-1}` is estimated from the same curves, and
standard errors that condition on it understate the uncertainty in `d` (Pagan, 1984). Nothing
here corrects for that; `se_ols` and `se_hac` are both conditional on the decomposition.

The MLE **shrinks nothing**. With a few hundred sessions and a `d` block as wide as the
retained components, the high-order directions are fitted to noise, and the reconstructed
`delta(u)` inherits it. `fit_lasso` is the diagnostic: it penalises the `d` and `k` blocks and
leaves the intercept and `gamma` alone, so the gap between the two estimates of `delta` is a
direct reading of what unpenalised estimation costs here.

`delta(u)` is the object to plot and to argue about. The vector `d` is an artefact of the basis
the curves happened to produce; `reconstruct_delta` puts it back on the domain, where it says
which part of yesterday's session predicts tonight's move.

Bosq's FAR(1) and this equation are the two halves of the same picture -- `far.far_est` models
the next curve from this one, and this models the scalar the curve does not contain.
`notebooks/overnight_jumps.ipynb` establishes what there is to find in the data; this is the
model written down.
"""

import numpy as np
from scipy import stats

__all__ = [
    "build_design",
    "JumpFit",
    "fit_mle",
    "fit_lasso",
    "reconstruct_delta",
]


def build_design(J, psi=None, Z=None):
    """Target and design matrix for the jump equation, with the lags aligned.

    Returns y (n - 1,) holding `J_2, ..., J_n`, X (n - 1, q) whose columns are the constant,
    `J_{t-1}`, `psi_{t-1}` and `z_t`, `names` a label per column, and `blocks` a dict of
    column slices keyed "const", "gamma", "d" and "k".

    The timing is the whole of the function. `psi` enters lagged and `z` contemporaneously,
    because `z_t` summarises the session that closed before the jump formed while `psi_t`
    describes the session the jump opens. One observation is lost at the front, there being no
    `J_0` to lag.

    PARAMETERS
    J   : (n,) array of overnight jumps in time order;
    psi : (n, M) array of principal component scores, one row per session, or None to leave
          the functional block out;
    Z   : (n, p) array of predetermined covariates, or None to leave that block out.
    """
    J = np.asarray(J, float).ravel()
    n = J.size
    if n < 3:
        raise ValueError(f"need at least three sessions to lag and fit, got {n}.")

    y = J[1:]
    columns = [np.ones(n - 1), J[:-1]]
    names = ["const", "gamma(J_lag)"]
    blocks = {"const": slice(0, 1), "gamma": slice(1, 2)}
    cursor = 2

    if psi is not None:
        psi = np.atleast_2d(np.asarray(psi, float))
        if psi.shape[0] == 1 and psi.shape[1] == n:
            psi = psi.T
        if psi.shape[0] != n:
            raise ValueError(f"psi has {psi.shape[0]} rows, J has {n}.")
        M = psi.shape[1]
        columns.append(psi[:-1])                       # psi_{t-1}
        names += [f"d[{k}]" for k in range(M)]
        blocks["d"] = slice(cursor, cursor + M)
        cursor += M

    if Z is not None:
        Z = np.atleast_2d(np.asarray(Z, float))
        if Z.shape[0] == 1 and Z.shape[1] == n:
            Z = Z.T
        if Z.shape[0] != n:
            raise ValueError(f"Z has {Z.shape[0]} rows, J has {n}.")
        p = Z.shape[1]
        columns.append(Z[1:])                          # z_t
        names += [f"k[{j}]" for j in range(p)]
        blocks["k"] = slice(cursor, cursor + p)

    X = np.column_stack([c if c.ndim == 2 else c[:, None] for c in columns])
    return y, X, names, blocks


def _hac_cov(X, resid, XtX_inv, lags=None):
    """Newey-West covariance of the coefficient vector, under Bartlett weights.

    Overnight jumps cluster in volatility even where the conditional mean is right, so the
    homoskedastic errors are the optimistic ones and these are the ones to quote on real data.
    The default truncation is the usual `4 (n/100)^(2/9)`.
    """
    n = X.shape[0]
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        G = u[lag:].T @ u[:-lag]
        S = S + weight * (G + G.T)
    return XtX_inv @ S @ XtX_inv


class JumpFit:
    """A fitted jump equation, holding both covariance estimates.

    Attributes are theta the coefficients, sigma2 the maximum likelihood innovation variance,
    cov_ols and cov_hac the two covariance matrices with se_ols and se_hac their square roots,
    names, blocks, resid, nobs, rank and loglik. `block`, `tstats`, `pvalues`, `wald` and
    `summary` read them back.
    """

    def __init__(self, theta, sigma2, cov_ols, cov_hac, names, blocks, resid, nobs, rank,
                 loglik):
        self.theta = theta
        self.sigma2 = sigma2
        self.cov_ols = cov_ols
        self.cov_hac = cov_hac
        self.se_ols = np.sqrt(np.clip(np.diag(cov_ols), 0.0, np.inf))
        self.se_hac = np.sqrt(np.clip(np.diag(cov_hac), 0.0, np.inf))
        self.names = names
        self.blocks = blocks
        self.resid = resid
        self.nobs = nobs
        self.rank = rank
        self.loglik = loglik

    def block(self, name):
        """The coefficients of one parameter block. Returns an array.

        PARAMETERS
        name : str, one of "const", "gamma", "d" and "k".
        """
        if name not in self.blocks:
            raise KeyError(f"no {name!r} block was fitted; have {sorted(self.blocks)}.")
        return self.theta[self.blocks[name]]

    def tstats(self, hac=True):
        """Coefficients over their standard errors. Returns an array.

        PARAMETERS
        hac : bool, use the Newey-West errors rather than the homoskedastic ones.
        """
        return self.theta / (self.se_hac if hac else self.se_ols)

    def pvalues(self, hac=True):
        """Two-sided p-values of `tstats`, against a t distribution on `nobs - rank` degrees
        of freedom. Returns an array.

        PARAMETERS
        hac : bool, use the Newey-West errors rather than the homoskedastic ones.
        """
        return 2.0 * stats.t.sf(np.abs(self.tstats(hac)), df=self.nobs - self.rank)

    def wald(self, name, hac=True):
        """Joint test that a whole block is zero. Returns the chi-squared statistic, its
        degrees of freedom and the p-value.

        This is the test the `d` block wants. Each `d_k` on its own is a loading on one
        principal component, and asking whether any single one is non-zero answers a question
        nobody has -- the question is whether `delta` is zero as a *function*, which is the
        hypothesis that the whole block is. Reading a block of eight t-statistics instead
        invites both errors at once: the block can be jointly significant with no coefficient
        individually so, and one coefficient in eight clears the 5% level by chance one time in
        three.

        PARAMETERS
        name : str, one of "const", "gamma", "d" and "k";
        hac  : bool, use the Newey-West covariance rather than the homoskedastic one.
        """
        if name not in self.blocks:
            raise KeyError(f"no {name!r} block was fitted; have {sorted(self.blocks)}.")
        where = self.blocks[name]
        theta = self.theta[where]
        cov = (self.cov_hac if hac else self.cov_ols)[where, where]
        statistic = float(theta @ np.linalg.solve(np.atleast_2d(cov), theta))
        df = int(theta.size)
        return statistic, df, float(stats.chi2.sf(statistic, df))

    def summary(self, hac=True):
        """The fit as a table, one row per coefficient. Returns a str.

        PARAMETERS
        hac : bool, use the Newey-West errors rather than the homoskedastic ones.
        """
        se = self.se_hac if hac else self.se_ols
        t = self.theta / se
        p = self.pvalues(hac)
        lines = [
            f"jump equation  |  n = {self.nobs}, q = {self.rank}, "
            f"sigma_J^2 = {self.sigma2:.4g}, logL = {self.loglik:.2f}",
            f"standard errors: {'Newey-West' if hac else 'homoskedastic'}",
            f"{'parameter':<16}{'coef':>12}{'se':>12}{'t':>9}{'p':>9}",
            "-" * 58,
        ]
        for name, c, s, tt, pp in zip(self.names, self.theta, se, t, p):
            lines.append(f"{name:<16}{c:>12.4f}{s:>12.4f}{tt:>9.3f}{pp:>9.4f}")
        return "\n".join(lines)


def fit_mle(J, psi=None, Z=None, hac_lags=None):
    """Conditional maximum likelihood of the jump equation, in closed form. Returns a
    `JumpFit`.

    Conditioning on the past makes every regressor predetermined, so the conditional
    likelihood factorises into that of a linear model and the maximiser is least squares.
    `sigma2` is the likelihood's own estimate, with divisor `n`; the standard errors use the
    unbiased `n - q` version, which is the usual convention and the one a regression table is
    read against.

    PARAMETERS
    J, psi, Z : arrays, as `build_design` takes them;
    hac_lags  : int or None, Newey-West truncation lag; the `4 (n/100)^(2/9)` rule by default.
    """
    y, X, names, blocks = build_design(J, psi, Z)
    n, q = X.shape
    if n <= q:
        raise ValueError(f"{n} usable sessions cannot identify {q} coefficients.")

    XtX_inv = np.linalg.inv(X.T @ X)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ theta
    rss = float(resid @ resid)

    cov_ols = (rss / (n - q)) * XtX_inv
    cov_hac = _hac_cov(X, resid, XtX_inv, lags=hac_lags)

    sigma2 = rss / n
    loglik = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
    return JumpFit(theta=theta, sigma2=sigma2, cov_ols=cov_ols, cov_hac=cov_hac, names=names,
                   blocks=blocks, resid=resid, nobs=n, rank=q, loglik=loglik)


def fit_lasso(J, psi, Z=None, n_folds=5, random_state=0):
    """Cross-validated lasso on the functional and covariate blocks, as an overfitting check.

    Returns a dict with alpha the selected penalty, coef a dict of every penalised
    coefficient, selected those it left non-zero, n_selected, and names_penalised.

    The intercept and `gamma` are left unpenalised and partialled out first (Frisch-Waugh), so
    the penalty falls only where the overfitting is -- the `d` block, whose width is set by how
    many components the decomposition retained rather than by how many of them carry signal.
    The remaining columns are standardised so one penalty is comparable across them, and the
    coefficients are returned on the original scale.

    Disagreement with `fit_mle` on `d` is the finding, not a nuisance: it is how much of the
    fitted `delta(u)` the unpenalised estimator is inventing.

    PARAMETERS
    J, psi, Z    : arrays, as `build_design` takes them; `psi` is required here;
    n_folds      : int, cross-validation folds;
    random_state : int or None, seed for the fold assignment.
    """
    from sklearn.linear_model import LassoCV

    y, X, names, _blocks = build_design(J, psi, Z)
    n, q = X.shape
    penalised = list(range(2, q))
    if not penalised:
        raise ValueError("nothing to penalise: pass psi, and optionally Z.")

    unpenalised = X[:, :2]
    projector = np.eye(n) - unpenalised @ np.linalg.solve(
        unpenalised.T @ unpenalised, unpenalised.T)
    Xp = projector @ X[:, penalised]
    yp = projector @ y

    sd = Xp.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    lasso = LassoCV(cv=n_folds, random_state=random_state, alphas=100, fit_intercept=False,
                    max_iter=100_000).fit((Xp - Xp.mean(axis=0)) / sd, yp)

    coef = lasso.coef_ / sd
    labels = [names[i] for i in penalised]
    return {"alpha": float(lasso.alpha_),
            "coef": dict(zip(labels, coef)),
            "selected": {lab: c for lab, c in zip(labels, coef) if abs(c) > 1e-8},
            "n_selected": int(np.sum(np.abs(coef) > 1e-8)),
            "names_penalised": labels}


def reconstruct_delta(d, phi):
    """The functional coefficient `delta(u) = sum_k d_k phi_k(u)`, on the grid `phi` is
    sampled at. Returns an (N,) array.

    PARAMETERS
    d   : (M,) array of loadings on the principal components;
    phi : (M, N) array of principal components, as `far.far_est.fpca` returns them.
    """
    return np.asarray(d, float) @ np.atleast_2d(np.asarray(phi, float))
