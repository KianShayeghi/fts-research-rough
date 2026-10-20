"""Estimating the joint jump-and-curve model of `joint_dgp`, one equation at a time.

    J_t    = alpha + rho J_{t-1} + int beta(u) X_{t-1}(u) du + eta_t,
    X_t(s) = mu(s) + b(s) J_t + int Psi(s, r) X_{t-1}(r) dr + eps_t(s),    b(s) = -gamma G(s).

The system is recursive and its two errors are independent, so the joint conditional
likelihood factorises and each equation can be fitted on its own without loss (`joint_dgp`
gives the argument). Both are linear once the curve regressor is expanded in its empirical
principal components, and both are fitted by least squares.

`fit_jump_equation` is `jump_eq.fit_mle` with the components of `X_{t-1}` as the score block,
and puts the loadings back on the domain as `beta_hat(u)`.

`fit_curve_equation` regresses `X_t(s)`, at every grid point, on `(1, J_t, xi_{t-1})` with
`xi_{t-1}` the scores of yesterday's curve -- the predictor-side principal component regression
of Yao, Müller & Wang (2005) with a scalar regressor alongside, which is a FAR with an exogenous
variable (Damon & Guillas, 2002). `J_t` is contemporaneous but exogenous: it is formed before
the session from the past and `eta_t`, and `eta_t` is independent of `eps_t`. The slope on it is
the unrestricted gap-fill profile `b_hat(s)`, and the notes' `gamma` is read off it in two ways:

- `gamma_ols`, the L2 projection of `-b_hat` onto `G`, which is least squares pooled over the
  whole grid;
- `gamma_gls`, which weights by the noise. The curve noise is an integral of Brownian motion, so
  its increments are independent with variances `sigma^2(v) dv`; generalised least squares on
  the increments of `b_hat` is therefore the Gaussian maximum likelihood estimator of `gamma`
  given the rest, with `sigma^2` estimated from the residual increments. Under constant
  volatility and `G(s) = s` it reduces to the slope of the close `X_t(1)` alone -- the drift of a
  Brownian motion is estimated by its endpoint -- and it is never less efficient than the
  projection, which weights the noisy late session most.

Both come with heteroskedasticity-robust standard errors, because a random `gamma_t` puts
`(gamma_t - gamma_bar) G(s) J_t` in the residual and makes it heteroskedastic in `J_t`; the
mean `gamma_bar` is still what both estimate, `gamma_t` being independent of `J_t`.
`gamma_variance` then reads `Var(gamma_t)` off that same heteroskedasticity.

`onset_profile` asks where `beta` switches on, under the parametric form `beta = kappa
1{u >= theta}`, by profiling the residual sum of squares over `theta`. It reads the onset on
the curve's level, as the notes write `beta`, or on its returns, and the difference between the
two is the most consequential thing in the notebook: on the level the onset is a kink in the
weight the jump puts on each intraday return, on returns it is a switch, and at equal signal
the switch is located far more sharply.

`nesting_test` asks the question `onset_profile` cannot: given an onset, is the jump driven
by the level after it or by the return after it? It fits both regressors at once, and each
t-test is a Davidson & MacKinnon (1981) J test of one model against the other.

`shape_test` asks whether `G(s) = s` is right at all: whether `b_hat` lies in the span of `G`.
Its statistic is the L2 norm of the part of `b_hat` orthogonal to `G`, whose null distribution
is a weighted sum of chi-squares (as in the functional F test of Shen & Faraway, 2004), and the
p-value comes from Imhof's (1961) inversion of its characteristic function rather than from a
moment-matched approximation.

Two things travel with every estimate here, as they do in `jump_eq`. The scores are generated
regressors, and the errors condition on the decomposition (Pagan, 1984). And truncation to `p`
components is not innocent in this model: whatever part of `beta` or `Psi` lies outside the
retained span is omitted, and the gap-fill term makes yesterday's curve correlated with
yesterday's jump, so the omission leaks into `rho_hat` and, through `beta`, into `gamma_hat`.
Both leaks shrink as `p` grows, and the Monte Carlo in `notebooks/jump_equation_v2.ipynb`
measures them rather than assuming them away.
"""

import numpy as np
from scipy import integrate

from ..core.quadrature import quadrature_weights
from ..far.far_est import fpca
from . import jump_eq as je
from .joint_dgp import step_functional

__all__ = [
    "fit_jump_equation",
    "fit_curve_equation",
    "gamma_variance",
    "shape_test",
    "onset_profile",
    "nesting_test",
    "imhof_sf",
    "hs_distance",
]


def _check_pair(J, X):
    J = np.asarray(J, float).ravel()
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != J.size:
        raise ValueError(f"J has {J.size} days, X has {X.shape[0]}.")
    if J.size < 4:
        raise ValueError("need at least four days to lag and fit.")
    return J, X


def fit_jump_equation(J, X, grid, n_components=5, hac_lags=None):
    """Fit `J_t = alpha + rho J_{t-1} + <beta, X_{t-1}> + eta_t` through the leading principal
    components of the curves. Returns a dict with alpha, rho, beta_hat (N,), sigma_eta, fit the
    `jump_eq.JumpFit`, and fpca the decomposition used.

    The scores are of centred curves, so the regression's own intercept is `alpha + <beta,
    X_bar>`; the mean curve's contribution is taken back off to report `alpha` itself.
    `sigma_eta` uses the unbiased divisor, `n - q`.

    PARAMETERS
    J            : (n,) jumps, `J[t]` opening day `t`;
    X            : (n, N) CIDR curves, `X[t]` the session of day `t`;
    grid         : (N,) array, the intraday domain;
    n_components : int, principal components of `X` the coefficient function is expanded in;
    hac_lags     : int or None, passed to `jump_eq.fit_mle`.
    """
    J, X = _check_pair(J, X)
    dec = fpca(X, grid, n_components=n_components)
    fit = je.fit_mle(J, dec["scores"], hac_lags=hac_lags)
    beta_hat = je.reconstruct_delta(fit.block("d"), dec["phi"])
    w = quadrature_weights(grid)
    resid = fit.resid
    return {"alpha": float(fit.theta[0] - beta_hat @ (w * dec["mean"])),
            "rho": float(fit.theta[1]), "beta_hat": beta_hat,
            "sigma_eta": float(np.sqrt(resid @ resid / (fit.nobs - fit.rank))),
            "fit": fit, "fpca": dec}


def fit_curve_equation(J, X, grid, n_components=5, G=None):
    """Fit `X_t(s) = mu(s) + b(s) J_t + int Psi(s, r) X_{t-1}(r) dr + eps_t(s)` by least squares
    at every grid point, and read `gamma` off `b`. Returns a dict with

        mu_hat (N,), b_hat (N,), kernel (N, N) the estimate of `Psi(s, r)`, resid (n-1, N),
        gamma_ols, se_ols, gamma_gls, se_gls, sigma2_hat (N-1,) the spot variance of the
        curve noise at each cell's midpoint, cov_b (N, N) the robust covariance of b_hat,
        J_tilde (n-1,) the jump with the other regressors partialled out, G, and fpca.

    `Psi` is estimated in the span of the `n_components` leading components of the curves,
    from the right: `Psi_hat(s, r) = sum_k C_k(s) phi_k(r)`. The left argument is unrestricted.

    Every standard error is the heteroskedasticity-robust one, over days. They are the honest
    ones whenever `gamma_t` is random, and agree with the homoskedastic ones when it is not.

    PARAMETERS
    J, X         : as `fit_jump_equation` takes them;
    grid         : (N,) array;
    n_components : int, principal components of yesterday's curve controlled for;
    G            : (N,) array or None, the gap-fill profile `gamma` multiplies; `G(s) = s` if
                   None, as in the notes.
    """
    J, X = _check_pair(J, X)
    grid = np.asarray(grid, float)
    G = grid.copy() if G is None else np.asarray(G, float)
    w = quadrature_weights(grid)
    dec = fpca(X, grid, n_components=n_components)
    xi_lag = dec["scores"][:-1]
    Y = X[1:]
    jump = J[1:]
    n = jump.size

    controls = np.column_stack([np.ones(n), xi_lag])
    design = np.column_stack([controls[:, :1], jump, xi_lag])
    coef, *_ = np.linalg.lstsq(design, Y, rcond=None)
    resid = Y - design @ coef
    b_hat = coef[1]
    kernel = coef[2:].T @ dec["phi"]
    mu_hat = coef[0] - kernel @ (w * dec["mean"])

    # Frisch-Waugh: the slope on J_t is the slope on J_t with the controls partialled out,
    # which is what every standard error for gamma is built from.
    J_tilde = jump - controls @ np.linalg.lstsq(controls, jump, rcond=None)[0]
    sxx = float(J_tilde @ J_tilde)
    q = design.shape[1]
    scale = n / (n - q)                                    # HC1 small-sample correction

    G_norm2 = float(G @ (w * G))
    gamma_ols = -float(b_hat @ (w * G)) / G_norm2
    e = resid @ (w * G)
    se_ols = np.sqrt(scale * np.sum(J_tilde ** 2 * e ** 2)) / (sxx * G_norm2)

    d_resid = np.diff(resid, axis=1)
    cell = np.diff(grid)
    v = np.sum(d_resid ** 2, axis=0) / (n - q)             # variance of each noise increment
    v = np.where(v > 0, v, np.min(v[v > 0]) if np.any(v > 0) else 1.0)
    dG = np.diff(G)
    D = float(np.sum(dG ** 2 / v))
    gamma_gls = -float(np.sum(np.diff(b_hat) * dG / v)) / D
    g = d_resid @ (dG / v)
    se_gls = np.sqrt(scale * np.sum(J_tilde ** 2 * g ** 2)) / (sxx * D)

    weighted = resid * J_tilde[:, None]
    cov_b = scale * (weighted.T @ weighted) / sxx ** 2

    return {"mu_hat": mu_hat, "b_hat": b_hat, "kernel": kernel, "resid": resid,
            "gamma_ols": gamma_ols, "se_ols": float(se_ols),
            "gamma_gls": gamma_gls, "se_gls": float(se_gls),
            "sigma2_hat": v / cell, "cov_b": cov_b, "J_tilde": J_tilde, "G": G,
            "grid": grid, "jump": jump, "fpca": dec}


def gamma_variance(curve_fit):
    """Estimate `Var(gamma_t)` from the heteroskedasticity a random `gamma_t` leaves behind.
    Returns a dict with var, se and t.

    With `gamma_t` random, the curve equation's residual is `eps_t(s) - (gamma_t - gamma_bar)
    G(s) J_t`, whose variance given `J_t` is `Var eps_t(s) + Var(gamma_t) G(s)^2 J_t^2`. The
    estimate is the common slope of the squared residuals on `G(s)^2 J_t^2`, with a separate
    intercept at every `s` for the noise variance and the grid pooled under the quadrature
    weights, and its standard error is robust over days.

    The identification is through the fourth moment of the jump: it needs `J_t^2` to vary,
    and it is only as strong as the random part is large next to the curve noise. On real
    data it is also confounded, because a large overnight move tends to be followed by a
    volatile session, which raises `Var eps_t(s)` with `J_t^2` exactly as a random `gamma_t`
    would. Here, where `eps_t` is independent of `J_t` by construction, it is not.

    PARAMETERS
    curve_fit : dict, the output of `fit_curve_equation`.
    """
    resid = curve_fit["resid"]
    G = curve_fit["G"]
    w = quadrature_weights(curve_fit["grid"])
    jump2 = curve_fit["jump"] ** 2
    x = jump2 - jump2.mean()
    r2 = resid ** 2
    u = (r2 - r2.mean(axis=0)) @ (w * G ** 2)
    g4 = float(np.sum(w * G ** 4))
    sxx = float(x @ x)
    var = float(x @ u) / (g4 * sxx)
    e = u - var * g4 * x
    se = float(np.sqrt(np.sum(x ** 2 * e ** 2)) / (g4 * sxx))
    return {"var": var, "se": se, "t": var / se if se > 0 else np.nan}


def imhof_sf(x, lam):
    """`P(sum_k lam_k Z_k^2 > x)` for independent standard normal `Z_k`, by Imhof's (1961)
    numerical inversion of the characteristic function. Returns a float in [0, 1].

    The weights and `x` are first divided by the largest weight, which leaves the probability
    unchanged and puts the integrand on a scale an absolute tolerance means something on; the
    weights of a functional test are the eigenvalues of a covariance of order `1/n`, and left
    as they are the integral is lost below the tolerance. The integrand is `sin(A(u) - x u/2)`
    over a decaying amplitude, with `A` slowly varying, so past the first half-period it is
    split into a cosine and a sine transform and handed to QUADPACK's routine for Fourier
    integrals on a half-line, which is built for exactly that tail.

    PARAMETERS
    x   : float, the point the upper tail is taken from;
    lam : array of non-negative weights; those below 1e-12 of the largest are dropped.
    """
    lam = np.asarray(lam, float).ravel()
    top = float(np.max(lam, initial=0.0))
    if top <= 0.0:
        return float(x < 0)
    if x <= 0:
        return 1.0
    lam = lam[lam > 1e-12 * top] / top
    omega = 0.5 * float(x) / top

    def phase(u):
        return 0.5 * np.sum(np.arctan(lam * u))

    def amplitude(u):
        return 1.0 / (u * np.exp(0.25 * np.sum(np.log1p((lam * u) ** 2))))

    def integrand(u):
        return np.sin(phase(u) - omega * u) * amplitude(u)

    split = np.pi / omega
    head = integrate.quad(integrand, 0.0, split, limit=200)[0]
    cos_part = integrate.quad(lambda u: np.sin(phase(u)) * amplitude(u), split, np.inf,
                              weight="cos", wvar=omega, limlst=200)[0]
    sin_part = integrate.quad(lambda u: np.cos(phase(u)) * amplitude(u), split, np.inf,
                              weight="sin", wvar=omega, limlst=200)[0]
    value = head + cos_part - sin_part
    return float(np.clip(0.5 + value / np.pi, 0.0, 1.0))


def shape_test(curve_fit, G=None):
    """Test whether the gap-fill profile has the assumed shape, `b(s) = -gamma G(s)` for some
    `gamma`. Returns a dict with stat, p_value, weights the eigenvalues of the null
    distribution, gamma the projection estimate, and b_orth (N,) the part of `b_hat`
    orthogonal to `G`.

    The statistic is `T = ||(I - P_G) b_hat||^2` in L2, where `P_G` projects onto `G`. Under
    the null `(I - P_G) b_hat` is asymptotically Gaussian with covariance `(I - P_G) C (I -
    P_G)'`, `C` the robust covariance of `b_hat`, so `T` is distributed as a sum of
    independent chi-squares weighted by that covariance's eigenvalues in the L2 metric. The
    p-value is the exact tail of that sum by `imhof_sf`.

    A rejection says the gap does not close in proportion to `G`. Where it does reject, `b_hat`
    itself is the estimate of the profile to look at, and `b_orth` is how it departs.

    PARAMETERS
    curve_fit : dict, the output of `fit_curve_equation`;
    G         : (N,) array or None, the profile under the null; the one the fit used if None.
    """
    G = curve_fit["G"] if G is None else np.asarray(G, float)
    w = quadrature_weights(curve_fit["grid"])
    b = curve_fit["b_hat"]
    N = b.size
    P = np.outer(G, w * G) / float(G @ (w * G))            # L2 projection onto G
    Q = np.eye(N) - P
    b_orth = Q @ b
    stat = float(b_orth @ (w * b_orth))
    S = Q @ curve_fit["cov_b"] @ Q.T
    sw = np.sqrt(w)
    weights = np.clip(np.linalg.eigvalsh(sw[:, None] * S * sw[None, :]), 0.0, None)
    return {"stat": stat, "p_value": imhof_sf(stat, weights), "weights": weights,
            "gamma": -float(b @ (w * G)) / float(G @ (w * G)), "b_orth": b_orth}


def onset_profile(J, X, grid, thetas=None, on="level"):
    """Profile least squares for the onset of a step coefficient function. Returns a dict with
    thetas, rss (one per candidate), theta_hat, kappa_hat, alpha_hat and rho_hat at the
    minimiser.

    At each candidate `theta` the jump is regressed on `(1, J_{t-1}, z_{t-1}(theta))`, where
    `z` is `int_theta^1 X(u) du` under the trapezium weights for `on="level"` -- the notes'
    `beta = kappa 1{u >= theta}` -- and `X(1) - X(theta)` for `on="return"`, the return over the
    session after `theta`. The estimate is the candidate with the smallest residual sum of
    squares. The model is a step, so this is a parametric estimator and the sharper for it;
    where the step is the wrong shape, Hall & Hooker's estimators in `truncFLR` assume much
    less.

    PARAMETERS
    J, X   : as `fit_jump_equation` takes them;
    grid   : (N,) array;
    thetas : array or None, candidate onsets, each moved to the first grid point at or after
             it; every interior grid point if None;
    on     : {"level", "return"}, what the step multiplies.
    """
    if on not in ("level", "return"):
        raise ValueError(f"on must be 'level' or 'return', got {on!r}.")
    J, X = _check_pair(J, X)
    grid = np.asarray(grid, float)
    thetas = grid[1:-1] if thetas is None else np.asarray(thetas, float)
    y = J[1:]
    base = np.column_stack([np.ones(y.size), J[:-1]])
    rss, coefs = [], []
    for theta in thetas:
        z = X[:-1] @ step_functional(grid, theta, on)
        design = np.column_stack([base, z])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        r = y - design @ coef
        rss.append(float(r @ r))
        coefs.append(coef)
    rss = np.array(rss)
    best = int(np.argmin(rss))
    theta_hat = float(grid[np.argmax(grid >= thetas[best])])
    return {"thetas": thetas, "rss": rss, "theta_hat": theta_hat,
            "kappa_hat": float(coefs[best][2]), "alpha_hat": float(coefs[best][0]),
            "rho_hat": float(coefs[best][1])}


def nesting_test(J, X, grid, theta, hac=False):
    """Which of the two step models the jump follows, at a known onset. Returns a dict with
    coef, se, t and p_value, each a dict keyed "level" and "return", the joint fit, and corr,
    the sample correlation of the two regressors.

    The jump is regressed on `(1, J_{t-1}, z_level, z_return)`, the regressors of the step on
    the level and the step on the returns at the same `theta`. This comprehensive model nests
    both, so a t-test on each slope asks whether that model adds anything to the other. Where
    each alternative differs from the shared regressors by one column, as here, that t-test is
    exactly the J test of Davidson & MacKinnon (1981) for one non-nested model against the
    other, so the two p-values answer the two directions of "is this one enough?": a level
    truth should reject the return model's sufficiency and not the reverse, and vice versa.

    The onset is taken as known -- it is the hypothesis being tested, not a parameter being
    searched for. Choosing it by `onset_profile` on the same data and then testing at the
    chosen value would make the p-values too small.

    PARAMETERS
    J, X  : as `fit_jump_equation` takes them;
    grid  : (N,) array;
    theta : float, the onset under both hypotheses;
    hac   : bool, Newey-West rather than homoskedastic standard errors.
    """
    J, X = _check_pair(J, X)
    grid = np.asarray(grid, float)
    Z = np.column_stack([X @ step_functional(grid, theta, "level"),
                         X @ step_functional(grid, theta, "return")])
    fit = je.fit_mle(J, Z)                         # build_design lags the score block
    d = fit.blocks["d"]
    coef, se = fit.theta[d], (fit.se_hac if hac else fit.se_ols)[d]
    p = fit.pvalues(hac)[d]
    keys = ("level", "return")
    return {"coef": dict(zip(keys, coef)), "se": dict(zip(keys, se)),
            "t": dict(zip(keys, coef / se)), "p_value": dict(zip(keys, p)),
            "fit": fit, "corr": float(np.corrcoef(Z[:-1].T)[0, 1])}


def hs_distance(K1, K2, grid):
    """Hilbert-Schmidt distance between two kernels sampled on `grid`, under the trapezium
    rule. Returns a float.

    PARAMETERS
    K1, K2 : (N, N) arrays, unweighted kernel values `K(s_i, r_j)`.
    """
    w = quadrature_weights(np.asarray(grid, float))
    D = np.asarray(K1, float) - np.asarray(K2, float)
    return float(np.sqrt(np.sum(D ** 2 * w[:, None] * w[None, :])))
