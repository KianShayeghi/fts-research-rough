"""The joint model for the overnight jump and the session that follows it, and its simulator.

A day is split into the two pieces `fdaSP.intraday` already splits it into. With `P_t(s)` the
log-price curve of session `t` in percent (`100 log` of the price) at a fraction `s` of the
session elapsed,

    J_t    = P_t(0) - P_{t-1}(1)           the overnight jump,
    X_t(s) = P_t(s) - P_t(0)               the cumulative intraday return (CIDR) curve,
    P_t(s) = P_{t-1}(1) + J_t + X_t(s)     which is the two of them put back together,

and the model is a recursive pair of equations, one for each piece:

    J_t    = alpha + rho J_{t-1} + int beta(u) X_{t-1}(u) du + eta_t,
    X_t(s) = mu(s) - gamma_t G(s) J_t + int Psi(s, r) X_{t-1}(r) dr + eps_t(s),
    eps_t(s) = int_0^s sigma(v) dW_t(v).

The first is a scalar-on-function regression with its own lag -- `jump_eq`'s equation without
covariates. The second is a FAR(1) on the CIDR curves with the jump entering as a
contemporaneous scalar regressor, so a FAR with an exogenous variable in the sense of Damon &
Guillas (2002). `gamma_t G(s)` is the fraction of the jump the session has given back by `s`:
`G(s) = s` fills the gap at a uniform rate, and `gamma_t` in `[0, 1]` is the part of it filled
by the close. `gamma_t` is fixed, or drawn i.i.d. independently of everything else.

`X_t(0) = 0` holds for every realisation only if `mu(0) = G(0) = 0`, `Psi(0, .) = 0` and
`eps_t(0) = 0`, and the last of those is why the curve noise is an integral of Brownian motion
rather than pointwise: `JointModel` checks the first three on construction, and the fourth
holds by construction. The integral also makes the noise a model of intraday returns rather
than of their measurement -- its increments are independent, with the variance profile
`sigma^2(v)` that Andersen & Bollerslev (1997) document as U-shaped over the session.

**Stationarity is joint.** Substituting the jump into the curve equation writes the pair as
one autoregression on `Y_t = (J_t, X_t)`,

    Y_t = L_t (c + B Y_{t-1} + w_t),
    B   = [[rho, <beta, .>], [0, Psi]],    L_t = [[1, 0], [-gamma_t G, I]],

with `c = (alpha, mu)` and `w_t = (eta_t, eps_t)`. `|rho| < 1` and `||Psi|| < 1` separately do
not make it stationary, because `L_t` couples them: with `Psi = 0` the jump's effective
persistence is `rho - gamma <beta, G>`, the part of yesterday's jump the session gave back
being fed into tonight's through `beta`. The condition is on the spectral radius of
`E A_t = L_bar B` (Bosq, 2000, Thm 3.1), and with random `gamma_t` it is on the mean-square
operator `S -> E[A_t S A_t']` as well (Nicholls & Quinn, 1982), which `stationarity` reports.
`moments` then solves for the stationary mean and covariance on the grid, so a simulation can
be checked against what it should produce rather than against itself.

**The system is recursive, so it estimates equation by equation.** `J_t` depends on the past
and `eta_t`; `X_t` on the past, `J_t` and `eps_t`; and the two errors are independent. The
joint conditional density therefore factorises, `f(J_t, X_t | past) = f(J_t | past) f(X_t |
J_t, past)`, and `J_t` is exogenous in the curve equation. `joint_est` fits the two halves
separately for that reason and loses nothing by it.

Everything here lives on a grid. Integral operators carry the trapezium weights of
`core.quadrature`, as `far.far_dgp` builds them, and the population moments are those of the
discretised process -- the one `simulate` actually draws from -- so theory and simulation
agree to Monte Carlo error rather than to discretisation error as well.
"""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from ..core.quadrature import quadrature_weights

__all__ = [
    "JointModel",
    "notes_kernel",
    "notes_beta",
    "gapfill_profile",
    "intraday_variance",
    "cell_variances",
    "noise_covariance",
    "notes_design",
    "stationarity",
    "moments",
    "calibrate",
    "scale_to_r2",
    "step_functional",
    "population_fit",
    "forecast",
    "simulate",
    "price_curves",
]

# Gauss-Legendre nodes for integrating sigma^2 over each grid cell: exact for polynomials of
# degree 15, which covers every variance profile here bar a pathological one.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(8)


class JointModel:
    """The jump-and-curve model on a grid, with its parameters checked on construction.

    Attributes are the parameters as given, plus `w` the quadrature weights and `Psi_op` the
    quadrature-weighted operator matrix, so that `Psi_op @ x` approximates `int Psi(s, r) x(r)
    dr`. `operators` returns the companion-form pieces, and `with_params` a copy with some
    parameters replaced.

    Raises ValueError where the curve equation would not keep `X_t(0) = 0`, which is the CIDR
    curve's definition rather than a modelling choice, or where `gamma_t`'s distribution
    cannot have the variance asked of it.

    PARAMETERS
    grid        : (N,) array, the intraday domain, starting at 0;
    alpha       : float, jump intercept;
    rho         : float, coefficient on yesterday's jump;
    beta        : (N,) array, the jump coefficient function on the grid;
    mu          : (N,) array, the intercept curve, zero at s = 0;
    kernel      : (N, N) array, `Psi(s_i, r_j)` unweighted, zero on its first row;
    G           : (N,) array, the gap-fill profile, zero at s = 0;
    gamma_mean  : float, mean of `gamma_t`;
    gamma_var   : float, variance of `gamma_t`; zero fixes it at its mean;
    gamma_dist  : {"fixed", "uniform"}, the law `gamma_t` is drawn from. "uniform" is the
                  uniform with the given mean and variance, so U(0, 1) is mean 1/2,
                  variance 1/12;
    sigma_eta   : float, standard deviation of the jump innovation;
    eta_df      : float or None, degrees of freedom of a Student t jump innovation, rescaled
                  to standard deviation `sigma_eta`; None for Gaussian;
    cell_var    : (N - 1,) array, variance of the curve noise accumulated over each grid cell,
                  `int sigma^2(v) dv` over it (see `cell_variances`).
    """

    def __init__(self, grid, alpha, rho, beta, mu, kernel, G, gamma_mean, sigma_eta, cell_var,
                 gamma_var=0.0, gamma_dist="fixed", eta_df=None):
        self.grid = np.asarray(grid, float)
        N = self.grid.size
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.beta = _on_grid(beta, N, "beta")
        self.mu = _on_grid(mu, N, "mu")
        self.kernel = np.asarray(kernel, float)
        self.G = _on_grid(G, N, "G")
        self.gamma_mean = float(gamma_mean)
        self.gamma_var = float(gamma_var)
        self.gamma_dist = gamma_dist
        self.sigma_eta = float(sigma_eta)
        self.eta_df = None if eta_df is None else float(eta_df)
        self.cell_var = np.asarray(cell_var, float)

        if self.grid[0] != 0.0 or np.any(np.diff(self.grid) <= 0):
            raise ValueError("the grid must be increasing and start at s = 0.")
        if self.kernel.shape != (N, N):
            raise ValueError(f"kernel is {self.kernel.shape}, the grid needs {(N, N)}.")
        if self.cell_var.shape != (N - 1,) or np.any(self.cell_var < 0):
            raise ValueError(f"cell_var must be {N - 1} non-negative cell variances.")
        if self.sigma_eta < 0:
            raise ValueError("sigma_eta must be non-negative.")
        if self.eta_df is not None and self.eta_df <= 2:
            raise ValueError("a t innovation needs more than 2 degrees of freedom to have a "
                             "variance.")
        violated = [name for name, value in (
            ("mu(0)", self.mu[0]), ("G(0)", self.G[0]),
            ("Psi(0, r)", np.max(np.abs(self.kernel[0])))) if value != 0.0]
        if violated:
            raise ValueError(f"X_t(0) = 0 needs {', '.join(violated)} to be zero.")
        if gamma_dist not in ("fixed", "uniform"):
            raise ValueError(f"gamma_dist must be 'fixed' or 'uniform', got {gamma_dist!r}.")
        if gamma_dist == "fixed" and self.gamma_var != 0.0:
            raise ValueError("a fixed gamma has no variance; use gamma_dist='uniform'.")
        if self.gamma_var < 0:
            raise ValueError("gamma_var must be non-negative.")

        self.w = quadrature_weights(self.grid)
        self.Psi_op = self.kernel * self.w[None, :]

    def operators(self):
        """The companion form `Y_t = L_t (c + B Y_{t-1} + w_t)` on the grid. Returns a dict
        with B (N+1, N+1), L_bar `E L_t`, E the matrix with `L_t = I - gamma_t E`, c the
        constant `(alpha, mu)`, and W the covariance of `w_t`."""
        N = self.grid.size
        B = np.zeros((N + 1, N + 1))
        B[0, 0] = self.rho
        B[0, 1:] = self.beta * self.w
        B[1:, 1:] = self.Psi_op
        E = np.zeros((N + 1, N + 1))
        E[1:, 0] = self.G
        W = np.zeros((N + 1, N + 1))
        W[0, 0] = self.sigma_eta ** 2
        W[1:, 1:] = noise_covariance(self.cell_var)
        return {"B": B, "L_bar": np.eye(N + 1) - self.gamma_mean * E, "E": E,
                "c": np.concatenate([[self.alpha], self.mu]), "W": W}

    def with_params(self, **changes):
        """A copy of this model with the named parameters replaced. Returns a JointModel.

        PARAMETERS
        changes : keyword arguments named as the constructor's.
        """
        params = {k: getattr(self, k) for k in (
            "grid", "alpha", "rho", "beta", "mu", "kernel", "G", "gamma_mean", "sigma_eta",
            "cell_var", "gamma_var", "gamma_dist", "eta_df")}
        params.update(changes)
        return JointModel(**params)


def _on_grid(values, N, name):
    values = np.asarray(values, float)
    if values.shape != (N,):
        raise ValueError(f"{name} has shape {values.shape}, the grid has {N} points.")
    return values


def noise_covariance(cell_var):
    """Covariance of `eps(s) = int_0^s sigma dW` on the grid, `min`-type as Brownian motion's.
    Returns an (N, N) array whose (i, k) entry is the variance accumulated up to the earlier
    of `s_i` and `s_k`.

    PARAMETERS
    cell_var : (N - 1,) array, variance accumulated over each grid cell.
    """
    cumulative = np.concatenate([[0.0], np.cumsum(cell_var)])
    idx = np.arange(cumulative.size)
    return cumulative[np.minimum.outer(idx, idx)]


# ------------------------------------------------------------------- the notes' parameters


def notes_kernel(grid, c=0.1, scale=1.0):
    """The FAR kernel `Psi(s, r) = scale * s * exp(-(s - r)^2 / c)`. Returns (N, N), unweighted.

    The factor `s` is what gives `Psi(0, r) = 0`, so yesterday's curve cannot move today's
    open. It also bounds the Hilbert-Schmidt norm: `|Psi| <= s`, so `||Psi||_HS^2 <= int s^2
    = 1/3` at unit `scale` whatever `c` is, and the curve equation on its own is stationary
    for every bandwidth. `c` sets how far along the session the dependence reaches.

    PARAMETERS
    c     : float, bandwidth of the Gaussian factor, in squared units of the domain;
    scale : float, multiplies the whole kernel; 1 is the kernel exactly as written.
    """
    S, R = np.meshgrid(grid, grid, indexing="ij")
    return scale * S * np.exp(-(S - R) ** 2 / float(c))


def notes_beta(grid, kind="indicator", onset=0.85, amplitude=1.0):
    """The jump coefficient function, `1{u >= onset}` or `{exp(u - onset) - 1} 1{u >= onset}`.
    Returns (N,).

    Both are zero before `onset`, which makes the coefficient function a truncated one in Hall
    & Hooker's sense, with `onset` the endpoint to recover. The second starts from zero at the
    onset and so is continuous, which the first is not; it is also far smaller, peaking at
    `exp(1 - onset) - 1` rather than 1, so at equal `amplitude` it carries a fraction of the
    first one's signal.

    Both act on the *level* of the curve, and that is not the same as "the last part of the
    session matters". Integrating by parts, with `X(0) = 0`,

        int beta(u) X(u) du = int B(v) dX(v),    B(v) = int_v^1 beta(u) du,

    so the indicator puts weight `1 - onset` on every return made *before* the onset and a
    weight falling linearly to zero on those after it: the onset is a kink in the weight on
    returns, not a switch. "return" is the switch itself, `amplitude * (X(1) - X(onset))`,
    the return over the late session alone. It is not a function in L2 -- it evaluates the
    curve at two points -- so here it is the grid functional that does that under the
    trapezium weights, spikes at the two points and zero elsewhere, which `simulate` and
    `moments` use exactly as they use any other `beta`.

    PARAMETERS
    kind      : {"indicator", "smooth", "return"}, which of the three;
    onset     : float, where the coefficient function switches on;
    amplitude : float, multiplies it.
    """
    grid = np.asarray(grid, float)
    on = grid >= onset
    if kind == "indicator":
        return amplitude * on.astype(float)
    if kind == "smooth":
        return amplitude * np.where(on, np.expm1(grid - onset), 0.0)
    if kind == "return":
        w = quadrature_weights(grid)
        start = int(np.argmax(on))
        functional = np.zeros(grid.size)
        functional[-1] += 1.0
        functional[start] -= 1.0
        return amplitude * functional / w
    raise ValueError(f"kind must be 'indicator', 'smooth' or 'return', got {kind!r}.")


def gapfill_profile(grid, kind="linear", tau=0.1):
    """The gap-fill profile `G(s)`, with `G(0) = 0` and `G(1) = 1`. Returns (N,).

    "linear" is the notes' `G(s) = s`, the gap closing at a uniform rate through the session.
    "front" is `(1 - exp(-s / tau)) / (1 - exp(-1 / tau))`, which does most of its filling in
    the first `tau` of the session, the way a gap is often said to be "filled in the first
    hour"; it is the alternative `joint_est.shape_test` is meant to have power against.

    PARAMETERS
    kind : {"linear", "front"};
    tau  : float, time scale of the "front" profile, as a fraction of the session.
    """
    grid = np.asarray(grid, float)
    if kind == "linear":
        return grid.copy()
    if kind == "front":
        return -np.expm1(-grid / tau) / -np.expm1(-1.0 / tau)
    raise ValueError(f"kind must be 'linear' or 'front', got {kind!r}.")


def intraday_variance(v, level=1.0, ratio=2.5):
    """A U-shaped spot variance `sigma^2(v) = level * (1 + k (12 (v - 1/2)^2 - 1))`. Returns an
    array shaped as `v`.

    `k` is set so the variance at either end of the session is `ratio` times the variance at
    its middle, and the bracket integrates to one, so `level` is the variance the noise
    accumulates over the whole session. `ratio = 1` is constant volatility, under which the
    curve noise is a scaled Brownian motion.

    PARAMETERS
    v     : array, points of the session;
    level : float, `int_0^1 sigma^2`;
    ratio : float, end-of-session to mid-session variance, at least 1.
    """
    if ratio < 1:
        raise ValueError("ratio must be at least 1 for a U shape.")
    k = (ratio - 1.0) / (ratio + 2.0)
    v = np.asarray(v, float)
    return level * (1.0 + k * (12.0 * (v - 0.5) ** 2 - 1.0))


def cell_variances(grid, sigma2):
    """`int sigma^2(v) dv` over each cell of the grid, by eight-point Gauss-Legendre. Returns
    (N - 1,).

    These are the exact variances of the increments of `eps(s) = int_0^s sigma dW` between
    neighbouring grid points, so drawing the increments independently with these variances
    samples the process on the grid without discretisation error.

    PARAMETERS
    sigma2 : callable or float, the spot variance `sigma^2(v)`; a float is constant.
    """
    grid = np.asarray(grid, float)
    if not callable(sigma2):
        return float(sigma2) * np.diff(grid)
    lo, hi = grid[:-1], grid[1:]
    half = 0.5 * (hi - lo)
    nodes = 0.5 * (hi + lo)[:, None] + half[:, None] * _GL_NODES[None, :]
    return half * (np.asarray(sigma2(nodes), float) @ _GL_WEIGHTS)


def notes_design(grid=None, n_grid=79, beta="indicator", onset=0.85, beta_amplitude=1.0,
                 c=0.1, psi_scale=1.0, gamma=0.3, gamma_dist="fixed", gamma_var=None,
                 gap="linear", tau=0.1, alpha=0.05, rho=0.1, drift=-0.02, sigma_eta=0.5,
                 noise_level=1.0, noise_ratio=2.5, eta_df=None):
    """The simulation design of the notes, with the parameters they leave blank filled in.
    Returns a JointModel.

    Given by the notes: `Psi(s, r) = s exp(-(s - r)^2 / c)`, `beta` the indicator or smooth
    onset at 0.85, `gamma = 0.3` or `gamma_t ~ U(0, 1)`, `G(s) = s`. Filled in here, in
    percent units, as a design rather than an estimate -- every value is a round number chosen
    for what it makes visible, and `notebooks/jump_equation_v2.ipynb` gives the reason for each:
    the grid, 79 points being five-minute bars over a 6.5-hour session; a session standard
    deviation of 1% and an overnight one of about half that; the U-shaped intraday variance
    of Andersen & Bollerslev (1997), its ends two and a half times its middle; a small
    positive overnight drift `alpha` and a small negative intraday one, `mu(s) = drift * s`,
    which is the sign pattern Cliff, Cooper & Gulen (2008) document; and `rho = 0.1`, small
    but not zero, so the jump's persistence is something to recover.

    PARAMETERS
    grid, n_grid   : the domain, or the number of equally spaced points on [0, 1];
    beta           : {"indicator", "smooth", "return"}, passed to `notes_beta` with `onset`;
    beta_amplitude : float, multiplies beta;
    c, psi_scale   : passed to `notes_kernel`;
    gamma          : float, `gamma_t`'s mean, or its value when fixed;
    gamma_dist     : {"fixed", "uniform"};
    gamma_var      : float or None, `gamma_t`'s variance; None is 1/12 under "uniform", the
                     variance of U(0, 1), and zero under "fixed";
    gap, tau       : passed to `gapfill_profile`;
    drift          : float, slope of `mu(s) = drift * s`;
    noise_level    : float, `int_0^1 sigma^2`, the curve-noise variance at the close;
    noise_ratio    : float, end-to-middle ratio of the U-shaped `sigma^2`;
    eta_df         : float or None, Student t degrees of freedom for the jump innovation.
    """
    grid = np.linspace(0.0, 1.0, int(n_grid)) if grid is None else np.asarray(grid, float)
    if gamma_var is None:
        gamma_var = 1.0 / 12.0 if gamma_dist == "uniform" else 0.0
    return JointModel(
        grid=grid, alpha=alpha, rho=rho,
        beta=notes_beta(grid, beta, onset=onset, amplitude=beta_amplitude),
        mu=drift * grid, kernel=notes_kernel(grid, c=c, scale=psi_scale),
        G=gapfill_profile(grid, gap, tau=tau), gamma_mean=gamma, gamma_var=gamma_var,
        gamma_dist=gamma_dist, sigma_eta=sigma_eta, eta_df=eta_df,
        cell_var=cell_variances(grid, lambda v: intraday_variance(v, noise_level,
                                                                  noise_ratio)))


# ----------------------------------------------------------------------------- the theory


def _ms_map(ops, gamma_var):
    """The mean-square map `S -> E[A_t S A_t']`, with `A_t = (I - gamma_t E) B`."""
    A_bar = ops["L_bar"] @ ops["B"]
    EB = ops["E"] @ ops["B"]
    return lambda S: A_bar @ S @ A_bar.T + gamma_var * (EB @ S @ EB.T)


def stationarity(model):
    """Whether the joint process has a stationary solution, and how far inside the boundary.
    Returns a dict with radius, the spectral radius of `E A_t`; ms_radius, that of the
    mean-square map `S -> E[A_t S A_t']`; decoupled, the largest of `|rho|` and the spectral
    radius of `Psi` alone, which is what checking the two equations separately would report;
    and stationary, `ms_radius < 1`.

    Under a fixed `gamma` the mean-square map is `S -> A S A'`, whose spectral radius is the
    square of `A`'s, and that is what is returned. Under a random one the map has a second
    term, `var(gamma) (E B) S (E B)'`, and its radius is found by ARPACK on the map as a linear
    operator on the `(N+1)^2` entries of `S`. Power iteration on the norm, the obvious
    alternative, does not converge here: the leading eigenvalues of `A` come as a complex pair,
    and the ratio of successive norms oscillates rather than settling.
    """
    from scipy.sparse.linalg import LinearOperator, eigs

    ops = model.operators()
    A_bar = ops["L_bar"] @ ops["B"]
    radius = float(np.max(np.abs(np.linalg.eigvals(A_bar))))
    if model.gamma_var == 0.0:
        ms = radius ** 2
    else:
        size = A_bar.shape[0]
        step = _ms_map(ops, model.gamma_var)
        operator = LinearOperator((size * size, size * size), dtype=float,
                                  matvec=lambda v: step(v.reshape(size, size)).ravel())
        start = np.eye(size).ravel()
        ms = float(np.max(np.abs(eigs(operator, k=1, which="LM", v0=start,
                                      return_eigenvectors=False, tol=1e-12))))
    psi_radius = float(np.max(np.abs(np.linalg.eigvals(model.Psi_op))))
    return {"radius": radius, "ms_radius": float(ms),
            "decoupled": max(abs(model.rho), psi_radius), "stationary": bool(ms < 1.0)}


def moments(model, tol=1e-14, max_iter=10_000):
    """Stationary mean and covariance of `Y_t = (J_t, X_t)` on the grid. Returns a dict with
    mean_J, mean_X (N,), var_J, cov_X (N, N), cov the whole (N+1, N+1) covariance, lag1 the
    lag-one autocovariance `Cov(Y_t, Y_{t-1})`, r2_jump the population R-squared of the jump
    equation, `1 - sigma_eta^2 / Var(J)`, and acf1_J the jump's lag-one autocorrelation.

    The mean solves `m = L_bar (c + B m)`. Written out, that is the notes' equation for the
    mean curve,

        m(s) = mu(s) - gamma_bar G(s) E J + int Psi(s, r) m(r) dr,

    together with the one the notes leave implicit, `E J = alpha + rho E J + <beta, m>`: the
    two are coupled, and neither determines its unknown without the other.

    The covariance solves `Sigma = L_bar (B Sigma B' + W) L_bar' + var(gamma) E (B Sigma B' + W
    + z z') E'` with `z = c + B m`, which is a discrete Lyapunov equation when `gamma` is fixed
    and is iterated to its fixed point otherwise. The second term is what a random `gamma_t`
    adds, and it involves the mean because `gamma_t` multiplies the whole of `J_t`, mean
    included.

    PARAMETERS
    tol      : float, convergence tolerance of the fixed-point iteration, relative;
    max_iter : int, iterations allowed before giving up.
    """
    check = stationarity(model)
    if not check["stationary"]:
        raise ValueError(f"no stationary solution: mean-square radius {check['ms_radius']:.4f}.")
    ops = model.operators()
    B, L, E, c, W = ops["B"], ops["L_bar"], ops["E"], ops["c"], ops["W"]
    A_bar = L @ B
    size = A_bar.shape[0]
    mean = np.linalg.solve(np.eye(size) - A_bar, L @ c)

    Q = L @ W @ L.T
    if model.gamma_var == 0.0:
        cov = solve_discrete_lyapunov(A_bar, Q)
    else:
        z = c + B @ mean
        extra = model.gamma_var * (E @ (W + np.outer(z, z)) @ E.T)
        step = _ms_map(ops, model.gamma_var)
        cov = Q + extra
        for _ in range(int(max_iter)):
            nxt = step(cov) + Q + extra
            if np.linalg.norm(nxt - cov) <= tol * np.linalg.norm(nxt):
                cov = nxt
                break
            cov = nxt
        else:
            raise RuntimeError("the covariance iteration did not converge.")
    cov = 0.5 * (cov + cov.T)
    lag1 = A_bar @ cov
    return {"mean_J": float(mean[0]), "mean_X": mean[1:], "var_J": float(cov[0, 0]),
            "cov_X": cov[1:, 1:], "cov": cov, "lag1": lag1,
            "r2_jump": 1.0 - model.sigma_eta ** 2 / cov[0, 0] if cov[0, 0] > 0 else np.nan,
            "acf1_J": float(lag1[0, 0] / cov[0, 0]) if cov[0, 0] > 0 else np.nan}


def calibrate(model, mean_jump, mean_close, var_jump, var_close):
    """Choose `alpha`, the drift, `sigma_eta` and the curve-noise level so that the stationary
    mean and variance of the jump and of the close `X_t(1)` match the four targets exactly.
    Returns a JointModel.

    Everything else is held at the model's values, which is what makes the problem linear.
    The stationary mean is linear in `(alpha, drift)` with `mu(s) = drift * s`. The stationary
    covariance is linear in `(sigma_eta^2, noise level)` under a fixed `gamma`, and affine in
    them under a random one, where the mean enters as well; so the means are matched first and
    the variances second, each by evaluating the model at three points and solving.

    Raises ValueError where the targets cannot be met with non-negative variances, which
    happens when the rest of the model already explains more variance than the target has.

    PARAMETERS
    mean_jump, mean_close : float, targets for `E J_t` and `E X_t(1)`;
    var_jump, var_close   : float, targets for `Var J_t` and `Var X_t(1)`.
    """
    grid = model.grid
    noise_shape = model.cell_var / model.cell_var.sum() if model.cell_var.sum() > 0 else \
        np.diff(grid)

    def means(alpha, drift):
        m = moments(model.with_params(alpha=alpha, mu=drift * grid))
        return np.array([m["mean_J"], m["mean_X"][-1]])

    base = means(0.0, 0.0)
    A = np.column_stack([means(1.0, 0.0) - base, means(0.0, 1.0) - base])
    alpha, drift = np.linalg.solve(A, np.array([mean_jump, mean_close]) - base)
    fitted = model.with_params(alpha=alpha, mu=drift * grid)

    def variances(s2_eta, level):
        m = moments(fitted.with_params(sigma_eta=np.sqrt(s2_eta), cell_var=level * noise_shape))
        return np.array([m["var_J"], m["cov_X"][-1, -1]])

    base = variances(0.0, 0.0)
    A = np.column_stack([variances(1.0, 0.0) - base, variances(0.0, 1.0) - base])
    s2_eta, level = np.linalg.solve(A, np.array([var_jump, var_close]) - base)
    if s2_eta < 0 or level < 0:
        raise ValueError(f"the targets need negative variances (sigma_eta^2 = {s2_eta:.4g}, "
                         f"noise level = {level:.4g}); the rest of the model already explains "
                         f"more than the target variance.")
    return fitted.with_params(sigma_eta=np.sqrt(s2_eta), cell_var=level * noise_shape)


def scale_to_r2(model, r2, lo=1e-6, hi=1e3):
    """Rescale `beta` so the jump equation's population R-squared is `r2`, holding the jump
    innovation fixed. Returns a JointModel.

    This is the way to compare two shapes of `beta` fairly -- or one shape at two signal
    strengths -- since the R-squared, not the amplitude, is what decides how much there is
    to recover. The R-squared counts what `rho J_{t-1}` explains as well, so it cannot be set
    below the value it takes with `beta` zero.

    PARAMETERS
    r2     : float, target population R-squared of the jump equation;
    lo, hi : float, bracket for the scale factor.
    """
    from scipy.optimize import brentq

    floor = moments(model.with_params(beta=np.zeros_like(model.beta)))["r2_jump"]
    if not floor < r2 < 1:
        raise ValueError(f"r2 must lie in ({floor:.4g}, 1), the floor being what rho explains.")

    def gap(f):
        candidate = model.with_params(beta=f * model.beta)
        if not stationarity(candidate)["stationary"]:
            return 1.0
        return moments(candidate)["r2_jump"] - r2

    return model.with_params(beta=brentq(gap, lo, hi, xtol=1e-12) * model.beta)


# ------------------------------------------------------------- levels, returns, forecasts


def step_functional(grid, theta, on="level"):
    """The grid functional of a step coefficient that switches on at `theta`. Returns (N,) `c`
    such that `X @ c` is the regressor the step builds from a curve `X`.

    On the level, `c @ X = int_theta^1 X(u) du` under the trapezium weights: the notes'
    `beta = 1{u >= theta}`. On the returns, `c @ X = X(1) - X(theta)`: the return made after
    `theta`, which is `int 1{v >= theta} dX(v)`. `theta` is moved to the first grid point at or
    after it in both, so the two steps switch on at the same place.

    The two are the same kind of object -- a linear functional of the day's path -- in two
    coordinate systems, and integrating by parts converts one into the other: any level
    coefficient `beta` is the return coefficient `B(v) = int_v^1 beta`, so the level step is
    the return weight `1 - theta` on every return before the onset, falling linearly to zero
    after it. Which one is the simple one is a statement about the world, and
    `notebooks/jump_equation_v2.ipynb` measures what it costs to get it wrong.

    PARAMETERS
    grid  : (N,) array;
    theta : float, the onset;
    on    : {"level", "return"}.
    """
    grid = np.asarray(grid, float)
    start = int(np.argmax(grid >= theta))
    c = np.zeros(grid.size)
    if on == "level":
        c[start:] = quadrature_weights(grid)[start:]
    elif on == "return":
        c[-1] += 1.0
        c[start] -= 1.0
    else:
        raise ValueError(f"on must be 'level' or 'return', got {on!r}.")
    return c


def population_fit(model, functionals, mom=None):
    """The population least-squares fit of the jump on its own lag and on chosen functionals
    of yesterday's curve. Returns a dict with r2, coef (k + 1,) the slopes on `J_{t-1}` and
    on each functional, and resid_var.

    This is what a regression of `J_t` on `(1, J_{t-1}, C X_{t-1})` would converge to with
    infinitely many days, computed from the stationary moments rather than estimated. With
    the true functional `beta * w` it returns the model's own R-squared, `1 - sigma_eta^2 /
    Var J`; with anything else it returns the best that restricted model can do, which is how
    a misspecified coefficient function is costed without sampling error in the way.

    PARAMETERS
    model       : JointModel;
    functionals : (k, N) array, one grid functional per row, so `C @ X_{t-1}` are the
                  regressors (see `step_functional`); a single (N,) row is accepted, and None
                  fits the lag alone, which is the floor every curve term is measured from;
    mom         : dict or None, the output of `moments(model)`, to avoid recomputing it.
    """
    mom = moments(model) if mom is None else mom
    N = model.grid.size
    if functionals is None:
        C = np.zeros((0, N))
    else:
        C = np.atleast_2d(np.asarray(functionals, float))
    if C.shape[1] != N:
        raise ValueError(f"functionals have {C.shape[1]} columns, the grid has {N} points.")
    M = np.zeros((C.shape[0] + 1, N + 1))
    M[0, 0] = 1.0
    M[1:, 1:] = C
    cov_r = M @ mom["cov"] @ M.T
    cov_rj = M @ mom["lag1"][0]
    coef = np.linalg.solve(cov_r, cov_rj)
    explained = float(coef @ cov_rj)
    return {"r2": explained / mom["var_J"], "coef": coef,
            "resid_var": mom["var_J"] - explained}


def forecast(model, J_prev, X_prev):
    """What the model expects of tomorrow, given today. Returns the expected jump (float)
    and the expected curve (N,), conditional on `J_{t-1} = J_prev` and `X_{t-1} = X_prev`.

    `E J_t = alpha + rho J_prev + <beta, X_prev>`, and the curve follows from it, `E X_t =
    mu - gamma_bar G E J_t + Psi X_prev`, because `gamma_t` and the noise are independent of
    the jump. The two pieces of the curve forecast are worth separating when reading it: the
    `Psi` term is how today's session carries into tomorrow's directly, and the `G` term is
    the share of tomorrow's expected gap the session is expected to give back.

    PARAMETERS
    model  : JointModel;
    J_prev : float, today's jump;
    X_prev : (N,) array, today's curve.
    """
    X_prev = np.asarray(X_prev, float)
    signal = float((model.beta * model.w) @ X_prev)
    expected_jump = model.alpha + model.rho * float(J_prev) + signal
    curve = model.mu - model.gamma_mean * model.G * expected_jump + model.Psi_op @ X_prev
    return expected_jump, curve


# ------------------------------------------------------------------------- the simulation


def _draw_gamma(model, n, rng):
    if model.gamma_dist == "fixed":
        return np.full(n, model.gamma_mean)
    half = np.sqrt(3.0 * model.gamma_var)
    return rng.uniform(model.gamma_mean - half, model.gamma_mean + half, size=n)


def _draw_eta(model, n, rng):
    if model.eta_df is None:
        return rng.normal(0.0, model.sigma_eta, size=n)
    df = model.eta_df
    return model.sigma_eta * np.sqrt((df - 2.0) / df) * rng.standard_t(df, size=n)


def simulate(model, n, burn_in=200, rng=None):
    """Draw `n` consecutive days of jumps and curves. Returns a dict with J (n,), X (n, N) the
    CIDR curves, gamma (n,) the gap-fill coefficients drawn, eta (n,), eps (n, N), grid, and
    model.

    Day `t` is the jump `J_t` that opens it followed by the curve `X_t`, so `J[t]` is formed
    from `J[t - 1]` and `X[t - 1]`, and `X[t]` from `J[t]` and `X[t - 1]` -- the timing the
    equations are written in, and the one `jump_eq.build_design` lags by. The recursion starts
    at the stationary mean and runs `burn_in` days before the first one returned.

    The curve noise is drawn exactly on the grid: its increments over the cells are
    independent Gaussians with variances `model.cell_var`, cumulated from zero.

    PARAMETERS
    n       : int, days returned;
    burn_in : int, days run and discarded first;
    rng     : int, Generator or None, seed.
    """
    rng = np.random.default_rng(rng)
    total = int(n) + int(burn_in)
    N = model.grid.size

    gamma = _draw_gamma(model, total, rng)
    eta = _draw_eta(model, total, rng)
    steps = rng.standard_normal((total, N - 1)) * np.sqrt(model.cell_var)[None, :]
    eps = np.concatenate([np.zeros((total, 1)), np.cumsum(steps, axis=1)], axis=1)

    start = moments(model) if stationarity(model)["stationary"] else None
    J_prev = start["mean_J"] if start else 0.0
    X_prev = start["mean_X"] if start else np.zeros(N)
    beta_w = model.beta * model.w

    J = np.empty(total)
    X = np.empty((total, N))
    for t in range(total):
        J[t] = model.alpha + model.rho * J_prev + beta_w @ X_prev + eta[t]
        X[t] = model.mu - gamma[t] * model.G * J[t] + model.Psi_op @ X_prev + eps[t]
        J_prev, X_prev = J[t], X[t]

    keep = slice(int(burn_in), total)
    return {"J": J[keep], "X": X[keep], "gamma": gamma[keep], "eta": eta[keep],
            "eps": eps[keep], "grid": model.grid, "model": model}


def price_curves(J, X, p0=0.0):
    """The log-price curves `P_t(s) = P_{t-1}(1) + J_t + X_t(s)`, laid end to end. Returns
    (n, N), in the units of `J` and `X`.

    This is the identity the notes build the model on, run forwards from `P_{-1}(1) = p0`, so
    `P_t(0) - P_{t-1}(1)` returns `J_t` and `P_t(s) - P_t(0)` returns `X_t(s)` exactly.

    PARAMETERS
    J  : (n,) array of jumps;
    X  : (n, N) array of CIDR curves;
    p0 : float, the close before the first jump.
    """
    J = np.asarray(J, float)
    X = np.atleast_2d(np.asarray(X, float))
    close_before = p0 + np.concatenate([[0.0], np.cumsum(J + X[:, -1])[:-1]])
    return (close_before + J)[:, None] + X
