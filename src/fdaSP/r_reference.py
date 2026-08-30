"""Reference outputs from CRAN `fdaSP::f2sSP`, cached under `artifacts/`.

Each function runs one R snippet from `R_SNIPPETS`, writes the result as a CSV under
`artifacts/`, and returns its path: `basis` the B-spline basis `f2sSP` builds internally,
`fixed_lambda_fit` its Tecator fit at a fixed grid of lambdas together with the design it
fitted on, `lambda_path` its whole automatically chosen lambda path on the same data, and
`tecator_cv` the cross-validation curve behind the paper's own tuning rule. The first three
are what `notebooks/LocSpaFLR_validation.ipynb` and `tests/test_f2ssp_match.py` compare
`fdaSP.SBC` against; the last is what `notebooks/tecator_application.ipynb` reproduces
section 62.5 with.

The CSVs are committed, so both run without R; `regenerate_all` (or
`python scripts/make_r_artifacts.py`) refreshes them where R is available.
"""

import os

from . import empirical as emp
from . import rbridge as rb

__all__ = [
    "M", "ORDER", "N_GRID", "LAMBDAS", "N_LAMBDA_PATH", "N_FOLDS_CV", "CV_SEED",
    "R_SNIPPETS", "ARTIFACTS",
    "basis", "fixed_lambda_fit", "lambda_path", "tecator_cv", "regenerate_all", "cached",
]

# Kept in step with the notebook: same basis size, spline order, grid and lambda grid.
M = 15
ORDER = 4
N_GRID = 100
LAMBDAS = (0.44, 0.30, 0.10, 0.03)
N_LAMBDA_PATH = 80
N_FOLDS_CV = 10
CV_SEED = 1

ARTIFACTS = {
    "basis": ["R_basis_bs.csv"],
    "fixed_lambda_fit": ["tecator_match_Rbeta.csv", "tecator_match_XB.csv",
                         "tecator_match_lam.csv"],
    "lambda_path": ["tecator_f2sSP_beta_path.csv", "tecator_f2sSP_lam_path.csv"],
    "tecator_cv": ["tecator_f2sSP_cv.csv", "tecator_f2sSP_cv_beta.csv"],
}

# Each snippet is a format template; `{...}` placeholders are filled by the functions below.
# They are exposed so the notebook can show the reader exactly what R was asked to do.
R_SNIPPETS = {
    "basis": """
library(splines)
s <- seq(0, 1, length.out = {n_grid})
write.csv(as.data.frame(bs(s, df = {M}, intercept = TRUE)), "{out_basis}", row.names = FALSE)
""",
    "fixed_lambda_fit": """
suppressMessages(library(fdaSP)); library(splines)
df <- read.csv("{tecator}")
xc <- grep("^x_", names(df), value = TRUE)
X  <- as.matrix(df[, xc]); y <- df$protein
lam <- c({lambdas})
fit <- f2sSP(vY = y, mX = X, M = {M}, lambda1 = lam, standardize.data = TRUE,
             intercept = FALSE, splOrd = {order})
s  <- seq(0, 1, length.out = {n_grid})
mH <- t(bs(s, df = {M}, intercept = TRUE))      # exactly the basis f2sSP builds internally
write.csv(as.data.frame(fit$sp.fun.path), "{out_beta}", row.names = FALSE)
write.csv(as.data.frame(tcrossprod(X, mH)), "{out_XB}", row.names = FALSE)
write.csv(data.frame(lambda = lam), "{out_lam}", row.names = FALSE)
""",
    "tecator_cv": """
suppressMessages(library(fdaSP))
df <- read.csv("{tecator}")
xc <- grep("^x_", names(df), value = TRUE)
X  <- as.matrix(df[, xc]); y <- df$protein
set.seed({seed})
cv <- f2sSP_cv(vY = y, mX = X, M = {M}, standardize.data = TRUE, intercept = FALSE,
               splOrd = {order}, nlambda1 = {n_lambda}, cv.fold = {n_folds})
i   <- which.min(cv$mse)
thr <- cv$mse[i] + cv$mse.sd[i] / sqrt({n_folds})
j   <- min(which(cv$mse <= thr))          # lambda1 descends, so the first is the largest
write.csv(data.frame(lambda = cv$lambda1, mse = cv$mse, mse_sd = cv$mse.sd,
                     idx_min = i, idx_1se = j), "{out_cv}", row.names = FALSE)
fit <- f2sSP(vY = y, mX = X, M = {M}, lambda1 = c(cv$lambda1[j], cv$lambda1[i]),
             standardize.data = TRUE, intercept = FALSE, splOrd = {order})
write.csv(as.data.frame(t(fit$sp.fun.path)), "{out_beta}", row.names = FALSE)
""",
    "lambda_path": """
suppressMessages(library(fdaSP))
df <- read.csv("{tecator}")
xc <- grep("^x_", names(df), value = TRUE)
X  <- as.matrix(df[, xc]); y <- df$protein
fit <- f2sSP(vY = y, mX = X, M = {M}, standardize.data = TRUE, intercept = FALSE,
             splOrd = {order}, nlambda1 = {n_lambda})
write.csv(as.data.frame(fit$sp.fun.path), "{out_beta}", row.names = FALSE)
write.csv(data.frame(lambda = fit$lambda1), "{out_lam}", row.names = FALSE)
""",
}


def cached(name):
    """Absolute path of a cached artifact. Raises if it has never been generated.

    PARAMETERS
    name : str, a CSV file name listed in `ARTIFACTS`.
    """
    path = rb.artifact_path(name, create=False)
    if not os.path.exists(path):
        known = sorted(n for names in ARTIFACTS.values() for n in names)
        hint = "" if name in known else f" (known artifacts: {known})"
        raise FileNotFoundError(
            f"cached R reference {name!r} not found at {path}{hint}. Regenerate it with "
            "`python scripts/make_r_artifacts.py` (needs R with fdaSP installed)."
        )
    return path


def _tecator_for_r():
    """Path to the Tecator CSV, as an R string literal. Downloads it if not yet cached."""
    return rb.r_path(emp.ensure_dataset("tecator.csv"))


def basis():
    """`splines::bs(s, df = M, intercept = TRUE)` on the shared grid. Returns the path."""
    out = rb.artifact_path("R_basis_bs.csv")
    rb.run_r(R_SNIPPETS["basis"].format(
        n_grid=N_GRID, M=M, out_basis=rb.r_path(out)))
    return out


def fixed_lambda_fit():
    """f2sSP at the four fixed lambdas, with the design it fits on. Returns three paths."""
    beta = rb.artifact_path("tecator_match_Rbeta.csv")
    XB = rb.artifact_path("tecator_match_XB.csv")
    lam = rb.artifact_path("tecator_match_lam.csv")
    rb.run_r(R_SNIPPETS["fixed_lambda_fit"].format(
        tecator=_tecator_for_r(), lambdas=", ".join(str(v) for v in LAMBDAS),
        M=M, order=ORDER, n_grid=N_GRID,
        out_beta=rb.r_path(beta), out_XB=rb.r_path(XB), out_lam=rb.r_path(lam)))
    return beta, XB, lam


def lambda_path():
    """f2sSP's own automatically chosen lambda path and its solutions. Returns two paths."""
    beta = rb.artifact_path("tecator_f2sSP_beta_path.csv")
    lam = rb.artifact_path("tecator_f2sSP_lam_path.csv")
    rb.run_r(R_SNIPPETS["lambda_path"].format(
        tecator=_tecator_for_r(), M=M, order=ORDER, n_lambda=N_LAMBDA_PATH,
        out_beta=rb.r_path(beta), out_lam=rb.r_path(lam)))
    return beta, lam


def tecator_cv():
    """`f2sSP_cv` on Tecator: the paper's own tuning rule, run by the authors' own package.

    Section 62.5 selects lambda by minimising cross-validated prediction error and then
    applying the one-standard-error rule, so this caches the whole CV curve -- lambda, the
    mean squared error and its spread across folds -- rather than only the selected fit,
    which is what lets the notebook apply either rule and see where each lands relative to
    the sparse part of the path. The two fitted coefficient functions come with it.

    Returns two paths.
    """
    cv = rb.artifact_path("tecator_f2sSP_cv.csv")
    beta = rb.artifact_path("tecator_f2sSP_cv_beta.csv")
    rb.run_r(R_SNIPPETS["tecator_cv"].format(
        tecator=_tecator_for_r(), M=M, order=ORDER, n_lambda=N_LAMBDA_PATH,
        n_folds=N_FOLDS_CV, seed=CV_SEED,
        out_cv=rb.r_path(cv), out_beta=rb.r_path(beta)))
    return cv, beta


def regenerate_all():
    """Rebuild every cached artifact from R. Returns the list of paths written.

    Raises `fdaSP.rbridge.RError` if R or the `fdaSP` package is unavailable, rather than
    silently leaving the cache as it was.
    """
    if not rb.r_available("fdaSP"):
        raise rb.RError(
            -1, "",
            f"R with the fdaSP package is required to regenerate artifacts. "
            f"Rscript found: {rb.find_rscript()}. Install R, then "
            "install.packages('fdaSP'), or set FTS_RSCRIPT to the right Rscript.",
        )
    return [basis(), *fixed_lambda_fit(), *lambda_path(), *tecator_cv()]
