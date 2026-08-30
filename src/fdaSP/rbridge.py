"""Locating and calling R, so `fdaSP.SBC` can be compared against CRAN `fdaSP::f2sSP`.

`find_rscript` locates an Rscript executable, `run_r` evaluates a snippet in it, and
`artifact_path` resolves the cached reference outputs under `artifacts/`. R's reference
outputs themselves are produced by `r_reference.py`.
"""

import glob
import os
import platform
import shutil
import subprocess

from .empirical import repo_root

__all__ = [
    "RError",
    "find_rscript",
    "r_available",
    "run_r",
    "r_path",
    "ARTIFACT_DIR",
    "artifact_path",
]

ARTIFACT_DIR = os.path.join(repo_root(), "artifacts")

_rscript_cache = []


def _candidate_dirs():
    """Directories that plausibly hold an Rscript executable, most-preferred first."""
    dirs = []
    r_home = os.environ.get("R_HOME")
    if r_home:
        dirs += [os.path.join(r_home, "bin", "x64"), os.path.join(r_home, "bin")]

    if platform.system() == "Windows":
        roots = [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "R"),
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "R"),
            os.path.join(os.environ.get("ProgramFiles(x86)", ""), "R"),
            r"C:\R",
        ]
        for root in roots:
            if root and os.path.isdir(root):
                # Newest R-x.y.z first, so a machine with several R versions picks the latest.
                for d in sorted(glob.glob(os.path.join(root, "R-*")), reverse=True):
                    dirs += [os.path.join(d, "bin", "x64"), os.path.join(d, "bin")]
    else:
        dirs += ["/usr/local/bin", "/usr/bin", "/opt/homebrew/bin"]
        for d in sorted(glob.glob("/Library/Frameworks/R.framework/Versions/*/Resources/bin"),
                        reverse=True):
            dirs.append(d)
    return [d for d in dirs if d and os.path.isdir(d)]


def find_rscript():
    """Absolute path to an `Rscript` executable, or None if R cannot be found.

    Checked in order: the `FTS_RSCRIPT` environment variable, `PATH`, `R_HOME`, then the
    standard install locations for the current platform. The result is cached.
    """
    if _rscript_cache:
        return _rscript_cache[0]

    override = os.environ.get("FTS_RSCRIPT")
    if override and os.path.isfile(override):
        _rscript_cache.append(override)
        return override

    found = shutil.which("Rscript")
    if found is None:
        exe = "Rscript.exe" if platform.system() == "Windows" else "Rscript"
        for d in _candidate_dirs():
            cand = os.path.join(d, exe)
            if os.path.isfile(cand):
                found = cand
                break

    if found is not None:
        _rscript_cache.append(found)
    return found


class RError(RuntimeError):
    """An R subprocess exited non-zero. Carries R's stdout and stderr."""

    def __init__(self, code, stdout, stderr):
        self.returncode, self.stdout, self.stderr = code, stdout, stderr
        detail = (stderr or stdout or "").strip() or "(no output)"
        super().__init__(f"Rscript exited with status {code}:\n{detail}")


def run_r(code, timeout=900, check=True):
    """Run an R snippet and return the `CompletedProcess`.

    The working directory is the repository root, and `--vanilla` is passed so a user's
    `.Rprofile`/`.RData` cannot change the result.

    PARAMETERS
    code    : str, R source to evaluate;
    timeout : float, seconds before the subprocess is killed;
    check   : bool, raise `RError` carrying R's own message on a non-zero exit (default True).
    """
    rscript = find_rscript()
    if rscript is None:
        raise RError(
            -1, "",
            "No Rscript found. Install R, put Rscript on PATH, or set the FTS_RSCRIPT "
            "environment variable to its full path.",
        )
    proc = subprocess.run(
        [rscript, "--vanilla", "-e", code],
        capture_output=True, text=True, timeout=timeout, cwd=repo_root(),
    )
    if check and proc.returncode != 0:
        raise RError(proc.returncode, proc.stdout, proc.stderr)
    return proc


def r_available(package="fdaSP"):
    """True if Rscript can be found and `package` is installed in it."""
    if find_rscript() is None:
        return False
    try:
        out = run_r(f'cat(requireNamespace("{package}", quietly = TRUE))',
                    timeout=120, check=False)
    except (OSError, subprocess.SubprocessError, RError):
        return False
    return out.returncode == 0 and out.stdout.strip().endswith("TRUE")


def r_path(path):
    """Absolute path as an R string literal, with forward slashes.

    R reads a backslash as an escape inside a string literal, so a Windows path pasted into R
    source is a syntax error. Forward slashes work on every platform.
    """
    return os.path.abspath(path).replace("\\", "/")


def artifact_path(name, create=True):
    """Absolute path of a cached R reference output under `artifacts/`.

    PARAMETERS
    name   : str, file name inside the artifacts directory;
    create : bool, create the artifacts directory if it does not exist (default True);
             R's `write.csv` fails outright if it is missing.
    """
    if create:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
    return os.path.join(ARTIFACT_DIR, name)
