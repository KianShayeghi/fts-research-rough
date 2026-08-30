"""Shared helpers for the test suite."""

import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tracked_notebooks():
    """Notebook paths under `notebooks/` that git is tracking, repo-relative and sorted.

    A notebook that is not yet committed is work in progress, so the documentation and
    structure checks leave it alone until it is added; everything committed is held to them.
    Where git is unavailable, or this is not a checkout, every notebook present is returned.

    Returns a list of str.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "notebooks/*.ipynb"],
            cwd=REPO, capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            names = [p for p in out.stdout.split("\0") if p.endswith(".ipynb")]
            return sorted(p for p in names if os.path.exists(os.path.join(REPO, p)))
    except (OSError, subprocess.SubprocessError):
        pass

    here = os.path.join(REPO, "notebooks")
    if not os.path.isdir(here):
        return []
    return sorted(f"notebooks/{n}" for n in os.listdir(here) if n.endswith(".ipynb"))
