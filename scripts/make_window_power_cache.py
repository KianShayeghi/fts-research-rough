"""Rebuild the cached window-DGP power study and estimator comparison in `artifacts/cache/`.

    python scripts/make_window_power_cache.py                    # only what is missing
    python scripts/make_window_power_cache.py --refresh          # re-run and overwrite
    python scripts/make_window_power_cache.py --only power       # just one of the two
    python scripts/make_window_power_cache.py --jobs 8           # worker processes

Three CSVs. `window_power.csv` holds one row per (n, replication, cell, test): the p-value of
each test of `beta = 0` and the M it used. `window_methods.csv` holds one row per (n,
replication, cell, method): every metric of `jump.window_power.score_methods`, kept per
replication so that paired comparisons need no refit. `window_methods_support.csv` holds how
often each method left each grid point in the support. `notebooks/window_jump_power.ipynb`
reads them back.

The design is `jump.window_power.CONFIG`, and every draw is seeded from it by
`jump.window_power.rng_of`, so a rebuild on the same configuration reproduces the committed
files. The power study takes minutes; the estimator comparison two to three hours on fifteen
workers. That run checkpoints every finished task, so if it is stopped the same command
resumes it, and while it runs it asks Windows not to sleep the machine for idleness (closing
the lid or choosing Sleep still does).
"""

import os
import sys
import time

from fts_research.core import paths
from fts_research.jump import window_power as wp

# SetThreadExecutionState flags: keep the system awake while this thread runs, then release.
_ES_CONTINUOUS, _ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def _stay_awake(on):
    """Ask Windows not to idle-sleep while the study runs, or release that request. Does
    nothing elsewhere."""
    if sys.platform == "win32":
        import ctypes
        flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)


def main(argv):
    """Rebuild the caches asked for, then report where each one was written."""
    refresh = "--refresh" in argv
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    n_jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else None
    if only not in (None, "power", "methods"):
        raise SystemExit(f"--only takes 'power' or 'methods', got {only!r}")

    started = time.time()
    names = []
    _stay_awake(True)
    try:
        if only in (None, "power"):
            wp.cached_power(refresh=refresh, n_jobs=n_jobs)
            names.append(wp.POWER_ARTIFACT)
        if only in (None, "methods"):
            wp.cached_methods(refresh=refresh, n_jobs=n_jobs)
            names += [wp.METHODS_ARTIFACT, wp.SUPPORT_ARTIFACT]
    finally:
        _stay_awake(False)

    print(f"cache to : {paths.CACHE_DIR}")
    for name in names:
        path = paths.cache_path(name)
        print(f"  {name:<28} {os.path.getsize(path) / 1024:9.1f} KB")
    print(f"in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
