"""Rebuild the cached `bench` study in `artifacts/cache/`.

    python scripts/make_study_cache.py            # only what is missing
    python scripts/make_study_cache.py --refresh  # re-run and overwrite

Three CSVs, written by one run so that they always describe the same fits: the per-replication
records, the frequency with which each method left each grid point in the support, and the
coefficient functions of the showcase replication. `notebooks/basis_and_coef_comp.ipynb` reads
them back rather than refitting, and would otherwise build them itself on first open; this is
the same work done headlessly, which is what makes the notebook's caches reproducible from a
command rather than only from the notebook.

The design is `bench.study.CONFIG` and every replication is seeded from it by
`bench.study.seed_of`, so a rebuild on the same configuration reproduces the committed CSVs.
"""

import os
import sys
import time

from fts_research.bench import study
from fts_research.core import paths


def main(argv):
    """Rebuild the three caches, then report where each one was written."""
    refresh = "--refresh" in argv
    started = time.time()
    study.cached_study(refresh=refresh)
    study.cached_curves(refresh=refresh)

    print(f"cache to : {paths.CACHE_DIR}")
    for name in (study.ARTIFACT, study.SUPPORT_ARTIFACT, study.CURVES_ARTIFACT):
        path = paths.cache_path(name)
        print(f"  {name:<28} {os.path.getsize(path) / 1024:9.1f} KB")
    print(f"in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
