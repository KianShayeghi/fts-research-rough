"""Rebuild the cached rolling-origin forecasts and updating sweep in `artifacts/cache/`.

    python scripts/make_ftsa_cache.py            # only what is missing
    python scripts/make_ftsa_cache.py --refresh  # re-run and overwrite

Four CSVs. The first two hold the one-step-ahead forecast and prediction band every method
produced for each held-out session, together with what was realised: one for the replication
design of `ftsa.evaluation.CONFIG`, the S&P 500 over the first half of 2017, and one for the
same evaluation across the nine instruments of the cached five-minute panel, which is what says
whether a conclusion drawn on the first is about the method or about the window.

The other two are the updating sweep of `ftsa.updating`, which revises each day's forecast at
nine points through the session. Those keep per-session sums and per-point averages rather than
every forecast, the cross of update points, methods and anchorings being wide enough that a
pointwise cache would repeat each realised curve thirteen times.

Every accuracy figure in `notebooks/shang_replication.ipynb` and
`notebooks/shang_updating.ipynb` is derived from these, so every method is compared on identical
splits and the notebooks open without re-running the evaluation, which takes minutes, most of
them spent searching ARIMA orders afresh at every split.

The design is `ftsa.evaluation.CONFIG` with `ftsa.updating.UPDATE_FRACTIONS`, and the bootstrap
is seeded from it, so a rebuild on the same configuration reproduces the committed files.
"""

import os
import sys
import time

from fts_research.core import paths
from fts_research.ftsa import evaluation as ev
from fts_research.ftsa import updating as up

# (files written, what builds them, what summarises the result for the report below).
BUILDS = (
    ((ev.ARTIFACT,), ev.cached_rolling, lambda frame: ev.summarise(frame, by="method")),
    ((ev.PANEL_ARTIFACT,), ev.cached_panel,
     lambda frame: ev.summarise(frame, by=["ticker", "method"])),
    ((up.ARTIFACT, up.PROFILE_ARTIFACT), up.cached_updating,
     lambda frame: up.summarise_updates(frame).xs(False, level="anchored")["skill"]
     .unstack("method")),
)
MAX_ROWS = 40


def main(argv):
    """Rebuild every cache, then report where each was written and what it says."""
    refresh = "--refresh" in argv
    started = time.time()
    print(f"cache to : {paths.CACHE_DIR}")

    for artifacts, build, summarise in BUILDS:
        frame = build(refresh=refresh)
        for artifact in artifacts:
            size = os.path.getsize(paths.cache_path(artifact)) / 1024
            print(f"\n  {artifact:<32} {size:9.1f} KB")
        table = summarise(frame)
        print(table.head(MAX_ROWS).to_string())

    print(f"\nin {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
