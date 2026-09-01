"""Rebuild the null-set selection study `notebooks/region_selection.ipynb` reads.

The study refits the overlap group lasso on a hundred subsamples per replication, which takes
tens of minutes over the four designs, so the result is committed under `artifacts/cache/` and
the notebook opens without recomputing it. Nothing here needs R or the network.

    python scripts/make_region_cache.py --refresh
    python scripts/make_region_cache.py --refresh --n-reps 50

Run without `--refresh` it reports what is cached and rebuilds nothing, which is what the
notebook and `tests/test_region.py` rely on.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from fts_research.core import paths                                            # noqa: E402
from fts_research.region import study                                          # noqa: E402

CACHE = "region_study.csv"


def build(n_reps, n, snr, n_pairs, n_lambda, n_folds, verbose=True):
    """Run the study and write it to `artifacts/cache/`. Returns the DataFrame.

    PARAMETERS
    n_reps                          : int, seeds per design;
    n, snr                          : int and float, the simulated sample and its
                                      signal-to-noise ratio;
    n_pairs, n_lambda, n_folds      : int, as `region.study.run_once` takes them;
    verbose                         : bool, report each replication as it starts.
    """
    started = time.time()

    def progress(design, seed):
        if verbose:
            print(f"  {design} seed {seed}  ({time.time() - started:6.1f}s)", flush=True)

    frame = study.run_study(n_reps=n_reps, n=n, snr=snr, n_pairs=n_pairs, n_lambda=n_lambda,
                            n_folds=n_folds, progress=progress)
    frame.to_csv(paths.cache_path(CACHE), index=False)
    if verbose:
        print(f"wrote {paths.cache_path(CACHE)} ({len(frame)} rows, "
              f"{time.time() - started:.0f}s)")
    return frame


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="recompute and overwrite")
    parser.add_argument("--n-reps", type=int, default=25)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--snr", type=float, default=4.0)
    parser.add_argument("--n-pairs", type=int, default=25)
    parser.add_argument("--n-lambda", type=int, default=30)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args(argv)

    where = paths.cache_path(CACHE)
    if not args.refresh:
        print(f"{where}: {'present' if os.path.exists(where) else 'absent'}")
        print("pass --refresh to rebuild it")
        return 0
    build(args.n_reps, args.n, args.snr, args.n_pairs, args.n_lambda, args.n_folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
