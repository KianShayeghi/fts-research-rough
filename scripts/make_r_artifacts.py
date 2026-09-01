"""Regenerate the cached CRAN `fdaSP::f2sSP` reference outputs in `artifacts/reference/`.

    python scripts/make_r_artifacts.py

Needs R with the `fdaSP` package installed. The R code itself lives in `fdaSP.r_reference`,
which `notebooks/LocSpaFLR_validation.ipynb` also runs from.
"""

import os
import sys

from fts_research.core import paths
from fts_research.fdaSP import r_reference, rbridge as rb


def main():
    """Regenerate every artifact, then report where each one was written."""
    try:
        written = r_reference.regenerate_all()
    except rb.RError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Rscript      : {rb.find_rscript()}")
    print(f"reference to : {paths.REFERENCE_DIR}")
    for path in written:
        print(f"  {os.path.basename(path):<32} {os.path.getsize(path) / 1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
