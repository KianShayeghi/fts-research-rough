"""Regenerate the cached CRAN `fdaSP::f2sSP` reference outputs in `artifacts/`.

    python scripts/make_r_artifacts.py

Needs R with the `fdaSP` package installed. The R code itself lives in `fdaSP.r_reference`,
which `notebooks/LocSpaFLR_validation.ipynb` also runs from.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fdaSP import r_reference, rbridge as rb   # noqa: E402


def main():
    """Regenerate every artifact, then report where each one was written."""
    try:
        written = r_reference.regenerate_all()
    except rb.RError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Rscript      : {rb.find_rscript()}")
    print(f"artifacts to : {rb.ARTIFACT_DIR}")
    for path in written:
        print(f"  {os.path.basename(path):<32} {os.path.getsize(path) / 1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
