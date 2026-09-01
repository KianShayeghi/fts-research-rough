"""What each file committed under `data/` actually is.

Git records that a CSV changed; it does not record what the new one is a snapshot *of*. The
intraday panels are the case that matters: the Yahoo Finance endpoint serves a window ending
on the day it is asked, so re-fetching does not return the file it replaces, and a number in
`notebooks/overnight_jumps.ipynb` computed on one snapshot cannot be compared with a number
computed on another. The manifest is what makes that checkable -- for each file, where it came
from, the day it was taken, the window it covers, and its SHA-256 -- and `verify` reads the
files on disk back against it.

Entries are written by whatever fetched the file, in `fdaSP.empirical.ensure_dataset` and
`fdaSP.intraday.ensure_panel`, so provenance is recorded at the moment it is known rather
than reconstructed afterwards. One file was not fetched at all: `spx_1m_2017H1.csv` is a fixed
historical window no public endpoint serves, and it carries `fetchable: false` so that a reader
is told the committed copy is the only one rather than left to infer it from a source that does
not resolve. `tests/test_provenance.py` allows that declaration and nothing else.

`python -m fts_research.core.provenance` prints the manifest and exits non-zero if the working
tree disagrees with it, which is what `tests/test_provenance.py` asserts.
"""

import datetime
import hashlib
import json
import os
import sys

from . import paths

__all__ = [
    "MANIFEST_NAME", "manifest_path", "sha256_of", "load", "entry_for", "record", "verify",
]

MANIFEST_NAME = "manifest.json"
_CHUNK = 1 << 20


def manifest_path():
    """Absolute path of the manifest, whether or not it exists yet."""
    return paths.data_path(MANIFEST_NAME)


def sha256_of(path):
    """SHA-256 of a file, read in chunks. Returns the digest as a hex str."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_count(path):
    """Data rows in a CSV, i.e. lines after the header."""
    with open(path, "rb") as fh:
        return max(sum(block.count(b"\n") for block in iter(lambda: fh.read(_CHUNK), b"")) - 1, 0)


def load():
    """The manifest as a dict keyed by file name, empty where it does not exist yet."""
    path = manifest_path()
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def entry_for(name):
    """The manifest entry for one file. Returns a dict, or None if it is not recorded."""
    return load().get(name)


def _write(manifest):
    path = manifest_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def record(name, path, source, **extra):
    """Record what was just written to `path`, and return the entry (dict).

    The date is today's, since this is called by the fetch itself. Anything a particular
    dataset needs beyond the common fields -- the session window of an intraday panel, say --
    is passed through `extra` and stored alongside them.

    PARAMETERS
    name   : str, the file name, which keys the manifest;
    path   : str, where the file was written;
    source : str, the URL or other origin it was taken from.
    """
    entry = {
        "source": source,
        "retrieved": datetime.date.today().isoformat(),
        "sha256": sha256_of(path),
        "bytes": os.path.getsize(path),
        "rows": _line_count(path),
    }
    entry.update(extra)
    manifest = load()
    manifest[name] = entry
    _write(manifest)
    return entry


def verify(name=None):
    """Check the files under `data/` against the manifest. Returns a list of problems, empty
    where the two agree.

    A recorded file that is missing, or whose digest has moved, is reported; so is a CSV
    present in the directory that nothing recorded, which is how a snapshot taken outside
    `ensure_dataset` and `ensure_panel` is caught before it reaches a commit.

    PARAMETERS
    name : str or None, check one recorded file rather than the whole directory.
    """
    manifest = load()
    problems = []
    names = [name] if name is not None else sorted(manifest)
    if name is not None and name not in manifest:
        return [f"{name} is not recorded in {MANIFEST_NAME}"]

    for key in names:
        entry = manifest[key]
        path = paths.data_path(key)
        if not os.path.exists(path):
            problems.append(f"{key} is recorded but not present at {path}")
            continue
        got = sha256_of(path)
        if got != entry["sha256"]:
            problems.append(
                f"{key} has changed since it was recorded: sha256 {got[:12]}... on disk, "
                f"{entry['sha256'][:12]}... recorded on {entry['retrieved']}"
            )

    if name is None:
        listing = os.listdir(paths.DATA_DIR) if os.path.isdir(paths.DATA_DIR) else []
        for extra in sorted(set(listing) - set(manifest) - {MANIFEST_NAME}):
            if extra.endswith(".csv"):
                problems.append(f"{extra} is in data/ but nothing recorded it")
    return problems


def _main(argv):
    manifest = load()
    print(f"data directory: {paths.DATA_DIR}")
    for name in sorted(manifest):
        e = manifest[name]
        span = f"  {e['span'][0]} to {e['span'][1]}" if "span" in e else ""
        print(f"  {name:<22} {e['bytes'] / 1024:9.1f} KB  {e['rows']:>6} rows  "
              f"taken {e['retrieved']}{span}")
        print(f"  {'':<22} sha256 {e['sha256']}")
    problems = verify()
    for p in problems:
        print(f"MISMATCH: {p}", file=sys.stderr)
    if "--check" in argv and not manifest:
        print("MISMATCH: the manifest is empty", file=sys.stderr)
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
