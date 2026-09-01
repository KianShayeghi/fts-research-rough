"""Checks that `data/manifest.json` still describes the files committed under `data/`.

Git records that a CSV changed; it does not record what the new one is a snapshot of. The
intraday panels are why that matters: the endpoint serves a window ending on the day it is
asked, so re-fetching replaces the file every number in `notebooks/overnight_jumps.ipynb` was
computed on, and the replacement looks exactly as legitimate. The manifest pins each file to a
digest, a retrieval date and, where it is a window of sessions, the window it covers; these
tests fail the moment the working tree and that record disagree.

One file is the other kind. `spx_1m_2017H1.csv` was imported rather than fetched, and no public
endpoint serves it, so it cannot be silently replaced and cannot be recovered either. It is
required to declare that in the record with `fetchable: false` rather than to be quietly
excused, and nothing else may.
"""

import datetime
import os
import re

import pytest

from fts_research.core import paths, provenance
from fts_research.fdaSP import empirical as emp, intraday as itd

MANIFEST = provenance.load()
NAMES = sorted(MANIFEST)
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def test_the_manifest_is_not_empty():
    assert NAMES, "data/manifest.json records nothing; run the fetchers to rebuild it"


@pytest.mark.parametrize("name", NAMES)
def test_every_recorded_file_is_present_and_unchanged(name):
    """The digest check. A file that has moved on since it was recorded is a different
    dataset, whatever its name says."""
    assert provenance.verify(name) == []


def test_nothing_in_the_data_directory_is_unrecorded():
    """A CSV fetched by hand, outside `ensure_dataset` and `ensure_panel`, has no provenance
    at all -- this is where that is caught, before it reaches a commit."""
    assert provenance.verify() == []


@pytest.mark.parametrize("name", NAMES)
def test_every_entry_carries_a_source_a_date_and_a_digest(name):
    """A dataset either cites the URL it was fetched from, or declares in the record itself
    that it cannot be fetched.

    The second is not a loophole. `spx_1m_2017H1.csv` is a fixed historical window that no
    public endpoint serves, so it has the opposite failure mode to the rolling snapshots: it
    will never be silently replaced, and it can never be recovered if it is lost. Marking it
    `fetchable: false` is the declaration of that, and it is still required to say in prose
    what it is -- an entry with an empty source fails either way.
    """
    entry = MANIFEST[name]
    if entry.get("fetchable", True):
        assert entry["source"].startswith("https://"), f"{name} must cite an https source"
    else:
        assert len(entry["source"]) > 60, (
            f"{name} declares itself unfetchable, so its source must say what it is instead")
    assert ISO_DATE.match(entry["retrieved"]), f"{name} needs an ISO retrieval date"
    assert len(entry["sha256"]) == 64
    assert entry["bytes"] > 0 and entry["rows"] > 0


def test_an_unfetchable_dataset_is_the_documented_exception():
    """Only the imported panel may skip the URL, and it must be the one `intraday` names."""
    unfetchable = sorted(n for n in NAMES if not MANIFEST[n].get("fetchable", True))
    assert unfetchable == [itd.SPX_2017["file"]], (
        f"an undeclared dataset has appeared without a source: {unfetchable}")


def test_the_imported_panel_records_the_window_it_covers():
    """Its window is what makes it usable as a replication design, so it is recorded and read
    back off the file the same way the snapshots' windows are."""
    entry = MANIFEST[itd.SPX_2017["file"]]
    sessions = itd.load_spx_2017()["session"]
    assert entry["span"] == [str(sessions.min()), str(sessions.max())]
    assert entry["span"] == ["2017-01-03", "2017-06-30"]


@pytest.mark.parametrize("name", NAMES)
def test_no_file_claims_to_have_been_fetched_in_the_future(name):
    taken = datetime.date.fromisoformat(MANIFEST[name]["retrieved"])
    assert taken <= datetime.date.today()


def test_every_dataset_the_loaders_know_about_is_recorded():
    """`SOURCES` and `PANELS` are what can be fetched; the manifest is what is committed. A
    name in the first and not the second means the repository does not ship that data."""
    expected = (set(emp.SOURCES) | {spec["file"] for spec in itd.PANELS.values()}
                | {itd.SPX_2017["file"]})
    assert expected <= set(MANIFEST), f"not recorded: {sorted(expected - set(MANIFEST))}"


@pytest.mark.parametrize("name", sorted(spec["file"] for spec in itd.PANELS.values()))
def test_each_intraday_panel_records_the_window_it_covers(name):
    """The panels are dated snapshots, so the window is the part that cannot be recovered
    from the file name or from git."""
    span = MANIFEST[name]["span"]
    assert len(span) == 2 and all(ISO_DATE.match(d) for d in span)
    assert span[0] < span[1]
    assert span[1] <= MANIFEST[name]["retrieved"]


def test_the_panel_windows_agree_with_the_files_themselves():
    """`snapshot_span` reads the window off the bars; the manifest claims it independently."""
    for key, spec in itd.PANELS.items():
        start, end = itd.snapshot_span(key)
        assert [str(start), str(end)] == MANIFEST[spec["file"]]["span"]


def test_the_five_minute_panel_is_the_shorter_window():
    """The endpoint trades resolution against history, which is why two panels are kept."""
    fine = MANIFEST[itd.PANELS["5m"]["file"]]["span"]
    long = MANIFEST[itd.PANELS["1h"]["file"]]["span"]
    assert long[0] < fine[0] and long[1] >= fine[1]


def test_sha256_of_reads_the_file_rather_than_its_name(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"x,y\n1,2\n")
    b.write_bytes(b"x,y\n1,2\n")
    assert provenance.sha256_of(str(a)) == provenance.sha256_of(str(b))
    b.write_bytes(b"x,y\n1,3\n")
    assert provenance.sha256_of(str(a)) != provenance.sha256_of(str(b))


def test_record_then_verify_round_trips(tmp_path, monkeypatch):
    """What a fetch writes is what a later check reads, and a change to the file is caught."""
    monkeypatch.setattr(paths, "DATA_DIR", str(tmp_path))
    target = tmp_path / "toy.csv"
    target.write_bytes(b"t,x\n0,1\n1,2\n")
    entry = provenance.record("toy.csv", str(target), "https://example.invalid/toy.csv",
                              span=["2026-01-01", "2026-01-02"])
    assert entry["rows"] == 2
    assert entry["retrieved"] == datetime.date.today().isoformat()
    assert provenance.verify() == []

    target.write_bytes(b"t,x\n0,9\n1,2\n")
    problems = provenance.verify()
    assert len(problems) == 1 and "has changed" in problems[0]


def test_verify_reports_a_file_that_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", str(tmp_path))
    target = tmp_path / "toy.csv"
    target.write_bytes(b"t,x\n0,1\n")
    provenance.record("toy.csv", str(target), "https://example.invalid/toy.csv")
    os.remove(target)
    problems = provenance.verify()
    assert len(problems) == 1 and "not present" in problems[0]


def test_verify_reports_an_unrecorded_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", str(tmp_path))
    (tmp_path / "stray.csv").write_bytes(b"t,x\n0,1\n")
    problems = provenance.verify()
    assert len(problems) == 1 and "nothing recorded it" in problems[0]


def test_verify_names_a_file_it_has_never_heard_of():
    assert provenance.verify("no_such_file.csv") == [
        "no_such_file.csv is not recorded in manifest.json"]
