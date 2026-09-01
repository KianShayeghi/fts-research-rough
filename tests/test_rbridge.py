"""Tests for `fdaSP.rbridge`, the layer that finds and calls R.

These are the checks that make the validation notebook portable: R must be discoverable
without a hard-coded path, paths handed to R must survive Windows backslashes, and a failing
R call must surface R's own error text rather than a bare exit status.
"""

import os

import pytest

from fts_research.core import paths
from fts_research.fdaSP import rbridge as rb


def test_r_path_uses_forward_slashes_and_is_absolute():
    """Backslashes are escape characters inside an R string literal, so they must not survive."""
    out = rb.r_path(os.path.join("data", "tecator.csv"))
    assert "\\" not in out
    assert out.endswith("data/tecator.csv")
    assert os.path.isabs(out.replace("/", os.sep))


def test_r_path_is_idempotent():
    once = rb.r_path("data/tecator.csv")
    assert rb.r_path(once) == once


def test_find_rscript_returns_a_real_file_or_none():
    found = rb.find_rscript()
    assert found is None or os.path.isfile(found)


def test_find_rscript_honours_the_override(monkeypatch, tmp_path):
    fake = tmp_path / "Rscript"
    fake.write_text("")
    monkeypatch.setattr(rb, "_rscript_cache", [])
    monkeypatch.setenv("FTS_RSCRIPT", str(fake))
    assert rb.find_rscript() == str(fake)


def test_run_r_without_R_raises_a_helpful_error(monkeypatch):
    monkeypatch.setattr(rb, "find_rscript", lambda: None)
    with pytest.raises(rb.RError, match="FTS_RSCRIPT"):
        rb.run_r("cat(1)")


def test_r_available_is_false_without_R(monkeypatch):
    monkeypatch.setattr(rb, "find_rscript", lambda: None)
    assert rb.r_available("fdaSP") is False


def test_r_error_carries_its_output():
    err = rb.RError(3, "some stdout", "some stderr")
    assert err.returncode == 3
    assert "some stderr" in str(err)


def test_cached_artifact_missing_raises_with_a_fix(monkeypatch, tmp_path):
    from fts_research.fdaSP import r_reference as rref
    monkeypatch.setattr(paths, "REFERENCE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="make_r_artifacts"):
        rref.cached("R_basis_bs.csv")


def test_r_snippets_cover_every_artifact():
    from fts_research.fdaSP import r_reference as rref
    assert set(rref.R_SNIPPETS) == set(rref.ARTIFACTS)
    for name in rref.ARTIFACTS:
        assert callable(getattr(rref, name))


@pytest.mark.needs_r
def test_run_r_executes_and_captures_stdout():
    out = rb.run_r('cat(2 + 2)')
    assert out.stdout.strip().endswith("4")


@pytest.mark.needs_r
def test_run_r_surfaces_R_error_messages():
    with pytest.raises(rb.RError, match="deliberate failure"):
        rb.run_r('stop("deliberate failure")')


@pytest.mark.needs_r
def test_run_r_can_write_to_an_absolute_windows_path(tmp_path):
    """The end-to-end path check: R must be able to write where Python says."""
    target = tmp_path / "out.csv"
    rb.run_r(f'write.csv(data.frame(x = 1:3), "{rb.r_path(str(target))}", row.names = FALSE)')
    assert target.exists()
    assert target.read_text().strip().splitlines()[0].strip('"') == '"x"'.strip('"')


@pytest.mark.needs_r
def test_run_r_runs_from_the_repo_root():
    out = rb.run_r("cat(file.exists('pyproject.toml'))")
    assert out.stdout.strip().endswith("TRUE")
