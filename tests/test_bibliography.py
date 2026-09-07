"""Checks on the one bibliography and on every citation in the repository that points at it.

`docs/references.toml` is the only place a journal, a volume, a page range or a DOI is written
down. Everything else cites by number, and the numbers are assigned alphabetically by
`scripts/bibliography.py`, which means they move whenever an entry is added: a citation is only
as good as the guarantee that the number in the prose is the number the bibliography now gives.
That guarantee is what the second half of this file is.

The first half tests the tool on data it makes up, because the interesting failures are in the
reading of prose rather than in the reading of TOML. Author names are wrapped over line ends,
written with and without their diacritics, abbreviated to `et al.`, made possessive, and put
either in the sentence or in brackets around it, and every one of those has to resolve to the
same entry or the check protects nothing.
"""

import json
import os
import re

import pytest

import bibliography as bib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRIES = bib.load()
REFS = bib.index(ENTRIES)
PROSE = bib.prose_files(REPO)

# The works this repository cites without ever having recorded a full citation for them. Both
# are deliberate -- the alternative was to invent bibliographic detail -- and both are pinned
# here so that a third cannot appear without someone saying so.
INCOMPLETE = {"bao-chen-ma-2025"}


def toml(body):
    """Write a bibliography to a temporary file and read it back.

    Returns the loaded entries.

    PARAMETERS
    body : str, the contents of the TOML.
    """
    path = os.path.join(TMP[0], "references.toml")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    return bib.load(path)


TMP = [None]


@pytest.fixture(autouse=True, scope="module")
def _tmp(tmp_path_factory):
    TMP[0] = str(tmp_path_factory.mktemp("bib"))


TWO = """
[hall-hooker-2016]
kind = "article"
authors = ["Hall, P.", "Hooker, G."]
year = 2016
title = "Truncated linear models for functional data"

[aue-norinho-hormann-2015]
kind = "article"
authors = ["Aue, A.", "Norinho, D. D.", "Hörmann, S."]
year = 2015
title = "On the prediction of stationary functional time series"
"""


# ------------------------------------------------------------------------------- the source


def test_the_numbering_is_alphabetical_and_starts_at_one():
    assert [e["number"] for e in ENTRIES] == list(range(1, len(ENTRIES) + 1))
    leads = [bib.fold(bib.surnames(e)[0] if bib.surnames(e) else e["title"]).lower()
             for e in ENTRIES]
    assert leads == sorted(leads)


def test_the_order_does_not_depend_on_the_order_of_the_file():
    """The TOML is kept in key order for reading; the numbering must not care."""
    forwards = [e["key"] for e in toml(TWO)]
    backwards = [e["key"] for e in toml("\n".join(reversed(TWO.strip().split("\n\n"))))]
    assert forwards == backwards == ["aue-norinho-hormann-2015", "hall-hooker-2016"]


def test_inserting_an_entry_renumbers_the_ones_after_it():
    """The reason the numbers are not written by hand anywhere.

    Adding a work whose first author sorts first pushes every later number up by one, in the
    TOML and in each of the sixteen files that cite one of them.
    """
    before = {e["key"]: e["number"] for e in toml(TWO)}
    after = {e["key"]: e["number"] for e in toml(TWO + """
[andersen-2007]
kind = "article"
authors = ["Andersen, T. G."]
year = 2007
title = "Roughing it up"
""")}
    assert before == {"aue-norinho-hormann-2015": 1, "hall-hooker-2016": 2}
    assert after == {"andersen-2007": 1, "aue-norinho-hormann-2015": 2, "hall-hooker-2016": 3}


@pytest.mark.parametrize("body,why", [
    ('[Hall2016]\nkind = "article"\nyear = 2016\ntitle = "x"\n', "slug"),
    ('[a-2016]\nkind = "article"\nyear = 2016\ntitle = "x"\njurnal = "y"\n', "unknown field"),
    ('[a-2016]\nkind = "article"\ntitle = "x"\n', "missing required field"),
    ('[a-2016]\nkind = "paper"\nyear = 2016\ntitle = "x"\n', "kind"),
    ('[a-2016]\nkind = "article"\nyear = 2016\n', "no title and no note"),
])
def test_load_refuses_an_entry_it_cannot_trust(body, why):
    with pytest.raises(ValueError):
        toml(body)
    assert why


def test_every_entry_carries_what_its_kind_needs():
    """A kind is a promise about which container fields mean something.

    An entry with no title is exempt from the first two: it is one this repository never
    chased down, and a journal it does not have is the same gap as the title.
    """
    wrong = []
    for entry in ENTRIES:
        complete = "title" in entry
        if complete and entry["kind"] == "article" and "journal" not in entry:
            wrong.append(f"{entry['key']} is an article with no journal")
        if complete and entry["kind"] == "chapter" and "booktitle" not in entry:
            wrong.append(f"{entry['key']} is a chapter with no book")
        if "issue" in entry and "volume" not in entry:
            wrong.append(f"{entry['key']} has an issue but no volume")
        if "editors" in entry and "booktitle" not in entry:
            wrong.append(f"{entry['key']} has editors but nothing they edited")
    assert not wrong, "\n".join(wrong)


def test_no_two_entries_share_authors_and_a_year_without_a_letter():
    """`Shang (2017a)` and `Shang (2017b)` are only tellable apart because the letters exist.

    Without them a citation in a docstring, which has no number to fall back on, names two
    papers at once and `resolve` has to pick one.
    """
    seen = {}
    for entry in ENTRIES:
        signature = (tuple(bib.fold(n).lower() for n in bib.surnames(entry)),
                     entry["year"], entry.get("letter", ""))
        assert signature not in seen, (
            f"{entry['key']} and {seen[signature]} are cited identically; give them "
            f"`letter = \"a\"` and `letter = \"b\"`")
        seen[signature] = entry["key"]


def test_the_entries_without_a_title_are_the_ones_that_are_meant_to_be():
    missing = {e["key"] for e in ENTRIES if "title" not in e}
    assert missing == INCOMPLETE, (
        f"entries with no recorded title: {sorted(missing)}. An entry may be incomplete, but "
        f"only on purpose: add it to INCOMPLETE here and say why in its `note`.")


def test_every_doi_and_url_is_well_formed():
    wrong = [f"{e['key']}: {e['doi']}" for e in ENTRIES
             if "doi" in e and not re.fullmatch(r"10\.\d{4,9}/\S+", e["doi"])]
    wrong += [f"{e['key']}: {e['url']}" for e in ENTRIES
              if "url" in e and not e["url"].startswith("https://")]
    assert not wrong, "\n".join(wrong)


# ------------------------------------------------------------------------------ short forms


@pytest.mark.parametrize("authors,expected", [
    (["Zou, H."], "Zou"),
    (["Hall, P.", "Hooker, G."], "Hall & Hooker"),
    (["Aue, A.", "Norinho, D. D.", "Hörmann, S."], "Aue, Norinho & Hörmann"),
    (["Xu, D.", "Li, B.", "Singh, T.", "Chen, X."], "Xu et al."),
])
def test_the_short_form_names_up_to_three_authors(authors, expected):
    assert bib.short_authors({"authors": authors}) == expected


def test_a_work_cited_by_title_has_no_short_author_form():
    assert bib.short_authors({"year": 2026}) == ""
    assert bib.short_form({"year": 2026}) == "(2026)"


# -------------------------------------------------------------------------- reading a cite


@pytest.mark.parametrize("prose", [
    "Aue, Norinho & Hörmann (2015)",
    "Aue, Norinho & Hormann (2015)",          # the ASCII spelling the Python uses
    "Aue, Norinho and Hörmann (2015)",
    "Aue, Norinho & Hörmann's (2015)",
    "Aue, Norinho and\nHörmann (2015)",       # wrapped after the joining word
    "Aue, Norinho &\nHörmann (2015)",
    "the operator of Aue, Norinho & Hörmann (2015)",
    "(Aue, Norinho & Hörmann, 2015)",
    "(Aue, Norinho & Hörmann, 2015, Thm 3.1)",
])
def test_one_work_is_recognised_however_the_prose_writes_it(prose):
    """Every one of these appears somewhere in this repository, and all mean the same paper."""
    found = bib.plain_citations(prose)
    assert len(found) == 1, f"{prose!r} read as {len(found)} citations"
    names, year, letter, _at, _replace = found[0]
    assert bib.resolve(names, year, letter, ENTRIES)["key"] == "aue-norinho-hormann-2015"


def test_et_al_resolves_only_where_the_work_has_authors_to_abbreviate():
    """`Xu et al.` is five authors. Two authors abbreviated that way is a different work."""
    assert bib.resolve("Xu et al.", 2025, "", ENTRIES)["key"] == "xu-etal-2025"
    assert bib.resolve("Hall et al.", 2016, "", ENTRIES) is None


def test_a_year_without_its_letter_resolves_to_nothing():
    """`fdaSP/intraday.py` said `Shang (2017)`, which is two papers, and this is what found it."""
    assert bib.resolve("Shang", 2017, "", ENTRIES) is None
    assert bib.resolve("Shang", 2017, "a", ENTRIES)["key"] == "shang-2017a"
    assert bib.resolve("Shang", 2017, "b", ENTRIES)["key"] == "shang-2017b"


def test_a_misspelled_or_wrong_surname_resolves_to_nothing():
    assert bib.resolve("Hall & Hooke", 2016, "", ENTRIES) is None
    assert bib.resolve("Hooker", 2016, "", ENTRIES) is None
    assert bib.resolve("Hall & Hooker", 2015, "", ENTRIES) is None


def test_names_a_repository_uses_for_an_unattributed_work_are_declared():
    """`cited_as` is how a work with no recorded authors is still resolvable from prose."""
    entry = bib.resolve("Bai and Li", 2026, "", ENTRIES)
    assert entry["key"] == "overnight-not-simultaneous-2026"
    assert "authors" not in entry, "the attribution is unconfirmed and must stay unrecorded"


def test_a_lowercase_word_is_not_read_as_an_author():
    assert bib.plain_citations("the kernel from chapter 3 of Bosq (2000)")[0][0] == "Bosq"
    assert not bib.plain_citations("the window ends in 2016 (see above)")


# ------------------------------------------------------------------------------- rendering


def test_the_marker_a_file_gets_depends_on_what_renders_it():
    """A notebook cell cannot see a link definition in the cell before it, so it links inline."""
    md = bib.rewrite("docs/x.md", "cited [[0]][zou-2006]\n<!-- references: cited -->\n"
                     "<!-- /references -->\n<!-- reference-links -->\n"
                     "<!-- /reference-links -->\n", ENTRIES)
    assert f"[[{REFS['zou-2006']['number']}]][zou-2006]" in md
    assert "[zou-2006]: references.md#ref-zou-2006" in md


def test_a_marker_renders_as_a_bracketed_number_that_links_to_the_entry():
    """The contract the whole scheme rests on, put to an actual markdown parser.

    `[[26]][key]` is a link whose text is the literal `[26]`, which only works because no link
    definition is ever labelled with a bare number: were one, the inner brackets would be
    parsed as a link of their own and the outer one would not form. `mistune` arrives with
    Jupyter; where it is absent the check is skipped rather than approximated.
    """
    mistune = pytest.importorskip("mistune")
    render = mistune.create_markdown()
    target = "../../../docs/references.md#ref-hall-hooker-2016"

    reference = render(f"Hall & Hooker [[26]][hall-hooker-2016] did it.\n\n"
                       f"[hall-hooker-2016]: {target}\n")
    inline = render("Bosq [[13]](../docs/references.md#ref-bosq-2000) says so.")
    assert f'<a href="{target}">[26]</a>' in reference
    assert '<a href="../docs/references.md#ref-bosq-2000">[13]</a>' in inline


def test_no_link_definition_is_labelled_with_a_number():
    """A definition labelled `26` would stop every marker in that file from forming a link."""
    numeric = []
    for rel in PROSE:
        if not rel.endswith(".md"):
            continue
        numeric += [f"{rel}: [{m.group(1)}]" for m in bib.LINK_DEF.finditer(bib.read(rel, REPO))
                    if m.group(1).isdigit()]
    assert not numeric, "\n".join(numeric)


@pytest.mark.parametrize("rel,expected", [
    ("README.md", "docs/references.md"),
    ("docs/research-log.md", "references.md"),
    ("notebooks/truncFLR.ipynb", "../docs/references.md"),
    ("src/fts_research/far/README.md", "../../../docs/references.md"),
])
def test_the_path_to_the_bibliography_is_relative_to_the_file(rel, expected):
    assert bib.relative_bibliography(rel) == expected


def test_a_file_lists_only_the_number_the_authors_and_the_title():
    """The point of the exercise: the journal and the DOI are written once, not once a file."""
    block = bib.render_block(["hall-hooker-2016"], REFS)
    assert "Hall & Hooker (2016)" in block
    assert "Truncated linear models for functional data" in block
    for absent in ("Royal Statistical", "637", "10.1111", "78(3)"):
        assert absent not in block


def test_the_bibtex_is_the_same_bibliography():
    bibtex = bib.render_bibtex(ENTRIES)
    for entry in ENTRIES:
        assert f"{{{entry['key']}," in bibtex, f"{entry['key']} is missing from the BibTeX"
    assert bibtex.count("@") == len(ENTRIES)


# ------------------------------------------------------------------------------- rewriting


def test_rewriting_a_file_that_is_in_step_changes_nothing():
    for rel in PROSE:
        text = bib.read(rel, REPO)
        assert bib.rewrite(rel, text, ENTRIES) == text, f"{rel} is not in step"


@pytest.mark.parametrize("rel", ["README.md", "src/fts_research/far/README.md",
                                 "docs/overnight-jumps-literature.md"])
def test_a_markdown_file_recovers_from_every_number_being_wrong(rel):
    """A renumbering is a rewrite of every citing file, so it has to land exactly.

    Every marker the checker can see is given a number no entry has, and the file must come
    back byte for byte: the numbers restored, and the prose around them untouched.
    """
    text = bib.read(rel, REPO)
    broken = text
    for _number, key, (start, end) in reversed(bib.markers(rel, text)):
        broken = broken[:start] + f"[[999]][{key}]" + broken[end:]
    assert broken != text
    assert bib.rewrite(rel, broken, ENTRIES) == text


def test_a_notebook_recovers_without_being_reformatted():
    """A notebook is rewritten as raw text, never through a JSON round trip.

    Re-serialising would restyle the whole file -- indentation, escapes, key order -- and put
    a diff of thousands of lines under a change of one number.
    """
    rel = "notebooks/shang_replication.ipynb"
    text = bib.read(rel, REPO)
    broken = bib.INLINE_MARKER.sub(lambda m: f"[[999]]({m.group(2)}#ref-{m.group(3)})", text)
    assert broken != text
    fixed = bib.rewrite(rel, broken, ENTRIES)
    assert fixed == text
    assert json.loads(fixed) == json.loads(text)


def test_a_marker_written_with_any_number_at_all_is_corrected():
    """`[[0]][key]` is how a citation is written by hand; the tool supplies the number."""
    text = ("Hall & Hooker [[0]][hall-hooker-2016].\n\n<!-- references: cited -->\n"
            "<!-- /references -->\n\n<!-- reference-links -->\n<!-- /reference-links -->\n")
    fixed = bib.rewrite("README.md", text, ENTRIES)
    assert f"[[{REFS['hall-hooker-2016']['number']}]][hall-hooker-2016]" in fixed
    assert bib.rewrite("README.md", fixed, ENTRIES) == fixed


def test_a_citation_inside_a_fenced_block_is_left_as_prose():
    """A link does not render inside a fence, so the layout listing in the root README keeps
    the author-year form and is checked rather than rewritten."""
    text = "```\nfoo.ipynb   Shang (2017a) rebuilt\n```\n"
    assert bib.rendered_prose("README.md", text).strip() == ""
    assert bib.plain_citations(bib.code_prose("README.md", text))


# ---------------------------------------------------------------------------- the whole repo


def test_every_citation_in_the_repository_is_in_step():
    """The check the whole arrangement rests on, run as a test.

    It fails when a number is stale, a key does not exist, a markdown citation has no link, a
    file cites a work its reference block does not list, an author-year citation in a
    docstring names nothing in the bibliography, or either generated file is out of date.
    """
    problems = bib.check(REPO, ENTRIES)
    assert not problems, "\n".join(problems)


def test_every_entry_is_cited_by_something():
    cited = set()
    for rel in PROSE:
        cited |= set(bib.linked_keys(rel, bib.read(rel, REPO)))
    assert not sorted(set(REFS) - cited), "entries nothing cites"


def test_only_the_bibliography_prints_a_journal_or_a_doi():
    """The invariant the whole change exists to create.

    Nine files used to carry Hall and Hooker's page range, two of them punctuated differently.
    A DOI outside `docs/references.md` now means a second copy has started.
    """
    guilty = []
    for rel in PROSE:
        if rel.startswith("scripts/"):
            continue  # the module that renders them
        text = bib.read(rel, REPO)
        for pattern in (r"doi:10\.", r"doi\.org/10\."):
            if re.search(pattern, text):
                guilty.append(f"{rel} prints a DOI; put it in {bib.SOURCE}")
    assert not guilty, "\n".join(guilty)


def test_every_package_readme_lists_the_works_its_package_rests_on():
    src = os.path.join(REPO, "src", "fts_research")
    empty = []
    for package in sorted(os.listdir(src)):
        rel = f"src/fts_research/{package}/README.md"
        if not os.path.exists(os.path.join(REPO, rel)):
            continue
        if not bib.declared_keys(rel, bib.read(rel, REPO)):
            empty.append(rel)
    assert not empty, f"package READMEs with no reference block: {empty}"


def test_no_citation_marker_is_hidden_from_the_checker():
    """A marker the checker cannot see is a number nothing keeps up to date.

    The root README explains the syntax by quoting a marker inside backticks, and that one is
    meant to be invisible. Anything else invisible would be a hole in the guarantee, so the
    two counts are required to add up rather than the quoted one being waved through.
    """
    for rel in PROSE:
        if rel.endswith(".py"):
            continue
        text = bib.read(rel, REPO)
        code = bib.code_prose(rel, text)
        raw = len(bib.REF_MARKER.findall(text)) + len(bib.INLINE_MARKER.findall(text))
        quoted = len(bib.REF_MARKER.findall(code)) + len(bib.INLINE_MARKER.findall(code))
        assert raw == len(bib.markers(rel, text)) + quoted, (
            f"{rel} has {raw} markers, of which {len(bib.markers(rel, text))} are checked and "
            f"{quoted} are quoted as examples")


def test_no_marker_is_left_dangling():
    """`[[26]]` with nothing after it renders as literal brackets and reads as a citation."""
    broken = []
    for rel in PROSE:
        text = bib.read(rel, REPO)
        for match in bib.BROKEN_MARKER.finditer(bib.rendered_prose(rel, text)):
            broken.append(f"{rel}:{text.count(chr(10), 0, match.start()) + 1}")
    assert not broken, f"markers with no link: {broken}"
