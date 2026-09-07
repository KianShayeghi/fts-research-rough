"""The repository's one bibliography, and the tool that keeps every citation pointing at it.

Every work cited anywhere here is described exactly once, in `docs/references.toml`. Nothing
else holds a journal name, a volume, a page range or a DOI. Prose cites by number, the number
is a link into `docs/references.md`, and both that file and `docs/references.bib` are written
from the TOML by this module. The numbering is alphabetical by first author, so adding a work
renumbers the ones after it; that is only safe because the renumbering is done here rather
than by hand, which is the whole reason this file exists.

The duplication it replaces was already drifting. `core/README.md` printed Hall and Hooker's
page range with en dashes and `bench/README.md` with hyphens; `fdaSP/README.md` called
Parkinson's journal *The Journal of Business* and `docs/overnight-jumps-literature.md` called
it *Journal of Business*; and four module docstrings under `src/` carried their own reference
lists, so a corrected DOI had to be found in nine places to be corrected once.

A citation takes one of three forms, and which one depends on what the reader is looking at.

- Markdown prose uses a reference-style link, `[[26]][hall-hooker-2016]`, whose definition is
  collected into a managed block at the end of the file. The visible number is generated; the
  label is the citation key, so the source stays readable, stays inside the line limit the
  prose is wrapped to, and survives renumbering untouched.
- A notebook uses the inline form, `[[26]](../docs/references.md#ref-hall-hooker-2016)`,
  because each markdown cell is rendered as a document of its own and a link definition in
  one cell does not reach the next.
- A docstring, a comment, a fenced code block and `CITATION.cff` cite in the author-year form,
  unlinked, because none of them is rendered as markdown and a bracketed number no reader can
  follow is worse than a name they can search for. These are still held to the bibliography:
  `check` resolves every author-year citation it finds to an entry, so a year missing the
  letter that tells two of an author's papers apart fails, and so does a misspelled surname.

Run `python scripts/bibliography.py --check` to report every disagreement between the prose
and the TOML, and `--write` to fix the ones that can be fixed mechanically: the generated
files, the numbers, the link definitions and the managed blocks. A marker may be written with
any number at all -- `[[0]][hall-hooker-2016]` is the convention -- and `--write` will set it.
`tests/test_bibliography.py` runs `--check` as a test, so the prose cannot drift from the
bibliography without the suite going red.
"""

import argparse
import json
import os
import re
import sys
import textwrap
import tomllib
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE = "docs/references.toml"
MARKDOWN = "docs/references.md"
BIBTEX = "docs/references.bib"

WIDTH = 99  # matches .flake8 and the wrapping of the prose this writes into

# What an entry may say. `kind` picks the BibTeX entry type and decides which of the container
# fields are meaningful; everything else is optional because the sources genuinely differ --
# a blog post has no volume and a book has no journal.
FIELDS = {
    "kind", "authors", "year", "letter", "title", "journal", "booktitle", "editors",
    "publisher", "series", "chapter", "volume", "issue", "pages", "doi", "url",
    "url_label", "note", "cited_as",
}
REQUIRED = ("kind", "year")
KINDS = {
    "article": "article",
    "chapter": "incollection",
    "book": "book",
    "software": "misc",
    "working-paper": "unpublished",
    "preprint": "misc",
    "blog": "misc",
}
# Kinds whose title is the work itself rather than a piece inside a journal or a volume, and
# so is set in italics the way a monograph is.
ITALIC_TITLE = {"book", "working-paper", "blog", "preprint"}

KEY = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# `[[26]][hall-hooker-2016]` and its definition, the markdown form.
REF_MARKER = re.compile(r"\[\[(\d+)\]\]\[([a-z0-9-]+)\]")
LINK_DEF = re.compile(r"(?m)^\[([a-z0-9-]+)\]: *(\S+) *$")
# `[[26]](../docs/references.md#ref-hall-hooker-2016)`, the notebook form.
INLINE_MARKER = re.compile(r"\[\[(\d+)\]\]\((\S*?)#ref-([a-z0-9-]+)\)")
# A marker half written, which renders as literal brackets and is easy not to notice.
BROKEN_MARKER = re.compile(r"\[\[\d+\]\](?!\[[a-z0-9-]+\]|\(\S*?#ref-[a-z0-9-]+\))")

# An author-year citation in running prose: a run of capitalised names, optionally possessive,
# followed by the year in brackets. `Shang`, `Hall & Hooker`, `Aue, Norinho & Hormann's` and
# `Xu et al.` all match; `chapter 3 of Bosq` stops at the lowercase `of`. A single line break
# inside the run is allowed, because the prose here is wrapped and a three-author citation is
# routinely broken across two lines; a blank line is not, so a run cannot cross a paragraph.
UPPER = r"A-ZÀ-ÖØ-Þ"
NAME = rf"[{UPPER}][\w'’-]*\.?"
GAP = r"(?:[ \t]+|[ \t]*\n[ \t]*)"
# `, `, ` and `, ` & ` -- and where the prose is wrapped the line break falls on either side of
# the joining word, which is why the gap is written on both.
JOIN = rf",?{GAP}(?:(?:and|&){GAP})?"
CITATION = re.compile(
    rf"(?P<names>{NAME}(?:{JOIN}(?:et al\.|{NAME}))*)"
    rf"{GAP}\((?P<year>(?:19|20)\d{{2}})(?P<letter>[ab]?)\)"
)
# The same citation written as a parenthetical aside: `(Hall & Hooker, 2016)`, and the form
# `(Bosq, 2000, Thm 3.1)` that carries a pointer into the work with it.
PARENTHETICAL = re.compile(
    rf"\((?P<names>{NAME}(?:{JOIN}(?:et al\.|{NAME}))*),"
    rf"{GAP}(?P<year>(?:19|20)\d{{2}})(?P<letter>[ab]?)(?=[,)])"
)

# A backticked span is code, not prose: the README documents the marker syntax by quoting it,
# and a quoted marker is an example rather than a citation of whatever key it names.
CODE_SPAN = re.compile(r"(?P<ticks>`+)(?:(?!(?P=ticks)).)+(?P=ticks)")

BLOCK = re.compile(
    r"(?ms)^<!-- references:(?P<spec>[^>]*?)-->\n(?P<body>.*?)^<!-- /references -->$"
)
LINKS_BLOCK = re.compile(
    r"(?ms)^<!-- reference-links -->\n(?P<body>.*?)^<!-- /reference-links -->$"
)

SKIP_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", ".ipynb_checkpoints", "build",
             "dist", ".github"}
GENERATED = (MARKDOWN, BIBTEX)

# Files nothing renders as markdown, which therefore cite by author and year and are checked
# rather than rewritten: the Python, and the citation metadata whose abstract names papers.
UNRENDERED = (".py", ".cff")

# The two files whose subject is citations, including ones that are meant not to resolve: this
# module, which quotes the forms it reads, and its tests, which quote the forms it must reject.
# Their author-year citations are examples rather than citations and are not checked. Every
# other file in the repository is.
SELF_REFERENTIAL = ("scripts/bibliography.py", "tests/test_bibliography.py")


# --------------------------------------------------------------------------- the bibliography


def fold(text):
    """`text` with its diacritics dropped, so `Hörmann` and `Hormann` are the same name.

    The markdown spells the surnames as their authors do and the Python spells them in ASCII,
    which is a house convention rather than a mistake; both have to resolve to one entry.

    Returns a str.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def surnames(entry):
    """The surnames of an entry's authors, in order, as written in the TOML.

    Returns a list of str, empty where the work is cited by title.
    """
    return [a.split(",")[0].strip() for a in entry.get("authors", ())]


def short_authors(entry):
    """`Hall & Hooker`, `Aue, Norinho & Hörmann`, `Berkman et al.` -- the form used in prose.

    Returns a str, empty where the work is cited by title.
    """
    names = surnames(entry)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) > 3:
        return f"{names[0]} et al."
    return f"{', '.join(names[:-1])} & {names[-1]}"


def year_label(entry):
    """`2016`, or `2017a` where two works by the same authors share a year.

    Returns a str.
    """
    return f"{entry['year']}{entry.get('letter', '')}"


def short_form(entry):
    """`Hall & Hooker (2016)`, the way an entry is named in a file's own reference list.

    Returns a str.
    """
    authors = short_authors(entry)
    return f"{authors} ({year_label(entry)})" if authors else f"({year_label(entry)})"


def _sort_key(key, entry):
    """Alphabetical by first author, then year, then letter, then title, then key.

    A work cited by title sorts under its title. The key is the last component only so that
    the order is total whatever the data, never so that it decides anything visible.
    """
    names = surnames(entry)
    title = entry.get("title", "")
    lead = fold(names[0] if names else title).lower()
    rest = " ".join(fold(n).lower() for n in names[1:])
    return (lead, entry["year"], entry.get("letter", ""), rest, title.lower(), key)


def load(path=None):
    """Every entry in `docs/references.toml`, in the order the numbering uses.

    An entry is its table from the file plus `key`, the table's name, and `number`, its
    position in the sorted order counting from one. Sorting here rather than trusting the
    order of the file means a new entry lands in one deterministic place and `--write` moves
    every number after it, wherever in the repository those numbers are written.

    Returns a list of dict.

    PARAMETERS
    path : str, the TOML to read. Defaults to `docs/references.toml` in this repository.
    """
    path = path or os.path.join(REPO, SOURCE)
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    entries = []
    for key, entry in raw.items():
        where = f"{os.path.basename(path)} [{key}]"
        if not KEY.fullmatch(key):
            raise ValueError(f"{where}: key is not a lowercase hyphenated slug")
        unknown = sorted(set(entry) - FIELDS)
        if unknown:
            raise ValueError(f"{where}: unknown field(s) {unknown}")
        missing = [f for f in REQUIRED if f not in entry]
        if missing:
            raise ValueError(f"{where}: missing required field(s) {missing}")
        if entry["kind"] not in KINDS:
            raise ValueError(f"{where}: kind {entry['kind']!r} is not one of {sorted(KINDS)}")
        if "title" not in entry and "note" not in entry:
            # The one reason an entry may have no title is that this repository never
            # recorded one, and then the note has to say so rather than the gap being silent.
            raise ValueError(f"{where}: has no title, so it must carry a note saying why")
        entries.append(dict(entry, key=key))

    entries.sort(key=lambda e: _sort_key(e["key"], e))
    for number, entry in enumerate(entries, start=1):
        entry["number"] = number
    return entries


def index(entries):
    """The entries by key.

    Returns a dict of str to dict.
    """
    return {e["key"]: e for e in entries}


# ------------------------------------------------------------------------------- rendering


def _wrap(text, indent=""):
    """`text` as lines no longer than the file's line limit, never breaking inside a URL."""
    return textwrap.fill(text, width=WIDTH, initial_indent="", subsequent_indent=indent,
                         break_long_words=False, break_on_hyphens=False)


def _authors_prose(entry):
    """`Hall, P. and Hooker, G.` -- the full author list as a bibliography prints it."""
    authors = list(entry.get("authors", ()))
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    return f"{', '.join(authors[:-1])} and {authors[-1]}"


def _link_label(entry):
    """The words a bare URL is shown as: `arxiv.org`, `SSRN 1004081` where the TOML says so."""
    if "url_label" in entry:
        return entry["url_label"]
    host = entry["url"].split("//", 1)[-1].split("/", 1)[0]
    return host[4:] if host.startswith("www.") else host


def _locus(entry):
    """`78(3), 637-653` -- where in a volume the work sits."""
    bits = []
    if "volume" in entry:
        bits.append(f"{entry['volume']}({entry['issue']})" if "issue" in entry
                    else str(entry["volume"]))
    if "pages" in entry:
        bits.append(str(entry["pages"]))
    return ", ".join(bits)


def _container(entry):
    """What the work appeared in, as markdown, ending without a full stop.

    A journal takes its volume and pages; a chapter takes its editors, its volume title and
    its publisher; a book, a working paper and a blog post have only the last of those.

    Returns a str, empty where the entry names no container at all.
    """
    parts = []
    if "journal" in entry:
        parts.append(f"*{entry['journal']}* {_locus(entry)}".strip())
    elif "booktitle" in entry:
        editors = _authors_prose({"authors": entry["editors"]}) if "editors" in entry else ""
        parts.append(f"In {editors} (eds.), *{entry['booktitle']}*" if editors
                     else f"In *{entry['booktitle']}*")
        if "series" in entry:
            parts.append(entry["series"])
        if "chapter" in entry:
            parts.append(f"chapter {entry['chapter']}")
        if "pages" in entry:
            parts.append(f"pp. {entry['pages']}")
    else:
        if "series" in entry:
            parts.append(entry["series"])
        if _locus(entry):
            parts.append(_locus(entry))
    text = ", ".join(parts)
    if "publisher" in entry:
        return f"{text}. {entry['publisher']}" if text else entry["publisher"]
    return text


def title_form(entry):
    """The title as prose sets it: italic for a work that stands on its own, plain otherwise.

    Returns a str, empty where this repository never recorded a title.
    """
    title = entry.get("title")
    if not title:
        return ""
    return f"*{title}*" if entry["kind"] in ITALIC_TITLE else title


def full_entry(entry, cited_in=()):
    """One entry of `docs/references.md`: anchor, number, the work, and who cites it.

    The anchor is the citation key rather than the number, so a link written anywhere in the
    repository keeps working when the numbering moves under it.

    Returns a str of wrapped markdown, without a trailing newline.

    PARAMETERS
    cited_in : sequence of str, repository-relative paths that cite this entry.
    """
    head = f'<a id="ref-{entry["key"]}"></a>**[{entry["number"]}]** '
    authors, title = _authors_prose(entry), title_form(entry)
    if authors:
        # The usual order. A work cited by title leads with the title instead, because
        # opening on a bare year reads as a missing word rather than as a deliberate gap.
        parts = [f"{authors} ({year_label(entry)})."] + ([f"{title}."] if title else [])
    else:
        parts = [f"{title} ({year_label(entry)})."]
    container = _container(entry)
    if container:
        parts.append(container + ".")
    if "doi" in entry:
        parts.append(f"[doi:{entry['doi']}](https://doi.org/{entry['doi']}).")
    elif "url" in entry:
        parts.append(f"[{_link_label(entry)}]({entry['url']}).")
    if "note" in entry:
        parts.append(entry["note"])

    lines = [_wrap(head + " ".join(parts))]
    if cited_in:
        named = ", ".join(f"`{p}`" for p in cited_in)
        lines.append(_wrap(f"Cited in {named}."))
    return "\n\n".join(lines)


HEADER = """<!-- Generated by scripts/bibliography.py from docs/references.toml. Do not edit. -->

# References

Every work this repository cites, numbered. The numbering is alphabetical by first author and
is regenerated whenever an entry is added, so cite by the link and never by the bare number:
markdown prose writes `[[26]][hall-hooker-2016]` and lets `scripts/bibliography.py --write`
set the number, notebooks write the same marker with an inline link, and docstrings and code
comments cite in the author-year form because nothing renders markdown for them.

`docs/references.bib` is the same list as BibTeX. `python scripts/bibliography.py --check`
fails if any citation in the repository has drifted from what is written here.

"""


def render_markdown(entries, citations):
    """`docs/references.md` in full.

    Returns a str.

    PARAMETERS
    citations : dict of key to sorted list of repository-relative paths that cite the key.
    """
    body = "\n\n".join(full_entry(e, citations.get(e["key"], ())) for e in entries)
    return HEADER + body + "\n"


def _bibtex_value(text):
    """A field value, braced against BibTeX's own case folding of titles."""
    return "{" + str(text).replace("&", r"\&") + "}"


def render_bibtex(entries):
    """`docs/references.bib`, the same bibliography for the thesis that will want it.

    Returns a str.
    """
    out = ["% Generated by scripts/bibliography.py from docs/references.toml. Do not edit.",
           "% One entry per work cited in this repository; the citation key is the key used",
           "% by the markdown links in docs/references.md.", ""]
    order = [("author", lambda e: " and ".join(e["authors"]) if e.get("authors") else None),
             ("title", lambda e: e.get("title")),
             ("journal", lambda e: e.get("journal")),
             ("booktitle", lambda e: e.get("booktitle")),
             ("editor", lambda e: " and ".join(e["editors"]) if e.get("editors") else None),
             ("series", lambda e: e.get("series")),
             ("chapter", lambda e: e.get("chapter")),
             ("volume", lambda e: e.get("volume")),
             ("number", lambda e: e.get("issue")),
             ("pages", lambda e: e.get("pages")),
             ("publisher", lambda e: e.get("publisher")),
             ("year", lambda e: e["year"]),
             ("doi", lambda e: e.get("doi")),
             ("url", lambda e: e.get("url")),
             ("note", lambda e: e.get("note"))]
    for entry in entries:
        out.append(f"@{KINDS[entry['kind']]}{{{entry['key']},")
        for name, get in order:
            value = get(entry)
            if value is not None:
                out.append(f"  {name:<9} = {_bibtex_value(value)},")
        out[-1] = out[-1].rstrip(",")
        out.append("}")
        out.append("")
    return "\n".join(out)


def render_block(keys, refs):
    """The list of works a file cites, for the managed block that file carries.

    Only the number, the short author-year form and the title are printed. Everything else
    about a work -- the journal, the volume, the pages, the DOI -- is in `docs/references.md`
    and nowhere else, which is the point of the exercise.

    Returns a str of markdown lines, without a trailing newline.

    PARAMETERS
    keys : sequence of str, the citation keys to list, in any order.
    refs : dict of key to entry.
    """
    lines = []
    for key in sorted(keys, key=lambda k: refs[k]["number"]):
        entry = refs[key]
        head, title = f"- **[{entry['number']}]**", title_form(entry)
        if title and short_authors(entry):
            line = f"{head} {short_form(entry)}, [{title}][{key}]."
        elif title:
            line = f"{head} [{title}][{key}] ({year_label(entry)})."
        else:
            line = f"{head} [{short_form(entry)}][{key}]."
        lines.append(_wrap(line, indent="  "))
    return "\n".join(lines)


def render_link_definitions(keys, rel, refs):
    """The link definitions a markdown file needs for the markers it carries.

    Returns a str of markdown lines, without a trailing newline.

    PARAMETERS
    rel : str, the repository-relative path of the file, which fixes the relative path.
    """
    target = relative_bibliography(rel)
    return "\n".join(f"[{key}]: {target}#ref-{key}"
                     for key in sorted(keys, key=lambda k: refs[k]["number"]))


def relative_bibliography(rel):
    """`docs/references.md` as seen from `rel`, with forward slashes.

    Returns a str.
    """
    here = os.path.dirname(rel) or "."
    return os.path.relpath(MARKDOWN, here).replace(os.sep, "/")


# ------------------------------------------------------------------------------- the prose


def prose_files(repo=REPO):
    """Every file in the repository that can carry a citation, repository-relative and sorted.

    Markdown and notebooks cite with links, Python with names; the generated bibliography
    itself is left out, because it is the thing the others are checked against.

    Returns a list of str.
    """
    out = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            rel = os.path.relpath(os.path.join(root, name), repo).replace(os.sep, "/")
            if rel in GENERATED:
                continue
            if name.endswith((".md", ".ipynb") + UNRENDERED):
                out.append(rel)
    return sorted(out)


def read(rel, repo=REPO):
    """The text of a repository file.

    Returns a str.
    """
    with open(os.path.join(repo, rel), encoding="utf-8") as fh:
        return fh.read()


def write(rel, text, repo=REPO):
    """Write `text` to a repository file with the newlines the checkout uses."""
    with open(os.path.join(repo, rel), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _fenced_spans(text):
    """(start, end) of every fenced code block, which renders as code and not as markdown."""
    spans, fence, start, pos = [], None, 0, 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if fence is None:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence, start = stripped[:3], pos
        elif stripped.startswith(fence):
            spans.append((start, pos + len(line)))
            fence = None
        pos += len(line)
    if fence is not None:
        spans.append((start, pos))
    return spans


def _blank(text, spans):
    """`text` with `spans` replaced by spaces, so offsets survive but nothing matches there."""
    out = list(text)
    for start, end in spans:
        for i in range(start, end):
            if out[i] != "\n":
                out[i] = " "
    return "".join(out)


def _managed_spans(text):
    """(start, end) of every managed block, whose contents this module wrote."""
    return ([m.span() for m in BLOCK.finditer(text)]
            + [m.span() for m in LINKS_BLOCK.finditer(text)])


def rendered_prose(rel, text):
    """The part of a file that is rendered as markdown, with everything else blanked out.

    Fenced code, backticked spans, managed blocks and -- for a notebook -- every code cell are
    removed, keeping the offsets of what is left so a match here can be applied to the file
    itself. A notebook is handled through its JSON, because its fences are inside string
    literals where a line-by-line reading of the raw file cannot see them.

    Returns a str the same length as `text`.

    PARAMETERS
    rel : str, the repository-relative path, which decides how the file is read.
    """
    if rel.endswith(UNRENDERED):
        return " " * len(text)
    if rel.endswith(".ipynb"):
        blanked = _notebook_prose(text)
    else:
        blanked = _blank(text, _fenced_spans(text) + _managed_spans(text))
    return _blank(blanked, [m.span() for m in CODE_SPAN.finditer(blanked)])


def _notebook_prose(text):
    """A notebook's markdown cells, blanked out of the raw JSON everywhere else.

    The spans are found by searching the raw text for each source line of each markdown cell,
    which keeps the offsets of the file itself; that matters because every rewrite this
    module makes to a notebook is a substitution on the raw text, never a JSON round trip,
    so no notebook is reformatted by being cited.
    """
    try:
        cells = json.loads(text)["cells"]
    except (ValueError, KeyError, TypeError):
        return " " * len(text)
    keep, pos = [], 0
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        for line in cell.get("source", []):
            encoded = json.dumps(line)[1:-1]
            found = text.find(encoded, pos)
            if found < 0:
                continue
            keep.append((found, found + len(encoded)))
            pos = found + len(encoded)
    blanked = [" " if ch != "\n" else "\n" for ch in text]
    for start, end in keep:
        chunk = list(text[start:end])
        # The newline a source line ends with is the two characters of a JSON escape, and the
        # quote, comma and indent joining one line to the next are blanked with everything
        # else. Blanking the escape too makes a cell read as the prose it is, so a citation
        # whose author names are wrapped over two source lines is still one citation here.
        for i in range(len(chunk) - 1):
            if chunk[i] == "\\" and chunk[i + 1] == "n":
                chunk[i] = chunk[i + 1] = " "
        blanked[start:end] = chunk
    return "".join(blanked)


def code_prose(rel, text):
    """The part of a file that is not rendered: docstrings, comments and fenced code.

    These cite by name rather than by number, and are checked rather than rewritten.

    Returns a str.
    """
    if rel.endswith(UNRENDERED):
        return text
    if rel.endswith(".ipynb"):
        return ""
    blanked = _blank(text, _fenced_spans(text) + _managed_spans(text))
    spans = _fenced_spans(text) + [m.span() for m in CODE_SPAN.finditer(blanked)]
    return "".join(text[start:end] for start, end in sorted(spans))


def markers(rel, text):
    """Every citation marker in the rendered part of a file.

    Returns a list of (number, key, span) with span a (start, end) pair into `text`.
    """
    prose = rendered_prose(rel, text)
    found = [(int(m.group(1)), m.group(2), m.span()) for m in REF_MARKER.finditer(prose)]
    found += [(int(m.group(1)), m.group(3), m.span()) for m in INLINE_MARKER.finditer(prose)]
    return sorted(found, key=lambda f: f[2])


def cited_keys(rel, text):
    """The citation keys a file uses, in the order they first appear.

    Returns a list of str.
    """
    seen = []
    for _number, key, _span in markers(rel, text):
        if key not in seen:
            seen.append(key)
    return seen


def declared_keys(rel, text):
    """The keys a file's managed blocks list.

    A block declares either `cited`, meaning whatever the file's own prose cites, or an
    explicit list. The explicit form is what most files here use, because a package README's
    reference list is a reading list: it names the papers the package is an implementation of
    whether or not the README happens to mention each of them in a sentence.

    Returns a list of str, in the order the blocks name them.
    """
    if not rel.endswith(".md"):
        return []
    out = []
    for match in BLOCK.finditer(text):
        keys = _spec_keys(match.group("spec"), rel, text)
        out += [k for k in keys if k not in out]
    return out


def _spec_keys(spec, rel, text):
    """The keys a block declaration names, `cited` resolved against the file's own markers."""
    if spec.split() == ["cited"]:
        return cited_keys(rel, text)
    return [k.strip() for k in spec.split(",") if k.strip()]


def _spec_comment(spec, keys):
    """The declaration comment, wrapped so a long list of keys stays inside the line limit."""
    if spec.split() == ["cited"]:
        return "<!-- references: cited -->"
    one_line = f"<!-- references: {', '.join(keys)} -->"
    if len(one_line) <= WIDTH:
        return one_line
    body = textwrap.fill(", ".join(keys), width=WIDTH, initial_indent="  ",
                         subsequent_indent="  ", break_long_words=False,
                         break_on_hyphens=False)
    return f"<!-- references:\n{body}\n-->"


def linked_keys(rel, text):
    """Every key a file needs a link definition for: cited in its prose, or listed in a block.

    Returns a list of str.
    """
    keys = list(cited_keys(rel, text))
    return keys + [k for k in declared_keys(rel, text) if k not in keys]


def plain_citations(text):
    """Every author-year citation in `text` that is not already a marker.

    Two forms are recognised, and this repository uses both: `Hall & Hooker (2016)` where the
    authors are part of the sentence, and `(Hall & Hooker, 2016)` where the whole citation is
    an aside. The second also appears as `(Bosq, 2000, Thm 3.1)`, carrying a pointer into the
    work, and the trailing part is left alone.

    Returns a list of (names, year, letter, at, replace), where `at` is where the citation
    starts and `replace` is the (start, end) of the part a marker stands in for: the year in
    brackets in the first form, the comma and the year in the second.
    """
    found = []
    for match in CITATION.finditer(text):
        opened = text.rindex("(", match.start(), match.end())
        found.append((match.group("names"), int(match.group("year")), match.group("letter"),
                      match.start(), (opened, match.end())))
    for match in PARENTHETICAL.finditer(text):
        names_end = match.start("names") + len(match.group("names"))
        found.append((match.group("names"), int(match.group("year")), match.group("letter"),
                      match.start(), (names_end, match.end())))
    return sorted(found, key=lambda citation: citation[3])


def resolve(names, year, letter, entries):
    """The entry an author-year citation names, or None where nothing matches it.

    A citation matches when the entry's surnames are the end of the ones written -- the end,
    not the whole, because a sentence opening `Follows Hall & Hooker (2016)` puts a word in
    front of the names that is not one of them. `First et al.` matches an entry of four or
    more authors beginning with that one. `cited_as` in the TOML matches outright, which is
    how a work whose authorship this repository has not confirmed can still be resolved from
    the names its prose happens to use.

    Returns a dict or None.

    PARAMETERS
    names  : str, the names as prose writes them, e.g. `Aue, Norinho & Hörmann's`.
    letter : str, the disambiguating letter, `a` or `b` or empty.
    """
    written = _prose_surnames(names)
    for entry in entries:
        if entry["year"] != year or entry.get("letter", "") != letter:
            continue
        if names.strip() in entry.get("cited_as", ()):
            return entry
        theirs = [fold(n).lower() for n in surnames(entry)]
        if not theirs:
            continue
        if len(written) >= len(theirs) and written[-len(theirs):] == theirs:
            return entry
        if len(theirs) > 3 and written[-2:] == [theirs[0], "et al."]:
            return entry
    return None


def _prose_surnames(names):
    """`Aue, Norinho & Hörmann's` as `["aue", "norinho", "hormann"]`.

    Initials are dropped, because the prose form does not use them and the bibliography form
    is not read by this; a trailing possessive is dropped, because `Samworth's Theorem 1` is
    a citation of Samworth; and `et al.` is kept as a name of its own so that an abbreviated
    citation can be told from a two-author one.

    Returns a list of str.
    """
    out = []
    for part in re.split(r"[,\s&]+", re.sub(r"\s+et al\.", " et-al", names)):
        part = re.sub(r"['’]s$", "", part.strip())
        if not part or part == "and" or re.fullmatch(r"[A-Z]\.?", part):
            continue
        out.append("et al." if part == "et-al" else fold(part).lower())
    return out


# -------------------------------------------------------------------------------- rewriting


def rewrite(rel, text, entries):
    """`text` with every number, link and managed block brought into line with the TOML.

    Nothing that is not generated is touched: the numbers inside markers, the link
    definitions, and the bodies of the managed blocks, and nothing else.

    Returns a str.

    PARAMETERS
    entries : list of dict, as `load` returns them.
    """
    refs = index(entries)
    cited = [k for k in cited_keys(rel, text) if k in refs]
    text = _rewrite_markers(rel, text, refs)
    if rel.endswith(".ipynb"):
        return text
    text = _rewrite_blocks(rel, text, refs, cited)
    linked = [k for k in linked_keys(rel, text) if k in refs]
    return _rewrite_link_definitions(rel, text, refs, linked)


def _rewrite_markers(rel, text, refs):
    """Set the visible number of every marker, and the path of every inline one."""
    target = relative_bibliography(rel)
    for _number, key, (start, end) in reversed(markers(rel, text)):
        if key not in refs:
            continue
        source = text[start:end]
        number = refs[key]["number"]
        fixed = (f"[[{number}]][{key}]" if source.endswith("]")
                 else f"[[{number}]]({target}#ref-{key})")
        text = text[:start] + fixed + text[end:]
    return text


def _rewrite_blocks(rel, text, refs, keys):
    """Fill each managed block with the entries its declaration names."""
    def replace(match):
        spec = match.group("spec")
        declared = _spec_keys(spec, rel, text)
        wanted = [k for k in (keys if spec.split() == ["cited"] else declared) if k in refs]
        head = _spec_comment(spec, declared)
        body = render_block(wanted, refs)
        return f"{head}\n{body}\n<!-- /references -->" if body else f"{head}\n<!-- /references -->"
    return BLOCK.sub(replace, text)


def _rewrite_link_definitions(rel, text, refs, keys):
    """Fill the file's one definitions block: a definition per key it cites or lists."""
    def replace(match):
        body = render_link_definitions(keys, rel, refs)
        return (f"<!-- reference-links -->\n{body}\n<!-- /reference-links -->" if body
                else "<!-- reference-links -->\n<!-- /reference-links -->")
    return LINKS_BLOCK.sub(replace, text)


# --------------------------------------------------------------------------------- checking


def check(repo=REPO, entries=None):
    """Every way the repository and `docs/references.toml` currently disagree.

    Returns a list of str, each one a complaint naming the file it is about; empty when the
    prose, the markers, the managed blocks and the two generated files are all in step.

    PARAMETERS
    repo    : str, the checkout to read.
    entries : list of dict, the bibliography. Loaded from `repo` when not given.
    """
    entries = entries if entries is not None else load(os.path.join(repo, SOURCE))
    refs = index(entries)
    problems, citations = [], {}

    for rel in prose_files(repo):
        text = read(rel, repo)
        problems += _check_file(rel, text, entries, refs)
        for key in linked_keys(rel, text):
            citations.setdefault(key, []).append(rel)

    problems += _check_generated(repo, entries, citations)

    orphans = sorted(set(refs) - set(citations))
    for key in orphans:
        problems.append(f"{SOURCE}: [{key}] is in the bibliography but nothing cites it")
    return problems


def _check_file(rel, text, entries, refs):
    """The complaints about one file: unknown keys, wrong numbers, unlinked and broken cites."""
    problems = []
    prose = rendered_prose(rel, text)
    target = relative_bibliography(rel)

    for number, key, (start, end) in markers(rel, text):
        line = text.count("\n", 0, start) + 1
        if key not in refs:
            problems.append(f"{rel}:{line} cites [{key}], which is not in {SOURCE}")
            continue
        if number != refs[key]["number"]:
            problems.append(f"{rel}:{line} shows [{number}] for [{key}], which is number "
                            f"{refs[key]['number']}")
        source = text[start:end]
        if source.endswith(")") and f"({target}#ref-{key})" not in source:
            problems.append(f"{rel}:{line} links [{key}] by a path that is not {target}")

    for match in BROKEN_MARKER.finditer(prose):
        line = text.count("\n", 0, match.start()) + 1
        problems.append(f"{rel}:{line} has a citation marker with no link after it")

    problems += _check_markdown_structure(rel, text, refs)

    for names, year, letter, at, _replace in plain_citations(prose):
        line = text.count("\n", 0, at) + 1
        problems.append(f"{rel}:{line} cites {names} ({year}{letter}) in prose without a "
                        f"link; what is rendered as markdown cites by marker")

    unchecked = rel in SELF_REFERENTIAL
    for names, year, letter, at, _replace in plain_citations(code_prose(rel, text)):
        if not unchecked and resolve(names, year, letter, entries) is None:
            line = text.count("\n", 0, at) + 1
            problems.append(f"{rel}:{line} cites {names} ({year}{letter}), which is not an "
                            f"entry in {SOURCE}")
    return problems


def _check_markdown_structure(rel, text, refs):
    """A markdown file must list every key it cites, and define every link it needs.

    The two are not the same set. A file may list a work it never mentions in a sentence --
    that is what a reading list is -- but it may not cite one it does not list, because the
    number would then have nothing on the page to say what it stands for.
    """
    if not rel.endswith(".md"):
        return []
    problems = []
    cited, declared = set(cited_keys(rel, text)), set(declared_keys(rel, text))
    if not cited and not declared:
        return []

    if cited and not BLOCK.search(text):
        problems.append(f"{rel} cites {len(cited)} work(s) but carries no "
                        f"`<!-- references: ... -->` block to list them in")
    missing = sorted(cited - declared)
    if missing and BLOCK.search(text):
        problems.append(f"{rel} cites {missing} but no reference block lists them")
    unknown = sorted(declared - set(refs))
    if unknown:
        problems.append(f"{rel} lists {unknown} in a reference block, but {SOURCE} has no "
                        f"such entry")

    needed = {k for k in cited | declared if k in refs}
    defined = {m.group(1) for m in LINK_DEF.finditer(text)}
    if not LINKS_BLOCK.search(text):
        problems.append(f"{rel} needs {len(needed)} link definition(s) but has no "
                        f"`<!-- reference-links -->` block to hold them")
    else:
        undefined = sorted(needed - defined)
        if undefined:
            problems.append(f"{rel} links {undefined} with no definition")
    return problems


def _check_generated(repo, entries, citations):
    """The two generated files, against what the TOML and the prose now say."""
    problems = []
    wanted = {MARKDOWN: render_markdown(entries, {k: sorted(v) for k, v in citations.items()}),
              BIBTEX: render_bibtex(entries)}
    for rel, expected in wanted.items():
        path = os.path.join(repo, rel)
        if not os.path.exists(path):
            problems.append(f"{rel} does not exist; run `python scripts/bibliography.py "
                            f"--write`")
            continue
        if read(rel, repo) != expected:
            problems.append(f"{rel} is out of date; run `python scripts/bibliography.py "
                            f"--write`")
    return problems


# ------------------------------------------------------------------------------------- cli


def apply_all(repo=REPO, entries=None):
    """Rewrite everything that is generated, and report which files changed.

    Returns a sorted list of the repository-relative paths that were rewritten.

    PARAMETERS
    repo    : str, the checkout to write into.
    entries : list of dict, the bibliography. Loaded from `repo` when not given.
    """
    entries = entries if entries is not None else load(os.path.join(repo, SOURCE))
    changed, citations = [], {}

    for rel in prose_files(repo):
        text = read(rel, repo)
        fixed = rewrite(rel, text, entries)
        if fixed != text:
            write(rel, fixed, repo)
            changed.append(rel)
        for key in linked_keys(rel, fixed):
            citations.setdefault(key, []).append(rel)

    citations = {k: sorted(v) for k, v in citations.items()}
    for rel, text in ((MARKDOWN, render_markdown(entries, citations)),
                      (BIBTEX, render_bibtex(entries))):
        path = os.path.join(repo, rel)
        if not os.path.exists(path) or read(rel, repo) != text:
            write(rel, text, repo)
            changed.append(rel)
    return sorted(changed)


def main(argv=None):
    """Run `--check` or `--write` over the checkout this file lives in.

    Returns 0 where the repository agrees with `docs/references.toml`, 1 where it does not.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="rewrite the generated files, numbers and blocks in place")
    args = parser.parse_args(argv)

    entries = load()
    if args.write:
        changed = apply_all(entries=entries)
        for rel in changed:
            print(f"rewrote {rel}")
        print(f"{len(entries)} entries, {len(changed)} file(s) rewritten")

    problems = check(entries=entries)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} problem(s); run with --write to fix what is mechanical",
              file=sys.stderr)
        return 1
    print(f"{len(entries)} entries, every citation in step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
