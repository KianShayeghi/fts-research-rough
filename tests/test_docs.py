"""Checks that the documentation stays true to the code and keeps to one house style.

Prose rots silently: a renamed parameter, a moved notebook or a stale README snippet costs
nothing at import time and is only found by the next reader. These checks read the docstrings,
the README code blocks and the notebook markdown, and hold them against the source.

The house style is the one used by `truncFLR.HH`, `truncFLR.HH_dgp`, `fdaSP.SBC` and
`fdaSP.SBC_sim`: a sentence saying what is computed, a `Returns` sentence, then a bare
`PARAMETERS` block naming only the arguments that need explaining. The three competing
conventions -- numpydoc's underlined headings, Google's argument headings and Sphinx's
cross-reference roles -- are rejected by `FOREIGN` below, so a docstring pasted in from
elsewhere is caught rather than quietly starting a second convention.
"""

import ast
import json
import math
import os
import re

import pytest

from _helpers import tracked_notebooks

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_LINE = 99  # matches .flake8
PACKAGES = ("truncFLR", "fdaSP", "far", "bench")


def _walk(*bases):
    for base in bases:
        for root, _dirs, files in os.walk(os.path.join(REPO, base)):
            if "__pycache__" in root or ".ipynb_checkpoints" in root:
                continue
            for name in sorted(files):
                yield os.path.relpath(os.path.join(root, name), REPO).replace("\\", "/")


PY_FILES = [p for p in _walk("src", "scripts", "tests") if p.endswith(".py")]
SRC_FILES = [p for p in PY_FILES if not p.startswith("tests/")]
MD_FILES = [p for p in _walk("src") if p.endswith("README.md")] + ["README.md"]
NB_FILES = tracked_notebooks()


def _tree(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        return ast.parse(fh.read())


def _defs(tree):
    """Module-level and class-level definitions only; nested closures are implementation."""
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(node)
        if isinstance(node, ast.ClassDef):
            out += [c for c in node.body if isinstance(c, ast.FunctionDef)]
    return out


def _signature(node):
    args = node.args
    return [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]


def _documented_parameters(doc):
    """Names introduced by a `PARAMETERS` block, e.g. `n_basis : int, ...` or `a, b : ...`."""
    if not doc or "PARAMETERS" not in doc:
        return []
    names = []
    for line in doc.split("PARAMETERS", 1)[1].splitlines():
        head, sep, _rest = line.partition(":")
        if not sep or not head.strip() or head != head.rstrip() and not head.strip():
            continue
        candidates = [c.strip() for c in head.split(",")]
        if candidates and all(re.fullmatch(r"\*{0,2}[A-Za-z_]\w*", c) for c in candidates):
            names += candidates
    return names


def _docstrings(rel):
    """(line number, docstring) for the module and each of its top-level definitions."""
    tree = _tree(rel)
    out = [(1, ast.get_docstring(tree, clean=False))]
    out += [(n.lineno, ast.get_docstring(n, clean=False)) for n in _defs(tree)]
    return [(ln, d) for ln, d in out if d]


# --------------------------------------------------------------------------- presence


@pytest.mark.parametrize("rel", SRC_FILES)
def test_every_module_and_public_definition_is_documented(rel):
    tree = _tree(rel)
    if os.path.getsize(os.path.join(REPO, rel)):
        assert ast.get_docstring(tree), f"{rel} has no module docstring"
    missing = [f"{rel}:{n.lineno} {n.name}" for n in _defs(tree)
               if not n.name.startswith("_") and not ast.get_docstring(n)]
    assert not missing, "undocumented public definitions: " + ", ".join(missing)


# --------------------------------------------------------------------------- accuracy


@pytest.mark.parametrize("rel", SRC_FILES)
def test_documented_parameters_are_real_parameters(rel):
    """A `PARAMETERS` entry must name an argument the function actually takes.

    This is the check that catches a rename: sim_dataset's `sigma` became `snr` while its
    docstring went on describing `sigma`.
    """
    wrong = []
    for node in _defs(_tree(rel)):
        if isinstance(node, ast.ClassDef):
            continue
        real = set(_signature(node))
        if node.args.vararg:
            real.add(node.args.vararg.arg)
        if node.args.kwarg:
            real.add(node.args.kwarg.arg)
        for name in _documented_parameters(ast.get_docstring(node, clean=False)):
            if name.lstrip("*") not in real:
                wrong.append(f"{rel}:{node.lineno} {node.name}() documents {name!r}, "
                             f"which is not in {sorted(real)}")
    assert not wrong, "\n".join(wrong)


SETTING = re.compile(r"\b([a-z_][a-z0-9_]{2,})\s*=\s*(?:True|False|None|[0-9])")


def _one_edit_apart(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long)):
        if long[:i] + long[i + 1:] == short:
            return True
    return False


@pytest.mark.parametrize("rel", SRC_FILES)
def test_prose_does_not_misspell_a_parameter_it_is_setting(rel):
    """`one_set=True` in a docstring whose argument is `one_se` reads as real until you try it.

    A `PARAMETERS` block is checked elsewhere; this catches the same mistake in running
    prose, where a near-miss of a real argument name is almost certainly a typo rather than
    a piece of mathematical notation. Only names within one edit of a genuine parameter are
    flagged, so `psi_1 = 1` and `u_hat = 1 - theta_hat` pass untouched.
    """
    wrong = []
    for node in _defs(_tree(rel)):
        if isinstance(node, ast.ClassDef):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            continue
        real = set(_signature(node))
        if node.args.vararg:
            real.add(node.args.vararg.arg)
        if node.args.kwarg:
            real.add(node.args.kwarg.arg)
        for name in set(SETTING.findall(doc)):
            if name in real:
                continue
            near = sorted(r for r in real if _one_edit_apart(name, r))
            if near:
                wrong.append(f"{rel}:{node.lineno} {node.name}() documents {name}=, "
                             f"one edit from its real argument {near}")
    assert not wrong, NEWLINE.join(wrong)


# --------------------------------------------------------------------------- house style


FOREIGN = [
    (re.compile(r":(?:func|class|data|mod|meth|attr|obj|exc|ref):`"),
     "Sphinx cross-reference role; this repository writes `name` in plain backticks"),
    (re.compile(r"(?m)^\s*(?:Parameters|Returns|Args|Raises|Attributes|Notes|Examples)"
                r"\s*:?\s*\n\s*-{3,}"),
     "numpydoc underlined heading; this repository uses a bare PARAMETERS block"),
    (re.compile(r"(?m)^\s*(?:Args|Arguments|Params|Keyword Args):\s*$"),
     "Google-style heading; this repository uses a bare PARAMETERS block"),
]


@pytest.mark.parametrize("rel", PY_FILES)
def test_docstrings_keep_to_the_house_style(rel):
    bad = []
    for lineno, doc in _docstrings(rel):
        for pattern, why in FOREIGN:
            if pattern.search(doc):
                bad.append(f"{rel}:{lineno} uses a {why}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("rel", PY_FILES)
def test_docstring_lines_fit_the_line_limit(rel):
    """flake8 does not measure inside a docstring, so wrapping is checked here."""
    long = [f"{rel}:{lineno} ({len(line)} chars) {line.strip()[:60]}..."
            for lineno, doc in _docstrings(rel)
            for line in doc.split("\n") if len(line) > MAX_LINE]
    assert not long, "\n".join(long)


# --------------------------------------------------------------------------- README code


def _python_blocks(text):
    return re.findall(r"```python\n(.*?)```", text, re.S)


def _api():
    """{"fdaSP.SBC.cv_fit": [parameter names]} read from the sources, without importing."""
    api = {}
    for rel in SRC_FILES:
        if not rel.startswith("src/"):
            continue
        module = rel[len("src/"):-len(".py")].replace("/", ".")
        for node in _defs(_tree(rel)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                api[f"{module}.{node.name}"] = (_signature(node), node.args.kwarg is not None)
    return api


@pytest.mark.parametrize("rel", MD_FILES)
def test_readme_python_blocks_parse(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        blocks = _python_blocks(fh.read())
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError as exc:
            pytest.fail(f"{rel}: python block does not parse: {exc}")


@pytest.mark.parametrize("rel", MD_FILES)
def test_readme_calls_use_real_keyword_arguments(rel):
    """A usage snippet must be runnable.

    The far package's README passed sim_dataset_window(..., sigma=0.1, ...) long after that
    argument was renamed to `snr`, so the documented call raised TypeError.
    """
    api = _api()
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        blocks = _python_blocks(fh.read())
    wrong = []
    for block in blocks:
        tree = ast.parse(block)
        alias = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in PACKAGES:
                for name in node.names:
                    alias[name.asname or name.name] = f"{node.module}.{name.name}"
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            owner = node.func.value
            if not (isinstance(owner, ast.Name) and owner.id in alias):
                continue
            target = f"{alias[owner.id]}.{node.func.attr}"
            if target not in api:
                wrong.append(f"{rel}: documents {target}(), which does not exist")
                continue
            params, takes_kwargs = api[target]
            if takes_kwargs:
                continue
            for kw in node.keywords:
                if kw.arg and kw.arg not in params:
                    wrong.append(f"{rel}: {target}(...) is passed {kw.arg!r}, "
                                 f"not one of {params}")
    assert not wrong, "\n".join(wrong)


PACKAGE_DIRS = sorted(
    d for d in os.listdir(os.path.join(REPO, "src"))
    if os.path.isfile(os.path.join(REPO, "src", d, "README.md"))
)


@pytest.mark.parametrize("package", PACKAGE_DIRS)
def test_every_module_is_described_in_its_package_readme(package):
    """A module nobody documents is a module nobody finds.

    `fdaSP`'s README described three of its five modules; `rbridge` and `r_reference` existed
    only in the root README's one-line summary, so a reader of the package's own README had
    no way to know the R bridge was there.
    """
    here = os.path.join(REPO, "src", package)
    with open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    missing = [m for m in sorted(os.listdir(here))
               if m.endswith(".py") and m != "__init__.py" and m not in readme]
    assert not missing, f"src/{package}/README.md never mentions {missing}"


# --------------------------------------------------------------------------- cross-refs


REFERENCE = re.compile(r"`([A-Za-z0-9_/.]+\.(?:py|ipynb|md|toml))`")
SEARCH_PATHS = ["", "notebooks/", "scripts/", "tests/", "src/",
                "src/truncFLR/", "src/fdaSP/", "src/far/", "src/bench/"]


def _exists_exactly(rel):
    """Case-sensitive existence, so a reference that only works on Windows still fails."""
    current = REPO
    for part in rel.split("/"):
        if part == "..":
            current = os.path.dirname(current)
            continue
        try:
            if part not in os.listdir(current):
                return False
        except OSError:
            return False
        current = os.path.join(current, part)
    return True


def _documentation_text():
    """(file, prose) for every place this repository writes prose about itself."""
    for rel in PY_FILES:
        yield rel, "\n".join(doc for _lineno, doc in _docstrings(rel))
    for rel in MD_FILES:
        with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
            yield rel, fh.read()
    for rel in NB_FILES:
        with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
            cells = json.load(fh)["cells"]
        yield rel, "\n".join("".join(c["source"]) for c in cells
                             if c["cell_type"] == "markdown")


@pytest.mark.parametrize("rel,text", list(_documentation_text()), ids=lambda v: str(v)[:60])
def test_referenced_repository_files_exist(rel, text):
    """Every module and notebook named in backticks in prose must resolve to a real file.

    The match is case-sensitive, so a reference that differs only in case -- which opens
    happily on Windows and nowhere else -- fails here.
    """
    here = os.path.dirname(rel)
    missing = sorted({
        name for name in REFERENCE.findall(text)
        if not any(_exists_exactly(prefix + name) for prefix in SEARCH_PATHS)
        and not (here and _exists_exactly(f"{here}/{name}"))
        and not (here and _exists_exactly(f"{here}/../{name}"))
    })
    assert not missing, f"{rel} refers to files that do not exist: {missing}"


# --------------------------------------------------------------------------- notebooks


NEWLINE = "\n"
QUOTED_FIGURE = re.compile(r"(?<![\w.])(\d+\.\d{3,})(?![\w])")
ANY_NUMBER = re.compile(r"(?<![\w.])(\d+\.\d+(?:[eE][-+]?\d+)?)(?![\w])")


def _supported(quoted, numbers):
    """True if some number rounds to `quoted` (a reported value) or truncates to it (a
    bound, as in "correlations exceed 0.999")."""
    decimals = len(quoted.split(".")[1])
    target = float(quoted)
    scale = 10.0 ** decimals
    return any(round(n, decimals) == target
               or math.floor(abs(n) * scale) / scale == target
               for n in numbers)


def _notebook_numbers(cells):
    """Every number the notebook itself puts on the page: code parameters and stored output."""
    text = []
    for c in cells:
        if c["cell_type"] != "code":
            continue
        text.append("".join(c["source"]))
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                text.append("".join(o.get("text", [])))
            elif o.get("output_type") in ("execute_result", "display_data"):
                plain = o.get("data", {}).get("text/plain")
                if plain:
                    text.append("".join(plain))
    return [float(x) for x in ANY_NUMBER.findall(NEWLINE.join(text))]


@pytest.mark.parametrize("rel", NB_FILES)
def test_notebook_prose_only_quotes_figures_the_notebook_produces(rel):
    """A result quoted in prose must be one the notebook's own cells actually show.

    Numbers to three or more decimals read as measurements, not as model parameters, and
    they go stale the moment a cell is re-run with a different seed or sample size. Three
    such claims had already drifted: a mean endpoint of 0.809 where the cell printed
    0.80683, an agreement of 1e-3 where the cell printed 8.4e-3, and a pair of
    30-seed figures quoted in a notebook whose cell runs 10 seeds.
    """
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        cells = json.load(fh)["cells"]
    numbers = _notebook_numbers(cells)
    stale = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "markdown":
            continue
        for quoted in sorted(set(QUOTED_FIGURE.findall("".join(c["source"])))):
            if not _supported(quoted, numbers):
                stale.append(f"{rel} cell {i} quotes {quoted}, which no cell produces")
    assert not stale, NEWLINE.join(stale)


@pytest.mark.parametrize("rel", NB_FILES)
def test_notebook_markdown_cells_are_not_empty(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        cells = json.load(fh)["cells"]
    empty = [i for i, c in enumerate(cells)
             if c["cell_type"] == "markdown" and not "".join(c["source"]).strip()]
    assert not empty, f"{rel} has empty markdown cells at {empty}"


@pytest.mark.parametrize("rel", NB_FILES)
def test_notebook_opens_with_a_titled_markdown_cell(rel):
    """Every notebook states what it is doing before it does anything."""
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        cells = json.load(fh)["cells"]
    assert cells, f"{rel} is empty"
    first = cells[0]
    assert first["cell_type"] == "markdown", f"{rel} opens with code, not an explanation"
    assert "".join(first["source"]).lstrip().startswith("#"), f"{rel} has no title"
