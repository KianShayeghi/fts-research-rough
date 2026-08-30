"""Structural checks on the notebooks.

Executing every notebook would take tens of minutes, so instead each one is checked to be
valid, to have code that parses, to carry no stored traceback from its last run, and to
depend on nothing outside the repository: no machine-specific path, no bare executable name,
and no CSV read outside `fdaSP.empirical`.
"""

import ast
import json
import os
import re

import pytest

from fdaSP import empirical as emp

from _helpers import tracked_notebooks

NB_DIR = os.path.join(emp.repo_root(), "notebooks")
NOTEBOOKS = [os.path.basename(p) for p in tracked_notebooks()]


def load(name):
    with open(os.path.join(NB_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def source_of(nb):
    return "\n".join("".join(c["source"]) for c in code_cells(nb))


def test_there_are_notebooks_to_check():
    assert NOTEBOOKS


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_valid_json_with_cells(name):
    nb = load(name)
    assert nb["nbformat"] == 4
    assert nb["cells"], "an empty notebook is not a notebook"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_code_parses(name):
    """IPython magics aside, every code cell must be syntactically valid Python."""
    for i, cell in enumerate(code_cells(load(name))):
        src = "".join(cell["source"])
        if any(ln.lstrip().startswith(("%", "!")) for ln in src.splitlines()):
            continue                                  # cell magic / shell escape
        try:
            ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(f"{name} cell {i} does not parse: {exc}")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_no_stored_errors(name):
    """A committed notebook must not carry a traceback from its last execution."""
    for i, cell in enumerate(code_cells(load(name))):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                pytest.fail(f"{name} cell {i} stores a {out.get('ename')}: {out.get('evalue')}")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_no_absolute_machine_paths(name):
    """No `C:/Users/...`, no `/home/...`: those break the notebook on any other machine."""
    src = source_of(load(name))
    bad = re.findall(r"""["'](?:[A-Za-z]:[\\/]Users[\\/]|/home/|/Users/)[^"']*["']""", src)
    assert not bad, f"{name} hard-codes machine-specific paths: {bad[:3]}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_does_not_shell_out_to_a_bare_Rscript(name):
    """`subprocess.run(["Rscript", ...])` fails wherever R is not on PATH. Notebooks must go
    through `fdaSP.rbridge` instead."""
    src = source_of(load(name))
    assert not re.search(r"""\[\s*["']Rscript["']""", src), (
        f"{name} calls a bare Rscript; use fdaSP.rbridge.run_r"
    )


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_imports_resolve(name):
    """Every module a notebook imports must actually be importable in this environment."""
    import importlib
    src = source_of(load(name))
    modules = set()
    for node in ast.walk(ast.parse("\n".join(
            ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!"))))):
        if isinstance(node, ast.Import):
            modules.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    for mod in sorted(modules):
        importlib.import_module(mod)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_data_access_goes_through_the_loaders(name):
    """Relative CSV paths resolve against the kernel's working directory, so data must come
    from `fdaSP.empirical` instead."""
    src = source_of(load(name))
    # Bare relative reads in *Python* (R snippets are handed absolute paths by rbridge).
    bad = re.findall(r"""pd\.read_csv\(\s*["'](?!\w+:)[^"']*\.csv["']""", src)
    assert not bad, f"{name} reads a CSV by relative path: {bad[:3]}"


@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive:UserWarning"
)
def test_validation_notebook_runs_without_R(monkeypatch, tmp_path):
    """The R-comparison notebook must run end to end on a machine with no R, from any working
    directory, using only the committed reference CSVs.

    Executes the notebook's own code cells with `R_AVAILABLE` forced False, so a cell that
    quietly depended on R being present (or on the kernel's CWD) fails here.
    """
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)

    nb = load("LocSpaFLR_validation.ipynb")
    namespace = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]).replace(
            'R_AVAILABLE = rb.r_available("fdaSP")', "R_AVAILABLE = False")
        try:
            exec(compile(src, f"{name_of_cell(i)}", "exec"), namespace)   # noqa: S102
        except Exception as exc:
            pytest.fail(f"validation notebook cell {i} failed without R: "
                        f"{type(exc).__name__}: {exc}")
    assert "beta_py" in namespace, "the comparison never produced a Python fit"


def name_of_cell(i):
    return f"LocSpaFLR_validation.ipynb:cell{i}"


def test_tracked_notebooks_are_the_ones_git_knows_about():
    """The checks in this file and in test_docs apply to committed notebooks only.

    A notebook that is not yet added is work in progress and is left alone until it is
    committed, so an unfinished draft in `notebooks/` does not fail the suite. Everything the
    repository actually ships is tracked, so CI sees all of it.
    """
    tracked = tracked_notebooks()
    assert tracked, "no tracked notebooks found; the helper's fallback may have fired"
    assert all(p.startswith("notebooks/") and p.endswith(".ipynb") for p in tracked)
    assert all(os.path.exists(os.path.join(emp.repo_root(), p)) for p in tracked)

    on_disk = {f for f in os.listdir(NB_DIR) if f.endswith(".ipynb")}
    assert {os.path.basename(p) for p in tracked} <= on_disk


def test_every_tracked_notebook_is_listed_in_the_root_readme():
    """A notebook nobody points at is a notebook nobody opens."""
    with open(os.path.join(emp.repo_root(), "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    missing = [os.path.basename(p) for p in tracked_notebooks()
               if os.path.basename(p) not in readme]
    assert not missing, f"README.md never mentions {missing}"
