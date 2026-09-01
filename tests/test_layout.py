"""Checks that the shape of the repository stays the shape it is documented to have.

The arrangement is not decoration. `core` holds what more than one chapter needs, so that a
quadrature rule or a basis has exactly one definition and two estimators cannot silently drift
apart; the method packages sit on top of it and know nothing of each other bar the dependences
declared below; and the repository ends in three packages nothing imports -- `bench`, because
comparing the three estimators is all it does, and `ftsa` and `jump`, which forecast the next
curve and the scalar the curve does not contain.
Left unchecked that ordering erodes one import at a time -- it had already, with `far` reading
`quadrature_weights` out of `truncFLR`'s private module and `bench` keeping a second copy of
the same six lines -- and nothing fails when it does.

`ALLOWED` is therefore the dependency graph as a statement rather than as a description, and
the tests here read the imports back against it. The same goes for the directories: only
`core.paths` may name one, and `artifacts/` is split into the half that comes from R and the
half this repository rebuilds itself, which nothing may quietly merge again.
"""

import ast
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src", "fts_research")

# Which packages each package may import. `core` depends on nothing, so it can be read on its
# own; `far` reads `fdaSP.SBC_sim`'s coefficient functions, because `far_dgp.sim_dataset_psi`
# is the FAR(1) analogue of that module's simulator and has to hit the same targets. `ftsa` and
# `jump` sit on `far` because both forecast the same principal component scores it does --
# `ftsa` forecasts them with `far_est.score_operator` itself, and `jump` regresses the next
# jump on them -- and `ftsa` reads `fdaSP.intraday` for the panel its replication design is
# defined on.
ALLOWED = {
    "core": set(),
    "truncFLR": {"core"},
    "fdaSP": {"core"},
    "far": {"core", "fdaSP"},
    "ftsa": {"core", "far", "fdaSP"},
    "jump": {"core", "far"},
    "bench": {"core", "truncFLR", "fdaSP"},
}

# Directory names only `core.paths` may join onto a path.
DIRECTORY_NAMES = {"data", "artifacts", "figures"}
PATHS_MODULE = "core/paths.py"
IN_A_PATH = re.compile(r"(?:^|[\\/])(?:data|artifacts|figures)(?:[\\/]|$)")


def _modules():
    """(package, relative path, absolute path) for every module under `src/fts_research/`."""
    out = []
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, SRC).replace("\\", "/")
            package = rel.split("/")[0] if "/" in rel else ""
            out.append((package, rel, path))
    return out


MODULES = _modules()
PACKAGES = sorted(ALLOWED)


def _tree(path):
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read())


def _imported_names(rel, tree):
    """Every `fts_research.*` dotted name this module imports, relative forms resolved."""
    here = ["fts_research"] + rel[:-len(".py")].split("/")
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = here[:-node.level]
                parts = base + (node.module.split(".") if node.module else [])
            elif node.module and node.module.split(".")[0] == "fts_research":
                parts = node.module.split(".")
            else:
                continue
            names += [".".join(parts + [alias.name]) for alias in node.names]
            names.append(".".join(parts))
        elif isinstance(node, ast.Import):
            names += [a.name for a in node.names if a.name.split(".")[0] == "fts_research"]
    return names


def _target_package(dotted):
    parts = dotted.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "fts_research" else None


# ------------------------------------------------------------------ the dependency graph


@pytest.mark.parametrize("package,rel,path", MODULES, ids=[m[1] for m in MODULES])
def test_module_only_imports_what_its_package_may_import(package, rel, path):
    """The layering, read straight off the import statements."""
    permitted = ALLOWED.get(package, set()) | {package} - {""}
    bad = sorted({
        dotted for dotted in _imported_names(rel, _tree(path))
        if (target := _target_package(dotted)) is not None and target not in permitted
    })
    assert not bad, (
        f"src/fts_research/{rel} imports {bad}, but {package or 'the top level'} may import "
        f"{sorted(ALLOWED.get(package, ())) or 'nothing'}. Either the shared piece belongs in "
        f"core, "
        f"or ALLOWED in this file is out of date and the change is deliberate."
    )


@pytest.mark.parametrize("package,rel,path", MODULES, ids=[m[1] for m in MODULES])
def test_no_package_reaches_into_another_packages_private_module(package, rel, path):
    """`far` once imported `truncFLR._common.quadrature_weights`. A leading underscore is a
    promise that the name may be moved without warning, and the promise only holds if it is
    kept inside the package that made it."""
    bad = []
    for dotted in _imported_names(rel, _tree(path)):
        parts = dotted.split(".")
        if _target_package(dotted) in (None, package):
            continue
        if any(p.startswith("_") and not p.startswith("__") for p in parts[2:]):
            bad.append(dotted)
    assert not bad, f"src/fts_research/{rel} imports the private {sorted(set(bad))}"


def test_the_declared_graph_is_acyclic_and_complete():
    """`ALLOWED` must name every package that exists, and admit no cycle."""
    on_disk = {d for d in os.listdir(SRC)
               if os.path.isdir(os.path.join(SRC, d)) and d != "__pycache__"}
    assert on_disk == set(ALLOWED), f"packages on disk {sorted(on_disk)} != {PACKAGES}"

    depth, seen = {}, set()

    def resolve(pkg, stack=()):
        if pkg in depth:
            return depth[pkg]
        assert pkg not in stack, f"dependency cycle: {' -> '.join(stack + (pkg,))}"
        depth[pkg] = 1 + max([resolve(d, stack + (pkg,)) for d in ALLOWED[pkg]], default=-1)
        seen.add(pkg)
        return depth[pkg]

    for pkg in ALLOWED:
        resolve(pkg)
    assert depth["core"] == 0, "core must sit under everything"

    # The packages nothing imports are where the repository ends, and there are exactly three:
    # `bench`, which compares the three scalar-on-function estimators and is all it does, and
    # `ftsa` and `jump`, the two halves of the functional time series chapter -- one forecasts
    # the next curve, the other the scalar the curve does not contain. A fourth sink would mean
    # a package nothing uses; one fewer would mean a comparison had grown a dependent.
    depended_on = set().union(*ALLOWED.values())
    assert sorted(set(ALLOWED) - depended_on) == ["bench", "ftsa", "jump"], (
        f"the packages nothing imports are {sorted(set(ALLOWED) - depended_on)}")


# ------------------------------------------------------------------------ the directories


def _code_strings(tree):
    """String constants that are not docstrings."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings]


def _joined_strings(tree):
    """String constants passed to `os.path.join`, which is how a directory name reaches a
    path. A dict key that happens to read `"cache"` is not that."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "join":
            out += [a.value for a in node.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    return out


@pytest.mark.parametrize("package,rel,path", MODULES, ids=[m[1] for m in MODULES])
def test_only_core_paths_names_a_repository_directory(package, rel, path):
    """One module decides where things are kept, so moving `artifacts/` is one edit."""
    if rel == PATHS_MODULE:
        return
    tree = _tree(path)
    named = sorted({s for s in _code_strings(tree)
                    if "://" not in s and IN_A_PATH.search(s)})
    named += sorted({s for s in _joined_strings(tree) if s in DIRECTORY_NAMES})
    assert not named, (
        f"src/fts_research/{rel} builds a path from {sorted(set(named))} itself; take it "
        f"from `core.paths` instead."
    )


def test_artifacts_is_split_into_reference_and_cache():
    """R's outputs and this repository's own caches are rebuilt by different commands, so
    they are not kept in one heap."""
    artifacts = os.path.join(REPO, "artifacts")
    entries = sorted(os.listdir(artifacts))
    assert entries == ["README.md", "cache", "reference"], (
        f"artifacts/ holds files outside the two halves: {entries}")
    assert os.listdir(os.path.join(artifacts, "reference"))
    assert os.listdir(os.path.join(artifacts, "cache"))


def test_every_package_has_a_readme():
    missing = [p for p in PACKAGES if not os.path.exists(os.path.join(SRC, p, "README.md"))]
    assert not missing, f"packages without a README: {missing}"


def _tracked(pattern):
    """Files matching a pathspec that git is tracking, or None where git cannot be reached."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", pattern], cwd=REPO,
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return sorted(p for p in out.stdout.split("\0") if p)


def test_generated_figures_are_not_committed():
    """`figures/` is rebuilt by the notebooks, so only its README belongs in the history."""
    tracked = _tracked("figures/")
    if tracked is None:
        pytest.skip("not a git checkout")
    assert tracked == ["figures/README.md"], f"figures/ has committed output: {tracked}"


# ----------------------------------------------------------------------------- the tests


def test_every_module_is_imported_by_some_test():
    """A module no test imports is a module nothing here is claiming to have checked."""
    here = os.path.dirname(os.path.abspath(__file__))
    imported = set()
    for name in sorted(os.listdir(here)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(here, name), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                if node.module.split(".")[0] != "fts_research":
                    continue
                imported.add(node.module)
                imported.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names
                                if a.name.split(".")[0] == "fts_research")

    missing = []
    for package, rel, _path in MODULES:
        stem = rel[:-len(".py")]
        if stem.endswith("__init__"):
            continue
        if f"fts_research.{stem.replace('/', '.')}" not in imported:
            missing.append(rel)
    assert not missing, f"modules no test imports: {missing}"
