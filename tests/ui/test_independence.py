"""The interface never imports aquakin or JAX, and aquakin never imports the interface."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
UI_FILES = sorted((ROOT / "aquakin_ui").rglob("*.py"))


def _module_name(path):
    """The dotted module name of a file under the repository root."""
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imported_roots(path):
    """
    The top-level package of every absolute import in a file.

    Every ``import`` and ``from ... import`` statement counts wherever it sits
    (module level, inside a function, under ``if TYPE_CHECKING``), as does an
    ``importlib.import_module`` or ``__import__`` call with a literal name.
    Relative imports stay inside the package and are skipped.

    Returns
    -------
    set of (str, int)
        ``(top-level package, line number)`` for each import found.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            first = node.args[0]
            if called in ("import_module", "__import__") and isinstance(first, ast.Constant):
                if isinstance(first.value, str):
                    names = [first.value]
        found.update((n.split(".")[0], node.lineno) for n in names)
    return found


def test_the_scan_sees_imports_inside_functions(tmp_path):
    """A lazy import in a function body is caught, which a runtime import cannot see."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os\n"
        "from . import sibling\n"
        "def later():\n"
        "    import aquakin.plant\n"
        "    from jax import numpy\n"
        "    import importlib\n"
        "    importlib.import_module('jax.numpy')\n"
    )
    roots = _imported_roots(probe)
    assert roots == {("os", 1), ("aquakin", 4), ("jax", 5), ("importlib", 6), ("jax", 7)}


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_interface_module_imports_aquakin_or_jax(path):
    bad = sorted(item for item in _imported_roots(path) if item[0] in ("aquakin", "jax"))
    assert bad == [], f"{path.relative_to(ROOT)} imports {bad}"


def test_importing_the_interface_loads_neither_aquakin_nor_jax():
    """Importing every interface module in a fresh interpreter pulls in neither package."""
    modules = [_module_name(p) for p in UI_FILES]
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "bad = [m for m in sys.modules if m in ('jax', 'aquakin') or m.startswith(('jax.', 'aquakin.'))]\n"
        "print(','.join(sorted(bad)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert out.stdout.strip() == ""


def test_the_fresh_interpreter_check_covers_every_module():
    """Every interface file is imported above, including ``__main__`` and subpackages."""
    modules = {_module_name(p) for p in UI_FILES}
    assert {
        "aquakin_ui",
        "aquakin_ui.__main__",
        "aquakin_ui.jobs",
        "aquakin_ui.widget.module",
    } <= modules
    assert len(modules) == len(UI_FILES)


def test_aquakin_never_imports_the_interface():
    for path in (ROOT / "aquakin").rglob("*.py"):
        bad = sorted(item for item in _imported_roots(path) if item[0] == "aquakin_ui")
        assert bad == [], f"{path.relative_to(ROOT)} imports {bad}"
