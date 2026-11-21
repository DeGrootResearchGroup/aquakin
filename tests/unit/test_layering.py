"""Imports between aquakin's subpackages point downward.

The layering is ``core`` < ``schema`` < ``integrate`` < ``plant`` < ``study``: a
package may import the ones below it, never one above. Every import statement in
the package is checked, including those inside function bodies and relative
imports, so an upward import cannot hide in a lazy import.

``utils`` is not in the chain: it and ``core`` import each other, and
``utils.composition`` reads the shipped models through ``schema.loader``.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[2] / "aquakin"

LAYERS = ["core", "schema", "integrate", "plant", "study"]

# The one sanctioned upward import: ``CompiledModel.influent`` is a convenience
# constructor for a plant influent, kept on the model because ``net.influent(...)``
# is how every plant example builds its feed. Keyed by (file, imported module).
ALLOWED = {("core/model.py", "aquakin.plant.influent")}


def _imported_modules(source: str, package: list[str]):
    """Yield ``(lineno, absolute module name)`` for every import in ``source``.

    ``package`` is the dotted package the file belongs to, as a list of parts
    (``["aquakin", "core"]`` for ``aquakin/core/model.py`` and for
    ``aquakin/core/__init__.py``), against which relative imports resolve.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                yield node.lineno, ".".join(base + ([node.module] if node.module else []))
            elif node.module:
                yield node.lineno, node.module


def _upward_imports():
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        layer = path.relative_to(PACKAGE).parts[0]
        if layer not in LAYERS:
            continue
        above = {f"aquakin.{name}" for name in LAYERS[LAYERS.index(layer) + 1 :]}
        rel = path.relative_to(PACKAGE).as_posix()
        package = ["aquakin", *path.relative_to(PACKAGE).parts[:-1]]
        for lineno, module in _imported_modules(path.read_text(), package):
            top = ".".join(module.split(".")[:2])
            if top in above and (rel, module) not in ALLOWED:
                found.append(f"{rel}:{lineno} imports {module}")
    return found


def test_no_subpackage_imports_a_higher_layer():
    assert _upward_imports() == []


def test_relative_imports_are_resolved():
    """The scanner reads relative imports as the absolute module they name."""
    source = "from ..plant.influent import X\nfrom . import model\nimport aquakin.study\n"
    names = [module for _, module in _imported_modules(source, ["aquakin", "core"])]
    assert names == ["aquakin.plant.influent", "aquakin.core", "aquakin.study"]
