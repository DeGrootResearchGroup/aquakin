"""Imports between aquakin's subpackages point downward.

The layering is ``core`` < ``schema`` < ``integrate`` < ``plant`` < ``study``: a
package may import the ones below it, never one above. ``utils`` sits on the
``core`` level (the two import each other). Every import statement in the
package is checked, including those inside function bodies and relative
imports, so an upward import cannot hide in a lazy import.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[2] / "aquakin"

LEVELS = {"core": 0, "utils": 0, "schema": 1, "integrate": 2, "plant": 3, "study": 4}

# The sanctioned upward imports, keyed by (file, imported module):
# - ``CompiledModel.influent`` is a convenience constructor for a plant influent,
#   kept on the model because ``net.influent(...)`` is how every plant example
#   builds its feed.
# - ``CompiledModel.composition()`` falls back to a shipped model's composition
#   table, and choosing that table compares the model's species with the shipped
#   model's, which only the loader can produce.
ALLOWED = {
    ("core/model.py", "aquakin.plant.influent"),
    ("utils/composition.py", "aquakin.schema.loader"),
}


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
        if layer not in LEVELS:
            continue
        above = {f"aquakin.{name}" for name, level in LEVELS.items() if level > LEVELS[layer]}
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
