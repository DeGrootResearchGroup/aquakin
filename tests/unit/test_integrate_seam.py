"""The plant layer reaches ``integrate`` internals only through ``aquakin.integrate.seam``.

``aquakin.integrate.seam`` lists the private names of ``aquakin.integrate``
that ``aquakin.plant`` reuses, so a rename inside ``integrate`` shows up as a
change to that one list. These checks keep it the only route, and keep the list
free of names nothing uses. Every import statement is checked, including those
inside function bodies.
"""

import ast
import importlib
from pathlib import Path

PACKAGE = Path(__file__).parents[2] / "aquakin"

SEAM = "aquakin.integrate.seam"

# The layers above ``integrate`` that may use the seam.
CONSUMERS = ("plant", "study")


def _bypasses_seam(module: str) -> bool:
    """Whether ``module`` is an ``aquakin.integrate`` module other than the seam."""
    return module.startswith("aquakin.integrate") and module != SEAM


def _consumer_imports():
    """Yield ``(file, lineno, module, name)`` for every ``from aquakin.integrate...
    import name`` in the consuming layers."""
    for layer in CONSUMERS:
        for path in sorted((PACKAGE / layer).rglob("*.py")):
            rel = path.relative_to(PACKAGE).as_posix()
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("aquakin.integrate"):
                        for alias in node.names:
                            yield rel, node.lineno, node.module, alias.name


def test_private_integrate_names_come_only_from_the_seam():
    """No plant or study module imports a private ``aquakin.integrate`` name
    from anywhere but the seam."""
    bypass = [
        f"{rel}:{lineno} imports {name} from {module}"
        for rel, lineno, module, name in _consumer_imports()
        if name.startswith("_") and _bypasses_seam(module)
    ]
    assert bypass == [], "import these from aquakin.integrate.seam:\n" + "\n".join(bypass)


def test_every_seam_name_has_a_plant_caller():
    """The seam lists exactly the names its consumers import from it: none is
    missing from ``__all__`` and none is left over."""
    seam = importlib.import_module(SEAM)
    used = {name for _rel, _lineno, module, name in _consumer_imports() if module == SEAM}
    assert sorted(seam.__all__) == sorted(used)


def test_the_check_sees_a_bypassing_import(tmp_path, monkeypatch):
    """The scan catches a private import from an ``integrate`` module,
    including one inside a function body (so the first test can fail)."""
    plant = tmp_path / "plant"
    plant.mkdir()
    (plant / "probe.py").write_text(
        "def f():\n    from aquakin.integrate.calibration.loss import _build_loss\n"
    )
    (tmp_path / "study").mkdir()
    monkeypatch.setitem(globals(), "PACKAGE", tmp_path)
    found = [
        (rel, module, name)
        for rel, _lineno, module, name in _consumer_imports()
        if name.startswith("_") and _bypasses_seam(module)
    ]
    assert found == [("plant/probe.py", "aquakin.integrate.calibration.loss", "_build_loss")]
