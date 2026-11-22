"""The plant layer reaches calibration internals only through ``aquakin.integrate.seam``.

``aquakin.integrate.seam`` lists the private names of the calibration family
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

# The modules whose private names the plant layer may only take from the seam.
FAMILY = {
    "aquakin.integrate.calibrate",
    "aquakin.integrate.fisher",
    "aquakin.integrate.profile",
    "aquakin.integrate.experiment_design",
    "aquakin.integrate.identifiability",
    "aquakin.integrate.candidate_design",
}

# The layers above ``integrate`` that may use the seam.
CONSUMERS = ("plant", "study")


def _in_family(module: str) -> bool:
    return module in FAMILY or module.startswith("aquakin.integrate.calibration")


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


def test_private_calibration_names_come_only_from_the_seam():
    """No plant or study module imports a private calibration-family name
    straight from its defining module."""
    bypass = [
        f"{rel}:{lineno} imports {name} from {module}"
        for rel, lineno, module, name in _consumer_imports()
        if name.startswith("_") and _in_family(module)
    ]
    assert bypass == [], "import these from aquakin.integrate.seam:\n" + "\n".join(bypass)


def test_every_seam_name_has_a_plant_caller():
    """The seam lists exactly the names its consumers import from it: none is
    missing from ``__all__`` and none is left over."""
    seam = importlib.import_module(SEAM)
    used = {name for _rel, _lineno, module, name in _consumer_imports() if module == SEAM}
    assert sorted(seam.__all__) == sorted(used)


def test_the_check_sees_a_bypassing_import(tmp_path, monkeypatch):
    """The scan catches a private import from a family module, including one
    inside a function body (so the first test can fail)."""
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
        if name.startswith("_") and _in_family(module)
    ]
    assert found == [("plant/probe.py", "aquakin.integrate.calibration.loss", "_build_loss")]
