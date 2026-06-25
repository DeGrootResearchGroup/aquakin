"""Single-source-of-truth guard for the ODE solver configuration.

Every implicit ODE solve in aquakin must build its diffrax solver and step
controller through the shared helpers ``build_implicit_solver`` /
``build_step_controller`` (``aquakin.integrate._common``). Those helpers are the
mechanism that keeps the reactor and plant solve paths on ONE common solver mode
-- the decoupled-Newton ESDIRK + PID controller -- instead of drifting apart (the
bug class that let the plant's stable forward-sensitivity config diverge from the
reactors').

This test fails if any module constructs a diffrax solver / step controller /
root finder *directly* (``diffrax.Kvaerno5(...)``, ``PIDController(...)``,
``VeryChord(...)``, ``with_stepsize_controller_tols(...)``) outside the small,
explicit allowlist below. To add a genuinely new construction site, add it to
``_ALLOW`` with a one-line reason -- the auditable exemption list, in lieu of
Python access control (which can't *prevent* the construction).
"""
import ast
import pathlib

import aquakin

_AQUAKIN = pathlib.Path(aquakin.__file__).resolve().parent

# The constructors that define the implicit-solve mode; building one outside the
# helpers is solver-config drift.
_FORBIDDEN = {"Kvaerno3", "Kvaerno5", "PIDController", "VeryChord",
              "with_stepsize_controller_tols"}

# Files allowed to construct these directly, each with the reason it is exempt.
_ALLOW = {
    "integrate/_common.py":
        "the single-source-of-truth builders (build_implicit_solver / "
        "build_step_controller) and the _CANONICAL_SOLVERS table live here",
}


def _direct_constructions(path):
    """``(lineno, name)`` for each direct forbidden constructor *call* in a file.

    Uses the AST, so a constructor named only in a docstring or comment (e.g. the
    ``Plant.solve(solver=diffrax.Kvaerno3(...))`` usage example) is NOT counted --
    only an actual call node is.
    """
    tree = ast.parse(path.read_text())
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = (f.attr if isinstance(f, ast.Attribute)
                else f.id if isinstance(f, ast.Name) else None)
        if name in _FORBIDDEN:
            hits.append((node.lineno, name))
    return hits


def test_no_solver_construction_outside_helpers():
    offenders = []
    for py in sorted(_AQUAKIN.rglob("*.py")):
        rel = py.relative_to(_AQUAKIN).as_posix()
        if rel in _ALLOW:
            continue
        for lineno, name in _direct_constructions(py):
            offenders.append(f"{rel}:{lineno}  {name}(...)")
    assert not offenders, (
        "diffrax solver / controller / root finder constructed outside the "
        "single-source-of-truth helpers:\n  " + "\n  ".join(offenders)
        + "\n\nBuild it through build_implicit_solver / build_step_controller "
        "(aquakin.integrate._common), or -- if it is a genuine new exception -- "
        "add the file to _ALLOW in this test with a reason.")


def test_allowlist_is_not_stale():
    """Each allowlisted file must exist and still construct a forbidden primitive,
    so a stale exemption (the construction was refactored away) surfaces and gets
    removed -- keeping the exemption list honest."""
    for rel, reason in _ALLOW.items():
        path = _AQUAKIN / rel
        assert path.exists(), f"allowlisted file does not exist: {rel}"
        assert reason, f"allowlist entry {rel} needs a reason"
        assert _direct_constructions(path), (
            f"allowlisted file {rel} no longer constructs a forbidden primitive; "
            f"remove it from _ALLOW.")
