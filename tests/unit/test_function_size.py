"""Functions in the package stay small enough to read, as a ratchet.

CLAUDE.md, *Keep modules and functions small enough to read*: "A function past
100 lines of body, or with more than eight parameters, gets split into named
steps or takes a config dataclass before it gets longer", and never gets under
the limit by removing a capability. This module checks it over
``aquakin/`` and ``aquakin_ui/``, as ``test_size_limits.py`` does for files:

- a function whose body (without its docstring) is over ``MAX_BODY_LINES``
  lines, or that takes over ``MAX_PARAMS`` parameters (``self`` / ``cls`` not
  counted; ``*args`` / ``**kwargs`` not counted), fails unless it is listed;
- a listed function may not grow: past ``LINE_ALLOWANCE`` lines over its
  recorded length (formatting churn), or by any parameter;
- an entry must be lowered when its function shrinks well below it, and deleted
  once the function is within the limit, so the debt only shrinks.

Functions are keyed ``path::Qualified.name`` (nested functions and methods by
their enclosing names). A function renamed or moved with its body unchanged
needs its entry renamed in the same change. Everything here is AST processing:
no JAX, no ``aquakin`` import.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]
PACKAGES = ("aquakin", "aquakin_ui")
MAX_BODY_LINES = 100
MAX_PARAMS = 8
LINE_ALLOWANCE = 5

# Pre-existing debt. Lower an entry when its function shrinks; delete it once
# the function is within the limit. Never raise one: split the function, or
# group its parameters into a config dataclass.
RECORDED_LINES = {
    "aquakin/core/model.py::compile_model": 140,
    "aquakin/core/ph_solver.py::solve_ph": 148,
    "aquakin/core/precipitation.py::build_precipitation_derived_fn": 124,
    "aquakin/core/speciation.py::build_ph_derived_fn": 129,
    "aquakin/integrate/biofilm.py::BiofilmReactor.__init__": 173,
    "aquakin/integrate/calibration/laplace.py::_laplace_posterior": 145,
    "aquakin/integrate/calibration/multistart.py::_run_multistart": 153,
    "aquakin/integrate/candidate_design.py::design_from_candidates": 150,
    "aquakin/integrate/discrete_adjoint.py::_discrete_adjoint_solve": 119,
    "aquakin/integrate/discrete_adjoint.py::esdirk_adjoint_solve": 146,
    "aquakin/integrate/experiment_design.py::optimize_experiment": 120,
    "aquakin/integrate/fisher.py::_FisherSetup.__init__": 101,
    "aquakin/integrate/fit.py::fit": 113,
    "aquakin/integrate/forward_solve.py::_integrate": 130,
    "aquakin/integrate/profile.py::profile_likelihood": 109,
    "aquakin/integrate/steady.py::arclength_continuation_solve": 111,
    "aquakin/integrate/steady.py::ptc_forward": 158,
    "aquakin/models/_make_khalil_balanced_biofilm_biomass.py::main": 101,
    "aquakin/plant/a2o.py::build_a2o": 156,
    "aquakin/plant/bsm/bsm1.py::build_bsm1": 171,
    "aquakin/plant/bsm/bsm2.py::build_bsm2": 354,
    "aquakin/plant/bsm/evaluation.py::evaluate_bsm2": 118,
    "aquakin/plant/calibrate.py::_resolve_plant_problem": 211,
    "aquakin/plant/dynamic_solve.py::solve": 178,
    "aquakin/plant/ifas.py::IFASUnit.__post_init__": 110,
    "aquakin/plant/interfaces.py::ASM1toADM1.translate": 127,
    "aquakin/plant/plant.py::Plant.__init__": 128,
    "aquakin/plant/sensitivity.py::solve_sensitivity": 104,
    "aquakin/plant/sensitivity.py::steady_state_dgsm": 132,
    "aquakin/plant/steady_solve.py::steady_state": 183,
    "aquakin/utils/units.py::_infer": 127,
}
RECORDED_PARAMS = {
    "aquakin/core/ph_solver.py::_adaptive_activity_solve": 12,
    "aquakin/core/ph_solver.py::charge_balance_residual": 12,
    "aquakin/core/ph_solver.py::charge_balance_residual_deriv": 10,
    "aquakin/core/ph_solver.py::solve_ph": 16,
    "aquakin/integrate/_solver.py::_run_diffeqsolve": 16,
    "aquakin/integrate/_solver.py::solve_chemistry": 16,
    "aquakin/integrate/batch.py::BatchReactor.solve_sensitivity": 10,
    "aquakin/integrate/biofilm.py::BiofilmReactor.__init__": 24,
    "aquakin/integrate/biofilm.py::BiofilmReactor.solve_sensitivity": 10,
    "aquakin/integrate/calibrate.py::calibrate": 21,
    "aquakin/integrate/calibration/objective.py::_residual_parts": 9,
    "aquakin/integrate/calibration/problem.py::_resolve_datasets": 10,
    "aquakin/integrate/calibration/problem.py::_resolve_problem": 21,
    # #686: the owner chose `draws=` on predictive_band; its deprecated, ignored
    # `eig_keep` is the ninth. Delete this entry when `eig_keep` is removed.
    "aquakin/integrate/calibration/result.py::CalibrationResult.predictive_band": 9,
    "aquakin/integrate/candidate_design.py::_solve_relaxed": 11,
    "aquakin/integrate/candidate_design.py::design_from_candidates": 14,
    "aquakin/integrate/design.py::optimize_design": 11,
    "aquakin/integrate/discrete_adjoint.py::_discrete_adjoint_solve": 13,
    "aquakin/integrate/discrete_adjoint.py::esdirk_adjoint_solve": 18,
    "aquakin/integrate/discrete_adjoint.py::implicit_euler_adjoint_solve": 11,
    "aquakin/integrate/events.py::_drive": 9,
    "aquakin/integrate/events.py::solve_with_events": 18,
    "aquakin/integrate/experiment_design.py::optimize_experiment": 18,
    "aquakin/integrate/experiment_design.py::rank_experiments": 12,
    "aquakin/integrate/fisher.py::_FisherSetup.__init__": 9,
    "aquakin/integrate/fit.py::fit": 9,
    "aquakin/integrate/forward_sensitivity.py::augmented_forward_sensitivity": 17,
    "aquakin/integrate/forward_sensitivity.py::build_jitted_sensitivity_solve": 11,
    "aquakin/integrate/forward_sensitivity.py::run_forward_sensitivity": 18,
    "aquakin/integrate/forward_solve.py::_staggered_attempt": 9,
    "aquakin/integrate/forward_solve.py::forward_solve": 10,
    "aquakin/integrate/forward_solve.py::staggered_sensitivity_solve": 15,
    "aquakin/integrate/particle.py::integrate_ensemble": 9,
    "aquakin/integrate/pfr.py::PlugFlowReactor.__init__": 9,
    "aquakin/integrate/profile.py::profile_likelihood": 25,
    "aquakin/integrate/steady.py::arclength_continuation_solve": 20,
    "aquakin/integrate/steady.py::arclength_continuation_solve._land_on_target": 9,
    "aquakin/integrate/steady.py::continuation_solve": 11,
    "aquakin/integrate/steady.py::make_arclength_kernels.corrector": 9,
    "aquakin/integrate/steady.py::ptc_forward": 16,
    "aquakin/plant/bsm/bsm1.py::build_bsm1": 11,
    "aquakin/plant/bsm/bsm2.py::build_bsm2": 15,
    "aquakin/plant/bsm/evaluation.py::evaluate_bsm1": 9,
    "aquakin/plant/bsm/evaluation.py::evaluate_bsm2": 11,
    "aquakin/plant/calibrate.py::_resolve_plant_problem": 20,
    "aquakin/plant/calibrate.py::calibrate_plant": 26,
    "aquakin/plant/characterize.py::characterize_influent": 12,
    "aquakin/plant/characterize.py::fractionate": 9,
    "aquakin/plant/design.py::size_activated_sludge": 10,
    "aquakin/plant/dynamic_solve.py::_build_jitted_solve": 13,
    "aquakin/plant/dynamic_solve.py::_esdirk_stable_adjoint": 14,
    "aquakin/plant/dynamic_solve.py::_solve_forward_fast": 10,
    "aquakin/plant/dynamic_solve.py::_solve_jax_adjoint": 18,
    "aquakin/plant/dynamic_solve.py::_solve_stable_adjoint": 18,
    "aquakin/plant/dynamic_solve.py::_solve_with_events": 16,
    "aquakin/plant/dynamic_solve.py::solve": 13,
    "aquakin/plant/identifiability.py::_PlantDesign.__init__": 14,
    "aquakin/plant/identifiability.py::plant_fisher_information": 20,
    "aquakin/plant/identifiability.py::plant_profile_likelihood": 18,
    "aquakin/plant/identifiability.py::plant_rank_sensor_sets": 20,
    "aquakin/plant/sensitivity.py::_build_sensitivity_solve": 10,
    "aquakin/plant/sensitivity.py::_dynamic_value_jac": 11,
    "aquakin/plant/sensitivity.py::dynamic_dgsm": 12,
    "aquakin/plant/sensitivity.py::dynamic_sensitivity": 10,
    "aquakin/plant/sensitivity.py::solve_sensitivity": 16,
    "aquakin/plant/sensitivity.py::steady_state_dgsm": 14,
    "aquakin/plant/sensitivity.py::steady_state_sensitivity": 10,
    "aquakin/plant/steady_solve.py::_steady_continuation_fallback": 13,
    "aquakin/plant/steady_solve.py::_steady_fallback_ladder": 18,
    "aquakin/plant/steady_solve.py::_steady_forward_design": 13,
    "aquakin/plant/steady_solve.py::_steady_forward_result": 9,
    "aquakin/plant/steady_solve.py::run_to_steady_state": 12,
    "aquakin/plant/steady_solve.py::steady_state": 18,
    "aquakin_ui/results.py::xy_chart": 11,
}

ADVICE = (
    "Split it into named steps, or group the parameters that belong together into "
    "a config dataclass (the established pattern, as IntegratorConfig and "
    "OptimizerConfig; not by itself an owner decision). Do not get under the "
    "limit by removing a capability (dropping a parameter, hard-coding a value a "
    "caller could set, or packing unrelated arguments together); if neither fix "
    "fits, stop and ask (CLAUDE.md, 'Keep modules and functions small enough to "
    "read')."
)


def body_lines(fn):
    """Lines from the first to the last statement of ``fn``'s body, docstring excluded."""
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if not body:
        return 0
    return body[-1].end_lineno - body[0].lineno + 1


def param_count(fn):
    """Named parameters of ``fn``, without ``self`` / ``cls`` and the ``*`` collectors."""
    a = fn.args
    names = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return len(names)


def function_sizes(source, path):
    """
    ``{key: (body lines, parameters)}`` for every function in ``source``.

    Returns
    -------
    dict of str to tuple of (int, int)
    """
    sizes = {}

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                sizes[f"{path}::{name}"] = (body_lines(child), param_count(child))
                visit(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(ast.parse(source), "")
    return sizes


def package_sizes(root=ROOT):
    """Every function's size across the packages."""
    sizes = {}
    for package in PACKAGES:
        for file in sorted((root / package).rglob("*.py")):
            sizes.update(function_sizes(file.read_text(), file.relative_to(root).as_posix()))
    return sizes


def _check(key, value, limit, recorded, allowance, what):
    if key in recorded:
        if value <= limit:
            return [
                f"{key} is now within the {limit}-{what} limit ({value}): delete its "
                f"RECORDED entry in {Path(__file__).name}."
            ]
        if value > recorded[key] + allowance:
            return [
                f"{key} has {value} {what}s, over its recorded {recorded[key]}"
                f"{f' + {allowance}' if allowance else ''}. {ADVICE}"
            ]
        if value < recorded[key] - allowance:
            return [
                f"{key} has shrunk to {value} {what}s from its recorded {recorded[key]}: "
                f"set its RECORDED entry in {Path(__file__).name} to {value}."
            ]
        return []
    if value > limit:
        return [f"{key} has {value} {what}s, over the {limit}-{what} limit. {ADVICE}"]
    return []


def size_problems(sizes, recorded_lines, recorded_params):
    """Messages for functions over a limit or a recorded ceiling, and stale entries."""
    problems = []
    for key, (lines, params) in sorted(sizes.items()):
        problems += _check(key, lines, MAX_BODY_LINES, recorded_lines, LINE_ALLOWANCE, "line")
        problems += _check(key, params, MAX_PARAMS, recorded_params, 0, "parameter")
    for key in sorted((set(recorded_lines) | set(recorded_params)) - set(sizes)):
        problems.append(
            f"{key} is recorded but no longer exists: rename its entry if the "
            "function moved, or delete it."
        )
    return problems


# --- the check on this repository -------------------------------------------


def test_functions_stay_within_their_size_limits():
    problems = size_problems(package_sizes(), RECORDED_LINES, RECORDED_PARAMS)
    assert not problems, "\n".join(problems)


# --- the check itself ---------------------------------------------------------

SAMPLE = '''
def small(a, b):
    """Doc."""
    return a + b

class C:
    def method(self, a, b, c, d, e, f, g, h, i):
        def inner():
            return 1
        return inner()
'''


def test_function_sizes_key_methods_and_nested_functions():
    sizes = function_sizes(SAMPLE, "pkg/m.py")
    assert sizes == {
        "pkg/m.py::small": (1, 2),
        "pkg/m.py::C.method": (3, 9),
        "pkg/m.py::C.method.inner": (1, 0),
    }


def test_size_problems_ratchet():
    sizes = {
        "m::new_long": (MAX_BODY_LINES + 1, 1),
        "m::listed_ok": (150, 1),
        "m::listed_grew": (150 + LINE_ALLOWANCE + 1, 1),
        "m::listed_shrank": (120, 1),
        "m::listed_fixed": (50, 1),
        "m::many": (5, MAX_PARAMS + 2),
        "m::many_grew": (5, 12),
    }
    recorded_lines = {
        "m::listed_ok": 150,
        "m::listed_grew": 150,
        "m::listed_shrank": 150,
        "m::listed_fixed": 150,
        "m::gone": 200,
    }
    recorded_params = {"m::many_grew": 11}
    problems = size_problems(sizes, recorded_lines, recorded_params)
    assert [p.split(" ")[0] for p in problems] == [
        "m::listed_fixed",
        "m::listed_grew",
        "m::listed_shrank",
        "m::many",
        "m::many_grew",
        "m::new_long",
        "m::gone",
    ]
    assert "delete its RECORDED entry" in problems[0]
    assert f"over its recorded 150 + {LINE_ALLOWANCE}" in problems[1]
    assert "set its RECORDED entry in test_function_size.py to 120" in problems[2]
    assert f"over the {MAX_PARAMS}-parameter limit" in problems[3]
    assert "has 12 parameters, over its recorded 11." in problems[4]
    assert "no longer exists" in problems[6]
