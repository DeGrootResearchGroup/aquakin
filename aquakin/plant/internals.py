"""The private ``Plant`` members that modules other than ``plant.py`` may use.

``Plant`` keeps its layouts, caches and the pieces of its right-hand side
behind a leading underscore: they are not public API. The plant's helper
objects and analyses still need some of them. The recycle resolver evaluates
the RHS pieces, the coloured Jacobian reads the state layout, the solve cache
clears the plant's per-structure caches, and the audits, results and calibration
read layouts and endpoints. This module lists, for each module, which of those
private members it reaches on a plant object, so the contract between
``Plant`` and the rest of the package is a list someone can read rather than
attribute accesses scattered through it.

The modules bound onto ``Plant`` as methods (``dynamic_solve``,
``steady_solve``) are the class itself: their ``self.<name>`` uses are not
listed. A bound module that takes the plant under another name (``plant``)
is listed like any other module.

``tests/unit/test_plant_internals.py`` checks that every private ``Plant``
member used outside ``plant.py`` is listed here for the module that uses it,
and that every listed name is a ``Plant`` member still used by that module.
Renaming or changing one of these is a change to the modules listed with it.
"""

SHARED_INTERNALS: dict[str, frozenset[str]] = {
    # Audits: the mass balance and the units check walk units in order and
    # read their layouts and parameter slices; the balance's reaction integral
    # rebuilds each unit's RHS inputs and signals at the saved states.
    "plant/balance.py": frozenset(
        {
            "_build_parameter_layout",
            "_build_state_layout",
            "_coerce_params",
            "_params_for_unit",
            "_parse_endpoint",
            "_recycle",
            "_split_state",
            "_state_layout",
            "_unit_inputs_and_signals",
            "_unit_order",
        }
    ),
    "plant/units_check.py": frozenset(
        {"_declared_time_unit", "_insertion_order", "_ordered_models"}
    ),
    # The BSM evaluation rebuilds streams from a stored trajectory.
    "plant/bsm/evaluation.py": frozenset(
        {
            "_build_parameter_layout",
            "_cached_streams",
            "_params_for_unit",
            "_parse_endpoint",
            "_resolve_streams",
            "_split_state",
        }
    ),
    # Calibration and identifiability address units and endpoints by name.
    "plant/calibrate.py": frozenset(
        {
            "_build_state_layout",
            "_is_concentration_unit",
            "_ordered_models",
            "_state_layout",
            "_unit_or_raise",
        }
    ),
    # Sludge metrics read the reactors' model parameters for the BOD load.
    "plant/design.py": frozenset({"_params_for_unit"}),
    "plant/identifiability.py": frozenset({"_parse_endpoint"}),
    # The coloured Jacobian maps the sparsity pattern onto the state layout.
    "plant/colored.py": frozenset(
        {
            "_is_concentration_unit",
            "_recycle",
            "_split_state",
            "_state_layout",
            "_total_state_size",
        }
    ),
    # The recycle resolver evaluates the plant's RHS pieces to resolve tears.
    "plant/recycle.py": frozenset(
        {
            "_coerce_params",
            "_compute_signals",
            "_inputs_by_unit",
            "_params_for_unit",
            "_recycle_conns",
            "_recycle_keys",
            "_recycle_seeds",
            "_rhs",
            "_split_state",
            "_sweep_outputs",
            "_unit_order",
            "_warn_if_flow_nonaffine",
        }
    ),
    "plant/results.py": frozenset({"_is_concentration_unit", "_state_layout", "_unit_or_raise"}),
    # Sensitivity builds its own compiled solves around the plant's RHS.
    "plant/sensitivity.py": frozenset(
        {
            "_build_parameter_layout",
            "_build_state_layout",
            "_coerce_params",
            "_colored",
            "_operating_design",
            "_parse_operating",
            "_recycle",
            "_rhs",
            "_solve_cache",
            "_split_state",
        }
    ),
    # The solve cache resets the plant's per-structure caches and flags.
    "plant/solve_cache.py": frozenset(
        {
            "_aeration_materialized",
            "_colored",
            "_dosing_materialized",
            "_flow_affinity_checked",
            "_params_unit_cache",
            "_recycle",
            "_recycle_convergence_checked",
            "_solve_cache",
        }
    ),
    # A plant built from a spec shares its compiled solves under its fingerprint.
    "plant/spec.py": frozenset({"_insertion_order", "_solve_cache"}),
    "plant/temperature.py": frozenset({"_temperature_units", "_temperature_volumes"}),
    "study/study.py": frozenset({"_parse_endpoint"}),
}
