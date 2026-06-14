"""Build a state-derived pH field from a network speciation declaration.

A network may declare a ``speciation:`` block that maps state species onto the
acid/base totals consumed by :func:`aquakin.core.ph_solver.solve_ph`. This
module turns that (already-validated, plain-data) declaration into a *derived
condition function*

    derived(C, params, condition_arrays, loc_idx) -> dict[str, scalar]

which the runtime evaluates once per RHS call and merges into
``condition_arrays`` before the rate callables run. The produced field (``pH``
by default) is then visible to ordinary ``{pH}`` / ``pH_switch(...)`` rate
expressions exactly as if it had been supplied externally — except it now
tracks the instantaneous state and is differentiable through ``solve_ph``.

This module lives in ``core`` and has no Pydantic dependency: it consumes plain
dicts/floats and the network's ``species_index``.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from aquakin.core.ph_solver import solve_ph

# Total acid/base systems understood by the pH solver, in solver-argument terms.
# Single source of truth: the Pydantic schema (schema/network_spec.py) imports
# this so the load-time validator and the runtime builder agree on the valid
# total keys.
VALID_TOTAL_KEYS = (
    "carbonate",
    "acetate",
    "propionate",
    "butyrate",
    "valerate",
    "ammonia",
    "phosphate",
    "sulfide",
)


def build_ph_derived_fn(
    config: dict,
    species_index: dict[str, int],
) -> tuple[Callable, str, set[str]]:
    """Compile a speciation declaration into a derived-condition callable.

    Parameters
    ----------
    config : dict
        Plain-data speciation declaration with keys:

        ``field`` : str
            Name of the produced condition field (default ``"pH"``).
        ``temperature_field`` : str
            Condition field carrying temperature.
        ``temperature_units`` : ``"celsius"`` or ``"kelvin"``
            How to interpret ``temperature_field`` (default ``"celsius"``).
        ``z_cation_eq`` : float or ``{"condition": name}``
            Net fixed cation-charge *offset* (eq/L), literal or read from a
            condition. Added to the ``strong_cations`` sum (default 0.0).
        ``n_iter`` : int
            Newton iteration count (default 40).
        ``totals`` : dict
            Map from total key (``carbonate``/``acetate``/``ammonia``/
            ``phosphate``/``sulfide``) to ``{"species": name,
            "molar_mass": g/mol-or-gCOD}``. The molar total is
            ``max(C[species], 0) / molar_mass``.
        ``strong_anions`` : list
            Each ``{"species": name, "molar_mass": .., "charge": ..}``;
            contributes ``charge * max(C, 0) / molar_mass`` eq/L of anionic
            charge.
        ``strong_cations`` : list
            Same form as ``strong_anions``; contributes
            ``charge * max(C, 0) / molar_mass`` eq/L of cationic charge (summed
            into the net cation charge). Lets a state species drive pH, e.g.
            ADM1's ``S_cat`` / ``S_an`` dynamic ion states.
    species_index : dict[str, int]
        Map from species name to index in ``C``.

    Returns
    -------
    (callable, produced_field, required_condition_fields)
        The derived-condition function, the name of the field it produces, and
        the set of condition fields it reads (so the network can require them).
    """
    field = config.get("field", "pH")
    temp_field = config["temperature_field"]
    temp_units = config.get("temperature_units", "celsius")
    n_iter = int(config.get("n_iter", 40))
    activity_model = config.get("activity_model", "none")

    if temp_units not in ("celsius", "kelvin"):
        raise ValueError(
            f"speciation temperature_units must be 'celsius' or 'kelvin', "
            f"got {temp_units!r}"
        )

    def _species_idx(name: str) -> int:
        if name not in species_index:
            raise KeyError(
                f"speciation references undeclared species {name!r}. "
                f"Declared: {sorted(species_index)}"
            )
        return species_index[name]

    # Resolve total-system species -> (solver_kw, index, molar_mass).
    totals_cfg = config.get("totals", {})
    unknown = set(totals_cfg) - set(VALID_TOTAL_KEYS)
    if unknown:
        raise ValueError(
            f"speciation 'totals' has unknown systems {sorted(unknown)}; "
            f"valid keys are {VALID_TOTAL_KEYS}"
        )
    total_terms: list[tuple[str, int, float]] = []
    for key, entry in totals_cfg.items():
        total_terms.append(
            (f"tot_{key}", _species_idx(entry["species"]), float(entry["molar_mass"]))
        )

    # Strong anions -> (index, molar_mass, charge).
    strong_terms: list[tuple[int, float, float]] = []
    for entry in config.get("strong_anions", []):
        strong_terms.append(
            (
                _species_idx(entry["species"]),
                float(entry["molar_mass"]),
                float(entry["charge"]),
            )
        )

    # Strong cations -> (index, molar_mass, charge). These are summed into the
    # net cation charge alongside the literal/condition offset below, so a
    # state cation species (e.g. ADM1's S_cat) drives pH dynamically.
    cation_terms: list[tuple[int, float, float]] = []
    for entry in config.get("strong_cations", []):
        cation_terms.append(
            (
                _species_idx(entry["species"]),
                float(entry["molar_mass"]),
                float(entry["charge"]),
            )
        )

    # Net fixed cation charge offset: literal or read from a condition field.
    # The total cation charge is this offset plus the strong_cations sum.
    z_spec = config.get("z_cation_eq", 0.0)
    z_condition: str | None = None
    z_literal = 0.0
    if isinstance(z_spec, dict):
        z_condition = z_spec["condition"]
    else:
        z_literal = float(z_spec)

    required_fields = {temp_field}
    if z_condition is not None:
        required_fields.add(z_condition)

    def derived(C, params, condition_arrays, loc_idx) -> dict:
        T = condition_arrays[temp_field][loc_idx]
        T_kelvin = T + 273.15 if temp_units == "celsius" else T

        kwargs = {k: 0.0 for k in (f"tot_{key}" for key in VALID_TOTAL_KEYS)}
        for solver_kw, idx, mm in total_terms:
            kwargs[solver_kw] = jnp.maximum(C[idx], 0.0) / mm

        strong_anion_eq = jnp.asarray(0.0)
        for idx, mm, charge in strong_terms:
            strong_anion_eq = strong_anion_eq + charge * jnp.maximum(C[idx], 0.0) / mm

        if z_condition is not None:
            z_offset = condition_arrays[z_condition][loc_idx]
        else:
            z_offset = jnp.asarray(z_literal)
        z_cation_eq = z_offset
        for idx, mm, charge in cation_terms:
            z_cation_eq = z_cation_eq + charge * jnp.maximum(C[idx], 0.0) / mm

        extra = {}
        if activity_model != "none":
            # The pH-independent strong-ion ionic strength 1/2 sum c*z^2. Only the
            # speciation layer knows each strong ion's charge (the solver receives
            # them lumped into the charge sums), so it is computed here. Seed from
            # the fixed cation-charge OFFSET (taken monovalent, z=1) -- NOT the
            # post-fold ``z_cation_eq``, which already contains the explicit strong
            # cations: using it here would count each strong cation twice (once
            # mis-weighted in the lump, once in its own z^2 term below).
            I_strong = 0.5 * jnp.abs(z_offset)
            for idx, mm, charge in strong_terms:
                I_strong = I_strong + 0.5 * charge * charge * jnp.maximum(C[idx], 0.0) / mm
            for idx, mm, charge in cation_terms:
                I_strong = I_strong + 0.5 * charge * charge * jnp.maximum(C[idx], 0.0) / mm
            extra = dict(activity_model=activity_model,
                         ionic_strength_strong=I_strong)

        pH = solve_ph(
            strong_anion_eq=strong_anion_eq,
            z_cation_eq=z_cation_eq,
            T_kelvin=T_kelvin,
            n_iter=n_iter,
            **kwargs,
            **extra,
        )
        return {field: pH}

    return derived, field, required_fields
