"""Model introspection, audit, and presentation.

Free functions that render or audit a :class:`~aquakin.core.model.CompiledModel`
— the human-readable summary, the LaTeX rate expressions, the dimensional
(``check_units``) and conservation (``check_conservation`` / ``check_nitrogen``)
audits, and the per-species composition table.

These are kept **out of the runtime dataclass** so ``CompiledModel`` stays
focused on the differentiable hot path (state → rates → dCdt); they change for
unrelated reasons (presentation, dimensional analysis, elemental balances) and
almost all delegate to a lower-level ``utils.*`` implementer. ``CompiledModel``
exposes them as thin delegating methods (``net.summary()``, ``net.check_units()``,
…), which remain the public API.

The ``utils.*`` imports are kept **lazy** (inside each function) exactly as they
were on the dataclass: ``utils`` modules import from ``core``, so importing them
at module top would form a cycle, and the whole point of the boundary is to keep
``utils`` out of the core import graph until an advisory function actually runs.
``CompiledModel`` in turn lazy-imports *this* module inside its delegators, so
``import aquakin`` never pulls it eagerly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel

# Long-name for the human-readable time unit, used in the summary header.
_TIME_UNIT_NAMES = {"s": "seconds", "d": "days", "h": "hours", "min": "minutes"}


def format_model_summary(model: "CompiledModel") -> str:
    """Render a human-readable table summarising ``model``.

    The body of :meth:`CompiledModel.summary`; see it for the public contract.
    """
    tu = model.time_unit
    if tu is not None:
        time_line = (
            f"  Time unit: {tu} (t_span / t_eval are in {_TIME_UNIT_NAMES.get(tu, tu)})"
        )
    else:
        time_line = "  Time unit: (could not infer from rate-constant units)"
    lines = [
        f"Model: {model.name}",
        f"  Description: {model.description}",
        time_line,
        f"  Species ({model.n_species}):",
    ]
    name_w = max((len(s) for s in model.species), default=0)
    unit_w = max((len(model.species_units.get(s, "")) for s in model.species), default=0)
    for s in model.species:
        units = model.species_units.get(s, "")
        desc = model.species_descriptions.get(s, "")
        line = f"    {s:<{name_w}}  [{units:<{unit_w}}]"
        if desc:
            line += f"  {desc}"
        lines.append(line)
    lines += [
        f"  Conditions required: {', '.join(model.conditions_required) or '(none)'}",
        f"  Reactions ({model.n_reactions}):",
    ]
    # Render against the stoichiometry evaluated at the default parameters, not
    # ``model.stoich_matrix`` (the static base, which holds zeros at every
    # parameter-dependent cell). Models with symbolic coefficients
    # (ASM1/ASM2d/ASM3/ADM1 yields, N-content, fractions) would otherwise have
    # those species silently dropped from the summary.
    stoich = model.compute_stoich(model._default_parameters)
    for i, rname in enumerate(model.reaction_names):
        stoich_terms = []
        for j, sp in enumerate(model.species):
            coef = float(stoich[i, j])
            if coef == 0:
                continue
            sign = "+" if coef > 0 else "-"
            mag = abs(coef)
            term = f"{sign} {mag:g} {sp}" if mag != 1 else f"{sign} {sp}"
            stoich_terms.append(term)
        lines.append(f"    [{i}] {rname}: " + " ".join(stoich_terms).lstrip("+ "))
    lines.append(f"  Parameters ({model.n_params}):")
    for p in model.parameters:
        bounds = model.parameter_bounds.get(p)
        bounds_s = f" bounds={bounds}" if bounds is not None else ""
        val = float(model._default_parameters[model.param_index[p]])
        lines.append(f"    {p} = {val:g}{bounds_s}")
    if model.references:
        lines.append("  References:")
        for ref in model.references:
            lines.append(f"    - {ref}")
    return "\n".join(lines)


def model_to_latex(model: "CompiledModel") -> "dict[str, str]":
    """``{reaction_name: LaTeX rate expression}`` (body of ``to_latex``)."""
    from aquakin.utils.latex import to_latex as _to_latex

    return {
        name: _to_latex(ast)
        for name, ast in zip(model.reaction_names, model.rate_asts, strict=True)
    }


def check_units(model: "CompiledModel", *, check_root: bool = True) -> list:
    """Dimensional-consistency audit of the rate expressions (body of
    ``check_units``)."""
    from aquakin.utils.units import check_model_units

    return check_model_units(model, check_root=check_root)


def model_composition(
    model: "CompiledModel", *, params=None, electron_acceptor_cod: bool = True
) -> "dict[str, dict[str, float]]":
    """Per-species conserved-quantity content table (body of ``composition``).

    Declared ``species[].composition`` metadata first, else the shipped
    role-based table, else empty.
    """
    if model.species_composition:
        return {sp: dict(c) for sp, c in model.species_composition.items()}
    from aquakin.utils.composition import composition_table

    try:
        return composition_table(model, electron_acceptor_cod=electron_acceptor_cod, params=params)
    except KeyError:
        return {}


def check_conservation(
    model: "CompiledModel",
    *,
    tol: float = 1e-2,
    params=None,
    quantities=None,
    composition=None,
    electron_acceptor_cod: bool = True,
) -> list:
    """Conservation violations above ``tol`` (body of ``check_conservation``)."""
    comp = (
        composition
        if composition is not None
        else model_composition(model, params=params, electron_acceptor_cod=electron_acceptor_cod)
    )
    if not comp:
        raise ValueError(
            f"model '{model.name}' has no composition metadata to check "
            f"against: declare a `composition:` per species in the YAML, or "
            f"pass an explicit composition=..."
        )
    from aquakin.utils.balance import check_conservation as _check

    return _check(model, comp, tol=tol, params=params, quantities=quantities)


def check_nitrogen(
    model: "CompiledModel",
    *,
    tol: float = 1e-2,
    params=None,
    composition=None,
    nitrate: str = "S_NO",
    n_key: str = "N",
) -> list:
    """Nitrogen-balance violations above ``tol`` (body of ``check_nitrogen``)."""
    comp = composition if composition is not None else model_composition(model, params=params)
    if not comp:
        raise ValueError(
            f"model '{model.name}' has no composition metadata to check "
            f"against: declare a `composition:` per species in the YAML, or "
            f"pass an explicit composition=..."
        )
    from aquakin.utils.balance import check_nitrogen as _check

    return _check(model, comp, tol=tol, params=params, nitrate=nitrate, n_key=n_key)
