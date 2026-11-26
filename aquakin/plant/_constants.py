"""Shared ASM1 particulate / TSS constants and species-mask helpers for the
plant units.

Single source of truth for the ASM1 particulate-species sets that the
clarifiers (settling) and the effluent metrics (TSS) would otherwise each
hardcode, plus the :func:`species_indices` / :func:`species_mask` helpers the
separator/clarifier family shares to resolve a species-name list against a
model. The two species sets are deliberately different:

- :data:`ASM1_SETTLING_SPECIES` are the particulates that *settle* in a
  secondary clarifier, including ``XND`` (organic nitrogen attached to the
  settling ``XS``).
- :data:`ASM1_TSS_SPECIES` are the particulates that contribute to *total
  suspended solids*. ``XND`` is excluded: it is nitrogen carried on ``XS``,
  not a separate solid (Copp 2002, ``TSS = 0.75 * (XS + XI + XBH + XBA + XP)``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from aquakin.core.model import CompiledModel

# Shared near-zero-denominator guard for the plant units' flow-/mass-weighted
# concentration divisions -- mass/flow and load/flow mixing, the capture split,
# a TSS-fraction denominator, and the derived flow/solids/volume ratios. One
# magnitude everywhere so a near-zero total flow yields a finite (~0)
# concentration instead of 0/0, rather than each unit picking its own epsilon.
# This is *only* the numerical divide-by-zero floor; a physically meaningful
# small-quantity floor (a minimum settler depth, a minimum flow that sets a
# maximum HRT, a log-domain floor) is a modelling choice, keeps its own value,
# and is deliberately not folded in here.
EPS_Q: float = 1e-12

# Time-unit conversion factors, shared by the aeration, disinfection, clarifier
# and metrics units so each conversion is defined once rather than re-hardcoded.
SECONDS_PER_DAY: float = 86400.0
HOURS_PER_DAY: float = 24.0
MINUTES_PER_DAY: float = 1440.0

# Pressure: the headspace digester's settings are in bar, the gas library in Pa.
PA_PER_BAR: float = 1.0e5

# Molar mass of methane (g/mol), as BSM2 rounds it in its biogas evaluation.
CH4_G_PER_MOL: float = 16.0

# ASM1 particulates that settle in a secondary clarifier (includes XND).
ASM1_SETTLING_SPECIES: tuple[str, ...] = ("XS", "XI", "XB_H", "XB_A", "XP", "XND")

# ASM1 particulates that contribute to TSS (excludes XND), and the COD->TSS
# conversion factor (Copp 2002).
ASM1_TSS_SPECIES: tuple[str, ...] = ("XS", "XI", "XB_H", "XB_A", "XP")
ASM1_TSS_FACTOR: float = 0.75


#: Names a model gives the state that carries its total suspended solids
#: (``XTSS`` in ASM2d, ``XSS`` in ASM3), which the reactions change through the
#: model's own particulate-to-TSS factors.
LUMPED_TSS_SPECIES: tuple[str, ...] = ("XTSS", "XSS")


def lumped_tss_weights(model: CompiledModel, params=None) -> dict[str, float]:
    """TSS per unit of each particulate of a model that tracks its TSS as a state.

    ASM2d and ASM3 carry the suspended solids as a state (``XTSS`` / ``XSS``):
    each reaction changes it by the TSS of the particulates it makes or
    consumes, through factors such as ``iTSS_XI`` (g TSS/g COD) or ``iTSS_XPP``
    (g TSS/g P). The weights returned are the ones under which the reactions
    change the TSS state by exactly the weighted change of the other
    particulates, solved from the stoichiometry, so they follow the model
    however its parameters are named. A particulate declared in the TSS state's
    own unit (a metal precipitate in g TSS/m³) is suspended solids already and
    weighs 1. The TSS state itself weighs 0, so a weighted sum over every
    particulate counts each solid once.

    The particulates are the species whose name starts with ``X``, as in
    :func:`solids_species`.

    Parameters
    ----------
    model : CompiledModel
        A model with a ``XTSS`` or ``XSS`` state.
    params : jnp.ndarray, optional
        Parameter vector whose stoichiometry is read; the model defaults if
        omitted.

    Returns
    -------
    dict of str to float
        Weight per particulate, in model order, the TSS state included at 0.

    Raises
    ------
    ValueError
        If the model has no TSS state, if its stoichiometry does not determine
        every weight (a particulate that changes in no reaction, or two that
        only ever change together), or if no weights reproduce the TSS
        state's stoichiometry.
    """
    import numpy as np

    from aquakin.utils.units import units_match

    tss = next((s for s in LUMPED_TSS_SPECIES if s in model.species_index), None)
    if tss is None:
        raise ValueError(
            f"model '{model.name}' has no TSS state ({', '.join(LUMPED_TSS_SPECIES)}); "
            "give the settler's TSS factors explicitly."
        )
    tss_unit = model.species_units.get(tss, "")
    parts = [s for s in model.species if s.startswith("X") and s != tss]
    fixed = [s for s in parts if units_match(model.species_units.get(s, ""), tss_unit) is True]
    free = [s for s in parts if s not in fixed]
    S = np.asarray(model.compute_stoich(model.default_parameters() if params is None else params))
    col = model.species_index
    A = S[:, [col[s] for s in free]]
    b = S[:, col[tss]] - S[:, [col[s] for s in fixed]].sum(axis=1)
    w, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < len(free):
        raise ValueError(
            f"model '{model.name}': the stoichiometry does not determine the TSS weight of "
            f"every particulate in {free}; give the settler's TSS factors explicitly."
        )
    if np.max(np.abs(A @ w - b), initial=0.0) > 1e-9 * max(np.max(np.abs(b), initial=0.0), 1.0):
        raise ValueError(
            f"model '{model.name}': no particulate TSS weights reproduce the stoichiometry "
            f"of {tss}; give the settler's TSS factors explicitly."
        )
    weights = dict(zip(free, map(float, w), strict=True)) | dict.fromkeys(fixed, 1.0)
    weights[tss] = 0.0
    return {s: weights[s] for s in model.species if s in weights}


def with_lumped_tss(model: CompiledModel, composition: dict[str, float]) -> dict[str, float]:
    """A composition with its TSS state set from its particulates.

    The TSS state (``XTSS`` / ``XSS``) is the :func:`lumped_tss_weights`-weighted
    sum of the composition's particulates, the value the model's reactions
    keep it at, unless ``composition`` gives it.

    Parameters
    ----------
    model : CompiledModel
        A model with a TSS state.
    composition : dict of str to float
        Concentrations by species name, in the model's units (unlisted species
        are 0).

    Returns
    -------
    dict of str to float
        A copy of ``composition`` with the TSS state added.
    """
    weights = lumped_tss_weights(model)
    tss = next(s for s in LUMPED_TSS_SPECIES if s in weights)
    out = dict(composition)
    out.setdefault(tss, sum(w * composition.get(s, 0.0) for s, w in weights.items()))
    return out


def solids_species(model: CompiledModel) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The ``(settling_species, tss_species)`` of an activated-sludge model.

    A model whose particulates (``X``-prefixed species) are exactly the ASM1
    settling species gets the ASM1 sets unchanged (:data:`ASM1_SETTLING_SPECIES`,
    :data:`ASM1_TSS_SPECIES`). Any other model (an ASM1 variant that splits the
    autotrophs into AOB and NOB, or one that adds a particulate to ASM1) is read
    by the ASM naming convention: every species whose name starts with ``X`` is
    a particulate and settles, and the particulates carrying COD are the TSS
    species. COD content comes from the model's composition table; without one,
    from the declared units: a particulate in a COD mass per volume counts as
    TSS, one in other units (moles, g P, g TSS) does not, and one whose unit
    cannot be read counts unless it is ``XND`` (nitrogen carried on ``XS``).

    Parameters
    ----------
    model : CompiledModel
        The activated-sludge model.

    Returns
    -------
    tuple of (tuple of str, tuple of str)
        The settling species and the TSS species, in model order for a non-ASM1
        model.

    Raises
    ------
    ValueError
        If the model has no species starting with ``X``.
    """
    settling = tuple(s for s in model.species if s.startswith("X"))
    if set(settling) == set(ASM1_SETTLING_SPECIES):
        return ASM1_SETTLING_SPECIES, ASM1_TSS_SPECIES
    if not settling:
        raise ValueError(
            f"model '{model.name}' has no particulate (X-prefixed) species; pass the "
            "settling and TSS species explicitly."
        )
    # Declared species composition, else the shipped table, else {}.
    comp = model.composition()
    if comp:
        tss = tuple(s for s in settling if comp.get(s, {}).get("COD", 0.0) > 0.0)
    else:
        tss = tuple(s for s in settling if _counts_as_cod_solid(s, model.species_units.get(s)))
    return settling, tss


def _counts_as_cod_solid(species: str, units: str | None) -> bool:
    """Whether a particulate without composition data counts towards TSS.

    The TSS factor converts COD to suspended solids, so only a particulate
    declared as a COD mass per volume (``g_COD/m3``, ``kgCOD/m3``) counts; one
    declared in other units -- moles of a precipitate, g P of stored
    polyphosphate, g TSS of a lumped solids state -- does not. A particulate
    whose unit cannot be read falls back to the naming convention: every one
    but ``XND`` (the nitrogen carried on ``XS``).
    """
    from aquakin.utils.units import _to_base_units, parse_units

    dim = parse_units(units or "")
    if dim is None:
        return species != "XND"
    tokens = dim.as_dict()
    if tokens.get("COD") != 1:
        return False
    base, _ = _to_base_units(dim)
    return base == _to_base_units(parse_units("g_COD/m3"))[0]


def species_indices(model: CompiledModel, names, *, what: str = "species") -> list[int]:
    """Resolve species ``names`` to their model indices, raising on any missing.

    The **single, decided** missing-species policy for the separator/clarifier
    family: a name in ``names`` that is not a species of ``model`` raises a
    :class:`ValueError` naming the offending species. This deliberately replaces
    the two contradictory behaviours these units used to have -- some *silently
    dropped* an unknown species (which under-settles / under-counts TSS without
    warning), others raised a bare ``KeyError``. A species you asked a unit to
    settle but the model does not define is almost always a misconfiguration, so
    it fails loudly and early, at construction, with a clear message.

    Parameters
    ----------
    model : CompiledModel
        The model whose ``species_index`` the names are resolved against.
    names : iterable of str
        Species names (e.g. a unit's ``settling_species`` / ``particulate_species``
        / ``tss_species``).
    what : str, optional
        A label for the names in the error message (e.g. ``"settling species"``).

    Returns
    -------
    list of int
        The flat model index of each name, in the order given.

    Raises
    ------
    ValueError
        If any name is not a species of ``model``.
    """
    idx = model.species_index
    missing = [s for s in names if s not in idx]
    if missing:
        raise ValueError(f"{what} {missing} not in model species {sorted(idx)}")
    return [idx[s] for s in names]


def species_mask(
    model: CompiledModel, names, *, weight: float = 1.0, what: str = "species"
) -> jnp.ndarray:
    """An ``(n_species,)`` array that is ``weight`` at each named species, 0 else.

    The shared particulate/settling/TSS-contribution mask the separator and
    clarifier units build. ``weight`` is the scalar value placed at every named
    species -- ``1.0`` for a 0/1 settling mask, the COD->TSS conversion factor for
    a TSS-contribution vector. Raises on any name not in ``model`` (the single
    missing-species policy; see :func:`species_indices`).
    """
    indices = species_indices(model, names, what=what)
    mask = jnp.zeros((model.n_species,))
    if indices:
        mask = mask.at[jnp.asarray(indices, dtype=int)].set(weight)
    return mask


def grams_per_m3_factors(model: CompiledModel, names) -> tuple[list, list]:
    """Per-species factors turning each named species' concentration, in the
    unit ``model`` declares for it, into ``g/m3``.

    A unit's built-in solids constants (a thickener's target underflow TSS, the
    Takács settling parameters) are per ``g/m3``; multiplying a species' TSS
    weight by its factor puts the TSS it contributes on that basis whatever the
    model declares (``kgCOD/m3`` gives 1000). A chemical currency in the model's
    unit is ignored (``g/m3`` names none). A blank or unreadable unit is taken
    to be ``g/m3`` already (factor 1).

    Returns
    -------
    factors : list of float
        One factor per name.
    unconverted : list of str
        Names whose unit is not a mass per volume (an amount in moles, say),
        which no scale factor can turn into ``g/m3``. Their factor is 1, so the
        unit uses them as if they were in ``g/m3``; it should report them as
        :meth:`~aquakin.plant.units.AssumedUnits.assumed_units` so
        :meth:`Plant.check_units` flags the mismatch.
    """
    from aquakin.utils.units import conversion_factor

    factors, unconverted = [], []
    for name in names:
        try:
            factor = conversion_factor(model.species_units.get(name, ""), "g/m3")
        except ValueError:
            factor = None
            unconverted.append(name)
        factors.append(1.0 if factor is None else float(factor))
    return factors, unconverted


def tss_concentration(C: jnp.ndarray, tss_vec: jnp.ndarray) -> jnp.ndarray:
    """Total suspended solids of a composition ``C`` given a TSS-weight vector.

    ``tss_vec`` is an ``(n_species,)`` vector carrying each species' COD->TSS
    contribution factor (0 for non-solids) -- build it with
    :func:`species_mask(model, tss_species, weight=tss_factor) <species_mask>`.
    The TSS is the weighted sum ``sum(tss_vec * C)`` over the last axis, so it
    vectorises over any leading (time / layer / batch) axes of ``C``. The single
    full-vector TSS form shared by the ideal separators and the effluent metrics.
    (The layered Takacs settler keeps a gather over its particulate indices in its
    per-layer hot loop -- the same weighted sum, restricted to the nonzero terms.)
    """
    return jnp.sum(tss_vec * C, axis=-1)


def check_positive(unit: str, name: str, value) -> None:
    """Refuse a size that is not a positive finite number (a volume, say).

    A zero or negative volume builds a unit whose mass balance divides by it,
    and a steady-state solve of such a plant does not end.

    Raises
    ------
    ValueError
        Naming the unit and the setting.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{unit}: {name} must be a positive number; got {value!r}") from None
    if not (v > 0.0 and v != float("inf")):
        raise ValueError(f"{unit}: {name} must be a positive number; got {value!r}")
