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

# ASM1 particulates that settle in a secondary clarifier (includes XND).
ASM1_SETTLING_SPECIES: tuple[str, ...] = ("XS", "XI", "XB_H", "XB_A", "XP", "XND")

# ASM1 particulates that contribute to TSS (excludes XND), and the COD->TSS
# conversion factor (Copp 2002).
ASM1_TSS_SPECIES: tuple[str, ...] = ("XS", "XI", "XB_H", "XB_A", "XP")
ASM1_TSS_FACTOR: float = 0.75


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


def grams_per_m3_factors(model: CompiledModel, names, *, what: str = "species") -> list:
    """Per-species factors turning each named species' concentration, in the
    unit ``model`` declares for it, into ``g/m3``.

    A unit's built-in solids constants (a thickener's target underflow TSS, the
    Takács settling parameters) are per ``g/m3``; multiplying a species' TSS
    weight by its factor puts the TSS it contributes on that basis whatever the
    model declares (``kgCOD/m3`` gives 1000). A chemical currency in the model's
    unit is ignored (``g/m3`` names none). A blank or unreadable unit is taken
    to be ``g/m3`` already (factor 1).

    Raises
    ------
    ValueError
        If a species is declared in a unit that is not a mass per volume (an
        amount in moles, say), which no scale factor can turn into ``g/m3``.
    """
    from aquakin.utils.units import conversion_factor

    factors = []
    for name in names:
        unit = model.species_units.get(name, "")
        try:
            factor = conversion_factor(unit, "g/m3")
        except ValueError:
            raise ValueError(
                f"{what} '{name}' is declared in '{unit}' by model '{model.name}', "
                f"which cannot be expressed in g/m3 (a mass per volume) without a "
                f"molar mass or composition."
            ) from None
        factors.append(1.0 if factor is None else float(factor))
    return factors


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
