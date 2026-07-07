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

# ASM1 particulates that settle in a secondary clarifier (includes XND).
ASM1_SETTLING_SPECIES: tuple[str, ...] = ("XS", "XI", "XB_H", "XB_A", "XP", "XND")

# ASM1 particulates that contribute to TSS (excludes XND), and the COD->TSS
# conversion factor (Copp 2002).
ASM1_TSS_SPECIES: tuple[str, ...] = ("XS", "XI", "XB_H", "XB_A", "XP")
ASM1_TSS_FACTOR: float = 0.75


def species_indices(model: "CompiledModel", names, *, what: str = "species") -> list[int]:
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
    model: "CompiledModel", names, *, weight: float = 1.0, what: str = "species"
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
