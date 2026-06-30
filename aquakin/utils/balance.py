"""Mass and electron (COD) conservation checks for reaction networks.

A reaction conserves a quantity ``Q`` when the stoichiometry-weighted sum of the
per-species content of ``Q`` is zero::

    sum_species  stoich[reaction, species] * composition[species][Q]  ==  0

This catches stoichiometry errors that are otherwise easy to miss: a wrong
electron-acceptor coefficient (e.g. the O2 demand of sulfide oxidation, or the
nitrate demand of a nitrate-driven oxidation) breaks the **COD / electron**
balance, while a wrong product split breaks an **elemental** balance (S, N, P,
Fe).

``composition`` maps each species to a dict of its content per unit of the
species' own measure -- e.g. ``{"COD": 1.0}`` for an organic (g COD per g COD),
``{"COD": -1.0}`` for dissolved oxygen (oxygen is an electron acceptor, i.e.
negative COD), ``{"COD": -2.86, "N": 1.0}`` for nitrate-N (2.86 g COD accepted
per g N reduced to N2), ``{"COD": 2.0, "S": 1.0}`` for sulfide (2 g COD per g S),
and so on. Species absent from a network are ignored, so one composition table
can serve a family of related networks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork

# species name -> {quantity name -> content per unit of the species' measure}
Composition = dict[str, dict[str, float]]


def _composition_matrix(
    network: "CompiledNetwork", composition: Composition
) -> tuple[list[str], np.ndarray]:
    """(n_quantities, n_species) content matrix and the quantity-name list."""
    quantities = sorted({q for c in composition.values() for q in c})
    sidx = network.species_index
    mat = np.zeros((len(quantities), len(sidx)))
    for j, q in enumerate(quantities):
        for sp, content in composition.items():
            if sp in sidx:
                mat[j, sidx[sp]] = content.get(q, 0.0)
    return quantities, mat


def conservation_residuals(
    network: "CompiledNetwork",
    composition: Composition,
    params: Optional[Any] = None,
) -> tuple[list[str], list[str], np.ndarray]:
    """Per-reaction conservation residual for every quantity in ``composition``.

    Parameters
    ----------
    network : CompiledNetwork
        The compiled reaction network.
    composition : dict
        ``{species_name: {quantity: content_per_unit}}``.
    params : array-like, optional
        Parameter vector (string-expression stoichiometry is evaluated with it).
        Defaults to ``network.default_parameters()``.

    Returns
    -------
    reaction_names : list[str]
    quantities : list[str]
    residuals : np.ndarray
        Shape ``(n_reactions, n_quantities)``; entry ``[i, j]`` is the residual
        of quantity ``j`` in reaction ``i`` (zero means conserved).
    """
    p = network.default_parameters() if params is None else params
    stoich = np.asarray(network.compute_stoich(p))  # (n_rxn, n_species)
    quantities, comp_mat = _composition_matrix(network, composition)
    residuals = stoich @ comp_mat.T  # (n_rxn, n_quantities)
    return list(network.reaction_names), quantities, residuals


def nitrogen_residuals(
    network: "CompiledNetwork",
    composition: Composition,
    *,
    params: Optional[Any] = None,
    nitrate: str = "S_NO",
    n_key: str = "N",
) -> tuple[list[str], np.ndarray]:
    """Per-reaction nitrogen residual, accounting for denitrification N2 gas.

    Nitrogen oxidation/reduction is outside the COD continuity, so it needs its
    own balance. In WATS/ASM models nitrate is purely an electron acceptor that
    is reduced to N2 -- an untracked gas product -- so the nitrogen that leaves
    as N2 equals the nitrate consumed. A reaction therefore conserves nitrogen
    when its tracked-species N residual plus that gassed-off nitrate is zero::

        residual = (stoich . N_content)  +  max(0, -stoich[nitrate])

    This is exact for both nitrification (no nitrate consumed -> the tracked
    residual must itself be zero, verifying NH3 -> NO3) and denitrification (the
    consumed nitrate is added back as the N2 that left).

    Returns ``(reaction_names, residuals)``.
    """
    p = network.default_parameters() if params is None else params
    stoich = np.asarray(network.compute_stoich(p))
    sidx = network.species_index
    n_content = np.zeros(stoich.shape[1])
    for sp, content in composition.items():
        if sp in sidx:
            n_content[sidx[sp]] = content.get(n_key, 0.0)
    residual = stoich @ n_content
    if nitrate in sidx:
        residual = residual + np.maximum(0.0, -stoich[:, sidx[nitrate]])
    return list(network.reaction_names), residual


def check_nitrogen(
    network: "CompiledNetwork",
    composition: Composition,
    *,
    tol: float = 1e-2,
    **kwargs: Any,
) -> list[tuple[str, float]]:
    """Return nitrogen-balance violations ``(reaction, residual)`` above ``tol``."""
    names, residual = nitrogen_residuals(network, composition, **kwargs)
    return [(names[i], float(residual[i])) for i in range(len(names)) if abs(residual[i]) > tol]


def check_conservation(
    network: "CompiledNetwork",
    composition: Composition,
    *,
    tol: float = 1e-2,
    params: Optional[Any] = None,
    quantities: Optional[list[str]] = None,
) -> list[tuple[str, str, float]]:
    """Return the list of conservation violations ``(reaction, quantity, residual)``.

    A reaction/quantity pair is reported when ``abs(residual) > tol``. The
    default ``tol`` (1e-2) tolerates the two-decimal rounding of published
    stoichiometric coefficients while still flagging genuine imbalances (which
    are typically order 0.1-1). Restrict to specific ``quantities`` if desired.
    """
    names, all_q, res = conservation_residuals(network, composition, params)
    keep = set(quantities) if quantities is not None else set(all_q)
    out = []
    for i, rxn in enumerate(names):
        for j, q in enumerate(all_q):
            if q in keep and abs(res[i, j]) > tol:
                out.append((rxn, q, float(res[i, j])))
    return out
