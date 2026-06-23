"""Conservation-derived stoichiometric coefficients (``auto`` / ``?``).

A coefficient written ``auto`` (or ``?``) in a reaction's stoichiometry is left
*unknown* and **solved from the network's declared conservation laws**, so it
cannot be written wrong -- the failure mode that has caused almost every
stoichiometry bug in this codebase (a hand-typed electron-acceptor demand, an
elemental-sulfur reduction donor, a product split). For each conserved quantity
``q`` the reaction conserves (its ``conserved_for``, or the network default), the
stoichiometry-weighted species content must sum to zero::

    sum_species  stoich[species] * composition[species][q]  ==  0

The known coefficients move to the right-hand side and the ``auto`` coefficients
are the unknowns of one small linear system, solved once at compile time. The
content comes from the per-species ``composition:`` metadata (a species with no
composition entry contributes zero content).

This module covers the **numeric** case (Phase 2 of issue #291): every known
coefficient in an ``auto`` reaction is a numeric literal, so the solve is purely
numeric and the resolved value is a constant baked into the stoichiometry matrix.
A parameter-expression neighbour would make the ``auto`` value parameter-dependent
(a later phase) and is rejected with a clear error.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Sentinels: a coefficient equal to one of these (after stripping) is solved from
# the conservation laws rather than read as a number or a parameter expression.
_AUTO_TOKENS = frozenset({"auto", "?"})

# Solve / consistency tolerance (absolute, scaled by the rhs magnitude).
_RESID_TOL = 1.0e-9


def is_auto(coef: Any) -> bool:
    """True if a stoichiometric coefficient is the ``auto`` / ``?`` sentinel."""
    return isinstance(coef, str) and coef.strip() in _AUTO_TOKENS


def resolve_auto_coefficients(
    reactions, species_composition: dict, network_conserved_for
) -> None:
    """Replace every ``auto`` / ``?`` coefficient with its conservation-derived
    numeric value, **in place** on each reaction's ``stoichiometry``.

    Parameters
    ----------
    reactions : list
        The validated reaction specs (each with ``name``, ``stoichiometry`` and
        an optional per-reaction ``conserved_for``). Mutated in place.
    species_composition : dict
        ``{species: {quantity: content}}`` -- the per-species conserved-quantity
        content (the declared ``composition:`` metadata).
    network_conserved_for : list[str] or None
        The network-level default list of conserved quantities, used for any
        reaction that does not declare its own ``conserved_for``.

    Raises
    ------
    ValueError
        If an ``auto`` reaction declares nothing to conserve it from, or the
        resulting system is under-determined, or the declared balances cannot all
        be satisfied at once.
    NotImplementedError
        If a known coefficient in an ``auto`` reaction is a parameter expression
        (which would make the ``auto`` value parameter-dependent -- a later phase).
    """
    for rxn in reactions:
        stoich = rxn.stoichiometry
        auto_species = [sp for sp, c in stoich.items() if is_auto(c)]
        if not auto_species:
            continue

        conserved = list(getattr(rxn, "conserved_for", None)
                          or network_conserved_for or [])
        if not conserved:
            raise ValueError(
                f"reaction '{rxn.name}' has an 'auto' stoichiometric coefficient "
                f"({auto_species}) but declares no quantities to conserve it from: "
                f"add a per-reaction `conserved_for: [COD, ...]` or a network-level "
                f"`conserved_for:`.")

        # Phase 2 is the numeric case: a parameter-expression neighbour would make
        # the auto value parameter-dependent (resolved via stoich_dynamic -- a
        # later phase of issue #291).
        for sp, c in stoich.items():
            if sp in auto_species or isinstance(c, (int, float)):
                continue
            raise NotImplementedError(
                f"reaction '{rxn.name}': resolving an 'auto' coefficient when "
                f"another coefficient ('{sp}' = {c!r}) is a parameter expression "
                f"is not yet supported (it would make the auto value "
                f"parameter-dependent). Use numeric coefficients alongside 'auto'.")

        def content(sp: str, q: str) -> float:
            return float(species_composition.get(sp, {}).get(q, 0.0))

        n_eq, n_unknown = len(conserved), len(auto_species)
        # M x = b : M[q, a] is auto-species a's content of quantity q; b[q] is the
        # negated content the known coefficients already contribute.
        M = np.array([[content(a, q) for a in auto_species] for q in conserved],
                     dtype=float)
        b = np.array(
            [-sum(float(stoich[sp]) * content(sp, q)
                  for sp in stoich if sp not in auto_species)
             for q in conserved], dtype=float)

        rank = int(np.linalg.matrix_rank(M)) if M.size else 0
        if rank < n_unknown:
            raise ValueError(
                f"reaction '{rxn.name}': cannot solve {n_unknown} 'auto' "
                f"coefficient(s) {auto_species} from the conserved quantity/"
                f"quantities {conserved} -- the system is under-determined "
                f"(rank {rank} < {n_unknown} unknowns). Each auto species must "
                f"carry content in enough conserved quantities; declare more "
                f"`conserved_for` quantities, or give the species a `composition:`.")

        x, *_ = np.linalg.lstsq(M, b, rcond=None)
        resid = float(np.max(np.abs(M @ x - b))) if b.size else 0.0
        if resid > _RESID_TOL * (1.0 + float(np.max(np.abs(b))) if b.size else 1.0):
            raise ValueError(
                f"reaction '{rxn.name}': the 'auto' coefficient(s) {auto_species} "
                f"cannot conserve all of {conserved} simultaneously -- the declared "
                f"balances are inconsistent (residual {resid:.3g}). Drop a conserved "
                f"quantity, or fix the known coefficients / composition.")

        for a, sp in enumerate(auto_species):
            stoich[sp] = float(x[a])
