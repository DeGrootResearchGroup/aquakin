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

Two cases (Phases 2 and 3 of issue #291):

- **numeric** -- every other coefficient in the ``auto`` reaction is a numeric
  literal, so the solve is purely numeric and the resolved value is a constant
  baked into the stoichiometry matrix.
- **parameter-expression (yield-dependent)** -- a neighbour is a string
  expression in the parameters (e.g. ``-1/Y_H``). Conservation is *linear* in the
  coefficients, so each ``auto`` coefficient is a numeric-weighted linear
  combination of the known coefficient expressions; this module emits that as a
  derived parameter-expression string, which the normal stoichiometry-expression
  machinery compiles into a parameter-dependent (``stoich_dynamic``) coefficient.
  Calibrating a yield therefore flows through to the derived coefficient, and the
  reaction conserves for *every* parameter value -- not just the nominal one.

The composition is numeric (the declared ``composition:`` literals); a species
with no composition entry contributes zero content.
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
        resulting system is under-determined, or (numeric case) the declared
        balances cannot all be satisfied at once, or (symbolic case) the system is
        not square.
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

        known = [sp for sp in stoich if sp not in auto_species]
        # A parameter-expression neighbour makes the auto value parameter-dependent
        # (the symbolic path); all-numeric neighbours fold to a constant.
        symbolic = any(not isinstance(stoich[sp], (int, float)) for sp in known)

        def content(sp: str, q: str) -> float:
            return float(species_composition.get(sp, {}).get(q, 0.0))

        n_eq, n_unknown = len(conserved), len(auto_species)
        # M[q, a] is auto-species a's content of quantity q.
        M = np.array([[content(a, q) for a in auto_species] for q in conserved],
                     dtype=float)
        rank = int(np.linalg.matrix_rank(M)) if M.size else 0
        if rank < n_unknown:
            raise ValueError(
                f"reaction '{rxn.name}': cannot solve {n_unknown} 'auto' "
                f"coefficient(s) {auto_species} from the conserved quantity/"
                f"quantities {conserved} -- the system is under-determined "
                f"(rank {rank} < {n_unknown} unknowns). Each auto species must "
                f"carry content in enough conserved quantities; declare more "
                f"`conserved_for` quantities, or give the species a `composition:`.")

        if not symbolic:
            # Numeric: M x = b with b[q] = -(known content). lstsq tolerates an
            # over-determined-but-consistent system; the residual check rejects an
            # inconsistent one. The resolved values are constants.
            b = np.array([-sum(float(stoich[sp]) * content(sp, q) for sp in known)
                          for q in conserved], dtype=float)
            x, *_ = np.linalg.lstsq(M, b, rcond=None)
            resid = float(np.max(np.abs(M @ x - b))) if b.size else 0.0
            scale = 1.0 + (float(np.max(np.abs(b))) if b.size else 0.0)
            if resid > _RESID_TOL * scale:
                raise ValueError(
                    f"reaction '{rxn.name}': the 'auto' coefficient(s) "
                    f"{auto_species} cannot conserve all of {conserved} "
                    f"simultaneously -- the declared balances are inconsistent "
                    f"(residual {resid:.3g}). Drop a conserved quantity, or fix the "
                    f"known coefficients / composition.")
            for a, sp in enumerate(auto_species):
                stoich[sp] = float(x[a])
            continue

        # Symbolic: a known coefficient is a parameter expression, so the auto
        # value is parameter-dependent. Conservation is linear in the coefficients:
        #     x = M^-1 b,   b[q] = -sum_known coeff[sp] * content(sp, q),
        # so each auto coefficient is a numeric-weighted linear combination of the
        # known coefficient expressions, x[a] = sum_known w[a, sp] * coeff[sp] with
        # w[a, sp] = -sum_q Minv[a, q] * content(sp, q). We emit that as a derived
        # expression string for the normal stoich-expression (stoich_dynamic) path.
        # This needs a unique inverse, i.e. a square full-rank system (an
        # over-determined symbolic system's consistency would be parameter-
        # dependent and cannot be guaranteed at compile time).
        if n_eq != n_unknown:
            raise ValueError(
                f"reaction '{rxn.name}': resolving 'auto' coefficient(s) "
                f"{auto_species} alongside a parameter-expression coefficient needs "
                f"a square system (#auto == #conserved_for); got {n_unknown} auto "
                f"vs {n_eq} conserved quantity/quantities {conserved}. Add or remove "
                f"a `conserved_for` quantity, or make the coefficients numeric.")
        Minv = np.linalg.inv(M)

        def coeff_str(sp: str) -> str:
            c = stoich[sp]
            return repr(float(c)) if isinstance(c, (int, float)) else str(c)

        for a, sp in enumerate(auto_species):
            terms = []
            for ksp in known:
                w = -float(sum(Minv[a, q] * content(ksp, qn)
                               for q, qn in enumerate(conserved)))
                if abs(w) > _RESID_TOL:               # drop structural-zero weights
                    terms.append(f"({w!r}) * ({coeff_str(ksp)})")
            stoich[sp] = " + ".join(terms) if terms else "0.0"
