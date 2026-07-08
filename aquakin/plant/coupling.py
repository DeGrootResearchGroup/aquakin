"""Structural coupling contract for plant units.

The plant's colored-Jacobian sparsity pattern must include every coupling the
RHS *can* express, for any influent -- otherwise a stiff coupling that switches
on only off the warm-start operating point (a Monod substrate entering its
limiting range, the Takacs settling velocity changing regime, an ASM<->ADM
interface branch flipping) is missing from the pattern and the chord-Newton
convergence collapses (a ~6x slowdown; see issue #388).

A numerical probe at one operating point cannot see those couplings (they are
numerically zero there but structurally present). So instead each component
**emits its own structural sparsity** from its equations:

- a reactor's kinetics from the rate AST (:func:`structural_sparsity_pattern`),
- a settler's settling law by AD over diverse physical states,
- a cross-model translator from its emitted ``coupling_pattern()``.

This module defines the contract (:class:`CouplingAware`, :class:`CouplingPattern`)
and a helper (:func:`ad_union`) for the AD-derived case. The plant assembles the
per-unit patterns into the full plant pattern (diagonal blocks from each unit's
``self_pattern``; off-diagonal blocks from each unit's ``inlet_pattern`` composed
with the translator couplings along the stream path that feeds it).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CouplingPattern:
    """Structural sparsity of a unit's RHS, as two boolean blocks.

    Attributes
    ----------
    self_pattern : np.ndarray
        ``(state_size, state_size)`` boolean -- which of the unit's own state
        variables each state-derivative ``d(state_i)/dt`` depends on. The
        diagonal block of the plant Jacobian.
    inlet_pattern : np.ndarray or None
        ``(state_size, n_inlet_species)`` boolean -- which inlet concentrations
        (in the unit's model species ordering) each state-derivative depends
        on. ``None`` for a unit with no concentration inlet (it then contributes
        no off-diagonal coupling). The plant composes this with the translator
        coupling(s) on the stream path feeding the unit to form the off-diagonal
        blocks.
    """

    self_pattern: np.ndarray
    inlet_pattern: np.ndarray | None = None


class CouplingAware(ABC):
    """Contract for a plant unit that emits its structural Jacobian sparsity.

    Implemented by every stateful unit so the plant can build the
    colored-Jacobian pattern from the equations (robust to any influent regime)
    rather than from a single-operating-point probe. Stateless units inherit a
    trivial empty pattern via :class:`aquakin.plant.units.StatelessUnit`.
    """

    @abstractmethod
    def coupling_pattern(self) -> CouplingPattern:
        """Return this unit's :class:`CouplingPattern`."""
        raise NotImplementedError


def ad_union(
    jac_fn, base, *, n_states: int = 64, decades: float = 2.0, seed: int = 0
) -> np.ndarray:
    """Boolean union of ``|jac_fn(x)| > 0`` over diverse positive inputs.

    Each component of ``base`` is independently scaled by
    ``10 ** U[-decades, decades]`` for ``n_states`` samples, so a smooth
    settling/transport nonlinearity has every branch exercised and the union is
    a structural superset.

    This is the right tool for a unit whose nonlinearity is a settling or
    transport law (the Takacs velocity), **not** for reaction kinetics: a Monod
    term ``S/(K+S)`` saturated at ``S >> K`` has a sensitivity ``~K/S^2`` that is
    numerically invisible at every probed state, so kinetics need the *syntactic*
    dependency from the rate AST instead (:func:`structural_sparsity_pattern`).

    Parameters
    ----------
    jac_fn : Callable
        ``x -> J`` returning a 2-D array (the Jacobian of some RHS block w.r.t.
        the perturbed input ``x``).
    base : array-like
        Reference input vector; sets the per-component probe scale.
    n_states, decades, seed : int, float, int
        Sample count, multiplicative spread (decades), and RNG seed.

    Returns
    -------
    np.ndarray
        Boolean array, the shape of one ``jac_fn`` output, true where any sample
        had a nonzero entry.
    """
    import jax
    import jax.numpy as jnp

    base = np.asarray(base, dtype=float)
    base = np.where(np.abs(base) > 0.0, np.abs(base), 1e-3)
    rng = np.random.default_rng(seed)
    fj = jax.jit(jac_fn)
    P = None
    for _ in range(n_states):
        x = jnp.asarray(base * 10.0 ** rng.uniform(-decades, decades, size=base.shape))
        Pi = np.abs(np.asarray(fj(x))) > 0.0
        P = Pi if P is None else (P | Pi)
    return P
