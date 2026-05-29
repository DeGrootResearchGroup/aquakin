"""Streams: the data passed between units in a plant flowsheet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


@dataclass(frozen=True)
class Stream:
    """A flow stream — bulk volumetric rate plus a concentration vector.

    Streams are produced by a unit's ``compute_outputs`` and consumed by
    downstream units' ``rhs`` / ``compute_outputs`` calls. They are
    intentionally immutable per evaluation: a connection delivers the
    upstream output directly, with optional :class:`StateTranslator`
    interposed for cross-network mappings.

    Attributes
    ----------
    Q : jnp.ndarray
        Volumetric flow rate (scalar), units must be consistent across the
        plant (typically m³/d for BSM-family plants).
    C : jnp.ndarray
        Concentration vector, shape ``(n_species,)`` where species ordering
        is ``network.species``.
    network : CompiledNetwork
        The kinetic network whose species ordering applies to ``C``.
    """

    Q: jnp.ndarray
    C: jnp.ndarray
    network: "CompiledNetwork"

    def mass_flow(self) -> jnp.ndarray:
        """Per-species mass flow rate ``Q * C``, shape ``(n_species,)``."""
        return self.Q * self.C

    def with_C(self, C: jnp.ndarray) -> "Stream":
        """Return a new stream with the same Q/network but a new C vector."""
        return Stream(Q=self.Q, C=C, network=self.network)

    def with_Q(self, Q: jnp.ndarray) -> "Stream":
        """Return a new stream with the same C/network but a new flow rate."""
        return Stream(Q=Q, C=self.C, network=self.network)
