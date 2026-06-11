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
    T : jnp.ndarray, optional
        Stream temperature (scalar, Kelvin). Carried algebraically through the
        flowsheet: mixers flow-weight it (a heat balance) and pass-through units
        propagate it unchanged, so a reactor can read its inlet temperature and
        feed it to temperature-dependent kinetics. ``None`` (the default) means
        the stream is temperature-agnostic; reactors then fall back to their
        static condition, so existing plants are unaffected. ``None``-ness is a
        static structural property (consistent across RHS calls), so it is
        jit-safe.
    """

    Q: jnp.ndarray
    C: jnp.ndarray
    network: "CompiledNetwork"
    T: "jnp.ndarray | None" = None

    def mass_flow(self) -> jnp.ndarray:
        """Per-species mass flow rate ``Q * C``, shape ``(n_species,)``."""
        return self.Q * self.C

    def with_C(self, C: jnp.ndarray) -> "Stream":
        """Return a new stream with the same Q/T/network but a new C vector."""
        return Stream(Q=self.Q, C=C, network=self.network, T=self.T)

    def with_Q(self, Q: jnp.ndarray) -> "Stream":
        """Return a new stream with the same C/T/network but a new flow rate."""
        return Stream(Q=Q, C=self.C, network=self.network, T=self.T)

    def with_T(self, T: "jnp.ndarray | None") -> "Stream":
        """Return a new stream with the same Q/C/network but a new temperature."""
        return Stream(Q=self.Q, C=self.C, network=self.network, T=T)


@dataclass(frozen=True)
class StreamSeries:
    """A stream's flow and concentration trajectory over time.

    Returned by :meth:`Plant.stream`, which reconstructs a named output stream
    (e.g. the clarifier effluent) from a solution's saved states -- the plant
    integrates unit *states*, not the inter-unit streams, so the effluent is
    recomputed after the fact.

    Attributes
    ----------
    t : jnp.ndarray
        Save times, shape ``(n_t,)``.
    Q : jnp.ndarray
        Volumetric flow rate at each time, shape ``(n_t,)``.
    C : jnp.ndarray
        Concentration over time, shape ``(n_t, n_species)`` in the network's
        species ordering.
    network : CompiledNetwork
        The kinetic network whose species ordering applies to ``C``.
    """

    t: jnp.ndarray
    Q: jnp.ndarray
    C: jnp.ndarray
    network: "CompiledNetwork"

    def C_named(self, species: str) -> jnp.ndarray:
        """Concentration trajectory of one species, shape ``(n_t,)``."""
        return self.C[:, self.network.species_index[species]]
