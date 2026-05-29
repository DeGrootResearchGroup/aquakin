"""State translators: convert a stream from one kinetic network to another.

For single-network plants like BSM1, every translator is the
:class:`IdentityTranslator`. The interface exists so that BSM2-style
plants — which use ASM1 in the activated-sludge tanks and ADM1 in the
anaerobic digester — can plug an ASM↔ADM mapping into the framework
without touching plant assembly code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


@runtime_checkable
class StateTranslator(Protocol):
    """Maps a concentration vector from one network's species ordering to
    another's. Must be AD-clean (used inside the plant RHS).

    Attributes
    ----------
    source_network : CompiledNetwork
        The kinetic network whose species ordering the input concentration
        vector follows.
    target_network : CompiledNetwork
        The kinetic network the output concentration vector is expressed
        in.
    """

    source_network: "CompiledNetwork"
    target_network: "CompiledNetwork"

    def translate(self, C_source: jnp.ndarray) -> jnp.ndarray: ...


class IdentityTranslator:
    """Pass-through translator for when source and target networks are the
    same — the only kind of translator BSM1 needs.

    The plant inserts one of these automatically on any connection whose
    source and target units share a network reference, so users don't
    normally need to instantiate it directly.
    """

    def __init__(self, network: "CompiledNetwork") -> None:
        self.source_network = network
        self.target_network = network

    def translate(self, C_source: jnp.ndarray) -> jnp.ndarray:
        return C_source
