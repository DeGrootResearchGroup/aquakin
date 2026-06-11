"""Unit Protocol: the contract every plant component must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.plant.streams import Stream


@runtime_checkable
class Unit(Protocol):
    """A plant unit operation.

    Units expose:

    - ``name``: a string identifier used for connections.
    - ``state_size``: the number of ODE state variables this unit owns
      (zero for stateless units like mixers / splitters). Exposed as a
      read-only ``@property`` on every shipped unit -- a constant ``0`` on
      stateless ones, derived from the unit's config on stateful ones.
    - ``input_ports`` / ``output_ports``: named stream ports.

    And implement:

    - :meth:`initial_state` — the ``(state_size,)`` initial state vector.
    - :meth:`compute_outputs` — given the current ``t``, internal ``state``,
      and input streams, return the output streams. Called by the plant in
      topological order on every RHS evaluation.
    - :meth:`rhs` — given the current ``t``, internal ``state``, and input
      streams, return ``dstate/dt`` of shape ``(state_size,)``. Called by
      the plant on every RHS evaluation after all output streams are known.

    Both ``compute_outputs`` and ``rhs`` must be AD-clean (no Python
    branching on traced values, no concretisation of ``t`` / ``state``).
    """

    name: str
    state_size: int
    input_ports: list[str]
    output_ports: list[str]

    def initial_state(self) -> jnp.ndarray: ...

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, "Stream"],
        params: jnp.ndarray,
    ) -> dict[str, "Stream"]: ...

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, "Stream"],
        params: jnp.ndarray,
    ) -> jnp.ndarray: ...
