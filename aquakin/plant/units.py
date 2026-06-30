"""Unit Protocol: the contract every plant component must satisfy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import jax.numpy as jnp

from aquakin.plant.coupling import CouplingAware

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.plant.streams import Stream


@dataclass(frozen=True)
class FlowContext:
    """Side information passed to :meth:`Unit.flow_outputs`.

    The recycle-flow solve evaluates each unit's linear flow rule with the
    unit's own internal ``state`` and the current time ``t`` held fixed (only
    the recycle back-edge flows vary), so the map stays affine. A unit whose
    flow split depends on its state (a variable-volume storage tank) or on the
    time (a scheduled pump) reads it from here; units whose split depends on
    neither simply ignore the context. Carrying both in one object keeps
    :meth:`Unit.flow_outputs` a single fixed signature for every unit.

    Attributes
    ----------
    state : jnp.ndarray, optional
        The unit's own internal state vector, or ``None`` when the flow solve
        is run without states.
    t : jnp.ndarray, optional
        The current time.
    """

    state: Optional[jnp.ndarray] = None
    t: Optional[jnp.ndarray] = None


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
      input streams, and the control-signal bus, return the output streams.
      Called by the plant in topological order on every RHS evaluation. It
      receives the same ``signals`` bus threaded into :meth:`rhs`, so a unit
      whose *output* stream depends on a control signal (e.g. a feedback-dosing
      unit) can read it; an uncontrolled unit ignores it.
    - :meth:`rhs` — given the current ``t``, internal ``state``, input streams,
      and the control-signal bus, return ``dstate/dt`` of shape
      ``(state_size,)``. Called by the plant on every RHS evaluation after all
      output streams are known.
    - :meth:`flow_outputs` — the unit's *linear* flow rule (output-port flows
      as a function of input-port flows), used by the plant's exact recycle-flow
      solve. Receives a :class:`FlowContext` so a state- or time-dependent
      split has one fixed signature.

    Every method receives the same fixed arguments for every unit -- the plant
    never branches its call on a per-unit capability flag. A unit ignores the
    arguments it does not use (``signals`` for an uncontrolled unit, the
    :class:`FlowContext` for a fixed-split unit).

    ``compute_outputs``, ``rhs`` and ``flow_outputs`` must be AD-clean (no
    Python branching on traced values, no concretisation of ``t`` / ``state``).

    Optional producer hook (duck-typed; only some units implement it):

    - ``signal_outputs(t, state, inputs, params) -> dict[str, jnp.ndarray]``:
      a unit that *produces* control signals (e.g. a PI controller) returns a
      mapping of signal name to scalar. The plant evaluates these each RHS call,
      gathers them into a shared signal bus, and threads that bus into every
      unit's :meth:`rhs` as ``signals``. A unit that produces no signals simply
      does not define this method.

    See :mod:`aquakin.plant.control` and ``Plant._rhs``.
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
        signals: Optional[dict] = None,
    ) -> dict[str, "Stream"]: ...

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, "Stream"],
        params: jnp.ndarray,
        signals: Optional[dict] = None,
    ) -> jnp.ndarray: ...

    def flow_outputs(
        self,
        input_flows: dict[str, jnp.ndarray],
        params: jnp.ndarray,
        ctx: FlowContext,
    ) -> dict[str, jnp.ndarray]: ...


class StatelessUnit(CouplingAware):
    """Mixin for units that own no ODE state (``state_size == 0``).

    A stateless unit transforms streams instantaneously -- a mixer, a splitter,
    an ideal separator -- so its only real work is :meth:`compute_outputs` and
    :meth:`flow_outputs`. This mixin supplies the three otherwise-identical state
    members (a zero state size, an empty initial state, and a no-op ``rhs``), so
    such a unit only writes the parts that actually differ and "stateless" is a
    named concept rather than three look-alike method bodies.

    It is a plain mixin, not part of the :class:`Unit` Protocol, so it composes
    with the ``@dataclass`` units: inherit it and the dataclass fields and the
    domain methods stay on the subclass. A unit author writing a new stateless
    unit inherits it the same way.
    """

    @property
    def state_size(self) -> int:
        return 0

    def initial_state(self) -> jnp.ndarray:
        return jnp.zeros((0,))

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, "Stream"],
        params: jnp.ndarray,
        signals: Optional[dict] = None,
    ) -> jnp.ndarray:
        # No state -> no derivative.
        return jnp.zeros((0,))

    def coupling_pattern(self):
        """No state -> no structural Jacobian contribution (issue #388)."""
        import numpy as np

        from aquakin.plant.coupling import CouplingPattern

        return CouplingPattern(self_pattern=np.zeros((0, 0), dtype=bool), inlet_pattern=None)
