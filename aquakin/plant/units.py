"""Unit Protocol: the contract every plant component must satisfy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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

    state: jnp.ndarray | None = None
    t: jnp.ndarray | None = None


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

    Optional capability hooks (a unit opts in by implementing the method; the
    plant detects it with ``isinstance``). Each has a named
    ``runtime_checkable`` Protocol below so the contract is explicit:

    - :class:`SignalProducer` (``signal_outputs``) -- produces control signals for
      the shared signal bus (e.g. a PI controller).
    - :class:`PHOperating` (``operating_pH``) -- exposes its state-derived pH to a
      pH-coupled interface translator.
    - :class:`LiquidVolumeUnit` (``liquid_volume``) -- a state-dependent liquid
      volume, read by the results-level mass balance.
    - :class:`ComponentInventoryUnit` (``component_inventory``) -- owns its
      COD/N/P inventory for a non-concentration-vector state layout.
    - :class:`CycleEventSource` (``cycle_events``) -- schedules located phase
      events (an SBR's fill/react/settle/decant boundaries).
    - :class:`TemperatureSettable` (``set_temperature``) -- its operating
      temperature can be set by :meth:`Plant.set_temperature`.

    A few further optional hooks are read as *values with a default* rather than
    checked for presence, so they stay plain attribute look-ups (not Protocols):
    ``signal_names`` / ``required_signals`` (the signals a unit publishes /
    consumes), ``flow_param_defaults`` (a flow-parameterized unit's setpoint
    defaults), and, on a :class:`StateTranslator`, ``needs_src_pH`` /
    ``needs_dest_pH`` (a pH-feedback interface).

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
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict[str, Stream]: ...

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray: ...

    def flow_outputs(
        self,
        input_flows: dict[str, jnp.ndarray],
        params: jnp.ndarray,
        ctx: FlowContext,
    ) -> dict[str, jnp.ndarray]: ...


# --- Optional unit-capability protocols --------------------------------------
#
# Beyond the core Unit contract above, a unit may implement one or more of these
# OPTIONAL hooks. Each is a ``runtime_checkable`` Protocol so the plant detects
# the capability with ``isinstance(unit, SignalProducer)`` rather than
# ``hasattr(unit, "signal_outputs")`` -- giving the contract a name and a unit
# author (or reader) an importable, documented handle on it. A unit opts in
# simply by implementing the method (structural typing -- no base class, no
# registration). Note: these check method *presence* only, like ``hasattr`` --
# their value is the explicit, discoverable contract, not a signature guarantee.


@runtime_checkable
class SignalProducer(Protocol):
    """A unit that *produces* control signals for the shared signal bus (a PI
    controller). ``signal_outputs`` returns a ``{signal name: scalar}`` map the
    plant gathers into the bus each RHS call and threads into every unit's
    :meth:`Unit.rhs`. (A producer also lists the names it publishes via a
    ``signal_names`` property so consumers can be wired before the solve.)"""

    def signal_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]: ...


@runtime_checkable
class PHOperating(Protocol):
    """A unit exposing its state-derived operating pH, so a pH-coupled interface
    translator (the ASM->ADM digester feed) can read the pH the unit is at."""

    def operating_pH(self, state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray: ...


@runtime_checkable
class LiquidVolumeUnit(Protocol):
    """A unit whose liquid volume depends on its state (a variable-volume storage
    tank / MBR / SBR). The results-level mass balance weights ``C`` by this
    volume to get the unit's component inventory."""

    def liquid_volume(self, state: jnp.ndarray) -> jnp.ndarray: ...


@runtime_checkable
class ComponentInventoryUnit(Protocol):
    """A unit that owns its component (COD/N/P) inventory, for a state layout that
    is not a plain concentration vector (the layered Takacs settler, the ADM1
    digester with its gas headspace). Takes precedence over the generic
    ``volume * C`` inventory in the mass balance."""

    def component_inventory(
        self, state: jnp.ndarray, content: dict, params: jnp.ndarray
    ) -> dict: ...


@runtime_checkable
class CycleEventSource(Protocol):
    """A unit that schedules located phase-transition events over a time span (an
    SBR's fill / react / settle / decant boundaries); the plant merges them into
    the integrator's event set so it lands exactly on every phase switch."""

    def cycle_events(self, t0: float, t1: float) -> list: ...


@runtime_checkable
class TemperatureSettable(Protocol):
    """A unit whose operating temperature can be set (the reactors), so
    :meth:`Plant.set_temperature` can update every temperature-bearing unit in
    one call (a heated fixed-``T`` unit like the digester does not implement it)."""

    def set_temperature(self, temperature_K: float) -> None: ...


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
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        # No state -> no derivative.
        return jnp.zeros((0,))

    def coupling_pattern(self):
        """No state -> no structural Jacobian contribution (issue #388)."""
        import numpy as np

        from aquakin.plant.coupling import CouplingPattern

        return CouplingPattern(self_pattern=np.zeros((0, 0), dtype=bool), inlet_pattern=None)
