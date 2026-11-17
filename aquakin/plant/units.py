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
    - :class:`LiquidVolumeUnit` (``liquid_volume``) -- a single well-mixed liquid
      volume whose state leads with its concentration vector, read by the
      results-level mass balance and the per-species readers.
    - :class:`ComponentInventoryUnit` (``component_inventory``) -- owns its
      COD/N/P inventory for a non-concentration-vector state layout.
    - :class:`CycleEventSource` (``cycle_events``) -- schedules located phase
      events (an SBR's fill/react/settle/decant boundaries).
    - :class:`TemperatureSettable` (``set_temperature``) -- its operating
      temperature can be set by :meth:`Plant.set_temperature`.
    - :class:`AssumedUnits` (``assumed_units``) -- declares the concentration
      basis its internal constants assume for some species, checked by
      :meth:`Plant.check_units`.
    - :class:`PlantTimeUnitAware` (``set_time_unit``) -- told the plant's time
      unit before a solve, so it can convert rates or built-in constants that
      are in another time unit (a kinetic model's rates, a UV reactor's
      seconds). :class:`ModelTimeScaling` implements it for kinetic units.

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
    """A unit holding a single well-mixed liquid volume, with state
    ``[C..., one-or-more scalars]``: a variable-volume storage tank / MBR / SBR,
    or the primary clarifier with its filtered-flow state. The results-level
    mass balance weights ``C`` by this volume to get the unit's component
    inventory, and the plant reads the concentration head by species."""

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
        """No state -> no structural Jacobian contribution."""
        import numpy as np

        from aquakin.plant.coupling import CouplingPattern

        return CouplingPattern(self_pattern=np.zeros((0, 0), dtype=bool), inlet_pattern=None)


@runtime_checkable
class AssumedUnits(Protocol):
    """A unit whose internal constants fix the units of some of its inputs, so
    :meth:`Plant.check_units` can confirm the plant supplies them.

    ``assumed_units`` returns ``{species: unit string}``: the concentration
    basis a built-in constant assumes, e.g. a settler whose settling parameters
    are in ``g/m3`` of solids. A species unit with no chemical currency
    (``"g/m3"``) constrains the mass and volume scale only. Only *fixed*
    assumptions belong here: a parameter the user supplies in any consistent
    unit (a volume, a flow setpoint, a ``kLa``) is not listed. A built-in time
    unit is converted instead (:class:`PlantTimeUnitAware`).
    """

    def assumed_units(self) -> dict[str, str]: ...


@runtime_checkable
class PlantTimeUnitAware(Protocol):
    """A unit with rates or built-in constants in a time unit of its own.

    The plant calls ``set_time_unit`` with its time unit (or ``None`` when it
    has none) before building a solve, and the unit converts what it needs to:
    a kinetic unit scales its model's reaction term, a UV reactor turns the
    residence time into seconds from that unit. Every parameter the user
    supplies to a unit (a ``kLa``, a filter time constant, a phase length) is
    already in the plant's time unit and is not converted.
    """

    def set_time_unit(self, time_unit: str | None) -> None: ...


class ModelTimeScaling:
    """Mixin for units whose ``rhs`` evaluates a kinetic model's rates.

    A model's rates are per its own time unit (:attr:`CompiledModel.time_unit`);
    the plant integrates in its own. :meth:`set_time_unit` stores the factor
    between the two and :meth:`_scale_rates` applies it to the reaction term
    (and anything else the model computes per its time unit, such as an ADM1
    digester's gas transfer). The factor is ``1.0`` when the units agree or
    either is unknown, and :meth:`_scale_rates` then returns its argument
    untouched, so a plant on its models' own time unit is unchanged.
    """

    _rate_factor: float = 1.0

    def set_time_unit(self, time_unit: str | None) -> None:
        """Set the factor from the model's time unit to the plant's."""
        self._rate_factor = model_rate_factor(self.model, time_unit)

    def _scale_rates(self, term: jnp.ndarray) -> jnp.ndarray:
        """``term`` (per the model's time unit) per the plant's time unit."""
        if self._rate_factor == 1.0:
            return term
        return term * self._rate_factor


def model_rate_factor(model, time_unit: str | None) -> float:
    """Factor turning ``model``'s rates (per its time unit) into rates per
    ``time_unit``; ``1.0`` when either is unknown or they agree."""
    from aquakin.utils.units import conversion_factor

    native = getattr(model, "time_unit", None)
    if native is None or time_unit is None or native == time_unit:
        return 1.0
    factor = conversion_factor(f"1/{native}", f"1/{time_unit}")
    return 1.0 if factor is None else float(factor)
