"""Chemical dosing: inject a reagent (metal salt, acid/base, external carbon)
into a stream at a fixed or feedback-controlled flow.

A :class:`DosingUnit` is an inline unit -- one stream in, the dosed stream out --
that adds a :class:`Reagent` (a fixed composition) at a dose flow. The flow is
either a constant setpoint or driven by the plant's control-signal bus: a
feedback dose declares a sensed reactor + measured species and a setpoint, and
the plant auto-wires a PI controller (the same one the aeration loop uses) that
manipulates the dose flow to hold the setpoint.

The dose only adds the reagent's *mass* to the stream. The reactive response --
the pH shift of an acid/base, metal-phosphate precipitation, the added COD's
oxygen demand -- is the downstream reactor's chemistry, not this unit's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant._constants import EPS_Q
from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


# eq=False: a Reagent holds a jnp array (unhashable by value), so use identity
# equality/hash -- honestly hashable, unlike the field-based __hash__ a frozen
# dataclass with eq=True would synthesize (which raises on the array at hash time).
@dataclass(frozen=True, eq=False)
class Reagent:
    """The neat composition of a dosing reagent: a concentration vector in the
    dosed stream's model, plus a label.

    Build it by name from the species the reagent actually contains -- a methanol
    carbon source is just readily-biodegradable COD, a ferric solution is just the
    metal -- everything else is zero::

        Reagent.from_species(asm1, SS=400000.0, label="methanol")
        Reagent.from_species(asm2d, S_Fe=130000.0, label="ferric chloride")

    Attributes
    ----------
    model : CompiledModel
        The model whose species ordering ``composition`` follows (the dosed
        stream's model).
    composition : jnp.ndarray
        Concentration vector of the neat reagent, shape ``(n_species,)``.
    label : str
        A human-readable name for the reagent (used in unit names / messages).
    """

    model: CompiledModel
    composition: jnp.ndarray
    label: str = "reagent"

    @classmethod
    def from_species(cls, model, overrides=None, /, *, label="reagent", **species) -> Reagent:
        """Build a reagent from named species concentrations (everything else 0).

        Thin wrapper over :meth:`CompiledModel.concentrations` with
        ``base="zero"`` -- the neat reagent contains only the species you name.

        Parameters
        ----------
        model : CompiledModel
            The model whose species indexing defines the composition vector.
        overrides : dict[str, float], optional
            Species name -> concentration, positional-only. Use it for names
            that are not valid Python identifiers (``"Br-"``, ``"NH4+"``); the
            keyword form below covers identifier-safe names.
        label : str, optional
            Human-readable reagent label (default ``"reagent"``).
        **species : float
            Convenience concentration overrides for identifier-safe species
            names (``SS=4e5``).

        Returns
        -------
        Reagent
            A reagent whose composition contains only the named species (every
            other species is 0).
        """
        comp = model.concentrations(overrides, base="zero", **species)
        return cls(model=model, composition=comp, label=label)


def dose_signal_name(controller_id: str) -> str:
    """The control-signal name a dosing controller publishes / a unit consumes."""
    return f"_dose_{controller_id}_flow"


@dataclass
class DosingUnit:
    """Inline chemical-dosing unit: ``in`` stream -> ``out`` = inlet + reagent dose.

    The dose adds ``Q_dose`` of the :class:`Reagent` to the through-stream
    (flow-mixing the compositions). Choose exactly one flow mode:

    **Fixed** -- a constant dose flow::

        DosingUnit("carbon", Reagent.from_species(asm1, SS=4e5), flow=2.0)

    **Feedback** -- a dissolved-target setpoint; the plant auto-wires a PI
    controller that senses ``measured_species`` in ``sensor`` and manipulates the
    dose flow to hold ``setpoint`` (reusing the aeration loop's controller and the
    signal bus)::

        DosingUnit("carbon", reagent, setpoint=1.0, measured_species="SNO",
                   sensor="anoxic", flow_max=5.0)

    The unit is stateless: a fixed dose needs no state, and a feedback dose's PI
    integral lives in the auto-wired controller. The controller senses the
    sensor's reactor state (so the dose-flow signal is available during the stream
    sweep, where this unit's output is computed); the sensor must therefore be a
    reactor whose output concentration is its state (a :class:`CSTRUnit`).

    Parameters
    ----------
    name : str
        Unit identifier.
    reagent : Reagent
        The dosed reagent. Its ``model`` is the unit's model.
    flow : float, optional
        Fixed dose flow (volume/time). Mutually exclusive with ``setpoint``.
    setpoint : float, optional
        Feedback target value for ``measured_species`` at ``sensor``. Mutually
        exclusive with ``flow``.
    measured_species : str, optional
        Species the feedback controller senses (required for feedback).
    sensor : str, optional
        Name of the reactor whose state the controller senses (required for
        feedback).
    controller : str, optional
        Shared-controller id: dosing units giving the same id share one PI
        controller (and the controller unit takes this name). ``None`` gives this
        unit its own controller.
    gain : float
        This unit's share of the controller's flow output (default 1.0), so one
        controller can drive several dose points.
    Kp, Ti, Tt, flow_offset, flow_min, flow_max : float
        Feedback PI tuning and dose-flow bounds. ``flow_min`` defaults to 0 (a
        dose cannot be negative); ``flow_offset`` is the bias flow.
    input_port, output_port : str
        Port names (defaults ``"in"`` / ``"out"``).
    """

    name: str
    reagent: Reagent
    flow: float | None = None
    setpoint: float | None = None
    measured_species: str | None = None
    sensor: str | None = None
    controller: str | None = None
    gain: float = 1.0
    Kp: float = 1.0
    Ti: float = 0.05
    Tt: float = 0.02
    flow_offset: float = 0.0
    flow_min: float = 0.0
    flow_max: float = 1.0e6
    input_port: str = "in"
    output_port: str = "out"

    def __post_init__(self) -> None:
        n_modes = (self.flow is not None) + (self.setpoint is not None)
        if n_modes != 1:
            raise ValueError(
                f"DosingUnit '{self.name}' requires exactly one of flow= (fixed) "
                f"or setpoint= (feedback)."
            )
        if self.flow is not None and self.flow < 0.0:
            raise ValueError(f"DosingUnit '{self.name}' flow must be >= 0, got {self.flow}.")
        if self.setpoint is not None and (self.measured_species is None or self.sensor is None):
            raise ValueError(
                f"DosingUnit '{self.name}' feedback dosing needs both "
                f"measured_species= and sensor=."
            )
        if (
            self.measured_species is not None
            and self.measured_species not in self.model.species_index
        ):
            raise ValueError(
                f"DosingUnit '{self.name}' measured species "
                f"'{self.measured_species}' not in the reagent's model."
            )

    # ----- identity / protocol -------------------------------------------
    @property
    def model(self) -> CompiledModel:
        return self.reagent.model

    @property
    def state_size(self) -> int:
        return 0  # stateless; a feedback dose's integral is in the controller

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port]

    @property
    def is_closed_loop(self) -> bool:
        return self.setpoint is not None

    def controller_id(self) -> str:
        """The id of the (shared or dedicated) controller for this dose."""
        return self.controller if self.controller is not None else f"{self.name}_dosing"

    @property
    def required_signals(self) -> tuple[str, ...]:
        """The dose-flow signal this unit reads from the bus (feedback only)."""
        if not self.is_closed_loop:
            return ()
        return (dose_signal_name(self.controller_id()),)

    def initial_state(self) -> jnp.ndarray:
        return jnp.zeros((0,))

    # ----- behaviour ------------------------------------------------------
    def _dose_flow(self, signals: dict | None) -> jnp.ndarray:
        """The dose flow this step: the fixed setpoint, or the controller's
        (gain-scaled) signal for a feedback dose."""
        if self.flow is not None:
            return jnp.asarray(float(self.flow))
        if signals is None:
            raise ValueError(
                f"DosingUnit '{self.name}' is feedback-controlled but no "
                f"control-signal bus was supplied; a controlled dose must be "
                f"solved inside its plant (which supplies the dose-flow signal "
                f"'{self.required_signals[0]}')."
            )
        return signals[self.required_signals[0]] * self.gain

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict[str, Stream]:
        s_in = inputs[self.input_port]
        q_dose = self._dose_flow(signals)
        q_out = s_in.Q + q_dose
        mass = s_in.Q * s_in.C + q_dose * self.reagent.composition
        c_out = mass / (q_out + EPS_Q)
        # The reagent carries no side-channels of its own; the dosed stream keeps
        # the through-stream's scalars (temperature, ...): a small ambient dose into
        # a large flow does not move the temperature appreciably.
        return {self.output_port: Stream(Q=q_out, C=c_out, model=self.model, scalars=s_in.scalars)}

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Output flow = inflow + dose flow. For a feedback dose the actual flow
        is signal- (and thus concentration-) dependent, so the linear flow solve
        uses the nominal ``flow_offset`` here and the exact value is applied in
        :meth:`compute_outputs` -- the same convention the concentration-dependent
        separator flows use. Dosing is feed-forward and the dose is small relative
        to the through-flow, so the nominal is a good seed."""
        q_in = input_flows[self.input_port]
        q_dose = self.flow if self.flow is not None else self.flow_offset
        return {self.output_port: q_in + jnp.asarray(float(q_dose))}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        return jnp.zeros((0,))  # stateless
