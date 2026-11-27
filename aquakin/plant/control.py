"""Feedback control units for plant-wide simulation.

Control is layered on the material flowsheet with a small *signal bus*: a
controller reads a measured variable from one of its (material) input streams,
carries any controller state (the PI integral) as part of the plant state, and
writes a named scalar **signal**; an actuated unit (e.g. an aerated CSTR under
dissolved-oxygen control) reads that signal in its ``rhs``. The plant evaluates
``signal_outputs`` on every controller each RHS call and threads the resulting
signal bus into every unit's ``rhs`` as ``signals`` -- see ``Plant._rhs``.

Two controllers ship: :class:`PIController`, a PI loop with back-calculation
anti-windup (the BSM1/BSM2 dissolved-oxygen / kLa controller form), and
:class:`ExpressionController`, a stateless control law written as an expression
(:mod:`aquakin.plant.control_law`). A law may read signals other controllers
publish; the plant runs each producer after those it reads
(:func:`aquakin.plant.signal_bus.order_signal_producers`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.core.hints import did_you_mean
from aquakin.core.time_expr import CONSTANTS
from aquakin.plant.control_law import CHANNELS, ControlLawError, compile_law
from aquakin.plant.errors import WiringError
from aquakin.plant.streams import Stream
from aquakin.plant.units import StatelessUnit

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


def check_sensed_model(owner: str, model, sensed, what: str) -> None:
    """Refuse a sensed stream whose model orders its species differently.

    A controller resolves species indices against its own ``model``; they index
    the right species only if the sensed stream shares that ordering. The models
    are static, so this is a trace-time check (it fires on the first solve, not
    every step); the ``is`` short-circuit keeps the common same-instance case
    free, while two instances of the same model (same species order) are still
    accepted.

    Parameters
    ----------
    owner : str
        The controller, for the message (``"PIController 'do'"``).
    model : CompiledModel
        The model its indices were resolved in.
    sensed : CompiledModel
        The sensed stream's model.
    what : str
        What was resolved, for the message.

    Raises
    ------
    ValueError
        If the two models list different species.
    """
    if sensed is not model and getattr(sensed, "species", None) != model.species:
        raise ValueError(
            f"{owner} senses a stream in model '{getattr(sensed, 'name', '?')}', but "
            f"{what} was resolved in model '{model.name}' with a different species "
            f"ordering -- it would read the wrong species. Build the controller with "
            f"the sensed unit's model."
        )


@dataclass
class PIController:
    """A PI controller with back-calculation anti-windup.

    Reads ``measured_species`` from the stream on ``input_port`` (e.g. the
    dissolved oxygen ``SO`` in the sensed reactor's outlet), drives the
    controller output toward ``setpoint``, and publishes the (saturated) output
    as the signal ``signal_name`` for actuated units to consume::

        u     = offset + Kp*e + x_i,   e = setpoint - measured
        u_sat = clip(u, out_min, out_max)
        dx_i/dt = (Kp/Ti)*e + (1/Tt)*(u_sat - u)      # tracking anti-windup

    The single integrator state ``x_i`` is the integral *contribution to the
    output* (already scaled), so the anti-windup term has consistent units.

    Parameters
    ----------
    name : str
    model : CompiledModel
        Model of the sensed stream (to resolve ``measured_species``).
    measured_species : str
        Species read from the sensed inlet stream (e.g. ``"SO"``).
    setpoint : float
        Target value for the measured variable.
    Kp : float
        Proportional gain (output per unit of the measured variable). BSM2's DO
        loop uses 25.
    Ti : float
        Integral time (d, on the plant's clock). BSM2's DO loop uses 0.002.
    Tt : float
        Anti-windup tracking time (d, on the plant's clock). BSM2's DO loop
        uses 0.001.
    offset : float
        Output bias (the output when ``e`` and the integral are zero).
    out_min, out_max : float
        Output saturation limits (e.g. kLa in [0, kLa_max]).
    signal_name : str
        Name under which the output is published on the signal bus.
    use_antiwindup : bool
        If False, the integrator is the plain ``(Kp/Ti)*e`` (no tracking term).
    input_port : str
    """

    name: str
    model: CompiledModel
    measured_species: str
    setpoint: float
    Kp: float
    Ti: float
    Tt: float
    offset: float
    out_min: float
    out_max: float
    signal_name: str
    use_antiwindup: bool = True
    input_port: str = "measured"

    def __post_init__(self) -> None:
        if self.measured_species not in self.model.species_index:
            raise ValueError(
                f"PIController '{self.name}': measured species "
                f"'{self.measured_species}' not in model."
            )
        self._meas_idx = self.model.species_index[self.measured_species]
        if self.Ti <= 0.0:
            raise ValueError(f"PIController '{self.name}': Ti must be > 0.")
        if self.use_antiwindup and self.Tt <= 0.0:
            raise ValueError(f"PIController '{self.name}': Tt must be > 0 with anti-windup.")

    @property
    def state_size(self) -> int:
        return 1  # the integral state x_i

    def state_names(self) -> tuple[str, ...]:
        """The one state entry: the controller's ``integral``."""
        return ("integral",)

    @property
    def signal_names(self) -> tuple[str, ...]:
        """Control-signal names this unit publishes to the bus (see
        :meth:`signal_outputs`). Lets the plant validate that consuming units'
        signal references resolve."""
        return (self.signal_name,)

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return []  # produces a signal, not a material stream

    def initial_state(self) -> jnp.ndarray:
        return jnp.zeros((1,))

    def _output(self, state: jnp.ndarray, inputs: dict[str, Stream]):
        """Return ``(u, u_sat, e)``: raw output, saturated output, error."""
        stream = inputs[self.input_port]
        # ``_meas_idx`` was resolved against ``self.model``; it indexes the right
        # species only if the sensed stream shares that model's species ordering.
        check_sensed_model(
            f"PIController '{self.name}'",
            self.model,
            stream.model,
            f"its measured_species '{self.measured_species}' index",
        )
        measured = stream.C[self._meas_idx]
        e = self.setpoint - measured
        u = self.offset + self.Kp * e + state[0]
        u_sat = jnp.clip(u, self.out_min, self.out_max)
        return u, u_sat, e

    def signal_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict:
        """Publish the saturated output; the bus so far (``signals``) is not read."""
        _, u_sat, _ = self._output(state, inputs)
        return {self.signal_name: u_sat}

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict:
        return {}  # no material outputs

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        return {}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        u, u_sat, e = self._output(state, inputs)
        dxi = (self.Kp / self.Ti) * e
        if self.use_antiwindup:
            dxi = dxi + (1.0 / self.Tt) * (u_sat - u)
        return jnp.reshape(dxi, (1,))


@dataclass
class ExpressionController(StatelessUnit):
    """A stateless controller whose output is a law written as an expression.

    The law is read once, at construction, into plain JAX arithmetic, and its
    value is published as the signal ``signal_name`` every right-hand-side call::

        ExpressionController(
            "nh4_loop", model=asm1, signal_name="kla_tank3",
            law="clip(offset + Kp*([SNH] - sp), 0, 240)",
            constants={"Kp": 40.0, "sp": 1.0, "offset": 120.0},
        )

    A law reads ``[species]`` and the measurement channels ``TSS``, ``COD``,
    ``TN``, ``TKN``, ``NOx`` and ``TP`` of the stream on its ``measured`` port
    (the channel weights follow the run's composition parameters), its
    ``constants``, the plant clock ``t``, ``pi`` and ``e``, and by bare name any
    signal another unit publishes, so one law can cascade from another's
    output; the plant runs it after the producers it reads. Limits go in the law
    with ``clip``; there is no ``if``. See :mod:`aquakin.plant.control_law`.
    Every setting but ``name`` is passed by keyword.

    Parameters
    ----------
    name : str
        Unit identifier.
    model : CompiledModel, optional
        Model of the sensed stream, which the law's species and channels are read
        in. ``None`` (the default) senses nothing: the controller then has no
        input port, and its law reads only constants, ``t`` and signals.
    signal_name : str
        Name under which the law's value is published on the signal bus.
    law : str
        The control law, e.g. ``"clip(offset + Kp*([SNH] - sp), 0, 240)"``.
    constants : dict[str, float]
        Named numbers the law uses (gains, setpoints, bounds). They are fixed
        numbers, not plant parameters.
    """

    name: str
    _: KW_ONLY
    model: CompiledModel | None = None
    signal_name: str
    law: str
    constants: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        what = f"ExpressionController '{self.name}'"
        if self.signal_name in CHANNELS:
            raise ControlLawError(
                f"{what}: signal_name '{self.signal_name}' is the name of a measurement "
                f"channel, which a law reads as the channel; choose another name."
            )
        self.constants = {str(k): float(v) for k, v in self.constants.items()}
        self._law = compile_law(self.law, self.constants, self.model, what=what)

    @property
    def signal_names(self) -> tuple[str, ...]:
        """The one signal this unit publishes."""
        return (self.signal_name,)

    @property
    def required_signals(self) -> tuple[str, ...]:
        """The signals the law reads from the bus, which the plant computes first."""
        return self._law.signals

    @property
    def input_ports(self) -> list[str]:
        return ["measured"] if self.model is not None else []

    @property
    def output_ports(self) -> list[str]:
        return []  # produces a signal, not a material stream

    def check_signals(self, published: Mapping[str, str]) -> None:
        """Check the law's names against the signals the plant publishes.

        Parameters
        ----------
        published : mapping of str to str
            Every published signal, to the unit that publishes it.

        Raises
        ------
        ControlLawError
            If the law reads a name that is none of its constants, a channel,
            ``t``, ``pi``, ``e`` or a published signal (with a "did you mean"),
            or a name that is both a published signal and a channel or constant.
        """
        what = f"ExpressionController '{self.name}'"
        for name in (*self._law.channels, *self.constants):
            if name in published:
                role = "measurement channel" if name in CHANNELS else "constant"
                raise ControlLawError(
                    f"{what}: '{name}' is both a {role} of its law and a signal "
                    f"'{published[name]}' publishes, so the law cannot tell which it "
                    f"reads. Rename the signal or the constant."
                )
        known = [*self.constants, *CHANNELS, "t", *CONSTANTS, *published]
        for name in self._law.signals:
            if name not in published:
                raise ControlLawError(
                    f"{what}: its law reads '{name}', which is not one of its constants, "
                    f"a measurement channel, 't', 'pi', 'e' or a signal a unit of this "
                    f"plant publishes.{did_you_mean(name, known)}"
                )

    def signal_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict:
        """Publish the law's value, read against the bus so far (``signals``)."""
        C = None
        if self._law.reads_stream:
            stream = inputs.get("measured")
            if stream is None:
                raise WiringError(
                    f"ExpressionController '{self.name}' reads the sensed stream, but "
                    f"nothing is connected to its 'measured' port. Connect the sensed "
                    f"unit: plant.connect('<unit>.out', '{self.name}.measured')."
                )
            check_sensed_model(
                f"ExpressionController '{self.name}'", self.model, stream.model, "its law"
            )
            C = stream.C
        return {self.signal_name: self._law(t, C, signals or {}, params)}

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict:
        return {}  # no material outputs

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        return {}
