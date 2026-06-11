"""Feedback control units for plant-wide simulation.

Control is layered on the material flowsheet with a small *signal bus*: a
controller reads a measured variable from one of its (material) input streams,
carries any controller state (the PI integral) as part of the plant state, and
writes a named scalar **signal**; an actuated unit (e.g. an aerated CSTR under
dissolved-oxygen control) reads that signal in its ``rhs``. The plant evaluates
``signal_outputs`` on every controller each RHS call and threads the resulting
signals to units that declare ``consumes_signals`` -- see ``Plant._rhs``.

The shipped controller is :class:`PIController`, a PI loop with back-calculation
anti-windup (the BSM1/BSM2 dissolved-oxygen / kLa controller form).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


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
    network : CompiledNetwork
        Network of the sensed stream (to resolve ``measured_species``).
    measured_species : str
        Species read from the sensed inlet stream (e.g. ``"SO"``).
    setpoint : float
        Target value for the measured variable.
    Kp, Ti, Tt : float
        Proportional gain, integral time, anti-windup tracking time (Ti, Tt in
        the plant's time unit, i.e. days for BSM). BSM2 DO loop: Kp=25,
        Ti=0.002, Tt=0.001.
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
    network: "CompiledNetwork"
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
        if self.measured_species not in self.network.species_index:
            raise ValueError(
                f"PIController '{self.name}': measured species "
                f"'{self.measured_species}' not in network."
            )
        self._meas_idx = self.network.species_index[self.measured_species]
        if self.Ti <= 0.0:
            raise ValueError(f"PIController '{self.name}': Ti must be > 0.")
        if self.use_antiwindup and self.Tt <= 0.0:
            raise ValueError(
                f"PIController '{self.name}': Tt must be > 0 with anti-windup."
            )

    @property
    def state_size(self) -> int:
        return 1  # the integral state x_i

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
        measured = inputs[self.input_port].C[self._meas_idx]
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
    ) -> dict:
        _, u_sat, _ = self._output(state, inputs)
        return {self.signal_name: u_sat}

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> dict:
        return {}  # no material outputs

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray) -> dict:
        return {}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        u, u_sat, e = self._output(state, inputs)
        dxi = (self.Kp / self.Ti) * e
        if self.use_antiwindup:
            dxi = dxi + (1.0 / self.Tt) * (u_sat - u)
        return jnp.reshape(dxi, (1,))
