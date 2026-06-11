"""Stateless flow-routing units: MixerUnit and SplitterUnit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


_EPS_Q = 1e-12  # guard against 0/0 when all inflows are zero


@dataclass
class MixerUnit:
    """Combines two or more input streams into a single output stream by
    mass balance: ``Q_out = sum(Q_in_i)``, ``C_out = sum(Q_in_i * C_in_i) / Q_out``.

    Stateless. All input streams must reference the same kinetic network
    (translators are applied upstream by the plant).

    Parameters
    ----------
    name : str
        Unit identifier.
    input_port_names : list[str]
        Names of the input ports. Order is not significant for the
        computation, but the plant uses them when wiring connections.
    network : CompiledNetwork
        Network of all input streams and the single output stream.
    """

    name: str
    input_port_names: list[str]
    network: "CompiledNetwork"

    @property
    def state_size(self) -> int:
        # Stateless: a mixer's output is an algebraic function of its inputs.
        return 0

    @property
    def input_ports(self) -> list[str]:
        return list(self.input_port_names)

    @property
    def output_ports(self) -> list[str]:
        # List-returning property to match the Unit Protocol (list[str]) and the
        # CSTR / clarifier units; the single output port is always "out".
        return ["out"]

    def initial_state(self) -> jnp.ndarray:
        return jnp.zeros((0,))

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> dict[str, Stream]:
        Q_total = jnp.zeros(())
        mass_total = jnp.zeros((self.network.n_species,))
        for name in self.input_port_names:
            s = inputs[name]
            Q_total = Q_total + s.Q
            mass_total = mass_total + s.Q * s.C
        C_out = mass_total / (Q_total + _EPS_Q)
        # Heat balance: the outlet temperature is the flow-weighted inlet
        # temperature. Only computed when every inlet carries a temperature
        # (a static, topology-determined property); otherwise the mixer stays
        # temperature-agnostic.
        T_out = None
        if all(inputs[name].T is not None for name in self.input_port_names):
            heat = jnp.zeros(())
            for name in self.input_port_names:
                s = inputs[name]
                heat = heat + s.Q * s.T
            T_out = heat / (Q_total + _EPS_Q)
        return {"out": Stream(Q=Q_total, C=C_out, network=self.network, T=T_out)}

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray) -> dict:
        """Output port flows from input port flows (the linear flow rule).

        Used by ``Plant`` to resolve the recycle-flow network cheaply and
        exactly, decoupled from the (expensive) concentration computation.
        """
        Q_total = jnp.zeros(())
        for name in self.input_port_names:
            Q_total = Q_total + input_flows[name]
        return {"out": Q_total}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.zeros((0,))


@dataclass
class SplitterUnit:
    """Splits one input stream into N output streams.

    Concentration is preserved across all outputs (passive splitter); only
    the flow rate is partitioned. Two partition modes, exactly one of which
    must be supplied:

    - **ratio mode** (``output_port_ratios``): each output gets a fixed
      *fraction* of the inlet flow. Fractions must sum to 1.
    - **flow mode** (``output_port_flows`` + ``remainder_port``): the named
      outputs are *flow-controlled pumps* delivering fixed absolute setpoint
      flows; ``remainder_port`` takes whatever is left
      (``Q_in - sum(setpoints)``). This is the correct model for the BSM
      recycle pumps (internal recycle, RAS, wastage), whose volumetric flows
      are held constant regardless of influent — see :func:`build_bsm1`. A
      fixed *fraction* of throughput, by contrast, makes the recycle-flow
      loop gain near-singular off the design influent and the plant blows up
      under dynamic flow.

    - **threshold mode** (``threshold`` + ``threshold_port`` +
      ``remainder_port``): inlet flow *above* ``threshold`` goes to
      ``threshold_port`` (``max(Q_in - threshold, 0)``) and the rest
      (``min(Q_in, threshold)``) to ``remainder_port``. This is the BSM2
      hydraulic influent bypass (flow above a limit diverted around the
      treatment). The split is piecewise-linear (a kink at ``threshold``), so
      the exact recycle-flow solve (:meth:`Plant._resolve_flows`) is only exact
      when the inlet flow is *independent of the recycle flows* -- e.g. fed
      directly by an external influent, as in :func:`build_bsm2`.

    Parameters
    ----------
    name : str
        Unit identifier.
    network : CompiledNetwork
    output_port_ratios : dict[str, float], optional
        Ratio mode: output port name -> fraction of inlet flow (sum to 1).
    output_port_flows : dict[str, float], optional
        Flow mode: output port name -> fixed setpoint flow (m³/d). Requires
        ``remainder_port``.
    threshold : float, optional
        Threshold mode: inlet-flow limit (m³/d). Requires ``threshold_port``
        and ``remainder_port``.
    threshold_port : str, optional
        Threshold mode: the output port carrying the above-threshold flow.
    remainder_port : str, optional
        Flow / threshold mode: the output port carrying the remaining flow.
    """

    name: str
    network: "CompiledNetwork"
    output_port_ratios: "dict[str, float] | None" = None
    output_port_flows: "dict[str, float] | None" = None
    threshold: "float | None" = None
    threshold_port: "str | None" = None
    remainder_port: "str | None" = None

    @property
    def state_size(self) -> int:
        # Stateless: a splitter routes its inlet flow by fixed ratios/flows.
        return 0

    @property
    def _mode(self) -> str:
        if self.output_port_ratios is not None:
            return "ratio"
        if self.output_port_flows is not None:
            return "flow"
        return "threshold"

    def __post_init__(self) -> None:
        n_modes = sum(m is not None for m in (
            self.output_port_ratios, self.output_port_flows, self.threshold))
        if n_modes != 1:
            raise ValueError(
                f"SplitterUnit '{self.name}': supply exactly one of "
                f"output_port_ratios, output_port_flows, or threshold."
            )
        if self._mode == "ratio":
            total = sum(self.output_port_ratios.values())
            if not (abs(total - 1.0) < 1e-9):
                raise ValueError(
                    f"SplitterUnit '{self.name}' ratios must sum to 1.0; got {total}"
                )
        elif self._mode == "flow":
            if self.remainder_port is None:
                raise ValueError(
                    f"SplitterUnit '{self.name}': output_port_flows requires "
                    f"remainder_port."
                )
            if self.remainder_port in self.output_port_flows:
                raise ValueError(
                    f"SplitterUnit '{self.name}': remainder_port "
                    f"'{self.remainder_port}' must not also be a setpoint port."
                )
        else:  # threshold
            if self.threshold_port is None or self.remainder_port is None:
                raise ValueError(
                    f"SplitterUnit '{self.name}': threshold requires both "
                    f"threshold_port and remainder_port."
                )
            if self.threshold_port == self.remainder_port:
                raise ValueError(
                    f"SplitterUnit '{self.name}': threshold_port and "
                    f"remainder_port must differ."
                )

    @property
    def input_ports(self) -> list[str]:
        # List-returning property to match the Unit Protocol (list[str]); the
        # single input port is always "in".
        return ["in"]

    @property
    def output_ports(self) -> list[str]:
        if self._mode == "ratio":
            return list(self.output_port_ratios.keys())
        if self._mode == "flow":
            return list(self.output_port_flows.keys()) + [self.remainder_port]
        return [self.threshold_port, self.remainder_port]

    def initial_state(self) -> jnp.ndarray:
        return jnp.zeros((0,))

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> dict[str, Stream]:
        s_in = inputs["in"]
        outputs: dict[str, Stream] = {}
        # A passive splitter preserves the inlet temperature on every outlet.
        if self._mode == "ratio":
            for port, ratio in self.output_port_ratios.items():
                outputs[port] = Stream(
                    Q=s_in.Q * jnp.asarray(ratio), C=s_in.C, network=self.network,
                    T=s_in.T,
                )
            return outputs
        if self._mode == "threshold":
            # Inlet flow above the limit is diverted; the rest passes through.
            limit = jnp.asarray(float(self.threshold))
            above = jnp.maximum(s_in.Q - limit, 0.0)
            outputs[self.threshold_port] = Stream(
                Q=above, C=s_in.C, network=self.network, T=s_in.T)
            outputs[self.remainder_port] = Stream(
                Q=jnp.minimum(s_in.Q, limit), C=s_in.C, network=self.network,
                T=s_in.T)
            return outputs
        # Flow mode: fixed setpoints, remainder takes what is left. Clamp the
        # remainder at zero so a feed transiently below the total setpoint does
        # not produce a negative-flow stream (the concentration-stage safeguard;
        # the linear flow_outputs below stays unclamped so _resolve_flows is
        # exact).
        total_set = jnp.zeros(())
        for port, q in self.output_port_flows.items():
            q = jnp.asarray(float(q))
            total_set = total_set + q
            outputs[port] = Stream(Q=q, C=s_in.C, network=self.network, T=s_in.T)
        outputs[self.remainder_port] = Stream(
            Q=jnp.maximum(s_in.Q - total_set, 0.0), C=s_in.C, network=self.network,
            T=s_in.T,
        )
        return outputs

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray) -> dict:
        """Output port flows from the inlet flow (the linear flow rule)."""
        Q_in = input_flows["in"]
        if self._mode == "ratio":
            return {port: Q_in * jnp.asarray(ratio)
                    for port, ratio in self.output_port_ratios.items()}
        if self._mode == "threshold":
            limit = jnp.asarray(float(self.threshold))
            return {self.threshold_port: jnp.maximum(Q_in - limit, 0.0),
                    self.remainder_port: jnp.minimum(Q_in, limit)}
        out = {port: jnp.asarray(float(q))
               for port, q in self.output_port_flows.items()}
        out[self.remainder_port] = Q_in - sum(out.values())
        return out

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.zeros((0,))
