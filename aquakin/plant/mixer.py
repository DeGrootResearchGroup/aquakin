"""Stateless flow-routing units: MixerUnit and SplitterUnit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant.flow_setpoint import FlowParameterized, FlowSetpoint
from aquakin.plant.streams import Stream, mixed_scalars
from aquakin.plant.units import StatelessUnit

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


_EPS_Q = 1e-12  # guard against 0/0 when all inflows are zero


@dataclass
class MixerUnit(StatelessUnit):
    """Combines two or more input streams into a single output stream by
    mass balance: ``Q_out = sum(Q_in_i)``, ``C_out = sum(Q_in_i * C_in_i) / Q_out``.

    Stateless. All input streams must reference the same kinetic model
    (translators are applied upstream by the plant).

    Parameters
    ----------
    name : str
        Unit identifier.
    input_port_names : list[str]
        Names of the input ports. Order is not significant for the
        computation, but the plant uses them when wiring connections.
    model : CompiledModel
        Model of all input streams and the single output stream.
    """

    name: str
    input_port_names: list[str]
    model: "CompiledModel"

    # state_size / initial_state / rhs come from StatelessUnit.

    @property
    def input_ports(self) -> list[str]:
        return list(self.input_port_names)

    @property
    def output_ports(self) -> list[str]:
        # List-returning property to match the Unit Protocol (list[str]) and the
        # CSTR / clarifier units; the single output port is always "out".
        return ["out"]

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> dict[str, Stream]:
        Q_total = jnp.zeros(())
        mass_total = jnp.zeros((self.model.n_species,))
        for name in self.input_port_names:
            s = inputs[name]
            Q_total = Q_total + s.Q
            mass_total = mass_total + s.Q * s.C
        C_out = mass_total / (Q_total + _EPS_Q)
        # Side-channel scalars: the outlet temperature is the flow-weighted inlet
        # temperature (a heat balance) and the indicator density the same
        # flow-weighted mass balance -- both from the one shared combiner, over the
        # inlets that carry each (an agnostic or zero-flow-seed inlet is ignored
        # rather than poisoning the mix).
        scalars_out = mixed_scalars(inputs, self.input_port_names)
        return {"out": Stream(Q=Q_total, C=C_out, model=self.model, scalars=scalars_out)}

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Output port flows from input port flows (the linear flow rule).

        Used by ``Plant`` to resolve the recycle-flow network cheaply and
        exactly, decoupled from the (expensive) concentration computation.
        """
        Q_total = jnp.zeros(())
        for name in self.input_port_names:
            Q_total = Q_total + input_flows[name]
        return {"out": Q_total}


@dataclass
class SplitterUnit(StatelessUnit, FlowParameterized):
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
      under dynamic flow. If the feed transiently drops *below* the total
      setpoint the **material** streams (``compute_outputs``) share the available
      flow proportionally (``q·min(1, Q_in/Σsetpoints)``) with a zero remainder,
      so the unit never carries more than it receives. The recycle-flow rule
      (``flow_outputs``) stays the exact *affine* ``Q_in − Σsetpoints`` remainder
      (which the linear recycle solve requires); the two coincide whenever
      ``Q_in ≥ Σsetpoints``, true at any steady state.

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
    model : CompiledModel
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
    model: "CompiledModel"
    output_port_ratios: "dict[str, float] | None" = None
    output_port_flows: "dict[str, float] | None" = None
    threshold: "float | None" = None
    threshold_port: "str | None" = None
    remainder_port: "str | None" = None

    # state_size / initial_state / rhs come from StatelessUnit.

    @property
    def _mode(self) -> str:
        if self.output_port_ratios is not None:
            return "ratio"
        if self.output_port_flows is not None:
            return "flow"
        return "threshold"

    def __post_init__(self) -> None:
        n_modes = sum(
            m is not None for m in (self.output_port_ratios, self.output_port_flows, self.threshold)
        )
        if n_modes != 1:
            raise ValueError(
                f"SplitterUnit '{self.name}': supply exactly one of "
                f"output_port_ratios, output_port_flows, or threshold."
            )
        if self._mode == "ratio":
            total = sum(self.output_port_ratios.values())
            if not (abs(total - 1.0) < 1e-9):
                raise ValueError(f"SplitterUnit '{self.name}' ratios must sum to 1.0; got {total}")
        elif self._mode == "flow":
            if self.remainder_port is None:
                raise ValueError(
                    f"SplitterUnit '{self.name}': output_port_flows requires remainder_port."
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
                    f"SplitterUnit '{self.name}': threshold_port and remainder_port must differ."
                )
        # Wrap the absolute flow setpoints (flow / threshold mode) as
        # FlowSetpoints so the recycle-flow rule and the material split read one
        # shared, differentiable value. Ratio mode has no absolute flow setpoint.
        if self._mode == "flow":
            self._setpoints = {
                port: FlowSetpoint(float(q), i)
                for i, (port, q) in enumerate(self.output_port_flows.items())
            }
        elif self._mode == "threshold":
            self._setpoints = {"threshold": FlowSetpoint(float(self.threshold), 0)}
        else:
            self._setpoints = {}

    def _flow_setpoints(self) -> "dict[str, FlowSetpoint]":
        return self._setpoints

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

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> dict[str, Stream]:
        s_in = inputs["in"]
        outputs: dict[str, Stream] = {}
        # A passive splitter preserves the inlet's side-channel scalars
        # (temperature, indicator density, ...) unchanged on every outlet.
        if self._mode == "ratio":
            for port, ratio in self.output_port_ratios.items():
                outputs[port] = Stream(
                    Q=s_in.Q * jnp.asarray(ratio),
                    C=s_in.C,
                    model=self.model,
                    scalars=s_in.scalars,
                )
            return outputs
        if self._mode == "threshold":
            # Inlet flow above the limit is diverted; the rest passes through.
            limit = self._setpoints["threshold"].resolve(self._flow_params(params))
            above = jnp.maximum(s_in.Q - limit, 0.0)
            outputs[self.threshold_port] = Stream(
                Q=above, C=s_in.C, model=self.model, scalars=s_in.scalars
            )
            outputs[self.remainder_port] = Stream(
                Q=jnp.minimum(s_in.Q, limit), C=s_in.C, model=self.model, scalars=s_in.scalars
            )
            return outputs
        # Flow mode: fixed setpoints, remainder takes what is left. When the feed
        # is below the total setpoint the setpoint ports share the available flow
        # proportionally (a flow-limited pump set), so the MATERIAL streams never
        # carry more than the unit receives -- it conserves mass -- and the
        # remainder is then zero. Identity when Q_in >= total setpoint (scale ==
        # 1), so steady-state behaviour is unchanged; the scale-down only bites in
        # a transient starve. (flow_outputs below stays the exact AFFINE rule the
        # recycle-flow solve requires -- it must, or the (I-A)x=b probe breaks --
        # and the two agree wherever the unit is not starved, i.e. at any steady
        # state.)
        fp = self._flow_params(params)
        setpts = {port: sp.resolve(fp) for port, sp in self._setpoints.items()}
        total_set = jnp.zeros(())
        for q in setpts.values():
            total_set = total_set + q
        scale = jnp.minimum(1.0, s_in.Q / jnp.maximum(total_set, 1e-12))
        for port, q in setpts.items():
            outputs[port] = Stream(Q=q * scale, C=s_in.C, model=self.model, scalars=s_in.scalars)
        outputs[self.remainder_port] = Stream(
            Q=jnp.maximum(s_in.Q - total_set, 0.0),
            C=s_in.C,
            model=self.model,
            scalars=s_in.scalars,
        )
        return outputs

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Output port flows from the inlet flow (the AFFINE flow rule the
        recycle-flow solve requires). The flow-mode remainder is the exact
        ``Q_in - sum(setpoints)`` and stays affine in ``Q_in`` (it may go negative
        in a transient starve, which is harmless for the linear flow solve); the
        conserving scale-down lives in :meth:`compute_outputs` (the material
        sweep), and the two agree wherever the unit is not starved."""
        Q_in = input_flows["in"]
        if self._mode == "ratio":
            return {
                port: Q_in * jnp.asarray(ratio) for port, ratio in self.output_port_ratios.items()
            }
        if self._mode == "threshold":
            limit = self._setpoints["threshold"].resolve(self._flow_params(params))
            return {
                self.threshold_port: jnp.maximum(Q_in - limit, 0.0),
                self.remainder_port: jnp.minimum(Q_in, limit),
            }
        fp = self._flow_params(params)
        out = {port: sp.resolve(fp) for port, sp in self._setpoints.items()}
        out[self.remainder_port] = Q_in - sum(out.values())
        return out
