"""Stateless flow-routing units: MixerUnit and the flow splitters."""

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
class _SplitterBase(StatelessUnit, FlowParameterized):
    """Shared base for the stateless flow splitters.

    A splitter routes one input stream (port ``"in"``) into several outputs,
    partitioning **only the flow**: concentration and every side-channel scalar
    pass through unchanged on each outlet (a passive split). The three concrete
    splitters differ only in *how* the flow is partitioned -- a fixed fraction
    (:class:`RatioSplitter`), fixed absolute pump setpoints with a remainder
    (:class:`SetpointSplitter`), or a threshold diversion
    (:class:`ThresholdSplitter`) -- so each is a distinct type carrying exactly
    the fields its rule needs, rather than one unit multiplexing on which of five
    mutually-exclusive optional fields happened to be supplied. Picking the class
    *is* picking the mode; the required fields are non-optional, so an incomplete
    or mixed configuration is a construction error, not a runtime guard.

    ``state_size`` / ``initial_state`` / ``rhs`` come from :class:`StatelessUnit`.
    """

    name: str
    model: "CompiledModel"

    @property
    def input_ports(self) -> list[str]:
        # List-returning property to match the Unit Protocol (list[str]); the
        # single input port is always "in".
        return ["in"]

    def _outlet(self, Q: jnp.ndarray, s_in: Stream) -> Stream:
        """One passive-split outlet: the partitioned flow ``Q`` carrying the
        inlet's concentration and side-channel ``scalars`` (temperature, indicator
        density, ...) unchanged."""
        return Stream(Q=Q, C=s_in.C, model=self.model, scalars=s_in.scalars)


@dataclass
class RatioSplitter(_SplitterBase):
    """Splits the inlet flow into fixed *fractions*: each output gets
    ``ratio * Q_in``, the fractions summing to 1. A passive split -- concentration
    and side-channel scalars are preserved on every outlet.

    Parameters
    ----------
    name : str
        Unit identifier.
    model : CompiledModel
    output_port_ratios : dict[str, float]
        Output port name -> fraction of inlet flow. Must sum to 1.
    """

    output_port_ratios: "dict[str, float]"

    def __post_init__(self) -> None:
        total = sum(self.output_port_ratios.values())
        if not (abs(total - 1.0) < 1e-9):
            raise ValueError(f"RatioSplitter '{self.name}' ratios must sum to 1.0; got {total}")

    @property
    def output_ports(self) -> list[str]:
        return list(self.output_port_ratios.keys())

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> dict[str, Stream]:
        s_in = inputs["in"]
        return {
            port: self._outlet(s_in.Q * jnp.asarray(ratio), s_in)
            for port, ratio in self.output_port_ratios.items()
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Output port flows: each a fixed fraction of the inlet flow (affine)."""
        Q_in = input_flows["in"]
        return {
            port: Q_in * jnp.asarray(ratio) for port, ratio in self.output_port_ratios.items()
        }


@dataclass
class SetpointSplitter(_SplitterBase):
    """Splits the inlet flow into fixed absolute *setpoint* flows plus a remainder.

    The named outputs are *flow-controlled pumps* delivering fixed absolute
    setpoint flows (m³/d); ``remainder_port`` takes whatever is left
    (``Q_in - sum(setpoints)``). This is the correct model for the BSM recycle
    pumps (internal recycle, RAS, wastage), whose volumetric flows are held
    constant regardless of influent -- see :func:`build_bsm1`. A fixed *fraction*
    of throughput (:class:`RatioSplitter`), by contrast, makes the recycle-flow
    loop gain near-singular off the design influent and the plant blows up under
    dynamic flow.

    If the feed transiently drops *below* the total setpoint the **material**
    streams (:meth:`compute_outputs`) share the available flow proportionally
    (``q·min(1, Q_in/Σsetpoints)``) with a zero remainder, so the unit never
    carries more than it receives. The recycle-flow rule (:meth:`flow_outputs`)
    stays the exact *affine* ``Q_in − Σsetpoints`` remainder (which the linear
    recycle solve requires); the two coincide whenever ``Q_in ≥ Σsetpoints``, true
    at any steady state. The setpoints are :class:`FlowSetpoint` s, so a plant is
    differentiable w.r.t. them (SRT / recycle-ratio design sweeps).

    Parameters
    ----------
    name : str
        Unit identifier.
    model : CompiledModel
    output_port_flows : dict[str, float]
        Output port name -> fixed setpoint flow (m³/d).
    remainder_port : str
        The output port carrying the remaining flow. Must not also be a setpoint
        port.
    """

    output_port_flows: "dict[str, float]"
    remainder_port: str

    def __post_init__(self) -> None:
        if self.remainder_port in self.output_port_flows:
            raise ValueError(
                f"SetpointSplitter '{self.name}': remainder_port "
                f"'{self.remainder_port}' must not also be a setpoint port."
            )
        # Wrap the absolute setpoints as FlowSetpoints (a fixed order) so the
        # recycle-flow rule and the material split read one shared, differentiable
        # value.
        self._setpoints = {
            port: FlowSetpoint(float(q), i)
            for i, (port, q) in enumerate(self.output_port_flows.items())
        }

    def _flow_setpoints(self) -> "dict[str, FlowSetpoint]":
        return self._setpoints

    @property
    def output_ports(self) -> list[str]:
        return list(self.output_port_flows.keys()) + [self.remainder_port]

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> dict[str, Stream]:
        # Fixed setpoints, remainder takes what is left. When the feed is below the
        # total setpoint the setpoint ports share the available flow proportionally
        # (a flow-limited pump set), so the MATERIAL streams never carry more than
        # the unit receives -- it conserves mass -- and the remainder is then zero.
        # Identity when Q_in >= total setpoint (scale == 1), so steady-state
        # behaviour is unchanged; the scale-down only bites in a transient starve.
        # (flow_outputs stays the exact AFFINE rule the recycle-flow solve requires
        # -- it must, or the (I-A)x=b probe breaks -- and the two agree wherever the
        # unit is not starved, i.e. at any steady state.)
        s_in = inputs["in"]
        fp = self._flow_params(params)
        setpts = {port: sp.resolve(fp) for port, sp in self._setpoints.items()}
        total_set = jnp.zeros(())
        for q in setpts.values():
            total_set = total_set + q
        scale = jnp.minimum(1.0, s_in.Q / jnp.maximum(total_set, 1e-12))
        outputs = {port: self._outlet(q * scale, s_in) for port, q in setpts.items()}
        outputs[self.remainder_port] = self._outlet(jnp.maximum(s_in.Q - total_set, 0.0), s_in)
        return outputs

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Output port flows: the fixed setpoints plus the exact AFFINE remainder
        ``Q_in - sum(setpoints)`` (which the linear recycle solve requires). The
        remainder may go negative in a transient starve, harmless for the linear
        flow solve; the conserving scale-down lives in :meth:`compute_outputs`, and
        the two agree wherever the unit is not starved."""
        Q_in = input_flows["in"]
        fp = self._flow_params(params)
        out = {port: sp.resolve(fp) for port, sp in self._setpoints.items()}
        out[self.remainder_port] = Q_in - sum(out.values())
        return out


@dataclass
class ThresholdSplitter(_SplitterBase):
    """Diverts inlet flow *above* a threshold, passing the rest through.

    Inlet flow above ``threshold`` goes to ``threshold_port``
    (``max(Q_in - threshold, 0)``) and the rest (``min(Q_in, threshold)``) to
    ``remainder_port``. This is the BSM2 hydraulic influent bypass (flow above a
    limit diverted around the treatment). The split is piecewise-linear (a kink at
    ``threshold``), so the exact recycle-flow solve (:meth:`Plant._resolve_flows`)
    is only exact when the inlet flow is *independent of the recycle flows* -- e.g.
    fed directly by an external influent, as in :func:`build_bsm2`. The threshold
    is a :class:`FlowSetpoint`, so a plant is differentiable w.r.t. it.

    Parameters
    ----------
    name : str
        Unit identifier.
    model : CompiledModel
    threshold : float
        Inlet-flow limit (m³/d).
    threshold_port : str
        The output port carrying the above-threshold flow.
    remainder_port : str
        The output port carrying the remaining (below-threshold) flow. Must differ
        from ``threshold_port``.
    """

    threshold: float
    threshold_port: str
    remainder_port: str

    def __post_init__(self) -> None:
        if self.threshold_port == self.remainder_port:
            raise ValueError(
                f"ThresholdSplitter '{self.name}': threshold_port and "
                f"remainder_port must differ."
            )
        self._setpoints = {"threshold": FlowSetpoint(float(self.threshold), 0)}

    def _flow_setpoints(self) -> "dict[str, FlowSetpoint]":
        return self._setpoints

    @property
    def output_ports(self) -> list[str]:
        return [self.threshold_port, self.remainder_port]

    def _limit(self, params) -> jnp.ndarray:
        return self._setpoints["threshold"].resolve(self._flow_params(params))

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> dict[str, Stream]:
        s_in = inputs["in"]
        limit = self._limit(params)
        return {
            self.threshold_port: self._outlet(jnp.maximum(s_in.Q - limit, 0.0), s_in),
            self.remainder_port: self._outlet(jnp.minimum(s_in.Q, limit), s_in),
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Output port flows: the above/below-threshold split (piecewise-linear)."""
        Q_in = input_flows["in"]
        limit = self._limit(params)
        return {
            self.threshold_port: jnp.maximum(Q_in - limit, 0.0),
            self.remainder_port: jnp.minimum(Q_in, limit),
        }
