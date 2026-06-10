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
    state_size: int = 0
    output_ports: tuple[str, ...] = ("out",)

    @property
    def input_ports(self) -> list[str]:
        return list(self.input_port_names)

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
        return {"out": Stream(Q=Q_total, C=C_out, network=self.network)}

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
    """Splits one input stream into N output streams by fixed flow ratios.

    Concentration is preserved across all outputs (passive splitter); only
    the flow rate is partitioned.

    Parameters
    ----------
    name : str
        Unit identifier.
    output_port_ratios : dict[str, float]
        Output port name -> fraction of inlet flow. Fractions must sum to 1
        (validated at construction; calibration of split ratios uses
        ``params`` instead — see :attr:`output_ports_dynamic`).
    network : CompiledNetwork
    """

    name: str
    output_port_ratios: dict[str, float]
    network: "CompiledNetwork"
    state_size: int = 0
    input_ports: tuple[str, ...] = ("in",)

    def __post_init__(self) -> None:
        total = sum(self.output_port_ratios.values())
        if not (abs(total - 1.0) < 1e-9):
            raise ValueError(
                f"SplitterUnit '{self.name}' ratios must sum to 1.0; got {total}"
            )

    @property
    def output_ports(self) -> list[str]:
        return list(self.output_port_ratios.keys())

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
        for port, ratio in self.output_port_ratios.items():
            outputs[port] = Stream(
                Q=s_in.Q * jnp.asarray(ratio),
                C=s_in.C,
                network=self.network,
            )
        return outputs

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray) -> dict:
        """Output port flows = inlet flow times each split ratio."""
        Q_in = input_flows["in"]
        return {port: Q_in * jnp.asarray(ratio)
                for port, ratio in self.output_port_ratios.items()}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.zeros((0,))
