"""Hydraulic delay: a first-order lag on a stream's flow and load.

A length of pipe or channel between unit operations delays and smooths the flow
and the pollutant load passing through it. The BSM2 hydraulic-delay element
models this as a first-order lag with a fixed time constant ``tau`` on the
**load** (mass flow ``Q*C``) and the **flow** ``Q`` -- not a fixed-volume tank,
whose residence time would vary with flow::

    d(Q*C_i)/dt = (Q_in*C_in,i - Q*C_i) / tau     # held load relaxes to inlet load
    dQ/dt       = (Q_in - Q) / tau                # held flow relaxes to inlet flow

so the outlet concentration is the lagged load over the lagged flow,
``C_i = (Q*C_i) / Q``. At steady state ``Q -> Q_in`` and ``C -> C_in`` (a pass
through); a flow or load pulse emerges delayed and rounded with time constant
``tau``.

The state is ``[load_0..load_{n-1}, Q]`` (the per-species loads plus the flow).
Because the *outlet flow* is the held-flow state, :meth:`flow_outputs` reads it
from the :class:`~aquakin.plant.units.FlowContext` the plant passes when
resolving the flow network.

The BSM2 reference uses ``tau`` ~ 1e-4 d -- a near-instantaneous lag whose role
is to break algebraic loops in a sequential-modular solver. ``aquakin`` resolves
recycles directly in one monolithic solve, so it does not need the delay for
that; the unit is here to model a *physical* transport delay (set ``tau`` to the
real hydraulic residence time) and to complete the BSM2 element set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant._constants import EPS_Q
from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


@dataclass
class HydraulicDelayUnit:
    """A first-order hydraulic lag on a stream's flow and load.

    Parameters
    ----------
    name : str
        Unit identifier.
    model : CompiledModel
        Model of the passed stream.
    tau : float
        Lag time constant (days). Smaller is faster (less delay).
    initial_flow : float, optional
        Initial held flow ``Q`` (m³/d) seeding the state. Default 0. **Set this
        to the expected influent flow** for a steady start: the outlet flow IS
        the held-flow state, so a default of 0 makes the unit emit no flow until
        ~``tau`` has elapsed -- a non-physical no-flow transient that starves the
        whole downstream plant on a cold start. ``build_bsm2(hydraulic_delay=...)``
        seeds it with the reference flow for exactly this reason.
    initial_concentrations : jnp.ndarray, optional
        Initial held concentrations, shape ``(n_species,)``. Defaults to the
        model's default concentrations.
    input_port, output_port : str, optional
        Port names (default ``"in"`` / ``"out"``).
    """

    name: str
    model: CompiledModel
    tau: float
    initial_flow: float = 0.0
    initial_concentrations: jnp.ndarray | None = None
    input_port: str = "in"
    output_port: str = "out"

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError(f"HydraulicDelayUnit '{self.name}': tau must be > 0; got {self.tau}")

    @property
    def state_size(self) -> int:
        return self.model.n_species + 1  # per-species loads + flow

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port]

    def initial_state(self) -> jnp.ndarray:
        C0 = (
            self.model.default_concentrations()
            if self.initial_concentrations is None
            else jnp.asarray(self.initial_concentrations)
        )
        Q0 = jnp.asarray(float(self.initial_flow))
        return jnp.concatenate([Q0 * C0, jnp.reshape(Q0, (1,))])

    def _flow_and_conc(self, state: jnp.ndarray):
        """Return ``(Q, C)`` from the held loads and flow."""
        Q = state[-1]
        loads = state[: self.model.n_species]
        C = loads / (Q + EPS_Q)
        return Q, C

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict[str, Stream]:
        Q, C = self._flow_and_conc(state)
        # Side-channel scalars pass straight through (T(out)=T(in), etc.).
        return {
            self.output_port: Stream(
                Q=Q, C=C, model=self.model, scalars=inputs[self.input_port].scalars
            )
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """The outlet flow is the held-flow state (the delayed flow), read from
        the unit's own state in ``ctx``."""
        return {self.output_port: jnp.asarray(ctx.state)[-1]}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        s_in = inputs[self.input_port]
        loads = state[: self.model.n_species]
        Q = state[-1]
        inv_tau = 1.0 / float(self.tau)
        dloads = (s_in.Q * s_in.C - loads) * inv_tau
        dQ = jnp.reshape((s_in.Q - Q) * inv_tau, (1,))
        return jnp.concatenate([dloads, dQ])
