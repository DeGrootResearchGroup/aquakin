"""Variable-volume storage tank with a level-gated overflow bypass.

A storage (equalization) tank buffers a stream of variable flow: it stores
liquid up to a maximum volume and releases it at a controlled output flow,
smoothing the load it passes downstream. In BSM2 it sits on the reject-water
recycle line, holding the high-ammonia thickener/dewatering liquor so it can be
returned to the plant front at a controlled rate rather than as an
uncontrolled spike.

The tank is a completely-mixed, variable-volume CSTR with **no reactions**::

    dV/dt   = Q_in_stored - Q_out
    dC_i/dt = Q_in_stored / V * (C_in,i - C_i)

with a level-gated automatic bypass that protects the tank's volume limits:

- normal (``empty_frac < V/Vmax < full_frac``): store the inflow, release the
  requested output flow ``Q_out``;
- full and filling (``V >= full_frac*Vmax`` and ``Q_in > Q_out``): divert the
  whole inflow to the bypass outlet (do not overfill);
- full and draining (``Q_in <= Q_out``): behave normally (the tank is emptying);
- empty (``V <= empty_frac*Vmax``): stop releasing (``Q_out -> 0``) and just
  fill, so the tank cannot drain below its lower limit.

The two outlets (``out`` -- the released stream at tank concentration, and
``bypass`` -- the diverted inflow at inlet concentration) are recombined
downstream. The flow split depends on the tank's own liquid volume (a state),
so the unit declares ``flow_needs_state`` and the plant passes its state into
:meth:`flow_outputs` when resolving the flow network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp

from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork

_EPS_V = 1e-9  # guard the 1/V mixing term


@dataclass
class StorageTank:
    """A variable-volume storage tank with a level-gated overflow bypass.

    Parameters
    ----------
    name : str
        Unit identifier.
    network : CompiledNetwork
        Network of the stored / passed streams.
    volume : float
        Maximum (total) tank volume ``Vmax`` (m³).
    output_flow : float, optional
        Requested release flow ``Q_out`` (m³/d), the controlled pump-out rate.
        Default 0 -- with no release the tank fills and bypasses (the BSM2
        open-loop default).
    initial_fraction : float, optional
        Initial liquid volume as a fraction of ``volume`` (default 0.5).
    full_fraction, empty_fraction : float, optional
        Upper / lower safety levels as fractions of ``volume`` (defaults 0.9 /
        0.1) at which the bypass / fill-only behaviour engages.
    initial_concentrations : jnp.ndarray, optional
        Initial tank concentrations, shape ``(n_species,)``. Defaults to the
        network's default concentrations.
    input_port : str, optional
        Inlet port name (default ``"in"``).
    """

    name: str
    network: "CompiledNetwork"
    volume: float
    output_flow: float = 0.0
    initial_fraction: float = 0.5
    full_fraction: float = 0.9
    empty_fraction: float = 0.1
    initial_concentrations: Optional[jnp.ndarray] = None
    input_port: str = "in"

    # The overflow bypass is gated by the liquid level (a state), so the plant
    # must hand this unit its state when resolving the flow network.
    flow_needs_state = True

    @property
    def state_size(self) -> int:
        return self.network.n_species + 1  # concentrations + liquid volume

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return ["out", "bypass"]

    def initial_state(self) -> jnp.ndarray:
        C0 = (self.network.default_concentrations()
              if self.initial_concentrations is None
              else jnp.asarray(self.initial_concentrations))
        V0 = jnp.asarray([self.initial_fraction * float(self.volume)])
        return jnp.concatenate([C0, V0])

    def _flow_split(self, V: jnp.ndarray, Q_in: jnp.ndarray):
        """Return ``(Q_out, Q_bypass, Q_in_stored)`` from the level and inflow."""
        Vmax = float(self.volume)
        Q_req = jnp.asarray(float(self.output_flow))
        full = V >= self.full_fraction * Vmax
        empty = V <= self.empty_fraction * Vmax
        filling_full = full & (Q_in > Q_req)
        Q_bypass = jnp.where(filling_full, Q_in, 0.0)
        Q_out = jnp.where(empty | filling_full, 0.0, Q_req)
        Q_in_stored = jnp.where(filling_full, 0.0, Q_in)
        return Q_out, Q_bypass, Q_in_stored

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> dict[str, Stream]:
        s_in = inputs[self.input_port]
        C_tank = state[: self.network.n_species]
        V = state[-1]
        Q_out, Q_bypass, _ = self._flow_split(V, s_in.Q)
        return {
            # Released stream carries the (well-mixed) tank concentration.
            "out": Stream(Q=Q_out, C=C_tank, network=self.network, T=s_in.T),
            # Bypassed inflow passes straight through at its inlet concentration.
            "bypass": Stream(Q=Q_bypass, C=s_in.C, network=self.network, T=s_in.T),
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, state) -> dict:
        """Flow split from the inlet flow and the current liquid level."""
        V = jnp.asarray(state)[-1]
        Q_out, Q_bypass, _ = self._flow_split(V, input_flows[self.input_port])
        return {"out": Q_out, "bypass": Q_bypass}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        s_in = inputs[self.input_port]
        C_tank = state[: self.network.n_species]
        V = state[-1]
        Q_out, _, Q_in_stored = self._flow_split(V, s_in.Q)
        V_safe = jnp.maximum(V, _EPS_V)
        dC = Q_in_stored / V_safe * (s_in.C - C_tank)
        dV = jnp.reshape(Q_in_stored - Q_out, (1,))
        return jnp.concatenate([dC, dV])
