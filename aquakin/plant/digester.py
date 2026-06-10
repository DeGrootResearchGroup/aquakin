"""ADM1 anaerobic digester unit for plant-wide (BSM2) simulation.

A continuously-fed, constant-volume mesophilic digester running the ``adm1``
network. It is a CSTR on the *liquid* phase with a gas headspace:

    dC/dt = f_reaction(C) + (Q_in / V_liq) * (C_in - C) * liquid_mask

The reaction term ``network.dCdt`` already contains the ADM1 biochemistry, the
gas–liquid transfer and the overpressure-driven biogas outflow, and the
state-derived charge-balance pH. The dilution term applies only to the liquid
states; the three gas-headspace states (``S_gas_*``) are not diluted by the
feed (``liquid_mask`` is 0 there) — they exchange with the liquid and leave as
biogas purely through the reaction network. The liquid effluent leaves at the
feed flow (constant liquid volume).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


# ADM1 gas-headspace states (BSM2 form): not part of the liquid throughput.
ADM1_GAS_SPECIES: tuple[str, ...] = ("S_gas_h2", "S_gas_ch4", "S_gas_co2")


@dataclass
class ADM1DigesterUnit:
    """Continuously-fed ADM1 digester (BSM2 mesophilic, 35 °C).

    Parameters
    ----------
    name : str
    network : CompiledNetwork
        The ``adm1`` network (26 liquid + 3 gas states).
    volume : float
        Liquid volume ``V_liq`` (m³); the dilution rate is ``Q_in / V_liq``.
    input_port_names : list[str]
        Incoming-stream ports; multiple inflows are Q-weighted-mixed.
    conditions : dict[str, float]
        Spatially-uniform condition values (e.g. ``{"T": 308.15}``). Defaults to
        the network's declared condition defaults. The digester pH is *derived*
        from the state (charge balance) and must not be supplied here.
    gas_species : tuple[str, ...]
        Headspace states excluded from feed dilution.
    output_port : str
        Liquid-effluent output port.
    """

    name: str
    network: "CompiledNetwork"
    volume: float
    input_port_names: list[str] = field(default_factory=lambda: ["inlet"])
    conditions: dict[str, float] = field(default_factory=dict)
    gas_species: tuple[str, ...] = ADM1_GAS_SPECIES
    output_port: str = "effluent"

    def __post_init__(self) -> None:
        # Fill in any unspecified required conditions from the network defaults.
        defaults = {
            name: float(arr[0])
            for name, arr in self.network.default_conditions().fields.items()
        }
        conds = {**defaults, **self.conditions}
        missing = set(self.network.conditions_required) - set(conds)
        if missing:
            raise ValueError(
                f"ADM1DigesterUnit '{self.name}' is missing condition values "
                f"for: {sorted(missing)}."
            )
        self._condition_arrays = {
            name: jnp.asarray([float(conds[name])])
            for name in self.network.conditions_required
        }
        # liquid_mask: 1.0 on liquid states, 0.0 on the gas-headspace states.
        mask = jnp.ones((self.network.n_species,))
        for sp in self.gas_species:
            if sp in self.network.species_index:
                mask = mask.at[self.network.species_index[sp]].set(0.0)
        self._liquid_mask = mask

    @property
    def state_size(self) -> int:
        return self.network.n_species

    @property
    def input_ports(self) -> list[str]:
        return list(self.input_port_names)

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port]

    def initial_state(self) -> jnp.ndarray:
        return self.network.default_concentrations()

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> dict[str, Stream]:
        # Constant liquid volume: effluent flow equals the total inflow.
        Q_total = jnp.zeros(())
        for name in self.input_port_names:
            Q_total = Q_total + inputs[name].Q
        return {self.output_port: Stream(Q=Q_total, C=state, network=self.network)}

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray) -> dict:
        Q_total = jnp.zeros(())
        for name in self.input_port_names:
            Q_total = Q_total + input_flows[name]
        return {self.output_port: Q_total}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        # Mix inflows (Q-weighted) into one feed composition.
        Q_total = jnp.zeros(())
        mass_total = jnp.zeros((self.network.n_species,))
        for name in self.input_port_names:
            s = inputs[name]
            Q_total = Q_total + s.Q
            mass_total = mass_total + s.Q * s.C
        C_in = mass_total / (Q_total + 1e-12)

        reaction = self.network.dCdt(state, params, self._condition_arrays, 0)
        dilution = (Q_total / self.volume) * (C_in - state) * self._liquid_mask
        return reaction + dilution
