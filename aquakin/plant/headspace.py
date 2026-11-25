"""An anaerobic digester whose gas headspace belongs to the unit, not the model.

:class:`HeadspaceDigesterUnit` runs any kinetic model that declares its
dissolved gases in a ``gases:`` block (:mod:`aquakin.core.gases`). The model
stays liquid-only, as it is in every other unit; the digester adds one
headspace state per declared gas and does the gas-liquid transfer and the
biogas outflow itself, in the form of BSM2 (Rosen & Jeppsson 2006):

    transfer_i     = kLa * (C_i,volatile - H_i(T) * p_i)       (per m3 of liquid)
    dC_i/dt        = reaction_i + Q/V_liq (C_in,i - C_i) - transfer_i
    dG_i/dt        = transfer_i * V_liq / V_gas - q_gas / V_gas * G_i
    q_gas          = k_p * max(0, P_gas - P_atm),   P_gas = sum_i p_i + p_H2O(T)

with ``G_i`` the gas-phase concentration in the dissolved species' own units
per m3 of gas, ``p_i`` its ideal-gas partial pressure, and one ``kLa`` for every
gas, as in BSM2. The headspace is part of the unit's state, so the plant
integrates it with everything else; the biogas leaving is reported to the mass
balance through :meth:`HeadspaceDigesterUnit.gas_export`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from aquakin.core.gases import (
    partial_pressure,
    saturation,
    temperature_and_pH,
    volatile_concentration,
    water_vapour_pressure,
)
from aquakin.plant._constants import PA_PER_BAR, check_positive
from aquakin.plant.coupling import CouplingAware
from aquakin.plant.kinetic import KineticUnit
from aquakin.plant.streams import Stream, mixed_feed, mixed_scalars, total_flow
from aquakin.plant.units import Biogas

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aquakin.core.model import CompiledModel


@dataclass
class HeadspaceDigesterUnit(KineticUnit, CouplingAware):
    """A continuously fed digester at a fixed temperature, with a gas headspace.

    Parameters
    ----------
    name : str
        Unit name.
    model : CompiledModel
        A kinetic model that declares its dissolved gases (``gases:``).
    volume : float
        Liquid volume ``V_liq`` (m³).
    gas_volume : float
        Headspace volume ``V_gas`` (m³). BSM2: 300.
    kla : float
        Gas-liquid transfer coefficient for every declared gas (1/d). BSM2: 200.
    k_p : float
        Biogas outflow coefficient (m³/(d·bar)), per bar of headspace
        overpressure. BSM2: 5e4.
    p_atm : float
        Pressure the biogas vents to (bar). BSM2: 1.013.
    input_port_names : list[str]
        Incoming-stream ports; several inflows are mixed by flow.
    conditions : dict[str, float]
        Condition values for the model, the temperature (and, for a model whose
        pH is a condition, the pH) among them.
    output_port : str
        Liquid effluent port.
    """

    name: str
    model: CompiledModel
    volume: float
    gas_volume: float = 300.0
    kla: float = 200.0
    k_p: float = 5.0e4
    p_atm: float = 1.013
    input_port_names: list[str] = field(default_factory=lambda: ["inlet"])
    conditions: dict[str, float] = field(default_factory=dict)
    output_port: str = "effluent"

    # Heated and held at its temperature, like the ADM1 digester.
    temperature_fixed = True

    def __post_init__(self) -> None:
        for setting in ("volume", "gas_volume", "kla", "k_p", "p_atm"):
            check_positive(self.name, setting, getattr(self, setting))
        if self.model.gases is None:
            raise ValueError(
                f"HeadspaceDigesterUnit '{self.name}': model '{self.model.name}' declares no "
                "dissolved gases; add a 'gases:' block to the model."
            )
        defaults = {
            name: float(arr[0]) for name, arr in self.model.default_conditions().fields.items()
        }
        self._init_conditions({**defaults, **self.conditions})
        self._gases = self.model.gases.entries
        self._gas_index = jnp.asarray([g.index for g in self._gases])

    # --- layout ----------------------------------------------------------------

    @property
    def n_species(self) -> int:
        return self.model.n_species

    @property
    def state_size(self) -> int:
        return self.model.n_species + len(self._gases)

    def state_names(self) -> tuple[str, ...]:
        """The species, then ``gas.<species>`` for each headspace state."""
        return tuple(self.model.species) + tuple(f"gas.{g.species}" for g in self._gases)

    @property
    def input_ports(self) -> list[str]:
        return list(self.input_port_names)

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port]

    def initial_state(self) -> jnp.ndarray:
        """The model's default concentrations and an empty headspace."""
        return jnp.concatenate([self.model.default_concentrations(), jnp.zeros(len(self._gases))])

    # --- gas phase -------------------------------------------------------------

    def _partial_pressures(self, G: jnp.ndarray, T) -> jnp.ndarray:
        """Partial pressure (Pa) of each headspace gas."""
        return jnp.stack([partial_pressure(g, T, G[k]) for k, g in enumerate(self._gases)])

    def _temperature(self):
        """The headspace temperature (K): the digester's fixed temperature."""
        return self.model.gases.kelvin(
            self._condition_arrays[self.model.gases.temperature_field][0]
        )

    def _vented(self, pressure):
        """Biogas flow (m³ per plant time unit) at headspace pressure ``pressure`` (Pa)."""
        return self.k_p * jnp.maximum(0.0, pressure - self.p_atm * PA_PER_BAR) / PA_PER_BAR

    def gas_flow(self, state: jnp.ndarray) -> jnp.ndarray:
        """Biogas flow leaving the headspace (m³ per plant time unit)."""
        return self.biogas(state).flow

    def biogas(self, state: jnp.ndarray) -> Biogas:
        """The biogas the headspace vents at ``state``
        (:class:`~aquakin.plant.units.BiogasHeadspaceUnit`).

        Parameters
        ----------
        state : jnp.ndarray
            The unit's state, shape ``(state_size,)``.

        Returns
        -------
        Biogas
            Its outflow, total and vent pressures, temperature and the partial
            pressure of each declared gas.
        """
        T = self._temperature()
        p = self._partial_pressures(state[self.n_species :], T)
        pressure = jnp.sum(p) + water_vapour_pressure(T)
        return Biogas(
            flow=self._vented(pressure),
            pressure=pressure,
            vent_pressure=self.p_atm * PA_PER_BAR,
            temperature=T,
            partial_pressures={g.gas: p[k] for k, g in enumerate(self._gases)},
        )

    def _transfer(self, C: jnp.ndarray, G: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Gas-liquid transfer of each declared gas, per m³ of liquid per plant
        time unit."""
        T, pH = temperature_and_pH(self.model, C, params, self._condition_arrays)
        p = self._partial_pressures(G, T)
        driving = jnp.stack(
            [
                volatile_concentration(g, C[g.index], T, pH) - saturation(g, T, p[k])
                for k, g in enumerate(self._gases)
            ]
        )
        return self.kla * driving

    def gas_export(self, state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """What the biogas carries out of the unit, per species, per plant time
        unit: shape ``(n_species,)``, in each species' units times m³.

        Indexed like the model's species, so a species' component content (COD,
        N, ...) applies to its entry; zero for species that are not gases.
        """
        G = state[self.n_species :]
        q = self.gas_flow(state)
        return jnp.zeros(self.n_species).at[self._gas_index].add(q * G)

    # --- the Unit contract -----------------------------------------------------

    def _reaction_conditions(self, inputs: dict[str, Stream], signals: dict | None) -> dict:
        """The static conditions: the heated digester runs at its fixed
        temperature, never a plant temperature."""
        return self._condition_arrays

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        n = self.n_species
        C, G = state[:n], state[n:]
        Q_total, C_in = mixed_feed(inputs, self.input_port_names)
        transfer = self._transfer(C, G, params)
        q_gas = self.gas_flow(state)
        dC = (
            self._reaction(C, params, self._reaction_conditions(inputs, signals))
            + (Q_total / self.volume) * (C_in - C)
            - jnp.zeros(n).at[self._gas_index].add(transfer)
        )
        dG = transfer * self.volume / self.gas_volume - q_gas / self.gas_volume * G
        return jnp.concatenate([dC, dG])

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict[str, Stream]:
        Q_total = total_flow(inputs[name].Q for name in self.input_port_names)
        scalars_out = mixed_scalars(inputs, self.input_port_names)
        return {
            self.output_port: Stream(
                Q=Q_total, C=state[: self.n_species], model=self.model, scalars=scalars_out
            )
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        return {self.output_port: total_flow(input_flows[name] for name in self.input_port_names)}

    def component_inventory(self, state, content, params):
        """Canonical-component inventory (``{component: grams}``): the liquid at
        ``V_liq`` and the headspace at ``V_gas``, each gas valued with the
        content of the species it is carried as."""
        n = self.n_species
        sv = np.asarray(state)
        gas_species = np.zeros(n)
        np.add.at(gas_species, np.asarray(self._gas_index), sv[n:])
        held = sv[:n] * float(self.volume) + gas_species * float(self.gas_volume)
        return {comp: float(np.dot(held, vec)) for comp, vec in content.items()}

    def coupling_pattern(self):
        """Structural sparsity: the model's kinetics, plus the gas phase.

        A headspace state depends on every species (the transfer's pH can come
        from the whole charge balance) and on every headspace state (through the
        total pressure); a gas-carrying species depends on all of them too.
        """
        from aquakin.integrate.colored_jacobian import structural_sparsity_pattern
        from aquakin.plant.coupling import CouplingPattern

        n, m = self.n_species, len(self._gases)
        size = n + m
        pattern = np.zeros((size, size), dtype=bool)
        pattern[:n, :n] = structural_sparsity_pattern(self.model)
        gas_rows = [g.index for g in self._gases] + list(range(n, size))
        pattern[gas_rows, :] = True
        inlet = np.zeros((size, n), dtype=bool)
        inlet[np.arange(n), np.arange(n)] = True
        output = np.zeros((n, size), dtype=bool)
        output[np.arange(n), np.arange(n)] = True
        return CouplingPattern(self_pattern=pattern, inlet_pattern=inlet, output_pattern=output)
