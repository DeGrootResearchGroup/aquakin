"""Continuous-flow stirred-tank reactor (CSTR) with kinetics + aeration.

The CSTR is the workhorse unit for activated-sludge plants: each ASM tank
in BSM1 is a CSTR. The mass-balance equation for species i is::

    dC_i/dt = (Q_in / V) * (C_in,i - C_i)        # convection
              + S^T r(C, p, conditions)[i]       # chemistry
              + kLa_i * (C_sat,i - C_i)          # mass transfer (DO in aerobic)

Per-species ``kLa`` and ``C_sat`` make it trivial to model both anoxic
tanks (kLa=0 everywhere) and aerobic tanks (kLa_DO > 0) with the same
unit class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


@dataclass
class CSTRUnit:
    """A single continuous-flow stirred tank with kinetics + aeration.

    Parameters
    ----------
    name : str
        Unit identifier.
    network : CompiledNetwork
        Kinetic network whose rate functions provide the chemistry term.
    volume : float
        Tank liquid volume.
    input_port_names : list[str]
        Names of the incoming-stream ports. Multiple inflows are summed
        (treated as a built-in mixer) before the mass balance.
    conditions : dict[str, float]
        Spatially-uniform condition values (e.g. ``{"T": 293.15}``). One
        value per condition declared by the network.
    kla : dict[str, float], optional
        Per-species ``kLa`` mass-transfer coefficients ``(1/time)``.
        Defaults to no aeration on any species. Typically only ``"SO"``
        is set for ASM1 / ASM2D / ASM3.
    C_sat : dict[str, float], optional
        Per-species saturation concentrations used in the aeration term.
        Defaults to 0 for any species without an entry.
    controlled_kla : dict[str, tuple[str, float]], optional
        Maps a species to ``(signal_name, gain)``: under closed-loop control its
        ``kLa`` is taken from the control signal ``signal_name`` (times ``gain``)
        each RHS call instead of the fixed ``kla`` entry, read from the
        ``signals`` bus the plant threads into every unit's ``rhs``. Used for
        DO/kLa control of the aerobic ASM tanks.
    output_port : str
        Name of the single output port.
    """

    name: str
    network: "CompiledNetwork"
    volume: float
    input_port_names: list[str]
    conditions: dict[str, float] = field(default_factory=dict)
    kla: dict[str, float] = field(default_factory=dict)
    C_sat: dict[str, float] = field(default_factory=dict)
    controlled_kla: dict[str, tuple[str, float]] = field(default_factory=dict)
    output_port: str = "out"

    def __post_init__(self) -> None:
        missing = set(self.network.conditions_required) - set(self.conditions)
        if missing:
            raise ValueError(
                f"CSTRUnit '{self.name}' is missing required condition values "
                f"for: {sorted(missing)}. Provided: {sorted(self.conditions)}"
            )
        for sp in self.kla:
            if sp not in self.network.species_index:
                raise ValueError(
                    f"CSTRUnit '{self.name}' kla refers to unknown species '{sp}'"
                )
        for sp in self.C_sat:
            if sp not in self.network.species_index:
                raise ValueError(
                    f"CSTRUnit '{self.name}' C_sat refers to unknown species '{sp}'"
                )
        for sp in self.controlled_kla:
            if sp not in self.network.species_index:
                raise ValueError(
                    f"CSTRUnit '{self.name}' controlled_kla refers to unknown "
                    f"species '{sp}'"
                )

        # Precompute (n_species,) aeration arrays once. These vectors are
        # used inside rhs() and don't change per integration step.
        kla_vec = jnp.zeros((self.network.n_species,))
        sat_vec = jnp.zeros((self.network.n_species,))
        for sp, val in self.kla.items():
            kla_vec = kla_vec.at[self.network.species_index[sp]].set(float(val))
        for sp, val in self.C_sat.items():
            sat_vec = sat_vec.at[self.network.species_index[sp]].set(float(val))
        self._kla_vec = kla_vec
        self._sat_vec = sat_vec

        # Condition arrays for the kinetics call: each declared condition
        # broadcast to a length-1 array so the rate functions index with
        # loc_idx=0 — same convention as BatchReactor.
        self._condition_arrays = {
            name: jnp.asarray([float(self.conditions[name])])
            for name in self.network.conditions_required
        }

    @property
    def state_size(self) -> int:
        return self.network.n_species

    def set_temperature(self, temperature_K: float) -> None:
        """Set this reactor's static operating temperature (Kelvin).

        Updates the ``T`` condition (and its precomputed rate-evaluation array)
        in place, so a re-solve runs the kinetics -- including any Arrhenius
        ``temperature_corrections`` -- at the new temperature. A no-op for a
        network that declares no ``T`` condition. The plant clears its compiled-
        solve cache after calling this; on a bare unit, rebuild any cached solve.
        """
        if "T" not in self.network.conditions_required:
            return
        self.conditions = {**self.conditions, "T": float(temperature_K)}
        self._condition_arrays = {
            **self._condition_arrays, "T": jnp.asarray([float(temperature_K)])}

    @property
    def input_ports(self) -> list[str]:
        return list(self.input_port_names)

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port]

    def initial_state(self) -> jnp.ndarray:
        return self.network.default_concentrations()

    def _mixed_inlet_T(self, inputs: dict[str, Stream]):
        """Flow-weighted inlet temperature, or ``None`` if any inlet is
        temperature-agnostic. The well-mixed reactor is taken to be at this
        temperature (no thermal lag — the hydraulic retention is hours, far
        shorter than the seasonal temperature variation)."""
        if not all(inputs[n].T is not None for n in self.input_port_names):
            return None
        Q_total = jnp.zeros(())
        heat = jnp.zeros(())
        for name in self.input_port_names:
            s = inputs[name]
            Q_total = Q_total + s.Q
            heat = heat + s.Q * s.T
        return heat / (Q_total + 1e-12)

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> dict[str, Stream]:
        # Total inflow Q. The outflow equals the total inflow (constant
        # volume assumption; no accumulation of water).
        Q_total = jnp.zeros(())
        for name in self.input_port_names:
            Q_total = Q_total + inputs[name].Q
        return {
            self.output_port: Stream(
                Q=Q_total, C=state, network=self.network,
                T=self._mixed_inlet_T(inputs),
            )
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Outflow equals total inflow (constant-volume reactor)."""
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
        signals: dict | None = None,
    ) -> jnp.ndarray:
        # Mix inflows (Q-weighted).
        Q_total = jnp.zeros(())
        mass_total = jnp.zeros((self.network.n_species,))
        for name in self.input_port_names:
            s = inputs[name]
            Q_total = Q_total + s.Q
            mass_total = mass_total + s.Q * s.C
        C_in = mass_total / (Q_total + 1e-12)

        # Convection.
        convection = (Q_total / self.volume) * (C_in - state)

        # Chemistry. If the inflow carries a temperature and the network uses a
        # 'T' condition, the reactor runs at that (flow-weighted) temperature --
        # so temperature-dependent kinetics track the influent through the
        # season; otherwise the static condition value is used.
        conditions = self._condition_arrays
        T_in = self._mixed_inlet_T(inputs)
        if T_in is not None and "T" in self._condition_arrays:
            conditions = {**self._condition_arrays, "T": jnp.reshape(T_in, (1,))}
        stoich = self.network.compute_stoich(params)
        rates = self.network.rates(state, params, conditions, 0)
        chemistry = stoich.T @ rates

        # Aeration (mass transfer). Zero on species without a kLa entry. Under
        # closed-loop control the kLa of a controlled species is overridden by
        # its control signal (times the per-tank gain).
        kla_vec = self._kla_vec
        if self.controlled_kla and signals is not None:
            for sp, (signal_name, gain) in self.controlled_kla.items():
                idx = self.network.species_index[sp]
                kla_vec = kla_vec.at[idx].set(signals[signal_name] * gain)
        aeration = kla_vec * (self._sat_vec - state)

        return convection + chemistry + aeration
