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


@dataclass(frozen=True)
class Aeration:
    """Aeration / dissolved-oxygen spec for a :class:`CSTRUnit`.

    The quantity a designer actually thinks in, instead of a raw mass-transfer
    coefficient on a state variable. Choose exactly one mode.

    **Open loop** -- a fixed mass-transfer coefficient (the saturation defaults
    sensibly, so the common case is one number)::

        Aeration(kla=120)                 # do_sat defaults to 8.0 gO2/m3
        Aeration(kla=120, do_sat=9.0)

    **Closed loop** -- a dissolved-oxygen setpoint. The plant auto-wires a PI
    controller that manipulates this tank's kLa to hold the setpoint::

        Aeration(do_setpoint=2.0)         # this tank holds DO = 2.0 on its own

    Several tanks share one controller by giving the same ``controller`` id -- the
    BSM2 design, one sensor driving several reactors at per-tank gains::

        Aeration(do_setpoint=2.0, controller="do", sensor="tank4", gain=1.0)
        Aeration(do_setpoint=2.0, controller="do", sensor="tank4", gain=0.5)

    Parameters
    ----------
    kla : float, optional
        Open-loop mass-transfer coefficient (1/time). Mutually exclusive with
        ``do_setpoint``.
    do_setpoint : float, optional
        Closed-loop dissolved-oxygen target (same units as ``species``).
    do_sat : float
        Saturation concentration in the aeration term ``kLa*(do_sat - C)``.
        Default 8.0 (gO2/m3, the usual ASM oxygen saturation).
    species : str
        The aerated species (default ``"SO"``).
    controller : str, optional
        Closed-loop only. Shared-controller id: tanks giving the same id share one
        PI controller (and the controller unit takes this name). ``None`` (the
        default) gives this tank its own dedicated controller.
    sensor : str, optional
        Closed-loop only. Name of the unit whose oxygen the controller measures.
        Defaults to the controlled tank itself (per-tank control).
    gain : float
        Closed-loop only. This tank's share of the controller's kLa output
        (default 1.0), so one shared controller can drive tanks at different
        rates.
    Kp, Ti, Tt, kla_offset, kla_min, kla_max : float
        Closed-loop PI tuning and output bounds. Defaults are the BSM2 DO loop:
        Kp=25, Ti=0.002 d, Tt=0.001 d, offset 120, bounds [0, 360] d^-1.
    """

    kla: float | None = None
    do_setpoint: float | None = None
    do_sat: float = 8.0
    species: str = "SO"
    controller: str | None = None
    sensor: str | None = None
    gain: float = 1.0
    Kp: float = 25.0
    Ti: float = 0.002
    Tt: float = 0.001
    kla_offset: float = 120.0
    kla_min: float = 0.0
    kla_max: float = 360.0

    def __post_init__(self) -> None:
        n_modes = (self.kla is not None) + (self.do_setpoint is not None)
        if n_modes != 1:
            raise ValueError(
                "Aeration requires exactly one of kla= (open loop) or "
                "do_setpoint= (closed loop)."
            )
        if self.kla is not None and self.kla < 0.0:
            raise ValueError(f"Aeration kla must be >= 0, got {self.kla}.")

    @property
    def is_closed_loop(self) -> bool:
        return self.do_setpoint is not None

    def controller_id(self, unit_name: str) -> str:
        """The id that groups tanks onto one controller: the shared ``controller``
        if given, else the tank's own name (a dedicated per-tank controller)."""
        return self.controller if self.controller is not None else unit_name

    def signal_name(self, unit_name: str) -> str:
        """The control-signal name the controlled tank reads and its controller
        publishes. Derived from the controller id so a shared controller's tanks
        all resolve to the same signal."""
        return _aeration_signal_name(self.controller_id(unit_name))


def _aeration_signal_name(controller_id: str) -> str:
    return f"_aer_{controller_id}_kla"


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
    aeration : Aeration, optional
        How the tank is aerated. ``None`` (default) is an anoxic/anaerobic tank
        (no aeration). ``Aeration(kla=120)`` is open-loop aeration at a fixed
        mass-transfer coefficient; ``Aeration(do_setpoint=2.0)`` is closed-loop
        dissolved-oxygen control, for which the plant auto-wires a PI controller
        on this tank's kLa (see :class:`Aeration`).
    output_port : str
        Name of the single output port.
    """

    name: str
    network: "CompiledNetwork"
    volume: float
    input_port_names: list[str]
    conditions: dict[str, float] = field(default_factory=dict)
    aeration: "Aeration | None" = None
    output_port: str = "out"

    def __post_init__(self) -> None:
        missing = set(self.network.conditions_required) - set(self.conditions)
        if missing:
            raise ValueError(
                f"CSTRUnit '{self.name}' is missing required condition values "
                f"for: {sorted(missing)}. Provided: {sorted(self.conditions)}"
            )
        # Translate the Aeration spec into the per-species vectors the RHS uses:
        # a fixed-kLa vector, a saturation vector, and -- for closed-loop control
        # -- a map species -> (signal_name, gain) read off the signal bus.
        aer = self.aeration
        controlled: dict[str, tuple[str, float]] = {}
        kla_vec = jnp.zeros((self.network.n_species,))
        sat_vec = jnp.zeros((self.network.n_species,))
        if aer is not None:
            if aer.species not in self.network.species_index:
                raise ValueError(
                    f"CSTRUnit '{self.name}' aeration species '{aer.species}' "
                    f"is not in the network."
                )
            idx = self.network.species_index[aer.species]
            sat_vec = sat_vec.at[idx].set(float(aer.do_sat))
            if aer.is_closed_loop:
                controlled[aer.species] = (aer.signal_name(self.name), aer.gain)
            else:
                kla_vec = kla_vec.at[idx].set(float(aer.kla))
        self._controlled_kla = controlled
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

    @property
    def required_signals(self) -> tuple[str, ...]:
        """Control-signal names this unit reads from the bus in ``rhs`` (the
        closed-loop aeration signal, if any). The plant validates these are
        published -- by the controller it auto-wires from the ``Aeration`` spec --
        before solving."""
        return tuple(
            signal_name for signal_name, _gain in self._controlled_kla.values()
        )

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
        if self._controlled_kla and signals is not None:
            for sp, (signal_name, gain) in self._controlled_kla.items():
                idx = self.network.species_index[sp]
                kla_vec = kla_vec.at[idx].set(signals[signal_name] * gain)
        aeration = kla_vec * (self._sat_vec - state)

        return convection + chemistry + aeration
