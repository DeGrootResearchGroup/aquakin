"""Diffuser / blower aeration-design physics layered on the kLa abstraction.

The kinetic model aerates a tank through a per-species mass-transfer coefficient
``kLa`` (see :class:`~aquakin.plant.cstr.Aeration`), and the Copp-2002 benchmark
scores aeration *energy* with a fixed correlation ``AE ∝ Σ V_i·kLa_i``. Real
aeration sizing and energy benchmarking need the blower/diffuser physics behind
that ``kLa``: how much **air** must be blown to produce it, and the **power** to
compress that air against the submergence head.

:class:`AerationSystem` is that design, kept **standalone** -- it does not change
the kinetic ``kLa`` interface. Given the ``kLa`` a solve produced (per tank, over
time) it computes:

* the **standard oxygen transfer rate** the air must deliver,
  ``SOTR = kLa · C_s,std · V`` (the clean-water transfer at zero DO that defines
  the diffuser airflow -- a given ``kLa`` needs a given airflow, independent of
  the operating DO deficit);
* the **airflow** ``Q_air = SOTR / (SOTE · o2_per_air)`` from the diffuser's
  standard oxygen-transfer efficiency ``SOTE`` (which rises with submergence);
* the blower **discharge pressure** ``p_atm + ρ_w·g·depth + headloss`` from the
  diffuser submergence; and
* the blower **power** by adiabatic compression,
  ``P = (Q·p1/η)·(γ/(γ−1))·[(p2/p1)^((γ−1)/γ) − 1]``.

Because the blower power is linear in airflow and airflow is linear in ``kLa``,
the aeration **energy** ``∫ Σ_i P_i dt`` has the same form as the Copp
correlation but with a mechanistic coefficient (SOTE / depth / blower efficiency)
in place of the fixed one -- so it is a principled refinement, and it stays
``jit`` / ``jax.grad`` clean. The α / β / temperature field corrections live on
:class:`~aquakin.plant.cstr.Aeration` (they shape the ``kLa`` and the driving
force in the solve); ``AerationSystem`` adds the diffuser-fouling factor ``F`` on
the standard efficiency and the blower curve.

All quantities are in days / metres / kPa / kW so they compose with the plant's
time unit and the existing energy kernels (kWh/d).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

# Standard physical constants (SI).
_RHO_WATER = 1000.0       # kg/m3
_G = 9.80665              # m/s2
_SECONDS_PER_DAY = 86400.0
_HOURS_PER_DAY = 24.0


@dataclass(frozen=True)
class AerationSystem:
    """Diffuser + blower design for the mechanistic aeration energy / airflow.

    Standalone: pass it to :func:`blower_energy` / :func:`design_summary`, or to
    ``evaluate_bsm1`` / ``evaluate_bsm2`` via ``aeration_system=`` (where it
    replaces the Copp aeration-energy term). It does **not** change the kinetic
    ``kLa`` interface.

    Parameters
    ----------
    depth : float
        Diffuser submergence (m). Sets both the standard transfer efficiency
        (when ``sote`` is derived from ``sote_per_meter``) and the blower
        discharge head.
    sote : float, optional
        Standard oxygen-transfer efficiency at ``depth`` (a fraction in
        ``(0, 1]``: the share of supplied oxygen that dissolves under standard
        conditions -- clean water, 20 °C, zero DO). If ``None`` (default) it is
        ``sote_per_meter · depth``.
    sote_per_meter : float
        Standard transfer efficiency per metre of submergence, used when ``sote``
        is ``None``. Default 0.06 (~6 %/m, typical fine-bubble diffusers).
    fouling_F : float
        Diffuser fouling factor multiplying the standard efficiency (``<= 1``;
        diffuser scaling/biofilm reduce transfer over time). Default 1.0.
    standard_do_sat : float
        Clean-water dissolved-oxygen saturation at the standard temperature
        (g O₂/m³). Default 9.09 (20 °C, 1 atm). The ``kLa·C_s,std·V`` standard
        transfer rate the airflow must deliver uses this.
    o2_per_air : float
        Mass of oxygen per cubic metre of supplied air at the blower's standard
        conditions (kg O₂/m³). Default 0.279 (air density ~1.204 kg/m³ at 20 °C ×
        0.2318 kg O₂/kg air).
    blower_efficiency : float
        Wire-to-air efficiency of the blower (motor × compression). Default 0.6.
    headloss_kpa : float
        Extra discharge head beyond the static submergence -- diffuser and piping
        losses (kPa). Default 0.
    p_atm_kpa : float
        Inlet (atmospheric) pressure (kPa). Default 101.325.
    gamma : float
        Ratio of specific heats for air, for the adiabatic compression. Default
        1.4.
    """

    depth: float
    sote: float | None = None
    sote_per_meter: float = 0.06
    fouling_F: float = 1.0
    standard_do_sat: float = 9.09
    o2_per_air: float = 0.279
    blower_efficiency: float = 0.6
    headloss_kpa: float = 0.0
    p_atm_kpa: float = 101.325
    gamma: float = 1.4

    def __post_init__(self) -> None:
        if self.depth <= 0.0:
            raise ValueError(f"AerationSystem depth must be > 0, got {self.depth}.")
        eff = self.effective_sote()
        if not (0.0 < eff <= 1.0):
            raise ValueError(
                f"AerationSystem effective SOTE must be in (0, 1], got {eff} "
                f"(sote={self.sote}, sote_per_meter={self.sote_per_meter}, "
                f"depth={self.depth}, fouling_F={self.fouling_F})."
            )
        for name in ("o2_per_air", "blower_efficiency", "standard_do_sat"):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"AerationSystem {name} must be > 0, got {getattr(self, name)}."
                )
        if self.gamma <= 1.0:
            raise ValueError(
                f"AerationSystem gamma must be > 1, got {self.gamma}."
            )

    def effective_sote(self) -> float:
        """The standard oxygen-transfer efficiency actually used: the declared
        ``sote`` (or ``sote_per_meter · depth``) reduced by the fouling factor."""
        base = self.sote if self.sote is not None else self.sote_per_meter * self.depth
        return base * self.fouling_F

    def discharge_pressure_kpa(self) -> float:
        """Blower discharge pressure: atmospheric + the static submergence head
        ``ρ_w·g·depth`` + the diffuser/piping ``headloss_kpa`` (kPa)."""
        static = _RHO_WATER * _G * self.depth / 1000.0   # Pa -> kPa
        return self.p_atm_kpa + static + self.headloss_kpa


def required_airflow(kla, volume, system: AerationSystem):
    """Air flow rate (m³/d) that produces ``kLa`` in a tank of ``volume`` m³.

    A given ``kLa`` is delivered by a given airflow: the standard oxygen transfer
    ``SOTR = kLa · C_s,std · V`` (g O₂/d, the clean-water transfer at zero DO)
    must equal ``SOTE · o2_per_air · Q_air``. So
    ``Q_air = SOTR / (SOTE · o2_per_air)``. The operating DO deficit does not
    enter -- it sets how much oxygen *dissolves* (the AOTR), not the air needed.

    Parameters
    ----------
    kla : float or jnp.ndarray
        Mass-transfer coefficient (1/d).
    volume : float or jnp.ndarray
        Tank liquid volume (m³).
    system : AerationSystem

    Returns
    -------
    jnp.ndarray
        Air flow rate (m³/d).
    """
    sotr_kg_per_d = jnp.asarray(kla) * system.standard_do_sat * jnp.asarray(volume) / 1000.0
    return sotr_kg_per_d / (system.effective_sote() * system.o2_per_air)


def blower_power_kw(airflow_m3_per_d, system: AerationSystem):
    """Blower shaft+motor power (kW) to compress ``airflow_m3_per_d`` from the
    inlet pressure to the diffuser discharge pressure (adiabatic).

    ``P = (Q·p1/η)·(γ/(γ−1))·[(p2/p1)^((γ−1)/γ) − 1]`` with ``Q`` in m³/s and the
    pressures in Pa, returned in kW. Linear in the airflow (the pressure ratio is
    fixed by the submergence), so AD-clean.
    """
    Q = jnp.asarray(airflow_m3_per_d) / _SECONDS_PER_DAY        # m3/s
    p1 = system.p_atm_kpa * 1000.0                              # Pa
    p2 = system.discharge_pressure_kpa() * 1000.0              # Pa
    n = (system.gamma - 1.0) / system.gamma
    watts = (Q * p1 / system.blower_efficiency) / n * ((p2 / p1) ** n - 1.0)
    return watts / 1000.0                                       # kW


def _time_average(values, t):
    """Trapezoidal mean of ``values`` over ``[t0, t1]`` (a single point returns
    itself -- the steady-state case)."""
    t = jnp.asarray(t)
    values = jnp.asarray(values)
    if t.shape[0] < 2:
        return values[0]
    return jnp.trapezoid(values, t, axis=0) / (t[-1] - t[0])


def blower_airflow_total(t, kla_history, volumes, system: AerationSystem) -> float:
    """Time-averaged total air flow over the window (m³/d), summed across tanks.

    ``kla_history`` is ``(n_t, n_tanks)`` and ``volumes`` is ``(n_tanks,)`` --
    the same arguments as :func:`aquakin.plant.metrics.aeration_energy`."""
    kla_history = jnp.asarray(kla_history)
    volumes = jnp.asarray(volumes)
    q = required_airflow(kla_history, volumes[None, :], system)  # (n_t, n_tanks)
    return float(_time_average(jnp.sum(q, axis=1), t))


def blower_energy(t, kla_history, volumes, system: AerationSystem) -> float:
    """Aeration energy (kWh/d) from the blower model, the mechanistic replacement
    for :func:`aquakin.plant.metrics.aeration_energy`.

    Sums the blower power across tanks at each save time and time-averages over
    the window, ×24 h to give energy per day. Same call signature as the Copp
    correlation (``t``, ``kla_history`` ``(n_t, n_tanks)``, ``volumes``
    ``(n_tanks,)``) so it is a drop-in.
    """
    kla_history = jnp.asarray(kla_history)
    volumes = jnp.asarray(volumes)
    q = required_airflow(kla_history, volumes[None, :], system)   # (n_t, n_tanks) m3/d
    power = blower_power_kw(q, system)                           # (n_t, n_tanks) kW
    total_power = jnp.sum(power, axis=1)                         # (n_t,) kW
    return float(_time_average(total_power, t) * _HOURS_PER_DAY)


@dataclass(frozen=True)
class AerationDesignPoint:
    """Sizing summary for one tank at one ``kLa`` (a design point)."""

    kla: float                  # 1/d
    volume: float               # m3
    sote: float                 # effective standard transfer efficiency (fraction)
    sotr: float                 # standard oxygen transfer rate (kg O2/d)
    airflow: float              # m3/d
    discharge_pressure: float   # kPa
    power: float                # kW

    def report(self) -> str:
        return (
            f"Aeration design point (kLa={self.kla:g} 1/d, V={self.volume:g} m3):\n"
            f"  SOTE                {self.sote * 100:7.2f} %\n"
            f"  SOTR                {self.sotr:10.1f} kg O2/d\n"
            f"  Air flow            {self.airflow:10.1f} m3/d "
            f"({self.airflow / 1440.0:.2f} m3/min)\n"
            f"  Discharge pressure  {self.discharge_pressure:10.2f} kPa\n"
            f"  Blower power        {self.power:10.2f} kW"
        )

    def __str__(self) -> str:  # pragma: no cover - thin delegation
        return self.report()


def design_summary(kla, volume, system: AerationSystem) -> AerationDesignPoint:
    """Size the air flow and blower power for one tank at a steady ``kLa``.

    The standalone sizing entry point: returns an :class:`AerationDesignPoint`
    with the effective SOTE, the standard oxygen transfer rate, the air flow, the
    blower discharge pressure and the blower power.
    """
    q = required_airflow(kla, volume, system)
    p = blower_power_kw(q, system)
    sotr = float(kla) * system.standard_do_sat * float(volume) / 1000.0
    return AerationDesignPoint(
        kla=float(kla), volume=float(volume), sote=system.effective_sote(),
        sotr=sotr, airflow=float(q),
        discharge_pressure=system.discharge_pressure_kpa(), power=float(p),
    )
