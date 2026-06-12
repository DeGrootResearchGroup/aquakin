"""Effluent quality, operational cost, and derived-quantity metrics.

The headline numbers BSM1 reports are:

- **EQI** (Effluent Quality Index) — weighted sum of total suspended
  solids, COD, BOD, TKN, and NO₃-N in the effluent, integrated over
  the simulation window. Lower is better.
- **OCI** (Operational Cost Index) — aeration energy + pumping energy +
  sludge production + mixing energy.

Both are defined in Copp 2002 / Alex 2008 with specific weighting factors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant._constants import ASM1_TSS_FACTOR, ASM1_TSS_SPECIES

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


# ASM1 → TSS conversion (Copp 2002): TSS = 0.75 * (X_S + X_I + X_BH + X_BA + X_P).
_TSS_FACTOR = ASM1_TSS_FACTOR
_TSS_SPECIES = ASM1_TSS_SPECIES

# EQI weighting factors (g pollutant / m³)⁻¹ from Copp 2002.
_EQI_WEIGHTS = {
    "TSS":  2.0,
    "COD":  1.0,
    "BOD":  2.0,
    "TKN": 20.0,  # total Kjeldahl nitrogen
    "NO":  20.0,  # nitrate-nitrogen
}


def _species_idx(network: "CompiledNetwork", names) -> jnp.ndarray:
    """Index array for the named species that exist in the network."""
    return jnp.asarray(
        [network.species_index[s] for s in names if s in network.species_index]
    )


# Every derived quantity below indexes ``C`` with ``C[..., i]`` (a scalar
# column) and, where it sums, sums over ``axis=-1``. That works for both a 1-D
# state vector ``(n_species,)`` -> scalar and a 2-D trajectory
# ``(n_t, n_species)`` -> vector, so no rank branch is needed.


def derived_TSS(C: jnp.ndarray, network: "CompiledNetwork") -> jnp.ndarray:
    """Total suspended solids from ASM1 particulate species.

    ``TSS = 0.75 × (XS + XI + XB_H + XB_A + XP)``. Scalar for 1-D ``C``, a
    leading-axis vector for 2-D ``C``.
    """
    return _TSS_FACTOR * jnp.sum(C[..., _species_idx(network, _TSS_SPECIES)], axis=-1)


def derived_COD(C: jnp.ndarray, network: "CompiledNetwork") -> jnp.ndarray:
    """Total COD = SI + SS + XI + XS + XB_H + XB_A + XP."""
    species = ("SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP")
    return jnp.sum(C[..., _species_idx(network, species)], axis=-1)


def derived_BOD(C: jnp.ndarray, network: "CompiledNetwork") -> jnp.ndarray:
    """BOD₅ proxy = 0.25 × (SS + XS + (1 - f_P) × (XB_H + XB_A))
    using Copp 2002 BOD relation with f_P ≈ 0.08."""
    f_P = 0.08
    i = network.species_index
    return 0.25 * (
        C[..., i["SS"]] + C[..., i["XS"]]
        + (1.0 - f_P) * (C[..., i["XB_H"]] + C[..., i["XB_A"]])
    )


def derived_TKN(C: jnp.ndarray, network: "CompiledNetwork") -> jnp.ndarray:
    """Total Kjeldahl Nitrogen = S_NH + S_ND + X_ND + i_XB × (XB_H + XB_A)
    + i_XP × (XP + XI). Uses standard ASM1 N-fractions i_XB=0.086, i_XP=0.06.
    """
    i_XB = 0.086
    i_XP = 0.06
    i = network.species_index
    return (
        C[..., i["SNH"]] + C[..., i["SND"]] + C[..., i["XND"]]
        + i_XB * (C[..., i["XB_H"]] + C[..., i["XB_A"]])
        + i_XP * (C[..., i["XP"]] + C[..., i["XI"]])
    )


def effluent_averages(
    t: jnp.ndarray,
    C_traj: jnp.ndarray,
    Q_traj: jnp.ndarray,
    network: "CompiledNetwork",
) -> dict[str, float]:
    """Time-flow-weighted average effluent concentrations.

    Parameters
    ----------
    t : jnp.ndarray
        Save-time vector, shape ``(n_t,)``.
    C_traj : jnp.ndarray
        Effluent concentration trajectory, shape ``(n_t, n_species)``.
    Q_traj : jnp.ndarray
        Effluent flow rate trajectory, shape ``(n_t,)``.
    network : CompiledNetwork

    Returns
    -------
    dict[str, float]
        Time-averaged COD, BOD, TSS, TKN, SNH, SNO (g/m³).
    """
    # Use trapezoidal integration over time.
    dt = jnp.diff(t)
    # Flow-weight via Q.
    weight = 0.5 * (Q_traj[:-1] + Q_traj[1:]) * dt  # (n_t - 1,)
    total_w = jnp.sum(weight)

    def time_avg(values: jnp.ndarray) -> float:
        v_mid = 0.5 * (values[:-1] + values[1:])
        return float(jnp.sum(v_mid * weight) / (total_w + 1e-12))

    return {
        "TSS": time_avg(derived_TSS(C_traj, network)),
        "COD": time_avg(derived_COD(C_traj, network)),
        "BOD": time_avg(derived_BOD(C_traj, network)),
        "TKN": time_avg(derived_TKN(C_traj, network)),
        "SNH": time_avg(C_traj[:, network.species_index["SNH"]]),
        "SNO": time_avg(C_traj[:, network.species_index["SNO"]]),
    }


def effluent_quality_index(
    t: jnp.ndarray,
    C_traj: jnp.ndarray,
    Q_traj: jnp.ndarray,
    network: "CompiledNetwork",
) -> float:
    """EQI per Copp 2002 / Alex 2008.

    ``EQI = (1 / T) × ∫ Q × (B_TSS×TSS + B_COD×COD + B_BOD×BOD
                              + B_TKN×TKN + B_NO×SNO) dt × 1e-3``

    Units: kg pollutant / day, averaged over the simulation window.
    """
    TSS_t = derived_TSS(C_traj, network)
    COD_t = derived_COD(C_traj, network)
    BOD_t = derived_BOD(C_traj, network)
    TKN_t = derived_TKN(C_traj, network)
    SNO_t = C_traj[:, network.species_index["SNO"]]

    integrand = Q_traj * (
        _EQI_WEIGHTS["TSS"] * TSS_t
        + _EQI_WEIGHTS["COD"] * COD_t
        + _EQI_WEIGHTS["BOD"] * BOD_t
        + _EQI_WEIGHTS["TKN"] * TKN_t
        + _EQI_WEIGHTS["NO"] * SNO_t
    )
    T_total = float(t[-1] - t[0])
    return float(jnp.trapezoid(integrand, t) * 1e-3 / (T_total + 1e-12))


def aeration_energy(
    t: jnp.ndarray,
    kla_history: jnp.ndarray,
    volumes: jnp.ndarray,
    saturation: float = 8.0,
) -> float:
    """Aeration energy (kWh/d) per Copp 2002 eq.

    AE = (S_sat / (T × 1.8 × 1000)) × ∫ Σ_i V_i × kLa_i(t) dt

    Parameters
    ----------
    t : (n_t,) save times in days
    kla_history : (n_t, n_aerated_tanks) kLa value at each save time
    volumes : (n_aerated_tanks,) liquid volume of each aerated tank
    saturation : float
        Dissolved-oxygen saturation concentration (mg/L).
    """
    kla_history = jnp.asarray(kla_history)
    volumes = jnp.asarray(volumes)
    integrand = jnp.sum(kla_history * volumes[None, :], axis=1)
    T_total = float(t[-1] - t[0])
    return float(
        saturation / (T_total * 1.8 * 1000.0) * jnp.trapezoid(integrand, t)
    )


def pumping_energy(
    t: jnp.ndarray,
    Q_internal: jnp.ndarray,
    Q_ras: jnp.ndarray,
    Q_was: jnp.ndarray,
) -> float:
    """Pumping energy (kWh/d) per Copp 2002 eq.

    PE = (1 / T) × ∫ (0.004 × Q_internal + 0.008 × Q_ras + 0.05 × Q_was) dt
    """
    integrand = 0.004 * Q_internal + 0.008 * Q_ras + 0.05 * Q_was
    T_total = float(t[-1] - t[0])
    return float(jnp.trapezoid(integrand, t) / (T_total + 1e-12))


def operational_cost_index(
    aeration: float,
    pumping: float,
    sludge_production: float,
) -> float:
    """OCI per Copp 2002 eq (BSM1 form):

    OCI = aeration + pumping + 5 × sludge_production

    Sludge_production is the time-averaged TSS mass flow leaving via
    wastage + the change in plant TSS inventory.
    """
    return float(aeration + pumping + 5.0 * sludge_production)


# ---- BSM2 OCI component kernels (Gernaey et al. 2014) -----------------------

# Default BSM2 pumping-energy factors (kWh/m³), one per pumped stream.
_BSM2_PUMP_FACTORS = {
    "internal": 0.004, "ras": 0.008, "wastage": 0.05,
    "primary_underflow": 0.075, "thickener_underflow": 0.060,
    "dewatering_underflow": 0.004,
}


def pumping_energy_bsm2(
    t: jnp.ndarray,
    flows: "dict[str, jnp.ndarray]",
    factors: "dict[str, float]" = None,
) -> float:
    """Pumping energy (kWh/d) for the full BSM2 pump set.

    ``PE = (1/T) × ∫ Σ_k PF_k × Q_k dt`` over the pumped streams: the AS internal
    recirculation, sludge recycle and wastage, plus the primary / thickener /
    dewatering underflows. ``flows`` maps a stream key to its ``(n_t,)`` flow
    trajectory; ``factors`` maps the same keys to per-m³ energy factors (default
    :data:`_BSM2_PUMP_FACTORS`). Keys present in ``flows`` but not ``factors``
    (or vice versa) are ignored.
    """
    factors = _BSM2_PUMP_FACTORS if factors is None else factors
    integrand = jnp.zeros_like(jnp.asarray(t))
    for key, Q in flows.items():
        if key in factors:
            integrand = integrand + float(factors[key]) * jnp.asarray(Q)
    T_total = float(t[-1] - t[0])
    return float(jnp.trapezoid(integrand, t) / (T_total + 1e-12))


def mixing_energy(
    t: jnp.ndarray,
    kla_history: jnp.ndarray,
    volumes: jnp.ndarray,
    digester_volume: float,
    kla_threshold: float = 20.0,
    reactor_unit: float = 0.005,
    digester_unit: float = 0.005,
) -> float:
    """Mixing energy (kWh/d) per Gernaey et al. 2014.

    A reactor needs mechanical mixing only while it is *not* aerated
    (``kLa < kla_threshold``); an aerated tank is mixed by the aeration. The
    anaerobic digester is always mechanically mixed. With unit mixing powers in
    kW/m³ (default 0.005 for both)::

        ME = 24 × [ Σ_i reactor_unit × V_i × frac_unaerated_i
                    + digester_unit × V_digester ]

    where ``frac_unaerated_i`` is the time fraction reactor ``i`` has
    ``kLa < kla_threshold``.

    Parameters
    ----------
    t : (n_t,) save times in days.
    kla_history : (n_t, n_reactors) kLa per reactor at each save time.
    volumes : (n_reactors,) reactor liquid volumes.
    digester_volume : float
        Anaerobic-digester liquid volume.
    """
    kla_history = jnp.asarray(kla_history)
    volumes = jnp.asarray(volumes)
    T_total = float(t[-1] - t[0])
    # Time fraction each reactor is below the aeration threshold.
    unaerated = (kla_history < kla_threshold).astype(jnp.float64)  # (n_t, n_reac)
    frac = jnp.trapezoid(unaerated, t, axis=0) / (T_total + 1e-12)  # (n_reac,)
    reactor_mix = reactor_unit * jnp.sum(volumes * frac)
    digester_mix = digester_unit * float(digester_volume)
    return float(24.0 * (reactor_mix + digester_mix))


def carbon_mass(
    t: jnp.ndarray,
    Q_carbon: jnp.ndarray,
    carbon_conc: float,
) -> float:
    """External-carbon mass dose (kg COD/d), time-averaged.

    ``= (1/T) × ∫ Q_carbon(t) × carbon_conc dt × 1e-3`` (the dose flow times the
    source COD concentration, g→kg).
    """
    integrand = jnp.asarray(Q_carbon) * float(carbon_conc) * 1e-3
    T_total = float(t[-1] - t[0])
    return float(jnp.trapezoid(integrand, t) / (T_total + 1e-12))


def heating_energy(
    t: jnp.ndarray,
    Q_feed: jnp.ndarray,
    T_feed_C: jnp.ndarray,
    T_target_C: float = 35.0,
    rho: float = 1000.0,
    cp: float = 4.186,
) -> float:
    """Digester sludge-heating energy (kWh/d) per Gernaey et al. 2014.

    Energy to raise the digester feed from ``T_feed_C`` to ``T_target_C``::

        Heatpower [kW] = (T_target − T_feed) × Q_feed × rho × cp / 86400
        HE [kWh/d]     = 24 × time-average(Heatpower)

    with water density ``rho`` (kg/m³) and specific heat ``cp`` (kJ/kg·°C).
    Temperatures are in **Celsius**. A feed already above the target contributes
    negative heating (no cooling credit is taken here; the OCI applies the
    methane offset separately).

    Parameters
    ----------
    Q_feed : (n_t,) digester feed flow (m³/d).
    T_feed_C : (n_t,) or scalar feed temperature (°C).
    """
    heatpower = ((float(T_target_C) - jnp.asarray(T_feed_C))
                 * jnp.asarray(Q_feed) * rho * cp / 86400.0)  # kW
    T_total = float(t[-1] - t[0])
    return float(24.0 * jnp.trapezoid(heatpower, t) / (T_total + 1e-12))


def operational_cost_index_bsm2(
    aeration: float,
    pumping: float,
    mixing: float,
    sludge_production: float,
    carbon: float,
    methane: float,
    heating: float,
) -> float:
    """Full BSM2 OCI (Gernaey et al. 2014):

    ``OCI = AE + PE + ME + 3·sludge + 3·carbon − 6·methane
            + max(0, heating − 7·methane)``

    Energies in kWh/d; sludge and carbon in kg/d; methane in kg CH₄/d. The
    methane credit and the methane-offset heating term reward biogas recovery.
    """
    return float(
        aeration + pumping + mixing
        + 3.0 * sludge_production + 3.0 * carbon
        - 6.0 * methane
        + max(0.0, heating - 7.0 * methane)
    )
