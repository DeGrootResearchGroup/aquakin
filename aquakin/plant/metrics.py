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


def derived_TSS(C: jnp.ndarray, network: "CompiledNetwork") -> jnp.ndarray:
    """Total suspended solids from ASM1 particulate species.

    ``TSS = 0.75 × (XS + XI + XB_H + XB_A + XP)``.

    Returns a scalar (if ``C`` is 1-D) or a vector along the leading
    axis (if 2-D).
    """
    indices = [network.species_index[s] for s in _TSS_SPECIES if s in network.species_index]
    idx = jnp.asarray(indices)
    if C.ndim == 1:
        return _TSS_FACTOR * jnp.sum(C[idx])
    return _TSS_FACTOR * jnp.sum(C[..., idx], axis=-1)


def derived_COD(C: jnp.ndarray, network: "CompiledNetwork") -> jnp.ndarray:
    """Total COD = SI + SS + XI + XS + XB_H + XB_A + XP."""
    species = ("SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP")
    idx = jnp.asarray(
        [network.species_index[s] for s in species if s in network.species_index]
    )
    if C.ndim == 1:
        return jnp.sum(C[idx])
    return jnp.sum(C[..., idx], axis=-1)


def derived_BOD(C: jnp.ndarray, network: "CompiledNetwork") -> jnp.ndarray:
    """BOD₅ proxy = 0.25 × (SS + XS + (1 - f_P) × (XB_H + XB_A))
    using Copp 2002 BOD relation with f_P ≈ 0.08."""
    f_P = 0.08
    idx = lambda s: network.species_index[s]
    if C.ndim == 1:
        return 0.25 * (
            C[idx("SS")] + C[idx("XS")]
            + (1.0 - f_P) * (C[idx("XB_H")] + C[idx("XB_A")])
        )
    return 0.25 * (
        C[..., idx("SS")] + C[..., idx("XS")]
        + (1.0 - f_P) * (C[..., idx("XB_H")] + C[..., idx("XB_A")])
    )


def derived_TKN(C: jnp.ndarray, network: "CompiledNetwork") -> jnp.ndarray:
    """Total Kjeldahl Nitrogen = S_NH + S_ND + X_ND + i_XB × (XB_H + XB_A)
    + i_XP × (XP + XI). Uses standard ASM1 N-fractions i_XB=0.086, i_XP=0.06.
    """
    i_XB = 0.086
    i_XP = 0.06
    idx = lambda s: network.species_index[s]
    if C.ndim == 1:
        return (
            C[idx("SNH")] + C[idx("SND")] + C[idx("XND")]
            + i_XB * (C[idx("XB_H")] + C[idx("XB_A")])
            + i_XP * (C[idx("XP")] + C[idx("XI")])
        )
    return (
        C[..., idx("SNH")] + C[..., idx("SND")] + C[..., idx("XND")]
        + i_XB * (C[..., idx("XB_H")] + C[..., idx("XB_A")])
        + i_XP * (C[..., idx("XP")] + C[..., idx("XI")])
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
    dt = jnp.diff(t)
    mid = 0.5 * (integrand[:-1] + integrand[1:])
    T_total = float(t[-1] - t[0])
    return float(jnp.sum(mid * dt) * 1e-3 / (T_total + 1e-12))


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
    dt = jnp.diff(t)
    mid = 0.5 * (integrand[:-1] + integrand[1:])
    T_total = float(t[-1] - t[0])
    return float(
        saturation / (T_total * 1.8 * 1000.0) * jnp.sum(mid * dt)
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
    dt = jnp.diff(t)
    mid = 0.5 * (integrand[:-1] + integrand[1:])
    T_total = float(t[-1] - t[0])
    return float(jnp.sum(mid * dt) / (T_total + 1e-12))


def operational_cost_index(
    aeration: float,
    pumping: float,
    sludge_production: float,
) -> float:
    """OCI per Copp 2002 eq:

    OCI = aeration + pumping + 5 × sludge_production

    Sludge_production is the time-averaged TSS mass flow leaving via
    wastage + the change in plant TSS inventory.
    """
    return float(aeration + pumping + 5.0 * sludge_production)
