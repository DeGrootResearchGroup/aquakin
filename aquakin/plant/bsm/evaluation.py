"""BSM2 performance evaluation: EQI / OCI from a plant solution.

The generic metric kernels live in :mod:`aquakin.plant.metrics`; this module
wires them to a concrete BSM2 flowsheet (built by :func:`build_bsm2`). It
reconstructs the quantities the indices need from a solved
:class:`~aquakin.plant.plant.PlantSolution` -- the secondary-effluent stream,
the (possibly control-varying) aeration kLa of the aerated reactors, the pumped
recycle flows, and the wasted-sludge mass flow -- and returns the headline
**EQI** (effluent quality index) and **OCI** (operational cost index).

The aeration term reads the *actual* kLa over the run: for an open-loop plant
that is each aerated tank's fixed ``kla``; under closed-loop DO control
(``build_bsm2(do_control=True)``) it is the controller's manipulated signal,
recovered per saved state via :meth:`Plant.signals_at`. So evaluating an
open- and a closed-loop run side by side quantifies what the control buys.

OCI is the **full BSM2 index** (Gernaey et al. 2014): aeration + pumping (over
the whole pump set) + mixing energy + 3·sludge production + 3·external carbon −
6·methane production + max(0, heating − 7·methane). The methane credit and the
sludge-heating term are reconstructed from the ADM1 digester's gas-headspace
state and feed flow (see ``_methane_production`` / ``heating_energy``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp

from aquakin.plant.metrics import (
    aeration_energy,
    carbon_mass,
    derived_TSS,
    effluent_averages,
    effluent_quality_index,
    heating_energy,
    mixing_energy,
    operational_cost_index_bsm2,
    pumping_energy_bsm2,
)

# Default BSM2 port names (as wired by build_bsm2).
_EFFLUENT_PORT = "settler.overflow"
_DISPOSAL_PORT = "dewatering.underflow"
_INTERNAL_RECYCLE_PORT = "tank5_split.internal_recycle"
_RAS_PORT = "underflow_split.ras"
_WASTE_PORT = "underflow_split.waste"
_PRIMARY_UNDERFLOW_PORT = "primary.underflow"
_THICKENER_UNDERFLOW_PORT = "thickener.underflow"
_DO_SATURATION = 8.0  # gO2/m^3
_DIGESTER_FEED_PORT = "sludge_mix.out"
_DIGESTER_TARGET_T_C = 35.0     # digester operating temperature (BSM2)
_DIGESTER_FEED_T_C = 15.0       # default feed temperature when streams carry no T


@dataclass
class BSM2Evaluation:
    """Headline BSM2 performance indices from a solved plant.

    Attributes
    ----------
    eqi : float
        Effluent Quality Index (kg pollutant / day), lower is better.
    oci : float
        Full BSM2 Operational Cost Index (Gernaey et al. 2014):
        ``AE + PE + ME + 3·sludge + 3·carbon − 6·methane + max(0, HE − 7·methane)``.
    aeration_energy : float
        Aeration energy AE (kWh/d).
    pumping_energy : float
        Pumping energy PE (kWh/d), over the full BSM2 pump set (AS internal
        recycle, RAS, wastage, and the primary / thickener / dewatering
        underflows).
    mixing_energy : float
        Mixing energy ME (kWh/d): mechanical mixing of the unaerated reactors and
        the digester.
    sludge_production : float
        Wasted-sludge TSS mass flow to disposal (kg TSS/d), time-averaged.
    carbon_mass : float
        External-carbon dose (kg COD/d), time-averaged.
    methane_production : float
        Digester methane production (kg CH₄/d) -- the OCI's biogas credit.
    heating_energy : float
        Digester sludge-heating energy HE (kWh/d).
    effluent : dict
        Time/flow-weighted average effluent concentrations (COD, BOD, TSS, TKN,
        SNH, SNO; g/m^3) from :func:`effluent_averages`.
    aerated_tanks : list[str]
        The reactors whose aeration was counted.
    oci_note : str
        Notes on the OCI computation.
    """

    eqi: float
    oci: float
    aeration_energy: float
    pumping_energy: float
    mixing_energy: float
    sludge_production: float
    carbon_mass: float
    methane_production: float
    heating_energy: float
    effluent: dict = field(default_factory=dict)
    aerated_tanks: list = field(default_factory=list)
    oci_note: str = (
        "Full BSM2 OCI (Gernaey et al. 2014): AE + PE + ME + 3*sludge + "
        "3*carbon - 6*methane + max(0, HE - 7*methane). Sludge production is the "
        "disposal TSS mass flow (plant TSS-inventory change neglected -- ~0 at "
        "steady state); the heating feed temperature defaults to 15 C unless "
        "supplied."
    )


def _as_reactors(plant) -> list:
    """The activated-sludge reactor CSTRs (anoxic and aerated), in plant order.

    Identified by the CSTR-only ``controlled_kla`` attribute (the digester and
    other units lack it). All of them are mechanically mixed when unaerated, so
    the mixing-energy term needs the full set, not just the aerated tanks.
    """
    return [name for name in plant._unit_order
            if hasattr(plant.units[name], "controlled_kla")]


def _kla_history(plant, solution, params, tanks) -> jnp.ndarray:
    """Reconstruct each reactor's kLa at every saved time, ``(n_t, n_tanks)``.

    An anoxic tank has ``kLa = 0``; an aerated tank under DO control reads its
    kLa from the control signal (via :meth:`Plant.signals_at`), otherwise its
    fixed ``kla``.
    """
    need_signals = any(plant.units[n].controlled_kla for n in tanks)
    rows = []
    for i in range(solution.t.shape[0]):
        sig = (plant.signals_at(solution.t[i], solution.state[i], params)
               if need_signals else {})
        row = []
        for n in tanks:
            unit = plant.units[n]
            if unit.controlled_kla and "SO" in unit.controlled_kla:
                signal_name, gain = unit.controlled_kla["SO"]
                row.append(float(sig[signal_name]) * gain)
            else:
                row.append(float(unit.kla.get("SO", 0.0)))
        rows.append(row)
    return jnp.asarray(rows)


def _time_average(t: jnp.ndarray, values: jnp.ndarray) -> float:
    """Trapezoidal time-average of ``values(t)`` over ``[t0, t1]``."""
    T = float(t[-1] - t[0])
    return float(jnp.trapezoid(values, t) / (T + 1e-12))


def _reconstruct(plant, solution, params_full, endpoints):
    """Reconstruct several output streams in one pass over the saved states.

    ``endpoints`` is a list of ``"unit.port"`` strings. Returns
    ``{endpoint: (Q (n_t,), C (n_t, n_species))}``. One :meth:`Plant.outputs_at`
    per saved time (resolving the whole flow + stream sweep once) instead of one
    full pass per stream -- the indices need ~8 streams, so this is ~8x cheaper.
    """
    keys = {ep: tuple(plant._parse_endpoint(ep, role="source")) for ep in endpoints}
    Qs = {ep: [] for ep in endpoints}
    Cs = {ep: [] for ep in endpoints}
    for i in range(solution.t.shape[0]):
        outs = plant.outputs_at(solution.t[i], solution.state[i], params_full)
        for ep, key in keys.items():
            s = outs[key]
            Qs[ep].append(s.Q)
            Cs[ep].append(s.C)
    return {ep: (jnp.stack(Qs[ep]), jnp.stack(Cs[ep])) for ep in endpoints}


def _methane_production(plant, solution, params_full) -> float:
    """Digester methane production (kg CH4/d), time-averaged.

    From the ADM1 headspace state and gas parameters: the partial pressures give
    the biogas flow ``Q_gas = k_P·(P_gas − P_atm)`` and the CH4 fraction, so
    ``CH4 = (p_ch4/P_gas)·P_atm·16/R_T · Q_gas`` (the BSM2 evaluation formula).
    """
    adm1 = plant.units["digester"].network
    p = plant._params_for_unit("digester", params_full)

    def pv(name):
        return float(p[adm1.parameters.index(name)])

    R_T, P_atm, k_P, p_h2o = pv("R_T"), pv("P_atm"), pv("k_P"), pv("p_h2o")
    s_h2 = solution.C_named("digester", "S_gas_h2")
    s_ch4 = solution.C_named("digester", "S_gas_ch4")
    s_co2 = solution.C_named("digester", "S_gas_co2")
    p_ch4 = R_T / 64.0 * s_ch4
    P_gas = R_T / 16.0 * s_h2 + R_T / 64.0 * s_ch4 + R_T * s_co2 + p_h2o
    Q_gas = k_P * (P_gas - P_atm)                       # m3/d
    ch4_density = (p_ch4 / P_gas) * P_atm * 16.0 / R_T  # kg CH4/m3
    return _time_average(solution.t, ch4_density * Q_gas)


def _feed_temperature_C(plant, solution, params_full, default_C):
    """Flow-weighted digester-feed temperature (°C) at the final state, falling
    back to ``default_C`` when the streams carry no temperature."""
    outs = plant.outputs_at(solution.t[-1], solution.state[-1], params_full)
    feed = outs.get(("sludge_mix", "out"))
    if feed is None or feed.T is None:
        return float(default_C)
    return float(feed.T) - 273.15  # Stream T is Kelvin


def evaluate_bsm2(
    plant,
    solution,
    params: Optional[jnp.ndarray] = None,
    *,
    effluent_port: Optional[str] = None,
    disposal_port: str = _DISPOSAL_PORT,
    internal_recycle_port: str = _INTERNAL_RECYCLE_PORT,
    ras_port: str = _RAS_PORT,
    waste_port: str = _WASTE_PORT,
    do_saturation: float = _DO_SATURATION,
    digester_feed_T_C: float = _DIGESTER_FEED_T_C,
) -> BSM2Evaluation:
    """Compute the BSM2 performance indices from a solved plant.

    Parameters
    ----------
    plant : Plant
        A BSM2 plant from :func:`build_bsm2` (open- or closed-loop).
    solution : PlantSolution
        A solution from ``plant.solve`` over the evaluation window. Use a fine
        enough ``t_eval`` to resolve the influent dynamics; the indices are
        trapezoidal time-integrals over the saved points.
    params : jnp.ndarray, optional
        The plant parameters used for the run (defaults to the plant defaults).
    effluent_port : str, optional
        Final-effluent stream to score. Defaults to ``"effluent_mix.out"`` when
        the plant has an influent bypass (the combined treated + bypassed flow),
        else ``"settler.overflow"``.
    disposal_port, internal_recycle_port, ras_port, waste_port : str, optional
        Stream endpoints to reconstruct; the defaults match ``build_bsm2``.
    do_saturation : float, optional
        DO saturation used in the aeration-energy formula (gO2/m^3).
    digester_feed_T_C : float, optional
        Digester-feed temperature (°C) for the heating term, used only when the
        plant's streams carry no temperature (the default constant-influent BSM2
        is temperature-agnostic). Default 15.

    Returns
    -------
    BSM2Evaluation
        EQI, OCI and all component terms.
    """
    network = plant.units["tank1"].network
    params_full = (plant.default_parameters() if params is None
                   else jnp.asarray(params))
    t = solution.t

    # The final effluent is the bypass combiner's outlet when an influent bypass
    # is present, otherwise the clarifier overflow.
    if effluent_port is None:
        effluent_port = ("effluent_mix.out" if "effluent_mix" in plant.units
                         else _EFFLUENT_PORT)
    # Reconstruct every needed output stream in a single pass over the states.
    streams = _reconstruct(plant, solution, params_full, [
        effluent_port, disposal_port, internal_recycle_port, ras_port,
        waste_port, _PRIMARY_UNDERFLOW_PORT, _THICKENER_UNDERFLOW_PORT,
        _DIGESTER_FEED_PORT,
    ])

    # ----- Effluent quality. -----
    eff_Q, eff_C = streams[effluent_port]
    eqi = effluent_quality_index(t, eff_C, eff_Q, network)
    averages = effluent_averages(t, eff_C, eff_Q, network)

    # ----- Aeration + mixing energy (actual kLa over the run). Both span all AS
    # reactors: anoxic tanks add no aeration (kLa=0) but do need mixing. -----
    reactors = _as_reactors(plant)
    kla_hist = _kla_history(plant, solution, params, reactors)
    volumes = jnp.asarray([float(plant.units[n].volume) for n in reactors])
    AE = aeration_energy(t, kla_hist, volumes, saturation=do_saturation)
    V_digester = float(plant.units["digester"].volume)
    ME = mixing_energy(t, kla_hist, volumes, V_digester)
    aerated = [reactors[i] for i in range(len(reactors))
               if float(jnp.max(kla_hist[:, i])) > 0.0]

    # ----- Pumping energy (the full BSM2 pump set). -----
    PE = pumping_energy_bsm2(t, {
        "internal": streams[internal_recycle_port][0],
        "ras": streams[ras_port][0],
        "wastage": streams[waste_port][0],
        "primary_underflow": streams[_PRIMARY_UNDERFLOW_PORT][0],
        "thickener_underflow": streams[_THICKENER_UNDERFLOW_PORT][0],
        "dewatering_underflow": streams[disposal_port][0],
    })

    # ----- Sludge production (TSS mass flow leaving to disposal, kg/d). -----
    disp_Q, disp_C = streams[disposal_port]
    tss_mass_flow = derived_TSS(disp_C, network) * disp_Q * 1e-3
    sludge = _time_average(t, tss_mass_flow)

    # ----- External-carbon dose (kg COD/d). -----
    carbon_influent = plant.influents.get("external_carbon")
    if carbon_influent is not None:
        ss_idx = network.species_index["SS"]
        Q_carbon = jnp.stack([carbon_influent.at(ti).Q for ti in t])
        conc = float(carbon_influent.at(t[0]).C[ss_idx])
        carbon = carbon_mass(t, Q_carbon, conc)
    else:
        carbon = 0.0

    # ----- Digester methane credit + sludge-heating energy. -----
    methane = _methane_production(plant, solution, params_full)
    Q_feed = streams[_DIGESTER_FEED_PORT][0]
    T_feed = _feed_temperature_C(plant, solution, params_full, digester_feed_T_C)
    HE = heating_energy(t, Q_feed, T_feed, T_target_C=_DIGESTER_TARGET_T_C)

    oci = operational_cost_index_bsm2(AE, PE, ME, sludge, carbon, methane, HE)

    return BSM2Evaluation(
        eqi=eqi, oci=oci, aeration_energy=AE, pumping_energy=PE, mixing_energy=ME,
        sludge_production=sludge, carbon_mass=carbon, methane_production=methane,
        heating_energy=HE, effluent=averages, aerated_tanks=aerated,
    )
