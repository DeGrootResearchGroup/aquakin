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

import textwrap
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
    operational_cost_index,
    operational_cost_index_bsm2,
    pumping_energy,
    pumping_energy_bsm2,
)

# Default BSM1 port names (as wired by build_bsm1).
_BSM1_EFFLUENT_PORT = "clarifier.overflow"
_BSM1_INTERNAL_RECYCLE_PORT = "tank5_split.internal_recycle"
_BSM1_RAS_PORT = "underflow_split.ras"
_BSM1_WASTE_PORT = "underflow_split.waste"

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

# Units for the keys returned by effluent_averages (g/m³, currency-specific).
_EFFLUENT_UNITS = {
    "COD": "g COD/m³", "BOD": "g BOD/m³", "TSS": "g SS/m³",
    "TKN": "g N/m³", "SNH": "g N/m³", "SNO": "g N/m³",
}


def _render_eval_report(title, eqi, oci, oci_formula, terms, effluent,
                        aerated_tanks, note):
    """Render a labeled, units-annotated EQI / OCI report.

    ``terms`` is a list of ``(label, value, unit, contribution)`` rows, where
    ``contribution`` is the term's signed addition to the OCI (``None`` for a row
    that enters the index non-linearly, whose contribution column is left blank).
    """
    width = max((len(lbl) for lbl, *_ in terms), default=0)
    lines = [
        title, "=" * len(title),
        f"  EQI  Effluent Quality Index = {eqi:14.1f}  kg poll.-units/d "
        f" (lower is better)",
        f"  OCI  Operational Cost Index = {oci:14.1f}  (weighted cost units)",
        "",
        f"  OCI = {oci_formula}",
        f"  {'term':<{width}}  {'value':>12}  {'unit':<9}  {'OCI +=':>12}",
    ]
    for lbl, val, unit, contrib in terms:
        c = "" if contrib is None else f"{contrib:12.1f}"
        lines.append(f"  {lbl:<{width}}  {val:12.1f}  {unit:<9}  {c:>12}")
    if effluent:
        lines += ["", "  Effluent quality (time/flow-weighted averages):"]
        for key, val in effluent.items():
            lines.append(f"    {key:<4} {val:9.2f}  {_EFFLUENT_UNITS.get(key, 'g/m³')}")
    if aerated_tanks:
        lines.append(f"  Aerated reactors counted: {', '.join(aerated_tanks)}")
    if note:
        lines.append("")
        lines += textwrap.wrap(note, width=76, initial_indent="  Note: ",
                               subsequent_indent="        ")
    return "\n".join(lines)


@dataclass
class BSM2Evaluation:
    """Headline BSM2 performance indices from a solved plant.

    ``str(eval)`` / :meth:`report` give a labeled, units-annotated breakdown of
    the EQI, the OCI and every component term (with its OCI contribution) plus
    the ``oci_note`` caveat; the raw fields below stay available for programmatic
    use.

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

    def report(self) -> str:
        """A labeled, units-annotated EQI / OCI breakdown (also ``str(eval)``).

        Shows each OCI term with its physical value, units, and signed
        contribution to the index, the effluent averages, and the ``oci_note``
        caveat -- so the headline numbers are not bare floats to misread against
        published values.
        """
        heat = max(0.0, self.heating_energy - 7.0 * self.methane_production)
        terms = [
            ("Aeration energy  AE", self.aeration_energy, "kWh/d",
             self.aeration_energy),
            ("Pumping energy   PE", self.pumping_energy, "kWh/d",
             self.pumping_energy),
            ("Mixing energy    ME", self.mixing_energy, "kWh/d",
             self.mixing_energy),
            ("Sludge prod.  (x3)", self.sludge_production, "kg TSS/d",
             3.0 * self.sludge_production),
            ("Ext. carbon   (x3)", self.carbon_mass, "kg COD/d",
             3.0 * self.carbon_mass),
            ("Methane      (x-6)", self.methane_production, "kg CH4/d",
             -6.0 * self.methane_production),
            ("Heating energy HE", self.heating_energy, "kWh/d", None),
            ("  net heating (>=0)", heat, "kWh/d", heat),
        ]
        return _render_eval_report(
            "BSM2 performance indices", self.eqi, self.oci,
            "AE + PE + ME + 3*sludge + 3*carbon - 6*methane + max(0, HE - 7*methane)",
            terms, self.effluent, self.aerated_tanks, self.oci_note)

    def __str__(self) -> str:
        return self.report()


def _as_reactors(plant) -> list:
    """The activated-sludge reactor CSTRs (anoxic and aerated), in plant order.

    Identified by the CSTR-only ``aeration`` attribute (the digester and other
    units lack it). All of them are mechanically mixed when unaerated, so the
    mixing-energy term needs the full set, not just the aerated tanks.
    """
    return [name for name in plant._unit_order
            if hasattr(plant.units[name], "aeration")]


def _kla_history(plant, solution, params, tanks) -> jnp.ndarray:
    """Reconstruct each reactor's kLa at every saved time, ``(n_t, n_tanks)``.

    An anoxic tank has ``kLa = 0``; an aerated tank under DO control reads its
    kLa from the control signal (via :meth:`Plant.signals_at`), otherwise its
    fixed ``kLa``.
    """
    need_signals = any(plant.units[n]._controlled_kla for n in tanks)
    rows = []
    for i in range(solution.t.shape[0]):
        sig = (plant.signals_at(solution.t[i], solution.state[i], params)
               if need_signals else {})
        row = []
        for n in tanks:
            unit = plant.units[n]
            controlled = unit._controlled_kla.get("SO")
            if controlled is not None:
                signal_name, gain = controlled
                row.append(float(sig[signal_name]) * gain)
            else:
                row.append(float(unit._kla_vec[unit.network.species_index["SO"]]))
        rows.append(row)
    return jnp.asarray(rows)


def _time_average(t: jnp.ndarray, values: jnp.ndarray) -> float:
    """Trapezoidal time-average of ``values(t)`` over ``[t0, t1]``.

    A single saved point (a steady-state solution) has a zero-width window; the
    average of a constant is that sample, so it is returned directly -- giving the
    instantaneous steady-state value rather than dividing by a zero window."""
    t = jnp.asarray(t)
    if t.shape[0] <= 1:
        return float(jnp.asarray(values)[0])
    return float(jnp.trapezoid(values, t) / (t[-1] - t[0]))


def _reconstruct(plant, solution, params_full, endpoints):
    """Reconstruct several output streams from the saved states.

    ``endpoints`` is a list of ``"unit.port"`` strings. Returns
    ``{endpoint: (Q (n_t,), C (n_t, n_species))}``. The whole output sweep is
    reconstructed once (resolving the flow + stream sweep per saved time) and
    cached on the solution by :meth:`Plant._cached_streams`, so the metric
    indices' ~8 streams and any later ``plant.stream`` call share one pass.
    """
    allstreams = plant._cached_streams(solution, params_full)
    return {ep: allstreams[plant._parse_endpoint(ep, role="source")]
            for ep in endpoints}


@dataclass
class DigesterGas:
    """The anaerobic digester's biogas trajectory (the semantic ``digester_gas``
    stream -- a *derived* output computed from the ADM1 headspace state, not a
    material port).

    Attributes
    ----------
    t : jnp.ndarray
        Save times, shape ``(n_t,)``.
    Q : jnp.ndarray
        Total biogas flow ``Q_gas`` (m³/d), shape ``(n_t,)``.
    p_ch4, p_co2, p_h2 : jnp.ndarray
        CH₄ / CO₂ / H₂ partial pressures (bar), shape ``(n_t,)``.
    ch4 : jnp.ndarray
        CH₄ mass flow (kg CH₄/d), shape ``(n_t,)`` -- the OCI biogas credit.
    """

    t: jnp.ndarray
    Q: jnp.ndarray
    p_ch4: jnp.ndarray
    p_co2: jnp.ndarray
    p_h2: jnp.ndarray
    ch4: jnp.ndarray

    def methane_production(self) -> float:
        """Time-averaged CH₄ production (kg CH₄/d) over the solution window."""
        return _time_average(self.t, self.ch4)


def _digester_unit_name(plant) -> str:
    """The name of the ADM1 anaerobic digester (the unit carrying the headspace
    gas states), or a clear error if the plant has none."""
    for name in plant.list_units():
        net = getattr(plant.units[name], "network", None)
        if net is not None and "S_gas_ch4" in net.species_index:
            return name
    raise ValueError(
        "This plant has no anaerobic digester (no unit with an ADM1 gas "
        "headspace), so it has no biogas. digester_gas() needs a build_bsm2 "
        "plant with an ADM1DigesterUnit."
    )


def digester_gas(plant, solution, params=None) -> DigesterGas:
    """The digester biogas trajectory (flow, partial pressures, CH₄ mass flow).

    From the ADM1 headspace state and gas parameters: the partial pressures give
    the biogas flow ``Q_gas = k_P·(P_gas − P_atm)`` and the CH₄ fraction, so
    ``CH4 = (p_ch4/P_gas)·P_atm·16/R_T · Q_gas`` (the BSM2 evaluation formula).
    Reached as ``plant.digester_gas(solution)``.
    """
    name = _digester_unit_name(plant)
    adm1 = plant.units[name].network
    params_full = (plant.default_parameters() if params is None
                   else jnp.asarray(params))
    plant._build_parameter_layout()
    p = plant._params_for_unit(name, params_full)

    def pv(pname):
        return float(p[adm1.parameters.index(pname)])

    R_T, P_atm, k_P, p_h2o = pv("R_T"), pv("P_atm"), pv("k_P"), pv("p_h2o")
    s_h2 = solution.C_named(name, "S_gas_h2")
    s_ch4 = solution.C_named(name, "S_gas_ch4")
    s_co2 = solution.C_named(name, "S_gas_co2")
    p_h2 = R_T / 16.0 * s_h2
    p_ch4 = R_T / 64.0 * s_ch4
    p_co2 = R_T * s_co2
    P_gas = p_h2 + p_ch4 + p_co2 + p_h2o
    Q_gas = k_P * (P_gas - P_atm)                       # m3/d
    ch4_density = (p_ch4 / P_gas) * P_atm * 16.0 / R_T  # kg CH4/m3
    return DigesterGas(t=solution.t, Q=Q_gas, p_ch4=p_ch4, p_co2=p_co2,
                       p_h2=p_h2, ch4=ch4_density * Q_gas)


def _methane_production(plant, solution, params_full) -> float:
    """Digester methane production (kg CH4/d), time-averaged (the OCI credit)."""
    return digester_gas(plant, solution, params_full).methane_production()


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

    # The final effluent is whatever the builder recorded on the plant (the
    # bypass combiner's outlet when an influent bypass is present, otherwise the
    # secondary overflow); fall back to detection for a plant with no recorded
    # endpoint.
    if effluent_port is None:
        effluent_port = (
            getattr(plant, "effluent_endpoint", None)
            or ("effluent_mix.out" if "effluent_mix" in plant.units
                else _EFFLUENT_PORT))
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


@dataclass
class BSM1Evaluation:
    """Headline BSM1 performance indices from a solved plant.

    ``str(eval)`` / :meth:`report` give a labeled, units-annotated breakdown of
    the EQI, the OCI and every component term plus the ``oci_note`` caveat; the
    raw fields below stay available for programmatic use.

    Attributes
    ----------
    eqi : float
        Effluent Quality Index (kg pollutant / day), lower is better.
    oci : float
        BSM1 Operational Cost Index (Copp 2002): ``AE + PE + 5·sludge``.
    aeration_energy : float
        Aeration energy AE (kWh/d).
    pumping_energy : float
        Pumping energy PE (kWh/d): the internal recycle, RAS and wastage pumps.
    sludge_production : float
        Wasted-sludge TSS mass flow (kg TSS/d), time-averaged.
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
    sludge_production: float
    effluent: dict = field(default_factory=dict)
    aerated_tanks: list = field(default_factory=list)
    oci_note: str = (
        "BSM1 OCI (Copp 2002): AE + PE + 5*sludge. Sludge production is the "
        "wastage TSS mass flow (plant TSS-inventory change neglected -- ~0 at "
        "steady state)."
    )

    def report(self) -> str:
        """A labeled, units-annotated EQI / OCI breakdown (also ``str(eval)``).

        Shows each OCI term with its value, units and signed contribution to the
        index, the effluent averages, and the ``oci_note`` caveat."""
        terms = [
            ("Aeration energy  AE", self.aeration_energy, "kWh/d",
             self.aeration_energy),
            ("Pumping energy   PE", self.pumping_energy, "kWh/d",
             self.pumping_energy),
            ("Sludge prod.  (x5)", self.sludge_production, "kg TSS/d",
             5.0 * self.sludge_production),
        ]
        return _render_eval_report(
            "BSM1 performance indices", self.eqi, self.oci,
            "AE + PE + 5*sludge", terms, self.effluent, self.aerated_tanks,
            self.oci_note)

    def __str__(self) -> str:
        return self.report()


def evaluate_bsm1(
    plant,
    solution,
    params: Optional[jnp.ndarray] = None,
    *,
    effluent_port: str = _BSM1_EFFLUENT_PORT,
    internal_recycle_port: str = _BSM1_INTERNAL_RECYCLE_PORT,
    ras_port: str = _BSM1_RAS_PORT,
    waste_port: str = _BSM1_WASTE_PORT,
    do_saturation: float = _DO_SATURATION,
) -> BSM1Evaluation:
    """Compute the BSM1 performance indices from a solved plant.

    Parameters
    ----------
    plant : Plant
        A BSM1 plant from :func:`build_bsm1` (open- or closed-loop).
    solution : PlantSolution
        A solution from ``plant.solve`` over the evaluation window. Use a fine
        enough ``t_eval`` to resolve the influent dynamics; the indices are
        trapezoidal time-integrals over the saved points.
    params : jnp.ndarray, optional
        The plant parameters used for the run (defaults to the plant defaults).
    effluent_port : str, optional
        Final-effluent stream to score. Defaults to ``"clarifier.overflow"``.
    internal_recycle_port, ras_port, waste_port : str, optional
        Pumped-stream endpoints; the defaults match ``build_bsm1``.
    do_saturation : float, optional
        DO saturation used in the aeration-energy formula (gO2/m^3).

    Returns
    -------
    BSM1Evaluation
        EQI, OCI and all component terms.
    """
    network = plant.units["tank1"].network
    params_full = (plant.default_parameters() if params is None
                   else jnp.asarray(params))
    t = solution.t

    # Reconstruct every needed output stream in a single pass over the states.
    streams = _reconstruct(plant, solution, params_full, [
        effluent_port, internal_recycle_port, ras_port, waste_port,
    ])

    # ----- Effluent quality. -----
    eff_Q, eff_C = streams[effluent_port]
    eqi = effluent_quality_index(t, eff_C, eff_Q, network)
    averages = effluent_averages(t, eff_C, eff_Q, network)

    # ----- Aeration energy (actual kLa over the run). -----
    reactors = _as_reactors(plant)
    kla_hist = _kla_history(plant, solution, params, reactors)
    volumes = jnp.asarray([float(plant.units[n].volume) for n in reactors])
    AE = aeration_energy(t, kla_hist, volumes, saturation=do_saturation)
    aerated = [reactors[i] for i in range(len(reactors))
               if float(jnp.max(kla_hist[:, i])) > 0.0]

    # ----- Pumping energy (internal recycle + RAS + wastage). -----
    PE = pumping_energy(
        t,
        streams[internal_recycle_port][0],
        streams[ras_port][0],
        streams[waste_port][0],
    )

    # ----- Sludge production (TSS mass flow leaving via wastage, kg/d). -----
    waste_Q, waste_C = streams[waste_port]
    tss_mass_flow = derived_TSS(waste_C, network) * waste_Q * 1e-3
    sludge = _time_average(t, tss_mass_flow)

    oci = operational_cost_index(AE, PE, sludge)

    return BSM1Evaluation(
        eqi=eqi, oci=oci, aeration_energy=AE, pumping_energy=PE,
        sludge_production=sludge, effluent=averages, aerated_tanks=aerated,
    )
