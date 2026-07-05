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

import jax
import jax.numpy as jnp

from aquakin.plant.aeration_system import blower_airflow_total, blower_energy
from aquakin.plant.metrics import (
    _EQI_WEIGHTS,
    _composition,
    aeration_energy,
    carbon_mass,
    derived_BOD,
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
from aquakin.plant.metrics import (
    _time_average as _metrics_time_average,
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
_DIGESTER_TARGET_T_C = 35.0  # digester operating temperature (BSM2)
_DIGESTER_FEED_T_C = 15.0  # default feed temperature when streams carry no T

# Units for the keys returned by effluent_averages (g/m³, currency-specific).
_EFFLUENT_UNITS = {
    "COD": "g COD/m³",
    "BOD": "g BOD/m³",
    "TSS": "g SS/m³",
    "TKN": "g N/m³",
    "SNH": "g N/m³",
    "SNO": "g N/m³",
}


def _render_eval_report(title, eqi, oci, oci_formula, terms, effluent, aerated_tanks, note):
    """Render a labeled, units-annotated EQI / OCI report.

    ``terms`` is a list of ``(label, value, unit, contribution)`` rows, where
    ``contribution`` is the term's signed addition to the OCI (``None`` for a row
    that enters the index non-linearly, whose contribution column is left blank).
    """
    width = max((len(lbl) for lbl, *_ in terms), default=0)
    lines = [
        title,
        "=" * len(title),
        f"  EQI  Effluent Quality Index = {eqi:14.1f}  kg poll.-units/d  (lower is better)",
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
        lines += textwrap.wrap(
            note, width=76, initial_indent="  Note: ", subsequent_indent="        "
        )
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
    air_flow : float or None
        Total blower air flow (m³/d, time-averaged) when an ``aeration_system``
        diffuser/blower design was supplied; ``None`` for the correlation default.
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
    air_flow: float | None = None
    oci_note: str = (
        "Full BSM2 OCI (Gernaey et al. 2014): AE + PE + ME + 3*sludge + "
        "3*carbon - 6*methane + max(0, HE - 7*methane). Sludge production is the "
        "disposal TSS mass flow (plant TSS-inventory change neglected -- ~0 at "
        "steady state); the heating feed temperature defaults to 15 C unless "
        "supplied."
    )

    def total_energy(self) -> float:
        """Total electricity draw (kWh/d) = aeration + pumping + mixing -- the
        energy basis for the GHG and cost layers."""
        return self.aeration_energy + self.pumping_energy + self.mixing_energy

    def kpis(self) -> dict:
        """Headline performance KPIs for a scenario comparison table."""
        return {
            "EQI (kg/d)": self.eqi,
            "OCI": self.oci,
            "Energy (kWh/d)": self.total_energy(),
            "Sludge (kgTSS/d)": self.sludge_production,
            "Methane (kgCH4/d)": self.methane_production,
            "SNH (gN/m3)": self.effluent.get("SNH", float("nan")),
            "SNO (gN/m3)": self.effluent.get("SNO", float("nan")),
        }

    def report(self) -> str:
        """A labeled, units-annotated EQI / OCI breakdown (also ``str(eval)``).

        Shows each OCI term with its physical value, units, and signed
        contribution to the index, the effluent averages, and the ``oci_note``
        caveat -- so the headline numbers are not bare floats to misread against
        published values.
        """
        heat = max(0.0, self.heating_energy - 7.0 * self.methane_production)
        ae_label = (
            "Aeration energy  AE (blower)" if self.air_flow is not None else "Aeration energy  AE"
        )
        terms = [
            (ae_label, self.aeration_energy, "kWh/d", self.aeration_energy),
        ]
        if self.air_flow is not None:
            terms.append(("  air flow", self.air_flow, "m3/d", None))
        terms += [
            ("Pumping energy   PE", self.pumping_energy, "kWh/d", self.pumping_energy),
            ("Mixing energy    ME", self.mixing_energy, "kWh/d", self.mixing_energy),
            (
                "Sludge prod.  (x3)",
                self.sludge_production,
                "kg TSS/d",
                3.0 * self.sludge_production,
            ),
            ("Ext. carbon   (x3)", self.carbon_mass, "kg COD/d", 3.0 * self.carbon_mass),
            (
                "Methane      (x-6)",
                self.methane_production,
                "kg CH4/d",
                -6.0 * self.methane_production,
            ),
            ("Heating energy HE", self.heating_energy, "kWh/d", None),
            ("  net heating (>=0)", heat, "kWh/d", heat),
        ]
        return _render_eval_report(
            "BSM2 performance indices",
            self.eqi,
            self.oci,
            "AE + PE + ME + 3*sludge + 3*carbon - 6*methane + max(0, HE - 7*methane)",
            terms,
            self.effluent,
            self.aerated_tanks,
            self.oci_note,
        )

    def __str__(self) -> str:
        return self.report()


def _as_reactors(plant) -> list:
    """The activated-sludge reactor CSTRs (anoxic and aerated), in plant order.

    Identified by the CSTR-only ``aeration`` attribute (the digester and other
    units lack it). All of them are mechanically mixed when unaerated, so the
    mixing-energy term needs the full set, not just the aerated tanks -- hence
    ``require_volume=False`` (an MBR-style reactor need not declare a volume).
    """
    return plant.activated_sludge_reactors(require_volume=False)


def _kla_history(plant, solution, params, tanks) -> jnp.ndarray:
    """Reconstruct each reactor's kLa at every saved time, ``(n_t, n_tanks)``.

    An anoxic tank has ``kLa = 0``; an aerated tank under DO control reads its
    kLa from the control signal (via :meth:`Plant.signals_at`), otherwise its
    fixed ``kLa``.
    """
    n_t = solution.t.shape[0]
    need_signals = any(plant.units[n]._controlled_kla for n in tanks)
    if not need_signals:
        # Every tank's kLa is fixed: one constant row, tiled over time.
        row = jnp.asarray(
            [
                float(plant.units[n]._kla_vec[plant.units[n].model.species_index["SO"]])
                for n in tanks
            ]
        )
        return jnp.broadcast_to(row, (n_t, len(tanks)))

    # Closed-loop DO control: the manipulated kLa comes from the control signal,
    # reconstructed per saved state. vmap it over all times in one sweep instead
    # of a per-step Python call to signals_at.
    def _row(t_i, state_row):
        sig = plant.signals_at(t_i, state_row, params)
        vals = []
        for n in tanks:
            unit = plant.units[n]
            controlled = unit._controlled_kla.get("SO")
            if controlled is not None:
                signal_name, gain = controlled
                vals.append(sig[signal_name] * gain)
            else:
                vals.append(jnp.asarray(float(unit._kla_vec[unit.model.species_index["SO"]])))
        return jnp.stack(vals)

    return jax.vmap(_row)(jnp.asarray(solution.t), jnp.asarray(solution.state))


def _time_average(t: jnp.ndarray, values: jnp.ndarray) -> float:
    """Trapezoidal time-average of ``values(t)`` over ``[t0, t1]`` (single source
    of truth: the shared :func:`aquakin.plant.metrics._time_average` kernel,
    which also returns the single sample for a one-point steady-state window).
    Wrapped here only to keep the local ``(t, values)`` argument order and the
    ``float`` return."""
    return float(_metrics_time_average(values, t))


def _reconstruct(plant, solution, params_full, endpoints):
    """Reconstruct several output streams from the saved states.

    ``endpoints`` is a list of ``"unit.port"`` strings. Returns
    ``{endpoint: (Q (n_t,), C (n_t, n_species))}``. The whole output sweep is
    reconstructed once (resolving the flow + stream sweep per saved time) and
    cached on the solution by :meth:`Plant._cached_streams`, so the metric
    indices' ~8 streams and any later ``plant.stream`` call share one pass.
    """
    allstreams = plant._cached_streams(solution, params_full)
    return {ep: allstreams[plant._parse_endpoint(ep, role="source")] for ep in endpoints}


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
        Total biogas flow ``Q_gas`` (m³/d), shape ``(n_t,)``, normalized to
        atmospheric pressure (the benchmark convention: the raw overpressure
        outflow ``k_P*(P_gas - P_atm)`` times ``P_gas/P_atm``).
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
        net = getattr(plant.units[name], "model", None)
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
    adm1 = plant.units[name].model
    params_full = plant.default_parameters() if params is None else jnp.asarray(params)
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
    # Headspace overpressure drives the raw outflow k_P*(P_gas - P_atm); this is
    # the flow the gas-phase ODE uses. The *reported* biogas flow is recalculated
    # to atmospheric pressure by the factor P_gas/P_atm, the benchmark
    # normalization (the BSM2 ADM1 reports q_gas*P_gas/P_atm as the gas flow, and
    # the methane-production / OCI credit is computed from it). Omitting the
    # normalization understates the biogas flow, and hence the methane production
    # and its OCI credit, by P_gas/P_atm (about 5% at the benchmark operating
    # point), while leaving the gas-phase concentrations unchanged.
    Q_gas = k_P * (P_gas - P_atm) * P_gas / P_atm  # m3/d, normalized to P_atm
    ch4_density = (p_ch4 / P_gas) * P_atm * 16.0 / R_T  # kg CH4/m3
    return DigesterGas(
        t=solution.t, Q=Q_gas, p_ch4=p_ch4, p_co2=p_co2, p_h2=p_h2, ch4=ch4_density * Q_gas
    )


def _methane_production(plant, solution, params_full) -> float:
    """Digester methane production (kg CH4/d), time-averaged (the OCI credit)."""
    return digester_gas(plant, solution, params_full).methane_production()


def _feed_temperature_C(plant, solution, params_full, default_C):
    """Digester-feed temperature (°C), per saved time, so ``heating_energy``
    time-averages it consistently with the feed flow (rather than using only the
    final instant). Falls back to ``default_C`` when the streams carry no
    temperature.

    Temperature presence is structural -- a temperature-agnostic influent leaves
    every stream ``T = None``, a temperature-carrying one leaves every stream with
    a ``T`` -- so a ``None`` at the final state means ``None`` throughout: the
    common (default BSM2) case returns the scalar default after one reconstruction
    and only a genuinely temperature-carrying run pays the per-step sweep."""
    final = plant.outputs_at(solution.t[-1], solution.state[-1], params_full)
    feed = final.get(("sludge_mix", "out"))
    if feed is None or feed.T is None:
        return float(default_C)
    # T is structurally present (a temperature-carrying influent leaves every
    # stream with a T), so vmap the digester-feed temperature over all saved
    # times in one vectorised sweep rather than a per-step Python loop.
    ts = jnp.asarray(solution.t)

    def _feed_T(t_i, state_row):
        states = plant._split_state(state_row)
        outs, _ = plant._resolve_streams(t_i, states, params_full)
        return outs[("sludge_mix", "out")].T

    return jax.vmap(_feed_T)(ts, jnp.asarray(solution.state)) - 273.15  # K -> C


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
    aeration_system=None,
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
        DO saturation used in the aeration-energy formula (gO2/m^3). This is the
        nominal saturation by BSM2 convention; it is **not** temperature-adjusted
        even when the plant runs with ``do_temperature_correction`` (where the
        reactor's actual driving-force saturation is scaled by ``C_s(T)/C_s(ref)``).
        So the reported AE reflects the BSM2 definition, not the temperature-
        corrected oxygen transfer.
    digester_feed_T_C : float, optional
        Digester-feed temperature (°C) for the heating term, used only when the
        plant's streams carry no temperature (the default constant-influent BSM2
        is temperature-agnostic). Default 15.

    Returns
    -------
    BSM2Evaluation
        EQI, OCI and all component terms.
    """
    model = plant.units["tank1"].model
    params_full = plant.default_parameters() if params is None else jnp.asarray(params)
    # Composition (i_XB / i_XP / f_P) is read with the ASM1 model's *local*
    # param_index, so it must be sliced from the ASM1 block of the concatenated
    # plant vector -- not indexed into the full vector (which is correct only while
    # ASM1 sits at block offset 0).
    params_asm = plant._params_for_unit("tank1", params_full)
    t = solution.t

    # The final effluent is whatever the builder recorded on the plant (the
    # bypass combiner's outlet when an influent bypass is present, otherwise the
    # secondary overflow); fall back to detection for a plant with no recorded
    # endpoint.
    if effluent_port is None:
        effluent_port = getattr(plant, "effluent_endpoint", None) or (
            "effluent_mix.out" if "effluent_mix" in plant.units else _EFFLUENT_PORT
        )
    # Reconstruct every needed output stream in a single pass over the states.
    streams = _reconstruct(
        plant,
        solution,
        params_full,
        [
            effluent_port,
            disposal_port,
            internal_recycle_port,
            ras_port,
            waste_port,
            _PRIMARY_UNDERFLOW_PORT,
            _THICKENER_UNDERFLOW_PORT,
            _DIGESTER_FEED_PORT,
        ],
    )

    # ----- Effluent quality. -----
    eff_Q, eff_C = streams[effluent_port]
    eqi = effluent_quality_index(t, eff_C, eff_Q, model, params=params_asm)
    averages = effluent_averages(t, eff_C, eff_Q, model, params=params_asm)

    # BSM2 weights the BOD of *bypassed* (untreated) flow at the raw-sewage 0.65
    # BOD5/BODu coefficient rather than the 0.25 applied to treated effluent. When
    # an influent bypass
    # is present the scored effluent is the treated + bypassed mix, so the flat
    # 0.25-coefficient BOD that effluent_averages returns understates it. Redo the
    # BOD average as the load-weighted split over the two component streams; this
    # branch is inert without a bypass (the validated no-bypass path is untouched).
    if "effluent_mix" in plant.units:
        # The two source streams feeding the bypass combiner: the treated
        # secondary-clarifier overflow and the diverted raw influent.
        comp = _reconstruct(plant, solution, params_full, [_EFFLUENT_PORT, "bypass_split.bypass"])
        Qt, Ct = comp[_EFFLUENT_PORT]
        Qb, Cb = comp["bypass_split.bypass"]
        _, _, f_P = _composition(model, params_asm)
        base_t = derived_BOD(Ct, model, f_P=f_P) / 0.25  # SS+XS+(1-fP)(XBH+XBA)
        base_b = derived_BOD(Cb, model, f_P=f_P) / 0.25
        bod_load = _time_average(t, 0.25 * base_t * Qt) + _time_average(t, 0.65 * base_b * Qb)
        total_flow = _time_average(t, Qt + Qb)
        averages = {**averages, "BOD": float(bod_load / total_flow)}
        # The flat-weight EQI scored the bypass BOD at the treated 0.25 coefficient
        # as well (it ran on the combined effluent); add the extra (0.65 − 0.25)
        # weight on the bypass BOD load so the scored EQI carries the benchmark
        # bypass coefficient too. derived_BOD is linear, so the combined-effluent
        # BOD load already equals 0.25·(base_t·Qt + base_b·Qb).
        eqi = eqi + float(
            _EQI_WEIGHTS["BOD"] * (0.65 - 0.25) * _time_average(t, base_b * Qb) * 1e-3
        )

    # ----- Aeration + mixing energy (actual kLa over the run). Both span all AS
    # reactors: anoxic tanks add no aeration (kLa=0) but do need mixing. -----
    reactors = _as_reactors(plant)
    kla_hist = _kla_history(plant, solution, params, reactors)
    volumes = jnp.asarray([float(plant.units[n].volume) for n in reactors])
    # When a diffuser/blower design is given, AE is the mechanistic blower energy
    # (SOTE / depth / blower curve) and the total air flow is reported; otherwise
    # the Copp-2002 aeration-energy correlation (the validated benchmark default).
    if aeration_system is not None:
        AE = blower_energy(t, kla_hist, volumes, aeration_system)
        air_flow = blower_airflow_total(t, kla_hist, volumes, aeration_system)
    else:
        AE = aeration_energy(t, kla_hist, volumes, saturation=do_saturation)
        air_flow = None
    V_digester = float(plant.units["digester"].volume)
    ME = mixing_energy(t, kla_hist, volumes, V_digester)
    aerated = [reactors[i] for i in range(len(reactors)) if float(jnp.max(kla_hist[:, i])) > 0.0]

    # ----- Pumping energy (the full BSM2 pump set). -----
    PE = pumping_energy_bsm2(
        t,
        {
            "internal": streams[internal_recycle_port][0],
            "ras": streams[ras_port][0],
            "wastage": streams[waste_port][0],
            "primary_underflow": streams[_PRIMARY_UNDERFLOW_PORT][0],
            "thickener_underflow": streams[_THICKENER_UNDERFLOW_PORT][0],
            "dewatering_underflow": streams[disposal_port][0],
        },
    )

    # ----- Sludge production (TSS mass flow leaving to disposal, kg/d). -----
    disp_Q, disp_C = streams[disposal_port]
    tss_mass_flow = derived_TSS(disp_C, model) * disp_Q * 1e-3
    sludge = _time_average(t, tss_mass_flow)

    # ----- External-carbon dose (kg COD/d). -----
    # The external carbon is dosed by the `external_carbon` DosingUnit: the dose
    # flow times the reagent's readily-biodegradable (SS) concentration.
    carbon_unit = plant.units.get("external_carbon")
    if carbon_unit is not None and hasattr(carbon_unit, "reagent"):
        ss_idx = model.species_index["SS"]
        conc = float(carbon_unit.reagent.composition[ss_idx])
        if carbon_unit.flow is not None:
            # Fixed dose: constant flow over the window.
            Q_carbon = jnp.full_like(jnp.asarray(t, dtype=float), float(carbon_unit.flow))
        else:
            # Feedback dose: the manipulated dose flow is the controller signal
            # (gain-scaled), reconstructed per saved state from the control bus.
            sig = carbon_unit.required_signals[0]
            Q_carbon = jnp.stack(
                [
                    plant.signals_at(ti, solution.state[i], params_full)[sig] * carbon_unit.gain
                    for i, ti in enumerate(t)
                ]
            )
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
        eqi=eqi,
        oci=oci,
        aeration_energy=AE,
        pumping_energy=PE,
        mixing_energy=ME,
        sludge_production=sludge,
        carbon_mass=carbon,
        methane_production=methane,
        heating_energy=HE,
        effluent=averages,
        aerated_tanks=aerated,
        air_flow=air_flow,
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
        BSM1 Operational Cost Index: ``AE + PE + ME + 5·sludge`` (the updated
        benchmark convention, which adds mixing energy to the original Copp 2002
        ``AE + PE + 5·sludge``).
    aeration_energy : float
        Aeration energy AE (kWh/d).
    pumping_energy : float
        Pumping energy PE (kWh/d): the internal recycle, RAS and wastage pumps.
    sludge_production : float
        Wasted-sludge TSS mass flow (kg TSS/d), time-averaged.
    mixing_energy : float
        Mixing energy ME (kWh/d): mechanical mixing of the unaerated reactors.
    effluent : dict
        Time/flow-weighted average effluent concentrations (COD, BOD, TSS, TKN,
        SNH, SNO; g/m^3) from :func:`effluent_averages`.
    aerated_tanks : list[str]
        The reactors whose aeration was counted.
    air_flow : float or None
        Total blower air flow (m³/d, time-averaged) when an ``aeration_system``
        diffuser/blower design was supplied; ``None`` for the correlation default.
    oci_note : str
        Notes on the OCI computation.
    """

    eqi: float
    oci: float
    aeration_energy: float
    pumping_energy: float
    sludge_production: float
    mixing_energy: float = 0.0
    effluent: dict = field(default_factory=dict)
    aerated_tanks: list = field(default_factory=list)
    air_flow: float | None = None
    oci_note: str = (
        "BSM1 OCI (updated benchmark): AE + PE + ME + 5*sludge, where ME is the "
        "mechanical-mixing energy of the unaerated reactors (the original Copp "
        "2002 index omits it). Sludge production is the wastage TSS mass flow "
        "(plant TSS-inventory change neglected -- ~0 at steady state)."
    )

    def total_energy(self) -> float:
        """Total electricity draw (kWh/d) = aeration + pumping + mixing -- the
        energy basis for the GHG and cost layers."""
        return self.aeration_energy + self.pumping_energy + self.mixing_energy

    def kpis(self) -> dict:
        """Headline performance KPIs for a scenario comparison table."""
        return {
            "EQI (kg/d)": self.eqi,
            "OCI": self.oci,
            "Energy (kWh/d)": self.total_energy(),
            "Sludge (kgTSS/d)": self.sludge_production,
            "SNH (gN/m3)": self.effluent.get("SNH", float("nan")),
            "SNO (gN/m3)": self.effluent.get("SNO", float("nan")),
        }

    def report(self) -> str:
        """A labeled, units-annotated EQI / OCI breakdown (also ``str(eval)``).

        Shows each OCI term with its value, units and signed contribution to the
        index, the effluent averages, and the ``oci_note`` caveat."""
        ae_label = (
            "Aeration energy  AE (blower)" if self.air_flow is not None else "Aeration energy  AE"
        )
        terms = [(ae_label, self.aeration_energy, "kWh/d", self.aeration_energy)]
        if self.air_flow is not None:
            terms.append(("  air flow", self.air_flow, "m3/d", None))
        terms += [
            ("Pumping energy   PE", self.pumping_energy, "kWh/d", self.pumping_energy),
            ("Mixing energy    ME", self.mixing_energy, "kWh/d", self.mixing_energy),
            (
                "Sludge prod.  (x5)",
                self.sludge_production,
                "kg TSS/d",
                5.0 * self.sludge_production,
            ),
        ]
        return _render_eval_report(
            "BSM1 performance indices",
            self.eqi,
            self.oci,
            "AE + PE + ME + 5*sludge",
            terms,
            self.effluent,
            self.aerated_tanks,
            self.oci_note,
        )

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
    aeration_system=None,
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
        DO saturation used in the aeration-energy formula (gO2/m^3). This is the
        nominal saturation by BSM2 convention; it is **not** temperature-adjusted
        even when the plant runs with ``do_temperature_correction`` (where the
        reactor's actual driving-force saturation is scaled by ``C_s(T)/C_s(ref)``).
        So the reported AE reflects the BSM2 definition, not the temperature-
        corrected oxygen transfer.

    Returns
    -------
    BSM1Evaluation
        EQI, OCI and all component terms.
    """
    model = plant.units["tank1"].model
    params_full = plant.default_parameters() if params is None else jnp.asarray(params)
    # Composition is read with the ASM1 model's local param_index, so slice the
    # ASM1 block from the concatenated plant vector (see evaluate_bsm2).
    params_asm = plant._params_for_unit("tank1", params_full)
    t = solution.t

    # Reconstruct every needed output stream in a single pass over the states.
    streams = _reconstruct(
        plant,
        solution,
        params_full,
        [
            effluent_port,
            internal_recycle_port,
            ras_port,
            waste_port,
        ],
    )

    # ----- Effluent quality. -----
    eff_Q, eff_C = streams[effluent_port]
    eqi = effluent_quality_index(t, eff_C, eff_Q, model, params=params_asm)
    averages = effluent_averages(t, eff_C, eff_Q, model, params=params_asm)

    # ----- Aeration energy (actual kLa over the run). -----
    reactors = _as_reactors(plant)
    kla_hist = _kla_history(plant, solution, params, reactors)
    volumes = jnp.asarray([float(plant.units[n].volume) for n in reactors])
    # Mechanistic blower energy when a diffuser/blower design is given, else the
    # Copp-2002 correlation (see evaluate_bsm2).
    if aeration_system is not None:
        AE = blower_energy(t, kla_hist, volumes, aeration_system)
        air_flow = blower_airflow_total(t, kla_hist, volumes, aeration_system)
    else:
        AE = aeration_energy(t, kla_hist, volumes, saturation=do_saturation)
        air_flow = None
    aerated = [reactors[i] for i in range(len(reactors)) if float(jnp.max(kla_hist[:, i])) > 0.0]

    # ----- Pumping energy (internal recycle + RAS + wastage). -----
    PE = pumping_energy(
        t,
        streams[internal_recycle_port][0],
        streams[ras_port][0],
        streams[waste_port][0],
    )

    # ----- Sludge production (TSS mass flow leaving via wastage, kg/d). -----
    waste_Q, waste_C = streams[waste_port]
    tss_mass_flow = derived_TSS(waste_C, model) * waste_Q * 1e-3
    sludge = _time_average(t, tss_mass_flow)

    # ----- Mixing energy (mechanical mixing of the unaerated reactors). -----
    # BSM1 has no digester, so the digester mixing volume is zero. The updated
    # benchmark OCI includes this term; the original Copp (2002) index omits it.
    ME = mixing_energy(t, kla_hist, volumes, 0.0)

    oci = operational_cost_index(AE, PE, sludge, mixing=ME)

    return BSM1Evaluation(
        eqi=eqi,
        oci=oci,
        aeration_energy=AE,
        pumping_energy=PE,
        sludge_production=sludge,
        mixing_energy=ME,
        effluent=averages,
        aerated_tanks=aerated,
        air_flow=air_flow,
    )


# ---- GHG / carbon-footprint coupling ---------------------------------------

# The dissolved N₂O state name in the N₂O kinetic models (Pocquet 2016 form).
_N2O_SPECIES = "SN2O"


def direct_n2o_emission(
    plant,
    solution,
    params: Optional[jnp.ndarray] = None,
    *,
    n2o_species: str = _N2O_SPECIES,
    kla_ratio: float = 1.0,
) -> float:
    """Direct N₂O stripped from the activated-sludge reactors (kg N₂O-N/d).

    The activated-sludge model must track a dissolved nitrous-oxide state
    (``n2o_species``, default ``"SN2O"`` -- present in the N₂O kinetic models,
    e.g. ``asm3_2step_n2o``). N₂O is stripped at the aeration mass-transfer rate,
    so only the aerated reactors emit; this reconstructs each reactor's oxygen
    ``kLa`` (the same control-aware reconstruction ``evaluate_bsm2`` uses) and its
    dissolved N₂O trajectory, and time-averages the stripping flux
    (:func:`aquakin.plant.ghg.stripped_n2o`).

    If the model has no ``n2o_species`` state (the standard ASM1 BSM2 plant,
    which does not resolve N₂O), the direct N₂O emission is **0** -- the model has
    no nitrous oxide to strip. Use an N₂O-capable activated-sludge model to get
    a non-zero direct footprint.

    Parameters
    ----------
    plant : Plant
        A plant whose activated-sludge reactors carry ``n2o_species``.
    solution : PlantSolution
        A solved trajectory over the evaluation window.
    params : jnp.ndarray, optional
        Plant parameters used for the run (defaults to the plant defaults).
    n2o_species : str
        Dissolved N₂O-N state name (default ``"SN2O"``).
    kla_ratio : float
        N₂O-to-O₂ mass-transfer-coefficient ratio (default 1.0).

    Returns
    -------
    float
        Time-averaged stripped N₂O-N mass flow (kg N/d).
    """
    from aquakin.plant.ghg import stripped_n2o

    reactors = _as_reactors(plant)
    # Reactors whose model resolves the dissolved N₂O state.
    n2o_reactors = [n for n in reactors if n2o_species in plant.units[n].model.species_index]
    if not n2o_reactors:
        return 0.0

    t = solution.t
    kla_hist = _kla_history(plant, solution, params, n2o_reactors)
    volumes = jnp.asarray([float(plant.units[n].volume) for n in n2o_reactors])
    s_n2o = jnp.stack([solution.C_named(n, n2o_species) for n in n2o_reactors], axis=1)
    return stripped_n2o(t, kla_hist, s_n2o, volumes, kla_ratio=kla_ratio)
