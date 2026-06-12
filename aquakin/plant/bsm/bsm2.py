"""BSM2 plant builder (open-loop, Gernaey et al. 2014 / Jeppsson et al. 2007).

BSM2 wraps the BSM1 activated-sludge core with a sludge train: a primary
clarifier ahead of the reactors, and downstream a thickener, an ADM1 anaerobic
digester (with ASM1<->ADM1 interfaces) and a dewatering unit, with the two
reject-water streams (thickener overflow + dewatering reject) recycled to the
plant front.

Flowsheet (open-loop, constant-influent steady state)::

    influent ─┐
              front_mix ─→ primary ─ effluent ─→ as_mix ─→ tank1..tank5 ─→ tank5_split
    reject ───┘                │                  ↑  ↑                          │  │
                               │      internal_recycle (Qintr) ─────────────────┘  │
                               │                  │                       to_settler
                               │                  │                          │
                               │                RAS (Qr) ── underflow_split ──┤
                               │                                  ↑       settler ─→ EFFLUENT
                               │                                waste (Qw)    │
                               │                                  └───────────┘ (underflow)
                               │                                       │
                               │  primary sludge                    thickener ─ overflow ─┐
                               └────────────────→ sludge_mix ←─ underflow                 │
                                                      │                                   │
                                            asm2adm → digester → adm2asm → dewatering      │
                                                                              │  │         │
                                                          disposal ←─ underflow  reject ───┤
                                                                                           │
                                                              reject_mix ←──────────────────┘
                                                                  └─→ front_mix:reject

All controlled (pumped) flows -- internal recycle ``Qintr``, RAS ``Qr``,
wastage ``Qw``, primary sludge ``f_PS·Q`` -- are fixed setpoints (see the
SplitterUnit flow mode / clarifier ``underflow_Q``). The thickener and
dewatering underflow flows are concentration-dependent but sit on the low-gain
reject loop. The storage tank / hydraulic delay / bypass and the controllers
are omitted (open-loop steady state).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import jax.numpy as jnp

from aquakin.plant.cstr import CSTRUnit
from aquakin.plant.digester import ADM1DigesterUnit
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.interfaces import ADM1toASM1, ASM1toADM1
from aquakin.plant.mixer import MixerUnit, SplitterUnit
from aquakin.plant.plant import Plant
from aquakin.plant.streams import Stream
from aquakin.plant.primary_clarifier import PrimaryClarifier
from aquakin.plant.separators import IdealThickener
from aquakin.plant.takacs import TakacsClarifier

# Reference BSM2 design values (Gernaey et al. 2014; asm1init/adm1init).
BSM2_Q_REF = 20648.0  # m³/d, reference dry-weather average (sizes the pumps)
BSM2_TANK_VOLUMES = (1500.0, 1500.0, 3000.0, 3000.0, 3000.0)  # m³
BSM2_KLA = (0.0, 0.0, 120.0, 120.0, 60.0)  # d⁻¹ (open-loop)
BSM2_DO_SATURATION = 8.0  # gO2/m³
BSM2_INTERNAL_RECYCLE = 3.0 * BSM2_Q_REF   # Qintr
BSM2_RAS = 1.0 * BSM2_Q_REF                # Qr
BSM2_WASTAGE = 300.0                       # Qw
# Scheduled (timed) wastage: the waste pump alternates between a low and a high
# rate over the 609-day evaluation to manage the sludge inventory (reginit
# Qw_low / Qw_high; step times from the reference wastage reference vector --
# ~182-day half-year blocks: low, high, low, high).
BSM2_WASTAGE_LOW = 300.0                    # Qw_low
BSM2_WASTAGE_HIGH = 450.0                   # Qw_high
BSM2_WASTAGE_STEPS = (182.0, 364.0, 546.0)  # d, schedule step times
BSM2_STORAGE_VOLUME = 160.0                # m³, reject equalisation tank (VOL_S)
BSM2_STORAGE_OUTFLOW = 0.0                 # m³/d, controlled release (Qstorage)
BSM2_STORAGE_OUTFLOW_MAX = 1500.0          # m³/d, release pump capacity (Qstorage_max)
# Closed-loop reject control: a proportional level controller on the storage
# release, holding the tank near a mid setpoint and releasing the reject
# smoothly (instead of fill-and-bypass).
BSM2_STORAGE_LEVEL_SETPOINT_FRAC = 0.5     # target level as a fraction of Vmax
BSM2_STORAGE_LEVEL_GAIN = 30.0             # m³/d release per m³ above setpoint
BSM2_BYPASS_Q = 60000.0                    # influent flow above this bypasses treatment
BSM2_PRIMARY_VOLUME = 900.0  # m³
BSM2_PRIMARY_FPS = 0.007
BSM2_CLARIFIER_AREA = 1500.0  # m²
BSM2_CLARIFIER_HEIGHT = 4.0   # m
BSM2_DIGESTER_VOLUME = 3400.0  # m³ liquid
BSM2_DIGESTER_T = 308.15       # K (35 °C)
BSM2_THICKENER_TSS_PERCENT = 7.0
BSM2_DEWATERING_TSS_PERCENT = 28.0
BSM2_SEPARATOR_REMOVAL = 98.0
BSM2_CARBON_FLOW = 2.0          # m³/d external carbon dosed to reactor 1
BSM2_CARBON_CONC = 400000.0     # gCOD/m³ readily-biodegradable (SS) carbon source
BSM2_AS_TEMPERATURE_K = 288.15  # K (15 °C) -- the BSM2 ASM1 reference temperature

# Closed-loop dissolved-oxygen / kLa control (reginit_bsm2). A PI controller
# senses SO in reactor 4 and manipulates its aeration kLa; reactors 3 and 5
# scale off the same signal. The constants are the reference DO loop tuning
# (Kp=KSO4, Ti=TiSO4, Tt=TtSO4 in days; the kLa offset and DO setpoint).
BSM2_DO_SETPOINT = 2.0          # gO2/m³ (SO4ref)
BSM2_DO_KP = 25.0               # PI proportional gain (KSO4)
BSM2_DO_TI = 0.002              # PI integral time, d (TiSO4)
BSM2_DO_TT = 0.001              # anti-windup tracking time, d (TtSO4)
BSM2_DO_KLA_OFFSET = 120.0      # kLa bias, d⁻¹ (KLa4offset)
BSM2_DO_KLA_MAX = 360.0         # kLa saturation upper bound, d⁻¹
# Per-tank kLa gains relative to the reactor-4 control signal (KLa{3,4,5}gain).
BSM2_DO_KLA_GAINS = {"tank3": 1.0, "tank4": 1.0, "tank5": 0.5}

# Published BSM2 constant-influent composition (the open-loop operating point;
# gCOD/m³ or gN/m³, SALK in mol/m³). Q is BSM2_Q_REF.
BSM2_CONSTANT_INFLUENT = {
    "SI": 27.2262, "SS": 58.1762, "XI": 92.499, "XS": 363.9435, "XB_H": 50.6833,
    "XB_A": 0.0, "XP": 0.0, "SO": 0.0, "SNO": 0.0, "SNH": 23.8595, "SND": 5.6516,
    "XND": 16.1298, "SALK": 7.0,
}

# BSM2 ASM1 kinetic/stoichiometric parameters (asm1init_bsm2, calibrated at
# 15 °C). ``KNH_H`` is set ~0 because the BSM/IWA ASM1 has no heterotroph
# ammonia-limitation term (aquakin's ASM1 adds one; disabling it recovers the
# benchmark behaviour). Names are aquakin's ASM1 parameter names.
BSM2_ASM1_PARAMETERS = {
    "muH": 4.0, "KS": 10.0, "KOH": 0.2, "KNO": 0.5, "KNH_H": 1e-6, "etag": 0.8,
    "muA": 0.5, "KNH_A": 1.0, "KOA": 0.4, "bH": 0.3, "bA": 0.05, "ka": 0.05,
    "kh": 3.0, "KX": 0.1, "etah": 0.8, "Y_H": 0.67, "Y_A": 0.24, "i_XB": 0.08,
    "i_XP": 0.06, "f_P": 0.08,
}


def bsm2_asm1_network(asm1_network=None):
    """ASM1 network configured for BSM2: temperature corrections re-referenced
    to 15 °C (the BSM2 ASM1 reference temperature, matching ``bsm2_parameters``).

    The shipped ``asm1`` corrections are referenced to 20 °C; this moves ``ref_T``
    to 288.15 K while keeping the (BSM2) slopes. Use the returned network for
    **both** ``build_bsm2`` and the influent (e.g. ``bsm2_constant_influent``)
    so their network identities match and a temperature-carrying influent drives
    the AS kinetics from the correct 15 °C base.
    """
    import aquakin
    asm1 = asm1_network if asm1_network is not None else aquakin.load_network("asm1")
    if not getattr(asm1, "temperature_corrections", None):
        return asm1
    return dataclasses.replace(asm1, temperature_corrections=[
        (idx, ln_theta, BSM2_AS_TEMPERATURE_K, cond)
        for (idx, ln_theta, _ref, cond) in asm1.temperature_corrections
    ])


def bsm2_asm1_parameter_vector(asm1_network):
    """ASM1 parameter vector with the BSM2 (15 °C) overrides applied."""
    p = asm1_network.default_parameters()
    for name, val in BSM2_ASM1_PARAMETERS.items():
        if name in asm1_network.parameters:
            p = p.at[asm1_network.parameters.index(name)].set(val)
    return p


def bsm2_parameters(asm1_network, adm1_network):
    """Full BSM2 plant parameter vector: BSM2 ASM1 block + default ADM1 block.

    Pass to ``plant.solve(params=...)``. The water-line block carries the BSM2
    ASM1 values (the network defaults are the BSM1/20 °C set); the digester block
    uses the ADM1 defaults, which are already the BSM2 values.
    """
    return jnp.concatenate([
        bsm2_asm1_parameter_vector(asm1_network),
        adm1_network.default_parameters(),
    ])


def bsm2_constant_influent(asm1_network, Q: float = BSM2_Q_REF) -> InfluentSeries:
    """The published BSM2 constant influent as an :class:`InfluentSeries`."""
    C = asm1_network.default_concentrations() * 0.0
    for sp, v in BSM2_CONSTANT_INFLUENT.items():
        C = C.at[asm1_network.species_index[sp]].set(float(v))
    return InfluentSeries(t=jnp.asarray([0.0, 1.0e4]), Q=jnp.full((2,), float(Q)),
                          C=jnp.tile(C, (2, 1)), network=asm1_network)


def bsm2_wastage_schedule(low: float = BSM2_WASTAGE_LOW,
                          high: float = BSM2_WASTAGE_HIGH,
                          steps=BSM2_WASTAGE_STEPS):
    """The BSM2 scheduled wastage flow ``Qw(t)`` as a
    :class:`~aquakin.plant.schedule.PiecewiseConstantSchedule`.

    The waste pump steps low → high → low → high at the ``steps`` times (the
    reference's ~182-day half-year blocks over the 609-day evaluation), managing
    the sludge inventory seasonally. Pass to ``build_bsm2(wastage_schedule=...)``.
    """
    from aquakin.plant.schedule import PiecewiseConstantSchedule
    values = [low, high, low, high]
    if len(values) != len(steps) + 1:
        raise ValueError("wastage schedule needs len(steps)+1 values")
    return PiecewiseConstantSchedule(list(steps), values)


def build_bsm2(
    asm1_network: Optional["object"] = None,
    adm1_network: Optional["object"] = None,
    *,
    Q_ref: float = BSM2_Q_REF,
    conditions: Optional[dict] = None,
    carbon_flow: float = BSM2_CARBON_FLOW,
    carbon_conc: float = BSM2_CARBON_CONC,
    do_control: bool = False,
    reject_storage: bool = False,
    reject_control: bool = False,
    storage_volume: float = BSM2_STORAGE_VOLUME,
    storage_output_flow: float = BSM2_STORAGE_OUTFLOW,
    influent_bypass: bool = False,
    bypass_threshold: float = BSM2_BYPASS_Q,
    wastage_schedule: Optional["object"] = None,
) -> Plant:
    """Assemble the BSM2 plant (open-loop by default; closed DO/kLa loop optional).

    Parameters
    ----------
    asm1_network : CompiledNetwork, optional
        ASM1 network for the water line. Defaults to ``load_network("asm1")``.
    adm1_network : CompiledNetwork, optional
        ADM1 network for the digester. Defaults to ``load_network("adm1")``.
    Q_ref : float
        Reference flow used to size the fixed recycle pumps.
    conditions : dict, optional
        Per-tank ASM1 condition values (e.g. ``{"T": ...}``). Defaults to the
        ASM1 network's declared defaults.
    do_control : bool, optional
        If True, close the dissolved-oxygen loop: a :class:`PIController` senses
        ``SO`` in reactor 4 and manipulates its aeration ``kLa`` (with reactors 3
        and 5 scaled off the same signal), instead of the fixed open-loop ``kLa``
        of reactors 3-5. Default False (open-loop fixed aeration).
    reject_storage : bool, optional
        If True, route the recycled reject water through a :class:`StorageTank`
        (a variable-volume equalisation tank with a level-gated overflow bypass)
        before returning it to the plant front, instead of recycling it directly.
        The tank buffers the reject load and releases it at
        ``storage_output_flow``; with the default 0 release it fills and bypasses,
        so the open-loop steady state is unchanged (all reject still reaches the
        front). Default False.
    reject_control : bool, optional
        If True, close the reject loop: the storage tank runs a proportional
        **level controller** on its release (holding a mid-level setpoint and
        releasing the reject smoothly, capped at the pump capacity) instead of a
        fixed ``storage_output_flow``. Implies a storage tank (so it does not
        fill and bypass). Default False.
    storage_volume : float, optional
        Maximum reject-storage-tank volume (m³). Used with ``reject_storage`` or
        ``reject_control``.
    storage_output_flow : float, optional
        Fixed release flow from the storage tank (m³/d), used when
        ``reject_storage=True`` and ``reject_control=False``. Default 0.
    influent_bypass : bool, optional
        If True, add the BSM2 hydraulic influent bypass: raw influent flow above
        ``bypass_threshold`` is diverted around the whole treatment train
        (primary, AS, secondary clarifier) and rejoined with the clarified
        effluent, modelling wet-weather hydraulic overload. This **changes the
        influent entry point**: wire the influent to ``"bypass_split.in"``
        (not ``"front_mix.fresh"``), and the final plant effluent becomes
        ``"effluent_mix.out"`` (``evaluate_bsm2`` auto-detects it). Default False.
    bypass_threshold : float, optional
        Influent flow limit (m³/d) above which the excess bypasses; the BSM2
        default is 60000. Only used when ``influent_bypass=True``.
    wastage_schedule : PiecewiseConstantSchedule, optional
        A time schedule for the wastage flow ``Qw(t)`` (see
        :func:`bsm2_wastage_schedule`). When given, the secondary-clarifier
        underflow follows ``Qr + Qw(t)`` so the waste pump steps on the schedule
        (the BSM2 timed-wastage strategy) rather than the constant ``Qw=300``.
        Default None (constant wastage).

    Returns
    -------
    Plant
        The wired BSM2 plant. The caller adds the influent and connects it to
        ``front_mix.fresh`` -- or to ``bypass_split.in`` when
        ``influent_bypass=True`` (mirroring :func:`build_bsm1`).
    """
    from aquakin.plant.control import PIController
    from aquakin.plant.storage import StorageTank
    import aquakin

    asm1 = asm1_network if asm1_network is not None else aquakin.load_network("asm1")
    adm1 = adm1_network if adm1_network is not None else aquakin.load_network("adm1")

    # The AS reactors operate at the temperature where their parameters are
    # defined -- the reference temperature of the ASM1 temperature corrections
    # (288.15 K for the BSM2-configured network from bsm2_asm1_network, 293.15 K
    # for the plain shipped asm1). Setting the static condition there makes the
    # correction unity at steady state, so a constant-temperature run reproduces
    # the (uncorrected) reference exactly; a temperature-carrying influent then
    # drives the correction away from it (seasonal kinetics).
    if conditions is None:
        conditions = {name: asm1._condition_defaults[name]
                      for name in asm1.conditions_required}
        if "T" in conditions and getattr(asm1, "temperature_corrections", None):
            conditions["T"] = float(asm1.temperature_corrections[0][2])

    Qintr = 3.0 * Q_ref
    Qr = 1.0 * Q_ref
    Qw = BSM2_WASTAGE
    # The secondary-clarifier underflow is RAS + wastage. With a wastage schedule
    # it becomes a time schedule Qr + Qw(t); the underflow_split then sends Qr to
    # RAS and the remainder (the scheduled Qw) to wastage. The IC operating point
    # uses the schedule's first value so the settled-blanket start is consistent.
    if wastage_schedule is not None:
        Q_settler_underflow = wastage_schedule.shifted(Qr)
        Q_settler_underflow_init = Qr + float(wastage_schedule.at(0.0))
    else:
        Q_settler_underflow = Qr + Qw
        Q_settler_underflow_init = Q_settler_underflow

    plant = Plant("BSM2")
    # Recycle seeds carry a nominal temperature so a temperature-aware influent
    # ignites T propagation around the reject loop from the first pass (the value
    # is overwritten within a couple of passes by the real recycle temperature).
    # For a temperature-agnostic influent the front mixer sees the (None) fresh
    # feed and the seed temperature is simply never used.
    seed = Stream(Q=jnp.asarray(0.0), C=asm1.default_concentrations(),
                  network=asm1, T=jnp.asarray(BSM2_AS_TEMPERATURE_K))

    # A storage tank is built when either the fixed-release storage or the
    # closed-loop reject controller is requested.
    use_storage = reject_storage or reject_control

    # ----- Influent bypass (optional): divert wet-weather peak flow around the
    # whole treatment train. The split is on the *raw influent* flow (an external
    # input, so the exact recycle-flow solve stays valid); the diverted raw flow
    # rejoins the clarified effluent downstream of the secondary clarifier.
    if influent_bypass:
        plant.add_unit(SplitterUnit(
            name="bypass_split", network=asm1, threshold=float(bypass_threshold),
            threshold_port="bypass", remainder_port="to_plant"))

    # ----- Front: combine raw influent with the recycled reject water. With a
    # reject storage tank the reject returns on two ports (the released stream
    # and the level-gated overflow bypass); otherwise on one combined port.
    front_reject_ports = (["storage_out", "storage_bypass"] if use_storage
                          else ["reject"])
    plant.add_unit(MixerUnit(name="front_mix",
                             input_port_names=["fresh"] + front_reject_ports,
                             network=asm1))
    plant.add_unit(PrimaryClarifier(name="primary", network=asm1,
                                    volume=BSM2_PRIMARY_VOLUME, f_PS=BSM2_PRIMARY_FPS))

    # ----- Activated sludge: mixer + 5 CSTRs + internal recycle. -----
    as_ports = ["primary_eff", "internal_recycle", "ras"]
    if carbon_flow > 0:
        as_ports.append("carbon")   # external carbon dosed to reactor 1
    plant.add_unit(MixerUnit(name="as_mix", input_port_names=as_ports, network=asm1))
    for i in range(5):
        tank = f"tank{i + 1}"
        aerated = BSM2_KLA[i] > 0
        kla = {"SO": BSM2_KLA[i]} if aerated else {}
        sat = {"SO": BSM2_DO_SATURATION} if aerated else {}
        controlled = {}
        if do_control and tank in BSM2_DO_KLA_GAINS:
            # The DO controller drives this tank's oxygen kLa; drop the fixed
            # kLa (the signal replaces it) but keep the DO saturation for the
            # aeration term.
            kla = {}
            controlled = {"SO": ("do_kla", BSM2_DO_KLA_GAINS[tank])}
        plant.add_unit(CSTRUnit(
            name=tank, network=asm1, volume=BSM2_TANK_VOLUMES[i],
            input_port_names=["inlet"], conditions=conditions, kla=kla, C_sat=sat,
            controlled_kla=controlled))

    # Closed DO loop: a PI controller sensing reactor 4's oxygen, publishing the
    # 'do_kla' signal the aerobic reactors consume.
    if do_control:
        plant.add_unit(PIController(
            name="do_control", network=asm1, measured_species="SO",
            setpoint=BSM2_DO_SETPOINT, Kp=BSM2_DO_KP, Ti=BSM2_DO_TI, Tt=BSM2_DO_TT,
            offset=BSM2_DO_KLA_OFFSET, out_min=0.0, out_max=BSM2_DO_KLA_MAX,
            signal_name="do_kla"))

    plant.add_unit(SplitterUnit(
        name="tank5_split", network=asm1,
        output_port_flows={"internal_recycle": Qintr}, remainder_port="to_settler"))

    # ----- Secondary clarifier (Takács) + RAS/wastage split. -----
    plant.add_unit(TakacsClarifier(
        name="settler", network=asm1, area=BSM2_CLARIFIER_AREA,
        height=BSM2_CLARIFIER_HEIGHT, underflow_Q=Q_settler_underflow,
        init_underflow_Q=Q_settler_underflow_init))
    plant.add_unit(SplitterUnit(
        name="underflow_split", network=asm1,
        output_port_flows={"ras": Qr}, remainder_port="waste"))

    # Final-effluent combiner: clarified effluent + the bypassed raw influent.
    if influent_bypass:
        plant.add_unit(MixerUnit(
            name="effluent_mix",
            input_port_names=["treated", "bypass"], network=asm1))

    # ----- Sludge train: thickener -> digester -> dewatering. -----
    plant.add_unit(IdealThickener(
        name="thickener", network=asm1, target_tss_percent=BSM2_THICKENER_TSS_PERCENT,
        tss_removal_percent=BSM2_SEPARATOR_REMOVAL, nominal_underflow_fraction=0.03))
    # Combine primary sludge + thickened secondary sludge into the digester feed.
    plant.add_unit(MixerUnit(
        name="sludge_mix",
        input_port_names=["primary_sludge", "thickener_under"], network=asm1))
    plant.add_unit(ADM1DigesterUnit(
        name="digester", network=adm1, volume=BSM2_DIGESTER_VOLUME,
        conditions={"T": BSM2_DIGESTER_T}))
    plant.add_unit(IdealThickener(
        name="dewatering", network=asm1, target_tss_percent=BSM2_DEWATERING_TSS_PERCENT,
        tss_removal_percent=BSM2_SEPARATOR_REMOVAL, nominal_underflow_fraction=0.02))

    # Combine the two reject-water streams for the recycle to the front.
    plant.add_unit(MixerUnit(
        name="reject_mix",
        input_port_names=["thickener_reject", "dewatering_reject"], network=asm1))

    # Optional reject equalisation tank: buffer the combined reject and release
    # it at a controlled rate, with a level-gated overflow bypass. Under
    # reject_control the release follows a proportional level controller (the
    # tank holds a mid setpoint and releases the reject smoothly); otherwise it
    # is the fixed storage_output_flow.
    if use_storage:
        if reject_control:
            storage = StorageTank(
                name="reject_storage", network=asm1, volume=storage_volume,
                level_setpoint=BSM2_STORAGE_LEVEL_SETPOINT_FRAC * storage_volume,
                level_gain=BSM2_STORAGE_LEVEL_GAIN,
                output_flow_max=BSM2_STORAGE_OUTFLOW_MAX)
        else:
            storage = StorageTank(
                name="reject_storage", network=asm1, volume=storage_volume,
                output_flow=storage_output_flow)
        plant.add_unit(storage)

    # Cross-network interfaces (ASM1 <-> ADM1).
    asm2adm = ASM1toADM1(source_network=asm1, target_network=adm1)
    adm2asm = ADM1toASM1(source_network=adm1, target_network=asm1)

    # ----- Wiring -----
    # Influent bypass: the raw influent enters the splitter; the within-capacity
    # flow goes to the plant, the excess skips the train and rejoins the effluent.
    if influent_bypass:
        plant.connect("bypass_split.to_plant", "front_mix.fresh")
        plant.connect("bypass_split.bypass", "effluent_mix.bypass")
        plant.connect("settler.overflow", "effluent_mix.treated")
    # Front line (bare endpoints use each unit's sole in/out port).
    plant.connect("front_mix", "primary")
    plant.connect("primary.effluent", "as_mix.primary_eff")
    plant.connect("as_mix", "tank1")
    plant.connect("tank1", "tank2")
    plant.connect("tank2", "tank3")
    plant.connect("tank3", "tank4")
    plant.connect("tank4", "tank5")
    plant.connect("tank5", "tank5_split")
    if do_control:
        # Sense reactor-4 oxygen for the DO controller (a measurement tap; the
        # controller produces no material stream).
        plant.connect("tank4", "do_control.measured")
    plant.connect("tank5_split.to_settler", "settler")
    plant.connect("settler.underflow", "underflow_split")
    # AS recycles (back-edges). Seeded with a temperature-carrying zero-flow
    # stream (not the default auto-seed, which is temperature-agnostic) so a
    # temperature-aware influent ignites T propagation around the loop.
    plant.connect("tank5_split.internal_recycle", "as_mix.internal_recycle",
                  initial_value=seed)
    plant.connect("underflow_split.ras", "as_mix.ras", initial_value=seed)
    # Sludge train (the digester crosses ASM1 <-> ADM1 via the interfaces).
    plant.connect("primary.underflow", "sludge_mix.primary_sludge")
    plant.connect("underflow_split.waste", "thickener")
    plant.connect("thickener.underflow", "sludge_mix.thickener_under")
    plant.connect("sludge_mix", "digester", translator=asm2adm)
    plant.connect("digester", "dewatering", translator=adm2asm)
    # Reject-water recycle to the front (back-edge; temperature-carrying seed).
    plant.connect("thickener.overflow", "reject_mix.thickener_reject")
    plant.connect("dewatering.overflow", "reject_mix.dewatering_reject")
    if use_storage:
        # reject_mix -> storage tank; the released stream and the overflow
        # bypass both return to the front (both back-edges, seeded).
        plant.connect("reject_mix", "reject_storage.in", initial_value=seed)
        plant.connect("reject_storage.out", "front_mix.storage_out",
                      initial_value=seed)
        plant.connect("reject_storage.bypass", "front_mix.storage_bypass",
                      initial_value=seed)
    else:
        plant.connect("reject_mix", "front_mix.reject", initial_value=seed)
    # dewatering:underflow -> sludge disposal (leaves the plant; not routed).

    # External carbon dosing to reactor 1 (a constant readily-biodegradable SS
    # source) -- supports denitrification in the anoxic tanks and is part of the
    # BSM2 plant design, so the builder adds it directly.
    if carbon_flow > 0:
        carbon_C = asm1.default_concentrations()
        carbon_C = (carbon_C * 0.0).at[asm1.species_index["SS"]].set(float(carbon_conc))
        carbon = InfluentSeries(
            t=jnp.asarray([0.0, 1.0e9]), Q=jnp.full((2,), float(carbon_flow)),
            C=jnp.tile(carbon_C, (2, 1)), network=asm1,
            T=jnp.full((2,), float(conditions.get("T", BSM2_AS_TEMPERATURE_K))))
        plant.add_influent("external_carbon", carbon, to="as_mix.carbon")

    return plant
