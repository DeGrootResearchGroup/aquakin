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

from aquakin.plant.cstr import Aeration, CSTRUnit
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
BSM2_HYDRAULIC_DELAY_TAU = 0.02            # d, influent hydraulic-lag time constant (~30 min)
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
    return asm1_network.influent(BSM2_CONSTANT_INFLUENT, Q=Q)


# ---------------------------------------------------------------------------
# Optional-feature option objects
# ---------------------------------------------------------------------------
# Each groups the coupled parameters of one optional BSM2 feature into a single
# object, so build_bsm2 takes a handful of feature objects instead of a dozen
# cross-coupled boolean/float flags. Passing the object (with its defaults)
# enables the feature; leaving the argument ``None`` leaves it off. Frozen so an
# instance is a safe shared default.

@dataclasses.dataclass(frozen=True)
class ExternalCarbon:
    """External-carbon dosing to reactor 1 (BSM2 default-on).

    A constant readily-biodegradable (``SS``) carbon source fed to the first
    anoxic reactor to support denitrification. ``build_bsm2(carbon=None)``
    disables it.

    Parameters
    ----------
    flow : float
        Dose flow (m³/d). Default 2.
    conc : float
        Source ``SS`` concentration (gCOD/m³). Default 4e5.
    """

    flow: float = BSM2_CARBON_FLOW
    conc: float = BSM2_CARBON_CONC


@dataclasses.dataclass(frozen=True)
class RejectStorage:
    """Reject-water equalisation tank on the recycle line.

    Routes the recycled reject water through a variable-volume
    :class:`~aquakin.plant.storage.StorageTank` with a level-gated overflow
    bypass before returning it to the front. With the default fixed release
    (``output_flow``) and ``control=False`` the tank fills and bypasses, so the
    open-loop steady state is unchanged; ``control=True`` runs a proportional
    **level controller** on the release instead (holding a mid-level setpoint and
    releasing the reject smoothly, capped at the pump capacity).

    Parameters
    ----------
    volume : float
        Maximum tank volume (m³). Default 160.
    output_flow : float
        Fixed release flow (m³/d) when ``control=False``. Default 0
        (fill-and-bypass).
    control : bool
        Close the reject loop with a proportional level controller instead of a
        fixed release. Default False.
    """

    volume: float = BSM2_STORAGE_VOLUME
    output_flow: float = BSM2_STORAGE_OUTFLOW
    control: bool = False


@dataclasses.dataclass(frozen=True)
class InfluentBypass:
    """Wet-weather hydraulic influent bypass.

    Raw influent flow above ``threshold`` is diverted around the whole treatment
    train (primary, AS, secondary clarifier) and rejoined with the clarified
    effluent. Enabling it moves the influent entry to ``bypass_split.in`` and the
    final effluent to ``effluent_mix.out`` -- but callers read
    :attr:`~aquakin.plant.plant.Plant.influent_endpoint` /
    :attr:`~aquakin.plant.plant.Plant.effluent_endpoint` rather than these
    literals, so the move is transparent.

    Parameters
    ----------
    threshold : float
        Influent flow limit (m³/d) above which the excess bypasses. Default
        60000.
    """

    threshold: float = BSM2_BYPASS_Q


@dataclasses.dataclass(frozen=True)
class HydraulicDelay:
    """First-order hydraulic lag on the raw influent.

    Inserts a :class:`~aquakin.plant.delay.HydraulicDelayUnit` (a first-order lag
    on flow and load) front-most, modelling the sewer/channel transport delay
    ahead of the works. Enabling it moves the influent entry to
    ``influent_delay.in`` (read it from
    :attr:`~aquakin.plant.plant.Plant.influent_endpoint`).

    Parameters
    ----------
    tau : float
        Lag time constant (days). Default ~0.02 (≈30 min).
    """

    tau: float = BSM2_HYDRAULIC_DELAY_TAU


# The BSM2 default carbon dose (a frozen instance -> safe as a default arg).
_DEFAULT_CARBON = ExternalCarbon()


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
    carbon: Optional["ExternalCarbon"] = _DEFAULT_CARBON,
    do_control: bool = False,
    reject: Optional["RejectStorage"] = None,
    bypass: Optional["InfluentBypass"] = None,
    hydraulic_delay: Optional["HydraulicDelay"] = None,
    wastage_schedule: Optional["object"] = None,
) -> Plant:
    """Assemble the BSM2 plant (open-loop by default; closed DO/kLa loop optional).

    The optional features are configured with small **option objects** -- pass
    the object to enable the feature, leave the argument ``None`` to leave it off
    -- so the builder takes a handful of feature objects instead of a dozen
    cross-coupled flags. The influent / effluent entry points move with some
    features, but the caller reads them from :attr:`Plant.influent_endpoint` /
    :attr:`Plant.effluent_endpoint` (set here) rather than hard-coding a port, so
    a feature can never silently mis-wire the influent.

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
    carbon : ExternalCarbon or None, optional
        External-carbon dosing to reactor 1 (default :class:`ExternalCarbon`,
        the BSM2 dose). Pass ``None`` to disable dosing.
    do_control : bool, optional
        If True, close the dissolved-oxygen loop: a :class:`PIController` senses
        ``SO`` in reactor 4 and manipulates its aeration ``kLa`` (with reactors 3
        and 5 scaled off the same signal), instead of the fixed open-loop ``kLa``
        of reactors 3-5. Default False (open-loop fixed aeration).
    reject : RejectStorage or None, optional
        Route the recycled reject water through an equalisation
        :class:`StorageTank` (see :class:`RejectStorage` for the fixed-release vs
        level-controlled options). Default ``None`` (reject recycled directly).
    bypass : InfluentBypass or None, optional
        Add the wet-weather hydraulic influent bypass (see :class:`InfluentBypass`).
        Moves the influent entry to ``bypass_split.in`` and the effluent to
        ``effluent_mix.out`` -- both reported on the plant's endpoint attributes.
        Default ``None``.
    hydraulic_delay : HydraulicDelay or None, optional
        Insert a first-order hydraulic lag on the raw influent (see
        :class:`HydraulicDelay`). Moves the influent entry to
        ``influent_delay.in`` (reported on :attr:`Plant.influent_endpoint`).
        Default ``None``.
    wastage_schedule : PiecewiseConstantSchedule, optional
        A time schedule for the wastage flow ``Qw(t)`` (see
        :func:`bsm2_wastage_schedule`). When given, the secondary-clarifier
        underflow follows ``Qr + Qw(t)`` so the waste pump steps on the schedule
        (the BSM2 timed-wastage strategy) rather than the constant ``Qw=300``.
        Default None (constant wastage).

    Returns
    -------
    Plant
        The wired BSM2 plant, with :attr:`Plant.influent_endpoint` /
        :attr:`Plant.effluent_endpoint` set. The caller adds the influent with
        ``plant.add_influent("feed", series)`` (which wires to the recorded
        front) or ``to=plant.influent_endpoint``. Mirrors :func:`build_bsm1`.
    """
    from aquakin.plant.delay import HydraulicDelayUnit
    from aquakin.plant.storage import StorageTank
    import aquakin

    # ----- Translate the feature option objects into the internal flags/values
    # the wiring below uses. A ``None`` argument means the feature is off.
    reject_storage = reject is not None
    reject_control = reject is not None and reject.control
    storage_volume = reject.volume if reject is not None else BSM2_STORAGE_VOLUME
    storage_output_flow = (reject.output_flow if reject is not None
                           else BSM2_STORAGE_OUTFLOW)
    influent_bypass = bypass is not None
    bypass_threshold = bypass.threshold if bypass is not None else BSM2_BYPASS_Q
    use_delay = hydraulic_delay is not None
    delay_tau = hydraulic_delay.tau if hydraulic_delay is not None else BSM2_HYDRAULIC_DELAY_TAU
    carbon_flow = carbon.flow if carbon is not None else 0.0
    carbon_conc = carbon.conc if carbon is not None else BSM2_CARBON_CONC

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

    # ----- Influent hydraulic delay (optional): a first-order lag on the raw
    # influent flow and load, modelling the sewer/channel transport delay. Added
    # front-most; its outlet feeds whatever the influent would otherwise enter.
    if use_delay:
        delay_C = asm1.concentrations(BSM2_CONSTANT_INFLUENT)
        plant.add_unit(HydraulicDelayUnit(
            name="influent_delay", network=asm1, tau=float(delay_tau),
            initial_flow=Q_ref, initial_concentrations=delay_C))

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
        if do_control and tank in BSM2_DO_KLA_GAINS:
            # Closed DO loop: one PI controller (named 'do_control') sensing
            # reactor 4's oxygen drives the aerobic reactors' kLa, at per-tank
            # gains. The plant auto-wires the shared controller from these specs.
            aeration = Aeration(
                do_setpoint=BSM2_DO_SETPOINT, do_sat=BSM2_DO_SATURATION,
                controller="do_control", sensor="tank4",
                gain=BSM2_DO_KLA_GAINS[tank], Kp=BSM2_DO_KP, Ti=BSM2_DO_TI,
                Tt=BSM2_DO_TT, kla_offset=BSM2_DO_KLA_OFFSET, kla_min=0.0,
                kla_max=BSM2_DO_KLA_MAX)
        elif BSM2_KLA[i] > 0:
            aeration = Aeration(kla=BSM2_KLA[i], do_sat=BSM2_DO_SATURATION)
        else:
            aeration = None
        plant.add_unit(CSTRUnit(
            name=tank, network=asm1, volume=BSM2_TANK_VOLUMES[i],
            input_port_names=["inlet"], conditions=conditions, aeration=aeration))

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
    # The raw influent enters at the bypass splitter if present, else the front
    # mixer; a hydraulic delay (if present) sits ahead of that and feeds it.
    fresh_entry = "bypass_split.in" if influent_bypass else "front_mix.fresh"
    if use_delay:
        plant.connect("influent_delay.out", fresh_entry)
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
    # The DO controller and its reactor-4 measurement tap are auto-wired from the
    # reactors' closed-loop Aeration specs (Plant._materialize_aeration).
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
        carbon_influent = asm1.influent(
            {"SS": carbon_conc}, Q=carbon_flow,
            T=conditions.get("T", BSM2_AS_TEMPERATURE_K))
        plant.add_influent("external_carbon", carbon_influent, to="as_mix.carbon")

    # Record the canonical entry / exit endpoints so callers never hard-code a
    # port: the influent enters the hydraulic delay (front-most) if present, else
    # the bypass splitter if present, else the front mixer; the final effluent is
    # the bypass combiner's outlet when bypassing, else the secondary overflow.
    plant.influent_endpoint = (
        "influent_delay.in" if use_delay
        else "bypass_split.in" if influent_bypass
        else "front_mix.fresh")
    plant.effluent_endpoint = (
        "effluent_mix.out" if influent_bypass else "settler.overflow")

    # Semantic stream shortcuts (plant.stream(sol, "effluent"), plant.list_streams())
    # so the engineer reads "effluent" / "ras" / "reject" / "primary_sludge" /
    # "digester_gas" (the last via plant.digester_gas) rather than the internal
    # "unit.port". "effluent" tracks the (option-dependent) effluent_endpoint.
    plant.register_stream("effluent", plant.effluent_endpoint)
    plant.register_stream("internal_recycle", "tank5_split.internal_recycle")
    plant.register_stream("ras", "underflow_split.ras")
    plant.register_stream("wastage", "underflow_split.waste")
    plant.register_stream("primary_effluent", "primary.effluent")
    plant.register_stream("primary_sludge", "primary.underflow")
    plant.register_stream("thickener_overflow", "thickener.overflow")
    plant.register_stream("reject", "reject_mix.out")
    plant.register_stream("dewatering_reject", "dewatering.overflow")
    plant.register_stream("disposal_sludge", "dewatering.underflow")

    return plant
