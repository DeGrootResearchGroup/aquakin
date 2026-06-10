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

from typing import Optional

import jax.numpy as jnp

from aquakin.plant.cstr import CSTRUnit
from aquakin.plant.digester import ADM1DigesterUnit
from aquakin.plant.interfaces import ADM1toASM1, ASM1toADM1
from aquakin.plant.mixer import MixerUnit, SplitterUnit
from aquakin.plant.plant import Plant
from aquakin.plant.primary_clarifier import PrimaryClarifier
from aquakin.plant.separators import IdealThickener
from aquakin.plant.streams import Stream
from aquakin.plant.takacs import TakacsClarifier

# Reference BSM2 design values (Gernaey et al. 2014; asm1init/adm1init).
BSM2_Q_REF = 20648.0  # m³/d, reference dry-weather average (sizes the pumps)
BSM2_TANK_VOLUMES = (1500.0, 1500.0, 3000.0, 3000.0, 3000.0)  # m³
BSM2_KLA = (0.0, 0.0, 120.0, 120.0, 60.0)  # d⁻¹ (open-loop)
BSM2_DO_SATURATION = 8.0  # gO2/m³
BSM2_INTERNAL_RECYCLE = 3.0 * BSM2_Q_REF   # Qintr
BSM2_RAS = 1.0 * BSM2_Q_REF                # Qr
BSM2_WASTAGE = 300.0                       # Qw
BSM2_PRIMARY_VOLUME = 900.0  # m³
BSM2_PRIMARY_FPS = 0.007
BSM2_CLARIFIER_AREA = 1500.0  # m²
BSM2_CLARIFIER_HEIGHT = 4.0   # m
BSM2_DIGESTER_VOLUME = 3400.0  # m³ liquid
BSM2_DIGESTER_T = 308.15       # K (35 °C)
BSM2_THICKENER_TSS_PERCENT = 7.0
BSM2_DEWATERING_TSS_PERCENT = 28.0
BSM2_SEPARATOR_REMOVAL = 98.0


def build_bsm2(
    asm1_network: Optional["object"] = None,
    adm1_network: Optional["object"] = None,
    *,
    Q_ref: float = BSM2_Q_REF,
    conditions: Optional[dict] = None,
) -> Plant:
    """Assemble the open-loop BSM2 plant.

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

    Returns
    -------
    Plant
        The wired BSM2 plant. The caller adds the influent and connects it to
        ``front_mix:fresh`` (mirroring :func:`build_bsm1`).
    """
    import aquakin

    asm1 = asm1_network if asm1_network is not None else aquakin.load_network("asm1")
    adm1 = adm1_network if adm1_network is not None else aquakin.load_network("adm1")
    if conditions is None:
        conditions = {name: asm1._condition_defaults[name]
                      for name in asm1.conditions_required}

    Qintr = 3.0 * Q_ref
    Qr = 1.0 * Q_ref
    Qw = BSM2_WASTAGE
    Q_settler_underflow = Qr + Qw

    plant = Plant("BSM2")
    seed = Stream(Q=jnp.asarray(0.0), C=asm1.default_concentrations(), network=asm1)

    # ----- Front: combine raw influent with the recycled reject water. -----
    plant.add_unit(MixerUnit(name="front_mix",
                             input_port_names=["fresh", "reject"], network=asm1))
    plant.add_unit(PrimaryClarifier(name="primary", network=asm1,
                                    volume=BSM2_PRIMARY_VOLUME, f_PS=BSM2_PRIMARY_FPS))

    # ----- Activated sludge: mixer + 5 CSTRs + internal recycle. -----
    plant.add_unit(MixerUnit(
        name="as_mix",
        input_port_names=["primary_eff", "internal_recycle", "ras"], network=asm1))
    for i in range(5):
        kla = {"SO": BSM2_KLA[i]} if BSM2_KLA[i] > 0 else {}
        sat = {"SO": BSM2_DO_SATURATION} if BSM2_KLA[i] > 0 else {}
        plant.add_unit(CSTRUnit(
            name=f"tank{i + 1}", network=asm1, volume=BSM2_TANK_VOLUMES[i],
            input_port_names=["inlet"], conditions=conditions, kla=kla, C_sat=sat))

    plant.add_unit(SplitterUnit(
        name="tank5_split", network=asm1,
        output_port_flows={"internal_recycle": Qintr}, remainder_port="to_settler"))

    # ----- Secondary clarifier (Takács) + RAS/wastage split. -----
    plant.add_unit(TakacsClarifier(
        name="settler", network=asm1, area=BSM2_CLARIFIER_AREA,
        height=BSM2_CLARIFIER_HEIGHT, underflow_Q=Q_settler_underflow,
        init_underflow_Q=Q_settler_underflow))
    plant.add_unit(SplitterUnit(
        name="underflow_split", network=asm1,
        output_port_flows={"ras": Qr}, remainder_port="waste"))

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

    # Cross-network interfaces (ASM1 <-> ADM1).
    asm2adm = ASM1toADM1(source_network=asm1, target_network=adm1)
    adm2asm = ADM1toASM1(source_network=adm1, target_network=asm1)

    # ----- Wiring -----
    # Front line.
    plant.connect("front_mix", "out", "primary", "inlet")
    plant.connect("primary", "effluent", "as_mix", "primary_eff")
    plant.connect("as_mix", "out", "tank1", "inlet")
    plant.connect("tank1", "out", "tank2", "inlet")
    plant.connect("tank2", "out", "tank3", "inlet")
    plant.connect("tank3", "out", "tank4", "inlet")
    plant.connect("tank4", "out", "tank5", "inlet")
    plant.connect("tank5", "out", "tank5_split", "in")
    plant.connect("tank5_split", "to_settler", "settler", "inlet")
    plant.connect("settler", "underflow", "underflow_split", "in")
    # AS recycles (back-edges; seeded).
    plant.connect("tank5_split", "internal_recycle", "as_mix", "internal_recycle",
                  initial_value=seed)
    plant.connect("underflow_split", "ras", "as_mix", "ras", initial_value=seed)
    # Sludge train.
    plant.connect("primary", "underflow", "sludge_mix", "primary_sludge")
    plant.connect("underflow_split", "waste", "thickener", "inlet")
    plant.connect("thickener", "underflow", "sludge_mix", "thickener_under")
    plant.connect("sludge_mix", "out", "digester", "inlet", translator=asm2adm)
    plant.connect("digester", "effluent", "dewatering", "inlet", translator=adm2asm)
    # Reject-water recycle to the front (back-edge; seeded).
    plant.connect("thickener", "overflow", "reject_mix", "thickener_reject")
    plant.connect("dewatering", "overflow", "reject_mix", "dewatering_reject")
    plant.connect("reject_mix", "out", "front_mix", "reject", initial_value=seed)
    # dewatering:underflow -> sludge disposal (leaves the plant; not routed).

    return plant
