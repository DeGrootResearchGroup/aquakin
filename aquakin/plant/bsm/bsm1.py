"""BSM1 plant builder.

The IWA BSM1 plant: 5 CSTRs (2 anoxic + 3 aerobic) followed by a Takács
secondary clarifier, with two recycle streams:

- **Internal recycle** ``Q_a`` from tank 5 to tank 1 (mixed liquor return).
- **External recycle** ``Q_r`` from the clarifier underflow to tank 1
  (return activated sludge), with the rest being wastage ``Q_w``.

Reference design parameters (Copp 2002 / Alex 2008 Table 1.1):

- Tank volumes: V₁ = V₂ = 1000 m³ (anoxic); V₃ = V₄ = V₅ = 1333 m³ (aerobic).
- Aerated tanks: kLa₃ = kLa₄ = 240 d⁻¹; kLa₅ = 84 d⁻¹.
- Oxygen saturation: S_O,sat = 8 g/m³.
- Recycle ratios at average dry weather: Q_a / Q_in = 3; Q_r / Q_in = 1;
  Q_w = 385 m³/d.
- Clarifier: A = 1500 m², H = 4 m, 10 layers, feed at layer 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aquakin.plant._builder_support import (
    add_secondary_clarifier,
    reactor_conditions,
    recycle_pump_flows,
    register_recycle_streams,
)
from aquakin.plant.cstr import Aeration, CSTRUnit
from aquakin.plant.mixer import MixerUnit, SetpointSplitter
from aquakin.plant.plant import Plant

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


# Reference design values (Copp 2002 Table 1.1 / Alex 2008 Table 1.7).
BSM1_TANK_VOLUMES = (1000.0, 1000.0, 1333.0, 1333.0, 1333.0)  # m³
BSM1_TANK_AEROBIC = (False, False, True, True, True)
BSM1_KLA = (0.0, 0.0, 240.0, 240.0, 84.0)  # d⁻¹ (open-loop reference)
BSM1_DO_SATURATION = 8.0  # g O2 / m³
BSM1_Q_AVG = 18446.0  # m³/d, dry-weather average
BSM1_INTERNAL_RECYCLE_RATIO = 3.0  # Q_a / Q_in
BSM1_EXTERNAL_RECYCLE_RATIO = 1.0  # Q_r / Q_in
BSM1_WASTAGE_FLOW = 385.0  # m³/d
BSM1_CLARIFIER_AREA = 1500.0  # m²
BSM1_CLARIFIER_HEIGHT = 4.0  # m


def build_bsm1(
    model: CompiledModel | None = None,
    *,
    Q_avg: float = BSM1_Q_AVG,
    wastage_flow: float = BSM1_WASTAGE_FLOW,
    closed_loop_do: bool = False,
    do_setpoint_tank5: float = 2.0,
    conditions: dict[str, float] | None = None,
    use_takacs: bool = False,
    settler_soluble_holdup: bool = True,
    settler_composition_mode: str = "lumped_tss",
) -> Plant:
    """Assemble the canonical BSM1 plant.

    Parameters
    ----------
    model : CompiledModel, optional
        ASM1 model. Defaults to ``aquakin.load_model("asm1")``.
    Q_avg : float
        Average dry-weather inlet flow. Used to size the recycle streams
        (internal recycle and RAS). Default 18446 m³/d per Copp 2002.
    wastage_flow : float
        Sludge-wastage pump flow ``Qw`` (m³/d). The clarifier underflow is
        ``Qr + Qw`` and the wastage is the free remainder after the RAS split.
        Default 385 per Copp 2002; vary it to hit a target solids retention time
        (see ``examples/bsm1_target_srt.py``).
    closed_loop_do : bool
        If True, attach a PI controller on tank 5 DO that adjusts kLa₅
        to maintain ``do_setpoint_tank5``. Open-loop default uses the
        constant kLa values from Alex 2008 Table 1.7.
    do_setpoint_tank5 : float
        DO setpoint when ``closed_loop_do=True``. Ignored otherwise.
    conditions : dict[str, float], optional
        Override the per-tank conditions vector (e.g. set ``T=288.15`` for
        winter). Defaults to model's declared defaults.
    use_takacs : bool
        If True, use the full Takács 1-D layered secondary clarifier (the BSM1
        reference settler, with its own per-layer solids state). If False
        (default), use the fast stateless ``IdealClarifier``. The Takács settler
        is stiffer, so its ``solve()`` needs a larger ``max_steps`` (the
        clarifier alone is cheap, but the full plant with recycles benefits from
        ``max_steps`` of a few hundred thousand).
    settler_soluble_holdup : bool
        When ``use_takacs=True``, carry the soluble species through the ten
        settler layers as well-mixed convective states, so the clarifier liquid
        volume damps the dynamic soluble effluent signal. Default True, matching
        the BSM1 reference settler ``settler1dv4`` (MODELTYPE=0, the COST
        benchmark). Set False for the simpler soluble pass-through. Ignored when
        ``use_takacs=False``. Leaves the steady state unchanged.
    settler_composition_mode : {"lumped_tss", "per_species"}
        When ``use_takacs=True``, how the settler tracks the particulate phase.
        ``"lumped_tss"`` (default) carries one total-suspended-solids value per
        layer and scales the outlet particulate composition from the
        instantaneous feed, reproducing the BSM1 reference ``settler1dv4``.
        ``"per_species"`` carries each particulate per layer (per-layer
        composition memory), matching the BSM2 reference ``settler1dv5``; it
        agrees with ``"lumped_tss"`` at steady state but diverges under dynamic
        flow. Ignored when ``use_takacs=False``.

    Returns
    -------
    Plant
        Fully wired BSM1 plant ready to ``solve()``.

    Notes
    -----
    The internal recycle (tank 5 → tank 1) and RAS recycle (clarifier
    underflow → tank 1) are seeded with the inlet's default
    concentration at zero flow on the first RHS pass. Steady-state
    behaviour is independent of this seed.
    """
    if closed_loop_do:
        raise NotImplementedError(
            "Closed-loop DO control will be added in a follow-up; set closed_loop_do=False for now."
        )

    if model is None:
        import aquakin

        model = aquakin.load_model("asm1")

    if conditions is None:
        conditions = reactor_conditions(model)

    plant = Plant("BSM1")

    # ----- Influent mixer -----
    # Mixes the external influent + internal recycle + RAS into tank 1's inlet.
    plant.add_unit(
        MixerUnit(
            name="inlet_mix",
            input_port_names=["fresh", "internal_recycle", "ras"],
            model=model,
        )
    )

    # ----- 5 CSTR tanks -----
    for i in range(5):
        aeration = Aeration(kla=BSM1_KLA[i], do_sat=BSM1_DO_SATURATION) if BSM1_KLA[i] > 0 else None
        plant.add_unit(
            CSTRUnit(
                name=f"tank{i + 1}",
                model=model,
                volume=BSM1_TANK_VOLUMES[i],
                input_port_names=["inlet"],
                conditions=conditions,
                aeration=aeration,
            )
        )

    # Controlled recycle-pump flows (BSM convention: constant volumetric setpoints
    # off the design flow ``Q_avg``, not fractions of throughput -- see
    # recycle_pump_flows / the SetpointSplitter docstring).
    Qa, Qr, Qw, Q_underflow = recycle_pump_flows(
        internal_ratio=BSM1_INTERNAL_RECYCLE_RATIO,
        ras_ratio=BSM1_EXTERNAL_RECYCLE_RATIO,
        Q_design=Q_avg,
        wastage=wastage_flow,
    )

    # ----- Internal recycle splitter (tank 5 outlet) -----
    # Tank 5 outlet splits into the fixed internal-recycle pump flow Qa and the
    # remainder (the clarifier feed, Q_in + Qr).
    plant.add_unit(
        SetpointSplitter(
            name="tank5_split",
            model=model,
            output_port_flows={"internal_recycle": Qa},
            remainder_port="to_clarifier",
        )
    )

    # ----- Clarifier (Takács 1-D settler, or the fast IdealClarifier) -----
    add_secondary_clarifier(
        plant,
        model=model,
        underflow_Q=Q_underflow,
        use_takacs=use_takacs,
        takacs_kwargs=dict(
            area=BSM1_CLARIFIER_AREA,
            height=BSM1_CLARIFIER_HEIGHT,
            # Settled-blanket initialization: the design underflow flow sets the
            # thickening ratio so the clarifier starts settled rather than uniform,
            # avoiding the violent startup transient.
            init_underflow_Q=Q_underflow,
            # The BSM1 reference settler (settler1dv4, MODELTYPE=0, the COST
            # benchmark) carries the solubles through the ten layers, so the
            # clarifier liquid volume damps the dynamic soluble effluent signal. On
            # by default to reproduce the reference; set False for the simpler
            # soluble pass-through. Leaves the steady state unchanged.
            soluble_holdup=settler_soluble_holdup,
            # The reference settler1dv4 sets the outlet particulate composition from
            # the instantaneous feed scaled by the boundary-layer TSS ratio
            # ("lumped_tss", the default here). The "per_species" alternative carries
            # per-layer composition memory the reference lacks, which diverges under
            # dynamic flow (same steady state); it matches the BSM2 settler1dv5.
            composition_mode=settler_composition_mode,
        ),
    )

    # ----- Underflow splitter (RAS + wastage) -----
    # The clarifier underflow (Qr + Qw) splits into the fixed RAS pump flow Qr
    # back to tank 1 and the remainder (wastage Qw), which leaves the plant.
    plant.add_unit(
        SetpointSplitter(
            name="underflow_split",
            model=model,
            output_port_flows={"ras": Qr},
            remainder_port="waste",
        )
    )

    # ----- Wire feed-forward edges -----
    # Influent and recycles will be added by caller via add_influent;
    # the wiring below assumes:
    #   inlet_mix:fresh  <- influent named "feed" (added by caller)
    #   tank1:inlet      <- inlet_mix:out
    #   tank2:inlet      <- tank1:out
    #   ...
    #   tank5_split:in   <- tank5:out
    #   clarifier:inlet  <- tank5_split:to_clarifier
    #   underflow_split:in <- clarifier:underflow
    # Recycles seeded with zero-flow streams:
    #   inlet_mix:internal_recycle <- tank5_split:internal_recycle
    #   inlet_mix:ras              <- underflow_split:ras
    # Note: clarifier.overflow is the effluent (not routed onward).

    # Reactor cascade: bare endpoints use each unit's sole in/out port.
    plant.connect("inlet_mix", "tank1")
    plant.connect("tank1", "tank2")
    plant.connect("tank2", "tank3")
    plant.connect("tank3", "tank4")
    plant.connect("tank4", "tank5")
    plant.connect("tank5", "tank5_split")
    plant.connect("tank5_split.to_clarifier", "clarifier")
    plant.connect("clarifier.underflow", "underflow_split")
    # Recycles (source evaluated after destination): auto-seeded with a
    # zero-flow stream, so no initial_value is needed.
    plant.connect("tank5_split.internal_recycle", "inlet_mix.internal_recycle")
    plant.connect("underflow_split.ras", "inlet_mix.ras")

    # Canonical entry / exit endpoints, so callers wire the influent with
    # ``plant.add_influent("feed", series)`` and read the effluent off
    # ``plant.effluent_endpoint`` instead of hard-coding a port.
    plant.influent_endpoint = "inlet_mix.fresh"
    plant.effluent_endpoint = "clarifier.overflow"

    # Semantic stream shortcuts (plant.stream(sol, "effluent"), plant.list_streams())
    # so the engineer reads by role rather than the internal "unit.port".
    register_recycle_streams(
        plant,
        internal_recycle="tank5_split.internal_recycle",
        ras="underflow_split.ras",
        wastage="underflow_split.waste",
    )

    return plant
