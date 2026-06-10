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

from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp

from aquakin.plant.clarifier import IdealClarifier
from aquakin.plant.cstr import CSTRUnit
from aquakin.plant.mixer import MixerUnit, SplitterUnit
from aquakin.plant.plant import Plant
from aquakin.plant.streams import Stream
from aquakin.plant.takacs import TakacsClarifier

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


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
    network: Optional["CompiledNetwork"] = None,
    *,
    Q_avg: float = BSM1_Q_AVG,
    closed_loop_do: bool = False,
    do_setpoint_tank5: float = 2.0,
    conditions: Optional[dict[str, float]] = None,
    use_takacs: bool = False,
) -> Plant:
    """Assemble the canonical BSM1 plant.

    Parameters
    ----------
    network : CompiledNetwork, optional
        ASM1 network. Defaults to ``aquakin.load_network("asm1")``.
    Q_avg : float
        Average dry-weather inlet flow. Used to size the recycle streams
        (internal recycle and RAS). Default 18446 m³/d per Copp 2002.
    closed_loop_do : bool
        If True, attach a PI controller on tank 5 DO that adjusts kLa₅
        to maintain ``do_setpoint_tank5``. Open-loop default uses the
        constant kLa values from Alex 2008 Table 1.7.
    do_setpoint_tank5 : float
        DO setpoint when ``closed_loop_do=True``. Ignored otherwise.
    conditions : dict[str, float], optional
        Override the per-tank conditions vector (e.g. set ``T=288.15`` for
        winter). Defaults to network's declared defaults.
    use_takacs : bool
        If True, use the full Takács 1-D layered secondary clarifier (the BSM1
        reference settler, with its own per-layer solids state). If False
        (default), use the fast stateless ``IdealClarifier``. The Takács settler
        is stiffer, so its ``solve()`` needs a larger ``max_steps`` (the
        clarifier alone is cheap, but the full plant with recycles benefits from
        ``max_steps`` of a few hundred thousand).

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
            "Closed-loop DO control will be added in a follow-up; "
            "set closed_loop_do=False for now."
        )

    if network is None:
        import aquakin
        network = aquakin.load_network("asm1")

    if conditions is None:
        conditions = {name: network._condition_defaults[name]
                      for name in network.conditions_required}

    plant = Plant("BSM1")

    # Two seed streams (zero flow, default C). Used for recycle initialisation.
    seed_zero = Stream(
        Q=jnp.asarray(0.0),
        C=network.default_concentrations(),
        network=network,
    )

    # ----- Influent mixer -----
    # Mixes the external influent + internal recycle + RAS into tank 1's inlet.
    plant.add_unit(
        MixerUnit(
            name="inlet_mix",
            input_port_names=["fresh", "internal_recycle", "ras"],
            network=network,
        )
    )

    # ----- 5 CSTR tanks -----
    for i in range(5):
        tank_kla = {"SO": BSM1_KLA[i]} if BSM1_KLA[i] > 0 else {}
        tank_sat = {"SO": BSM1_DO_SATURATION} if BSM1_KLA[i] > 0 else {}
        upstream = "inlet_mix" if i == 0 else f"tank{i}"
        plant.add_unit(
            CSTRUnit(
                name=f"tank{i + 1}",
                network=network,
                volume=BSM1_TANK_VOLUMES[i],
                input_port_names=["inlet"],
                conditions=conditions,
                kla=tank_kla,
                C_sat=tank_sat,
            )
        )

    # Controlled recycle-pump flows (BSM convention: constant volumetric
    # setpoints, not fractions of throughput). ``Qa`` internal recycle, ``Qr``
    # RAS, ``Qw`` wastage -- all fixed regardless of influent, defined off the
    # design flow ``Q_avg``. The clarifier underflow is ``Qr + Qw`` and the
    # effluent is the free remainder (``Q_e = Q_f - Q_u``). Modelling these as
    # fixed *fractions* instead makes the recycle-flow loop gain near-singular
    # off the design influent, so the plant blows up under dynamic flow (see
    # the SplitterUnit flow-mode docstring).
    Qa = BSM1_INTERNAL_RECYCLE_RATIO * Q_avg
    Qr = BSM1_EXTERNAL_RECYCLE_RATIO * Q_avg
    Qw = BSM1_WASTAGE_FLOW
    Q_underflow = Qr + Qw

    # ----- Internal recycle splitter (tank 5 outlet) -----
    # Tank 5 outlet splits into the fixed internal-recycle pump flow Qa and the
    # remainder (the clarifier feed, Q_in + Qr).
    plant.add_unit(
        SplitterUnit(
            name="tank5_split",
            network=network,
            output_port_flows={"internal_recycle": Qa},
            remainder_port="to_clarifier",
        )
    )

    # ----- Clarifier -----
    # ``use_takacs`` selects the full Takács 1-D layered secondary clarifier
    # (the BSM1 reference model); the default ``IdealClarifier`` is a fast,
    # stateless ~99.8%-capture separator. Both expose the same overflow /
    # underflow ports, so the rest of the plant graph is identical.
    if use_takacs:
        plant.add_unit(
            TakacsClarifier(
                name="clarifier",
                network=network,
                area=BSM1_CLARIFIER_AREA,
                height=BSM1_CLARIFIER_HEIGHT,
                # Fixed underflow pump flow (Qr + Qw); the effluent overflow is
                # the remainder and tracks the feed.
                underflow_Q=Q_underflow,
                # Settled-blanket initialization: the design underflow flow sets
                # the thickening ratio so the clarifier starts settled rather
                # than uniform, avoiding the violent startup transient.
                init_underflow_Q=Q_underflow,
            )
        )
    else:
        plant.add_unit(
            IdealClarifier(
                name="clarifier",
                network=network,
                underflow_Q=Q_underflow,
                capture_efficiency=0.998,
            )
        )

    # ----- Underflow splitter (RAS + wastage) -----
    # The clarifier underflow (Qr + Qw) splits into the fixed RAS pump flow Qr
    # back to tank 1 and the remainder (wastage Qw), which leaves the plant.
    plant.add_unit(
        SplitterUnit(
            name="underflow_split",
            network=network,
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

    plant.connect("inlet_mix", "out", "tank1", "inlet")
    plant.connect("tank1", "out", "tank2", "inlet")
    plant.connect("tank2", "out", "tank3", "inlet")
    plant.connect("tank3", "out", "tank4", "inlet")
    plant.connect("tank4", "out", "tank5", "inlet")
    plant.connect("tank5", "out", "tank5_split", "in")
    plant.connect("tank5_split", "to_clarifier", "clarifier", "inlet")
    plant.connect("clarifier", "underflow", "underflow_split", "in")
    plant.connect("tank5_split", "internal_recycle", "inlet_mix", "internal_recycle",
                  initial_value=seed_zero)
    plant.connect("underflow_split", "ras", "inlet_mix", "ras",
                  initial_value=seed_zero)

    return plant
