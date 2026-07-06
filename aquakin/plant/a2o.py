"""A²O biological-nutrient-removal plant builder (ASM2d).

The **A²O** (Anaerobic–Anoxic–Oxic) process is the canonical biological
nutrient-removal flowsheet: an anaerobic selector at the front (where
phosphorus-accumulating organisms release phosphate and store fermentation
products), an anoxic zone (denitrification), and an aerated zone (nitrification
and luxury phosphate uptake), followed by a secondary clarifier. Two recycles
close the loop:

- **Internal (mixed-liquor / nitrate) recycle** ``Q_a`` from the end of the
  aerobic zone back to the start of the anoxic zone — returns nitrate to be
  denitrified.
- **Return activated sludge** ``Q_r`` from the clarifier underflow back to the
  anaerobic zone, with the remainder wasted as ``Q_w``.

It runs the shipped ``asm2d`` model, which already carries the biological
phosphorus model (PAO / poly-P / PHA) and ASM2d's own simple metal-phosphate
precipitation (``XMeOH`` / ``XMeP``), so the plant simultaneously removes carbon,
nitrogen and phosphorus. It is the home for the chemical-phosphorus (metal-salt
dosing) demonstration that the BSM plants — built on the phosphorus-free ASM1 —
cannot host.

Unlike BSM1/BSM2 this is **not** a standardised benchmark; the default sizing is
a defensible municipal A²O design, not a published reference set, so use it as a
worked nutrient-removal flowsheet rather than a validation target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from aquakin.plant._builder_support import (
    add_secondary_clarifier,
    reactor_conditions,
    recycle_pump_flows,
    register_recycle_streams,
)
from aquakin.plant.cstr import Aeration, CSTRUnit
from aquakin.plant.dosing import DosingUnit, Reagent
from aquakin.plant.mixer import MixerUnit, SetpointSplitter
from aquakin.plant.plant import Plant

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel
    from aquakin.plant.influent import InfluentSeries


# ASM2d particulate species — settle in the clarifier (everything that is not a
# soluble S* component). XTSS is the lumped solids tracer; XMeOH / XMeP are the
# metal-hydroxide / metal-phosphate precipitate solids.
ASM2D_PARTICULATES = (
    "XI",
    "XS",
    "XH",
    "XPAO",
    "XPP",
    "XPHA",
    "XAUT",
    "XTSS",
    "XMeOH",
    "XMeP",
)

# Default A²O design (a defensible municipal layout, not a published benchmark).
A2O_Q_AVG = 18446.0  # m³/d, dry-weather average (BSM scale, for a familiar size)
A2O_ANAEROBIC_VOLUMES = (750.0, 750.0)  # m³ — anaerobic selector
A2O_ANOXIC_VOLUMES = (750.0, 750.0)  # m³
A2O_AEROBIC_VOLUMES = (1333.0, 1333.0, 1333.0)  # m³
A2O_AEROBIC_KLA = (360.0, 360.0, 240.0)  # d⁻¹ (open-loop; high enough that
#   nitrifiers and PAOs coexist)
A2O_DO_SATURATION = 8.0  # g O₂ / m³
A2O_INTERNAL_RECYCLE_RATIO = 1.0  # Q_a / Q_in (aerobic → anoxic)
A2O_RAS_RATIO = 0.75  # Q_r / Q_in (underflow → anaerobic)
A2O_WASTAGE_FLOW = 120.0  # m³/d (a long SRT, to retain PAOs)
A2O_CLARIFIER_AREA = 1500.0  # m²
A2O_CLARIFIER_HEIGHT = 4.0  # m

# A typical municipal ASM2d raw-influent composition (g/m³ except SALK in
# mol HCO₃⁻/m³). The fermentation product SA (volatile fatty acids) is what
# drives biological P removal in the anaerobic selector, so it is non-trivial.
A2O_INFLUENT = {
    "SO2": 0.0,
    "SF": 40.0,  # fermentable COD
    "SA": 50.0,  # volatile fatty acids (the bio-P electron / carbon donor)
    "SNH4": 25.0,  # ammonia N
    "SNO3": 0.0,
    "SPO4": 8.0,  # soluble inorganic P
    "SI": 30.0,  # soluble inert COD
    "SALK": 7.0,  # alkalinity
    "SN2": 0.0,
    "XI": 30.0,  # particulate inert COD
    "XS": 180.0,  # slowly biodegradable COD
    "XH": 40.0,  # heterotroph seed
}


@dataclass(frozen=True)
class FerricDose:
    """Ferric (or alum) metal-salt dose for simultaneous chemical-P removal.

    The metal is dosed as metal hydroxide ``XMeOH`` into the aerobic zone, where
    the ASM2d precipitation reaction (``kPRE·[SPO4]·[XMeOH]``) precipitates
    phosphate as metal phosphate ``XMeP`` -- the chemical-P polish that runs
    alongside biological P removal. Dosing more metal drives the effluent
    phosphate lower (until metal is in excess).

    Parameters
    ----------
    flow : float
        Reagent dose flow (m³/d).
    xmeoh_conc : float
        Metal-hydroxide concentration of the neat reagent (g/m³ as ``XMeOH``);
        the metal mass dosed is ``flow · xmeoh_conc``. Default a concentrated
        stock so a small flow delivers the dose.
    """

    flow: float
    xmeoh_conc: float = 1.0e5


def a2o_influent(
    model: "CompiledModel",
    *,
    Q: float = A2O_Q_AVG,
    T: float = 293.15,
    overrides: Optional[dict[str, float]] = None,
) -> "InfluentSeries":
    """A constant municipal ASM2d influent for the A²O plant.

    Parameters
    ----------
    model : CompiledModel
        The ASM2d model (must match the one passed to :func:`build_a2o`).
    Q : float
        Influent flow (m³/d). Default the dry-weather average ``A2O_Q_AVG``.
    T : float
        Influent temperature (K). Default 20 °C.
    overrides : dict, optional
        Per-species concentration overrides on the default
        :data:`A2O_INFLUENT` composition (e.g. a higher P load).

    Returns
    -------
    InfluentSeries
        A constant-in-time, zero-based influent (unlisted species are 0).
    """
    comp = dict(A2O_INFLUENT)
    if overrides:
        comp.update(overrides)
    return model.influent(comp, Q=Q, T=T)


# A healthy bio-P mixed-liquor seed (g/m³): an established EBPR sludge with a
# large PAO population carrying stored poly-P. Seeding from a healthy sludge (as
# a real plant is) avoids the slow, seed-sensitive cold-start establishment of the
# PAO population; the solve relaxes it to the steady state, so the exact values
# affect settling time, not the operating point.
A2O_WARM_REACTOR_COMPOSITION = {
    "SO2": 2.0,
    "SF": 2.0,
    "SA": 2.0,
    "SNH4": 8.0,
    "SNO3": 2.0,
    "SPO4": 4.0,
    "SI": 30.0,
    "SALK": 7.0,
    "SN2": 5.0,
    "XI": 1500.0,
    "XS": 80.0,
    "XH": 1500.0,
    "XPAO": 900.0,
    "XPP": 250.0,
    "XPHA": 20.0,
    "XAUT": 150.0,
    "XTSS": 5500.0,
    "XMeOH": 0.0,
    "XMeP": 0.0,
}


def a2o_warm_start(plant: Plant) -> "jnp.ndarray":
    """A warm-start state vector for an :func:`build_a2o` plant.

    Seeds every activated-sludge reactor (the anaerobic/anoxic/aerobic CSTRs)
    with a healthy EBPR mixed liquor (:data:`A2O_WARM_REACTOR_COMPOSITION`) and
    every other unit at its default, so a ``plant.solve(y0=a2o_warm_start(plant))``
    starts from an established bio-P population rather than cold.

    Parameters
    ----------
    plant : Plant
        A plant built by :func:`build_a2o`.

    Returns
    -------
    jnp.ndarray
        Flat initial-state vector for ``plant.solve(y0=...)``.
    """
    reactors = [
        n
        for n in plant.list_units()
        if n.startswith(("anaer", "anox", "aer")) and "mix" not in n and "split" not in n
    ]
    if not reactors:  # pragma: no cover - a build_a2o plant always has reactors
        raise ValueError("no activated-sludge reactor found in the plant")
    model = plant.units[reactors[0]].model
    ml = model.concentrations(A2O_WARM_REACTOR_COMPOSITION, base="zero")
    return plant.initial_state(overrides=dict.fromkeys(reactors, ml))


def build_a2o(
    model: Optional["CompiledModel"] = None,
    *,
    Q_avg: float = A2O_Q_AVG,
    wastage_flow: float = A2O_WASTAGE_FLOW,
    internal_recycle_ratio: float = A2O_INTERNAL_RECYCLE_RATIO,
    ras_ratio: float = A2O_RAS_RATIO,
    ferric: Optional[FerricDose] = None,
    conditions: Optional[dict[str, float]] = None,
    use_takacs: bool = False,
) -> Plant:
    """Assemble an A²O biological-nutrient-removal plant on ASM2d.

    Flowsheet (7 reactors): an anaerobic selector (2 tanks) → anoxic zone
    (2 tanks) → aerobic zone (3 tanks) → secondary clarifier, with the
    mixed-liquor internal recycle (aerobic → anoxic) and RAS (underflow →
    anaerobic) closing the loop.

    Parameters
    ----------
    model : CompiledModel, optional
        ASM2d model. Defaults to ``aquakin.load_model("asm2d")``.
    Q_avg : float
        Design inlet flow used to size the recycle pumps (m³/d).
    wastage_flow : float
        Sludge-wastage pump flow ``Q_w`` (m³/d). The clarifier underflow is
        ``Q_r + Q_w`` and the wastage is the free remainder after the RAS split.
    internal_recycle_ratio, ras_ratio : float
        ``Q_a / Q_avg`` (aerobic → anoxic nitrate recycle) and ``Q_r / Q_avg``
        (underflow → anaerobic return sludge).
    conditions : dict, optional
        Per-tank condition overrides (e.g. ``{"T": 288.15}`` for winter).
        Defaults to the model's declared defaults.
    use_takacs : bool
        Use the layered Takács secondary clarifier instead of the fast
        stateless ``IdealClarifier`` (the default).

    Returns
    -------
    Plant
        Fully wired A²O plant. Add the influent with
        ``plant.add_influent("feed", a2o_influent(model))`` and read the
        effluent off ``plant.effluent_endpoint``.

    Notes
    -----
    Not a standardised benchmark — the sizing is a representative municipal A²O
    design. Use :func:`a2o_influent` for a matching constant influent.
    """
    if model is None:
        import aquakin

        model = aquakin.load_model("asm2d")

    if conditions is None:
        conditions = reactor_conditions(model)

    plant = Plant("A2O")

    # Controlled recycle-pump flows (constant volumetric setpoints off the design
    # flow, like the BSM plants -- see recycle_pump_flows / the SetpointSplitter
    # docstring).
    Qa, Qr, Qw, Q_underflow = recycle_pump_flows(
        internal_ratio=internal_recycle_ratio,
        ras_ratio=ras_ratio,
        Q_design=Q_avg,
        wastage=wastage_flow,
    )

    # ----- Front mixer: fresh influent + RAS -> anaerobic selector -----
    plant.add_unit(
        MixerUnit(
            name="front_mix",
            input_port_names=["fresh", "ras"],
            model=model,
        )
    )

    # ----- Anaerobic selector (no aeration) -----
    for i, vol in enumerate(A2O_ANAEROBIC_VOLUMES):
        plant.add_unit(
            CSTRUnit(
                name=f"anaer{i + 1}",
                model=model,
                volume=vol,
                input_port_names=["inlet"],
                conditions=conditions,
                aeration=None,
            )
        )

    # ----- Anoxic mixer: anaerobic outlet + internal (nitrate) recycle -----
    plant.add_unit(
        MixerUnit(
            name="anoxic_mix",
            input_port_names=["upstream", "internal_recycle"],
            model=model,
        )
    )

    # ----- Anoxic zone (denitrification, no aeration) -----
    for i, vol in enumerate(A2O_ANOXIC_VOLUMES):
        plant.add_unit(
            CSTRUnit(
                name=f"anox{i + 1}",
                model=model,
                volume=vol,
                input_port_names=["inlet"],
                conditions=conditions,
                aeration=None,
            )
        )

    # ----- Aerobic zone (nitrification + luxury P uptake) -----
    for i, vol in enumerate(A2O_AEROBIC_VOLUMES):
        plant.add_unit(
            CSTRUnit(
                name=f"aer{i + 1}",
                model=model,
                volume=vol,
                input_port_names=["inlet"],
                conditions=conditions,
                aeration=Aeration(kla=A2O_AEROBIC_KLA[i], do_sat=A2O_DO_SATURATION, species="SO2"),
            )
        )

    n_aer = len(A2O_AEROBIC_VOLUMES)

    # ----- Aerobic outlet splitter: internal recycle Qa + clarifier feed -----
    plant.add_unit(
        SetpointSplitter(
            name="aer_split",
            model=model,
            output_port_flows={"internal_recycle": Qa},
            remainder_port="to_clarifier",
        )
    )

    # ----- Secondary clarifier (Takács 1-D settler, or the fast IdealClarifier) -----
    add_secondary_clarifier(
        plant,
        model=model,
        underflow_Q=Q_underflow,
        use_takacs=use_takacs,
        takacs_kwargs=dict(
            area=A2O_CLARIFIER_AREA,
            height=A2O_CLARIFIER_HEIGHT,
            init_underflow_Q=Q_underflow,
            particulate_species=list(ASM2D_PARTICULATES),
        ),
        ideal_kwargs=dict(particulate_species=list(ASM2D_PARTICULATES)),
    )

    # ----- Underflow splitter: RAS Qr + wastage Qw -----
    plant.add_unit(
        SetpointSplitter(
            name="underflow_split",
            model=model,
            output_port_flows={"ras": Qr},
            remainder_port="waste",
        )
    )

    # ----- Feed-forward edges -----
    plant.connect("front_mix", "anaer1")
    for i in range(1, len(A2O_ANAEROBIC_VOLUMES)):
        plant.connect(f"anaer{i}", f"anaer{i + 1}")
    plant.connect(f"anaer{len(A2O_ANAEROBIC_VOLUMES)}", "anoxic_mix.upstream")
    plant.connect("anoxic_mix", "anox1")
    for i in range(1, len(A2O_ANOXIC_VOLUMES)):
        plant.connect(f"anox{i}", f"anox{i + 1}")
    last_anox = f"anox{len(A2O_ANOXIC_VOLUMES)}"
    if ferric is not None:
        # Insert a metal-salt dosing unit on the line into the aerobic zone, so
        # the dosed metal hydroxide precipitates phosphate (chemical-P) in the
        # aerated reactors alongside the biological uptake.
        reagent = Reagent.from_species(model, {"XMeOH": ferric.xmeoh_conc}, label="ferric")
        plant.add_unit(DosingUnit("ferric_dose", reagent, flow=ferric.flow))
        plant.connect(last_anox, "ferric_dose.in")
        plant.connect("ferric_dose.out", "aer1")
    else:
        plant.connect(last_anox, "aer1")
    for i in range(1, n_aer):
        plant.connect(f"aer{i}", f"aer{i + 1}")
    plant.connect(f"aer{n_aer}", "aer_split")
    plant.connect("aer_split.to_clarifier", "clarifier")
    plant.connect("clarifier.underflow", "underflow_split")

    # ----- Recycles (auto-seeded zero-flow back-edges) -----
    plant.connect("aer_split.internal_recycle", "anoxic_mix.internal_recycle")
    plant.connect("underflow_split.ras", "front_mix.ras")

    # Canonical entry / exit endpoints.
    plant.influent_endpoint = "front_mix.fresh"
    plant.effluent_endpoint = "clarifier.overflow"

    # Semantic stream shortcuts.
    register_recycle_streams(
        plant,
        internal_recycle="aer_split.internal_recycle",
        ras="underflow_split.ras",
        wastage="underflow_split.waste",
    )

    return plant
