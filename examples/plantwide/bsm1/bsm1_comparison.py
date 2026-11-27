"""The plant-wide core (``plantwide_core``) and ASM2d on the same BSM1 plant.

The core's aerobic and anoxic processes are ASM2d's; its own choices are
fermentation as heterotroph growth, slower heterotroph decay without oxygen or
nitrate, and a lower anaerobic hydrolysis factor (``etafe``). Both models run
on ``build_bsm1`` (open loop, BSM1's kLa, ideal clarifier) with BSM1's
constant dry-weather influent, carried into each model's species:

- readily biodegradable substrate ``SS`` as fermentable ``SF``, inert products
  ``XP`` with the particulate inerts ``XI``;
- total nitrogen kept: ammonium is what the influent's total N leaves after the
  model's own organic N contents;
- alkalinity as inorganic carbon for the core (12 g C per mol); ASM2d gets
  alkalinity and phosphate in excess (its growth is limited by both, the core's
  by neither).

Two comparisons:

- *aligned*: the core's own anaerobic heterotroph processes switched off
  (``q_FE`` 0, ``eta_b_AN`` 1), ASM2d's fermentation switched off and its
  ``etafe`` set to the core's. The two then share every process, and their
  steady states should agree;
- *as shipped*: each model with its defaults, to measure what the core's
  choices change on a nitrifying-denitrifying plant.

Run as a script to print both (about 2 minutes).
"""

from __future__ import annotations

from pathlib import Path

import aquakin
from aquakin.plant.bsm import build_bsm1
from aquakin.plant.bsm.bsm1 import BSM1_DO_SATURATION, BSM1_KLA, BSM1_Q_AVG
from aquakin.plant.bsm.warmstart import BSM1_WARM_REACTOR_COMPOSITION
from aquakin.plant.cstr import Aeration

CORE = Path(__file__).resolve().parents[1] / "models" / "plantwide_core.yaml"

# BSM1's constant (flow-weighted dry-weather) influent, ASM1 species, g/m3
# (alkalinity mol/m3): Copp (2002), the COST benchmark, Table 5.1.
BSM1_INFLUENT = {
    "SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17, "XB_A": 0.0,
    "XP": 0.0, "SO": 0.0, "SNO": 0.0, "SNH": 31.56, "SND": 6.95, "XND": 10.59,
    "SALK": 7.0,
}  # fmt: skip
# ASM1's organic N contents (g N/g COD): biomass 0.086 (i_XB), inert products
# 0.06 (i_XP); the remaining organic N is the influent's SND and XND.
ASM1_I_XB, ASM1_I_XP = 0.086, 0.06
SALK_TO_SIC = 12.0  # g C per mol of alkalinity (bicarbonate)
EXCESS = 1.0e6  # ASM2d's phosphate and alkalinity, far above their half-saturations
# Phosphate for the plant-wide levels that carry it: BSM1's influent has none,
# and 10 g P/m3 keeps growth unlimited by it (half-saturation 0.01 g P/m3).
PLANTWIDE_PO4 = 10.0
# AOB share of ASM1's nitrifiers, by the two groups' yields in the nitrite
# level (Hiatt & Grady 2008: 0.18 and 0.06 g COD/g N).
AOB_SHARE = 0.18 / (0.18 + 0.06)


def asm1_total_n(values):
    """Total nitrogen (g N/m3) of an ASM1 composition."""
    return (
        values["SNH"] + values["SNO"] + values["SND"] + values["XND"]
        + ASM1_I_XB * (values["XB_H"] + values["XB_A"]) + ASM1_I_XP * values["XP"]
    )  # fmt: skip


def carry_over(model, values):
    """An ASM1 composition in ``model``'s species (a plant-wide level or
    ``asm2d``), its total N kept by putting the difference in ammonium.

    A level with two-step nitrification gets ASM1's nitrifiers as AOB and NOB
    in the ratio of their yields; a level with phosphate but no alkalinity
    state (from the bio-P level on) gets phosphate at a non-limiting
    ``PLANTWIDE_PO4``."""
    out = {
        "SF": values["SS"], "SI": values["SI"], "XS": values["XS"], "XH": values["XB_H"],
        "XI": values["XI"] + values["XP"], "SO2": values["SO"], "SNO3": values["SNO"],
    }  # fmt: skip
    if "XAUT" in model.species_index:
        out["XAUT"] = values["XB_A"]
    else:
        out["XAOB"] = AOB_SHARE * values["XB_A"]
        out["XNOB"] = (1.0 - AOB_SHARE) * values["XB_A"]
    if "SIC" in model.species_index:
        out["SIC"] = values["SALK"] * SALK_TO_SIC
        if "SPO4" in model.species_index:
            out["SPO4"] = PLANTWIDE_PO4
    else:
        out["SALK"] = EXCESS
        out["SPO4"] = EXCESS
    comp = model.composition()
    oxidized = ("SNO3", "SNO2")
    organic_n = sum(comp.get(s, {}).get("N", 0.0) * v for s, v in out.items() if s not in oxidized)
    out["SNH4"] = asm1_total_n(values) - organic_n - values["SNO"]
    return out


def aligned_parameters(plant, model, core_etafe):
    """The plant's parameters with the processes the two models do not share
    switched off (see the module docstring)."""
    p = plant.default_parameters()
    if model.name == "plantwide_core":
        settings = {"q_FE": 1e-9, "eta_b_AN": 1.0}
    else:
        settings = {"qfe": 0.0, "etafe": core_etafe}
    for name, value in settings.items():
        p = p.at[plant.parameter_index(f"{model.name}.{name}")].set(value)
    return p


def bsm1_plant(model):
    """BSM1 on ``model``, aerated in tanks 3 to 5 at BSM1's kLa, its influent added."""
    # A model that declares dissolved gases has them stripped by the air
    # (the core's hydrogen, methane, nitrogen and CO2); ASM2d declares none.
    stripped = "model" if model.gases is not None else None
    aeration = {
        f"tank{i + 1}": Aeration(
            kla=kla, do_sat=BSM1_DO_SATURATION, species="SO2", stripped=stripped
        )
        for i, kla in enumerate(BSM1_KLA)
        if kla > 0
    }
    plant = build_bsm1(model, aeration=aeration)
    plant.add_influent(
        "feed",
        model.influent(carry_over(model, BSM1_INFLUENT), Q=BSM1_Q_AVG),
        to=plant.influent_endpoint,
    )
    warm = model.concentrations(carry_over(model, BSM1_WARM_REACTOR_COMPOSITION), base="zero")
    y0 = plant.initial_state(overrides=dict.fromkeys(plant.activated_sludge_reactors(), warm))
    return plant, y0


def run(model, params=None):
    plant, y0 = bsm1_plant(model)
    params = plant.default_parameters() if params is None else params
    steady = plant.steady_state(params=params, y0=y0, tol=1e-9, max_iter=2000)
    return dict(plant=plant, model=model, steady=steady, params=params)


def summary(result):
    """Tank 5 and the effluent at the steady state (the ideal clarifier passes
    the solubles through, so tank 5's solubles are the effluent's)."""
    plant, model = result["plant"], result["model"]
    tank5 = plant.states_by_unit(result["steady"].state)["tank5"]

    def at(species):
        return float(tank5[model.species_index[species]]) if species in model.species_index else 0.0

    return dict(
        SNH4=at("SNH4"),
        SNO3=at("SNO3"),
        SF=at("SF"),
        XH=at("XH"),
        XAUT=at("XAUT"),
        XS=at("XS"),
        XI=at("XI"),
        SO2=at("SO2"),
        SPRO=at("SPRO"),
        SH2=at("SH2"),
    )


if __name__ == "__main__":
    core = aquakin.load_model_from_file(CORE)
    asm2d = aquakin.load_model("asm2d")
    core_etafe = float(core.default_parameters()[core.param_index["etafe"]])
    rows = {}
    for label, aligned in (("aligned", True), ("as shipped", False)):
        for model in (core, asm2d):
            plant, _ = bsm1_plant(model)
            params = aligned_parameters(plant, model, core_etafe) if aligned else None
            rows[f"{model.name}, {label}"] = summary(run(model, params))
    keys = list(next(iter(rows.values())))
    print(f"{'':28s}" + "".join(f"{k:>9s}" for k in keys))
    for label, r in rows.items():
        print(f"{label:28s}" + "".join(f"{r[k]:9.3f}" for k in keys))
