"""Write ``plantwide_nitrite.yaml``: the nitrite level of the plant-wide model.

It extends ``plantwide_bio_p`` with nitrite as an intermediate: two-step
nitrification by ammonia- and nitrite-oxidizing bacteria (replacing the
core's one-step nitrifiers), anammox, and two-step denitrification by every
group that reduces nitrate. The design and the owner's decisions (17-20) are in
``.claude/notes/plantwide-model-design.md``, *The nitrite level*. Run from
this directory: ``python _make_plantwide_nitrite.py``.

The level is generated because the two-step reduction rewrites every inherited
process that consumes nitrate (the heterotrophs' and the PAOs' and GAOs'
anoxic processes): each becomes a pair, one reducing nitrate to nitrite and one
nitrite to N2, with the same rate law on its own switch and the same electrons,
so the pair does what the one process did, through nitrite. The inherited
processes are read from the core and bio-P files, merged as ``extends:``
merges them.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from aquakin.schema.inheritance import merge_spec, pop_inheritance_keys

HERE = Path(__file__).resolve().parent
CORE = HERE / "plantwide_core.yaml"
BIO_P = HERE / "plantwide_bio_p.yaml"
OUT = HERE / "plantwide_nitrite.yaml"

# Electrons accepted per g N, as g COD: NO3 -> N2 (5 e), NO3 -> NO2 (2 e),
# NO2 -> N2 (3 e). A process's nitrate coefficient c (to N2) carries
# c * 40/14 g COD of electrons, so the same electrons reduce c * 5/2 g N of
# nitrate to nitrite, or c * 5/3 g N of nitrite to N2.
TO_NO2 = (40.0 / 14.0) / (16.0 / 14.0)
FROM_NO2 = (40.0 / 14.0) / (24.0 / 14.0)
# COD content of nitrite-N relative to ammonium, as for nitrate (-64/14): the
# oxygen its 6 electrons (ammonium to nitrite) took, negative.
NO2_COD = -48.0 / 14.0

ANOXIC = re.compile(r"\banoxic\b")
# The balances the bio-P level closes in every reaction (its model-level list).
CONSERVED = ["COD", "N", "C", "P"]


def num(x: float) -> float:
    return float(f"{x:.10g}")


def merged_bio_p() -> dict:
    """The bio-P level as ``extends:`` resolves it: the core with the bio-P
    file merged onto it."""
    core = yaml.safe_load(CORE.read_text())
    bio_p = yaml.safe_load(BIO_P.read_text())
    pop_inheritance_keys(bio_p, BIO_P.name)
    return merge_spec(core, bio_p, source=BIO_P.name)


def _scaled(coefficient, factor: float):
    """``factor`` times a stoichiometric coefficient, a number or an expression."""
    if isinstance(coefficient, (int, float)):
        return num(factor * coefficient)
    return f"{num(factor)} * ({coefficient})"


def nitrate_consumers(reactions: list[dict]) -> list[dict]:
    """The inherited processes that reduce nitrate under the anoxic switch."""
    return [
        r
        for r in reactions
        if ANOXIC.search(str(r.get("rate", "")))
        and r.get("stoichiometry", {}).get("SNO3") not in (None, "auto")
    ]


def two_step(reaction: dict) -> list[dict]:
    """The pair replacing a nitrate-consuming process: NO3 -> NO2, then NO2 -> N2."""
    stoich = dict(reaction["stoichiometry"])
    c = stoich.pop("SNO3")
    stoich.pop("SN2", None)
    rate = str(reaction["rate"])
    conserved = reaction.get("conserved_for", CONSERVED)
    first = dict(
        name=f"{reaction['name']}_NO3_NO2",
        rate=ANOXIC.sub("anoxic_NO3", rate),
        stoichiometry={**stoich, "SNO3": _scaled(c, TO_NO2), "SNO2": _scaled(c, -TO_NO2)},
        # The nitrate and nitrite coefficients carry the electrons, so COD
        # closes by construction; the rest close on their auto species.
        conserved_for=[q for q in conserved if q != "COD"],
    )
    second = dict(
        name=f"{reaction['name']}_NO2_N2",
        rate=ANOXIC.sub("anoxic_NO2", rate),
        stoichiometry={**stoich, "SNO2": _scaled(c, FROM_NO2), "SN2": "auto"},
        conserved_for=list(conserved),
    )
    return [first, second]


def species(base: dict) -> list[dict]:
    """Nitrite and the three new groups, which take the one-step nitrifiers'
    composition (the core's biomass, with the bio-P level's phosphorus)."""
    biomass = next(s for s in base["species"] if s["name"] == "XAUT")["composition"]

    def group(name, text):
        return dict(
            name=name,
            units="g_COD/m3",
            default_concentration=50.0,
            description=text,
            composition=dict(biomass),
        )
    return [
        dict(name="SNO3", description="Nitrate nitrogen"),
        dict(
            name="SNO2",
            units="g_N/m3",
            default_concentration=0.1,
            description="Nitrite nitrogen",
            composition={"COD": num(NO2_COD), "N": 1.0},
        ),
        group("XAOB", "Ammonia-oxidizing bacteria (NH4 to NO2)"),
        group("XNOB", "Nitrite-oxidizing bacteria (NO2 to NO3)"),
        {**group("XAMX", "Anaerobic ammonium-oxidizing bacteria (anammox)"), "default_concentration": 10.0},
    ]


def parameters() -> dict:
    p = {}

    def add(name, value, units, bounds=None, theta=None, transform="positive_log"):
        entry = dict(value=value, units=units)
        if bounds is not None:
            entry.update(bounds=list(bounds), transform=transform)
        if theta is not None:
            entry["temperature"] = dict(theta=theta, ref_T=293.15)
        p[name] = entry

    # Heterotroph nitrite half-saturation (decision 18).
    add("KNO2_H", 0.05, "g_N/m3", bounds=(0.005, 0.5))
    # Nitrifiers and anammox (decision 17), at 20 C.
    add("muAOB", 0.9, "1/d", bounds=(0.2, 3.0), theta=1.072)
    add("bAOB", 0.17, "1/d", bounds=(0.02, 1.0), theta=1.03)
    add("KNH4_AOB", 0.7, "g_N/m3", bounds=(0.05, 5.0))
    add("KO2_AOB", 0.25, "g_O2/m3", bounds=(0.05, 2.0))
    add("YAOB", 0.15, "g_COD/g_N", bounds=(0.05, 0.4))
    add("muNOB", 0.65, "1/d", bounds=(0.1, 2.0), theta=1.06)
    add("bNOB", 0.15, "1/d", bounds=(0.02, 1.0), theta=1.03)
    add("KNO2_NOB", 0.1, "g_N/m3", bounds=(0.01, 2.0))
    add("KO2_NOB", 0.25, "g_O2/m3", bounds=(0.05, 2.0))
    add("YNOB", 0.09, "g_COD/g_N", bounds=(0.02, 0.3))
    add("muAMX", 0.1, "1/d", bounds=(0.02, 0.5), theta=1.01)
    add("bAMX", 0.019, "1/d", bounds=(0.002, 0.1), theta=1.03)
    add("KNH4_AMX", 2.0, "g_N/m3", bounds=(0.05, 10.0))
    add("KNO2_AMX", 1.0, "g_N/m3", bounds=(0.05, 10.0))
    add("KI_O2_AMX", 0.4, "g_O2/m3", bounds=(0.01, 2.0))
    # Strous et al. (1998): mol NO2 consumed and NO3 made per mol NH4 oxidized.
    add("Y_AMX_NO2", 1.32, "-", bounds=(1.1, 1.6))
    add("Y_AMX_NO3", 0.26, "-", bounds=(0.1, 0.4))
    # Decay of the autotrophs with nitrite or nitrate, relative to aerobic (the
    # same model's 0.5); without either they use the core's eta_b_AN.
    add("eta_b_NOX_AUT", 0.5, "-", bounds=(0.05, 1.0), transform="logit")
    return p


def expressions() -> dict:
    return {
        # Nitrate is reduced while nitrite is scarce, nitrite whenever present;
        # aerobic + anoxic + anaerobic stays 1, and without nitrite every switch
        # is the core's.
        "anoxic_NO3": "monod_inh([SO2], KO2_H) * monod([SNO3], KNO3_H) * monod_inh([SNO2], KNO2_H)",
        "anoxic_NO2": "monod_inh([SO2], KO2_H) * monod([SNO2], KNO2_H)",
        "anoxic": "anoxic_NO3 + anoxic_NO2",
        "anaerobic": "monod_inh([SO2], KO2_H) * monod_inh([SNO3], KNO3_H) * monod_inh([SNO2], KNO2_H)",
        "anaerobic_only": "monod_inh([SO2], KI_O2_AN) * monod_inh([SNO3] + [SNO2], KI_NO3_AN)",
        "decay_switch_AUT": "aerobic + eta_b_NOX_AUT * anoxic + eta_b_AN * anaerobic",
    }


def autotroph_reactions() -> list[dict]:
    p_limit = "monod([SPO4], KP_H)"
    closures = {"SO2": "auto", "SNH4": "auto", "SIC": "auto", "SPO4": "auto"}
    # Anammox per unit biomass COD (the Strous stoichiometry with the biomass's
    # 0.07 g N/g COD): nitrite consumed and nitrate made; ammonium and N2 close
    # the N and COD balances.
    amx = "(3 * 0.07 * 16 + 2 * 14) / 16 / (5 * Y_AMX_NO3 + 3 - 3 * Y_AMX_NO2)"

    def decay(group, b):
        return dict(
            name=f"decay_of_{group}",
            rate=f"{b} * decay_switch_AUT * [{group}]",
            stoichiometry={group: -1.0, "XS": "1 - fXI", "XI": "fXI", "SNH4": "auto", "SIC": "auto", "SPO4": "auto"},
            conserved_for=["N", "C", "P"],
        )

    return [
        dict(
            name="growth_of_AOB",
            description="Nitritation: ammonium to nitrite.",
            rate=f"muAOB * monod([SO2], KO2_AOB) * monod([SNH4], KNH4_AOB) * {p_limit} * [XAOB]",
            stoichiometry={"XAOB": 1.0, "SNO2": "1 / YAOB", **closures},
        ),
        dict(
            name="growth_of_NOB",
            description="Nitratation: nitrite to nitrate.",
            rate=(
                "muNOB * monod([SO2], KO2_NOB) * monod([SNO2], KNO2_NOB) * monod([SNH4], KNH4_H) "
                f"* {p_limit} * [XNOB]"
            ),
            stoichiometry={"XNOB": 1.0, "SNO2": "-1 / YNOB", "SNO3": "1 / YNOB", **closures},
        ),
        dict(
            name="growth_of_anammox",
            description="Anaerobic ammonium oxidation with nitrite, part of the nitrite to nitrate.",
            rate=(
                "muAMX * monod([SNH4], KNH4_AMX) * monod([SNO2], KNO2_AMX) * monod_inh([SO2], KI_O2_AMX) "
                f"* {p_limit} * [XAMX]"
            ),
            stoichiometry={
                "XAMX": 1.0,
                "SNO2": f"-Y_AMX_NO2 * {amx}",
                "SNO3": f"Y_AMX_NO3 * {amx}",
                "SNH4": "auto",
                "SN2": "auto",
                "SIC": "auto",
                "SPO4": "auto",
            },
        ),
        decay("XAOB", "bAOB"),
        decay("XNOB", "bNOB"),
        decay("XAMX", "bAMX"),
    ]


def build() -> dict:
    base = merged_bio_p()
    consumers = nitrate_consumers(base["reactions"])
    reactions = autotroph_reactions()
    for r in consumers:
        reactions += two_step(r)
    one_step = ["growth_of_nitrifiers", "decay_of_nitrifiers"]
    return dict(
        model=dict(
            name="plantwide_nitrite",
            extends="plantwide_bio_p.yaml",
            version="0.1",
            description=(
                "Nitrite level of the plant-wide model: plantwide_bio_p with nitrite as an "
                "intermediate. Ammonia- and nitrite-oxidizing bacteria replace the one-step "
                "nitrifiers, anammox oxidizes ammonium with nitrite, and every group that reduces "
                "nitrate (heterotrophs, PAOs, both GAO groups) does it in two steps, nitrate to "
                "nitrite and nitrite to N2, the first slowed while nitrite is present. Nitrifier "
                "and anammox kinetics and the two-step form are those of the plant-wide model of "
                "Varga et al. (2018), with the anammox stoichiometry of Strous et al. (1998). "
                "Generated by _make_plantwide_nitrite.py."
            ),
            references=[
                "Varga, Hauduc, Barnard, Dunlap, Jimenez, Menniti, Schauer, Lopez Vazquez, Gu, Sharp & Takacs (2018). Recent advances in bio-P modelling - a new approach verified by full-scale observations. Water Sci. Technol. 78(10), 2119-2130. doi:10.2166/wst.2018.490",
                "Kaelin, Manser, Rieger, Eugster, Rottermann & Siegrist (2009). Extension of ASM3 for two-step nitrification and denitrification and its calibration and validation with batch tests and pilot scale data. Water Res. 43(6), 1680-1692.",
                "Strous, Heijnen, Kuenen & Jetten (1998). The sequencing batch reactor as a powerful tool for the study of slowly growing anaerobic ammonium-oxidizing microorganisms. Appl. Microbiol. Biotechnol. 50(5), 589-596.",
            ],
        ),
        species=species(base),
        parameters=parameters(),
        expressions=expressions(),
        reactions=reactions,
        remove=dict(
            species=["XAUT"],
            reactions=one_step + [r["name"] for r in consumers],
            parameters=["muAUT", "bAUT", "KO2_AUT", "KNH4_AUT", "YAUT"],
        ),
    )


def main() -> None:
    header = "# Auto-generated by _make_plantwide_nitrite.py -- do not edit by hand.\n"
    OUT.write_text(header + yaml.safe_dump(build(), sort_keys=False, width=100, allow_unicode=True))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
