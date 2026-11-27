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

import math
import re
from pathlib import Path

import numpy as np
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
        rate=ANOXIC.sub("f_NO3_NO2 * anoxic_NO3", rate),
        stoichiometry={**stoich, "SNO3": _scaled(c, TO_NO2), "SNO2": _scaled(c, -TO_NO2)},
        # The nitrate and nitrite coefficients carry the electrons, so COD
        # closes by construction; the rest close on their auto species.
        conserved_for=[q for q in conserved if q != "COD"],
    )
    second = dict(
        name=f"{reaction['name']}_NO2_N2",
        rate=ANOXIC.sub("f_NO2_N2 * anoxic_NO2", rate),
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


# Net growth (growth minus decay) of AOB and NOB on municipal pilot sludge,
# measured between 5 and 25 C: exp(k (T - 20)) (Kaelin et al. 2009).
KAELIN_NET_AOB, KAELIN_NET_NOB = 0.120, 0.078


def growth_theta(mu20: float, b20: float, b_theta: float, net_k: float) -> float:
    """The growth theta that makes net growth ``mu - b`` follow Kaelin et al.'s
    measured ``exp(net_k (T - 20))`` between 10 and 25 C (least squares in
    log), with decay on its own measured theta.

    Kaelin et al. measured the temperature dependence of net growth; decay
    depends weakly on temperature (Friedrich et al. 2017), so growth carries
    the rest. The result keeps their order: above about 19 C AOB outgrow NOB,
    below it NOB outgrow AOB, as Hellinga et al. (1998, 1999) describe."""
    d = np.arange(-10.0, 5.5, 1.0)
    mu = (mu20 - b20) * np.exp(net_k * d) + b20 * b_theta**d
    return num(math.exp(np.polyfit(d, np.log(mu / mu20), 1)[0]))


def parameters() -> dict:
    p = {}

    def add(name, value, units, bounds=None, theta=None, transform="positive_log", ref_T=293.15):
        entry = dict(value=value, units=units)
        if bounds is not None:
            entry.update(bounds=list(bounds), transform=transform)
        if theta is not None:
            entry["temperature"] = dict(theta=theta, ref_T=ref_T)
        p[name] = entry

    # Two-step denitrification (decision 18). Hiatt & Grady (2008) give each
    # heterotroph reduction step its own anoxic growth factor, relative to the
    # aerobic maximum: 0.28 for nitrate to nitrite and 0.16 for nitrite to NO,
    # the slowest of the three steps to N2 (0.16, 0.35, 0.35, one electron
    # each). Nitrite to N2 here is those three steps lumped at the slowest
    # one's capacity, 3 * 0.16 = 0.48. Every inherited process keeps its own
    # anoxic factor (the heterotrophs' ASM2d 0.8) and takes these as fractions
    # of it: 0.28 / 0.8 and 0.48 / 0.8.
    add("f_NO3_NO2", 0.35, "-", bounds=(0.05, 1.0), transform="logit")
    add("f_NO2_N2", 0.6, "-", bounds=(0.05, 1.0), transform="logit")
    # Heterotroph nitrite half-saturation below the NOB's: Latimer et al.
    # (2009) found ASMN-like values (heterotrophs above NOB) let nitrite cycle
    # back to nitrate and over-predict a full-scale plant's effluent N; the
    # revised 0.05 (with NOB 0.1) matched it.
    add("KNO2_H", 0.05, "g_N/m3", bounds=(0.005, 2.0))
    # Nitrifiers (decision 17), at 20 C, on total ammonium and total nitrite,
    # with free ammonia and free nitrous acid as inhibitors (Hiatt & Grady
    # 2008's inhibition constants, as N). Oxygen half-saturations are apparent
    # floc values, measured together on conventional activated sludge at 20 C
    # (Daebel et al. 2007): a model without diffusion takes apparent values,
    # for which AOB's exceeds NOB's (Picioreanu et al. 2016).
    # AOB growth and decay are a pair: the WERF growth rate 0.9 1/d (0.80-1.0
    # over several plants and methods) with the decay 0.17 1/d it was derived
    # with (Jones et al. 2005); decay theta 1.029 (WERF, as tabulated by
    # Friedrich et al. 2017). Ammonium half-saturation from batch tests on
    # full-scale sludge (Mannucci et al. 2020). NOB decay is the measured loss
    # of NOB activity, 0.089 1/d with theta 1.036 (Friedrich et al. 2017); NOB
    # growth is Hiatt & Grady's (2008); NOB nitrite half-saturation the middle
    # of Mehrani et al.'s (2021) measured 0.06-0.14 g N/m3. Growth thetas: see
    # growth_theta.
    aob_theta = growth_theta(0.9, 0.17, 1.029, KAELIN_NET_AOB)
    nob_theta = growth_theta(0.78, 0.089, 1.036, KAELIN_NET_NOB)
    add("muAOB", 0.9, "1/d", bounds=(0.2, 3.0), theta=aob_theta)
    add("bAOB", 0.17, "1/d", bounds=(0.02, 0.5), theta=1.029)
    add("KNH4_AOB", 0.6, "g_N/m3", bounds=(0.05, 5.0))
    add("KI_FA_AOB", 1.0, "g_N/m3", bounds=(0.1, 10.0))
    add("KI_FNA_AOB", 0.1, "g_N/m3", bounds=(0.01, 1.0))
    add("KO2_AOB", 0.65, "g_O2/m3", bounds=(0.05, 2.0))
    add("YAOB", 0.18, "g_COD/g_N", bounds=(0.05, 0.4))
    add("muNOB", 0.78, "1/d", bounds=(0.1, 3.0), theta=nob_theta)
    add("bNOB", 0.089, "1/d", bounds=(0.02, 0.5), theta=1.036)
    add("KNO2_NOB", 0.1, "g_N/m3", bounds=(0.01, 2.0))
    add("KI_FNA_NOB", 0.04, "g_N/m3", bounds=(0.004, 0.4))
    add("KI_FA_NOB", 0.2, "g_N/m3", bounds=(0.02, 2.0))
    add("KO2_NOB", 0.23, "g_O2/m3", bounds=(0.02, 2.0))
    add("YNOB", 0.06, "g_COD/g_N", bounds=(0.02, 0.3))
    # Temperatures in the free-ammonia and free-nitrous-acid equilibria
    # (Anthonisen et al. 1976): Kb/Kw = exp(6344/T), Ka = exp(-2300/T).
    add("T_FA", 6344.0, "K")
    add("T_FNA", 2300.0, "K")
    # Anammox: Hao et al. (2002) at 30 C, from Strous et al. (1998); the
    # activation energy of 70 kJ/mol (Strous et al. 1999) gives
    # theta = exp(70000 / 8.314 * (1/293.15 - 1/303.15) / 10) between 20 and 30 C.
    amx_theta = num(math.exp(70000.0 / 8.314 * (1.0 / 293.15 - 1.0 / 303.15) / 10.0))
    add("muAMX", 0.072, "1/d", bounds=(0.01, 0.3), theta=amx_theta, ref_T=303.15)
    add("bAMX", 0.003, "1/d", bounds=(0.0003, 0.03), theta=amx_theta, ref_T=303.15)
    add("KNH4_AMX", 0.07, "g_N/m3", bounds=(0.01, 1.0))
    add("KNO2_AMX", 0.05, "g_N/m3", bounds=(0.01, 1.0))
    add("KI_O2_AMX", 0.01, "g_O2/m3", bounds=(0.001, 0.2))
    # Strous et al. (1998): mol NO2 consumed and NO3 made per mol NH4 oxidized.
    add("Y_AMX_NO2", 1.32, "-", bounds=(1.1, 1.6))
    add("Y_AMX_NO3", 0.26, "-", bounds=(0.1, 0.4))
    # Decay of the autotrophs with nitrite or nitrate, relative to aerobic:
    # Hao et al. (2002), after Koch et al. (2000); Munz et al. (2011) measured
    # 0.57 (AOB) and 0.45 (NOB) on aerobically grown sludge. Without either
    # they use the core's eta_b_AN.
    add("eta_b_NOX_AUT", 0.5, "-", bounds=(0.05, 1.0), transform="logit")
    return p


def expressions() -> dict:
    return {
        # Each reduction step on its own acceptor, as Hiatt & Grady (2008):
        # the steps run side by side, neither inhibits the other.
        "anoxic_NO3": "monod_inh([SO2], KO2_H) * monod([SNO3], KNO3_H)",
        "anoxic_NO2": "monod_inh([SO2], KO2_H) * monod([SNO2], KNO2_H)",
        # The switches of the processes that do not reduce nitrogen (hydrolysis,
        # decay, fermentation, anaerobic storage) see nitrate and nitrite
        # together; aerobic + anoxic + anaerobic stays 1, and without nitrite
        # every switch is the core's.
        "anoxic": "monod_inh([SO2], KO2_H) * monod([SNO3] + [SNO2], KNO3_H)",
        "anaerobic": "monod_inh([SO2], KO2_H) * monod_inh([SNO3] + [SNO2], KNO3_H)",
        "anaerobic_only": "monod_inh([SO2], KI_O2_AN) * monod_inh([SNO3] + [SNO2], KI_NO3_AN)",
        "decay_switch_AUT": "aerobic + eta_b_NOX_AUT * anoxic + eta_b_AN * anaerobic",
        # Free ammonia and free nitrous acid, as N (Anthonisen et al. 1976, in
        # the form of Hiatt & Grady 2008).
        "free_ammonia": "[SNH4] * 10 ** {pH} / (exp(T_FA / {T}) + 10 ** {pH})",
        "free_nitrous_acid": "[SNO2] / (exp(-T_FNA / {T}) * 10 ** {pH})",
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
            rate=(
                "muAOB * monod([SNH4], KNH4_AOB) * monod([SO2], KO2_AOB) "
                "* monod_inh(free_ammonia, KI_FA_AOB) * monod_inh(free_nitrous_acid, KI_FNA_AOB) "
                f"* {p_limit} * [XAOB]"
            ),
            stoichiometry={"XAOB": 1.0, "SNO2": "1 / YAOB", **closures},
        ),
        dict(
            name="growth_of_NOB",
            description="Nitratation: nitrite to nitrate.",
            rate=(
                "muNOB * monod([SNO2], KNO2_NOB) * monod([SO2], KO2_NOB) "
                "* monod_inh(free_nitrous_acid, KI_FNA_NOB) * monod_inh(free_ammonia, KI_FA_NOB) "
                "* monod([SNH4], KNH4_H) "
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
                "nitrifiers, growing on ammonium and nitrite, inhibited by free ammonia and free "
                "nitrous acid (Hiatt & Grady 2008), with oxygen affinities measured on activated "
                "sludge (Daebel et al. 2007), AOB growth and decay from WERF measurements (Jones et al. "
                "2005; Friedrich et al. 2017) and NOB temperature dependence from Mehrani et al. (2021); anammox oxidizes ammonium with nitrite, with the kinetics of "
                "Hao et al. (2002) and the stoichiometry of Strous et al. (1998); and every group "
                "that reduces nitrate (heterotrophs, PAOs, both GAO groups) does it in two steps, "
                "nitrate to nitrite and nitrite to N2, at the step capacities of Hiatt & Grady (2008). "
                "Generated by _make_plantwide_nitrite.py."
            ),
            references=[
                "Hiatt & Grady (2008). An updated process model for carbon oxidation, nitrification, and denitrification. Water Environ. Res. 80(11), 2145-2156. doi:10.2175/106143008X304776",
                "Hao, Heijnen & van Loosdrecht (2002). Sensitivity analysis of a biofilm model describing a one-stage completely autotrophic nitrogen removal (CANON) process. Biotechnol. Bioeng. 77(3), 266-277. doi:10.1002/bit.10105",
                "Strous, Heijnen, Kuenen & Jetten (1998). The sequencing batch reactor as a powerful tool for the study of slowly growing anaerobic ammonium-oxidizing microorganisms. Appl. Microbiol. Biotechnol. 50(5), 589-596.",
                "Strous, Kuenen & Jetten (1999). Key physiology of anaerobic ammonium oxidation. Appl. Environ. Microbiol. 65(7), 3248-3250.",
                "Daebel, Manser & Gujer (2007). Exploring temporal variations of oxygen saturation constants of nitrifying bacteria. Water Res. 41(5), 1094-1102. doi:10.1016/j.watres.2006.11.011",
                "Picioreanu, Perez & van Loosdrecht (2016). Impact of cell cluster size on apparent half-saturation coefficients for oxygen in nitrifying sludge and biofilms. Water Res. 106, 371-382. doi:10.1016/j.watres.2016.10.017",
                "Mehrani, Lu, Kowal, Sobotka & Makinia (2021). Incorporation of the complete ammonia oxidation (comammox) process for modeling nitrification in suspended growth wastewater treatment systems. J. Environ. Manage. 297, 113223. doi:10.1016/j.jenvman.2021.113223",
                "Jones, Bye & Dold (2005). Nitrification parameter measurement for plant design: experience and experimental issues with new methods. Water Sci. Technol. 52(10-11), 461-468. doi:10.2166/wst.2005.0724",
                "Friedrich, Jimenez, Pruden, Miller, Metch & Takacs (2017). Rethinking growth and decay kinetics in activated sludge - towards a new adaptive kinetics approach. Water Sci. Technol. 75(3), 501-506. doi:10.2166/wst.2016.439",
                "Latimer, Pitt, Dold, Takacs & Lynch (2009). Kinetic parameters for modeling two-step nitrification and denitrification: a case study. Water Sci. Technol. 59(4), 631-638. doi:10.2166/wst.2009.593",
                "Kaelin, Manser, Rieger, Eugster, Rottermann & Siegrist (2009). Extension of ASM3 for two-step nitrification and denitrification and its calibration and validation with batch tests and pilot scale data. Water Res. 43(6), 1680-1692. doi:10.1016/j.watres.2008.12.039",
                "Hellinga, Schellen, Mulder, van Loosdrecht & Heijnen (1998). The SHARON process: an innovative method for nitrogen removal from ammonium-rich waste water. Water Sci. Technol. 37(9), 135-142. doi:10.1016/S0273-1223(98)00281-9",
                "Mannucci, Caretti, Ducci, Lubello, Gori, Coppini & Munz (2020). Ammonia oxidizing bacteria (AOB) kinetic parameters estimated using the actual maximum ammonia oxidation rate. Int. Biodeterior. Biodegrad. 147, 104876. doi:10.1016/j.ibiod.2019.104876",
                "Munz, Lubello & Oleszkiewicz (2011). Factors affecting the growth rates of ammonium and nitrite oxidizing bacteria. Chemosphere 83(5), 720-725. doi:10.1016/j.chemosphere.2011.01.058",
                "Anthonisen, Loehr, Prakasam & Srinath (1976). Inhibition of nitrification by ammonia and nitrous acid. J. Water Pollut. Control Fed. 48(5), 835-852.",
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
