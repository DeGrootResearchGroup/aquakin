"""Write ``plantwide_bio_p.yaml``: the bio-P level of the plant-wide model.

It extends ``plantwide_core`` with phosphate, phosphorus contents and three
carbon-storing populations after López-Vázquez et al. (2009): PAOs
(*Accumulibacter*, metabolic TU Delft form) and two GAO groups
(*Competibacter*, *Alphaproteobacteria*-GAO). The design and every choice
below are in ``.claude/notes/plantwide-model-design.md`` (decisions 8-12);
the source and its corrections are in ``../lopez_vazquez_2009/``.

The stoichiometric coefficients are computed here from the source's metabolic
yields (its Appendices 2-4, corrected) and converted from C-mmol to the
model's g COD, g N, g P and g C, so they are numbers in the YAML; the
closures (N, P, C, and O2 for COD) are left to ``auto``. Run from this
directory: ``python _make_plantwide_bio_p.py``.

Form (decision 10): each population stores PHA in two pools by origin, made
from acetate (``..._AC``) and from propionate (``..._PR``), each with its
composition at pH 7, so the aerobic yields are constants. Every pH effect is
carried in a rate: a "transport energy" process runs at the uptake rate times
alpha(pH), with a COD-balanced stoichiometry (in g COD the anaerobic
stoichiometry is linear in alpha). The PAO aerobic rate constants follow the
acetate share of the PAO's stored PHA, where the source uses the influent's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lopez_vazquez_2009"))
import stoich  # noqa: E402  (the reference's yield formulas, one implementation)

CORE = HERE / "plantwide_core.yaml"
OUT = HERE / "plantwide_bio_p.yaml"

# g COD per C-mol (degree of reduction x 8) and molar masses.
COD = dict(HAc=32.0, HPr=112.0 / 3.0, GLY=32.0, PHB=36.0, PHV=38.4, PH2MV=40.0)
COD_X = 8.0 * stoich.REDOX_BM  # biomass CH1.84 O0.5 N0.19 P0.015
P_G, N_G, C_G = 31.0, 14.0, 12.0
NO3_COD = 40.0 / 14.0  # g COD accepted per g NO3-N reduced to N2
PH_REF = 7.0
DELTA_AEROBIC = 1.85
# Anoxic ATP/NADH ratio giving the TU Delft anoxic/aerobic yield ratio on PHB
# (asm2d_tud: 1.39 / 1.72); the anoxic P and glycogen yields then land within
# about 10% of its values.
DELTA_ANOXIC = 0.928

POPS = dict(
    PAO=dict(X="XPAO", GLY="XGLY_PAO", AC="XPHA_PAO_AC", PR="XPHA_PAO_PR", label="PAOs (Accumulibacter)"),
    GAOC=dict(X="XGAOC", GLY="XGLY_GAOC", AC="XPHA_GAOC_AC", PR="XPHA_GAOC_PR", label="Competibacter GAOs"),
    GAOA=dict(X="XGAOA", GLY="XGLY_GAOA", AC="XPHA_GAOA_AC", PR="XPHA_GAOA_PR", label="Alphaproteobacteria-GAOs"),
)


def num(x: float) -> float:
    return float(f"{x:.10g}")


def pool_composition(pop: str, origin: str) -> tuple[float, float, float]:
    """(PHB, PHV, PH2MV) C-fractions of a pool at pH 7 (source Table 1 and
    Appendix 2)."""
    if pop == "PAO":
        return (1.0, 0.0, 0.0) if origin == "AC" else (0.0, 0.56 / 1.23, 0.67 / 1.23)
    if origin == "AC":
        a = 0.057 * PH_REF - 0.34
        d = 5 / 3 + 4 * a / 3
        parts = (
            (9 / 6 + 2 * a / 3) ** 2 / d,
            2.5 * (9 / 6 + 2 * a / 3) * (1 / 6 + 2 * a / 3) / d,
            1.5 * (1 / 6 + 2 * a / 3) ** 2 / d,
        )
    else:
        parts = (0.10, 0.58, 0.82)
    tot = sum(parts)
    return tuple(p / tot for p in parts)


def pool_cod(comp) -> float:
    """g COD per C-mol of a pool."""
    return sum(f * COD[k] for f, k in zip(comp, ("PHB", "PHV", "PH2MV")))


def pool_yields(pop: str, origin: str, delta: float) -> dict:
    stoich.DELTA = delta
    return stoich.yields(*pool_composition(pop, origin), pao=(pop == "PAO"))


def m_s(pop: str, origin: str) -> float:
    """Aerobic maintenance on PHA, C-mol/(C-mol h) (Appendix 3)."""
    lam, beta, _ = stoich.phase(*pool_composition(pop, origin))
    d = DELTA_AEROBIC
    return 12 * 0.019 / (6 * lam + 27 * lam * d + 8 * beta + 30 * beta * d)


# --- species -----------------------------------------------------------------

def biomass_composition() -> dict:
    return dict(COD=1.0, N=num(0.19 * N_G / COD_X), P=num(0.015 * P_G / COD_X), C=num(C_G / COD_X))


def species() -> list[dict]:
    out = [
        dict(name="SPO4", units="g_P/m3", default_concentration=5.0,
             description="Ortho-phosphate", composition=dict(P=1.0)),
        # Phosphorus contents of the core's organics and biomass (ASM2d).
        *(dict(name=n, composition=dict(P=p)) for n, p in (
            ("SF", 0.01), ("SI", 0.0), ("XI", 0.01), ("XS", 0.01), ("XH", 0.02), ("XAUT", 0.02),
            ("XFE", 0.02), ("XPRO", 0.02), ("XAC", 0.02), ("XH2M", 0.02))),
        dict(name="XPP", units="g_P/m3", default_concentration=10.0,
             description="Polyphosphate stored by the PAOs", composition=dict(P=1.0)),
    ]
    for pop, s in POPS.items():
        out.append(dict(name=s["X"], units="g_COD/m3", default_concentration=1.0,
                        description=s["label"] + "; storage polymers excluded",
                        composition=biomass_composition()))
        out.append(dict(name=s["GLY"], units="g_COD/m3", default_concentration=0.1,
                        description=f"Glycogen stored by the {s['label']}",
                        composition=dict(COD=1.0, C=num(C_G / COD["GLY"]))))
        for origin, what in (("AC", "acetate"), ("PR", "propionate")):
            c = pool_cod(pool_composition(pop, origin))
            out.append(dict(name=s[origin], units="g_COD/m3", default_concentration=0.0,
                            description=f"PHA made from {what} by the {s['label']} (fixed composition at pH 7)",
                            composition=dict(COD=1.0, C=num(C_G / c))))
    return out


# --- parameters ----------------------------------------------------------------

def per_day(q_cmol_per_h: float, cod_numerator: float) -> float:
    """C-mol/(C-mol X h) of a compound -> g COD (or g P) per g COD X per day."""
    return num(q_cmol_per_h * 24.0 * cod_numerator / COD_X)


def parameters() -> dict:
    p = {}

    def add(name, value, units, note=None, bounds=None, transform="positive_log"):
        entry = dict(value=num(value), units=units)
        if bounds:
            entry.update(bounds=[num(b) for b in bounds], transform=transform)
        p[name] = entry

    # Anaerobic uptake (Appendix 5), per g COD biomass per day at 20 C.
    add("q_PAO_HAc", per_day(0.20, COD["HAc"]), "1/d", bounds=(1, 20))
    add("q_PAO_HPr", per_day(0.20, COD["HPr"]), "1/d", bounds=(1, 20))
    add("q_GAOC_HAc", per_day(0.22, COD["HAc"]), "1/d", bounds=(1, 20))
    add("q_GAOC_HPr", per_day(0.01, COD["HPr"]), "1/d", bounds=(0.01, 5))
    add("q_GAOA_HAc", per_day(0.11, COD["HAc"]), "1/d", bounds=(0.5, 10))
    add("q_GAOA_HPr", per_day(0.22, COD["HPr"]), "1/d", bounds=(1, 20))
    # Above 20 C the GAO uptake rates stop rising (0.24 or 0.13 C-mmol/(C-mmol h)).
    add("qmax_ratio_GAOC", 0.24 / 0.22, "-")
    add("qmax_ratio_GAOA_HAc", 0.13 / 0.11, "-")
    add("qmax_ratio_GAOA_HPr", 0.24 / 0.22, "-")
    # Transport energy alpha(pH) (P-mol or C-mol glycogen per C-mol VFA):
    # PAO on acetate 0.16 pH - 0.7985 (Filipe et al. 2001); PAO on propionate
    # 0.18 (+0.20 per pH unit above 7.5); GAO on acetate 0.057 pH - 0.34.
    add("alpha_PAO_HAc_slope", 0.16, "1/pH")
    add("alpha_PAO_HAc_int", -0.7985, "-", bounds=(-0.95, -0.6), transform="none")
    add("alpha_PAO_HPr", 0.18, "-")
    add("alpha_GAO_HAc_slope", 0.057, "1/pH")
    add("alpha_GAO_HAc_int", -0.34, "-")
    # Anaerobic maintenance: PAO poly-P (2.35e-3 P-mmol/(C-mmol h)); GAO glycogen
    # (4.7e-3 C-mmol/(C-mmol h), Zeng et al. 2003a).
    add("m_ANA_PAO", per_day(2.35e-3, P_G), "g_P/g_COD/d")
    add("m_ANA_GAO", per_day(4.7e-3, COD["GLY"]), "1/d")
    # Aerobic storage and growth constants at 20 C, C-mol/(C-mol h) (Appendix 5,
    # Table 4): the PAO's by the origin of the PHA it uses.
    for name, v in dict(
        k_PHA_PAO_AC=0.80, k_PHA_PAO_PR=0.33, k_GLY_PAO_AC=0.015, k_GLY_PAO_PR=0.008,
        k_PP_PAO_AC=0.020, k_PP_PAO_PR=0.002, k_PHA_GAOC=0.80 * 1.10, k_GLY_GAOC=0.30,
        k_PHA_GAOA=0.80 * 0.50, k_GLY_GAOA=0.06,
    ).items():
        add(name, v, "C-mol/(C-mol h)" if "PP" not in name else "P-mol/(C-mol h)")
    # Storage limits and switches (Appendix 7), in C-mol (P-mol) per C-mol biomass.
    add("fGLY_max_PAO", 0.27, "C-mol/C-mol")
    add("fGLY_max_GAO", 0.35, "C-mol/C-mol")
    add("fPP_max", 0.30, "P-mol/C-mol")
    add("K_fPHA", 0.01, "C-mol/C-mol")
    add("K_store", 0.01, "C-mol/C-mol")
    add("K_VFA_BIOP", 0.001 * 32.0, "g_COD/m3")
    add("K_PHA_BIOP", 0.01 * 36.0, "g_COD/m3")
    add("K_GLY_BIOP", 0.01 * 32.0, "g_COD/m3")
    add("K_PP_BIOP", 0.01 * P_G, "g_P/m3")
    add("K_PO4_BIOP", 0.01 * P_G, "g_P/m3")
    # Phosphate limitation of the core's growth (ASM2d K_P).
    add("KP_H", 0.01, "g_P/m3")
    # Anoxic reduction of the PAO and GAO aerobic rates (TU Delft etaNO_PAO; the
    # GAOs' is assumed equal).
    add("etaNO3_BIOP", 0.8, "-", bounds=(0.1, 1.0), transform="logit")
    # Decay without O2 and NO3 (decision 12; ASM2d b_PAO at 20 C).
    add("b_BIOP", 0.2, "1/d", bounds=(0.05, 1.0))
    return p


# --- expressions -----------------------------------------------------------------

THETA = dict(  # Appendix 5 / Table 2
    PAO_upt=1.095, PAO_ana=1.096, PAO_PHA=1.129, PAO_GLY=1.125, PAO_PP=1.031, PAO_maint=1.064,
    GAO_upt=1.211, GAO_ana=1.028, GAO_PHA=1.109, GAO_GLY=1.071, GAO_maint=1.046, decay=1.0717735,
)


def expressions() -> dict:
    e = {
        "nutrient_limit": "monod([SNH4], KNH4_H) * monod([SPO4], KP_H)",
        "nutrient_limit_AN": "monod([SNH4], K_S_IN) * monod([SPO4], KP_H)",
        # Temperature factors; the PAO's uptake and aerobic rates are capped at 20 C.
        "dT": "{T} - 293.15",
        "dT_capped": "min({T}, 293.15) - 293.15",
        # GAO uptake falls with pH (Filipe et al. 2001a, as the source).
        "ph_gao": "1 / (1 + 1.2e-8 * 10 ** {pH})",
        "alpha_PAO_HAc": "max(alpha_PAO_HAc_slope * {pH} + alpha_PAO_HAc_int, 0)",
        "alpha_PAO_HPr_pH": "alpha_PAO_HPr + 0.2 * max({pH} - 7.5, 0)",
        "alpha_GAO_HAc": "max(alpha_GAO_HAc_slope * {pH} + alpha_GAO_HAc_int, 0)",
        "decay_switch": "anaerobic",
    }
    for pop, s in POPS.items():
        c_ac = num(pool_cod(pool_composition(pop, "AC")))
        c_pr = num(pool_cod(pool_composition(pop, "PR")))
        X = s["X"]
        e[f"pha_C_{pop}"] = f"[{s['AC']}] / {c_ac} + [{s['PR']}] / {c_pr}"
        e[f"x_C_{pop}"] = f"[{X}] / {num(COD_X)}"
        e[f"f_PHA_{pop}"] = f"safe_div(pha_C_{pop}, x_C_{pop})"
        e[f"f_GLY_{pop}"] = f"safe_div([{s['GLY']}] / 32.0, x_C_{pop})"
        e[f"share_AC_{pop}"] = f"safe_div([{s['AC']}] / {c_ac}, pha_C_{pop})"
        e[f"share_PR_{pop}"] = f"safe_div([{s['PR']}] / {c_pr}, pha_C_{pop})"
        fmax = "fGLY_max_PAO" if pop == "PAO" else "fGLY_max_GAO"
        e[f"room_GLY_{pop}"] = f"max({fmax} - f_GLY_{pop}, 0)"
        # Murnleitner kinetics on the storage fractions (Appendix 6), C-mol/h
        # per unit rate constant.
        e[f"pha_drive_{pop}"] = (
            f"f_PHA_{pop} ** (2 / 3) * x_C_{pop} * monod(f_PHA_{pop}, K_fPHA)"
            " * monod([SPO4], K_PO4_BIOP)"
        )
        e[f"gly_drive_{pop}"] = (
            f"f_PHA_{pop} ** (2 / 3) / max(f_GLY_{pop}, 0.001) * x_C_{pop}"
            f" * room_GLY_{pop} / (room_GLY_{pop} + K_store)"
            f" * monod([{s['AC']}] + [{s['PR']}], K_PHA_BIOP)"
        )
        store = f"monod([{s['GLY']}], K_GLY_BIOP)"
        if pop == "PAO":
            store += " * monod([XPP], K_PP_BIOP)"
        e[f"ana_store_{pop}"] = store
    e["f_PP"] = "safe_div([XPP] / 31.0, x_C_PAO)"
    e["room_PP"] = "max(fPP_max - f_PP, 0)"
    e["pp_drive"] = (
        "1 / max(f_PP, 0.001) * room_PP / (room_PP + K_store) * x_C_PAO"
        " * monod([XPHA_PAO_AC] + [XPHA_PAO_PR], K_PHA_BIOP) * monod([SPO4], K_PO4_BIOP)"
    )
    return e


# --- reactions -------------------------------------------------------------------

def core_overrides() -> list[dict]:
    """Every core reaction also closes phosphorus on SPO4."""
    core = yaml.safe_load(CORE.read_text())
    out = []
    for r in core["reactions"]:
        entry = dict(name=r["name"], stoichiometry=dict(SPO4="auto"))
        if "conserved_for" in r:
            entry["conserved_for"] = list(r["conserved_for"]) + ["P"]
        out.append(entry)
    return out


def anoxic_acceptor(stoich_known: dict, cod: dict) -> float:
    """SNO3 coefficient closing COD with N2 as the product: the COD of the known
    terms (taken as their own units, g COD) over 40/14."""
    return num(sum(v * cod[k] for k, v in stoich_known.items()) / NO3_COD)


def anaerobic_reactions(pop: str) -> list[dict]:
    s = POPS[pop]
    X, GLY, AC, PR = s["X"], s["GLY"], s["AC"], s["PR"]
    rx = []
    upt_T = "1.095 ** dT_capped" if pop == "PAO" else None
    for acid, S, pool in (("HAc", "SA", AC), ("HPr", "SPRO", PR)):
        if pop == "PAO":
            rate = f"q_PAO_{acid} * {upt_T} * monod([{S}], K_VFA_BIOP) * [{X}] * ana_store_PAO * anaerobic"
        else:
            ratio = "qmax_ratio_GAOC" if pop == "GAOC" else f"qmax_ratio_GAOA_{acid}"
            rate = (f"q_{pop}_{acid} * min(1.211 ** dT, {ratio}) * ph_gao"
                    f" * monod([{S}], K_VFA_BIOP) * [{X}] * ana_store_{pop} * anaerobic")
        name = f"{pop}_uptake_of_{'acetate' if acid == 'HAc' else 'propionate'}"
        # Base stoichiometry at alpha = 0, per g COD acid, COD-balanced.
        if pop == "PAO" and acid == "HAc":
            gly, pp = 0.5, 0.25 * P_G / COD["HAc"]
        elif pop == "PAO":
            gly, pp = (1 / 3) * COD["GLY"] / COD["HPr"], (2 / 9) * P_G / COD["HPr"]
        elif acid == "HAc":
            gly, pp = 1.0, 0.0
        else:
            gly, pp = (2 / 3) * COD["GLY"] / COD["HPr"], 0.0
        st = {S: -1.0, GLY: -num(gly), pool: num(1.0 + gly), "SIC": "auto"}
        if pp:
            st.update({"XPP": -num(pp), "SPO4": "auto"})
        rx.append(dict(name=name, rate=rate, stoichiometry=st, conserved_for=["COD", "C", "P"]))
        # The pH-dependent transport energy, as its own process.
        if pop == "PAO":
            a = "alpha_PAO_HAc" if acid == "HAc" else "alpha_PAO_HPr_pH"
            rx.append(dict(
                name=f"{name}_transport_energy",
                description="Poly-P hydrolysed for the pH-dependent VFA transport, per unit uptake times alpha(pH).",
                rate=f"{a} * ({rate})",
                stoichiometry={"XPP": -num(P_G / COD[acid]), "SPO4": "auto"},
                conserved_for=["P"]))
            if acid == "HPr":
                rx.append(dict(
                    name="PAO_propionate_glycogen_above_pH_7",
                    description="Less glycogen used per propionate above pH 7 (0.2 C-mol per pH unit); PHA follows by COD balance.",
                    rate=f"0.2 * max({{pH}} - 7, 0) * ({rate})",
                    stoichiometry={GLY: num(COD["GLY"] / COD["HPr"]), PR: -num(COD["GLY"] / COD["HPr"]), "SIC": "auto"},
                    conserved_for=["COD", "C"]))
        elif acid == "HAc":
            rx.append(dict(
                name=f"{name}_transport_energy",
                description="Glycogen used for the pH-dependent acetate transport (2 C-mol per alpha), its reducing power stored as PHA.",
                rate=f"alpha_GAO_HAc * ({rate})",
                stoichiometry={GLY: -2.0, AC: 2.0, "SIC": "auto"},
                conserved_for=["COD", "C"]))
    if pop == "PAO":
        rx.append(dict(name="PAO_anaerobic_maintenance", rate="m_ANA_PAO * 1.096 ** dT * [XPAO] * monod([XPP], K_PP_BIOP) * anaerobic",
                       stoichiometry={"XPP": -1.0, "SPO4": "auto"}, conserved_for=["P"]))
    else:
        rx.append(dict(name=f"{pop}_anaerobic_maintenance",
                       description="Glycogen to propionyl-rich PHA (Zeng et al. 2003a, Eq. 21).",
                       rate=f"m_ANA_GAO * 1.028 ** dT * [{X}] * monod([{GLY}], K_GLY_BIOP) * anaerobic",
                       stoichiometry={GLY: -1.0, PR: 1.0, "SIC": "auto"}, conserved_for=["COD", "C"]))
    return rx


def oxidative_reactions(pop: str) -> list[dict]:
    """Aerobic and anoxic PHA use, glycogen and poly-P storage and maintenance,
    per PHA pool, each with its pool's constant yields."""
    s = POPS[pop]
    X, GLY = s["X"], s["GLY"]
    rx = []
    is_pao = pop == "PAO"
    th = (lambda k: f"{THETA['PAO_' + k]} ** dT_capped") if is_pao else (lambda k: f"{THETA['GAO_' + k]} ** dT")
    for origin in ("AC", "PR"):
        pool = s[origin]
        c_pool = pool_cod(pool_composition(pop, origin))
        share = f"share_{origin}_{pop}"
        k_pha = f"k_PHA_PAO_{origin}" if is_pao else f"k_PHA_{pop}"
        k_gly = f"k_GLY_PAO_{origin}" if is_pao else f"k_GLY_{pop}"
        for zone, delta in (("aerobic", DELTA_AEROBIC), ("anoxic", DELTA_ANOXIC)):
            y = pool_yields(pop, origin, delta)
            zf = "aerobic" if zone == "aerobic" else "etaNO3_BIOP * anoxic"
            cod = {pool: 1.0, X: 1.0, GLY: 1.0}

            def acceptor(known):
                if zone == "aerobic":
                    return {"SO2": "auto"}
                return {"SNO3": anoxic_acceptor(known, cod), "SN2": "auto"}

            # PHA degradation: growth on the pool (rate: g COD PHA per m3 per d).
            known = {pool: -1.0, X: num(y["Y_SX"] * COD_X / c_pool)}
            rx.append(dict(
                name=f"{pop}_{zone}_growth_on_PHA_{origin}",
                rate=f"{k_pha} * 24 * {num(c_pool)} * {th('PHA')} * pha_drive_{pop} * {share} * {zf}",
                stoichiometry={**known, **acceptor(known), "SNH4": "auto", "SPO4": "auto", "SIC": "auto"}))
            # Glycogen storage, diverting PHA from growth (rate: g COD glycogen).
            known = {GLY: 1.0, X: -num(y["Y_SX"] / y["Y_SGLY"] * COD_X / COD["GLY"])}
            rx.append(dict(
                name=f"{pop}_{zone}_glycogen_storage_{origin}",
                rate=f"{k_gly} * 24 * 32.0 * {th('GLY')} * gly_drive_{pop} * {share} * {zf}",
                stoichiometry={**known, **acceptor(known), "SNH4": "auto", "SPO4": "auto", "SIC": "auto"}))
            if is_pao:
                # Poly-P storage (rate: g P per m3 per d).
                known = {X: -num(y["Y_SX"] / y["Y_SPP"] * COD_X / P_G)}
                rx.append(dict(
                    name=f"PAO_{zone}_polyP_storage_{origin}",
                    rate=f"k_PP_PAO_{origin} * 24 * 31.0 * {th('PP')} * pp_drive * {share} * {zf}",
                    stoichiometry={"XPP": 1.0, **known, **acceptor(known), "SNH4": "auto", "SPO4": "auto", "SIC": "auto"}))
            # Maintenance: biomass oxidized at m_S Y_SX (rate: g COD biomass).
            m = num(m_s(pop, origin) * y["Y_SX"] * 24)
            known = {X: -1.0}
            th_m = f"{THETA['PAO_maint' if is_pao else 'GAO_maint']} ** dT"
            rx.append(dict(
                name=f"{pop}_{zone}_maintenance_{origin}",
                rate=f"{m} * {th_m} * {share} * [{X}] * {zf}",
                stoichiometry={**known, **acceptor(known), "SNH4": "auto", "SPO4": "auto", "SIC": "auto"}))
    return rx


def decay_reactions(pop: str) -> list[dict]:
    """Death-regeneration without O2 and NO3, releasing the storage pools
    (decision 12)."""
    s = POPS[pop]
    rate = f"b_BIOP * {THETA['decay']} ** dT * decay_switch"
    rx = [dict(name=f"decay_of_{pop}", rate=f"{rate} * [{s['X']}]",
               stoichiometry={s["X"]: -1.0, "XS": "1 - fXI", "XI": "fXI", "SNH4": "auto", "SPO4": "auto", "SIC": "auto"},
               conserved_for=["N", "C", "P"])]
    for origin in ("AC", "PR"):
        rx.append(dict(name=f"lysis_of_{s[origin]}", rate=f"{rate} * [{s[origin]}]",
                       stoichiometry={s[origin]: -1.0, "SA": 1.0, "SIC": "auto"}, conserved_for=["COD", "C"]))
    rx.append(dict(name=f"lysis_of_{s['GLY']}", rate=f"{rate} * [{s['GLY']}]",
                   stoichiometry={s["GLY"]: -1.0, "XS": 1.0, "SNH4": "auto", "SPO4": "auto", "SIC": "auto"},
                   conserved_for=["COD", "N", "C", "P"]))
    if pop == "PAO":
        rx.append(dict(name="lysis_of_XPP", rate=f"{rate} * [XPP]",
                       stoichiometry={"XPP": -1.0, "SPO4": "auto"}, conserved_for=["P"]))
    return rx


def build() -> dict:
    reactions = core_overrides()
    for pop in POPS:
        reactions += anaerobic_reactions(pop) + oxidative_reactions(pop) + decay_reactions(pop)
    return dict(
        model=dict(
            name="plantwide_bio_p",
            extends="plantwide_core.yaml",
            version="0.1",
            description=(
                "Bio-P level of the plant-wide model: plantwide_core plus phosphate, phosphorus "
                "contents and the PAO-GAO competition of Lopez-Vazquez et al. (2009), PAOs in the "
                "metabolic TU Delft form and two GAO groups. PHA is held in two pools per population "
                "by origin (acetate, propionate) with fixed compositions, so every yield is a "
                "constant; pH effects are carried in the rates. Anoxic PAO/GAO processes use the "
                "TU Delft anoxic reduction (rates by etaNO3_BIOP, yields by an ATP/NADH ratio of "
                "0.928); PAOs and GAOs keep the source's maintenance and decay only without O2 and "
                "NO3. Generated by _make_plantwide_bio_p.py."
            ),
            references=[
                "Lopez-Vazquez, Oehmen, Hooijmans, Brdjanovic, Gijzen, Yuan & van Loosdrecht (2009). Modeling the PAO-GAO competition: effects of carbon source, pH and temperature. Water Res. 43(2), 450-462. doi:10.1016/j.watres.2008.10.032",
                "Smolders, van der Meij, van Loosdrecht & Heijnen (1994). Model of the anaerobic metabolism of the biological phosphorus removal process: stoichiometry and pH influence. Biotechnol. Bioeng. 43(6), 461-470.",
                "Filipe, Daigger & Grady (2001). Stoichiometry and kinetics of acetate uptake under anaerobic conditions by an enriched culture of phosphorus-accumulating organisms at different pHs. Biotechnol. Bioeng. 76(1), 32-43.",
                "Zeng, van Loosdrecht, Yuan & Keller (2003). Metabolic model for glycogen-accumulating organisms in anaerobic/aerobic activated sludge systems. Biotechnol. Bioeng. 81(1), 92-105. doi:10.1002/bit.10455",
                "Murnleitner, Kuba, van Loosdrecht & Heijnen (1997). An integrated metabolic model for the aerobic and denitrifying biological phosphorus removal. Biotechnol. Bioeng. 54(5), 434-450.",
                "Meijer, van Loosdrecht & Heijnen (2001). Metabolic modelling of full-scale biological nitrogen and phosphorus removing WWTP's. Water Res. 35(11), 2711-2723. doi:10.1016/S0043-1354(00)00567-4",
            ],
        ),
        conserved_for=["COD", "N", "C", "P"],
        species=species(),
        parameters=parameters(),
        expressions=expressions(),
        reactions=reactions,
    )


def main() -> None:
    header = "# Auto-generated by _make_plantwide_bio_p.py -- do not edit by hand.\n"
    OUT.write_text(header + yaml.safe_dump(build(), sort_keys=False, width=100, allow_unicode=True))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
