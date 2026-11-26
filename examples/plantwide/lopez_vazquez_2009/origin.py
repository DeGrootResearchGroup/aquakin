"""Decision 10's simplification of the reference model: PHA in two pools per
population by origin (from acetate, from propionate; GAO maintenance PHA,
propionyl-rich, joins the propionate pool), each with a fixed composition at
pH_REF, so every yield is a constant as the engine requires. Exact in the
engine and kept exact here: pH in the P release and glycogen use (linear in
alpha) and in the PAO propionate PHA total. Fixed at pH_REF: the composition
of each pool (aerobic yields, maintenance) and the GAO acetate PHA yield.
"""

import model as ref
import numpy as np
from stoich import iP, phase, yields

PH_REF = 7.0
# Fix the GAO acetate PHA yield at PH_REF (True), or carry its pH dependence
# in the rate, which the engine can do exactly (False).
GAO_ACETATE_AT_REF = False
POPS = ref.POPS
FIELDS = ("X", "PHAac", "PHApr", "GLY")
NAMES = ["HAc", "HPr", "PO4"] + [f"{p}_{f}" for p in POPS for f in FIELDS] + ["PAO_PP"]
IX = {n: i for i, n in enumerate(NAMES)}
MAINT = dict(PHB=1 / 6, PHV=5 / 12, PH2MV=1 / 4)


def composition(d):
    tot = sum(d.get(k, 0.0) for k in ("PHB", "PHV", "PH2MV"))
    return tot, tuple(d.get(k, 0.0) / tot for k in ("PHB", "PHV", "PH2MV"))


# Fixed pool compositions and their constant yields, per population.
POOL = {}
for pop in POPS:
    _, fac = composition(ref.anaerobic_stoich(pop, "HAc", PH_REF))
    _, fpr = composition(ref.anaerobic_stoich(pop, "HPr", PH_REF))
    POOL[pop] = {}
    for pool, fr in (("PHAac", fac), ("PHApr", fpr)):
        lam, beta, _ = phase(*fr)
        m_s = 12 * 0.019 / (6 * lam + 27 * lam * 1.85 + 8 * beta + 30 * beta * 1.85)
        POOL[pop][pool] = dict(y=yields(*fr, pao=(pop == "PAO")), m_s=m_s)


def anaerobic(pop, acid, pH):
    """(PHA total to its pool, glycogen, P) per C-mmol acid; PHA total at the
    actual pH except the GAO acetate yield (nonlinear in alpha), at PH_REF."""
    d = ref.anaerobic_stoich(pop, acid, pH)
    tot, _ = composition(d)
    if GAO_ACETATE_AT_REF and pop != "PAO" and acid == "HAc":
        tot, _ = composition(ref.anaerobic_stoich(pop, acid, PH_REF))
    return tot, d["GLY"], d.get("PO4", 0.0)


def rhs(t, y, aerobic, T, pH, f_hac):
    y = np.maximum(y, 0.0)
    dy = np.zeros_like(y)
    S = {a: y[IX[a]] for a in ("HAc", "HPr")}
    PO4, PP = y[IX["PO4"]], y[IX["PAO_PP"]]
    for pop in POPS:
        g = {f: y[IX[f"{pop}_{f}"]] for f in FIELDS}
        X = g["X"]
        if X <= 0:
            continue
        pha = g["PHAac"] + g["PHApr"]
        if not aerobic:
            store = ref.mono(g["GLY"], ref.K["GLY"]) * (
                ref.mono(PP, ref.K["PP"]) if pop == "PAO" else 1.0
            )
            for acid, pool in (("HAc", "PHAac"), ("HPr", "PHApr")):
                r = ref.uptake_rate(pop, acid, T, pH) * ref.mono(S[acid], ref.K[acid]) * X * store
                tot, gly, p = anaerobic(pop, acid, pH)
                dy[IX[acid]] -= r
                dy[IX[f"{pop}_{pool}"]] += tot * r
                dy[IX[f"{pop}_GLY"]] += gly * r
                if pop == "PAO":
                    dy[IX["PAO_PP"]] -= p * r
                    dy[IX["PO4"]] += p * r
            if pop == "PAO":
                r = 2.35e-3 * 1.096 ** (T - 20.0) * X * ref.mono(PP, ref.K["PP"])
                dy[IX["PAO_PP"]] -= r
                dy[IX["PO4"]] += r
            else:
                r = 4.7e-3 * 1.028 ** (T - 20) * X * ref.mono(g["GLY"], ref.K["GLY"])
                dy[IX[f"{pop}_GLY"]] -= r
                dy[IX[f"{pop}_PHApr"]] += sum(MAINT.values()) * r
            continue
        k_pha, k_gly, k_pp, th_m = ref.aerobic_constants(pop, T, f_hac)
        f_pha, f_gly = pha / X, g["GLY"] / X
        r4 = k_pha * f_pha ** (2 / 3) * X * ref.mono(f_pha, ref.K["fPHA"]) * ref.mono(PO4, ref.K["PO4"])
        room = max(ref.FMAX_GLY[pop] - f_gly, 0.0)
        r5 = (k_gly * f_pha ** (2 / 3) / max(f_gly, 1e-3) * X
              * room / (room + ref.K["GLY"]) * ref.mono(pha, ref.K["PHA"]))
        if pop == "PAO":
            f_pp = PP / X
            room = max(ref.FMAX_PP - f_pp, 0.0)
            r6 = (k_pp / max(f_pp, 1e-3) * room / (room + ref.K["GLY"]) * X
                  * ref.mono(pha, ref.K["PHA"]) * ref.mono(PO4, ref.K["PO4"]))
        else:
            r6 = 0.0
        # Each process split over the pools in proportion; yields per pool.
        for pool in ("PHAac", "PHApr"):
            share = g[pool] / pha if pha > 0 else 0.5
            c = POOL[pop][pool]
            Y = c["y"]
            dy[IX[f"{pop}_{pool}"]] -= share * r4
            dy[IX[f"{pop}_X"]] += Y["Y_SX"] * share * r4
            dy[IX["PO4"]] -= iP * Y["Y_SX"] * share * r4
            dy[IX[f"{pop}_GLY"]] += share * r5
            dy[IX[f"{pop}_X"]] -= Y["Y_SX"] / Y["Y_SGLY"] * share * r5
            dy[IX["PO4"]] += iP * Y["Y_SX"] / Y["Y_SGLY"] * share * r5
            if pop == "PAO":
                dy[IX["PAO_PP"]] += share * r6
                dy[IX[f"{pop}_X"]] -= Y["Y_SX"] / Y["Y_SPP"] * share * r6
                dy[IX["PO4"]] += (-1.0 + iP * Y["Y_SX"] / Y["Y_SPP"]) * share * r6
            r7 = c["m_s"] * th_m ** (T - 20) * X * share
            dy[IX[f"{pop}_X"]] -= Y["Y_SX"] * r7
            dy[IX["PO4"]] += iP * Y["Y_SX"] * r7
    return dy


def run_sbr(T, pH, f_hac, **kw):
    """The simplified model in the same SBR."""
    return ref.operate(rhs, NAMES, IX, T, pH, f_hac, **kw)


if __name__ == "__main__":
    import json

    from grid import classify

    rows = json.load(open("grid.json"))
    worst, agree_ref, agree_paper = 0.0, 0, 0
    for r in rows:
        fr, _ = run_sbr(r["T"], r["pH"], r["f_hac"])
        dev = max(abs(fr[p] - r[p]) for p in POPS)
        worst = max(worst, dev)
        got = classify(fr)
        agree_ref += got == r["got"]
        agree_paper += got == r["paper"]
        flag = "" if dev < 0.05 else "  <-- >5 points"
        print(f"HAc {r['f_hac']:4.2f} {r['T']:2d}C pH{r['pH']}: origin PAO {fr['PAO']:.2f} CMP {fr['CMP']:.2f} ALP {fr['ALP']:.2f} | ref {r['PAO']:.2f} {r['CMP']:.2f} {r['ALP']:.2f}  max dev {dev:.3f}{flag}", flush=True)
    print(f"worst population-share deviation from the faithful model: {worst:.3f}")
    print(f"classification agrees with faithful model {agree_ref}/36, with the paper {agree_paper}/36")
