"""Reference implementation of Lopez-Vazquez et al. (2009), with the
corrections established from its sources (design note, 2026-11-26):
PAO P release 1/4 + alpha per acetate and 2/9 + alpha per propionate,
storage rows reduce biomass, electron-balanced aerobic maintenance, GAO
anaerobic maintenance per Zeng et al. (2003a) Eq. 21.

Faithful where the engine version will simplify: PHA is tracked by type
(PHB, PHV, PH2MV) per population and the aerobic yields follow the current
PHA composition. Units: C-mmol/L, P-mmol/L, hours.
"""

import numpy as np
from scipy.integrate import solve_ivp
from stoich import iP, phase, yields

POPS = ("PAO", "CMP", "ALP")
FIELDS = ("X", "PHB", "PHV", "PH2MV", "GLY")
NAMES = ["HAc", "HPr", "PO4"] + [f"{p}_{f}" for p in POPS for f in FIELDS] + ["PAO_PP"]
IX = {n: i for i, n in enumerate(NAMES)}
K = dict(HAc=0.001, HPr=0.001, PHA=0.01, GLY=0.01, PP=0.01, PO4=0.01, fPHA=0.01)
FMAX_GLY = dict(PAO=0.27, CMP=0.35, ALP=0.35)
FMAX_PP = 0.30
REDOX = dict(PHB=4.5, PHV=4.8, PH2MV=5.0)


def mono(s, k):
    return s / (s + k) if s > 0 else 0.0


def theta_below20(value, theta, T):
    """value * theta**(T-20) up to 20 C, constant above (Appendix 5)."""
    return value * theta ** (min(T, 20.0) - 20.0)


def gao_ph(pH):
    h = 10.0 ** -pH
    return h / (1.2e-8 + h)


def anaerobic_stoich(pop, acid, pH):
    """{component: coefficient} per C-mmol acid taken up (PHA by type)."""
    if pop == "PAO":
        if acid == "HAc":
            a = 0.16 * pH - 0.7985
            p = 0.25 + a
            return dict(GLY=-0.5, PHB=4 / 3, PP=-p, PO4=p)
        if pH <= 7.0:
            phv, ph2mv, gly = 0.56, 0.67, -1 / 3
        else:
            phv, ph2mv, gly = 2.0217 - 0.21 * pH, 1.1716 - 0.07 * pH, -(1.73 - 0.2 * pH)
        p = 2 / 9 + 0.18 if pH <= 7.5 else 0.20 * pH - 1.1
        return dict(GLY=gly, PHV=phv, PH2MV=ph2mv, PP=-p, PO4=p)
    if acid == "HAc":
        a = 0.057 * pH - 0.34
        d = 5 / 3 + 4 * a / 3
        return dict(
            GLY=-(1 + 2 * a),
            PHB=(9 / 6 + 2 * a / 3) ** 2 / d,
            PHV=2.5 * (9 / 6 + 2 * a / 3) * (1 / 6 + 2 * a / 3) / d,
            PH2MV=1.5 * (1 / 6 + 2 * a / 3) ** 2 / d,
        )
    # Propionate: Table 1 composition (Appendix 2's Alphaproteobacteria PH2MV
    # term does not reproduce it); glycogen 2/3 (alpha_HPr = 0).
    return dict(GLY=-2 / 3, PHB=0.10, PHV=0.58, PH2MV=0.82)


def uptake_rate(pop, acid, T, pH):
    if pop == "PAO":
        return theta_below20(0.20, 1.095, T)
    q20, q_hi = dict(
        CMP=dict(HAc=(0.22, 0.24), HPr=(0.01, 0.01)),
        ALP=dict(HAc=(0.11, 0.13), HPr=(0.22, 0.24)),
    )[pop][acid]
    q = q20 * 1.211 ** (T - 20.0) if T <= 20 else q_hi
    return q * gao_ph(pH)


def aerobic_constants(pop, T, f_hac):
    """k_PHA, k_GLY, k_PP (per h) and maintenance theta (Appendix 5; PAO
    glycogen on propionate 0.008 from Table 4)."""
    if pop == "PAO":
        k_pha = theta_below20((0.80 - 0.33) * f_hac + 0.33, 1.129, T)
        k_gly = theta_below20((0.015 - 0.008) * f_hac + 0.008, 1.125, T)
        k_pp = theta_below20((0.020 - 0.002) * f_hac + 0.002, 1.031, T)
        return k_pha, k_gly, k_pp, 1.064
    k_pha20, k_gly20 = dict(CMP=(1.10, 0.30), ALP=(0.50, 0.06))[pop]
    return 0.80 * k_pha20 * 1.109 ** (T - 20), k_gly20 * 1.071 ** (T - 20), 0.0, 1.046


def rhs(t, y, aerobic, T, pH, f_hac):
    y = np.maximum(y, 0.0)
    dy = np.zeros_like(y)
    S = {a: y[IX[a]] for a in ("HAc", "HPr")}
    PO4 = y[IX["PO4"]]
    PP = y[IX["PAO_PP"]]
    for pop in POPS:
        g = {f: y[IX[f"{pop}_{f}"]] for f in FIELDS}
        X = g["X"]
        if X <= 0:
            continue
        pha = g["PHB"] + g["PHV"] + g["PH2MV"]
        if not aerobic:
            store = mono(g["GLY"], K["GLY"]) * (mono(PP, K["PP"]) if pop == "PAO" else 1.0)
            for acid in ("HAc", "HPr"):
                r = uptake_rate(pop, acid, T, pH) * mono(S[acid], K[acid]) * X * store
                dy[IX[acid]] -= r
                for comp, c in anaerobic_stoich(pop, acid, pH).items():
                    key = "PAO_PP" if comp == "PP" else ("PO4" if comp == "PO4" else f"{pop}_{comp}")
                    dy[IX[key]] += c * r
            if pop == "PAO":
                r = 2.35e-3 * 1.096 ** (T - 20.0) * X * mono(PP, K["PP"])
                dy[IX["PAO_PP"]] -= r
                dy[IX["PO4"]] += r
            else:
                r = 4.7e-3 * 1.028 ** (T - 20) * X * mono(g["GLY"], K["GLY"])
                dy[IX[f"{pop}_GLY"]] -= r
                for comp, c in dict(PHB=1 / 6, PHV=5 / 12, PH2MV=1 / 4).items():
                    dy[IX[f"{pop}_{comp}"]] += c * r
            continue
        if pha <= 0:
            fr = (1.0, 0.0, 0.0)
        else:
            fr = (g["PHB"] / pha, g["PHV"] / pha, g["PH2MV"] / pha)
        yl = yields(*fr, pao=(pop == "PAO"))
        k_pha, k_gly, k_pp, th_m = aerobic_constants(pop, T, f_hac)
        f_pha, f_gly = pha / X, g["GLY"] / X
        # PHA degradation (growth), shared by the three PHA types in proportion.
        r4 = k_pha * f_pha ** (2 / 3) * X * mono(f_pha, K["fPHA"]) * mono(PO4, K["PO4"])
        for comp, f in zip(("PHB", "PHV", "PH2MV"), fr):
            dy[IX[f"{pop}_{comp}"]] -= f * r4
        dy[IX[f"{pop}_X"]] += yl["Y_SX"] * r4
        dy[IX["PO4"]] -= iP * yl["Y_SX"] * r4
        # Glycogen formation, diverting PHA from growth.
        room = max(FMAX_GLY[pop] - f_gly, 0.0)
        r5 = (
            k_gly * f_pha ** (2 / 3) / max(f_gly, 1e-3) * X
            * room / (room + K["GLY"]) * mono(pha, K["PHA"])
        )
        dy[IX[f"{pop}_GLY"]] += r5
        dy[IX[f"{pop}_X"]] -= yl["Y_SX"] / yl["Y_SGLY"] * r5
        dy[IX["PO4"]] += iP * yl["Y_SX"] / yl["Y_SGLY"] * r5
        if pop == "PAO":
            f_pp = PP / X
            room = max(FMAX_PP - f_pp, 0.0)
            r6 = (
                k_pp / max(f_pp, 1e-3) * room / (room + K["GLY"]) * X
                * mono(pha, K["PHA"]) * mono(PO4, K["PO4"])
            )
            dy[IX["PAO_PP"]] += r6
            dy[IX[f"{pop}_X"]] -= yl["Y_SX"] / yl["Y_SPP"] * r6
            dy[IX["PO4"]] += (-1.0 + iP * yl["Y_SX"] / yl["Y_SPP"]) * r6
        # Aerobic maintenance on PHA, diverting it from growth.
        lam, beta, _ = phase(*fr)
        m_s = 12 * 0.019 / (6 * lam + 27 * lam * 1.85 + 8 * beta + 30 * beta * 1.85)
        r7 = m_s * th_m ** (T - 20) * X
        dy[IX[f"{pop}_X"]] -= yl["Y_SX"] * r7
        dy[IX["PO4"]] += iP * yl["Y_SX"] * r7
    return dy


def initial_state(names, ix, x0):
    """Every population at x0 with glycogen (and the PAO's poly-P) at 90% of
    its maximum fraction, as the source starts its simulations."""
    y = np.zeros(len(names))
    for pop in POPS:
        y[ix[f"{pop}_X"]] = x0
        y[ix[f"{pop}_GLY"]] = 0.9 * FMAX_GLY[pop] * x0
    y[ix["PAO_PP"]] = 0.9 * FMAX_PP * x0
    return y


def operate(rhs_fn, names, ix, T, pH, f_hac, days=50.0, x0=30.0, srt_d=8.0,
            exchange=0.5, t_ana=2.0, t_aer=3.5, t_idle=0.5, vfa=5.0, po4_in=0.40):
    """Run the reconstructed SBR (fill, anaerobic, aerobic, settle/idle; solids
    wasted at the end of the aerobic phase) and return the biomass fractions
    and the final state."""
    y = initial_state(names, ix, x0)
    cycle = t_ana + t_aer + t_idle
    waste = cycle / (srt_d * 24.0)
    soluble = [ix["HAc"], ix["HPr"], ix["PO4"]]
    particulate = [i for i in range(len(names)) if i not in soluble]
    feed = np.zeros(len(names))
    feed[ix["HAc"]] = vfa * f_hac / exchange
    feed[ix["HPr"]] = vfa * (1 - f_hac) / exchange
    feed[ix["PO4"]] = po4_in / exchange
    for _ in range(round(days * 24 / cycle)):
        y[soluble] = (1 - exchange) * y[soluble] + exchange * feed[soluble]
        for aerobic, dur in ((False, t_ana), (True, t_aer)):
            sol = solve_ivp(rhs_fn, (0, dur), y, method="LSODA", args=(aerobic, T, pH, f_hac),
                            rtol=1e-6, atol=1e-9)
            y = np.maximum(sol.y[:, -1], 0.0)
        y[particulate] *= 1 - waste
    xs = {p: y[ix[f"{p}_X"]] for p in POPS}
    tot = sum(xs.values())
    return {p: xs[p] / tot for p in POPS}, y


def run_sbr(T, pH, f_hac, **kw):
    """The faithful model in the reconstructed SBR."""
    return operate(rhs, NAMES, IX, T, pH, f_hac, **kw)
