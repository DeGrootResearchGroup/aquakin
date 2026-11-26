"""Aerobic stoichiometry of Lopez-Vazquez et al. (2009), Appendices 1, 3 and 4,
with electron and phosphorus balances to settle the signs of the storage rows.

Units: C-mmol for carbon compounds and biomass, P-mmol for P, O2-mmol for O2.
Electrons per C-mol (degree of reduction): glycogen (CH10/6 O5/6) 4,
PHB (CH1.5 O0.5) 4.5, PHV (CH1.6 O0.4) 4.8, PH2MV (CH1.67 O0.33) 5, biomass
4 + iH - 2 iO - 3 iN + 5 iP as Appendix 3 prints it.
"""

import itertools

# Appendix 3 / 4.
iH, iO, iN, iP = 1.84, 0.50, 0.19, 0.015
DELTA, K1, K2, EPS = 1.85, 1.7, 1.38, 7.0
REDOX_BM = 4 + iH - 2 * iO - 3 * iN + 5 * iP
REDOX_GLY = 4.0


def phase(a, b, c):
    """lambda, beta and PHA degree of reduction from PHB/PHV/PH2MV fractions."""
    lam = a + 2 * b / 5
    beta = c + 3 * b / 5
    redox_pha = 4.5 * a + 4.8 * b + 5 * c
    return lam, beta, redox_pha


def yields(a, b, c, pao=True):
    lam, beta, redox_pha = phase(a, b, c)
    d = DELTA
    num = 250 * (106 * lam + 127 * beta) * (6 * lam + 27 * lam * d + 8 * beta + 30 * beta * d)
    den = (
        201930 * lam + 318000 * K1 * lam + 678771 * lam * d
        + 813435 * beta * d + 269240 * beta + 381000 * K2 * beta
    )
    Y_SX = num / den
    Y_SGLY = (3 * lam + 4 * beta) * (6 * lam + 27 * lam * d + 8 * beta + 30 * beta * d) / (
        24 * (2 * lam + 3 * lam * d + 2 * beta + 4 * beta * d)
    )
    Y_SPP = EPS * (6 * lam + 27 * lam * d + 8 * beta + 30 * beta * d) / (12 * (EPS + d)) if pao else None
    Y_OX = 1 / (redox_pha / 4 / Y_SX - REDOX_BM / 4)
    Y_OGLY = 1 / (redox_pha / 4 / Y_SGLY - 1)
    Y_OPP = 1 / (redox_pha / 4 / Y_SPP) if pao else None
    return dict(Y_SX=Y_SX, Y_SGLY=Y_SGLY, Y_SPP=Y_SPP, Y_OX=Y_OX, Y_OGLY=Y_OGLY,
                Y_OPP=Y_OPP, redox_pha=redox_pha)


def rows(a, b, c, sign_store):
    """Aerobic rows (PHA degradation, glycogen and poly-P formation, maintenance)
    as {component: coefficient}; O2 entries are O2 produced (negative =
    consumed). sign_store is the sign applied to the biomass entry of the two
    storage rows: +1 as Appendix 1 prints it literally, -1 the alternative."""
    y = yields(a, b, c)
    Y_SX, Y_SGLY, Y_SPP, Y_OX, Y_OGLY, Y_OPP = (
        y[k] for k in ("Y_SX", "Y_SGLY", "Y_SPP", "Y_OX", "Y_OGLY", "Y_OPP")
    )
    return y["redox_pha"], {
        "PHA_degradation": dict(O2=-Y_SX / Y_OX, PO4=-iP * Y_SX, X=Y_SX, PHA=-1.0),
        # Phosphate follows from the P balance (structural); the electron
        # balance tests the biomass sign together with the printed O2 entry.
        "glycogen_formation": dict(
            O2=Y_SX / (Y_OX * Y_SGLY) - 1 / Y_OGLY,
            PO4=-iP * sign_store * Y_SX / Y_SGLY,
            X=sign_store * Y_SX / Y_SGLY,
            GLY=1.0,
        ),
        "polyP_formation": dict(
            O2=Y_SX / (Y_OX * Y_SPP) - 1 / Y_OPP,
            PO4=-1.0 - iP * sign_store * Y_SX / Y_SPP,
            X=sign_store * Y_SX / Y_SPP,
            PP=1.0,
        ),
        "maintenance": dict(O2=Y_SX / Y_OX - 1, PO4=iP * Y_SX, X=-Y_SX),
    }


def balances(redox_pha, row):
    """Electron imbalance (implicit CO2 and water carry none; O2 takes 4 e-/mol)
    and phosphorus imbalance of one row; both zero when it conserves."""
    e = (
        row.get("X", 0) * REDOX_BM + row.get("GLY", 0) * REDOX_GLY
        + row.get("PHA", 0) * redox_pha - row.get("O2", 0) * 4
    )
    p = row.get("PO4", 0) + row.get("PP", 0) + iP * row.get("X", 0)
    return e, p


if __name__ == "__main__":
    for comp, sign in itertools.product([(1, 0, 0), (0, 0.45, 0.55)], (+1, -1)):
        redox_pha, rr = rows(*comp, sign)
        print(f"PHA {comp}, storage-row biomass sign {sign:+d}")
        for name, row in rr.items():
            e, p = balances(redox_pha, row)
            print(f"  {name:20s} electrons {e:+.3e}  P {p:+.3e}")
    y = yields(1, 0, 0)
    print("PAO on PHB:", {k: round(v, 4) for k, v in y.items() if v is not None})
