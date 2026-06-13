"""COD / N / P continuity (Gujer-matrix conservation) for the ASM family (#210).

Commercial simulators enforce composition-matrix continuity (COD, N, P close to
machine precision) as a first-class invariant. aquakin's ASM stoichiometry is
live-symbolic, so continuity holds under yield calibration *iff* the coefficients
are right -- and the asm2d / asm3 coefficients are generator-derived. This pins
that: for each ASM network, every reaction conserves COD, N and (where modelled)
P, given a per-species composition vector, evaluated against the live
``compute_stoich`` so the symbolic coefficients are exercised.

Conventions (matching ``aquakin/utils/balance.py`` and the WATS coverage in
``test_mass_balance.py``):

- Organic COD carriers (substrates, biomass, storage, inerts) carry ``COD = 1``;
  dissolved oxygen carries ``COD = -1`` (an electron acceptor).
- Nitrate carries ``COD = iCOD_NO3 = -4.571`` g COD/g N -- the NH4-referenced
  electron equivalent the ASM2d/3 stoichiometry uses (8 e- for NH4 -> NO3). The
  models that track dissolved N2 give it ``COD = iCOD_NO3 + iNO3_N2`` so
  denitrification (NO3 -> N2) closes; ASM1 does not track N2, so its single
  denitrification reaction is excluded from the COD check (the electrons leave
  with the untracked N2 gas) and the N balance accounts for it via
  ``check_nitrogen``.
- N / P contents of organics are the model's own ``i_XB`` / ``iN_*`` / ``iP_*``
  parameters, read from the network so the test tracks the model's constants.
"""
import pytest

import aquakin
from aquakin.utils.balance import check_conservation, check_nitrogen

_MODELS = ["asm1", "asm2d", "asm2d_tud", "asm3", "asm3_biop"]
_TOL = 1.0e-2

# NH4-referenced COD of nitrate (g COD / g N); ASM1 has no such parameter, so the
# canonical value is used there.
_ICOD_NO3 = -32.0 / 7.0          # = -4.571428...

_NITRATE = {"asm1": "SNO", "asm2d": "SNO3", "asm2d_tud": "SNO",
            "asm3": "SNOX", "asm3_biop": "SNO"}
# Models that carry dissolved N2 as a state (so N2 is in the COD/N balance, not
# lost as gas). ASM1 is the exception.
_TRACKS_N2 = {"asm2d", "asm2d_tud", "asm3", "asm3_biop"}
# Reactions excluded from the COD check: ASM1 denitrification, whose electrons
# leave with the untracked N2 gas (the N balance covers its nitrogen).
_COD_EXCLUDED = {"asm1": {"anoxic_growth_heterotrophs"}}

_BIOMASS = {"XB_H", "XB_A", "XH", "XPAO", "XAUT", "XA"}
_STORAGE = {"SA", "XPHA", "XSTO", "XGLY"}          # COD = 1, no N / P
_NPOOL = {"SNH", "SNH4", "SND", "XND"}             # N = 1
_PPOOL = {"SPO4", "SPO", "XPP"}                    # P = 1
_OXYGEN = {"SO", "SO2"}


def _p(net, name, default=0.0):
    if name in net.param_index:
        return float(net.default_parameters()[net.param_index[name]])
    return default


def _composition(net) -> dict:
    """Per-species COD / N / P content for an ASM network, from its own
    composition parameters."""
    iN_BM = _p(net, "iN_BM", _p(net, "i_XB"))
    iN_SF, iN_SS = _p(net, "iN_SF"), _p(net, "iN_SS")
    iN_SI, iN_XI, iN_XS = _p(net, "iN_SI"), _p(net, "iN_XI"), _p(net, "iN_XS")
    iP_BM, iP_SF, iP_SI = _p(net, "iP_BM"), _p(net, "iP_SF"), _p(net, "iP_SI")
    iP_XI, iP_XS = _p(net, "iP_XI"), _p(net, "iP_XS")
    iXP = _p(net, "i_XP")
    icod_no3 = _p(net, "iCOD_NO3", _ICOD_NO3)
    n2_cod = icod_no3 + _p(net, "iNO3_N2")          # 0 if no iNO3_N2 (ASM1)
    fMeP = _p(net, "fMeP_PO4_MW")

    comp: dict = {}
    for sp in net.species:
        c: dict = {}
        if sp in _BIOMASS:
            c = {"COD": 1.0, "N": iN_BM, "P": iP_BM}
        elif sp == "SF":
            c = {"COD": 1.0, "N": iN_SF, "P": iP_SF}
        elif sp == "SS":
            c = {"COD": 1.0, "N": iN_SS}
        elif sp == "SI":
            c = {"COD": 1.0, "N": iN_SI, "P": iP_SI}
        elif sp == "XI":
            c = {"COD": 1.0, "N": iN_XI, "P": iP_XI}
        elif sp == "XS":
            c = {"COD": 1.0, "N": iN_XS, "P": iP_XS}
        elif sp == "XP":
            c = {"COD": 1.0, "N": iXP}
        elif sp in _STORAGE:
            c = {"COD": 1.0}
        elif sp in _OXYGEN:
            c = {"COD": -1.0}
        elif sp in _NITRATE.values():
            c = {"COD": icod_no3, "N": 1.0}
        elif sp == "SN2":
            c = {"COD": n2_cod, "N": 1.0}
        elif sp in _NPOOL:
            c = {"N": 1.0}
        elif sp in _PPOOL:
            c = {"P": 1.0}
        elif sp == "XMeP":                          # precipitated phosphate
            c = {"P": 1.0 / fMeP} if fMeP else {}
        # alkalinity, TSS, metal hydroxide -> no COD/N/P content
        comp[sp] = {k: v for k, v in c.items() if v != 0.0}
    return comp


@pytest.mark.parametrize("model", _MODELS)
def test_cod_continuity(model):
    net = aquakin.load_network(model)
    excl = _COD_EXCLUDED.get(model, set())
    viol = [(r, v) for r, _, v in
            check_conservation(net, _composition(net), tol=_TOL, quantities=["COD"])
            if r not in excl]
    assert not viol, f"{model} COD imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, v in viol)


@pytest.mark.parametrize("model", _MODELS)
def test_nitrogen_continuity(model):
    net = aquakin.load_network(model)
    comp = _composition(net)
    if model in _TRACKS_N2:
        # N2 is a tracked state carrying N=1, so the plain balance closes.
        viol = [(r, v) for r, _, v in
                check_conservation(net, comp, tol=_TOL, quantities=["N"])]
    else:
        # ASM1: nitrate reduced to (untracked) N2 gas -- check_nitrogen credits
        # the consumed nitrate back as the gas leaving.
        viol = check_nitrogen(net, comp, tol=_TOL, nitrate=_NITRATE[model])
    assert not viol, f"{model} N imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, v in viol)


@pytest.mark.parametrize("model", ["asm2d", "asm2d_tud", "asm3_biop"])
def test_phosphorus_continuity(model):
    net = aquakin.load_network(model)
    viol = check_conservation(net, _composition(net), tol=_TOL, quantities=["P"])
    assert not viol, f"{model} P imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, _, v in viol)


def test_cod_continuity_holds_under_yield_calibration():
    # The point of the live-symbolic stoichiometry: continuity must survive a
    # changed yield. Move YH well off its default and re-check COD closes.
    net = aquakin.load_network("asm2d")
    p = net.default_parameters().at[net.param_index["YH"]].set(0.5)
    viol = check_conservation(net, _composition(net), tol=_TOL,
                              quantities=["COD"], params=p)
    assert not viol, "asm2d COD imbalance at YH=0.5: " + "; ".join(
        f"{r} {v:+.3f}" for r, _, v in viol)
