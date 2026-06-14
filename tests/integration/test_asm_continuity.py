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
from aquakin.utils.composition import composition_table

# asm1_ammonia_limitation differs from asm1 only in a rate factor; its
# stoichiometry (which the continuity checks read) is identical, so it must
# conserve COD/N exactly as asm1 does.
_MODELS = ["asm1", "asm1_ammonia_limitation", "asm3_2step", "asm2d", "asm2d_tud",
           "asm3", "asm3_biop"]
_TOL = 1.0e-2

_NITRATE = {"asm1": "SNO", "asm1_ammonia_limitation": "SNO", "asm3_2step": "SNO3",
            "asm2d": "SNO3", "asm2d_tud": "SNO", "asm3": "SNOX", "asm3_biop": "SNO"}
# Models that carry dissolved N2 as a state (so N2 is in the COD/N balance, not
# lost as gas). The ASM1 networks are the exception. asm3_2step carries SN2 and
# its two-step denitrification (NO3->NO2->N2) conserves COD/N exactly.
_TRACKS_N2 = {"asm3_2step", "asm2d", "asm2d_tud", "asm3", "asm3_biop"}
# Reactions excluded from the COD check: ASM1 denitrification, whose electrons
# leave with the untracked N2 gas (the N balance covers its nitrogen).
_COD_EXCLUDED = {"asm1": {"anoxic_growth_heterotrophs"},
                 "asm1_ammonia_limitation": {"anoxic_growth_heterotrophs"}}


@pytest.mark.parametrize("model", _MODELS)
def test_cod_continuity(model):
    net = aquakin.load_network(model)
    excl = _COD_EXCLUDED.get(model, set())
    viol = [(r, v) for r, _, v in
            check_conservation(net, composition_table(net), tol=_TOL, quantities=["COD"])
            if r not in excl]
    assert not viol, f"{model} COD imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, v in viol)


@pytest.mark.parametrize("model", _MODELS)
def test_nitrogen_continuity(model):
    net = aquakin.load_network(model)
    comp = composition_table(net)
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
    viol = check_conservation(net, composition_table(net), tol=_TOL, quantities=["P"])
    assert not viol, f"{model} P imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, _, v in viol)


def test_cod_continuity_holds_under_yield_calibration():
    # The point of the live-symbolic stoichiometry: continuity must survive a
    # changed yield. Move YH well off its default and re-check COD closes.
    net = aquakin.load_network("asm2d")
    p = net.default_parameters().at[net.param_index["YH"]].set(0.5)
    viol = check_conservation(net, composition_table(net, params=p), tol=_TOL,
                              quantities=["COD"], params=p)
    assert not viol, "asm2d COD imbalance at YH=0.5: " + "; ".join(
        f"{r} {v:+.3f}" for r, _, v in viol)
