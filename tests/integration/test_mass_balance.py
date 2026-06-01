"""Mass / electron (COD) conservation tests for the WATS sewer models.

Every reaction must conserve electrons (COD) and each chemical element. These
checks catch the class of stoichiometry bug that is otherwise easy to miss --
a wrong electron-acceptor coefficient (the O2 demand of a sulfide oxidation, the
nitrate demand of a nitrate-driven oxidation) breaks the COD balance, while a
wrong product split breaks an elemental (S / Fe) balance.

Composition convention (content per unit of the species' own measure):
  - organics: COD = 1 (g COD per g COD);
  - dissolved O2: COD = -1 (an electron acceptor is negative COD);
  - nitrate-N: COD = -2.86 (2.86 g COD accepted per g N reduced to N2);
  - sulfide: COD = 2, elemental S: COD = 1.5, sulfate: COD = 0 (g COD per g S);
  - Fe(II): COD = 0.143 g COD per g Fe.
Organic micro-contents of S/N/P (the i_s_*, i_n_bio, i_p_bio coefficients the
model balances through S_SO4 / S_NH / S_PO4) are included so the heterotrophic
balances close exactly.
"""

import os

import pytest

import aquakin
from aquakin.utils.balance import check_conservation, check_nitrogen

_NDIR = os.path.join(os.path.dirname(aquakin.__file__), "networks")

# Biomass / substrate micro-composition (literature stoichiometric coefficients).
_I_S_SB, _I_S_BIO, _I_S_XB = 0.0015, 0.01, 0.0015
_I_N_BIO, _I_N_XB, _I_P_BIO = 0.07, 0.07, 0.02

WATS_COMPOSITION = {
    # organic COD carriers (+ small S content of substrate / biomass). Particulate
    # substrate also carries organic N (i_n_xb), released as ammonia on hydrolysis.
    "S_B":   {"COD": 1.0, "S": _I_S_SB},
    "S_VFA": {"COD": 1.0},                       # acetate carries no sulfur
    "X_S1":  {"COD": 1.0, "S": _I_S_XB, "N": _I_N_XB},
    "X_S2":  {"COD": 1.0, "S": _I_S_XB, "N": _I_N_XB},
    "X_BH":  {"COD": 1.0, "N": _I_N_BIO, "P": _I_P_BIO, "S": _I_S_BIO},
    "X_BA":  {"COD": 1.0, "N": _I_N_BIO, "P": _I_P_BIO, "S": _I_S_BIO},
    "S_CH4": {"COD": 1.0},
    "S_H2":  {"COD": 1.0},
    "S_I":   {"COD": 1.0},
    "X_I":   {"COD": 1.0},
    # electron acceptors
    "S_O":   {"COD": -1.0},
    "S_NO":  {"COD": -2.86, "N": 1.0},
    # nutrient pools
    "S_NH":  {"N": 1.0},
    "S_PO4": {"P": 1.0},
    # sulfur cycle (COD = oxygen demand of the reduced sulfur)
    "sumS":  {"COD": 2.0, "S": 1.0},
    "S_SO4": {"S": 1.0},
    "X_S0":  {"COD": 1.5, "S": 1.0},
    # iron sulfide (precipitation is not a redox step; Fe(II) = 0.143 gCOD/gFe)
    "S_Fe2": {"COD": 0.143, "Fe": 1.0},
    "X_FeS": {"COD": 0.818, "S": 32.0 / 88.0, "Fe": 56.0 / 88.0},
}

_MODELS = [
    "wats_sewer",
    "wats_sewer_extended",
    "wats_sewer_khalil_paper",
    "wats_sewer_khalil_paper_balanced",
    "wats_sewer_khalil_thesis",
]

# Nitrification / autotroph decay oxidise nitrogen (NH3 -> NO3), which is not
# part of the carbon/sulfur COD continuity (its electron transfer is carried by
# the explicit O2 term and balanced in the nitrogen continuity instead). They
# are excluded from the COD check only.
_COD_EXCLUDED = {"nitrification", "nitrifier_decay"}

# The faithful Khalil model reproduces his undocumented anoxic-VFA-uptake throttle
# (0.01/Y, present only in his source code, not the paper), which lets
# denitrification consume nitrate without the matching VFA -- this does NOT
# conserve COD. The _balanced model reverts it to the standard 1/Y. Excluded from
# the faithful's COD check (and asserted separately below as a documented finding).
_KHALIL_THROTTLE_RXNS = {"anox_growth_VFA_bulk", "anox_growth_VFA_biofilm"}


def _cod_excluded(model):
    excl = set(_COD_EXCLUDED)
    if model == "wats_sewer_khalil_paper":
        excl |= _KHALIL_THROTTLE_RXNS
    return excl

# Generous absolute tolerance: published coefficients are rounded to 2-3 decimals
# (e.g. nitrate demand 0.175 vs the exact 0.1748), so true imbalances (order
# 0.1-1) are flagged while rounding (order 1e-3) is not.
_TOL = 1.0e-2


def _net(name):
    return aquakin.load_network_from_file(os.path.join(_NDIR, name + ".yaml"))


@pytest.mark.parametrize("model", _MODELS)
def test_cod_electron_balance(model):
    net = _net(model)
    excl = _cod_excluded(model)
    viol = [(r, q, v) for r, q, v in check_conservation(net, WATS_COMPOSITION,
                                                        tol=_TOL, quantities=["COD"])
            if r not in excl]
    assert not viol, f"{model} COD imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, _, v in viol)


def test_faithful_reproduces_khalil_nonconserving_throttle():
    """The faithful model reproduces Khalil's undocumented anoxic-VFA throttle,
    which does not conserve COD; the _balanced model reverts it and conserves.
    This documents the contrast (and guards it against regression)."""
    cod = {r: v for r, q, v in check_conservation(
        _net("wats_sewer_khalil_paper"), WATS_COMPOSITION, tol=_TOL, quantities=["COD"])}
    for rxn in _KHALIL_THROTTLE_RXNS:
        assert rxn in cod, f"faithful model should NOT conserve COD on {rxn!r} " \
                           "(reproduces Khalil's throttle)"
    bal = [r for r, q, v in check_conservation(
        _net("wats_sewer_khalil_paper_balanced"), WATS_COMPOSITION, tol=_TOL,
        quantities=["COD"]) if r not in _COD_EXCLUDED]
    assert not bal, f"balanced model COD imbalance: {bal}"


@pytest.mark.parametrize("model", _MODELS)
def test_sulfur_balance(model):
    net = _net(model)
    viol = check_conservation(net, WATS_COMPOSITION, tol=_TOL, quantities=["S"])
    assert not viol, f"{model} sulfur imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, _, v in viol)


@pytest.mark.parametrize("model", _MODELS)
def test_iron_balance(model):
    net = _net(model)
    viol = check_conservation(net, WATS_COMPOSITION, tol=_TOL, quantities=["Fe"])
    assert not viol, f"{model} iron imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, _, v in viol)


def test_balanced_model_conserves_nitrogen():
    """The mass/electron-balanced model conserves nitrogen (accounting for
    denitrification N2 gas) -- the full WATS N-content terms are restored. The
    faithful wats_sewer_khalil_paper deliberately omits N tracking (matching the
    published model), so it does NOT conserve N; that contrast is the point of
    the two-model pair."""
    balanced = _net("wats_sewer_khalil_paper_balanced")
    viol = check_nitrogen(balanced, WATS_COMPOSITION, tol=_TOL)
    assert not viol, "balanced model N imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, v in viol)

    # The faithful paper model does NOT track nitrogen -- assert the contrast so
    # the simplification is documented and regressions are caught.
    faithful = _net("wats_sewer_khalil_paper")
    assert check_nitrogen(faithful, WATS_COMPOSITION, tol=_TOL), \
        "faithful model unexpectedly conserves N (it should reproduce the " \
        "published model's N-simplification)"
