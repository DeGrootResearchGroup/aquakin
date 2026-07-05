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
model balances through S_SO4 / S_NH / S_PO4) close the heterotrophic balances.

The composition table is no longer a literal dict here: each WATS model now
declares its own per-species ``composition:`` in the YAML, read back through
``model.composition()`` and checked with ``model.check_conservation()`` /
``model.check_nitrogen()`` (the conservation table is a first-class model
property, so a value can no longer drift out of sync with the model it describes).
"""

import os

import pytest

import aquakin

_NDIR = os.path.join(os.path.dirname(aquakin.__file__), "models")

_MODELS = [
    "wats_sewer",
    "wats_sewer_extended",
    "wats_sewer_khalil_paper",
    "wats_sewer_khalil_paper_balanced",
    "wats_sewer_khalil_thesis",
    # Structural variants of both Khalil bases (reviewer-feedback alternatives).
    # The faithful variants carry the faithful base's VFA throttle (excluded from
    # the COD check); the balanced variants conserve everything. These guard the
    # variant-generator stoichiometry (the directsulfate one-step nitrate demand
    # and the S_B-driven elemental-S reduction coefficient).
    "wats_sewer_khalil_paper_halforder",
    "wats_sewer_khalil_paper_directsulfate",
    "wats_sewer_khalil_paper_srbsubstrate",
    "wats_sewer_khalil_paper_combined",
    "wats_sewer_khalil_paper_stopatS0",
    "wats_sewer_khalil_paper_balanced_halforder",
    "wats_sewer_khalil_paper_balanced_directsulfate",
    "wats_sewer_khalil_paper_balanced_combined",
    "wats_sewer_khalil_paper_balanced_stopatS0",
    # Areal {A_V} biofilm split of the balanced base (bulk/biofilm phase split,
    # spatially uniform biofilm activity). Same stoichiometry as the lumped
    # balanced base, so it conserves COD/S/Fe identically.
    "wats_sewer_khalil_paper_balanced_biofilm",
    # Per-layer-biomass biofilm variant: biofilm processes driven by local
    # volumetric [X_BH] instead of the areal {A_V} lump. Same stoichiometry as
    # the balanced base, so it conserves COD/S/Fe identically.
    "wats_sewer_khalil_paper_balanced_biofilm_biomass",
    # Full multispecies biofilm: sulfur/methane processes grow their own
    # functional-group biomass (X_SRB/X_MA/X_SOB) at literature yields with
    # COD/S/N-conserving stoichiometry.
    "wats_sewer_khalil_paper_balanced_biofilm_multispecies",
]

# Shipped wats_sewer*.yaml models deliberately NOT under the conservation
# checks, each with the reason. The completeness guard below asserts that every
# shipped WATS model is either in _MODELS (checked) or listed here, so a new
# variant from the _make_* generators cannot be added without a conservation
# decision. The extended-model structural variants carry COD imbalances in the
# sulfur-oxidation / elemental-S reactions (the same class the Khalil-family
# generator later fixed but which was never back-ported to these earlier extended
# variants); the conserving `wats_sewer_extended` base itself IS checked.
_BALANCE_EXEMPT = {
    "wats_sewer_extended_v0":
        "early extended-model prototype; superseded, not conservation-vetted",
    "wats_sewer_extended_combined":
        "extended-base structural variant with un-back-ported sulfur-stoich "
        "COD imbalances",
    "wats_sewer_extended_directsulfate":
        "extended-base structural variant with un-back-ported sulfur-stoich "
        "COD imbalances",
    "wats_sewer_extended_halforder":
        "extended-base structural variant with un-back-ported sulfur-stoich "
        "COD imbalances",
    "wats_sewer_extended_srbsubstrate":
        "extended-base structural variant with un-back-ported sulfur-stoich "
        "COD imbalances",
}


def test_every_wats_model_is_checked_or_exempt():
    """Completeness guard: every shipped ``wats_sewer*.yaml`` is either under the
    conservation checks (``_MODELS``) or explicitly exempt (``_BALANCE_EXEMPT``)
    with a reason. Catches the failure mode where a new variant added to the
    ``_make_*`` generators is silently left unchecked -- the hand-maintained list
    can otherwise drift out of sync with the directory."""
    import glob
    on_disk = {os.path.basename(p)[:-len(".yaml")]
               for p in glob.glob(os.path.join(_NDIR, "wats_sewer*.yaml"))}
    accounted = set(_MODELS) | set(_BALANCE_EXEMPT)
    unaccounted = on_disk - accounted
    assert not unaccounted, (
        "shipped WATS models neither checked nor exempt (add to _MODELS if they "
        f"conserve, else to _BALANCE_EXEMPT with a reason): {sorted(unaccounted)}"
    )
    # And no stale entries pointing at deleted models.
    stale = accounted - on_disk
    assert not stale, f"_MODELS/_BALANCE_EXEMPT reference missing models: {sorted(stale)}"

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
    # The faithful base and all its structural variants carry the throttle; the
    # _balanced base and its variants revert it.
    if model.startswith("wats_sewer_khalil_paper") and "_balanced" not in model:
        excl |= _KHALIL_THROTTLE_RXNS
    return excl

# Generous absolute tolerance: published coefficients are rounded to 2-3 decimals
# (e.g. nitrate demand 0.175 vs the exact 0.1748), so true imbalances (order
# 0.1-1) are flagged while rounding (order 1e-3) is not.
_TOL = 1.0e-2


def _net(name):
    return aquakin.load_model_from_file(os.path.join(_NDIR, name + ".yaml"))


@pytest.mark.parametrize("model", _MODELS)
def test_cod_electron_balance(model):
    net = _net(model)
    excl = _cod_excluded(model)
    viol = [(r, q, v) for r, q, v in net.check_conservation(tol=_TOL,
                                                            quantities=["COD"])
            if r not in excl]
    assert not viol, f"{model} COD imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, _, v in viol)


def test_faithful_reproduces_khalil_nonconserving_throttle():
    """The faithful model reproduces Khalil's undocumented anoxic-VFA throttle,
    which does not conserve COD; the _balanced model reverts it and conserves.
    This documents the contrast (and guards it against regression)."""
    cod = {r: v for r, q, v in _net("wats_sewer_khalil_paper").check_conservation(
        tol=_TOL, quantities=["COD"])}
    for rxn in _KHALIL_THROTTLE_RXNS:
        assert rxn in cod, f"faithful model should NOT conserve COD on {rxn!r} " \
                           "(reproduces Khalil's throttle)"
    bal = [r for r, q, v in _net("wats_sewer_khalil_paper_balanced").check_conservation(
        tol=_TOL, quantities=["COD"]) if r not in _COD_EXCLUDED]
    assert not bal, f"balanced model COD imbalance: {bal}"


def test_yield_change_preserves_conservation():
    """Lowering the anoxic growth yield must not break conservation: the growth
    coefficients are derived from the COD/electron balance (S_B = -1/Y_H,
    S_NO = (Y_H-1)/(2.86 Y_H)), so they conserve for any Y_H. Guards the
    yield-robustness check (a lower WATS denitrifying yield Y_H=0.35 leaves the
    balanced model fully conserving and the faithful model's only COD imbalance
    the documented VFA throttle)."""
    import numpy as np
    for model, conserves_n in [("wats_sewer_khalil_paper_balanced", True),
                               ("wats_sewer_khalil_paper_directsulfate", False)]:
        net = _net(model)
        p = np.array(net.default_parameters())
        p[net.param_index["y_h"]] = 0.35
        excl = _cod_excluded(model)
        cod = [(r, v) for r, q, v in net.check_conservation(
            tol=_TOL, params=p, quantities=["COD"]) if r not in excl]
        S = net.check_conservation(tol=_TOL, params=p, quantities=["S"])
        assert not cod, f"{model} COD imbalance at Y_H=0.35: {cod}"
        assert not S, f"{model} S imbalance at Y_H=0.35: {S}"
        if conserves_n:
            assert not net.check_nitrogen(tol=_TOL, params=p), \
                f"{model} N imbalance at Y_H=0.35"


@pytest.mark.parametrize("model", _MODELS)
def test_sulfur_balance(model):
    net = _net(model)
    viol = net.check_conservation(tol=_TOL, quantities=["S"])
    assert not viol, f"{model} sulfur imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, _, v in viol)


@pytest.mark.parametrize("model", _MODELS)
def test_iron_balance(model):
    net = _net(model)
    viol = net.check_conservation(tol=_TOL, quantities=["Fe"])
    assert not viol, f"{model} iron imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, _, v in viol)


def test_balanced_model_conserves_nitrogen():
    """The mass/electron-balanced model conserves nitrogen (accounting for
    denitrification N2 gas) -- the full WATS N-content terms are restored. The
    faithful wats_sewer_khalil_paper deliberately omits N tracking (matching the
    published model), so it does NOT conserve N; that contrast is the point of
    the two-model pair."""
    balanced = _net("wats_sewer_khalil_paper_balanced")
    viol = balanced.check_nitrogen(tol=_TOL)
    assert not viol, "balanced model N imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, v in viol)

    # The faithful paper model does NOT track nitrogen -- assert the contrast so
    # the simplification is documented and regressions are caught.
    faithful = _net("wats_sewer_khalil_paper")
    assert faithful.check_nitrogen(tol=_TOL), \
        "faithful model unexpectedly conserves N (it should reproduce the " \
        "published model's N-simplification)"
