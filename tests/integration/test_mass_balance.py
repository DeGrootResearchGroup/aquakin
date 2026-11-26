"""Mass / electron (COD) conservation tests for the WATS sewer model.

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

The composition table is no longer a literal dict here: the WATS model
declares its own per-species ``composition:`` in the YAML, read back through
``model.composition()`` and checked with ``model.check_conservation()`` /
``model.check_nitrogen()`` (the conservation table is a first-class model
property, so a value can no longer drift out of sync with the model it describes).
"""

import os

import pytest

import aquakin

_NDIR = os.path.join(os.path.dirname(aquakin.__file__), "models")

_MODELS = ["wats_sewer"]


def test_every_wats_model_is_checked():
    """Completeness guard: every shipped ``wats_sewer*.yaml`` is under the
    conservation checks (``_MODELS``), so a new sewer model cannot be added
    without them, and ``_MODELS`` names no model that is gone."""
    import glob
    on_disk = {os.path.basename(p)[:-len(".yaml")]
               for p in glob.glob(os.path.join(_NDIR, "wats_sewer*.yaml"))}
    assert on_disk == set(_MODELS)


# Nitrification / autotroph decay oxidise nitrogen (NH3 -> NO3), which is not
# part of the carbon/sulfur COD continuity (its electron transfer is carried by
# the explicit O2 term and balanced in the nitrogen continuity instead). They
# are excluded from the COD check only.
_COD_EXCLUDED = {"nitrification", "nitrifier_decay"}

# Generous absolute tolerance: published coefficients are rounded to 2-3 decimals
# (e.g. nitrate demand 0.175 vs the exact 0.1748), so true imbalances (order
# 0.1-1) are flagged while rounding (order 1e-3) is not.
_TOL = 1.0e-2


def _net(name):
    return aquakin.load_model_from_file(os.path.join(_NDIR, name + ".yaml"))


@pytest.mark.parametrize("model", _MODELS)
def test_cod_electron_balance(model):
    net = _net(model)
    viol = [(r, q, v) for r, q, v in net.check_conservation(tol=_TOL,
                                                            quantities=["COD"])
            if r not in _COD_EXCLUDED]
    assert not viol, f"{model} COD imbalance: " + "; ".join(
        f"{r} {v:+.3f}" for r, _, v in viol)


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
