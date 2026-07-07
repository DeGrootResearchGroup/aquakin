"""Fast unit tests for the pure cores extracted from ``evaluate_bsm2``.

``bsm2_oci_terms`` (the single-source BSM2 OCI weighting) and
``_bypass_bod_correction`` (the influent-bypass BOD re-weighting) are pure
functions, so their weights, the net-heating clamp, and the load-weighting are
checked here on hand-built arrays -- no BSM2 solve. The full ``evaluate_bsm2``
wiring stays covered by the slow suite.
"""

import jax.numpy as jnp
import pytest

from aquakin.plant.bsm import bsm2_asm1_model
from aquakin.plant.bsm.evaluation import _bypass_bod_correction
from aquakin.plant.metrics import (
    _composition,
    bsm2_oci_terms,
    derived_BOD,
    operational_cost_index_bsm2,
)


def test_bsm2_oci_terms_are_the_single_weight_source():
    vals = dict(
        aeration=3784.2,
        pumping=1689.0,
        mixing=768.0,
        sludge_production=2280.5,
        carbon=800.0,
        methane=1010.3,
        heating=4200.0,
    )
    d = {k: (v, c) for k, v, c in bsm2_oci_terms(**vals)}
    # The Gernaey-2014 weights.
    assert d["sludge"][1] == pytest.approx(3.0 * 2280.5)
    assert d["carbon"][1] == pytest.approx(3.0 * 800.0)
    assert d["methane"][1] == pytest.approx(-6.0 * 1010.3)
    assert d["heating"][1] is None  # non-linear term; no direct contribution
    # net heating clamps to 0 here (HE - 7*methane < 0).
    assert d["net_heating"][0] == 0.0
    # The scalar OCI is exactly the sum of the itemized contributions.
    total = sum(c for _, _, c in bsm2_oci_terms(**vals) if c is not None)
    assert total == pytest.approx(operational_cost_index_bsm2(**vals))


def test_bsm2_oci_net_heating_positive_branch():
    # Large heating, small methane -> net heating positive (HE - 7*methane > 0).
    d = {k: (v, c) for k, v, c in bsm2_oci_terms(1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 200.0)}
    assert d["net_heating"][0] == pytest.approx(200.0 - 70.0)
    assert d["net_heating"][1] == pytest.approx(130.0)


def test_bypass_bod_correction_zero_and_positive_bypass():
    model = bsm2_asm1_model()
    _, _, f_P = _composition(model, model.default_parameters())
    Ct = model.concentrations({"SS": 2.0, "XS": 5.0, "XB_H": 1.0}, base="zero")  # treated
    Cb = model.concentrations({"SS": 60.0, "XS": 200.0, "XB_H": 30.0}, base="zero")  # raw bypass
    t = jnp.array([0.0, 1.0])
    Ct2 = jnp.tile(Ct, (2, 1))
    Cb2 = jnp.tile(Cb, (2, 1))
    Qt = jnp.array([1000.0, 1000.0])
    treated_bod = float(derived_BOD(Ct, model, f_P=f_P))
    eqi_flat = 5000.0

    # No bypass flow -> EQI unchanged; the BOD average is just the treated BOD.
    eqi0, bod0 = _bypass_bod_correction(t, eqi_flat, Qt, Ct2, jnp.zeros(2), Cb2, model, f_P)
    assert eqi0 == pytest.approx(eqi_flat)
    assert bod0 == pytest.approx(treated_bod)

    # A high-BOD bypass raises both the scored EQI and the reported BOD average.
    eqi1, bod1 = _bypass_bod_correction(
        t, eqi_flat, Qt, Ct2, jnp.array([200.0, 200.0]), Cb2, model, f_P
    )
    assert eqi1 > eqi_flat
    assert bod1 > treated_bod
