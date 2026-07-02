"""Tests for the Arrhenius-style parameter temperature correction."""

import math

import jax
import jax.numpy as jnp
import pytest

import aquakin


_TNET = """
model:
  name: ttest
  version: "1.0"
  description: "temperature correction test"
species:
  - {name: A, units: mol/L, default_concentration: 1.0}
  - {name: B, units: mol/L, default_concentration: 0.0}
conditions:
  - {name: T, description: "temperature K", default: 293.15}
parameters:
  k:
    value: 2.0
    temperature: {theta: 1.1, ref_T: 293.15}
reactions:
  - name: decay
    rate: "k * [A]"
    parameters: {}
    stoichiometry: {A: -1, B: 1}
"""


def _load(text, tmp_path, name="t.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return aquakin.load_model_from_file(str(p))


def _rate(net, T):
    C = net.default_concentrations()
    p = net.default_parameters()
    return float(net.rates(C, p, {"T": jnp.array([float(T)])}, 0)[0])


def test_unity_at_reference_temperature(tmp_path):
    net = _load(_TNET, tmp_path)
    assert len(net.temperature_corrections) == 1
    # At ref_T the correction is exactly 1: rate = k * [A] = 2.0.
    assert _rate(net, 293.15) == pytest.approx(2.0, rel=1e-12)


def test_arrhenius_scaling(tmp_path):
    net = _load(_TNET, tmp_path)
    assert _rate(net, 303.15) == pytest.approx(2.0 * 1.1 ** 10, rel=1e-9)
    assert _rate(net, 283.15) == pytest.approx(2.0 * 1.1 ** -10, rel=1e-9)


def test_correction_is_differentiable(tmp_path):
    net = _load(_TNET, tmp_path)
    C = net.default_concentrations()
    p = net.default_parameters()
    g = jax.grad(lambda T: net.rates(C, p, {"T": jnp.array([T])}, 0)[0])(300.0)
    assert jnp.isfinite(g)
    # d/dT [k*theta^(T-ref)] = rate * ln(theta); check the analytic value.
    expected = _rate(net, 300.0) * math.log(1.1)
    assert float(g) == pytest.approx(expected, rel=1e-6)


def test_stoichiometry_untouched_by_temperature(tmp_path):
    """The correction is confined to rate constants; compute_stoich is unaffected
    by temperature (it has no T argument and uses the raw parameters)."""
    net = _load(_TNET, tmp_path)
    s = net.compute_stoich(net.default_parameters())
    assert s.shape == (1, 2)
    assert float(s[0, net.species_index["A"]]) == pytest.approx(-1.0)


def test_negative_theta_rejected(tmp_path):
    bad = _TNET.replace("theta: 1.1", "theta: -1.0")
    with pytest.raises(Exception):
        _load(bad, tmp_path, name="bad.yaml")


def test_model_without_temperature_unaffected():
    """A model with no temperature specs has no corrections (back-compat)."""
    net = aquakin.load_model_from_file  # sanity: shipped simple model
    from importlib.resources import files
    # The fixtures model has no temperature spec.
    import aquakin as ak
    n = ak.load_model("uv_h2o2")
    assert n.temperature_corrections == []


# --- ASM1 (shipped model) -------------------------------------------------

def test_asm1_unchanged_at_reference():
    """At the default 20 °C condition the ASM1 rates are identical to the
    uncorrected values (the correction is unity)."""
    net = aquakin.load_model("asm1")
    assert len(net.temperature_corrections) == 6
    C = net.concentrations(SS=10.0, SO=2.0, XB_H=2000.0)
    p = net.default_parameters()
    r = net.rates(C, p, {"T": jnp.array([293.15])}, 0)
    # Recompute the heterotroph aerobic growth by hand at 20 °C (theta^0 = 1).
    muH = float(p[net.parameters.index("muH")])
    KS = float(p[net.parameters.index("KS")])
    KOH = float(p[net.parameters.index("KOH")])
    # Textbook ASM1 heterotroph growth: no ammonia (N-source) availability switch.
    expect = (muH * 10.0 / (KS + 10.0) * 2.0 / (KOH + 2.0) * 2000.0)
    assert float(r[0]) == pytest.approx(expect, rel=1e-9)


def test_asm1_nitrification_slows_in_the_cold():
    """Autotroph growth (nitrification) is the most temperature-sensitive: at
    10 °C it drops to ~36% of the 20 °C rate (theta_muA^-10)."""
    net = aquakin.load_model("asm1")
    C = net.concentrations(SO=2.0, SNH=5.0, XB_A=150.0)
    p = net.default_parameters()
    i = net.reaction_names.index("aerobic_growth_autotrophs")
    r20 = float(net.rates(C, p, {"T": jnp.array([293.15])}, 0)[i])
    r10 = float(net.rates(C, p, {"T": jnp.array([283.15])}, 0)[i])
    assert r10 / r20 == pytest.approx((0.3 / 0.5) ** 2, rel=1e-4)  # 0.36
