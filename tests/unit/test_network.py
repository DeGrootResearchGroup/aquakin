"""CompiledNetwork unit tests."""

import jax.numpy as jnp
import pytest

import aquakin


def test_stoich_matrix_shape(simple_network):
    assert simple_network.stoich_matrix.shape == (1, 2)
    # A -> B with stoich A:-1, B:+1
    a_idx = simple_network.species_index["A"]
    b_idx = simple_network.species_index["B"]
    assert float(simple_network.stoich_matrix[0, a_idx]) == -1.0
    assert float(simple_network.stoich_matrix[0, b_idx]) == 1.0


def test_default_arrays(simple_network):
    C0 = simple_network.default_concentrations()
    assert C0.shape == (2,)
    p0 = simple_network.default_parameters()
    assert p0.shape == (1,)
    assert float(p0[0]) == pytest.approx(0.1)


def test_namespaced_param_keys(simple_network):
    assert "A_to_B.k" in simple_network.param_index
    assert simple_network.parameters == ["A_to_B.k"]


def test_rates_signature(simple_network):
    C = jnp.asarray([1.0, 0.0])
    params = simple_network.default_parameters()
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    r = simple_network.rates(C, params, conditions.fields, 0)
    assert r.shape == (1,)
    assert float(r[0]) == pytest.approx(0.1)


def test_dCdt(simple_network):
    C = jnp.asarray([1.0, 0.0])
    params = simple_network.default_parameters()
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    rhs = simple_network.dCdt(C, params, conditions.fields, 0)
    # dA/dt = -0.1, dB/dt = +0.1
    assert float(rhs[0]) == pytest.approx(-0.1)
    assert float(rhs[1]) == pytest.approx(0.1)


def test_summary_smoke(simple_network):
    out = simple_network.summary()
    assert "simple_decay" in out
    assert "A_to_B" in out


def test_to_latex_smoke(simple_network):
    latex = simple_network.to_latex()
    assert "A_to_B" in latex
    assert "mathrm" in latex["A_to_B"]
