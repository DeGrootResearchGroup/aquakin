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


def test_compile_stage_helpers():
    """The compile_network stage helpers behave as a pipeline: parameter index
    is network-level-then-reaction-local in order, and _unresolved_params flags
    only names resolving to neither scope."""
    from aquakin.core.network import _build_param_index, _unresolved_params
    from aquakin.core.parser import parse_rate_expression

    # ASM1 has both network-level (Y_H, ...) and reaction-local params.
    net = aquakin.load_network("asm1")
    # Network-level params come before reaction-local (which are dotted).
    bare = [p for p in net.parameters if "." not in p]
    dotted = [p for p in net.parameters if "." in p]
    if bare and dotted:
        assert net.parameters.index(bare[-1]) < net.parameters.index(dotted[0])

    pidx = {"k": 0, "r1.kf": 1}
    ast = parse_rate_expression("k * kf / nope")
    # 'k' resolves network-level; 'r1.kf' resolves reaction-local; 'nope' doesn't.
    # (order is unspecified -- param_names() is a set -- so compare as sets.)
    assert set(_unresolved_params(ast, "r1", pidx)) == {"nope"}
    assert set(_unresolved_params(ast, "r2", pidx)) == {"kf", "nope"}  # r2.kf absent
