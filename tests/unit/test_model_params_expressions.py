"""Tests for model-level shared parameters and named rate expressions."""

import textwrap

import jax.numpy as jnp
import pytest

import aquakin


def _yaml(tmp_path, body: str):
    p = tmp_path / "net.yaml"
    p.write_text(textwrap.dedent(body))
    return p


# ---------- Model-level shared parameters ----------


def test_model_level_parameter_used_by_reaction(tmp_path):
    p = _yaml(tmp_path, """
        model: {name: shared, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
          - {name: B, default_concentration: 0.0}
        parameters:
          k: {value: 0.5, transform: positive_log}
        reactions:
          - name: r1
            rate: "k * [A]"
            stoichiometry: {A: -1, B: 1}
    """)
    net = aquakin.load_model_from_file(p)
    assert net.parameters == ["k"]  # bare name, no namespacing
    assert net.parameter_transforms["k"] == "positive_log"
    assert float(net.default_parameters()[0]) == 0.5


def test_model_param_shared_across_reactions(tmp_path):
    """The same k is used in two reactions; single parameter slot."""
    p = _yaml(tmp_path, """
        model: {name: shared2, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
          - {name: B, default_concentration: 0.0}
          - {name: C, default_concentration: 0.0}
        parameters:
          k: {value: 0.5}
        reactions:
          - name: r1
            rate: "k * [A]"
            stoichiometry: {A: -1, B: 1}
          - name: r2
            rate: "k * [B]"
            stoichiometry: {B: -1, C: 1}
    """)
    net = aquakin.load_model_from_file(p)
    assert net.parameters == ["k"]
    assert net.n_params == 1


def test_local_param_shadowing_global_rejected(tmp_path):
    p = _yaml(tmp_path, """
        model: {name: bad, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        parameters:
          k: {value: 0.5}
        reactions:
          - name: r1
            rate: "k * [A]"
            parameters:
              k: {value: 1.0}
            stoichiometry: {A: -1}
    """)
    with pytest.raises(ValueError):
        aquakin.load_model_from_file(p)


# ---------- Named rate expressions ----------


def test_expression_reference_inlined(tmp_path):
    """A reaction whose rate is just an expression name evaluates correctly."""
    p = _yaml(tmp_path, """
        model: {name: exp1, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
          - {name: B, default_concentration: 0.0}
        parameters:
          k: {value: 0.3}
        expressions:
          rho_decay: "k * [A]"
        reactions:
          - name: r1
            rate: "rho_decay"
            stoichiometry: {A: -1, B: 1}
    """)
    net = aquakin.load_model_from_file(p)
    rate = net.rates(
        jnp.asarray([2.0, 0.0]),
        net.default_parameters(),
        net.default_conditions().fields,
        0,
    )
    assert float(rate[0]) == pytest.approx(0.3 * 2.0)


def test_expression_referencing_another_expression(tmp_path):
    """Expression chain: rho2 references rho1."""
    p = _yaml(tmp_path, """
        model: {name: exp2, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
          - {name: B, default_concentration: 0.0}
        parameters:
          k: {value: 0.5}
          a: {value: 2.0}
        expressions:
          rho1: "k * [A]"
          rho2: "a * rho1"
        reactions:
          - name: r1
            rate: "rho2"
            stoichiometry: {A: -1, B: 1}
    """)
    net = aquakin.load_model_from_file(p)
    rate = net.rates(
        jnp.asarray([3.0, 0.0]),
        net.default_parameters(),
        net.default_conditions().fields,
        0,
    )
    assert float(rate[0]) == pytest.approx(2.0 * 0.5 * 3.0)


def test_expression_used_in_inline_formula(tmp_path):
    """A reaction's rate can mix an expression reference with other terms."""
    p = _yaml(tmp_path, """
        model: {name: exp3, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        parameters:
          k: {value: 1.0}
        expressions:
          rho: "k * [A]"
        reactions:
          - name: r1
            rate: "2.0 * rho + 1.0"
            stoichiometry: {A: -1}
    """)
    net = aquakin.load_model_from_file(p)
    rate = net.rates(
        jnp.asarray([3.0]),
        net.default_parameters(),
        net.default_conditions().fields,
        0,
    )
    assert float(rate[0]) == pytest.approx(2.0 * 1.0 * 3.0 + 1.0)


def test_expression_cycle_rejected(tmp_path):
    p = _yaml(tmp_path, """
        model: {name: cyc, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        parameters:
          k: {value: 1.0}
        expressions:
          a: "k * b"
          b: "k * a"
        reactions:
          - name: r1
            rate: "a"
            stoichiometry: {A: -1}
    """)
    with pytest.raises(ValueError, match="[Cc]ycle"):
        aquakin.load_model_from_file(p)


def test_expression_self_reference_rejected(tmp_path):
    p = _yaml(tmp_path, """
        model: {name: self_ref, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        parameters:
          k: {value: 1.0}
        expressions:
          rho: "k * rho"
        reactions:
          - name: r1
            rate: "rho"
            stoichiometry: {A: -1}
    """)
    with pytest.raises(ValueError):
        aquakin.load_model_from_file(p)


def test_expression_name_collides_with_global_param_rejected(tmp_path):
    p = _yaml(tmp_path, """
        model: {name: collide1, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        parameters:
          rho: {value: 1.0}
        expressions:
          rho: "rho * [A]"
        reactions:
          - name: r1
            rate: "rho"
            stoichiometry: {A: -1}
    """)
    with pytest.raises(ValueError, match="parameters and expressions"):
        aquakin.load_model_from_file(p)


def test_expression_name_collides_with_local_param_rejected(tmp_path):
    p = _yaml(tmp_path, """
        model: {name: collide2, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        expressions:
          rho: "1.0 * [A]"
        reactions:
          - name: r1
            rate: "rho * 2"
            parameters:
              rho: {value: 0.5}
            stoichiometry: {A: -1}
    """)
    with pytest.raises(ValueError, match="collide with named expression"):
        aquakin.load_model_from_file(p)


def test_unknown_identifier_after_resolution_errors(tmp_path):
    p = _yaml(tmp_path, """
        model: {name: unknown, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        reactions:
          - name: r1
            rate: "undefined_thing * [A]"
            stoichiometry: {A: -1}
    """)
    with pytest.raises((KeyError, ValueError)):
        aquakin.load_model_from_file(p)
