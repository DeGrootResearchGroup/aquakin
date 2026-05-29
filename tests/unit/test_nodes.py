"""ASTNode unit tests."""

import jax
import jax.numpy as jnp
import pytest

from aquakin.core.context import CompileContext
from aquakin.core.nodes import (
    AddNode,
    ArrheniusNode,
    ConstantNode,
    ConditionNode,
    DivideNode,
    MultiplyNode,
    NegateNode,
    ParamNode,
    PowerNode,
    SpeciesNode,
    SubtractNode,
    pHSwitchNode,
    GAS_CONSTANT,
)


def _ctx(**overrides):
    base = dict(
        species_index={"A": 0, "B": 1},
        param_index={"r.k": 0, "r.A": 1, "r.Ea": 2, "r.pKa": 3},
        condition_fields=frozenset({"pH", "T"}),
        reaction_name="r",
    )
    base.update(overrides)
    return CompileContext(**base)


def _eval(node, C=(1.0, 2.0), params=(0.5, 1.0, 5000.0, 8.8), pH=7.5, T=293.15):
    fn = node.compile(_ctx())
    return float(
        fn(
            jnp.asarray(C),
            jnp.asarray(params),
            {"pH": jnp.asarray([pH]), "T": jnp.asarray([T])},
            0,
        )
    )


def test_constant():
    assert _eval(ConstantNode(3.14)) == pytest.approx(3.14)


def test_species_lookup():
    assert _eval(SpeciesNode("A")) == pytest.approx(1.0)
    assert _eval(SpeciesNode("B")) == pytest.approx(2.0)


def test_species_unknown_raises():
    with pytest.raises(KeyError):
        SpeciesNode("X").compile(_ctx())


def test_param_lookup_namespaces():
    assert _eval(ParamNode("k")) == pytest.approx(0.5)


def test_param_unknown_raises():
    with pytest.raises(KeyError):
        ParamNode("missing").compile(_ctx())


def test_condition_lookup():
    assert _eval(ConditionNode("pH")) == pytest.approx(7.5)
    assert _eval(ConditionNode("T")) == pytest.approx(293.15)


def test_condition_unknown_raises():
    with pytest.raises(KeyError):
        ConditionNode("salinity").compile(_ctx())


def test_binary_ops():
    node = AddNode(ConstantNode(2.0), SubtractNode(ConstantNode(5.0), ConstantNode(1.0)))
    assert _eval(node) == pytest.approx(6.0)

    node = MultiplyNode(ConstantNode(3.0), DivideNode(ConstantNode(8.0), ConstantNode(4.0)))
    assert _eval(node) == pytest.approx(6.0)

    node = PowerNode(ConstantNode(2.0), ConstantNode(3.0))
    assert _eval(node) == pytest.approx(8.0)


def test_negate():
    assert _eval(NegateNode(ConstantNode(2.5))) == pytest.approx(-2.5)


def test_arrhenius():
    node = ArrheniusNode(ParamNode("A"), ParamNode("Ea"))
    # A=1.0, Ea=5000, T=293.15
    expected = 1.0 * jnp.exp(-5000.0 / (GAS_CONSTANT * 293.15))
    assert _eval(node) == pytest.approx(float(expected))


def test_pH_switch():
    node = pHSwitchNode(ParamNode("pKa"))
    # pKa=8.8, pH=7.5
    expected = 1.0 / (1.0 + 10.0 ** (7.5 - 8.8))
    assert _eval(node) == pytest.approx(float(expected))


def test_arrhenius_requires_T():
    ctx = _ctx(condition_fields=frozenset({"pH"}))
    with pytest.raises(KeyError):
        ArrheniusNode(ParamNode("A"), ParamNode("Ea")).compile(ctx)


def test_monod_evaluates_correctly():
    from aquakin.core.nodes import MonodNode

    node = MonodNode(SpeciesNode("A"), ConstantNode(1.0))
    # A = 1.0, K = 1.0 -> 1/(1+1) = 0.5
    assert _eval(node) == pytest.approx(0.5)


def test_monod_inhibition_evaluates_correctly():
    from aquakin.core.nodes import MonodInhibitionNode

    node = MonodInhibitionNode(SpeciesNode("A"), ConstantNode(1.0))
    # A = 1.0, K = 1.0 -> 1/(1+1) = 0.5; same as Monod at A=K.
    assert _eval(node) == pytest.approx(0.5)


def test_monod_plus_inhibition_equals_one():
    """A property of the Monod / inhibition pair: M(X,K) + M_inh(X,K) = 1."""
    from aquakin.core.nodes import MonodInhibitionNode, MonodNode

    A_node = SpeciesNode("A")
    K_node = ConstantNode(0.7)
    m = _eval(MonodNode(A_node, K_node))
    m_inh = _eval(MonodInhibitionNode(A_node, K_node))
    assert m + m_inh == pytest.approx(1.0)


def test_monod_ratio_evaluates_correctly():
    from aquakin.core.nodes import MonodRatioNode

    # (A/B) / (K + A/B) with A=2, B=1, K=1 -> 2/(1+2) = 2/3
    node = MonodRatioNode(SpeciesNode("A"), SpeciesNode("B"), ConstantNode(1.0))
    ctx = _ctx()
    fn = node.compile(ctx)
    result = float(
        fn(
            jnp.asarray([2.0, 1.0]),
            jnp.asarray([0.5, 1.0, 5000.0, 8.8]),
            {"pH": jnp.asarray([7.5]), "T": jnp.asarray([293.15])},
            0,
        )
    )
    assert result == pytest.approx(2.0 / 3.0)


def test_monod_ratio_handles_zero_denominator():
    """A=0, B=0 -> 0/0 mathematically. Our form A/(K*B+A) returns 0/0 = NaN
    only when both are exactly zero; in practice solvers don't visit that
    point. Test the well-defined case A=0, B>0 -> 0."""
    from aquakin.core.nodes import MonodRatioNode

    node = MonodRatioNode(ConstantNode(0.0), ConstantNode(1.0), ConstantNode(0.5))
    assert _eval(node) == pytest.approx(0.0)


def test_grad_through_params():
    # d/d(k) of k * [A] = [A]
    node = MultiplyNode(ParamNode("k"), SpeciesNode("A"))
    fn = node.compile(_ctx())

    def f(p):
        return fn(
            jnp.asarray([1.0, 2.0]),
            p,
            {"pH": jnp.asarray([7.5]), "T": jnp.asarray([293.15])},
            0,
        )

    g = jax.grad(f)(jnp.asarray([0.5, 1.0, 5000.0, 8.8]))
    assert float(g[0]) == pytest.approx(1.0)
