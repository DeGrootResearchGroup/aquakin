"""Parser unit tests."""

import jax.numpy as jnp
import pytest

from aquakin.core.context import CompileContext
from aquakin.core.nodes import (
    AddNode,
    ArrheniusNode,
    ConditionNode,
    ConstantNode,
    MultiplyNode,
    NegateNode,
    ParamNode,
    PowerNode,
    SpeciesNode,
    SubtractNode,
    pHSwitchNode,
)
from aquakin.core.parser import ParseError, parse_rate_expression


def test_number_literal():
    tree = parse_rate_expression("1.5")
    assert isinstance(tree, ConstantNode)
    assert tree.value == 1.5


def test_scientific_notation():
    tree = parse_rate_expression("1.6e2")
    assert isinstance(tree, ConstantNode)
    assert tree.value == 160.0


def test_species_with_charge():
    tree = parse_rate_expression("[Br-]")
    assert isinstance(tree, SpeciesNode)
    assert tree.name == "Br-"


def test_species_with_digits_and_charge():
    tree = parse_rate_expression("[BrO3-]")
    assert isinstance(tree, SpeciesNode)
    assert tree.name == "BrO3-"


def test_param_reference():
    tree = parse_rate_expression("k1")
    assert isinstance(tree, ParamNode)
    assert tree.name == "k1"


def test_simple_product():
    tree = parse_rate_expression("k1 * [O3] * [Br-]")
    assert isinstance(tree, MultiplyNode)


def test_operator_precedence():
    # 1 + 2 * 3 must parse as 1 + (2*3)
    tree = parse_rate_expression("1 + 2 * 3")
    assert isinstance(tree, AddNode)
    assert isinstance(tree.right, MultiplyNode)


def test_power_right_associative():
    tree = parse_rate_expression("2 ** 3 ** 2")
    assert isinstance(tree, PowerNode)
    assert isinstance(tree.right, PowerNode)  # right-assoc


def test_unary_minus():
    tree = parse_rate_expression("-k * [A]")
    assert isinstance(tree, MultiplyNode)
    assert isinstance(tree.left, NegateNode)


def test_parenthesised_expression():
    tree = parse_rate_expression("(1 + 2) * 3")
    assert isinstance(tree, MultiplyNode)
    assert isinstance(tree.left, AddNode)


def test_subtract():
    tree = parse_rate_expression("k - 1")
    assert isinstance(tree, SubtractNode)


def test_condition_reference_simple():
    tree = parse_rate_expression("{pH}")
    assert isinstance(tree, ConditionNode)
    assert tree.field_name == "pH"


def test_condition_reference_in_expression():
    tree = parse_rate_expression("k * [O3] * 10 ** ({pH} - 14)")
    assert isinstance(tree, MultiplyNode)


def test_condition_reference_underscore_name():
    tree = parse_rate_expression("{fluence_rate}")
    assert isinstance(tree, ConditionNode)
    assert tree.field_name == "fluence_rate"


def test_unclosed_condition_brace():
    with pytest.raises(ParseError):
        parse_rate_expression("{pH")


def test_empty_condition_brace():
    with pytest.raises(ParseError):
        parse_rate_expression("{}")


def test_arrhenius_function():
    tree = parse_rate_expression("arrhenius(A, Ea)")
    assert isinstance(tree, ArrheniusNode)


def test_pH_switch_function():
    tree = parse_rate_expression("pH_switch(pKa)")
    assert isinstance(tree, pHSwitchNode)


def test_monod_function():
    from aquakin.core.nodes import MonodNode
    tree = parse_rate_expression("monod([SS], KS)")
    assert isinstance(tree, MonodNode)


def test_monod_inh_function():
    from aquakin.core.nodes import MonodInhibitionNode
    tree = parse_rate_expression("monod_inh([SO], KOH)")
    assert isinstance(tree, MonodInhibitionNode)


def test_monod_ratio_function():
    from aquakin.core.nodes import MonodRatioNode
    tree = parse_rate_expression("monod_ratio([XS], [XH], KX)")
    assert isinstance(tree, MonodRatioNode)


def test_monod_wrong_arity():
    with pytest.raises(ParseError):
        parse_rate_expression("monod([SS])")
    with pytest.raises(ParseError):
        parse_rate_expression("monod_ratio([A], [B])")


def test_unknown_function_rejected():
    with pytest.raises(ParseError):
        parse_rate_expression("foo(1, 2)")


def test_unexpected_character():
    with pytest.raises(ParseError):
        parse_rate_expression("k * @")


def test_unbalanced_parens():
    with pytest.raises(ParseError):
        parse_rate_expression("(k * [A]")


def test_trailing_tokens():
    with pytest.raises(ParseError):
        parse_rate_expression("k * [A] foo")


def test_full_expression_compiles_and_evaluates():
    """End-to-end compile + evaluate of a parsed expression."""
    tree = parse_rate_expression("k1 * [A] + 2.0")
    ctx = CompileContext(
        species_index={"A": 0},
        param_index={"rxn.k1": 0},
        condition_fields=frozenset(),
        reaction_name="rxn",
    )
    fn = tree.compile(ctx)
    C = jnp.asarray([3.0])
    params = jnp.asarray([5.0])
    result = float(fn(C, params, {}, 0))
    assert result == pytest.approx(5.0 * 3.0 + 2.0)
