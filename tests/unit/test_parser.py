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


@pytest.mark.parametrize("expr", ["1e", "1e+", "2.5e", "3E-"])
def test_malformed_exponent_is_parse_error(expr):
    """A number with an exponent marker but no exponent digits must fail as a
    positioned ParseError, not a bare float() ValueError from the primary."""
    with pytest.raises(ParseError):
        parse_rate_expression(expr)


@pytest.mark.parametrize("expr", ["1e3", "1.6e2", "1.5e-2", "2E+3"])
def test_valid_exponent_still_parses(expr):
    assert parse_rate_expression(expr) is not None


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


def test_safe_div_function():
    from aquakin.core.nodes import SafeDivideNode, SpeciesNode

    tree = parse_rate_expression("safe_div([S_va], [S_va] + [S_bu])")
    assert isinstance(tree, SafeDivideNode)
    assert isinstance(tree.num, SpeciesNode) and tree.num.name == "S_va"


def test_monod_wrong_arity():
    with pytest.raises(ParseError):
        parse_rate_expression("monod([SS])")
    with pytest.raises(ParseError):
        parse_rate_expression("monod_ratio([A], [B])")
    with pytest.raises(ParseError):
        parse_rate_expression("safe_div([A])")


def test_unknown_function_rejected():
    with pytest.raises(ParseError):
        parse_rate_expression("foo(1, 2)")


def test_unknown_function_message_lists_all_builtins():
    """The 'unknown function' message is derived from the registry, so it lists
    every built-in -- including pH_inhibit, which the old hand-written message
    omitted."""
    with pytest.raises(ParseError) as exc:
        parse_rate_expression("phinhibit(a)")
    msg = str(exc.value)
    for fn in ("arrhenius", "pH_switch", "pH_inhibit", "monod",
               "monod_inh", "monod_ratio", "monod_inh_ratio"):
        assert fn in msg


def test_pH_inhibit_wrong_arity():
    with pytest.raises(ParseError, match="pH_inhibit.. takes 2"):
        parse_rate_expression("pH_inhibit(a)")


def test_unexpected_character():
    with pytest.raises(ParseError):
        parse_rate_expression("k * @")


def test_unbalanced_parens():
    with pytest.raises(ParseError):
        parse_rate_expression("(k * [A]")


def test_trailing_tokens():
    with pytest.raises(ParseError):
        parse_rate_expression("k * [A] foo")


@pytest.mark.parametrize("expr", ["", "   ", "\t \n"])
def test_empty_or_whitespace_expression_is_parse_error(expr):
    """An empty or whitespace-only expression has no primary, so the parser hits
    EOF in `_primary` and raises rather than returning None."""
    with pytest.raises(ParseError):
        parse_rate_expression(expr)


@pytest.mark.parametrize("expr", ["[]", "[3A]", "[Br -]"])
def test_malformed_species_bracket_is_parse_error(expr):
    """`[` must be followed by an identifier species name: empty brackets and a
    digit-leading name fail; whitespace splitting a charge suffix (`[Br -]`) ends
    the name early and leaves a stray `-]` that fails to close the bracket. (The
    `{}` counterpart is already covered.)"""
    with pytest.raises(ParseError):
        parse_rate_expression(expr)


@pytest.mark.parametrize("expr", ["{3}", "{a b}", "{1pH}"])
def test_malformed_condition_brace_is_parse_error(expr):
    """`{` must wrap exactly one identifier: a number, two idents, or a digit-led
    name all fail."""
    with pytest.raises(ParseError):
        parse_rate_expression(expr)


@pytest.mark.parametrize("expr", ["monod([SS], KS,)", "max()", "max(1,)"])
def test_bad_arglist_is_parse_error(expr):
    """A trailing comma or an empty argument list leaves a primary with no
    expression to parse, which raises (before the arity check is even reached)."""
    with pytest.raises(ParseError):
        parse_rate_expression(expr)


@pytest.mark.parametrize("expr", ["1 2", "1.2.3", "[A] [B]"])
def test_adjacent_primaries_without_operator_is_parse_error(expr):
    """Two primaries with no operator between them leave trailing tokens after the
    first expression parses, which `parse()` rejects."""
    with pytest.raises(ParseError):
        parse_rate_expression(expr)


def test_max_function_parses():
    """`max(a, b)` is a built-in but had no parse test (only an eval test)."""
    from aquakin.core.nodes import MaxNode
    tree = parse_rate_expression("max(0, [A] - 3.0)")
    assert isinstance(tree, MaxNode)


def test_monod_inh_ratio_function_parses():
    """`monod_inh_ratio(A, B, K)` is a built-in but had no direct parse test."""
    from aquakin.core.nodes import MonodInhibitionRatioNode
    tree = parse_rate_expression("monod_inh_ratio([XPHA], [XPAO], KPHA)")
    assert isinstance(tree, MonodInhibitionRatioNode)


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
