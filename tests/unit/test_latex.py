"""LaTeX renderer unit tests."""

import pytest

from aquakin.core.nodes import (
    AddNode,
    ArrheniusNode,
    ConditionNode,
    ConstantNode,
    DivideNode,
    MultiplyNode,
    NegateNode,
    ParamNode,
    PowerNode,
    SpeciesNode,
    SubtractNode,
    pHSwitchNode,
)
from aquakin.utils.latex import to_latex


def test_constant():
    assert to_latex(ConstantNode(3.14)) == "3.14"


def test_species_charge_rendering():
    assert to_latex(SpeciesNode("Br-")) == r"[\mathrm{Br^{-}}]"
    assert to_latex(SpeciesNode("BrO3-")) == r"[\mathrm{BrO3^{-}}]"
    assert to_latex(SpeciesNode("H+")) == r"[\mathrm{H^{+}}]"


def test_param_rendering():
    assert to_latex(ParamNode("k1")) == "k1"


def test_condition_pH_special_cased():
    assert to_latex(ConditionNode("pH")) == r"\mathrm{pH}"


def test_condition_T_special_cased():
    assert to_latex(ConditionNode("T")) == "T"


def test_condition_other_rendered_italic():
    out = to_latex(ConditionNode("fluence_rate"))
    assert out.startswith(r"\mathit{")
    # The underscore is escaped so the name renders literally (not as a subscript).
    assert r"fluence\_rate" in out


def test_negate():
    assert to_latex(NegateNode(ConstantNode(2.0))) == "-2"


def test_add_subtract():
    assert to_latex(AddNode(ConstantNode(1), ConstantNode(2))) == "1 + 2"
    assert to_latex(SubtractNode(ConstantNode(1), ConstantNode(2))) == "1 - 2"


def test_multiply_uses_cdot():
    out = to_latex(MultiplyNode(ParamNode("k"), SpeciesNode("A")))
    assert r"\cdot" in out


def test_divide_uses_frac():
    out = to_latex(DivideNode(ParamNode("a"), ParamNode("b")))
    assert out.startswith(r"\frac")


def test_power():
    out = to_latex(PowerNode(ConstantNode(2), ConstantNode(3)))
    assert out == "2^{3}"


def test_arrhenius():
    out = to_latex(ArrheniusNode(ParamNode("A"), ParamNode("Ea")))
    assert r"\exp" in out
    assert "R" in out and "T" in out


def test_pH_switch():
    out = to_latex(pHSwitchNode(ParamNode("pKa")))
    assert r"\frac" in out
    assert "10^" in out


def test_multiply_with_add_left_parenthesises():
    out = to_latex(MultiplyNode(AddNode(ConstantNode(1), ConstantNode(2)), ParamNode("k")))
    assert r"\left(" in out and r"\right)" in out


def test_species_underscore_escaped():
    """Underscore-delimited names (S_NO, X_BH) must escape the underscore so it
    renders literally rather than as a LaTeX subscript operator."""
    assert to_latex(SpeciesNode("S_NO")) == r"[\mathrm{S\_NO}]"
    assert to_latex(SpeciesNode("X_BH")) == r"[\mathrm{X\_BH}]"


def test_power_atomic_base_not_parenthesised():
    # Species / param / non-negative constant bases need no parentheses.
    assert to_latex(PowerNode(SpeciesNode("A"), ConstantNode(2))) == r"[\mathrm{A}]^{2}"
    assert to_latex(PowerNode(ParamNode("k"), ConstantNode(2))) == "k^{2}"


def test_power_parenthesises_multiply_base():
    """(k * [A]) ** 2 must wrap the product so the exponent binds to all of it."""
    base = MultiplyNode(ParamNode("k"), SpeciesNode("A"))
    out = to_latex(PowerNode(base, ConstantNode(2)))
    assert out == r"\left(k \cdot [\mathrm{A}]\right)^{2}"


def test_power_parenthesises_negate_base():
    """(-x) ** 2 must wrap the negation so it is not read as -(x^2)."""
    out = to_latex(PowerNode(NegateNode(ParamNode("x")), ConstantNode(2)))
    assert out == r"\left(-x\right)^{2}"


def test_power_parenthesises_negative_constant_base():
    out = to_latex(PowerNode(ConstantNode(-3.0), ConstantNode(2)))
    assert out == r"\left(-3\right)^{2}"


def test_power_parenthesises_divide_base():
    base = DivideNode(ParamNode("a"), ParamNode("b"))
    out = to_latex(PowerNode(base, ConstantNode(2)))
    assert out.startswith(r"\left(\frac") and out.endswith(r"\right)^{2}")


def test_unknown_node_raises():
    class _Bogus:
        pass

    with pytest.raises(TypeError):
        to_latex(_Bogus())
