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
    assert "fluence_rate" in to_latex(ConditionNode("fluence_rate"))


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


def test_unknown_node_raises():
    class _Bogus:
        pass

    with pytest.raises(TypeError):
        to_latex(_Bogus())
