"""Render a rate-expression AST as a LaTeX string."""

from __future__ import annotations

from aquakin.core.nodes import (
    AddNode,
    ArrheniusNode,
    ASTNode,
    ConstantNode,
    ConditionNode,
    DivideNode,
    MonodInhibitionNode,
    MonodInhibitionRatioNode,
    MonodNode,
    MonodRatioNode,
    MultiplyNode,
    NegateNode,
    ParamNode,
    PowerNode,
    SpeciesNode,
    SubtractNode,
    pHSwitchNode,
)


def _species_latex(name: str) -> str:
    """Render a species name as ``[\\mathrm{...}]``, escaping charges."""
    escaped = name.replace("+", "^{+}").replace("-", "^{-}")
    return r"[\mathrm{" + escaped + r"}]"


def _paren(inner: str) -> str:
    return rf"\left({inner}\right)"


def to_latex(node: ASTNode) -> str:
    """Render a rate-expression AST node as a LaTeX math string."""
    if isinstance(node, ConstantNode):
        return f"{node.value:g}"
    if isinstance(node, SpeciesNode):
        return _species_latex(node.name)
    if isinstance(node, ParamNode):
        return node.name
    if isinstance(node, ConditionNode):
        if node.field_name == "pH":
            return r"\mathrm{pH}"
        if node.field_name == "T":
            return "T"
        return rf"\mathit{{{node.field_name}}}"
    if isinstance(node, NegateNode):
        return f"-{to_latex(node.operand)}"
    if isinstance(node, AddNode):
        return f"{to_latex(node.left)} + {to_latex(node.right)}"
    if isinstance(node, SubtractNode):
        return f"{to_latex(node.left)} - {to_latex(node.right)}"
    if isinstance(node, MultiplyNode):
        return rf"{_maybe_paren_addsub(node.left)} \cdot {_maybe_paren_addsub(node.right)}"
    if isinstance(node, DivideNode):
        return rf"\frac{{{to_latex(node.left)}}}{{{to_latex(node.right)}}}"
    if isinstance(node, PowerNode):
        return f"{_maybe_paren_addsub(node.left)}^{{{to_latex(node.right)}}}"
    if isinstance(node, ArrheniusNode):
        A = to_latex(node.A)
        Ea = to_latex(node.Ea)
        return rf"{A} \exp\!\left(-\frac{{{Ea}}}{{R\,T}}\right)"
    if isinstance(node, pHSwitchNode):
        pKa = to_latex(node.pKa)
        return rf"\frac{{1}}{{1 + 10^{{\mathrm{{pH}} - {pKa}}}}}"
    if isinstance(node, MonodNode):
        return rf"\frac{{{to_latex(node.X)}}}{{{to_latex(node.K)} + {to_latex(node.X)}}}"
    if isinstance(node, MonodInhibitionNode):
        return rf"\frac{{{to_latex(node.K)}}}{{{to_latex(node.K)} + {to_latex(node.X)}}}"
    if isinstance(node, MonodRatioNode):
        ratio = rf"{to_latex(node.A)}/{to_latex(node.B)}"
        return rf"\frac{{{ratio}}}{{{to_latex(node.K)} + {ratio}}}"
    if isinstance(node, MonodInhibitionRatioNode):
        ratio = rf"{to_latex(node.A)}/{to_latex(node.B)}"
        return rf"\frac{{{to_latex(node.K)}}}{{{to_latex(node.K)} + {ratio}}}"
    raise TypeError(f"No LaTeX renderer for node type {type(node).__name__}")


def _maybe_paren_addsub(node: ASTNode) -> str:
    """Wrap +/- subtrees in parentheses so multiplication renders correctly."""
    rendered = to_latex(node)
    if isinstance(node, (AddNode, SubtractNode)):
        return _paren(rendered)
    return rendered
