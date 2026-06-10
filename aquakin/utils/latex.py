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
    pHInhibitNode,
    pHSwitchNode,
)


# LaTeX text-mode special characters that must be escaped when a species or
# condition name is rendered literally (e.g. the underscore in ``S_NO``, which
# would otherwise be read as a subscript operator and mis-render the name).
_LATEX_ESCAPES = {
    "_": r"\_", "#": r"\#", "%": r"\%", "&": r"\&",
    "$": r"\$", "{": r"\{", "}": r"\}",
}


def _escape_latex(name: str) -> str:
    """Escape LaTeX text-mode specials so an identifier renders literally."""
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in name)


def _species_latex(name: str) -> str:
    """Render a species name as ``[\\mathrm{...}]``.

    LaTeX specials (e.g. the underscore in ``S_NO``) are escaped so the name
    renders literally, and a trailing ionic charge (``+`` / ``-``) is raised to
    a superscript (e.g. ``Br-`` -> ``[\\mathrm{Br^{-}}]``).
    """
    escaped = _escape_latex(name).replace("+", "^{+}").replace("-", "^{-}")
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
        return rf"\mathit{{{_escape_latex(node.field_name)}}}"
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
        return f"{_paren_pow_base(node.left)}^{{{to_latex(node.right)}}}"
    if isinstance(node, ArrheniusNode):
        A = to_latex(node.A)
        Ea = to_latex(node.Ea)
        return rf"{A} \exp\!\left(-\frac{{{Ea}}}{{R\,T}}\right)"
    if isinstance(node, pHSwitchNode):
        pKa = to_latex(node.pKa)
        return rf"\frac{{1}}{{1 + 10^{{\mathrm{{pH}} - {pKa}}}}}"
    if isinstance(node, pHInhibitNode):
        # Hill lower-pH inhibition in its stable sigmoid closed form:
        #   I = 1 / (1 + 10^{n (m - pH)}),  n = 3/(ul - ll),  m = (ul + ll)/2
        # -- 1 at high pH, 0 at low pH. Matches the implemented sigmoid exactly.
        ll = to_latex(node.pH_LL)
        ul = to_latex(node.pH_UL)
        n = rf"\frac{{3}}{{{ul} - {ll}}}"
        m = rf"\frac{{{ul} + {ll}}}{{2}}"
        exponent = rf"{n}\left({m} - \mathrm{{pH}}\right)"
        return rf"\frac{{1}}{{1 + 10^{{{exponent}}}}}"
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


def _paren_pow_base(node: ASTNode) -> str:
    """Parenthesise the base of a power unless it is a single self-delimiting
    atom, so the exponent binds to the whole base.

    Without this, ``(k * [A]) ** 2`` renders as ``k \\cdot [A]^{2}`` (the
    exponent binds only to ``[A]``) and ``(-x) ** 2`` as ``-x^{2}`` (i.e.
    ``-(x^2)``). Atoms that need no parentheses are species, parameters,
    conditions, and non-negative constants; everything else (products,
    quotients, sums, negations, nested powers, negative constants, ...) is
    wrapped.
    """
    rendered = to_latex(node)
    atomic = (
        isinstance(node, (SpeciesNode, ParamNode, ConditionNode))
        or (isinstance(node, ConstantNode) and node.value >= 0)
    )
    return rendered if atomic else _paren(rendered)
