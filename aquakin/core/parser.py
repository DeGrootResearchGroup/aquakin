"""Recursive-descent parser for rate expressions.

Grammar (lowest to highest precedence)::

    expr      := term (('+' | '-') term)*
    term      := factor (('*' | '/') factor)*
    factor    := unary ('**' factor)?          # right-associative
    unary     := ('+' | '-') unary | primary
    primary   := number
               | '[' species_name ']'
               | '{' condition_name '}'
               | identifier ( '(' arglist ')' )?
               | '(' expr ')'
    arglist   := expr (',' expr)*

Identifiers without parentheses are rate-constant references (``ParamNode``).
The built-in function names ``arrhenius`` and ``pH_switch`` emit their
respective domain nodes. Any other identifier with parentheses is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from aquakin.core.nodes import (
    AddNode,
    ArrheniusNode,
    ASTNode,
    ConditionNode,
    ConstantNode,
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


class ParseError(ValueError):
    """Raised when a rate expression cannot be parsed."""


# --- Tokenizer ---------------------------------------------------------


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int


_SINGLE_CHAR_TOKENS = {
    "+": "PLUS",
    "-": "MINUS",
    "*": "STAR",  # may be promoted to "POW" if doubled
    "/": "SLASH",
    "(": "LPAREN",
    ")": "RPAREN",
    ",": "COMMA",
    "[": "LBRACK",
    "]": "RBRACK",
    "{": "LBRACE",
    "}": "RBRACE",
}


def _tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == "*" and i + 1 < n and text[i + 1] == "*":
            tokens.append(Token("POW", "**", i))
            i += 2
            continue
        if c in _SINGLE_CHAR_TOKENS:
            tokens.append(Token(_SINGLE_CHAR_TOKENS[c], c, i))
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            saw_dot = False
            saw_exp = False
            while j < n:
                cj = text[j]
                if cj.isdigit():
                    j += 1
                elif cj == "." and not saw_dot and not saw_exp:
                    saw_dot = True
                    j += 1
                elif cj in ("e", "E") and not saw_exp:
                    saw_exp = True
                    j += 1
                    if j < n and text[j] in "+-":
                        j += 1
                else:
                    break
            tokens.append(Token("NUMBER", text[i:j], i))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(Token("IDENT", text[i:j], i))
            i = j
            continue
        raise ParseError(f"Unexpected character {c!r} at position {i}")
    tokens.append(Token("EOF", "", n))
    return tokens


# --- Parser ------------------------------------------------------------


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.i = 0

    def _peek(self) -> Token:
        return self.tokens[self.i]

    def _eat(self, kind: str | Iterable[str]) -> Token:
        tok = self._peek()
        kinds = (kind,) if isinstance(kind, str) else tuple(kind)
        if tok.kind not in kinds:
            raise ParseError(
                f"Expected {kinds} at position {tok.pos}, got {tok.kind} ({tok.value!r}) "
                f"in expression {self.text!r}"
            )
        self.i += 1
        return tok

    def parse(self) -> ASTNode:
        node = self._expr()
        if self._peek().kind != "EOF":
            tok = self._peek()
            raise ParseError(
                f"Unexpected token {tok.value!r} at position {tok.pos} in expression "
                f"{self.text!r}"
            )
        return node

    def _expr(self) -> ASTNode:
        node = self._term()
        while self._peek().kind in ("PLUS", "MINUS"):
            op = self._eat(("PLUS", "MINUS"))
            right = self._term()
            node = AddNode(node, right) if op.kind == "PLUS" else SubtractNode(node, right)
        return node

    def _term(self) -> ASTNode:
        node = self._factor()
        while self._peek().kind in ("STAR", "SLASH"):
            op = self._eat(("STAR", "SLASH"))
            right = self._factor()
            node = MultiplyNode(node, right) if op.kind == "STAR" else DivideNode(node, right)
        return node

    def _factor(self) -> ASTNode:
        node = self._unary()
        if self._peek().kind == "POW":
            self._eat("POW")
            right = self._factor()  # right-associative
            node = PowerNode(node, right)
        return node

    def _unary(self) -> ASTNode:
        if self._peek().kind == "PLUS":
            self._eat("PLUS")
            return self._unary()
        if self._peek().kind == "MINUS":
            self._eat("MINUS")
            return NegateNode(self._unary())
        return self._primary()

    def _primary(self) -> ASTNode:
        tok = self._peek()
        if tok.kind == "NUMBER":
            self._eat("NUMBER")
            return ConstantNode(float(tok.value))
        if tok.kind == "LPAREN":
            self._eat("LPAREN")
            node = self._expr()
            self._eat("RPAREN")
            return node
        if tok.kind == "LBRACE":
            self._eat("LBRACE")
            name_tok = self._eat("IDENT")
            self._eat("RBRACE")
            return ConditionNode(name_tok.value)
        if tok.kind == "LBRACK":
            self._eat("LBRACK")
            name_tok = self._peek()
            if name_tok.kind != "IDENT":
                # Allow species names that include things like '-' or digits-after-letters?
                # Species names are constrained to identifiers; charge suffixes like 'Br-' or
                # 'BrO3-' need special handling.
                raise ParseError(
                    f"Expected species name after '[' at position {tok.pos}, "
                    f"got {name_tok.kind} {name_tok.value!r}"
                )
            name = self._read_species_name()
            self._eat("RBRACK")
            return SpeciesNode(name)
        if tok.kind == "IDENT":
            self._eat("IDENT")
            name = tok.value
            if self._peek().kind == "LPAREN":
                self._eat("LPAREN")
                args = self._arglist()
                self._eat("RPAREN")
                return self._make_function_call(name, args)
            return ParamNode(name)
        raise ParseError(
            f"Unexpected token {tok.value!r} at position {tok.pos} in expression "
            f"{self.text!r}"
        )

    def _read_species_name(self) -> str:
        """
        Read a species name inside ``[...]``.

        Species names can include charge suffixes such as ``Br-``, ``BrO3-``,
        ``H+``, ``Ca2+``. Adjacent tokens (no intervening whitespace) are
        concatenated. Whitespace inside the brackets ends the species name
        and any trailing non-``]`` content raises a ``ParseError``.
        """
        parts: list[str] = []
        # First piece must be an identifier (letters/digits/underscore).
        ident = self._eat("IDENT")
        parts.append(ident.value)
        prev_end = ident.pos + len(ident.value)
        # Then any number of adjacent '+', '-', integer, or ident tokens.
        while self._peek().kind in ("PLUS", "MINUS", "NUMBER", "IDENT"):
            tok = self._peek()
            if tok.pos != prev_end:
                # Whitespace separated the previous token from this one — stop.
                break
            if tok.kind == "PLUS":
                self._eat("PLUS")
                parts.append("+")
            elif tok.kind == "MINUS":
                self._eat("MINUS")
                parts.append("-")
            elif tok.kind == "NUMBER":
                # Only accept pure-integer numbers as part of a species name.
                if "." in tok.value or "e" in tok.value.lower():
                    break
                self._eat("NUMBER")
                parts.append(tok.value)
            elif tok.kind == "IDENT":
                self._eat("IDENT")
                parts.append(tok.value)
            else:  # pragma: no cover
                break
            prev_end = tok.pos + len(tok.value)
        return "".join(parts)

    def _arglist(self) -> list[ASTNode]:
        args = [self._expr()]
        while self._peek().kind == "COMMA":
            self._eat("COMMA")
            args.append(self._expr())
        return args

    def _make_function_call(self, name: str, args: list[ASTNode]) -> ASTNode:
        if name == "arrhenius":
            if len(args) != 2:
                raise ParseError(
                    f"arrhenius() takes 2 arguments (A, Ea), got {len(args)}"
                )
            return ArrheniusNode(args[0], args[1])
        if name == "pH_switch":
            if len(args) != 1:
                raise ParseError(
                    f"pH_switch() takes 1 argument (pKa), got {len(args)}"
                )
            return pHSwitchNode(args[0])
        if name == "pH_inhibit":
            if len(args) != 2:
                raise ParseError(
                    f"pH_inhibit() takes 2 arguments (pH_LL, pH_UL), got {len(args)}"
                )
            return pHInhibitNode(args[0], args[1])
        if name == "monod":
            if len(args) != 2:
                raise ParseError(
                    f"monod() takes 2 arguments (X, K), got {len(args)}"
                )
            return MonodNode(args[0], args[1])
        if name == "monod_inh":
            if len(args) != 2:
                raise ParseError(
                    f"monod_inh() takes 2 arguments (X, K), got {len(args)}"
                )
            return MonodInhibitionNode(args[0], args[1])
        if name == "monod_ratio":
            if len(args) != 3:
                raise ParseError(
                    f"monod_ratio() takes 3 arguments (A, B, K), got {len(args)}"
                )
            return MonodRatioNode(args[0], args[1], args[2])
        if name == "monod_inh_ratio":
            if len(args) != 3:
                raise ParseError(
                    f"monod_inh_ratio() takes 3 arguments (A, B, K), got {len(args)}"
                )
            return MonodInhibitionRatioNode(args[0], args[1], args[2])
        raise ParseError(
            f"Unknown function '{name}'. Built-ins are: arrhenius, pH_switch, "
            f"monod, monod_inh, monod_ratio, monod_inh_ratio."
        )


def parse_rate_expression(text: str) -> ASTNode:
    """Parse a rate-expression string into an AST.

    Parameters
    ----------
    text : str
        Rate expression, e.g. ``"k1 * [O3] * [Br-]"``.

    Returns
    -------
    ASTNode
        Root of the parsed expression tree.

    Raises
    ------
    ParseError
        On syntax errors.
    """
    return _Parser(text).parse()
