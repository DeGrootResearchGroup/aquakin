"""Recursive-descent parser for rate expressions.

Grammar (lowest to highest precedence)::

    expr      := term (('+' | '-') term)*
    term      := unary (('*' | '/') unary)*
    unary     := ('+' | '-') unary | power
    power     := primary ('**' unary)?         # right-associative
    primary   := number
               | '[' species_name ']'
               | '{' condition_name '}'
               | identifier ( '(' arglist ')' )?
               | '(' expr ')'
    arglist   := expr (',' expr)*

An identifier without parentheses is a rate-constant reference (``ParamNode``);
``[name]`` is a species and ``{name}`` a condition field. The built-in function
names ``arrhenius``, ``pH_switch``, ``pH_inhibit``, ``monod``, ``monod_inh``,
``monod_ratio``, ``monod_inh_ratio``, ``safe_div`` and ``max`` emit their
respective domain nodes. Any other identifier with parentheses is rejected.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields

from aquakin.core.lexer import Token, tokenize
from aquakin.core.nodes import (
    AddNode,
    ASTNode,
    ConditionNode,
    ConstantNode,
    DivideNode,
    MultiplyNode,
    NegateNode,
    ParamNode,
    PowerNode,
    SpeciesNode,
    SubtractNode,
)


def _build_function_registry() -> dict:
    """Built-in domain functions: name -> (node class, argument names).

    Discovered by walking the AST node hierarchy for every node that declares a
    ``FUNCTION_NAME`` -- its call spelling -- with the argument names read
    straight off the node's dataclass fields (the constructor takes them
    positionally, in order). So a new domain-function node registers its grammar,
    kernel and interning all from its own class body, with no edit here. Drives
    both ``_make_function_call`` (arity + argument-name errors) and the
    "unknown function" message, so neither can go stale.
    """
    registry: dict = {}
    stack = list(ASTNode.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        name = cls.__dict__.get("FUNCTION_NAME")
        if name is not None:
            registry[name] = (cls, tuple(f.name for f in fields(cls)))
    return registry


_FUNCTIONS = _build_function_registry()


class ParseError(ValueError):
    """Raised when a rate expression cannot be parsed."""


# --- Tokenizer ---------------------------------------------------------


# --- Parser ------------------------------------------------------------


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = tokenize(text, error=ParseError)
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
                f"Unexpected token {tok.value!r} at position {tok.pos} in expression {self.text!r}"
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
        node = self._unary()
        while self._peek().kind in ("STAR", "SLASH"):
            op = self._eat(("STAR", "SLASH"))
            right = self._unary()
            node = MultiplyNode(node, right) if op.kind == "STAR" else DivideNode(node, right)
        return node

    def _unary(self) -> ASTNode:
        """A sign, or a power.

        A sign binds **looser** than ``**``, as in Python and ordinary
        mathematical notation: ``-2**2`` is ``-(2**2)``, not ``(-2)**2``.
        """
        if self._peek().kind == "PLUS":
            self._eat("PLUS")
            return self._unary()
        if self._peek().kind == "MINUS":
            self._eat("MINUS")
            return NegateNode(self._unary())
        return self._power()

    def _power(self) -> ASTNode:
        node = self._primary()
        if self._peek().kind == "POW":
            self._eat("POW")
            # Right-associative, and the exponent may carry its own sign, so
            # ``2**-1`` and ``2**3**2`` both read as they do in Python.
            node = PowerNode(node, self._unary())
        return node

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
                # A species name must START with an identifier token; its charge
                # suffixes ('Br-', 'BrO3-', 'Ca2+') are read by _read_species_name
                # below. A '[' followed by a non-identifier is malformed.
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
            f"Unexpected token {tok.value!r} at position {tok.pos} in expression {self.text!r}"
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
        spec = _FUNCTIONS.get(name)
        if spec is None:
            raise ParseError(f"Unknown function '{name}'. Built-ins are: {', '.join(_FUNCTIONS)}.")
        node_cls, arg_names = spec
        if len(args) != len(arg_names):
            raise ParseError(
                f"{name}() takes {len(arg_names)} argument(s) "
                f"({', '.join(arg_names)}), got {len(args)}"
            )
        return node_cls(*args)


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
