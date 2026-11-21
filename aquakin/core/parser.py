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
from functools import reduce

from aquakin.core.lexer import Token
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
from aquakin.core.precedence import PrecedenceParser, accepts, spell_arity

#: The module a *built-in* node lives in. Discovery is restricted to it so a
#: stray subclass defined anywhere else cannot reach the grammar.
_NODES_MODULE = ASTNode.__module__

#: Rate functions registered from outside the package, name -> node class.
#: Populated **only** by :func:`aquakin.register_rate_function`; see there for
#: why this is an explicit table rather than more discovery.
_EXTRA_FUNCTIONS: dict[str, type] = {}


def _build_function_registry() -> dict:
    """Callable rate functions: name -> (node class, argument names).

    Built-ins are discovered by walking the AST node hierarchy for every node
    **defined in** :mod:`aquakin.core.nodes` that declares a ``FUNCTION_NAME``
    -- its call spelling -- with the argument names read straight off the
    node's dataclass fields (the constructor takes them positionally, in
    order). So a new built-in node registers its grammar, kernel and interning
    all from its own class body, with no edit here. Drives both
    ``_make_function_call`` (arity + argument-name errors) and the "unknown
    function" message, so neither can go stale.

    Nodes defined **elsewhere** are deliberately *not* discovered, and reach
    the grammar only through ``_EXTRA_FUNCTIONS``. Merely defining a class must
    not change what a rate expression means: discovery walks live subclasses,
    so without this restriction a class that was defined and never registered
    could take a built-in's name on the next rebuild -- silently, and depending
    on traversal order -- which would bypass the collision check that
    registration does.
    """
    registry: dict = {}
    stack = list(ASTNode.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.__module__ != _NODES_MODULE:
            continue
        name = cls.__dict__.get("FUNCTION_NAME")
        if name is not None:
            registry[name] = (cls, tuple(f.name for f in fields(cls)))
    for name, cls in _EXTRA_FUNCTIONS.items():
        registry[name] = (cls, tuple(f.name for f in fields(cls)))
    return registry


_FUNCTIONS = _build_function_registry()


def refresh_function_registry() -> None:
    """Rebuild :data:`_FUNCTIONS` from the current node hierarchy.

    The registry is built once at import, so a node class defined *later* --
    a user-registered rate function (:func:`aquakin.register_rate_function`) --
    is invisible to the parser until it is rebuilt. Rebuilding is the whole
    mechanism: discovery is already generic, so nothing else about the parser
    knows or cares that a function came from outside the package.

    Rebinding the module global is enough because ``_make_function_call`` looks
    the registry up at call time.
    """
    global _FUNCTIONS
    _FUNCTIONS = _build_function_registry()


class ParseError(ValueError):
    """Raised when a rate expression cannot be parsed."""


# --- Tokenizer ---------------------------------------------------------


# --- Parser ------------------------------------------------------------


class _Parser(PrecedenceParser):
    """The shared precedence, building the rate-expression AST.

    Only the leaves are this language's own: ``[species]``, ``{condition}``, a
    bare identifier as a rate constant, and the domain functions discovered from
    the AST node classes.
    """

    error = ParseError

    def __init__(self, text: str) -> None:
        super().__init__(text)

    def _eat(self, kind: str | Iterable[str]) -> Token:
        tok = self.peek()
        kinds = (kind,) if isinstance(kind, str) else tuple(kind)
        if tok.kind not in kinds:
            raise ParseError(
                f"Expected {kinds} at position {tok.pos}, got {tok.kind} ({tok.value!r}) "
                f"in expression {self.text!r}"
            )
        self.i += 1
        return tok

    def trailing_message(self, token: Token) -> str:
        return (
            f"Unexpected token {token.value!r} at position {token.pos} in expression {self.text!r}"
        )

    # --- the arithmetic, as AST nodes -------------------------------------

    def add(self, left, right, token):
        return AddNode(left, right)

    def subtract(self, left, right, token):
        return SubtractNode(left, right)

    def multiply(self, left, right, token):
        return MultiplyNode(left, right)

    def divide(self, left, right, token):
        return DivideNode(left, right)

    def negate(self, operand, token):
        return NegateNode(operand)

    def power(self, base, exponent, token):
        return PowerNode(base, exponent)

    # --- its leaves -------------------------------------------------------

    def primary(self) -> ASTNode:
        tok = self.peek()
        if tok.kind == "NUMBER":
            self._eat("NUMBER")
            return ConstantNode(float(tok.value))
        if tok.kind == "LPAREN":
            self._eat("LPAREN")
            node = self.expression()
            self._eat("RPAREN")
            return node
        if tok.kind == "LBRACE":
            self._eat("LBRACE")
            name_tok = self._eat("IDENT")
            self._eat("RBRACE")
            return ConditionNode(name_tok.value)
        if tok.kind == "LBRACK":
            self._eat("LBRACK")
            name_tok = self.peek()
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
            if self.peek().kind == "LPAREN":
                self._eat("LPAREN")
                args = self.arglist()
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
        while self.peek().kind in ("PLUS", "MINUS", "NUMBER", "IDENT"):
            tok = self.peek()
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

    def _make_function_call(self, name: str, args: list[ASTNode]) -> ASTNode:
        spec = _FUNCTIONS.get(name)
        if spec is None:
            available = ", ".join(sorted(_FUNCTIONS))
            raise ParseError(
                f"Unknown function '{name}'. Available functions are: {available}. "
                "A function defined outside aquakin has to be registered with "
                "aquakin.register_rate_function() before the model is loaded."
            )
        node_cls, arg_names = spec
        arities = (len(arg_names),)
        if not accepts(name, arities, len(args)):
            raise ParseError(
                f"{name}() takes {spell_arity(name, arities)} argument(s) "
                f"({', '.join(arg_names)}), got {len(args)}"
            )
        if len(args) > len(arg_names):
            # An associative built-in called on more than it is defined for:
            # fold left, so max(a, b, c) builds max(max(a, b), c). Exact, since
            # only associative functions reach here.
            return reduce(node_cls, args)
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
