"""The operator precedence every aquakin expression language shares.

Three languages read arithmetic — rate expressions
(:mod:`aquakin.core.parser`), expressions of time
(:mod:`aquakin.plant.time_expr`) and measure expressions
(:mod:`aquakin.plant.measures`). They differ in what a *leaf* is (a species, the
clock, a measurement keyword) and in what they *build* (an AST, a callable, an
affine weighted sum). They do not differ in how ``+ - * / **`` and a sign relate
to one another, and they must not: text that reads one way in a rate expression
has to read the same way everywhere.

So the precedence lives here once::

    expression := term (('+' | '-') term)*
    term       := unary (('*' | '/') unary)*
    unary      := ('+' | '-') unary | power
    power      := primary ('**' unary)?

A sign binds **looser** than a power, as in Python and ordinary mathematical
notation, so ``-2**2`` is ``-(2**2)``; ``**`` is right-associative and its
exponent may carry its own sign, so ``2**3**2`` and ``2**-1`` read as they do in
Python. All three languages had this backwards until #699, and it had to be
fixed in each by hand — which is why it is no longer written down more than once.

A language subclasses :class:`PrecedenceParser`, supplies :meth:`primary` for
its own leaves, and implements one hook per operation. A language that does not
*allow* an operation says so by raising in its hook — that is how a measure
expression stays affine, rather than by having a grammar of its own.
"""

from __future__ import annotations

from typing import ClassVar

from aquakin.core.lexer import Token, spell, tokenize

__all__ = ["ASSOCIATIVE_CALLS", "PrecedenceParser", "accepts", "spell_arity"]


#: Built-ins that are associative, so a call may carry more arguments than the
#: two they are defined on and fold left -- ``max(a, b, c)`` is
#: ``max(max(a, b), c)``. Python's own ``max`` and ``min`` take any number of
#: arguments, and someone writing a three-way bound expects the same here. It is
#: a fact about the functions rather than about any one language, so every
#: language that offers them reads it from here instead of keeping its own list.
ASSOCIATIVE_CALLS = frozenset({"min", "max"})


def accepts(name: str, arities: tuple[int, ...], n: int) -> bool:
    """Whether a call of ``name`` with ``n`` arguments is allowed.

    ``arities`` is what the function declares. An associative built-in declared
    on two arguments also accepts more; everything else takes what it declares.
    """
    if n in arities:
        return True
    return n > 2 and 2 in arities and name in ASSOCIATIVE_CALLS


def spell_arity(name: str, arities: tuple[int, ...]) -> str:
    """How many arguments a message should say ``name`` takes."""
    wanted = " or ".join(str(a) for a in arities)
    return f"{wanted} or more" if 2 in arities and name in ASSOCIATIVE_CALLS else wanted


class PrecedenceParser:
    """Recursive descent over the shared precedence, building whatever a
    subclass's hooks build.

    Attributes
    ----------
    error : type
        The exception this language raises, used for both tokenizing and
        parsing so a reader sees one error type for one language.
    """

    error: ClassVar[type[Exception]] = ValueError

    def __init__(self, text: str, *, offset: int = 0) -> None:
        self.text = text
        self.tokens = tokenize(text, error=self.error, offset=offset)
        self.i = 0

    # --- tokens -----------------------------------------------------------

    def peek(self) -> Token:
        """The token about to be read."""
        return self.tokens[self.i]

    def advance(self) -> Token:
        """Read the next token."""
        token = self.tokens[self.i]
        self.i += 1
        return token

    def expect(self, kind: str) -> Token:
        """Read the next token, which must be ``kind``."""
        token = self.peek()
        if token.kind != kind:
            raise self.error(self.expected_message(kind, token))
        self.i += 1
        return token

    # --- messages (override to phrase them in the language's own terms) ----

    def describe(self, token: Token) -> str:
        """A token as a message should name it."""
        return spell("EOF") if token.kind == "EOF" else f"{token.value!r}"

    def expected_message(self, kind: str, token: Token) -> str:
        return f"expected {spell(kind)}, got {self.describe(token)} at position {token.pos}"

    def trailing_message(self, token: Token) -> str:
        return f"unexpected {self.describe(token)} at position {token.pos}"

    # --- the grammar ------------------------------------------------------

    def parse(self):
        """Read the whole text, which must be one expression and nothing more."""
        node = self.expression()
        if self.peek().kind != "EOF":
            raise self.error(self.trailing_message(self.peek()))
        return node

    def expression(self):
        node = self.term()
        while self.peek().kind in ("PLUS", "MINUS"):
            operator = self.advance()
            right = self.term()
            node = (
                self.add(node, right, operator)
                if operator.kind == "PLUS"
                else self.subtract(node, right, operator)
            )
        return node

    def term(self):
        node = self.unary()
        while self.peek().kind in ("STAR", "SLASH"):
            operator = self.advance()
            right = self.unary()
            node = (
                self.multiply(node, right, operator)
                if operator.kind == "STAR"
                else self.divide(node, right, operator)
            )
        return node

    def unary(self):
        token = self.peek()
        if token.kind in ("PLUS", "MINUS"):
            self.advance()
            operand = self.unary()
            return operand if token.kind == "PLUS" else self.negate(operand, token)
        return self.power_rule()

    def power_rule(self):
        base = self.primary()
        if self.peek().kind == "POW":
            operator = self.advance()
            # Right-associative through ``unary``, so the exponent may carry a
            # sign and ``2**3**2`` groups to the right, as in Python.
            return self.power(base, self.unary(), operator)
        return base

    def arglist(self):
        """The arguments of a call, between an already-read ``(`` and its ``)``.

        A trailing comma is allowed, as Python allows one, so a call written
        across several lines does not make its last argument special. Two commas
        in a row are not: the comma is read and an argument is then required.
        """
        args = [self.expression()]
        while self.peek().kind == "COMMA":
            self.advance()
            if self.peek().kind == "RPAREN":
                break
            args.append(self.expression())
        return args

    # --- what each language supplies --------------------------------------

    def primary(self):
        """This language's leaves, and anything bracketed."""
        raise NotImplementedError

    def add(self, left, right, token: Token):
        raise NotImplementedError

    def subtract(self, left, right, token: Token):
        raise NotImplementedError

    def multiply(self, left, right, token: Token):
        raise NotImplementedError

    def divide(self, left, right, token: Token):
        raise NotImplementedError

    def negate(self, operand, token: Token):
        raise NotImplementedError

    def power(self, base, exponent, token: Token):
        """A language without powers refuses them here, in its own words."""
        raise self.error(f"'**' is not allowed here (at position {token.pos})")
