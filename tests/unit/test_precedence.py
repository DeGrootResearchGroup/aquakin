"""The operator precedence every aquakin expression language shares.

Each language has its own tests; these pin the precedence itself, once, on a
throwaway language that records what it was asked to build. The three real
languages had this backwards until it lived in one place.
"""

import pytest

from aquakin.core.precedence import PrecedenceParser


class _Sexpr(PrecedenceParser):
    """A language whose leaves are numbers and names, building s-expressions."""

    def add(self, left, right, token):
        return f"(+ {left} {right})"

    def subtract(self, left, right, token):
        return f"(- {left} {right})"

    def multiply(self, left, right, token):
        return f"(* {left} {right})"

    def divide(self, left, right, token):
        return f"(/ {left} {right})"

    def negate(self, operand, token):
        return f"(neg {operand})"

    def power(self, base, exponent, token):
        return f"(** {base} {exponent})"

    def primary(self):
        token = self.peek()
        if token.kind in ("NUMBER", "IDENT"):
            self.advance()
            return token.value
        if token.kind == "LPAREN":
            self.advance()
            inner = self.expression()
            self.expect("RPAREN")
            return inner
        raise self.error(self.expected_message("NUMBER", token))


def read(text):
    return _Sexpr(text).parse()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # + and - are left-associative and bind loosest.
        ("a + b + c", "(+ (+ a b) c)"),
        ("a - b - c", "(- (- a b) c)"),
        ("a + b*c", "(+ a (* b c))"),
        # * and / are left-associative and bind tighter than + -.
        ("a/b/c", "(/ (/ a b) c)"),
        ("a*b + c", "(+ (* a b) c)"),
        # ** binds tighter than * /, and is right-associative.
        ("a*b**c", "(* a (** b c))"),
        ("a**b**c", "(** a (** b c))"),
        # A sign binds LOOSER than **, as in Python: -a**b is -(a**b).
        ("-a**b", "(neg (** a b))"),
        ("-a*b", "(* (neg a) b)"),
        ("-a + b", "(+ (neg a) b)"),
        # ...and an exponent may carry its own sign.
        ("a**-b", "(** a (neg b))"),
        ("a**-b**c", "(** a (neg (** b c)))"),
        # A leading + is a no-op, and signs stack.
        ("+a", "a"),
        ("--a", "(neg (neg a))"),
        # Brackets override all of it.
        ("(a + b)*c", "(* (+ a b) c)"),
        ("(-a)**b", "(** (neg a) b)"),
    ],
)
def test_precedence(text, expected):
    assert read(text) == expected


def test_a_trailing_token_is_refused():
    with pytest.raises(ValueError, match="unexpected"):
        read("a b")


def test_a_language_refuses_an_operation_by_raising_in_its_hook():
    """How a measure expression stays affine: not a grammar of its own, but a
    hook that says no."""

    class NoPowers(_Sexpr):
        def power(self, base, exponent, token):
            raise self.error(f"no powers here (at position {token.pos})")

    assert NoPowers("a*b").parse() == "(* a b)"
    with pytest.raises(ValueError, match=r"no powers here \(at position 1\)"):
        NoPowers("a**b").parse()


def test_the_error_class_is_the_language_s_own():
    class Mine(ValueError):
        pass

    class Lang(_Sexpr):
        error = Mine

    # Both tokenizing and parsing raise it, so one language means one error type.
    with pytest.raises(Mine, match="Unexpected character '@' at position 2"):
        Lang("a @ b").parse()
    with pytest.raises(Mine, match="unexpected 'b' at position 2"):
        Lang("a b").parse()


def test_positions_survive_an_offset():
    """A caller that has already taken a prefix off the text still gets
    positions pointing into what the user wrote."""

    class Lang(_Sexpr):
        def power(self, base, exponent, token):
            raise self.error(f"at {token.pos}")

    with pytest.raises(ValueError, match="at 13"):
        Lang("a**b", offset=12).parse()


def test_a_language_that_implements_no_hook_says_so():
    """The hooks are the contract: a subclass that forgets one fails loudly
    rather than building something wrong."""

    class Bare(PrecedenceParser):
        def primary(self):
            token = self.advance()
            return token.value

    for text in ("a + b", "a - b", "a*b", "a/b", "-a"):
        with pytest.raises(NotImplementedError):
            Bare(text).parse()
    # primary() is a hook too.
    with pytest.raises(NotImplementedError):
        PrecedenceParser("a").parse()


def test_powers_are_refused_by_default():
    """So a language gets a sensible refusal before it has thought about
    powers, rather than a NotImplementedError leaking to a user."""

    class Bare(PrecedenceParser):
        def primary(self):
            return self.advance().value

    with pytest.raises(ValueError, match=r"'\*\*' is not allowed here \(at position 1\)"):
        Bare("a**b").parse()
