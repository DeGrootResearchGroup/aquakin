"""Scalar expressions of time, for an operating value that varies on a schedule.

An influent flow, a feed concentration, a temperature -- anything the flowsheet
reads as a number -- may instead be written as an expression of the plant clock::

    "18446 + 4000*sin(2*pi*t)"      # a diurnal flow, on a plant running on days

The expression is read once, at load, into a callable evaluated inside the
solve. It compiles to plain JAX arithmetic with no Python control flow, so it is
``jit``-able and differentiable like any other term of the right-hand side, and
it stays a **string** in a flowsheet spec -- it fingerprints, diffs and round
trips, where a Python callable would make opening someone else's flowsheet
arbitrary code execution.

``t`` is the only variable. It is the plant's own clock, so an expression
written for a plant running on days means something different on one running on
hours; :func:`compile_time_expression` takes a ``time_scale`` for that, and the
callers that know both clocks (an influent's ``in_time_unit``) pass it.

The grammar is ordinary arithmetic -- ``+ - * / **``, brackets, and the
functions in :data:`FUNCTIONS` -- with no comparison or conditional. A value
that has to switch is written with ``clip`` or ``step``, which stay
differentiable, rather than an ``if`` that a traced value cannot take.

One precedence to know: a sign binds **tighter** than a power, so ``-2**t`` is
``(-2)**t`` rather than Python's ``-(2**t)``. That is deliberate -- it is what a
rate expression already does (:mod:`aquakin.core.parser` shares the grammar), and
the two languages must not read the same text differently. Write ``0 - 2**t`` for
the other reading.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

__all__ = [
    "CONSTANTS",
    "FUNCTIONS",
    "TimeExpressionError",
    "compile_time_expression",
    "is_time_expression",
]


class TimeExpressionError(ValueError):
    """Raised when an expression of time cannot be read.

    A :class:`ValueError`, so a caller validating user input with a plain
    ``except ValueError`` keeps working.
    """


def _step(x):
    """0 below zero, 1 from zero up. The AD-safe stand-in for an ``if``."""
    return jnp.heaviside(x, 1.0)


#: The functions an expression may call, by name and arity.
FUNCTIONS: dict[str, tuple[Callable, tuple[int, ...]]] = {
    "sin": (jnp.sin, (1,)),
    "cos": (jnp.cos, (1,)),
    "tan": (jnp.tan, (1,)),
    "exp": (jnp.exp, (1,)),
    "log": (jnp.log, (1,)),
    "log10": (jnp.log10, (1,)),
    "sqrt": (jnp.sqrt, (1,)),
    "abs": (jnp.abs, (1,)),
    "floor": (jnp.floor, (1,)),
    "min": (jnp.minimum, (2,)),
    "max": (jnp.maximum, (2,)),
    "mod": (jnp.mod, (2,)),
    "clip": (jnp.clip, (3,)),
    "step": (_step, (1,)),
}

#: The named constants an expression may use.
CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

_PUNCTUATION = "+-*/(),"
_NAME = re.compile(r"[A-Za-z_]\w*\Z")


@dataclass(frozen=True)
class _Token:
    kind: str  # "number", "name", "pow", "end", or the punctuation character
    text: str
    pos: int


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == "*" and i + 1 < n and text[i + 1] == "*":
            tokens.append(_Token("pow", "**", i))
            i += 2
        elif c in _PUNCTUATION:
            tokens.append(_Token(c, c, i))
            i += 1
        elif c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            # An exponent only when a digit follows it (optionally after a sign),
            # so "1e" stays a number followed by a name and reads as a clean error.
            if j < n and text[j] in "eE":
                k = j + 1
                if k < n and text[k] in "+-":
                    k += 1
                if k < n and text[k].isdigit():
                    j = k
                    while j < n and text[j].isdigit():
                        j += 1
            try:
                float(text[i:j])
            except ValueError:
                raise TimeExpressionError(
                    f"'{text[i:j]}' at position {i} is not a number"
                ) from None
            tokens.append(_Token("number", text[i:j], i))
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(_Token("name", text[i:j], i))
            i = j
        else:
            raise TimeExpressionError(f"unexpected character {c!r} at position {i}")
    tokens.append(_Token("end", "", n))
    return tokens


class _Parser:
    """Recursive descent over an expression of time, building a callable.

    Each rule returns ``f(t) -> value``; the tree is built once at load and the
    leaves close over constants, so evaluating it is plain JAX arithmetic::

        expr    := term (('+' | '-') term)*
        term    := factor (('*' | '/') factor)*
        factor  := unary ('**' factor)?          # right-associative
        unary   := ('+' | '-') unary | primary
        primary := number | 't' | constant | name '(' args ')' | '(' expr ')'
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.i = 0

    def _peek(self) -> _Token:
        return self.tokens[self.i]

    def _take(self, kind: str) -> _Token:
        token = self._peek()
        if token.kind != kind:
            raise TimeExpressionError(self._unexpected(token, f"expected {kind!r}"))
        self.i += 1
        return token

    def _unexpected(self, token: _Token, expected: str) -> str:
        what = "the end of the expression" if token.kind == "end" else f"{token.text!r}"
        return f"{expected}, got {what} at position {token.pos}"

    def parse(self) -> Callable:
        if self._peek().kind == "end":
            raise TimeExpressionError("the expression is empty")
        fn = self._expression()
        token = self._peek()
        if token.kind != "end":
            raise TimeExpressionError(self._unexpected(token, "expected an operator"))
        return fn

    def _expression(self) -> Callable:
        fn = self._term()
        while self._peek().kind in "+-":
            plus = self._peek().kind == "+"
            self.i += 1
            right = self._term()
            left = fn
            fn = (
                (lambda t, a=left, b=right: a(t) + b(t))
                if plus
                else (lambda t, a=left, b=right: a(t) - b(t))
            )
        return fn

    def _term(self) -> Callable:
        fn = self._factor()
        while self._peek().kind in "*/":
            times = self._peek().kind == "*"
            self.i += 1
            right = self._factor()
            left = fn
            fn = (
                (lambda t, a=left, b=right: a(t) * b(t))
                if times
                else (lambda t, a=left, b=right: a(t) / b(t))
            )
        return fn

    def _factor(self) -> Callable:
        fn = self._unary()
        if self._peek().kind == "pow":
            self.i += 1
            right = self._factor()  # right-associative
            return lambda t, a=fn, b=right: a(t) ** b(t)
        return fn

    def _unary(self) -> Callable:
        token = self._peek()
        if token.kind in "+-":
            self.i += 1
            fn = self._unary()
            return fn if token.kind == "+" else (lambda t, a=fn: -a(t))
        return self._primary()

    def _primary(self) -> Callable:
        token = self._peek()
        if token.kind == "number":
            self.i += 1
            return lambda t, v=float(token.text): v
        if token.kind == "(":
            self.i += 1
            fn = self._expression()
            self._take(")")
            return fn
        if token.kind == "name":
            self.i += 1
            return self._name(token)
        raise TimeExpressionError(self._unexpected(token, "expected a number, 't' or '('"))

    def _name(self, token: _Token) -> Callable:
        name = token.text
        if self._peek().kind == "(":
            return self._call(token)
        if name == "t":
            return lambda t: t
        if name in CONSTANTS:
            return lambda t, v=CONSTANTS[name]: v
        known = ["t", *CONSTANTS, *FUNCTIONS]
        raise TimeExpressionError(
            f"'{name}' at position {token.pos} is not a value an expression of time can "
            f"read. Use 't', {', '.join(sorted(CONSTANTS))}, or one of "
            f"{', '.join(sorted(FUNCTIONS))}.{_hint(name, known)}"
        )

    def _call(self, token: _Token) -> Callable:
        name = token.text
        spec = FUNCTIONS.get(name)
        if spec is None:
            raise TimeExpressionError(
                f"'{name}' at position {token.pos} is not a function an expression of time "
                f"can call. Built-ins are: {', '.join(sorted(FUNCTIONS))}."
                f"{_hint(name, list(FUNCTIONS))}"
            )
        func, arities = spec
        self._take("(")
        args = [self._expression()]
        while self._peek().kind == ",":
            self.i += 1
            args.append(self._expression())
        self._take(")")
        if len(args) not in arities:
            wanted = " or ".join(str(a) for a in arities)
            raise TimeExpressionError(
                f"{name}() takes {wanted} argument(s), got {len(args)} (at position {token.pos})"
            )
        return lambda t, f=func, a=tuple(args): f(*(x(t) for x in a))


def _hint(name: str, known: list[str]) -> str:
    from aquakin.core.hints import did_you_mean

    return did_you_mean(name, known)


def is_time_expression(value) -> bool:
    """Whether a value is an expression rather than a plain number.

    A number -- or a string that is only a number -- is not an expression; it is
    a constant, and the caller can keep it as one.
    """
    if not isinstance(value, str):
        return False
    try:
        float(value)
    except ValueError:
        return True
    return False


def compile_time_expression(value, *, time_scale: float = 1.0, what: str = "value") -> Callable:
    """Read ``value`` as a function of time.

    Parameters
    ----------
    value : str or float
        The expression, or a plain number (which compiles to a constant).
    time_scale : float, optional
        What to multiply the caller's ``t`` by before the expression sees it --
        the factor from the caller's clock to the clock the expression was
        written on. An expression written in days, evaluated by a plant running
        on hours, takes ``time_scale = 1/24``.
    what : str, optional
        What the value is, for the error message (``"the flow of influent
        'feed'"``).

    Returns
    -------
    callable
        ``f(t) -> value``, JAX-traceable and differentiable.

    Raises
    ------
    TimeExpressionError
        If the expression cannot be read.

    Examples
    --------
    >>> f = compile_time_expression("18446 + 4000*sin(2*pi*t)")
    >>> float(f(0.0))
    18446.0
    >>> constant = compile_time_expression(2000.0)
    >>> float(constant(123.0))
    2000.0
    """
    if not is_time_expression(value):
        number = float(value)
        return lambda t, v=number: v
    try:
        fn = _Parser(value).parse()
    except TimeExpressionError as exc:
        raise TimeExpressionError(f"{what}: {exc.args[0]}") from None
    if time_scale == 1.0:
        return fn
    return lambda t, f=fn, s=time_scale: f(t * s)
