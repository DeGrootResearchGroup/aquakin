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

Precedence follows Python and ordinary mathematical notation throughout,
including that a sign binds **looser** than a power: ``-2**t`` is ``-(2**t)``.
A rate expression reads the same way (:mod:`aquakin.core.parser` shares the
grammar), so the two languages never disagree about the same text.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax.numpy as jnp

from aquakin.core.lexer import Token
from aquakin.core.precedence import PrecedenceParser

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


class _Parser(PrecedenceParser):
    """The shared precedence, building a callable of ``t``.

    Each rule returns ``f(t) -> value``; the leaves close over their constants,
    so evaluating the result is plain JAX arithmetic.
    """

    error = TimeExpressionError

    def parse(self) -> Callable:
        if self.peek().kind == "EOF":
            raise TimeExpressionError("the expression is empty")
        return super().parse()

    def trailing_message(self, token) -> str:
        return f"expected an operator, got {self.describe(token)} at position {token.pos}"

    # --- the arithmetic, as closures --------------------------------------

    def add(self, left, right, token):
        return lambda t, a=left, b=right: a(t) + b(t)

    def subtract(self, left, right, token):
        return lambda t, a=left, b=right: a(t) - b(t)

    def multiply(self, left, right, token):
        return lambda t, a=left, b=right: a(t) * b(t)

    def divide(self, left, right, token):
        return lambda t, a=left, b=right: a(t) / b(t)

    def negate(self, operand, token):
        return lambda t, a=operand: -a(t)

    def power(self, base, exponent, token):
        return lambda t, a=base, b=exponent: a(t) ** b(t)

    # --- its leaves -------------------------------------------------------

    def primary(self) -> Callable:
        token = self.peek()
        if token.kind == "NUMBER":
            self.advance()
            return lambda t, v=float(token.value): v
        if token.kind == "LPAREN":
            self.advance()
            fn = self.expression()
            self.expect("RPAREN")
            return fn
        if token.kind == "IDENT":
            self.advance()
            return self._name(token)
        raise TimeExpressionError(
            f"expected a number, 't' or '(', got {self.describe(token)} at position {token.pos}"
        )

    def _name(self, token: Token) -> Callable:
        name = token.value
        if self.peek().kind == "LPAREN":
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

    def _call(self, token: Token) -> Callable:
        name = token.value
        spec = FUNCTIONS.get(name)
        if spec is None:
            raise TimeExpressionError(
                f"'{name}' at position {token.pos} is not a function an expression of time "
                f"can call. Built-ins are: {', '.join(sorted(FUNCTIONS))}."
                f"{_hint(name, list(FUNCTIONS))}"
            )
        func, arities = spec
        self.expect("LPAREN")
        args = [self.expression()]
        while self.peek().kind == "COMMA":
            self.advance()
            args.append(self.expression())
        self.expect("RPAREN")
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
