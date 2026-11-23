"""Expressions of time: the shared engine behind an operating value that varies
on a schedule (an influent's flow, a setpoint).

The reading needs no plant, so these are pure; that an influent actually follows
one is ``tests/integration``.
"""

import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquakin.plant.time_expr import (
    CONSTANTS,
    FUNCTIONS,
    TimeExpressionError,
    compile_time_expression,
    is_time_expression,
)


def value(text, t, **kw):
    return float(compile_time_expression(text, **kw)(t))


@pytest.mark.parametrize(
    ("text", "t", "expected"),
    [
        ("1", 0.0, 1.0),
        ("t", 3.5, 3.5),
        ("-t", 3.5, -3.5),
        ("+t", 3.5, 3.5),
        ("2*t + 1", 3.0, 7.0),
        ("t/4", 3.0, 0.75),
        ("(t + 1)*2", 3.0, 8.0),
        ("1e3*t", 2.0, 2000.0),
        # A signed exponent takes a different path through the scanner than an
        # unsigned one, so both are exercised.
        ("1e-3*t", 2.0, 0.002),
        ("1E+3*t", 2.0, 2000.0),
        # ** is right-associative, as in Python: 2**(3**2), not (2**3)**2.
        ("2**3**2", 0.0, 512.0),
        ("-2**2", 0.0, -4.0),
        ("2**-1", 0.0, 0.5),
        ("pi", 0.0, math.pi),
        ("e", 0.0, math.e),
    ],
)
def test_the_arithmetic(text, t, expected):
    assert value(text, t) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "t", "expected"),
    [
        ("sin(0)", 0.0, 0.0),
        ("cos(0)", 0.0, 1.0),
        ("exp(0)", 0.0, 1.0),
        ("log(e)", 0.0, 1.0),
        ("log10(100)", 0.0, 2.0),
        ("sqrt(9)", 0.0, 3.0),
        ("abs(0-3)", 0.0, 3.0),
        ("floor(2.7)", 0.0, 2.0),
        ("min(2, 5)", 0.0, 2.0),
        ("max(2, 5)", 0.0, 5.0),
        ("mod(7, 4)", 0.0, 3.0),
        ("clip(t, 0, 10)", 99.0, 10.0),
        ("clip(t, 0, 10)", -5.0, 0.0),
        # step is the AD-safe stand-in for an if: 1 from zero up, 0 below.
        ("step(t)", 1.0, 1.0),
        ("step(t)", 0.0, 1.0),
        ("step(t)", -1.0, 0.0),
    ],
)
def test_the_functions(text, t, expected):
    assert value(text, t) == pytest.approx(expected, rel=1e-6, abs=0)


def test_every_declared_function_and_constant_can_be_called():
    """So the tables and the parser cannot drift apart."""
    for name, (_, arities) in FUNCTIONS.items():
        args = ", ".join(["1"] * min(arities))
        assert np.isfinite(value(f"{name}({args})", 0.0))
    for name in CONSTANTS:
        assert value(name, 0.0) == pytest.approx(CONSTANTS[name])


def test_a_diurnal_flow():
    """The motivating case: a daily swing about a mean, on a plant on days."""
    f = compile_time_expression("18446 + 4000*sin(2*pi*t)")
    assert float(f(0.0)) == pytest.approx(18446.0)
    assert float(f(0.25)) == pytest.approx(22446.0)
    assert float(f(0.75)) == pytest.approx(14446.0)
    # A whole day later it is back where it started.
    assert float(f(1.0)) == pytest.approx(float(f(0.0)))


def test_a_number_is_a_constant_not_an_expression():
    """So a plain setting keeps costing nothing and reads the same either way."""
    assert not is_time_expression(2000.0)
    assert not is_time_expression("2000")
    assert not is_time_expression("1e4")
    assert is_time_expression("2000 + t")
    assert not is_time_expression(None)
    for given in (2000.0, "2000", 2000):
        assert value(given, 12345.0) == 2000.0


def test_the_time_scale_puts_an_expression_on_the_callers_clock():
    """An expression written in days, read by a plant running on hours: six
    hours in is a quarter of the way through the day's cycle."""
    f = compile_time_expression("sin(2*pi*t)", time_scale=1 / 24)
    assert float(f(6.0)) == pytest.approx(1.0)
    assert float(f(24.0)) == pytest.approx(0.0, abs=1e-12)
    # A scale of exactly 1 is the identity, so the common case adds no work.
    assert value("t", 3.0, time_scale=1.0) == 3.0


def test_it_is_jit_grad_and_vmap_clean():
    """It is evaluated inside the solve, so it has to trace, differentiate and
    batch like any other term of the right-hand side."""
    f = compile_time_expression("100 + 10*t**2")
    assert float(jax.jit(f)(2.0)) == pytest.approx(140.0)
    assert float(jax.grad(f)(2.0)) == pytest.approx(40.0)
    np.testing.assert_allclose(
        np.asarray(jax.vmap(f)(jnp.array([0.0, 1.0, 2.0]))), [100.0, 110.0, 140.0]
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "the expression is empty"),
        ("   ", "the expression is empty"),
        ("t +", "expected a number, 't' or '('"),
        ("sin(", "expected a number, 't' or '('"),
        ("(t", "expected ')'"),
        ("t 2", "expected an operator"),
        ("t @ 2", "Unexpected character '@'"),
        ("1.2.3", "is not a number"),
        # "1e" is deliberately the number 1 followed by the name "e", so it
        # reads as a parse error rather than a bare float() failure.
        ("1e", "expected an operator"),
        ("1e+", "expected an operator"),
        # Arity is checked against the table, so a typo'd call is caught here.
        ("sin(t, 1)", "sin() takes 1 argument(s), got 2"),
        ("min(t)", "min() takes 2 or more argument(s), got 1"),
        ("clip(t, 1, 2, 3)", "clip() takes 3 argument(s), got 4"),
        ("clip(t, 1)", "clip() takes 3 argument(s), got 2"),
    ],
)
def test_what_is_refused(text, message):
    with pytest.raises(TimeExpressionError, match=re.escape(message)):
        compile_time_expression(text)


def test_an_unknown_name_is_hinted():
    with pytest.raises(TimeExpressionError, match="Did you mean: sin"):
        compile_time_expression("sinn(t)")
    with pytest.raises(TimeExpressionError, match="Did you mean: pi"):
        compile_time_expression("2*pii")


def test_an_unknown_name_says_what_it_could_have_been():
    with pytest.raises(TimeExpressionError, match=re.escape("is not a value an expression")):
        compile_time_expression("x + 1")
    with pytest.raises(TimeExpressionError, match=re.escape("is not a function an expression")):
        compile_time_expression("wobble(t)")


def test_an_error_names_what_was_being_read():
    """The message reaches a user who gave one field of one unit, so it has to
    say which."""
    with pytest.raises(TimeExpressionError, match=re.escape("the flow of influent 'feed':")):
        compile_time_expression("t +", what="the flow of influent 'feed'")


def test_a_syntax_error_is_a_value_error():
    assert issubclass(TimeExpressionError, ValueError)
    with pytest.raises(ValueError, match="empty"):
        compile_time_expression("")
