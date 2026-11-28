"""The expression languages read as Python does, except where they must not.

Someone writing ``-k**2`` or ``max(a, b, c)`` in a model file is reading it with
the Python in their head. Where aquakin's answer differs from Python's, it is
either a bug (``-2**2`` was one, fixed in #699) or a decision worth stating out
loud -- there is no third case, and this file is where the line is drawn.

The check is differential: the same text is handed to Python's own ``eval`` and
to :func:`~aquakin.core.time_expr.compile_time_expression`, with the function
names bound to the same callables on each side, so nothing is compared except
syntax and operator semantics. ``eval`` here runs only the literal corpus below,
never anything a user supplies.

The time language is the vehicle because it is scalar arithmetic over one
variable, which Python can evaluate directly. Its parser is the *shared*
:class:`~aquakin.core.precedence.PrecedenceParser`, so what holds for it holds
for rate expressions too -- pinned at the bottom by parsing the same text into a
rate AST and evaluating that.
"""

import math
import re

import pytest

from aquakin.core.nodes import ASTNode
from aquakin.core.parser import ParseError, parse_rate_expression
from aquakin.core.time_expr import TimeExpressionError, compile_time_expression
from aquakin.plant.measures import MeasureSyntaxError, parse_measure

T = 3.0


def _step(x):
    return 1.0 if x >= 0 else 0.0


def _clip(x, lo, hi):
    return min(max(x, lo), hi)


#: The same names on the Python side, so only the grammar is under test.
_PY_NS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
    "floor": math.floor,
    "min": min,
    "max": max,
    "mod": lambda a, b: a % b,
    "clip": _clip,
    "step": _step,
    "pi": math.pi,
    "e": math.e,
    "t": T,
}

#: ``(expression, python_spelling_or_None)``. ``None`` means Python reads the
#: very same text; a string is the Python spelling of what aquakin writes as a
#: call (Python has no ``mod`` function, it has ``%``).
AGREES_WITH_PYTHON = [
    # precedence and associativity -- the #699 regression lives here
    ("-2**2", None),
    ("2**3**2", None),
    ("2**-1", None),
    ("-2**-2", None),
    ("(-2)**2", None),
    ("2*3**2", None),
    ("-2*3**2", None),
    ("2**2*3", None),
    ("-t**2", None),
    ("2**t", None),
    ("-2**t", None),
    ("1-2-3", None),
    ("8/4/2", None),
    ("8/4*2", None),
    ("2*8/4", None),
    ("1+2*3", None),
    ("(1+2)*3", None),
    ("-3*-4", None),
    ("1/2", None),
    # stacked signs
    ("2--3", None),
    ("2---3", None),
    ("+-2", None),
    ("--2", None),
    ("---2", None),
    ("-+-+2", None),
    ("t--1", None),
    ("t*-2", None),
    # numeric literals, including Python's underscore separators
    ("1.", None),
    (".5", None),
    ("1e3", None),
    ("1E3", None),
    ("1e+3", None),
    ("1e-3", None),
    ("1.5e2", None),
    ("1e400", None),
    ("1_000", None),
    ("1_000+0", "1_000+0"),
    ("18446_000/2", None),
    ("1_0e1_0", None),
    # calls: fixed arity, variadic max/min, and a trailing comma
    ("max(1,2)", None),
    ("min(3,1)", None),
    ("max(1,2,3)", None),
    ("min(4,2,9,1)", None),
    ("max(1,2,)", None),
    ("clip(5,0,3,)", None),
    ("abs(-2)", None),
    ("sqrt(4)", None),
    ("exp(0)", None),
    ("log(e)", None),
    ("log10(100)", None),
    ("floor(2.7)", None),
    ("floor(-2.3)", None),
    ("sin(pi/2)", None),
    ("max(1,min(5,3))", None),
    # aquakin spells these as calls where Python has an operator
    ("mod(-7,3)", "(-7) % 3"),
    ("mod(7,-3)", "7 % -3"),
]

#: Text Python accepts that aquakin refuses, the reason it refuses it, and the
#: fragment of the error that names the refusal.
REFUSED_ON_PURPOSE = [
    ("t<1", "a comparison cannot be traced, so there are none",
     "Unexpected character '<' at position 1"),
    ("t==1", "a comparison cannot be traced, so there are none",
     "Unexpected character '=' at position 1"),
    ("1 if t else 2", "no conditional; a value that switches uses step or clip",
     "expected an operator, got 'if' at position 2"),
    ("not t", "no boolean operators",
     "'not' at position 0 is not a value an expression of time can read"),
    ("7%3", "spelled mod(7, 3); '%' is not an operator here",
     "Unexpected character '%' at position 1"),
    ("7//2", "no floor division", "got '/' at position 2"),
    (
        "2^3",
        "'^' is XOR in Python and a power to most readers -- refused "
        "rather than quietly read as either",
        "Unexpected character '^' at position 1",
    ),
    ("0x10", "no hexadecimal literals", "expected an operator, got 'x10' at position 1"),
    ("1__0", "an underscore may only sit between digits, as in Python",
     "'1__0' at position 0 is not a number"),
]


def _py(expr):
    return float(eval(expr, {"__builtins__": {}}, dict(_PY_NS)))


def _aquakin(expr):
    return float(compile_time_expression(expr)(T))


@pytest.mark.parametrize("expr,python", AGREES_WITH_PYTHON)
def test_reads_as_python_does(expr, python):
    expected = _py(python if python is not None else expr)
    got = _aquakin(expr)
    assert got == pytest.approx(expected, rel=1e-12, abs=0.0, nan_ok=True), expr


@pytest.mark.parametrize("expr,why,message", REFUSED_ON_PURPOSE)
def test_what_is_refused_on_purpose(expr, why, message):
    with pytest.raises(TimeExpressionError, match=re.escape(message)):
        compile_time_expression(expr)


def test_a_leading_zero_is_read_as_a_number_where_python_refuses_it():
    """The one place aquakin is the more permissive of the two.

    Python 3 rejects ``007`` outright (a Python 2 octal holdover). Nobody
    writing it in a model file means anything but seven, so it is read as seven
    rather than inheriting a restriction whose reason does not apply here.
    """
    with pytest.raises(SyntaxError, match="leading zeros in decimal integer literals"):
        eval("007")
    assert _aquakin("007") == 7.0


# --- the rate language reads the same text the same way ---------------------


def _value(node: ASTNode) -> float:
    """Evaluate a constant-only rate AST through the nodes' own contract."""
    children = node.children()
    if not children:
        return float(node.value)
    return float(node.op(tuple(_value(child) for child in children)))


@pytest.mark.parametrize(
    "expr",
    [
        "-2**2",
        "2**3**2",
        "2**-1",
        "1-2-3",
        "8/4/2",
        "1+2*3",
        "2*3**2",
        "-2*3**2",
        "(0-2)**2",
        "2---3",
        "max(1,2,3)",
        "max(1,2,)",
        "1_000+0",
    ],
)
def test_the_rate_language_agrees_with_the_time_language(expr):
    """One precedence, one tokenizer: the two must not read the same text
    differently, which is the whole reason they share a parser."""
    assert _value(parse_rate_expression(expr)) == pytest.approx(_py(expr), rel=1e-12)


def test_a_measure_refuses_what_it_cannot_be():
    """A measure is affine on purpose -- the ranking's linear map depends on it
    -- so it refuses arithmetic the other two languages allow."""
    for text, message in (
        ("B = [A]**2", "so it has no powers (at position 7)"),
        ("B = [A]*[C]", "cannot multiply two measured quantities (at position 7)"),
        ("B = 2/[A]", "can only divide by a number (at position 5)"),
        ("B = [A] + 1", "adds the constant 1"),
    ):
        with pytest.raises(MeasureSyntaxError, match=re.escape(message)):
            parse_measure(text)


def test_a_double_comma_is_still_an_error():
    """A trailing comma is allowed, as in Python; two in a row are not."""
    with pytest.raises(TimeExpressionError, match="got ',' at position 6"):
        compile_time_expression("max(1,,2)")
    with pytest.raises(ParseError, match="Unexpected token ',' at position 6"):
        parse_rate_expression("max(1,,2)")
