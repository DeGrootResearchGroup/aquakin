"""Measure expressions: ``"BOD5 = 0.65*[SS] + 0.65*[XS]"`` read as a ``Measure``.

The reading is model-independent, so these need no plant; that a custom channel
then weighs a stream correctly -- and follows the composition parameters -- is
``tests/integration/test_plant_measures.py``.
"""

import re

import pytest

from aquakin.plant import Measure, channel_name, parse_measure
from aquakin.plant.measures import KEYWORDS, MeasureSyntaxError, as_measure, is_measure_expression


def test_a_weighted_sum_of_species():
    assert parse_measure("BOD5 = 0.65*[SS] + 0.65*[XS]") == Measure(
        "BOD5", {"SS": 0.65, "XS": 0.65}
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The weight may sit on either side of the '*', or be left out entirely.
        ("x = [SS]*2", Measure("x", {"SS": 2.0})),
        ("x = 2*[SS]", Measure("x", {"SS": 2.0})),
        ("x = [SS]", Measure("x", {"SS": 1.0})),
        ("x = -[SS]", Measure("x", {"SS": -1.0})),
        ("x = [SS]/4", Measure("x", {"SS": 0.25})),
        ("x = [SS] - [XS]", Measure("x", {"SS": 1.0, "XS": -1.0})),
        # Brackets and a sign in front of them distribute over the sum.
        ("x = -(2*[SS] - [XS])/4", Measure("x", {"SS": -0.5, "XS": 0.25})),
        ("x = 1e-3*[SS]", Measure("x", {"SS": 1e-3})),
        # Repeated terms accumulate rather than overwrite.
        ("x = [SS] + [SS]", Measure("x", {"SS": 2.0})),
        # Flow and temperature are their own fields, not species weights.
        ("x = Q/1000", Measure("x", flow=1e-3)),
        ("x = T", Measure("x", temperature=1.0)),
        # Every other keyword is carried as a keyword.
        ("x = COD - [XS]", Measure("x", {"XS": -1.0}, keywords={"COD": 1.0})),
        ("x = 0.5*TN + 0.5*TKN", Measure("x", keywords={"TKN": 0.5, "TN": 0.5})),
    ],
)
def test_the_arithmetic_of_a_weighted_sum(text, expected):
    assert parse_measure(text) == expected


def test_whitespace_and_the_name_are_free_form():
    assert parse_measure("  sCOD=COD-[XS]  ") == parse_measure("sCOD = COD - [XS]")
    assert parse_measure("_x2 = [SS]").name == "_x2"


@pytest.mark.parametrize("keyword", KEYWORDS)
def test_every_keyword_is_a_term(keyword):
    assert parse_measure(f"x = {keyword}") == Measure(
        "x",
        keywords={keyword: 1.0} if keyword not in ("Q", "T") else {},
        flow=1.0 if keyword == "Q" else 0.0,
        temperature=1.0 if keyword == "T" else 0.0,
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("COD", "has no '='"),
        ("= [SS]", "no name"),
        ("x = ", "empty"),
        ("1x = [SS]", "not a usable channel name"),
        ("x.y = [SS]", "not a usable channel name"),
        ("x y = [SS]", "not a usable channel name"),
        # Linear only: the measurement matrix the ranking and Fisher tools use
        # is a column per channel, so a product or a reciprocal has no column.
        ("x = [SS]*[XS]", "cannot multiply two measured quantities"),
        ("x = [SS]*COD", "cannot multiply two measured quantities"),
        ("x = 1/[SS]", "only divide by a number"),
        ("x = [SS]/0", "division by zero"),
        # A channel reads the stream, and only the stream.
        ("x = [SS] + 3", "no constant term"),
        ("x = 2", "no constant term"),
        ("x = [SS] - [SS]", "reads nothing of the stream"),
        ("x = 0*[SS]", "reads nothing of the stream"),
        # Syntax.
        ("x = SS", "write a species in brackets, as '[SS]'"),
        ("x = [SS", "expected ']'"),
        ("x = ([SS]", "expected ')'"),
        ("x = [SS] )", "expected '+' or '-'"),
        ("x = + ", "expected a number"),
        ("x = [SS] +", "expected a number"),
        ("x = [2]", "expected 'name'"),
        ("x = [SS] @ 2", "unexpected character '@'"),
    ],
)
def test_what_is_refused(text, message):
    # Escaped: the messages quote brackets and operators, which are regex syntax.
    with pytest.raises(MeasureSyntaxError, match=re.escape(message)):
        parse_measure(text)


def test_a_syntax_error_is_a_value_error():
    """So a caller validating user input with ``except ValueError`` still fits."""
    assert issubclass(MeasureSyntaxError, ValueError)
    with pytest.raises(ValueError, match="no constant term"):
        parse_measure("x = 2")


@pytest.mark.parametrize(
    ("text", "position"),
    [("x = [SS]*[XS]", 8), ("x = 1/[SS]", 5), ("x = SS", 4), ("x = [SS] @ 2", 9)],
)
def test_a_position_points_into_the_text_the_user_wrote(text, position):
    """Not into the expression after the name, which is what the parser sees --
    the editor shows the whole line."""
    with pytest.raises(MeasureSyntaxError, match=f"position {position}"):
        parse_measure(text)


def test_an_unknown_keyword_is_hinted():
    """A keyword given straight to ``Measure`` is checked too, not only a parsed one."""
    with pytest.raises(ValueError, match="Did you mean: TN"):
        Measure("x", keywords={"TNN": 1.0})


def test_a_measure_is_hashable_like_any_channel():
    """An observable's channels are a hashable tuple, so a measure must be too."""
    assert hash(parse_measure("x = COD + [SS]")) == hash(
        Measure("x", {"SS": 1.0}, keywords={"COD": 1.0})
    )


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        ("SNH", "SNH"),
        ("COD", "COD"),
        ("BOD5 = 0.65*[SS]", "BOD5"),
        ("  spaced   =  [SS] ", "spaced"),
        (Measure("m", {"SS": 1.0}), "m"),
    ],
)
def test_a_channel_is_reported_under_its_name(channel, expected):
    assert channel_name(channel) == expected


def test_only_an_expression_holds_an_equals_sign():
    assert is_measure_expression("x = [SS]")
    assert not is_measure_expression("COD")
    assert not is_measure_expression(Measure("m"))
    # So only an expression is converted; every other channel is left alone.
    assert as_measure("COD") == "COD"
    assert as_measure("x = [SS]") == Measure("x", {"SS": 1.0})


def test_a_non_string_is_refused_clearly():
    with pytest.raises(TypeError, match="a measure expression is a string"):
        parse_measure(3.0)
