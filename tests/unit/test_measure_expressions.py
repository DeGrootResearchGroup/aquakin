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


@pytest.mark.parametrize("text", ["x = 1e5*[SS]", "x = 1E5*[SS]", "x = 1e+5*[SS]"])
def test_an_exponent_is_read_with_or_without_its_sign(text):
    """The scanner takes the sign only when one is there, so an unsigned
    exponent has to reach the digits just the same."""
    assert parse_measure(text) == Measure("x", {"SS": 1e5})


@pytest.mark.parametrize("text", ["x = 1e*[SS]", "x = 1e+*[SS]"])
def test_an_e_with_no_digits_is_not_an_exponent(text):
    """Deliberate: the scanner only takes an exponent when a digit follows, so
    "1e" is the number 1 followed by the name "e" -- which reads as a parse
    error rather than a bare float() ValueError."""
    with pytest.raises(MeasureSyntaxError, match=re.escape("got 'e' at position 5")):
        parse_measure(text)


def test_a_malformed_number_is_named():
    """The scanner takes digits and dots together, so a second dot reaches
    ``float`` and must come back as a measure error, not a bare ValueError."""
    with pytest.raises(
        MeasureSyntaxError, match=re.escape("'1.2.3' at position 4 is not a number")
    ):
        parse_measure("x = 1.2.3*[SS]")


def test_flow_and_temperature_may_be_given_as_keywords():
    """``parse_measure`` puts Q and T in their own fields, but a Measure built
    by hand may name them as keywords, and must weigh the same thing."""
    import aquakin
    from aquakin.plant.measures import measure_map

    model = aquakin.load_model("asm1")
    by_keyword = measure_map(model, (Measure("x", keywords={"Q": 2.0, "T": 3.0}),))
    by_field = measure_map(model, (Measure("x", flow=2.0, temperature=3.0),))
    assert by_keyword.q == by_field.q == 2.0
    assert by_keyword.t == by_field.t == 3.0
    assert by_keyword.uses_temperature


def test_tss_is_weighed_like_any_other_keyword_in_a_custom_channel():
    """TSS comes from the model's own species rather than its composition table,
    so a channel built on it takes a different route than COD or TN."""
    import aquakin
    from aquakin.plant.measures import measure_map

    model = aquakin.load_model("asm1")
    scaled = measure_map(model, ("x = 0.75*TSS",)).W[:, 0]
    plain = measure_map(model, ("TSS",)).W[:, 0]
    assert scaled.any()
    assert scaled == pytest.approx(0.75 * plain)


def test_a_keyword_the_model_cannot_give_names_the_measure():
    """asm1 carries no phosphorus, so a channel built on TP cannot be weighed --
    the error has to say which measure asked for it."""
    import aquakin
    from aquakin.plant.measures import measure_map

    model = aquakin.load_model("asm1")
    with pytest.raises(KeyError, match="measure 'x' weighs TP"):
        measure_map(model, ("x = TP",))
    with pytest.raises(KeyError, match="carries nothing measured as TP"):
        measure_map(model, ("x = TP",))


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
