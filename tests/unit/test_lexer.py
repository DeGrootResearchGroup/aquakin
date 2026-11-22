"""The one tokenizer behind every aquakin expression language.

It was three near-identical copies until the languages were put on a single
lexer; these pin the behaviour all three relied on, in one place.
"""

import re

import pytest

from aquakin.core.lexer import SYMBOLS, Token, spell, tokenize


def kinds(text):
    return [t.kind for t in tokenize(text)]


def test_every_symbol_is_read():
    """So a grammar that uses a symbol gets a token for it, and one that does
    not refuses it where it parses rather than here."""
    for char, kind in SYMBOLS.items():
        assert kinds(char) == [kind, "EOF"]
    assert kinds("**") == ["POW", "EOF"]
    # ** is taken before *, so a power is never two multiplications.
    assert kinds("a**b") == ["IDENT", "POW", "IDENT", "EOF"]
    assert kinds("a*b") == ["IDENT", "STAR", "IDENT", "EOF"]


def test_a_stream_of_everything():
    assert kinds("2.5e-3*[SS] + {pH} / f(x, y)") == [
        "NUMBER", "STAR", "LBRACK", "IDENT", "RBRACK",
        "PLUS", "LBRACE", "IDENT", "RBRACE",
        "SLASH", "IDENT", "LPAREN", "IDENT", "COMMA", "IDENT", "RPAREN", "EOF",
    ]  # fmt: skip


@pytest.mark.parametrize("text", ["1", "1.5", ".5", "1e3", "1E3", "1e-3", "1e+3", "12.34e-5"])
def test_numbers_that_are_one_token(text):
    assert kinds(text) == ["NUMBER", "EOF"]
    assert float(tokenize(text)[0].value) == float(text)


@pytest.mark.parametrize("text", ["1e", "1e+", "1E-"])
def test_an_exponent_needs_a_digit(text):
    """The subtle rule, and the reason this lives in one place: "1e" is the
    number 1 followed by the name "e", so a caller reports a parse error naming
    the token instead of this raising on a malformed float."""
    assert kinds(text)[0] == "NUMBER"
    assert tokenize(text)[0].value == "1"
    assert "IDENT" in kinds(text)


def test_whitespace_is_skipped_and_positions_are_the_text_s():
    tokens = tokenize("  a +  b")
    assert [t.kind for t in tokens] == ["IDENT", "PLUS", "IDENT", "EOF"]
    assert [t.pos for t in tokens] == [2, 4, 7, 8]


def test_the_offset_shifts_every_position():
    """For a caller that has already taken a prefix off the text, so a position
    in a message points into what the user actually wrote."""
    plain = tokenize("a + b")
    shifted = tokenize("a + b", offset=10)
    assert [t.pos + 10 for t in plain] == [t.pos for t in shifted]


def test_each_language_raises_its_own_error():
    class Mine(ValueError):
        pass

    with pytest.raises(Mine, match=re.escape("Unexpected character '@' at position 2")):
        tokenize("a @ b", error=Mine)
    with pytest.raises(Mine, match=re.escape("'1.2.3' at position 0 is not a number")):
        tokenize("1.2.3", error=Mine)
    # The default keeps it a plain ValueError.
    with pytest.raises(ValueError, match="Unexpected character '@' at position 0"):
        tokenize("@")


def test_a_kind_is_spelled_as_a_message_should_name_it():
    """A kind is an internal name; a reader needs the character they typed."""
    assert spell("RPAREN") == "')'"
    assert spell("POW") == "'**'"
    assert spell("NUMBER") == "a number"
    assert spell("IDENT") == "a name"
    assert spell("EOF") == "the end of the expression"
    assert spell("WHATEVER") == "'WHATEVER'"


def test_a_token_carries_what_it_was_read_from():
    token = tokenize("abc")[0]
    assert token == Token("IDENT", "abc", 0)
