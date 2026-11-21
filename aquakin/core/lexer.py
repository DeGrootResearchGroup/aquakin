"""The one tokenizer behind aquakin's expression languages.

aquakin reads several small expression languages, each with its own grammar and
its own idea of what a name means:

- **rate expressions** (:mod:`aquakin.core.parser`) -- ``[species]``,
  ``{condition}``, a bare identifier as a rate constant, domain functions;
- **expressions of time** (:mod:`aquakin.plant.time_expr`) -- ``t`` and a table
  of maths functions, compiled to a callable;
- **measure expressions** (:mod:`aquakin.plant.measures`) -- a deliberately
  affine weighted sum of a stream.

What they do *not* differ in is how text becomes tokens: the same numbers, the
same identifiers, the same symbols. That part lives here, once, so a fix to it
is a fix everywhere.

Keeping it in one place matters more than it looks. The number scanner takes an
exponent **only when a digit follows it** (optionally after a sign), so ``1e``
is the number ``1`` followed by the name ``e`` and reads as a clean parse error
rather than a bare ``float()`` failure. That rule is easy to get subtly wrong
and was previously written out three times.

Each language keeps its own error type -- a message should name the language the
reader is writing in -- so :func:`tokenize` takes the class to raise.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SYMBOLS", "Token", "spell", "tokenize"]

#: Single-character symbols, by the token kind they produce. ``**`` is read as
#: ``POW`` before ``*`` is considered.
SYMBOLS = {
    "+": "PLUS",
    "-": "MINUS",
    "*": "STAR",
    "/": "SLASH",
    "(": "LPAREN",
    ")": "RPAREN",
    ",": "COMMA",
    "[": "LBRACK",
    "]": "RBRACK",
    "{": "LBRACE",
    "}": "RBRACE",
}


#: How a token kind is written in a message meant for whoever typed the
#: expression. A kind is an internal name; ``expected ')'`` is what a reader
#: needs, not ``expected 'RPAREN'``.
_SPELLING = {kind: f"'{char}'" for char, kind in SYMBOLS.items()} | {
    "POW": "'**'",
    "NUMBER": "a number",
    "IDENT": "a name",
    "EOF": "the end of the expression",
}


def spell(kind: str) -> str:
    """``kind`` as a message should name it (``RPAREN`` -> ``')'``)."""
    return _SPELLING.get(kind, repr(kind))


@dataclass(frozen=True)
class Token:
    """One token: its ``kind`` (a value of :data:`SYMBOLS`, or ``NUMBER``,
    ``IDENT`` or ``EOF``), the text it was read from, and where it started."""

    kind: str
    value: str
    pos: int


def tokenize(text: str, *, error: type[Exception] = ValueError, offset: int = 0) -> list[Token]:
    """Read ``text`` into tokens, ending with an ``EOF``.

    Every symbol of :data:`SYMBOLS` is read whether or not the caller's grammar
    uses it; a grammar that has no use for ``{`` refuses it where it parses,
    with a message in its own terms, rather than here.

    Parameters
    ----------
    text : str
        The expression.
    error : type, optional
        The exception class raised for an unreadable character or number --
        each language's own, so the message names what the reader is writing.
    offset : int, optional
        Added to every token's position. For a caller that has already taken a
        prefix off the text (a measure's ``"name = "``), so a position in a
        message points into what the user actually wrote.

    Returns
    -------
    list of Token

    Raises
    ------
    error
        If a character cannot start a token, or a run of digits and dots is not
        a number.

    Examples
    --------
    >>> [t.kind for t in tokenize("2*[SS]")]
    ['NUMBER', 'STAR', 'LBRACK', 'IDENT', 'RBRACK', 'EOF']
    """
    tokens: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == "*" and i + 1 < n and text[i + 1] == "*":
            tokens.append(Token("POW", "**", i + offset))
            i += 2
        elif c in SYMBOLS:
            tokens.append(Token(SYMBOLS[c], c, i + offset))
            i += 1
        elif c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = _scan_number(text, i, n)
            try:
                float(text[i:j])
            except ValueError:
                raise error(f"'{text[i:j]}' at position {i + offset} is not a number") from None
            tokens.append(Token("NUMBER", text[i:j], i + offset))
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(Token("IDENT", text[i:j], i + offset))
            i = j
        else:
            raise error(f"Unexpected character {c!r} at position {i + offset}")
    tokens.append(Token("EOF", "", n + offset))
    return tokens


def _scan_number(text: str, i: int, n: int) -> int:
    """Where the number starting at ``i`` ends.

    Digits and dots, then an exponent **only when a digit follows** it
    (optionally after a sign). So ``1e`` ends at the ``e``, leaving it to be
    read as a name: the caller then reports a parse error naming the token,
    instead of this raising on a malformed float.
    """
    j = i
    while j < n and (text[j].isdigit() or text[j] == "."):
        j += 1
    if j < n and text[j] in "eE":
        k = j + 1
        if k < n and text[k] in "+-":
            k += 1
        if k < n and text[k].isdigit():
            j = k
            while j < n and text[j].isdigit():
                j += 1
    return j
