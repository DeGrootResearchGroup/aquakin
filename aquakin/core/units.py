"""Format unit strings for display.

Network YAML keeps units in a plain ASCII form that is easy for an engineer to
type (``g_COD/m3``, ``M-1 s-1``). When those units are carried onto the compiled
network they are *prettified*: an exponent written as a trailing (optionally
signed) integer on a unit symbol becomes a Unicode superscript
(``m3`` -> ``m³``, ``s-1`` -> ``s⁻¹``). Chemical formulae embedded in a species
token are deliberately left untouched -- the ``O2`` in ``g_O2`` is a subscript,
not an exponent, so it stays ``g_O2`` (never ``g_O²``).
"""

from __future__ import annotations

import re

_SUPERSCRIPT = str.maketrans(
    {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "-": "⁻",
        "+": "⁺",
    }
)

# Unit symbols that may carry an exponent. Longer symbols are listed first so
# the alternation prefers ``mol`` over ``m`` and ``min`` over ``m``.
_UNIT_SYMBOLS = r"(?:mol|min|kg|Pa|m|s|d|h|L|g|M)"

# A unit symbol at a token boundary -- not preceded by a letter, digit, or
# underscore, so the ``O2`` inside ``g_O2`` (preceded by ``_``) is never matched
# -- directly followed by a signed integer exponent.
_EXP_RE = re.compile(rf"(?<![A-Za-z0-9_])({_UNIT_SYMBOLS})([+-]?\d+)")


def prettify_units(units: str) -> str:
    """Convert plain-ASCII exponents in a unit string to Unicode superscripts.

    Parameters
    ----------
    units : str
        A plain-ASCII unit string as written in a network YAML, e.g.
        ``"g_COD/m3"``, ``"M-1 s-1"``, ``"mol/m3"``, ``"-"``.

    Returns
    -------
    str
        The same string with unit exponents rendered as Unicode superscripts
        (``"g_COD/m³"``, ``"M⁻¹ s⁻¹"``, ``"mol/m³"``, ``"-"``). Chemical
        formulae embedded in a species token are unchanged, and the function is
        idempotent (already-prettified input is returned verbatim).

    Examples
    --------
    >>> prettify_units("g_O2/m3")
    'g_O2/m³'
    >>> prettify_units("M-1 s-1")
    'M⁻¹ s⁻¹'
    """
    if not units:
        return units
    return _EXP_RE.sub(lambda m: m.group(1) + m.group(2).translate(_SUPERSCRIPT), units)
