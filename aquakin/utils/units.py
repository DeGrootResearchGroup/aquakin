"""Dimensional ("unit") consistency checking for rate expressions.

A *currency-aware* dimensional analysis. Units are modelled as a free abelian
group over **currency tokens** -- ``{g, mol, m, L, d, s, COD, N, O2, P, S, C,
...}`` -- where ``COD`` / ``N`` / ``O2`` are **distinct base dimensions**, not
labels on mass. This catches the bug a plain SI 7-vector cannot: ``g_COD/m3``
and ``g_N/m3`` are both mass/volume, but they carry different *currencies*, so
mixing them (a dropped factor, a Monod term comparing two different currencies)
is a real authoring error.

Scope is the **rate-expression layer only**. Stoichiometry consistency is a
*conservation* question -- ASM/ADM stoichiometric coefficients are deliberately
cross-currency (a yield is ``g_COD/g_COD``, ``i_N`` is ``g_N/g_COD``) -- and is
handled by per-currency conservation in :mod:`aquakin.utils.balance`, not here.

The check is **opt-in and advisory**: :meth:`CompiledModel.check_units` never
fails a load. A unit string that is blank or that this module cannot parse is
treated as *unknown* and skipped, so the uneven parameter annotations in the
shipped models never raise a false alarm. Only an actual inconsistency between
two *known* units is reported.

This is distinct from :mod:`aquakin.core.units`, which only *formats* unit
strings for display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple

from aquakin.core.nodes import (
    AddNode,
    ArrheniusNode,
    ASTNode,
    ConditionNode,
    ConstantNode,
    DivideNode,
    ExpNode,
    LogNode,
    MaxNode,
    MinNode,
    MonodInhibitionNode,
    MonodInhibitionRatioNode,
    MonodNode,
    MonodRatioNode,
    MultiplyNode,
    NegateNode,
    ParamNode,
    PowerNode,
    SafeDivideNode,
    SpeciesNode,
    SqrtNode,
    SubtractNode,
    pHInhibitNode,
    pHSwitchNode,
)

# --- the currency-token dimension algebra -----------------------------------


@dataclass(frozen=True)
class Dimension:
    """A free-abelian-group element over currency tokens: ``token -> exponent``.

    Exponents are :class:`fractions.Fraction` so half-order kinetics
    (``[S]**0.5``) are exact. An empty map is *dimensionless*. ``Dimension`` is
    immutable and hashable; ``*`` / ``/`` add / subtract exponents and ``**``
    scales them, so it forms the unit group used to propagate units through a
    rate expression. ``None`` (never a ``Dimension``) is reserved for *unknown*.
    """

    tokens: frozenset  # frozenset of (token, Fraction) pairs with nonzero exp

    def __init__(self, mapping: dict | None = None):
        cleaned = {k: Fraction(v) for k, v in (mapping or {}).items() if Fraction(v) != 0}
        object.__setattr__(self, "tokens", frozenset(cleaned.items()))

    def as_dict(self) -> dict:
        return dict(self.tokens)

    @property
    def is_dimensionless(self) -> bool:
        return not self.tokens

    def __mul__(self, other: Dimension) -> Dimension:
        d = self.as_dict()
        for k, v in other.tokens:
            d[k] = d.get(k, Fraction(0)) + v
        return Dimension(d)

    def __truediv__(self, other: Dimension) -> Dimension:
        d = self.as_dict()
        for k, v in other.tokens:
            d[k] = d.get(k, Fraction(0)) - v
        return Dimension(d)

    def __pow__(self, exponent) -> Dimension:
        e = Fraction(exponent)
        return Dimension({k: v * e for k, v in self.tokens})

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "-"
        items = sorted(self.tokens)
        num = [(k, v) for k, v in items if v > 0]
        den = [(k, -v) for k, v in items if v < 0]

        def fmt(pairs):
            return ".".join(k if v == 1 else f"{k}^{v}" for k, v in pairs)

        if not den:
            return fmt(num)
        return f"{fmt(num) or '1'}/{fmt(den)}"

    def __repr__(self) -> str:
        return f"Dimension({self})"


DIMENSIONLESS = Dimension()


# --- the tolerant unit-string parser ----------------------------------------

# Unit symbols that may carry an exponent (a trailing signed integer, e.g. ``m3``
# = m^3, ``s-1`` = s^-1, or a caret exponent ``^0.5``). Longest first so the
# scan prefers ``mol`` over ``m`` and ``min`` over ``m``. ``M`` is molarity and
# is normalised to ``mol/L`` after parsing.
_UNIT_SYMBOLS = [
    "kmol",
    "mmol",
    "mol",
    "min",
    "kg",
    "mg",
    "bar",
    "Pa",
    "g",
    "m",
    "L",
    "d",
    "s",
    "h",
    "M",
    "K",
]
# Chemical "currency" tokens. ``O2`` carries the meaning of oxygen-as-O2;
# ``TSS`` is suspended-solids mass (a distinct currency in the ASM2d/ASM3
# models). These are the distinct base dimensions the check exists to keep
# apart. Longest first (``COD`` before ``C``, ``TSS`` before ``S``).
_CURRENCY_TOKENS = ["COD", "TSS", "O2", "N", "P", "S", "C"]

_SUPERSCRIPT_TO_ASCII = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "⁺": "+",
    }
)

_WORD_RE = re.compile(r"[A-Za-z0-9_^.+\-]+")
_NUMBER_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")


def _match_longest(word: str, pos: int, options) -> str | None:
    for opt in options:
        if word.startswith(opt, pos):
            return opt
    return None


def _read_exponent(word: str, pos: int):
    """Read an exponent at ``pos``: ``^<frac>`` or a trailing signed integer.

    Returns ``(Fraction, new_pos)``; defaults to ``(1, pos)`` when none is
    present. ``None`` signals a malformed exponent (the whole unit is unknown).
    """
    if pos < len(word) and word[pos] == "^":
        m = re.match(r"\^([+-]?\d+(?:\.\d+)?)", word[pos:])
        if not m:
            return None
        return Fraction(m.group(1)), pos + m.end()
    m = re.match(r"[+-]?\d+", word[pos:])
    if m and m.group(0) not in ("",):
        return Fraction(m.group(0)), pos + m.end()
    return Fraction(1), pos


def _tokenize_word(word: str) -> Dimension | None:
    """Turn one space/operator-free unit word into a :class:`Dimension`.

    Handles the regular dialects: ``g_COD``, ``gCOD``, ``g_O2``, ``m3``, ``mol``,
    ``M-1``, ``s-1``, ``gO2^0.5``. A bare number is dimensionless. Returns
    ``None`` (unknown) for anything it cannot fully consume -- e.g. the dotted
    ``COD.m-3`` ADM dialect -- so unparseable annotations are skipped, not
    flagged.
    """
    if _NUMBER_RE.match(word):
        return DIMENSIONLESS
    d: dict = {}
    pos = 0
    n = len(word)
    while pos < n:
        if word[pos] == "_":
            pos += 1
            continue
        sym = _match_longest(word, pos, _UNIT_SYMBOLS)
        if sym is not None:
            pos += len(sym)
            exp = _read_exponent(word, pos)
            if exp is None:
                return None
            e, pos = exp
            d[sym] = d.get(sym, Fraction(0)) + e
            continue
        cur = _match_longest(word, pos, _CURRENCY_TOKENS)
        if cur is not None:
            pos += len(cur)
            # A caret exponent may bind to a currency token (``O2^0.5``).
            if pos < n and word[pos] == "^":
                exp = _read_exponent(word, pos)
                if exp is None:
                    return None
                e, pos = exp
            else:
                e = Fraction(1)
            d[cur] = d.get(cur, Fraction(0)) + e
            continue
        return None
    return Dimension(d)


def _normalise_molarity(dim: Dimension) -> Dimension:
    """Rewrite the molarity token ``M`` as ``mol/L`` so the ``mol/L`` and
    ``M-1 s-1`` dialects resolve to the same group elements."""
    md = dim.as_dict()
    m = md.pop("M", None)
    if m is None:
        return dim
    out = Dimension(md)
    return out * Dimension({"mol": m, "L": -m})


def parse_units(text: str) -> Dimension | None:
    """Parse a unit string into a :class:`Dimension`, or ``None`` if unknown.

    Parameters
    ----------
    text : str
        A unit string in any of the shipped dialects, e.g. ``"g_COD/m3"``,
        ``"1/d"``, ``"M-1 s-1"``, ``"m3/(g_COD*d)"``, ``"gO2^0.5/m/d"``,
        ``"mol/L"``. The display form is accepted too: Unicode superscripts
        (``m³``) and the multiplication dot (``g_COD·m⁻³`` == ``g_COD*m-3``).

    Returns
    -------
    Dimension or None
        The parsed dimension. ``"-"`` (or ``"1"``) is the dimensionless
        ``Dimension()``. A blank string, or any string this tolerant parser
        cannot fully consume, returns ``None`` (treated as *unknown* / skipped by
        the check) rather than raising.
    """
    if text is None:
        return None
    # Accept the display multiplication dot (U+00B7 '·', U+22C5 '⋅') as '*', so
    # the prettified form round-trips through the parser.
    s = text.translate(_SUPERSCRIPT_TO_ASCII).replace("·", "*").replace("⋅", "*").strip()
    if s == "":
        return None
    if s == "-":
        return DIMENSIONLESS
    dim = _parse_product(s)
    if dim is None:
        return None
    return _normalise_molarity(dim)


def _parse_product(s: str) -> Dimension | None:
    """Parse a product/quotient of unit words with ``*`` ``/`` and parentheses.

    ``*`` and juxtaposition multiply; ``/`` divides; both associate left to
    right (``a/b/c`` == ``a.b^-1.c^-1``); parentheses group. Returns ``None`` if
    any word or the bracket structure is unparseable.
    """
    toks = _lex(s)
    if toks is None:
        return None
    parser = _ProductParser(toks)
    try:
        dim = parser.parse()
    except _ParseError:
        return None
    if parser.pos != len(toks):
        return None
    return dim


class _ParseError(Exception):
    pass


def _lex(s: str):
    """Split a unit string into ``(`` ``)`` ``/`` ``*`` and word tokens.

    A space separates words (implicit multiply). Returns ``None`` on an
    unexpected character so the whole unit is treated as unknown.
    """
    toks = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c in "()/*":
            toks.append(c)
            i += 1
        else:
            m = _WORD_RE.match(s, i)
            if not m or m.start() != i:
                return None
            toks.append(m.group(0))
            i = m.end()
    return toks


class _ProductParser:
    """Tiny left-to-right parser for the unit-string product grammar."""

    def __init__(self, toks):
        self.toks = toks
        self.pos = 0

    def _peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def parse(self) -> Dimension:
        dim = self._factor()
        while True:
            t = self._peek()
            if t == "/":
                self.pos += 1
                dim = dim / self._factor()
            elif t == "*":
                self.pos += 1
                dim = dim * self._factor()
            elif t is not None and t != ")":
                # juxtaposition, e.g. "M-1 s-1" or "(..)(..)" -> multiply
                dim = dim * self._factor()
            else:
                break
        return dim

    def _factor(self) -> Dimension:
        t = self._peek()
        if t == "(":
            self.pos += 1
            dim = self.parse()
            if self._peek() != ")":
                raise _ParseError("unbalanced parenthesis")
            self.pos += 1
            return dim
        if t is None or t in ")/*":
            raise _ParseError("expected a unit word")
        self.pos += 1
        word = _tokenize_word(t)
        if word is None:
            raise _ParseError(f"unparseable unit word {t!r}")
        return word


# --- scale-aware comparison ---------------------------------------------------

#: Spellings of a Celsius temperature unit. They are not parsed (``C`` is the
#: carbon currency), so code reading a declared temperature unit checks this set.
CELSIUS_SPELLINGS = frozenset(
    {"c", "degc", "deg_c", "°c", "℃", "celsius", "degree_celsius", "degrees_celsius"}
)

# Each scaled symbol -> (base symbol, base exponent, factor), so that one unit of
# the symbol equals ``factor * base**exponent``. Unlisted symbols (``g``, ``mol``,
# ``m``, ``s``, ``Pa``, ``K``) are already base symbols.
_TO_BASE = {
    "kg": ("g", 1, 1.0e3),
    "mg": ("g", 1, 1.0e-3),
    "kmol": ("mol", 1, 1.0e3),
    "mmol": ("mol", 1, 1.0e-3),
    "L": ("m", 3, 1.0e-3),
    "d": ("s", 1, 86400.0),
    "h": ("s", 1, 3600.0),
    "min": ("s", 1, 60.0),
    "bar": ("Pa", 1, 1.0e5),
}


def _to_base_units(dim: Dimension) -> tuple[Dimension, float]:
    """Express ``dim`` in base symbols, returning ``(base_dimension, factor)``.

    One unit of ``dim`` equals ``factor`` units of ``base_dimension``: for
    example ``kg/m3`` is ``(g/m3, 1000)`` and ``mg/L`` is ``(g/m3, 1)``.
    Chemical currency tokens (``COD``, ``N``, ...) carry no scale and pass
    through unchanged.
    """
    out: dict = {}
    factor = 1.0
    for tok, exp in dim.tokens:
        base, base_exp, f = _TO_BASE.get(tok, (tok, 1, 1.0))
        out[base] = out.get(base, Fraction(0)) + exp * base_exp
        factor *= f ** float(exp)
    return Dimension(out), factor


def _drop_currency(dim: Dimension) -> Dimension:
    return Dimension({k: v for k, v in dim.tokens if k not in _CURRENCY_TOKENS})


def _has_currency(dim: Dimension) -> bool:
    return any(k in _CURRENCY_TOKENS for k, _ in dim.tokens)


def _comparable(a: str, b: str):
    """Parse two unit strings into ``((base_a, factor_a), (base_b, factor_b))``
    for comparison, dropping the chemical currencies unless both state one;
    ``None`` when either string is blank or unparseable."""
    da, db = parse_units(a), parse_units(b)
    if da is None or db is None:
        return None
    if not (_has_currency(da) and _has_currency(db)):
        da, db = _drop_currency(da), _drop_currency(db)
    return _to_base_units(da), _to_base_units(db)


def conversion_factor(from_unit: str, to_unit: str) -> float | None:
    """Factor ``f`` with ``x_to = f * x_from`` between two units of one quantity.

    Converts through base symbols, so ``kgCOD/m3`` to ``g_COD/m3`` is ``1000``
    and ``mol/m3`` to ``kmol/m3`` is ``0.001``. Chemical currencies are compared
    only when both strings name one, as in :func:`units_match`. No molar mass is
    applied: a mass and an amount of substance are different quantities.

    Parameters
    ----------
    from_unit, to_unit : str
        Unit strings in any dialect :func:`parse_units` accepts.

    Returns
    -------
    float or None
        The factor, or ``None`` when either string is blank or cannot be
        parsed.

    Raises
    ------
    ValueError
        If the two units measure different quantities (``g/m3`` and
        ``mol/m3``, or ``g_N/m3`` and ``g_COD/m3``).

    Examples
    --------
    >>> conversion_factor("kgCOD/m3", "g_COD/m3")
    1000.0
    """
    pair = _comparable(from_unit, to_unit)
    if pair is None:
        return None
    (base_a, fa), (base_b, fb) = pair
    if base_a != base_b:
        raise ValueError(
            f"cannot convert '{from_unit}' to '{to_unit}': they measure different "
            f"quantities ({base_a} vs {base_b})."
        )
    return fa / fb


def units_match(a: str, b: str, *, rtol: float = 1e-9) -> bool | None:
    """Whether two unit strings denote the same unit, scale included.

    Unlike the rate-expression check (which compares symbols exactly), this
    converts both sides to base symbols first, so ``mg/L`` matches ``g/m3`` and
    ``m3/d`` matches ``m3/d`` but ``kgCOD/m3`` does not match ``g_COD/m3`` (a
    factor of 1000) and ``1/s`` does not match ``1/d``. Chemical currencies are
    compared only when **both** strings name one, so a bare ``g/m3`` or
    ``mg/L`` matches ``g_COD/m3`` (the currency is simply unstated).

    Parameters
    ----------
    a, b : str
        Unit strings in any dialect :func:`parse_units` accepts.
    rtol : float, optional
        Relative tolerance on the scale factor.

    Returns
    -------
    bool or None
        ``True``/``False``, or ``None`` when either string is blank or cannot be
        parsed (unknown, so not comparable).

    Examples
    --------
    >>> units_match("mg/L", "g_COD/m3")
    True
    >>> units_match("kgCOD/m3", "g_COD/m3")
    False
    """
    pair = _comparable(a, b)
    if pair is None:
        return None
    (base_a, fa), (base_b, fb) = pair
    if base_a != base_b:
        return False
    return abs(fa - fb) <= rtol * max(abs(fa), abs(fb))


# --- unit propagation through a rate AST + the check ------------------------


class UnitWarning(NamedTuple):
    """One dimensional-consistency finding for a rate expression.

    Attributes
    ----------
    reaction : str
        The reaction whose ``rate:`` expression the finding is in.
    location : str
        A short description of the offending sub-expression / node.
    detail : str
        What was expected versus what was found.
    """

    reaction: str
    location: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.reaction}] {self.location}: {self.detail}"


@dataclass
class _Ctx:
    reaction: str
    species_dim: dict  # species name -> Dimension|None
    param_dim: dict  # namespaced param name -> Dimension|None
    condition_dim: dict  # condition field -> Dimension|None
    warnings: list

    def warn(self, location: str, detail: str) -> None:
        self.warnings.append(UnitWarning(self.reaction, location, detail))


def _param_dim(ctx: _Ctx, local_name: str) -> Dimension | None:
    """Resolve a (local) parameter name to its declared dimension, trying the
    reaction-local namespaced key first, then the model-level key."""
    for key in (f"{ctx.reaction}.{local_name}", local_name):
        if key in ctx.param_dim:
            return ctx.param_dim[key]
    return None


def _additive_terms(node: ASTNode):
    """Yield the terms of a (possibly nested) ``a + b + ...`` sum."""
    if isinstance(node, AddNode):
        yield from _additive_terms(node.left)
        yield from _additive_terms(node.right)
    else:
        yield node


def _is_literal_half_saturation(node: DivideNode) -> bool:
    """True for ``c / (c + X + ...)`` with the same numeric literal ``c`` as the
    numerator and as one term of the additive denominator -- a Monod inhibition
    term ``K / (K + X)`` whose constant ``K`` is written inline."""
    if not isinstance(node.left, ConstantNode) or not isinstance(node.right, AddNode):
        return False
    return any(
        isinstance(t, ConstantNode) and t.value == node.left.value
        for t in _additive_terms(node.right)
    )


def _infer(node: ASTNode, ctx: _Ctx) -> Dimension | None:
    """Infer the dimension of a rate sub-expression, accumulating warnings.

    Returns ``None`` when a leaf's units are unknown (undeclared / unparseable),
    which propagates so the root check is skipped rather than guessed.
    """
    if isinstance(node, ConstantNode):
        return DIMENSIONLESS
    if isinstance(node, SpeciesNode):
        return ctx.species_dim.get(node.name)
    if isinstance(node, ParamNode):
        return _param_dim(ctx, node.name)
    if isinstance(node, ConditionNode):
        return ctx.condition_dim.get(node.field_name)
    if isinstance(node, NegateNode):
        return _infer(node.operand, ctx)

    if isinstance(node, (AddNode, SubtractNode)):
        lo = _infer(node.left, ctx)
        ro = _infer(node.right, ctx)
        # A bare numeric literal added to / subtracted from a dimensioned term is
        # a deliberate regulariser (``[S_va] + [S_bu] + 1e-6`` guards a division),
        # not a dropped-factor bug, so it adopts the sibling's dimension silently.
        # A *dimensionless-returning function* (Monod, pH switch) is not a literal
        # and is still flagged against a dimensioned sibling.
        if isinstance(node.left, ConstantNode):
            return ro
        if isinstance(node.right, ConstantNode):
            return lo
        if lo is not None and ro is not None and lo != ro:
            op = "+" if isinstance(node, AddNode) else "-"
            ctx.warn(f"'{op}' operands", f"added/subtracted terms differ in units: {lo} vs {ro}")
        # Result carries either side's dimension (they should agree); prefer a
        # known one so the rest of the expression can still be checked.
        return lo if lo is not None else ro
    if isinstance(node, (MaxNode, MinNode)):
        # max(a, b) / min(a, b) are dimensionally like add/subtract: the
        # operands must share a unit and the result carries it. A bare literal
        # (the ``0`` in a ``max(0, .)`` clip, the ceiling in a ``min(c, .)``)
        # adopts the sibling's dimension silently.
        name = "max" if isinstance(node, MaxNode) else "min"
        lo = _infer(node.a, ctx)
        ro = _infer(node.b, ctx)
        if isinstance(node.a, ConstantNode):
            return ro
        if isinstance(node.b, ConstantNode):
            return lo
        if lo is not None and ro is not None and lo != ro:
            ctx.warn(f"'{name}' operands", f"{name}() operands differ in units: {lo} vs {ro}")
        return lo if lo is not None else ro
    if isinstance(node, (ExpNode, LogNode)):
        # Transcendental: the argument has to be dimensionless (you cannot
        # exponentiate a concentration), and the result is dimensionless. This
        # is the check that catches an attenuation written ``exp(-a * z)`` with
        # ``a`` in the wrong units, or a free-energy term missing its RT.
        name = "exp" if isinstance(node, ExpNode) else "log"
        dx = _infer(node.x, ctx)
        if dx is not None and not dx.is_dimensionless:
            ctx.warn(f"'{name}' argument", f"{name}() argument is not dimensionless: {dx}")
        return DIMENSIONLESS
    if isinstance(node, SqrtNode):
        # sqrt halves the argument's dimension, exactly as ``** 0.5`` does.
        dx = _infer(node.x, ctx)
        return None if dx is None else dx ** Fraction(1, 2)
    if isinstance(node, MultiplyNode):
        lo = _infer(node.left, ctx)
        ro = _infer(node.right, ctx)
        return None if lo is None or ro is None else lo * ro
    if isinstance(node, DivideNode):
        ro = _infer(node.right, ctx)
        if _is_literal_half_saturation(node):
            # ``c / (c + X)``: an inhibition switch with its half-saturation
            # constant written as a bare literal. The literal carries the
            # summand's units (it is added to ``X``), so the ratio is
            # dimensionless rather than ``1 / [X]``.
            return None if ro is None else DIMENSIONLESS
        lo = _infer(node.left, ctx)
        return None if lo is None or ro is None else lo / ro
    if isinstance(node, SafeDivideNode):
        # safe_div(num, denom) is num/denom dimensionally (the zero-guard only
        # affects the value at denom == 0, not the units).
        lo = _infer(node.num, ctx)
        ro = _infer(node.denom, ctx)
        return None if lo is None or ro is None else lo / ro
    if isinstance(node, PowerNode):
        base = _infer(node.left, ctx)
        if base is None:
            return None
        if isinstance(node.right, ConstantNode):
            return base ** Fraction(node.right.value).limit_denominator(1000)
        # A non-constant exponent has no static dimension; cannot check.
        return None

    if isinstance(node, (MonodNode, MonodInhibitionNode)):
        dx = _infer(node.X, ctx)
        dk = _infer(node.K, ctx)
        if dx is not None and dk is not None and dx != dk:
            ctx.warn(
                "Monod term",
                f"saturation argument and half-saturation constant differ in units: {dx} vs {dk}",
            )
        return DIMENSIONLESS
    if isinstance(node, (MonodRatioNode, MonodInhibitionRatioNode)):
        da = _infer(node.A, ctx)
        db = _infer(node.B, ctx)
        dk = _infer(node.K, ctx)
        ratio = None if da is None or db is None else da / db
        if ratio is not None and dk is not None and ratio != dk:
            ctx.warn(
                "Monod-ratio term",
                f"saturation ratio and half-saturation constant differ in units: {ratio} vs {dk}",
            )
        return DIMENSIONLESS
    if isinstance(node, (pHSwitchNode, pHInhibitNode)):
        return DIMENSIONLESS
    if isinstance(node, ArrheniusNode):
        # ``A * exp(-Ea / (R*T))``: the exponential is dimensionless, so the node
        # carries the dimension of its prefactor ``A``.
        return _infer(node.A, ctx)

    return None


# Tokens recognised in the canonical "concentration per time" root form.
_TIME_TOKENS = {"d", "s", "h", "min"}
_MASS_TOKENS = {"g", "kg", "mg", "mol", "kmol", "mmol"}
_VOLUME_LITRE = "L"
_LENGTH_TOKEN = "m"


def _root_issue(dim: Dimension) -> str | None:
    """Return a message if ``dim`` is not a ``currency / volume / time`` rate.

    A well-formed reaction rate is a concentration per time: ``g_X/m3/d`` (mass
    of a currency per cubic metre per day) or ``mol/L/s``. Returns ``None`` if
    ``dim`` matches that form, else a description of how it differs.
    """
    d = dim.as_dict()

    # time: exactly one time token at exponent -1.
    time = {k: v for k, v in d.items() if k in _TIME_TOKENS}
    if len(time) != 1 or next(iter(time.values())) != -1:
        return f"expected one inverse-time factor, found units {dim}"
    rest = {k: v for k, v in d.items() if k not in _TIME_TOKENS}

    # volume: either L^-1, or m^-3 (the length token to the -3).
    if rest.get(_VOLUME_LITRE, 0) == -1:
        rest.pop(_VOLUME_LITRE)
    elif rest.get(_LENGTH_TOKEN, 0) == -3:
        rest.pop(_LENGTH_TOKEN)
    else:
        return f"expected one inverse-volume factor (m3 or L), found units {dim}"

    # currency: one mass-unit^+1, optionally times one chemical currency^+1
    # (``mol`` alone is a valid molar currency).
    masses = {k: v for k, v in rest.items() if k in _MASS_TOKENS}
    chems = {k: v for k, v in rest.items() if k in _CURRENCY_TOKENS}
    other = {k: v for k, v in rest.items() if k not in _MASS_TOKENS and k not in _CURRENCY_TOKENS}
    if other:
        return f"unexpected factor(s) {Dimension(other)} in rate units {dim}"
    if len(masses) != 1 or next(iter(masses.values())) != 1:
        return f"expected one currency-mass factor, found units {dim}"
    if chems and (len(chems) != 1 or next(iter(chems.values())) != 1):
        return f"expected a single chemical currency, found units {dim}"
    return None


def check_rate_units(
    ast: ASTNode,
    reaction: str,
    species_dim: dict,
    param_dim: dict,
    condition_dim: dict,
    *,
    check_root: bool = True,
) -> list:
    """Check one rate AST for dimensional consistency; return ``UnitWarning``\\ s.

    The local rules (matching ``+``/``-`` operands, single-currency Monod terms)
    always run. The ``currency/volume/time`` root check runs when
    ``check_root`` is true *and* the whole expression's dimension is known (no
    unknown leaf), so an incomplete annotation never produces a spurious root
    warning.
    """
    ctx = _Ctx(reaction, species_dim, param_dim, condition_dim, [])
    root = _infer(ast, ctx)
    if check_root and root is not None:
        issue = _root_issue(root)
        if issue is not None:
            ctx.warn("rate root", issue)
    return ctx.warnings


def _same_quantity(a: Dimension, b: Dimension) -> bool:
    """Whether two dimensions are the same unit, scale included, comparing the
    chemical currencies only when both state one (as :func:`units_match`)."""
    if not (_has_currency(a) and _has_currency(b)):
        a, b = _drop_currency(a), _drop_currency(b)
    (base_a, fa), (base_b, fb) = _to_base_units(a), _to_base_units(b)
    return base_a == base_b and abs(fa - fb) <= 1e-9 * max(abs(fa), abs(fb))


def _unit_stoichiometry_species(model) -> dict:
    """``{reaction index: [species name, ...]}`` of the species whose
    stoichiometric coefficient is a fixed ``+1`` or ``-1``.

    Read from the static stoichiometry matrix, which holds the numeric
    coefficients only: a coefficient written as an expression is a zero
    placeholder there (it is evaluated per solve), so it never counts.
    """
    import numpy as np

    stoich = np.asarray(model.stoich_matrix)
    out: dict = {}
    for i, j in zip(*np.nonzero(np.isclose(np.abs(stoich), 1.0, rtol=0.0, atol=1e-12))):
        out.setdefault(int(i), []).append(model.species[int(j)])
    return out


def _stoichiometry_issue(root: Dimension, species: list, species_dim: dict) -> str | None:
    """Compare a rate's concentration unit with its unit-coefficient species.

    A species whose coefficient is ``+1`` or ``-1`` changes by exactly the rate,
    so the rate must be in that species' concentration per time. A reaction may
    have several such species in different currencies (one is enough to fix the
    rate's unit), so this reports only when the rate matches none of them.
    Returns a message, or ``None`` when consistent or not determinable.
    """
    time = [k for k, v in root.as_dict().items() if k in _TIME_TOKENS and v == -1]
    if len(time) != 1:
        return None
    conc = root * Dimension({time[0]: 1})
    known = [(sp, species_dim[sp]) for sp in species if species_dim.get(sp) is not None]
    if not known or any(_same_quantity(conc, dim) for _, dim in known):
        return None
    listing = ", ".join(f"'{sp}' in {dim}" for sp, dim in known)
    return (
        f"the rate is in {conc} per {time[0]}, but each species with a +/-1 "
        f"coefficient changes by exactly the rate and none is in {conc} ({listing})"
    )


def check_model_units(
    model,
    *,
    check_root: bool = True,
    check_stoichiometry: bool = True,
    include_acknowledged: bool = False,
) -> list:
    """Run :func:`check_rate_units` over every reaction in a compiled model.

    Parameters
    ----------
    model : CompiledModel
        The compiled model to check.
    check_root : bool, default True
        Whether to also assert each rate resolves to ``currency/volume/time``.
    check_stoichiometry : bool, default True
        Whether to also check each rate against the species whose
        stoichiometric coefficient is a fixed ``+1`` or ``-1``: such a species
        changes by exactly the rate, so the rate must be in its concentration
        per time. A reaction is flagged only when its rate matches none of them.
        Other coefficients are not checked (yields and conversion factors
        legitimately carry cross-currency units).
    include_acknowledged : bool, default False
        Also return the findings of reactions whose YAML declares
        ``units_acknowledged:`` (a written reason the reaction's units are
        knowingly unconventional, e.g. a molar mass written as a bare number).
        By default they are left out, so the result lists only what still needs
        attention; the reasons are in ``model.units_acknowledged``.

    Returns
    -------
    list of UnitWarning
        Every dimensional-consistency finding across all reactions, in
        reaction order. An empty list means no inconsistency was found among the
        *declared, parseable* units (it is not a proof of correctness, since
        unknown units are skipped).
    """
    species_dim = {name: parse_units(u) for name, u in model.species_units.items()}
    param_dim = {name: parse_units(u) for name, u in model.parameter_units.items()}
    condition_dim = {name: parse_units(u) for name, u in model.condition_units.items()}
    warnings: list = []
    inv_time: dict = {}
    unit_species = _unit_stoichiometry_species(model) if check_stoichiometry else {}
    for i, (name, ast) in enumerate(zip(model.reaction_names, model.rate_asts)):
        warnings.extend(
            check_rate_units(
                ast,
                name,
                species_dim,
                param_dim,
                condition_dim,
                check_root=check_root,
            )
        )
        # Record each rate's inverse-time token for the cross-reaction check
        # below (reuse the same root inference; cheap and advisory).
        root = _infer(ast, _Ctx(name, species_dim, param_dim, condition_dim, []))
        if root is not None:
            inv = [k for k, v in root.as_dict().items() if k in _TIME_TOKENS and v < 0]
            if len(inv) == 1:
                inv_time[name] = inv[0]
            if i in unit_species:
                issue = _stoichiometry_issue(root, unit_species[i], species_dim)
                if issue is not None:
                    warnings.append(UnitWarning(name, "stoichiometry", issue))

    # Cross-reaction time-unit consistency. Every rate constant drives dC/dt
    # against the *same* integration time, so all rates must share one
    # inverse-time unit. A model mixing, say, 1/d and 1/s rates is malformed --
    # its RHS sums terms on inconsistent time bases -- yet each such rate passes
    # the per-rate root check on its own. Flag the disagreement once, at model
    # scope, so an author sees it. (Runs whenever the roots are determinable,
    # independent of ``check_root``.)
    distinct = set(inv_time.values())
    if len(distinct) > 1:
        detail = (
            "rate constants disagree on the time unit, so the RHS is not "
            "dimensionally consistent: "
            + ", ".join(f"{r} -> 1/{u}" for r, u in sorted(inv_time.items()))
        )
        warnings.append(UnitWarning("(model)", "time unit", detail))
    if not include_acknowledged:
        acknowledged = getattr(model, "units_acknowledged", None) or {}
        warnings = [w for w in warnings if w.reaction not in acknowledged]
    return warnings
