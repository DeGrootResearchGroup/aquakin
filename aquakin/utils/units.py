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

The check is **opt-in and advisory**: :meth:`CompiledNetwork.check_units` never
fails a load. A unit string that is blank or that this module cannot parse is
treated as *unknown* and skipped, so the uneven parameter annotations in the
shipped networks never raise a false alarm. Only an actual inconsistency between
two *known* units is reported.

This is distinct from :mod:`aquakin.core.units`, which only *formats* unit
strings for display.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple, Optional

from aquakin.core.nodes import (
    AddNode,
    MaxNode,
    ArrheniusNode,
    ASTNode,
    ConditionNode,
    ConstantNode,
    DivideNode,
    MonodInhibitionNode,
    MonodInhibitionRatioNode,
    MonodNode,
    MonodRatioNode,
    MultiplyNode,
    NegateNode,
    ParamNode,
    PowerNode,
    SafeDivideNode,
    pHInhibitNode,
    pHSwitchNode,
    SpeciesNode,
    SubtractNode,
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

    def __init__(self, mapping: Optional[dict] = None):
        cleaned = {
            k: Fraction(v) for k, v in (mapping or {}).items() if Fraction(v) != 0
        }
        object.__setattr__(self, "tokens", frozenset(cleaned.items()))

    def as_dict(self) -> dict:
        return dict(self.tokens)

    @property
    def is_dimensionless(self) -> bool:
        return not self.tokens

    def __mul__(self, other: "Dimension") -> "Dimension":
        d = self.as_dict()
        for k, v in other.tokens:
            d[k] = d.get(k, Fraction(0)) + v
        return Dimension(d)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        d = self.as_dict()
        for k, v in other.tokens:
            d[k] = d.get(k, Fraction(0)) - v
        return Dimension(d)

    def __pow__(self, exponent) -> "Dimension":
        e = Fraction(exponent)
        return Dimension({k: v * e for k, v in self.tokens})

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "-"
        items = sorted(self.tokens)
        num = [(k, v) for k, v in items if v > 0]
        den = [(k, -v) for k, v in items if v < 0]

        def fmt(pairs):
            return ".".join(
                k if v == 1 else f"{k}^{v}" for k, v in pairs
            )

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
_UNIT_SYMBOLS = ["kmol", "mol", "min", "kg", "bar", "Pa", "g", "m", "L", "d",
                 "s", "h", "M", "K"]
# Chemical "currency" tokens. ``O2`` carries the meaning of oxygen-as-O2;
# ``TSS`` is suspended-solids mass (a distinct currency in the ASM2d/ASM3
# models). These are the distinct base dimensions the check exists to keep
# apart. Longest first (``COD`` before ``C``, ``TSS`` before ``S``).
_CURRENCY_TOKENS = ["COD", "TSS", "O2", "N", "P", "S", "C"]

_SUPERSCRIPT_TO_ASCII = str.maketrans(
    {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
     "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+"}
)

_WORD_RE = re.compile(r"[A-Za-z0-9_^.+\-]+")
_NUMBER_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")


def _match_longest(word: str, pos: int, options) -> Optional[str]:
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


def _tokenize_word(word: str) -> Optional[Dimension]:
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


def parse_units(text: str) -> Optional[Dimension]:
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
    s = (text.translate(_SUPERSCRIPT_TO_ASCII)
         .replace("·", "*").replace("⋅", "*").strip())
    if s == "" :
        return None
    if s == "-":
        return DIMENSIONLESS
    dim = _parse_product(s)
    if dim is None:
        return None
    return _normalise_molarity(dim)


def _parse_product(s: str) -> Optional[Dimension]:
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
    species_dim: dict          # species name -> Dimension|None
    param_dim: dict            # namespaced param name -> Dimension|None
    condition_dim: dict        # condition field -> Dimension|None
    warnings: list

    def warn(self, location: str, detail: str) -> None:
        self.warnings.append(UnitWarning(self.reaction, location, detail))


def _param_dim(ctx: _Ctx, local_name: str) -> Optional[Dimension]:
    """Resolve a (local) parameter name to its declared dimension, trying the
    reaction-local namespaced key first, then the network-level key."""
    for key in (f"{ctx.reaction}.{local_name}", local_name):
        if key in ctx.param_dim:
            return ctx.param_dim[key]
    return None


def _infer(node: ASTNode, ctx: _Ctx) -> Optional[Dimension]:
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
            ctx.warn(f"'{op}' operands",
                     f"added/subtracted terms differ in units: "
                     f"{lo} vs {ro}")
        # Result carries either side's dimension (they should agree); prefer a
        # known one so the rest of the expression can still be checked.
        return lo if lo is not None else ro
    if isinstance(node, MaxNode):
        # max(a, b) is dimensionally like add/subtract: the operands must share a
        # unit and the result carries it. A bare literal (the ``0`` in a
        # ``max(0, .)`` clip) adopts the sibling's dimension silently.
        lo = _infer(node.a, ctx)
        ro = _infer(node.b, ctx)
        if isinstance(node.a, ConstantNode):
            return ro
        if isinstance(node.b, ConstantNode):
            return lo
        if lo is not None and ro is not None and lo != ro:
            ctx.warn("'max' operands",
                     f"max() operands differ in units: {lo} vs {ro}")
        return lo if lo is not None else ro
    if isinstance(node, MultiplyNode):
        lo = _infer(node.left, ctx)
        ro = _infer(node.right, ctx)
        return None if lo is None or ro is None else lo * ro
    if isinstance(node, DivideNode):
        lo = _infer(node.left, ctx)
        ro = _infer(node.right, ctx)
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
            ctx.warn("Monod term",
                     f"saturation argument and half-saturation constant differ "
                     f"in units: {dx} vs {dk}")
        return DIMENSIONLESS
    if isinstance(node, (MonodRatioNode, MonodInhibitionRatioNode)):
        da = _infer(node.A, ctx)
        db = _infer(node.B, ctx)
        dk = _infer(node.K, ctx)
        ratio = None if da is None or db is None else da / db
        if ratio is not None and dk is not None and ratio != dk:
            ctx.warn("Monod-ratio term",
                     f"saturation ratio and half-saturation constant differ in "
                     f"units: {ratio} vs {dk}")
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
_MASS_TOKENS = {"g", "kg", "mol", "kmol"}
_VOLUME_LITRE = "L"
_LENGTH_TOKEN = "m"


def _root_issue(dim: Dimension) -> Optional[str]:
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
    other = {k: v for k, v in rest.items()
             if k not in _MASS_TOKENS and k not in _CURRENCY_TOKENS}
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


def check_network_units(network, *, check_root: bool = True) -> list:
    """Run :func:`check_rate_units` over every reaction in a compiled network.

    Parameters
    ----------
    network : CompiledNetwork
        The compiled network to check.
    check_root : bool, default True
        Whether to also assert each rate resolves to ``currency/volume/time``.

    Returns
    -------
    list of UnitWarning
        Every dimensional-consistency finding across all reactions, in
        reaction order. An empty list means no inconsistency was found among the
        *declared, parseable* units (it is not a proof of correctness, since
        unknown units are skipped).
    """
    species_dim = {name: parse_units(u)
                   for name, u in network.species_units.items()}
    param_dim = {name: parse_units(u)
                 for name, u in network.parameter_units.items()}
    condition_dim = {name: parse_units(u)
                     for name, u in network.condition_units.items()}
    warnings: list = []
    inv_time: dict = {}
    for name, ast in zip(network.reaction_names, network.rate_asts):
        warnings.extend(check_rate_units(
            ast, name, species_dim, param_dim, condition_dim,
            check_root=check_root,
        ))
        # Record each rate's inverse-time token for the cross-reaction check
        # below (reuse the same root inference; cheap and advisory).
        root = _infer(ast, _Ctx(name, species_dim, param_dim, condition_dim, []))
        if root is not None:
            inv = [k for k, v in root.as_dict().items()
                   if k in _TIME_TOKENS and v < 0]
            if len(inv) == 1:
                inv_time[name] = inv[0]

    # Cross-reaction time-unit consistency. Every rate constant drives dC/dt
    # against the *same* integration time, so all rates must share one
    # inverse-time unit. A network mixing, say, 1/d and 1/s rates is malformed --
    # its RHS sums terms on inconsistent time bases -- yet each such rate passes
    # the per-rate root check on its own. Flag the disagreement once, at network
    # scope, so an author sees it. (Runs whenever the roots are determinable,
    # independent of ``check_root``.)
    distinct = set(inv_time.values())
    if len(distinct) > 1:
        detail = ("rate constants disagree on the time unit, so the RHS is not "
                  "dimensionally consistent: "
                  + ", ".join(f"{r} -> 1/{u}" for r, u in sorted(inv_time.items())))
        warnings.append(UnitWarning("(network)", "time unit", detail))
    return warnings
