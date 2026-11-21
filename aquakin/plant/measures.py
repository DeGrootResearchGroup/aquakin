"""What a sensor on a plant stream measures, as a linear map of the stream.

A plant observable (:class:`~aquakin.plant.calibrate.PlantObservable`) reads
*channels* of a stream. Each channel is a weighted sum of the stream's species
concentrations plus multiples of its flow and temperature, so every measurement
a calibration, a Fisher information or a sensor ranking uses is linear in the
stream and differentiable in the plant state and parameters. The channels are
meant to correspond to what a laboratory or an online sensor measures:

- a **species** name of the stream's model (that concentration);
- ``"Q"``: the stream's flow; ``"T"``: its temperature (the stream must carry
  one, i.e. an influent with a temperature);
- ``"COD"``: total COD, as a COD test measures it: electron acceptors
  (dissolved oxygen, nitrate) count none, nor do dissolved hydrogen and
  methane;
- ``"TKN"``: total Kjeldahl nitrogen, ammonium plus organic nitrogen;
- ``"NOx"``: nitrate plus nitrite nitrogen;
- ``"TN"``: total nitrogen, ``TKN + NOx``. Dissolved nitrogen gas, nitrous oxide
  and nitric oxide are in none of the nitrogen channels (a TN analysis does
  not see them);
- ``"TP"``: total phosphorus;
- ``"TSS"``: total suspended solids, from the model's own TSS species (``XTSS``
  or ``XSS``) or, for the ASM1 family, the benchmark convention
  ``0.75 × (XS + XI + XB_H + XB_A + XP)``;
- a :class:`Measure`: any other weighted sum (a custom TSS factor, BOD₅, ...);
- a **measure expression**, the same weighted sum written as text --
  ``"BOD5 = 0.65*[SS] + 0.65*[XS]"`` -- so a custom channel can be given in a
  study file or typed into the flowsheet editor. Species go in brackets, the
  measurements above are named bare, and the combination must stay linear (see
  :func:`parse_measure`).

COD, TKN, NOx, TN and TP come from the model's composition table
(:func:`aquakin.composition_table`), in g/m³. Their weights depend on
composition parameters such as ``i_XB``; a :class:`MeasureMap` carries the
weights at the model's defaults and their dependence on those parameters (exact
for the content fractions, which enter affinely; linearized at the defaults
otherwise), so a channel follows the parameters a run uses, and a fit of
``i_XB`` sees total nitrogen move with it. Gas-phase states (a digester's
``S_gas_*`` headspace) are in no measurement, and dissolved hydrogen and
methane are not in COD (they leave a sample before its COD is measured).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from aquakin.core.hints import did_you_mean
from aquakin.core.lexer import Token, spell, tokenize

__all__ = [
    "KEYWORDS",
    "Measure",
    "MeasureMap",
    "MeasureSyntaxError",
    "channel_name",
    "measure_map",
    "parse_measure",
]

#: The channel names that are measurements rather than species.
KEYWORDS = ("Q", "T", "COD", "TKN", "NOx", "TN", "TP", "TSS")
_COMPOSITION = ("COD", "TKN", "NOx", "TN", "TP")
_TSS_SPECIES = ("XTSS", "XSS")
# Dissolved gases carrying COD that a sample loses before its COD is measured.
_GAS_COD = ("S_h2", "S_ch4", "SH2", "SCH4")


@dataclass(frozen=True)
class Measure:
    """A custom measured channel: a weighted sum of species plus flow and temperature.

    Parameters
    ----------
    name : str
        A label for the channel (used in messages).
    weights : mapping of str to float, optional
        Weight per species of the stream's model; species not listed weigh 0.
    keywords : mapping of str to float, optional
        Weight per measurement of :data:`KEYWORDS` -- so a channel can be built
        on ``COD``, ``TN``, ``TSS``, ... rather than on species alone (soluble
        COD, a BOD estimated from COD). Their own weights follow the model's
        composition parameters, so a custom channel built on ``TN`` moves with a
        fitted ``i_XB`` exactly as ``TN`` itself does.
    flow : float, optional
        Weight of the stream's flow.
    temperature : float, optional
        Weight of the stream's temperature.

    Examples
    --------
    >>> bod5 = Measure("BOD5", {"SS": 0.65, "XS": 0.65})
    >>> PlantObservable("effluent", ("SNH", bod5))
    >>> soluble_cod = Measure("sCOD", {"XS": -1.0, "XI": -1.0}, keywords={"COD": 1.0})

    See Also
    --------
    parse_measure : the same measure written as text, ``"BOD5 = 0.65*[SS] + 0.65*[XS]"``.
    """

    name: str
    weights: Mapping[str, float] = field(default_factory=dict)
    keywords: Mapping[str, float] = field(default_factory=dict)
    flow: float = 0.0
    temperature: float = 0.0

    def __post_init__(self):
        # Stored as sorted pairs so the measure is hashable like the rest of an observable.
        object.__setattr__(self, "weights", tuple(sorted(dict(self.weights).items())))
        object.__setattr__(self, "keywords", tuple(sorted(dict(self.keywords).items())))
        for keyword, _ in self.keywords:
            if keyword not in KEYWORDS:
                hint = did_you_mean(keyword, list(KEYWORDS))
                raise ValueError(
                    f"measure '{self.name}' weighs '{keyword}', which is not one of "
                    f"{KEYWORDS}; a species goes in weights=.{hint}"
                )


@dataclass(frozen=True)
class MeasureMap:
    """The linear map from a stream to its measured channels.

    A stream with concentrations ``C``, flow ``Q`` and temperature ``T`` measures
    ``C @ W(θ) + Q * q + T * t``, where ``W(θ) = W + Σ_i (θ_i - θ0_i) dW[i]`` over
    the model parameters ``params`` the weights depend on (exact when the
    dependence is affine, as for every content fraction).

    Attributes
    ----------
    W : numpy.ndarray
        Species weights at the model's default parameters, ``(n_species, k)``.
    q, t : numpy.ndarray
        Flow and temperature weights, ``(k,)``.
    params : tuple of str
        The model's parameter names the weights depend on.
    theta0 : numpy.ndarray
        Their default values, ``(P,)``.
    dW : numpy.ndarray
        ``dW/dθ`` for each, ``(P, n_species, k)``.
    """

    W: np.ndarray
    q: np.ndarray
    t: np.ndarray
    params: tuple[str, ...]
    theta0: np.ndarray
    dW: np.ndarray

    @property
    def uses_temperature(self) -> bool:
        """Whether a channel reads the stream's temperature."""
        return bool(np.any(self.t))


# ---------------------------------------------------------------------------
# Measure expressions
# ---------------------------------------------------------------------------


class MeasureSyntaxError(ValueError):
    """Raised when a measure expression cannot be read.

    A :class:`ValueError`, so a caller that validates user input with a plain
    ``except ValueError`` keeps working.
    """


_NAME = re.compile(r"[A-Za-z_]\w*\Z")


@dataclass
class _Terms:
    """A combination being built: its species, measurement and constant parts."""

    species: dict[str, float] = field(default_factory=dict)
    keywords: dict[str, float] = field(default_factory=dict)
    constant: float = 0.0

    @property
    def measured(self) -> bool:
        """Whether anything of the stream is read (so the terms are not a number)."""
        return bool(self.species or self.keywords)

    def scaled(self, factor: float) -> _Terms:
        return _Terms(
            {s: w * factor for s, w in self.species.items()},
            {k: w * factor for k, w in self.keywords.items()},
            self.constant * factor,
        )

    def added(self, other: _Terms, sign: float) -> _Terms:
        out = _Terms(dict(self.species), dict(self.keywords), self.constant + sign * other.constant)
        for part, source in ((out.species, other.species), (out.keywords, other.keywords)):
            for key, weight in source.items():
                part[key] = part.get(key, 0.0) + sign * weight
        return out


class _ExpressionParser:
    """Recursive descent over a measure expression, evaluating to :class:`_Terms`.

    The grammar is the arithmetic of a weighted sum::

        expr    := term (('+' | '-') term)*
        term    := factor (('*' | '/') factor)*
        factor  := ('+' | '-') factor | primary
        primary := number | '[' species ']' | measurement | '(' expr ')'

    A product needs one side to be a plain number and a divisor must be one, so
    whatever the expression says is linear in the stream -- which is what lets a
    channel be a column of the measurement matrix, and what the sensor ranking
    and Fisher information rely on.
    """

    def __init__(self, text: str, offset: int = 0) -> None:
        self.text = text
        self.tokens = tokenize(text, error=MeasureSyntaxError, offset=offset)
        self.i = 0

    def _peek(self) -> Token:
        return self.tokens[self.i]

    def _take(self, kind: str) -> Token:
        token = self._peek()
        if token.kind != kind:
            raise MeasureSyntaxError(self._unexpected(token, f"expected {spell(kind)}"))
        self.i += 1
        return token

    def _unexpected(self, token: Token, expected: str) -> str:
        what = spell("EOF") if token.kind == "EOF" else f"{token.value!r}"
        return f"{expected}, got {what} at position {token.pos}"

    def parse(self) -> _Terms:
        if self._peek().kind == "EOF":
            raise MeasureSyntaxError("the expression is empty")
        terms = self._expression()
        token = self._peek()
        if token.kind != "EOF":
            raise MeasureSyntaxError(self._unexpected(token, "expected '+' or '-'"))
        return terms

    def _expression(self) -> _Terms:
        terms = self._term()
        while self._peek().kind in ("PLUS", "MINUS"):
            sign = 1.0 if self._peek().kind == "PLUS" else -1.0
            self.i += 1
            terms = terms.added(self._term(), sign)
        return terms

    def _term(self) -> _Terms:
        terms = self._factor()
        while self._peek().kind in ("STAR", "SLASH"):
            operator = self._peek()
            self.i += 1
            right = self._factor()
            if operator.kind == "STAR":
                if not terms.measured:
                    terms = right.scaled(terms.constant)
                elif not right.measured:
                    terms = terms.scaled(right.constant)
                else:
                    raise MeasureSyntaxError(
                        f"a measure is a weighted sum, so it cannot multiply two measured "
                        f"quantities (at position {operator.pos}); one side of a '*' must "
                        f"be a number"
                    )
            else:
                if right.measured:
                    raise MeasureSyntaxError(
                        f"a measure is a weighted sum, so it can only divide by a number "
                        f"(at position {operator.pos})"
                    )
                if right.constant == 0.0:
                    raise MeasureSyntaxError(f"division by zero at position {operator.pos}")
                terms = terms.scaled(1.0 / right.constant)
        return terms

    def _factor(self) -> _Terms:
        token = self._peek()
        if token.kind in ("PLUS", "MINUS"):
            self.i += 1
            factor = self._factor()
            return factor if token.kind == "PLUS" else factor.scaled(-1.0)
        return self._primary()

    def _primary(self) -> _Terms:
        token = self._peek()
        if token.kind == "NUMBER":
            self.i += 1
            return _Terms(constant=float(token.value))
        if token.kind == "LBRACK":
            self.i += 1
            name = self._take("IDENT").value
            self._take("RBRACK")
            return _Terms(species={name: 1.0})
        if token.kind == "LPAREN":
            self.i += 1
            terms = self._expression()
            self._take("RPAREN")
            return terms
        if token.kind == "IDENT":
            self.i += 1
            if token.value not in KEYWORDS:
                hint = did_you_mean(token.value, list(KEYWORDS))
                raise MeasureSyntaxError(
                    f"'{token.value}' at position {token.pos} is not one of {KEYWORDS}; "
                    f"write a species in brackets, as '[{token.value}]'.{hint}"
                )
            return _Terms(keywords={token.value: 1.0})
        raise MeasureSyntaxError(self._unexpected(token, "expected a number, '[species]' or '('"))


def parse_measure(text: str) -> Measure:
    """Read a measure expression -- ``"name = weighted sum"`` -- as a :class:`Measure`.

    The expression names the stream's species in brackets and the measurements
    of :data:`KEYWORDS` bare, combines them with ``+``, ``-``, ``*``, ``/`` and
    brackets, and must stay **linear** in the stream: a ``*`` needs a plain
    number on one side and a ``/`` a plain number on the right. A constant term
    is refused -- a channel that reads nothing of the stream measures nothing.

    The reading is independent of any model, so an expression can be checked for
    sense when it is written (in a study file, or in the flowsheet editor) and
    the species it names resolved later, against the model of the stream it is
    read on.

    Parameters
    ----------
    text : str
        The expression, e.g. ``"BOD5 = 0.65*[SS] + 0.65*[XS]"``.

    Returns
    -------
    Measure

    Raises
    ------
    MeasureSyntaxError
        If the expression has no name, cannot be read, is not linear, or leaves
        a constant term.

    Examples
    --------
    >>> parse_measure("BOD5 = 0.65*[SS] + 0.65*[XS]")
    Measure(name='BOD5', weights=(('SS', 0.65), ('XS', 0.65)), keywords=(), flow=0.0, temperature=0.0)
    >>> parse_measure("sCOD = COD - [XS] - [XI]").keywords
    (('COD', 1.0),)
    >>> parse_measure("Qm3 = Q / 1000").flow
    0.001
    """
    if not isinstance(text, str):
        raise TypeError(f"a measure expression is a string; got {text!r}")
    name, equals, body = text.partition("=")
    name = name.strip()
    if not equals:
        raise MeasureSyntaxError(
            f"a measure is written 'name = weighted sum', as "
            f"'BOD5 = 0.65*[SS] + 0.65*[XS]'; {text!r} has no '='"
        )
    if not name:
        raise MeasureSyntaxError(f"the measure in {text!r} has no name before its '='")
    if not _NAME.match(name):
        raise MeasureSyntaxError(
            f"'{name}' is not a usable channel name: a name starts with a letter or '_' and "
            f"holds only letters, digits and '_' (it becomes a column of the results)"
        )
    # The name and '=' are already taken off, so the parser is told where the
    # body starts: a position in a message points into the text the user wrote.
    terms = _ExpressionParser(body, len(text) - len(body)).parse()
    terms.species = {s: w for s, w in terms.species.items() if w}
    terms.keywords = {k: w for k, w in terms.keywords.items() if w}
    if terms.constant:
        raise MeasureSyntaxError(
            f"measure '{name}' adds the constant {terms.constant:g}; a channel is a weighted "
            f"sum of what the stream carries, with no constant term"
        )
    if not terms.measured:
        raise MeasureSyntaxError(f"measure '{name}' reads nothing of the stream")
    flow = terms.keywords.pop("Q", 0.0)
    temperature = terms.keywords.pop("T", 0.0)
    return Measure(
        name,
        weights=terms.species,
        keywords=terms.keywords,
        flow=flow,
        temperature=temperature,
    )


def is_measure_expression(channel) -> bool:
    """Whether a channel is a measure expression rather than a plain name.

    An expression is told apart by its ``'='``, which no species or measurement
    name can hold.
    """
    return isinstance(channel, str) and "=" in channel


def channel_name(channel) -> str:
    """The name a channel is reported under (its results column).

    A species or measurement is its own name, a :class:`Measure` its ``name``,
    and an expression the name before its ``'='``. Cheap enough to call while
    naming columns: an expression is not parsed, only split.
    """
    if isinstance(channel, Measure):
        return channel.name
    if is_measure_expression(channel):
        return channel.split("=", 1)[0].strip()
    return channel


def as_measure(channel):
    """A measure expression as a :class:`Measure`; any other channel unchanged."""
    return parse_measure(channel) if is_measure_expression(channel) else channel


# ---------------------------------------------------------------------------
# The channel definitions
# ---------------------------------------------------------------------------


def _gas_phase(model) -> set[str]:
    return {s for s in model.species if s.startswith("S_gas_")}


def _nitrogen_classes(model) -> tuple[set[str], set[str]]:
    """``(NOx species, dissolved nitrogen gases)`` of a model, by species name.

    ``SNO`` is nitrate in the ASM1 family but nitric oxide in a model that names
    nitrate ``SNO3``.
    """
    species = set(model.species)
    nitric_oxide_is_sno = "SNO3" in species
    nox = {"SNO3", "SNOX", "SNO2"} & species
    gases = {"SN2", "SN2O"} & species
    if "SNO" in species:
        (gases if nitric_oxide_is_sno else nox).add("SNO")
    return nox, gases


def _composition_weights(model, keyword: str, params) -> np.ndarray:
    from aquakin.utils.composition import canonical_content

    component = {"COD": "COD", "TP": "P"}.get(keyword, "N")
    try:
        w = np.asarray(
            canonical_content(model, component, electron_acceptor_cod=False, params=params), float
        )
    except KeyError as exc:
        raise KeyError(
            f"no {keyword} for model '{model.name}': {exc.args[0] if exc.args else exc}"
        ) from None
    excluded = set(_gas_phase(model))
    nox, gases = _nitrogen_classes(model)
    if keyword == "COD":
        w = np.maximum(w, 0.0)  # electron acceptors (oxygen, nitrate) count none
        excluded |= set(_GAS_COD) & set(model.species)
    elif keyword == "TKN":
        excluded |= nox | gases
    elif keyword == "NOx":
        excluded |= set(model.species) - nox
    elif keyword == "TN":
        excluded |= gases
    for s in excluded:
        w[model.species_index[s]] = 0.0
    return w


def _tss_weights(model) -> np.ndarray:
    from aquakin.plant._constants import ASM1_TSS_FACTOR, ASM1_TSS_SPECIES

    w = np.zeros(model.n_species)
    for name in _TSS_SPECIES:
        if name in model.species_index:
            w[model.species_index[name]] = 1.0
            return w
    if all(s in model.species_index for s in ASM1_TSS_SPECIES):
        for s in ASM1_TSS_SPECIES:
            w[model.species_index[s]] = ASM1_TSS_FACTOR
        return w
    raise KeyError(
        f"model '{model.name}' has no TSS species ({', '.join(_TSS_SPECIES)}) and is not of the "
        f"ASM1 family; give its TSS as Measure('TSS', {{species: factor, ...}})"
    )


def _keyword(model, keyword: str, params) -> tuple[np.ndarray, float, float]:
    """One measurement keyword's weights, with no species/keyword ambiguity.

    A :class:`Measure` names its keywords explicitly, so unlike a bare channel
    name a keyword here is never shadowed by a species of the same name.
    """
    zero = np.zeros(model.n_species)
    if keyword == "Q":
        return zero, 1.0, 0.0
    if keyword == "T":
        return zero, 0.0, 1.0
    if keyword in _COMPOSITION:
        w = _composition_weights(model, keyword, params)
        if not np.any(w):
            raise KeyError(f"model '{model.name}' carries nothing measured as {keyword}")
        return w, 0.0, 0.0
    if keyword == "TSS":
        return _tss_weights(model), 0.0, 0.0
    raise KeyError(f"'{keyword}' is not one of {KEYWORDS}")  # pragma: no cover - Measure checks


def _measure(model, channel: Measure, params) -> tuple[np.ndarray, float, float]:
    """A custom measure's weights: its species, plus each keyword's own weights."""
    w = np.zeros(model.n_species)
    for species, weight in channel.weights:
        if species not in model.species_index:
            hint = did_you_mean(species, list(model.species))
            raise KeyError(
                f"measure '{channel.name}' weighs '{species}', which is not a species of "
                f"model '{model.name}'.{hint}"
            )
        w[model.species_index[species]] = float(weight)
    q, t = float(channel.flow), float(channel.temperature)
    for keyword, weight in channel.keywords:
        try:
            kw, kq, kt = _keyword(model, keyword, params)
        except KeyError as exc:
            message = exc.args[0] if exc.args else str(exc)
            raise KeyError(f"measure '{channel.name}' weighs {keyword}, but {message}") from None
        w = w + float(weight) * kw
        q += float(weight) * kq
        t += float(weight) * kt
    return w, q, t


def _channel(model, channel, params) -> tuple[np.ndarray, float, float]:
    """One channel's species weights ``(n_species,)``, flow and temperature weights."""
    zero = np.zeros(model.n_species)
    if isinstance(channel, Measure):
        return _measure(model, channel, params)
    if not isinstance(channel, str):
        raise TypeError(
            f"a channel is a species name, one of {KEYWORDS}, or a Measure; got {channel!r}"
        )
    if channel in model.species_index:
        if channel in KEYWORDS:
            raise KeyError(
                f"'{channel}' is both a species of model '{model.name}' and a measurement "
                f"keyword; use Measure('{channel}', {{'{channel}': 1.0}}) for the species"
            )
        w = zero.copy()
        w[model.species_index[channel]] = 1.0
        return w, 0.0, 0.0
    if channel == "Q":
        return zero, 1.0, 0.0
    if channel == "T":
        return zero, 0.0, 1.0
    if channel in _COMPOSITION:
        w = _composition_weights(model, channel, params)
        if not np.any(w):
            raise KeyError(f"model '{model.name}' carries nothing measured as {channel}")
        return w, 0.0, 0.0
    if channel == "TSS":
        return _tss_weights(model), 0.0, 0.0
    hint = did_you_mean(channel, [*model.species, *KEYWORDS])
    raise KeyError(
        f"Unknown observed channel '{channel}': neither a species of model '{model.name}' "
        f"nor one of {KEYWORDS}.{hint}"
    )


def _weights(model, channels, params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = [_channel(model, c, params) for c in channels]
    return (
        np.stack([w for w, _, _ in cols], axis=1),
        np.asarray([q for _, q, _ in cols]),
        np.asarray([t for _, _, t in cols]),
    )


def _follows_composition(model, channel) -> bool:
    """Whether a channel's weights move with the model's composition parameters."""
    if isinstance(channel, Measure):
        return any(k in _COMPOSITION for k, _ in channel.keywords)
    return channel in _COMPOSITION and channel not in model.species_index


def measure_map(model, channels) -> MeasureMap:
    """The linear map from a stream of ``model`` to its measured ``channels``.

    Parameters
    ----------
    model : CompiledModel
        The stream's model.
    channels : sequence or None
        The channels (see the module docstring) -- species names, measurement
        keywords, :class:`Measure` objects or measure expressions; ``None``
        measures every species.

    Returns
    -------
    MeasureMap

    Raises
    ------
    KeyError
        If a channel names no species or measurement of ``model``, or a
        measurement the model cannot give (``TSS`` of a digester).
    ValueError
        If ``channels`` is empty, or a measure expression cannot be read
        (:class:`MeasureSyntaxError`).
    """
    n = model.n_species
    if channels is None:
        return MeasureMap(np.eye(n), np.zeros(n), np.zeros(n), (), np.zeros(0), np.zeros((0, n, n)))
    # Expressions are read once here, so every consumer below (and the dW probe,
    # which re-weighs the same channels) works on resolved measures.
    channels = tuple(as_measure(c) for c in channels)
    if not channels:
        raise ValueError("an observable must measure at least one channel")
    theta0 = np.asarray(model.default_parameters(), float)
    W, q, t = _weights(model, channels, theta0)
    params, values, slopes = [], [], []
    if any(_follows_composition(model, c) for c in channels):
        # Which parameters move the weights, and by how much. Composition contents
        # are affine in their fractions, so a central difference gives the exact
        # slope; a dependence that is not affine (a molecular weight in a
        # precipitate's P content) is linearized at the defaults.
        for i, name in enumerate(model.parameters):
            step = max(abs(theta0[i]) * 1e-3, 1e-6)
            up, down = theta0.copy(), theta0.copy()
            up[i] += step
            down[i] -= step
            slope = (_weights(model, channels, up)[0] - _weights(model, channels, down)[0]) / (
                2 * step
            )
            slope[np.abs(slope) < 1e-12 * max(1.0, float(np.abs(W).max()))] = 0.0
            if not np.any(slope):
                continue
            params.append(name)
            values.append(theta0[i])
            slopes.append(slope)
    dW = np.stack(slopes) if slopes else np.zeros((0, n, len(channels)))
    return MeasureMap(W, q, t, tuple(params), np.asarray(values, float), dW)
