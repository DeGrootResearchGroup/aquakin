"""Control laws written as expressions, for :class:`~aquakin.plant.control.ExpressionController`.

A control law is a stateless output map: the value a controller publishes on
the signal bus, written as text over what the controller can see::

    "clip(offset + Kp*([SNH] - sp), 0, 240)"

The law reads

- ``[species]``, a species of the sensed stream's model;
- a measurement channel of :mod:`aquakin.plant.measures` by its bare name
  (``TSS``, ``COD``, ``TN``, ``TKN``, ``NOx``, ``TP``), with weights that
  follow the composition parameters of the run, as a calibration observable's
  do;
- the controller's ``constants`` by name;
- ``t``, the plant clock, and ``pi`` and ``e``;
- any other bare name as a **signal** another unit publishes, which lets one
  law cascade from another's output.

The grammar and the functions (``clip``, ``step``, ``min``, ``max``, the guarded
``sqrt`` and ``log``, ...) are those of an expression of time
(:mod:`aquakin.plant.time_expr`), read by the same parser. The law compiles once
to plain JAX arithmetic with no Python control flow, so it runs inside the
jitted, differentiated right-hand side.

The sensed stream's flow is not readable: signals are computed before the
stream sweep, from the sensed unit's state, so no flow exists yet.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from aquakin.core.hints import did_you_mean
from aquakin.core.lexer import Token
from aquakin.plant.measures import KEYWORDS, measure_map, weights_at
from aquakin.plant.time_expr import CONSTANTS, FUNCTIONS, ClosureParser

__all__ = ["CHANNELS", "ControlLaw", "ControlLawError", "compile_law"]

#: The measurement channels a law reads by bare name. The flow ``Q`` and
#: temperature ``T`` are not among them: signals are computed before the stream
#: sweep, from the sensed unit's state, which carries neither.
CHANNELS = tuple(k for k in KEYWORDS if k not in ("Q", "T"))

#: Names a law gives a meaning of its own, so a constant cannot take them.
_RESERVED = ("t", *CONSTANTS, *FUNCTIONS, *CHANNELS)


class ControlLawError(ValueError):
    """Raised when a control law cannot be read or does not fit its controller.

    A :class:`ValueError`, so a caller validating user input with a plain
    ``except ValueError`` keeps working.
    """


class _Inputs(NamedTuple):
    """What one evaluation of a law reads."""

    t: object
    C: object
    channels: object
    signals: Mapping


class _LawParser(ClosureParser):
    """A control law: the shared grammar, with the law's own leaves."""

    error = ControlLawError
    language = "a control law"

    def __init__(self, text: str, constants: Mapping[str, float], species: Mapping[str, int]):
        super().__init__(text)
        self.constants = constants
        self.species_index = species
        self.species: dict[str, int] = {}
        self.channels: list[str] = []
        self.signals: list[str] = []

    def primary_message(self, token: Token) -> str:
        return (
            f"expected a number, a name, '[species]' or '(', got {self.describe(token)} "
            f"at position {token.pos}"
        )

    def variable(self, token: Token) -> Callable:
        name = token.value
        if name in self.constants:
            return lambda x, v=float(self.constants[name]): v
        if name == "t":
            return lambda x: x.t
        if name in CHANNELS:
            if name not in self.channels:
                self.channels.append(name)
            return lambda x, j=self.channels.index(name): x.channels[j]
        if name not in self.signals:
            self.signals.append(name)
        return lambda x, s=name: x.signals[s]

    def bracket(self, token: Token) -> Callable:
        self.advance()
        name = self.expect("IDENT")
        self.expect("RBRACK")
        if self.species_index and name.value not in self.species_index:
            hint = did_you_mean(name.value, list(self.species_index))
            raise ControlLawError(
                f"'[{name.value}]' at position {token.pos} is not a species of the sensed "
                f"model.{hint}"
            )
        self.species.setdefault(name.value, self.species_index.get(name.value, -1))
        return lambda x, i=self.species[name.value]: x.C[i]


@dataclass(frozen=True)
class ControlLaw:
    """A compiled control law.

    Attributes
    ----------
    text : str
        The law as written.
    species : tuple of str
        The species it reads, in brackets.
    channels : tuple of str
        The measurement channels it reads.
    signals : tuple of str
        The signals it reads from the bus.
    """

    text: str
    species: tuple[str, ...]
    channels: tuple[str, ...]
    signals: tuple[str, ...]
    _fn: Callable
    _weights: tuple | None

    @property
    def reads_stream(self) -> bool:
        """Whether the law reads the sensed stream (a species or a channel)."""
        return bool(self.species or self.channels)

    def __call__(self, t, C, signals: Mapping, params) -> jnp.ndarray:
        """The law's value.

        Parameters
        ----------
        t : scalar
            The plant clock.
        C : jnp.ndarray or None
            The sensed stream's concentrations, ``(n_species,)``; ``None`` when
            the law reads nothing of it.
        signals : mapping
            The signals published so far.
        params : jnp.ndarray
            The sensed model's parameter vector, which the channel weights follow.

        Returns
        -------
        jnp.ndarray
            A scalar.
        """
        channels = None
        if self._weights is not None:
            W, idx, theta0, dW = self._weights
            channels = C @ weights_at(W, idx, theta0, dW, params)
        return jnp.asarray(self._fn(_Inputs(t, C, channels, signals)), dtype=float)


def compile_law(text: str, constants: Mapping[str, float], model=None, *, what: str) -> ControlLaw:
    """Read a control law.

    Parameters
    ----------
    text : str
        The law, e.g. ``"clip(offset + Kp*([SNH] - sp), 0, 240)"``.
    constants : mapping of str to float
        The named numbers the law uses.
    model : CompiledModel, optional
        The sensed stream's model; ``None`` when the controller senses nothing,
        and then the law may read no species or channel.
    what : str
        What the law belongs to, for messages (``"ExpressionController 'x'"``).

    Returns
    -------
    ControlLaw

    Raises
    ------
    ControlLawError
        If the law cannot be read, a constant takes a reserved name, or the law
        reads a species or channel the controller cannot sense.

    Examples
    --------
    >>> law = compile_law("0.5*kla_tank3 + offset", {"offset": 10.0}, what="tank4")
    >>> law.signals
    ('kla_tank3',)
    >>> float(law(0.0, None, {"kla_tank3": 100.0}, None))
    60.0
    """
    for name in constants:
        if name in _RESERVED:
            raise ControlLawError(
                f"{what}: the constant '{name}' takes a name the law already gives a "
                f"meaning (the clock 't', 'pi', 'e', a function or a measurement "
                f"channel); rename it"
            )
    parser = _LawParser(text, constants, model.species_index if model is not None else {})
    try:
        fn = parser.parse()
    except ControlLawError as exc:
        raise ControlLawError(f"{what}: {exc.args[0]}") from None
    read = [f"[{s}]" for s in parser.species] + parser.channels
    if read and model is None:
        raise ControlLawError(
            f"{what}: the law reads {', '.join(read)}, but the controller senses no "
            f"stream. Give it model= (the sensed stream's model) and connect the sensed "
            f"unit to its 'measured' port."
        )
    weights = _channel_weights(model, parser.channels, what) if parser.channels else None
    return ControlLaw(
        text, tuple(parser.species), tuple(parser.channels), tuple(parser.signals), fn, weights
    )


def _channel_weights(model, channels: list[str], what: str) -> tuple:
    """``(W, idx, theta0, dW)`` of the channels in ``model``, for :func:`weights_at`.

    ``idx`` addresses the model's own parameter vector, which is what the plant
    hands a unit with that model.
    """
    try:
        mm = measure_map(model, channels)
    except KeyError as exc:
        raise ControlLawError(f"{what}: {exc.args[0]}") from None
    idx = np.asarray([model.param_index[name] for name in mm.params], int)
    return jnp.asarray(mm.W), idx, jnp.asarray(mm.theta0), jnp.asarray(mm.dW)
