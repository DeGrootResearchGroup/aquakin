"""What a sensor on a plant stream measures, as a linear map of the stream.

A plant observable (:class:`~aquakin.plant.calibrate.PlantObservable`) reads
*channels* of a stream. Each channel is a weighted sum of the stream's species
concentrations plus, optionally, a multiple of its flow, so every measurement a
calibration, a Fisher information or a sensor ranking uses is linear in the
stream and differentiable in the plant state and parameters. A channel is one
of:

- a **species** name of the stream's model (that concentration);
- ``"Q"``: the stream's flow;
- ``"COD"``, ``"TN"``, ``"TP"``: total COD (as measured: electron acceptors such
  as dissolved oxygen and nitrate count none), total nitrogen and total
  phosphorus, in g/m³, from the model's composition table
  (:func:`aquakin.composition_table`);
- ``"TSS"``: total suspended solids, from the model's own TSS species (``XTSS``
  or ``XSS``) or, for the ASM1 family, the benchmark convention
  ``0.75 × (XS + XI + XB_H + XB_A + XP)``;
- a :class:`Measure`: any other weighted sum (a custom TSS factor, BOD₅, ...).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from aquakin.core.hints import did_you_mean

__all__ = ["KEYWORDS", "Measure", "measure_columns"]

#: The channel names that are measurements rather than species.
KEYWORDS = ("Q", "COD", "TN", "TP", "TSS")
_COMPONENT = {"COD": "COD", "TN": "N", "TP": "P"}
_TSS_SPECIES = ("XTSS", "XSS")


@dataclass(frozen=True)
class Measure:
    """A custom measured channel: a weighted sum of species plus flow.

    Parameters
    ----------
    name : str
        A label for the channel (used in messages).
    weights : mapping of str to float, optional
        Weight per species of the stream's model; species not listed weigh 0.
    flow : float, optional
        Weight of the stream's flow.

    Examples
    --------
    >>> bod5 = Measure("BOD5", {"SS": 0.65, "XS": 0.65})
    >>> PlantObservable("effluent", ("SNH", bod5))
    """

    name: str
    weights: Mapping[str, float] = field(default_factory=dict)
    flow: float = 0.0

    def __post_init__(self):
        # Stored as sorted pairs so the measure is hashable like the rest of an observable.
        object.__setattr__(self, "weights", tuple(sorted(dict(self.weights).items())))


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


def _component_weights(model, keyword: str) -> np.ndarray:
    from aquakin.utils.composition import canonical_content

    component = _COMPONENT[keyword]
    try:
        w = np.asarray(canonical_content(model, component, electron_acceptor_cod=False), float)
    except KeyError as exc:
        raise KeyError(
            f"no {keyword} for model '{model.name}': {exc.args[0] if exc.args else exc}"
        ) from None
    if not np.any(w):
        raise KeyError(f"model '{model.name}' carries no {component}, so it has no {keyword}")
    # A balance gives electron acceptors (dissolved oxygen) negative COD; a COD
    # measurement does not count them.
    return np.maximum(w, 0.0)


def _channel(model, channel) -> tuple[np.ndarray, float]:
    """One channel's species weights ``(n_species,)`` and flow weight."""
    if isinstance(channel, Measure):
        w = np.zeros(model.n_species)
        for species, weight in channel.weights:
            if species not in model.species_index:
                hint = did_you_mean(species, list(model.species))
                raise KeyError(
                    f"measure '{channel.name}' weighs '{species}', which is not a species of "
                    f"model '{model.name}'.{hint}"
                )
            w[model.species_index[species]] = float(weight)
        return w, float(channel.flow)
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
        w = np.zeros(model.n_species)
        w[model.species_index[channel]] = 1.0
        return w, 0.0
    if channel == "Q":
        return np.zeros(model.n_species), 1.0
    if channel in _COMPONENT:
        return _component_weights(model, channel), 0.0
    if channel == "TSS":
        return _tss_weights(model), 0.0
    hint = did_you_mean(channel, [*model.species, *KEYWORDS])
    raise KeyError(
        f"Unknown observed channel '{channel}': neither a species of model '{model.name}' "
        f"nor one of {KEYWORDS}.{hint}"
    )


def measure_columns(model, channels) -> tuple[np.ndarray, np.ndarray]:
    """The linear map from a stream to its measured channels.

    Parameters
    ----------
    model : CompiledModel
        The stream's model.
    channels : sequence or None
        The channels (see the module docstring); ``None`` measures every species.

    Returns
    -------
    W : numpy.ndarray
        Species weights, shape ``(n_species, n_channels)``.
    q : numpy.ndarray
        Flow weights, shape ``(n_channels,)``. A stream with concentrations
        ``C`` and flow ``Q`` measures ``C @ W + Q * q``.

    Raises
    ------
    KeyError
        If a channel names no species or measurement of ``model``, or a
        measurement the model cannot give (``TSS`` of a digester).
    """
    if channels is None:
        return np.eye(model.n_species), np.zeros(model.n_species)
    cols = [_channel(model, c) for c in channels]
    if not cols:
        raise ValueError("an observable must measure at least one channel")
    return np.stack([w for w, _ in cols], axis=1), np.asarray([q for _, q in cols])
