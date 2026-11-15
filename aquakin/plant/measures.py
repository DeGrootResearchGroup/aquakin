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
- a :class:`Measure`: any other weighted sum (a custom TSS factor, BOD₅, ...).

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

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from aquakin.core.hints import did_you_mean

__all__ = ["KEYWORDS", "Measure", "MeasureMap", "measure_map"]

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
    flow : float, optional
        Weight of the stream's flow.
    temperature : float, optional
        Weight of the stream's temperature.

    Examples
    --------
    >>> bod5 = Measure("BOD5", {"SS": 0.65, "XS": 0.65})
    >>> PlantObservable("effluent", ("SNH", bod5))
    """

    name: str
    weights: Mapping[str, float] = field(default_factory=dict)
    flow: float = 0.0
    temperature: float = 0.0

    def __post_init__(self):
        # Stored as sorted pairs so the measure is hashable like the rest of an observable.
        object.__setattr__(self, "weights", tuple(sorted(dict(self.weights).items())))


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


def _channel(model, channel, params) -> tuple[np.ndarray, float, float]:
    """One channel's species weights ``(n_species,)``, flow and temperature weights."""
    zero = np.zeros(model.n_species)
    if isinstance(channel, Measure):
        w = zero.copy()
        for species, weight in channel.weights:
            if species not in model.species_index:
                hint = did_you_mean(species, list(model.species))
                raise KeyError(
                    f"measure '{channel.name}' weighs '{species}', which is not a species of "
                    f"model '{model.name}'.{hint}"
                )
            w[model.species_index[species]] = float(weight)
        return w, float(channel.flow), float(channel.temperature)
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


def measure_map(model, channels) -> MeasureMap:
    """The linear map from a stream of ``model`` to its measured ``channels``.

    Parameters
    ----------
    model : CompiledModel
        The stream's model.
    channels : sequence or None
        The channels (see the module docstring); ``None`` measures every species.

    Returns
    -------
    MeasureMap

    Raises
    ------
    KeyError
        If a channel names no species or measurement of ``model``, or a
        measurement the model cannot give (``TSS`` of a digester).
    ValueError
        If ``channels`` is empty.
    """
    n = model.n_species
    if channels is None:
        return MeasureMap(np.eye(n), np.zeros(n), np.zeros(n), (), np.zeros(0), np.zeros((0, n, n)))
    channels = tuple(channels)
    if not channels:
        raise ValueError("an observable must measure at least one channel")
    theta0 = np.asarray(model.default_parameters(), float)
    W, q, t = _weights(model, channels, theta0)
    params, values, slopes = [], [], []
    if any(
        isinstance(c, str) and c in _COMPOSITION and c not in model.species_index for c in channels
    ):
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
