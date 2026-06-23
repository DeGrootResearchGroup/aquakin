"""Spatially varying condition fields (pH, temperature, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import jax.numpy as jnp


@dataclass
class SpatialConditions:
    """
    Container for spatially varying condition fields.

    Each entry in ``fields`` is a JAX array of shape ``(n_locations,)`` giving
    the value of that field at each spatial location. Reactors index into
    these arrays with ``loc_idx`` at runtime.

    Attributes
    ----------
    fields : dict[str, jnp.ndarray]
        Mapping from field name (e.g. ``"pH"``) to a 1-D JAX array.
    """

    fields: dict[str, jnp.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalised: dict[str, jnp.ndarray] = {}
        n_locations: int | None = None
        for name, value in self.fields.items():
            arr = jnp.asarray(value)
            if arr.ndim == 0:
                arr = arr[None]
            elif arr.ndim != 1:
                raise ValueError(
                    f"Condition field '{name}' must be 1-D, got shape {arr.shape}"
                )
            if n_locations is None:
                n_locations = int(arr.shape[0])
            elif int(arr.shape[0]) != n_locations:
                raise ValueError(
                    f"Condition field '{name}' has length {arr.shape[0]}, "
                    f"expected {n_locations}"
                )
            normalised[name] = arr
        self.fields = normalised

    @property
    def n_locations(self) -> int:
        """Number of spatial locations represented."""
        if not self.fields:
            return 0
        return int(next(iter(self.fields.values())).shape[0])

    @classmethod
    def uniform(cls, n_locations: int = 1, **kwargs: float) -> "SpatialConditions":
        """
        Build a spatially homogeneous ``SpatialConditions`` object.

        Parameters
        ----------
        n_locations : int, optional
            Number of spatial locations (default 1). Must be at least 1; the
            default suits a 0-D batch reactor, so a batch user writes
            ``SpatialConditions.uniform(pH=7.5, T=293.15)``. For the 0-D case the
            :class:`OperatingConditions` alias reads more naturally still.
        **kwargs : float
            Field name -> scalar value. The scalar is broadcast to
            ``(n_locations,)``.

        Returns
        -------
        SpatialConditions
        """
        if n_locations < 1:
            raise ValueError(f"n_locations must be >= 1, got {n_locations}")
        # Broadcast (don't ``float()``-coerce) so a *traced* condition value -- a
        # user differentiating a solve w.r.t. pH/T -- flows through instead of
        # raising a ConcretizationTypeError. Identity for a concrete value.
        fields = {name: jnp.broadcast_to(jnp.asarray(value), (n_locations,))
                  for name, value in kwargs.items()}
        return cls(fields=fields)

    def with_(self, **kwargs: float) -> "SpatialConditions":
        """Return a copy with some fields overridden (or added).

        The common edit-from-defaults pattern: start from
        ``network.default_conditions()`` and change only what differs, e.g.::

            conditions = network.default_conditions().with_(T=283.15)   # cold

        Scalar overrides are broadcast to this object's location count (so the
        result keeps the same ``n_locations``); a length-``n_locations`` array
        override is used as-is. Fields not named are carried over unchanged. The
        original is not modified.

        Parameters
        ----------
        **kwargs : float or array-like
            Field name -> new value (scalar, broadcast to ``n_locations``, or a
            length-``n_locations`` array).

        Returns
        -------
        SpatialConditions
            A new ``SpatialConditions`` (always the base type, so it stays valid
            for every reactor) with the merged fields.
        """
        n = self.n_locations or 1
        merged = dict(self.fields)
        for name, value in kwargs.items():
            arr = jnp.asarray(value)
            # Broadcast a scalar rather than ``float()``-coercing it, so a traced
            # override (a gradient w.r.t. a condition) is not concretized.
            merged[name] = jnp.broadcast_to(arr, (n,)) if arr.ndim == 0 else arr
        return SpatialConditions(fields=merged)

    def validate_required(self, required: Iterable[str]) -> None:
        """
        Raise ``ValueError`` if any required field is missing.

        Parameters
        ----------
        required : iterable of str
            Field names that must be present.
        """
        missing = sorted(set(required) - set(self.fields.keys()))
        if missing:
            raise ValueError(
                f"SpatialConditions is missing required fields: {missing}. "
                f"Provided: {sorted(self.fields.keys())}"
            )


class OperatingConditions(SpatialConditions):
    """Operating conditions for the 0-D (single stirred tank) case.

    A single stirred tank has no spatial extent, so the spatially-varying
    :class:`SpatialConditions` (with its ``n_locations`` array model) reads as
    over-machinery for the most basic setup. ``OperatingConditions`` is the same
    object specialised to one location, constructed directly from scalar field
    values::

        conditions = aquakin.OperatingConditions(pH=7.5, T=293.15)

    It **is** a :class:`SpatialConditions` (one location), so it works unchanged
    in every reactor; for a spatially varying PFR/CFD case use
    :class:`SpatialConditions` (or :meth:`SpatialConditions.uniform`) directly.
    To start from a network's declared defaults instead, use
    ``network.default_conditions()`` and :meth:`SpatialConditions.with_` to edit.

    Parameters
    ----------
    **kwargs : float
        Condition field name (e.g. ``pH``, ``T``) -> scalar value.
    """

    def __init__(self, **kwargs: float) -> None:
        # ``jnp.asarray(value)`` (not ``float(value)``) so a traced condition --
        # a gradient w.r.t. an operating condition -- flows through rather than
        # raising a ConcretizationTypeError. Identity for a concrete value.
        super().__init__(
            fields={name: jnp.asarray(value)
                    for name, value in kwargs.items()})
