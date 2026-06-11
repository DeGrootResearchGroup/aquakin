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
            ``SpatialConditions.uniform(pH=7.5, T=293.15)``.
        **kwargs : float
            Field name -> scalar value. The scalar is broadcast to
            ``(n_locations,)``.

        Returns
        -------
        SpatialConditions
        """
        if n_locations < 1:
            raise ValueError(f"n_locations must be >= 1, got {n_locations}")
        fields = {name: jnp.full((n_locations,), float(value)) for name, value in kwargs.items()}
        return cls(fields=fields)

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
