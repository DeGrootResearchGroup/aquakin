"""Python side of the OpenFOAM coupling.

This module is the seam between OpenFOAM cell-field data and a
:class:`~aquakin.core.conditions.SpatialConditions` object. The actual C++
plugin (Option C in CLAUDE.md) lives in a separate repository; the C++ side is
responsible for populating a ``cell_fields`` mapping from OpenFOAM's
``volScalarField`` data each transport sub-step and calling
:func:`from_cell_fields` to produce the ``SpatialConditions`` the reactor
consumes.

This is the Option-C (runtime-coupling) seam from CLAUDE.md. The offline /
Lagrangian Option-A path is provided by :mod:`aquakin.transport.openfoam.tracks`
instead.
"""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp

from aquakin.core.conditions import SpatialConditions


def from_cell_fields(
    cell_fields: Mapping[str, jnp.ndarray],
    n_cells: int,
) -> SpatialConditions:
    """Build a ``SpatialConditions`` object from per-cell field arrays.

    The single function of the OpenFOAM Option-C coupling seam: given the CFD
    cell fields the C++ plugin populated, validate their shapes and wrap them as
    a :class:`~aquakin.core.conditions.SpatialConditions` the reactor can index.

    Parameters
    ----------
    cell_fields : mapping str -> array
        Each value must be a 1-D array of length ``n_cells`` giving the field
        value at every CFD cell.
    n_cells : int
        Number of CFD cells.

    Returns
    -------
    SpatialConditions

    Raises
    ------
    ValueError
        If any field's shape is not ``(n_cells,)``.
    """
    fields = {}
    for name, arr in cell_fields.items():
        a = jnp.asarray(arr)
        if a.shape != (n_cells,):
            raise ValueError(f"Field '{name}' has shape {a.shape}, expected ({n_cells},)")
        fields[name] = a
    return SpatialConditions(fields=fields)
