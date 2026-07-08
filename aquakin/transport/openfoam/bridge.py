"""Python side of the OpenFOAM coupling.

This module is the seam between OpenFOAM cell-field data and a
:class:`~aquakin.core.conditions.SpatialConditions` object. The actual C++
plugin (Option C in CLAUDE.md) lives in a separate repository; this stub
documents the contract it must satisfy.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from aquakin.core.conditions import SpatialConditions


class OpenFOAMBridge:
    """
    Bridge between OpenFOAM cell fields and ``SpatialConditions``.

    The C++ plugin is responsible for populating ``cell_fields`` from
    OpenFOAM's volScalarField data each transport sub-step, then calling
    :meth:`from_cell_fields` to produce a ``SpatialConditions`` object that
    the reactor can consume.

    This is the Option-C (runtime-coupling) seam from CLAUDE.md. The
    offline / Lagrangian Option-A path is provided by
    :mod:`aquakin.transport.openfoam.tracks` instead.
    """

    @classmethod
    def from_cell_fields(
        cls,
        cell_fields: Mapping[str, jnp.ndarray],
        n_cells: int,
    ) -> SpatialConditions:
        """
        Build a ``SpatialConditions`` object from per-cell field arrays.

        Parameters
        ----------
        cell_fields : mapping str -> array
            Each value must be a 1-D array of length ``n_cells`` giving the
            field value at every CFD cell.
        n_cells : int
            Number of CFD cells.

        Returns
        -------
        SpatialConditions
        """
        fields = {}
        for name, arr in cell_fields.items():
            a = jnp.asarray(arr)
            if a.shape != (n_cells,):
                raise ValueError(f"Field '{name}' has shape {a.shape}, expected ({n_cells},)")
            fields[name] = a
        return SpatialConditions(fields=fields)
