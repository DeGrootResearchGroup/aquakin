"""OpenFOAM coupling adapter.

Option A (offline / Lagrangian) is implemented here via :mod:`tracks`.

Option C (runtime coupling via pybind11) uses
:class:`aquakin.integrate.cfd.CFDReactor` as its Python entry point;
re-exported from this module for discoverability. The C++ ``fvOptions``
plugin lives in a separate repository.
"""

from aquakin.integrate.cfd import CFDReactor
from aquakin.transport.openfoam.bridge import from_cell_fields
from aquakin.transport.openfoam.tracks import read_tracks_csv, write_tracks_csv

__all__ = ["CFDReactor", "from_cell_fields", "read_tracks_csv", "write_tracks_csv"]
