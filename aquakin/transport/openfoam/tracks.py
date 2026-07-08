"""Read and write Lagrangian particle tracks in the aquakin CSV format.

The CSV schema is the user-facing contract for offline OpenFOAM coupling:

::

    particle_id,t,<field1>,<field2>,...
    0,0.0,7.5,293.15,5.0e4
    0,0.5,7.5,293.15,5.0e4
    ...
    1,0.0,7.5,293.15,5.0e4
    ...

- ``particle_id`` is an integer.
- ``t`` is the time at which the particle samples this state, in seconds.
  Within each particle the rows must be strictly ascending in ``t``.
- Remaining columns are condition field values at that ``(particle, t)``
  sample. The column names become the field names of the resulting
  :class:`~aquakin.integrate.particle.Track`.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path

import jax.numpy as jnp

from aquakin.integrate.particle import Track

PathLike = str | Path

_REQUIRED_COLUMNS = ("particle_id", "t")


def read_tracks_csv(path: PathLike) -> dict[int, Track]:
    """
    Load Lagrangian particle tracks from a CSV file.

    Parameters
    ----------
    path : str or Path
        CSV file with the schema documented in this module.

    Returns
    -------
    dict[int, Track]
        Mapping from ``particle_id`` to :class:`Track`.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Track file not found: {p}")

    with p.open() as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty track file: {p}") from None

        header = [col.strip() for col in raw_header]

        for required in _REQUIRED_COLUMNS:
            if required not in header:
                raise ValueError(
                    f"Track file {p} is missing required column '{required}'. Got header: {header}"
                )
        pid_col = header.index("particle_id")
        t_col = header.index("t")
        field_cols = [(i, name) for i, name in enumerate(header) if name not in _REQUIRED_COLUMNS]

        def _parse_float(row_no: int, col_name: str, raw: str) -> float:
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(
                    f"Track file {p} row {row_no}: column '{col_name}' has "
                    f"non-numeric value {raw!r}."
                ) from None
            if not math.isfinite(value):
                raise ValueError(
                    f"Track file {p} row {row_no}: column '{col_name}' has "
                    f"non-finite value {raw!r}."
                )
            return value

        grouped: dict[int, list[tuple[float, list[float]]]] = {}
        for row_no, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"Track file {p} row {row_no} has {len(row)} columns, expected {len(header)}"
                )
            try:
                pid = int(row[pid_col])
            except ValueError:
                raise ValueError(
                    f"Track file {p} row {row_no}: particle_id must be integer, "
                    f"got {row[pid_col]!r}"
                ) from None
            t_value = _parse_float(row_no, "t", row[t_col])
            field_values = [_parse_float(row_no, name, row[i]) for i, name in field_cols]
            grouped.setdefault(pid, []).append((t_value, field_values))

    tracks: dict[int, Track] = {}
    for pid, samples in grouped.items():
        ts = jnp.asarray([s[0] for s in samples])
        diffs = jnp.diff(ts)
        if bool(jnp.any(diffs == 0)):
            raise ValueError(f"Track file {p}: particle_id {pid} has duplicate t values.")
        if not bool(jnp.all(diffs > 0)):
            raise ValueError(
                f"Track file {p}: particle_id {pid} samples are not strictly ascending in t."
            )
        fields = {
            name: jnp.asarray([s[1][col_index] for s in samples])
            for col_index, (_, name) in enumerate(field_cols)
        }
        tracks[pid] = Track(t=ts, fields=fields)
    return tracks


def write_tracks_csv(path: PathLike, tracks: Mapping[int, Track]) -> None:
    """
    Write an ensemble of tracks to a CSV file in the standard schema.

    Tracks are written sorted by ``particle_id``. The field column order is
    taken from the first track encountered; all tracks must share the same
    set of field names.
    """
    if not tracks:
        raise ValueError("Cannot write an empty tracks mapping.")
    field_order: list[str] | None = None
    for track in tracks.values():
        names = sorted(track.fields)
        if field_order is None:
            field_order = names
        elif names != field_order:
            raise ValueError(
                f"All tracks must share the same field names; got {names} vs {field_order}"
            )
    assert field_order is not None

    p = Path(path)
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["particle_id", "t", *field_order])
        for pid in sorted(tracks):
            track = tracks[pid]
            ts = track.t
            for i in range(track.n_points):
                writer.writerow(
                    [
                        pid,
                        f"{float(ts[i]):.10g}",
                        *[f"{float(track.fields[name][i]):.10g}" for name in field_order],
                    ]
                )
