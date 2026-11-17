"""Result files of a study, every value keyed by name.

A study's results are plain files a person can open: parameters and plant
states as YAML mappings, trajectories and matrices as CSV tables with named
columns (and rows). Nothing is stored by position, so a result still reads
correctly after the plant's units, models or parameter vector are reordered.
Floats are written with Python's ``repr``, so every file reads back bit for bit.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

__all__ = [
    "read_matrix",
    "read_named_values",
    "read_trajectory",
    "write_matrix",
    "write_named_values",
    "write_trajectory",
]


def _number(value) -> float:
    return float(np.asarray(value).item())


def _text(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)


def _write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _plain(values: Mapping) -> dict:
    out = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError(f"a result is keyed by name; got the key {key!r}")
        out[key] = _plain(value) if isinstance(value, Mapping) else _number(value)
    return out


def write_named_values(path, values: Mapping) -> Path:
    """Write values keyed by name (parameters, a plant state) as a YAML mapping.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to write (folders are created).
    values : mapping
        ``{name: number}``, or nested mappings of them, such as a plant state
        ``{unit: {state name: value}}`` from :meth:`Plant.named_state` or
        parameters ``{"<model>.<param>": value}``.

    Returns
    -------
    pathlib.Path
    """
    return _write_text(path, yaml.safe_dump(_plain(values), sort_keys=False))


def read_named_values(path) -> dict:
    """Read a file written by :func:`write_named_values`.

    Returns
    -------
    dict
        The same (nested) mapping of names to floats.
    """
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} holds no named values")
    return data


def _write_csv(path, header: Sequence[str], rows) -> Path:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return _write_text(path, buffer.getvalue())


def _read_csv(path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows[0], rows[1:]


def write_trajectory(path, t, columns: Mapping) -> Path:
    """Write named time series as a CSV table: a ``t`` column, then one per name.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to write.
    t : array-like
        Times, shape ``(n_t,)``.
    columns : mapping of str to array-like
        ``{name: values}``, each of shape ``(n_t,)``, such as
        ``{"effluent.SNH": ..., "effluent.TN": ...}``.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    ValueError
        If a column's length differs from ``t``'s, or a name is ``t`` or repeats.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    names = list(columns)
    if "t" in names:
        raise ValueError("'t' is the time column; give the series another name")
    data = []
    for name in names:
        values = np.asarray(columns[name], dtype=float).reshape(-1)
        if values.shape != t.shape:
            raise ValueError(f"column '{name}' has {values.size} values for {t.size} times")
        data.append(values)
    rows = ([_text(t[i]), *(_text(v[i]) for v in data)] for i in range(t.size))
    return _write_csv(path, ["t", *names], rows)


def read_trajectory(path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read a file written by :func:`write_trajectory`.

    Returns
    -------
    t : numpy.ndarray
    columns : dict of str to numpy.ndarray
    """
    header, rows = _read_csv(path)
    if not header or header[0] != "t":
        raise ValueError(f"{path} is not a trajectory table (its first column is not 't')")
    values = np.asarray([[float(x) for x in row] for row in rows], dtype=float).reshape(
        len(rows), len(header)
    )
    return values[:, 0], {name: values[:, j] for j, name in enumerate(header[1:], start=1)}


def write_matrix(path, matrix, rows: Sequence[str], columns: Sequence[str] | None = None) -> Path:
    """Write a matrix with named rows and columns (a Fisher information, a covariance).

    Parameters
    ----------
    path : str or pathlib.Path
        The file to write.
    matrix : array-like
        Shape ``(len(rows), len(columns))``.
    rows : sequence of str
        Row names.
    columns : sequence of str, optional
        Column names (default: the row names, for a square matrix).

    Returns
    -------
    pathlib.Path
    """
    columns = list(rows if columns is None else columns)
    rows = list(rows)
    m = np.asarray(matrix, dtype=float)
    if m.shape != (len(rows), len(columns)):
        raise ValueError(f"the matrix has shape {m.shape} for {len(rows)} x {len(columns)} names")
    return _write_csv(
        path, ["", *columns], ([name, *(_text(v) for v in m[i])] for i, name in enumerate(rows))
    )


def read_matrix(path) -> tuple[np.ndarray, list[str], list[str]]:
    """Read a file written by :func:`write_matrix`.

    Returns
    -------
    matrix : numpy.ndarray
    rows, columns : list of str
    """
    header, body = _read_csv(path)
    columns = header[1:]
    names = [row[0] for row in body]
    m = np.asarray([[float(x) for x in row[1:]] for row in body], dtype=float).reshape(
        len(names), len(columns)
    )
    return m, names, columns
