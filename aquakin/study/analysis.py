"""What a study's analyses share, whatever they study.

A plant study and a reactor study call different analyses (``calibrate_plant``
or ``calibrate``, ...), but read their observations, set up their fit, and write
their Fisher information and profile with the functions here, so the two
targets give the same files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from aquakin.study.common import (
    FISHER_FILE,
    FISHER_SUMMARY_FILE,
    PARAMETERS_FILE,
    PROFILE_FILE,
    PROFILE_SUMMARY_FILE,
    StudyError,
)
from aquakin.utils.result_files import write_matrix, write_named_values, write_yaml

if TYPE_CHECKING:
    from aquakin.study.study import _Study


def _fit_sigma(study: _Study):
    """The fit's ``sigma``: ``"estimate"``, one error for every observation, or an
    array of one per channel (broadcast over the times; a list would mean one
    per dataset)."""
    import numpy as np

    if study.estimate_sigma:
        return "estimate"
    if isinstance(study.sigma, list):
        return np.asarray(study.sigma, dtype=float)
    return study.sigma


def _flattened(ref, name: str, base_dir: Path) -> str:
    """A model's YAML with its whole ``extends`` chain merged in."""
    from importlib.resources import files

    from aquakin.schema.loader import _resolve_inheritance, parse_model_yaml

    if ref["source"] == "shipped":
        text = (files("aquakin.models") / f"{name}.yaml").read_text()
        base = None
    else:
        path = Path(ref["source"])
        path = path if path.is_absolute() else base_dir / path
        text, base = path.read_text(), path.parent
    data = _resolve_inheritance(parse_model_yaml(text), name, base, seen=())
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _fit_options(study: _Study) -> dict:
    """The settings of a study's fit that do not depend on what it fits: the
    transforms, the likelihood with its measurement error, the priors and the
    optimizer."""
    from aquakin import OptimizerConfig

    return dict(
        transforms=dict(study.transforms) or None,
        loss="nll",
        sigma=_fit_sigma(study),
        use_priors=study.use_priors,
        optimizer=OptimizerConfig(n_starts=study.n_starts, max_iter=study.max_iter),
    )


def _read_observations(path: Path, names: list[str], where: str):
    """The named columns of a data table.

    Parameters
    ----------
    path : pathlib.Path
        A table written as :func:`~aquakin.utils.result_files.write_trajectory`
        writes one: a ``t`` column, then one column per name.
    names : list of str
        The columns to read, in order.
    where : str
        The setting that names the file, for messages.

    Returns
    -------
    t : numpy.ndarray
        ``(n_t,)``.
    observations : numpy.ndarray
        ``(n_t, len(names))``.

    Raises
    ------
    StudyError
        If the file is missing, is not such a table, or lacks a column.
    """
    import numpy as np

    from aquakin.utils.result_files import read_trajectory

    if not path.is_file():
        raise StudyError(f"{where}: {path} does not exist")
    try:
        t, columns = read_trajectory(path)
    except ValueError as exc:
        raise StudyError(f"{where}: {exc}") from None
    missing = [n for n in names if n not in columns]
    if missing:
        raise StudyError(
            f"{where}: {path.name} has no column {', '.join(missing)} (its columns: "
            f"{', '.join(columns) or 'none'})"
        )
    return t, np.stack([columns[n] for n in names], axis=1)


def _write_fisher_files(out: Path, fim, parameters: dict) -> list[Path]:
    """Write a Fisher information's ``fisher.csv`` and ``summary.yaml`` (values,
    transforms, standard deviations or ``None`` when it is singular, the four
    design criteria), and the parameters it was evaluated at."""
    import numpy as np

    from aquakin import design_criterion

    criteria = {}
    for c in ("D", "A", "E", "modified_E"):
        try:
            criteria[c] = float(design_criterion(fim, c))
        except (np.linalg.LinAlgError, ValueError):
            criteria[c] = None
    try:
        std = fim.std()
    except np.linalg.LinAlgError:
        std = None
    summary = {
        "values": dict(fim.values),
        "transforms": dict(zip(fim.names, fim.transforms)),
        "std": std,
        "criteria": criteria,
    }
    return [
        write_matrix(out / FISHER_FILE, fim.matrix, fim.names),
        write_yaml(out / FISHER_SUMMARY_FILE, summary),
        write_named_values(out / PARAMETERS_FILE, parameters),
    ]


def _write_profile_files(out: Path, study: _Study, prof) -> list[Path]:
    """Write a profile likelihood's ``profile.csv`` (the grid, the loss and its
    rise from the best) and ``profile.yaml`` (the best value and the interval)."""
    import numpy as np

    from aquakin.utils.result_files import write_table

    written = [
        write_table(
            out / PROFILE_FILE,
            {
                study.profile: list(prof.grid),
                "loss": [None if not np.isfinite(v) else v for v in prof.loss],
                "delta_loss": [None if not np.isfinite(v) else v for v in prof.delta_loss],
            },
        )
    ]
    lo, hi = prof.ci
    summary = {
        "parameter": study.profile,
        "mle": None if not np.isfinite(prof.mle) else float(prof.mle),
        "ci": [None if lo is None else float(lo), None if hi is None else float(hi)],
        "delta": study.delta,
    }
    written.append(write_yaml(out / PROFILE_SUMMARY_FILE, summary))
    return written
