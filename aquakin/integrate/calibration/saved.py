"""A fit saved to a folder, and read back: :meth:`CalibrationResult.save` and
:func:`aquakin.load_calibration`.

The folder holds plain files a person can open, written with
:mod:`aquakin.utils.result_files` (every value keyed by name, floats with
``repr`` so they read back bit for bit)::

    fit.yaml               the summary: the fitted rates, their std, the loss and
                           the optimizer's status, the noise estimate, the
                           transforms, priors and information-criteria counts,
                           the fitted initial pools, the model fingerprints, the
                           format and the aquakin version
    parameters.yaml        the full parameter vector, by name
    covariance.csv         posterior_cov             (with a Laplace posterior)
    hessian.csv            hessian_unconstrained     (with a Laplace posterior)
    joint_covariance.csv   joint_cov, joint_hessian  (with include_ic=True)
    joint_hessian.csv
    initial_states.csv     C0_fitted, one row per dataset (with free_ic)

A calibrate study (:mod:`aquakin.study`) writes ``parameters.yaml``,
``fit.yaml`` and ``covariance.csv`` with the same writer,
:func:`write_fit_files`, its ``fit.yaml`` holding only :func:`fit_summary`.
The file names are part of the stored format and do not change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from aquakin.core.model import CompiledModel
from aquakin.utils.result_files import (
    read_matrix,
    read_named_values,
    write_matrix,
    write_named_values,
    write_yaml,
)

if TYPE_CHECKING:
    from aquakin.integrate.calibration.result import CalibrationResult

__all__ = [
    "COVARIANCE_FILE",
    "FIT_FILE",
    "FORMAT",
    "HESSIAN_FILE",
    "INITIAL_STATES_FILE",
    "JOINT_COVARIANCE_FILE",
    "JOINT_HESSIAN_FILE",
    "PARAMETERS_FILE",
    "fit_summary",
    "write_fit_files",
]

FIT_FILE = "fit.yaml"
PARAMETERS_FILE = "parameters.yaml"
COVARIANCE_FILE = "covariance.csv"
HESSIAN_FILE = "hessian.csv"
JOINT_COVARIANCE_FILE = "joint_covariance.csv"
JOINT_HESSIAN_FILE = "joint_hessian.csv"
INITIAL_STATES_FILE = "initial_states.csv"
#: The format of a saved fit, recorded in its ``fit.yaml``. Raise it when the
#: layout changes; :func:`aquakin.load_calibration` refuses any other.
FORMAT = 1

# The optional matrices: (file, result field, row/column names field).
_MATRICES = (
    (COVARIANCE_FILE, "posterior_cov", "parameter_names"),
    (HESSIAN_FILE, "hessian_unconstrained", "parameter_names"),
    (JOINT_COVARIANCE_FILE, "joint_cov", "joint_names"),
    (JOINT_HESSIAN_FILE, "joint_hessian", "joint_names"),
)


def _floats(values: Mapping | None) -> dict | None:
    return None if values is None else {name: float(v) for name, v in values.items()}


def _float_or_none(value) -> float | None:
    return None if value is None else float(value)


def fit_summary(result: CalibrationResult, outputs: Sequence[str]) -> dict:
    """The summary of a fit that a calibrate study writes as its ``fit.yaml``.

    Parameters
    ----------
    result : CalibrationResult
        The fit.
    outputs : sequence of str
        A name for each observed output, keying ``sigma_estimate``.

    Returns
    -------
    dict
        ``fitted`` and ``std`` (by parameter name; ``std`` is ``None`` without
        a Laplace posterior), ``loss``, ``converged``, ``message``,
        ``iterations``, ``at_bound`` and ``sigma_estimate`` (by output name, or
        ``None``), as plain Python values.

    Raises
    ------
    ValueError
        If ``sigma_estimate`` does not have one entry per name in ``outputs``.
    """
    sigma = None
    if result.sigma_estimate is not None:
        values = [float(v) for v in np.ravel(result.sigma_estimate)]
        if len(values) != len(outputs):
            raise ValueError(
                f"sigma_estimate has {len(values)} entries for {len(outputs)} output names"
            )
        sigma = dict(zip(outputs, values))
    return {
        "fitted": _floats(result.params_named),
        "std": _floats(result.params_named_std),
        "loss": float(result.loss),
        "converged": bool(result.converged),
        "message": str(result.message),
        "iterations": int(result.n_iter),
        "at_bound": dict(result.at_bound),
        "sigma_estimate": sigma,
    }


def write_fit_files(
    folder, result: CalibrationResult, summary: Mapping, parameters: Mapping
) -> list[Path]:
    """Write a fit's ``parameters.yaml``, ``fit.yaml`` and, with a Laplace
    posterior, ``covariance.csv``.

    Parameters
    ----------
    folder : str or pathlib.Path
        Where to write them (created if needed).
    result : CalibrationResult
        The fit; its ``posterior_cov`` is written by ``parameter_names``.
    summary : mapping
        The content of ``fit.yaml``, plain Python values (see
        :func:`fit_summary`).
    parameters : mapping of str to float
        The full parameter vector by name.

    Returns
    -------
    list of pathlib.Path
        The files written, in that order.
    """
    folder = Path(folder)
    written = [
        write_named_values(folder / PARAMETERS_FILE, parameters),
        write_yaml(folder / FIT_FILE, summary),
    ]
    if result.posterior_cov is not None:
        written.append(
            write_matrix(
                folder / COVARIANCE_FILE, np.asarray(result.posterior_cov), result.parameter_names
            )
        )
    return written


# ---------------------------------------------------------------------------
# What a saved fit records of its model
# ---------------------------------------------------------------------------


def _ordered(index: Mapping[str, int], what: str) -> list[str]:
    """The names of a vector in its order, from ``{name: position}``."""
    names = sorted(index, key=index.__getitem__)
    if [index[n] for n in names] != list(range(len(names))):
        raise ValueError(f"the {what} names do not cover every entry of the vector")
    return names


def _model_record(model, problem=None) -> dict:
    """What a saved fit needs from its model: the names of the parameter vector
    and of the initial-state vector in order, each model's fingerprint, and a
    name for each observed output.

    ``model`` is a :class:`~aquakin.CompiledModel`, or the plant's parameter
    namespace of a plant fit, which offers the same ``param_index`` /
    ``species_index`` (the plant state by ``"<unit>.<state>"``) and lists its
    models in ``models``.
    """
    compiled = [model] if isinstance(model, CompiledModel) else list(model.models)
    fingerprints = {}
    for m in compiled:
        source = tuple(getattr(m, "_aquakin_source", ()) or ())
        fingerprints[m.name] = source[2] if len(source) == 3 else None
    outputs = None
    if problem is not None:
        n_out = int(problem.n_observed)
        if problem.observation_fn is None and isinstance(model, CompiledModel):
            outputs = [model.species[int(i)] for i in np.asarray(problem.obs_species_indices)]
        else:
            outputs = [f"output[{j}]" for j in range(n_out)]
    return {
        "parameters": _ordered(model.param_index, "parameter"),
        "states": _ordered(model.species_index, "state"),
        "models": fingerprints,
        "outputs": outputs,
    }


def _record_of(result: CalibrationResult, model) -> dict:
    """The model record of a fit (its own), a loaded result (read with it), or
    else of the ``model`` given."""
    if result._fit_record is not None:
        problem = result._fit_record.problem
        return _model_record(problem.model, problem)
    if result._model_record is not None:
        return result._model_record
    if model is not None:
        return _model_record(model)
    raise ValueError(
        "This CalibrationResult carries no fit to name its parameter vector by; "
        "pass the calibrated model: result.save(folder, model=reactor.model)."
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def _pools(result: CalibrationResult) -> dict | None:
    if result.ic_named is None:
        return None

    def per_dataset(pools):
        return None if pools is None else [_floats(p) for p in pools]

    return {
        "fitted": per_dataset(result.ic_named),
        "std": per_dataset(result.ic_named_std),
        "fractions": per_dataset(result.ic_fractions),
        "fractions_std": per_dataset(result.ic_fractions_std),
    }


def save_calibration(result: CalibrationResult, folder, model=None) -> Path:
    """Write ``result`` to ``folder``; see :meth:`CalibrationResult.save`."""
    from aquakin import __version__

    folder = Path(folder)
    taken = [
        name
        for name in (FIT_FILE, PARAMETERS_FILE, INITIAL_STATES_FILE, *(m[0] for m in _MATRICES))
        if (folder / name).exists()
    ]
    if taken:
        raise FileExistsError(
            f"{folder} already holds {', '.join(taken)}; save to a new folder, or delete "
            "the saved fit there first"
        )
    record = _record_of(result, model)
    params = np.asarray(result.params, dtype=float)
    if params.shape != (len(record["parameters"]),):
        raise ValueError(
            f"the result has {params.size} parameters; the model names {len(record['parameters'])}"
        )
    n_out = 0 if result.sigma_estimate is None else np.size(result.sigma_estimate)
    outputs = record["outputs"] or [f"output[{j}]" for j in range(n_out)]
    matrices = [m for m in _MATRICES if getattr(result, m[1]) is not None]
    summary = {
        "format": FORMAT,
        "aquakin": __version__,
        **fit_summary(result, outputs),
        "transforms": dict(zip(result.parameter_names, result.transforms)),
        "std_unconstrained": None
        if result.posterior_std_unconstrained is None
        else dict(
            zip(result.parameter_names, map(float, np.ravel(result.posterior_std_unconstrained)))
        ),
        "priors": {n: [float(m), float(s)] for n, (m, s) in result.priors_applied.items()},
        "log_likelihood": _float_or_none(result.log_likelihood),
        "n_observations": result.n_observations,
        "n_estimated": result.n_estimated,
        "observations_key": result._observations_key,
        "prior_in_loss": bool(result._prior_in_loss),
        "initial_pools": _pools(result),
        "joint_names": result.joint_names,
        "models": record["models"],
        "files": [m[0] for m in matrices]
        + ([INITIAL_STATES_FILE] if result.C0_fitted is not None else []),
    }
    parameters = dict(zip(record["parameters"], map(float, params)))
    write_fit_files(folder, result, summary, parameters)  # covariance.csv among them
    for file, attr, names in matrices:
        if file != COVARIANCE_FILE:
            write_matrix(folder / file, np.asarray(getattr(result, attr)), getattr(result, names))
    if result.C0_fitted is not None:
        write_matrix(
            folder / INITIAL_STATES_FILE,
            np.stack([np.asarray(c, dtype=float) for c in result.C0_fitted]),
            [f"dataset {k}" for k in range(len(result.C0_fitted))],
            record["states"],
        )
    return folder


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


class _Damaged(ValueError):
    pass


def _read_summary(folder: Path) -> dict:
    import yaml

    from aquakin import __version__

    path = folder / FIT_FILE
    if not folder.is_dir():
        raise FileNotFoundError(f"no saved fit at {folder}: the folder does not exist")
    if not path.is_file():
        raise FileNotFoundError(
            f"{folder} holds no {FIT_FILE}, so it is not a fit saved by CalibrationResult.save"
        )
    try:
        summary = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise _Damaged(f"{path} is damaged: {exc}") from None
    if not isinstance(summary, dict):
        raise _Damaged(f"{path} is damaged: it holds no summary of a fit")
    version = summary.get("format")
    if version != FORMAT:
        raise ValueError(
            f"{path} is in saved-fit format {version!r}; this aquakin ({__version__}) reads "
            f"format {FORMAT}. A fit saved by a newer aquakin needs that version to read it."
        )
    return summary


def _listed(folder: Path, file: str) -> Path:
    path = folder / file
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: it is part of the fit saved in {folder}")
    return path


def _matrix(folder: Path, file: str, names: list[str]):
    path = _listed(folder, file)
    m, rows, columns = read_matrix(path)
    if rows != names or columns != names:
        raise _Damaged(f"{path} is damaged: its rows or columns are not the ones {FIT_FILE} names")
    return m


def _pool_fields(pools: dict | None) -> dict:
    keys = {"fitted": "ic_named", "std": "ic_named_std"}
    keys |= {"fractions": "ic_fractions", "fractions_std": "ic_fractions_std"}
    if pools is None:
        return {}
    return {field: pools[key] for key, field in keys.items()}


def read_calibration(folder) -> dict:
    """The fields of a :class:`CalibrationResult` saved in ``folder``; see
    :func:`aquakin.load_calibration`."""
    folder = Path(folder)
    summary = _read_summary(folder)
    try:
        return _fields(folder, summary)
    except _Damaged:
        raise
    except KeyError as exc:
        raise ValueError(f"{folder / FIT_FILE} is damaged: it has no {exc}") from None
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"the saved fit in {folder} is damaged: {exc}") from None


def _fields(folder: Path, s: dict) -> dict:
    names = list(s["transforms"])
    files = list(s["files"])
    parameters = read_named_values(_listed(folder, PARAMETERS_FILE))
    std_u = s["std_unconstrained"]
    sigma = s["sigma_estimate"]
    fields = {
        "params": jnp.asarray([float(v) for v in parameters.values()]),
        "params_named": {n: float(s["fitted"][n]) for n in names},
        "loss": float(s["loss"]),
        "converged": bool(s["converged"]),
        "message": str(s["message"]),
        "n_iter": int(s["iterations"]),
        "parameter_names": names,
        "transforms": [s["transforms"][n] for n in names],
        "posterior_std_unconstrained": None if std_u is None else jnp.asarray(list(std_u.values())),
        "params_named_std": s["std"],
        "priors_applied": {n: (float(m), float(sd)) for n, (m, sd) in s["priors"].items()},
        "joint_names": s["joint_names"],
        "at_bound": dict(s["at_bound"]),
        "sigma_estimate": None if sigma is None else np.asarray(list(sigma.values()), dtype=float),
        "log_likelihood": s["log_likelihood"],
        "n_observations": s["n_observations"],
        "n_estimated": s["n_estimated"],
        "_observations_key": s["observations_key"],
        "_prior_in_loss": bool(s["prior_in_loss"]),
        **_pool_fields(s["initial_pools"]),
    }
    for file, attr, labels in _MATRICES:
        if file in files:
            fields[attr] = jnp.asarray(_matrix(folder, file, fields[labels]))
    states = None
    if INITIAL_STATES_FILE in files:
        c0, _rows, states = read_matrix(_listed(folder, INITIAL_STATES_FILE))
        fields["C0_fitted"] = [jnp.asarray(row) for row in c0]
    fields["_model_record"] = {
        "parameters": list(parameters),
        "states": states,
        "models": dict(s["models"]),
        "outputs": None if sigma is None else list(sigma),
    }
    return fields
