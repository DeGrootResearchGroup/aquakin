"""A study of a batch reactor: the ``reactor:`` target of ``study.yaml``.

A reactor study runs its analysis on a batch reactor built from one model, in
place of a plant built from the flowsheet::

    study:
      kind: calibrate
      reactor:
        model: ../../models/ozone_demand.yaml   # a model file, or a shipped model's name
        conditions: {T: 293.15, pH: 7.5}
        integrator: {rtol: 1.0e-6}              # optional
      experiments:
        - {data: run1.csv, initial: {O3: 1.0, D: 12.0}}
        - {data: run2.csv, initial: {O3: 2.0, D: 12.0}}
      observed: [O3]
      free: [fast.k1, slow.k2]
      estimate_sigma: true

Each experiment starts from its ``initial`` concentrations (the model's
defaults for the species it does not name) and, for a fit, reads a data table:
a ``t`` column and one column per observed species. An experiment is named by
its data file (``run1``), and results that differ by experiment are keyed
``<experiment>.<species>``. The ``calibrate``, ``profile`` and ``fisher`` kinds
take a reactor; they share the fit settings and the result files of a plant
study (:mod:`aquakin.study.analysis`).

A model file is fingerprinted with its ``extends`` chain merged in, a shipped
model by its shipped content, and each data table by its bytes: the study is
stale once one changes (:func:`reactor_inputs`). Freezing copies the model,
flattened, and the data tables into ``snapshot/``, and a frozen study reads
them from there.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aquakin.study.analysis import (
    _fit_options,
    _flattened,
    _read_observations,
    _write_fisher_files,
    _write_profile_files,
)
from aquakin.study.common import FITTED_FILE, FROZEN_FILE, SNAPSHOT, StudyError, _times
from aquakin.utils.result_files import write_trajectory

if TYPE_CHECKING:
    from aquakin.study.study import _Study

#: The kinds a reactor study can be, each with the settings it needs.
REQUIRED = {
    "calibrate": ("free", "experiments", "observed"),
    "profile": ("free", "experiments", "observed", "profile", "grid"),
    "fisher": ("free", "experiments", "observed", "t_obs", "sigma"),
}

#: The settings of a flowsheet study that a reactor study has no use for.
_FLOWSHEET_ONLY = (
    "initial_state",
    "time",
    "streams",
    "observables",
    "t_span",
    "relative_sigma",
    "sigma_floor",
    "sensitivity",
    "data",
    "candidates",
    "base",
)

#: Inside ``snapshot/``: the flattened model, and the data tables.
_MODELS = "models"
_DATA = "data"


# ---------------------------------------------------------------------------
# Checking the settings
# ---------------------------------------------------------------------------


def check_reactor_study(study: _Study, where: str) -> None:
    """Refuse a reactor study its target cannot run, as ``study.yaml`` is read.

    Parameters
    ----------
    study : _Study
        The study's settings, with a ``reactor``.
    where : str
        The file or setting, for messages.

    Raises
    ------
    StudyError
        If the kind takes no reactor, a setting is a flowsheet study's, the
        experiments do not fit the kind (a profile has one, a Fisher
        information reads no data, a fit reads one table each), or two data
        tables share a name.
    """
    if study.kind not in REQUIRED:
        raise StudyError(f"{where}: the {study.kind} kind does not take a reactor target yet")
    fields = type(study).model_fields
    unused = [n for n in _FLOWSHEET_ONLY if getattr(study, n) != fields[n].get_default()]
    if unused:
        raise StudyError(
            f"{where}: {', '.join(unused)} {'is a setting' if len(unused) == 1 else 'are settings'}"
            f" of a flowsheet study; a reactor study has no use for "
            f"{'it' if len(unused) == 1 else 'them'}"
        )
    if study.kind == "fisher":
        if any(e.data is not None for e in study.experiments):
            raise StudyError(
                f"{where}: a fisher study reads no data; its experiments give their initial "
                f"concentrations, and t_obs the sampling times"
            )
        return
    without = [i for i, e in enumerate(study.experiments) if e.data is None]
    if without:
        raise StudyError(
            f"{where}: experiments {', '.join(map(str, without))} give no data table to fit"
        )
    if study.kind == "profile" and len(study.experiments) != 1:
        raise StudyError(
            f"{where}: a profile study of a reactor takes one experiment, not "
            f"{len(study.experiments)}"
        )
    names = experiment_names(study)
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise StudyError(
            f"{where}: the experiments are named by their data tables, and "
            f"{', '.join(twice)} names more than one"
        )


def experiment_names(study: _Study) -> list[str]:
    """Each experiment's name: its data table's, without the extension."""
    return [
        Path(e.data).stem if e.data is not None else f"experiment {i + 1}"
        for i, e in enumerate(study.experiments)
    ]


# ---------------------------------------------------------------------------
# Where the model and the data are, and what they are now
# ---------------------------------------------------------------------------


def _model_source(folder: Path, study: _Study, frozen: bool) -> tuple[str, str | Path]:
    """``("shipped", name)`` or ``("file", path)``: the model the study reads,
    the snapshot's flattened copy when ``frozen``."""
    from aquakin.plant.spec import _is_file_base, _shipped_names

    ref = study.reactor.model
    if frozen:
        path = folder / SNAPSHOT / _MODELS / f"{Path(ref).stem}.yaml"
    elif _is_file_base(ref):
        path = Path(ref) if Path(ref).is_absolute() else folder / ref
    elif ref in _shipped_names():
        return "shipped", ref
    else:
        raise StudyError(
            f"reactor.model: {ref!r} is neither a model file (a path ending in .yaml) nor "
            f"a shipped model"
        )
    if not path.is_file():
        raise StudyError(f"reactor.model: {path} does not exist")
    return "file", path


def _data_file(folder: Path, data: str, frozen: bool) -> Path:
    return folder / SNAPSHOT / _DATA / Path(data).name if frozen else folder / data


def reactor_inputs(folder: Path, study: _Study, *, frozen: bool) -> tuple[dict[str, str], str]:
    """What a reactor study's results depend on besides its settings.

    Parameters
    ----------
    folder : pathlib.Path
        The study folder.
    study : _Study
        Its settings.
    frozen : bool
        Read the snapshot's model and data rather than the live ones.

    Returns
    -------
    inputs : dict
        ``{"model:<reactor.model>": fingerprint, "data:<file name>": sha256}``:
        the model with its ``extends`` chain merged in (a shipped model by its
        shipped content), and each data table by its bytes.
    fingerprint : str
        One sha256 of them all.

    Raises
    ------
    StudyError
        If the model or a data table is missing.
    """
    from aquakin.plant.spec import _model_fingerprint, _shipped_fingerprint

    kind, source = _model_source(folder, study, frozen)
    model = _shipped_fingerprint(source) if kind == "shipped" else _model_fingerprint(source)
    inputs = {f"model:{study.reactor.model}": model}
    for i, experiment in enumerate(study.experiments):
        if experiment.data is None:
            continue
        path = _data_file(folder, experiment.data, frozen)
        if not path.is_file():
            raise StudyError(f"experiments[{i}].data: {path} does not exist")
        inputs[f"data:{Path(experiment.data).name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    text = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return inputs, hashlib.sha256(text.encode()).hexdigest()


def freeze_reactor(folder: Path, study: _Study, snapshot: Path) -> None:
    """Copy the study's model, flattened (its ``extends`` chain merged in, a
    shipped model included), and its data tables into ``snapshot``."""
    kind, source = _model_source(folder, study, frozen=False)
    (snapshot / _MODELS).mkdir(parents=True)
    stem = Path(study.reactor.model).stem
    ref = {"source": "shipped"} if kind == "shipped" else {"source": str(Path(source).resolve())}
    (snapshot / _MODELS / f"{stem}.yaml").write_text(_flattened(ref, stem, folder))
    (snapshot / _DATA).mkdir()
    for i, experiment in enumerate(study.experiments):
        if experiment.data is None:
            continue
        path = _data_file(folder, experiment.data, frozen=False)
        if not path.is_file():
            raise StudyError(f"experiments[{i}].data: {path} does not exist")
        shutil.copyfile(path, snapshot / _DATA / path.name)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class _Batch:
    """A reactor study set up to run: the reactor, the parameters, and each
    experiment's initial state, with the factor that takes the study's times
    to the model's."""

    model: object
    reactor: object
    params: object
    initial: list
    names: list[str]
    to_model_time: float


def _setting(where: str, read):
    """``read()``, its ``KeyError`` or ``ValueError`` (an unknown name, a bad
    unit) given as a :class:`StudyError` that names the setting."""
    try:
        return read()
    except (KeyError, ValueError) as exc:
        raise StudyError(f"{where}: {exc.args[0] if exc.args else exc}") from None


def _set_up(folder: Path, study: _Study, frozen: bool) -> _Batch:
    from aquakin import BatchReactor, IntegratorConfig, load_model, load_model_from_file
    from aquakin.integrate._common import native_time_factor

    kind, source = _model_source(folder, study, frozen)
    model = load_model(source) if kind == "shipped" else load_model_from_file(source)
    unknown = [s for s in study.observed if s not in model.species_index]
    if unknown:
        raise StudyError(
            f"observed: the model has no species {', '.join(unknown)} (its species: "
            f"{', '.join(model.species)})"
        )
    settings = study.reactor.integrator.model_dump(exclude_none=True)
    tolerances = {k: settings.pop(k) for k in ("rtol", "atol") if k in settings}
    conditions = _setting(
        "reactor.conditions", lambda: model.default_conditions().with_(**study.reactor.conditions)
    )
    params = _setting("parameters", lambda: model.parameter_values(study.parameters))
    initial = [
        _setting(f"experiments[{i}].initial", lambda e=e: model.concentrations(e.initial))
        for i, e in enumerate(study.experiments)
    ]
    factor = _setting("time_unit", lambda: native_time_factor(model.time_unit, study.time_unit))
    reactor = BatchReactor(model, conditions, integrator=IntegratorConfig(**settings), **tolerances)
    return _Batch(model, reactor, params, initial, experiment_names(study), factor)


def _named(model, params) -> dict[str, float]:
    values = np.asarray(params, dtype=float)
    return {name: float(values[model.param_index[name]]) for name in model.parameters}


def _observations(folder: Path, study: _Study, frozen: bool) -> list[tuple]:
    """Each experiment's ``(t, observations (n_t, n_observed))`` from its table."""
    return [
        _read_observations(
            _data_file(folder, e.data, frozen), list(study.observed), f"experiments[{i}].data"
        )
        for i, e in enumerate(study.experiments)
    ]


def _on_one_grid(series: list[tuple[str, np.ndarray, dict]]) -> tuple[np.ndarray, dict]:
    """Experiments' series, each ``(name, t, {column: values})``, as one table:
    every time any of them has, ``<name>.<column>`` columns, and ``nan`` where
    an experiment has no value."""
    t_all = np.unique(np.concatenate([np.asarray(t, dtype=float) for _, t, _ in series]))
    columns = {}
    for name, t, values in series:
        at = np.searchsorted(t_all, np.asarray(t, dtype=float))
        for column, v in values.items():
            full = np.full(t_all.shape, np.nan)
            full[at] = np.asarray(v, dtype=float)
            columns[f"{name}.{column}"] = full
    return t_all, columns


def observed_data(folder: Path, study: _Study) -> tuple[np.ndarray, dict] | None:
    """The observed columns of every experiment's data table on one grid (see
    :func:`_on_one_grid`), for a front end to show beside the fit; ``None``
    when no table can be read."""
    from aquakin.utils.result_files import read_trajectory

    frozen = (folder / FROZEN_FILE).is_file()
    series = []
    for name, e in zip(experiment_names(study), study.experiments):
        path = None if e.data is None else _data_file(folder, e.data, frozen)
        if path is not None and path.is_file():
            t, columns = read_trajectory(path)
            series.append((name, t, {s: columns[s] for s in study.observed if s in columns}))
    return _on_one_grid(series) if series else None


def _run_calibrate(batch: _Batch, folder, study, out: Path, frozen: bool, progress) -> list[Path]:
    from aquakin import calibrate
    from aquakin.progress import report

    data = _observations(folder, study, frozen)
    report(progress, "fit", 0, 1, "fitting")
    result = calibrate(
        batch.reactor,
        batch.initial,
        [y for _, y in data],
        [t * batch.to_model_time for t, _ in data],
        list(study.free),
        initial_params=batch.params,
        observed_species=list(study.observed),
        laplace=study.laplace,
        **_fit_options(study),
    )
    report(progress, "fit", 1, 1, "writing the results")
    result.save(out)
    written = sorted(out.iterdir())
    # The fitted reactor's observed species at each experiment's data times.
    series = []
    columns = [batch.model.species_index[s] for s in study.observed]
    for name, c0, (t, _) in zip(batch.names, batch.initial, data):
        t_model = t * batch.to_model_time
        sol = batch.reactor.solve(
            c0, (0.0, float(t_model[-1])), t_eval=t_model, params=result.params
        )
        values = np.asarray(sol.C)[:, columns]
        series.append((name, t, dict(zip(study.observed, values.T))))
    written.append(write_trajectory(out / FITTED_FILE, *_on_one_grid(series)))
    return written


def _run_profile(batch: _Batch, folder, study, out: Path, frozen: bool, progress) -> list[Path]:
    from aquakin import profile_likelihood
    from aquakin.progress import report

    ((t, observations),) = _observations(folder, study, frozen)
    report(progress, "profile", 0, 1, "profiling")
    prof = profile_likelihood(
        batch.reactor,
        batch.initial[0],
        observations,
        t * batch.to_model_time,
        list(study.free),
        grid=np.asarray(_times(study.grid)),
        profile_param=study.profile,
        delta=study.delta,
        initial_params=batch.params,
        observed_species=list(study.observed),
        **_fit_options(study),
    )
    return _write_profile_files(out, study, prof)


def _run_fisher(batch: _Batch, folder, study, out: Path, frozen: bool, progress) -> list[Path]:
    import jax.numpy as jnp

    from aquakin import Experiment, fisher_information
    from aquakin.progress import report

    t_obs = jnp.asarray(_times(study.t_obs)) * batch.to_model_time
    sigma = np.asarray(study.sigma, dtype=float) if isinstance(study.sigma, list) else study.sigma
    design = [
        Experiment(c0, t_obs=t_obs, sigma=sigma, observed_species=list(study.observed))
        for c0 in batch.initial
    ]
    report(progress, "fisher", 0, 1, "computing the information")
    fim = fisher_information(batch.reactor, design, list(study.free), params=batch.params)
    return _write_fisher_files(out, fim, _named(batch.model, batch.params))


#: Each kind's reactor runner; one per entry of :data:`REQUIRED`.
_RUNNERS = {"calibrate": _run_calibrate, "profile": _run_profile, "fisher": _run_fisher}


def run_reactor_study(folder: Path, study: _Study, out: Path, progress) -> list[Path]:
    """Run a reactor study, writing its results into ``out``.

    Parameters
    ----------
    folder : pathlib.Path
        The study folder.
    study : _Study
        Its settings, with a ``reactor``.
    out : pathlib.Path
        Where to write the results.
    progress : callable or None
        Called with each :class:`aquakin.Progress` report.

    Returns
    -------
    list of pathlib.Path
        The files written.

    Raises
    ------
    StudyError
        If the model, a species, a condition or a data column it names is not
        there.
    """
    from aquakin.progress import report

    report(progress, "setup", 0, None, "building the reactor")
    frozen = (folder / FROZEN_FILE).is_file()
    batch = _set_up(folder, study, frozen)
    return _RUNNERS[study.kind](batch, folder, study, out, frozen, progress)
