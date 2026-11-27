"""A study: one analysis of a flowsheet project, saved in the project with its results.

A study lives in ``<project>/studies/<name>/``::

    study.yaml      what to run (written by the user or the editor; never rewritten)
    run.yaml        the record of the last run (written by :func:`run_study`)
    results/        what the run produced, every value keyed by name
    frozen.yaml     present while the study is frozen: what the snapshot holds
    snapshot/       the frozen plant: flowsheet.yaml, models/, data/

A **live** study builds the plant from the project's ``flowsheet.yaml`` and is
stale once what that simulates has changed since the run
(:func:`aquakin.plant.spec.spec_inputs` says which input changed). A **frozen**
study builds it from its snapshot, in which every model is a flattened copy
(shipped models too), so neither an edit to the project nor an aquakin upgrade
can change it. A change of aquakin version is noted, never counted as stale.

The kinds of study: ``simulate`` (a steady state, or a dynamic run from a
chosen initial state), ``fisher`` (the Fisher information of a sensor design),
``calibrate`` (a fit to observations in a data table), ``profile`` (the profile
likelihood of one parameter) and ``rank`` (sensor sets ranked by the
information they give).

A ``calibrate``, ``profile`` or ``fisher`` study can instead study a batch
reactor, given by a ``reactor:`` block in place of the flowsheet
(:mod:`aquakin.study.reactor`): the runners dispatch on the target, and the
results, their staleness and freezing work as for a plant.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ValidationError, field_validator

from aquakin.integrate.calibration.saved import fit_summary, write_fit_files
from aquakin.study.analysis import (
    _fit_options,
    _flattened,
    _read_observations,
    _write_fisher_files,
    _write_profile_files,
)

# Re-exported: the tests and the worker take them from here.
from aquakin.study.analysis import _fit_sigma as _fit_sigma
from aquakin.study.common import (
    _STRICT,
    ATTEMPT_FILE,
    FINAL_STATE_FILE,
    FITTED_FILE,
    FROZEN_FILE,
    KINDS,
    MEASURED_FILE,
    MEASURED_STEADY_FILE,
    PARAMETERS_FILE,
    PARTIAL,
    RANKING_FILE,
    RESULT_STATES,
    RESULTS,
    RUN_FILE,
    SNAPSHOT,
    SPEC_FILE,
    STREAMS_FILE,
    STREAMS_STEADY_FILE,
    STUDY_FILE,
    StudyError,
    _Grid,
    _now,
    _times,
    _to_float,
)
from aquakin.study.common import COVARIANCE_FILE as COVARIANCE_FILE
from aquakin.study.common import FISHER_FILE as FISHER_FILE
from aquakin.study.common import FISHER_SUMMARY_FILE as FISHER_SUMMARY_FILE
from aquakin.study.common import FIT_FILE as FIT_FILE
from aquakin.study.common import PROFILE_FILE as PROFILE_FILE
from aquakin.study.common import PROFILE_SUMMARY_FILE as PROFILE_SUMMARY_FILE
from aquakin.study.common import STATES as STATES
from aquakin.study.common import STUDIES_DIR as STUDIES_DIR
from aquakin.study.reactor import REQUIRED as REACTOR_REQUIRED
from aquakin.study.reactor import (
    check_reactor_study,
    freeze_reactor,
    reactor_inputs,
    run_reactor_study,
)
from aquakin.utils.result_files import (
    read_named_values,
    write_named_values,
    write_trajectory,
    write_yaml,
)

__all__ = [
    "StudyError",
    "StudyStatus",
    "check_study",
    "copy_study",
    "freeze_study",
    "load_study",
    "run_study",
    "unfreeze_study",
]

# ---------------------------------------------------------------------------
# What a study runs (study.yaml)
# ---------------------------------------------------------------------------


class _Measure(BaseModel):
    model_config = _STRICT
    measure: str
    weights: dict[str, float] = {}
    keywords: dict[str, float] = {}
    flow: float = 0.0
    temperature: float = 0.0

    @field_validator("weights", "keywords", mode="before")
    @classmethod
    def _int_weights(cls, v):
        return {k: _to_float(x) for k, x in v.items()} if isinstance(v, dict) else v

    _floats = field_validator("flow", "temperature", mode="before")(_to_float)


class _Observable(BaseModel):
    model_config = _STRICT
    stream: str
    channels: list[str | _Measure] | None = None

    @field_validator("channels")
    @classmethod
    def _readable_measures(cls, v):
        """Refuse an unreadable measure expression as the study is written.

        Only the syntax: the species an expression names are resolved against
        the model of the stream it reads, which needs the flowsheet, so they are
        checked when the study runs.
        """
        from aquakin.plant.measures import is_measure_expression, parse_measure

        for channel in v or []:
            if is_measure_expression(channel):
                parse_measure(channel)
        return v


class _Time(BaseModel):
    model_config = _STRICT
    t_span: tuple[float, float]
    t_eval: _Grid | list[float] | None = None
    time_unit: str | None = None

    @field_validator("t_span", mode="before")
    @classmethod
    def _span(cls, v):
        return tuple(_to_float(x) for x in v) if isinstance(v, (list, tuple)) else v

    @field_validator("t_eval", mode="before")
    @classmethod
    def _eval(cls, v):
        return [_to_float(x) for x in v] if isinstance(v, list) else v


class _Candidate(BaseModel):
    """One sensor set of a ``rank`` study."""

    model_config = _STRICT
    name: str
    observables: list[_Observable]
    sigma: float | list[float]
    relative_sigma: bool = False
    sigma_floor: float = 0.0

    @field_validator("sigma", "sigma_floor", mode="before")
    @classmethod
    def _int_sigma(cls, v):
        return [_to_float(x) for x in v] if isinstance(v, list) else _to_float(v)


def _float_values(v):
    return {k: _to_float(x) for k, x in v.items()} if isinstance(v, dict) else v


class _Integrator(BaseModel):
    """A reactor study's integrator settings; each left out keeps the reactor's own."""

    model_config = _STRICT
    rtol: float | None = None
    atol: float | None = None
    order: Literal[3, 5] | None = None
    dtmax: float | None = None
    max_steps: int | None = None

    _floats = field_validator("rtol", "atol", "dtmax", mode="before")(_to_float)


class _Reactor(BaseModel):
    """The batch reactor a reactor study runs: its model (a model file relative
    to the study folder, or a shipped model's name), its conditions by name, and
    its integrator settings."""

    model_config = _STRICT
    model: str
    conditions: dict[str, float] = {}
    integrator: _Integrator = _Integrator()

    _conditions = field_validator("conditions", mode="before")(_float_values)


class _Experiment(BaseModel):
    """One batch of a reactor study: its initial concentrations by species, and
    the table of its observations (relative to the study folder) for a fit."""

    model_config = _STRICT
    data: str | None = None
    initial: dict[str, float] = {}

    _initial = field_validator("initial", mode="before")(_float_values)


class _Study(BaseModel):
    model_config = _STRICT
    kind: Literal[KINDS]
    description: str | None = None
    # The flowsheet, relative to the study folder (a live study's plant).
    flowsheet: str = f"../../{SPEC_FILE}"
    # In place of the flowsheet: a batch reactor (calibrate, profile, fisher),
    # its experiments and the species observed in them.
    reactor: _Reactor | None = None
    experiments: list[_Experiment] = []
    observed: list[str] = []
    # Plant parameters changed from the flowsheet's, "<model>.<param>": value.
    parameters: dict[str, float] = {}
    # "default" (each unit's own), "steady" (the steady state at the
    # parameters), or a named-state file relative to the study folder.
    initial_state: str = "default"
    # simulate: a dynamic run when given, otherwise the steady state.
    time: _Time | None = None
    # simulate: streams saved whole (flow, temperature, every species).
    streams: list[str] = []
    # simulate: measured channels saved; fisher: the sensor design.
    observables: list[_Observable] = []
    # fisher
    free: list[str] = []
    t_obs: _Grid | list[float] | None = None
    t_span: tuple[float, float] | None = None
    sigma: float | list[float] | None = None
    relative_sigma: bool = False
    sigma_floor: float = 0.0
    sensitivity: Literal["forward", "reverse"] = "forward"
    time_unit: str | None = None
    # calibrate / profile: the observations, a CSV table relative to the study
    # folder with a "t" column and one "<stream>.<channel>" column per channel.
    data: str | None = None
    # The fit is always the Gaussian likelihood of the data, so it needs each
    # channel's measurement error (``sigma``) or ``estimate_sigma``: least squares
    # ignores the error, which leaves uncertainties, profile intervals and the
    # weight of priors without a scale.
    estimate_sigma: bool = False
    transforms: dict[str, Literal["none", "positive_log", "logit"]] = {}
    use_priors: bool = True
    n_starts: int = 1
    max_iter: int = 500
    laplace: bool = False
    # profile
    profile: str | None = None
    grid: _Grid | list[float] | None = None
    delta: float = 1.92
    # rank
    candidates: list[_Candidate] = []
    base: _Candidate | None = None
    criterion: Literal["D", "A", "E", "modified_E"] = "D"

    @field_validator("grid", mode="before")
    @classmethod
    def _int_grid(cls, v):
        return [_to_float(x) for x in v] if isinstance(v, list) else v

    _delta = field_validator("delta", mode="before")(_to_float)

    @field_validator("parameters", mode="before")
    @classmethod
    def _int_parameters(cls, v):
        return {k: _to_float(x) for k, x in v.items()} if isinstance(v, dict) else v

    @field_validator("t_obs", mode="before")
    @classmethod
    def _int_times(cls, v):
        return [_to_float(x) for x in v] if isinstance(v, list) else v

    @field_validator("t_span", mode="before")
    @classmethod
    def _int_span(cls, v):
        return tuple(_to_float(x) for x in v) if isinstance(v, (list, tuple)) else v

    @field_validator("sigma", "sigma_floor", mode="before")
    @classmethod
    def _int_sigma(cls, v):
        return [_to_float(x) for x in v] if isinstance(v, list) else _to_float(v)


_REQUIRED = {
    # A simulation needs streams or observables, either one (see _check_kind).
    "simulate": (),
    "fisher": ("free", "observables", "t_obs", "sigma"),
    "calibrate": ("free", "observables", "data"),
    "profile": ("free", "observables", "data", "profile", "grid"),
    "rank": ("free", "candidates", "t_obs"),
}


def _check_target(study: _Study, where: str) -> None:
    """A study studies the flowsheet (by default the project's) or a reactor."""
    if study.reactor is not None:
        if "flowsheet" in study.model_fields_set:
            raise StudyError(f"{where}: give a flowsheet or a reactor to study, not both")
        check_reactor_study(study, where)
    elif study.experiments or study.observed:
        raise StudyError(
            f"{where}: experiments and observed are the settings of a reactor study; "
            f"give the reactor (its model and conditions) in a reactor: block"
        )


def _check_kind(study: _Study, where: str) -> None:
    _check_target(study, where)
    if study.kind == "simulate":
        if not study.streams and not study.observables:
            raise StudyError(f"{where}: a simulate study saves at least one stream or observable")
        return
    required = _REQUIRED[study.kind] if study.reactor is None else REACTOR_REQUIRED[study.kind]
    missing = [
        name
        for name in required
        if getattr(study, name) is None or getattr(study, name) == [] or getattr(study, name) == ""
    ]
    if missing:
        raise StudyError(f"{where}: a {study.kind} study needs {', '.join(missing)}")
    if study.kind == "fisher" and study.reactor is not None:
        _check_sigma_count(study, where)
    if study.kind in ("calibrate", "profile"):
        if any(o.channels is None for o in study.observables):
            raise StudyError(
                f"{where}: a {study.kind} study names the channels of each observable, so "
                f"each can be found in the data"
            )
        if study.estimate_sigma and study.sigma is not None:
            raise StudyError(f"{where}: give sigma or estimate_sigma, not both")
        if not study.estimate_sigma and study.sigma is None:
            raise StudyError(
                f"{where}: a {study.kind} study fits the likelihood of the data, so give "
                f"the measurement error (sigma) or estimate it (estimate_sigma)"
            )
        _check_sigma_count(study, where)


def _check_sigma_count(study: _Study, where: str) -> None:
    """A ``sigma`` list gives one error per measured channel (observed species)."""
    if not isinstance(study.sigma, list):
        return
    n_channels = (
        len(study.observed)
        if study.reactor is not None
        else sum(len(_channel_names(_channels(o))) for o in study.observables)
    )
    if len(study.sigma) != n_channels:
        raise StudyError(
            f"{where}: sigma gives {len(study.sigma)} errors for {n_channels} measured "
            f"channel{'s' if n_channels != 1 else ''}; give one for all, or one per channel"
        )


def load_study(path) -> _Study:
    """Read and check a study's ``study.yaml``.

    Parameters
    ----------
    path : str or pathlib.Path
        The study folder (or its ``study.yaml``).

    Returns
    -------
    The validated study inputs.

    Raises
    ------
    StudyError
        If the file is missing or does not describe a study.
    """
    folder = _folder(path)
    where = str(folder / STUDY_FILE)
    try:
        data = yaml.safe_load((folder / STUDY_FILE).read_text())
    except FileNotFoundError:
        raise StudyError(f"{where} does not exist") from None
    except yaml.YAMLError as exc:
        raise StudyError(f"{where} is not valid YAML: {exc}") from None
    if not isinstance(data, dict) or "study" not in data:
        raise StudyError(f"{where} has no 'study:' section")
    try:
        study = _Study.model_validate(data["study"])
    except ValidationError as exc:
        lines = [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors(include_url=False)
        ]
        raise StudyError(f"{where}:\n  " + "\n  ".join(lines)) from None
    _check_kind(study, where)
    return study


def _folder(path) -> Path:
    path = Path(path)
    return path.parent if path.name == STUDY_FILE else path


def _study_fingerprint(folder: Path, study: _Study) -> str:
    """What the study's own inputs are: its settings, and any file they read."""
    inputs: dict[str, Any] = {"study": study.model_dump(mode="json")}
    files = {"data": study.data}
    if study.initial_state not in ("default", "steady"):
        files["initial_state"] = study.initial_state
    for key, rel in files.items():
        if rel is not None and (folder / rel).is_file():
            inputs[key] = hashlib.sha256((folder / rel).read_bytes()).hexdigest()
    text = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _read_yaml(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else None


def _flowsheet(folder: Path, study: _Study) -> Path:
    """The spec the study builds its plant from: the snapshot when frozen."""
    if (folder / FROZEN_FILE).is_file():
        return folder / SNAPSHOT / SPEC_FILE
    return (folder / study.flowsheet).resolve()


def _inputs(folder: Path, study: _Study) -> tuple[dict[str, str], str]:
    """What the study's target simulates: its inputs' fingerprints by name (the
    flowsheet, each model and table), and one fingerprint of them all."""
    from aquakin.plant.spec import spec_fingerprint, spec_inputs

    if study.reactor is not None:
        return reactor_inputs(folder, study, frozen=(folder / FROZEN_FILE).is_file())
    spec = _flowsheet(folder, study)
    return spec_inputs(spec), spec_fingerprint(spec)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@dataclass
class StudyStatus:
    """Where a study stands.

    Attributes
    ----------
    state : {"not run", "fresh", "stale", "damaged", "failed", "cancelled"}
        ``"fresh"``: the results are those of the study's inputs and plant.
        ``"stale"``: something they were computed from has changed (see
        ``reasons``). ``"damaged"``: a result file is missing or was edited.
        ``"failed"``: the last run ended in an error (``reasons`` holds it).
        ``"cancelled"``: the last run was stopped before it finished.
    frozen : bool
        Whether the study runs on its snapshot.
    reasons : list of str
        Why the study is stale, damaged or failed.
    notes : list of str
        What changed without making it stale (the aquakin version).
    """

    state: str
    frozen: bool
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_results(self) -> bool:
        """Whether the last run finished, so there are results to show or replace."""
        return self.state in RESULT_STATES


def _changed_inputs(before: dict, now: dict) -> list[str]:
    out = []
    for key in sorted(set(before) | set(now)):
        if key not in now:
            out.append(f"{_describe(key)} is no longer used")
        elif key not in before:
            out.append(f"{_describe(key)} is new")
        elif before[key] != now[key]:
            out.append(f"{_describe(key)} has changed")
    return out


def _describe(key: str) -> str:
    if key == "flowsheet":
        return "the flowsheet"
    kind, _, name = key.partition(":")
    if kind == "model":
        return f"model '{name}'"
    if kind == "data":
        return f"data table '{name}'"
    return f"the influent table of '{name}'"


def check_study(path) -> StudyStatus:
    """Whether a study's results still are those of its inputs and its plant.

    Parameters
    ----------
    path : str or pathlib.Path
        The study folder.

    Returns
    -------
    StudyStatus
    """
    from aquakin.plant.spec import aquakin_version

    folder = _folder(path)
    study = load_study(folder)
    frozen = _read_yaml(folder / FROZEN_FILE)
    run = _read_yaml(folder / RUN_FILE)
    if run is None:
        return StudyStatus("not run", frozen is not None)
    status = StudyStatus("fresh", frozen is not None)
    attempt = _read_yaml(folder / ATTEMPT_FILE)
    if attempt is not None:
        what = "was cancelled" if attempt.get("outcome") == "cancelled" else "failed"
        error = "" if attempt.get("outcome") == "cancelled" else f" ({attempt.get('error')})"
        status.notes.append(
            f"the last run, started {attempt.get('started')}, {what}{error}; these are the "
            f"results of the run before"
        )
    if run.get("aquakin") != aquakin_version():
        status.notes.append(
            f"run with aquakin {run.get('aquakin')}; this is aquakin {aquakin_version()}"
        )
    if run.get("outcome") == "cancelled":
        status.state = "cancelled"
        return status
    if run.get("outcome") != "finished":
        status.state = "failed"
        status.reasons.append(str(run.get("error", "the run did not finish")))
        return status
    for rel, digest in (run.get("results") or {}).items():
        target = folder / rel
        if not target.is_file():
            status.reasons.append(f"{rel} is missing")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            status.reasons.append(f"{rel} has been edited")
    if status.reasons:
        status.state = "damaged"
        return status
    if run.get("study") != _study_fingerprint(folder, study):
        status.reasons.append("the study's settings have changed")
    if frozen is not None:
        if run.get("spec_fingerprint") != frozen.get("spec_fingerprint"):
            status.reasons.append("the study was frozen on another plant than its results'")
    else:
        target = "flowsheet" if study.reactor is None else "reactor's model or data"
        try:
            now = _inputs(folder, study)[0]
        except Exception as exc:
            now = None
            status.reasons.append(f"the {target} cannot be read: {exc}")
        if now is not None:
            status.reasons.extend(_changed_inputs(run.get("spec_inputs") or {}, now))
    if status.reasons:
        status.state = "stale"
    return status


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _channels(observable: _Observable):
    from aquakin.plant import Measure, PlantObservable

    channels = None
    if observable.channels is not None:
        channels = tuple(
            c
            if isinstance(c, str)
            else Measure(
                c.measure, c.weights, keywords=c.keywords, flow=c.flow, temperature=c.temperature
            )
            for c in observable.channels
        )
    return PlantObservable(observable.stream, channels)


def _channel_names(observable) -> list[str]:
    """The names a resolved observable's channels are reported under.

    A measure expression is reported under its own name, not the text that
    defines it, so ``"BOD5 = 0.65*[SS]"`` becomes the column ``"<stream>.BOD5"``.
    """
    from aquakin.plant import channel_name

    if observable.channels is None:
        return []
    return [channel_name(c) for c in observable.channels]


def _initial_state(plant, folder: Path, study: _Study, params):
    if study.initial_state == "default":
        return plant.initial_state()
    if study.initial_state == "steady":
        result = plant.steady_state(params)
        if not bool(result.converged):
            raise StudyError("the steady state used as the initial state did not converge")
        return result.state
    path = folder / study.initial_state
    if not path.is_file():
        raise StudyError(f"initial_state: {path} does not exist")
    return plant.state_from_named(read_named_values(path))


def _measured_columns(plant, sol, observables, params) -> dict:
    """``{"<stream>.<channel>": values}`` of measured channels over a solution."""
    import numpy as np

    from aquakin.plant.calibrate import _resolve_observable_columns, measured

    columns = {}
    resolved = _resolve_observable_columns(plant, observables)[0]
    for observable, (_, m) in zip(observables, resolved):
        s = plant.stream(sol, observable.stream, params)
        values = np.asarray(measured(s.C, s.Q, s.T, m, params))
        names = _channel_names(observable) or list(s.model.species)
        for j, channel in enumerate(names):
            columns[f"{observable.stream}.{channel}"] = values[:, j]
    return columns


def _run_simulate(
    plant, folder: Path, study: _Study, params, y0, out: Path, progress
) -> list[Path]:
    import numpy as np

    from aquakin.plant.calibrate import _resolve_observable_columns, measured
    from aquakin.progress import report

    written = [write_named_values(out / PARAMETERS_FILE, _named_parameters(plant, params))]
    observables = [_channels(o) for o in study.observables]
    resolved = _resolve_observable_columns(plant, observables)[0] if observables else ()
    if study.time is None:
        report(progress, "simulate", 0, 1, "finding the steady state")
        result = plant.steady_state(params, y0=y0)
        if not bool(result.converged):
            raise StudyError("the steady state did not converge")
        state = result.state
        outputs = plant.outputs_at(0.0, state, params)
        streams = {}
        for name in study.streams:
            s = outputs[plant._parse_endpoint(plant.named_streams.get(name, name), role="source")]
            values = {"Q": s.Q, **{sp: s.C[i] for i, sp in enumerate(s.model.species)}}
            if s.scalars.get("T") is not None:
                values["T"] = s.scalars["T"]
            streams[name] = values
        if streams:
            written.append(write_named_values(out / STREAMS_STEADY_FILE, streams))
        if resolved:
            channels = {}
            for observable, (endpoint, m) in zip(observables, resolved):
                s = outputs[plant._parse_endpoint(endpoint, role="source")]
                values = np.asarray(measured(s.C, s.Q, s.scalars.get("T"), m, params))
                names = _channel_names(observable) or list(s.model.species)
                channels[observable.stream] = dict(zip(names, values))
            written.append(write_named_values(out / MEASURED_STEADY_FILE, channels))
        written.append(write_named_values(out / FINAL_STATE_FILE, plant.named_state(state)))
        return written

    t_eval = None if study.time.t_eval is None else np.asarray(_times(study.time.t_eval))
    report(progress, "simulate", 0, 1, "solving")
    sol = plant.solve(
        study.time.t_span, t_eval=t_eval, params=params, y0=y0, time_unit=study.time.time_unit
    )
    columns = {}
    for name in study.streams:
        s = plant.stream(sol, name, params)
        columns[f"{name}.Q"] = s.Q
        if s.T is not None:
            columns[f"{name}.T"] = s.T
        for i, sp in enumerate(s.model.species):
            columns[f"{name}.{sp}"] = np.asarray(s.C)[:, i]
    report(progress, "simulate", 1, 1, "writing the results")
    if columns:
        written.append(write_trajectory(out / STREAMS_FILE, sol.t, columns))
    if resolved:
        measured_columns = _measured_columns(plant, sol, observables, params)
        written.append(write_trajectory(out / MEASURED_FILE, sol.t, measured_columns))
    final = np.asarray(sol.state)[-1]
    written.append(write_named_values(out / FINAL_STATE_FILE, plant.named_state(final)))
    return written


def _run_fisher(plant, folder: Path, study: _Study, params, y0, out: Path, progress) -> list[Path]:
    import numpy as np

    from aquakin.plant import plant_fisher_information

    observables = [_channels(o) for o in study.observables]
    fim = plant_fisher_information(
        plant,
        observables,
        np.asarray(_times(study.t_obs)),
        list(study.free),
        sigma=study.sigma,
        relative_sigma=study.relative_sigma,
        sigma_floor=study.sigma_floor,
        t_span=study.t_span,
        y0=y0,
        params=params,
        mode=study.sensitivity,
        time_unit=study.time_unit,
        progress=progress,
    )
    return _write_fisher_files(out, fim, _named_parameters(plant, params))


def _observations(folder: Path, study: _Study):
    """``(t, observations (n_t, n_channels), channel names)`` from the study's data table."""
    names = [
        f"{o.stream}.{name}" for o in study.observables for name in _channel_names(_channels(o))
    ]
    t, observations = _read_observations(folder / study.data, names, "data")
    return t, observations, names


def _fit_arguments(plant, study: _Study, params, y0, progress) -> dict:
    return dict(
        observables=[_channels(o) for o in study.observables],
        t_span=study.t_span,
        y0=y0,
        params=params,
        time_unit=study.time_unit,
        progress=progress,
        **_fit_options(study),
    )


def _run_calibrate(
    plant, folder: Path, study: _Study, params, y0, out: Path, progress
) -> list[Path]:
    import numpy as np

    from aquakin.plant import calibrate_plant

    t, observations, names = _observations(folder, study)
    result = calibrate_plant(
        plant,
        observations,
        t,
        list(study.free),
        laplace=study.laplace,
        **_fit_arguments(plant, study, params, y0, progress),
    )
    fitted = np.asarray(result.params)
    written = write_fit_files(
        out, result, fit_summary(result, names), _named_parameters(plant, fitted)
    )
    # The fitted plant's channels at the observation times, beside the data.
    span = study.t_span or (0.0, float(t[-1]))
    sol = plant.solve(span, t_eval=t, params=fitted, y0=y0, time_unit=study.time_unit)
    observables = [_channels(o) for o in study.observables]
    written.append(
        write_trajectory(
            out / FITTED_FILE, sol.t, _measured_columns(plant, sol, observables, fitted)
        )
    )
    return written


def _run_profile(plant, folder: Path, study: _Study, params, y0, out: Path, progress) -> list[Path]:
    import numpy as np

    from aquakin.plant import plant_profile_likelihood

    t, observations, _ = _observations(folder, study)
    arguments = _fit_arguments(plant, study, params, y0, progress)
    prof = plant_profile_likelihood(
        plant,
        observations,
        t,
        list(study.free),
        grid=np.asarray(_times(study.grid)),
        profile_param=study.profile,
        delta=study.delta,
        **arguments,
    )
    return _write_profile_files(out, study, prof)


def _sensor_set(candidate: _Candidate):
    from aquakin.plant import SensorSet

    return SensorSet(
        [_channels(o) for o in candidate.observables],
        sigma=candidate.sigma,
        relative_sigma=candidate.relative_sigma,
        sigma_floor=candidate.sigma_floor,
        name=candidate.name,
    )


def _run_rank(plant, folder: Path, study: _Study, params, y0, out: Path, progress) -> list[Path]:
    import numpy as np

    from aquakin.plant import plant_rank_sensor_sets
    from aquakin.utils.result_files import write_table

    ranked = plant_rank_sensor_sets(
        plant,
        [_sensor_set(c) for c in study.candidates],
        np.asarray(_times(study.t_obs)),
        list(study.free),
        base=None if study.base is None else _sensor_set(study.base),
        criterion=study.criterion,
        params=params,
        t_span=study.t_span,
        y0=y0,
        mode=study.sensitivity,
        time_unit=study.time_unit,
        progress=progress,
    )
    written = [
        write_table(
            out / RANKING_FILE,
            {
                "rank": list(range(1, len(ranked) + 1)),
                "candidate": [study.candidates[c.index].name for c in ranked],
                study.criterion: [c.value for c in ranked],
                "improvement": [c.improvement for c in ranked],
            },
        ),
        write_named_values(out / PARAMETERS_FILE, _named_parameters(plant, params)),
    ]
    return written


_RUNNERS = {
    "simulate": _run_simulate,
    "fisher": _run_fisher,
    "calibrate": _run_calibrate,
    "profile": _run_profile,
    "rank": _run_rank,
}


def _named_parameters(plant, params) -> dict[str, float]:
    import numpy as np

    values = np.asarray(params, dtype=float)
    return {name: float(values[plant.parameter_index(name)]) for name in plant.parameter_names()}


def _run_plant_study(folder: Path, study: _Study, out: Path, progress) -> list[Path]:
    """Build the study's plant from its flowsheet and run its kind's runner."""
    import numpy as np

    from aquakin.plant import Plant
    from aquakin.progress import report

    report(progress, "setup", 0, None, "building the plant")
    plant = Plant.from_spec(_flowsheet(folder, study))
    params = plant.parameter_values(study.parameters) if study.parameters else None
    params = plant.default_parameters() if params is None else params
    y0 = _initial_state(plant, folder, study, params)
    runner = _RUNNERS[study.kind]
    return runner(plant, folder, study, params, np.asarray(y0), out, progress)


def run_study(path, *, replace: bool = False, progress=None) -> StudyStatus:
    """Run a study and write its results and the record of the run.

    Parameters
    ----------
    path : str or pathlib.Path
        The study folder.
    replace : bool, optional
        Replace results already there. Without it a study that has results is
        refused, so they are not lost by accident: copy the study first
        (:func:`copy_study`) to keep them.
    progress : callable, optional
        Passed to the analysis (see :mod:`aquakin.progress`); raising
        :class:`aquakin.Cancelled` from it stops the run, which is then
        recorded as cancelled.

    Returns
    -------
    StudyStatus
        ``"fresh"`` after a successful run.

    Raises
    ------
    StudyError
        If the study has results and ``replace`` is not set, or cannot be read.
        A failure during the run itself is recorded in ``run.yaml`` and
        re-raised.
    """
    from aquakin.plant.spec import aquakin_version

    folder = _folder(path)
    study = load_study(folder)
    out = folder / RESULTS
    previous = _read_yaml(folder / RUN_FILE)
    has_results = previous is not None and previous.get("outcome") == "finished"
    if has_results and not replace:
        raise StudyError(
            f"study {folder.name} already has results; run it with replace=True to replace "
            f"them (copy the study first to keep them)"
        )
    inputs, fingerprint = _inputs(folder, study)
    record: dict[str, Any] = {
        "started": _now(),
        "aquakin": aquakin_version(),
        "spec_fingerprint": fingerprint,
        "spec_inputs": inputs,
        "study": _study_fingerprint(folder, study),
    }
    # The run writes beside the results it would replace, which stay until it
    # succeeds: a run that fails, is cancelled or is killed loses nothing.
    partial = folder / PARTIAL
    if partial.exists():
        shutil.rmtree(partial)  # left by a run that was killed
    try:
        if study.reactor is not None:
            written = run_reactor_study(folder, study, partial, progress)
        else:
            written = _run_plant_study(folder, study, partial, progress)
    except BaseException as exc:
        from aquakin.progress import Cancelled

        cancelled = isinstance(exc, Cancelled)
        record |= {
            "finished": _now(),
            "outcome": "cancelled" if cancelled else "failed",
            "error": "cancelled" if cancelled else f"{type(exc).__name__}: {exc}",
        }
        shutil.rmtree(partial, ignore_errors=True)
        if has_results:
            write_yaml(folder / ATTEMPT_FILE, record)  # the results stand
        else:
            if out.exists():
                shutil.rmtree(out)  # files no finished run recorded
            write_yaml(folder / RUN_FILE, record)
        raise
    digests = {
        (Path(RESULTS) / p.relative_to(partial)).as_posix(): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in written
    }
    if out.exists():
        shutil.rmtree(out)
    partial.rename(out)
    (folder / ATTEMPT_FILE).unlink(missing_ok=True)
    record |= {"finished": _now(), "outcome": "finished", "results": digests}
    write_yaml(folder / RUN_FILE, record)
    return check_study(folder)


# ---------------------------------------------------------------------------
# Freezing, unfreezing and copying
# ---------------------------------------------------------------------------


def _freeze_plant(folder: Path, study: _Study, snapshot: Path) -> None:
    """Write the flowsheet into ``snapshot`` with every model flattened."""
    from aquakin.plant.spec import _located, load_spec_file, write_spec

    source = (folder / study.flowsheet).resolve()
    spec = load_spec_file(source)
    (snapshot / "models").mkdir(parents=True)
    models = {}
    for name, ref in (spec.get("models") or {}).items():
        flat = snapshot / "models" / f"{name}.yaml"
        if ref.get("source", "shipped") != "shipped":
            ref = {**ref, "source": str(_located(ref["source"], source.parent, name))}
        flat.write_text(_flattened(ref, name, source.parent))
        models[name] = {"source": str(flat)}
        if ref.get("source", "shipped") == "shipped":
            models[name]["copy_of"] = "shipped"
    write_spec(dict(spec, models=models), snapshot / SPEC_FILE, base_dir=source.parent)


def freeze_study(path) -> dict:
    """Freeze a study on a snapshot of what it studies as that is now.

    The snapshot (``snapshot/``) of a flowsheet study holds the flowsheet, every
    influent table, and every model flattened (its ``extends`` chain merged in,
    shipped models included); that of a reactor study, its model flattened and
    its data tables. Neither an edit to the project nor an aquakin upgrade can
    then change what the study runs. Results already there stay, and stay fresh
    only if they were computed on the same plant or reactor.

    Parameters
    ----------
    path : str or pathlib.Path
        The study folder.

    Returns
    -------
    dict
        The freeze record (also written to ``frozen.yaml``).

    Raises
    ------
    StudyError
        If the study is already frozen.
    """
    from aquakin.plant.spec import aquakin_version, spec_fingerprint, spec_inputs

    folder = _folder(path)
    study = load_study(folder)
    if (folder / FROZEN_FILE).exists():
        raise StudyError(f"study {folder.name} is already frozen")
    snapshot = folder / SNAPSHOT
    if snapshot.exists():
        shutil.rmtree(snapshot)
    try:
        if study.reactor is not None:
            freeze_reactor(folder, study, snapshot)
            origin = study.reactor.model
            inputs, fingerprint = reactor_inputs(folder, study, frozen=True)
        else:
            _freeze_plant(folder, study, snapshot)
            origin = study.flowsheet
            inputs = spec_inputs(snapshot / SPEC_FILE)
            fingerprint = spec_fingerprint(snapshot / SPEC_FILE)
        record = {
            "frozen": _now(),
            "aquakin": aquakin_version(),
            "from": origin,
            "spec_fingerprint": fingerprint,
            "spec_inputs": inputs,
        }
    except BaseException:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise
    write_yaml(folder / FROZEN_FILE, record)
    return record


def unfreeze_study(path, *, discard_results: bool = False) -> None:
    """Make a frozen study follow the project's flowsheet again.

    Its snapshot is deleted, and so are its results and run record, since they
    were computed on the snapshot. Copy the study first (:func:`copy_study`) to
    keep them.

    Parameters
    ----------
    path : str or pathlib.Path
        The study folder.
    discard_results : bool
        Must be ``True`` when the study has results: unfreezing deletes them.

    Raises
    ------
    StudyError
        If the study is not frozen, or has results and ``discard_results`` is
        not set.
    """
    folder = _folder(path)
    load_study(folder)
    if not (folder / FROZEN_FILE).exists():
        raise StudyError(f"study {folder.name} is not frozen")
    has_results = (folder / RESULTS).exists() or (folder / RUN_FILE).exists()
    if has_results and not discard_results:
        raise StudyError(
            f"unfreezing study {folder.name} deletes its results, which were computed on its "
            f"snapshot; copy the study first to keep them, then unfreeze with "
            f"discard_results=True"
        )
    shutil.rmtree(folder / RESULTS, ignore_errors=True)
    shutil.rmtree(folder / PARTIAL, ignore_errors=True)
    (folder / RUN_FILE).unlink(missing_ok=True)
    (folder / ATTEMPT_FILE).unlink(missing_ok=True)
    shutil.rmtree(folder / SNAPSHOT, ignore_errors=True)
    (folder / FROZEN_FILE).unlink()


def copy_study(path, name: str) -> Path:
    """Copy a study, with its results and any snapshot, beside it under ``name``.

    Parameters
    ----------
    path : str or pathlib.Path
        The study folder.
    name : str
        The copy's folder name (letters, digits, ``_`` and ``-``).

    Returns
    -------
    pathlib.Path
        The new study folder.

    Raises
    ------
    StudyError
        If ``name`` is not a valid name or a study of that name exists.
    """
    from aquakin.plant.errors import NAME_PATTERN

    folder = _folder(path)
    load_study(folder)
    if not NAME_PATTERN.match(name):
        raise StudyError(
            f"study name {name!r} must start with a letter or '_' and hold only letters, "
            f"digits, '_' and '-'"
        )
    dest = folder.parent / name
    if dest.exists():
        raise StudyError(f"{dest} already exists")
    shutil.copytree(folder, dest, ignore=shutil.ignore_patterns(PARTIAL))
    return dest
