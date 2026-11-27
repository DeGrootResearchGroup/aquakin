"""What a front end needs to let a person set up a study.

:func:`study_form` describes every setting of ``study.yaml``: its type (which a
front end turns into a widget), help text, default, and the study kinds that
use it. :func:`plant_choices` lists what a study of one flowsheet may name: its
streams, its parameters, and the channels each stream can measure.
:func:`create_study` and :func:`update_study` write a study from plain settings,
checked exactly as :func:`~aquakin.study.run_study` reads them, and copy any
data or initial-state file into the study folder so the study holds its own
inputs.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import yaml

from aquakin.study.common import (
    FISHER_FILE,
    FISHER_SUMMARY_FILE,
    FIT_FILE,
    FITTED_FILE,
    MEASURED_FILE,
    MEASURED_STEADY_FILE,
    PROFILE_FILE,
    PROFILE_SUMMARY_FILE,
    RANKING_FILE,
    RESULTS,
    STREAMS_FILE,
    STREAMS_STEADY_FILE,
    STUDIES_DIR,
    STUDY_FILE,
    StudyError,
)
from aquakin.study.reactor import REQUIRED as REACTOR_REQUIRED
from aquakin.study.reactor import experiment_names, observed_data
from aquakin.study.study import (
    _REQUIRED,
    _channel_names,
    _channels,
    _check_kind,
    _flowsheet,
    _folder,
    _Study,
    check_study,
    load_study,
)
from aquakin.utils.result_files import read_matrix, read_named_values, read_table, read_trajectory

__all__ = [
    "create_study",
    "plant_choices",
    "read_study",
    "simulation_study",
    "study_form",
    "study_results",
    "update_study",
]

_COMMON = ("description", "parameters", "initial_state")
_FIT = ("free", "observables", "data", "t_span", "sigma", "estimate_sigma")
_FIT_OPTIONS = ("transforms", "use_priors", "n_starts", "max_iter", "time_unit")

#: The fields a kind shows when it studies a batch reactor (``reactor:``) in place
#: of the flowsheet; a kind not listed takes no reactor.
_REACTOR_FIELDS = {
    "fisher": (
        "description",
        "reactor",
        "experiments",
        "observed",
        "parameters",
        "free",
        "t_obs",
        "sigma",
        "time_unit",
    ),
    "calibrate": (
        "description",
        "reactor",
        "experiments",
        "observed",
        "parameters",
        "free",
        "sigma",
        "estimate_sigma",
        *_FIT_OPTIONS,
        "laplace",
    ),
    "profile": (
        "description",
        "reactor",
        "experiments",
        "observed",
        "parameters",
        "free",
        "sigma",
        "estimate_sigma",
        "profile",
        "grid",
        "delta",
        *_FIT_OPTIONS,
    ),
}

_KINDS = {
    "simulate": (
        "Simulation",
        "A steady state, or a dynamic run, with chosen streams and measured channels saved.",
        (*_COMMON, "time", "streams", "observables"),
    ),
    "fisher": (
        "Fisher information",
        "How much a sensor design tells about chosen parameters.",
        (
            *_COMMON,
            "free",
            "observables",
            "t_obs",
            "t_span",
            "sigma",
            "relative_sigma",
            "sigma_floor",
            "sensitivity",
            "time_unit",
        ),
    ),
    "calibrate": (
        "Calibration",
        "Fit chosen parameters to observations in a data table.",
        (*_COMMON, *_FIT, *_FIT_OPTIONS, "laplace"),
    ),
    "profile": (
        "Profile likelihood",
        "Fit the other parameters at each value of one, to see how well the data fix it.",
        (*_COMMON, *_FIT, "profile", "grid", "delta", *_FIT_OPTIONS),
    ),
    "rank": (
        "Sensor ranking",
        "Rank candidate sensor sets by the information they give.",
        (
            *_COMMON,
            "free",
            "candidates",
            "base",
            "t_obs",
            "t_span",
            "criterion",
            "sensitivity",
            "time_unit",
        ),
    ),
}

# (type, label, help, default). Types: text, number, integer, boolean, choice
# (with choices), span (two numbers), times (an even grid or a list of times),
# run_time (a span, output times and their unit), numbers (one number or one per
# channel), parameter / parameters (plant parameter names), parameter_values
# (name -> number), transforms (name -> choice), streams, observables (list of
# {stream, channels}), candidates / candidate (named sensor sets), file, initial_state,
# reactor ({model, conditions, integrator}), experiments (list of {data, initial}),
# species (species names).
_FIELDS = {
    "description": ("text", "Description", "What the study is for.", None),
    "reactor": (
        "reactor",
        "Reactor",
        "A batch reactor to study in place of the flowsheet: its model (a model file, "
        "or a shipped model's name), its conditions by name, and optionally its "
        "integrator settings (rtol, atol, order, dtmax, max_steps).",
        None,
    ),
    "experiments": (
        "experiments",
        "Experiments",
        "The reactor's batch experiments: each its initial concentrations by species "
        "(the model's defaults for the others) and, for a fit, a CSV table with a t "
        "column and one column per observed species, copied into the study.",
        [],
    ),
    "observed": (
        "species",
        "Observed species",
        "The species measured in the reactor's experiments.",
        [],
    ),
    "parameters": (
        "parameter_values",
        "Parameters",
        "Plant parameters set to other values than the flowsheet's for this study (for "
        "a fit, its starting point).",
        {},
    ),
    "initial_state": (
        "initial_state",
        "Initial state",
        "Where the run starts: each unit's own default state, the steady state at the "
        "parameters, or a state saved in a file (a final_state.yaml of another study).",
        "default",
    ),
    "time": (
        "run_time",
        "Run",
        "Leave empty for the steady state; give a time span for a dynamic run, with the "
        "times to save.",
        None,
    ),
    "streams": (
        "streams",
        "Streams saved",
        "Streams saved whole: flow, temperature and every species.",
        [],
    ),
    "observables": (
        "observables",
        "Measured channels",
        "What is measured, and on which stream: species, or a measurement such as COD, "
        "TKN, NOx, TN, TP, TSS, flow (Q) or temperature (T). A channel of your own is "
        "written as a weighted sum, 'BOD5 = 0.65*[SS] + 0.65*[XS]', with species in "
        "brackets and the measurements above named as they are.",
        [],
    ),
    "free": (
        "parameters",
        "Free parameters",
        "The parameters to estimate.",
        [],
    ),
    "t_obs": (
        "times",
        "Sampling times",
        "When the channels are measured.",
        None,
    ),
    "t_span": (
        "span",
        "Time span",
        "The span the plant is run over (by default, from 0 to the last sampling time).",
        None,
    ),
    "sigma": (
        "numbers",
        "Measurement error",
        "The standard deviation of each channel's measurements: one number for all, or "
        "one per channel.",
        None,
    ),
    "relative_sigma": (
        "boolean",
        "Relative error",
        "Read the measurement error as a fraction of each value.",
        False,
    ),
    "sigma_floor": (
        "number",
        "Smallest error",
        "With a relative error, the smallest standard deviation used.",
        0.0,
    ),
    "sensitivity": (
        "choice",
        "Sensitivity",
        "How the sensitivity of the channels is computed: forward (a variational solve) "
        "or reverse (through the solve the calibration differentiates).",
        "forward",
    ),
    "time_unit": (
        "text",
        "Time unit",
        "The unit of the times given here (by default, the plant's).",
        None,
    ),
    "data": (
        "file",
        "Observations",
        "A CSV table with a t column and one <stream>.<channel> column per measured "
        "channel. It is copied into the study.",
        None,
    ),
    "estimate_sigma": (
        "boolean",
        "Estimate the error",
        "Estimate each channel's measurement error from the fit instead of giving it.",
        False,
    ),
    "transforms": (
        "transforms",
        "Transforms",
        "Fit a parameter in log space (positive_log, for a positive one) or logit space "
        "(for a fraction between 0 and 1).",
        {},
    ),
    "use_priors": (
        "boolean",
        "Use the models' priors",
        "Add the literature priors the models declare on their parameters.",
        True,
    ),
    "n_starts": ("integer", "Starts", "Optimizer starts (the first from the start point).", 1),
    "max_iter": ("integer", "Iterations", "Most optimizer iterations per start.", 500),
    "laplace": (
        "boolean",
        "Uncertainty",
        "Estimate each fitted parameter's uncertainty (a Laplace approximation).",
        False,
    ),
    "profile": (
        "parameter",
        "Profiled parameter",
        "The parameter fixed at each grid value while the others are fitted.",
        None,
    ),
    "grid": ("times", "Grid", "The values of the profiled parameter.", None),
    "delta": (
        "number",
        "Threshold",
        "The likelihood-ratio threshold of the confidence interval (1.92 for 95 %).",
        1.92,
    ),
    "candidates": (
        "candidates",
        "Candidates",
        "The sensor sets to rank, each named, with its channels and measurement error. "
        "A channel of your own is written as a weighted sum, "
        "'BOD5 = 0.65*[SS] + 0.65*[XS]'.",
        [],
    ),
    "base": (
        "candidate",
        "Sensors in place",
        "Sensors already installed; each candidate is scored as an addition to them.",
        None,
    ),
    "criterion": (
        "choice",
        "Criterion",
        "D (overall information), A (average variance), E (worst direction) or "
        "modified_E (conditioning).",
        "D",
    ),
}

_CHOICES = {
    "sensitivity": ["forward", "reverse"],
    "criterion": ["D", "A", "E", "modified_E"],
    "transforms": ["none", "positive_log", "logit"],
    "initial_state": ["default", "steady"],
}


def study_form() -> dict:
    """Every setting of a study, and which kinds use which, as plain data.

    Returns
    -------
    dict
        ``{"kinds": [{"kind", "label", "summary", "fields", "required",
        "reactor_fields", "reactor_required"}], "fields": {name: {"type",
        "label", "help", "default", "choices"}}}``. ``fields`` of a kind are
        those of a study of the flowsheet, in the order a form shows them;
        ``reactor_fields`` those of a study of a batch reactor (empty for a
        kind that takes no reactor).
    """

    kinds = [
        {
            "kind": kind,
            "label": label,
            "summary": summary,
            "fields": list(fields),
            "required": list(_REQUIRED.get(kind, ())),
            "reactor_fields": list(_REACTOR_FIELDS.get(kind, ())),
            "reactor_required": list(REACTOR_REQUIRED.get(kind, ())),
        }
        for kind, (label, summary, fields) in _KINDS.items()
    ]
    fields = {
        name: {
            "type": kind,
            "label": label,
            "help": help_text,
            "default": default,
            "choices": _CHOICES.get(name, []),
        }
        for name, (kind, label, help_text, default) in _FIELDS.items()
    }
    return {"kinds": kinds, "fields": fields}


def plant_choices(spec, *, base_dir=None) -> dict:
    """What a study of the flowsheet ``spec`` may name.

    Parameters
    ----------
    spec : dict or str or pathlib.Path
        The flowsheet, or its file.
    base_dir : str or pathlib.Path, optional
        Where a mapping's relative paths point from.

    Returns
    -------
    dict
        ``{"streams": [...], "parameters": [...], "channels": {stream: [...]}}``:
        the named streams then every ``unit.port`` output, the plant's parameter
        names, and per stream the measurements its model gives then its species
        (temperature only when an influent carries one).
    """
    from aquakin.plant import Plant

    plant = Plant.from_spec(spec, base_dir=base_dir)
    streams = [*plant.list_streams(), *plant.list_ports()]
    channels = {s: [*_measures(plant, s), *_model_of(plant, s).species] for s in streams}
    return {"streams": streams, "parameters": plant.parameter_names(), "channels": channels}


def _model_of(plant, stream: str):
    unit = plant.named_streams.get(stream, stream).split(".", 1)[0]
    return plant.units[unit].model


def _measures(plant, stream: str) -> list[str]:
    """The measurement keywords the model of ``stream`` gives (temperature only
    when an influent carries one)."""
    from aquakin.plant.measures import KEYWORDS, measure_map

    warm = any(series.T is not None for series in plant.influents.values())
    model = _model_of(plant, stream)
    given = []
    for keyword in KEYWORDS:
        if keyword == "T" and not warm:
            continue
        try:
            measure_map(model, (keyword,))
        except (KeyError, ValueError):
            continue
        given.append(keyword)
    return given


def simulation_study(spec, *, base_dir=None) -> dict:
    """The settings of a plain simulation of the flowsheet ``spec``.

    The plant at steady state, saving every stream that leaves it (an output
    port nothing is connected to, by its stream name when it has one) and the
    measurements its model gives on each (flow, COD, TKN, TN, TSS, ...).

    Parameters
    ----------
    spec : dict or str or pathlib.Path
        The flowsheet, or its file.
    base_dir : str or pathlib.Path, optional
        Where a mapping's relative paths point from.

    Returns
    -------
    dict
        Study settings for :func:`create_study`.

    Raises
    ------
    StudyError
        If no stream leaves the plant and none is named.
    """
    from aquakin.plant import Plant

    plant = Plant.from_spec(spec, base_dir=base_dir)
    fed = {f"{c.from_unit}.{c.from_port}" for c in plant.connections if c.from_unit is not None}
    named = {endpoint: name for name, endpoint in plant.list_streams().items()}
    streams = [
        named.get(port, port)
        for port in plant.list_ports()
        if port not in fed and hasattr(plant.units[port.split(".", 1)[0]], "model")
    ]
    if not streams:
        streams = list(plant.list_streams())
    if not streams:
        raise StudyError("no stream leaves the plant: connect an outlet or name a stream")
    # Every model gives at least the flow, so each stream has a measurement.
    return {
        "kind": "simulate",
        "description": "The plant at steady state: the streams that leave it.",
        "streams": streams,
        "observables": [{"stream": s, "channels": _measures(plant, s)} for s in streams],
    }


def _finite(value):
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    value = float(value)
    return value if math.isfinite(value) else None


def _by_stream(columns: dict, streams: list[str]) -> dict:
    """``{"<stream>.<channel>": values}`` as ``{stream: {channel: values}}`` (a
    stream name may hold a dot, so the longest stream that prefixes a column wins)."""
    out: dict = {}
    longest = sorted(streams, key=len, reverse=True)
    for column, values in columns.items():
        stream = next((s for s in longest if column.startswith(s + ".")), None)
        if stream is None:
            stream, _, channel = column.rpartition(".")
        else:
            channel = column[len(stream) + 1 :]
        out.setdefault(stream, {})[channel] = values
    return out


def _units(folder: Path, study, streams: list[str]) -> tuple[dict, str | None]:
    """Each stream's units by channel, and the plant's clock; empty when the
    plant no longer builds (the results are shown without units then)."""
    from aquakin.plant import Plant

    try:
        plant = Plant.from_spec(_flowsheet(folder, study))
        clock = plant.time_unit
        units = {}
        for stream in streams:
            model = _model_of(plant, stream)
            species = {sp: model.units_of(sp) for sp in model.species}
            units[stream] = {
                "Q": f"m³/{clock}" if clock else "",
                "T": "°C",
                **_measure_units(species.values()),
                **species,
            }
        return units, clock
    except Exception:  # units are a courtesy: shown without them when the plant no longer builds
        return {}, None


def _measure_units(species_units) -> dict:
    """The units of the measurement keywords, read off the species' own: COD in
    the unit of the COD species, the nitrogen and phosphorus measures in those of
    the N and P species, TSS in the COD species' mass per volume."""
    units = list(species_units)

    def first(tag: str) -> str:
        return next((u for u in units if tag in u), "")

    cod, n, p = first("COD"), first("_N/"), first("_P/")
    out = {"COD": cod, "TKN": n, "NOx": n, "TN": n, "TP": p}
    if cod:
        out["TSS"] = cod.replace("_COD", "")
    return {k: v for k, v in out.items() if v}


def study_results(path) -> dict:
    """A study's results as plain data, for a front end to show.

    Parameters
    ----------
    path : str or pathlib.Path
        The study folder.

    Returns
    -------
    dict
        ``{"kind", "files"}`` (the result file names), the study's standing as
        :func:`~aquakin.study.check_study` gives it (``"state"``,
        ``"has_results"``, ``"reasons"``, ``"notes"``), and what its kind's
        reader finds (``_READERS``). A ``simulate`` study gives ``"steady"`` and
        per stream its values by channel: ``"streams"`` (flow, species,
        temperature) and ``"measured"`` (the observables' channels), each
        ``{stream: {channel: value}}`` at a steady state or
        ``{stream: {channel: [values]}}`` with ``"t"`` and ``"time_unit"`` for a
        run in time; ``"units"`` gives ``{stream: {channel: unit}}`` where known.
        Temperature is in °C (the files keep the plant's kelvin). Fisher, rank,
        calibrate and profile studies give their summaries, tables and matrices
        (see :func:`_read_fisher` and its neighbours). A value that is not finite
        is ``None``.

    Raises
    ------
    StudyError
        If the study has no results.
    """
    folder = _folder(path)
    study = load_study(folder)
    out = folder / RESULTS
    if not out.is_dir():
        raise StudyError(f"study {folder.name} has no results; run it first")
    status = check_study(folder)
    result: dict = {
        "kind": study.kind,
        "files": sorted(p.name for p in out.iterdir()),
        "state": status.state,
        "has_results": status.has_results,
        "reasons": status.reasons,
        "notes": status.notes,
    }
    result.update(_reader(study.kind)(folder, study, out))
    return result


def _reader(kind: str):
    """The function that reads a ``kind`` study's results (``_READERS``)."""
    try:
        return _READERS[kind]
    except KeyError:
        raise StudyError(f"no result reader for a {kind!r} study") from None


def _yaml_file(path: Path):
    return _finite_tree(yaml.safe_load(path.read_text())) if path.is_file() else None


def _read_simulate(folder: Path, study, out: Path) -> dict:
    """A simulation's streams and measured channels, by stream, with their units."""
    streams = [*study.streams, *(o.stream for o in study.observables)]
    units, clock = _units(folder, study, list(dict.fromkeys(streams)))
    found: dict = {"units": units, "steady": study.time is None}
    files = (
        {"streams": STREAMS_STEADY_FILE, "measured": MEASURED_STEADY_FILE}
        if study.time is None
        else {"streams": STREAMS_FILE, "measured": MEASURED_FILE}
    )
    for key, name in files.items():
        file = out / name
        if not file.is_file():
            continue
        if study.time is None:
            found[key] = _finite(read_named_values(file))
        else:
            t, columns = read_trajectory(file)
            found["t"] = _finite(list(t))
            found[key] = _by_stream(
                {name: _finite(list(v)) for name, v in columns.items()}, streams
            )
    for key in ("streams", "measured"):
        for channels in (found.get(key) or {}).values():
            if "T" in channels:
                channels["T"] = _celsius(channels["T"])
    if study.time is not None:
        found["time_unit"] = study.time.time_unit or clock
    return found


def _read_fisher(folder: Path, study, out: Path) -> dict:
    """The Fisher summary (values, standard deviations, criteria) and its matrix."""
    found: dict = {"summary": _yaml_file(out / FISHER_SUMMARY_FILE)}
    if (out / FISHER_FILE).is_file():
        matrix, rows, _ = read_matrix(out / FISHER_FILE)
        found["matrix"] = {"names": rows, "rows": _finite(matrix.tolist())}
    return found


def _read_rank(folder: Path, study, out: Path) -> dict:
    """The ranking table and the criterion it was ranked by."""
    found: dict = {"criterion": study.criterion}
    if (out / RANKING_FILE).is_file():
        found["table"] = _finite_tree(read_table(out / RANKING_FILE))
    return found


def _read_calibrate(folder: Path, study, out: Path) -> dict:
    """The fit's summary, and the fitted and the observed channels by stream (by
    experiment, for a reactor)."""
    found: dict = {"summary": _yaml_file(out / FIT_FILE)}
    if study.reactor is not None:
        streams = experiment_names(study)
        observed = observed_data(folder, study)
    else:
        streams = [o.stream for o in study.observables]
        observed = _plant_data(folder, study)
    if (out / FITTED_FILE).is_file():
        t, columns = read_trajectory(out / FITTED_FILE)
        found["fitted"] = {
            "t": _finite(list(t)),
            "values": _by_stream({n: _finite(list(v)) for n, v in columns.items()}, streams),
        }
    if observed is not None:
        t, columns = observed
        found["data"] = {
            "t": _finite(list(t)),
            "values": _by_stream({n: _finite(list(v)) for n, v in columns.items()}, streams),
        }
    return found


def _plant_data(folder: Path, study):
    """The measured channels of a flowsheet study's data table, or ``None``."""
    data = folder / (study.data or "")
    if not (study.data and data.is_file()):
        return None
    names = [f"{o.stream}.{n}" for o in study.observables for n in _channel_names(_channels(o))]
    t, columns = read_trajectory(data)
    return t, {n: v for n, v in columns.items() if n in names}


def _read_profile(folder: Path, study, out: Path) -> dict:
    """The profile's summary (best value, interval) and its table."""
    found: dict = {"summary": _yaml_file(out / PROFILE_SUMMARY_FILE)}
    if (out / PROFILE_FILE).is_file():
        found["table"] = _finite_tree(read_table(out / PROFILE_FILE))
    return found


#: Each kind's result reader, ``(folder, study, results folder) -> dict``; one per
#: entry of :data:`~aquakin.study.study.KINDS`, like ``_RUNNERS``.
_READERS = {
    "simulate": _read_simulate,
    "fisher": _read_fisher,
    "calibrate": _read_calibrate,
    "profile": _read_profile,
    "rank": _read_rank,
}


def _finite_tree(value):
    """A YAML or table value with every number finite or ``None`` (text kept)."""
    if isinstance(value, dict):
        return {k: _finite_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_tree(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _celsius(kelvin):
    if isinstance(kelvin, list):
        return [_celsius(v) for v in kelvin]
    return None if kelvin is None else kelvin - 273.15


def _bring_in(folder: Path, study: dict, key: str, project: Path) -> None:
    """Copy the file ``study[key]`` names into the study folder, keeping its name."""
    value = study.get(key)
    if not value or (key == "initial_state" and value in ("default", "steady")):
        return
    study[key] = _bring_in_file(folder, value, key, project)


def _bring_in_file(folder: Path, value: str, key: str, project: Path) -> str:
    """Copy the file ``value`` into the study folder, keeping its name, and
    return that name. A relative path is read from the study folder, else the
    project."""
    source = Path(value)
    if not source.is_absolute():
        inside = folder / source
        source = inside if inside.is_file() else project / source
    if not source.is_file():
        raise StudyError(f"{key}: {value} does not exist")
    dest = folder / source.name
    if source.resolve() != dest.resolve():
        if dest.exists() and dest.read_bytes() != source.read_bytes():
            raise StudyError(f"{key}: the study already holds another {source.name}")
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    return source.name


def _reactor_files(folder: Path, study: dict, project: Path) -> None:
    """Copy each experiment's data table into the study, and store a model file
    relative to the study folder (an absolute one, or one relative to the
    project), as a run reads it."""
    from aquakin.plant.spec import _is_file_base

    experiments = []
    for i, experiment in enumerate(study.get("experiments") or []):
        experiment = dict(experiment)
        if experiment.get("data"):
            experiment["data"] = _bring_in_file(
                folder, experiment["data"], f"experiments[{i}].data", project
            )
        experiments.append(experiment)
    if experiments:
        study["experiments"] = experiments
    reactor = study.get("reactor")
    if not reactor or not _is_file_base(reactor.get("model", "")):
        return
    model = Path(reactor["model"])
    if not model.is_absolute():
        if (folder / model).is_file():
            return
        model = project / model
    if not model.is_file():
        raise StudyError(f"reactor.model: {reactor['model']} does not exist")
    try:
        relative = Path(os.path.relpath(model.resolve(), folder.resolve())).as_posix()
    except ValueError:  # another drive (Windows): keep the absolute path
        relative = str(model)
    study["reactor"] = {**reactor, "model": relative}


def _relative_flowsheet(folder: Path, study: dict) -> None:
    """Store an absolute ``flowsheet`` relative to the study folder, and leave it
    out when it is the default, so a front end can name the spec file it has
    open without knowing where studies sit in a project."""
    value = study.get("flowsheet")
    if not value or not Path(value).is_absolute():
        return
    try:
        relative = Path(os.path.relpath(Path(value).resolve(), folder.resolve())).as_posix()
    except ValueError:  # another drive (Windows): keep the absolute path
        return
    if relative == _Study.model_fields["flowsheet"].default:
        del study["flowsheet"]
    else:
        study["flowsheet"] = relative


def _write(folder: Path, study: dict, project: Path) -> Path:
    from pydantic import ValidationError

    study = dict(study)
    try:
        checked = _Study.model_validate(study)
    except ValidationError as exc:
        lines = [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors(include_url=False)
        ]
        raise StudyError("the study is not valid:\n  " + "\n  ".join(lines)) from None
    _check_kind(checked, "the study")
    for key in ("data", "initial_state"):
        _bring_in(folder, study, key, project)
    _reactor_files(folder, study, project)
    _relative_flowsheet(folder, study)
    folder.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump({"study": study}, sort_keys=False, allow_unicode=True)
    (folder / STUDY_FILE).write_text(text)
    return folder


def create_study(project, name: str, study: dict) -> Path:
    """Write a new study ``project/studies/<name>/`` from plain settings.

    The settings are checked exactly as :func:`~aquakin.study.run_study` reads
    them; a ``data`` table or an ``initial_state`` file outside the study folder
    is copied into it, so the study holds its own inputs.

    Parameters
    ----------
    project : str or pathlib.Path
        The project folder (the one holding ``flowsheet.yaml``).
    name : str
        The study's folder name (letters, digits, ``_`` and ``-``).
    study : dict
        The ``study:`` settings (see :func:`study_form`). A relative file path
        is read from the project folder. An absolute ``flowsheet`` is stored
        relative to the study folder (and left out when it is the project's
        ``flowsheet.yaml``).

    Returns
    -------
    pathlib.Path
        The study folder.

    Raises
    ------
    StudyError
        If the name is taken or not a valid name, the settings are not a study,
        or a file it names does not exist.
    """
    from aquakin.plant.errors import NAME_PATTERN

    project = Path(project)
    if not NAME_PATTERN.match(name):
        raise StudyError(
            f"study name {name!r} must start with a letter or '_' and hold only letters, "
            f"digits, '_' and '-'"
        )
    folder = project / STUDIES_DIR / name
    if folder.exists():
        raise StudyError(f"a study named {name} already exists")
    try:
        return _write(folder, study, project)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def update_study(path, study: dict) -> Path:
    """Replace a study's settings (its ``study.yaml``); results already there stay.

    A changed setting makes the results stale (see
    :func:`~aquakin.study.check_study`), so the study shows that it needs to
    run again.

    Parameters
    ----------
    path : str or pathlib.Path
        The study folder.
    study : dict
        The new ``study:`` settings; an absolute ``flowsheet`` is stored as for
        :func:`create_study`.

    Returns
    -------
    pathlib.Path
        The study folder.
    """
    folder = _folder(path)
    load_study(folder)  # it is a study
    return _write(folder, study, folder.parent.parent)


def read_study(path) -> dict:
    """A study's settings as plain data, defaults left out (for a form to edit).

    Parameters
    ----------
    path : str or pathlib.Path
        The study folder.

    Returns
    -------
    dict
    """
    study = load_study(path)
    return {"kind": study.kind, **study.model_dump(mode="json", exclude_defaults=True)}
