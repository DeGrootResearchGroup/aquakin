"""Tools that change a project: write a flowsheet, and create, edit and run studies.

Each calls the function the editor's ``aquakin serve`` worker calls
(:func:`~aquakin.plant.spec.write_spec`, :mod:`aquakin.study`), so a flowsheet
or study the assistant writes is the file the editor would have written, and
the editor shows it. The refusals of those functions stand: a study with
results is not run again without ``replace``, and a frozen study is not
unfrozen without ``discard_results``. Writing a flowsheet over an existing file
also needs ``replace``.
"""

from __future__ import annotations

from pathlib import Path

from aquakin.mcp.runs import StudyRuns

#: The runs this server has started (one server per process).
RUNS = StudyRuns()


def write_flowsheet(
    path: str, spec: dict, base_dir: str | None = None, replace: bool = False
) -> dict:
    """Write a plant flowsheet file, checking every unit's settings first.

    The file the editor saves: settings filled in from the catalogue
    defaults, models and data tables from outside the folder copied in, the
    file replaced atomically. A spec that does not check out writes nothing.
    An editor with the file open shows the new version.

    Parameters
    ----------
    path : str
        The flowsheet file to write (``.yaml``, or ``.json``).
    spec : dict
        The flowsheet: units, connections and settings, as ``read_flowsheet``
        returns it.
    base_dir : str, optional
        The folder relative paths in ``spec`` are relative to (default: the
        folder of ``path``).
    replace : bool, optional
        Write over an existing file. Without it an existing file is refused.

    Returns
    -------
    dict
        The flowsheet as written.
    """
    from aquakin.plant.spec import write_spec

    if Path(path).exists() and not replace:
        raise FileExistsError(
            f"{path} already exists; read it with read_flowsheet, then write it again "
            f"with replace=true to overwrite it"
        )
    return write_spec(spec, path, base_dir=base_dir)


def list_studies(project: str) -> list[dict]:
    """List a project's studies with where each stands (fresh, stale, not run, ...).

    Parameters
    ----------
    project : str
        The project folder: the one holding the flowsheet and ``studies/``.

    Returns
    -------
    list of dict
        Per study: ``name``, ``path`` (its folder, what the other study tools
        take), ``kind``, ``description``, ``state``, ``has_results``,
        ``frozen``, ``flowsheet``, ``reasons`` and ``notes``.
    """
    from aquakin.study import list_studies as listed

    return listed(project)


def study_settings() -> dict:
    """Describe every kind of study (simulate, fisher, calibrate, profile, rank)
    and every setting a study can have: its type, default, help, which kinds
    use it and which are required.

    Returns
    -------
    dict
        ``{"kinds": [{"kind", "label", "summary", "fields", "required"}],
        "fields": [{"name", "type", "label", "help", "default", ...}]}``.
    """
    from aquakin.study import study_form

    return study_form()


def study_choices(flowsheet: str) -> dict:
    """List what a study of a flowsheet may name: its streams, its parameters
    (as ``"<model>.<parameter>"``) and each stream's measured channels.

    Parameters
    ----------
    flowsheet : str
        The flowsheet file.

    Returns
    -------
    dict
        ``{"streams": [...], "parameters": [...], "channels": {stream: [...]}}``.
    """
    from aquakin.study import plant_choices

    return plant_choices(flowsheet)


def read_study(path: str) -> dict:
    """Read a study's settings.

    Parameters
    ----------
    path : str
        The study folder (from ``list_studies``).

    Returns
    -------
    dict
        ``{"kind", ...settings}``, as ``create_study`` and ``update_study`` take
        it.
    """
    from aquakin.study import read_study as read

    return read(path)


def create_study(project: str, name: str, study: dict) -> str:
    """Create a study in a project, following the project's flowsheet.

    Parameters
    ----------
    project : str
        The project folder.
    name : str
        The study's name (its folder under ``studies/``); must be new.
    study : dict
        ``{"kind": ..., ...settings}``: the settings ``study_settings`` lists
        for that kind, naming streams, parameters and channels from
        ``study_choices``.

    Returns
    -------
    str
        The new study's folder.
    """
    from aquakin.study import create_study as create

    return str(create(project, name, study))


def update_study(path: str, study: dict) -> str:
    """Replace a study's settings. Its results stay, and show as stale until it
    is run again.

    Parameters
    ----------
    path : str
        The study folder.
    study : dict
        ``{"kind": ..., ...settings}``, the whole of the new settings.

    Returns
    -------
    str
        The study's folder.
    """
    from aquakin.study import update_study as update

    return str(update(path, study))


def run_study(path: str, replace: bool = False) -> dict:
    """Start running a study in the background and return at once; ask
    ``study_status`` how it is going. The editor shows the results when it ends.

    Parameters
    ----------
    path : str
        The study folder.
    replace : bool, optional
        Replace the results the study has. Without it a study with results is
        refused (copy_study keeps them).

    Returns
    -------
    dict
        The study's status (see ``study_status``).
    """
    from aquakin.study import check_study

    folder = Path(path)
    if check_study(folder).has_results and not replace:
        raise ValueError(
            f"study {folder.name} already has results; run it with replace=true to "
            f"replace them, or copy_study first to keep them"
        )
    RUNS.start(folder, replace=replace)
    return study_status(path)


def study_status(path: str) -> dict:
    """Say where a study stands and, if this server is running it, how far it has got.

    Parameters
    ----------
    path : str
        The study folder.

    Returns
    -------
    dict
        ``state``, ``has_results``, ``reasons`` and ``notes`` (as
        ``list_studies`` gives them), ``running``, and for a run started here
        ``progress`` (the latest ``stage``, ``done``, ``total``, ``fraction``,
        ``message``) and, once it has ended, ``outcome`` (``finished``,
        ``cancelled``, or ``failed`` with the ``error``).
    """
    from aquakin.study import check_study

    status = check_study(path)
    out = {
        "state": status.state,
        "has_results": status.has_results,
        "reasons": status.reasons,
        "notes": status.notes,
        "running": False,
    }
    run = RUNS.get(path)
    if run is not None:
        out |= {"running": run.running, "progress": run.progress}
        if run.outcome is not None:
            out |= run.outcome
    return out


def cancel_study(path: str) -> dict:
    """Stop a study this server is running, at its next checkpoint. Results it
    had before the run are kept.

    Parameters
    ----------
    path : str
        The study folder.

    Returns
    -------
    dict
        The study's status (see ``study_status``).
    """
    run = RUNS.get(path)
    if run is None or not run.running:
        raise ValueError(f"study {Path(path).name} is not running here")
    run.cancel()
    return study_status(path)


def study_results(path: str) -> dict:
    """Read a study's results as data: per stream its values by channel for a
    simulation; the summaries, tables and matrices of an analysis.

    Parameters
    ----------
    path : str
        The study folder.

    Returns
    -------
    dict
        ``kind``, ``files``, the study's standing (``state``, ``has_results``,
        ``reasons``, ``notes``) and its kind's results.
    """
    from aquakin.study import study_results as results

    return results(path)


def copy_study(path: str, name: str) -> str:
    """Copy a study, results included, to a new study of the same project.

    Parameters
    ----------
    path : str
        The study folder.
    name : str
        The copy's name; must be new.

    Returns
    -------
    str
        The copy's folder.
    """
    from aquakin.study import copy_study as copy

    return str(copy(path, name))


def freeze_study(path: str) -> dict:
    """Freeze a study on a snapshot of its plant (flowsheet, models, data), so
    later edits of the flowsheet do not make it stale.

    Parameters
    ----------
    path : str
        The study folder.

    Returns
    -------
    dict
        The study's status (see ``study_status``).
    """
    from aquakin.study import freeze_study as freeze

    freeze(path)
    return study_status(path)


def unfreeze_study(path: str, discard_results: bool = False) -> dict:
    """Make a frozen study follow the project's flowsheet again. This deletes
    its results, which describe the snapshot, not the live plant; copy_study
    first to keep them.

    Parameters
    ----------
    path : str
        The study folder.
    discard_results : bool, optional
        Agree to delete the results; without it a study with results is refused.

    Returns
    -------
    dict
        The study's status (see ``study_status``).
    """
    from aquakin.study import unfreeze_study as unfreeze

    unfreeze(path, discard_results=discard_results)
    return study_status(path)


#: The tools here and what each may do to the user's files: ``"read"`` reads
#: only, ``"write"`` adds files or records, ``"overwrite"`` may replace or
#: delete what is there (given its ``replace`` / ``discard_results``).
TOOLS = (
    (write_flowsheet, "overwrite"),
    (list_studies, "read"),
    (study_settings, "read"),
    (study_choices, "read"),
    (read_study, "read"),
    (create_study, "write"),
    (update_study, "overwrite"),
    (run_study, "overwrite"),
    (study_status, "read"),
    (cancel_study, "write"),
    (study_results, "read"),
    (copy_study, "write"),
    (freeze_study, "write"),
    (unfreeze_study, "overwrite"),
)
