"""The ``aquakin`` command line.

``aquakin catalog`` prints the unit catalogue (see :mod:`aquakin.plant.catalog`)
as JSON. ``aquakin mcp`` serves aquakin's tools to an AI assistant over the
Model Context Protocol (see :mod:`aquakin.mcp`). ``aquakin serve`` answers requests from a front end, one JSON object
per line on standard input, one reply per line on standard output::

    {"id": 1, "command": "catalog"}
    {"id": 1, "ok": true, "result": {...}}

Commands: ``ping``, ``catalog``, ``models`` (the shipped kinetic models),
``model`` (``name``, optional ``path`` of a model file: one model's species
and conditions), ``model_document`` (``name``, optional ``path``: its reactions
and parameters as text, for editing), ``check_expression`` (``expression``,
``model``, optional ``path``, ``kind`` ``rate`` or ``stoichiometry``,
``reaction``, ``parameters`` the edit adds: ``ok``, ``error`` and its
``position``, the names it ``references`` and the ``unknown`` ones),
``check_model_edits`` (``model``, optional ``path``, ``edits``: ``ok`` and the
``errors`` by field), ``save_model_edits`` (``spec``, ``path`` of the
flowsheet, ``model``, ``edits``, optional ``name`` for a model made from a
shipped one, ``sha256``, ``base_dir``, ``dry_run``; see
:mod:`aquakin.plant.model_edit`), ``check_unit`` (``kind``, ``name``, ``settings``, optional
``model_files`` and ``base_dir``: build a unit and report its real ports or
the constructor's error), ``check_spec`` (``spec``, optional ``base_dir`` for
model file paths: build the whole plant from a flowsheet spec and report
``Plant.check``), ``fingerprint`` (``spec``, optional ``base_dir``: the spec's
:func:`~aquakin.plant.spec.spec_fingerprint`), ``read_spec`` (``path``: a spec
file read as YAML 1.2, see :func:`~aquakin.plant.spec.load_spec_file`) and
``write_spec`` (``spec``, ``path``, optional ``base_dir``: write it with
:func:`~aquakin.plant.spec.write_spec`, returning what was written),
``list_studies`` (``project``: every study of a project folder with its
status), ``freeze_study`` / ``unfreeze_study`` (``path``, ``discard_results``)
``copy_study`` (``path``, ``name``), ``cancel_study`` (``path``: stop its run at
the next checkpoint, whichever program started it), ``study_form`` (every study setting, for a
form), ``plant_choices`` (``spec``, ``base_dir``: the streams, parameters and
channels a study of it may name), ``read_study`` (``path``), ``create_study``
(``project``, ``name``, ``study``), ``update_study`` (``path``, ``study``),
``simulation_study`` (``spec``, ``base_dir``: the settings of a plain simulation)
and ``study_results`` (``path``: the results as plain data). A study is run in a process of its
own, ``aquakin study run --json <path>``, which streams progress events.
A failed request is answered
``{"id": ..., "ok": false, "error": message, "type": class name}`` (the message
without the class name in front, see :func:`aquakin.plant.errors.error_reply`;
no ``type`` for a request that is not JSON or names no known command) and the
worker keeps serving. Each study listed carries its ``state`` (one of
:data:`aquakin.study.study.STATES`, or ``"unreadable"``), ``has_results``, its
folder (``path``) and the spec it builds its plant from (``flowsheet``: the
snapshot's for a frozen study; ``None`` for a study of a batch reactor, whose
``reactor`` is then its model), so a front end never rebuilds study paths or
decides from a state's name what it implies.
Standard output carries replies only; anything else goes to standard error.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

# Named here too: the worker's replies carry it, and its protocol test reads it here.
from aquakin.study.form import UNREADABLE as UNREADABLE


def _study_module():
    from aquakin import study

    return study


def _study_call(function: str, req: dict, **kwargs):
    """Call ``aquakin.study.<function>`` on the request's study folder."""
    from aquakin import study as st

    result = getattr(st, function)(req["path"], **kwargs)
    return (
        None if function == "unfreeze_study" else (result if function != "freeze_study" else True)
    )


def _model_edit():
    from aquakin.plant import model_edit

    return model_edit


def _handlers():
    from aquakin import __version__
    from aquakin.plant import catalog as cat
    from aquakin.plant.spec import check_spec, load_spec_file, spec_fingerprint, write_spec

    return {
        "ping": lambda req: {"version": __version__},
        "catalog": lambda req: cat.catalog(),
        "models": lambda req: cat.shipped_models(),
        "model": lambda req: cat.model_info(req["name"], req.get("path")),
        "model_document": lambda req: _model_edit().model_document(req["name"], req.get("path")),
        "check_expression": lambda req: _model_edit().check_expression(
            req["expression"],
            model=req["model"],
            path=req.get("path"),
            kind=req.get("kind", "rate"),
            reaction=req.get("reaction"),
            parameters=req.get("parameters", ()),
        ),
        "check_model_edits": lambda req: _model_edit().check_model_edits(
            req["model"], req.get("path"), req["edits"]
        ),
        "save_model_edits": lambda req: _model_edit().save_model_edits(
            req["spec"],
            req["path"],
            model=req["model"],
            edits=req["edits"],
            name=req.get("name"),
            sha256=req.get("sha256"),
            base_dir=req.get("base_dir"),
            dry_run=bool(req.get("dry_run")),
        ),
        "check_spec": lambda req: check_spec(req["spec"], base_dir=req.get("base_dir")),
        "read_spec": lambda req: load_spec_file(req["path"]),
        "list_studies": lambda req: _study_module().list_studies(req["project"]),
        "freeze_study": lambda req: _study_call("freeze_study", req),
        "unfreeze_study": lambda req: _study_call(
            "unfreeze_study", req, discard_results=bool(req.get("discard_results"))
        ),
        "copy_study": lambda req: str(_study_call("copy_study", req, name=req["name"])),
        "cancel_study": lambda req: _study_call("cancel_study", req),
        "study_form": lambda req: _study_module().study_form(),
        "plant_choices": lambda req: _study_module().plant_choices(
            req["spec"], base_dir=req.get("base_dir")
        ),
        "read_study": lambda req: _study_module().read_study(req["path"]),
        "simulation_study": lambda req: _study_module().simulation_study(
            req["spec"], base_dir=req.get("base_dir")
        ),
        "study_results": lambda req: _study_module().study_results(req["path"]),
        "create_study": lambda req: str(
            _study_module().create_study(req["project"], req["name"], req["study"])
        ),
        "update_study": lambda req: str(_study_module().update_study(req["path"], req["study"])),
        "write_spec": lambda req: write_spec(
            req["spec"], req["path"], base_dir=req.get("base_dir")
        ),
        "fingerprint": lambda req: spec_fingerprint(req["spec"], base_dir=req.get("base_dir")),
        "check_unit": lambda req: cat.check_unit(
            req["kind"],
            req.get("name", "unit"),
            req.get("settings", {}),
            req.get("model_files"),
            req.get("base_dir"),
        ),
    }


def _answer(handlers, line: str) -> dict:
    try:
        req = json.loads(line)
    except json.JSONDecodeError as exc:
        return {"id": None, "ok": False, "error": f"not JSON: {exc}"}
    rid = req.get("id") if isinstance(req, dict) else None
    command = req.get("command") if isinstance(req, dict) else None
    if command not in handlers:
        return {
            "id": rid,
            "ok": False,
            "error": f"unknown command {command!r}; known: {sorted(handlers)}",
        }
    from aquakin.plant.errors import error_reply

    try:
        # A unit or model may print while it is built; keep stdout for replies.
        with contextlib.redirect_stdout(sys.stderr):
            result = handlers[command](req)
        return {"id": rid, "ok": True, "result": result}
    except Exception as exc:
        return {"id": rid, "ok": False, **error_reply(exc)}


def serve(stdin=None, stdout=None) -> None:
    """Answer JSON-line requests until standard input closes.

    Parameters
    ----------
    stdin, stdout : file, optional
        The request and reply streams (default the process's own). With the
        process's own standard output, replies go to a private copy of it and
        file descriptor 1 is pointed at standard error, so nothing else written
        to standard output (by a compiled extension, say) can corrupt a reply.
    """
    stdin = stdin or sys.stdin
    if stdout is None:
        sys.stdout.flush()
        stdout = os.fdopen(os.dup(1), "w", buffering=1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
    handlers = _handlers()
    for line in stdin:
        if not line.strip():
            continue
        reply = _answer(handlers, line)
        try:
            text = json.dumps(reply, allow_nan=False)
        except (TypeError, ValueError) as exc:
            # A result with NaN/inf or a non-JSON value must not end the worker.
            text = json.dumps(
                {"id": reply.get("id"), "ok": False, "error": f"reply is not valid JSON: {exc}"}
            )
        stdout.write(text + "\n")
        stdout.flush()


def _study_job(path, replace: bool) -> int:
    """Run a study for a front end: JSON-line events out, ``cancel`` in.

    Events: ``{"event": "progress", "stage", "done", "total", "fraction",
    "message"}`` while it runs, then one of ``{"event": "done", "state",
    "has_results", "reasons", "notes"}``, ``{"event": "cancelled"}`` or
    ``{"event": "error", "error", "type"}`` (as a refused request's, see
    :func:`aquakin.plant.errors.error_reply`). Standard output carries events
    only (as for ``serve``).
    """
    import threading

    from aquakin import study as st
    from aquakin.plant.errors import error_reply
    from aquakin.progress import Cancelled

    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    cancel = threading.Event()

    def listen():
        for line in sys.stdin:
            if line.strip() == "cancel":
                cancel.set()
        # A closed input means the front end has gone: stop as well.
        cancel.set()

    threading.Thread(target=listen, daemon=True).start()

    def emit(event):
        out.write(json.dumps(event, allow_nan=False) + "\n")

    def progress(p):
        if cancel.is_set():
            raise Cancelled()
        emit(
            {
                "event": "progress",
                "stage": p.stage,
                "done": p.done,
                "total": p.total,
                "fraction": p.fraction,
                "message": p.message,
            }
        )

    try:
        status = st.run_study(path, replace=replace, progress=progress)
    except Cancelled:
        emit({"event": "cancelled"})
        return 1
    except Exception as exc:
        emit({"event": "error", **error_reply(exc)})
        return 2
    emit(
        {
            "event": "done",
            "state": status.state,
            "has_results": status.has_results,
            "reasons": status.reasons,
            "notes": status.notes,
        }
    )
    return 0


def _study_command(args) -> int:
    from aquakin import study as st

    if args.action == "run" and args.json:
        return _study_job(args.path, args.replace)
    try:
        if args.action == "run":
            status = st.run_study(args.path, replace=args.replace)
        elif args.action == "freeze":
            st.freeze_study(args.path)
            status = st.check_study(args.path)
        elif args.action == "unfreeze":
            st.unfreeze_study(args.path, discard_results=args.discard_results)
            status = st.check_study(args.path)
        elif args.action == "copy":
            print(st.copy_study(args.path, args.name))
            return 0
        elif args.action == "cancel":
            st.cancel_study(args.path)
            print("asked to stop at its next checkpoint")
            return 0
        else:
            status = st.check_study(args.path)
    except st.StudyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    frozen = ", frozen" if status.frozen else ""
    print(f"{status.state}{frozen}")
    if status.running is not None:
        progress = status.running["progress"] or {}
        print(f"  running since {status.running['started']}: {progress.get('message', '')}")
    for line in status.reasons:
        print(f"  - {line}")
    for line in status.notes:
        print(f"  note: {line}")
    return 0 if status.state in ("fresh", "not run") else 1


def main(argv=None) -> int:
    """Run the ``aquakin`` command line."""
    parser = argparse.ArgumentParser(prog="aquakin", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="answer JSON-line requests from a front end")
    sub.add_parser("catalog", help="print the unit catalogue as JSON")
    mcp = sub.add_parser(
        "mcp",
        help='serve aquakin\'s tools to an AI assistant over MCP (needs the "mcp" extra)',
    )
    mcp.add_argument(
        "--allow-python",
        action="store_true",
        help="also let the assistant run Python scripts it writes (each run is approved "
        "in the assistant's app)",
    )
    study = sub.add_parser("study", help="run, check, freeze, unfreeze or copy a study")
    action = study.add_subparsers(dest="action", required=True)
    run = action.add_parser("run", help="run a study and write its results")
    run.add_argument("path", help="the study folder")
    run.add_argument("--replace", action="store_true", help="replace results already there")
    run.add_argument(
        "--json",
        action="store_true",
        help="report progress as JSON lines on standard output; 'cancel' on standard "
        "input stops the run at its next checkpoint",
    )
    check = action.add_parser("check", help="say whether a study's results are up to date")
    check.add_argument("path", help="the study folder")
    freeze = action.add_parser("freeze", help="freeze a study on a snapshot of its plant")
    freeze.add_argument("path", help="the study folder")
    unfreeze = action.add_parser(
        "unfreeze", help="make a frozen study follow the flowsheet again (deletes its results)"
    )
    unfreeze.add_argument("path", help="the study folder")
    unfreeze.add_argument(
        "--discard-results", action="store_true", help="agree to delete the study's results"
    )
    cancel = action.add_parser(
        "cancel", help="stop a study's run at its next checkpoint, whatever started it"
    )
    cancel.add_argument("path", help="the study folder")
    copy = action.add_parser("copy", help="copy a study, results included")
    copy.add_argument("path", help="the study folder")
    copy.add_argument("name", help="the copy's name")
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve()
    elif args.command == "mcp":
        from aquakin.mcp.server import main as serve_mcp

        serve_mcp(allow_python=args.allow_python)
    elif args.command == "study":
        return _study_command(args)
    else:
        from aquakin.plant.catalog import catalog

        json.dump(catalog(), sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
