"""The ``aquakin`` command line.

``aquakin catalog`` prints the unit catalogue (see :mod:`aquakin.plant.catalog`)
as JSON. ``aquakin serve`` answers requests from a front end, one JSON object
per line on standard input, one reply per line on standard output::

    {"id": 1, "command": "catalog"}
    {"id": 1, "ok": true, "result": {...}}

Commands: ``ping``, ``catalog``, ``models`` (the shipped kinetic models),
``model`` (``name``: one model's species and conditions) and ``check_unit``
(``kind``, ``name``, ``settings``: build a unit and report its real ports or
the constructor's error). A failed request is answered
``{"id": ..., "ok": false, "error": message}`` and the worker keeps serving.
Standard output carries replies only; anything else goes to standard error.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys


def _handlers():
    from aquakin import __version__
    from aquakin.plant import catalog as cat

    return {
        "ping": lambda req: {"version": __version__},
        "catalog": lambda req: cat.catalog(),
        "models": lambda req: cat.shipped_models(),
        "model": lambda req: cat.model_info(req["name"]),
        "check_unit": lambda req: cat.check_unit(
            req["kind"], req.get("name", "unit"), req.get("settings", {})
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
    try:
        # A unit or model may print while it is built; keep stdout for replies.
        with contextlib.redirect_stdout(sys.stderr):
            result = handlers[command](req)
        return {"id": rid, "ok": True, "result": result}
    except Exception as exc:
        return {"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def serve(stdin=None, stdout=None) -> None:
    """Answer JSON-line requests until standard input closes.

    Parameters
    ----------
    stdin, stdout : file, optional
        The request and reply streams (default the process's own).
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    handlers = _handlers()
    for line in stdin:
        if not line.strip():
            continue
        stdout.write(json.dumps(_answer(handlers, line), allow_nan=False) + "\n")
        stdout.flush()


def main(argv=None) -> int:
    """Run the ``aquakin`` command line."""
    parser = argparse.ArgumentParser(prog="aquakin", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="answer JSON-line requests from a front end")
    sub.add_parser("catalog", help="print the unit catalogue as JSON")
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve()
    else:
        from aquakin.plant.catalog import catalog

        json.dump(catalog(), sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
