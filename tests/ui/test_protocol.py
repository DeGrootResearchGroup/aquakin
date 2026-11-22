"""The contract between ``aquakin_ui`` and the ``aquakin serve`` worker.

The two sides share no imports (the independence rule), so what they must agree
on (command names, study states, each study kind's view, the shape of an error)
is pinned here, where a test may import both.
"""

import ast
import io
import json
from pathlib import Path

import pytest

import aquakin_ui
from aquakin.__main__ import UNREADABLE, _handlers, serve
from aquakin.study import study as study_module

UI = Path(aquakin_ui.__file__).parent

#: Calls whose first argument is a worker command name.
_SENDERS = {"request", "_worker_study"}


def _sent_commands(root: Path = UI) -> tuple[set[str], list[str]]:
    """``(command names, calls whose command is not a literal)`` across a package."""
    names: set[str] = set()
    opaque: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _SENDERS
                and node.args
            ):
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names.add(first.value)
            else:
                opaque.append(f"{path.name}:{node.lineno}")
    return names, opaque


def test_every_command_the_interface_sends_is_one_the_worker_answers():
    names, opaque = _sent_commands()
    # The scan found the interface's requests at all (it would pass vacuously otherwise).
    assert {"catalog", "list_studies", "study_results", "freeze_study"} <= names
    assert sorted(names - set(_handlers())) == []
    # Only the one pass-through (the worker's study commands) names its command
    # by variable; its callers are scanned as `_worker_study("...")`.
    assert [where.split(":")[0] for where in opaque] == ["app.py"]


def test_the_scan_sees_a_command_the_worker_lacks(tmp_path):
    """The command check can fail: a request for an unknown command is found."""
    package = tmp_path / "aquakin_ui"
    package.mkdir()
    (package / "x.py").write_text("worker.request('frobnicate', path='p')\n")
    names, _ = _sent_commands(package)
    assert names - set(_handlers()) == {"frobnicate"}


def test_the_interface_colours_every_state_a_study_can_be_in():
    from aquakin_ui import study_page as app

    assert set(app.STATE_COLOURS) == {*study_module.STATES, UNREADABLE}
    assert app.UNREADABLE == UNREADABLE


def test_the_state_table_goes_into_a_template_without_double_quotes():
    """The chip colour is a template expression inside a double-quoted HTML
    attribute: a double quote there ends the attribute and the page renders
    blank (found in the browser)."""
    from aquakin_ui.study_page import STATE_COLOURS, _js

    text = _js(STATE_COLOURS)
    assert '"' not in text and text.startswith("{'fresh': 'success'")
    assert _js(None) == "null"
    with pytest.raises(ValueError, match="holds a quote"):
        _js("it's")


def test_every_study_kind_has_a_view():
    from aquakin_ui.results import ANALYSIS_VIEWS, SIMULATE

    assert {SIMULATE, *ANALYSIS_VIEWS} == set(study_module.KINDS)
    assert SIMULATE not in ANALYSIS_VIEWS


def test_a_kind_without_a_view_is_refused_not_drawn_as_another():
    from aquakin_ui.results import analysis_view

    with pytest.raises(ValueError, match="no view for a 'survey' study"):
        analysis_view({"kind": "survey"})


def _serve(*requests) -> list[dict]:
    out = io.StringIO()
    serve(io.StringIO("".join(json.dumps(r) + "\n" for r in requests)), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_a_refusal_carries_its_message_and_type_apart(tmp_path):
    missing = tmp_path / "none"
    (reply,) = _serve({"id": 1, "command": "study_results", "path": str(missing)})
    assert reply == {
        "id": 1,
        "ok": False,
        "error": f"{missing / 'study.yaml'} does not exist",
        "type": "StudyError",
    }


def test_the_client_keeps_the_type_off_the_message(tmp_path):
    from aquakin_ui.worker import AquakinWorker, WorkerError

    missing = tmp_path / "none"
    w = AquakinWorker()
    try:
        with pytest.raises(WorkerError) as info:
            w.request("study_results", path=str(missing))
        assert str(info.value) == f"{missing / 'study.yaml'} does not exist"
        assert info.value.type == "StudyError"
        with pytest.raises(WorkerError, match="unknown command 'nope'") as info:
            w.request("nope")
        assert info.value.type is None
    finally:
        w.stop()
