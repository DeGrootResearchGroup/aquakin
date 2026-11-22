"""``aquakin-ui``: its arguments, its failure messages, and a real start-up.

The argument tests replace the trame server and the editor with recorders, so
nothing is served and no browser opens. The start-up test serves the real page
on a port the system picks (``--port 0``) and reads it over HTTP.
"""

import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
import trame.app

import aquakin_ui
import aquakin_ui.app
from aquakin_ui.__main__ import main

from .conftest import PROJECTS


class _Recorder:
    """Stands in for the trame server and the editor, recording what ``main`` did."""

    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with
        recorder = self

        class Server:
            def start(self, **kwargs):
                recorder.calls.append(("start", kwargs))
                if recorder.fail_with is not None:
                    raise recorder.fail_with

        class Worker:
            def stop(self):
                recorder.calls.append(("worker.stop", None))

        class Editor:
            def __init__(self, server, path=None):
                recorder.calls.append(("Editor", path))
                self.worker = Worker()

            def stop_runs(self):
                recorder.calls.append(("stop_runs", None))

        self.server = Server()
        self.editor = Editor

    def get_server(self, client_type):
        self.calls.append(("get_server", client_type))
        return self.server


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(trame.app, "get_server", rec.get_server)
    monkeypatch.setattr(aquakin_ui.app, "Editor", rec.editor)
    return rec


def test_with_no_arguments_an_empty_editor_is_served_locally_on_a_free_port(recorder):
    assert main([]) == 0
    assert recorder.calls == [
        ("get_server", "vue3"),
        ("Editor", None),
        ("start", {"port": 0, "host": "127.0.0.1", "open_browser": True}),
        ("stop_runs", None),
        ("worker.stop", None),
    ]


def test_the_flowsheet_port_and_browser_options_reach_the_editor_and_server(recorder):
    assert main(["plant.yaml", "--port", "8123", "--no-browser"]) == 0
    assert ("Editor", "plant.yaml") in recorder.calls
    assert ("start", {"port": 8123, "host": "127.0.0.1", "open_browser": False}) in recorder.calls


def test_runs_and_the_worker_are_stopped_when_the_server_stops_with_an_error(recorder):
    """Ctrl-C (or a port already taken) ends the server with an exception: the
    runs are cancelled and the worker stopped before it propagates."""
    recorder.fail_with = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        main(["--no-browser"])
    assert [c for c, _ in recorder.calls[-2:]] == ["stop_runs", "worker.stop"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--port", "eighty"], "argument --port: invalid int value: 'eighty'"),
        (["a.yaml", "b.yaml"], "unrecognized arguments: b.yaml"),
        (["--browser"], "unrecognized arguments: --browser"),
    ],
)
def test_bad_arguments_are_refused_before_anything_starts(recorder, capsys, argv, message):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: aquakin-ui [-h] [--port PORT] [--no-browser] [flowsheet]")
    assert message in err
    assert recorder.calls == []


def test_help_names_every_option(recorder, capsys):
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    for text in ("a flowsheet spec file to open", "port (default: a free one)", "--no-browser"):
        assert text in out
    assert recorder.calls == []


def test_the_editor_starts_and_serves_its_page(tmp_path):
    """``python -m aquakin_ui flowsheet.yaml --no-browser --port 0`` serves the
    page and the canvas bundle on this machine, at the port it prints."""
    env = {**os.environ, "AQUAKIN_UI_CONFIG": str(tmp_path / "ui.json")}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aquakin_ui",
            str(PROJECTS / "bsm1" / "flowsheet.yaml"),
            "--no-browser",
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        seen = []
        for line in proc.stdout:
            seen.append(line)
            found = re.search(r"http://(localhost|127\.0\.0\.1):(\d+)/", line)
            if found:
                break
        assert found, "".join(seen)
        port = int(found.group(2))
        assert port != 0
        base = f"http://127.0.0.1:{port}"
        page = urllib.request.urlopen(base + "/", timeout=30).read().decode()
        assert 'data-app-name="trame"' in page
        # The canvas is served from the package as built (module.serve).
        serve = Path(aquakin_ui.__file__).parent / "widget" / "serve"
        for name in ("aquakin-flowsheet.umd.js", "aquakin-flowsheet.css"):
            served = urllib.request.urlopen(f"{base}/__aquakin_flowsheet/{name}", timeout=30)
            assert served.read() == (serve / name).read_bytes()
    finally:
        proc.terminate()
        proc.wait(30)
