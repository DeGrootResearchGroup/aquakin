"""The editor served on a free port and driven in Chromium with Playwright.

These tests run in the ``browser`` CI job, not in the required gate (which runs
``tests/unit``, ``tests/integration`` and ``tests/ui``); run them with
``pytest tests/browser``. They need the ``browser`` extra (Playwright) and its
Chromium (``python -m playwright install chromium``); without Playwright they
are skipped.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECTS = Path(__file__).parents[2] / "examples" / "projects"


@pytest.fixture(scope="session")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


class Relay:
    """The page's websocket, passed through here so a test can hold back what
    Python sends (``hold``) and deliver it later (``release``): what the page
    does while an answer is on its way."""

    def __init__(self):
        self._held = None

    def route(self, ws):
        server = ws.connect_to_server()

        def to_page(message):
            if self._held is None:
                ws.send(message)
            else:
                self._held.append((ws, message))

        server.on_message(to_page)
        ws.on_message(server.send)

    def hold(self):
        self._held = []

    def release(self):
        held, self._held = self._held or [], None
        for ws, message in held:
            ws.send(message)


@pytest.fixture
def relay():
    return Relay()


@pytest.fixture
def page(browser, tmp_path, relay):
    """The editor on a copy of the BSM1 project, drawn in a fresh page, its
    websocket passed through ``relay``."""
    project = tmp_path / "bsm1"
    shutil.copytree(PROJECTS / "bsm1", project)
    env = {**os.environ, "AQUAKIN_UI_CONFIG": str(tmp_path / "ui.json")}
    cmd = [sys.executable, "-m", "aquakin_ui", str(project / "flowsheet.yaml")]
    proc = subprocess.Popen(
        [*cmd, "--no-browser", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        seen = []
        for line in proc.stdout:
            seen.append(line)
            found = re.search(r"http://127\.0\.0\.1:(\d+)/", line)
            if found:
                break
        else:
            raise AssertionError("the editor did not start:\n" + "".join(seen))
        pg = browser.new_page(viewport={"width": 1400, "height": 900})
        pg.route_web_socket(re.compile(".*"), relay.route)
        pg.goto(f"http://127.0.0.1:{found.group(1)}/")
        pg.wait_for_selector(".vue-flow__node", timeout=60_000)
        yield pg
        pg.close()
    finally:
        proc.terminate()
        proc.wait(30)
