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


def checks_finished(page):
    """Wait until aquakin's background checks have finished: the editor's
    ``checked`` state is set and ``checking`` is not (the state the Checks tab
    draws from), read from the page without changing what it shows.

    The checks share the one worker with every other request, so a request
    made while a round runs (Open's ``read_spec``, the study dialog's
    ``study_form``) is answered only after it, and a round on a fresh worker
    builds every unit cold, which takes many seconds on a loaded machine.
    The timeout is the time a whole round may take, like the ``page``
    fixture's wait for the first drawing."""
    page.wait_for_function(
        "() => window.trame.state.get('checked') && !window.trame.state.get('checking')",
        timeout=60_000,
    )


def checks_done(page):
    """Wait until aquakin's background checks have finished (after which no
    republish of the flowsheet is pending), and show that they found nothing:
    the Checks tab says so."""
    checks_finished(page)
    expect = pytest.importorskip("playwright.sync_api").expect
    page.get_by_role("tab", name=re.compile(r"^Checks")).click()
    expect(page.get_by_text("None: aquakin built every unit and the plant.")).to_be_visible()


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
    websocket passed through ``relay``, once the checks it starts with have
    finished (:func:`checks_finished`): a test's own requests then find the
    worker free, and its waits time only them."""
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
        checks_finished(pg)
        yield pg
        pg.close()
    finally:
        proc.terminate()
        proc.wait(30)
