"""Fixtures for the browser interface's tests."""

import threading
from pathlib import Path

import pytest

from aquakin.plant.catalog import catalog, shipped_models
from aquakin_ui import studies
from aquakin_ui.catalog import Catalog
from aquakin_ui.worker import AquakinWorker

from .inprocess import InProcessRun

PROJECTS = Path(__file__).parents[2] / "examples" / "projects"


@pytest.fixture(scope="session")
def real_worker():
    """One real ``python -m aquakin serve`` worker, shared by the tests whose
    subject is the worker process (each start imports JAX)."""
    w = AquakinWorker()
    yield w
    w.stop()


class InProcessRuns:
    """The runs an editor started, and a gate that holds them back.

    Attributes
    ----------
    started : list of InProcessRun
        Every run, in the order started.
    gate : threading.Event
        Set (open) unless a test calls :meth:`hold`: a run starts its solve
        only once the gate is open, so a test can look at a run while it is
        certainly still running (a run in this process can end within
        milliseconds).
    """

    def __init__(self):
        self.started: list[InProcessRun] = []
        self.gate = threading.Event()
        self.gate.set()

    def start(self, *args, **kwargs) -> InProcessRun:
        self.started.append(InProcessRun(*args, gate=self.gate, **kwargs))
        return self.started[-1]

    def hold(self) -> None:
        """Hold back the runs started from now on until :meth:`release`."""
        self.gate.clear()

    def release(self) -> None:
        self.gate.set()


@pytest.fixture
def in_process_runs(monkeypatch):
    """The editor's study runs made :class:`~tests.ui.inprocess.InProcessRun`
    instances; every run started is cancelled and waited for at the end, so
    none outlives its test."""
    runs = InProcessRuns()
    monkeypatch.setattr(studies, "StudyRun", runs.start)
    yield runs
    runs.release()
    for run in runs.started:
        run.cancel()
        assert run.wait(300) is not None, f"the run of {run.folder} did not end"


@pytest.fixture(scope="session")
def ui_catalog():
    """The catalogue as the interface holds it (built in-process, as the worker would send it)."""
    return Catalog(catalog(), shipped_models())


@pytest.fixture(autouse=True)
def _own_preferences(tmp_path, monkeypatch):
    """Each test keeps its preferences in a file of its own, never the user's."""
    monkeypatch.setenv("AQUAKIN_UI_CONFIG", str(tmp_path / "preferences" / "ui.json"))


def study_project(tmp_path):
    """A one-tank project with one simulate study, written with aquakin."""
    import aquakin
    from aquakin.plant import CSTRUnit, InfluentSeries, Plant

    m = aquakin.load_model("asm1")
    p = Plant("tank")
    p.add_unit(CSTRUnit("tank", m, 1000.0, ["in"], conditions={"T": 293.15}))
    p.add_influent("feed", InfluentSeries.constant(m, Q=2000.0, SS=60.0, SNH=30.0), to="tank.in")
    project = tmp_path / "project"
    p.to_spec(project / "flowsheet.yaml")
    study = project / "studies" / "short"
    study.mkdir(parents=True)
    (study / "study.yaml").write_text(
        "study:\n  kind: simulate\n  description: half a day\n"
        "  time: {t_span: [0, 0.5], t_eval: {start: 0, stop: 0.5, num: 3}}\n"
        "  streams: [tank]\n"
    )
    return project
