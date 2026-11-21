"""Fixtures for the browser interface's tests."""

from pathlib import Path

import pytest

from aquakin.plant.catalog import catalog, shipped_models
from aquakin_ui.catalog import Catalog

PROJECTS = Path(__file__).parents[2] / "examples" / "projects"


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
