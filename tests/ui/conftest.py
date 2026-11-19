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
