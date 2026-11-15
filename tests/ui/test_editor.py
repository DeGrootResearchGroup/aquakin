"""The editor page, built and driven without a browser."""

import pytest
from trame.app import get_server

from aquakin_ui.app import Editor

from .conftest import PROJECTS


@pytest.fixture(scope="module")
def editor():
    server = get_server(name="aquakin-ui-test", client_type="vue3")
    ed = Editor(server, path=PROJECTS / "bsm1" / "flowsheet.yaml")
    yield ed
    ed.worker.stop()


def test_the_page_draws_the_flowsheet(editor):
    s = editor.state
    assert len(s.nodes) == 10 and len(s.edges) == 11
    assert {n["id"] for n in s.nodes} >= {"influent", "tank1", "clarifier"}
    tank3 = next(n for n in s.nodes if n["id"] == "tank3")
    assert tank3["data"]["label"] == "Aerobic zone 1" and tank3["data"]["icon"] == "cstr-aerated"
    assert sorted(s.products) == ["clarifier.overflow", "underflow_split.waste"]


def test_checks_come_from_aquakin(editor):
    for uid in editor.doc.units:
        assert editor._check_unit(uid) is None
    assert editor._check_plant()["ok"]
    editor.publish(checks=False)
    assert editor.state.errors == [] and editor.state.n_recycles == 2


def test_a_dropped_unit_is_built_by_aquakin_and_its_error_shown(editor):
    editor.on_dropped({"kind": "cstr", "position": {"x": 0, "y": 600}})
    uid = editor.state.selected
    assert editor._check_unit(uid) == "ValueError: 'volume' is required"
    editor.publish(checks=False)
    assert f"{uid}: ValueError: 'volume' is required" in editor.state.errors
    editor.on_remove({"nodes": [uid], "edges": []})
    assert uid not in editor.doc.units


def test_removing_a_connected_port_asks_first(editor):
    from aquakin_ui import forms

    editor.on_select({"nodes": ["inlet_mix"], "edges": []})
    rows = forms.rows_for(editor.doc, "inlet_mix", editor.model_info)
    for r in rows:
        if r["key"] == "input_port_names":
            r["value"] = "fresh, ras"
    before = list(editor.doc.connections)
    editor.apply_properties(rows)
    assert editor.state.confirm_open
    assert editor.state.confirm_lines == [
        "tank5_split.internal_recycle -> inlet_mix.internal_recycle"
    ]
    assert editor.doc.connections == before  # nothing changed yet
    editor.cancel_apply()
    assert not editor.state.confirm_open and editor.doc.connections == before
    editor.apply_properties(rows)
    editor.confirm_apply()
    assert not editor.state.confirm_open and len(editor.doc.connections) == len(before) - 1
    assert editor.doc.ports("inlet_mix")[0] == ["fresh", "ras"]
