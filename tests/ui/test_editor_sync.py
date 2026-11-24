"""The editor's sync rules (``.claude/rules/ui.md``) that Python owns, driven
through an ``Editor`` without a browser.

The rules that live in the canvas (diffing Python's updates, wire validation on
the handles, no echoes, one removal for a unit and its wires, Delete but not
Backspace) are in ``tests/browser``, which drives the real page.
"""

import asyncio
import itertools
import json
import threading

import pytest
import yaml
from trame.app import get_server

from aquakin_ui import forms
from aquakin_ui.app import Editor
from aquakin_ui.checks import CHECK_DELAY
from aquakin_ui.icons import SVG

from .conftest import PROJECTS
from .inprocess import InProcessWorker

BSM1 = PROJECTS / "bsm1" / "flowsheet.yaml"
_names = itertools.count()


@pytest.fixture
def make_editor():
    """``make_editor(path=None) -> (editor, worker)`` on a server of its own, with
    an :class:`InProcessWorker` that records every request."""

    def make(path=None):
        server = get_server(name=f"aquakin-ui-sync-{next(_names)}", client_type="vue3")
        worker = InProcessWorker()
        return Editor(server, worker=worker, path=path), worker

    return make


async def _until(condition, timeout=60.0):
    """Wait on the running loop until ``condition()`` holds."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not condition():
        if loop.time() > end:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


# -- aquakin's checks run in the background ---------------------------------


def test_checks_wait_for_quiet_then_run_once_off_the_loop(make_editor):
    """Two edits in quick succession give one round of checks, after
    ``CHECK_DELAY``; ``checking`` shows meanwhile and ``checked`` (the page's
    licence to say "None") is set only when the round finished."""
    ed, worker = make_editor(BSM1)
    n_units = len(ed.doc.units)
    threads = []
    ask = ed._ask_unit
    ed._ask_unit = lambda *a: (threads.append(threading.current_thread()), ask(*a))[1]

    async def go():
        assert not ed.state.checked and not ed.state.checking
        ed.schedule_checks()
        ed.schedule_checks()  # a second edit within the quiet spell
        assert ed.state.checking and not ed.state.checked
        await asyncio.sleep(CHECK_DELAY / 2)
        assert worker.count("check_unit") == 0  # still quiet
        await _until(lambda: not ed.state.checking)

    asyncio.run(go())
    assert ed.state.checked
    assert worker.count("check_unit") == n_units and worker.count("check_spec") == 1
    assert threading.main_thread() not in threads and len(threads) == n_units
    assert ed.state.errors == [] and ed.state.n_recycles == 2


def test_a_unit_error_is_published_and_skips_the_plant_check(make_editor):
    """A unit that does not build is shown on its node and in the list, and the
    plant is not checked until every unit builds."""
    ed, worker = make_editor(BSM1)

    async def go():
        ed.on_dropped({"kind": "cstr", "position": {"x": 0, "y": 600}})  # no volume
        await _until(lambda: ed.state.checked and not ed.state.checking)

    asyncio.run(go())
    uid = ed.state.selected
    assert ed.state.errors == [f"{uid}: 'volume' is required", f"{uid}.in is not fed"]
    node = next(n for n in ed.state.nodes if n["id"] == uid)
    assert node["data"]["error"] == "'volume' is required"
    assert worker.count("check_spec") == 0


def test_each_check_is_cached_by_the_units_spec(make_editor):
    """A unit is checked again only when its settings change, and the plant only
    when the spec does; the other units' answers stand."""
    ed, worker = make_editor(BSM1)
    for uid in ed.doc.units:
        ed._check_unit(uid)
    ed._check_plant()
    assert worker.count("check_unit") == len(ed.doc.units)
    for uid in ed.doc.units:
        assert ed._check_unit(uid) is None
    assert ed._check_plant()["ok"] and worker.count("check_spec") == 1
    assert worker.count("check_unit") == len(ed.doc.units)

    ed.state.selected = "tank1"
    ed.refresh_selection()
    ed.apply_properties(_with(ed.state.sel_rows, "volume", "1200"))
    for uid in ed.doc.units:
        ed._check_unit(uid)
    ed._check_plant()
    assert worker.count("check_unit") == len(ed.doc.units) + 1
    assert (
        worker.sent[-2][1]["name"] == "tank1" and worker.sent[-2][1]["settings"]["volume"] == 1200
    )
    assert worker.count("check_spec") == 2


def test_a_background_round_asks_only_for_what_changed(make_editor):
    """The background round uses the same caches: a second round with nothing
    edited sends the worker nothing, and after one unit's edit only that unit
    and the plant are asked again."""
    ed, worker = make_editor(BSM1)
    n_units = len(ed.doc.units)

    async def round_of_checks():
        ed.schedule_checks()
        await _until(lambda: not ed.state.checking)

    asyncio.run(round_of_checks())
    assert worker.count("check_unit") == n_units and worker.count("check_spec") == 1
    asyncio.run(round_of_checks())
    assert worker.count("check_unit") == n_units and worker.count("check_spec") == 1

    ed.state.selected = "tank1"
    ed.refresh_selection()
    ed.apply_properties(_with(ed.state.sel_rows, "volume", "1200"))
    asyncio.run(round_of_checks())
    assert worker.count("check_unit") == n_units + 1 and worker.count("check_spec") == 2
    asked = [f["name"] for c, f in worker.sent if c == "check_unit"]
    assert asked[-1] == "tank1"
    assert ed.state.errors == [] and ed.state.checked


def test_a_move_republishes_without_a_check(make_editor):
    ed, worker = make_editor(BSM1)

    async def go():
        ed.on_move([{"id": "tank1", "position": {"x": 11.0, "y": 22.0}}])
        assert not ed.state.checking and ed._check_task is None

    asyncio.run(go())
    tank1 = next(n for n in ed.state.nodes if n["id"] == "tank1")
    assert tank1["position"] == {"x": 11.0, "y": 22.0}
    assert worker.count("check_unit") == 0 and worker.count("check_spec") == 0


def test_a_move_of_a_unit_python_no_longer_has_is_ignored(make_editor):
    ed, _ = make_editor(BSM1)
    before = {uid: u.position for uid, u in ed.doc.units.items()}
    ed.on_move([{"id": "gone", "position": {"x": 1.0, "y": 2.0}}])
    assert {uid: u.position for uid, u in ed.doc.units.items()} == before
    assert "gone" not in ed.doc.units


# -- Open and Save run off the page's thread ---------------------------------


def test_save_runs_off_the_page_and_keeps_an_edit_made_meanwhile(make_editor, tmp_path):
    """``busy`` is set while the worker writes; an edit made meanwhile is kept
    (not replaced by what was written) and stays unsaved."""
    ed, worker = make_editor(BSM1)
    target = tmp_path / "copy.yaml"
    ed.state.spec_path = str(target)

    async def go():
        ed.save()
        assert ed.state.busy
        ed.on_move([{"id": "tank1", "position": {"x": 5.0, "y": 7.0}}])
        await _until(lambda: not ed.state.busy)

    asyncio.run(go())
    assert ed.state.message == f"saved {target}"
    assert ed.doc.units["tank1"].position == {"x": 5.0, "y": 7.0}
    written = yaml.safe_load(target.read_text())
    assert written["layout"]["tank1"] != [5.0, 7.0]  # written before the move
    assert ed._unsaved()


def test_a_save_the_worker_refuses_is_reported_and_frees_the_buttons(make_editor, tmp_path):
    ed, worker = make_editor(BSM1)
    worker.refuse["write_spec"] = "the disk is full"
    target = tmp_path / "copy.yaml"
    ed.state.spec_path = str(target)

    async def go():
        ed.save()
        await _until(lambda: not ed.state.busy)

    asyncio.run(go())
    assert ed.state.message == f"could not save {target}: the disk is full"
    assert not target.exists()


def test_a_flowsheet_that_cannot_be_read_leaves_the_editor_empty(make_editor, tmp_path):
    """``aquakin-ui missing.yaml`` starts with an empty flowsheet and says why."""
    missing = tmp_path / "missing.yaml"
    ed, _ = make_editor(missing)
    assert ed.doc.units == {} and ed.state.nodes == []
    assert ed.state.message.startswith(f"could not open {missing}: ")
    assert "No such file" in ed.state.message or "does not exist" in ed.state.message
    assert "Error:" not in ed.state.message  # no exception class in front


# -- what the page is sent -------------------------------------------------------


def test_the_spec_text_is_sent_only_while_the_spec_tab_is_open(make_editor):
    ed, _ = make_editor(BSM1)
    ed.state.ready()
    ed.on_move([{"id": "tank1", "position": {"x": 3.0, "y": 4.0}}])
    assert ed.state.spec_text == ""
    ed.state.panel = "spec"
    ed.state.flush()
    shown = yaml.safe_load(ed.state.spec_text)
    assert shown["layout"]["tank1"] == [3.0, 4.0]
    assert set(shown["units"]) == set(ed.doc.units)
    ed.on_move([{"id": "tank1", "position": {"x": 8.0, "y": 9.0}}])
    assert yaml.safe_load(ed.state.spec_text)["layout"]["tank1"] == [8.0, 9.0]


def test_opening_the_studies_tab_lists_the_studies(make_editor):
    ed, worker = make_editor(BSM1)
    ed.state.ready()
    before = worker.count("list_studies")
    ed.state.panel = "studies"
    ed.state.flush()
    assert worker.count("list_studies") == before + 1
    assert worker.sent[-1][1] == {"project": str(BSM1.parent)}


def test_icons_are_sent_once_and_nodes_name_them(make_editor):
    ed, _ = make_editor(BSM1)
    assert ed.state.icons == SVG
    assert all(n["data"]["icon"] in SVG for n in ed.state.nodes)
    assert "<svg" not in json.dumps(ed.state.nodes)


def test_a_node_summarizes_its_main_settings(make_editor):
    """The node's reminder line: required numbers, then the aeration (kLa or a
    DO set-point), at most three parts."""
    ed, _ = make_editor(BSM1)
    ed.on_dropped({"kind": "cstr", "position": {"x": 0, "y": 600}})
    uid = ed.state.selected
    ed.doc.apply(uid, {**ed.doc.units[uid].settings, "volume": 1500.0}, None)
    ed.publish(checks=False)
    assert _node(ed, uid)["data"]["summary"] == "volume 1500"
    aeration = {"do_setpoint": 2.0, "species": "SO"}
    ed.doc.apply(uid, {**ed.doc.units[uid].settings, "aeration": aeration}, None)
    ed.publish(checks=False)
    assert _node(ed, uid)["data"]["summary"] == "volume 1500 · DO 2"
    aeration = {"kla": 240.0, "species": "SO"}
    ed.doc.apply(uid, {**ed.doc.units[uid].settings, "aeration": aeration}, None)
    ed.publish(checks=False)
    assert _node(ed, uid)["data"]["summary"] == "volume 1500 · kLa 240"


def test_the_palette_marks_kinds_the_form_cannot_finish(make_editor):
    ed, _ = make_editor()
    marked = {k["kind"] for cat in ed.state.palette for k in cat["kinds"] if k["python"]}
    assert marked == {"sbr"}  # a dosing unit's reagent has a form of its own
    sbr = next(k for cat in ed.state.palette for k in cat["kinds"] if k["kind"] == "sbr")
    assert sbr["summary"].endswith("can only be set from Python.")


def test_a_new_flowsheet_runs_on_days(make_editor):
    ed, _ = make_editor()
    ed.on_dropped({"kind": "cstr", "position": {"x": 0, "y": 0}})
    assert ed.doc.to_spec()["options"] == {"time_unit": "d"}


# -- the unit form ---------------------------------------------------------------


def test_selecting_one_unit_opens_its_form_and_several_select_none(make_editor):
    ed, _ = make_editor(BSM1)
    ed.state.panel = "studies"
    ed.on_select({"nodes": ["tank1", "tank2"], "edges": []})
    assert ed.state.selected is None and ed.state.panel == "studies"
    assert ed.state.sel_rows == []
    ed.on_select({"nodes": ["tank1"], "edges": []})
    assert ed.state.selected == "tank1" and ed.state.panel == "properties"
    assert ed.state.sel_kind == "CSTR"
    assert _with(ed.state.sel_rows, "volume", None) is not None


def test_a_refused_selection_is_sent_back_to_the_canvas(make_editor):
    """When the form's edits cannot be read, Python keeps their unit selected
    and sends the canvas nodes marking it, each time a list that differs from
    the last one sent (trame sends nothing for an unchanged value)."""
    ed, _ = make_editor(BSM1)
    ed.on_select({"nodes": ["tank1"], "edges": []})
    rows = _with(ed.state.sel_rows, "volume", "lots")
    ed.on_select({"nodes": ["tank2"], "edges": []}, pending=rows)
    first = ed.state.nodes
    assert ed.state.selected == "tank1"
    assert [n["id"] for n in first if n["selected"]] == ["tank1"]
    ed.on_select({"nodes": ["tank3"], "edges": []}, pending=rows)
    second = ed.state.nodes
    assert [n["id"] for n in second if n["selected"]] == ["tank1"]
    assert second != first
    assert [n for n in second if n["id"] != "tank1"] == [n for n in first if n["id"] != "tank1"]
    ed.publish(checks=False)  # an ordinary redraw carries no revision
    assert all("revision" not in n for n in ed.state.nodes)


def test_selection_is_marked_by_python(make_editor):
    ed, _ = make_editor(BSM1)
    ed.on_select({"nodes": ["tank2"], "edges": []})
    ed.publish(checks=False)
    assert [n["id"] for n in ed.state.nodes if n["selected"]] == ["tank2"]


def test_a_reset_form_carries_a_new_revision_and_clears_dirty(make_editor):
    """``cancel_apply`` marks the first row with a new revision each time (so
    the page drops the declined edits), and only the first row."""
    ed, _ = make_editor(BSM1)
    ed.on_select({"nodes": ["tank1"], "edges": []})
    plain = forms.rows_for(ed.doc, "tank1", ed.model_info)
    ed.state.sel_dirty = True
    ed.cancel_apply()
    first = ed.state.sel_rows
    assert first[0] == {**plain[0], "revision": 1} and first[1:] == plain[1:]
    assert not ed.state.sel_dirty
    ed.cancel_apply()
    assert ed.state.sel_rows[0]["revision"] == 2


def test_a_reset_with_nothing_selected_sends_no_rows(make_editor):
    ed, _ = make_editor(BSM1)
    ed.cancel_apply()
    assert ed.state.sel_rows == [] and not ed.state.confirm_open


def test_confirming_with_nothing_pending_changes_nothing(make_editor):
    ed, _ = make_editor(BSM1)
    ed.on_select({"nodes": ["tank1"], "edges": []})
    before = ed.doc.snapshot()
    ed.confirm_apply()
    assert ed.doc.snapshot() == before and not ed.state.can_undo


def test_a_model_change_that_cannot_be_read_changes_nothing(make_editor):
    """``reshape_form`` with an unreadable field elsewhere in the form keeps the
    unit as it was and names the field."""
    ed, _ = make_editor(BSM1)
    ed.on_select({"nodes": ["tank1"], "edges": []})
    before = ed.doc.snapshot()
    rows = _with(ed.state.sel_rows, "volume", "lots", copy=True)
    i = next(i for i, r in enumerate(rows) if r["key"] == "model")
    ed.reshape_form(rows, i, "asm3")
    assert ed.doc.snapshot() == before
    assert set(ed.state.sel_errors) == {"volume"}


def test_removing_a_unit_keeps_another_selection(make_editor):
    ed, _ = make_editor(BSM1)
    ed.on_select({"nodes": ["tank1"], "edges": []})
    ed.on_remove({"nodes": ["tank5"], "edges": []})
    assert ed.state.selected == "tank1" and "tank5" not in ed.doc.units
    ed.on_remove({"nodes": ["tank1"], "edges": []})
    assert ed.state.selected is None


def test_a_model_the_worker_cannot_describe_is_asked_once(make_editor):
    """A model the worker refuses has no description (no condition or species
    rows); a dropped unit still gets the model, and the refusal is remembered."""
    ed, worker = make_editor()
    worker.refuse["model"] = "no such model"
    assert ed.model_info("asm1") is None
    assert ed.model_info("asm1") is None
    assert worker.count("model") == 1
    ed.on_dropped({"kind": "cstr", "position": {"x": 0, "y": 0}})
    assert ed.doc.units[ed.state.selected].settings == {"model": "asm1"}


def test_a_kind_without_a_model_is_dropped_with_no_settings(make_editor):
    """A dosing unit has no model setting: nothing is filled in, nothing asked."""
    ed, worker = make_editor()
    ed.on_dropped({"kind": "dosing", "position": {"x": 0, "y": 0}})
    assert ed.doc.units[ed.state.selected].settings == {}
    assert worker.count("model") == 0


def test_a_reagent_set_in_the_form_builds_and_survives_save_and_open(make_editor, tmp_path):
    """Choosing the reagent's model redraws the composition with its species;
    the reagent composed there builds, and Save then Open give it back."""
    from aquakin.plant.catalog import model_info

    ed, _ = make_editor()
    ed.on_dropped({"kind": "dosing", "position": {"x": 0, "y": 0}})
    uid = ed.state.selected
    assert ed._check_unit(uid) == "'reagent' is required"
    rows = ed.state.sel_rows
    ed.reshape_form(rows, next(i for i, r in enumerate(rows) if r["key"] == "reagent.model"), "asm3")
    composition = _with(ed.state.sel_rows, "reagent.composition", None)
    assert composition["items"] == [s["name"] for s in model_info("asm3")["species"]]
    entries = [{"species": "SS", "value": "250000"}]
    rows = _with(_with(ed.state.sel_rows, "reagent.composition", entries), "flow", "3")
    assert ed.apply_properties(rows)
    reagent = {"model": "asm3", "composition": {"SS": 250000.0}}
    assert ed.doc.units[uid].settings == {"flow": 3.0, "reagent": reagent}
    assert ed._check_unit(uid) is None

    ed.state.spec_path = str(tmp_path / "flowsheet.yaml")
    ed.save()
    ed.open(tmp_path / "flowsheet.yaml")
    assert ed.doc.units[uid].settings["reagent"] == reagent
    assert yaml.safe_load((tmp_path / "flowsheet.yaml").read_text())["units"][uid]["reagent"] == (
        reagent
    )


def _node(ed, uid):
    return next(n for n in ed.state.nodes if n["id"] == uid)


def _with(rows, key, value, copy=False):
    """``rows`` with ``key``'s value set (a copy when ``copy``); with ``value``
    ``None``, the row itself (``None`` when absent)."""
    if value is None:
        return next((r for r in rows if r["key"] == key), None)
    return [{**r, "value": value} if r["key"] == key else dict(r) for r in rows]
