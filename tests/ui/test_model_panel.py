"""The model panel: a model's reactions and parameters edited, checked in the
background and saved, driven through an ``Editor`` without a browser
(``tests/browser/test_model_panel.py`` drives the page itself).

The draft logic (:mod:`aquakin_ui.model_form`) is pure and tested on its own;
the editor tests answer through aquakin in this process (``InProcessWorker``).
"""

import asyncio
import copy
import itertools
import json

import pytest
import yaml
from trame.app import get_server

from aquakin_ui import model_form
from aquakin_ui.app import Editor

from .conftest import PROJECTS
from .inprocess import InProcessWorker

_names = itertools.count()

#: A small model document, as the worker's ``model_document`` gives it.
DOCUMENT = {
    "name": "m",
    "source": "shipped",
    "path": None,
    "sha256": None,
    "extends": None,
    "reactions": [
        {
            "name": "growth",
            "description": "",
            "rate": "mu * [S]",
            "stoichiometry": [{"species": "S", "value": "-1 / Y"}, {"species": "X", "value": "1.0"}],
        },
        {"name": "decay", "description": "", "rate": "b * [X]",
         "stoichiometry": [{"species": "X", "value": "-1.0"}]},
    ],
    "parameters": [
        {"name": "mu", "reaction": None, "value": "6.0", "units": "1/d", "description": ""},
        {"name": "b", "reaction": "decay", "value": "0.6", "units": "1/d", "description": ""},
    ],
    "species": ["S", "X"],
    "conditions": ["T"],
    "expressions": [],
}


# -- the draft ---------------------------------------------------------------


def test_an_untouched_draft_has_no_edits_and_whitespace_is_not_an_edit():
    draft = model_form.draft_of(DOCUMENT)
    assert model_form.edits(DOCUMENT, draft) == {}
    draft["reactions"][0]["rate"] = "  mu * [S] "
    draft["parameters"][0]["value"] = "6.0 "
    assert model_form.edits(DOCUMENT, draft) == {}


def test_the_edits_hold_only_what_changed_where_aquakin_reads_it():
    draft = model_form.draft_of(DOCUMENT)
    draft["reactions"][0]["rate"] = "2 * mu * [S]"
    draft["reactions"][0]["stoichiometry"][0]["value"] = "-1/Y "
    draft["parameters"][0]["units"] = "1/day"
    draft["parameters"][1]["value"] = "0.5"
    draft = model_form.add_parameter(draft, "k")
    draft["parameters"][-1]["value"] = "3"
    draft["parameters"][-1]["description"] = "half saturation"
    assert model_form.edits(DOCUMENT, draft) == {
        "reactions": {
            "growth": {"rate": "2 * mu * [S]", "stoichiometry": {"S": "-1/Y"}},
            "decay": {"parameters": {"b": {"value": "0.5"}}},
        },
        "parameters": {
            "mu": {"units": "1/day"},
            "k": {"value": "3", "description": "half saturation"},
        },
    }
    assert model_form.added(draft) == ["k"]
    assert model_form.expressions(model_form.edits(DOCUMENT, draft)) == [
        {"field": "reactions.growth.rate", "expression": "2 * mu * [S]", "kind": "rate",
         "reaction": "growth"},
        {"field": "reactions.growth.stoichiometry.S", "expression": "-1/Y",
         "kind": "stoichiometry", "reaction": "growth"},
    ]


def test_errors_go_under_their_field_or_the_model():
    draft = model_form.add_parameter(model_form.draft_of(DOCUMENT), "k")
    errors = [
        {"field": "reactions.growth.rate", "error": "bad rate"},
        {"field": "reactions.decay.stoichiometry.X", "error": "bad coefficient"},
        {"field": "parameters.mu", "error": "outside bounds"},
        {"field": "reactions.decay.parameters.b.value", "error": "not a number"},
        {"field": "parameters.k.value", "error": "needs a value"},
        {"field": "model", "error": "does not compile"},
        {"field": "reactions.gone", "error": "no such reaction"},
    ]
    shown, general = model_form.field_errors(errors, draft)
    assert shown == {
        "rate.0": "bad rate",
        "stoich.1.0": "bad coefficient",
        "param.0": "outside bounds",
        "param.1": "not a number",
        "param.2": "needs a value",
    }
    assert general == ["does not compile", "reactions.gone: no such reaction"]


def test_a_caret_marks_the_position_and_uses_are_one_line():
    assert model_form.pointer("mu * ([S]", 9) == "mu * ([S]\n         ^"
    assert model_form.pointer("mu", None) == ""
    refs = {"species": ["S"], "parameters": ["mu", "K"], "conditions": [], "expressions": []}
    assert model_form.uses(refs) == "species S · parameters mu, K"


@pytest.mark.parametrize(
    "flowsheet, name", [("BSM1", "asm1_bsm1"), ("My plant (v2)", "asm1_my_plant_v2"), ("--", "asm1_edited")]
)
def test_the_new_model_name_proposed_follows_the_flowsheet(flowsheet, name):
    assert model_form.proposed_name("asm1", flowsheet) == name


# -- the editor --------------------------------------------------------------


@pytest.fixture
def bsm1(tmp_path):
    """A writable copy of the BSM1 project, and an editor on it."""
    path = tmp_path / "flowsheet.yaml"
    path.write_bytes((PROJECTS / "bsm1" / "flowsheet.yaml").read_bytes())
    server = get_server(name=f"aquakin-ui-model-{next(_names)}", client_type="vue3")
    worker = InProcessWorker()
    return Editor(server, worker=worker, path=path), worker, path


async def _until(condition, timeout=60.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not condition():
        if loop.time() > end:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


def _check(ed, draft):
    """Send the page's draft as a field losing focus does, and wait for the checks."""

    async def go():
        ed.model_edited(draft)
        await _until(lambda: not ed.state.checking)

    asyncio.run(go())


def _draft(ed):
    return copy.deepcopy(ed.state.model_draft)


def _reaction(draft, name):
    return next(i for i, r in enumerate(draft["reactions"]) if r["name"] == name)


def test_the_panel_offers_the_flowsheets_models_and_shows_one(bsm1):
    ed, worker, _ = bsm1
    assert ed.state.model_choices == ["asm1"]
    ed.state.panel = "model"
    ed.state.ready()
    ed.state.flush()  # the tab's change handler opens the flowsheet's model
    assert ed.state.model_name == "asm1"
    assert "is a shipped model" in ed.state.model_source
    draft = ed.state.model_draft
    assert [r["name"] for r in draft["reactions"]][:2] == [
        "aerobic_growth_heterotrophs", "anoxic_growth_heterotrophs"
    ]
    assert "revision" in draft and not ed.state.model_dirty
    assert ed.state.model_names["conditions"] == ["T"]
    assert worker.count("model_document") == 1


def test_an_edit_is_checked_in_the_background_with_its_error_placed(bsm1):
    ed, worker, _ = bsm1
    ed.open_model("asm1")
    draft = _draft(ed)
    i = _reaction(draft, "decay_heterotrophs")
    draft["reactions"][i]["rate"] = "bH * ([XB_H]"
    _check(ed, draft)
    key = f"rate.{i}"
    assert "at position 12" in ed.state.model_errors[key]
    assert ed.state.model_pointers[key] == "bH * ([XB_H]\n            ^"
    assert not ed.state.model_valid
    assert worker.count("check_model_edits") == 0  # an unreadable rate is not loaded whole

    draft["reactions"][i]["rate"] = "2 * bH * [XB_H]"
    _check(ed, draft)
    assert ed.state.model_errors == {} and ed.state.model_general == []
    assert ed.state.model_uses[key] == "species XB_H · parameters bH"
    assert ed.state.model_valid and worker.count("check_model_edits") == 1

    _check(ed, draft)  # nothing new to ask: every answer is cached
    assert worker.count("check_expression") == 2 and worker.count("check_model_edits") == 1


def test_an_unknown_parameter_is_offered_as_a_new_row(bsm1):
    ed, _, _ = bsm1
    ed.open_model("asm1")
    draft = _draft(ed)
    i = _reaction(draft, "decay_heterotrophs")
    draft["reactions"][i]["rate"] = "k_decay * [XB_H]"
    _check(ed, draft)
    assert ed.state.model_offers == ["k_decay"]
    assert ed.state.model_errors[f"rate.{i}"].startswith("'k_decay' is not a parameter")

    async def add():
        ed.add_model_parameter(_draft(ed), "k_decay")
        await _until(lambda: not ed.state.checking)

    asyncio.run(add())
    draft = _draft(ed)
    row = draft["parameters"][-1]
    assert row == {"name": "k_decay", "reaction": None, "value": "", "units": "",
                   "description": "", "new": True}
    assert ed.state.model_offers == [] and ed.state.model_dirty
    assert ed.state.model_errors == {f"param.{len(draft['parameters']) - 1}": "a new parameter needs a value"}
    row["value"] = "0.3"
    _check(ed, draft)
    assert ed.state.model_valid


def test_a_whole_model_error_is_shown_and_blocks_apply(bsm1):
    ed, worker, path = bsm1
    ed.open_model("asm1")
    draft = _draft(ed)
    muH = next(p for p in draft["parameters"] if p["name"] == "muH")
    muH["value"] = "100"
    _check(ed, draft)
    k = draft["parameters"].index(muH)
    assert "outside bounds" in ed.state.model_errors[f"param.{k}"]
    before = path.read_bytes()
    ed.apply_model(draft)
    assert ed.state.message == "the edits do not make a valid model; the panel shows why"
    assert path.read_bytes() == before and not (path.parent / "models").exists()
    assert worker.count("save_model_edits") == 0


def test_a_shipped_model_is_saved_as_a_new_model_named_in_a_dialog(bsm1):
    ed, worker, path = bsm1
    ed.open_model("asm1")
    draft = _draft(ed)
    i = _reaction(draft, "decay_heterotrophs")
    draft["reactions"][i]["rate"] = "2 * bH * [XB_H]"
    ed.apply_model(draft)  # checked and planned at once, unchecked before
    assert ed.state.model_dialog_open
    assert ed.state.model_new_name == "asm1_bsm1"
    assert ed.state.model_dialog_units == [u for u, x in ed.doc.units.items()
                                           if x.settings.get("model") == "asm1"]
    assert not (path.parent / "models").exists()  # the plan wrote nothing

    ed.plan_model_name("asm3")
    assert "'asm3' is the name of a shipped model" in ed.state.model_dialog_error
    ed.answer_model_dialog("save", "asm3")
    assert ed.state.model_dialog_open and "shipped model" in ed.state.model_dialog_error
    ed.plan_model_name("asm1_bsm1")
    assert ed.state.model_dialog_error == "" and len(ed.state.model_dialog_units) == 10

    ed.answer_model_dialog("save", "asm1_bsm1")
    assert not ed.state.model_dialog_open
    written = yaml.safe_load((path.parent / "models" / "asm1_bsm1.yaml").read_text())
    assert written["reactions"] == [{"name": "decay_heterotrophs", "rate": "2 * bH * [XB_H]"}]
    spec = yaml.safe_load(path.read_text())
    assert list(spec["models"]) == ["asm1_bsm1"]
    assert {u.settings["model"] for u in ed.doc.units.values()} == {"asm1_bsm1"}
    assert not ed._unsaved()
    # The panel now shows the new project model, the edit part of it.
    assert ed.state.model_name == "asm1_bsm1" and "Apply writes the edits into" in ed.state.model_source
    assert ed.state.model_draft["reactions"][i]["rate"] == "2 * bH * [XB_H]"
    assert ed.state.message.startswith("saved model asm1_bsm1")


def test_a_project_model_is_saved_in_place_without_a_dialog(bsm1):
    ed, worker, path = bsm1
    ed.open_model("asm1")
    draft = _draft(ed)
    draft["parameters"][0]["value"] = "5.0"
    ed.apply_model(draft)
    ed.answer_model_dialog("save", "asm1_bsm1")
    model_file = path.parent / "models" / "asm1_bsm1.yaml"
    draft = _draft(ed)
    draft["parameters"][0]["description"] = "maximum growth rate"
    ed.apply_model(draft)
    assert not ed.state.model_dialog_open
    assert yaml.safe_load(model_file.read_text())["parameters"]["muH"] == {
        "value": 5.0, "description": "maximum growth rate"
    }
    assert ed.state.model_draft["parameters"][0]["description"] == "maximum growth rate"
    assert json.loads(json.dumps(yaml.safe_load(path.read_text())["models"]))["asm1_bsm1"][
        "source"
    ] == "models/asm1_bsm1.yaml"


def test_cancelling_the_dialog_writes_nothing(bsm1):
    ed, _, path = bsm1
    ed.open_model("asm1")
    draft = _draft(ed)
    draft["parameters"][0]["value"] = "5.0"
    ed.apply_model(draft)
    ed.answer_model_dialog("cancel")
    assert not ed.state.model_dialog_open and not (path.parent / "models").exists()
    ed.plan_model_name("x")  # no plan any more: nothing to say
    assert ed.state.model_new_name == "asm1_bsm1"


def test_apply_says_why_it_cannot_save(bsm1, tmp_path):
    ed, _, path = bsm1
    ed.open_model("asm1")
    ed.apply_model(_draft(ed))
    assert ed.state.message == "the model has no edits to apply"
    draft = _draft(ed)
    draft["parameters"][0]["value"] = "5.0"
    path.write_text(path.read_text() + "\n# changed elsewhere\n")
    ed.apply_model(draft)
    assert "has changed on disk since it was opened here" in ed.state.message

    server = get_server(name=f"aquakin-ui-model-{next(_names)}", client_type="vue3")
    new = Editor(server, worker=InProcessWorker())
    new.on_dropped({"kind": "cstr", "position": {"x": 0, "y": 0}})
    new.open_model("asm1")
    new.apply_model(draft)
    assert new.state.message.startswith("type where to keep the flowsheet")


def test_a_refused_request_is_shown_not_raised(bsm1):
    ed, worker, _ = bsm1
    worker.refuse["model_document"] = "no such model"
    ed.open_model("asm1")
    assert ed.state.message == "could not read model asm1: no such model"
    del worker.refuse["model_document"]
    ed.open_model("asm1")
    draft = _draft(ed)
    draft["parameters"][0]["value"] = "5.0"
    worker.refuse["check_model_edits"] = "the worker stopped"
    ed.apply_model(draft)
    assert ed.state.message == "could not save model asm1: the worker stopped"
    _check(ed, draft)
    assert ed.state.model_general == ["the worker stopped"]
    worker.refuse["check_expression"] = "the worker stopped"
    i = _reaction(draft, "decay_heterotrophs")
    draft["reactions"][i]["rate"] = "2 * bH * [XB_H]"
    _check(ed, draft)
    assert ed.state.model_errors[f"rate.{i}"] == "the worker stopped"


def test_discarding_draws_the_model_as_it_is(bsm1):
    ed, _, _ = bsm1
    ed.open_model("asm1")
    first = ed.state.model_draft["revision"]
    draft = _draft(ed)
    draft["parameters"][0]["value"] = "5.0"
    _check(ed, draft)
    ed.discard_model_edits()
    assert ed.state.model_draft["parameters"][0]["value"] == "6.0"
    assert ed.state.model_draft["revision"] > first  # reaches a page holding the edit
    assert not ed.state.model_dirty and ed.state.model_errors == {}


def test_the_default_model_is_the_selected_units(bsm1):
    from aquakin_ui.model_panel import _default_model

    ed, _, _ = bsm1
    ed.doc.meta["models"]["adm1"] = {"source": "shipped"}
    ed.doc.units["tank5"].settings["model"] = "adm1"
    ed.publish(checks=False)
    assert ed.state.model_choices == ["asm1", "adm1"]
    ed.state.selected = "tank5"
    assert _default_model(ed) == "adm1"
    ed.state.selected = None
    assert _default_model(ed) == "asm1"


def test_without_a_model_open_discard_and_apply_do_nothing(bsm1):
    ed, worker, _ = bsm1
    ed.discard_model_edits()
    ed.apply_model({"reactions": [], "parameters": []})
    assert ed.state.model_draft == {"reactions": [], "parameters": []}
    assert worker.sent[-1][0] != "check_model_edits"


def test_answers_for_a_model_no_longer_shown_are_dropped(bsm1):
    """A model opened while the checks of another were on their way: their
    answers are not shown against the new one's draft."""
    from aquakin_ui.model_panel import ModelPanel

    ed, worker, _ = bsm1
    ed.open_model("asm1")
    draft = _draft(ed)
    i = _reaction(draft, "decay_heterotrophs")
    draft["reactions"][i]["rate"] = "bH * ([XB_H]"
    ask = worker.request

    def opening_another(command, **fields):
        if command == "check_expression":
            ed._model = ModelPanel()  # as open_model's answer would, meanwhile
        return ask(command, **fields)

    worker.request = opening_another
    _check(ed, draft)
    assert ed.state.model_errors == {}


def test_a_project_model_changed_on_disk_is_not_overwritten(bsm1):
    ed, _, path = bsm1
    ed.open_model("asm1")
    draft = _draft(ed)
    draft["parameters"][0]["value"] = "5.0"
    ed.apply_model(draft)
    ed.answer_model_dialog("save", "asm1_bsm1")
    model_file = path.parent / "models" / "asm1_bsm1.yaml"
    model_file.write_text(model_file.read_text() + "# edited elsewhere\n")
    draft = _draft(ed)
    draft["parameters"][0]["value"] = "4.0"
    ed.apply_model(draft)
    assert "has changed on disk since it was opened for editing" in ed.state.message
    assert model_file.read_text().endswith("# edited elsewhere\n")


def test_a_model_the_flowsheet_stops_using_leaves_the_panel(bsm1):
    ed, _, _ = bsm1
    ed.open_model("asm1")
    ed.doc.meta["models"]["adm1"] = {"source": "shipped"}
    for unit in ed.doc.units.values():
        unit.settings["model"] = "adm1"
    ed.publish(checks=False)
    assert ed.state.model_choices == ["adm1"]
    assert ed.state.model_name is None and ed.state.model_draft["reactions"] == []
    assert ed._model.document is None


def test_a_commented_project_model_asks_before_its_comments_go(tmp_path):
    """BSM2's ``asm1_bsm2.yaml`` is commented: Apply shows the save dialog with
    the warning (no name to choose), and saves only when told to."""
    import shutil

    project = tmp_path / "bsm2"
    shutil.copytree(PROJECTS / "bsm2", project)
    server = get_server(name=f"aquakin-ui-model-{next(_names)}", client_type="vue3")
    ed = Editor(server, worker=InProcessWorker(), path=project / "flowsheet.yaml")
    model_file = project / "models" / "asm1_bsm2.yaml"
    before = model_file.read_text()
    ed.open_model("asm1_bsm2")
    draft = _draft(ed)
    draft["parameters"][0]["value"] = "4.5"
    ed.apply_model(draft)
    assert ed.state.model_dialog_open and not ed.state.model_dialog_new
    assert ed.state.model_dialog_warning == (
        "Comments in models/asm1_bsm2.yaml will not be kept when it is saved."
    )
    assert model_file.read_text() == before  # nothing written while it asks
    ed.answer_model_dialog("save")
    assert not ed.state.model_dialog_open
    assert yaml.safe_load(model_file.read_text())["parameters"]["muH"]["value"] == 4.5
    assert "#" not in model_file.read_text()


def test_a_model_without_comments_or_made_from_a_shipped_one_has_no_warning(bsm1):
    ed, _, _ = bsm1
    ed.open_model("asm1")
    draft = _draft(ed)
    draft["parameters"][0]["value"] = "5.0"
    ed.apply_model(draft)
    assert ed.state.model_dialog_new and ed.state.model_dialog_warning == ""
    ed.answer_model_dialog("save", "asm1_bsm1")
    draft = _draft(ed)  # now asm1_bsm1, a file written without comments
    draft["parameters"][0]["value"] = "4.0"
    ed.apply_model(draft)
    assert not ed.state.model_dialog_open and ed.state.model_dialog_warning == ""
    assert ed.state.message.startswith("saved model asm1_bsm1")
