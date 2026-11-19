"""The unit form: rows from the catalogue, and the settings they give back."""

import pytest
import yaml

from aquakin.plant import Plant
from aquakin.plant.catalog import model_info
from aquakin_ui import forms
from aquakin_ui.document import Flowsheet

from .conftest import PROJECTS


@pytest.fixture
def bsm1(ui_catalog):
    path = PROJECTS / "bsm1" / "flowsheet.yaml"
    return Flowsheet.from_spec(ui_catalog, yaml.safe_load(path.read_text()), base_dir=path.parent)


def _info(name):
    return model_info(name)


def _rows(doc, uid):
    return {r["key"]: r for r in forms.rows_for(doc, uid, _info)}


def test_rows_come_from_the_catalogue(bsm1):
    rows = _rows(bsm1, "tank3")
    assert rows["id"]["widget"] == "readonly" and rows["id"]["value"] == "tank3"
    assert rows["label"]["value"] == "Aerobic zone 1"
    assert rows["model"]["widget"] == "select" and "asm1" in rows["model"]["items"]
    assert rows["volume"]["value"] == "1333" and rows["volume"]["label"] == "volume *"
    assert rows["conditions.T"]["value"] == "293.15"
    assert rows["aeration.mode"]["value"] == "kla" and rows["aeration.value"]["value"] == "240"
    assert "Tank liquid volume" in rows["volume"]["hint"]


def test_an_unedited_form_gives_back_the_same_plant(bsm1):
    before = Plant.from_spec(bsm1.to_spec(), base_dir=PROJECTS / "bsm1").to_spec()
    for uid in list(bsm1.units):
        label, settings = forms.read_rows(bsm1, uid, forms.rows_for(bsm1, uid, _info))
        bsm1.apply(uid, settings, label)
    after = Plant.from_spec(bsm1.to_spec(), base_dir=PROJECTS / "bsm1").to_spec()
    assert after == before


def test_edits_keep_what_the_form_does_not_show(bsm1):
    rows = forms.rows_for(bsm1, "tank3", _info)
    for r in rows:
        if r["key"] == "volume":
            r["value"] = "1500"
        if r["key"] == "aeration.value":
            r["value"] = "180"
    label, settings = forms.read_rows(bsm1, "tank3", rows)
    assert settings["volume"] == 1500.0
    assert settings["aeration"]["kla"] == 180.0 and settings["aeration"]["Kp"] == 25.0


def test_a_value_of_the_wrong_type_is_refused(bsm1):
    rows = forms.rows_for(bsm1, "tank3", _info)
    for r in rows:
        if r["key"] == "volume":
            r["value"] = "large"
    with pytest.raises(ValueError):
        forms.read_rows(bsm1, "tank3", rows)


@pytest.mark.parametrize("text", ["nan", "NaN", "1e999"])
def test_nan_and_overflowing_numbers_are_refused(bsm1, text):
    rows = forms.rows_for(bsm1, "tank3", _info)
    for r in rows:
        if r["key"] in ("volume", "conditions.T"):
            r["value"] = text
            with pytest.raises(ValueError, match="not a finite number"):
                forms.read_rows(bsm1, "tank3", rows)
            r["value"] = "1000"
    for r in rows:
        if r["key"] == "volume":
            r["value"] = "inf"
    assert forms.read_rows(bsm1, "tank3", rows)[1]["volume"] == "inf"


def test_numbers_are_shown_exactly_so_apply_changes_nothing_unedited(ui_catalog):
    path = PROJECTS / "bsm2" / "flowsheet.yaml"
    doc = Flowsheet.from_spec(ui_catalog, yaml.safe_load(path.read_text()), base_dir=path.parent)
    influent = doc.units["influent"]
    rows = forms.rows_for(doc, "influent", lambda name: None)  # its model is a project file
    assert {r["key"]: r["value"] for r in rows}["T"] == "288.00808"
    _, settings = forms.read_rows(doc, "influent", rows)
    assert (
        settings["T"] == 288.00808 and settings["composition"] == influent.settings["composition"]
    )
    assert forms._text(0.1 + 0.2) == "0.30000000000000004" and forms._text(1e-7) == "1e-07"


def test_switching_aeration_mode_does_not_carry_the_old_number(bsm1):
    rows = forms.rows_for(bsm1, "tank3", _info)  # kLa 240
    mode = next(r for r in rows if r["key"] == "aeration.mode")
    mode["value"] = "do_setpoint"
    _, settings = forms.read_rows(bsm1, "tank3", rows)
    assert settings["aeration"]["do_setpoint"] == 0.0 and "kla" not in settings["aeration"]
    # In the same mode the field is read as usual.
    rows = forms.rows_for(bsm1, "tank3", _info)
    next(r for r in rows if r["key"] == "aeration.value")["value"] = "200"
    assert forms.read_rows(bsm1, "tank3", rows)[1]["aeration"]["kla"] == 200.0


def test_conditions_are_kept_when_the_model_cannot_be_described(bsm1):
    rows = forms.rows_for(bsm1, "tank3", lambda name: None)
    assert not any(r["key"].startswith("conditions.") for r in rows)
    _, settings = forms.read_rows(bsm1, "tank3", rows)
    assert settings["conditions"] == {"T": 293.15}
