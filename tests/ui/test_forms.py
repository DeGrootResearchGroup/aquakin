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
    bsm1.units["tank3"].settings.pop("conditions")  # none to keep
    assert "conditions" not in forms.read_rows(bsm1, "tank3", rows)[1]


def test_a_bad_value_is_named_by_its_field(bsm1):
    for key, text, message in (
        ("volume", "0,5", "volume: '0,5' is not a number"),
        ("conditions.T", "warm", "condition T: 'warm' is not a number"),
        ("aeration.value", "lots", "kla: 'lots' is not a number"),
    ):
        rows = forms.rows_for(bsm1, "tank3", _info)
        next(r for r in rows if r["key"] == key)["value"] = text
        with pytest.raises(ValueError, match=message):
            forms.read_rows(bsm1, "tank3", rows)
    rows = forms.rows_for(bsm1, "clarifier", _info)
    next(r for r in rows if r["key"] == "n_layers")["value"] = "2.5"
    with pytest.raises(ValueError, match=r"n_layers: '2\.5' is not a whole number"):
        forms.read_rows(bsm1, "clarifier", rows)
    rows = forms.rows_for(bsm1, "tank5_split", _info)
    row = next(r for r in rows if r["widget"] == "text" and ":" in str(r["value"]))
    row["value"] = "internal_recycle 55338"
    with pytest.raises(ValueError, match="write each entry as 'name: value'"):
        forms.read_rows(bsm1, "tank5_split", rows)


def test_the_aerated_species_is_chosen_from_the_units_model(ui_catalog):
    doc = Flowsheet(ui_catalog)
    uid = doc.add("cstr", {"x": 0, "y": 0}, {"model": "asm3", "volume": 500.0})
    doc.units[uid].settings["aeration"] = {"kla": 100.0, "species": "SO"}  # ASM1's name
    rows = {r["key"]: r for r in forms.rows_for(doc, uid, _info)}
    assert rows["aeration.species"]["value"] == "SO2"  # ASM3's oxygen, suggested
    assert "SO2" in rows["aeration.species"]["items"]
    _, settings = forms.read_rows(doc, uid, list(rows.values()))
    assert settings["aeration"]["species"] == "SO2" and settings["aeration"]["kla"] == 100.0
    assert forms._oxygen(["SA", "S_O2_aq"]) == "S_O2_aq" and forms._oxygen(["SA"]) is None


def test_help_text_is_plain_and_names_the_unit(bsm1):
    from aquakin_ui.catalog import plain

    assert plain("Sums ``Q_in``, see :class:`~aquakin.plant.Aeration` and *note*.") == (
        "Sums Q_in, see Aeration and note."
    )
    assert "``" not in bsm1.catalog.kind("mixer")["summary"]
    assert forms.unit_in("Clarifier liquid volume (m³); sets the HRT.") == "m³"
    assert forms.unit_in("Bulk liquid volume (m^3).") == "m³"
    assert forms.unit_in("Fixed underflow flow rate, m³/d (the pump).") == "m³/d"
    assert forms.unit_in("A ratio (e.g. 0.5).") == ""
    rows = {r["key"]: r for r in forms.rows_for(bsm1, "tank3", _info)}
    assert rows["volume"]["suffix"] == "m³" and rows["conditions.T"]["suffix"] == "K"
    assert rows["aeration.value"]["suffix"] == "1/d"


def test_a_bad_value_names_its_row(bsm1):
    rows = forms.rows_for(bsm1, "tank3", _info)
    next(r for r in rows if r["key"] == "volume")["value"] = "big"
    with pytest.raises(forms.FieldError) as caught:
        forms.read_rows(bsm1, "tank3", rows)
    assert caught.value.key == "volume"


def test_a_bad_value_is_reported_under_its_label(bsm1):
    """The message names the field as the form shows it, without the required
    mark ("volume *" is "volume"; aeration's value is "kla", not its row key),
    then says what is wrong with the text."""
    for key, label in [("volume", "volume"), ("aeration.value", "kla")]:
        rows = forms.rows_for(bsm1, "tank3", _info)
        next(r for r in rows if r["key"] == key)["value"] = "big"
        with pytest.raises(forms.FieldError, match="is not a number") as caught:
            forms.read_rows(bsm1, "tank3", rows)
        assert str(caught.value) == f"{label}: 'big' is not a number"
        assert caught.value.key == key


def test_settling_species_follow_the_units_model():
    asm3 = ["SO2", "SS", "XI", "XS", "XH"]
    settling = {"name": "particulate_species", "default": ["XS", "XB_H"]}
    assert forms.suggest_species(settling, None, asm3) == ["XI", "XS", "XH"]
    assert forms.suggest_species(settling, ["XS"], asm3) is None  # all in the model
    assert forms.suggest_species({**settling, "default": []}, None, asm3) == ["XI", "XS", "XH"]
    assert forms.suggest_species({"name": "gas_species", "default": ["X"]}, None, asm3) is None


def test_units_on_the_plants_clock_follow_it(ui_catalog):
    assert forms.unit_in("Fixed dose flow (m³/d).", "h") == "m³/h"
    assert forms.unit_in("Integral time (d, on the plant's clock).", "min") == "min"
    assert forms.unit_in("Decay rate (1/d). Default 0.", "s") == "1/s"
    assert forms.unit_in("Applied dose (g/m³, i.e. mg/L of chlorine).", "h") == "g/m³"
    assert forms.unit_in("Surface (m²/m³: biofilm area per volume).") == "m²/m³"
    # Every shipped unit kind names the unit of its main sizes and flows.
    named = {
        (k, s["name"]): forms.unit_in(s["help"])
        for k, kind in ui_catalog.kinds.items()
        for s in kind["settings"]
    }
    assert named[("cstr", "volume")] == "m³" and named[("influent", "Q")] == "m³/d"
    assert named[("ideal_clarifier", "underflow_Q")] == "m³/d"
    assert named[("dosing", "flow")] == "m³/d" and named[("pi_controller", "Ti")] == "d"


def test_a_scheduled_value_keeps_a_number_a_number_and_an_expression_text():
    """An influent's Q, T and composition may each be a number or an expression
    of time. A number has to stay a number, so a constant feed is written to the
    spec exactly as it was before expressions existed."""
    from aquakin_ui.forms import _parse

    number = {"type": "scheduled_number"}
    assert _parse(number, "2000") == 2000.0
    assert _parse(number, " 2000 ") == 2000.0
    assert _parse(number, "1e4") == 1e4
    assert _parse(number, "") is None
    # Anything that is not a number is kept verbatim for aquakin to read --
    # aquakin_ui cannot read it (the independence rule), so it does not try.
    assert _parse(number, "2000 + 500*sin(2*pi*t)") == "2000 + 500*sin(2*pi*t)"
    assert _parse(number, "wobble(t)") == "wobble(t)"

    composition = {"type": "scheduled_species_values"}
    assert _parse(composition, "SS: 60, SNH: 30 + 10*sin(2*pi*t)") == {
        "SS": 60.0,
        "SNH": "30 + 10*sin(2*pi*t)",
    }
    with pytest.raises(ValueError, match="write each entry as 'name: value'"):
        _parse(composition, "SS 60")
