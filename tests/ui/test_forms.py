"""The unit form: rows from the catalogue, and the settings they give back."""

import re

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
    with pytest.raises(forms.FieldError, match="volume: 'large' is not a number"):
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
    with pytest.raises(forms.FieldError, match="'big' is not a number") as caught:
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


def test_a_pump_map_entry_may_be_an_expression_with_commas_inside():
    """A setpoint splitter's flows are a scheduled port map: each entry a number
    or an expression, and a comma inside an expression's brackets does not end
    the entry."""
    from aquakin_ui.forms import _parse

    pumps = {"type": "scheduled_port_map"}
    assert _parse(pumps, "ras: 300 + 100*clip(t - 10, 0, 1), waste: 40") == {
        "ras": "300 + 100*clip(t - 10, 0, 1)",
        "waste": 40.0,
    }
    with pytest.raises(ValueError, match=re.escape("(not 'ras 300')")):
        _parse(pumps, "ras 300")
    # The same splitting serves a composition.
    composition = {"type": "scheduled_species_values"}
    assert _parse(composition, "SNH: max(20, 30*sin(2*pi*t)), SS: 60") == {
        "SNH": "max(20, 30*sin(2*pi*t))",
        "SS": 60.0,
    }


def test_setpoint_expressions_are_shown_and_read_back_as_text(bsm1):
    """The clarifier underflow and a pump flow of BSM1 written as expressions:
    the form shows the text, gives it back unchanged, and the plant it describes
    builds with them."""
    bsm1.apply("clarifier", {**bsm1.units["clarifier"].settings, "underflow_Q": "18831 + 385*sin(2*pi*t)"}, None)
    split = bsm1.units["tank5_split"].settings
    flows = {**split["output_port_flows"]}
    port = next(iter(flows))
    flows[port] = "55338*(1 + 0.1*step(t - 7))"
    bsm1.apply("tank5_split", {**split, "output_port_flows": flows}, None)

    rows = _rows(bsm1, "clarifier")
    assert rows["underflow_Q"]["value"] == "18831 + 385*sin(2*pi*t)"
    assert rows["underflow_Q"]["suffix"] == "m³/d"
    _, settings = forms.read_rows(bsm1, "clarifier", list(rows.values()))
    assert settings["underflow_Q"] == "18831 + 385*sin(2*pi*t)"

    rows = _rows(bsm1, "tank5_split")
    assert rows["output_port_flows"]["value"] == f"{port}: 55338*(1 + 0.1*step(t - 7))"
    _, settings = forms.read_rows(bsm1, "tank5_split", list(rows.values()))
    assert settings["output_port_flows"] == flows

    plant = Plant.from_spec(bsm1.to_spec(), base_dir=PROJECTS / "bsm1")
    assert plant.units["clarifier"].scheduled_setpoints() == ("underflow_Q",)
    assert plant.units["tank5_split"].scheduled_setpoints() == (port,)


# -- a dosing unit's reagent --------------------------------------------------


@pytest.fixture
def dosing(ui_catalog):
    """A flowsheet holding one dosing unit with no reagent yet."""
    return Flowsheet.from_spec(
        ui_catalog,
        {"format": 2, "name": "p", "units": {"dose": {"kind": "dosing", "flow": 2.0}}},
    )


def _reagent_rows(doc, model="", label="", entries=()):
    """The dosing unit's rows with the reagent's fields set as a user would."""
    rows = forms.rows_for(doc, "dose", _info)
    edits = {"reagent.model": model, "reagent.label": label, "reagent.composition": entries}
    return [{**r, "value": edits[r["key"]]} if r["key"] in edits else r for r in rows]


def test_a_new_reagent_is_drawn_empty_and_left_unset(dosing):
    rows = _rows(dosing, "dose")
    assert rows["reagent.model"]["value"] == "" and rows["reagent.model"]["reshape"]
    assert rows["reagent.label"]["placeholder"] == "reagent (default)"
    composition = rows["reagent.composition"]
    assert composition["widget"] == "composition"
    assert composition["value"] == [] and composition["items"] == []
    # A wholly empty entry (an "Add a species" left blank) is no entry.
    rows = _reagent_rows(dosing, entries=[{"species": "", "value": ""}])
    _, settings = forms.read_rows(dosing, "dose", rows)
    assert settings == {"flow": 2.0}


def test_a_reagent_composed_in_the_form_builds_that_reagent(dosing):
    """The reagent the form gives is the one aquakin builds: the concentrations
    typed in, every other species of the model zero."""
    from aquakin import load_model
    from aquakin.plant.catalog import build_unit

    rows = _reagent_rows(dosing, model="asm1")
    assert rows[-1]["items"] == []  # the species come with the model, on the redraw
    dosing.apply("dose", forms.read_rows(dosing, "dose", rows)[1], None)
    assert dosing.units["dose"].settings["reagent"] == {"model": "asm1", "composition": {}}
    composition = _rows(dosing, "dose")["reagent.composition"]
    assert composition["items"] == list(load_model("asm1").species)
    assert composition["units"]["SNO"] == "g_N/m³"

    entries = [{"species": "SS", "value": "4e5"}, {"species": "SNO", "value": " 1000 "}]
    rows = _reagent_rows(dosing, "asm1", "methanol", entries)
    _, settings = forms.read_rows(dosing, "dose", rows)
    assert settings["reagent"] == {
        "model": "asm1",
        "label": "methanol",
        "composition": {"SS": 400000.0, "SNO": 1000.0},
    }
    unit = build_unit("dosing", "dose", settings)
    asm1 = load_model("asm1")
    expected = [{"SS": 4e5, "SNO": 1000.0}.get(s, 0.0) for s in asm1.species]
    assert [float(c) for c in unit.reagent.composition] == expected
    assert unit.reagent.label == "methanol"


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([{"species": "", "value": "5"}], "choose the species 5 is the concentration of"),
        ([{"species": "SS", "value": ""}], "give SS a concentration"),
        ([{"species": "SS", "value": None}], "give SS a concentration"),
        ([{"species": "SS", "value": "1"}, {"species": "SS", "value": "2"}], "SS is listed twice"),
        ([{"species": "SS", "value": "lots"}], "'lots' is not a number"),
        ([{"species": "SS", "value": "nan"}], "'nan' is not a finite number"),
        ([{"species": "SFOO", "value": "1"}], "'SFOO' is not a species of the reagent's model"),
    ],
)
def test_an_unreadable_composition_entry_is_named(dosing, entries, message):
    dosing.apply("dose", {"flow": 2.0, "reagent": {"model": "asm1"}}, None)
    rows = _reagent_rows(dosing, "asm1", entries=entries)
    with pytest.raises(forms.FieldError, match=f"^reagent composition: {re.escape(message)}$") as caught:
        forms.read_rows(dosing, "dose", rows)
    assert caught.value.key == "reagent.composition"


@pytest.mark.parametrize(
    ("label", "entries"), [("", [{"species": "SS", "value": "1"}]), ("methanol", [])]
)
def test_a_reagent_without_its_model_is_refused_at_the_model(dosing, label, entries):
    rows = _reagent_rows(dosing, label=label, entries=entries)
    with pytest.raises(
        forms.FieldError, match=r"^reagent model: choose the model the reagent is written in$"
    ) as caught:
        forms.read_rows(dosing, "dose", rows)
    assert caught.value.key == "reagent.model"


def test_a_reagent_on_a_model_that_cannot_be_described_takes_any_species(dosing):
    """With no description of the model there is no species list to refuse a
    name against; aquakin's check reports one the model lacks."""
    dosing.apply("dose", {"flow": 2.0, "reagent": {"model": "mine"}}, None)
    rows = forms.rows_for(dosing, "dose", lambda name: None)
    rows = [
        {**r, "value": [{"species": "SANY", "value": "3"}]}
        if r["key"] == "reagent.composition"
        else r
        for r in rows
    ]
    _, settings = forms.read_rows(dosing, "dose", rows)
    assert settings["reagent"] == {"model": "mine", "composition": {"SANY": 3.0}}


def test_the_choices_that_redraw_the_form_are_marked(bsm1):
    marked = {r["key"] for r in forms.rows_for(bsm1, "tank3", _info) if r.get("reshape")}
    assert marked == {"model", "aeration.mode"}


# -- an SBR's phase table and settling model ----------------------------------


_PHASES = [
    {"name": "fill", "duration": 0.1, "feed": True},
    {"name": "react", "duration": 0.3, "kla": 240.0},
    {"name": "decant", "duration": 0.1, "decant": True, "mixed": False},
]
_SBR = {
    "kind": "sbr",
    "model": "asm1",
    "conditions": {"T": 293.15},
    "full_volume": 1000.0,
    "feed_flow": 5000.0,
    "decant_flow": 5000.0,
    "feed": "batch",
    "phases": _PHASES,
    "settling": {"kind": "interface", "v_settle": 400.0, "area": 200.0},
    # ASM1's particulates, which the form would otherwise suggest for an empty list.
    "particulate_species": ["XI", "XS", "XB_H", "XB_A", "XP", "XND"],
}


@pytest.fixture
def sbr(ui_catalog):
    """A flowsheet holding one SBR, its phases written tersely by hand."""
    return Flowsheet.from_spec(ui_catalog, {"format": 2, "name": "p", "units": {"sbr": _SBR}})


def _edit(rows, **values):
    """``rows`` with the given values (keys with ``.`` written as ``__``)."""
    values = {k.replace("__", "."): v for k, v in values.items()}
    return [{**r, "value": values[r["key"]]} if r["key"] in values else r for r in rows]


def test_the_phases_are_a_table_and_the_settling_model_its_kind_and_arguments(sbr):
    rows = _rows(sbr, "sbr")
    phases = rows["phases"]
    assert phases["widget"] == "records" and phases["item"] == "phase"
    assert [f["name"] for f in phases["fields"]] == [
        "name", "duration", "feed", "decant", "kla", "settle", "mixed",
    ]  # fmt: skip
    fields = {f["name"]: f for f in phases["fields"]}
    assert fields["duration"]["suffix"] == "d" and fields["kla"]["suffix"] == "1/d"
    assert fields["kla"]["placeholder"] == "0 (default)"
    assert phases["value"][1] == {
        "name": "react",
        "duration": "0.3",
        "feed": False,
        "decant": False,
        "kla": "240",
        "settle": False,
        "mixed": "",
    }
    assert phases["value"][2]["mixed"] == "false"
    assert phases["blank"] == {
        "name": "",
        "duration": "",
        "feed": False,
        "decant": False,
        "kla": "",
        "settle": False,
        "mixed": "",
    }
    kind = rows["settling.kind"]
    assert (kind["widget"], kind["value"], kind["reshape"]) == ("select", "interface", True)
    assert kind["items"] == ["interface", "layered"]
    assert rows["settling.v_settle"]["value"] == "400"
    assert rows["settling.v_settle"]["suffix"] == "m/d"
    assert rows["settling.remix_rate"]["placeholder"] == "1000 (default)"
    assert "settling.n_layers" not in rows
    # Choosing the other kind draws its arguments.
    sbr.apply("sbr", {**sbr.units["sbr"].settings, "settling": {"kind": "layered"}}, None)
    assert _rows(sbr, "sbr")["settling.n_layers"]["placeholder"] == "4 (default)"


def test_an_unedited_sbr_form_gives_back_the_same_settings_and_plant(sbr):
    before = sbr.units["sbr"].settings
    _, settings = forms.read_rows(sbr, "sbr", forms.rows_for(sbr, "sbr", _info))
    # Every phase reads back with its boxes and its "mixed" choice written out.
    assert settings["phases"][0] == {
        "name": "fill",
        "duration": 0.1,
        "feed": True,
        "decant": False,
        "settle": False,
        "mixed": None,
    }
    assert settings["settling"] == before["settling"]
    full = Plant.from_spec(sbr.to_spec()).to_spec()
    sbr.apply("sbr", settings, None)
    assert Plant.from_spec(sbr.to_spec()).to_spec() == full


def test_the_feed_mode_is_a_choice_that_shows_only_its_own_settings(sbr):
    """``feed`` is a select of the three modes that redraws the form; the fill
    pump's flow shows with a storage or batch feed, the holding volume's
    settings with a storage feed only, and neither with a continuous feed."""
    rows = _rows(sbr, "sbr")
    feed = rows["feed"]
    assert (feed["widget"], feed["value"], feed["reshape"]) == ("select", "batch", True)
    assert feed["items"] == ["continuous", "storage", "batch"]
    assert feed["label"] == "feed *" and feed["placeholder"] == "required"
    storage = {"storage_volume", "storage_initial_fraction", "overflow_port"}
    assert "feed_flow" in rows and not storage & set(rows)
    settings = sbr.units["sbr"].settings
    sbr.apply("sbr", {**settings, "feed": "storage", "storage_volume": 80.0}, None)
    rows = _rows(sbr, "sbr")
    assert storage <= set(rows) and "feed_flow" in rows
    assert rows["storage_volume"]["value"] == "80" and rows["storage_volume"]["suffix"] == "m³"
    assert rows["storage_initial_fraction"]["placeholder"] == "0.5 (default)"
    # Python-only, like the tank's own initial concentrations.
    assert rows["storage_initial_concentrations"]["widget"] == "readonly"
    sbr.apply("sbr", {**settings, "feed": "continuous", "feed_flow": None}, None)
    rows = _rows(sbr, "sbr")
    assert not (storage | {"feed_flow"}) & set(rows)


def test_switching_feed_mode_leaves_the_other_modes_settings_unset(sbr):
    """Read back after the mode changes, the settings only the old mode had are
    dropped (they would be refused), and the ports follow the mode."""
    settings = sbr.units["sbr"].settings
    sbr.apply("sbr", {**settings, "feed": "storage", "storage_volume": 80.0}, None)
    assert sbr.catalog.ports_for("sbr", sbr.units["sbr"].settings)[1] == ["effluent", "overflow"]
    rows = _edit(forms.rows_for(sbr, "sbr", _info), feed="continuous")
    _, back = forms.read_rows(sbr, "sbr", rows)
    assert back["feed"] == "continuous"
    assert not {"feed_flow", "storage_volume"} & set(back)
    sbr.apply("sbr", back, None)
    assert sbr.catalog.ports_for("sbr", back)[1] == ["effluent"]
    assert Plant.from_spec(sbr.to_spec()).units["sbr"].feed == "continuous"


def test_phases_edited_reordered_and_added_in_the_form_build_those_phases(sbr):
    """The table's entries, in the order shown, are the SBR's phases; an entry
    added and left blank is no phase."""
    from aquakin.plant import InterfaceSettling, LayeredSettling, SBRPhase
    from aquakin.plant.catalog import build_unit

    rows = forms.rows_for(sbr, "sbr", _info)
    shown = _rows(sbr, "sbr")["phases"]
    fill, react, decant = shown["value"]
    settle = {**shown["blank"], "name": "settle", "duration": "0.05", "settle": True}
    entries = [fill, {**react, "kla": "180", "mixed": "true"}, settle, decant, shown["blank"]]
    rows = _edit(
        rows, phases=entries, settling__kind="layered", settling__v_settle="50", settling__area=""
    )
    _, settings = forms.read_rows(sbr, "sbr", rows)
    assert settings["settling"] == {"kind": "layered", "v_settle": 50.0}
    unit = build_unit("sbr", "sbr", settings)
    assert tuple(unit.phases) == (
        SBRPhase("fill", 0.1, feed=True),
        SBRPhase("react", 0.3, kla=180.0, mixed=True),
        SBRPhase("settle", 0.05, settle=True),
        SBRPhase("decant", 0.1, decant=True, mixed=False),
    )
    assert type(unit.settling) is LayeredSettling
    assert (unit.settling.n_layers, unit.settling.v_settle, unit.settling.area) == (4, 50.0, 1.0)
    assert not isinstance(unit.settling, InterfaceSettling)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"duration": "long"}, "phases: phase 2 (react): duration: 'long' is not a number"),
        ({"duration": "nan"}, "phases: phase 2 (react): duration: 'nan' is not a finite number"),
        ({"duration": ""}, "phases: phase 2 (react): give duration"),
        ({"name": "", "duration": "0.2"}, "phases: phase 2: give name"),
    ],
)
def test_an_unreadable_phase_is_named_with_its_field(sbr, entry, message):
    rows = forms.rows_for(sbr, "sbr", _info)
    entries = [dict(e) for e in _rows(sbr, "sbr")["phases"]["value"]]
    entries[1].update(entry)
    with pytest.raises(forms.FieldError, match=f"^{re.escape(message)}$") as caught:
        forms.read_rows(sbr, "sbr", _edit(rows, phases=entries))
    assert caught.value.key == "phases"


def test_a_settling_argument_that_cannot_be_read_names_its_field(sbr):
    rows = _edit(forms.rows_for(sbr, "sbr", _info), settling__area="wide")
    with pytest.raises(
        forms.FieldError, match=r"^settling area: 'wide' is not a number$"
    ) as caught:
        forms.read_rows(sbr, "sbr", rows)
    assert caught.value.key == "settling.area"


def test_a_new_sbr_is_drawn_with_an_empty_table_and_no_settling_model(ui_catalog):
    doc = Flowsheet.from_spec(
        ui_catalog, {"format": 2, "name": "p", "units": {"s": {"kind": "sbr", "model": "asm1"}}}
    )
    rows = _rows(doc, "s")
    assert rows["phases"]["value"] == [] and rows["phases"]["label"] == "phases *"
    assert rows["settling.kind"]["value"] == "" and rows["settling.kind"]["placeholder"] == "required"
    assert not any(k.startswith("settling.") and k != "settling.kind" for k in rows)
    _, settings = forms.read_rows(doc, "s", forms.rows_for(doc, "s", _info))
    assert "phases" not in settings and "settling" not in settings


# -- the expression controller and the signal-driven modes ---------------------


@pytest.fixture
def controllers(ui_catalog):
    """A tank aerated from a signal and an expression controller publishing it."""
    units = {
        "tank": {
            "kind": "cstr",
            "model": "asm1",
            "volume": 1000.0,
            "conditions": {"T": 293.15},
            "aeration": {"kla_signal": "kla"},
        },
        "loop": {
            "kind": "expression_controller",
            "model": "asm1",
            "signal_name": "kla",
            "law": "clip(offset + Kp*([SNH] - sp), 0, 240)",
            "constants": {"Kp": 40.0, "sp": 1.0, "offset": 120.0},
        },
        "dose": {"kind": "dosing", "flow_signal": "q"},
    }
    return Flowsheet.from_spec(ui_catalog, {"format": 2, "name": "p", "units": units})


def test_a_law_is_a_text_field_and_its_constants_named_number_rows(controllers):
    rows = _rows(controllers, "loop")
    assert rows["law"]["widget"] == "text"
    assert rows["law"]["value"] == "clip(offset + Kp*([SNH] - sp), 0, 240)"
    assert rows["constants"]["widget"] == "named"
    assert rows["constants"]["value"] == [
        {"name": "Kp", "value": "40"},
        {"name": "sp", "value": "1"},
        {"name": "offset", "value": "120"},
    ]
    assert rows["signal_name"]["label"] == "signal_name *"


def test_constants_edited_in_the_form_build_that_controller(controllers):
    from aquakin.plant.catalog import build_unit

    rows = forms.rows_for(controllers, "loop", _info)
    for r in rows:
        if r["key"] == "constants":
            r["value"] = [
                {"name": "Kp", "value": "20"},
                {"name": "", "value": ""},  # an added row left blank is no entry
                {"name": "sp", "value": "2.5"},
            ]
        if r["key"] == "law":
            r["value"] = "Kp*([SNH] - sp)"
    _, settings = forms.read_rows(controllers, "loop", rows)
    assert settings["constants"] == {"Kp": 20.0, "sp": 2.5}
    unit = build_unit("expression_controller", "loop", settings)
    assert unit.constants == {"Kp": 20.0, "sp": 2.5} and unit.law == "Kp*([SNH] - sp)"


def test_constants_all_left_blank_read_back_unset(controllers):
    rows = forms.rows_for(controllers, "loop", _info)
    rows = [{**r, "value": [{"name": "", "value": ""}]} if r["key"] == "constants" else r for r in rows]
    _, settings = forms.read_rows(controllers, "loop", rows)
    assert "constants" not in settings


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([{"name": "", "value": "3"}], "constants: give the name of the value 3"),
        ([{"name": "Kp", "value": ""}], "constants: give Kp a value"),
        ([{"name": "Kp", "value": "1"}, {"name": "Kp", "value": "2"}], "Kp is listed twice"),
        ([{"name": "Kp", "value": "big"}], "'big' is not a number"),
    ],
)
def test_an_unreadable_constant_is_named(controllers, entries, message):
    rows = forms.rows_for(controllers, "loop", _info)
    rows = [{**r, "value": entries} if r["key"] == "constants" else r for r in rows]
    with pytest.raises(forms.FieldError, match=message):
        forms.read_rows(controllers, "loop", rows)


def test_the_measured_port_exists_only_with_a_model(ui_catalog, controllers):
    assert ui_catalog.ports_for("expression_controller", {"model": "asm1"}) == (["measured"], [])
    assert ui_catalog.ports_for("expression_controller", {}) == ([], [])
    assert _rows(controllers, "loop")["model"]["widget"] == "select"


def test_an_aeration_read_from_a_signal_is_its_name(controllers):
    rows = _rows(controllers, "tank")
    assert rows["aeration.mode"]["value"] == "kla_signal"
    assert "kla_signal" in rows["aeration.mode"]["items"]
    assert rows["aeration.value"]["value"] == "kla" and rows["aeration.value"]["suffix"] == ""
    rows["aeration.value"]["value"] = "kla_tank3"
    _, settings = forms.read_rows(controllers, "tank", list(rows.values()))
    assert settings["aeration"]["kla_signal"] == "kla_tank3"
    rows["aeration.value"]["value"] = " "
    with pytest.raises(forms.FieldError, match="give the name of the signal that sets the kLa"):
        forms.read_rows(controllers, "tank", list(rows.values()))


def test_switching_to_a_signal_mode_leaves_its_name_to_fill_in(bsm1):
    rows = forms.rows_for(bsm1, "tank3", _info)  # kLa 240
    next(r for r in rows if r["key"] == "aeration.mode")["value"] = "kla_signal"
    _, settings = forms.read_rows(bsm1, "tank3", rows)
    assert settings["aeration"]["kla_signal"] == "" and "kla" not in settings["aeration"]


def test_a_dose_read_from_a_signal_is_a_text_field(controllers):
    rows = _rows(controllers, "dose")
    assert rows["flow_signal"]["widget"] == "text" and rows["flow_signal"]["value"] == "q"
    _, settings = forms.read_rows(controllers, "dose", list(rows.values()))
    assert settings["flow_signal"] == "q" and "flow" not in settings
