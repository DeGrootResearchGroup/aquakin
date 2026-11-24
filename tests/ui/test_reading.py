"""Reading a spec into the document, and the form rows of the settings no other
test reaches (a species choice, a reagent, settling species on another model)."""

import pytest
import yaml

from aquakin.plant.catalog import model_info
from aquakin_ui import forms
from aquakin_ui.document import Flowsheet

from .conftest import PROJECTS

BSM2 = PROJECTS / "bsm2" / "flowsheet.yaml"


def _spec(units, connections=(), **extra):
    return {"format": 2, "name": "p", "units": units, "connections": list(connections), **extra}


TANKS = {
    "a": {"kind": "cstr", "model": "asm1", "volume": 1.0},
    "s": {"kind": "splitter", "model": "asm1", "output_port_names": ["x", "y"]},
    "b": {"kind": "cstr", "model": "asm1", "volume": 1.0},
}


def test_only_format_2_is_read(ui_catalog):
    with pytest.raises(ValueError, match=r"^spec format 1 is not 2$"):
        Flowsheet.from_spec(ui_catalog, {"format": 1, "units": {}})
    with pytest.raises(ValueError, match=r"^spec format None is not 2$"):
        Flowsheet.from_spec(ui_catalog, {"units": {}})


def test_a_connection_to_a_unit_that_is_not_there_is_refused(ui_catalog):
    with pytest.raises(ValueError, match=r"connection endpoint 'c.in' names no unit"):
        Flowsheet.from_spec(ui_catalog, _spec(TANKS, ["a.out -> c.in"]))


def test_a_connection_may_leave_out_the_only_port(ui_catalog):
    """``a -> s`` means ``a.out -> s.in``: each side has just one port."""
    doc = Flowsheet.from_spec(ui_catalog, _spec(TANKS, ["a -> s", "s.y -> b"]))
    assert doc.connections == [("a", "out", "s", "in"), ("s", "y", "b", "in")]


def test_a_unit_with_several_ports_must_name_one(ui_catalog):
    with pytest.raises(ValueError, match=r"^'s' has 2 output ports; name one$"):
        Flowsheet.from_spec(ui_catalog, _spec(TANKS, ["s -> b"]))


def test_options_are_read_and_written_back(ui_catalog):
    doc = Flowsheet.from_spec(ui_catalog, _spec(TANKS, options={"time_unit": "h"}))
    assert doc.to_spec()["options"] == {"time_unit": "h"}
    assert "options" not in Flowsheet.from_spec(ui_catalog, _spec(TANKS)).to_spec()


def test_new_units_take_the_next_free_number(ui_catalog):
    doc = Flowsheet.from_spec(ui_catalog, _spec({"cstr1": TANKS["a"], "cstr2": TANKS["b"]}))
    assert doc.add("cstr", {"x": 0, "y": 0}) == "cstr3"
    assert doc.add("mixer", {"x": 0, "y": 0}) == "mixer1"


@pytest.fixture
def bsm2(ui_catalog):
    return Flowsheet.from_spec(ui_catalog, yaml.safe_load(BSM2.read_text()), base_dir=BSM2.parent)


def _info(doc):
    return lambda name: model_info(name, doc.model_files().get(name))


def test_a_reagent_is_shown_as_its_model_label_and_composition(bsm2):
    rows = {r["key"]: r for r in forms.rows_for(bsm2, "external_carbon", _info(bsm2))}
    assert "reagent" not in rows
    model = rows["reagent.model"]
    # The project's own ASM1 (at 15 °C), a model file, is offered beside the shipped ones.
    assert model["widget"] == "select" and model["value"] == "asm1_bsm2" and model["reshape"]
    assert {"asm1_bsm2", "asm1"} <= set(model["items"]) and model["label"] == "reagent model *"
    assert rows["reagent.label"]["value"] == "external carbon"
    composition = rows["reagent.composition"]
    assert composition["widget"] == "composition"
    assert composition["value"] == [{"species": "SS", "value": "400000"}]
    species = _info(bsm2)("asm1_bsm2")["species"]
    assert composition["items"] == [s["name"] for s in species]
    assert composition["units"]["SS"] == "g_COD/m³"


def test_an_unedited_reagent_is_kept_on_apply(bsm2):
    before = bsm2.units["external_carbon"].settings["reagent"]
    rows = forms.rows_for(bsm2, "external_carbon", _info(bsm2))
    _, settings = forms.read_rows(bsm2, "external_carbon", rows)
    assert settings["reagent"] == before


def test_a_species_setting_chooses_from_the_units_model(ui_catalog):
    doc = Flowsheet.from_spec(
        ui_catalog,
        _spec({"c": {"kind": "pi_controller", "model": "asm3", "measured_species": "SNH4"}}),
    )
    rows = {r["key"]: r for r in forms.rows_for(doc, "c", model_info)}
    row = rows["measured_species"]
    assert row["widget"] == "select" and row["value"] == "SNH4"
    assert row["items"] == [s["name"] for s in model_info("asm3")["species"]]


def test_settling_species_on_another_model_are_offered_its_own(ui_catalog):
    """A clarifier on an ASM3 plant still holding ASM1's settling species is
    shown ASM3's particulates (``X`` species) instead."""
    doc = Flowsheet.from_spec(ui_catalog, _spec({"k": {"kind": "clarifier", "model": "asm3"}}))
    rows = {r["key"]: r for r in forms.rows_for(doc, "k", model_info)}
    row = rows["particulate_species"]
    assert row["widget"] == "multi"
    assert row["value"] == ["XI", "XS", "XH", "XSTO", "XA", "XSS"]
