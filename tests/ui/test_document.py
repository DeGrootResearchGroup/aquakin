"""The flowsheet document behind the canvas (no page, no worker)."""

import pytest
import yaml

from aquakin.plant import Plant
from aquakin_ui.document import Flowsheet, edge_id

from .conftest import PROJECTS


@pytest.fixture
def bsm1(ui_catalog):
    path = PROJECTS / "bsm1" / "flowsheet.yaml"
    return Flowsheet.from_spec(ui_catalog, yaml.safe_load(path.read_text()), base_dir=path.parent)


def test_a_spec_opened_and_saved_builds_the_same_plant(bsm1):
    original = Plant.from_spec(PROJECTS / "bsm1" / "flowsheet.yaml")
    saved = Plant.from_spec(bsm1.to_spec(), base_dir=PROJECTS / "bsm1")
    assert saved.to_spec() == original.to_spec()
    assert saved.labels["tank3"] == "Aerobic zone 1"


def test_a_spec_without_a_layout_is_laid_out_left_to_right(bsm1):
    line = ("influent", "inlet_mix", "tank1", "tank2", "tank5")
    xs = [bsm1.units[u].position["x"] for u in line]
    assert xs == sorted(xs) and len(set(xs)) == len(xs)
    # A long line wraps into a second band below the first.
    assert bsm1.units["clarifier"].position["y"] > bsm1.units["tank1"].position["y"]
    spots = {(u.position["x"], u.position["y"]) for u in bsm1.units.values()}
    assert len(spots) == len(bsm1.units)


def test_ports_follow_settings_and_dropped_connections_are_reported(bsm1):
    assert bsm1.ports("inlet_mix") == (["fresh", "internal_recycle", "ras"], ["out"])
    settings = {**bsm1.units["inlet_mix"].settings, "input_port_names": ["fresh", "ras"]}
    dropped = bsm1.apply("inlet_mix", settings, "Front mixer")
    assert dropped == ["tank5_split.internal_recycle -> inlet_mix.internal_recycle"]
    assert bsm1.units["inlet_mix"].label == "Front mixer"
    unfed, dangling = bsm1.port_status()
    assert unfed["inlet_mix"] == [] and "internal_recycle" in dangling["tank5_split"]


def test_wiring_rules(bsm1):
    uid = bsm1.add("cstr", {"x": 0, "y": 400}, {"model": "asm1"})
    assert uid == "cstr1"
    assert bsm1.connect("tank1", "out", "tank2", "inlet") == "tank2.inlet is already fed"
    assert "already feeds a unit" in bsm1.connect("tank1", "out", uid, "in")
    assert "do not exist" in bsm1.connect("tank1", "nope", uid, "in")
    assert bsm1.connect(uid, "out", uid, "in") == "a unit cannot feed itself"
    bsm1.remove(edges=["tank1.out->tank2.inlet"])
    assert bsm1.connect("tank1", "out", uid, "in") is None


def test_removing_a_unit_drops_its_connections_streams_and_endpoints(bsm1):
    bsm1.remove(units=["clarifier"])
    assert all("clarifier" not in edge_id(c) for c in bsm1.connections)
    assert "effluent" not in bsm1.meta["streams"] and bsm1.meta["effluent_endpoint"] is None


def test_recycles_are_aquakins_when_known(bsm1):
    recycles = Plant.from_spec(PROJECTS / "bsm1" / "flowsheet.yaml").check().recycles
    back = bsm1.back_edges(recycles)
    assert {f"{c[0]}.{c[1]}" for c in back} == set(recycles)
    assert back == bsm1.back_edges()  # the depth-first walk agrees on BSM1


def test_project_models_are_listed_first(ui_catalog):
    path = PROJECTS / "bsm2" / "flowsheet.yaml"
    doc = Flowsheet.from_spec(ui_catalog, yaml.safe_load(path.read_text()), base_dir=path.parent)
    assert doc.model_names()[0] == "asm1_bsm2"
    assert doc.model_files()["asm1_bsm2"].endswith("models/asm1_bsm2.yaml")
    assert (
        Plant.from_spec(doc.to_spec(), base_dir=path.parent).to_spec()
        == Plant.from_spec(path).to_spec()
    )


def test_a_unit_of_an_unknown_kind_is_refused_on_reading(ui_catalog):
    spec = {"format": 2, "units": {"t": {"kind": "cstrr"}}, "connections": []}
    with pytest.raises(ValueError, match="unit 't' is of kind 'cstrr'"):
        Flowsheet.from_spec(ui_catalog, spec)
