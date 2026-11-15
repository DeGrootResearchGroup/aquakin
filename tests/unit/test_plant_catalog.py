"""The unit catalogue a front end reads (``aquakin.plant.catalog``) and the
``aquakin serve`` worker that sends it.

The catalogue's port rules are hand-declared beside settings read from the
unit classes, so the central check is that they agree with the units: for every
kind, the ports the rules give for a set of settings equal the ports of the
unit built from those settings.
"""

import io
import json

import jax.numpy as jnp
import pytest

import aquakin
import aquakin.plant as plant
from aquakin.__main__ import serve
from aquakin.plant.catalog import (
    CATEGORIES,
    SETTING_TYPES,
    catalog,
    check_unit,
    model_info,
    shipped_models,
    unit_kind,
    unit_kinds,
)

T = {"T": 293.15}

# Settings that build each kind from the front end (required values filled in).
BUILDABLE = {
    "cstr": {"model": "asm1", "volume": 1000.0, "conditions": T},
    "mbr": {"model": "asm1", "volume": 1000.0, "conditions": T},
    "ifas": {
        "model": "asm1",
        "volume": 1000.0,
        "conditions": T,
        "specific_surface_area": 500.0,
        "fill_fraction": 0.5,
        "biofilm_thickness": 1e-4,
    },
    "digester": {"model": "adm1", "volume": 3400.0, "conditions": {"T": 308.15, "T_ref": 298.15}},
    "clarifier": {"model": "asm1", "area": 1500.0, "height": 4.0, "underflow_Q": 18000.0},
    "primary_clarifier": {"model": "asm1", "volume": 900.0},
    "ideal_clarifier": {"model": "asm1", "underflow_Q": 18000.0},
    "thickener": {"model": "asm1", "target_tss_percent": 7.0},
    "mixer": {"model": "asm1"},
    "splitter": {"model": "asm1"},
    "setpoint_splitter": {"model": "asm1"},
    "threshold_splitter": {"model": "asm1", "threshold": 1000.0},
    "delay": {"model": "asm1", "tau": 0.01},
    "storage": {"model": "asm1", "volume": 100.0},
    "uv": {"model": "asm1", "volume": 10.0, "intensity": 1.0, "d10": 5.0},
    "chlorine_contact": {"model": "asm1", "volume": 10.0, "dose": 5.0, "ct_per_log": 10.0},
    "pi_controller": {
        "model": "asm1",
        "measured_species": "SO",
        "setpoint": 2.0,
        "Kp": 1.0,
        "Ti": 0.1,
        "Tt": 0.05,
        "offset": 0.0,
        "out_min": 0.0,
        "out_max": 360.0,
        "signal_name": "kla",
    },
    "influent": {"model": "asm1", "Q": 18000.0, "composition": {"SS": 69.5}},
}

# Port settings changed from their defaults, to check the rules follow them.
RENAMED = {
    "cstr": {"input_port_names": ["feed", "ras"], "output_port": "mixed_liquor"},
    "mbr": {"input_port": "in", "permeate_port": "p", "waste_port": "w"},
    "clarifier": {"input_port": "feed", "overflow_port": "eff", "underflow_port": "ras"},
    "mixer": {"input_port_names": ["a", "b", "c"]},
    "splitter": {"output_port_ratios": {"a": 0.2, "b": 0.3, "c": 0.5}},
    "setpoint_splitter": {"output_port_flows": {"was": 385.0}, "remainder_port": "ras"},
    "threshold_splitter": {"threshold_port": "over", "remainder_port": "under"},
    "primary_clarifier": {"input_port_names": ["a", "b"], "effluent_port": "e", "sludge_port": "s"},
}


def test_every_kind_is_tested_and_every_category_known():
    assert set(BUILDABLE) | {"sbr", "dosing"} == set(unit_kinds())
    assert {k.category for k in unit_kinds().values()} <= set(CATEGORIES)


@pytest.mark.parametrize("kind", sorted(BUILDABLE))
def test_port_rules_give_the_built_units_ports(kind):
    settings = BUILDABLE[kind]
    built = check_unit(kind, "u", settings)
    assert built["ok"], built
    assert unit_kind(kind).ports_for(settings) == (built["inputs"], built["outputs"])


@pytest.mark.parametrize("kind", sorted(RENAMED))
def test_port_rules_follow_renamed_ports(kind):
    settings = {**BUILDABLE[kind], **RENAMED[kind]}
    built = check_unit(kind, "u", settings)
    assert built["ok"], built
    assert unit_kind(kind).ports_for(settings) == (built["inputs"], built["outputs"])


def test_port_rules_of_kinds_built_only_from_python():
    m = aquakin.load_model("asm1")
    sbr = plant.SBRUnit(
        "s",
        m,
        [plant.SBRPhase("fill", 0.1, feed=True)],
        100.0,
        1.0,
        1.0,
        plant.InterfaceSettling(1.0, 10.0),
        conditions=T,
        input_port="in",
        output_port="decant",
    )
    assert unit_kind("sbr").ports_for({"input_port": "in", "output_port": "decant"}) == (
        sbr.input_ports,
        sbr.output_ports,
    )
    dosing = plant.DosingUnit(
        "d", plant.Reagent(m, jnp.zeros(m.n_species)), flow=1.0, input_port="a"
    )
    assert unit_kind("dosing").ports_for({"input_port": "a"}) == (
        dosing.input_ports,
        dosing.output_ports,
    )


def test_the_catalogue_covers_every_exported_unit_class():
    covered = {k.python_class.rpartition(".")[2] for k in unit_kinds().values()}
    units = {
        name
        for name in plant.__all__
        if isinstance(getattr(plant, name), type)
        and hasattr(getattr(plant, name), "input_ports")
        and hasattr(getattr(plant, name), "output_ports")
        and name not in ("StatelessUnit", "Unit")
    }
    # MBBRUnit is another name for IFASUnit, so the "ifas" kind covers it.
    assert units - covered <= {"MBBRUnit"}
    assert plant.MBBRUnit is plant.IFASUnit


def test_settings_come_from_the_constructor_and_its_docstring():
    cstr = unit_kind("cstr")
    by_name = {s.name: s for s in cstr.settings}
    assert list(by_name) == [
        "model",
        "volume",
        "input_port_names",
        "conditions",
        "aeration",
        "output_port",
    ]
    assert by_name["volume"].type == "number" and by_name["volume"].required
    assert by_name["aeration"].type == "aeration" and by_name["aeration"].optional
    assert "Tank liquid volume" in by_name["volume"].help
    assert cstr.summary.startswith("A single continuous-flow stirred tank")
    for kind in unit_kinds().values():
        assert all(s.type in SETTING_TYPES for s in kind.settings), kind.kind


def test_exactly_one_of_is_reported():
    out = check_unit("clarifier", "c", {"model": "asm1", "area": 1500.0, "height": 4.0})
    assert not out["ok"] and "exactly one of overflow_Q, underflow_Q" in out["error"]


def test_a_constructor_error_is_reported_not_raised():
    out = check_unit("cstr", "c", {"model": "asm1", "volume": 1.0})
    assert not out["ok"] and "missing required condition values" in out["error"]
    out = check_unit("sbr", "s", {"model": "asm1"})
    assert not out["ok"] and "only be set from Python" in out["error"]
    assert not check_unit("no_such_kind", "x", {})["ok"]


def test_the_catalogue_is_strict_json():
    text = json.dumps(catalog(), allow_nan=False)
    assert json.loads(text)["kinds"][0]["kind"] == "influent"
    assert unit_kind("storage").defaults()["output_flow_max"] == "inf"


def test_models_are_listed_without_the_generator_scripts():
    names = [m["name"] for m in shipped_models()]
    assert "asm1" in names and "adm1" in names
    assert not any(n.startswith("_") for n in names)
    info = model_info("asm1")
    assert info["conditions"] == [{"name": "T", "default": 293.15, "unit": ""}]
    assert info["species"][0]["name"] == "SI"


def test_serve_answers_requests_and_survives_bad_ones():
    requests = "\n".join(
        [
            json.dumps({"id": 1, "command": "ping"}),
            "not json",
            json.dumps({"id": 3, "command": "nope"}),
            json.dumps(
                {
                    "id": 4,
                    "command": "check_unit",
                    "kind": "splitter",
                    "settings": {"model": "asm1"},
                }
            ),
            json.dumps({"id": 5, "command": "model", "name": "no_such_model"}),
        ]
    )
    out = io.StringIO()
    serve(io.StringIO(requests + "\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, None, 3, 4, 5]
    assert [r["ok"] for r in replies] == [True, False, False, True, False]
    assert replies[3]["result"] == {"ok": True, "inputs": ["in"], "outputs": ["out1", "out2"]}


def test_serve_checks_a_whole_spec():
    spec = {
        "units": {
            "feed": {"kind": "influent", "model": "asm1", "Q": 100.0},
            "tank": {"kind": "cstr", "model": "asm1", "volume": 10.0, "conditions": T},
        },
        "connections": ["feed.out -> tank.in"],
    }
    broken = {"units": {"tank": {"kind": "cstr", "model": "asm1", "volume": 10.0}}}
    requests = "\n".join(
        json.dumps({"id": i, "command": "check_spec", "spec": s})
        for i, s in enumerate((spec, broken))
    )
    out = io.StringIO()
    serve(io.StringIO(requests + "\n"), out)
    good, bad = (json.loads(line)["result"] for line in out.getvalue().splitlines())
    assert good == {"ok": True, "unfed": [], "dangling": ["tank.out"], "recycles": []}
    assert not bad["ok"] and bad["error"].startswith("units.tank:")
