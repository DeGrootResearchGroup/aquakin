"""The unit catalogue a front end reads (``aquakin.plant.catalog``) and the
``aquakin serve`` worker that sends it.

The catalogue's port rules are hand-declared beside settings read from the
unit classes, so the central check is that they agree with the units: for every
kind, the ports the rules give for a set of settings equal the ports of the
unit built from those settings.
"""

import io
import json
import re
from pathlib import Path

import jax.numpy as jnp
import pytest

import aquakin
import aquakin.plant as plant
from aquakin.__main__ import serve
from aquakin.plant.catalog import (
    CATEGORIES,
    SETTING_TYPES,
    build_unit,
    catalog,
    check_unit,
    model_info,
    shipped_models,
    unit_kind,
    unit_kinds,
    unit_settings,
)

T = {"T": 293.15}

# Project models some kinds need: no shipped model declares dissolved gases yet,
# so the headspace digester is built on the plant-wide example model.
MODEL_FILES = {
    "plantwide_core": str(
        Path(__file__).parents[2] / "examples" / "plantwide" / "models" / "plantwide_core.yaml"
    )
}

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
    "headspace_digester": {
        "model": "plantwide_core",
        "volume": 3400.0,
        "conditions": {"T": 308.15, "pH": 7.1},
    },
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
    "sbr": {
        "model": "asm1",
        "conditions": T,
        "phases": [
            {"name": "fill", "duration": 0.1, "feed": True},
            {"name": "decant", "duration": 0.1, "decant": True},
        ],
        "full_volume": 100.0,
        "feed_flow": 1.0,
        "decant_flow": 1.0,
        "settling": {"kind": "interface", "v_settle": 1.0, "area": 10.0},
        "feed": "batch",
    },
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
    "sbr": {"input_port": "in", "output_port": "decant"},
}


def test_every_kind_is_tested_and_every_category_known():
    assert set(BUILDABLE) | {"dosing"} == set(unit_kinds())
    assert {k.category for k in unit_kinds().values()} <= set(CATEGORIES)


@pytest.mark.parametrize("kind", sorted(BUILDABLE))
def test_port_rules_give_the_built_units_ports(kind):
    settings = BUILDABLE[kind]
    built = check_unit(kind, "u", settings, model_files=MODEL_FILES)
    assert built["ok"], built
    assert unit_kind(kind).ports_for(settings) == (built["inputs"], built["outputs"])


@pytest.mark.parametrize("kind", sorted(RENAMED))
def test_port_rules_follow_renamed_ports(kind):
    settings = {**BUILDABLE[kind], **RENAMED[kind]}
    built = check_unit(kind, "u", settings, model_files=MODEL_FILES)
    assert built["ok"], built
    assert unit_kind(kind).ports_for(settings) == (built["inputs"], built["outputs"])


def test_port_rules_of_kinds_built_only_from_python():
    m = aquakin.load_model("asm1")
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
    # The class travels apart from the message, which a front end shows as it stands.
    assert out["type"] == "ValueError" and not out["error"].startswith("ValueError")


def test_aeration_values_are_type_checked():
    base = BUILDABLE["cstr"]
    for aeration, message in (
        ({"kla": "240"}, "'aeration.kla' must be a number"),
        ({"kla": 240.0, "temperature_correction": "false"}, "must be true or false"),
        ({"kla": 240.0, "do_sat": None}, "'aeration.do_sat' cannot be empty"),
        ({"kla": 240.0, "species": 3}, "'aeration.species' must be text"),
        ({"kla": 240.0, "stripped": 3}, "'aeration.stripped' must be a mapping or a list or text"),
        ({"kla": 240.0, "air": "dry"}, "'aeration.air' must be a mapping or a list,"),
        ({"kal": 240.0}, "'aeration.kal' is not an Aeration setting"),
    ):
        out = check_unit("cstr", "t", {**base, "aeration": aeration})
        assert not out["ok"] and message in out["error"], (aeration, out)
    assert check_unit("cstr", "t", {**base, "aeration": {"kla": 240, "controller": None}})["ok"]


def test_a_constructor_error_is_reported_not_raised():
    out = check_unit("cstr", "c", {"model": "asm1", "volume": 1.0})
    assert not out["ok"] and "missing required condition values" in out["error"]
    out = check_unit("sbr", "s", {"model": "asm1"})
    assert not out["ok"] and out["error"] == "'phases' is required"
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
    models = {"asm1": {"source": "shipped"}}
    spec = {
        "format": 2,
        "models": models,
        "units": {
            "feed": {"kind": "influent", "model": "asm1", "Q": 100.0},
            "tank": {"kind": "cstr", "model": "asm1", "volume": 10.0, "conditions": T},
        },
        "connections": ["feed.out -> tank.in"],
    }
    broken = {
        "format": 2,
        "models": models,
        "units": {"tank": {"kind": "cstr", "model": "asm1", "volume": 10.0}},
    }
    requests = "\n".join(
        json.dumps({"id": i, "command": "check_spec", "spec": s})
        for i, s in enumerate((spec, broken))
    )
    out = io.StringIO()
    serve(io.StringIO(requests + "\n"), out)
    good, bad = (json.loads(line)["result"] for line in out.getvalue().splitlines())
    assert good == {"ok": True, "unfed": [], "dangling": ["tank.out"], "recycles": []}
    assert not bad["ok"] and bad["error"].startswith("units.tank:")
    assert bad["type"] == "SpecError"


def test_a_reply_that_is_not_json_does_not_end_the_worker(monkeypatch):
    import aquakin.__main__ as cli

    handlers = cli._handlers()
    monkeypatch.setattr(cli, "_handlers", lambda: {**handlers, "nan": lambda req: float("nan")})
    out = io.StringIO()
    serve(io.StringIO('{"id": 1, "command": "nan"}\n{"id": 2, "command": "ping"}\n'), out)
    first, second = (json.loads(line) for line in out.getvalue().splitlines())
    assert (
        first == {"id": 1, "ok": False, "error": first["error"]}
        and "not valid JSON" in first["error"]
    )
    assert second["ok"] and second["id"] == 2


def test_writes_to_standard_output_while_serving_do_not_reach_the_replies():
    import subprocess
    import sys

    code = (
        "import os, aquakin.__main__ as cli\n"
        "base = cli._handlers()\n"
        "def noisy(req):\n"
        "    os.write(1, b'noise from a compiled extension\\n')\n"
        "    print('noise from python')\n"
        "    return 'ok'\n"
        "cli._handlers = lambda: {**base, 'noisy': noisy}\n"
        "cli.serve()\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        input='{"id": 1, "command": "noisy"}\n',
        capture_output=True,
        text=True,
        check=True,
    )
    assert [json.loads(line) for line in out.stdout.splitlines()] == [
        {"id": 1, "ok": True, "result": "ok"}
    ]
    assert "noise from a compiled extension" in out.stderr and "noise from python" in out.stderr


def test_nan_is_no_setting():
    out = check_unit("cstr", "t", {**BUILDABLE["cstr"], "volume": float("nan")})
    assert not out["ok"] and "'volume' must be a real number, got nan" in out["error"]
    out = check_unit("cstr", "t", {**BUILDABLE["cstr"], "conditions": {"T": float("nan")}})
    assert not out["ok"] and "'conditions.T' must be a number" in out["error"]
    assert check_unit("storage", "s", {**BUILDABLE["storage"], "output_flow_max": "inf"})["ok"]


def test_a_cached_model_file_follows_an_edit_to_its_base(tmp_path):
    from aquakin.plant.catalog import load_model_file

    (tmp_path / "base.yaml").write_text(
        "model:\n  name: base\n  extends: asm1\nparameters:\n  muH: {value: 5.0}\n"
    )
    (tmp_path / "top.yaml").write_text("model:\n  name: top\n  extends: base.yaml\n")
    first = load_model_file(tmp_path / "top.yaml")
    assert float(first.parameter_values({})[first.param_index["muH"]]) == 5.0
    (tmp_path / "base.yaml").write_text(
        "model:\n  name: base\n  extends: asm1\nparameters:\n  muH: {value: 6.0}\n"
    )
    again = load_model_file(tmp_path / "top.yaml")
    assert float(again.parameter_values({})[again.param_index["muH"]]) == 6.0


def test_settings_that_are_not_well_formed_are_refused():
    from aquakin.plant.catalog import validate_settings

    cstr, dosing, store = unit_kind("cstr"), unit_kind("dosing"), unit_kind("storage")
    with pytest.raises(ValueError, match="settings must be a mapping"):
        validate_settings(cstr, [("volume", 1.0)])
    with pytest.raises(ValueError, match="'volume' is required"):
        validate_settings(cstr, {"volume": None})
    with pytest.raises(ValueError, match="'conditions' must be a mapping"):
        validate_settings(cstr, {"conditions": [293.15]})
    with pytest.raises(ValueError, match="'conditions' key 1 must be a string"):
        validate_settings(cstr, {"conditions": {1: 293.15}})
    with pytest.raises(ValueError, match="'reagent' must be"):
        validate_settings(dosing, {"reagent": {"model": "asm1", "colour": "red"}})
    with pytest.raises(ValueError, match=r"'reagent\.composition\.SS' must be a number"):
        validate_settings(dosing, {"reagent": {"model": "asm1", "composition": {"SS": "a lot"}}})
    with pytest.raises(ValueError, match="can only be set from Python"):
        validate_settings(store, {"initial_concentrations": [1.0]})
    validate_settings(store, {"output_flow_max": "inf", "initial_concentrations": None})


def test_an_influent_series_is_read_relative_to_the_spec(tmp_path):
    from aquakin.plant import write_influent_table
    from aquakin.plant.bsm import load_bsm1_influent
    from aquakin.plant.catalog import build_unit

    m = aquakin.load_model("asm1")
    write_influent_table(load_bsm1_influent("dry", m), tmp_path / "data" / "feed.csv")
    series = build_unit(
        "influent", "feed", {"model": "asm1", "series": "data/feed.csv"}, base_dir=tmp_path
    )
    assert series.Q.shape[0] > 100
    out = check_unit(
        "influent", "feed", {"model": "asm1", "series": "data/feed.csv", "T": 288.0}, None, tmp_path
    )
    assert not out["ok"] and "set none of them" in out["error"]


def test_a_unit_that_cannot_be_written_as_settings_says_why():
    import dataclasses

    from aquakin.plant.catalog import unit_settings

    m = aquakin.load_model("asm1")
    tank = plant.CSTRUnit("t", dataclasses.replace(m), 100.0, ["in"], conditions=T)
    with pytest.raises(ValueError, match=r"is not a shipped model as aquakin\.load_model"):
        unit_settings(tank)
    store = plant.StorageTank("s", m, 10.0, initial_concentrations=m.default_concentrations())
    with pytest.raises(ValueError, match=r"unit 's': 'initial_concentrations' .* has no plain form"):
        unit_settings(store)

    class Custom(plant.CSTRUnit):
        pass

    with pytest.raises(ValueError, match="no unit kind for Custom"):
        unit_settings(Custom("c", m, 100.0, ["in"], conditions=T))


def test_docstrings_and_plain_values():
    import numpy as np

    from aquakin.plant.catalog import _docstring_parts, _plain

    class NoParameters:
        """Summary line.

        More text but no parameter list.
        """

    assert _docstring_parts(NoParameters) == ("Summary line.", {})
    assert _plain(np.float64(2.0)) == 2.0 and _plain(-np.inf) == "-inf"
    assert _plain([1, 2.5, "a"]) == [1, 2.5, "a"]
    assert _plain({"a": 1.0, "b": float("inf")}) == {"a": 1.0, "b": "inf"}
    assert _plain({"a": [1]}) is None and _plain(object()) is None
    with pytest.raises(ValueError, match="holds NaN"):
        _plain(float("nan"))


def test_a_port_setting_left_empty_gives_no_ports():
    kind = unit_kind("mixer")
    assert kind.ports_for({"model": "asm1", "input_port_names": None})[0] == []


def test_settings_read_from_classes_that_document_less():
    from aquakin.plant.catalog import Setting, _docstring_parts, _generic_help, _settings_of

    class Plain:
        """A unit that is not a dataclass.

        Parameters
        ----------
        size : float
            How big it is.

        Returns
        -------
        nothing
        """

        def __init__(self, name, size: float = 1.0, inlet_port: str = "in"):
            pass

    summary, helps = _docstring_parts(Plain)
    assert summary == "A unit that is not a dataclass." and helps == {"size": "How big it is."}
    by_name = {s.name: s for s in _settings_of(Plain)}
    assert by_name["size"].default == 1.0
    assert by_name["inlet_port"].help == _generic_help("inlet_port", "port_name")
    assert _generic_help("inlet_port", "port_name") == "Name of the inlet port."
    assert _generic_help("size", "number") == ""  # nothing generic to say
    assert isinstance(by_name["size"], Setting)


def test_cached_model_files_and_plain_setting_values(tmp_path):
    from aquakin.plant.catalog import _setting_value, load_model_file, unit_settings

    (tmp_path / "m.yaml").write_text("model:\n  name: m\n  extends: asm1\n")
    assert load_model_file(tmp_path / "m.yaml") is load_model_file(tmp_path / "m.yaml")

    m = aquakin.load_model("asm1")
    tank = plant.CSTRUnit("t", m, 100.0, ["in"], conditions=T, aeration=plant.Aeration(kla=240.0))
    # Without full, an aeration lists only the fields away from their defaults.
    assert unit_settings(tank)[1]["aeration"] == {"kla": 240.0}
    number = unit_kind("cstr").settings[1]
    with pytest.raises(ValueError, match=r"'volume' .* has no plain form"):
        _setting_value(number, object())


def test_aeration_must_be_a_mapping_and_series_may_be_absolute(tmp_path):
    from aquakin.plant import write_influent_table
    from aquakin.plant.bsm import load_bsm1_influent
    from aquakin.plant.catalog import build_unit, validate_settings

    with pytest.raises(ValueError, match="'aeration' must be"):
        validate_settings(unit_kind("cstr"), {"aeration": 240.0})
    m = aquakin.load_model("asm1")
    table = write_influent_table(load_bsm1_influent("dry", m), tmp_path / "feed.csv")
    series = build_unit("influent", "feed", {"model": "asm1", "series": str(table)})
    assert series.Q.shape[0] > 100


@pytest.mark.parametrize("value", [[1, 2], {"a": 1}, True])
def test_a_scheduled_setting_is_a_number_or_an_expression_and_nothing_else(value):
    """An influent's Q may be either; a bool is not a number here, and a list or
    mapping is neither. (``None`` is not refused -- it means the setting is
    unset, which an optional setting is allowed to be.)"""
    from aquakin.plant.catalog import unit_kind, validate_settings

    kind = unit_kind("influent")
    with pytest.raises(ValueError, match="a real number, or an expression of time"):
        validate_settings(kind, {"model": "asm1", "Q": value})


def test_a_scheduled_setting_takes_a_number_or_a_readable_expression():
    from aquakin.plant.catalog import unit_kind, validate_settings

    kind = unit_kind("influent")
    for good in (2000.0, 2000, "2000", "2000 + 500*sin(2*pi*t)"):
        validate_settings(kind, {"model": "asm1", "Q": good})
    # Each composition entry is read the same way, and names the species.
    validate_settings(
        kind, {"model": "asm1", "Q": 1.0, "composition": {"SNH": "30 + t", "SS": 60.0}}
    )
    with pytest.raises(ValueError, match=re.escape("'composition.SNH'")):
        validate_settings(kind, {"model": "asm1", "Q": 1.0, "composition": {"SNH": "wobble(t)"}})


@pytest.mark.parametrize("value", ["not a mapping", [1, 2], 3.0])
def test_a_scheduled_composition_must_be_a_mapping(value):
    from aquakin.plant.catalog import unit_kind, validate_settings

    with pytest.raises(ValueError, match="'composition' must be a mapping"):
        validate_settings(unit_kind("influent"), {"model": "asm1", "Q": 1.0, "composition": value})


@pytest.mark.parametrize("value", [[1], {"a": 1}, True, None])
def test_a_scheduled_composition_value_is_a_number_or_an_expression(value):
    """Each entry is read like Q itself, so a value that is neither names the
    species rather than only the composition."""
    from aquakin.plant.catalog import unit_kind, validate_settings

    with pytest.raises(ValueError, match=re.escape("'composition.SNH' must be")):
        validate_settings(
            unit_kind("influent"), {"model": "asm1", "Q": 1.0, "composition": {"SNH": value}}
        )


def test_a_scheduled_composition_is_keyed_by_species_name():
    from aquakin.plant.catalog import unit_kind, validate_settings

    with pytest.raises(ValueError, match="key 3 must be a string"):
        validate_settings(
            unit_kind("influent"), {"model": "asm1", "Q": 1.0, "composition": {3: 1.0}}
        )


@pytest.mark.parametrize(
    ("kind", "setting", "expected"),
    [
        ("ideal_clarifier", "underflow_Q", "scheduled_number"),
        ("ideal_clarifier", "overflow_Q", "scheduled_number"),
        ("clarifier", "underflow_Q", "scheduled_number"),
        ("primary_clarifier", "f_PS", "scheduled_number"),
        ("threshold_splitter", "threshold", "scheduled_number"),
        ("setpoint_splitter", "output_port_flows", "scheduled_port_map"),
        ("splitter", "output_port_ratios", "port_map"),
    ],
)
def test_flow_setpoints_may_be_expressions_of_time(kind, setting, expected):
    """Every flow setpoint is a number-or-expression setting; a ratio split is not."""
    assert {s.name: s.type for s in unit_kind(kind).settings}[setting] == expected
    assert expected in SETTING_TYPES


def test_a_setpoint_expression_builds_and_reads_back_as_written():
    """A unit built from an expression setting holds it as text, and writes it
    back as the text; a number is read as a float."""
    settings = {
        "model": "asm1",
        "output_port_flows": {"ras": "300 + 100*step(t - 10)", "waste": 40},
        "remainder_port": "rest",
    }
    unit = build_unit("setpoint_splitter", "split", settings)
    assert unit.output_port_flows == {"ras": "300 + 100*step(t - 10)", "waste": 40.0}
    assert unit.scheduled_setpoints() == ("ras",)
    assert unit_settings(unit)[1]["output_port_flows"] == {
        "ras": "300 + 100*step(t - 10)",
        "waste": 40.0,
    }
    clarifier = build_unit(
        "ideal_clarifier", "c", {"model": "asm1", "underflow_Q": "400 + 75*sin(2*pi*t)"}
    )
    assert unit_settings(clarifier)[1]["underflow_Q"] == "400 + 75*sin(2*pi*t)"


def test_a_scheduled_port_map_names_the_port_that_holds_a_bad_value():
    from aquakin.plant.catalog import validate_settings

    kind = unit_kind("setpoint_splitter")
    base = {"model": "asm1", "remainder_port": "rest"}
    with pytest.raises(ValueError, match=re.escape("'output_port_flows.ras'") + ".*wobble"):
        validate_settings(kind, {**base, "output_port_flows": {"ras": "wobble(t)"}})
    with pytest.raises(ValueError, match=re.escape("'output_port_flows.ras' must be")):
        validate_settings(kind, {**base, "output_port_flows": {"ras": [1]}})
    with pytest.raises(ValueError, match="key 3 must be a string \\(a port name\\)"):
        validate_settings(kind, {**base, "output_port_flows": {3: 1.0}})
    assert check_unit(
        "setpoint_splitter", "s", {**base, "output_port_flows": {"ras": "1/t"}}
    ) == {
        "ok": False,
        "error": "SetpointSplitter 's': 'ras': '1/t' is not finite at t = 0",
        "type": "ValueError",
    }


# -- an SBR's feed mode: a required choice, and settings of one mode ---------


def test_an_sbrs_feed_is_a_required_choice_and_its_mode_settings_say_when():
    """``feed`` is a choice of the three modes with no default; feed_flow applies
    with a fill pump (storage, batch) and the holding volume's settings and
    overflow port with a storage feed only."""
    settings = {s.name: s for s in unit_kind("sbr").settings}
    feed = settings["feed"]
    assert (feed.type, feed.required, feed.optional, feed.default) == ("choice", True, False, None)
    assert feed.choices == ["continuous", "storage", "batch"]
    assert feed.to_json()["choices"] == ["continuous", "storage", "batch"]
    pumped = {"setting": "feed", "values": ["storage", "batch"]}
    storage = {"setting": "feed", "values": ["storage"]}
    assert (settings["feed_flow"].required, settings["feed_flow"].when) == (False, pumped)
    for name in ("storage_volume", "storage_initial_fraction", "storage_initial_concentrations",
                 "overflow_port"):  # fmt: skip
        assert settings[name].when == storage, name
    assert settings["decant_flow"].when is None
    kind = unit_kind("sbr")
    assert kind.ports_for({"feed": "storage"}) == (["feed"], ["effluent", "overflow"])
    assert kind.ports_for({"feed": "storage", "overflow_port": "spill"})[1] == ["effluent", "spill"]
    for feed_mode in ("continuous", "batch", None):
        assert kind.ports_for({"feed": feed_mode}) == (["feed"], ["effluent"])


def test_a_storage_fed_sbrs_port_rules_give_the_built_units_ports():
    settings = {**BUILDABLE["sbr"], "feed": "storage", "storage_volume": 50.0,
                "overflow_port": "spill"}  # fmt: skip
    out = check_unit("sbr", "s", settings)
    assert out["ok"], out
    assert (out["inputs"], out["outputs"]) == (["feed"], ["effluent", "spill"])
    assert unit_kind("sbr").ports_for(settings) == (out["inputs"], out["outputs"])
    built = build_unit("sbr", "s", settings)
    assert (built.storage_volume, built.storage_initial_fraction) == (50.0, 0.5)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"feed": None}, "'feed' is required: one of 'continuous', 'storage', 'batch'"),
        ({"feed": "pump"}, "'feed' must be one of 'continuous', 'storage', 'batch', got 'pump'"),
        ({"feed": "continuous"}, "feed_flow does not apply"),
        ({"feed": "storage"}, "storage_volume must be a positive number; got None"),
        ({"storage_volume": 5.0}, "so storage_volume does not apply"),
    ],
)
def test_an_sbrs_feed_settings_are_checked(change, message):
    settings = {k: v for k, v in {**BUILDABLE["sbr"], **change}.items() if v is not None}
    out = check_unit("sbr", "s", settings)
    assert not out["ok"] and message in out["error"], out


def test_a_continuous_sbr_needs_no_feed_flow():
    settings = {k: v for k, v in BUILDABLE["sbr"].items() if k != "feed_flow"}
    out = check_unit("sbr", "s", {**settings, "feed": "continuous"})
    assert out["ok"] and out["outputs"] == ["effluent"], out
    assert build_unit("sbr", "s", {**settings, "feed": "continuous"}).feed_flow is None


# -- settings that are records: an SBR's phases and settling model -----------


def test_an_sbrs_phases_and_settling_describe_their_fields():
    """The phases are records of SBRPhase's fields; the settling model is one
    kind of two, each with its own arguments (and none of its private state)."""
    settings = {s.name: s for s in unit_kind("sbr").settings}
    phases, settling = settings["phases"], settings["settling"]
    assert (phases.type, phases.required, settling.type, settling.required) == (
        "records",
        True,
        "tagged",
        True,
    )
    fields = {f["name"]: f for f in phases.fields}
    assert list(fields) == ["name", "duration", "feed", "decant", "kla", "settle", "mixed"]
    assert fields["duration"]["type"] == "number" and fields["duration"]["required"]
    assert fields["mixed"]["type"] == "boolean" and fields["mixed"]["optional"]
    assert fields["kla"]["default"] == 0.0 and "(1/d)" in fields["kla"]["help"]
    kinds = {k: [f["name"] for f in v] for k, v in settling.fields.items()}
    assert kinds == {
        "interface": ["v_settle", "area", "remix_rate"],
        "layered": ["n_layers", "v_settle", "area", "remix_rate"],
    }
    layers = settling.fields["layered"][0]
    assert (layers["type"], layers["default"]) == ("integer", 4)
    assert set(SETTING_TYPES) >= {"records", "tagged"}
    assert json.loads(json.dumps(catalog(), allow_nan=False))  # still strict JSON


def test_an_sbr_built_from_plain_settings_has_those_phases_and_settling():
    from aquakin.plant.catalog import build_unit, unit_settings

    sbr = build_unit("sbr", "s", BUILDABLE["sbr"])
    assert tuple(sbr.phases) == (
        plant.SBRPhase("fill", 0.1, feed=True),
        plant.SBRPhase("decant", 0.1, decant=True),
    )
    assert type(sbr.settling) is plant.InterfaceSettling
    assert (sbr.settling.v_settle, sbr.settling.area, sbr.settling.remix_rate) == (
        1.0,
        10.0,
        1000.0,
    )
    _, written = unit_settings(sbr)
    assert written["phases"][1] == {
        "name": "decant",
        "duration": 0.1,
        "feed": False,
        "decant": True,
        "kla": 0.0,
        "settle": False,
        "mixed": None,
    }
    assert written["settling"] == {
        "kind": "interface",
        "v_settle": 1.0,
        "area": 10.0,
        "remix_rate": 1000.0,
    }


def test_a_settling_model_of_no_kind_cannot_be_written():
    """A settling model the catalogue does not list (one written in Python) is
    refused by name rather than written as something else."""
    import dataclasses

    from aquakin.plant.catalog import build_unit, unit_settings

    @dataclasses.dataclass
    class Mine(plant.InterfaceSettling):
        pass

    sbr = build_unit("sbr", "s", BUILDABLE["sbr"])
    own = dataclasses.replace(sbr, settling=Mine(v_settle=1.0, area=10.0))
    with pytest.raises(ValueError, match=r"^unit 's': 'settling' Mine cannot be written in a spec$"):
        unit_settings(own)


def test_a_phase_with_no_plain_form_cannot_be_written():
    import dataclasses

    from aquakin.plant.catalog import build_unit, unit_settings

    sbr = build_unit("sbr", "s", BUILDABLE["sbr"])
    odd = dataclasses.replace(sbr.phases[0], duration=jnp.asarray([0.1, 0.2]))
    object.__setattr__(sbr, "phases", (odd, sbr.phases[1]))
    with pytest.raises(
        ValueError, match=r"^unit 's': 'phases' entry 1 setting 'duration' has no plain form$"
    ):
        unit_settings(sbr)
