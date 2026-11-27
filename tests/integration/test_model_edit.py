"""Editing a model's reactions and parameters (``aquakin.plant.model_edit``) and
the worker commands behind the editor's model panel.

What a user types is checked against the model it belongs to, a whole set of
edits is loaded and compiled as the file it would be written as, and a save
writes the model and the flowsheet together: a project model in place, a
shipped model as a new project model that extends it.
"""

import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

import aquakin
from aquakin.__main__ import serve
from aquakin.core.lexer import tokenize
from aquakin.core.parser import ParseError, parse_rate_expression
from aquakin.plant import Plant
from aquakin.plant.model_edit import (
    ModelEditError,
    check_expression,
    check_model_edits,
    model_document,
    save_model_edits,
)
from aquakin.plant.spec import SpecError, load_spec_file, write_spec

BSM1 = Path(__file__).parents[2] / "examples" / "projects" / "bsm1" / "flowsheet.yaml"


def _serve(*requests) -> list[dict]:
    out = io.StringIO()
    serve(io.StringIO("".join(json.dumps(r) + "\n" for r in requests)), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


@pytest.fixture
def decay_project(tmp_path, make_cstr_plant):
    """A project of the one-tank plant: ``flowsheet.yaml`` and the A -> B model
    copied into ``models/simple_decay.yaml`` (its rate ``k * [A]``, ``k`` a
    parameter of the reaction, 0.1)."""
    path = tmp_path / "decay" / "flowsheet.yaml"
    path.parent.mkdir()
    make_cstr_plant().to_spec(path)
    return path


@pytest.fixture
def bsm1_project(tmp_path):
    path = tmp_path / "bsm1" / "flowsheet.yaml"
    path.parent.mkdir()
    path.write_bytes(BSM1.read_bytes())
    return path


# ---------------------------------------------------------------------------
# Parse errors carry their position
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, position, message",
    [
        ("k * [A", 6, "Expected \\('RBRACK',\\) at position 6"),
        ("k * * 2", 4, "Unexpected token '\\*' at position 4"),
        ("k $ 2", 2, "Unexpected character '\\$' at position 2"),
        ("k + 1..2", 4, "'1..2' at position 4 is not a number"),
        ("[ 1]", 2, "Expected species name after '\\['"),
        ("2 * foo([A])", 4, "Unknown function 'foo'"),
        ("1 + max(k)", 4, "max\\(\\) takes 2 or more"),
        ("k k", 2, "Unexpected token 'k' at position 2"),
    ],
)
def test_a_parse_error_carries_the_position_its_message_names(text, position, message):
    with pytest.raises(ParseError, match=message) as caught:
        parse_rate_expression(text)
    assert caught.value.position == position


def test_the_shared_tokenizer_locates_its_errors_for_any_language():
    with pytest.raises(KeyError) as caught:
        tokenize("a ? b", error=KeyError)
    assert caught.value.position == 2


# ---------------------------------------------------------------------------
# One expression
# ---------------------------------------------------------------------------


def test_a_valid_rate_lists_what_it_references():
    found = check_expression("muH * [SS] / (KS + [SS]) * {T} * rho_hetero_decay", model="asm1")
    assert found == {
        "ok": True,
        "error": None,
        "position": None,
        "references": {
            "species": ["SS"],
            "parameters": ["KS", "muH"],
            "conditions": ["T"],
            "expressions": ["rho_hetero_decay"],
        },
        "unknown": {"species": [], "parameters": [], "conditions": []},
    }


def test_a_syntax_error_is_reported_with_its_position():
    found = check_expression("muH * [SS] / (KS + [SS]", model="asm1")
    assert not found["ok"]
    assert found["position"] == 23
    assert "at position 23" in found["error"]


@pytest.mark.parametrize(
    "text, kind, error, position, unknown",
    [
        ("k_new * [SS]", "rate", "'k_new' is not a parameter or named expression of asm1", 0,
         {"species": [], "parameters": ["k_new"], "conditions": []}),
        ("bH * [SZ]", "rate", "'SZ' is not a species of asm1", 5,
         {"species": ["SZ"], "parameters": [], "conditions": []}),
        ("bH * {Temp}", "rate", "'Temp' is not a condition of asm1", 5,
         {"species": [], "parameters": [], "conditions": ["Temp"]}),
        ("-1 / Y_HH", "stoichiometry", "'Y_HH' is not a parameter of asm1", 5,
         {"species": [], "parameters": ["Y_HH"], "conditions": []}),
    ],
)
def test_a_name_the_model_does_not_define_is_reported_where_it_stands(
    text, kind, error, position, unknown
):
    found = check_expression(text, model="asm1", kind=kind)
    assert not found["ok"]
    assert found["error"].startswith(error)
    assert found["position"] == position
    assert found["unknown"] == unknown


def test_a_parameter_the_edit_adds_is_known():
    assert check_expression("k_new * [SS]", model="asm1", parameters=["k_new"])["ok"]


def test_a_reaction_may_use_its_own_local_parameters_only(decay_project):
    path = decay_project.parent / "models" / "simple_decay.yaml"
    model = "simple_decay"
    assert check_expression("2 * k * [A]", model=model, path=path, reaction="A_to_B")["ok"]
    elsewhere = check_expression("2 * k * [A]", model=model, path=path)
    assert elsewhere["unknown"]["parameters"] == ["k"]


@pytest.mark.parametrize("text", ["-1.5", "auto", "?", "1e-3", "-(1 - Y_H) / Y_H"])
def test_a_coefficient_may_be_a_number_auto_or_an_expression_in_parameters(text):
    assert check_expression(text, model="asm1", kind="stoichiometry")["ok"]


@pytest.mark.parametrize(
    "text, error",
    [
        ("[SS] / Y_H", "uses an unsupported expression element \\(SpeciesNode\\)"),
        ("rho_hetero_decay", "'rho_hetero_decay' is a named expression"),
        ("", "write a coefficient"),
    ],
)
def test_a_coefficient_may_not_use_the_state_or_named_expressions(text, error):
    found = check_expression(text, model="asm1", kind="stoichiometry")
    assert not found["ok"]
    assert re.search(error, found["error"])


def test_an_empty_rate_and_an_unknown_kind_are_refused():
    assert check_expression("  ", model="asm1")["error"] == "write an expression"
    with pytest.raises(ValueError, match="kind must be one of \\['rate', 'stoichiometry'\\]"):
        check_expression("k", model="asm1", kind="speciation")


def test_conditions_a_model_derives_are_known():
    """A speciation block makes pH a condition rates may use."""
    assert "pH" in model_document("adm1")["conditions"]
    assert check_expression("k_dis * [X_c] * {pH}", model="adm1")["references"]["conditions"] == [
        "pH"
    ]


# ---------------------------------------------------------------------------
# Reading a model
# ---------------------------------------------------------------------------


def test_the_document_holds_each_reaction_and_parameter_as_text():
    doc = model_document("asm1")
    assert (doc["source"], doc["path"], doc["sha256"], doc["extends"]) == (
        "shipped",
        None,
        None,
        None,
    )
    decay = next(r for r in doc["reactions"] if r["name"] == "decay_heterotrophs")
    assert decay["rate"] == "rho_hetero_decay"
    assert decay["stoichiometry"][:2] == [
        {"species": "XB_H", "value": "-1.0"},
        {"species": "XS", "value": "1 - f_P"},
    ]
    muH = next(p for p in doc["parameters"] if p["name"] == "muH")
    assert muH == {"name": "muH", "reaction": None, "value": "6.0", "units": "1/d",
                   "description": ""}
    assert doc["species"][:2] == ["SI", "SS"]


def test_a_project_model_is_described_with_its_file_and_base(bsm1_project, decay_project):
    path = decay_project.parent / "models" / "simple_decay.yaml"
    doc = model_document("simple_decay", path)
    assert doc["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert doc["parameters"] == [
        {"name": "k", "reaction": "A_to_B", "value": "0.1", "units": "s-1", "description": ""}
    ]
    bsm2 = Path(__file__).parents[2] / "examples" / "projects" / "bsm2" / "models"
    derived = model_document("asm1_bsm2", bsm2 / "asm1_bsm2.yaml")
    assert derived["extends"] == "asm1"
    assert len(derived["reactions"]) == 8  # the base's, merged in


def test_a_model_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="Built-in model 'asm9' not found"):
        model_document("asm9")
    with pytest.raises(SpecError, match=r"model file .*none\.yaml does not exist"):
        model_document("x", tmp_path / "none.yaml")


# ---------------------------------------------------------------------------
# A set of edits
# ---------------------------------------------------------------------------


def test_valid_edits_pass():
    edits = {
        "reactions": {"decay_heterotrophs": {"rate": "2 * bH * [XB_H]",
                                             "stoichiometry": {"XP": "f_P * 1.0"}}},
        "parameters": {"bH": {"value": "0.5", "units": "1/d", "description": "decay"}},
    }
    assert check_model_edits("asm1", None, edits) == {"ok": True, "errors": []}


@pytest.mark.parametrize(
    "edits, field, message, position",
    [
        ({"reactions": {"decay_heterotrophs": {"rate": "bH * ([XB_H]"}}},
         "reactions.decay_heterotrophs.rate", "Expected \\('RPAREN',\\)", 12),
        ({"reactions": {"decay_heterotrophs": {"rate": "bH2 * [XB_H]"}}},
         "reactions.decay_heterotrophs.rate", "'bH2' is not a parameter .* Did you mean: bH", 0),
        ({"reactions": {"decay_heterotrophs": {"stoichiometry": {"XP": "[XP]"}}}},
         "reactions.decay_heterotrophs.stoichiometry.XP", "unsupported expression element", None),
        ({"reactions": {"decay": {"rate": "bH"}}},
         "reactions.decay", "asm1 has no reaction 'decay'", None),
        ({"reactions": {"decay_heterotrophs": {"parameters": {"k": {"value": "1"}}}}},
         "reactions.decay_heterotrophs.parameters.k", "has no local parameter 'k'", None),
        ({"reactions": {"decay_heterotrophs": {"reference": "x"}}},
         "reactions.decay_heterotrophs", "'reference' is not a field an edit may set", None),
        ({"parameters": {"bH": {"value": "fast"}}},
         "parameters.bH.value", "'fast' is not a number", None),
        ({"parameters": {"bH": {"value": "nan"}}},
         "parameters.bH.value", "'nan' is not a number", None),
        ({"parameters": {"bH": {"bounds": [0, 1]}}},
         "parameters.bH", "'bounds' is not a field an edit may set", None),
        ({"parameters": {"bH": {"value": "100"}}},
         "parameters.bH", "parameter value 100.0 is outside bounds", None),
        ({"parameters": {"k new": {"value": "1"}}},
         "parameters.k new", "'k new' cannot be a parameter's name", None),
        ({"parameters": {"k_new": {"units": "1/d"}}},
         "parameters.k_new.value", "a new parameter needs a value", None),
        ({"species": {"SS": {}}}, "species", "only reactions and parameters can be edited", None),
        ({"parameters": {"rho_hetero_decay": {"value": "1"}}},
         "model", "Names appear in both model parameters and expressions", None),
    ],
)
def test_each_problem_with_an_edit_is_reported_by_field(edits, field, message, position):
    found = check_model_edits("asm1", None, edits)
    assert not found["ok"]
    (error,) = [e for e in found["errors"] if e["field"] == field]
    assert re.search(message, error["error"]), error
    assert error["position"] == position


def test_a_compile_error_is_reported_against_the_model():
    """A coefficient the parser takes but the compiler refuses (``auto`` with
    nothing to solve it from) is the model's error."""
    found = check_model_edits(
        "asm1", None, {"reactions": {"decay_heterotrophs": {"stoichiometry": {"XP": "auto"}}}}
    )
    assert found["errors"] == [{
        "field": "model",
        "error": "reaction 'decay_heterotrophs' has an 'auto' stoichiometric coefficient "
        "(['XP']) but declares no quantities to conserve it from: add a per-reaction "
        "`conserved_for: [COD, ...]` or a model-level `conserved_for:`.",
        "position": None,
    }]


def test_a_reaction_the_schema_refuses_is_named_by_its_name_not_its_index():
    zero = dict.fromkeys(["XB_A", "XS", "XP", "XND"], "0")
    found = check_model_edits("asm1", None, {"reactions": {"decay_autotrophs": {"stoichiometry": zero}}})
    assert found["errors"] == [{
        "field": "reactions.decay_autotrophs",
        "error": "Reaction 'decay_autotrophs' has all-zero stoichiometric coefficients; this "
        "reaction would contribute nothing to dC/dt.",
        "position": None,
    }]


@pytest.mark.parametrize(
    "text, message",
    [("- a\n", "Top-level YAML in x must be a mapping, got list"),
     ("a: [\n", "Failed to parse YAML from x")],
)
def test_resolving_a_model_that_is_not_a_mapping_is_refused(text, message):
    from aquakin.schema.loader import resolve_model_data

    with pytest.raises(ValueError, match=re.escape(message)):
        resolve_model_data(text, "x")


def test_a_new_parameter_used_by_an_edited_rate_makes_a_valid_model():
    edits = {
        "reactions": {"decay_heterotrophs": {"rate": "k_new * [XB_H]"}},
        "parameters": {"k_new": {"value": "0.3", "units": "1/d"}},
    }
    assert check_model_edits("asm1", None, edits)["ok"]


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def test_a_shipped_model_is_saved_as_a_project_model_that_extends_it(bsm1_project):
    spec = load_spec_file(bsm1_project)
    before = bsm1_project.read_bytes()
    edits = {
        "reactions": {"decay_heterotrophs": {"rate": "2 * bH * [XB_H]"}},
        "parameters": {"bA": {"value": "0.06"}, "k2": {"value": "1.5", "units": "1/d"}},
    }
    planned = save_model_edits(
        spec, bsm1_project, model="asm1", edits=edits, name="asm1_bsm1", dry_run=True
    )
    target = bsm1_project.parent / "models" / "asm1_bsm1.yaml"
    units = [u for u, e in spec["units"].items() if e.get("model") == "asm1"]
    assert planned == {"model": "asm1_bsm1", "model_file": str(target), "units": units,
                       "spec": None}
    assert not target.exists() and bsm1_project.read_bytes() == before

    saved = save_model_edits(spec, bsm1_project, model="asm1", edits=edits, name="asm1_bsm1")
    assert yaml.safe_load(target.read_text()) == {
        "model": {"name": "asm1_bsm1", "extends": "asm1"},
        "parameters": {"bA": {"value": 0.06}, "k2": {"value": 1.5, "units": "1/d"}},
        "reactions": [{"name": "decay_heterotrophs", "rate": "2 * bH * [XB_H]"}],
    }
    written = load_spec_file(bsm1_project)
    assert written == saved["spec"]
    assert list(written["models"]) == ["asm1_bsm1"]
    assert written["models"]["asm1_bsm1"]["source"] == "models/asm1_bsm1.yaml"
    assert all(written["units"][u]["model"] == "asm1_bsm1" for u in units)
    assert len(units) == 10

    edited = aquakin.load_model_from_file(target)
    assert float(edited.default_parameters()[edited.param_index["k2"]]) == 1.5
    i = edited.reaction_names.index("decay_heterotrophs")
    assert edited.rate_asts[i].param_names() == {"bH"}
    assert float(edited.default_parameters()[edited.param_index["bA"]]) == 0.06


def test_a_reagent_written_in_the_shipped_model_switches_too():
    from aquakin.plant.model_edit import _switched

    spec = {
        "models": {"asm1": {"source": "shipped"}, "adm1": {"source": "shipped"}},
        "units": {
            "dose": {"kind": "dosing", "model": "adm1", "reagent": {"model": "asm1"}},
            "dig": {"kind": "digester", "model": "adm1"},
        },
    }
    out, used = _switched(spec, "asm1", "asm1_x")
    assert used == ["dose"]
    assert out["units"]["dose"]["reagent"]["model"] == "asm1_x"
    assert out["units"]["dose"]["model"] == "adm1"
    assert list(out["models"]) == ["adm1"]


def test_a_project_model_is_edited_in_place_and_a_solve_follows_it(decay_project):
    """Doubling the one rate of the A -> B model doubles its rate constant: the
    tank's steady state ``A* = (Q/V) / (Q/V + k)`` goes from 0.1/0.2 to 0.1/0.3."""
    model_file = decay_project.parent / "models" / "simple_decay.yaml"
    spec = load_spec_file(decay_project)
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    edits = {"reactions": {"A_to_B": {"rate": "2 * k * [A]",
                                      "parameters": {"k": {"description": "first order"}}}}}
    out = save_model_edits(spec, decay_project, model="simple_decay", edits=edits, sha256=digest)
    assert out["model"] == "simple_decay" and out["units"] == ["feed", "tank"]
    data = yaml.safe_load(model_file.read_text())
    (reaction,) = data["reactions"]
    assert reaction["rate"] == "2 * k * [A]"
    assert reaction["stoichiometry"] == {"A": -1, "B": 1}  # kept as it was
    assert reaction["parameters"]["k"] == {
        "value": 0.1, "units": "s-1", "bounds": [1.0e-3, 10.0], "description": "first order"
    }
    # The flowsheet records the edited model's content: it reads back without a warning.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plant = Plant.from_spec(decay_project)
    steady = plant.steady_state()
    np.testing.assert_allclose(float(steady.state[0]), 0.1 / 0.3, rtol=1e-8)


def test_an_extends_file_gets_only_the_changed_fields_as_overrides(tmp_path):
    (tmp_path / "models").mkdir()
    model_file = tmp_path / "models" / "asm1_x.yaml"
    model_file.write_text(
        "model: {name: asm1_x, extends: asm1}\nparameters:\n  muH: {value: 4.0}\n"
    )
    spec = {"format": 2, "name": "x", "models": {"asm1_x": {"source": "models/asm1_x.yaml"}},
            "units": {}, "connections": []}
    flowsheet = tmp_path / "flowsheet.yaml"
    save_model_edits(
        spec, flowsheet, model="asm1_x",
        edits={"parameters": {"muH": {"units": "1/day"}, "KS": {"value": "15"}},
               "reactions": {"decay_heterotrophs": {"stoichiometry": {"XP": "0.1"}}}},
    )
    assert yaml.safe_load(model_file.read_text()) == {
        "model": {"name": "asm1_x", "extends": "asm1"},
        "parameters": {"muH": {"value": 4.0, "units": "1/day"}, "KS": {"value": 15.0}},
        "reactions": [{"name": "decay_heterotrophs", "stoichiometry": {"XP": 0.1}}],
    }
    model = aquakin.load_model_from_file(model_file)
    i, j = model.reaction_names.index("decay_heterotrophs"), model.species_index["XS"]
    # The override kept the rest of the base reaction's stoichiometry.
    assert np.asarray(model.compute_stoich(model.default_parameters()))[i, j] == pytest.approx(
        1 - 0.08, rel=1e-12
    )
    assert load_spec_file(flowsheet)["models"]["asm1_x"]["source"] == "models/asm1_x.yaml"


@pytest.mark.parametrize(
    "name, existing, message",
    [
        ("asm3", None, "'asm3' is the name of a shipped model"),
        ("bad name", None, "model name 'bad name' must start with a letter"),
        (None, None, "give the model made from the shipped 'asm1' a name"),
        ("asm1_bsm1", "other: 1\n", "models/asm1_bsm1.yaml already exists with different"),
    ],
)
def test_a_name_that_clashes_is_refused_and_nothing_is_written(
    bsm1_project, name, existing, message
):
    target = bsm1_project.parent / "models" / "asm1_bsm1.yaml"
    if existing is not None:
        target.parent.mkdir()
        target.write_text(existing)
    before = bsm1_project.read_bytes()
    with pytest.raises(SpecError, match=message):
        save_model_edits(
            load_spec_file(bsm1_project), bsm1_project, model="asm1",
            edits={"parameters": {"bA": {"value": "0.06"}}}, name=name,
        )
    assert bsm1_project.read_bytes() == before
    assert (target.read_text() if target.exists() else None) == existing


def test_a_name_the_flowsheet_already_gives_another_model_is_refused(bsm1_project):
    spec = load_spec_file(bsm1_project)
    spec["models"]["mine"] = {"source": "elsewhere/mine.yaml"}
    with pytest.raises(SpecError, match="already has a model named 'mine'"):
        save_model_edits(
            spec, bsm1_project, model="asm1", name="mine",
            edits={"parameters": {"bA": {"value": "0.06"}}}, dry_run=True,
        )


def test_saving_the_same_edit_again_under_the_same_name_is_allowed(bsm1_project):
    """The file already holds exactly what would be written: no clash."""
    edits = {"parameters": {"bA": {"value": "0.06"}}}
    spec = load_spec_file(bsm1_project)
    save_model_edits(spec, bsm1_project, model="asm1", edits=edits, name="asm1_bsm1")
    again = save_model_edits(
        spec, bsm1_project, model="asm1", edits=edits, name="asm1_bsm1", dry_run=True
    )
    assert again["model"] == "asm1_bsm1"


def test_an_invalid_edit_writes_nothing(bsm1_project):
    before = bsm1_project.read_bytes()
    with pytest.raises(ModelEditError, match=re.escape("reactions.decay_heterotrophs.rate: 'bH2'")) as caught:
        save_model_edits(
            load_spec_file(bsm1_project), bsm1_project, model="asm1", name="asm1_bsm1",
            edits={"reactions": {"decay_heterotrophs": {"rate": "bH2 * [XB_H]"}}},
        )
    assert caught.value.errors[0]["field"] == "reactions.decay_heterotrophs.rate"
    assert bsm1_project.read_bytes() == before
    assert not (bsm1_project.parent / "models").exists()


def test_a_project_model_changed_on_disk_or_outside_the_project_is_refused(
    decay_project, tmp_path_factory
):
    spec = load_spec_file(decay_project)
    edits = {"reactions": {"A_to_B": {"rate": "2 * k * [A]"}}}
    with pytest.raises(SpecError, match="has changed on disk since it was opened for editing"):
        save_model_edits(spec, decay_project, model="simple_decay", edits=edits, sha256="0" * 64)
    # As the message says: open the model again, and its edits save.
    path = decay_project.parent / "models" / "simple_decay.yaml"
    again = model_document("simple_decay", path)["sha256"]
    save_model_edits(spec, decay_project, model="simple_decay", edits=edits, sha256=again)
    assert yaml.safe_load(path.read_text())["reactions"][0]["rate"] == "2 * k * [A]"

    outside = tmp_path_factory.mktemp("elsewhere") / "simple_decay.yaml"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    spec = load_spec_file(decay_project)
    spec["models"]["simple_decay"] = {"source": str(outside)}
    edits = {"reactions": {"A_to_B": {"rate": "3 * k * [A]"}}}
    with pytest.raises(SpecError, match="outside the flowsheet's folder; save the flowsheet"):
        save_model_edits(spec, decay_project, model="simple_decay", edits=edits)
    # As the message says: saving the flowsheet copies the model in, then it saves.
    written = write_spec(spec, decay_project)
    save_model_edits(written, decay_project, model="simple_decay", edits=edits)
    assert yaml.safe_load(path.read_text())["reactions"][0]["rate"] == "3 * k * [A]"
    assert yaml.safe_load(outside.read_text())["reactions"][0]["rate"] == "2 * k * [A]"


def test_a_flowsheet_that_cannot_be_written_puts_the_model_file_back(decay_project, bsm1_project):
    model_file = decay_project.parent / "models" / "simple_decay.yaml"
    before = model_file.read_bytes()
    spec = load_spec_file(decay_project)
    spec["units"]["tank"]["volume"] = "large"
    with pytest.raises(SpecError, match=r"units\.tank"):
        save_model_edits(
            spec, decay_project, model="simple_decay",
            edits={"reactions": {"A_to_B": {"rate": "2 * k * [A]"}}},
        )
    assert model_file.read_bytes() == before
    spec = load_spec_file(bsm1_project)
    spec["units"]["tank1"]["volume"] = "large"
    with pytest.raises(SpecError, match=r"units\.tank1"):
        save_model_edits(
            spec, bsm1_project, model="asm1", name="asm1_bsm1",
            edits={"parameters": {"bA": {"value": "0.06"}}},
        )
    assert not (bsm1_project.parent / "models" / "asm1_bsm1.yaml").exists()


# ---------------------------------------------------------------------------
# Through the worker
# ---------------------------------------------------------------------------


def test_the_worker_answers_the_model_panel_commands(bsm1_project):
    edits = {"reactions": {"decay_heterotrophs": {"rate": "2 * bH * [XB_H]"}}}
    doc, expr, checked, planned, refused = _serve(
        {"id": 1, "command": "model_document", "name": "asm1"},
        {"id": 2, "command": "check_expression", "expression": "bH * [XB_H",
         "model": "asm1", "reaction": "decay_heterotrophs"},
        {"id": 3, "command": "check_model_edits", "model": "asm1", "edits": edits},
        {"id": 4, "command": "save_model_edits", "spec": load_spec_file(bsm1_project),
         "path": str(bsm1_project), "model": "asm1", "edits": edits, "name": "asm1_bsm1",
         "dry_run": True},
        {"id": 5, "command": "save_model_edits", "spec": load_spec_file(bsm1_project),
         "path": str(bsm1_project), "model": "asm1", "edits": edits, "name": "asm3"},
    )
    assert doc["ok"] and doc["result"]["name"] == "asm1"
    assert expr["result"]["position"] == 10 and not expr["result"]["ok"]
    assert checked["result"] == {"ok": True, "errors": []}
    assert planned["result"]["model_file"] == str(bsm1_project.parent / "models" / "asm1_bsm1.yaml")
    assert refused == {"id": 5, "ok": False, "type": "SpecError",
                       "error": "'asm3' is the name of a shipped model; choose another name"}
