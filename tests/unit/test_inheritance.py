"""Declarative YAML model inheritance: model.extends + add/modify/remove (#246)."""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.schema.inheritance import apply_remove, merge_spec, pop_inheritance_keys


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _derived(tmp_path, body):
    return _write(tmp_path, "derived.yaml", body)


# --- end-to-end: a derived model extending the shipped asm1 ----------------


def test_extends_inherits_the_whole_base(tmp_path):
    """A derived model that only renames inherits every species, reaction and
    parameter of its base."""
    net = aquakin.load_model_from_file(
        _derived(
            tmp_path,
            """
model:
  name: asm1_renamed
  extends: asm1
""",
        )
    )
    base = aquakin.load_model("asm1")
    assert net.name == "asm1_renamed"
    assert net.species == base.species
    assert net.reaction_names == base.reaction_names
    assert net.parameters == base.parameters
    assert jnp.allclose(net.default_parameters(), base.default_parameters())


def test_extends_adds_parameter_and_overrides_expression(tmp_path):
    """The motivating case (#208/#246): add a parameter and override the two
    heterotroph rate expressions; everything else inherited."""
    net = aquakin.load_model_from_file(
        _derived(
            tmp_path,
            """
model:
  name: asm1_nutrient
  extends: asm1
parameters:
  KNH_H: {value: 0.05, units: "g_N/m3", bounds: [0.001, 1.0]}
expressions:
  rho_hetero_aerobic: "muH * [SS] / (KS + [SS]) * [SO] / (KOH + [SO]) * [SNH] / (KNH_H + [SNH]) * [XB_H]"
""",
        )
    )
    base = aquakin.load_model("asm1")
    assert "KNH_H" in net.param_index and "KNH_H" not in base.param_index
    assert net.species == base.species  # rest inherited
    assert net.reaction_names == base.reaction_names
    # The shipped asm1_ammonia_limitation is exactly this pattern.
    amm = aquakin.load_model("asm1_ammonia_limitation")
    assert "KNH_H" in amm.param_index


def test_extends_reaction_field_merge_keeps_base_stoichiometry(tmp_path):
    """Overriding a reaction by name merges fields: change the rate, keep the
    inherited stoichiometry."""
    net = aquakin.load_model_from_file(
        _derived(
            tmp_path,
            """
model:
  name: asm1_rate_tweak
  extends: asm1
parameters:
  scale: {value: 2.0}
reactions:
  - name: aerobic_growth_heterotrophs
    rate: "scale * rho_hetero_aerobic"
""",
        )
    )
    base = aquakin.load_model("asm1")
    i = net.reaction_names.index("aerobic_growth_heterotrophs")
    p = net.default_parameters()
    # Same stoichiometry as the base (inherited), only the rate scaled.
    assert jnp.allclose(net.compute_stoich(p)[i], base.compute_stoich(base.default_parameters())[i])


def test_extends_add_new_reaction(tmp_path):
    net = aquakin.load_model_from_file(
        _derived(
            tmp_path,
            """
model:
  name: asm1_extra
  extends: asm1
parameters:
  k_extra: {value: 0.1}
reactions:
  - name: extra_decay
    rate: "k_extra * [XP]"
    stoichiometry: {XP: -1, XI: 1}
""",
        )
    )
    assert "extra_decay" in net.reaction_names
    assert len(net.reaction_names) == len(aquakin.load_model("asm1").reaction_names) + 1


def test_remove_reaction_and_parameter(tmp_path):
    net = aquakin.load_model_from_file(
        _derived(
            tmp_path,
            """
model:
  name: asm1_no_ammonification
  extends: asm1
remove:
  reactions: [ammonification]
""",
        )
    )
    assert "ammonification" not in net.reaction_names
    assert "ammonification" in aquakin.load_model("asm1").reaction_names


def test_remove_block_explicit_null_is_treated_as_empty(tmp_path):
    """A `remove:` block set to an explicit null (e.g. `reactions: null`) removes
    nothing for that block -- `remove.get(block) or []` collapses None to empty --
    rather than erroring. The base is inherited intact."""
    net = aquakin.load_model_from_file(
        _derived(
            tmp_path,
            """
model:
  name: asm1_remove_null
  extends: asm1
remove:
  reactions: null
""",
        )
    )
    base = aquakin.load_model("asm1")
    assert net.reaction_names == base.reaction_names  # nothing removed
    assert net.species == base.species


# --- error handling ----------------------------------------------------------


def test_unknown_base_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="base model 'no_such_model' not found"):
        aquakin.load_model_from_file(
            _derived(
                tmp_path,
                """
model: {name: x, extends: no_such_model}
""",
            )
        )


def test_cyclic_extends_errors(tmp_path):
    _write(
        tmp_path,
        "a.yaml",
        "model: {name: a, extends: ./b.yaml}\n"
        "species: [{name: A, default_concentration: 1.0}]\n"
        "reactions: [{name: r, rate: '1.0', stoichiometry: {A: -1}}]\n",
    )
    _write(tmp_path, "b.yaml", "model: {name: b, extends: ./a.yaml}\n")
    with pytest.raises(ValueError, match="cyclic 'extends' chain"):
        aquakin.load_model_from_file(tmp_path / "a.yaml")


def test_remove_without_extends_errors(tmp_path):
    with pytest.raises(ValueError, match="'remove:' has no effect without 'extends:'"):
        aquakin.load_model_from_file(
            _derived(
                tmp_path,
                """
model: {name: x}
species: [{name: A, default_concentration: 1.0}]
reactions: [{name: r, rate: "1.0", stoichiometry: {A: -1}}]
remove: {reactions: [r]}
""",
            )
        )


def test_extends_declared_twice_errors(tmp_path):
    with pytest.raises(ValueError, match="declare 'extends' once"):
        aquakin.load_model_from_file(
            _derived(
                tmp_path,
                """
extends: asm1
model: {name: x, extends: asm1}
""",
            )
        )


def test_remove_nonexistent_reaction_errors(tmp_path):
    with pytest.raises(ValueError, match=r"remove\.reactions names .* not in the base"):
        aquakin.load_model_from_file(
            _derived(
                tmp_path,
                """
model: {name: x, extends: asm1}
remove: {reactions: [nope]}
""",
            )
        )


def test_relative_path_extends(tmp_path):
    """``extends`` resolves a relative path against the extending file's dir."""
    _write(
        tmp_path,
        "base.yaml",
        """
model: {name: tiny}
species:
  - {name: A, default_concentration: 1.0}
  - {name: B, default_concentration: 0.0}
parameters: {k: {value: 0.5}}
reactions: [{name: decay, rate: "k * [A]", stoichiometry: {A: -1, B: 1}}]
""",
    )
    net = aquakin.load_model_from_file(
        _derived(
            tmp_path,
            """
model: {name: tiny_fast, extends: ./base.yaml}
parameters: {k: {value: 2.0}}
""",
        )
    )
    assert net.name == "tiny_fast"
    assert float(net.default_parameters()[net.param_index["k"]]) == 2.0  # overridden


# --- unit: the pure merge helpers --------------------------------------------


def test_merge_spec_named_list_field_merge():
    base = {
        "reactions": [
            {"name": "r1", "rate": "a", "stoichiometry": {"A": -1}},
            {"name": "r2", "rate": "b"},
        ]
    }
    derived = {
        "reactions": [
            {"name": "r1", "rate": "c"},  # override field
            {"name": "r3", "rate": "d"},
        ]
    }  # append
    out = merge_spec(base, derived, source="<t>")
    r = {e["name"]: e for e in out["reactions"]}
    assert r["r1"] == {
        "name": "r1",
        "rate": "c",
        "stoichiometry": {"A": -1},
    }  # rate overridden, stoich kept
    assert r["r2"]["rate"] == "b"  # untouched
    assert r["r3"]["rate"] == "d"  # appended
    assert [e["name"] for e in out["reactions"]] == ["r1", "r2", "r3"]


def test_pop_inheritance_keys_reads_model_extends():
    data = {"model": {"name": "x", "extends": "asm1"}, "remove": {"reactions": ["r"]}}
    extends, remove = pop_inheritance_keys(data, "<t>")
    assert extends == "asm1" and remove == {"reactions": ["r"]}
    assert "extends" not in data["model"] and "remove" not in data


def test_apply_remove_unknown_block_errors():
    with pytest.raises(ValueError, match="unknown block"):
        apply_remove({"reactions": []}, {"widgets": ["x"]}, "<t>")


def test_merge_named_list_entry_without_name_errors():
    """A named-list entry lacking a 'name' cannot be merged by name and errors
    (inheritance.py line 85)."""
    base = {"reactions": [{"rate": "a"}]}
    derived = {"reactions": [{"name": "r1", "rate": "b"}]}
    with pytest.raises(ValueError, match="must be a mapping with a 'name'"):
        merge_spec(base, derived, source="<t>")


def test_apply_remove_non_mapping_errors():
    """A ``remove:`` that is not a mapping is rejected (inheritance.py line 107)."""
    with pytest.raises(ValueError, match="'remove:' must be a mapping"):
        apply_remove({"reactions": []}, ["reactions"], "<t>")


def test_remove_nonexistent_parameter_errors():
    """Removing a parameter absent from the base model errors
    (inheritance.py line 135)."""
    with pytest.raises(ValueError, match=r"remove\.parameters 'nope' is not in the base"):
        apply_remove({"parameters": {"k": {"value": 1.0}}}, {"parameters": ["nope"]}, "<t>")


def test_apply_remove_tolerates_nameless_base_entry():
    # A base named-list entry lacking a 'name' must be un-targetable by remove:
    # rather than tripping a bare KeyError; removing a present name still works and
    # the nameless sibling is retained.
    data = {"reactions": [{"name": "a", "rate": "1"}, {"rate": "2"}, {"name": "b", "rate": "3"}]}
    apply_remove(data, {"reactions": ["a"]}, "t")
    names = [e.get("name") for e in data["reactions"]]
    assert "a" not in names and "b" in names and None in names  # nameless retained
