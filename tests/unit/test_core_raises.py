"""Regression tests for validation ``raise`` paths in the core compile pipeline.

These cover the load-time / builder-time guards that reject a malformed model:
unknown species/parameter lookups on a :class:`CompiledModel`, bad rate- and
stoichiometry-expression references, malformed ``speciation:`` /
``precipitation:`` declarations, and the vectorized-kernel index lookups. Each
test triggers exactly one guard via the most natural API (a small YAML load, a
``CompiledModel`` method, or the derived-fn builder called directly).
"""

import importlib.util
import os
import tempfile

import pytest

import aquakin
from aquakin.core.parser import parse_rate_expression
from aquakin.core.precipitation import build_precipitation_derived_fn
from aquakin.core.precipitation_equilibrium import (
    build_precipitation_equilibrium_derived_fn,
)
from aquakin.core.speciation import build_ph_derived_fn
from aquakin.core.vector_kernel import build_vectorized_rates


def _load_yaml(text: str):
    """Write ``text`` to a temp YAML file and load it as a model."""
    p = tempfile.mktemp(suffix=".yaml")
    with open(p, "w") as f:
        f.write(text)
    return aquakin.load_model_from_file(p)


_SPECIES_A = "species:\n  - {name: A, units: mol/L, default_concentration: 1.0}\n"


# --- CompiledModel public-method lookups (core/model.py) ---------------------


def test_concentrations_unknown_species_raises():
    model = aquakin.load_model("asm1")
    with pytest.raises(KeyError, match="Unknown species 'Zzz'"):
        model.concentrations({"Zzz": 1.0})


def test_concentrations_bad_base_raises():
    model = aquakin.load_model("asm1")
    with pytest.raises(ValueError, match="base must be 'defaults' or 'zero'"):
        model.concentrations(base="bogus")


def test_override_vector_non_dict_raises():
    model = aquakin.load_model("asm1")
    with pytest.raises(TypeError, match="overrides must be a dict"):
        model.concentrations([1.0, 2.0])


def test_units_of_unknown_species_raises():
    model = aquakin.load_model("asm1")
    with pytest.raises(KeyError, match="Unknown species 'Zzz'"):
        model.units_of("Zzz")


def test_description_of_unknown_species_raises():
    model = aquakin.load_model("asm1")
    with pytest.raises(KeyError, match="Unknown species 'Zzz'"):
        model.description_of("Zzz")


def test_precipitation_equilibrium_without_block_raises():
    # asm1 declares no precipitation: block, so the equilibrium projection has
    # no equilibrium-mode mineral to solve for.
    model = aquakin.load_model("asm1")
    with pytest.raises(ValueError, match="requires a precipitation: block"):
        model.precipitation_equilibrium()


# --- compile-time reference validation (core/model.py) -----------------------


def test_rate_expression_undeclared_species_raises():
    with pytest.raises(KeyError, match="undeclared species 'Zzz'"):
        _load_yaml(
            "model: {name: b, description: d}\n"
            + _SPECIES_A
            + """conditions: []
reactions:
  - name: R
    rate: "k * [A] * [Zzz]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1}
"""
        )


def test_rate_expression_undeclared_condition_raises():
    with pytest.raises(KeyError, match="undeclared condition 'Tzz'"):
        _load_yaml(
            "model: {name: b, description: d}\n"
            + _SPECIES_A
            + """conditions: []
reactions:
  - name: R
    rate: "k * [A] * {Tzz}"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1}
"""
        )


def test_rate_expression_unresolved_parameter_raises():
    with pytest.raises(KeyError, match="kbogus"):
        _load_yaml(
            "model: {name: b, description: d}\n"
            + _SPECIES_A
            + """conditions: []
reactions:
  - name: R
    rate: "kbogus * [A]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1}
"""
        )


def test_named_expression_undeclared_species_raises():
    # An expression no reaction consumes is still validated at compile time.
    with pytest.raises(KeyError, match=r"Named expression 'unused'.*undeclared species 'Zzz'"):
        _load_yaml(
            "model: {name: b, description: d}\n"
            + _SPECIES_A
            + """conditions: []
expressions:
  unused: "[Zzz] * 2"
reactions:
  - name: R
    rate: "k * [A]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1}
"""
        )


def test_named_expression_unknown_condition_raises():
    with pytest.raises(
        ValueError, match=r"Named expression 'unused'.*unknown condition field 'Tzz'"
    ):
        _load_yaml(
            "model: {name: b, description: d}\n"
            + _SPECIES_A
            + """conditions: []
expressions:
  unused: "{Tzz} * 2"
reactions:
  - name: R
    rate: "k * [A]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1}
"""
        )


def test_stoichiometry_expression_unresolved_parameter_raises():
    with pytest.raises(KeyError, match="bogusparam"):
        _load_yaml(
            "model: {name: b, description: d}\n"
            "species:\n"
            "  - {name: A, units: mol/L, default_concentration: 1.0}\n"
            "  - {name: B, units: mol/L, default_concentration: 0.0}\n"
            """conditions: []
reactions:
  - name: R
    rate: "k * [A]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1, B: "bogusparam * 2"}
"""
        )


# --- vectorized kernel index lookups (core/vector_kernel.py) -----------------


def test_vector_kernel_undeclared_species_raises():
    ast = parse_rate_expression("[Zzz]")
    with pytest.raises(KeyError, match="Species 'Zzz' not declared"):
        build_vectorized_rates([ast], ["R"], {}, {})


def test_vector_kernel_unresolved_parameter_raises():
    ast = parse_rate_expression("kbogus")
    with pytest.raises(KeyError, match=r"Parameter 'kbogus'.*not found"):
        build_vectorized_rates([ast], ["R"], {}, {})


# --- speciation builder validation (core/speciation.py) ----------------------


def test_speciation_bad_temperature_units_raises():
    with pytest.raises(ValueError, match="temperature_units must be 'celsius' or 'kelvin'"):
        build_ph_derived_fn({"temperature_field": "T", "temperature_units": "fahrenheit"}, {"A": 0})


def test_speciation_unknown_totals_key_raises():
    with pytest.raises(ValueError, match="'totals' has unknown systems"):
        build_ph_derived_fn(
            {"temperature_field": "T", "totals": {"bogus": {"species": "A", "molar_mass": 1.0}}},
            {"A": 0},
        )


def test_speciation_undeclared_species_raises():
    with pytest.raises(KeyError, match="undeclared species 'Zzz'"):
        build_ph_derived_fn(
            {
                "temperature_field": "T",
                "totals": {"carbonate": {"species": "Zzz", "molar_mass": 1.0}},
            },
            {"A": 0},
        )


# --- kinetic precipitation builder validation (core/precipitation.py) --------


def test_precipitation_unknown_fraction_raises():
    with pytest.raises(ValueError, match="unknown fraction 'bogus'"):
        build_precipitation_derived_fn(
            {
                "minerals": [
                    {
                        "name": "m",
                        "pKsp": 1.0,
                        "order": 1,
                        "ions": [
                            {
                                "species": "A",
                                "molar_mass": 1.0,
                                "count": 1,
                                "charge": 1,
                                "fraction": "bogus",
                            }
                        ],
                    }
                ]
            },
            {"A": 0},
        )


def test_precipitation_ion_missing_species_raises():
    with pytest.raises(ValueError, match="needs a 'species'"):
        build_precipitation_derived_fn(
            {
                "minerals": [
                    {
                        "name": "m",
                        "pKsp": 1.0,
                        "order": 1,
                        "ions": [
                            {
                                "molar_mass": 1.0,
                                "count": 1,
                                "charge": 1,
                                "fraction": "carbonate",
                            }
                        ],
                    }
                ]
            },
            {"A": 0},
        )


# --- equilibrium precipitation builder validation ---------------------------


def test_precipitation_equilibrium_missing_solid_raises():
    with pytest.raises(ValueError, match="needs a 'solid:' species"):
        build_precipitation_equilibrium_derived_fn(
            {
                "minerals": [
                    {
                        "name": "m",
                        "mode": "equilibrium",
                        "pKsp": 1.0,
                        "ions": [{"species": "A", "molar_mass": 1.0, "count": 1, "charge": 1}],
                    }
                ]
            },
            {"A": 0},
        )


def test_precipitation_equilibrium_undeclared_solid_raises():
    with pytest.raises(KeyError, match="solid 'Zzz' is not a declared"):
        build_precipitation_equilibrium_derived_fn(
            {
                "minerals": [
                    {
                        "name": "m",
                        "mode": "equilibrium",
                        "solid": "Zzz",
                        "pKsp": 1.0,
                        "ions": [{"species": "A", "molar_mass": 1.0, "count": 1, "charge": 1}],
                    }
                ]
            },
            {"A": 0},
        )


# --- model-generator name lookups (models/_make_khalil_*.py) -----------------


def _load_generator(module_name: str):
    """Import a (package-private, manually-run) ``_make_*`` generator module by
    path; these are not importable as ``aquakin.models.<name>``."""
    base = os.path.dirname(aquakin.__file__)
    path = os.path.join(base, "models", module_name + ".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_khalil_variants_rxn_unknown_name_raises():
    module = _load_generator("_make_khalil_variants")
    with pytest.raises(KeyError, match="nope"):
        module.rxn({"reactions": [{"name": "a"}]}, "nope")


def test_khalil_paper_by_name_unknown_name_raises():
    # _make_khalil_paper imports ruamel at module scope (a manual-generator dep,
    # not a runtime dep); skip cleanly when it is unavailable.
    pytest.importorskip("ruamel.yaml")
    module = _load_generator("_make_khalil_paper")
    with pytest.raises(KeyError, match="nope"):
        module.by_name([{"name": "a"}], "nope")


# --- more compile-time reference validation (core/model.py) ------------------


def test_unparseable_named_expression_raises():
    # A malformed named-expression formula surfaces at compile with the
    # offending expression name, wrapping the parser error.
    with pytest.raises(ValueError, match="Failed to parse named expression 'bad'"):
        _load_yaml(
            "model: {name: b, description: d}\n"
            + _SPECIES_A
            + """conditions: []
expressions:
  bad: "1 +"
reactions:
  - name: R
    rate: "k * [A]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1}
"""
        )


def test_unparseable_stoichiometry_expression_raises():
    # A malformed stoichiometric-coefficient expression (as opposed to an
    # unresolved parameter) is reported at compile with the reaction/species.
    with pytest.raises(ValueError, match="Failed to parse stoichiometric coefficient 'R'"):
        _load_yaml(
            "model: {name: b, description: d}\n"
            "species:\n"
            "  - {name: A, units: mol/L, default_concentration: 1.0}\n"
            "  - {name: B, units: mol/L, default_concentration: 0.0}\n"
            """conditions: []
reactions:
  - name: R
    rate: "k * [A]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1, B: "k *"}
"""
        )


def test_speciation_reads_undeclared_condition_raises():
    # The speciation block's temperature_field must be a declared condition.
    with pytest.raises(ValueError, match="speciation block reads condition field"):
        _load_yaml(
            "model: {name: b, description: d}\n"
            + _SPECIES_A
            + """conditions: []
speciation:
  temperature_field: T
  totals:
    carbonate: {species: A, molar_mass: 12000}
reactions:
  - name: R
    rate: "k * [A]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1}
"""
        )


def test_speciation_produces_declared_condition_raises():
    # A speciation-produced field (here pH) must not also be declared in
    # 'conditions:' -- it is computed, not supplied.
    with pytest.raises(ValueError, match="speciation produces condition field"):
        _load_yaml(
            "model: {name: b, description: d}\n"
            + _SPECIES_A
            + """conditions:
  - {name: T, default: 298.15}
  - {name: pH, default: 7.0}
speciation:
  field: pH
  temperature_field: T
  temperature_units: kelvin
  totals:
    carbonate: {species: A, molar_mass: 12000}
reactions:
  - name: R
    rate: "k * [A]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1}
"""
        )


_PRECIP_MINERAL = """precipitation:
  pH_field: pH
  temperature_field: T
  temperature_units: kelvin
  minerals:
    - name: struvite
      pKsp: 13.26
      order: 1
      solid: X_str
      rate_constant: {value: 1.0, units: "1/d"}
      ions:
        - {species: S_Mg, molar_mass: 1000, count: 1, charge: 2}
"""

_PRECIP_SPECIES = (
    "species:\n"
    "  - {name: S_Mg, units: mol/L, default_concentration: 1.0}\n"
    "  - {name: X_str, units: mol/L, default_concentration: 0.0}\n"
)


def test_precipitation_reads_undeclared_condition_raises():
    # The precipitation block reads pH, which here is neither declared nor
    # produced by a speciation: block.
    with pytest.raises(ValueError, match=r"precipitation .* reads condition field"):
        _load_yaml(
            "model: {name: b, description: d}\n"
            + _PRECIP_SPECIES
            + "conditions:\n  - {name: T, default: 298.15}\n"
            + _PRECIP_MINERAL
        )


def test_precipitation_produces_declared_condition_raises():
    # A precipitation-produced field (SI_<name>) must not collide with a
    # declared condition.
    with pytest.raises(ValueError, match=r"precipitation .* produces field"):
        _load_yaml(
            "model: {name: b, description: d}\n" + _PRECIP_SPECIES + "conditions:\n"
            "  - {name: T, default: 298.15}\n"
            "  - {name: pH, default: 7.0}\n"
            "  - {name: SI_struvite, default: 0.0}\n" + _PRECIP_MINERAL
        )


# --- introspection / composition-metadata guards (core/introspect.py) --------


def test_check_nitrogen_without_composition_metadata_raises():
    # A model with no per-species composition: metadata cannot be nitrogen-
    # balanced; the guard asks the user to declare it or pass composition=.
    model = aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")
    with pytest.raises(ValueError, match="no composition metadata"):
        model.check_nitrogen()
