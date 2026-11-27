"""Loader / schema unit tests."""

import textwrap
from pathlib import Path

import pytest

import aquakin


def _write(tmp_path, body: str):
    p = tmp_path / "net.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_builtin_ozone_bromate():
    model = aquakin.load_model("ozone_bromate")
    assert model.name == "ozone_bromate"
    assert "O3" in model.species
    assert "BrO3-" in model.species


def test_load_simple_from_file(simple_model):
    assert simple_model.name == "simple_decay"
    assert simple_model.species == ["A", "B"]


def test_unknown_builtin_raises():
    with pytest.raises(FileNotFoundError, match="Built-in model 'does_not_exist' not found"):
        aquakin.load_model("does_not_exist")


def test_undeclared_species_in_stoichiometry_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        model: {name: bad, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        reactions:
          - name: r1
            rate: "k * [A]"
            parameters:
              k: {value: 1.0}
            stoichiometry:
              A: -1
              X: +1
        """,
    )
    with pytest.raises(ValueError, match="stoichiometry references undeclared species 'X'"):
        aquakin.load_model_from_file(p)


def test_undeclared_condition_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        model: {name: bad, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        reactions:
          - name: r1
            rate: "k * [A] * pH_switch(8.0)"
            parameters:
              k: {value: 1.0}
            stoichiometry:
              A: -1
        """,
    )
    with pytest.raises(KeyError, match="references undeclared condition 'pH'"):
        aquakin.load_model_from_file(p)


def test_bounds_validation(tmp_path):
    p = _write(
        tmp_path,
        """
        model: {name: bad, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        reactions:
          - name: r1
            rate: "k * [A]"
            parameters:
              k:
                value: 10.0
                bounds: [0.1, 1.0]
            stoichiometry:
              A: -1
        """,
    )
    with pytest.raises(ValueError, match=r"parameter value 10\.0 is outside bounds"):
        aquakin.load_model_from_file(p)


def test_duplicate_species_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        model: {name: bad, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
          - {name: A, default_concentration: 2.0}
        reactions:
          - name: r1
            rate: "k * [A]"
            parameters:
              k: {value: 1.0}
            stoichiometry:
              A: -1
        """,
    )
    with pytest.raises(ValueError, match="Duplicate species names"):
        aquakin.load_model_from_file(p)


def test_unreferenced_expression_bad_species_rejected(tmp_path):
    """A named expression that NO reaction consumes is still validated -- a typo'd
    species reference in it must be rejected, not loaded silently (it used to be
    checked only when a reaction inlined the expression)."""
    p = _write(
        tmp_path,
        """
        model: {name: bad, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        expressions:
          unused: "[X]"
        reactions:
          - name: r1
            rate: "k * [A]"
            parameters:
              k: {value: 1.0}
            stoichiometry:
              A: -1
        """,
    )
    with pytest.raises(KeyError, match="undeclared species"):
        aquakin.load_model_from_file(p)


def test_unreferenced_expression_valid_refs_loads(tmp_path):
    """An unused expression with valid species/condition references loads fine."""
    p = _write(
        tmp_path,
        """
        model: {name: ok, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        expressions:
          unused: "[A] + 1"
        reactions:
          - name: r1
            rate: "k * [A]"
            parameters:
              k: {value: 1.0}
            stoichiometry:
              A: -1
        """,
    )
    net = aquakin.load_model_from_file(p)
    assert net.species == ["A"]


def test_missing_file():
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        aquakin.load_model_from_file("/no/such/file.yaml")


def test_non_validation_error_propagates(tmp_path, monkeypatch):
    """A genuine bug during validation must propagate, not be relabelled as an
    invalid model specification."""
    from aquakin.schema import loader

    def _boom(_data):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(loader.ModelSpec, "model_validate", staticmethod(_boom))
    p = _write(
        tmp_path,
        """
        model: {name: ok, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        reactions:
          - name: r1
            rate: "k * [A]"
            parameters:
              k: {value: 1.0}
            stoichiometry:
              A: -1
        """,
    )
    with pytest.raises(RecursionError, match="maximum recursion depth"):
        aquakin.load_model_from_file(p)


def test_extends_base_not_a_mapping_rejected(tmp_path):
    """A base file whose top-level YAML is not a mapping (here a list) is a clear
    error, not an opaque downstream failure."""
    (tmp_path / "base.yaml").write_text("- 1\n- 2\n- 3\n")
    derived = tmp_path / "derived.yaml"
    derived.write_text(
        textwrap.dedent("""
        model: {name: d, extends: base.yaml}
        """)
    )
    with pytest.raises(ValueError, match="must be a mapping"):
        aquakin.load_model_from_file(derived)


def test_empty_composition_quantity_name_rejected(tmp_path):
    """A composition entry keyed by the empty string is rejected at validation."""
    p = _write(
        tmp_path,
        """
        model: {name: bad}
        species:
          - {name: A, default_concentration: 1.0, composition: {"": 1.0}}
        reactions:
          - name: r1
            rate: "k * [A]"
            parameters: {k: {value: 1.0}}
            stoichiometry: {A: -1}
        """,
    )
    with pytest.raises(ValueError, match="empty composition"):
        aquakin.load_model_from_file(p)


@pytest.mark.parametrize("bad", [".nan", ".inf", "-.inf"])
def test_non_finite_composition_value_rejected(tmp_path, bad):
    """NaN / +-inf composition content is rejected -- such a value would poison the
    conservation check it feeds."""
    body = (
        "model: {name: bad}\n"
        "species:\n"
        "  - {name: A, default_concentration: 1.0, composition: {COD: " + bad + "}}\n"
        "reactions:\n"
        "  - name: r1\n"
        '    rate: "k * [A]"\n'
        "    parameters: {k: {value: 1.0}}\n"
        "    stoichiometry: {A: -1}\n"
    )
    p = tmp_path / "net.yaml"
    p.write_text(body)
    with pytest.raises(ValueError, match="must be finite"):
        aquakin.load_model_from_file(p)


# ----- speciation activity_model override (issue #205) ---------------------


def test_load_model_activity_override_shifts_ph_and_keeps_default():
    """load_model(..., activity_model=) overrides the speciation activity
    model on a shipped model; the cached default is untouched."""
    import jax.numpy as jnp

    from aquakin import load_model

    base = load_model("adm1")  # cached default (none)
    dav = load_model("adm1", activity_model="davies")
    assert dav is not base
    assert load_model("adm1") is base  # default still cached/unchanged

    C = base.default_concentrations()
    conds = {f: jnp.asarray([v]) for f, v in base._condition_defaults.items()}

    def ph(net):
        return float(net.derived_condition_fn(C, net.default_parameters(), conds, 0)["pH"])

    ph_base, ph_dav = ph(base), ph(dav)
    # The default (no activity) pH is the plain charge balance of the speciation
    # block's totals: VFAs from kgCOD/m3 by their COD per mole, the rest molar.
    from aquakin.core.ph_solver import solve_ph

    si = base.species_index

    def c(name):
        return float(C[si[name]])

    expected = solve_ph(
        tot_carbonate=c("S_IC"), tot_ammonia=c("S_IN"),
        tot_acetate=c("S_ac") / 64.0, tot_propionate=c("S_pro") / 112.0,
        tot_butyrate=c("S_bu") / 160.0, tot_valerate=c("S_va") / 208.0,
        z_cation_eq=c("S_cat"), strong_anion_eq=c("S_an"),
        T_kelvin=float(base._condition_defaults["T"]),
    )
    assert ph_base == pytest.approx(float(expected), abs=1e-9)
    assert abs(ph_dav - ph_base) > 0.02  # activity-shifted


def test_activity_override_requires_speciation_block():
    from aquakin import load_model

    with pytest.raises(ValueError, match="speciation"):
        load_model("asm1", activity_model="davies")  # no pH solver


def test_activity_override_validates_model_name():
    from aquakin import load_model

    with pytest.raises(ValueError, match="activity_model"):
        load_model("adm1", activity_model="bogus")


def test_speciation_activity_model_validated_in_yaml(tmp_path):
    """A bad activity_model in a speciation: block is rejected at load."""
    body = """
    model:
      name: t
      version: "1"
    species:
      - {name: S_IC, units: mol/L, default_concentration: 1.0e-3}
    conditions:
      - {name: T, default: 25.0}
    speciation:
      temperature_field: T
      activity_model: bogus
      totals:
        carbonate: {species: S_IC, molar_mass: 1.0}
    reactions:
      - {name: r, rate: "0.0 * [S_IC]", stoichiometry: {S_IC: -1}}
    """
    with pytest.raises(ValueError, match="activity_model"):
        aquakin.load_model_from_file(_write(tmp_path, body))


def test_clear_model_cache_is_exported():
    # Documented public API: must be importable from the top-level package and in
    # __all__ (it was missing despite being referenced in the docs).
    import aquakin

    assert "clear_model_cache" in aquakin.__all__
    assert callable(aquakin.clear_model_cache)
    # Cached identity, then cleared.
    a = aquakin.load_model("asm1")
    assert aquakin.load_model("asm1") is a
    aquakin.clear_model_cache()
    assert aquakin.load_model("asm1") is not a


def test_top_level_yaml_not_a_mapping_rejected(tmp_path):
    """A model file whose top-level YAML is not a mapping (here a list) is
    rejected with a clear error (loader.py line 85)."""
    p = tmp_path / "net.yaml"
    p.write_text("- 1\n- 2\n- 3\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        aquakin.load_model_from_file(p)


def test_malformed_yaml_rejected(tmp_path):
    """Unparseable YAML is reported as a parse failure, not an opaque crash
    (loader.py line 83)."""
    p = tmp_path / "net.yaml"
    # A bare unmatched flow-mapping brace is a YAML syntax error.
    p.write_text("model: {name: bad\nspecies: [\n")
    with pytest.raises(ValueError, match="Failed to parse YAML"):
        aquakin.load_model_from_file(p)


def test_extends_relative_base_file_not_found_rejected(tmp_path):
    """A relative ``extends:`` pointing at a missing base file is rejected
    (loader.py line 34)."""
    with pytest.raises(FileNotFoundError, match="extends: base file not found"):
        aquakin.load_model_from_file(
            _write(
                tmp_path,
                """
                model: {name: d, extends: ./no_such_base.yaml}
                """,
            )
        )


def test_extends_base_malformed_yaml_rejected(tmp_path):
    """A base file with unparseable YAML is reported as a base-parse failure
    (loader.py line 67)."""
    (tmp_path / "base.yaml").write_text("model: {name: base\nspecies: [\n")
    derived = tmp_path / "derived.yaml"
    derived.write_text(
        textwrap.dedent("""
        model: {name: d, extends: base.yaml}
        """)
    )
    with pytest.raises(ValueError, match="failed to parse base"):
        aquakin.load_model_from_file(derived)


def test_duplicate_stoichiometry_key_rejected_naming_key_and_lines(tmp_path):
    """A mapping that repeats a key is refused, naming the key and both lines,
    rather than loaded as whichever value came last (``asm2d`` once listed
    XMeOH / XMeP twice in a stoichiometry)."""
    p = _write(
        tmp_path,
        """
        model: {name: dup, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
          - {name: B, default_concentration: 0.0}
        reactions:
          - name: r1
            rate: "k * [A]"
            parameters:
              k: {value: 1.0}
            stoichiometry:
              A: -1
              B: 1
              A: -2
        """,
    )
    with pytest.raises(ValueError, match=r"duplicate key 'A' \(first given on line 12\)") as err:
        aquakin.load_model_from_file(p)
    assert "line 14" in str(err.value)  # where the repeat is


def test_duplicate_key_in_an_extended_base_rejected(tmp_path):
    (tmp_path / "base.yaml").write_text(
        textwrap.dedent("""
        model: {name: base, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        reactions:
          - name: r1
            rate: "k * [A]"
            rate: "2 * k * [A]"
            parameters:
              k: {value: 1.0}
            stoichiometry:
              A: -1
        """)
    )
    derived = tmp_path / "derived.yaml"
    derived.write_text("model: {name: d, extends: base.yaml}\n")
    with pytest.raises(ValueError, match=r"(?s)failed to parse base 'base\.yaml'.*duplicate key 'rate'"):
        aquakin.load_model_from_file(derived)


def test_model_yaml_parser_keeps_safe_loader_semantics():
    """Merge keys still let an explicit key override a merged one, and the safe
    loader's own errors (an unhashable key, a ``!!map`` scalar) are unchanged."""
    import yaml

    from aquakin.schema.loader import parse_model_yaml

    doc = parse_model_yaml("base: &b {x: 1, y: 2}\nd:\n  <<: *b\n  x: 3\n")
    assert doc["d"] == {"x": 3, "y": 2}
    with pytest.raises(yaml.constructor.ConstructorError, match="found unhashable key"):
        parse_model_yaml("? [a, b]\n: 1\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="expected a mapping node"):
        parse_model_yaml("a: !!map scalar\n")


_REPO = Path(__file__).resolve().parents[2]
_ALL_YAML = sorted(
    p.relative_to(_REPO).as_posix()
    for pattern in ("aquakin/models/**/*.yaml", "tests/fixtures/**/*.yaml", "examples/**/*.yaml")
    for p in _REPO.glob(pattern)
)


@pytest.mark.parametrize("path", _ALL_YAML)
def test_shipped_fixture_and_example_yaml_have_no_duplicate_keys(path):
    """Every shipped model, test fixture and example YAML parses under the
    duplicate-refusing loader (a repeat raises, naming the key)."""
    from aquakin.schema.loader import parse_model_yaml

    assert parse_model_yaml((_REPO / path).read_text(encoding="utf-8")) is not None


def test_asm1_adm1_ship_literature_priors():
    # asm1 and adm1 carry literature-grounded Gaussian priors (physical-space
    # mean/std centred on the nominal, relative std = a literature coefficient of
    # variation). ASM1: Hauduc (2011) activated-sludge database V% (wastewater-
    # specific). ADM1: Brun (2002) uncertainty classes around the BSM2 municipal-
    # sludge nominal (kinetics 50%, yields 5%) -- deliberately NOT Mo (2023)'s
    # across-substrate ranges, which measure digester-type variation (grass
    # silage, food/agricultural/industrial waste) rather than the uncertainty of
    # a mesophilic municipal-sludge digester.
    asm1 = aquakin.load_model("asm1")
    adm1 = aquakin.load_model("adm1")
    assert len(asm1.parameter_priors) == 19
    assert len(adm1.parameter_priors) == 38
    # Every prior is physically valid and centred on the parameter's nominal.
    for net in (asm1, adm1):
        for name, (mean, std) in net.parameter_priors.items():
            assert std > 0.0, name
            assert mean > 0.0, name
            nominal = float(net.default_parameters()[net.param_index[name]])
            assert mean == pytest.approx(nominal, rel=1e-6), name

    # ASM1 muH: narrow Hauduc V=6% spread.
    mean, std = asm1.parameter_priors["muH"]
    assert std / mean == pytest.approx(0.06, rel=1e-3)
    # ASM1 KNO: the wide half-saturation V=80%.
    mean, std = asm1.parameter_priors["KNO"]
    assert std / mean == pytest.approx(0.80, rel=1e-3)
    # ADM1 kinetics: Brun class-3 (50%) around the municipal-sludge nominal. The
    # H2-inhibition constant is centred on its default (mean == 1e-5), NOT the
    # cross-substrate Mo range whose linear midpoint (~5e-6) would pull it toward
    # the grass-silage extreme of 5e-8.
    mean, std = adm1.parameter_priors["K_I_h2_c4"]
    assert std / mean == pytest.approx(0.5, rel=1e-3)
    assert mean == pytest.approx(1e-5, rel=1e-3)
    # ADM1 yields: Brun class-1 (5%).
    mean, std = adm1.parameter_priors["Y_su"]
    assert std / mean == pytest.approx(0.05, rel=1e-3)
