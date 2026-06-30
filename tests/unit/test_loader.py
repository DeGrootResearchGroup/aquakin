"""Loader / schema unit tests."""

import textwrap

import pytest

import aquakin


def _write(tmp_path, body: str):
    p = tmp_path / "net.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_builtin_ozone_bromate():
    network = aquakin.load_network("ozone_bromate")
    assert network.name == "ozone_bromate"
    assert "O3" in network.species
    assert "BrO3-" in network.species


def test_load_simple_from_file(simple_network):
    assert simple_network.name == "simple_decay"
    assert simple_network.species == ["A", "B"]


def test_unknown_builtin_raises():
    with pytest.raises(FileNotFoundError):
        aquakin.load_network("does_not_exist")


def test_undeclared_species_in_stoichiometry_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        network: {name: bad, version: "1.0"}
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
    with pytest.raises(ValueError):
        aquakin.load_network_from_file(p)


def test_undeclared_species_in_rate_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        network: {name: bad, version: "1.0"}
        species:
          - {name: A, default_concentration: 1.0}
        reactions:
          - name: r1
            rate: "k * [X]"
            parameters:
              k: {value: 1.0}
            stoichiometry:
              A: -1
        """,
    )
    with pytest.raises((KeyError, ValueError)):
        aquakin.load_network_from_file(p)


def test_undeclared_condition_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        network: {name: bad, version: "1.0"}
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
    with pytest.raises((KeyError, ValueError)):
        aquakin.load_network_from_file(p)


def test_bounds_validation(tmp_path):
    p = _write(
        tmp_path,
        """
        network: {name: bad, version: "1.0"}
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
    with pytest.raises(ValueError):
        aquakin.load_network_from_file(p)


def test_duplicate_species_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        network: {name: bad, version: "1.0"}
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
    with pytest.raises(ValueError):
        aquakin.load_network_from_file(p)


def test_unreferenced_expression_bad_species_rejected(tmp_path):
    """A named expression that NO reaction consumes is still validated -- a typo'd
    species reference in it must be rejected, not loaded silently (it used to be
    checked only when a reaction inlined the expression)."""
    p = _write(
        tmp_path,
        """
        network: {name: bad, version: "1.0"}
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
        aquakin.load_network_from_file(p)


def test_unreferenced_expression_valid_refs_loads(tmp_path):
    """An unused expression with valid species/condition references loads fine."""
    p = _write(
        tmp_path,
        """
        network: {name: ok, version: "1.0"}
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
    net = aquakin.load_network_from_file(p)
    assert net.species == ["A"]


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        aquakin.load_network_from_file("/no/such/file.yaml")


def test_non_validation_error_propagates(tmp_path, monkeypatch):
    """A genuine bug during validation must propagate, not be relabelled as an
    invalid network specification."""
    from aquakin.schema import loader

    def _boom(_data):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(loader.NetworkSpec, "model_validate", staticmethod(_boom))
    p = _write(
        tmp_path,
        """
        network: {name: ok, version: "1.0"}
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
    with pytest.raises(RecursionError):
        aquakin.load_network_from_file(p)


def test_extends_base_not_a_mapping_rejected(tmp_path):
    """A base file whose top-level YAML is not a mapping (here a list) is a clear
    error, not an opaque downstream failure."""
    (tmp_path / "base.yaml").write_text("- 1\n- 2\n- 3\n")
    derived = tmp_path / "derived.yaml"
    derived.write_text(textwrap.dedent("""
        network: {name: d, extends: base.yaml}
        """))
    with pytest.raises(ValueError, match="must be a mapping"):
        aquakin.load_network_from_file(derived)


def test_empty_composition_quantity_name_rejected(tmp_path):
    """A composition entry keyed by the empty string is rejected at validation."""
    p = _write(
        tmp_path,
        """
        network: {name: bad}
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
        aquakin.load_network_from_file(p)


@pytest.mark.parametrize("bad", [".nan", ".inf", "-.inf"])
def test_non_finite_composition_value_rejected(tmp_path, bad):
    """NaN / +-inf composition content is rejected -- such a value would poison the
    conservation check it feeds."""
    body = (
        "network: {name: bad}\n"
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
        aquakin.load_network_from_file(p)


# ----- speciation activity_model override (issue #205) ---------------------

def test_load_network_activity_override_shifts_ph_and_keeps_default():
    """load_network(..., activity_model=) overrides the speciation activity
    model on a shipped network; the cached default is untouched."""
    import jax.numpy as jnp
    from aquakin import load_network

    base = load_network("adm1")                      # cached default (none)
    dav = load_network("adm1", activity_model="davies")
    assert dav is not base
    assert load_network("adm1") is base              # default still cached/unchanged

    C = base.default_concentrations()
    conds = {f: jnp.asarray([v]) for f, v in base._condition_defaults.items()}

    def ph(net):
        return float(net.derived_condition_fn(
            C, net.default_parameters(), conds, 0)["pH"])

    ph_base, ph_dav = ph(base), ph(dav)
    assert ph_base == pytest.approx(7.27, abs=0.05)  # validated BSM2 digester pH
    assert abs(ph_dav - ph_base) > 0.02              # activity-shifted


def test_activity_override_requires_speciation_block():
    from aquakin import load_network
    with pytest.raises(ValueError, match="speciation"):
        load_network("asm1", activity_model="davies")   # no pH solver


def test_activity_override_validates_model_name():
    from aquakin import load_network
    with pytest.raises(ValueError, match="activity_model"):
        load_network("adm1", activity_model="bogus")


def test_speciation_activity_model_validated_in_yaml(tmp_path):
    """A bad activity_model in a speciation: block is rejected at load."""
    body = """
    network:
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
        aquakin.load_network_from_file(_write(tmp_path, body))


def test_clear_network_cache_is_exported():
    # Documented public API: must be importable from the top-level package and in
    # __all__ (it was missing despite being referenced in the docs).
    import aquakin
    assert "clear_network_cache" in aquakin.__all__
    assert callable(aquakin.clear_network_cache)
    # Cached identity, then cleared.
    a = aquakin.load_network("asm1")
    assert aquakin.load_network("asm1") is a
    aquakin.clear_network_cache()
    assert aquakin.load_network("asm1") is not a


def test_asm1_adm1_ship_literature_priors():
    # asm1 and adm1 carry literature-grounded Gaussian priors (physical-space
    # mean/std centred on the nominal, relative std = a literature coefficient of
    # variation). ASM1: Hauduc (2011) activated-sludge database V% (wastewater-
    # specific). ADM1: Brun (2002) uncertainty classes around the BSM2 municipal-
    # sludge nominal (kinetics 50%, yields 5%) -- deliberately NOT Mo (2023)'s
    # across-substrate ranges, which measure digester-type variation (grass
    # silage, food/agricultural/industrial waste) rather than the uncertainty of
    # a mesophilic municipal-sludge digester.
    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")
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
