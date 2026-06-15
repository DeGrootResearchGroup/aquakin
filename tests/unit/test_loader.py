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
