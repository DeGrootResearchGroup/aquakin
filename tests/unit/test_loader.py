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
