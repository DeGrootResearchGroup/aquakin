"""SpatialConditions, the OperatingConditions 0-D alias, and ``with_``."""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.core.conditions import OperatingConditions, SpatialConditions


# ----- OperatingConditions (0-D alias) ------------------------------------

def test_operating_conditions_is_single_location_spatial():
    oc = OperatingConditions(pH=7.5, T=293.15)
    assert isinstance(oc, SpatialConditions)
    assert oc.n_locations == 1
    assert float(oc.fields["pH"][0]) == 7.5
    assert float(oc.fields["T"][0]) == pytest.approx(293.15)
    # Scalar field values are stored as length-1 arrays.
    assert oc.fields["pH"].shape == (1,)


def test_operating_conditions_exported_at_top_level():
    assert aquakin.OperatingConditions is OperatingConditions


def test_operating_conditions_validates_required():
    oc = OperatingConditions(pH=7.0)
    oc.validate_required(["pH"])
    with pytest.raises(ValueError):
        oc.validate_required(["pH", "T"])


def test_operating_conditions_works_in_reactor():
    network = aquakin.load_network("ozone_bromate")
    reactor = aquakin.BatchReactor(
        network, OperatingConditions(pH=7.5, T=293.15, OH_scavenging=5.0e4))
    C0 = network.concentrations({"O3": 1.0e-4, "Br-": 1.0e-5})
    sol = reactor.solve(C0, t_span=(0.0, 600.0), t_eval=jnp.array([0.0, 600.0]))
    assert jnp.all(jnp.isfinite(sol.C))


# ----- with_ ---------------------------------------------------------------

def test_with_overrides_field_and_leaves_original():
    base = SpatialConditions.uniform(pH=7.0, T=293.15)
    cold = base.with_(T=283.15)
    assert float(cold.fields["T"][0]) == pytest.approx(283.15)
    assert float(cold.fields["pH"][0]) == 7.0           # carried over
    assert float(base.fields["T"][0]) == pytest.approx(293.15)  # unchanged
    assert type(cold) is SpatialConditions


def test_with_adds_new_field():
    c = SpatialConditions.uniform(pH=7.0).with_(T=300.0)
    assert set(c.fields) == {"pH", "T"}


def test_with_on_default_conditions():
    net = aquakin.load_network("asm1")
    cold = net.default_conditions().with_(T=283.15)
    assert float(cold.fields["T"][0]) == pytest.approx(283.15)


def test_with_broadcasts_scalar_and_accepts_array_for_multilocation():
    sc = SpatialConditions.uniform(3, pH=7.0)
    out = sc.with_(pH=8.0, T=jnp.array([280.0, 285.0, 290.0]))
    assert out.n_locations == 3
    assert jnp.allclose(out.fields["pH"], 8.0)          # scalar broadcast
    assert jnp.allclose(out.fields["T"], jnp.array([280.0, 285.0, 290.0]))


def test_with_on_operating_conditions_returns_single_location():
    oc = OperatingConditions(pH=7.5, T=293.15)
    out = oc.with_(pH=8.0)
    assert out.n_locations == 1
    assert float(out.fields["pH"][0]) == 8.0
    assert float(out.fields["T"][0]) == pytest.approx(293.15)
