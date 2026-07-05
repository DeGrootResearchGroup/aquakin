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
    model = aquakin.load_model("ozone_bromate")
    reactor = aquakin.BatchReactor(
        model, OperatingConditions(pH=7.5, T=293.15, OH_scavenging=5.0e4))
    C0 = model.concentrations({"O3": 1.0e-4, "Br-": 1.0e-5})
    sol = reactor.solve(C0, t_span=(0.0, 600.0), t_eval=jnp.array([0.0, 600.0]))
    assert jnp.all(jnp.isfinite(sol.C))


# ----- construction-time validation ---------------------------------------

def test_mismatched_field_lengths_rejected():
    """Every field must span the same number of locations; a length mismatch is a
    construction-time ValueError, not a silent broadcast."""
    with pytest.raises(ValueError, match="length"):
        SpatialConditions(fields={"pH": jnp.array([7.0, 7.1]),
                                  "T": jnp.array([293.15])})


def test_non_1d_field_rejected():
    """A condition field must be scalar (promoted to length-1) or 1-D; a 2-D array
    is rejected."""
    with pytest.raises(ValueError, match="1-D"):
        SpatialConditions(fields={"pH": jnp.ones((2, 2))})


def test_uniform_rejects_nonpositive_location_count():
    with pytest.raises(ValueError, match="n_locations"):
        SpatialConditions.uniform(0, pH=7.0)


def test_n_locations_zero_for_empty():
    assert SpatialConditions().n_locations == 0


def test_with_rejects_wrong_length_array_override():
    """A 1-D override is used as-is, so it must match the object's location count;
    a wrong length is caught when the merged object is re-validated."""
    sc = SpatialConditions.uniform(3, pH=7.0)
    with pytest.raises(ValueError, match="length"):
        sc.with_(T=jnp.array([280.0, 285.0]))


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
    net = aquakin.load_model("asm1")
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


def test_condition_builders_are_ad_safe():
    """A *traced* condition value (a gradient w.r.t. an operating condition such
    as pH/T) must flow through the SpatialConditions builders instead of being
    float()-concretized -- which raised ConcretizationTypeError. Covers
    ``uniform``, ``with_`` and ``OperatingConditions``, including under jit."""
    import jax
    import numpy as np

    def field0(value, build):
        return build(value).fields["pH"][0]

    builders = {
        "uniform": lambda v: SpatialConditions.uniform(2, pH=v),
        "with_": lambda v: SpatialConditions.uniform(2, pH=7.0).with_(pH=v),
        "operating": lambda v: OperatingConditions(pH=v, T=293.15),
    }
    for name, build in builders.items():
        g = jax.grad(lambda v: field0(v, build))(jnp.asarray(7.5))
        assert np.isfinite(float(g)) and float(g) == pytest.approx(1.0), name
        # also composes under jit
        assert np.isfinite(float(jax.jit(jax.grad(lambda v: field0(v, build)))(
            jnp.asarray(7.5))))
        # value is unchanged for a concrete input
        assert float(build(8.0).fields["pH"][0]) == pytest.approx(8.0), name
