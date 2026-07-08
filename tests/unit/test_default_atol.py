"""Per-component absolute-tolerance builder (``default_atol``).

The solver uses a vector ``atol`` scaled to each state's typical magnitude
(SUNDIALS "vector atol" / Hairer "atol proportional to typical value"). Every
component of the returned floor must stay strictly positive -- a zero ``atol_i``
removes the noise floor on that component, which the solver literature warns
against.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.integrate._common import default_atol, resolve_layered_atol


def test_positive_for_normal_scales():
    atol = default_atol(jnp.asarray([1.0, 100.0, 5000.0]))
    assert jnp.all(atol > 0.0)


def test_all_zero_scale_falls_back_to_unit_floor():
    """When every magnitude is zero there is no scale to floor against, so the
    relative floor ``floor_frac*char`` would itself be 0. The unit-scale
    fallback keeps every component strictly positive."""
    atol = default_atol(jnp.zeros(5))
    assert jnp.all(atol > 0.0)
    # floor_frac (1e-6) * unit char (1.0) * atol_factor (1e-6) = 1e-12.
    assert jnp.allclose(atol, 1e-12)


def test_identity_for_nonzero_scale_input():
    """The fallback only fires when the characteristic scale is zero, so an
    input with any nonzero magnitude is unaffected by it."""
    scale = jnp.asarray([0.0, 2.0, 0.0])
    atol = default_atol(scale)
    # char = max = 2.0; floor = floor_frac*char = 2e-6; typ = max(|scale|, 2e-6)
    # = [2e-6, 2.0, 2e-6]; atol = atol_factor * typ.
    expected = 1e-6 * jnp.asarray([2e-6, 2.0, 2e-6])
    assert jnp.allclose(atol, expected)


def test_reference_raises_the_floor():
    """A per-species reference magnitude lifts the floor where the operating
    scale alone would be too small."""
    atol = default_atol(jnp.zeros(2), reference=jnp.asarray([10.0, 0.0]))
    # char = 10; species 0 floored at 10, species 1 at floor_frac*char = 1e-5.
    assert atol[0] == pytest.approx(1e-6 * 10.0)
    assert atol[1] == pytest.approx(1e-6 * 1e-5)


def test_atol_is_gradient_detached():
    # atol is a solver noise floor, never a differentiated quantity. It is
    # stop_gradient'd so a tolerance derived from a *traced* magnitude (e.g. the
    # initial state y0, when differentiating w.r.t. y0) carries no gradient and
    # cannot leak into the integrator's step controller (issue #420).
    import jax
    import numpy as np
    g = jax.grad(lambda y: jnp.sum(default_atol(y)))(jnp.asarray([1.0, 100.0, 5.0]))
    assert np.all(np.asarray(g) == 0.0)
    # ... and the value is unchanged (stop_gradient is identity on the value).
    val = default_atol(jnp.asarray([1.0, 100.0, 5.0]))
    assert np.all(np.isfinite(np.asarray(val))) and float(val[1]) == pytest.approx(1e-4)


# --- resolve_layered_atol: the layered (BiofilmReactor) atol resolver ---------


def test_layered_atol_none_tiles_per_component_floor():
    """atol=None yields the per-component default_atol tiled across compartments
    (shape (n_comp, n_species), matching the reactor's 2-D state), not a fixed
    scalar ~9 orders too tight for g/m3 states."""
    net = aquakin.load_model("asm1")
    n = net.n_species
    per_species = default_atol(net.default_concentrations())
    got = resolve_layered_atol(net, None, 4)
    assert got.shape == (4, n)
    assert jnp.allclose(got, jnp.tile(per_species, (4, 1)))
    # a g/m3 model's floor is far above the old fixed 1e-9 scalar
    assert float(got.max()) > 1e-9


def test_layered_atol_scalar_broadcasts_to_full_state():
    net = aquakin.load_model("asm1")
    n = net.n_species
    got = resolve_layered_atol(net, 1e-8, 3)
    assert got.shape == (3, n)
    assert jnp.allclose(got, 1e-8)


def test_layered_atol_per_species_array_is_tiled():
    net = aquakin.load_model("asm1")
    n = net.n_species
    per_species = jnp.arange(1.0, n + 1.0)
    got = resolve_layered_atol(net, per_species, 2)
    assert got.shape == (2, n)
    assert jnp.allclose(got, jnp.tile(per_species, (2, 1)))


def test_layered_atol_rejects_wrong_shape():
    net = aquakin.load_model("asm1")
    with pytest.raises(ValueError, match="atol must be a scalar"):
        resolve_layered_atol(net, jnp.ones(net.n_species + 1), 3)


def test_biofilm_reactor_default_atol_is_per_component():
    """The layered BiofilmReactor no longer defaults to the fixed 1e-9 scalar;
    it inherits the per-component floor over its (n_layers+1, n_species) state."""
    net = aquakin.load_model("asm1")
    reactor = aquakin.BiofilmReactor(
        net,
        aquakin.SpatialConditions.uniform(T=293.15),
        n_layers=3,
        thickness=8e-4,
        area_per_volume=50.0,
        diffusivity=1e-4,
        boundary_layer=1e-4,
    )
    assert reactor.atol.shape == (4, net.n_species)
    assert float(reactor.atol.min()) > 1e-9  # not the old too-tight scalar


# --- CompiledModel.atol: by-name per-species atol builder ---------------------


def test_model_atol_default_is_per_component():
    """net.atol() now starts from the per-component floor, not a uniform 1e-9."""
    net = aquakin.load_model("asm1")
    got = net.atol()
    assert got.shape == (net.n_species,)
    assert jnp.allclose(got, default_atol(net.default_concentrations()))
    # not the old uniform 1e-9 floor
    assert not jnp.allclose(got, 1e-9)


def test_model_atol_named_override_on_per_component_base():
    net = aquakin.load_model("asm1")
    got = net.atol({"SNH": 1e-15})
    assert float(got[net.species_index["SNH"]]) == pytest.approx(1e-15)
    # the rest keep the per-component floor
    base = default_atol(net.default_concentrations())
    keep = [i for s, i in net.species_index.items() if s != "SNH"]
    assert jnp.allclose(got[jnp.asarray(keep)], base[jnp.asarray(keep)])


def test_model_atol_explicit_scalar_default_still_uniform():
    net = aquakin.load_model("asm1")
    got = net.atol(default=1e-12)
    assert jnp.allclose(got, 1e-12)
