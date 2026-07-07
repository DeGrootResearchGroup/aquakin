"""Fast-tier coverage for :class:`aquakin.BiofilmReactor`.

The thorough biofilm suite (`tests/integration/test_biofilm.py`) is blanket
``@pytest.mark.slow``, so none of it runs on the PR gate -- yet a tiny 2-layer
solve is ~sub-second. These fast, no-frills tests cover the constructor
validation, the solve/`t_eval` guards, the reactive-particulate warning, an
AD-through-solve check, the well-mixed limit, and the result accessors
(including the biofilm-specific ``to_dataframe(profile=True)``). The expensive
paths (`solve_sensitivity`, `steady_state`) stay in the slow suite.
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def net():
    return aquakin.load_model_from_file(_FIXTURES / "simple_model.yaml")


@pytest.fixture()
def cond():
    return aquakin.SpatialConditions.uniform(T=293.15)


def _tiny(net, cond, **kw):
    """A minimal 2-layer biofilm reactor; override any field via kw."""
    d = dict(
        n_layers=2,
        thickness=8e-4,
        area_per_volume=50.0,
        diffusivity=1e-4,
        boundary_layer=1e-4,
    )
    d.update(kw)
    return aquakin.BiofilmReactor(net, cond, **d)


def test_64bit_precision_enabled():
    assert jax.config.x64_enabled


def test_build_and_solve_smoke(net, cond):
    r = _tiny(net, cond)
    sol = r.solve(jnp.array([1.0, 0.0]), t_span=(0.0, 1.0))
    assert isinstance(sol, aquakin.BiofilmSolution)
    assert sol.C.shape == (1, net.n_species)
    assert sol.profile.shape == (1, r.n_layers + 1, net.n_species)
    assert sol.depth.shape == (r.n_layers,)
    assert np.all(np.isfinite(np.asarray(sol.C)))


def test_solve_t_eval_shapes_and_accessors(net, cond):
    r = _tiny(net, cond)
    t_eval = jnp.linspace(0.0, 1.0, 3)
    sol = r.solve(jnp.array([1.0, 0.0]), t_span=(0.0, 1.0), t_eval=t_eval)
    assert sol.C.shape == (3, net.n_species)
    assert sol.profile.shape == (3, r.n_layers + 1, net.n_species)
    assert sol.profile_named("A").shape == (3, r.n_layers + 1)
    # Bulk (C) is compartment 0 of the profile.
    assert np.allclose(np.asarray(sol.C), np.asarray(sol.profile[:, 0, :]))
    assert np.allclose(np.asarray(sol.C_named("A")), np.asarray(sol.profile_named("A")[:, 0]))


def test_c0_broadcast_and_full_profile_accepted(net, cond):
    r = _tiny(net, cond)
    n_comp = r.n_layers + 1
    sol_bc = r.solve(jnp.zeros(net.n_species), t_span=(0.0, 0.5))
    sol_full = r.solve(jnp.zeros((n_comp, net.n_species)), t_span=(0.0, 0.5))
    assert sol_bc.profile.shape[1:] == (n_comp, net.n_species)
    assert sol_full.profile.shape[1:] == (n_comp, net.n_species)


def test_solve_rejects_bad_c0_shape(net, cond):
    r = _tiny(net, cond)
    with pytest.raises(ValueError, match="C0 has shape"):
        r.solve(jnp.zeros(net.n_species + 1), t_span=(0.0, 1.0))


def test_solve_rejects_bad_params_shape(net, cond):
    r = _tiny(net, cond)
    with pytest.raises(ValueError, match="params has shape"):
        r.solve(jnp.zeros(net.n_species), params=jnp.zeros(net.n_params + 1), t_span=(0.0, 1.0))


def test_solve_requires_t_span(net, cond):
    r = _tiny(net, cond)
    with pytest.raises(ValueError, match="t_span"):
        r.solve(jnp.array([1.0, 0.0]))


def test_solve_rejects_nonincreasing_t_span(net, cond):
    r = _tiny(net, cond)
    with pytest.raises(ValueError, match="end must exceed start"):
        r.solve(jnp.array([1.0, 0.0]), t_span=(1.0, 0.0))


def test_solve_rejects_out_of_span_t_eval(net, cond):
    r = _tiny(net, cond)
    with pytest.raises(ValueError, match="within t_span"):
        r.solve(jnp.array([1.0, 0.0]), t_span=(0.0, 1.0), t_eval=jnp.array([0.0, 2.0]))


@pytest.mark.parametrize(
    "kw",
    [
        dict(n_layers=0),
        dict(thickness=-1.0),
        dict(area_per_volume=0.0),
        dict(boundary_layer=0.0),
    ],
)
def test_constructor_rejects_bad_geometry(net, cond, kw):
    with pytest.raises(ValueError):
        _tiny(net, cond, **kw)


@pytest.mark.parametrize("mask_name", ["soluble_mask", "fixed_mask", "attach_mask", "detach_mask"])
def test_constructor_rejects_wrong_mask_shape(net, cond, mask_name):
    with pytest.raises(ValueError, match="must have shape"):
        _tiny(net, cond, **{mask_name: jnp.array([True])})


def test_unknown_biofilm_reaction_name_raises(net, cond):
    with pytest.raises(ValueError, match="Unknown biofilm reaction"):
        _tiny(net, cond, biofilm_reactions=["not_a_reaction"])


def test_biofilm_reactions_mask_wrong_shape_raises(net, cond):
    with pytest.raises(ValueError, match="biofilm_reactions mask must have shape"):
        _tiny(net, cond, biofilm_reactions=jnp.array([True, False, True]))


def test_default_fixed_mask_warns_on_reactive_particulate(net, cond):
    # Marking B a (reactive) particulate with no explicit fixed_mask should warn.
    with pytest.warns(UserWarning, match="reactive particulate"):
        _tiny(net, cond, soluble_mask=jnp.array([True, False]))


def test_grad_flows_through_solve(net, cond):
    r = _tiny(net, cond)
    ki = net.param_index["A_to_B.k"]
    ai = net.species_index["A"]
    y0 = jnp.zeros((r.n_layers + 1, net.n_species)).at[0, ai].set(1.0)

    def final_bulk_A(k):
        p = net.default_parameters().at[ki].set(k)
        return r.solve(y0, params=p, t_span=(0.0, 1.0)).C_named("A")[-1]

    g = jax.grad(final_bulk_A)(0.1)
    assert np.isfinite(float(g))
    assert float(g) < 0.0  # more decay with larger k -> lower final bulk A


def test_well_mixed_limit_matches_batch(net, cond):
    # A uniform IC with uniform kinetics has zero transport, so the bulk follows
    # the batch decay exactly.
    r = _tiny(net, cond)
    batch = aquakin.BatchReactor(net, cond)
    C0 = jnp.array([1.0, 0.0])
    t_eval = jnp.linspace(0.0, 5.0, 6)
    bio = r.solve(C0, t_span=(0.0, 5.0), t_eval=t_eval)
    bat = batch.solve(C0, t_span=(0.0, 5.0), t_eval=t_eval, params=net.default_parameters())
    assert np.allclose(
        np.asarray(bio.C_named("A")), np.asarray(bat.C_named("A")), rtol=1e-5, atol=1e-7
    )


def test_profile_named_unknown_species_raises(net, cond):
    r = _tiny(net, cond)
    sol = r.solve(jnp.array([1.0, 0.0]), t_span=(0.0, 1.0))
    with pytest.raises(KeyError):
        sol.profile_named("ZZZ")


def test_to_dataframe_bulk(net, cond):
    pytest.importorskip("pandas")
    r = _tiny(net, cond)
    sol = r.solve(jnp.array([1.0, 0.0]), t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 3))
    df = sol.to_dataframe()
    assert list(df.columns) == ["A", "B"]
    assert df.index.name == "t"
    assert len(df) == 3


def test_to_dataframe_profile(net, cond):
    pytest.importorskip("pandas")
    r = _tiny(net, cond)
    sol = r.solve(jnp.array([1.0, 0.0]), t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 3))
    pdf = sol.to_dataframe(profile=True)
    assert list(pdf.index.names) == ["t", "compartment"]
    assert "depth" in pdf.columns
    assert len(pdf) == 3 * (r.n_layers + 1)
