"""Integration tests for the layered biofilm reactor (diffusion--reaction).

Uses only the toy two-species decay fixture (A -> B). The biofilm reactor adds
1-D diffusion and a bulk boundary layer on top of the per-compartment chemistry,
so these check the transport operator (conservation, the well-mixed limit, and
diffusion limitation) and that gradients flow through the layered solve.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "simple_network.yaml"
)


@pytest.fixture(scope="module")
def net():
    return aquakin.load_network_from_file(FIXTURE)


@pytest.fixture
def cond():
    return aquakin.SpatialConditions.uniform(1, T=293.15)


def _reactor(net, cond, **kw):
    defaults = dict(
        n_layers=6, thickness=8e-4, area_per_volume=50.0,
        diffusivity=1e-4, boundary_layer=1e-4,
    )
    defaults.update(kw)
    return aquakin.BiofilmReactor(net, cond, **defaults)


def test_pure_diffusion_conserves_and_equilibrates(net, cond):
    # With the reaction switched off (k=0), diffusion alone must conserve the
    # volume-weighted total and relax every compartment to a common value.
    r = _reactor(net, cond, diffusivity=1e-3)
    p = net.default_parameters().at[net.param_index["A_to_B.k"]].set(0.0)
    n_comp = r.n_layers + 1
    # bulk full of A, biofilm empty.
    y0 = jnp.zeros((n_comp, net.n_species))
    y0 = y0.at[0, net.species_index["A"]].set(1.0)
    sol = r.solve(y0, p, t_span=(0.0, 2.0))
    final = np.asarray(sol.profile[-1])  # (n_comp, n_species)
    a = final[:, net.species_index["A"]]
    # all compartments equal at steady state
    assert np.ptp(a) < 1e-4
    # volume-weighted A conserved: bulk weight 1/A_V, each layer weight dz
    dz = r.thickness / r.n_layers
    w = np.array([1.0 / r.area_per_volume] + [dz] * r.n_layers)
    total0 = (1.0 / r.area_per_volume) * 1.0
    total1 = float(np.sum(w * a))
    assert total1 == pytest.approx(total0, rel=1e-4)


def test_boundary_diffusivity_sets_transfer_and_defaults(net, cond):
    # The external boundary layer is liquid: its mass-transfer coefficient k_L
    # must come from the supplied (free-water) boundary diffusivity, not the
    # density-reduced in-biofilm value. Passing it equal to ``diffusivity``
    # reproduces the default (None) behaviour; a larger value speeds bulk<->film
    # exchange.
    d_film = 1e-4
    r_default = _reactor(net, cond, diffusivity=d_film)
    r_equal = _reactor(net, cond, diffusivity=d_film, boundary_diffusivity=d_film)
    r_fast = _reactor(net, cond, diffusivity=d_film, boundary_diffusivity=10 * d_film)
    # k_L = D_bl / boundary_layer, applied to solubles only.
    sidx = net.species_index["A"]  # A is soluble in the toy network
    assert float(r_default._kL[sidx]) == pytest.approx(float(r_equal._kL[sidx]))
    assert float(r_fast._kL[sidx]) == pytest.approx(10 * float(r_default._kL[sidx]))

    # A faster boundary layer equilibrates the bulk with the film sooner. Bulk
    # full, film empty, reaction off: the bulk drains faster with the larger k_L.
    p = net.default_parameters().at[net.param_index["A_to_B.k"]].set(0.0)
    n_comp = r_default.n_layers + 1
    y0 = jnp.zeros((n_comp, net.n_species)).at[0, sidx].set(1.0)
    bulk_default = float(r_default.solve(y0, p, t_span=(0.0, 0.5)).C_named("A")[-1])
    bulk_fast = float(r_fast.solve(y0, p, t_span=(0.0, 0.5)).C_named("A")[-1])
    assert bulk_fast < bulk_default


def test_reactive_particulate_evolves_and_conserves(net, cond):
    # A reactive particulate does not diffuse (soluble_mask False) but must still
    # react (fixed_mask False) -- the two roles are decoupled. Freezing such a
    # species (the old default) turns it into a non-depleting source/sink and
    # breaks mass balance. Here B is made a non-diffusing reactive product of
    # A -> B; with B reactive it accumulates, frozen it cannot.
    ia, ib = net.species_index["A"], net.species_index["B"]
    soluble = jnp.array([True, False])  # A diffuses, B does not
    p = net.default_parameters()  # A_to_B.k > 0
    n_comp = cond_layers = 5
    y0 = jnp.zeros((n_comp, net.n_species)).at[:, ia].set(1.0)

    reactive = _reactor(net, cond, n_layers=n_comp - 1,
                        soluble_mask=soluble, fixed_mask=jnp.array([False, False]))
    frozen = _reactor(net, cond, n_layers=n_comp - 1,
                      soluble_mask=soluble, fixed_mask=jnp.array([False, True]))
    sr = reactive.solve(y0, p, t_span=(0.0, 1.0))
    sf = frozen.solve(y0, p, t_span=(0.0, 1.0))
    # Reactive B grows from zero; frozen B stays put.
    assert float(sr.profile[-1, 0, ib]) > 0.05
    assert float(sf.profile[-1, 0, ib]) == pytest.approx(0.0, abs=1e-12)
    # A -> B is 1:1, so the volume-weighted total A+B is conserved when B reacts.
    dz = reactive.thickness / reactive.n_layers
    w = np.array([1.0 / reactive.area_per_volume] + [dz] * reactive.n_layers)
    tot0 = float(np.sum(w * np.asarray(y0[:, ia] + y0[:, ib])))  # from the IC
    tot1 = float(np.sum(w * (np.asarray(sr.profile[-1, :, ia])
                             + np.asarray(sr.profile[-1, :, ib]))))
    assert tot1 == pytest.approx(tot0, rel=1e-5)


def test_well_mixed_limit_matches_batch(net, cond):
    # Started uniform with a uniform reaction, every compartment stays identical
    # and transport is zero, so the bulk must follow the batch first-order decay.
    r = _reactor(net, cond)
    batch = aquakin.BatchReactor(net, cond)
    C0 = jnp.array([1.0, 0.0])  # A, B
    p = net.default_parameters()
    t_eval = jnp.linspace(0.0, 5.0, 11)
    bio = r.solve(C0, p, t_span=(0.0, 5.0), t_eval=t_eval)  # uniform IC broadcast
    bat = batch.solve(C0, p, t_span=(0.0, 5.0), t_eval=t_eval)
    assert np.allclose(np.asarray(bio.C_named("A")), np.asarray(bat.C_named("A")),
                       rtol=1e-5, atol=1e-7)


def test_diffusion_limitation_starves_deep_layers(net, cond):
    # Bulk starts full, biofilm empty, reaction on. With fast diffusion A
    # penetrates and fills the deepest layer; with slow diffusion it is consumed
    # in the outer layers before reaching the wall, so the deepest layer stays
    # starved. This depth dependence is the whole point of resolving the biofilm.
    p = net.default_parameters()  # slow reaction (k=0.1): bulk stays ~full
    n_comp = 7
    y0 = jnp.zeros((n_comp, net.n_species)).at[0, net.species_index["A"]].set(1.0)
    fast = _reactor(net, cond, diffusivity=1e-3)
    slow = _reactor(net, cond, diffusivity=1e-6)
    # Short time: fast diffusion (timescale L^2/D ~ 6e-4 d) fills the deepest
    # layer; slow diffusion (timescale ~0.6 d) has barely penetrated.
    deep_fast = float(fast.solve(y0, p, t_span=(0.0, 0.1)).profile_named("A")[-1, -1])
    deep_slow = float(slow.solve(y0, p, t_span=(0.0, 0.1)).profile_named("A")[-1, -1])
    assert deep_fast > deep_slow + 0.3


def test_phase_mask_confines_reaction_to_biofilm(net, cond):
    # With A_to_B marked biofilm-only, the bulk compartment has no reaction, so
    # the (large) bulk A pool falls only by diffusion into the (small) reacting
    # biofilm -- much slower than when the reaction also runs in the bulk. And
    # bulk B is then produced only in the layers, so it lags.
    p = net.default_parameters()
    C0 = jnp.array([1.0, 0.0])  # uniform A everywhere
    everywhere = _reactor(net, cond)                       # runs in every compartment
    film_only = _reactor(net, cond, biofilm_reactions=["A_to_B"])
    se = everywhere.solve(C0, p, t_span=(0.0, 2.0))
    sf = film_only.solve(C0, p, t_span=(0.0, 2.0))
    assert float(sf.C_named("A")[-1]) > float(se.C_named("A")[-1]) + 1e-2
    assert float(sf.C_named("B")[-1]) < float(se.C_named("B")[-1])


def test_unknown_biofilm_reaction_name_raises(net, cond):
    with pytest.raises(ValueError, match="Unknown biofilm reaction"):
        _reactor(net, cond, biofilm_reactions=["not_a_reaction"])


def test_solution_shapes_and_named_accessors(net, cond):
    r = _reactor(net, cond)
    p = net.default_parameters()
    t_eval = jnp.linspace(0.0, 1.0, 5)
    sol = r.solve(jnp.array([1.0, 0.0]), p, t_span=(0.0, 1.0), t_eval=t_eval)
    assert sol.C.shape == (5, net.n_species)               # bulk trajectory
    assert sol.profile.shape == (5, r.n_layers + 1, net.n_species)
    assert sol.depth.shape == (r.n_layers,)
    assert sol.profile_named("A").shape == (5, r.n_layers + 1)
    # bulk is compartment 0 of the profile
    assert np.allclose(np.asarray(sol.C), np.asarray(sol.profile[:, 0, :]))


def test_grad_flows_through_layered_solve(net, cond):
    # jax.grad must flow through the diffusion-reaction solve without NaNs.
    r = _reactor(net, cond)
    ki = net.param_index["A_to_B.k"]
    n_comp = r.n_layers + 1
    y0 = jnp.zeros((n_comp, net.n_species)).at[0, net.species_index["A"]].set(1.0)

    def final_bulk_A(k):
        p = net.default_parameters().at[ki].set(k)
        return r.solve(y0, p, t_span=(0.0, 1.0)).C_named("A")[-1]

    g = jax.grad(final_bulk_A)(0.1)
    assert np.isfinite(float(g))
    # more decay with larger k -> final bulk A decreases
    assert float(g) < 0.0


def test_64bit_precision_enabled():
    import jax
    assert jax.config.x64_enabled
