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
