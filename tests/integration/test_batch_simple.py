"""Integration tests: batch reactor against analytical first-order decay."""

import jax
import jax.numpy as jnp
import pytest

import aquakin


def test_x64_enabled():
    assert jax.config.x64_enabled


def test_first_order_decay_matches_analytical(simple_network):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(simple_network, conditions)
    C0 = jnp.asarray([1.0, 0.0])
    params = simple_network.default_parameters()
    k = float(params[0])
    t_eval = jnp.linspace(0.0, 20.0, 21)

    sol = reactor.solve(C0, params, t_span=(0.0, 20.0), t_eval=t_eval)

    analytical_A = jnp.exp(-k * t_eval)
    analytical_B = 1.0 - analytical_A

    assert jnp.allclose(sol.C_named("A"), analytical_A, atol=1e-5, rtol=1e-4)
    assert jnp.allclose(sol.C_named("B"), analytical_B, atol=1e-5, rtol=1e-4)


def test_grad_through_solve_finite(simple_network):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(simple_network, conditions)
    C0 = jnp.asarray([1.0, 0.0])

    def loss(params):
        sol = reactor.solve(C0, params, t_span=(0.0, 10.0), t_eval=jnp.linspace(0.0, 10.0, 11))
        return jnp.sum(sol.C_named("B") ** 2)

    g = jax.grad(loss)(simple_network.default_parameters())
    assert jnp.all(jnp.isfinite(g))
    # d/dk should be positive (increasing k drives more B).
    assert float(g[0]) > 0.0


def test_C0_shape_validated(simple_network):
    reactor = aquakin.BatchReactor(simple_network, aquakin.SpatialConditions.uniform(1, T=293.15))
    with pytest.raises(ValueError):
        reactor.solve(jnp.asarray([1.0]), simple_network.default_parameters(), (0.0, 1.0))


def test_t_span_must_be_ascending(simple_network):
    reactor = aquakin.BatchReactor(simple_network, aquakin.SpatialConditions.uniform(1, T=293.15))
    with pytest.raises(ValueError):
        reactor.solve(jnp.asarray([1.0, 0.0]), simple_network.default_parameters(), (1.0, 1.0))
    with pytest.raises(ValueError):
        reactor.solve(jnp.asarray([1.0, 0.0]), simple_network.default_parameters(), (2.0, 1.0))


def test_t_eval_out_of_span_rejected(simple_network):
    reactor = aquakin.BatchReactor(simple_network, aquakin.SpatialConditions.uniform(1, T=293.15))
    C0, p = jnp.asarray([1.0, 0.0]), simple_network.default_parameters()
    with pytest.raises(ValueError):   # below t0
        reactor.solve(C0, p, (0.0, 10.0), t_eval=jnp.asarray([-1.0, 5.0]))
    with pytest.raises(ValueError):   # above t1
        reactor.solve(C0, p, (0.0, 10.0), t_eval=jnp.asarray([5.0, 11.0]))


def test_t_eval_must_be_ascending(simple_network):
    reactor = aquakin.BatchReactor(simple_network, aquakin.SpatialConditions.uniform(1, T=293.15))
    C0, p = jnp.asarray([1.0, 0.0]), simple_network.default_parameters()
    with pytest.raises(ValueError):   # not ascending
        reactor.solve(C0, p, (0.0, 10.0), t_eval=jnp.asarray([5.0, 2.0, 8.0]))
    with pytest.raises(ValueError):   # repeated (not strictly ascending)
        reactor.solve(C0, p, (0.0, 10.0), t_eval=jnp.asarray([2.0, 2.0]))


def test_t_eval_valid_accepted(simple_network):
    reactor = aquakin.BatchReactor(simple_network, aquakin.SpatialConditions.uniform(1, T=293.15))
    C0, p = jnp.asarray([1.0, 0.0]), simple_network.default_parameters()
    sol = reactor.solve(C0, p, (0.0, 10.0), t_eval=jnp.linspace(0.0, 10.0, 6))
    assert jnp.all(jnp.isfinite(sol.C))


def test_uniform_rejects_zero_locations():
    with pytest.raises(ValueError):
        aquakin.SpatialConditions.uniform(0, T=293.15)


def test_missing_required_condition_rejected(simple_network):
    # simple_network requires 'T'; passing nothing must fail.
    with pytest.raises(ValueError):
        aquakin.BatchReactor(simple_network, aquakin.SpatialConditions(fields={}))


def test_ozone_bromate_runs():
    network = aquakin.load_network("ozone_bromate")
    conditions = aquakin.SpatialConditions.uniform(
        1, pH=7.5, T=293.15, OH_scavenging=5.0e4
    )
    atol = jnp.full((network.n_species,), 1e-12)
    atol = atol.at[network.species_index["OH"]].set(1e-20)
    reactor = aquakin.BatchReactor(network, conditions, atol=atol)
    C0 = network.default_concentrations()
    C0 = C0.at[network.species_index["Br-"]].set(1e-5)
    C0 = C0.at[network.species_index["O3"]].set(1e-4)
    sol = reactor.solve(
        C0,
        network.default_parameters(),
        t_span=(0.0, 600.0),
        t_eval=jnp.linspace(0.0, 600.0, 11),
    )
    # Ozone should decrease monotonically.
    o3 = sol.C_named("O3")
    assert jnp.all(jnp.diff(o3) <= 1e-12)
    # Bromate should be non-negative.
    bro3 = sol.C_named("BrO3-")
    assert jnp.all(bro3 >= -1e-15)
