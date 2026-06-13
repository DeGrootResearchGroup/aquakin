"""Integration tests: PFR against analytical first-order decay along x."""

import jax
import jax.numpy as jnp
import pytest

import aquakin


def test_pfr_first_order_decay(simple_network):
    """For uniform conditions, PFR decay matches batch decay with t = x/v."""
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    velocity = 0.5  # m/s
    length = 5.0  # m -> residence time 10 s
    reactor = aquakin.PlugFlowReactor(
        simple_network,
        conditions,
        n_points=11,
        length=length,
        velocity=velocity,
    )
    C0 = jnp.asarray([1.0, 0.0])
    params = simple_network.default_parameters()
    k = float(params[0])

    sol = reactor.solve(C0, params=params)

    tau = sol.x / velocity
    analytical_A = jnp.exp(-k * tau)
    assert jnp.allclose(sol.C_named("A"), analytical_A, atol=1e-5, rtol=1e-4)


def test_pfr_inlet_matches_C0(simple_network):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.PlugFlowReactor(
        simple_network,
        conditions,
        n_points=5,
        length=2.0,
        velocity=1.0,
    )
    C0 = jnp.asarray([0.75, 0.25])
    sol = reactor.solve(C0, params=simple_network.default_parameters())
    assert float(sol.C[0, 0]) == 0.75
    assert float(sol.C[0, 1]) == 0.25


def test_pfr_grad_through_solve_finite(simple_network):
    """CLAUDE.md requires an AD-grad test for every integration test suite."""
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.PlugFlowReactor(
        simple_network, conditions, n_points=11, length=5.0, velocity=0.5
    )
    C0 = jnp.asarray([1.0, 0.0])

    def loss(params):
        sol = reactor.solve(C0, params=params)
        return jnp.sum(sol.C_named("B") ** 2)

    g = jax.grad(loss)(simple_network.default_parameters())
    assert jnp.all(jnp.isfinite(g))
    # d/dk should be positive (faster decay -> more B everywhere downstream).
    assert float(g[0]) > 0.0


def test_pfr_rejects_invalid_geometry(simple_network):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    with pytest.raises(ValueError):
        aquakin.PlugFlowReactor(simple_network, conditions, n_points=1, length=1.0, velocity=1.0)
    with pytest.raises(ValueError):
        aquakin.PlugFlowReactor(simple_network, conditions, n_points=5, length=-1.0, velocity=1.0)
    with pytest.raises(ValueError):
        aquakin.PlugFlowReactor(simple_network, conditions, n_points=5, length=1.0, velocity=0.0)


def test_pfr_conditions_override_shape_mismatch_rejected(simple_network):
    """If the override has a different n_locations the x_grid is invalid."""
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.PlugFlowReactor(
        simple_network, conditions, n_points=5, length=2.0, velocity=1.0
    )
    overlay = aquakin.SpatialConditions.uniform(3, T=293.15)
    with pytest.raises(ValueError):
        reactor.solve(
            jnp.asarray([1.0, 0.0]),
            params=simple_network.default_parameters(),
            conditions=overlay,
        )


def test_pfr_direct_adjoint_enables_forward_mode(simple_network):
    """With adjoint=DirectAdjoint the PFR solve is forward-mode differentiable
    (jacfwd), which the default RecursiveCheckpointAdjoint (a reverse-only
    custom_vjp) rejects. Closes the adjoint-asymmetry gap with BatchReactor."""
    import diffrax

    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.PlugFlowReactor(
        simple_network, conditions, n_points=5, length=10.0, velocity=1.0,
        adjoint=diffrax.DirectAdjoint(),
    )
    C0 = jnp.asarray([1.0, 0.0])

    def out(p):
        return jnp.sum(reactor.solve(C0, params=p).C)

    J = jax.jacfwd(out)(simple_network.default_parameters())
    assert jnp.all(jnp.isfinite(J))
