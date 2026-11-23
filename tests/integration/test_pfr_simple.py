"""Integration tests: PFR against analytical first-order decay along x."""

import jax
import jax.numpy as jnp
import pytest

import aquakin


def test_pfr_first_order_decay(simple_model):
    """For uniform conditions, PFR decay matches batch decay with t = x/v."""
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    velocity = 0.5  # m/s
    length = 5.0  # m -> residence time 10 s
    reactor = aquakin.PlugFlowReactor(
        simple_model,
        conditions,
        n_points=11,
        length=length,
        velocity=velocity,
    )
    C0 = jnp.asarray([1.0, 0.0])
    params = simple_model.default_parameters()
    k = float(params[0])

    sol = reactor.solve(C0, params=params)

    tau = sol.x / velocity
    analytical_A = jnp.exp(-k * tau)
    assert jnp.allclose(sol.C_named("A"), analytical_A, atol=1e-5, rtol=1e-4)


def test_pfr_inlet_matches_C0(simple_model):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.PlugFlowReactor(
        simple_model,
        conditions,
        n_points=5,
        length=2.0,
        velocity=1.0,
    )
    C0 = jnp.asarray([0.75, 0.25])
    sol = reactor.solve(C0, params=simple_model.default_parameters())
    assert float(sol.C[0, 0]) == 0.75
    assert float(sol.C[0, 1]) == 0.25


def test_pfr_grad_through_solve_matches_closed_form(simple_model):
    """``jax.grad`` through the PFR solve gives the closed-form gradient.

    With ``B(x) = 1 - exp(-k tau)``, ``tau = x / v``, the loss
    ``sum_x B(x)**2`` has ``dL/dk = sum_x 2 B(x) tau exp(-k tau)``.
    """
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.PlugFlowReactor(
        simple_model, conditions, n_points=11, length=5.0, velocity=0.5
    )
    C0 = jnp.asarray([1.0, 0.0])

    def loss(params):
        sol = reactor.solve(C0, params=params)
        return jnp.sum(sol.C_named("B") ** 2)

    params = simple_model.default_parameters()
    g = jax.grad(loss)(params)
    tau = reactor.solve(C0, params=params).x / 0.5
    decay = jnp.exp(-float(params[0]) * tau)
    expected = float(jnp.sum(2.0 * (1.0 - decay) * tau * decay))
    # The forward solve matches the closed form to rtol 1e-4 (see above); the
    # gradient through the same adaptive solve is held a decade looser.
    assert float(g[0]) == pytest.approx(expected, rel=1e-3)


def test_pfr_rejects_invalid_geometry(simple_model):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    with pytest.raises(ValueError, match="n_points must be >= 2"):
        aquakin.PlugFlowReactor(simple_model, conditions, n_points=1, length=1.0, velocity=1.0)
    with pytest.raises(ValueError, match="length must be positive"):
        aquakin.PlugFlowReactor(simple_model, conditions, n_points=5, length=-1.0, velocity=1.0)
    with pytest.raises(ValueError, match="velocity must be positive"):
        aquakin.PlugFlowReactor(simple_model, conditions, n_points=5, length=1.0, velocity=0.0)


def test_pfr_conditions_override_shape_mismatch_rejected(simple_model):
    """If the override has a different n_locations the x_grid is invalid."""
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.PlugFlowReactor(
        simple_model, conditions, n_points=5, length=2.0, velocity=1.0
    )
    overlay = aquakin.SpatialConditions.uniform(3, T=293.15)
    with pytest.raises(ValueError, match="conditions override must have n_locations=1"):
        reactor.solve(
            jnp.asarray([1.0, 0.0]),
            params=simple_model.default_parameters(),
            conditions=overlay,
        )


def test_pfr_direct_adjoint_enables_forward_mode(simple_model):
    """With adjoint=DirectAdjoint the PFR solve is forward-mode differentiable
    (jacfwd), which the default RecursiveCheckpointAdjoint (a reverse-only
    custom_vjp) rejects. Closes the adjoint-asymmetry gap with BatchReactor.

    The output is the summed product ``sum_x B(x)``, whose closed-form derivative
    is ``sum_x tau exp(-k tau)`` with ``tau = x / v``."""

    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.PlugFlowReactor(
        simple_model, conditions, n_points=5, length=10.0, velocity=1.0,
        diff=aquakin.DifferentiationConfig(mode="forward", method="through_solve"),
    )
    C0 = jnp.asarray([1.0, 0.0])

    def out(p):
        return jnp.sum(reactor.solve(C0, params=p).C_named("B"))

    params = simple_model.default_parameters()
    J = jax.jacfwd(out)(params)
    tau = reactor.solve(C0, params=params).x / 1.0
    expected = float(jnp.sum(tau * jnp.exp(-float(params[0]) * tau)))
    assert float(J[0]) == pytest.approx(expected, rel=1e-3)
