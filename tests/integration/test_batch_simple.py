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


def test_max_steps_is_exposed_and_enforced(simple_network):
    # max_steps is now a constructor knob on BatchReactor (it used to be hardwired
    # to 100k in _run_diffeqsolve). A budget far too small must make the solve
    # raise; the generous default completes.
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    C0 = jnp.asarray([1.0, 0.0])
    params = simple_network.default_parameters()
    t_eval = jnp.linspace(0.0, 20.0, 21)

    r_tight = aquakin.BatchReactor(simple_network, conditions, max_steps=2)
    assert r_tight.max_steps == 2
    with pytest.raises(Exception):
        r_tight.solve(C0, params, t_span=(0.0, 20.0), t_eval=t_eval)

    r_ok = aquakin.BatchReactor(simple_network, conditions, max_steps=100_000)
    sol = r_ok.solve(C0, params, t_span=(0.0, 20.0), t_eval=t_eval)
    assert jnp.all(jnp.isfinite(sol.C))


def test_conditions_uniform_defaults_to_one_location():
    """SpatialConditions.uniform defaults n_locations=1 for the batch case."""
    c = aquakin.SpatialConditions.uniform(T=293.15, pH=7.5)
    assert c.n_locations == 1
    assert float(c.fields["T"][0]) == 293.15
    # Explicit n_locations still works.
    assert aquakin.SpatialConditions.uniform(3, T=293.15).n_locations == 3


def test_solve_defaults_params_to_network_defaults(simple_network):
    """reactor.solve(C0, t_span=...) uses network.default_parameters() when
    params is omitted, matching an explicit pass."""
    conditions = aquakin.SpatialConditions.uniform(T=293.15)
    reactor = aquakin.BatchReactor(simple_network, conditions)
    C0 = jnp.asarray([1.0, 0.0])
    t_eval = jnp.linspace(0.0, 20.0, 21)

    auto = reactor.solve(C0, t_span=(0.0, 20.0), t_eval=t_eval)
    explicit = reactor.solve(
        C0, simple_network.default_parameters(), t_span=(0.0, 20.0), t_eval=t_eval
    )
    assert jnp.allclose(auto.C, explicit.C)


def test_solve_requires_t_span(simple_network):
    """Omitting t_span (now that params is optional) is a clear error, not a
    silent misbinding."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(T=293.15)
    )
    with pytest.raises(ValueError, match="t_span"):
        reactor.solve(jnp.asarray([1.0, 0.0]))


def test_solve_tuple_as_params_is_caught(simple_network):
    """The footgun reactor.solve(C0, (0, 20)) -- a t_span tuple landing in the
    params slot -- is caught by the params shape check, not run silently."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(T=293.15)
    )
    with pytest.raises(ValueError, match="params has shape"):
        reactor.solve(jnp.asarray([1.0, 0.0]), (0.0, 20.0))


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
    atol = network.atol({"OH": 1e-20}, default=1e-12)
    reactor = aquakin.BatchReactor(network, conditions, atol=atol)
    C0 = network.concentrations({"Br-": 1e-5, "O3": 1e-4})
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


def test_solve_chemistry_helper_matches_analytical_and_scales(simple_network):
    """The shared _common.solve_chemistry factory (used by every reactor) must
    reproduce first-order decay, and rate_scale must linearly scale the rate
    (the PFR's 1/velocity device): a span of 2*T at scale 0.5 equals a span of
    T at scale 1."""
    import diffrax
    from aquakin.integrate._common import solve_chemistry

    net = simple_network
    cond = aquakin.SpatialConditions.uniform(1, T=293.15).fields
    C0 = jnp.asarray([1.0, 0.0])
    p = net.default_parameters()
    k = float(p[0])
    T = 8.0
    kw = dict(cond_fn=lambda t: cond, rtol=1e-8, atol=1e-10)

    sol = solve_chemistry(net, C0, p, saveat=diffrax.SaveAt(ts=jnp.asarray([T])),
                          t0=0.0, t1=T, **kw)
    assert float(sol.ys[-1, 0]) == pytest.approx(jnp.exp(-k * T), rel=1e-5)

    # rate_scale=0.5 over 2T reaches the same state as scale=1 over T.
    sol_scaled = solve_chemistry(net, C0, p, rate_scale=0.5,
                                 saveat=diffrax.SaveAt(ts=jnp.asarray([2 * T])),
                                 t0=0.0, t1=2 * T, **kw)
    assert float(sol_scaled.ys[-1, 0]) == pytest.approx(float(sol.ys[-1, 0]), rel=1e-5)


def test_batch_step_ceiling_gives_friendly_error():
    """A BatchReactor that hits the integrator step budget re-raises with a
    domain-level remedy (loosen rtol / raise max_steps), not a raw Equinox
    runtime traceback -- the catch works through the reactor's jitted solve."""
    net = aquakin.load_network_from_file("tests/fixtures/simple_network.yaml")
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    reactor = aquakin.BatchReactor(net, cond, max_steps=1)
    with pytest.raises(RuntimeError, match="step budget"):
        reactor.solve(net.concentrations(A=1.0), t_span=(0.0, 1000.0),
                      t_eval=jnp.linspace(0.0, 1000.0, 50))
