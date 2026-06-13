"""Integration tests for the built-in UV/H2O2 AOP network."""

import jax.numpy as jnp
import pytest

import aquakin


@pytest.fixture
def network():
    return aquakin.load_network("uv_h2o2")


def _atol_for(network):
    return network.atol({"OH": 1e-20}, default=1e-12)


def _solve(network, *, fluence_rate, t_end=600.0, n=61):
    conditions = aquakin.SpatialConditions.uniform(
        1, fluence_rate=fluence_rate, OH_scavenging=5.0e4
    )
    reactor = aquakin.BatchReactor(network, conditions, atol=_atol_for(network))
    return reactor.solve(
        network.default_concentrations(),
        params=network.default_parameters(),
        t_span=(0.0, t_end),
        t_eval=jnp.linspace(0.0, t_end, n),
    )


def test_network_shape(network):
    assert network.n_species == 4
    assert network.n_reactions == 4
    assert set(network.conditions_required) == {"fluence_rate", "OH_scavenging"}


def test_uv_on_decays_H2O2_and_target(network):
    sol = _solve(network, fluence_rate=1.0)
    h2o2 = sol.C_named("H2O2")
    target = sol.C_named("target")
    assert float(h2o2[-1]) < float(h2o2[0])
    assert float(target[-1]) < float(target[0])
    # Monotone non-increasing within tolerance.
    assert jnp.all(jnp.diff(h2o2) <= 1e-12)
    assert jnp.all(jnp.diff(target) <= 1e-12)


def test_uv_off_holds_steady(network):
    sol = _solve(network, fluence_rate=0.0)
    # Without photolysis there is no OH source, so H2O2 and target are conserved.
    assert float(sol.C_named("H2O2")[-1]) == pytest.approx(
        float(sol.C_named("H2O2")[0]), rel=1e-8
    )
    assert float(sol.C_named("target")[-1]) == pytest.approx(
        float(sol.C_named("target")[0]), rel=1e-8
    )


def test_higher_fluence_destroys_more_target(network):
    sol_low = _solve(network, fluence_rate=0.5)
    sol_high = _solve(network, fluence_rate=5.0)
    assert float(sol_high.C_named("target")[-1]) < float(sol_low.C_named("target")[-1])
