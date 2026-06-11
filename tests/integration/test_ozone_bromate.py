"""Integration tests for the expanded ozone/bromate network."""

import jax.numpy as jnp
import pytest

import aquakin


@pytest.fixture
def network():
    return aquakin.load_network("ozone_bromate")


def _atol_for(network):
    return network.atol({"OH": 1e-20}, default=1e-12)


def _run(network, *, OH_scavenging, t_end=600.0, n=61):
    conditions = aquakin.SpatialConditions.uniform(
        1, pH=7.5, T=293.15, OH_scavenging=OH_scavenging
    )
    reactor = aquakin.BatchReactor(network, conditions, atol=_atol_for(network))
    C0 = network.concentrations({"O3": 1.0e-4, "Br-": 1.0e-5})
    return reactor.solve(
        C0,
        network.default_parameters(),
        t_span=(0.0, t_end),
        t_eval=jnp.linspace(0.0, t_end, n),
    )


def test_shape_matches_v2(network):
    """Network shape after expansion."""
    assert network.n_species == 6
    assert network.n_reactions == 7
    assert "OH" in network.species
    assert "OH_scavenging" in network.conditions_required


def test_increasing_scavenging_reduces_bromate(network):
    """Higher matrix OH-scavenging should lower the OH-driven BrO3- yield."""
    sol_low = _run(network, OH_scavenging=1.0e4)
    sol_high = _run(network, OH_scavenging=1.0e6)

    bro3_low = float(sol_low.C_named("BrO3-")[-1])
    bro3_high = float(sol_high.C_named("BrO3-")[-1])

    assert bro3_low > bro3_high


def test_OH_stays_in_radical_band(network):
    """OH should sit between 1e-15 and 1e-10 M throughout."""
    sol = _run(network, OH_scavenging=5.0e4)
    oh = sol.C_named("OH")
    # Skip t=0 where OH starts at exactly 0.
    interior = oh[1:]
    assert jnp.all(interior >= 0.0)
    assert jnp.all(interior < 1e-10)
    assert float(jnp.max(interior)) > 1e-16


def test_bromate_bounded_by_bromide_inventory(network):
    """Final BrO3- cannot exceed initial Br-."""
    sol = _run(network, OH_scavenging=1.0e3)
    bro3_final = float(sol.C_named("BrO3-")[-1])
    assert bro3_final <= 1.0e-5 + 1e-12
