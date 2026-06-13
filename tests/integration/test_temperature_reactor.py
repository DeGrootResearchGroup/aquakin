"""Integration: parameter temperature correction flows through a reactor."""

import jax.numpy as jnp
import pytest

import aquakin


@pytest.fixture
def asm1():
    return aquakin.load_network("asm1")


def _batch_final(asm1, T_kelvin):
    """Run a short ASM1 batch (substrate + biomass, aerobic) at temperature T,
    holding DO via a large initial SO, and return the final state."""
    C0 = asm1.concentrations(SS=50.0, SO=8.0, SNH=20.0, XB_H=800.0)
    cond = aquakin.SpatialConditions.uniform(T=T_kelvin)
    reactor = aquakin.BatchReactor(asm1, cond)
    sol = reactor.solve(C0, params=asm1.default_parameters(), t_span=(0.0, 0.5))
    return sol.C[-1]


def test_batch_slower_in_the_cold(asm1):
    """At 10 °C the heterotrophs consume substrate more slowly than at 20 °C, so
    more readily-biodegradable SS remains -- the temperature correction reaches
    the kinetics through the reactor's T condition."""
    ss = asm1.species_index["SS"]
    SS_20 = float(_batch_final(asm1, 293.15)[ss])
    SS_10 = float(_batch_final(asm1, 283.15)[ss])
    assert SS_10 > SS_20 + 1.0   # slower consumption -> more SS left in the cold


def test_batch_matches_uncorrected_at_reference(asm1):
    """At the reference 20 °C the corrected run equals an explicitly-uncorrected
    network (temperature is unity at ref_T)."""
    # Build an ASM1 with the corrections stripped, compare final states at 20 °C.
    import dataclasses
    bare = dataclasses.replace(asm1, temperature_corrections=[])
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    C0 = asm1.concentrations(SS=50.0, SO=8.0, XB_H=800.0)
    a = aquakin.BatchReactor(asm1, cond).solve(C0, (0.0, 0.5), params=asm1.default_parameters())
    b = aquakin.BatchReactor(bare, cond).solve(C0, (0.0, 0.5), params=bare.default_parameters())
    assert jnp.allclose(a.C[-1], b.C[-1], rtol=1e-8, atol=1e-8)
