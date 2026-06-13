"""Reference warm-start helpers for the BSM plants.

These build a flat ``y0`` (no plant solve), so they run in the fast PR gate: the
helper must seed exactly the activated-sludge reactors with the reference
composition and leave every other unit at its default.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm import (
    BSM1_WARM_REACTOR_COMPOSITION,
    BSM2_WARM_REACTOR_COMPOSITION,
    bsm1_warm_start,
    bsm2_warm_start,
    bsm2_asm1_network,
    bsm2_constant_influent,
    build_bsm1,
    build_bsm2,
)

_TANKS = ("tank1", "tank2", "tank3", "tank4", "tank5")


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


def test_compositions_are_valid_asm1_species(asm1):
    """Every key is a real ASM1 species (so concentrations() accepts them)."""
    for comp in (BSM1_WARM_REACTOR_COMPOSITION, BSM2_WARM_REACTOR_COMPOSITION):
        for sp in comp:
            assert sp in asm1.species_index
        # Builds a finite concentration vector.
        assert jnp.all(jnp.isfinite(asm1.concentrations(comp)))


def test_bsm2_warm_start_matches_manual_seed():
    """bsm2_warm_start == the hand-written initial_state(overrides=...) seed."""
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1, adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")

    y0 = bsm2_warm_start(plant)
    warm = asm1.concentrations(BSM2_WARM_REACTOR_COMPOSITION)
    y0_manual = plant.initial_state(overrides={t: warm for t in _TANKS})
    assert jnp.allclose(y0, y0_manual)


def test_bsm2_warm_start_only_seeds_reactors():
    """Non-reactor units (digester, clarifiers, recycle units) keep their
    default initial state; only the five AS reactors are overridden."""
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1, adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")

    cold = plant.initial_state()
    warm_y0 = bsm2_warm_start(plant)
    cold_by_unit = plant.states_by_unit(cold)
    warm_by_unit = plant.states_by_unit(warm_y0)
    for name, vec in warm_by_unit.items():
        if name in _TANKS:
            assert not jnp.allclose(vec, cold_by_unit[name])  # overridden
        else:
            assert jnp.allclose(vec, cold_by_unit[name])       # untouched


def test_bsm1_warm_start_shape_and_finite(asm1):
    plant = build_bsm1(network=asm1)
    y0 = bsm1_warm_start(plant)
    assert y0.shape == (sum(u.state_size for u in plant.units.values()),)
    assert jnp.all(jnp.isfinite(y0))
    # Seeds the reactors with the reference biomass.
    warm = asm1.concentrations(BSM1_WARM_REACTOR_COMPOSITION)
    by_unit = plant.states_by_unit(y0)
    assert jnp.allclose(by_unit["tank3"], warm)


def test_warm_start_accepts_explicit_network(asm1):
    """An explicit asm1_network gives the same result as auto-detection."""
    plant = build_bsm1(network=asm1)
    assert jnp.allclose(bsm1_warm_start(plant),
                        bsm1_warm_start(plant, asm1_network=asm1))
