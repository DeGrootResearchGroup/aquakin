"""Plant.set_temperature and the influent/plant network-mismatch error."""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm import (
    build_bsm1,
    build_bsm2,
    bsm2_asm1_network,
    bsm2_constant_influent,
)
from aquakin.plant.influent import InfluentSeries


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


# ----- set_temperature (construction only; fast) --------------------------

def test_set_temperature_sets_all_reactors_in_kelvin(asm1):
    plant = build_bsm1(asm1)
    ret = plant.set_temperature(10.0)
    assert ret is plant  # chainable
    for i in range(1, 6):
        assert plant.units[f"tank{i}"].conditions["T"] == pytest.approx(283.15)
        # the precomputed rate-evaluation array tracks it too
        assert float(plant.units[f"tank{i}"]._condition_arrays["T"][0]) == \
            pytest.approx(283.15)


def test_set_temperature_clears_compiled_cache(asm1):
    plant = build_bsm1(asm1)
    plant.add_influent("feed", asm1.influent({"SS": 60.0}, Q=18446.0))
    plant._jit_cache["sentinel"] = object()   # pretend a solve was compiled
    plant.set_temperature(15.0)
    assert plant._jit_cache == {}


def test_set_temperature_leaves_heated_digester_untouched():
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1, adm1)
    dig_T = plant.units["digester"].conditions["T"]
    plant.set_temperature(12.0)
    assert plant.units["tank1"].conditions["T"] == pytest.approx(285.15)
    assert plant.units["digester"].conditions["T"] == dig_T  # unchanged (heated)


def test_set_temperature_explicit_units(asm1):
    plant = build_bsm1(asm1)
    plant.set_temperature(14.0, units=["tank1", "tank2"])
    assert plant.units["tank1"].conditions["T"] == pytest.approx(287.15)
    assert plant.units["tank3"].conditions["T"] != pytest.approx(287.15)  # not set


def test_set_temperature_rejects_bad_units(asm1):
    plant = build_bsm1(asm1)
    with pytest.raises(ValueError, match="Unknown unit"):
        plant.set_temperature(10.0, units=["nope"])
    with pytest.raises(ValueError, match="does not support"):
        plant.set_temperature(10.0, units=["clarifier"])  # a separator, no T


# ----- network-mismatch error --------------------------------------------

def test_influent_network_instance_mismatch_is_clear():
    """A different *instance* of the same model gives a 'reuse one object'
    error, not the misleading 'supply a translator'."""
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(bsm2_asm1_network(), adm1)
    with pytest.raises(ValueError, match="different \\*instances\\* of the same"):
        # bsm2_asm1_network() builds a fresh instance each call.
        plant.add_influent("feed", bsm2_constant_influent(bsm2_asm1_network()))


def test_genuine_cross_network_still_asks_for_translator(asm1):
    """An influent of a genuinely different model still raises the
    cross-networks/translator error (the instances message does not apply)."""
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm1(asm1)
    bad = InfluentSeries(
        t=jnp.array([0.0, 1.0]), Q=jnp.array([1.0, 1.0]),
        C=jnp.tile(adm1.default_concentrations(), (2, 1)), network=adm1)
    with pytest.raises(ValueError, match="crosses networks"):
        plant.add_influent("feed", bad)


def test_matched_network_instance_wires_fine(asm1):
    plant = build_bsm1(asm1)
    plant.add_influent("feed", asm1.influent({"SS": 60.0}, Q=18446.0))  # same object
    assert any(c.from_port == "feed" for c in plant.connections)


# ----- functional: temperature drives nitrification (slow) ----------------

@pytest.mark.slow
def test_colder_suppresses_nitrification(asm1):
    inf = {"SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
           "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0}

    def steady_snh(celsius):
        plant = build_bsm1(asm1)
        plant.add_influent("feed", asm1.influent(inf, Q=18446.0))
        plant.set_temperature(celsius)
        ss = plant.run_to_steady_state(max_time=300.0)
        return float(plant.stream(ss.solution, plant.effluent_endpoint)
                     .C_named("SNH")[-1])

    warm = steady_snh(20.0)
    cold = steady_snh(10.0)
    assert warm < 2.0          # warm: nitrified
    assert cold > warm + 5.0   # cold: ammonia breaks through
