"""build_bsm2 option objects + plant entry/exit endpoint properties.

All fast (plant construction only, no solve): the option objects must enable the
right units and the plant must record the canonical influent/effluent endpoints
so callers never hard-code a moving port.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm import (
    ExternalCarbon,
    HydraulicDelay,
    InfluentBypass,
    RejectStorage,
    build_bsm1,
    build_bsm2,
    bsm2_asm1_network,
    bsm2_constant_influent,
)


@pytest.fixture(scope="module")
def nets():
    return bsm2_asm1_network(), aquakin.load_network("adm1")


# ----- Endpoints move with the front-end features -------------------------

@pytest.mark.parametrize("kwargs,influent,effluent", [
    ({}, "front_mix.fresh", "settler.overflow"),
    ({"bypass": InfluentBypass()}, "bypass_split.in", "effluent_mix.out"),
    ({"hydraulic_delay": HydraulicDelay()}, "influent_delay.in", "settler.overflow"),
])
def test_endpoints_reflect_features(nets, kwargs, influent, effluent):
    asm1, adm1 = nets
    plant = build_bsm2(asm1, adm1, **kwargs)
    assert plant.influent_endpoint == influent
    assert plant.effluent_endpoint == effluent
    # The recorded ports are real plant endpoints.
    avail = {f"{n}.{p}" for n, u in plant.units.items() for p in u.output_ports}
    in_avail = {f"{n}.{p}" for n, u in plant.units.items() for p in u.input_ports}
    assert plant.influent_endpoint in in_avail
    assert plant.effluent_endpoint in avail


def test_build_bsm1_sets_endpoints():
    plant = build_bsm1()
    assert plant.influent_endpoint == "inlet_mix.fresh"
    assert plant.effluent_endpoint == "clarifier.overflow"


def test_add_influent_defaults_to_endpoint(nets):
    """add_influent with no `to` wires to the plant's influent_endpoint."""
    asm1, adm1 = nets
    plant = build_bsm2(asm1, adm1, bypass=InfluentBypass())
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    fed = [c for c in plant.connections if c.from_port == "feed"]
    assert len(fed) == 1
    assert (fed[0].to_unit, fed[0].to_port) == ("bypass_split", "in")


# ----- Option objects enable the right units / config ---------------------

def test_reject_storage_object_adds_tank(nets):
    asm1, adm1 = nets
    assert "reject_storage" not in build_bsm2(asm1, adm1).units
    plant = build_bsm2(asm1, adm1,
                       reject=RejectStorage(volume=200.0, output_flow=50.0))
    tank = plant.units["reject_storage"]
    assert tank.volume == 200.0 and tank.output_flow == 50.0
    assert tank.level_setpoint is None  # fixed-release, not controlled


def test_reject_control_object_is_level_controlled(nets):
    asm1, adm1 = nets
    plant = build_bsm2(asm1, adm1, reject=RejectStorage(control=True))
    tank = plant.units["reject_storage"]
    assert tank.level_setpoint is not None  # proportional level controller


def test_bypass_object_threshold_and_units(nets):
    asm1, adm1 = nets
    plant = build_bsm2(asm1, adm1, bypass=InfluentBypass(threshold=70000.0))
    assert "bypass_split" in plant.units and "effluent_mix" in plant.units
    assert plant.units["bypass_split"].threshold == 70000.0


def test_hydraulic_delay_object_tau(nets):
    asm1, adm1 = nets
    plant = build_bsm2(asm1, adm1, hydraulic_delay=HydraulicDelay(tau=0.05))
    assert "influent_delay" in plant.units
    assert plant.units["influent_delay"].tau == 0.05


def test_carbon_default_on_and_disablable(nets):
    asm1, adm1 = nets
    # External carbon is now a DosingUnit on the as_mix -> tank1 line (default on).
    from aquakin.plant.dosing import DosingUnit
    default = build_bsm2(asm1, adm1)
    assert "external_carbon" in default.units
    assert isinstance(default.units["external_carbon"], DosingUnit)
    assert "external_carbon" not in build_bsm2(asm1, adm1, carbon=None).units
    plant = build_bsm2(asm1, adm1, carbon=ExternalCarbon(flow=3.0, conc=5e5))
    dose = plant.units["external_carbon"]
    assert dose.flow == 3.0
    assert float(dose.reagent.composition[asm1.species_index["SS"]]) == 5e5


def test_option_objects_are_frozen():
    """The option objects are immutable (safe as shared defaults)."""
    with pytest.raises(Exception):
        InfluentBypass().threshold = 1.0
