"""The reactor units share one reaction term and one operating temperature.

Every kinetic plant unit (CSTR, MBR, SBR, IFAS/MBBR, ADM1 digester) builds its
reaction term from ``CompiledModel.dCdt`` -- the canonical term with the
negative-state clip and the positivity limiter -- and every unit whose
temperature the plant tracks runs its kinetics at the temperature the plant
hands it. These tests pin that by single RHS evaluations (no stiff solve):

- an MBR's and an SBR's reaction term equals ``model.dCdt`` for a model with a
  positivity limiter (ASM2d), at a state where the limiter is active;
- under a heat-balance temperature model an MBR's and an IFAS tank's kinetics
  run at their own lagged temperature state, not the inlet temperature;
- ``Plant.set_temperature`` reaches an MBR and an SBR.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import (
    Aeration,
    IFASUnit,
    InterfaceSettling,
    MBRUnit,
    Plant,
    SBRPhase,
    SBRUnit,
)
from aquakin.plant.bsm.bsm2 import bsm2_asm1_model
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.streams import Stream

_ASM1_PARTS = ["XI", "XS", "XB_H", "XB_A", "XP", "XND"]


@pytest.fixture(scope="module")
def asm2d():
    return aquakin.load_model("asm2d")


@pytest.fixture(scope="module")
def asm1():
    # The BSM2 ASM1 carries the Arrhenius temperature corrections, so its rates
    # depend on the T condition.
    return bsm2_asm1_model()


def _limited_state(model):
    """A substrate-rich state with oxygen and nitrate below the positivity
    threshold, so the limiter throttles their (net) consumption."""
    C = np.asarray(model.default_concentrations(), dtype=float).copy()
    thr = model.positivity_threshold
    values = {"SF": 50.0, "SA": 50.0, "SNH4": 20.0, "SPO4": 5.0, "SALK": 5.0}
    values.update({"XH": 2000.0, "XPAO": 500.0, "XAUT": 100.0, "XPHA": 50.0, "XPP": 50.0})
    values.update({"SO2": 0.2 * thr, "SNO3": 0.2 * thr})
    for sp, v in values.items():
        C[model.species_index[sp]] = v
    return jnp.asarray(C)


def _limiter_is_active(model, C, conditions):
    """The limited and unlimited terms must differ, or the test proves nothing."""
    params = model.default_parameters()
    raw = model.compute_stoich(params).T @ model.rates(C, params, conditions, 0)
    limited = model.dCdt(C, params, conditions, 0)
    return float(jnp.max(jnp.abs(raw - limited)))


# ----- positivity limiter ----------------------------------------------------


def test_mbr_reaction_term_is_model_dCdt(asm2d):
    """With no flow and no aeration an MBR's species derivative is exactly its
    reaction term, which must be the limited ``model.dCdt``."""
    mbr = MBRUnit(
        "mbr",
        asm2d,
        volume=1000.0,
        waste_flow=0.0,
        membrane_area=500.0,
        conditions={"T": 293.15},
    )
    C = _limited_state(asm2d)
    conditions = {"T": jnp.asarray([293.15])}
    assert _limiter_is_active(asm2d, C, conditions) > 1e-3
    params = asm2d.default_parameters()
    feed = Stream(Q=jnp.asarray(0.0), C=jnp.zeros(asm2d.n_species), model=asm2d)
    state = jnp.concatenate([C, jnp.zeros(1)])
    d = mbr.rhs(jnp.asarray(0.0), state, {"feed": feed}, params)
    expected = asm2d.dCdt(C, params, conditions, 0)
    np.testing.assert_allclose(d[: asm2d.n_species], expected, rtol=1e-12, atol=0.0)


def test_sbr_reaction_term_is_model_dCdt(asm2d):
    """In an unfed, unaerated, undecanted phase an SBR's species derivative is
    exactly its reaction term, which must be the limited ``model.dCdt``."""
    sbr = SBRUnit(
        "sbr",
        asm2d,
        [SBRPhase("react", 1.0)],
        full_volume=1000.0,
        feed_flow=0.0,
        decant_flow=0.0,
        settling=InterfaceSettling(v_settle=1.0, area=1.0),
        conditions={"T": 293.15},
        oxygen_species="SO2",
    )
    C = _limited_state(asm2d)
    conditions = {"T": jnp.asarray([293.15])}
    assert _limiter_is_active(asm2d, C, conditions) > 1e-3
    params = asm2d.default_parameters()
    state = jnp.concatenate([C, jnp.asarray([500.0]), sbr.settling.initial_extra_state()])
    feed = Stream(Q=jnp.asarray(0.0), C=jnp.zeros(asm2d.n_species), model=asm2d)
    d = sbr.rhs(jnp.asarray(0.5), state, {"feed": feed}, params)
    expected = asm2d.dCdt(C, params, conditions, 0)
    np.testing.assert_allclose(d[: asm2d.n_species], expected, rtol=1e-12, atol=0.0)


# ----- heat-balance operating temperature -----------------------------------

_T_UNIT = 293.15  # the unit's condition, so its heat-balance state starts here
_T_FEED = 283.15  # a colder influent


def _single_unit_plant(model, unit, port, T_feed, temperature_model=None):
    plant = Plant("one", time_unit="d")
    plant.add_unit(unit)
    infl = InfluentSeries.constant(
        model, SS=300.0, SNH=40.0, XS=200.0, XB_H=80.0, SO=1.0, Q=1000.0
    )
    infl = dataclasses.replace(infl, T=jnp.full_like(infl.Q, T_feed))
    plant.add_influent("feed", infl, to=f"{unit.name}.{port}")
    if temperature_model is not None:
        plant.set_temperature_model(temperature_model)
    plant._build_state_layout()
    return plant


def _unit_derivative(plant, name, conc_state):
    """The unit's block of the plant derivative at the given unit state (the
    heat-balance temperature, if any, left at its seed)."""
    y0 = plant.initial_state(overrides={name: conc_state})
    d = plant.derivative(y0, params=plant.default_parameters())
    start, size = plant._state_layout[name]
    return d[start : start + size]


def _mbr(model):
    return MBRUnit(
        "mbr",
        model,
        volume=1000.0,
        waste_flow=20.0,
        membrane_area=500.0,
        particulate_species=_ASM1_PARTS,
        rejection=0.999,
        aeration=Aeration(kla=240.0),
        conditions={"T": _T_UNIT},
    )


def _ifas(model):
    return IFASUnit(
        name="ifas",
        model=model,
        volume=1000.0,
        input_port_names=["in"],
        specific_surface_area=500.0,
        fill_fraction=0.4,
        biofilm_thickness=5e-4,
        n_layers=4,
        aeration=Aeration(kla=240.0),
        conditions={"T": _T_UNIT},
    )


@pytest.mark.parametrize("make, port", [(_mbr, "feed"), (_ifas, "in")], ids=["mbr", "ifas"])
def test_heat_balance_kinetics_run_at_the_unit_temperature_state(asm1, make, port):
    """Under the heat-balance model the unit's temperature state starts at its
    293.15 K condition while the influent is at 283.15 K. Its kinetics must run
    at the state, so its derivative equals that of the same unit in an
    algebraic-temperature plant whose influent is at 293.15 K (where the inlet
    temperature is the operating temperature) -- and differs from the one fed at
    283.15 K, which is what reading the inlet temperature would give."""
    unit_hb = make(asm1)
    hb = _single_unit_plant(asm1, unit_hb, port, _T_FEED, aquakin.HeatBalanceTemperature())
    assert hb._temperature_units == [unit_hb.name]
    state = unit_hb.initial_state()
    state = state.at[asm1.species_index["XB_A"]].set(150.0)
    state = state.at[asm1.species_index["SNH"]].set(20.0)

    at_state = _unit_derivative(
        _single_unit_plant(asm1, make(asm1), port, _T_UNIT), unit_hb.name, state
    )
    at_inlet = _unit_derivative(
        _single_unit_plant(asm1, make(asm1), port, _T_FEED), unit_hb.name, state
    )
    # The temperature must matter for this check to discriminate.
    assert float(jnp.max(jnp.abs(at_state - at_inlet))) > 1e-3

    got = _unit_derivative(hb, unit_hb.name, state)
    np.testing.assert_allclose(got, at_state, rtol=1e-12, atol=1e-12)


# ----- Plant.set_temperature ------------------------------------------------


def test_plant_set_temperature_reaches_mbr_and_sbr(asm1):
    """``Plant.set_temperature`` sets the T condition of an MBR and an SBR, and
    their kinetics then match a unit built at that temperature."""
    mbr = _mbr(asm1)
    sbr = SBRUnit(
        "sbr",
        asm1,
        [SBRPhase("react", 1.0, kla=240.0)],
        full_volume=1000.0,
        feed_flow=0.0,
        decant_flow=0.0,
        settling=InterfaceSettling(v_settle=1.0, area=1.0),
        particulate_species=_ASM1_PARTS,
        conditions={"T": _T_UNIT},
    )
    plant = Plant("two", time_unit="d")
    plant.add_unit(mbr)
    plant.add_unit(sbr)
    plant.add_influent(
        "feed", InfluentSeries.constant(asm1, SS=300.0, Q=1000.0), to="mbr.feed"
    )
    plant.connect("mbr.permeate", "sbr.feed")
    plant.set_temperature(10.0)
    assert mbr.conditions["T"] == pytest.approx(283.15)
    assert sbr.conditions["T"] == pytest.approx(283.15)

    params = asm1.default_parameters()
    C = asm1.default_concentrations().at[asm1.species_index["XB_A"]].set(150.0)
    feed = Stream(Q=jnp.asarray(0.0), C=jnp.zeros(asm1.n_species), model=asm1)
    t = jnp.asarray(0.5)

    sbr_ref = dataclasses.replace(sbr, conditions={"T": 283.15})
    s_state = jnp.concatenate([C, jnp.asarray([500.0]), sbr.settling.initial_extra_state()])
    np.testing.assert_allclose(
        sbr.rhs(t, s_state, {"feed": feed}, params),
        sbr_ref.rhs(t, s_state, {"feed": feed}, params),
        rtol=1e-12,
        atol=0.0,
    )
    mbr_ref = dataclasses.replace(mbr, conditions={"T": 283.15})
    m_state = jnp.concatenate([C, jnp.zeros(1)])
    np.testing.assert_allclose(
        mbr.rhs(t, m_state, {"feed": feed}, params),
        mbr_ref.rhs(t, m_state, {"feed": feed}, params),
        rtol=1e-12,
        atol=0.0,
    )


# ----- mass balance ---------------------------------------------------------


def test_mass_balance_reaction_term_is_the_units_reaction_term(asm2d):
    """``plant.mass_balance`` integrates each reactor's reaction term; it must be
    the limited ``model.dCdt`` the reactor's ``rhs`` integrates, not the raw
    ``stoich.T @ rates``."""
    from aquakin.plant import CSTRUnit
    from aquakin.plant.balance import _reaction_term

    tank = CSTRUnit("tank", asm2d, volume=1000.0, input_port_names=["in"],
                    conditions={"T": 293.15})
    plant = Plant("one", time_unit="d")
    plant.add_unit(tank)
    plant.add_influent("feed", InfluentSeries.constant(asm2d, SF=50.0, Q=1000.0), to="tank.in")
    plant._build_state_layout()
    plant._build_parameter_layout()
    C = _limited_state(asm2d)
    conditions = {"T": jnp.asarray([293.15])}
    assert _limiter_is_active(asm2d, C, conditions) > 1e-3
    params = plant.default_parameters()
    got = _reaction_term(plant, "tank", C, params)
    expected = asm2d.dCdt(C, asm2d.default_parameters(), conditions, 0)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)
