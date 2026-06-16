"""The Aeration abstraction on CSTRUnit (issue #137).

``Aeration`` replaces the raw per-species ``kla`` / ``C_sat`` dicts with the
quantity a designer thinks in -- a fixed mass-transfer coefficient (open loop) or
a dissolved-oxygen setpoint (closed loop), the latter auto-wiring a PI controller
on the plant. Covers the spec validation, the open-loop translation, and the
closed-loop auto-wiring (per-tank and shared-controller).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import Aeration
from aquakin.plant.cstr import CSTRUnit, oxygen_saturation
from aquakin.plant.plant import Plant
from aquakin.plant.streams import Stream


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


def _tank(asm1, name, aeration, **kw):
    return CSTRUnit(name=name, network=asm1, volume=1000.0,
                    input_port_names=["in"], conditions={"T": 293.15},
                    aeration=aeration, **kw)


# --- spec validation --------------------------------------------------------

def test_requires_exactly_one_mode():
    with pytest.raises(ValueError, match="exactly one"):
        Aeration()                                   # neither
    with pytest.raises(ValueError, match="exactly one"):
        Aeration(kla=120.0, do_setpoint=2.0)         # both
    with pytest.raises(ValueError, match="kla must be"):
        Aeration(kla=-1.0)


# --- open loop --------------------------------------------------------------

def test_open_loop_sets_kla_and_saturation(asm1):
    tank = _tank(asm1, "t", Aeration(kla=240.0, do_sat=8.0))
    so = asm1.species_index["SO"]
    assert float(tank._kla_vec[so]) == 240.0
    assert float(tank._sat_vec[so]) == 8.0
    assert tank.required_signals == ()                # no control signal
    assert tank._controlled_kla == {}


def test_do_sat_defaults_to_eight(asm1):
    tank = _tank(asm1, "t", Aeration(kla=120.0))
    assert float(tank._sat_vec[asm1.species_index["SO"]]) == 8.0


def test_anoxic_tank_has_no_aeration(asm1):
    tank = _tank(asm1, "t", None)
    assert float(tank._kla_vec.max()) == 0.0
    assert float(tank._sat_vec.max()) == 0.0
    assert tank.required_signals == ()


# --- closed loop: per-tank auto-wiring --------------------------------------

def test_per_tank_do_setpoint_auto_wires_its_own_controller(asm1):
    p = Plant(name="pertank")
    p.add_unit(_tank(asm1, "reactor", Aeration(do_setpoint=2.0)))
    p._build_state_layout()                           # materialises + validates
    # a dedicated controller, named off the tank, sensing the tank itself
    assert "reactor_aeration" in p.units
    ctrl = p.units["reactor_aeration"]
    assert ctrl.setpoint == 2.0 and ctrl.measured_species == "SO"
    assert ctrl.signal_names == ("_aer_reactor_kla",)
    assert p.units["reactor"].required_signals == ("_aer_reactor_kla",)
    # the sensor tap was wired: reactor -> controller.measured
    assert any(c.from_unit == "reactor" and c.to_unit == "reactor_aeration"
               for c in p.connections)


def test_materialisation_is_idempotent(asm1):
    p = Plant(name="idem")
    p.add_unit(_tank(asm1, "reactor", Aeration(do_setpoint=2.0)))
    p._build_state_layout()
    p._build_state_layout()                           # second solve setup
    n_ctrl = sum(1 for u in p.units.values() if hasattr(u, "setpoint"))
    assert n_ctrl == 1                                # not added twice


def test_controlled_rhs_without_signal_bus_raises(asm1):
    """A closed-loop reactor's rhs called without the control-signal bus raises a
    clear error rather than silently running unaerated. The plant always supplies
    the bus, so a ``None`` here is misuse -- and a controlled tank has no open-loop
    kLa to fall back on, so a silent fallback would quietly leave it anoxic."""
    p = Plant(name="noBus")
    p.add_unit(_tank(asm1, "reactor", Aeration(do_setpoint=2.0)))
    p._build_state_layout()                # materialises the controlled aeration
    tank = p.units["reactor"]
    assert tank.required_signals           # sanity: it really is controlled
    C0 = asm1.default_concentrations()
    inp = {"in": Stream(Q=jnp.asarray(1000.0), C=C0, network=asm1)}
    with pytest.raises(ValueError, match="control-signal bus"):
        # signals defaults to None -- the misuse this guards against.
        tank.rhs(jnp.asarray(0.0), C0, inp, asm1.default_parameters())


# --- closed loop: shared controller -----------------------------------------

def test_shared_controller_drives_several_tanks(asm1):
    p = Plant(name="shared")
    p.add_unit(_tank(asm1, "tankA",
               Aeration(do_setpoint=2.0, controller="do", sensor="tankA", gain=1.0)))
    p.add_unit(_tank(asm1, "tankB",
               Aeration(do_setpoint=2.0, controller="do", sensor="tankA", gain=0.5)))
    p._build_state_layout()
    # exactly one controller, named after the shared id, sensing tankA
    ctrls = [n for n, u in p.units.items() if hasattr(u, "setpoint")]
    assert ctrls == ["do"]
    # both tanks read the same signal; gains differ
    assert p.units["tankA"]._controlled_kla["SO"] == ("_aer_do_kla", 1.0)
    assert p.units["tankB"]._controlled_kla["SO"] == ("_aer_do_kla", 0.5)


def test_shared_controller_disagreement_raises(asm1):
    p = Plant(name="bad")
    p.add_unit(_tank(asm1, "tankA",
               Aeration(do_setpoint=2.0, controller="do", sensor="tankA")))
    p.add_unit(_tank(asm1, "tankB",
               Aeration(do_setpoint=1.5, controller="do", sensor="tankA")))  # differs
    with pytest.raises(ValueError, match="must agree"):
        p._build_state_layout()


def test_sensor_must_exist(asm1):
    p = Plant(name="nosensor")
    p.add_unit(_tank(asm1, "tankA",
               Aeration(do_setpoint=2.0, controller="do", sensor="ghost")))
    with pytest.raises(ValueError, match="senses unit 'ghost'"):
        p._build_state_layout()


# --- oxygen-transfer corrections (issue #206) -------------------------------

def _aeration_term(asm1, aeration, T_in, signals=None):
    """The aeration contribution to the SO rhs, isolated by subtracting an
    otherwise-identical un-aerated tank at the same inlet temperature (so the
    convection + temperature-dependent chemistry terms cancel exactly)."""
    so = asm1.species_index["SO"]
    C = asm1.default_concentrations()
    s = Stream(Q=jnp.asarray(100.0), C=C, network=asm1, T=jnp.asarray(float(T_in)))
    p = asm1.default_parameters()
    aer_tank = _tank(asm1, "t", aeration)
    bare = _tank(asm1, "t0", None)
    r_aer = aer_tank.rhs(jnp.asarray(0.0), C, {"in": s}, p, signals)
    r_bare = bare.rhs(jnp.asarray(0.0), C, {"in": s}, p)
    return float(r_aer[so] - r_bare[so]), float(C[so])


def test_oxygen_saturation_benson_krause():
    """The Benson--Krause correlation reproduces the standard saturation curve:
    ~9.09 mg/L at 20 degC, ~7.56 at 30 degC -- a ~17% swing."""
    assert float(oxygen_saturation(293.15)) == pytest.approx(9.09, abs=0.02)
    assert float(oxygen_saturation(303.15)) == pytest.approx(7.56, abs=0.02)
    # Monotonically decreasing with temperature.
    assert float(oxygen_saturation(283.15)) > float(oxygen_saturation(303.15))


def test_oxygen_saturation_bsm2_normalised_to_eight_at_15C():
    """The IWA benchmark saturation is normalised to 8.0 mg/L at 15 degC and
    decreases with temperature (~9.0 at 9.5 degC, ~7.2 at 20.5 degC)."""
    from aquakin.plant.cstr import oxygen_saturation_bsm2
    assert float(oxygen_saturation_bsm2(288.15)) == pytest.approx(8.0, abs=1e-4)
    assert float(oxygen_saturation_bsm2(282.65)) == pytest.approx(9.0, abs=0.05)
    assert float(oxygen_saturation_bsm2(293.65)) == pytest.approx(7.19, abs=0.05)
    assert (float(oxygen_saturation_bsm2(282.65))
            > float(oxygen_saturation_bsm2(293.65)))


def test_saturation_model_selects_correction_curve(asm1):
    """``saturation_model='bsm2'`` uses the benchmark curve for the correction
    ratio; an unknown model is rejected at construction."""
    from aquakin.plant.cstr import (
        aeration_transfer, build_aeration_vectors, oxygen_saturation_bsm2)
    with pytest.raises(ValueError):
        Aeration(kla=120.0, saturation_model="nope")
    aer = Aeration(kla=120.0, do_sat=8.0, temperature_correction=True,
                   ref_T=288.15, kla_theta=1.024, saturation_model="bsm2")
    av = build_aeration_vectors(aer, asm1, "t")
    so = asm1.species_index["SO"]
    C = jnp.zeros(asm1.n_species)
    # At 10 degC the saturation is scaled by the BSM2 ratio C_s(283.15)/C_s(288.15)
    # and the open-loop kLa by theta**(T-ref); both apply to the aeration term.
    term = aeration_transfer(av, C, 283.15, None, asm1)
    ratio = float(oxygen_saturation_bsm2(283.15) / oxygen_saturation_bsm2(288.15))
    kla_eff = 120.0 * 1.024 ** (283.15 - 288.15)
    assert float(term[so]) == pytest.approx(kla_eff * 8.0 * ratio, rel=1e-6)


def test_default_aeration_is_bit_faithful(asm1):
    """All corrections off by default: the saturation/kLa vectors are the raw
    constants and the rhs ignores the inlet temperature (the IWA benchmark)."""
    tank = _tank(asm1, "t", Aeration(kla=240.0, do_sat=8.0))
    so = asm1.species_index["SO"]
    assert float(tank._sat_vec[so]) == 8.0 and float(tank._kla_vec[so]) == 240.0
    # Aeration term is the same whether the inlet is warm or at the reference.
    warm, _ = _aeration_term(asm1, Aeration(kla=240.0), 303.15)
    ref, _ = _aeration_term(asm1, Aeration(kla=240.0), 293.15)
    assert warm == pytest.approx(ref)


def test_constant_factors_fold_into_vectors(asm1):
    """alpha scales the open-loop kLa; beta and pressure_factor scale the
    saturation -- all temperature-independent, folded at construction."""
    tank = _tank(asm1, "t", Aeration(kla=240.0, do_sat=8.0,
                                     alpha=0.6, beta=0.95, pressure_factor=0.9))
    so = asm1.species_index["SO"]
    assert float(tank._kla_vec[so]) == pytest.approx(240.0 * 0.6)
    assert float(tank._sat_vec[so]) == pytest.approx(8.0 * 0.95 * 0.9)


def test_temperature_correction_identity_at_ref_T(asm1):
    """With temperature_correction on, the reference temperature is the unity
    point -- it reproduces the uncorrected aeration exactly."""
    on = Aeration(kla=240.0, temperature_correction=True, ref_T=293.15)
    off = Aeration(kla=240.0)
    a_on, _ = _aeration_term(asm1, on, 293.15)
    a_off, _ = _aeration_term(asm1, off, 293.15)
    assert a_on == pytest.approx(a_off)


def test_temperature_correction_warm_lowers_saturation_raises_kla(asm1):
    """At 30 degC the saturation falls by C_s(T)/C_s(ref) and the open-loop kLa
    rises by theta**(T-ref) -- both applied to the aeration term."""
    on = Aeration(kla=240.0, do_sat=8.0, temperature_correction=True,
                  ref_T=293.15, kla_theta=1.024)
    term, so_val = _aeration_term(asm1, on, 303.15)
    ratio = float(oxygen_saturation(303.15) / oxygen_saturation(293.15))
    kla_eff = 240.0 * 1.024 ** (303.15 - 293.15)
    expected = kla_eff * (8.0 * ratio - so_val)
    assert term == pytest.approx(expected, rel=1e-6)


def test_temperature_correction_falls_back_to_static_T(asm1):
    """When the inlet carries no temperature, the correction uses the tank's
    static T condition (the same source the kinetics fall back to)."""
    so = asm1.species_index["SO"]
    C = asm1.default_concentrations()
    s = Stream(Q=jnp.asarray(100.0), C=C, network=asm1, T=None)  # no inlet T
    p = asm1.default_parameters()
    tank = CSTRUnit(name="t", network=asm1, volume=1000.0, input_port_names=["in"],
                    conditions={"T": 303.15},
                    aeration=Aeration(kla=240.0, do_sat=8.0,
                                      temperature_correction=True, ref_T=293.15))
    bare = CSTRUnit(name="t0", network=asm1, volume=1000.0, input_port_names=["in"],
                    conditions={"T": 303.15}, aeration=None)
    term = float(tank.rhs(jnp.asarray(0.0), C, {"in": s}, p)[so]
                 - bare.rhs(jnp.asarray(0.0), C, {"in": s}, p)[so])
    ratio = float(oxygen_saturation(303.15) / oxygen_saturation(293.15))
    kla_eff = 240.0 * 1.024 ** (303.15 - 293.15)
    assert term == pytest.approx(kla_eff * (8.0 * ratio - float(C[so])), rel=1e-6)


def test_closed_loop_kla_not_theta_scaled_but_saturation_is(asm1):
    """A controlled kLa comes from the control signal and is NOT theta-scaled
    (the controller already manipulates it), but its driving-force saturation
    still gets the C_s(T) correction."""
    aer = Aeration(do_setpoint=2.0, temperature_correction=True, ref_T=293.15)
    signals = {"_aer_t_kla": 100.0}
    term, so_val = _aeration_term(asm1, aer, 303.15, signals=signals)
    ratio = float(oxygen_saturation(303.15) / oxygen_saturation(293.15))
    # kLa is the raw signal (gain 1.0), NOT 100*theta**10.
    expected = 100.0 * (8.0 * ratio - so_val)
    assert term == pytest.approx(expected, rel=1e-6)


def test_correction_factor_validation():
    with pytest.raises(ValueError, match="alpha must be"):
        Aeration(kla=120.0, alpha=-0.1)
    with pytest.raises(ValueError, match="beta must be"):
        Aeration(kla=120.0, beta=-1.0)
    with pytest.raises(ValueError, match="pressure_factor must be"):
        Aeration(kla=120.0, pressure_factor=-0.5)
    with pytest.raises(ValueError, match="kla_theta must be"):
        Aeration(kla=120.0, kla_theta=0.0)


def test_build_bsm2_do_temperature_correction_flag():
    """build_bsm2 keeps the IWA-faithful constant aeration by default, and the
    opt-in flag enables the temperature correction on the aerated tanks with the
    reactors' static T as the unity reference."""
    from aquakin.plant.bsm.bsm2 import build_bsm2

    base = build_bsm2()
    t3 = base.units["tank3"]
    assert t3._av.temp_correct is False
    assert float(t3._sat_vec[t3.network.species_index["SO"]]) == 8.0

    corr = build_bsm2(do_temperature_correction=True)
    t3c = corr.units["tank3"]
    assert t3c._av.temp_correct is True
    assert t3c._av.ref_T == float(corr.units["tank3"].conditions["T"])
    # An unaerated (anoxic) tank is untouched -- no aeration to correct.
    assert corr.units["tank1"].aeration is None


def test_temperature_corrected_aeration_is_ad_clean(asm1):
    """jax.grad flows through the corrected aeration term w.r.t. the inlet
    temperature without error or NaN (it stays in the monolithic plant solve)."""
    so = asm1.species_index["SO"]
    C = asm1.default_concentrations()
    p = asm1.default_parameters()
    tank = _tank(asm1, "t", Aeration(kla=240.0, do_sat=8.0,
                                     temperature_correction=True))

    def so_rhs(T):
        s = Stream(Q=jnp.asarray(100.0), C=C, network=asm1, T=T)
        return tank.rhs(jnp.asarray(0.0), C, {"in": s}, p)[so]

    g = jax.grad(so_rhs)(jnp.asarray(300.0))
    assert np.isfinite(float(g))
