"""Diffuser / blower aeration-design physics (issue #279).

``AerationSystem`` turns the kLa a solve produced into the air flow and blower
power that produce it: ``SOTR = kLa·C_s,std·V`` -> air flow via SOTE -> blower
power by adiabatic compression against the submergence head. These checks pin the
physics against its closed-form definition, the SOTE/depth/fouling handling, the
validation, and AD-cleanliness of the differentiable primitives. The wiring into
``evaluate_bsm1``/``evaluate_bsm2`` (where it replaces the Copp correlation) is
covered in ``test_bsm2_evaluation.py``.
"""

import jax
import jax.numpy as jnp
import pytest

from aquakin import (
    AerationSystem,
    blower_energy,
    blower_power_kw,
    design_summary,
    required_airflow,
)

_RHO_G = 9.80665 * 1000.0 / 1000.0   # rho_w * g (kPa per metre of water)


# --- diffuser: SOTE, airflow ------------------------------------------------

def test_effective_sote_from_depth_and_fouling():
    # explicit SOTE
    assert AerationSystem(depth=5.0, sote=0.25).effective_sote() == pytest.approx(0.25)
    # derived from sote_per_meter * depth (default 6 %/m)
    assert AerationSystem(depth=5.0).effective_sote() == pytest.approx(0.30)
    # fouling factor reduces it
    assert AerationSystem(depth=5.0, sote=0.25, fouling_F=0.8).effective_sote() \
        == pytest.approx(0.20)


def test_required_airflow_matches_definition():
    """Q_air = SOTR / (SOTE·o2_per_air), SOTR = kLa·C_s,std·V / 1000 (kg/d)."""
    s = AerationSystem(depth=5.0, sote=0.20, standard_do_sat=9.09, o2_per_air=0.279)
    kla, V = 100.0, 1000.0
    sotr = kla * 9.09 * V / 1000.0                     # kg O2/d
    expected = sotr / (0.20 * 0.279)                   # m3/d
    assert float(required_airflow(kla, V, s)) == pytest.approx(expected, rel=1e-9)


def test_airflow_is_proportional_to_kla():
    """A given kLa needs a given airflow (linear); the DO deficit does not enter."""
    s = AerationSystem(depth=5.0, sote=0.20)
    q1 = float(required_airflow(100.0, 1000.0, s))
    q2 = float(required_airflow(250.0, 1000.0, s))
    assert q2 / q1 == pytest.approx(2.5, rel=1e-9)


# --- blower: discharge pressure + adiabatic power ---------------------------

def test_discharge_pressure_from_submergence():
    s = AerationSystem(depth=5.0, sote=0.20, headloss_kpa=3.0)
    assert s.discharge_pressure_kpa() == pytest.approx(
        101.325 + _RHO_G * 5.0 + 3.0, rel=1e-9)


def test_blower_power_adiabatic_closed_form():
    """P = (Q·p1/η)·(γ/(γ−1))·[(p2/p1)^((γ−1)/γ) − 1], Q in m³/s, p in Pa -> kW."""
    s = AerationSystem(depth=5.0, sote=0.20, blower_efficiency=0.6, gamma=1.4)
    q_m3_d = 16290.3
    Q = q_m3_d / 86400.0
    p1 = 101.325e3
    p2 = s.discharge_pressure_kpa() * 1e3
    n = (1.4 - 1.0) / 1.4
    expected_kw = (Q * p1 / 0.6) / n * ((p2 / p1) ** n - 1.0) / 1000.0
    assert float(blower_power_kw(q_m3_d, s)) == pytest.approx(expected_kw, rel=1e-9)
    assert expected_kw > 0.0


def test_blower_power_is_linear_in_airflow():
    s = AerationSystem(depth=5.0, sote=0.20)
    p1 = float(blower_power_kw(1000.0, s))
    p2 = float(blower_power_kw(3000.0, s))
    assert p2 / p1 == pytest.approx(3.0, rel=1e-9)


# --- energy kernel (drop-in for aeration_energy) ----------------------------

def test_blower_energy_is_time_averaged_power_over_a_day():
    s = AerationSystem(depth=5.0, sote=0.20)
    t = jnp.array([0.0, 1.0])
    volumes = jnp.array([1000.0, 1333.0])
    kla = jnp.array([[100.0, 200.0], [100.0, 200.0]])   # constant in time
    # constant power -> energy = power * 24 h
    q = required_airflow(kla[0], volumes, s)
    power_kw = float(jnp.sum(blower_power_kw(q, s)))
    assert blower_energy(t, kla, volumes, s) == pytest.approx(power_kw * 24.0, rel=1e-9)


# --- sizing summary ---------------------------------------------------------

def test_design_summary_is_self_consistent():
    s = AerationSystem(depth=5.0, sote=0.20)
    dp = design_summary(100.0, 1000.0, s)
    assert dp.sote == pytest.approx(0.20)
    assert dp.sotr == pytest.approx(100.0 * 9.09 * 1000.0 / 1000.0, rel=1e-9)
    assert dp.airflow == pytest.approx(float(required_airflow(100.0, 1000.0, s)), rel=1e-12)
    assert dp.power == pytest.approx(float(blower_power_kw(dp.airflow, s)), rel=1e-12)
    assert "Blower power" in dp.report() and "SOTE" in dp.report()


# --- validation -------------------------------------------------------------

def test_validation():
    with pytest.raises(ValueError, match="depth must be > 0"):
        AerationSystem(depth=0.0, sote=0.2)
    with pytest.raises(ValueError, match="effective SOTE must be in"):
        AerationSystem(depth=5.0, sote=1.5)                 # > 1
    with pytest.raises(ValueError, match="effective SOTE must be in"):
        AerationSystem(depth=5.0, sote=0.0)                 # 0
    with pytest.raises(ValueError, match="o2_per_air must be > 0"):
        AerationSystem(depth=5.0, sote=0.2, o2_per_air=0.0)
    with pytest.raises(ValueError, match="gamma must be > 1"):
        AerationSystem(depth=5.0, sote=0.2, gamma=1.0)


# --- AD ---------------------------------------------------------------------

def test_grad_through_blower_primitives():
    """jax.grad flows through required_airflow -> blower_power_kw, w.r.t. both the
    kLa (the operating point) and a design parameter (depth)."""
    volumes = jnp.array([1000.0, 1333.0])

    def power(scale):
        s = AerationSystem(depth=5.0, sote=0.20)
        q = required_airflow(scale * jnp.array([100.0, 80.0]), volumes, s)
        return jnp.sum(blower_power_kw(q, s))

    g = jax.grad(power)(1.0)
    assert jnp.isfinite(g)
    # power is linear in the scale, so grad == the value at scale 1
    assert float(g) == pytest.approx(float(power(1.0)), rel=1e-9)

    def power_by_depth(depth):
        s = AerationSystem(depth=depth, sote=0.20)
        return jnp.sum(blower_power_kw(
            required_airflow(jnp.array([100.0]), jnp.array([1000.0]), s), s))

    gd = jax.grad(power_by_depth)(5.0)
    fd = (power_by_depth(5.001) - power_by_depth(4.999)) / 0.002
    assert jnp.isfinite(gd)
    assert float(gd) == pytest.approx(float(fd), rel=1e-4)
