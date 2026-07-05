"""Unit tests for the shared core helpers consolidated from scattered copies.

* ``core/temperature.py`` -- the van't Hoff / Arrhenius primitives the pH solver,
  both precipitation engines and the model rate-constant corrections route
  through (the formula used to be inlined verbatim at four sites).
* ``core/hints.py`` -- the ``did_you_mean`` close-match suffix shared by every
  unknown-name error across core / integrate / utils / plant.
"""

import math

import jax.numpy as jnp

from aquakin.core.hints import did_you_mean
from aquakin.core.temperature import (
    LN10,
    R_GAS,
    T_REF_THERMO,
    arrhenius_factor,
    van_t_hoff_factor,
)


# --------------------------------------------------------------------------
# core/temperature.py
# --------------------------------------------------------------------------

def test_temperature_constants():
    assert R_GAS == 8.314462618
    assert T_REF_THERMO == 298.15
    assert float(LN10) == float(jnp.log(10.0))


def test_van_t_hoff_factor_closed_form():
    Tk = 283.15
    expected = (1.0 / T_REF_THERMO - 1.0 / Tk) / R_GAS
    assert abs(float(van_t_hoff_factor(Tk)) - expected) < 1e-18


def test_van_t_hoff_factor_zero_at_reference():
    # The factor vanishes at T_ref, so exp(dH * factor) == 1 for any enthalpy --
    # a constant referenced at 25 degC is unchanged there.
    assert float(van_t_hoff_factor(T_REF_THERMO)) == 0.0


def test_van_t_hoff_factor_custom_reference():
    Tk, Tref = 290.0, 280.0
    expected = (1.0 / Tref - 1.0 / Tk) / R_GAS
    assert abs(float(van_t_hoff_factor(Tk, Tref)) - expected) < 1e-18


def test_arrhenius_factor_closed_form():
    theta = 1.072
    got = float(arrhenius_factor(283.15, 293.15, math.log(theta)))
    assert abs(got - theta ** (283.15 - 293.15)) < 1e-12


def test_arrhenius_factor_unity_at_reference():
    assert float(arrhenius_factor(293.15, 293.15, math.log(1.072))) == 1.0


def test_arrhenius_factor_vectorized():
    ref = jnp.array([293.15, 293.15])
    ln_theta = jnp.log(jnp.array([1.05, 1.10]))
    got = arrhenius_factor(283.15, ref, ln_theta)
    assert got.shape == (2,)
    assert abs(float(got[0]) - 1.05 ** (283.15 - 293.15)) < 1e-12


# --------------------------------------------------------------------------
# core/hints.py
# --------------------------------------------------------------------------

def test_did_you_mean_close_match():
    assert did_you_mean("SNHH", ["SNH", "SNO", "SO"]) == " Did you mean: SNH?"


def test_did_you_mean_no_match():
    assert did_you_mean("zzzzz", ["SNH", "SNO"]) == ""


def test_did_you_mean_caps_at_n():
    out = did_you_mean("ka", ["ka1", "ka2", "ka3", "ka4"], n=2)
    assert out.count(",") <= 1  # at most two suggestions -> at most one comma


def test_did_you_mean_accepts_any_iterable():
    # dict keys (a common call form) work without pre-listing.
    out = did_you_mean("muH", {"muH_x": 1, "bH": 2})
    assert "muH_x" in out
