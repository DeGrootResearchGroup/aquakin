"""Plant(time_unit=...): one clock, with models, influents and unit operations
converted to it.

Each test compares two plants describing the same physical system on different
clocks (or checks a converted constant directly), so the conversions are pinned
by physics rather than by a hand-copied factor.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import aquakin
from aquakin.plant import (
    CSTRUnit,
    Plant,
    PrimaryClarifier,
    UnitsError,
    UVUnit,
)
from aquakin.plant.takacs import TakacsClarifier


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


@pytest.fixture(scope="module")
def ozone():
    return aquakin.load_model("ozone_bromate")


def _cstr(name, model, volume=1000.0):
    return CSTRUnit(
        name,
        model,
        volume=volume,
        input_port_names=["inlet"],
        conditions=model.condition_defaults(),
    )


def _tank_plant(model, time_unit, Q, volume=1000.0, **feed):
    plant = Plant("tank", time_unit=time_unit)
    plant.add_unit(_cstr("tank", model, volume=volume))
    plant.add_influent("feed", model.influent(feed, Q=Q), to="tank")
    return plant


def _final(plant, t_end):
    sol = plant.solve(t_span=(0.0, t_end), t_eval=np.array([t_end]), rtol=1e-9, atol=1e-11)
    return np.asarray(sol.state[-1])


# ----- the clock --------------------------------------------------------------


def test_a_model_runs_the_same_on_any_clock(asm1):
    # ASM1's rates are per day. Two days on a day clock and 48 hours on an hour
    # clock are the same run: the rates and the feed flow (given in m3/d, the
    # model's unit) are converted to the plant's clock.
    feed = {"SS": 60.0, "SNH": 25.0, "XB_H": 100.0}
    days = _final(_tank_plant(asm1, "d", 2000.0, **feed), 2.0)
    hours = _final(_tank_plant(asm1, "h", 2000.0, **feed), 48.0)
    np.testing.assert_allclose(hours, days, rtol=1e-6, atol=1e-9)


def test_models_on_different_time_units_share_a_declared_clock(asm1, ozone):
    # A per-second model next to a per-day one: on a day clock the ozone tank's
    # chemistry is scaled to per-day rates. Its derivative is the per-second
    # one times 86400.
    plant = Plant("mixed_clock", time_unit="d")
    plant.add_unit(_cstr("a", asm1))
    plant.add_unit(_cstr("o", ozone, volume=10.0))
    plant.add_influent("fa", asm1.influent(Q=100.0), to="a")
    plant.add_influent("fo", ozone.influent(Q=0.0), to="o")
    report = plant.check_units()
    assert report.ok and report.time_unit == "d"

    alone = Plant("ozone_alone", time_unit="s")
    alone.add_unit(_cstr("o", ozone, volume=10.0))
    alone.add_influent("fo", ozone.influent(Q=0.0), to="o")

    y_mixed = plant.initial_state()
    d_mixed = plant.states_by_unit(plant.derivative(y_mixed))["o"]
    d_alone = alone.states_by_unit(alone.derivative(alone.initial_state()))["o"]
    np.testing.assert_allclose(d_mixed, 86400.0 * np.asarray(d_alone), rtol=1e-12)
    assert np.any(np.asarray(d_alone) != 0.0)


def test_a_plant_on_its_models_clock_is_unchanged(asm1):
    # A factor of 1 leaves the reaction term untouched (bit-identical).
    declared = _tank_plant(asm1, "d", 2000.0, SS=60.0, XB_H=100.0)
    y0 = declared.initial_state()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        inferred = _tank_plant(asm1, None, 2000.0, SS=60.0, XB_H=100.0)
        d_inferred = inferred.derivative(y0)
    assert np.array_equal(np.asarray(declared.derivative(y0)), np.asarray(d_inferred))
    assert declared.units["tank"]._rate_factor == 1.0


def test_time_unit_must_be_a_time(asm1):
    with pytest.raises(ValueError, match="unit of time"):
        Plant("p", time_unit="kg")
    with pytest.raises(ValueError, match="unit of time"):
        Plant("p", time_unit="fortnights")


# ----- deprecation of a missing time_unit -----------------------------------------


def test_missing_time_unit_warns_once_and_uses_the_models(asm1):
    plant = _tank_plant(asm1, None, 2000.0, SS=60.0)
    with pytest.warns(FutureWarning, match=r"time_unit='d'"):
        plant.solve(t_span=(0.0, 0.1))
    assert plant.time_unit == "d"
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plant.solve(t_span=(0.0, 0.1))


def test_missing_time_unit_warns_once_even_when_the_plant_changes(asm1):
    plant = _tank_plant(asm1, None, 2000.0, SS=60.0)
    with pytest.warns(FutureWarning):
        plant._build_state_layout()
    # A new influent changes what is put on the clock, so the work is redone,
    # but the warning is not repeated.
    plant.add_influent("extra", asm1.influent({"SNH": 5.0}, Q=10.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plant._build_state_layout()


def test_declared_time_unit_does_not_warn(asm1):
    plant = _tank_plant(asm1, "d", 2000.0, SS=60.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plant.solve(t_span=(0.0, 0.1))


def test_missing_time_unit_with_disagreeing_models_is_an_error(asm1, ozone):
    plant = Plant("mixed_clock")
    plant.add_unit(_cstr("a", asm1))
    plant.add_unit(_cstr("o", ozone, volume=10.0))
    plant.add_influent("fa", asm1.influent(Q=100.0), to="a")
    plant.add_influent("fo", ozone.influent(Q=1.0), to="o")
    report = plant.check_units()
    assert not report.ok and report.time_unit is None
    (err,) = report.errors
    assert "'asm1' in 1/d" in err.detail and "'ozone_bromate' in 1/s" in err.detail
    assert "time_unit" in err.detail
    with pytest.raises(UnitsError, match="disagree on the time unit"):
        plant.solve(t_span=(0.0, 1.0))


def test_missing_time_unit_without_an_inferable_unit_warns(tmp_path):
    path = tmp_path / "no_units.yaml"
    path.write_text(
        """
model:
  name: no_units
  description: "rate constants without units"
species:
  - {name: A, units: g_COD/m3, default_concentration: 1.0}
conditions: []
reactions:
  - name: R
    rate: "k * [A]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1}
"""
    )
    net = aquakin.load_model_from_file(str(path))
    plant = _tank_plant(net, None, 1.0, A=1.0)
    with pytest.warns(FutureWarning, match="no time unit could be inferred"):
        plant.solve(t_span=(0.0, 0.1))
    assert plant.time_unit is None


# ----- influents ----------------------------------------------------------------


def test_influents_are_put_on_the_plant_clock(asm1):
    series = asm1.influent({"SS": 60.0}, Q=2400.0)
    plant = Plant("p", time_unit="h")
    plant.add_unit(_cstr("tank", asm1))
    plant.add_influent("feed", series, to="tank")
    feed = plant.influents["feed"]
    np.testing.assert_allclose(feed.Q, 100.0)
    np.testing.assert_allclose(feed.t, np.asarray(series.t) * 24.0)
    assert feed.units == {"t": "h", "Q": "m3/h"}
    np.testing.assert_array_equal(feed.C, series.C)
    # On the model's own clock the series is used as it is.
    assert series.in_time_unit("d") is series
    assert series.in_time_unit(None) is series


def test_influent_with_declared_times_and_unknown_model_clock(tmp_path):
    # The model declares no time unit, so the series keeps the units it was
    # given; the plant converts from those.
    path = tmp_path / "bare.yaml"
    path.write_text(
        """
model:
  name: bare
  description: "no declared units"
species:
  - {name: A, units: "", default_concentration: 1.0}
conditions: []
reactions:
  - name: R
    rate: "k * [A]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1}
"""
    )
    net = aquakin.load_model_from_file(str(path))
    series = net.influent({"A": 1.0}, Q=48.0, units={"Q": "m3/d"})
    moved = series.in_time_unit("h")
    np.testing.assert_allclose(moved.Q, 2.0)
    np.testing.assert_array_equal(moved.t, series.t)
    assert moved.units == {"Q": "m3/h"}
    # Declared times but no flow unit: only the times move.
    timed = net.influent({"A": 1.0}, Q=48.0, units={"t": "h"})
    moved = timed.in_time_unit("d")
    np.testing.assert_array_equal(moved.Q, timed.Q)
    assert moved.units == {"t": "d"}
    # Nothing declared and no model clock: there is nothing to convert from.
    plain = net.influent({"A": 1.0}, Q=48.0)
    assert plain.in_time_unit("h") is plain


def test_mass_balance_agrees_across_clocks(asm1):
    feed = {"SS": 60.0, "SNH": 25.0, "XB_H": 100.0}
    days = _tank_plant(asm1, "d", 2000.0, **feed)
    hours = _tank_plant(asm1, "h", 2000.0, **feed)
    sol_d = days.solve(t_span=(0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 49), rtol=1e-9)
    sol_h = hours.solve(t_span=(0.0, 24.0), t_eval=np.linspace(0.0, 24.0, 49), rtol=1e-9)
    mb_d = days.mass_balance(sol_d)
    mb_h = hours.mass_balance(sol_h)
    for q in ("COD", "N"):
        np.testing.assert_allclose(mb_h[q].inflow, mb_d[q].inflow, rtol=1e-6)
        np.testing.assert_allclose(mb_h[q].gas, mb_d[q].gas, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(mb_h[q].accumulation, mb_d[q].accumulation, rtol=1e-5)


# ----- unit operations with built-in time constants ----------------------------


def _resolved(plant):
    plant._build_state_layout()
    return plant


def test_uv_exposure_is_in_seconds_on_any_clock(ozone):
    def uv_plant(time_unit):
        plant = Plant("uv", time_unit=time_unit)
        plant.add_unit(UVUnit("uv", ozone, volume=1.0, intensity=10.0, d10=5.0))
        plant.add_influent("f", ozone.influent(Q=1.0), to="uv")
        return _resolved(plant)

    in_seconds = uv_plant("s").units["uv"]
    in_days = uv_plant("d").units["uv"]
    # 1 m3 at 0.1 m3/s is 10 s; 0.1 m3/s is 8640 m3/d.
    np.testing.assert_allclose(in_seconds.exposure_seconds(0.1), 10.0, rtol=1e-9)
    np.testing.assert_allclose(in_days.exposure_seconds(8640.0), 10.0, rtol=1e-9)


def test_primary_clarifier_removal_is_the_same_on_any_clock(asm1):
    def pc_plant(time_unit):
        plant = Plant("pc", time_unit=time_unit)
        plant.add_unit(PrimaryClarifier("pc", asm1, volume=900.0))
        plant.add_influent("f", asm1.influent(Q=20000.0), to="pc")
        return _resolved(plant).units["pc"]

    in_days = pc_plant("d")._removal_fraction(20000.0)
    in_hours = pc_plant("h")._removal_fraction(20000.0 / 24.0)
    np.testing.assert_allclose(in_hours, in_days, rtol=1e-3)


def test_takacs_default_velocities_follow_the_clock(asm1):
    def settler(time_unit, **kw):
        plant = Plant("st", time_unit=time_unit)
        plant.add_unit(TakacsClarifier("st", asm1, area=1500.0, height=4.0, underflow_Q=500.0, **kw))
        plant.add_influent("f", asm1.influent(Q=20000.0), to="st")
        return _resolved(plant).units["st"]

    in_days = settler("d")
    in_hours = settler("h")
    assert in_days._v0 == 474.0 and in_days._vmax == 250.0
    np.testing.assert_allclose(in_hours._v0, 474.0 / 24.0)
    np.testing.assert_allclose(in_hours._vmax, 250.0 / 24.0)
    # Parameters the user supplies are already in the plant's units.
    own = dict(v0=20.0, vmax=10.0, rh=5.76e-4, rp=2.86e-3, fns=2.28e-3, X_threshold=3000.0)
    assert settler("h", settling_params=own)._v0 == 20.0
