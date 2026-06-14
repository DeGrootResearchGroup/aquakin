"""Single-stage partial-nitritation/anammox (PN/A) sidestream deammonification.

A continuously-fed low-DO CSTR on the asm3_2step_anammox network must reach an
autotrophic-nitrogen-removal steady state: anammox is retained, NOB are out-
competed for nitrite and wash out, and most of the ammonium leaves as N2 with no
organic carbon. This is a long (slow-anammox) plant solve, so it runs in the
merge-only slow suite.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import Aeration, CSTRUnit, InfluentSeries, Plant

pytestmark = pytest.mark.slow


def _solve_pna(t_end=400.0):
    net = aquakin.load_network("asm3_2step_anammox")
    tank = CSTRUnit(
        name="reactor", network=net, volume=3000.0, input_port_names=["in"],
        conditions={"T": 303.15},
        aeration=Aeration(kla=4.0, species="SO2"), output_port="out",
    )
    plant = Plant("pna")
    plant.add_unit(tank)
    feed = InfluentSeries.constant(net, {"SNH4": 500.0, "SALK": 0.065}, Q=100.0,
                                   base="zero")
    plant.add_influent("feed", feed, to="reactor.in")
    seed = net.concentrations(
        {"XAOB": 100.0, "XNOB": 10.0, "XAMX": 300.0, "SO2": 0.3, "SALK": 0.065},
        base="zero")
    y0 = plant.initial_state(overrides={"reactor": seed})
    sol = plant.solve(t_span=(0.0, t_end),
                      t_eval=jnp.linspace(0.0, t_end, int(t_end) + 1),
                      y0=y0, max_steps=400_000)
    eff = plant.stream(sol, "reactor.out")
    return {s: float(eff.C_named(s)[-1])
            for s in ("SNH4", "SNO2", "SNO3", "SN2", "XAOB", "XNOB", "XAMX")}


def test_pna_autotrophic_nitrogen_removal():
    e = _solve_pna()
    assert all(np.isfinite(v) for v in e.values())
    NH4_in = 500.0
    tin_eff = e["SNH4"] + e["SNO2"] + e["SNO3"]
    removal = (NH4_in - tin_eff) / NH4_in
    assert removal > 0.7                              # autotrophic N removal
    # Anammox retained; NOB out-competed and washed out (PN/A signature).
    assert e["XAMX"] > 5.0
    assert e["XNOB"] < 1.0
    # Not full nitrification: nitrate stays near the anammox byproduct level,
    # well below what nitrifying all the ammonium would give.
    assert e["SNO3"] < 0.3 * NH4_in
    # N2 is the dominant product.
    assert e["SN2"] > 0.6 * NH4_in


def test_pna_is_a_steady_state():
    # The reported state must be a genuine steady state, not a transient: the
    # effluent at 400 d and 800 d agree.
    e400 = _solve_pna(400.0)
    e800 = _solve_pna(800.0)
    for s in ("SNH4", "SNO2", "SNO3", "XAMX", "XNOB"):
        assert e800[s] == pytest.approx(e400[s], abs=1.0)
