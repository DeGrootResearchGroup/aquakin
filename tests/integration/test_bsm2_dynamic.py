"""Dynamic-influent BSM2 integration tests.

Drives the open-loop BSM2 plant with a time-varying influent and checks it
integrates efficiently to a finite, stable trajectory. This is the BSM2-scale
counterpart of the BSM1 dynamic test and exercises the fixed-flow-pump fix on
the 167-state, two-network (ASM1 + ADM1) plant: the recycle flows stay bounded
under diurnal / wet-weather forcing, so the monolithic solve does not blow up.

The influent files are synthesised (see ``scripts/generate_bsm2_influent.py``),
so these assert *qualitative* behaviour (finite, stable, plant stays healthy),
not published BSM2 dynamic metrics.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm.bsm2 import (
    build_bsm2,
    bsm2_constant_influent,
    bsm2_parameters,
)
from aquakin.plant.influent import load_bsm2_influent


_WARM = {"SI": 28.06, "SS": 2.0, "XI": 1532.3, "XS": 45.0, "XB_H": 2244.0,
         "XB_A": 167.0, "XP": 967.0, "SO": 1.0, "SNO": 7.0, "SNH": 3.0,
         "SND": 0.7, "XND": 3.0, "SALK": 5.0}

_SS_CACHE = {}


def _networks():
    return aquakin.load_network("asm1"), aquakin.load_network("adm1")


def _build(asm1, adm1, influent):
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", influent, to="front_mix.fresh")
    return plant


def _steady_state(asm1, adm1):
    """A warm BSM2 steady state (computed once, cached) used as the dynamic-run
    initial condition."""
    if "y" in _SS_CACHE:
        return _SS_CACHE["y"]
    plant = _build(asm1, adm1, bsm2_constant_influent(asm1))
    warm = asm1.concentrations(_WARM)
    tanks = ("tank1", "tank2", "tank3", "tank4", "tank5")
    y0 = plant.initial_state(overrides={tk: warm for tk in tanks})
    sol = plant.solve(t_span=(0.0, 150.0), t_eval=jnp.array([0.0, 150.0]),
                      params=bsm2_parameters(asm1, adm1), y0=y0,
                      rtol=1e-5, atol=1e-3, max_steps=500_000)
    _SS_CACHE["y"] = sol.state[-1]
    return _SS_CACHE["y"]


def _run_dynamic(profile, t_end=14.0):
    asm1, adm1 = _networks()
    y_ss = _steady_state(asm1, adm1)
    plant = _build(asm1, adm1, load_bsm2_influent(profile, asm1))
    n_save = int(t_end) + 1
    sol = plant.solve(
        t_span=(0.0, t_end), t_eval=jnp.linspace(0.0, t_end, n_save),
        params=bsm2_parameters(asm1, adm1), y0=y_ss,
        rtol=1e-4, atol=1e-3, max_steps=200_000,
    )
    return plant, sol


def test_bsm2_dry_weather_runs_dynamic():
    """The dynamic dry-weather influent drives the full plant efficiently to a
    finite, healthy trajectory (the recycle flows stay bounded -- issue #30 at
    BSM2 scale)."""
    plant, sol = _run_dynamic("dry", t_end=14.0)
    assert jnp.all(jnp.isfinite(sol.state))
    # Activated sludge stays healthy under diurnal forcing.
    assert float(sol.C_named("tank5", "XB_H")[-1]) > 1500.0
    assert float(sol.C_named("tank5", "SNH")[-1]) < 3.0
    assert float(sol.C_named("tank5", "SNO")[-1]) > 3.0
    # Digester keeps producing methane.
    adm1 = plant.units["digester"].network
    start, size = plant._state_layout["digester"]
    dstate = sol.state[-1, start:start + size]
    assert float(dstate[adm1.species_index["S_gas_ch4"]]) > 1.0


def test_bsm2_rain_weather_stable():
    """A wet-weather (rain) event -- doubling the influent flow -- stays finite
    and stable (the fixed recycle pumps keep throughput bounded)."""
    plant, sol = _run_dynamic("rain", t_end=14.0)
    assert jnp.all(jnp.isfinite(sol.state))
    assert float(sol.C_named("tank5", "XB_H")[-1]) > 1500.0
