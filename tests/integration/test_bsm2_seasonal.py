"""Seasonal-temperature BSM2 test: cold influent slows nitrification.

Exercises the end-to-end temperature path -- a temperature-carrying influent
flows through the plant (mixers heat-balance it, units pass it through), each AS
reactor reads its inlet temperature, and the ASM1 temperature corrections
(re-referenced to the BSM2 15 degC base by ``bsm2_asm1_network``) slow the
kinetics in the cold. At the 15 degC reference temperature the plant reproduces
the validated steady state exactly (the correction is unity there); colder
influent leaves more residual ammonia (nitrification, the most
temperature-sensitive process, slows), warmer leaves less.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm import bsm2_warm_start
from aquakin.plant.bsm.bsm2 import (
    BSM2_CONSTANT_INFLUENT,
    BSM2_Q_REF,
    build_bsm2,
    bsm2_asm1_network,
    bsm2_parameters,
)
from aquakin.plant.influent import InfluentSeries


def _tank5_snh_at(asm1, adm1, params, T_kelvin):
    C = asm1.default_concentrations() * 0.0
    for sp, v in BSM2_CONSTANT_INFLUENT.items():
        C = C.at[asm1.species_index[sp]].set(v)
    infl = InfluentSeries(t=jnp.array([0.0, 1e4]), Q=jnp.full((2,), BSM2_Q_REF),
                          C=jnp.tile(C, (2, 1)), network=asm1,
                          T=jnp.full((2,), float(T_kelvin)))
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", infl)
    y0 = bsm2_warm_start(plant)
    sol = plant.solve(t_span=(0.0, 150.0), t_eval=jnp.array([0.0, 150.0]),
                      params=params, y0=y0,
                      rtol=1e-5, atol=1e-3,
                      integrator=aquakin.IntegratorConfig(max_steps=500_000))
    assert jnp.all(jnp.isfinite(sol.state))
    return float(sol.C_named("tank5", "SNH")[-1])


@pytest.mark.validation
def test_bsm2_cold_influent_slows_nitrification():
    asm1 = bsm2_asm1_network()           # temperature corrections referenced to 15 °C
    adm1 = aquakin.load_network("adm1")
    params = bsm2_parameters(asm1, adm1)
    snh_cold = _tank5_snh_at(asm1, adm1, params, 283.15)   # 10 °C
    snh_warm = _tank5_snh_at(asm1, adm1, params, 293.15)   # 20 °C
    # Colder water nitrifies more slowly, so more ammonia escapes.
    assert snh_cold > snh_warm
    # Both stay in the well-nitrified regime (this loading is over-designed).
    assert snh_warm < snh_cold < 5.0


def test_temperature_influent_rhs_finite_and_active():
    """A cheap RHS check: a temperature-carrying influent gives a finite RHS,
    and the tank-5 ammonia derivative differs between a cold and a warm influent
    (the temperature reaches the kinetics through the recycle loop -- it does not
    if the seeded recycle stream drops its temperature)."""
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    params = bsm2_parameters(asm1, adm1)
    C = asm1.default_concentrations() * 0.0
    for sp, v in BSM2_CONSTANT_INFLUENT.items():
        C = C.at[asm1.species_index[sp]].set(v)

    def rhs_dsnh(T_kelvin):
        infl = InfluentSeries(t=jnp.array([0.0, 1e4]), Q=jnp.full((2,), BSM2_Q_REF),
                              C=jnp.tile(C, (2, 1)), network=asm1,
                              T=jnp.full((2,), float(T_kelvin)))
        plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
        plant.add_influent("feed", infl)
        y0 = bsm2_warm_start(plant)
        d = plant.derivative(y0, params)          # dstate/dt, no full solve
        assert jnp.all(jnp.isfinite(d))
        tank5_rate = plant.states_by_unit(d)["tank5"]
        return float(tank5_rate[asm1.species_index["SNH"]])

    # Faster nitrification when warm -> more negative SNH derivative.
    assert rhs_dsnh(293.15) < rhs_dsnh(283.15) - 1.0
