"""Validation: the ADM1DigesterUnit reproduces the BSM2 digester steady state.

Same published operating point as ``test_adm1_bsm2_steadystate`` (3400 m3
liquid, 35 degC, fed 178.47 m3/d), but driven through the plant
:class:`ADM1DigesterUnit.rhs` rather than an inline RHS — so it checks the unit
wrapper (feed mixing, liquid-only dilution mask, gas headspace) is faithful.
"""

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.digester import ADM1DigesterUnit
from aquakin.plant.streams import Stream
from tests.validation.bsm2_digester_reference import (
    INFLUENT,
    Q_FEED,
    REFERENCE_SS,
    T_OP,
    V_LIQ,
    perturbed_start,
    state_vector,
    worst_relative_error,
)

# Every test here runs in the validation job, which selects by this marker.
pytestmark = pytest.mark.validation


def _solve_unit():
    net = aquakin.load_model("adm1")
    unit = ADM1DigesterUnit(name="digester", model=net, volume=V_LIQ,
                            conditions={"T": T_OP})
    params = net.default_parameters()
    s_in = Stream(Q=jnp.asarray(Q_FEED), C=jnp.asarray(state_vector(net, INFLUENT)), model=net)

    def rhs(t, C, args):
        return unit.rhs(t, C, {"inlet": s_in}, params)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Kvaerno5(),
        t0=0.0, t1=600.0, dt0=None, y0=jnp.asarray(perturbed_start(net)),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=300_000, saveat=diffrax.SaveAt(t1=True),
    )
    return net, np.asarray(sol.ys[-1])


def test_digester_unit_reproduces_bsm2_steady_state():
    net, C = _solve_unit()
    si = net.species_index
    worst, worst_name = worst_relative_error(net, C)
    # Every state within 2% of the published BSM2 digester steady state, reached
    # from a start away from it. The stoichiometry is verified identical to the
    # official adm1_ODE_bsm2.c and the constants to adm1init_bsm2.m; the residual
    # ~1% is the charge-balance vs reference-DAE pH difference. Methane is tight.
    assert worst < 0.02, f"{worst_name} off by {worst:.1%}"
    # Methane output (defining digester quantity).
    assert float(C[si["S_gas_ch4"]]) == pytest.approx(REFERENCE_SS["S_gas_ch4"], rel=0.015)


def test_digester_effluent_flow_equals_feed():
    """The liquid effluent leaves at the feed flow (constant liquid volume)."""
    net = aquakin.load_model("adm1")
    unit = ADM1DigesterUnit(name="d", model=net, volume=V_LIQ)
    s_in = Stream(Q=jnp.asarray(Q_FEED), C=net.default_concentrations(), model=net)
    out = unit.compute_outputs(jnp.asarray(0.0), net.default_concentrations(),
                               {"inlet": s_in}, net.default_parameters())
    assert float(out["effluent"].Q) == pytest.approx(Q_FEED, rel=1e-9)


def test_digester_effluent_temperature_is_flow_weighted():
    """The effluent temperature is the flow-weighted inlet temperature (a heat
    balance, like every other multi-inlet unit) -- not the first inlet's T, which
    would ignore a second feed at a different temperature."""
    net = aquakin.load_model("adm1")
    C = net.default_concentrations()
    unit = ADM1DigesterUnit(name="d", model=net, volume=V_LIQ,
                            input_port_names=["feed", "reject"])
    inputs = {
        "feed":   Stream(Q=jnp.asarray(100.0), C=C, model=net, scalars={"T": jnp.asarray(308.15)}),
        "reject": Stream(Q=jnp.asarray(50.0),  C=C, model=net, scalars={"T": jnp.asarray(290.15)}),
    }
    out = unit.compute_outputs(jnp.asarray(0.0), C, inputs, net.default_parameters())
    expected = (100.0 * 308.15 + 50.0 * 290.15) / 150.0
    assert float(out["effluent"].scalars["T"]) == pytest.approx(expected, rel=1e-9)
    # A temperature-agnostic inlet is IGNORED, not allowed to force the whole mix
    # agnostic: the effluent carries the temperature-bearing feed's T. (This is
    # what lets temperature propagate around a loop seeded with an agnostic
    # zero-flow recycle stream -- see streams.mixed_scalars.)
    inputs["reject"] = Stream(Q=jnp.asarray(50.0), C=C, model=net)
    out_partial = unit.compute_outputs(jnp.asarray(0.0), C, inputs,
                                       net.default_parameters())
    assert float(out_partial["effluent"].scalars["T"]) == pytest.approx(308.15, rel=1e-9)
    # Only when NO inlet carries a temperature is the effluent agnostic.
    inputs["feed"] = Stream(Q=jnp.asarray(100.0), C=C, model=net)
    out_none = unit.compute_outputs(jnp.asarray(0.0), C, inputs,
                                    net.default_parameters())
    assert out_none["effluent"].scalars.get("T") is None


def test_gas_transfer_scales_with_digester_volume():
    """The gas-transfer headspace gain uses V_liq/V_gas, with V_liq slaved to the
    unit's liquid volume -- so a digester of a different size transfers
    proportionally (the old code hard-coded the BSM2 3400/300 ratio). With the
    headspace empty (no back-pressure, no outflow), dS_gas/dt is pure transfer-in
    and scales linearly with the liquid volume."""
    adm1 = aquakin.load_model("adm1")
    # A liquid state above the Henry equilibrium so transfer is into the
    # headspace; gas states zeroed so p_gas = 0 (no back-pressure / outflow).
    C = adm1.default_concentrations()
    for sp in ("S_gas_h2", "S_gas_ch4", "S_gas_co2"):
        C = C.at[adm1.species_index[sp]].set(0.0)
    C = C.at[adm1.species_index["S_ch4"]].set(0.06)
    inp = {"inlet": Stream(Q=jnp.asarray(0.0), C=C, model=adm1)}  # no dilution
    gi = adm1.species_index["S_gas_ch4"]

    def gas_rate(volume):
        d = ADM1DigesterUnit(name="d", model=adm1, volume=volume,
                             conditions={"T": 308.15})
        return float(d.rhs(0.0, C, inp, adm1.default_parameters())[gi])

    r_full, r_half = gas_rate(3400.0), gas_rate(1700.0)
    assert r_full > 0.0
    assert r_full == pytest.approx(2.0 * r_half, rel=1e-6)  # 3400/1700


def test_biogas_outflow_clipped_when_subatmospheric():
    """The overpressure outflow k_P*max(0, P_gas - P_atm) holds the valve shut
    (no gas drawn back) if the headspace is transiently below atmospheric, while
    being identical to the un-clipped form at the operating point (P_gas > P_atm)."""
    adm1 = aquakin.load_model("adm1")
    p = adm1.default_parameters()
    conds = {f: jnp.asarray([v]) for f, v in adm1._condition_defaults.items()}
    ri = adm1.reaction_names.index("gas_outflow_ch4")

    def outflow(C):
        return float(adm1.rates(C, p, conds, 0)[ri])

    C = adm1.default_concentrations()
    assert outflow(C) > 0.0                       # operating point: P_gas > P_atm
    # Empty headspace -> P_gas = p_h2o << P_atm -> clipped to zero (no backflow).
    for sp in ("S_gas_h2", "S_gas_ch4", "S_gas_co2"):
        C = C.at[adm1.species_index[sp]].set(0.0)
    assert outflow(C) == 0.0
