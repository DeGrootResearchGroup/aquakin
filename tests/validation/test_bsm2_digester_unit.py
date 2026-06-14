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


Q_FEED = 178.4674
V_LIQ = 3400.0

INFLUENT = {
    "S_aa": 0.0468, "S_IC": 0.00791, "S_IN": 0.00197, "S_I": 0.0281,
    "X_ch": 3.7244, "X_pr": 16.2784, "X_li": 8.0068, "X_I": 17.0172,
    "S_an": 0.0052,
}

REFERENCE_SS = {
    "S_su": 0.0124, "S_aa": 0.0055, "S_fa": 0.1074, "S_va": 0.0123,
    "S_bu": 0.0140, "S_pro": 0.0176, "S_ac": 0.0893, "S_ch4": 0.0555,
    "S_IC": 0.0951, "S_IN": 0.0945, "S_I": 0.1309, "X_ch": 0.0205,
    "X_pr": 0.0842, "X_li": 0.0436, "X_su": 0.3122, "X_aa": 0.9317,
    "X_fa": 0.3384, "X_c4": 0.3258, "X_pro": 0.1011, "X_ac": 0.6772,
    "X_h2": 0.2848, "X_I": 17.2162, "S_gas_ch4": 1.6535, "S_gas_co2": 0.0135,
}


def _solve_unit():
    net = aquakin.load_network("adm1")
    si = net.species_index
    unit = ADM1DigesterUnit(name="digester", network=net, volume=V_LIQ,
                            conditions={"T": 308.15})
    params = net.default_parameters()

    C0 = np.zeros(net.n_species)
    for name, val in REFERENCE_SS.items():
        C0[si[name]] = val
    u = np.zeros(net.n_species)
    for name, val in INFLUENT.items():
        u[si[name]] = val
    s_in = Stream(Q=jnp.asarray(Q_FEED), C=jnp.asarray(u), network=net)

    def rhs(t, C, args):
        return unit.rhs(t, C, {"inlet": s_in}, params)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Kvaerno5(),
        t0=0.0, t1=600.0, dt0=None, y0=jnp.asarray(C0),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=300_000, saveat=diffrax.SaveAt(t1=True),
    )
    return net, np.asarray(sol.ys[-1])


@pytest.mark.validation
def test_digester_unit_reproduces_bsm2_steady_state():
    net, C = _solve_unit()
    si = net.species_index
    worst, worst_name = 0.0, ""
    for name, ref in REFERENCE_SS.items():
        if ref < 1e-4:
            continue
        rel = abs(float(C[si[name]]) - ref) / ref
        if rel > worst:
            worst, worst_name = rel, name
    # Every state within ~6%; the biochemical stoichiometry is verified identical
    # to the official BSM2 adm1_ODE_bsm2.c and the kinetic/equilibrium constants
    # to adm1init_bsm2.m. The residual few-percent spread is the charge-balance
    # vs reference-DAE pH difference, amplified by the inhibition-sensitive
    # acetate/C4 methanogens. Methane is tight.
    assert worst < 0.06, f"{worst_name} off by {worst:.1%}"
    # Methane output (defining digester quantity).
    assert float(C[si["S_gas_ch4"]]) == pytest.approx(REFERENCE_SS["S_gas_ch4"], rel=0.015)


@pytest.mark.validation
def test_digester_effluent_flow_equals_feed():
    """The liquid effluent leaves at the feed flow (constant liquid volume)."""
    net = aquakin.load_network("adm1")
    unit = ADM1DigesterUnit(name="d", network=net, volume=V_LIQ)
    s_in = Stream(Q=jnp.asarray(Q_FEED), C=net.default_concentrations(), network=net)
    out = unit.compute_outputs(jnp.asarray(0.0), net.default_concentrations(),
                               {"inlet": s_in}, net.default_parameters())
    assert float(out["effluent"].Q) == pytest.approx(Q_FEED, rel=1e-9)


def test_gas_transfer_scales_with_digester_volume():
    """The gas-transfer headspace gain uses V_liq/V_gas, with V_liq slaved to the
    unit's liquid volume -- so a digester of a different size transfers
    proportionally (the old code hard-coded the BSM2 3400/300 ratio). With the
    headspace empty (no back-pressure, no outflow), dS_gas/dt is pure transfer-in
    and scales linearly with the liquid volume."""
    adm1 = aquakin.load_network("adm1")
    # A liquid state above the Henry equilibrium so transfer is into the
    # headspace; gas states zeroed so p_gas = 0 (no back-pressure / outflow).
    C = adm1.default_concentrations()
    for sp in ("S_gas_h2", "S_gas_ch4", "S_gas_co2"):
        C = C.at[adm1.species_index[sp]].set(0.0)
    C = C.at[adm1.species_index["S_ch4"]].set(0.06)
    inp = {"inlet": Stream(Q=jnp.asarray(0.0), C=C, network=adm1)}  # no dilution
    gi = adm1.species_index["S_gas_ch4"]

    def gas_rate(volume):
        d = ADM1DigesterUnit(name="d", network=adm1, volume=volume,
                             conditions={"T": 308.15})
        return float(d.rhs(0.0, C, inp, adm1.default_parameters())[gi])

    r_full, r_half = gas_rate(3400.0), gas_rate(1700.0)
    assert r_full > 0.0
    assert r_full == pytest.approx(2.0 * r_half, rel=1e-6)  # 3400/1700


def test_biogas_outflow_clipped_when_subatmospheric():
    """The overpressure outflow k_P*max(0, P_gas - P_atm) holds the valve shut
    (no gas drawn back) if the headspace is transiently below atmospheric, while
    being identical to the un-clipped form at the operating point (P_gas > P_atm)."""
    adm1 = aquakin.load_network("adm1")
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
