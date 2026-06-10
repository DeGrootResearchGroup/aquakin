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
