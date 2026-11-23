"""Validation: ADM1 reproduces the BSM2 anaerobic-digester steady state.

The BSM2 digester (35 degC, 3400 m3 liquid, fed 178.47 m3/d; the published
operating point and steady state are in ``bsm2_digester_reference``) is a
standard reference point for ADM1 implementations.

This test runs the shipped ``adm1`` model as that CSTR -- the reaction RHS
plus a dilution term ``(Q/V_liq)*(C_in - C)`` on the liquid states -- integrates
to steady state from a start well away from the published composition, and
checks it settles onto it. It also checks the state-derived charge-balance pH
against the reference electroneutrality relation.

The feed is what the full plant delivers to the digester at steady state, so the
standalone CSTR run is the same boundary-value problem the benchmark solves.

References
----------
Rosen, C. & Jeppsson, U. (2006). Aspects on ADM1 Implementation within the BSM2
Framework. Dept. IEA, Lund University.
Gernaey, K.V. et al. (2014). Benchmarking of Control Strategies for Wastewater
Treatment Plants. IWA Scientific and Technical Report No. 23.
Batstone, D.J. et al. (2002). Anaerobic Digestion Model No. 1 (ADM1). IWA STR 13.
"""

import math

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from tests.validation.bsm2_digester_reference import (
    GAS,
    INFLUENT,
    Q_FEED,
    REFERENCE_SS,
    T_OP,
    V_LIQ,
    perturbed_start,
    state_vector,
    worst_relative_error,
)


def _cstr_rhs(net):
    """The BSM2 digester as a CSTR: reactions plus dilution of the liquid states
    (the headspace is not diluted), and the feed vector."""
    p = net.default_parameters()
    cond_fields = net.default_conditions().fields
    u = jnp.asarray(state_vector(net, INFLUENT))
    liquid = jnp.array([0.0 if n in GAS else 1.0 for n in net.species])
    D = Q_FEED / V_LIQ

    def rhs(t, C, args):
        return net.dCdt(C, p, cond_fields, 0) + D * (u - C) * liquid

    return rhs, np.asarray(u), D


@pytest.fixture(scope="module")
def digester_cstr_solution():
    """Integrate the adm1 model as the BSM2 CSTR for 600 days (~31 hydraulic
    retention times) from ``perturbed_start`` (shared by the tests that read the
    converged state)."""
    net = aquakin.load_model("adm1")
    rhs, _, _ = _cstr_rhs(net)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Kvaerno5(),
        t0=0.0, t1=600.0, dt0=None, y0=jnp.asarray(perturbed_start(net)),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=300_000,
        saveat=diffrax.SaveAt(t1=True),
    )
    return net, np.asarray(sol.ys[-1])


@pytest.mark.validation
def test_adm1_reproduces_bsm2_digester_steady_state(digester_cstr_solution):
    """The CSTR settles onto the published BSM2 steady state to within 2% on
    every state (measured worst 1.27%, S_gas_co2) and methane within 1.5%.

    The biochemical stoichiometry is verified identical to the official BSM2
    ``adm1_ODE_bsm2.c`` (every reaction's coefficients, including the inorganic
    C/N balances, match to machine precision), and the kinetic / equilibrium
    constants match ``adm1init_bsm2.m`` (including the carbonate Ka1 van't Hoff
    enthalpy). With the exact published feed, the residual ~1% is the difference
    between this model's charge-balance pH solver and the reference DAE (the
    pH differs by 0.004; see ``test_adm1_ph_matches_reference_charge_balance``)."""
    net, C = digester_cstr_solution
    si = net.species_index

    worst, worst_name = worst_relative_error(net, C)
    assert worst < 0.02, f"{worst_name} off by {worst:.1%} from BSM2 steady state"

    # Methane (the defining digester output): dissolved + headspace within 1.5%.
    assert float(C[si["S_ch4"]]) == pytest.approx(REFERENCE_SS["S_ch4"], rel=0.015)
    assert float(C[si["S_gas_ch4"]]) == pytest.approx(REFERENCE_SS["S_gas_ch4"], rel=0.015)
    assert float(C[si["S_gas_co2"]]) == pytest.approx(REFERENCE_SS["S_gas_co2"], rel=0.015)


@pytest.mark.validation
def test_adm1_steady_state_is_stationary(digester_cstr_solution):
    """The state the run ends on is a fixed point, so the comparison above is of
    the steady state and not of a slow transient.

    Every state's net rate is measured against its own throughput
    ``D * max(|C|, |C_in|)`` (S_cat, zero in both, must have a zero rate): at the converged state the largest ratio is 3e-9
    (measured), so 1e-6 bounds it with margin. The published composition itself
    is *not* a fixed point of this model (its CO2 headspace rate is ~0.18), which
    is why the run starts away from it and must find the steady state itself."""
    net, C = digester_cstr_solution
    rhs, u, D = _cstr_rhs(net)
    residual = np.abs(np.asarray(rhs(0.0, jnp.asarray(C), None)))
    throughput = D * np.maximum(np.abs(C), np.abs(u))
    fed = throughput > 0.0
    assert np.max(residual[fed] / throughput[fed]) < 1e-6
    # A state neither fed nor present (S_cat) must not move at all.
    assert np.all(residual[~fed] == 0.0)


@pytest.mark.validation
def test_adm1_ph_matches_reference_charge_balance():
    """The state-derived pH matches the ADM1 reference electroneutrality
    relation evaluated at the same composition.

    The reference relation treats carbonate as monoprotic (HCO3- only) and gives
    7.2631, the published BSM2 digester pH. The model's charge balance also
    carries CO3--, which lowers the pH by 0.004 here; the same relation with
    that one term added reproduces the model to 4e-7 (measured). So the model is
    within 0.01 of the reference, and the whole difference is that term."""
    net = aquakin.load_model("adm1")
    C = state_vector(net, REFERENCE_SS)
    pH_model = float(
        net.derived_condition_fn(
            jnp.array(C), net.default_parameters(),
            net.default_conditions().fields, 0,
        )["pH"]
    )

    # Reference charge balance (VFA as COD/charge):
    #   phi = S_cat + NH4+ - HCO3- - Ac/64 - Pro/112 - Bu/160 - Va/208 - S_an
    #   [H+] - Kw/[H+] = -phi
    # with ``with_carbonate`` adding the 2 CO3-- term to the carbonate charge.
    R, T_base = 0.083145, 298.15
    f = (1.0 / T_base - 1.0 / T_OP) / (100.0 * R)
    Kw = 10 ** -14 * math.exp(55900 * f)
    Ka_ac, Ka_pro, Ka_bu, Ka_va = (10 ** -4.76, 10 ** -4.88, 10 ** -4.82, 10 ** -4.86)
    Ka_co2 = 10 ** -6.35 * math.exp(7646 * f)
    Ka_hco3 = 10 ** -10.33 * math.exp(14900 * f)
    Ka_IN = 10 ** -9.25 * math.exp(51965 * f)
    g = REFERENCE_SS

    def reference_ph(with_carbonate):
        def phi(H):
            nh4 = g["S_IN"] * H / (H + Ka_IN)
            if with_carbonate:
                denom = H * H + Ka_co2 * H + Ka_co2 * Ka_hco3
                ic_charge = g["S_IC"] * (Ka_co2 * H + 2.0 * Ka_co2 * Ka_hco3) / denom
            else:
                ic_charge = g["S_IC"] * Ka_co2 / (H + Ka_co2)
            return (g["S_cat"] + nh4 - ic_charge
                    - g["S_ac"] / 64 * Ka_ac / (H + Ka_ac)
                    - g["S_pro"] / 112 * Ka_pro / (H + Ka_pro)
                    - g["S_bu"] / 160 * Ka_bu / (H + Ka_bu)
                    - g["S_va"] / 208 * Ka_va / (H + Ka_va)
                    - g["S_an"])

        lo, hi = 1e-10, 1e-4
        for _ in range(200):
            H = math.sqrt(lo * hi)
            if H - Kw / H + phi(H) > 0:
                hi = H
            else:
                lo = H
        return -math.log10(H)

    assert pH_model == pytest.approx(reference_ph(with_carbonate=False), abs=0.01)
    assert pH_model == pytest.approx(reference_ph(with_carbonate=True), abs=1e-5)
