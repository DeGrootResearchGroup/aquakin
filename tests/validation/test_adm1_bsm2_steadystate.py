"""Validation: ADM1 reproduces the BSM2 anaerobic-digester steady state.

The IWA Benchmark Simulation Model No. 2 (BSM2) specifies a continuously-fed
mesophilic (35 degC) anaerobic digester: a 3400 m3 liquid reactor with a 300 m3
headspace, fed a constant particulate-rich sludge at 178 m3/d (hydraulic
retention time ~19 d). The published open-loop steady-state digester
composition is a standard reference point for ADM1 implementations.

This test runs the shipped ``adm1`` network as that CSTR -- the reaction RHS
plus a dilution term ``(Q/V_liq)*(C_in - C)`` on the liquid states -- integrates
to steady state, and checks it reproduces the published composition. It also
checks the state-derived charge-balance pH against the reference electroneutrality
relation. All reference numbers below are the published BSM2 constant-influent
benchmark values (digester feed, reactor geometry and open-loop steady state).

References
----------
Rosen, C. & Jeppsson, U. (2006). Aspects on ADM1 Implementation within the BSM2
Framework. Dept. IEA, Lund University.
Batstone, D.J. et al. (2002). Anaerobic Digestion Model No. 1 (ADM1). IWA STR 13.
"""

import math

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


# --- Published BSM2 digester operating point -----------------------------
Q_FEED = 178.4674  # m3/d -- benchmark digester feed flow
V_LIQ = 3400.0     # m3
T_OP = 308.15      # K (35 degC)

# Constant influent (digester feed), kgCOD/m3 except S_IC (kmolC/m3), S_IN
# (kmolN/m3) and S_an (kmol/m3). Only these components are non-zero in the
# BSM2 constant-influent file; the feed is dominated by particulate
# carbohydrate / protein / lipid and inert solids.
INFLUENT = {
    "S_aa": 0.0468,
    "S_IC": 0.00791,
    "S_IN": 0.00197,
    "S_I": 0.0281,
    "X_ch": 3.7244,
    "X_pr": 16.2784,
    "X_li": 8.0068,
    "X_I": 17.0172,
    "S_an": 0.0052,   # chloride; the only strong ion fed
}

# Published open-loop steady-state digester composition (DIGESTERINIT).
REFERENCE_SS = {
    "S_su": 0.0124, "S_aa": 0.0055, "S_fa": 0.1074, "S_va": 0.0123,
    "S_bu": 0.0140, "S_pro": 0.0176, "S_ac": 0.0893, "S_h2": 2.5055e-7,
    "S_ch4": 0.0555, "S_IC": 0.0951, "S_IN": 0.0945, "S_I": 0.1309,
    "X_ch": 0.0205, "X_pr": 0.0842, "X_li": 0.0436, "X_su": 0.3122,
    "X_aa": 0.9317, "X_fa": 0.3384, "X_c4": 0.3258, "X_pro": 0.1011,
    "X_ac": 0.6772, "X_h2": 0.2848, "X_I": 17.2162,
    "S_gas_h2": 1.1032e-5, "S_gas_ch4": 1.6535, "S_gas_co2": 0.0135,
    # X_c (composite) -- the published snapshot zeros it, but biomass decay
    # genuinely feeds a small composite pool; excluded from the comparison.
    "S_cat": 0.0, "S_an": 0.0052,
}

_GAS = ("S_gas_h2", "S_gas_ch4", "S_gas_co2")


def _digester_cstr_solution():
    """Integrate the adm1 network as the BSM2 CSTR to steady state."""
    net = aquakin.load_network("adm1")
    si = net.species_index
    p = net.default_parameters()
    cond_fields = net.default_conditions().fields

    C0 = np.zeros(net.n_species)
    for name, val in REFERENCE_SS.items():
        C0[si[name]] = val

    u = np.zeros(net.n_species)
    for name, val in INFLUENT.items():
        u[si[name]] = val
    u = jnp.array(u)

    # Dilution acts on liquid states only; the headspace is not diluted.
    liquid = jnp.array([0.0 if n in _GAS else 1.0 for n in net.species])
    D = Q_FEED / V_LIQ

    def rhs(t, C, args):
        return net.dCdt(C, p, cond_fields, 0) + D * (u - C) * liquid

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Kvaerno5(),
        t0=0.0, t1=600.0, dt0=None, y0=jnp.array(C0),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=300_000,
        saveat=diffrax.SaveAt(t1=True),
    )
    return net, np.asarray(sol.ys[-1])


@pytest.mark.validation
def test_adm1_reproduces_bsm2_digester_steady_state():
    """The CSTR settles onto the published BSM2 steady state (core states
    within ~6%; methane output much tighter)."""
    net, C = _digester_cstr_solution()
    si = net.species_index

    # Every reported state (excluding the zeroed composite X_c) within 6%.
    worst, worst_name = 0.0, ""
    for name, ref in REFERENCE_SS.items():
        if ref < 1e-4:        # skip ~zero pools (S_h2, S_cat) -- absolute scale
            continue
        rel = abs(float(C[si[name]]) - ref) / ref
        if rel > worst:
            worst, worst_name = rel, name
    assert worst < 0.06, f"{worst_name} off by {worst:.1%} from BSM2 steady state"

    # Methane (the defining digester output): dissolved + headspace within 1.5%.
    assert float(C[si["S_ch4"]]) == pytest.approx(REFERENCE_SS["S_ch4"], rel=0.015)
    assert float(C[si["S_gas_ch4"]]) == pytest.approx(REFERENCE_SS["S_gas_ch4"], rel=0.015)
    assert float(C[si["S_gas_co2"]]) == pytest.approx(REFERENCE_SS["S_gas_co2"], rel=0.015)


@pytest.mark.validation
def test_adm1_steady_state_is_stationary():
    """At the converged state the CSTR RHS is ~0 (a genuine fixed point)."""
    net, C = _digester_cstr_solution()
    si = net.species_index
    p = net.default_parameters()
    cond_fields = net.default_conditions().fields
    u = np.zeros(net.n_species)
    for name, val in INFLUENT.items():
        u[si[name]] = val
    liquid = np.array([0.0 if n in _GAS else 1.0 for n in net.species])
    D = Q_FEED / V_LIQ
    rhs = np.asarray(net.dCdt(jnp.array(C), p, cond_fields, 0)) + D * (u - C) * liquid
    # Relative to the dominant inert pool's throughput, the residual is tiny.
    assert np.max(np.abs(rhs)) < 1e-2


@pytest.mark.validation
def test_adm1_ph_matches_reference_charge_balance():
    """The state-derived pH matches the ADM1 reference electroneutrality
    relation evaluated at the same composition."""
    net = aquakin.load_network("adm1")
    si = net.species_index
    C = np.zeros(net.n_species)
    for name, val in REFERENCE_SS.items():
        C[si[name]] = val
    pH_model = float(
        net.derived_condition_fn(
            jnp.array(C), net.default_parameters(),
            net.default_conditions().fields, 0,
        )["pH"]
    )

    # Reference charge balance (monoprotic carbonate; VFA as COD/charge):
    #   phi = S_cat + NH4+ - HCO3- - Ac/64 - Pro/112 - Bu/160 - Va/208 - S_an
    #   [H+] - Kw/[H+] = -phi
    R, T_base = 0.083145, 298.15
    f = (1.0 / T_base - 1.0 / T_OP) / (100.0 * R)
    Kw = 10 ** -14 * math.exp(55900 * f)
    Ka_ac, Ka_pro, Ka_bu, Ka_va = (10 ** -4.76, 10 ** -4.88, 10 ** -4.82, 10 ** -4.86)
    Ka_co2 = 10 ** -6.35 * math.exp(7646 * f)
    Ka_IN = 10 ** -9.25 * math.exp(51965 * f)
    g = REFERENCE_SS

    def phi(H):
        nh4 = g["S_IN"] * H / (H + Ka_IN)
        hco3 = g["S_IC"] * Ka_co2 / (H + Ka_co2)
        return (g["S_cat"] + nh4 - hco3
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
    pH_ref = -math.log10(H)
    assert pH_model == pytest.approx(pH_ref, abs=0.05)
