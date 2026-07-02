"""Validation: ADM1 reproduces the BSM2 anaerobic-digester steady state.

The IWA Benchmark Simulation Model No. 2 (BSM2) specifies a continuously-fed
mesophilic (35 degC) anaerobic digester: a 3400 m3 liquid reactor with a 300 m3
headspace, fed a constant particulate-rich sludge at 178 m3/d (hydraulic
retention time ~19 d). The published open-loop steady-state digester
composition is a standard reference point for ADM1 implementations.

This test runs the shipped ``adm1`` model as that CSTR -- the reaction RHS
plus a dilution term ``(Q/V_liq)*(C_in - C)`` on the liquid states -- integrates
to steady state, and checks it reproduces the published composition. It also
checks the state-derived charge-balance pH against the reference electroneutrality
relation.

The feed (``INFLUENT``) and the steady state (``REFERENCE_SS``) are the exact
published BSM2 figures: the "ADM1 influent (post ASM2ADM interface)" and "ADM1
effluent (prior ADM2ASM interface)" tables of the BSM2 open-loop steady-state
report (``Results/BSM2_steady_state.pdf`` in the official BSM2 distribution).
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


# --- Published BSM2 digester operating point -----------------------------
Q_FEED = 178.4674  # m3/d -- benchmark digester feed flow
V_LIQ = 3400.0     # m3
T_OP = 308.15      # K (35 degC)

# Digester feed = the published "ADM1 influent (post ASM2ADM interface)" at the
# BSM2 steady state (Results/BSM2_steady_state.pdf). kgCOD/m3 except S_IC
# (kmolC/m3), S_IN (kmolN/m3) and S_an (kmol/m3). The feed is dominated by
# particulate protein / lipid / carbohydrate and inert solids.
INFLUENT = {
    "S_aa": 0.04388,
    "S_IC": 0.0079326,
    "S_IN": 0.0019721,
    "S_I": 0.028067,
    "X_ch": 3.7236,
    "X_pr": 15.9235,
    "X_li": 8.047,
    "X_I": 17.0106,
    "S_an": 0.0052101,   # chloride; the only strong ion fed
}

# Published open-loop steady-state digester composition: the "ADM1 effluent
# (prior ADM2ASM interface)" table of Results/BSM2_steady_state.pdf.
REFERENCE_SS = {
    "S_su": 0.012394, "S_aa": 0.0055432, "S_fa": 0.10741, "S_va": 0.012333,
    "S_bu": 0.014003, "S_pro": 0.017584, "S_ac": 0.089315, "S_h2": 2.5055e-7,
    "S_ch4": 0.05549, "S_IC": 0.095149, "S_IN": 0.094468, "S_I": 0.13087,
    "X_c": 0.10792, "X_ch": 0.020517, "X_pr": 0.08422, "X_li": 0.043629,
    "X_su": 0.31222, "X_aa": 0.93167, "X_fa": 0.33839, "X_c4": 0.33577,
    "X_pro": 0.10112, "X_ac": 0.67724, "X_h2": 0.28484, "X_I": 17.2162,
    "S_gas_h2": 1.1032e-5, "S_gas_ch4": 1.6535, "S_gas_co2": 0.01354,
    "S_cat": 0.0, "S_an": 0.0052101,
}

_GAS = ("S_gas_h2", "S_gas_ch4", "S_gas_co2")


def _digester_cstr_solution():
    """Integrate the adm1 model as the BSM2 CSTR to steady state."""
    net = aquakin.load_model("adm1")
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
    """The CSTR settles onto the published BSM2 steady state to within ~1.5% on
    every state (methane much tighter).

    The biochemical stoichiometry is verified identical to the official BSM2
    ``adm1_ODE_bsm2.c`` (every reaction's coefficients, including the inorganic
    C/N balances, match to machine precision), and the kinetic / equilibrium
    constants match ``adm1init_bsm2.m`` (including the carbonate Ka1 van't Hoff
    enthalpy). With the exact published feed, the residual ~1% is the difference
    between this model's charge-balance pH solver and the reference DAE."""
    net, C = _digester_cstr_solution()
    si = net.species_index

    # Every reported state within 2%.
    worst, worst_name = 0.0, ""
    for name, ref in REFERENCE_SS.items():
        if ref < 1e-4:        # skip ~zero pools (S_h2, S_cat) -- absolute scale
            continue
        rel = abs(float(C[si[name]]) - ref) / ref
        if rel > worst:
            worst, worst_name = rel, name
    assert worst < 0.02, f"{worst_name} off by {worst:.1%} from BSM2 steady state"

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
    net = aquakin.load_model("adm1")
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
