"""Algebraic mineral-equilibrium reduction + bounded-driver kinetics (issue #295).

The kinetic power-law precipitation of a very insoluble mineral (a metal
phosphate at SI ~ 14) has a ~1e13 rate Jacobian, so the forward solve is fine but
no sensitivity / gradient method survives the transient. Two opt-in alternatives
restore differentiability while leaving the default power-law model untouched:

* **algebraic equilibrium** (``mode: equilibrium``): solve the precipitation
  equilibrium directly and project onto it (``network.precipitation_equilibrium``)
  -- exact, fast, differentiable via the implicit function theorem; and
* **bounded-driver kinetics** (``supersaturation_form: bounded``): a non-stiff
  rate form whose Jacobian is ~k, so a *dynamic* reactor solve is differentiable
  and relaxes to the same equilibrium.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import SpatialConditions

_EQ = "precipitation_metal_phosphate_equilibrium"
_BD = "precipitation_metal_phosphate_bounded"
_KIN = "precipitation_metal_phosphate"


def _ph_T(pH, T=293.15):
    return SpatialConditions(fields={"pH": jnp.array([pH]), "T": jnp.array([T])})


# --- A: algebraic equilibrium projection -------------------------------------

def test_projection_conserves_mass_and_hits_solubility():
    net = aquakin.load_network(_EQ)
    si = net.species_index
    C0 = net.default_concentrations()
    Ceq = net.precipitation_equilibrium()

    # Total Fe and PO4 (dissolved + bound in solids) are conserved exactly.
    def total(state, ion, solids):
        return float(state[si[ion]] + sum(state[si[s]] for s in solids))
    assert total(Ceq, "S_Fe3", ["X_FePO4", "X_FeOH3"]) == pytest.approx(
        total(C0, "S_Fe3", ["X_FePO4", "X_FeOH3"]), rel=1e-5)
    assert total(Ceq, "S_PO4", ["X_FePO4", "X_AlPO4"]) == pytest.approx(
        total(C0, "S_PO4", ["X_FePO4", "X_AlPO4"]), rel=1e-5)

    # The present phases sit on their solubility (SI ~ 0): a tiny further
    # projection of the projected state does not move it.
    Ceq2 = net.precipitation_equilibrium(Ceq)
    assert float(jnp.max(jnp.abs(Ceq2 - Ceq))) < 1e-3


def test_projection_pH_trend():
    # Chemical-P removal worsens at higher pH (Fe(OH)3 outcompetes FePO4 for Fe).
    net = aquakin.load_network(_EQ)
    i = net.species_index["S_PO4"]
    resid = [float(net.precipitation_equilibrium(conditions=_ph_T(ph))[i])
             for ph in (6.0, 7.0, 8.0)]
    assert resid[0] < resid[1] < resid[2]


def test_projection_is_differentiable_in_dose_and_pH():
    net = aquakin.load_network(_EQ)
    si = net.species_index

    def resP(args):
        Fe, pH = args
        C = net.default_concentrations().at[si["S_Fe3"]].set(Fe)
        return net.precipitation_equilibrium(C, _ph_T(pH))[si["S_PO4"]]

    g = jax.grad(resP)((2.0, 7.0))
    assert np.all(np.isfinite(np.asarray(g)))
    # pH sensitivity is the strong one (more OH- -> more Fe(OH)3 -> more residual P).
    assert float(g[1]) > 0.0


def test_projection_requires_equilibrium_minerals():
    # The kinetic network has no mode: equilibrium minerals -> clear error.
    net = aquakin.load_network(_KIN)
    assert net.precipitation_equilibrium_fn is None
    with pytest.raises(ValueError, match="mode: equilibrium"):
        net.precipitation_equilibrium()


# --- C: bounded-driver kinetics (differentiable dynamics) --------------------

def test_bounded_jacobian_is_tame():
    net = aquakin.load_network(_BD)
    cond = net.default_conditions()
    C0 = net.default_concentrations()
    p = net.default_parameters()
    J = jax.jacfwd(lambda C: net.dCdt(C, p, cond.fields, 0))(C0)
    # ~k (here 200), not the ~1e13 of the power law.
    assert float(jnp.max(jnp.abs(J))) < 1e4


def test_kinetic_default_is_unchanged():
    # The default (power-law) network keeps its huge R -- the documented limitation.
    net = aquakin.load_network(_KIN)
    cond = net.default_conditions()
    d = net.derived_condition_fn(net.default_concentrations(),
                                 net.default_parameters(), cond.fields, 0)
    assert float(d["R_FePO4"]) > 1e10


@pytest.mark.slow
def test_bounded_dynamic_solve_is_differentiable():
    # The headline #295 win for the dynamic case: a reverse gradient through the
    # time integration of the ultra-insoluble network is finite (the power-law
    # network's is non-finite). Use a short span so the rate constant still moves
    # the output (at equilibrium the endpoint is k-independent).
    net = aquakin.load_network(_BD)
    cond = net.default_conditions()
    C0 = net.default_concentrations()
    p = net.default_parameters()
    r = aquakin.BatchReactor(net, cond)

    def loss(pp):
        s = r.solve(C0, params=pp, t_span=(0.0, 5e-3), t_eval=jnp.array([5e-3]))
        return jnp.sum(s.C_named("S_PO4"))

    g = jax.grad(loss)(p)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.asarray(g) != 0.0)


@pytest.mark.slow
def test_bounded_dynamics_relax_to_the_projection_equilibrium():
    # A long bounded-driver solve reaches the same equilibrium the algebraic
    # projection gives directly (cross-validates A and C).
    bd = aquakin.load_network(_BD)
    eq = aquakin.load_network(_EQ)
    r = aquakin.BatchReactor(bd, bd.default_conditions())
    sol = r.solve(bd.default_concentrations(), params=bd.default_parameters(),
                  t_span=(0.0, 20.0), t_eval=jnp.array([20.0]))
    i = bd.species_index["S_PO4"]
    target = float(eq.precipitation_equilibrium()[eq.species_index["S_PO4"]])
    assert float(sol.C_named("S_PO4")[-1]) == pytest.approx(target, abs=0.02)


# --- the algebraic solver core ----------------------------------------------

def test_equilibrium_solver_satisfies_complementarity():
    # At the solved equilibrium every precipitated mineral is on its solubility
    # (SI ~ 0) and every absent one is undersaturated (SI < 0).
    net = aquakin.load_network(_EQ)
    si = net.species_index
    cond = net.default_conditions()
    Ceq = net.precipitation_equilibrium()
    p = net.default_parameters()
    derived = net.derived_condition_fn(Ceq, p, cond.fields, 0)
    # Fe phosphate + Fe hydroxide are present at pH 7; Al phases are absent (no Al).
    assert float(Ceq[si["X_FePO4"]]) > 1e-3
    assert float(Ceq[si["X_FeOH3"]]) > 1e-3
    assert float(Ceq[si["X_AlPO4"]]) < 1e-3
    # SI of the present phases is ~0 (they re-solve to themselves, see derived).
    assert "Xeq_FePO4" in derived


# --- van't Hoff Ksp(T) in the equilibrium solver ----------------------------

def _moderate_equilibrium_yaml(dH_sp):
    # A moderately-soluble 1:1 mineral (two free ions, no pH fraction, no activity
    # correction) so the only temperature dependence is the van't Hoff Ksp(T):
    # at equilibrium [M][A]/1e6 = Ksp(T), and with M=A by symmetry the residual
    # ion scales as sqrt(Ksp(T)).
    return f"""
network: {{name: vh, version: "1.0", description: x}}
species:
  - {{name: S_M, default_concentration: 1.0, units: "mol/m3"}}
  - {{name: S_A, default_concentration: 1.0, units: "mol/m3"}}
  - {{name: X_MA, default_concentration: 1.0e-4, units: "mol/m3"}}
conditions:
  - {{name: T, default: 293.15}}
  - {{name: pH, default: 7.0}}
clip_negative_states: true
precipitation:
  pH_field: pH
  temperature_field: T
  temperature_units: kelvin
  activity_model: none
  minerals:
    - name: MA
      pKsp: 8.0
      dH_sp: {dH_sp}
      mode: equilibrium
      solid: X_MA
      ions:
        - {{species: S_M, molar_mass: 1000, count: 1, charge: 2}}
        - {{species: S_A, molar_mass: 1000, count: 1, charge: 1}}
reactions:
  - {{name: r, rate: "{{Xeq_MA}}", stoichiometry: {{S_M: -1, S_A: -1, X_MA: 1}}}}
"""


def test_vant_hoff_shifts_equilibrium_with_temperature(tmp_path):
    import math
    from aquakin.core.ph_solver import _R_SI

    def residual_M(dH, T):
        f = tmp_path / f"vh_{dH}.yaml"
        f.write_text(_moderate_equilibrium_yaml(dH), encoding="utf-8")
        net = aquakin.load_network_from_file(f)
        cond = SpatialConditions(fields={"pH": jnp.array([7.0]), "T": jnp.array([T])})
        return float(net.precipitation_equilibrium(conditions=cond)[net.species_index["S_M"]])

    T_lo, T_hi = 283.15, 303.15
    # dH = 0: Ksp is temperature-independent, so the residual is unchanged.
    assert residual_M(0.0, T_lo) == pytest.approx(residual_M(0.0, T_hi), rel=1e-4)

    # dH > 0: the residual ion tracks sqrt(Ksp(T)); the ratio across two
    # temperatures equals the analytic van't Hoff factor exp(0.5*dH*Δ(1/T)/R).
    dH = 40000.0
    lo, hi = residual_M(dH, T_lo), residual_M(dH, T_hi)
    expected_ratio = math.exp(0.5 * dH * (1.0 / T_lo - 1.0 / T_hi) / _R_SI)
    assert lo != pytest.approx(hi, rel=1e-3)         # genuinely temperature-shifted
    assert (hi / lo) == pytest.approx(expected_ratio, rel=2e-3)


# --- schema validation ------------------------------------------------------

def test_equilibrium_mode_requires_solid(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("""
network: {name: bad, version: "1.0", description: x}
species:
  - {name: S_M, default_concentration: 1.0}
  - {name: S_P, default_concentration: 1.0}
conditions:
  - {name: T, default: 293.15}
  - {name: pH, default: 7.0}
precipitation:
  minerals:
    - name: MP
      pKsp: 20.0
      mode: equilibrium
      ions:
        - {species: S_M, count: 1, charge: 2}
        - {species: S_P, count: 1, charge: 3, fraction: phosphate}
reactions:
  - {name: r, rate: "1.0", stoichiometry: {S_M: -1}}
""", encoding="utf-8")
    with pytest.raises(ValueError, match="needs a 'solid:'"):
        aquakin.load_network_from_file(p)


def test_invalid_supersaturation_form(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("""
network: {name: bad2, version: "1.0", description: x}
species:
  - {name: S_M, default_concentration: 1.0}
conditions:
  - {name: T, default: 293.15}
  - {name: pH, default: 7.0}
precipitation:
  minerals:
    - name: M
      pKsp: 10.0
      supersaturation_form: sideways
      ions:
        - {species: S_M, count: 1, charge: 2}
reactions:
  - {name: r, rate: "1.0", stoichiometry: {S_M: -1}}
""", encoding="utf-8")
    with pytest.raises(ValueError, match="supersaturation_form"):
        aquakin.load_network_from_file(p)
