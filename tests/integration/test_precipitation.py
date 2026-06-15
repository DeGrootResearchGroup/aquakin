"""Mineral precipitation: the ``precipitation:`` block (Kazadi Mbamba et al. 2015).

Covers the shipped struvite + calcite network (precipitation under
supersaturation, exact elemental mass balance, relaxation to saturation,
dissolution when undersaturated, pH dependence, gradients) and the
speciation->precipitation composition (a state-derived pH feeding the saturation
index), plus the schema validation.
"""
import os
import tempfile
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


@pytest.fixture
def net():
    return aquakin.load_network("precipitation_struvite_calcite")


def _solve(net, t_end=1.0, n=11, T=308.15, pH=7.8, **C0):
    cond = aquakin.SpatialConditions.uniform(T=T, pH=pH)
    r = aquakin.BatchReactor(net, cond)
    C = net.concentrations(C0) if C0 else net.default_concentrations()
    sol = r.solve(C, params=net.default_parameters(), t_span=(0.0, t_end),
                  t_eval=jnp.linspace(0.0, t_end, n))
    return sol


def test_structure(net):
    assert net.n_species == 7 and net.n_reactions == 2
    # The precipitation block exposes SI_/R_ fields per mineral as conditions.
    assert set(net.derived_fields) == {"SI_struvite", "R_struvite",
                                       "SI_calcite", "R_calcite"}
    assert "X_struvite" in net.species and "X_calcite" in net.species


def test_precipitation_and_exact_mass_balance(net):
    sol = _solve(net)
    s = lambda n: np.asarray(sol.C_named(n))
    # Ions consumed, solids formed.
    assert s("S_PO4")[-1] < s("S_PO4")[0] and s("S_Ca")[-1] < s("S_Ca")[0]
    assert s("X_struvite")[-1] > 1.0 and s("X_calcite")[-1] > 1.0
    # Each element is conserved between the aqueous total and the solid (one mole
    # of mineral carries one mole of each constituent).
    for el, sol_sp, solid in [("P", "S_PO4", "X_struvite"),
                              ("Mg", "S_Mg", "X_struvite"),
                              ("N", "S_NH", "X_struvite"),
                              ("Ca", "S_Ca", "X_calcite"),
                              ("IC", "S_IC", "X_calcite")]:
            tot = s(sol_sp) + s(solid)
            assert np.allclose(tot, tot[0], rtol=1e-5), f"{el} not conserved"


def test_relaxes_to_saturation(net):
    # As ions deplete the supersaturation falls toward the saturation index 0.
    sol = _solve(net)
    p = net.default_parameters()
    def SI(C, name):
        d = net.derived_condition_fn(jnp.asarray(C), p,
                                     {"T": jnp.array([308.15]), "pH": jnp.array([7.8])}, 0)
        return float(d[name])
    assert SI(sol.C[0], "SI_struvite") > 0.5          # starts supersaturated
    assert 0.0 <= SI(sol.C[-1], "SI_struvite") < 0.2   # ends near saturation
    assert 0.0 <= SI(sol.C[-1], "SI_calcite") < 0.2


def test_undersaturated_solid_dissolves(net):
    # Solid present but the solution dilute (undersaturated): the sign-preserving
    # rate runs in reverse -- the solid dissolves and ions return to solution.
    sol = _solve(net, S_Mg=0.2, S_PO4=0.1, S_NH=5.0, S_Ca=0.1, S_IC=2.0,
                 X_struvite=2.0, X_calcite=2.0)
    s = lambda n: np.asarray(sol.C_named(n))
    assert s("X_struvite")[-1] < s("X_struvite")[0]    # dissolves
    assert s("S_PO4")[-1] > s("S_PO4")[0]              # ions return


def test_struvite_favoured_at_higher_pH(net):
    # Struvite supersaturation rises with pH (the deprotonated PO4^3- fraction
    # grows), so the same supernatant precipitates more struvite at pH 8.5 than 7.
    hi = _solve(net, pH=8.5)
    lo = _solve(net, pH=7.0)
    assert float(hi.C_named("X_struvite")[-1]) > float(lo.C_named("X_struvite")[-1])


def test_grad_through_solve_is_finite(net):
    cond = aquakin.SpatialConditions.uniform(T=308.15, pH=7.8)
    r = aquakin.BatchReactor(net, cond)
    C0 = net.default_concentrations()

    def loss(p):
        sol = r.solve(C0, params=p, t_span=(0.0, 0.2),
                      t_eval=jnp.linspace(0.0, 0.2, 4))
        return jnp.sum(sol.C_named("X_struvite"))

    g = jax.grad(loss)(net.default_parameters())
    assert jnp.all(jnp.isfinite(g))
    assert float(g[net.param_index["k_struvite"]]) != 0.0


# --- speciation -> precipitation composition ---------------------------------

_COMPOSED_YAML = textwrap.dedent("""
network: {name: precip_speciation, version: "1.0", description: "x"}
species:
  - {name: S_Ca,  default_concentration: 3.0,  units: "mol/m3"}
  - {name: S_IC,  default_concentration: 50.0, units: "mol/m3"}
  - {name: S_cat, default_concentration: 60.0, units: "mol/m3"}
  - {name: X_calcite, default_concentration: 1.0e-3, units: "mol/m3"}
conditions:
  - {name: T, default: 298.15}
speciation:
  field: pH
  temperature_field: T
  temperature_units: kelvin
  totals:
    carbonate: {species: S_IC, molar_mass: 1000}
  strong_cations:
    - {species: S_cat, molar_mass: 1000, charge: 1}
precipitation:
  pH_field: pH
  temperature_field: T
  temperature_units: kelvin
  minerals:
    - name: calcite
      pKsp: 8.48
      order: 2
      ions:
        - {species: S_Ca, molar_mass: 1000, count: 1, charge: 2}
        - {species: S_IC, molar_mass: 1000, count: 1, charge: 2, fraction: carbonate}
parameters:
  k_calcite: {value: 10.0, units: "1/d"}
reactions:
  - name: calcite_precip
    rate: "k_calcite * [X_calcite] * {R_calcite}"
    stoichiometry: {S_Ca: -1.0, S_IC: -1.0, X_calcite: 1.0}
""")


def test_speciation_pH_feeds_precipitation():
    # The pH is NOT supplied as a condition -- it is solved from the charge
    # balance (speciation:) and then read by the precipitation saturation index
    # (the two derived functions compose). Calcite precipitates as the
    # solved pH makes the supernatant supersaturated.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(_COMPOSED_YAML)
        path = f.name
    try:
        net = aquakin.load_network_from_file(path)
        assert "pH" not in net.conditions_required          # pH is derived, not supplied
        assert "SI_calcite" in net.derived_fields and "pH" in net.derived_fields
        # The composed derived fn produces both pH and the saturation index.
        d = net.derived_condition_fn(jnp.asarray(net.default_concentrations()),
                                     net.default_parameters(),
                                     {"T": jnp.array([298.15])}, 0)
        assert np.isfinite(float(d["pH"])) and 5.0 < float(d["pH"]) < 11.0
        assert np.isfinite(float(d["SI_calcite"]))
        # Run it: calcite precipitates (supersaturated at the solved pH).
        r = aquakin.BatchReactor(net, aquakin.SpatialConditions.uniform(T=298.15))
        sol = r.solve(net.default_concentrations(), params=net.default_parameters(),
                      t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 6))
        assert float(sol.C_named("X_calcite")[-1]) > 1.0
        assert np.all(np.isfinite(sol.C))
    finally:
        os.unlink(path)


# --- schema validation -------------------------------------------------------

def _load_bad(block):
    head = textwrap.dedent("""
    network: {name: bad, version: "1.0", description: "x"}
    species: [{name: S_Ca, default_concentration: 1.0, units: "mol/m3"}]
    conditions: [{name: T, default: 298.15}, {name: pH, default: 8.0}]
    """)
    tail = textwrap.dedent("""
    parameters: {k: {value: 1.0}}
    reactions: [{name: r, rate: "k * [S_Ca]", stoichiometry: {S_Ca: -1.0}}]
    """)
    yaml = head + textwrap.dedent(block) + tail
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml)
        path = f.name
    try:
        aquakin.load_network_from_file(path)
    finally:
        os.unlink(path)


def test_schema_rejects_unknown_fraction():
    block = textwrap.dedent("""
    precipitation:
      minerals:
        - {name: m, pKsp: 8.0, order: 2, ions: [
            {species: S_Ca, molar_mass: 1000, count: 1, charge: 2, fraction: bogus}]}
    """)
    with pytest.raises(Exception, match="fraction"):
        _load_bad(block)


def test_schema_rejects_undeclared_species():
    block = textwrap.dedent("""
    precipitation:
      minerals:
        - {name: m, pKsp: 8.0, order: 2, ions: [
            {species: S_ghost, molar_mass: 1000, count: 1, charge: 2}]}
    """)
    with pytest.raises(Exception, match="ghost|undeclared"):
        _load_bad(block)
