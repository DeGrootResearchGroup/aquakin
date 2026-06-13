"""Integration test: state-derived pH via a speciation block.

Exercises the full path YAML speciation -> compiled derived-condition fn ->
``{pH}`` / ``pH_switch`` rate expressions -> Diffrax solve, including AD.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import aquakin
from aquakin import BatchReactor, SpatialConditions, compile_network
from aquakin.core.ph_solver import solve_ph
from aquakin.schema.network_spec import NetworkSpec

# A minimal sewer-like network: a carbonate/ammonia/sulfide pool sets pH, and a
# tracer X is removed by a pH-gated (pH_switch) process. pH is never supplied
# externally -- it is derived from the state every RHS step.
YAML = """
network:
  name: ph_toy
  version: "1.0"
  description: "Toy network exercising state-derived pH."

species:
  - {name: S_CO2, units: mgC/L, default_concentration: 24.0}
  - {name: S_NH,  units: mgN/L, default_concentration: 28.0}
  - {name: sumS,  units: mgS/L, default_concentration: 3.2}
  - {name: S_SO4, units: mgS/L, default_concentration: 16.0}
  - {name: X,     units: mg/L,  default_concentration: 100.0}

conditions:
  - {name: T, description: "Temperature (degC)", default: 20.0}

speciation:
  field: pH
  temperature_field: T
  temperature_units: celsius
  z_cation_eq: 3.28e-3
  n_iter: 40
  totals:
    carbonate: {species: S_CO2, molar_mass: 12000}
    ammonia:   {species: S_NH,  molar_mass: 14000}
    sulfide:   {species: sumS,  molar_mass: 32000}
  strong_anions:
    - {species: S_SO4, molar_mass: 32000, charge: 2}

reactions:
  - name: X_removal
    description: "pH-gated first-order removal of X."
    rate: "k * [X] * (1 - pH_switch(pKa))"
    parameters:
      k:   {value: 0.5}
      pKa: {value: 7.0}
    stoichiometry:
      X: -1
"""


def _build():
    spec = NetworkSpec.model_validate(yaml.safe_load(YAML))
    return compile_network(spec)


def test_pH_is_not_a_required_condition():
    net = _build()
    # pH is produced, not supplied; only T is required from the user.
    assert net.conditions_required == ["T"]
    assert net.derived_fields == ["pH"]


def test_derived_pH_matches_solver():
    net = _build()
    C0 = net.default_concentrations()
    cond = SpatialConditions.uniform(1, T=20.0)
    derived = net.derived_condition_fn(C0, net.default_parameters(), cond.fields, 0)
    pH = float(derived["pH"])

    expected = float(
        solve_ph(
            tot_carbonate=24.0 / 12000,
            tot_ammonia=28.0 / 14000,
            tot_sulfide=3.2 / 32000,
            strong_anion_eq=2 * 16.0 / 32000,
            z_cation_eq=3.28e-3,
            T_kelvin=293.15,
        )
    )
    assert pH == pytest.approx(expected, abs=1e-9)
    assert 4.0 < pH < 11.0  # physically sane


def test_solve_runs_and_removes_X():
    net = _build()
    reactor = BatchReactor(net, SpatialConditions.uniform(1, T=20.0))
    C0 = net.default_concentrations()
    sol = reactor.solve(C0, params=net.default_parameters(), t_span=(0.0, 5.0))
    X_final = float(sol.C_named("X")[-1])
    assert 0.0 < X_final < 100.0  # some removal, but not unphysical


@pytest.mark.slow  # heavy: jax.grad through state-derived pH solve
def test_grad_flows_through_pH():
    net = _build()
    reactor = BatchReactor(net, SpatialConditions.uniform(1, T=20.0))
    C0 = net.default_concentrations()
    params = net.default_parameters()

    def final_X(C0):
        sol = reactor.solve(C0, params=params, t_span=(0.0, 5.0))
        return sol.C_named("X")[-1]

    # Sensitivity of final X to the initial carbonate (which moves pH and thus
    # the pH-gated removal rate) must be finite and non-zero.
    g = jax.grad(final_X)(C0)
    assert np.all(np.isfinite(np.asarray(g)))
    assert abs(float(g[net.species_index["S_CO2"]])) > 0.0


# A network whose net cation charge comes from a strong-cation STATE (S_cat),
# in addition to the strong-anion state (S_an) -- the ADM1 explicit-ion pattern.
YAML_IONS = """
network:
  name: ph_ions_toy
  version: "1.0"
  description: "Toy network with explicit strong-ion states driving pH."

species:
  - {name: S_CO2, units: mgC/L, default_concentration: 24.0}
  - {name: S_cat, units: mol/L, default_concentration: 3.0e-3}
  - {name: S_an,  units: mol/L, default_concentration: 1.0e-3}

conditions:
  - {name: T, description: "Temperature (degC)", default: 20.0}

speciation:
  field: pH
  temperature_field: T
  temperature_units: celsius
  n_iter: 40
  totals:
    carbonate: {species: S_CO2, molar_mass: 12000}
  strong_cations:
    - {species: S_cat, molar_mass: 1.0, charge: 1}
  strong_anions:
    - {species: S_an,  molar_mass: 1.0, charge: 1}

reactions:
  - name: co2_decay
    description: "First-order CO2 removal; the ion states stay conservative."
    rate: "k * [S_CO2]"
    parameters:
      k: {value: 0.1}
    stoichiometry:
      S_CO2: -1
"""


def test_strong_cation_state_drives_pH():
    """A strong-cation STATE feeds the net cation charge, equivalent to a
    z_cation_eq offset of (S_cat - S_an); raising S_cat raises pH."""
    spec = NetworkSpec.model_validate(yaml.safe_load(YAML_IONS))
    net = compile_network(spec)
    si = net.species_index
    C0 = net.default_concentrations()
    p = net.default_parameters()
    cond = SpatialConditions.uniform(1, T=20.0)

    pH = float(net.derived_condition_fn(C0, p, cond.fields, 0)["pH"])
    # The S_cat (+3 mM) / S_an (-1 mM) split is a net +2 mM cation charge.
    expected = float(
        solve_ph(tot_carbonate=24.0 / 12000, z_cation_eq=2.0e-3, T_kelvin=293.15)
    )
    assert pH == pytest.approx(expected, abs=1e-9)

    # More strong cation -> higher pH; differentiable through the solver.
    C_more = C0.at[si["S_cat"]].add(2.0e-3)
    pH_more = float(net.derived_condition_fn(C_more, p, cond.fields, 0)["pH"])
    assert pH_more > pH
    g = jax.grad(
        lambda scat: net.derived_condition_fn(
            C0.at[si["S_cat"]].set(scat), p, cond.fields, 0
        )["pH"]
    )(3.0e-3)
    assert np.isfinite(float(g)) and float(g) > 0.0
