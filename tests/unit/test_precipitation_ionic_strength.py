"""Ionic strength in the precipitation and speciation blocks.

Pins four behaviours:

* the kinetic path counts each dissolved ion once in the ionic strength, even
  when several minerals share it;
* the equilibrium path reads ``ionic_strength_field`` when one is set, and
  otherwise adds the mineral ions to the background electrolyte;
* ``CompiledModel.precipitation_equilibrium()`` works when the pH comes from a
  ``speciation:`` block rather than a declared condition; and
* a speciation block without an activity model still exports the real ionic
  strength.
"""

import textwrap
from importlib.resources import files

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import aquakin
from aquakin.core.ph_solver import solve_ph
from aquakin.core.precipitation import (
    build_precipitation_derived_fn,
    ionic_strength_ions,
)
from aquakin.core.precipitation_equilibrium import (
    build_precipitation_equilibrium_derived_fn,
)
from aquakin.core.speciation import build_ph_derived_fn


def _precip_block(name):
    text = (files("aquakin.models") / f"{name}.yaml").read_text()
    return yaml.safe_load(text)["precipitation"]


# --- kinetic path: each ion counted once --------------------------------------


def test_shared_ions_counted_once_in_kinetic_ionic_strength():
    # Fe3+ and PO4 appear in several minerals; the FePO4 saturation index must
    # not depend on how many other minerals share those ions.
    net = aquakin.load_model("precipitation_metal_phosphate")
    C = net.default_concentrations()
    cf = net.default_conditions().fields
    pc = _precip_block("precipitation_metal_phosphate")
    full, _, _ = build_precipitation_derived_fn(pc, net.species_index)
    alone, _, _ = build_precipitation_derived_fn(
        {**pc, "minerals": [pc["minerals"][0]]}, net.species_index
    )
    si_full = float(full(C, None, cf, 0)["SI_FePO4"])
    si_alone = float(alone(C, None, cf, 0)["SI_FePO4"])
    assert si_full == pytest.approx(si_alone, rel=1e-12)


def test_ionic_strength_ions_are_unique():
    net = aquakin.load_model("precipitation_metal_phosphate")
    pc = _precip_block("precipitation_metal_phosphate")
    ions = ionic_strength_ions(pc["minerals"], net.species_index)
    keys = [(idx, frac) for idx, _mm, _z2, frac in ions]
    assert len(keys) == len(set(keys))
    # Every state-backed ion of the block is represented.
    declared = {
        (net.species_index[ion["species"]], ion.get("fraction"))
        for m in pc["minerals"]
        for ion in m["ions"]
        if ion.get("fraction") not in ("proton", "hydroxide")
    }
    assert set(keys) == declared


# --- equilibrium path: ionic_strength_field and the mineral ions --------------

_EQ = "precipitation_metal_phosphate_equilibrium"


def _eq_conditions(pH=6.0, **extra):
    fields = {"T": jnp.array([293.15]), "pH": jnp.array([pH])}
    fields.update({k: jnp.array([v]) for k, v in extra.items()})
    return fields


def test_equilibrium_reads_ionic_strength_field():
    net = aquakin.load_model(_EQ)
    pc = {**_precip_block(_EQ), "ionic_strength_field": "I_aq"}
    fn, _, required, _ = build_precipitation_equilibrium_derived_fn(pc, net.species_index)
    assert "I_aq" in required
    C = net.default_concentrations()
    low = float(fn(C, None, _eq_conditions(I_aq=0.001), 0)["Xeq_FePO4"])
    high = float(fn(C, None, _eq_conditions(I_aq=0.5), 0)["Xeq_FePO4"])
    # Stronger electrolyte -> smaller activity coefficients -> less precipitate.
    assert high < low


def test_equilibrium_ionic_strength_field_not_required_without_activity():
    net = aquakin.load_model(_EQ)
    pc = {**_precip_block(_EQ), "ionic_strength_field": "I_aq", "activity_model": "none"}
    _, _, required, _ = build_precipitation_equilibrium_derived_fn(pc, net.species_index)
    assert "I_aq" not in required


def test_equilibrium_adds_mineral_ions_to_background():
    # Without a field the dissolved mineral ions add to ionic_strength_offset, so
    # the result differs from reading just the offset through a field. At pH 2.2
    # with no phosphate, a millimolar Fe3+ stays dissolved over Fe(OH)3 and
    # raises the ionic strength.
    net = aquakin.load_model(_EQ)
    si = net.species_index
    pc = _precip_block(_EQ)
    own, _, _, _ = build_precipitation_equilibrium_derived_fn(pc, si)
    offset_only, _, _, _ = build_precipitation_equilibrium_derived_fn(
        {**pc, "ionic_strength_field": "I_aq"}, si
    )
    C = net.default_concentrations().at[si["S_Fe3"]].set(5.0).at[si["S_PO4"]].set(0.0)
    I0 = pc["ionic_strength_offset"]
    x_own = float(own(C, None, _eq_conditions(pH=2.2), 0)["Xeq_FeOH3"])
    x_off = float(offset_only(C, None, _eq_conditions(pH=2.2, I_aq=I0), 0)["Xeq_FeOH3"])
    assert np.isfinite(x_own)
    # More ions -> smaller activity coefficients -> less Fe(OH)3.
    assert x_off - x_own > 1e-6 * x_off


def test_equilibrium_counts_kinetic_only_ions_at_the_current_state():
    # With AlPO4 kinetic, S_Al is not conserved by the equilibrium solve, but it
    # still adds to the ionic strength, at its current concentration: more
    # dissolved Al3+ -> smaller activity coefficients -> less FePO4.
    net = aquakin.load_model(_EQ)
    si = net.species_index
    pc = _precip_block(_EQ)
    minerals = [
        {**m, "mode": "kinetic"} if m["name"] == "AlPO4" else m
        for m in pc["minerals"]
        if m["name"] != "AlOH3"
    ]
    fn, _, _, _ = build_precipitation_equilibrium_derived_fn({**pc, "minerals": minerals}, si)
    C = net.default_concentrations()
    no_al = float(fn(C.at[si["S_Al"]].set(0.0), None, _eq_conditions(), 0)["Xeq_FePO4"])
    al = float(fn(C.at[si["S_Al"]].set(20.0), None, _eq_conditions(), 0)["Xeq_FePO4"])
    assert np.isfinite(no_al) and np.isfinite(al)
    assert al < no_al


# --- precipitation_equilibrium() with a speciation-derived pH ------------------

_SPEC_EQ_YAML = """
model: {name: spec_eq}
species:
  - {name: S_Fe3, default_concentration: 2.0, units: "mol/m3"}
  - {name: S_PO4, default_concentration: 2.0, units: "mol/m3"}
  - {name: S_IC, default_concentration: 5.0, units: "mol/m3"}
  - {name: X_FePO4, default_concentration: 1.0e-4, units: "mol/m3"}
conditions:
  - {name: T, default: 293.15}
speciation:
  temperature_field: T
  temperature_units: kelvin
  z_cation_eq: 0.005
  totals:
    carbonate: {species: S_IC, molar_mass: 1000}
precipitation:
  pH_field: pH
  temperature_field: T
  temperature_units: kelvin
  minerals:
    - name: FePO4
      pKsp: 26.4
      mode: equilibrium
      solid: X_FePO4
      ions:
        - {species: S_Fe3, molar_mass: 1000, count: 1, charge: 3}
        - {species: S_PO4, molar_mass: 1000, count: 1, charge: 3, fraction: phosphate}
reactions:
  - name: relax
    rate: "k * ({Xeq_FePO4} - [X_FePO4])"
    parameters: {k: {value: 10.0}}
    stoichiometry: {X_FePO4: 1, S_Fe3: -1, S_PO4: -1}
"""


def test_precipitation_equilibrium_with_speciation_ph(tmp_path):
    path = tmp_path / "spec_eq.yaml"
    path.write_text(textwrap.dedent(_SPEC_EQ_YAML))
    net = aquakin.load_model_from_file(str(path))
    assert "pH" not in net.default_conditions().fields  # pH is derived, not declared
    C0 = net.default_concentrations()
    Ceq = net.precipitation_equilibrium()
    si = net.species_index
    assert np.all(np.isfinite(np.asarray(Ceq)))
    # Mass is conserved and FePO4 precipitates.
    for ion in ("S_Fe3", "S_PO4"):
        assert float(Ceq[si[ion]] + Ceq[si["X_FePO4"]]) == pytest.approx(
            float(C0[si[ion]] + C0[si["X_FePO4"]]), rel=1e-6
        )
    assert float(Ceq[si["X_FePO4"]]) > 1.0


# --- speciation without an activity model exports the real ionic strength ------


def test_speciation_without_activity_exports_ionic_strength():
    # 50 mM NaHCO3: I ~ 1/2 (Na+ + HCO3- + 4 CO3--) ~ 0.05 mol/L, not 0.
    cfg = {
        "field": "pH",
        "temperature_field": "T",
        "temperature_units": "kelvin",
        "activity_model": "none",
        "ionic_strength_field": "I_aq",
        "totals": {"carbonate": {"species": "A", "molar_mass": 1.0}},
        "strong_cations": [{"species": "B", "molar_mass": 1.0, "charge": 1.0}],
    }
    fn, produced, _ = build_ph_derived_fn(cfg, {"A": 0, "B": 1})
    assert produced == ["pH", "I_aq"]
    out = fn(jnp.array([0.05, 0.05]), None, {"T": jnp.array([298.15])}, 0)
    assert float(out["I_aq"]) == pytest.approx(0.0505, rel=0.01)
    # The pH itself is unchanged by exporting I.
    ph_only, _, _ = build_ph_derived_fn(
        {k: v for k, v in cfg.items() if k != "ionic_strength_field"}, {"A": 0, "B": 1}
    )
    ph = ph_only(jnp.array([0.05, 0.05]), None, {"T": jnp.array([298.15])}, 0)["pH"]
    assert float(out["pH"]) == float(ph)


def test_solve_ph_without_activity_returns_solution_ionic_strength():
    pH, I = solve_ph(tot_carbonate=0.05, z_cation_eq=0.05, return_ionic_strength=True)
    assert float(pH) == float(solve_ph(tot_carbonate=0.05, z_cation_eq=0.05))
    # Weak-acid ions + water contribute even with no strong-ion term supplied.
    assert float(I) == pytest.approx(0.025, rel=0.02)
