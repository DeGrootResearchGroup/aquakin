"""Aeration strips the gases a model declares (``Aeration(stripped="model")``).

Checked against independent references:

- the transfer of each declared gas equals ``ratio * kLa * (H(T) y P - C_vol)``
  computed by hand from the library's published constants;
- a tank without biomass reaches the closed-form steady state of dilution
  against transfer;
- the mass balance counts the nitrogen the liquid takes up from the air, and
  at steady state that equals what leaves in the effluent;
- a reverse-mode gradient through a plant solve matches a central difference.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.core.ph_solver import equilibrium_constants
from aquakin.plant import Aeration, CSTRUnit, Plant
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.streams import Stream

CORE = Path(__file__).parents[2] / "examples" / "plantwide" / "models" / "plantwide_core.yaml"

# Henry solubility at 298.15 K (mol m-3 Pa-1), d ln H / d(1/T) (K) and
# diffusivity (m2/s) as published (Sander 2023; CRC Handbook), the molar mass
# that converts each species' units to mol/L, and its mole fraction in dry air.
PUBLISHED = {
    "SN2": (6.4e-6, 1300.0, 2.0e-9, 28000.0, 0.78084),
    "SCH4": (1.4e-5, 1600.0, 1.84e-9, 64000.0, 1.9e-6),
    "SH2": (7.8e-6, 530.0, 5.11e-9, 16000.0, 5.3e-7),
    "SIC": (3.4e-4, 2300.0, 1.91e-9, 12000.0, 4.2e-4),
}
D_O2 = 2.42e-9
T, PH, KLA = 288.15, 7.2, 100.0


@pytest.fixture(scope="module")
def core():
    return aquakin.load_model_from_file(CORE)


def _tank(core, **aeration):
    return CSTRUnit(
        name="tank",
        model=core,
        volume=1000.0,
        input_port_names=["inlet"],
        conditions={"T": T, "pH": PH},
        aeration=Aeration(kla=KLA, species="SO2", stripped="model", **aeration),
    )


def _saturation(species, y=None):
    henry, dlnh, _d, molar_mass, y_air = PUBLISHED[species]
    y = y_air if y is None else y
    return henry * math.exp(dlnh * (1.0 / T - 1.0 / 298.15)) * y * 101325.0 * molar_mass / 1e3


def _ratio(species):
    return math.sqrt(PUBLISHED[species][2] / D_O2)


def _volatile_fraction(species):
    """1, or for inorganic carbon its H2CO3* fraction at ``PH``."""
    if species != "SIC":
        return 1.0
    K = equilibrium_constants(T)
    h = 10.0**-PH
    k1, k2 = float(K["co3_1"]), float(K["co3_2"])
    return h * h / (h * h + k1 * h + k1 * k2)


def _hand_transfer(species, C, y=None):
    return _ratio(species) * KLA * (_saturation(species, y) - _volatile_fraction(species) * C)


@pytest.mark.parametrize("air", [None, {"CO2": 0.02}])
def test_transfer_of_each_declared_gas_equals_the_hand_calculation(core, air):
    """Every declared gas but oxygen transfers at its library ratio toward its
    share of the air; oxygen keeps ``do_sat``; nothing else transfers."""
    unit = _tank(core, air=air)
    values = {"SO2": 2.0, "SN2": 10.0, "SCH4": 0.5, "SH2": 0.01, "SIC": 50.0}
    C = jnp.array([values.get(s, 1.0) for s in core.species])
    inlet = Stream(Q=jnp.asarray(10.0), C=C, model=core)
    term = np.asarray(unit.aeration_term(C, {"inlet": inlet}, core.default_parameters(), None))

    idx = core.species_index
    for sp in PUBLISHED:
        share = (air or {}).get("CO2") if sp == "SIC" else None
        np.testing.assert_allclose(term[idx[sp]], _hand_transfer(sp, values[sp], share), rtol=1e-9)
    np.testing.assert_allclose(term[idx["SO2"]], KLA * (8.0 - 2.0), rtol=1e-12)
    others = [i for s, i in idx.items() if s not in values]
    np.testing.assert_array_equal(term[others], 0.0)
    assert set(unit.stripped_gases) == set(PUBLISHED)


def _single_tank_plant(core, feed, conditions, aeration):
    plant = Plant("aerated", time_unit="d")
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=core,
            volume=1000.0,
            input_port_names=["inlet"],
            conditions=conditions,
            aeration=aeration,
        )
    )
    C = jnp.array([feed.get(s, 0.0) for s in core.species])
    influent = InfluentSeries(
        t=jnp.asarray([0.0, 10.0]),
        Q=jnp.asarray([1000.0, 1000.0]),
        C=jnp.stack([C, C]),
        model=core,
    )
    plant.add_influent("feed", influent, to="tank.inlet")
    return plant


@pytest.fixture(scope="module")
def abiotic(core):
    """A tank with no biomass, fed water holding only inorganic carbon and
    aerated for long enough (40 transfer times) to reach its steady state."""
    plant = _single_tank_plant(
        core, {"SIC": 60.0}, {"T": T, "pH": PH}, Aeration(kla=KLA, species="SO2", stripped="model")
    )
    y0 = plant.initial_state(overrides={"tank": jnp.zeros(core.n_species)})
    # Fine sampling: the balance integrates the transfer by the trapezoid rule
    # and the start-up transient lasts about 1/80 d.
    t = np.linspace(0.0, 2.0, 2001)
    return plant, plant.solve(t_span=(0.0, 2.0), y0=y0, t_eval=t)


def test_tank_without_biomass_reaches_the_closed_form_steady_state(core, abiotic):
    """Dilution against transfer: ``C = (D C_in + k sat) / (D + k f)`` with
    ``D = Q/V``, ``k`` the gas's kLa and ``f`` its volatile fraction."""
    plant, sol = abiotic
    final = {s: float(sol.state[-1][core.species_index[s]]) for s in core.species}
    D = 1.0
    for sp in PUBLISHED:
        k = _ratio(sp) * KLA
        C_in = 60.0 if sp == "SIC" else 0.0
        expected = (D * C_in + k * _saturation(sp)) / (D + k * _volatile_fraction(sp))
        np.testing.assert_allclose(final[sp], expected, rtol=1e-5, err_msg=sp)
    np.testing.assert_allclose(final["SO2"], KLA * 8.0 / (D + KLA), rtol=1e-5)


def test_mass_balance_counts_the_nitrogen_taken_up_from_the_air(core, abiotic):
    """The liquid takes up N2 from the air, which the balance reports as
    negative stripped N; with it counted, N closes. The fed water holds no
    nitrogen, so all the N leaving or held came in from the air.
    """
    plant, sol = abiotic
    mb = plant.mass_balance(sol)
    uptake = -mb.gas_detail["stripped_N"]
    assert mb["N"].inflow == 0.0
    assert uptake > 0.0
    # The balance integrates the transfer by the trapezoid rule, whose error on
    # the start-up transient (time constant ~0.01 d, sampled every 0.001 d) is
    # 2e-4 of the uptake; it falls as the step squared (1e-5 at a quarter of it).
    assert uptake == pytest.approx(mb["N"].outflow + mb["N"].accumulation, rel=1e-3)
    assert mb.closed(rtol=1e-3), mb.summary()


def test_gradient_through_a_solve_matches_a_central_difference(core):
    """d(final inorganic carbon)/d(muH) on an aerated, seeded tank, where the
    heterotrophs make the CO2 the aeration strips.

    About 30 s, mostly compiling the reverse-mode solve of the 19-state model;
    kept in the fast tier because it is this feature's only gradient check
    through a solve.
    """
    feed = {"SA": 100.0, "SNH4": 30.0, "SIC": 60.0, "SN2": 15.0}
    plant = _single_tank_plant(
        core, feed, {"T": T, "pH": PH}, Aeration(kla=KLA, species="SO2", stripped="model")
    )
    seeds = dict.fromkeys(core.species, 0.0)
    seeds.update({"XH": 2000.0, "SA": 50.0, "SIC": 60.0, "SO2": 2.0})
    y0 = plant.initial_state(overrides={"tank": jnp.array([seeds[s] for s in core.species])})
    k = plant.parameter_index(f"{core.name}.muH")
    sic = plant._state_layout["tank"][0] + core.species_index["SIC"]
    p0 = plant.parameter_values()

    def final_sic(p):
        # A tight solver tolerance, so the difference quotient is not solver noise.
        sol = plant.solve(
            t_span=(0.0, 0.05), y0=y0, t_eval=jnp.asarray([0.05]), params=p, rtol=1e-10
        )
        return sol.state[-1][sic]

    grad = float(jax.grad(final_sic)(p0)[k])
    h = 1e-4 * float(p0[k])
    fd = (float(final_sic(p0.at[k].add(h))) - float(final_sic(p0.at[k].add(-h)))) / (2 * h)
    assert abs(fd) > 1e-3
    np.testing.assert_allclose(grad, fd, rtol=1e-4)


def test_errors_name_the_problem(core):
    with pytest.raises(ValueError, match="stripped must be 'model'"):
        Aeration(kla=1.0, stripped="all")
    with pytest.raises(ValueError, match="it needs stripped='model'"):
        Aeration(kla=1.0, air={"CO2": 0.02})
    with pytest.raises(ValueError, match="unknown gas 'Ar'"):
        Aeration(kla=1.0, stripped="model", air={"Ar": 0.01})
    with pytest.raises(ValueError, match="fraction of 'CO2' must be in"):
        Aeration(kla=1.0, stripped="model", air={"CO2": 1.5})
    asm1 = aquakin.load_model("asm1")
    with pytest.raises(ValueError, match="declares no dissolved gases"):
        CSTRUnit(
            name="t",
            model=asm1,
            volume=1.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
            aeration=Aeration(kla=1.0, stripped="model", species="SO"),
        )
    unit = _tank(core)
    C = core.default_concentrations()
    with pytest.raises(ValueError, match="needs the tank's conditions"):
        from aquakin.plant.cstr import aeration_transfer

        aeration_transfer(unit._av, C, None, None, core)


def test_settings_round_trip_through_the_catalogue(core):
    """``stripped="model"`` and ``air`` are written as plain settings and build
    the same aeration back."""
    from aquakin.plant.catalog import build_unit, unit_settings

    unit = _tank(core, air={"CO2": 0.02})
    kind, settings = unit_settings(unit, model_ref=lambda m: m.name)
    assert settings["aeration"]["stripped"] == "model"
    assert settings["aeration"]["air"] == [["CO2", 0.02]]
    rebuilt = build_unit(kind, "tank", settings, models={core.name: core})
    assert rebuilt.aeration == unit.aeration


def test_hand_listed_stripped_gases_round_trip_through_the_catalogue():
    """The ``{species: ratio}`` form is written as (species, ratio, sat)
    records rather than dropped."""
    from aquakin.plant.catalog import build_unit, unit_settings

    n2o = aquakin.load_model("asm3_2step_n2o")
    unit = CSTRUnit(
        name="tank",
        model=n2o,
        volume=1000.0,
        input_port_names=["inlet"],
        conditions={"T": 293.15, "pH": 7.0},
        aeration=Aeration(kla=100.0, species="SO2", stripped={"SN2O": (0.9, 0.001)}),
    )
    kind, settings = unit_settings(unit)
    assert settings["aeration"]["stripped"] == [["SN2O", 0.9, 0.001]]
    assert build_unit(kind, "tank", settings).aeration == unit.aeration
