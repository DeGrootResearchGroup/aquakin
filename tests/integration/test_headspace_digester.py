"""The digester whose gas headspace belongs to the unit (HeadspaceDigesterUnit).

Checked against independent references:

- the shipped ``adm1`` model's own BSM2 headspace: with its headspace states and
  gas reactions removed and BSM2's Henry constants declared through ``gases:``,
  the unit's gas transfer and biogas outflow equal ADM1's at the same state;
- conservation: a continuously fed digester on the plant-wide core model, solved
  to steady state, closes its COD and N balances with the vented biogas counted
  as gas;
- the structural Jacobian pattern covers the dense Jacobian;
- a reverse-mode gradient through a plant solve matches a central difference.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import Plant
from aquakin.plant.headspace import HeadspaceDigesterUnit
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.streams import Stream

CORE = Path(__file__).parents[2] / "examples" / "plantwide" / "models" / "plantwide_core.yaml"
GAS_REACTIONS = (
    "transfer_h2",
    "transfer_ch4",
    "transfer_co2",
    "gas_outflow_h2",
    "gas_outflow_ch4",
    "gas_outflow_co2",
)
# BSM2's Henry temperature dependence: van't Hoff enthalpies divided by its R.
_R_BSM2 = 8.3145


@pytest.fixture(scope="module")
def core():
    return aquakin.load_model_from_file(CORE)


@pytest.fixture(scope="module")
def adm1_without_headspace(tmp_path_factory):
    """ADM1 with its headspace removed and BSM2's gas constants declared."""
    path = tmp_path_factory.mktemp("adm1") / "adm1_no_headspace.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            model: {{name: adm1_no_headspace, extends: adm1}}
            remove:
              species: [S_gas_h2, S_gas_ch4, S_gas_co2]
              reactions: {list(GAS_REACTIONS)}
              expressions: [p_gas_h2, p_gas_ch4, p_gas_co2, P_gas, q_gas_norm, S_co2_diss]
            gases:
              temperature_field: T
              pH_field: pH
              species:
                S_h2: {{gas: H2, molar_mass: 16, henry: 7.8e-6,
                        henry_temperature: {4180 / _R_BSM2}}}
                S_ch4: {{gas: CH4, molar_mass: 64, henry: 1.4e-5,
                         henry_temperature: {14240 / _R_BSM2}}}
                S_IC: {{gas: CO2, molar_mass: 1, henry: 3.5e-4,
                        henry_temperature: {19410 / _R_BSM2}, volatile_fraction: carbonate}}
            """
        )
    )
    return aquakin.load_model_from_file(path)


def test_gas_transfer_and_outflow_equal_adm1s_bsm2_headspace(adm1_without_headspace):
    """At ADM1's reference state the unit's transfer and outflow are ADM1's.

    The two differ only in the volatile carbonate fraction (ADM1 ignores the
    second dissociation and rounds pK1: 9e-4 relative at this state) and in
    the gas constant's last digits. A transfer is kLa times a small difference
    of two near-equal terms (for CO2 here, 1% of either), so its tolerance is
    1e-3 of the terms, kLa * C, not of the difference.
    """
    from aquakin.core.gases import volatile_concentration
    adm1 = aquakin.load_model("adm1")
    unit = HeadspaceDigesterUnit(
        name="d", model=adm1_without_headspace, volume=3400.0, conditions={"T": 308.15}
    )
    full = adm1.default_concentrations()
    n = adm1_without_headspace.n_species
    C = full[:n]
    G = jnp.asarray([full[adm1.species_index[s]] for s in ("S_gas_h2", "S_gas_ch4", "S_gas_co2")])

    rates = adm1.rates(full, adm1.default_parameters(), adm1.default_conditions().fields, 0)
    adm1_rate = {name: float(rates[adm1.reaction_names.index(name)]) for name in GAS_REACTIONS}

    params = adm1_without_headspace.default_parameters()
    transfer = np.asarray(unit._transfer(C, G, params))
    outflow = float(unit.gas_flow(jnp.concatenate([C, G]))) / unit.gas_volume * np.asarray(G)
    pH = float(adm1.derived_condition_fn(full, adm1.default_parameters(), {"T": [308.15]}, 0)["pH"])
    for k, gas in enumerate(("h2", "ch4", "co2")):
        entry = unit._gases[k]
        terms = unit.kla * float(volatile_concentration(entry, C[entry.index], 308.15, pH))
        np.testing.assert_allclose(
            transfer[k], adm1_rate[f"transfer_{gas}"], rtol=0, atol=1e-3 * terms
        )
        np.testing.assert_allclose(outflow[k], adm1_rate[f"gas_outflow_{gas}"], rtol=1e-4)


FEED = dict(
    SF=300.0, SA=300.0, SI=100.0, SNH4=40.0, SIC=60.0, XI=14000.0, XS=22000.0,
    XH=9000.0, XAUT=500.0, XFE=1.0, XPRO=1.0, XAC=1.0, XH2M=1.0,
)  # fmt: skip
SEED = {**FEED, "XFE": 500.0, "XPRO": 100.0, "XAC": 500.0, "XH2M": 200.0}


def _vector(model, values):
    return jnp.array([values.get(s, 0.0) for s in model.species])


def _digester_plant(core, time_unit="d"):
    plant = Plant("digester", time_unit=time_unit)
    unit = HeadspaceDigesterUnit(
        name="digester",
        model=core,
        volume=2000.0,
        gas_volume=200.0,
        conditions={"T": 308.15, "pH": 7.1},
    )
    plant.add_unit(unit)
    feed = _vector(core, FEED)
    plant.add_influent(
        "sludge",
        InfluentSeries(
            t=jnp.asarray([0.0, 2000.0]),
            Q=jnp.asarray([100.0, 100.0]),
            C=jnp.stack([feed, feed]),
            model=core,
        ),
        to="digester.inlet",
    )
    y0 = plant.initial_state(
        overrides={"digester": jnp.concatenate([_vector(core, SEED), jnp.zeros(5)])}
    )
    return plant, unit, y0


@pytest.fixture(scope="module")
def steady_digester(core):
    plant, unit, y0 = _digester_plant(core)
    sol = plant.solve(t_span=(0.0, 800.0), t_eval=jnp.linspace(700.0, 800.0, 11), y0=y0)
    return plant, unit, sol


def test_steady_digester_closes_cod_and_n_with_the_biogas(steady_digester):
    """Feed in = effluent out + biogas vented (+ the small change held), for COD
    and N, with the headspace export counted as gas."""
    plant, _, sol = steady_digester
    mb = plant.mass_balance(sol)
    assert mb.closed(rtol=1e-6)
    assert mb.gas_detail["gas_export_COD"] == pytest.approx(mb["COD"].gas, rel=1e-12)
    assert mb["COD"].gas > 0.4 * mb["COD"].inflow


def test_steady_biogas_is_methane_and_carbon_dioxide(core, steady_digester):
    """The headspace holds methane and CO2, methane the larger share; the
    dissolved methane is supersaturated relative to it, as at BSM2's reference
    state (ADM1's defaults: 0.0555 dissolved against 0.0492 at saturation); and
    at steady state each gas crosses into the headspace as fast as it leaves
    with the biogas."""
    from aquakin.core.gases import partial_pressure, saturation

    _, unit, sol = steady_digester
    state = np.asarray(sol.unit_state("digester"))[-1]
    n = core.n_species
    T = 308.15
    p = {g.gas: float(partial_pressure(g, T, state[n + k])) for k, g in enumerate(unit._gases)}
    dry = sum(p.values())
    assert 0.55 < p["CH4"] / dry < 0.75
    assert p["CH4"] + p["CO2"] == pytest.approx(dry, rel=1e-3)
    ch4 = core.gases.by_gas("CH4")
    assert state[ch4.index] > float(saturation(ch4, T, p["CH4"]))
    C, G = jnp.asarray(state[:n]), jnp.asarray(state[n:])
    transferred = np.asarray(unit._transfer(C, G, core.default_parameters())) * unit.volume
    vented = float(unit.gas_flow(jnp.asarray(state))) * np.asarray(G)
    np.testing.assert_allclose(transferred[2:], vented[2:], rtol=1e-4)  # CH4, H2, CO2


def test_digester_gas_reports_the_headspace_in_the_bsm2_convention(core, steady_digester):
    """``plant.digester_gas`` reads the headspace digester. Checked by hand from
    the saved headspace states: each partial pressure is the ideal-gas
    ``G / M * R T`` in bar; the flow is the vented ``k_p (P - p_atm)``
    normalized to the vent pressure by ``P / p_atm``, with ``P`` the gases plus
    BSM2's water vapour; and the methane mass flow is the methane COD the
    biogas carries out (``gas_export``) at 4 g COD per g CH4."""
    plant, unit, sol = steady_digester
    gas = plant.digester_gas(sol)
    states = np.asarray(sol.unit_state("digester"))
    n, T, R = core.n_species, 308.15, 8.314462618
    bar = {
        g.gas: states[:, n + k] * 1000.0 / g.molar_mass * R * T / 1e5
        for k, g in enumerate(unit._gases)
    }
    P = sum(bar.values()) + 0.0313 * np.exp(5290.0 * (1.0 / 298.15 - 1.0 / T))
    np.testing.assert_array_equal(gas.t, sol.t)
    for name, got in (("CH4", gas.p_ch4), ("CO2", gas.p_co2), ("H2", gas.p_h2)):
        np.testing.assert_allclose(got, bar[name], rtol=1e-12, err_msg=name)
    np.testing.assert_allclose(gas.Q, unit.k_p * (P - unit.p_atm) * P / unit.p_atm, rtol=1e-12)
    params = core.default_parameters()
    export = np.stack([np.asarray(unit.gas_export(jnp.asarray(s), params)) for s in states])
    ch4_cod = export[:, core.species_index["SCH4"]]  # g COD per day
    np.testing.assert_allclose(gas.ch4, ch4_cod / 4.0 / 1e3, rtol=1e-12)
    assert gas.methane_production() > 0.0


def test_digester_gas_is_per_day_on_a_plant_in_hours(core, steady_digester):
    """On a plant in hours the headspace vents m3 per hour; ``digester_gas``
    reports it per day. Started from the steady digester, so gas is venting."""
    _, _, steady = steady_digester
    plant, unit, _ = _digester_plant(core, time_unit="h")
    y0 = jnp.asarray(steady.unit_state("digester"))[-1]
    sol = plant.solve(t_span=(0.0, 1.0), t_eval=jnp.asarray([0.0, 1.0]), y0=y0)
    gas = plant.digester_gas(sol)
    vented = unit.biogas(jnp.asarray(sol.unit_state("digester"))[-1])
    assert float(vented.flow) > 0.0
    per_hour = float(vented.flow * vented.pressure / vented.vent_pressure)
    assert float(gas.Q[-1]) == pytest.approx(24.0 * per_hour, rel=1e-12)


def test_coupling_pattern_covers_the_dense_jacobian(core):
    unit = HeadspaceDigesterUnit(
        name="d", model=core, volume=2000.0, conditions={"T": 308.15, "pH": 7.1}
    )
    cp = unit.coupling_pattern()
    size = unit.state_size
    assert cp.self_pattern.shape == (size, size)
    inlet = {"inlet": Stream(Q=jnp.asarray(100.0), C=_vector(core, FEED), model=core)}
    params = core.default_parameters()
    jac = jax.jit(jax.jacfwd(lambda y: unit.rhs(jnp.asarray(0.0), y, inlet, params)))
    rng = np.random.default_rng(3)
    base = np.concatenate([np.asarray(_vector(core, SEED)) + 1.0, np.full(5, 50.0)])
    for _ in range(4):
        J = np.asarray(jac(jnp.asarray(base * 10.0 ** rng.uniform(-1, 1, size=size))))
        active = np.abs(J) > 1e-9 * np.abs(J).max()
        assert not (active & ~cp.self_pattern).any()


def test_a_gas_without_a_ph_dependence_transfers_against_its_henry_saturation(tmp_path):
    """A model whose gases need no pH: the transfer is kLa (C - H(T) p M/1000)."""
    from aquakin.core.gases import henry

    path = tmp_path / "ch4.yaml"
    path.write_text(
        textwrap.dedent(
            """
            model: {name: methane_only}
            species: [{name: SCH4, units: g_COD/m3}]
            conditions: [{name: T, default: 308.15}]
            parameters: {k: {value: 0.0, units: 1/d}}
            gases: {species: {SCH4: {gas: CH4, molar_mass: 64000}}}
            reactions: [{name: r, rate: "k * [SCH4]", stoichiometry: {SCH4: -1}}]
            """
        )
    )
    model = aquakin.load_model_from_file(path)
    unit = HeadspaceDigesterUnit(name="d", model=model, volume=10.0)
    T, C, G = 308.15, jnp.asarray([50.0]), jnp.asarray([400.0])  # g COD per m3
    p = 400.0 * 1000.0 / 64000.0 * 8.314462618 * T  # Pa, ideal gas
    expected = unit.kla * (50.0 - float(henry(model.gases.entries[0], T)) * p * 64.0)
    transfer = unit._transfer(C, G, model.default_parameters())
    assert float(transfer[0]) == pytest.approx(expected, rel=1e-12)


def test_layout_and_construction_errors(core):
    unit = HeadspaceDigesterUnit(name="d", model=core, volume=10.0, conditions={"T": 308.15})
    assert unit.state_names()[-1] == "gas.SIC"
    assert unit.state_size == core.n_species + 5
    assert float(jnp.sum(unit.initial_state()[core.n_species :])) == 0.0
    with pytest.raises(ValueError, match="gas_volume must be a positive number"):
        HeadspaceDigesterUnit(name="d", model=core, volume=10.0, gas_volume=0.0)
    with pytest.raises(ValueError, match="declares no dissolved gases"):
        HeadspaceDigesterUnit(name="d", model=aquakin.load_model("asm1"), volume=10.0)


def test_gradient_of_biogas_matches_a_central_difference(core):
    """d(biogas flow after one day)/d(k_m_ac) by reverse mode equals a central
    difference with a relative step of 1e-3. About 20 s, nearly all compiling
    the reverse-mode plant solve."""
    plant, unit, y0 = _digester_plant(core)
    k = plant.parameter_index(f"{core.name}.k_m_ac")
    n = core.n_species

    def biogas(params):
        sol = plant.solve(t_span=(0.0, 1.0), t_eval=jnp.asarray([1.0]), params=params, y0=y0)
        return unit.gas_flow(jnp.asarray(sol.state)[-1][-(n + 5) :])

    p = plant.default_parameters()
    grad = jax.grad(biogas)(p)[k]
    h = 1e-3 * p[k]
    fd = (biogas(p.at[k].add(h)) - biogas(p.at[k].add(-h))) / (2 * h)
    assert float(grad) == pytest.approx(float(fd), rel=1e-4)
    assert float(grad) > 0.0
