"""Dissolved gases co-stripped with oxygen by ``Aeration(stripped=...)``.

Each listed gas gets the mass-transfer term ``ratio * kLa_O2 * (sat - C)`` in the
tank's balance, with ``kLa_O2`` the fixed open-loop kLa or, under DO control, the
controller signal times the tank's gain. Covers the spec, the per-species vectors
and transfer term in both modes, the temperature correction, and, on a one-tank
plant with a small toy model, the steady state, the mass balance of the stripped
gas, the agreement of ``direct_n2o_emission`` with the in-solve flux, and a
finite gradient through the solve.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant import Aeration
from aquakin.plant.bsm import direct_n2o_emission
from aquakin.plant.cstr import CSTRUnit, aeration_transfer
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.plant import Plant

FIXTURE = Path(__file__).parent.parent / "fixtures" / "stripped_gas_model.yaml"

V = 1000.0  # m3
Q = 2000.0  # m3/d
KLA = 120.0  # 1/d
RATIO = 0.9
SN2O_IN = 2.0  # g N/m3


@pytest.fixture(scope="module")
def toy():
    return aquakin.load_model_from_file(FIXTURE)


def _tank(model, aeration, name="tank"):
    return CSTRUnit(
        name=name,
        model=model,
        volume=V,
        input_port_names=["in"],
        conditions={"T": 293.15},
        aeration=aeration,
    )


# --- spec ---------------------------------------------------------------------


def test_stripped_spec_normalizes_to_triples():
    a = Aeration(kla=KLA, stripped={"SN2O": 1.0, "SNO": (0.8, 0.01)})
    assert a.stripped == (("SN2O", 1.0, 0.0), ("SNO", 0.8, 0.01))
    hash(a)  # stays hashable (frozen dataclass)
    assert Aeration(kla=KLA).stripped == ()


@pytest.mark.parametrize(
    "stripped, match",
    [
        ({"SO": 1.0}, "aerated species itself"),
        ({"SN2O": -1.0}, "ratio"),
        ({"SN2O": (1.0, -0.1)}, "saturation"),
    ],
)
def test_stripped_spec_validation(stripped, match):
    with pytest.raises(ValueError, match=match):
        Aeration(kla=KLA, stripped=stripped)


def test_unknown_stripped_species_raises(toy):
    with pytest.raises(ValueError, match="stripped gas 'SNO'"):
        _tank(toy, Aeration(kla=KLA, stripped={"SNO": 1.0}))


# --- vectors and transfer term ------------------------------------------------


def test_open_loop_gas_kla_is_ratio_times_o2_kla(toy):
    tank = _tank(toy, Aeration(kla=KLA, alpha=0.5, stripped={"SN2O": RATIO}))
    i_o, i_g = toy.species_index["SO"], toy.species_index["SN2O"]
    assert float(tank._kla_vec[i_o]) == pytest.approx(0.5 * KLA)
    assert float(tank._kla_vec[i_g]) == pytest.approx(0.5 * KLA * RATIO)
    assert float(tank._sat_vec[i_g]) == 0.0
    C = toy.default_concentrations()
    transfer = aeration_transfer(tank._av, C, None, None, toy)
    assert float(transfer[i_g]) == pytest.approx(-0.5 * KLA * RATIO * float(C[i_g]))
    # Species that are neither oxygen nor stripped see no transfer.
    assert float(transfer[toy.species_index["SN2"]]) == 0.0


def test_closed_loop_gas_reads_signal_times_gain_times_ratio(toy):
    tank = _tank(toy, Aeration(do_setpoint=2.0, gain=0.5, stripped={"SN2O": RATIO}))
    signal = "_aer_tank_kla"
    assert tank.required_signals == (signal,)  # one signal, not one per gas
    assert tank._controlled_kla["SN2O"] == (signal, 0.5 * RATIO)
    C = toy.default_concentrations()
    transfer = aeration_transfer(tank._av, C, None, {signal: jnp.asarray(200.0)}, toy)
    i_o, i_g = toy.species_index["SO"], toy.species_index["SN2O"]
    assert float(transfer[i_g]) == pytest.approx(-200.0 * 0.5 * RATIO * float(C[i_g]))
    assert float(transfer[i_o]) == pytest.approx(200.0 * 0.5 * (8.0 - float(C[i_o])))


def test_temperature_correction_scales_gas_kla_not_gas_saturation(toy):
    aer = Aeration(
        kla=KLA, temperature_correction=True, stripped={"SN2O": (RATIO, 0.3)}
    )
    tank = _tank(toy, aer)
    i_o, i_g = toy.species_index["SO"], toy.species_index["SN2O"]
    C = jnp.zeros(toy.n_species)
    T = aer.ref_T + 10.0
    transfer = aeration_transfer(tank._av, C, T, None, toy)
    theta = aer.kla_theta**10.0
    # Gas: theta-scaled kLa, uncorrected saturation.
    assert float(transfer[i_g]) == pytest.approx(KLA * RATIO * theta * 0.3)
    # Oxygen: still gets the C_s(T) saturation correction (lower when warm).
    assert float(transfer[i_o]) < KLA * theta * 8.0


# --- one-tank plant ------------------------------------------------------------


def _plant(model, aeration):
    plant = Plant("strip")
    plant.add_unit(_tank(model, aeration))
    C_in = model.concentrations({"SS": 200.0, "SO": 0.0, "SN2O": SN2O_IN, "SN2": 0.0})
    influent = InfluentSeries(
        t=jnp.asarray([0.0, 100.0]),
        Q=jnp.full((2,), Q),
        C=jnp.tile(C_in, (2, 1)),
        model=model,
    )
    plant.add_influent("feed", influent, to="tank.in")
    return plant


def _solve(plant, params=None, t_end=5.0, n=201, y0=None):
    return plant.solve(
        t_span=(0.0, t_end),
        t_eval=jnp.linspace(0.0, t_end, n),
        params=params,
        y0=y0,
        rtol=1e-8,
        atol=1e-9,
    )


def test_open_loop_steady_state_matches_analytic(toy):
    plant = _plant(toy, Aeration(kla=KLA, stripped={"SN2O": RATIO}))
    sol = _solve(plant)
    k_n = float(toy.default_parameters()[toy.param_index["n2o_reduction.k_n"]])
    # Steady CSTR balance: Q (C_in - C) = V (k_n + ratio kLa) C.
    expected = Q * SN2O_IN / (Q + V * (k_n + RATIO * KLA))
    assert float(sol.C_named("tank", "SN2O")[-1]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("closed_loop", [False, True])
def test_stripped_gas_mass_balance_and_emission(toy, closed_loop):
    aer = (
        Aeration(do_setpoint=2.0, stripped={"SN2O": RATIO})
        if closed_loop
        else Aeration(kla=KLA, stripped={"SN2O": RATIO})
    )
    plant = _plant(toy, aer)
    # Settle first, then balance a window from the settled state: the balance's
    # flux integrals are trapezoids over the saved times, which a fast start-up
    # transient would under-resolve.
    settled = _solve(plant, t_end=5.0, n=2).final_state
    sol = _solve(plant, t_end=1.0, n=11, y0=settled)

    # The effluent is named: the DO sensor tap also reads tank.out, so it is not
    # a dangling (terminal) output of the closed-loop plant.
    mb = plant.mass_balance(sol, components=("N", "COD"), effluent_ports=["tank.out"])
    assert mb["N"].relative_imbalance == pytest.approx(0.0, abs=1e-6)
    assert mb["COD"].relative_imbalance == pytest.approx(0.0, abs=1e-6)

    # The stripped N (g over the window) is the time integral of the in-solve
    # flux, which direct_n2o_emission time-averages (kg N/d) from the tank's own
    # ratio. So they agree, and the gas term is all stripping (the toy model
    # makes no N gas by reaction).
    t = sol.t
    emission = direct_n2o_emission(plant, sol)  # kg N/d
    stripped_g = mb.gas_detail["stripped_N"]
    assert stripped_g > 0.0
    assert emission * 1e3 * float(t[-1] - t[0]) == pytest.approx(stripped_g, rel=1e-6)
    assert mb["N"].gas == pytest.approx(stripped_g, rel=1e-9)

    if closed_loop:
        # The controller holds DO, and the gas kLa follows the controlled kLa:
        # at the end, flux / C == ratio * signal.
        sig = plant.signals_at(sol.t_native[-1], sol.state[-1])["_aer_tank_kla"]
        c_g = float(sol.C_named("tank", "SN2O")[-1])
        assert float(sol.C_named("tank", "SO")[-1]) == pytest.approx(2.0, abs=1e-3)
        tank = plant.units["tank"]
        C_end = plant.states_by_unit(sol.final_state)["tank"]
        transfer = aeration_transfer(tank._av, C_end, None, {"_aer_tank_kla": sig}, toy)
        assert float(transfer[toy.species_index["SN2O"]]) == pytest.approx(
            -RATIO * float(sig) * c_g, rel=1e-9
        )


def test_gradient_through_stripping_solve_is_finite(toy):
    plant = _plant(toy, Aeration(do_setpoint=2.0, stripped={"SN2O": RATIO}))
    p0 = plant.default_parameters()
    idx = plant.parameter_index("stripped_gas_toy.n2o_reduction.k_n")

    def final_n2o(p):
        sol = _solve(plant, params=p, t_end=2.0, n=3)
        return sol.C_named("tank", "SN2O")[-1]

    g = jax.grad(final_n2o)(p0)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(g[idx]) < 0.0  # faster reduction, less dissolved N2O
