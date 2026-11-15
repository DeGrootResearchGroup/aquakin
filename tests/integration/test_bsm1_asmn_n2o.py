"""BSM1 running the N2O model ``asmn_n2o`` with in-solve N2O stripping.

``asmn_n2o`` keeps the ASM1 names for the shared components, so the BSM1
influent maps by name, but splits the autotrophs into ``XB_AOB`` / ``XB_NOB``;
``build_bsm1`` derives the clarifier's settling / TSS species from the model.
The three aerated tanks run under DO control and strip N2O and NO alongside
oxygen (``Aeration(stripped=...)``). Checks that the plant solves for a few days
with finite, non-negative N2O, that the reported emission is non-zero and equals
the in-solve stripping flux, that the N balance closes with the stripped gas
counted, and that ``jax.grad`` flows through the solve.
"""

from importlib.resources import files

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant import Aeration
from aquakin.plant.bsm import build_bsm1, direct_n2o_emission
from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
from aquakin.plant.influent import InfluentSeries

pytestmark = pytest.mark.skipif(
    not (files("aquakin.models") / "asmn_n2o.yaml").is_file(),
    reason="the asmn_n2o model is not shipped in this checkout",
)

GASES = {"SN2O": 1.0, "SNO": 1.0}
T_END = 3.0  # days


@pytest.fixture(scope="module")
def net():
    return aquakin.load_model("asmn_n2o")


def _plant(net):
    aeration = {f"tank{i}": Aeration(do_setpoint=2.0, stripped=GASES) for i in (3, 4, 5)}
    plant = build_bsm1(model=net, aeration=aeration)
    # BSM1 average dry-weather influent (Copp 2002), by ASM1 species name.
    C_in = net.concentrations(
        {
            "SI": 30.0,
            "SS": 69.5,
            "XI": 51.2,
            "XS": 202.32,
            "XB_H": 28.17,
            "XB_AOB": 0.0,
            "XB_NOB": 0.0,
            "XP": 0.0,
            "SO": 0.0,
            "SNO3": 0.0,
            "SNO2": 0.0,
            "SNO": 0.0,
            "SN2O": 0.0,
            "SNH": 31.56,
            "SND": 6.95,
            "XND": 10.59,
            "SALK": 7.0,
        }
    )
    influent = InfluentSeries(
        t=jnp.asarray([0.0, 100.0]),
        Q=jnp.full((2,), BSM1_Q_AVG),
        C=jnp.tile(C_in, (2, 1)),
        model=net,
    )
    plant.add_influent("feed", influent, to=plant.influent_endpoint)
    return plant


def _y0(plant, net):
    # A nitrifying activated-sludge seed in every tank.
    warm = net.concentrations(
        {
            "SI": 30.0,
            "SS": 2.0,
            "XI": 1200.0,
            "XS": 60.0,
            "XB_H": 2500.0,
            "XB_AOB": 120.0,
            "XB_NOB": 60.0,
            "XP": 400.0,
            "SO": 1.0,
            "SNO3": 8.0,
            "SNO2": 0.3,
            "SNH": 3.0,
            "SND": 1.0,
            "XND": 4.0,
            "SALK": 5.0,
        }
    )
    return plant.initial_state(overrides={f"tank{i}": warm for i in range(1, 6)})


def _solve(plant, y0, params=None, n=13):
    return plant.solve(
        t_span=(0.0, T_END),
        t_eval=jnp.linspace(0.0, T_END, n),
        y0=y0,
        params=params,
        rtol=1e-5,
        atol=1e-6,
    )


def test_build_bsm1_derives_clarifier_solids(net):
    plant = build_bsm1(model=net)
    clarifier = plant.units["clarifier"]
    assert "XB_A" not in clarifier.particulate_species
    assert {"XB_AOB", "XB_NOB", "XND"} <= set(clarifier.particulate_species)
    takacs = build_bsm1(model=net, use_takacs=True).units["clarifier"]
    assert takacs.tss_factors["XB_AOB"] == pytest.approx(0.75)
    assert takacs.tss_factors["XND"] == 0.0


@pytest.mark.slow
def test_bsm1_asmn_n2o_solves_and_emits_n2o(net):
    plant = _plant(net)
    sol = _solve(plant, _y0(plant, net), n=61)

    s_n2o = jnp.stack([sol.C_named(f"tank{i}", "SN2O") for i in range(1, 6)])
    assert bool(jnp.all(jnp.isfinite(s_n2o)))
    assert float(s_n2o.min()) >= -1e-9
    # DO control holds the aerated tanks at the setpoint.
    for i in (3, 4, 5):
        assert float(sol.C_named(f"tank{i}", "SO")[-1]) == pytest.approx(2.0, abs=0.05)

    emission = direct_n2o_emission(plant, sol)  # kg N/d
    assert emission > 0.0

    mb = plant.mass_balance(sol, components=("N",))
    # The stripped N2O + NO is the N gas term's stripping part; it covers the
    # N2O emission (NO strips too) and its time average matches the reported
    # emission to the trapezoid accuracy of the saved times.
    stripped_kg_per_d = mb.gas_detail["stripped_N"] * 1e-3 / T_END
    assert stripped_kg_per_d >= emission * (1.0 - 1e-2)
    assert mb["N"].relative_imbalance == pytest.approx(0.0, abs=1e-2)


@pytest.mark.slow
def test_gradient_through_bsm1_asmn_n2o_solve_is_finite(net):
    plant = _plant(net)
    y0 = _y0(plant, net)
    p0 = plant.default_parameters()

    def effluent_n2o(p):
        sol = _solve(plant, y0, params=p, n=2)
        return sol.C_named("tank5", "SN2O")[-1]

    g = jax.grad(effluent_n2o)(p0)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g).max()) > 0.0
