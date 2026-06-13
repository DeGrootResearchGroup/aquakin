"""BSM2 hydraulic influent-bypass tests.

Two layers: the ``SplitterUnit`` threshold mode (fast, no plant solve) and the
wired ``build_bsm2(bypass=InfluentBypass())`` plant (one module-scoped solve at a
high influent flow, since the suite runs near the CI runner's limit).
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm.bsm2 import (
    BSM2_BYPASS_Q,
    build_bsm2,
    InfluentBypass,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
)
from aquakin.plant.bsm import evaluate_bsm2
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.mixer import SplitterUnit
from aquakin.plant.streams import Stream


# ----- SplitterUnit threshold mode (no plant solve) -----------------------

@pytest.fixture(scope="module")
def asm1():
    return bsm2_asm1_network()


def _threshold_splitter(asm1, limit=60000.0):
    return SplitterUnit(
        name="bypass", network=asm1, threshold=limit,
        threshold_port="bypass", remainder_port="to_plant")


def test_threshold_split_above_limit(asm1):
    """Flow above the limit goes to the threshold port, the rest to remainder."""
    split = _threshold_splitter(asm1)
    flows = split.flow_outputs({"in": jnp.asarray(90000.0)}, None)
    assert float(flows["bypass"]) == pytest.approx(30000.0)
    assert float(flows["to_plant"]) == pytest.approx(60000.0)


def test_threshold_split_below_limit_is_inactive(asm1):
    """Below the limit nothing is diverted."""
    split = _threshold_splitter(asm1)
    flows = split.flow_outputs({"in": jnp.asarray(40000.0)}, None)
    assert float(flows["bypass"]) == pytest.approx(0.0)
    assert float(flows["to_plant"]) == pytest.approx(40000.0)


def test_threshold_split_preserves_concentration(asm1):
    """A passive split preserves concentration on both outlets."""
    split = _threshold_splitter(asm1)
    C = asm1.default_concentrations()
    s_in = {"in": Stream(Q=jnp.asarray(90000.0), C=C, network=asm1)}
    outs = split.compute_outputs(0.0, jnp.zeros((0,)), s_in, None)
    assert jnp.allclose(outs["bypass"].C, C)
    assert jnp.allclose(outs["to_plant"].C, C)
    # Flows conserve total.
    assert float(outs["bypass"].Q + outs["to_plant"].Q) == pytest.approx(90000.0)


def test_threshold_mode_validation(asm1):
    with pytest.raises(ValueError, match="threshold requires"):
        SplitterUnit(name="b", network=asm1, threshold=1.0, threshold_port="x")
    with pytest.raises(ValueError, match="exactly one"):
        SplitterUnit(name="b", network=asm1, threshold=1.0, threshold_port="x",
                     remainder_port="y", output_port_ratios={"x": 1.0})


# ----- Wired BSM2 plant with bypass ---------------------------------------

@pytest.fixture(scope="module")
def adm1():
    return aquakin.load_network("adm1")


@pytest.fixture(scope="module")
def bypass_run(asm1, adm1):
    """Build the bypass plant, drive it above the threshold, solve once."""
    params = bsm2_parameters(asm1, adm1)
    C = bsm2_constant_influent(asm1).C[0]
    Q_in = 1.5 * BSM2_BYPASS_Q  # 90000: half again the bypass limit
    influent = InfluentSeries(t=jnp.array([0.0, 1e4]), Q=jnp.full((2,), Q_in),
                              C=jnp.tile(C, (2, 1)), network=asm1)
    plant = build_bsm2(asm1, adm1, bypass=InfluentBypass())
    plant.add_influent("feed", influent)
    sol = plant.solve((0.0, 10.0), t_eval=jnp.linspace(0.0, 10.0, 11),
                      params=params, rtol=1e-4, atol=1e-3, max_steps=400_000)
    return plant, sol, params, Q_in


def test_bypass_plant_is_finite(bypass_run):
    _, sol, _, _ = bypass_run
    assert jnp.all(jnp.isfinite(sol.state))


def test_bypass_flow_split(bypass_run):
    """The diverted flow is max(Q_in - threshold, 0); the plant gets the rest."""
    plant, sol, params, Q_in = bypass_run
    bp = plant.stream(sol, "bypass_split.bypass", params)
    pl = plant.stream(sol, "bypass_split.to_plant", params)
    assert float(bp.Q[-1]) == pytest.approx(Q_in - BSM2_BYPASS_Q, rel=1e-6)
    assert float(pl.Q[-1]) == pytest.approx(BSM2_BYPASS_Q, rel=1e-6)


def test_effluent_is_treated_plus_bypass(bypass_run):
    """The final effluent flow is the clarified flow plus the bypassed flow."""
    plant, sol, params, _ = bypass_run
    eff = plant.stream(sol, plant.effluent_endpoint, params)
    treated = plant.stream(sol, "settler.overflow", params)
    bp = plant.stream(sol, "bypass_split.bypass", params)
    assert float(eff.Q[-1]) == pytest.approx(
        float(treated.Q[-1] + bp.Q[-1]), rel=1e-6)


def test_bypass_degrades_effluent(bypass_run):
    """Routing raw influent around treatment raises the effluent pollutant load
    above the clarified stream's."""
    from aquakin.plant.metrics import derived_COD
    plant, sol, params, _ = bypass_run
    eff = plant.stream(sol, plant.effluent_endpoint, params)
    treated = plant.stream(sol, "settler.overflow", params)
    cod_eff = float(derived_COD(eff.C[-1], plant.units["tank1"].network))
    cod_treated = float(derived_COD(treated.C[-1], plant.units["tank1"].network))
    assert cod_eff > cod_treated


def test_evaluate_autodetects_bypass_effluent(bypass_run):
    """evaluate_bsm2 scores the combined effluent (effluent_mix.out) when the
    bypass is present, not the clarifier overflow alone."""
    plant, sol, params, _ = bypass_run
    ev = evaluate_bsm2(plant, sol, params)
    eff = plant.stream(sol, plant.effluent_endpoint, params)
    from aquakin.plant.metrics import effluent_averages
    expected = effluent_averages(eff.t, eff.C, eff.Q, plant.units["tank1"].network)
    assert ev.effluent["COD"] == pytest.approx(expected["COD"], rel=1e-6)
    assert jnp.isfinite(ev.eqi)
