"""BSM1 plant integration tests.

Verifies the end-to-end behaviour of the BSM1 reference plant under
constant and dynamic influent. Since the shipped influent CSVs are
synthesised (not the canonical IWA files), the tests assert on the
*qualitative* BSM1 behaviour — nitrification active, biomass alive,
flow balance, AD-grad cleanness — rather than on absolute EQI / OCI
values that would need the canonical files for ~1% comparison.
"""

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin import (
    BSM1Evaluation,
    derived_TSS,
    effluent_averages,
    effluent_quality_index,
    evaluate_bsm1,
)
from aquakin.plant.bsm import build_bsm1, load_bsm1_influent
from aquakin.plant.influent import InfluentSeries

# Slow module: full BSM1 plant solves (5 reactors + clarifier + recycles).
# Excluded from the fast PR gate; runs in the merge-to-main suite.
pytestmark = pytest.mark.slow


@pytest.fixture
def asm1():
    return aquakin.load_model("asm1")


@pytest.fixture
def constant_influent(asm1):
    """Constant inlet at the documented BSM1 average composition."""
    from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
    n_t = 2
    t = jnp.asarray([0.0, 100.0])
    Q = jnp.full((n_t,), BSM1_Q_AVG)
    # The documented Table 5.1 inlet composition (rest = model defaults).
    C0 = asm1.concentrations({
        "SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32,
        "XB_H": 28.17, "XB_A": 0.0, "XP": 0.0,
        "SO": 0.0, "SNO": 0.0, "SNH": 31.56,
        "SND": 6.95, "XND": 10.59, "SALK": 7.0,
    })
    C = jnp.tile(C0, (n_t, 1))
    return InfluentSeries(t=t, Q=Q, C=C, model=asm1)


def _run(plant, t_end=30.0, n_save=4, **kwargs):
    return plant.solve(
        t_span=(0.0, t_end),
        t_eval=jnp.linspace(0.0, t_end, n_save),
        rtol=1e-4, atol=1e-3,
        **kwargs,
    )


def test_bsm1_builds_and_solves(asm1, constant_influent):
    plant = build_bsm1(model=asm1)
    plant.add_influent("feed", constant_influent, to="inlet_mix.fresh")
    sol = _run(plant, t_end=10.0)
    assert jnp.all(jnp.isfinite(sol.state))


@pytest.mark.parametrize("use_takacs", [False, True])
def test_recycle_presolve_makes_mopup_passes_irrelevant(
    asm1, constant_influent, use_takacs
):
    """The recycle back-edges are seeded with their *exact* affine fixed point
    (``_resolve_recycle_concentrations``) before the Gauss-Seidel mop-up, so the
    BSM1 RHS is identical at 1, 2 and 10 mop-up passes -- the seed is already the
    answer, and the pass count does no work. (Before the pre-solve this needed 2
    passes; now it is exact at any count, gain-independent.)"""
    import numpy as np

    plant = build_bsm1(model=asm1, use_takacs=use_takacs)
    plant.add_influent("feed", constant_influent, to="inlet_mix.fresh")
    y0 = plant.initial_state()

    def dstate(n):
        plant.recycle_passes = n
        return np.asarray(plant.derivative(y0))

    d1, d2, d10 = dstate(1), dstate(2), dstate(10)
    tol = 1e-6 * (np.linalg.norm(d10) + 1.0)
    assert np.allclose(d1, d10, rtol=1e-8, atol=tol)
    assert np.allclose(d2, d10, rtol=1e-8, atol=tol)


def test_recycle_passes_validated_and_configurable(asm1):
    from aquakin.plant import Plant

    with pytest.raises(ValueError):
        Plant("p", recycle_passes=0)
    assert Plant("p", recycle_passes=5).recycle_passes == 5
    assert Plant("p").recycle_passes == 3  # default


def test_bsm1_nitrification_active(asm1, constant_influent):
    """Under aerobic conditions in tank 5, NH4 should be largely oxidised."""
    plant = build_bsm1(model=asm1)
    plant.add_influent("feed", constant_influent, to="inlet_mix.fresh")
    sol = _run(plant, t_end=15.0)
    tank5_SNH = float(sol.C_named("tank5", "SNH")[-1])
    tank5_SNO = float(sol.C_named("tank5", "SNO")[-1])
    # Influent SNH = 31.56; tank 5 SNH should be <50% of that.
    assert tank5_SNH < 0.5 * 31.56, f"SNH not nitrified: {tank5_SNH}"
    assert tank5_SNO > 1.0, f"SNO not produced: {tank5_SNO}"


def test_bsm1_biomass_sustained(asm1, constant_influent):
    """RAS recycle should keep biomass concentrations elevated."""
    plant = build_bsm1(model=asm1)
    plant.add_influent("feed", constant_influent, to="inlet_mix.fresh")
    sol = _run(plant, t_end=15.0)
    # Heterotrophic biomass should grow well above the influent value
    # (which is 28.17) via the recycled mass.
    xbh = float(sol.C_named("tank3", "XB_H")[-1])
    assert xbh > 500.0, f"XB_H washed out: {xbh}"
    # Autotrophs must persist for nitrification to happen.
    xba = float(sol.C_named("tank5", "XB_A")[-1])
    assert xba > 10.0, f"XB_A washed out: {xba}"


def test_bsm1_aerobic_anoxic_separation(asm1, constant_influent):
    """Aerobic tanks should have positive SO; anoxic tanks ~zero."""
    plant = build_bsm1(model=asm1)
    plant.add_influent("feed", constant_influent, to="inlet_mix.fresh")
    sol = _run(plant, t_end=10.0)
    so1 = float(sol.C_named("tank1", "SO")[-1])
    so3 = float(sol.C_named("tank3", "SO")[-1])
    so5 = float(sol.C_named("tank5", "SO")[-1])
    assert so1 < 0.1, f"anoxic tank 1 has DO: {so1}"
    # Aerobic tanks must have non-trivial DO. Tank 3 (first aerobic) often
    # sits near 0.3-0.5 mg/L with open-loop kLa under heavy load; tank 5
    # tends higher because by then most of the BOD has been consumed.
    assert so3 > 0.1, f"aerobic tank 3 starved: {so3}"
    assert so5 > 0.1, f"aerobic tank 5 starved: {so5}"
    # Either way, aerobic tanks must be much higher than the anoxic.
    assert so3 > 5 * so1
    assert so5 > 5 * so1


def test_bsm1_grad_through_plant(asm1, constant_influent):
    """jax.grad through plant.solve must produce finite gradients."""
    plant = build_bsm1(model=asm1)
    plant.add_influent("feed", constant_influent, to="inlet_mix.fresh")

    def loss(params):
        # The default reverse-mode gradient (diff.method="stable", the cap-free
        # hand-written discrete adjoint) stays finite on this stiff plant with no
        # dtmax cap -- the cap was only ever needed by the old through-the-solve
        # adjoint (diff.method="through_solve").
        sol = plant.solve(
            t_span=(0.0, 5.0), t_eval=jnp.asarray([0.0, 5.0]),
            params=params, rtol=1e-3, atol=1e-2,
        )
        # Sum SNO across all tanks at endpoint (a quantity that depends
        # on every nitrification-related parameter).
        total = jnp.zeros(())
        for name in ("tank1", "tank2", "tank3", "tank4", "tank5"):
            total = total + sol.C_named(name, "SNO")[-1]
        return total

    g = jax.grad(loss)(plant.default_parameters())
    assert jnp.all(jnp.isfinite(g))


def test_bsm1_takacs_reaches_steady_state(asm1, constant_influent):
    """The full Takács 1-D clarifier plant integrates to the correct BSM1
    steady state (not just the fast stateless IdealClarifier). This exercises
    the decoupled recycle-flow resolution: without it the high-gain recycle
    flow loop is under-resolved, the underflow is starved, and the plant washes
    out. The Takács result should match the IdealClarifier's healthy steady
    state."""
    plant = build_bsm1(model=asm1, use_takacs=True)
    plant.add_influent("feed", constant_influent, to="inlet_mix.fresh")
    sol = plant.solve(
        t_span=(0.0, 150.0), t_eval=jnp.asarray([0.0, 150.0]),
        rtol=1e-4, atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=300_000),
    )
    assert jnp.all(jnp.isfinite(sol.state))
    # Healthy steady state (not washed out): elevated biomass, nitrified.
    assert float(sol.C_named("tank5", "XB_H")[-1]) > 1000.0
    assert float(sol.C_named("tank5", "SNH")[-1]) < 5.0
    assert float(sol.C_named("tank5", "SNO")[-1]) > 1.0


def test_bsm1_dry_weather_runs(asm1):
    """The dry-weather influent CSV drives the plant efficiently to a healthy
    state. This is the dynamic-influent counterpart of the steady-state test
    and the regression guard for the recycle-flow-control fix: the recycle
    pumps (internal recycle, RAS, wastage) deliver fixed setpoint flows, so the
    throughput tracks the influent smoothly instead of the near-singular
    fixed-fraction gain that blew the throughput up to ~20x Qin and made the
    monolithic solve hit the step ceiling under diurnal forcing (issue #30)."""
    plant = build_bsm1(model=asm1)
    plant.add_influent("feed", load_bsm1_influent("dry", asm1), to="inlet_mix.fresh")
    sol = _run(plant, t_end=14.0, n_save=8)
    assert jnp.all(jnp.isfinite(sol.state))
    # Healthy plant under the diurnal load: biomass sustained, nitrified.
    assert float(sol.C_named("tank5", "XB_H")[-1]) > 1000.0
    assert float(sol.C_named("tank5", "SNH")[-1]) < 5.0
    assert float(sol.C_named("tank5", "SNO")[-1]) > 1.0


def test_bsm1_takacs_dry_weather_runs(asm1):
    """The full Takács 1-D clarifier plant also integrates the dynamic dry
    influent to a healthy state (issue #30): the fixed-setpoint recycle pumps
    keep the layered settler's flows bounded under diurnal forcing."""
    plant = build_bsm1(model=asm1, use_takacs=True)
    plant.add_influent("feed", load_bsm1_influent("dry", asm1), to="inlet_mix.fresh")
    sol = plant.solve(
        t_span=(0.0, 14.0), t_eval=jnp.linspace(0.0, 14.0, 8),
        rtol=1e-4, atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=200_000),
    )
    assert jnp.all(jnp.isfinite(sol.state))
    assert float(sol.C_named("tank5", "XB_H")[-1]) > 1000.0
    assert float(sol.C_named("tank5", "SNH")[-1]) < 5.0


def test_metrics_compute_finite(asm1, constant_influent):
    """The metrics module produces finite values on a BSM1 trajectory."""
    plant = build_bsm1(model=asm1)
    plant.add_influent("feed", constant_influent, to="inlet_mix.fresh")
    sol = _run(plant, t_end=10.0, n_save=11)

    # Effluent = the clarifier overflow, reconstructed from the saved states.
    eff = plant.stream(sol, "clarifier.overflow")
    averages = effluent_averages(eff.t, eff.C, eff.Q, asm1)
    for key, val in averages.items():
        assert val >= 0.0, f"{key} is negative: {val}"
        assert val < 1e4, f"{key} unreasonably large: {val}"

    # TSS conversion.
    tss = derived_TSS(sol.state[-1, :asm1.n_species], asm1)
    assert float(tss) > 0.0


def test_metrics_accept_stream_series(asm1, constant_influent):
    """The metric kernels take a StreamSeries directly (model from the stream),
    giving the same result as the unpacked-array call."""
    plant = build_bsm1(model=asm1)
    plant.add_influent("feed", constant_influent, to="inlet_mix.fresh")
    sol = _run(plant, t_end=10.0, n_save=11)
    eff = plant.stream(sol, "clarifier.overflow")

    # StreamSeries form == explicit-array form.
    eqi_stream = effluent_quality_index(eff)
    eqi_arrays = effluent_quality_index(eff.t, eff.C, eff.Q, asm1)
    assert eqi_stream == pytest.approx(eqi_arrays)

    avg_stream = effluent_averages(eff)
    avg_arrays = effluent_averages(eff.t, eff.C, eff.Q, asm1)
    assert avg_stream == pytest.approx(avg_arrays)

    # derived_* take a StreamSeries (model from it) and match the array form.
    tss_stream = derived_TSS(eff)
    tss_arrays = derived_TSS(eff.C, asm1)
    assert jnp.allclose(tss_stream, tss_arrays)


def test_evaluate_bsm1_indices(asm1, constant_influent):
    """evaluate_bsm1 returns finite, positive EQI / OCI and component terms."""
    plant = build_bsm1(model=asm1)
    plant.add_influent("feed", constant_influent, to="inlet_mix.fresh")
    # Settle toward steady state so the indices are representative.
    sol = plant.solve(
        t_span=(0.0, 60.0), t_eval=jnp.linspace(50.0, 60.0, 6),
        rtol=1e-4, atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=200_000),
    )
    ev = evaluate_bsm1(plant, sol)
    assert isinstance(ev, BSM1Evaluation)
    assert ev.eqi > 0.0 and jnp.isfinite(ev.eqi)
    assert ev.aeration_energy > 0.0
    assert ev.pumping_energy > 0.0
    assert ev.sludge_production > 0.0
    # The two unaerated reactors (tanks 1-2) are mechanically mixed.
    assert ev.mixing_energy > 0.0
    # OCI is the updated BSM1 form AE + PE + ME + 5*sludge.
    assert ev.oci == pytest.approx(
        ev.aeration_energy + ev.pumping_energy + ev.mixing_energy
        + 5.0 * ev.sludge_production)
    # The three aerated tanks (tanks 3-5) are counted.
    assert ev.aerated_tanks == ["tank3", "tank4", "tank5"]


def test_evaluate_bsm1_on_single_point_steady_state(asm1, constant_influent):
    """The natural 'run to steady state, then evaluate' flow used to crash with
    ZeroDivisionError, because run_to_steady_state returns a one-point solution
    and aeration_energy divided by a zero window. It now returns finite, positive
    indices -- the instantaneous steady-state values (issue #180)."""
    plant = build_bsm1(model=asm1)
    plant.add_influent("feed", constant_influent, to="inlet_mix.fresh")
    ss = plant.run_to_steady_state()
    assert ss.solution.t.shape[0] == 1            # the degenerate single point
    ev = evaluate_bsm1(plant, ss.solution)         # no ZeroDivisionError
    for name in ("eqi", "oci", "aeration_energy", "pumping_energy",
                 "sludge_production"):
        v = getattr(ev, name)
        assert jnp.isfinite(v) and v > 0.0, name
    assert ev.oci == pytest.approx(
        ev.aeration_energy + ev.pumping_energy + ev.mixing_energy
        + 5.0 * ev.sludge_production)
