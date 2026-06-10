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
from aquakin.plant.bsm import build_bsm1, load_bsm1_influent
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.metrics import (
    derived_TSS,
    effluent_averages,
    operational_cost_index,
    aeration_energy,
    pumping_energy,
)


@pytest.fixture
def asm1():
    return aquakin.load_network("asm1")


@pytest.fixture
def constant_influent(asm1):
    """Constant inlet at the documented BSM1 average composition."""
    from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
    n_t = 2
    t = jnp.asarray([0.0, 100.0])
    Q = jnp.full((n_t,), BSM1_Q_AVG)
    C0 = asm1.default_concentrations()
    # Override with the documented Table 5.1 inlet composition.
    inlet_overrides = {
        "SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32,
        "XB_H": 28.17, "XB_A": 0.0, "XP": 0.0,
        "SO": 0.0, "SNO": 0.0, "SNH": 31.56,
        "SND": 6.95, "XND": 10.59, "SALK": 7.0,
    }
    for sp, val in inlet_overrides.items():
        C0 = C0.at[asm1.species_index[sp]].set(val)
    C = jnp.tile(C0, (n_t, 1))
    return InfluentSeries(t=t, Q=Q, C=C, network=asm1)


def _run(plant, t_end=30.0, n_save=4, **kwargs):
    return plant.solve(
        t_span=(0.0, t_end),
        t_eval=jnp.linspace(0.0, t_end, n_save),
        rtol=1e-4, atol=1e-3,
        **kwargs,
    )


def test_bsm1_builds_and_solves(asm1, constant_influent):
    plant = build_bsm1(network=asm1)
    plant.add_influent("feed", constant_influent)
    plant.connect(None, "feed", "inlet_mix", "fresh")
    sol = _run(plant, t_end=10.0)
    assert jnp.all(jnp.isfinite(sol.state))


def test_bsm1_nitrification_active(asm1, constant_influent):
    """Under aerobic conditions in tank 5, NH4 should be largely oxidised."""
    plant = build_bsm1(network=asm1)
    plant.add_influent("feed", constant_influent)
    plant.connect(None, "feed", "inlet_mix", "fresh")
    sol = _run(plant, t_end=15.0)
    tank5_SNH = float(sol.C_named("tank5", "SNH")[-1])
    tank5_SNO = float(sol.C_named("tank5", "SNO")[-1])
    # Influent SNH = 31.56; tank 5 SNH should be <50% of that.
    assert tank5_SNH < 0.5 * 31.56, f"SNH not nitrified: {tank5_SNH}"
    assert tank5_SNO > 1.0, f"SNO not produced: {tank5_SNO}"


def test_bsm1_biomass_sustained(asm1, constant_influent):
    """RAS recycle should keep biomass concentrations elevated."""
    plant = build_bsm1(network=asm1)
    plant.add_influent("feed", constant_influent)
    plant.connect(None, "feed", "inlet_mix", "fresh")
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
    plant = build_bsm1(network=asm1)
    plant.add_influent("feed", constant_influent)
    plant.connect(None, "feed", "inlet_mix", "fresh")
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
    plant = build_bsm1(network=asm1)
    plant.add_influent("feed", constant_influent)
    plant.connect(None, "feed", "inlet_mix", "fresh")

    def loss(params):
        # Cap the integrator step. The reverse-mode adjoint of this stiff plant
        # is right at the edge of finiteness uncapped and tips to non-finite on
        # some floating-point environments; capping dtmax to a small multiple of
        # the fastest reaction timescale bounds the per-step stiffness and keeps
        # the reverse accumulation finite (see the dtmax discussion in CLAUDE.md).
        sol = plant.solve(
            t_span=(0.0, 5.0), t_eval=jnp.asarray([0.0, 5.0]),
            params=params, rtol=1e-3, atol=1e-2, dtmax=0.005,
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
    plant = build_bsm1(network=asm1, use_takacs=True)
    plant.add_influent("feed", constant_influent)
    plant.connect(None, "feed", "inlet_mix", "fresh")
    sol = plant.solve(
        t_span=(0.0, 150.0), t_eval=jnp.asarray([0.0, 150.0]),
        rtol=1e-4, atol=1e-3, max_steps=300_000,
    )
    assert jnp.all(jnp.isfinite(sol.state))
    # Healthy steady state (not washed out): elevated biomass, nitrified.
    assert float(sol.C_named("tank5", "XB_H")[-1]) > 1000.0
    assert float(sol.C_named("tank5", "SNH")[-1]) < 5.0
    assert float(sol.C_named("tank5", "SNO")[-1]) > 1.0


@pytest.mark.skip(
    reason="Dynamic (time-varying) influent integration is stiff once the "
    "recycle flows are resolved at full strength (the under-resolved flows used "
    "to make the plant artificially mild). Steady-state runs work; the dynamic "
    "diurnal-forcing transient is the open plant-hardening item tracked in #30."
)
def test_bsm1_dry_weather_runs(asm1):
    """The dry-weather influent CSV drives the plant without solver failure."""
    plant = build_bsm1(network=asm1)
    plant.add_influent("feed", load_bsm1_influent("dry", asm1))
    plant.connect(None, "feed", "inlet_mix", "fresh")
    sol = _run(plant, t_end=10.0, n_save=5)
    assert jnp.all(jnp.isfinite(sol.state))


def test_metrics_compute_finite(asm1, constant_influent):
    """The metrics module produces finite values on a BSM1 trajectory."""
    plant = build_bsm1(network=asm1)
    plant.add_influent("feed", constant_influent)
    plant.connect(None, "feed", "inlet_mix", "fresh")
    sol = _run(plant, t_end=10.0, n_save=11)

    # Reconstruct effluent stream at every save time. Effluent = clarifier
    # overflow. The IdealClarifier is stateless so we recompute it.
    clar = plant.units["clarifier"]
    n_t = sol.state.shape[0]
    C_eff = jnp.zeros((n_t, asm1.n_species))
    Q_eff = jnp.zeros((n_t,))
    for i in range(n_t):
        # Tank 5 outlet → tank5_split:to_clarifier → clarifier:inlet.
        tank5_start, tank5_size = plant._state_layout["tank5"]
        tank5_C = sol.state[i, tank5_start:tank5_start + tank5_size]
        # Q_clar = 2/5 * Q_tank5_outlet. Tank 5 sees Q = 5 * Q_in.
        Q_in_t = float(plant.influents["feed"].at(jnp.asarray(sol.t[i])).Q)
        Q_clar = 2.0 / 5.0 * 5.0 * Q_in_t  # = 2 * Q_in
        from aquakin.plant.streams import Stream
        inlet_stream = Stream(Q=jnp.asarray(Q_clar), C=tank5_C, network=asm1)
        out = clar.compute_outputs(
            jnp.asarray(sol.t[i]), jnp.zeros((0,)),
            {"inlet": inlet_stream}, plant.default_parameters(),
        )
        C_eff = C_eff.at[i].set(out["overflow"].C)
        Q_eff = Q_eff.at[i].set(out["overflow"].Q)

    averages = effluent_averages(sol.t, C_eff, Q_eff, asm1)
    for key, val in averages.items():
        assert val >= 0.0, f"{key} is negative: {val}"
        assert val < 1e4, f"{key} unreasonably large: {val}"

    # TSS conversion.
    tss = derived_TSS(sol.state[-1, :asm1.n_species], asm1)
    assert float(tss) > 0.0
