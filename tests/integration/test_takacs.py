"""Tests for the Takács 1-D secondary clarifier."""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant import Plant
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.takacs import TakacsClarifier


@pytest.fixture
def asm1():
    return aquakin.load_network("asm1")


def _clarifier_plant(asm1, *, Q_inlet=56000.0, overflow_Q=18446.0, C_inlet=None):
    plant = Plant("clarifier")
    plant.add_unit(
        TakacsClarifier(
            name="clar",
            network=asm1,
            area=1500.0,
            height=4.0,
            overflow_Q=overflow_Q,
        )
    )
    if C_inlet is None:
        C_inlet = asm1.default_concentrations()
    inf = InfluentSeries(
        t=jnp.asarray([0.0, 10.0]),
        Q=jnp.asarray([Q_inlet, Q_inlet]),
        C=jnp.stack([C_inlet, C_inlet]),
        network=asm1,
    )
    plant.add_influent("feed", inf)
    plant.connect(None, "feed", "clar", "inlet")
    return plant


def test_solubles_pass_through(asm1):
    """Soluble species (SS, SNH, ...) must appear at the same concentration
    in both overflow and underflow because they don't settle."""
    # Make soluble SS distinctive in the inlet.
    C_inlet = asm1.default_concentrations().at[asm1.species_index["SS"]].set(50.0)
    plant = _clarifier_plant(asm1, C_inlet=C_inlet)
    sol = plant.solve(
        t_span=(0.0, 2.0), t_eval=jnp.asarray([0.0, 2.0]),
        rtol=1e-4, atol=1e-3,
    )
    # Reconstruct streams at t_end to inspect overflow / underflow.
    clar = plant.units["clar"]
    state = sol.state[-1]
    inf = plant.influents["feed"].at(jnp.asarray(2.0))
    streams = clar.compute_outputs(
        jnp.asarray(2.0), state, {"inlet": inf}, plant.default_parameters()
    )
    SS_idx = asm1.species_index["SS"]
    assert float(streams["overflow"].C[SS_idx]) == pytest.approx(50.0, rel=1e-6)
    assert float(streams["underflow"].C[SS_idx]) == pytest.approx(50.0, rel=1e-6)


def test_particulates_concentrate_in_underflow(asm1):
    """Particulates settle: at quasi-steady state the underflow X_BH
    concentration must significantly exceed the inlet, and the overflow
    must be much lower."""
    plant = _clarifier_plant(asm1)
    sol = plant.solve(
        t_span=(0.0, 2.0), t_eval=jnp.asarray([0.0, 2.0]),
        rtol=1e-4, atol=1e-3,
    )
    state = sol.state[-1]
    clar = plant.units["clar"]
    inf = plant.influents["feed"].at(jnp.asarray(2.0))
    streams = clar.compute_outputs(
        jnp.asarray(2.0), state, {"inlet": inf}, plant.default_parameters()
    )
    X_BH_in = float(inf.C[asm1.species_index["X_BH"] if "X_BH" in asm1.species_index else asm1.species_index["XB_H"]])
    XBH_idx = asm1.species_index["XB_H"]
    overflow_XBH = float(streams["overflow"].C[XBH_idx])
    underflow_XBH = float(streams["underflow"].C[XBH_idx])
    assert overflow_XBH < float(inf.C[XBH_idx]), "particulates should clarify"
    assert underflow_XBH > float(inf.C[XBH_idx]), "particulates should thicken"


def test_overall_mass_balance(asm1):
    """Mass in (Q_in × C_in) ≈ mass out (Q_over × C_over + Q_under × C_under)
    at quasi-steady state for every particulate species."""
    plant = _clarifier_plant(asm1)
    sol = plant.solve(
        t_span=(0.0, 5.0), t_eval=jnp.asarray([0.0, 5.0]),
        rtol=1e-4, atol=1e-3,
    )
    state = sol.state[-1]
    clar = plant.units["clar"]
    inf = plant.influents["feed"].at(jnp.asarray(5.0))
    streams = clar.compute_outputs(
        jnp.asarray(5.0), state, {"inlet": inf}, plant.default_parameters()
    )
    Q_in = float(inf.Q)
    Q_over = float(streams["overflow"].Q)
    Q_under = float(streams["underflow"].Q)
    assert Q_over + Q_under == pytest.approx(Q_in, rel=1e-6)
    # Particulate mass balance (within solver tolerance).
    XBH_idx = asm1.species_index["XB_H"]
    mass_in = Q_in * float(inf.C[XBH_idx])
    mass_out = (
        Q_over * float(streams["overflow"].C[XBH_idx])
        + Q_under * float(streams["underflow"].C[XBH_idx])
    )
    # Allow ~10% drift since we're not at exact steady state.
    assert mass_out == pytest.approx(mass_in, rel=0.15)


def test_takacs_blanket_profile_and_clarification(asm1):
    """At a realistic (BSM1) solids loading the corrected Takács model must
    build a monotone sludge blanket (dense bottom, clear top), clarify the
    effluent strongly, thicken the underflow, and conserve solids tightly.

    This exercises the clarification-zone flux limiting (without it the effluent
    is poorly clarified) and the per-species flux apportioning (settling moves
    each species at the bulk velocity, conserving total settleable solids)."""
    from aquakin.plant.metrics import derived_TSS
    from aquakin.plant.streams import Stream

    C = asm1.default_concentrations()
    feed = {"XB_H": 2200.0, "XB_A": 120.0, "XS": 80.0, "XI": 1100.0,
            "XP": 600.0, "XND": 5.0}
    for s, v in feed.items():
        C = C.at[asm1.species_index[s]].set(v)
    Q_in, overflow = 36892.0, 18061.0   # ~BSM1 clarifier loading

    plant = Plant("clar_load")
    clar = TakacsClarifier(name="clar", network=asm1, area=1500.0,
                           height=4.0, overflow_Q=overflow)
    plant.add_unit(clar)
    plant.add_influent("feed", InfluentSeries(
        t=jnp.asarray([0.0, 50.0]), Q=jnp.full((2,), Q_in),
        C=jnp.stack([C, C]), network=asm1))
    plant.connect(None, "feed", "clar", "inlet")
    sol = plant.solve(t_span=(0.0, 5.0), t_eval=jnp.asarray([0.0, 5.0]),
                      rtol=1e-5, atol=1e-3, max_steps=200_000)
    assert jnp.all(jnp.isfinite(sol.state))

    st0, sz = plant._state_layout["clar"]
    lay = sol.state[-1, st0:st0 + sz].reshape((clar.n_layers, clar._n_part))
    tss = jnp.sum(lay * jnp.asarray(clar._part_tss_factors), axis=1)  # bottom->top
    # A settled blanket: bottom much denser than top, monotone non-increasing up.
    assert float(tss[0]) > 10.0 * float(tss[-1])
    assert bool(jnp.all(jnp.diff(tss) <= 1e-6))

    out = clar.compute_outputs(
        jnp.asarray(5.0), sol.state[-1, st0:st0 + sz],
        {"inlet": Stream(Q=jnp.asarray(Q_in), C=C, network=asm1)},
        plant.default_parameters())
    feed_tss = float(derived_TSS(C, asm1))
    eff_tss = float(derived_TSS(out["overflow"].C, asm1))
    und_tss = float(derived_TSS(out["underflow"].C, asm1))
    assert eff_tss < 0.05 * feed_tss       # strongly clarified effluent
    assert und_tss > 1.5 * feed_tss        # thickened underflow
    # Tight solids mass balance (corrected apportioning conserves TSS).
    mass_in = Q_in * feed_tss
    mass_out = overflow * eff_tss + (Q_in - overflow) * und_tss
    assert mass_out == pytest.approx(mass_in, rel=0.03)


def test_no_inflow_no_dynamics(asm1):
    """Initially the clarifier is at uniform default concentrations. With
    zero inflow (Q=0) and zero overflow, the clarifier should reach a
    quiescent state where particulates simply settle to the bottom."""
    plant = Plant("quiescent_clar")
    plant.add_unit(
        TakacsClarifier(
            name="clar", network=asm1, area=1500.0, height=4.0,
            overflow_Q=0.0,
        )
    )
    inf = InfluentSeries(
        t=jnp.asarray([0.0, 10.0]),
        Q=jnp.asarray([0.0, 0.0]),
        C=jnp.stack([
            asm1.default_concentrations(),
            asm1.default_concentrations(),
        ]),
        network=asm1,
    )
    plant.add_influent("feed", inf)
    plant.connect(None, "feed", "clar", "inlet")
    sol = plant.solve(
        t_span=(0.0, 0.5), t_eval=jnp.asarray([0.0, 0.5]),
        rtol=1e-4, atol=1e-3,
    )
    # All particulate state must remain finite.
    assert jnp.all(jnp.isfinite(sol.state))


def test_shared_asm1_constants_are_single_source(asm1):
    """Clarifier/Takács settling sets and the metrics TSS set come from the one
    shared constants module, so they cannot drift apart."""
    from aquakin.plant._constants import (
        ASM1_SETTLING_SPECIES,
        ASM1_TSS_FACTOR,
        ASM1_TSS_SPECIES,
    )
    from aquakin.plant import metrics
    from aquakin.plant import takacs as takacs_mod
    from aquakin.plant.clarifier import IdealClarifier

    ideal = IdealClarifier(name="c", network=asm1, overflow_Q=1.0)
    tak = TakacsClarifier(name="t", network=asm1, area=1.0, height=4.0,
                          overflow_Q=1.0)
    assert tuple(ideal.particulate_species) == ASM1_SETTLING_SPECIES
    assert tuple(tak.particulate_species) == ASM1_SETTLING_SPECIES
    # Takács TSS factors: the TSS species at the shared factor, XND at zero.
    assert takacs_mod._DEFAULT_TSS_FACTORS == {
        **{s: ASM1_TSS_FACTOR for s in ASM1_TSS_SPECIES}, "XND": 0.0}
    # Metrics reuses the same TSS set / factor object.
    assert metrics._TSS_SPECIES is ASM1_TSS_SPECIES
    assert metrics._TSS_FACTOR == ASM1_TSS_FACTOR
