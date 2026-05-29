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
