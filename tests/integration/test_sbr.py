"""Sequencing batch reactor: cyclic phase scheduling, variable volume, settling.

The SBR cycles fill -> react -> settle -> decant -> idle in one tank, with the
volume rising on fill and falling on decant, the biology reacting throughout,
aeration switched per phase, and the settle phase clarifying the decanted
effluent. Phase transitions are located events the plant lands on exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import (
    InterfaceSettling,
    LayeredSettling,
    SBRPhase,
    SBRUnit,
)
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.plant import Plant

_PARTS = ["XI", "XS", "XB_H", "XB_A", "XP", "XND"]


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


def _phases():
    return [
        SBRPhase("fill", 0.10, feed=True),
        SBRPhase("react", 0.30, kla=240.0),
        SBRPhase("settle", 0.10, settle=True),
        SBRPhase("decant", 0.10, decant=True),
        SBRPhase("idle", 0.05),
    ]


def _sbr(asm1, settling=None, **kw):
    settling = settling or InterfaceSettling(v_settle=400.0, area=200.0)
    return SBRUnit("sbr", asm1, _phases(), full_volume=1000.0, feed_flow=5000.0,
                   decant_flow=5000.0, settling=settling,
                   particulate_species=_PARTS, initial_fraction=0.5,
                   conditions={"T": 293.15}, **kw)


def _plant(asm1, sbr):
    p = Plant("sbr_plant")
    p.add_unit(sbr)
    p.add_influent("feed", InfluentSeries.constant(
        asm1, SS=200.0, SNH=40.0, XS=150.0, XB_H=50.0, Q=5000.0), to="sbr.feed")
    return p


# --- construction / validation ---------------------------------------------

def test_state_layout(asm1):
    sbr = _sbr(asm1)
    assert sbr.state_size == asm1.n_species + 1 + 1   # C + V + 1 settling state
    s0 = sbr.initial_state()
    assert s0.shape == (sbr.state_size,)
    assert float(s0[asm1.n_species]) == pytest.approx(500.0)   # initial_fraction*V


def test_spec_validation(asm1):
    with pytest.raises(ValueError, match="at least one phase"):
        SBRUnit("s", asm1, [], 1000.0, 1.0, 1.0,
                InterfaceSettling(1.0, 1.0), conditions={"T": 293.15})
    with pytest.raises(ValueError, match="durations must be > 0"):
        SBRUnit("s", asm1, [SBRPhase("p", 0.0)], 1000.0, 1.0, 1.0,
                InterfaceSettling(1.0, 1.0), conditions={"T": 293.15})
    with pytest.raises(ValueError, match="initial_fraction"):
        SBRUnit("s", asm1, _phases(), 1000.0, 1.0, 1.0,
                InterfaceSettling(1.0, 1.0), particulate_species=_PARTS,
                initial_fraction=1.5, conditions={"T": 293.15})
    with pytest.raises(ValueError, match="missing condition"):
        SBRUnit("s", asm1, _phases(), 1000.0, 1.0, 1.0,
                InterfaceSettling(1.0, 1.0))
    with pytest.raises(ValueError, match="particulate species"):
        SBRUnit("s", asm1, _phases(), 1000.0, 1.0, 1.0,
                InterfaceSettling(1.0, 1.0), particulate_species=["NOPE"],
                conditions={"T": 293.15})


def test_cycle_events_at_phase_boundaries(asm1):
    sbr = _sbr(asm1)                                  # period 0.65
    evs = sbr.cycle_events(0.0, 1.3)
    assert len(evs) == 1
    times = evs[0].at_times
    # phase starts within a cycle: 0.1, 0.4, 0.5, 0.6 (and the next cycle's 0.65).
    assert 0.1 in times and 0.4 in times and 0.5 in times and 0.6 in times
    assert 0.65 in times and 1.15 in times
    # a sub-phase span has no interior boundary
    assert sbr.cycle_events(0.0, 0.05) == []


# --- the cycle: volume + clarification --------------------------------------
#
# These run real event-driven plant solves. Like the other whole-plant solves
# (test_bsm1/test_bsm2_dynamic/test_biofilm) they are marked ``slow`` so they run
# on the merge-only, pytest-split-sharded job that bounds per-process memory; on
# the unsharded ``-n auto`` fast gate their accumulated XLA compilation cache +
# live JAX buffers push the hosted runner into an OOM reclaim. The cheap
# construction / unit-level tests above and below stay on the fast gate.


@pytest.mark.slow
def test_volume_cycles_fill_and_decant(asm1):
    sbr = _sbr(asm1)
    p = _plant(asm1, sbr)
    sol = p.solve(t_span=(0.0, 1.3), t_eval=jnp.linspace(0.005, 1.295, 80))
    V = np.asarray(sol.state[:, asm1.n_species])
    assert np.all(np.isfinite(np.asarray(sol.state)))
    assert V.max() == pytest.approx(1000.0, abs=5.0)   # fills to full
    assert V.min() == pytest.approx(500.0, abs=5.0)    # decants back to the heel


@pytest.mark.slow
def test_event_aligned_save_grid_is_finite(asm1):
    """A save grid that lands exactly on phase-boundary event times must stay
    finite -- the segment-endpoint emission, not a dense edge-evaluation."""
    sbr = _sbr(asm1)
    p = _plant(asm1, sbr)
    sol = p.solve(t_span=(0.0, 1.3), t_eval=jnp.linspace(0.0, 1.3, 131))  # hits 0.1,0.4,...
    assert np.all(np.isfinite(np.asarray(sol.state)))
    assert sol.events_log and len(sol.events_log) >= 8   # phase switches logged


@pytest.mark.slow
def test_decant_is_clarified(asm1):
    """During decant the effluent particulates are far below the bulk (settled);
    solubles pass through unchanged (they do not settle)."""
    sbr = _sbr(asm1)
    p = _plant(asm1, sbr)
    sol = p.solve(t_span=(0.0, 1.3), t_eval=jnp.linspace(0.0, 1.3, 131))
    eff = p.stream(sol, "sbr.effluent")
    q = np.asarray(eff.Q)
    decant = np.where(q > 1e-6)[0]
    assert decant.size > 0
    i = int(decant[decant.size // 2])
    xbh, ss = asm1.species_index["XB_H"], asm1.species_index["SS"]
    assert float(eff.C[i, xbh]) < 0.1 * float(sol.state[i, xbh])    # clarified
    assert float(eff.C[i, ss]) == pytest.approx(float(sol.state[i, ss]), rel=1e-3)


@pytest.mark.slow
def test_layered_settling_solves_finite(asm1):
    sbr = _sbr(asm1, settling=LayeredSettling(n_layers=4, v_settle=400.0, area=200.0))
    assert sbr.state_size == asm1.n_species + 1 + 4
    p = _plant(asm1, sbr)
    sol = p.solve(t_span=(0.0, 1.3), t_eval=jnp.linspace(0.0, 1.3, 131))
    assert np.all(np.isfinite(np.asarray(sol.state)))
    eff = p.stream(sol, "sbr.effluent")
    q = np.asarray(eff.Q)
    i = int(np.where(q > 1e-6)[0][0])
    xbh = asm1.species_index["XB_H"]
    assert float(eff.C[i, xbh]) < float(sol.state[i, xbh])          # clarified


@pytest.mark.slow
def test_grad_through_a_cycle(asm1):
    sbr = _sbr(asm1)
    p = _plant(asm1, sbr)
    base = p.default_parameters()

    def loss(scale):
        sol = p.solve(t_span=(0.0, 0.65), t_eval=jnp.array([0.65]),
                      params=base * scale)
        return jnp.sum(sol.state ** 2)

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)


# --- settling models (unit level) -------------------------------------------

def test_interface_settling_clarifies_particulates_only(asm1):
    m = InterfaceSettling(v_settle=400.0, area=200.0)
    m.bind(asm1, _PARTS)
    C = asm1.default_concentrations()
    # fully clarified (c=1): particulates -> 0, solubles unchanged
    mult = m.decant_multiplier(C, jnp.asarray(1000.0), jnp.asarray([1.0]))
    assert float(mult[asm1.species_index["XB_H"]]) == pytest.approx(0.0)
    assert float(mult[asm1.species_index["SS"]]) == pytest.approx(1.0)
    # mixed (c=0): no clarification
    mult0 = m.decant_multiplier(C, jnp.asarray(1000.0), jnp.asarray([0.0]))
    assert float(mult0[asm1.species_index["XB_H"]]) == pytest.approx(1.0)


def test_interface_settling_grows_while_settling(asm1):
    m = InterfaceSettling(v_settle=400.0, area=200.0)
    m.bind(asm1, _PARTS)
    C = asm1.default_concentrations()
    # settling active -> clarity rises; not settling -> relaxes toward 0
    dc_on = m.extra_rhs(C, jnp.asarray(1000.0), jnp.asarray([0.0]), jnp.asarray(1.0))
    dc_off = m.extra_rhs(C, jnp.asarray(1000.0), jnp.asarray([0.5]), jnp.asarray(0.0))
    assert float(dc_on[0]) > 0.0
    assert float(dc_off[0]) < 0.0


def test_layered_settling_average_ratio_conserved(asm1):
    """Settling redistributes the particulate profile downward but conserves its
    cross-layer average (mass moves between layers, none leaves the profile)."""
    m = LayeredSettling(n_layers=5, v_settle=400.0, area=200.0)
    m.bind(asm1, _PARTS)
    C = asm1.default_concentrations()
    dr = m.extra_rhs(C, jnp.asarray(1000.0), jnp.ones((5,)), jnp.asarray(1.0))
    assert float(jnp.sum(dr)) == pytest.approx(0.0, abs=1e-9)   # mean ratio held
    assert float(dr[0]) < 0.0 and float(dr[-1]) > 0.0           # top down, bottom up
