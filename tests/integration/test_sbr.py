"""Sequencing batch reactor: cyclic phase scheduling, variable volume, settling.

The SBR cycles fill -> react -> settle -> decant -> idle in one tank, with the
volume rising on fill and falling on decant, the biology reacting throughout,
aeration switched per phase, and the settle phase clarifying the decanted
effluent. Phase transitions are located events the plant lands on exactly.
"""

import dataclasses

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
from aquakin.plant.influent import InfluentExpression, InfluentSeries
from aquakin.plant.plant import Plant

_PARTS = ["XI", "XS", "XB_H", "XB_A", "XP", "XND"]


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


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
                   conditions={"T": 293.15}, feed="batch", **kw)


#: 5000 m3/d during the 0.1 d fill of each 0.65 d cycle, nothing otherwise.
_FILL_ONLY_Q = "5000*(1 - step(mod(t, 0.65) - 0.1))"


def _plant(asm1, sbr):
    p = Plant("sbr_plant", time_unit="d")
    p.add_unit(sbr)
    p.add_influent("feed", InfluentExpression(
        asm1, Q=_FILL_ONLY_Q, composition=dict(SS=200.0, SNH=40.0, XS=150.0, XB_H=50.0)),
        to="sbr.feed")
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
                InterfaceSettling(1.0, 1.0), conditions={"T": 293.15}, feed="batch")
    with pytest.raises(ValueError, match="durations must be > 0"):
        SBRUnit("s", asm1, [SBRPhase("p", 0.0)], 1000.0, 1.0, 1.0,
                InterfaceSettling(1.0, 1.0), conditions={"T": 293.15}, feed="batch")
    with pytest.raises(ValueError, match="initial_fraction"):
        SBRUnit("s", asm1, _phases(), 1000.0, 1.0, 1.0,
                InterfaceSettling(1.0, 1.0), particulate_species=_PARTS,
                initial_fraction=1.5, conditions={"T": 293.15}, feed="batch")
    with pytest.raises(ValueError, match="missing required condition values"):
        SBRUnit("s", asm1, _phases(), 1000.0, 1.0, 1.0,
                InterfaceSettling(1.0, 1.0), feed="batch")
    with pytest.raises(ValueError, match="particulate species"):
        SBRUnit("s", asm1, _phases(), 1000.0, 1.0, 1.0,
                InterfaceSettling(1.0, 1.0), particulate_species=["NOPE"],
                conditions={"T": 293.15}, feed="batch")


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

# The save grids the cycle tests read. All three come from one solve of the same
# plant over (0, 1.3) on their union: the event solver integrates each segment
# with an adaptive step sequence that does not depend on ``t_eval`` and reads
# every save point independently from that segment's dense output (or its exact
# end state), so each grid's rows are the values its own solve would give.
_GRID_VOLUME = jnp.linspace(0.005, 1.295, 80)
_GRID_EVENT_ALIGNED = jnp.linspace(0.0, 1.3, 131)  # hits 0.1, 0.4, ...
# A fine grid with several interior samples inside each decant phase
# ([0.5, 0.6) and [1.15, 1.25)), none relying on a phase boundary.
_GRID_DECANT = jnp.linspace(0.0, 1.3, 261)


@pytest.fixture(scope="module")
def cycle_run(asm1):
    """The default SBR plant solved once over two cycles on the union of the
    cycle tests' save grids.

    Returns a function ``on(grid) -> (plant, sol)`` giving the plant and the
    solution restricted to the rows of ``grid``.
    """
    p = _plant(asm1, _sbr(asm1))
    grids = [np.asarray(g) for g in (_GRID_VOLUME, _GRID_EVENT_ALIGNED, _GRID_DECANT)]
    t_all = np.unique(np.concatenate(grids))
    sol = p.solve(t_span=(0.0, 1.3), t_eval=jnp.asarray(t_all))

    def on(grid):
        grid = np.asarray(grid)
        idx = np.searchsorted(t_all, grid)
        assert np.array_equal(t_all[idx], grid)   # every point is in the union
        return p, dataclasses.replace(sol, t=sol.t[idx], state=sol.state[idx])

    return on


@pytest.mark.slow
def test_volume_cycles_fill_and_decant(asm1, cycle_run):
    _, sol = cycle_run(_GRID_VOLUME)
    V = np.asarray(sol.state[:, asm1.n_species])
    assert np.all(np.isfinite(np.asarray(sol.state)))
    assert V.max() == pytest.approx(1000.0, abs=5.0)   # fills to full
    assert V.min() == pytest.approx(500.0, abs=5.0)    # decants back to the heel


@pytest.mark.slow
def test_event_aligned_save_grid_follows_the_volume_closed_form(asm1, cycle_run):
    """A save grid that lands exactly on phase-boundary event times reads the
    segment-endpoint emission, not a dense edge-evaluation: every row is finite
    and the volume equals its piecewise-linear closed form there too."""
    p, sol = cycle_run(_GRID_EVENT_ALIGNED)
    assert np.all(np.isfinite(np.asarray(sol.state)))
    V = np.asarray(sol.state[:, asm1.n_species])
    # dV/dt is piecewise constant between located events; the implicit solve
    # (default rtol 1e-6, coupled to the stiff biology) holds V to ~1e-5 of
    # it, so the check is two decades above the solve's rtol. A 1% error in
    # the fill or decant flow moves V by 2e-2.
    np.testing.assert_allclose(V, _volume_closed_form(p.units["sbr"], _GRID_EVENT_ALIGNED),
                               rtol=1e-4)
    assert sol.events_log and len(sol.events_log) >= 8   # phase switches logged


@pytest.mark.slow
def test_decant_is_clarified(asm1, cycle_run):
    """*Every* decant draw -- not just the phase boundary -- is clarified: the
    particulates stay far below the bulk throughout the decant, and solubles pass
    through unchanged. Sampling the whole decant span (not one grid-lucky point)
    is what pins that clarity is held across the draw, rather than washed out by
    re-mixing once the quiescent decant begins."""
    p, sol = cycle_run(_GRID_DECANT)
    eff = p.stream(sol, "sbr.effluent")
    q = np.asarray(eff.Q)
    decant = np.where(q > 1e-6)[0]
    assert decant.size >= 6                                          # multiple draws
    xbh, ss = asm1.species_index["XB_H"], asm1.species_index["SS"]
    for i in decant:
        i = int(i)
        # particulates clarified at every decant instant (held across the draw)
        assert float(eff.C[i, xbh]) < 0.1 * float(sol.state[i, xbh])
        # solubles unchanged (they do not settle)
        assert float(eff.C[i, ss]) == pytest.approx(
            float(sol.state[i, ss]), rel=1e-3)


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


# --- a small SBR in the fast tier ---------------------------------------------
#
# The cycle tests above solve ASM1 and are ``slow``. This SBR runs the same
# event-driven fill / react / settle / decant / idle cycle on the two-species
# A -> B decay model (in seconds, B taken as the settling particulate), which
# compiles and solves in seconds, so the fast gate exercises the SBR solve and
# its gradient too (the plant is `make_small_sbr_plant`, in `tests/conftest.py`).

def _volume_closed_form(sbr, t):
    """The SBR's liquid volume from its schedule alone: the initial heel, rising
    at ``feed_flow`` through each fill phase and falling at ``decant_flow``
    through each decant phase, constant otherwise (piecewise linear in t)."""
    t = np.asarray(t, dtype=float)
    durations = np.array([p.duration for p in sbr.phases])
    starts = np.concatenate([[0.0], np.cumsum(durations)])
    period = starts[-1]
    n_cycles, tau = np.divmod(t, period)
    V = np.full_like(t, sbr.initial_fraction * sbr.full_volume)
    for k, phase in enumerate(sbr.phases):
        rate = sbr.feed_flow * phase.feed - sbr.decant_flow * phase.decant
        in_phase = np.clip(tau - starts[k], 0.0, durations[k])
        V += rate * (n_cycles * durations[k] + in_phase)
    return V


def test_small_sbr_volume_follows_fill_and_decant(sbr_model, make_small_sbr_plant):
    """Over one cycle the small SBR's volume equals its piecewise-linear closed
    form at every save point, the phase boundaries included; the decant is
    clarified (B, the particulate, well below the bulk) while A passes through."""
    p = make_small_sbr_plant()
    t = np.linspace(0.0, 27.0, 55)                # 0.5 s grid, on every boundary
    sol = p.solve(t_span=(0.0, 27.0), t_eval=jnp.asarray(t), rtol=1e-8)
    V = np.asarray(sol.state[:, sbr_model.n_species])
    # dV/dt is piecewise constant between located events; the implicit solve
    # reproduces the closed form to within its tolerance (rtol 1e-8).
    np.testing.assert_allclose(V, _volume_closed_form(p.units["sbr"], t), rtol=1e-6)
    assert V.max() == pytest.approx(p.units["sbr"].full_volume) and V[-1] == pytest.approx(50.0)

    eff = p.stream(sol, "sbr.effluent")
    decant = np.where(np.asarray(eff.Q) > 0.0)[0]
    assert decant.size >= 8                       # the decant phase is sampled
    a, b = sbr_model.species_index["A"], sbr_model.species_index["B"]
    C = np.asarray(sol.state)
    C_eff = np.asarray(eff.C)
    assert np.all(C_eff[decant, b] < 0.5 * C[decant, b])
    np.testing.assert_allclose(C_eff[decant, a], C[decant, a], rtol=1e-12)


def test_small_sbr_grad_through_settle_and_decant_matches_finite_difference(sbr_model, make_small_sbr_plant):
    """d(A + 2B at the end of the decant)/dk from a full tank through the settle
    and decant phases (two located-event segments, the settling state and the
    clarified draw) matches a central finite difference."""
    p = make_small_sbr_plant()
    base = p.default_parameters()
    a, b = sbr_model.species_index["A"], sbr_model.species_index["B"]
    full = p.units["sbr"].initial_state().at[sbr_model.n_species].set(p.units["sbr"].full_volume)
    y0 = p.initial_state(overrides={"sbr": full})

    def loss(k):
        sol = p.solve(t_span=(15.0, 25.0), t_eval=jnp.array([25.0]),
                      params=base.at[0].set(k), y0=y0, rtol=1e-10, atol=1e-12)
        return sol.state[-1, a] + 2.0 * sol.state[-1, b]

    k0 = float(base[0])
    h = 1e-4 * k0
    g = float(jax.grad(loss)(k0))
    fd = (float(loss(k0 + h)) - float(loss(k0 - h))) / (2.0 * h)
    # Central difference: O(h^2) ~ 1e-8 relative truncation, plus the solver's
    # 1e-10 tolerance over a 1e-4 step, ~1e-6.
    assert g == pytest.approx(fd, rel=1e-5)
    assert g != 0.0


def test_small_sbr_solve_traces_under_jit_and_matches_eager(sbr_model, make_small_sbr_plant):
    """A whole-cycle SBR solve wrapped in ``jax.jit`` -- the default ``y0``,
    traced parameters and a save grid built inside the trace -- traces and gives
    the eager solve's trajectory (one computation compiled whole instead of
    segment by segment, so they agree to rounding)."""
    p = make_small_sbr_plant()
    base = p.default_parameters()

    def trajectory(params):
        return p.solve(t_span=(0.0, 27.0), t_eval=jnp.linspace(0.0, 27.0, 55),
                       params=params).state

    eager = trajectory(base)
    np.testing.assert_allclose(np.asarray(jax.jit(trajectory)(base)), np.asarray(eager),
                               rtol=1e-10, atol=1e-12)
    # The jitted cycle is the SBR cycle: the volume follows its closed form
    # (an adaptive solve at the default rtol = 1e-6: two decades above it).
    np.testing.assert_allclose(np.asarray(eager[:, sbr_model.n_species]),
                               _volume_closed_form(p.units["sbr"], np.linspace(0.0, 27.0, 55)),
                               rtol=1e-4)


def test_small_sbr_jitted_gradient_matches_finite_difference(sbr_model, make_small_sbr_plant):
    """``jax.jit(jax.grad(...))`` of the settle + decant loss of the
    finite-difference test above (a save grid built inside the trace) matches a
    central finite difference of the eager loss. (About 15 s, like that test:
    the cost is compiling the reverse pass through the stiff solve, and this is
    the fast tier's only check that a jitted SBR gradient traces.)"""
    p = make_small_sbr_plant()
    base = p.default_parameters()
    a, b = sbr_model.species_index["A"], sbr_model.species_index["B"]
    full = p.units["sbr"].initial_state().at[sbr_model.n_species].set(p.units["sbr"].full_volume)
    y0 = p.initial_state(overrides={"sbr": full})

    def loss(k):
        sol = p.solve(t_span=(15.0, 25.0), t_eval=jnp.array([25.0]),
                      params=base.at[0].set(k), y0=y0, rtol=1e-10, atol=1e-12)
        return sol.state[-1, a] + 2.0 * sol.state[-1, b]

    k0 = float(base[0])
    h = 1e-4 * k0
    fd = (float(loss(k0 + h)) - float(loss(k0 - h))) / (2.0 * h)
    # As in the finite-difference test above: O(h^2) truncation plus the
    # solver's 1e-10 tolerance over a 1e-4 step, ~1e-6.
    assert float(jax.jit(jax.grad(loss))(k0)) == pytest.approx(fd, rel=1e-5)


# --- the feed modes (#857) ------------------------------------------------------
#
# The small SBR (tests/conftest.py) in each mode: a fill-only influent for
# batch, the same 50 m3 per 27 s cycle arriving continuously for continuous and
# storage (a 60 m3 holding volume, half full, drawn at 10 m3/s during the 5 s
# fill).

def _fill_time(t, start, duration, period):
    """Time spent inside the phase ``[start, start + duration)`` of each cycle
    up to each ``t`` (closed form)."""
    n_cycles, tau = np.divmod(np.asarray(t, dtype=float), period)
    return n_cycles * duration + np.clip(tau - start, 0.0, duration)


def test_omitting_the_feed_mode_raises_and_lists_the_modes(sbr_model):
    """An SBR has no default feed mode: leaving it out, or naming another, is
    refused with the three modes listed; following the message (naming one)
    builds the unit."""
    args = ("s", sbr_model, [SBRPhase("fill", 1.0, feed=True)], 100.0, 1.0, 1.0,
            InterfaceSettling(1.0, 1.0))
    kw = {"conditions": {"T": 293.15}}
    for feed in (None, "pump"):
        with pytest.raises(ValueError, match=r"'continuous' \(.*\); 'storage' \(.*\); "
                                             r"'batch' \(.*\)") as err:
            SBRUnit(*args, **kw, feed=feed)
        assert ("needs a feed mode" if feed is None else "is not a feed mode") in str(err.value)
    assert SBRUnit(*args, **kw, feed="batch").feed_flow == 1.0


def test_a_feed_mode_refuses_the_settings_it_does_not_use(sbr_model):
    """feed_flow does not apply to a continuous feed, a storage or batch feed
    needs it, and only a storage feed takes the holding-volume settings."""
    phases = [SBRPhase("fill", 1.0, feed=True)]

    def build(feed, feed_flow, **kw):
        return SBRUnit("s", sbr_model, phases, 100.0, feed_flow, 1.0,
                       InterfaceSettling(1.0, 1.0), conditions={"T": 293.15}, feed=feed, **kw)

    with pytest.raises(ValueError, match="feed_flow does not apply"):
        build("continuous", 1.0)
    with pytest.raises(ValueError, match="feed_flow must be a positive number; got None"):
        build("batch", None)
    with pytest.raises(ValueError, match="storage_volume must be a positive number; got None"):
        build("storage", 1.0)
    with pytest.raises(ValueError, match="storage_initial_fraction must be in"):
        build("storage", 1.0, storage_volume=10.0, storage_initial_fraction=0.0)
    with pytest.raises(ValueError, match="so storage_volume, storage_initial_fraction does not"):
        build("batch", 1.0, storage_volume=10.0, storage_initial_fraction=0.2)
    with pytest.raises(ValueError, match="so storage_initial_concentrations does not apply"):
        build("continuous", None, storage_initial_concentrations=jnp.ones(2))
    assert build("continuous", None).output_ports == ["effluent"]
    storage = build("storage", 1.0, storage_volume=10.0)
    assert storage.output_ports == ["effluent", "overflow"]
    assert storage.state_names()[-3:] == ("storage.A", "storage.B", "storage.V")
    np.testing.assert_array_equal(np.asarray(storage.initial_state()[-3:]), [1.0, 0.0, 5.0])


@pytest.fixture(scope="module")
def feed_mode_runs(make_small_sbr_plant):
    """The small SBR solved over two cycles in each feed mode, on a 0.05 s grid:
    ``{feed: (plant, solution)}``."""
    t = jnp.linspace(0.0, 54.0, 1081)
    runs = {}
    for feed in ("batch", "continuous", "storage"):
        p = make_small_sbr_plant(feed)
        runs[feed] = p, p.solve(t_span=(0.0, 54.0), t_eval=t, rtol=1e-8, atol=1e-10)
    return runs


@pytest.mark.parametrize("feed", ["batch", "continuous", "storage"])
def test_each_feed_mode_conserves_mass(feed_mode_runs, feed):
    """What the influent brings in leaves through the decant (and overflow) or
    is held in the tank and the holding volume: the COD balance closes. The
    reaction conserves COD, so no gas term enters. The flow integrals are
    trapezoids over the 0.05 s saves, across steps at the phase boundaries, so
    the closure is set by the sampling (1e-3 to 2e-3 here, ten times smaller on
    a ten times finer grid); before the feed modes the discarded inflow left an
    imbalance of order one."""
    p, sol = feed_mode_runs[feed]
    mb = p.mass_balance(sol, components=("COD",))
    assert mb["COD"].inflow == pytest.approx(100.0, rel=1e-9)   # 50 m3 of A = 1 per cycle
    assert abs(mb["COD"].relative_imbalance) < 3e-3


def test_a_continuous_feed_fills_the_tank_with_the_inflow_minus_the_decant(
    sbr_model, feed_mode_runs
):
    """With feed='continuous' the volume follows V0 + Q_in t - decant_flow x
    (time spent decanting), whatever the phase (the fill pump does not apply)."""
    p, sol = feed_mode_runs["continuous"]
    t = np.asarray(sol.t)
    V = np.asarray(sol.state[:, sbr_model.n_species])
    expected = 50.0 + (50.0 / 27.0) * t - 10.0 * _fill_time(t, 20.0, 5.0, 27.0)
    # dV/dt is piecewise constant between located events: the solve (rtol 1e-8)
    # holds it to its tolerance.
    np.testing.assert_allclose(V, expected, rtol=1e-6)


def test_a_holding_volume_level_follows_the_inflow_minus_the_draw(make_small_sbr_plant):
    """Started at 90 % (54 m3), the holding volume neither runs empty nor
    overflows: its level is 54 + Q_in t - feed_flow x (time spent filling), and
    the report says nothing overflowed and nothing was short."""
    from aquakin.plant import storage_report

    p = make_small_sbr_plant("storage", storage_initial_fraction=0.9)
    t = np.linspace(0.0, 54.0, 217)
    sol = p.solve(t_span=(0.0, 54.0), t_eval=jnp.asarray(t), rtol=1e-8, atol=1e-10)
    report = storage_report(p, sol, "sbr")
    expected = 54.0 + (50.0 / 27.0) * t - 10.0 * _fill_time(t, 0.0, 5.0, 27.0)
    np.testing.assert_allclose(report.level, expected, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(sol.unit_state("sbr")[:, -1]), report.level)
    assert report.overflow_volume == 0.0 and report.shortfall_volume == 0.0
    assert report.capacity == 60.0


def test_a_full_holding_volume_overflows_through_its_port(sbr_model, make_small_sbr_plant):
    """At 4 m3/s (108 m3 per cycle against a 50 m3 draw) the holding volume
    fills: its level never exceeds its capacity, the surplus leaves through the
    overflow port, and the mass balance closes with it as an outflow. The
    reported overflow volume equals what the volumes alone say overflowed: the
    inflow, less what the tank drew (its volume change plus the decant), less
    the holding volume's change."""
    from aquakin.plant import storage_report

    p = make_small_sbr_plant("storage", Q=4.0, storage_initial_fraction=0.9)
    t = np.linspace(0.0, 54.0, 1081)
    sol = p.solve(t_span=(0.0, 54.0), t_eval=jnp.asarray(t), rtol=1e-8, atol=1e-10)
    report = storage_report(p, sol, "sbr")
    # Full, to the solve's tolerance (rtol 1e-8): the overflow takes all the
    # surplus at the top of its band.
    assert report.level.max() == pytest.approx(60.0, rel=1e-6)
    overflow = p.stream(sol, "sbr.overflow")
    np.testing.assert_array_equal(np.asarray(overflow.Q), report.overflow)
    V = np.asarray(sol.unit_state("sbr")[:, sbr_model.n_species])
    drawn = V[-1] - V[0] + 10.0 * 10.0                    # two 5 s decants at 10 m3/s
    expected = 4.0 * 54.0 - drawn - (report.level[-1] - report.level[0])
    # The report integrates the overflow by trapezoids over the 0.05 s saves, and
    # the overflow steps by 4 m3/s where a fill starts and ends while full: each
    # step costs up to 0.1 m3 of the 110 m3, so 5e-3.
    assert report.overflow_volume == pytest.approx(expected, rel=5e-3)
    assert expected > 50.0
    mb = p.mass_balance(sol, components=("COD",))
    assert abs(mb["COD"].relative_imbalance) < 3e-3   # sampling, as in the test above
    assert "overflowed: 1" in report.summary()


def test_an_empty_holding_volume_limits_the_draw_and_reports_it(
    sbr_model, make_small_sbr_plant
):
    """Half full (30 m3) and fed 9.3 m3 during the 50 m3 fill, the holding
    volume runs empty: it stops at its pump's dead volume (1 % of 60 m3), the
    draw falls to the inflow, the tank fills short of full, and the shortfall
    reported over the cycle is what the tank did not receive."""
    from aquakin.plant import storage_report

    p = make_small_sbr_plant("storage")
    t = np.linspace(0.0, 27.0, 2701)
    sol = p.solve(t_span=(0.0, 27.0), t_eval=jnp.asarray(t), rtol=1e-8, atol=1e-10)
    report = storage_report(p, sol, "sbr")
    assert report.level.min() == pytest.approx(0.6, rel=1e-6)
    # By the end of the fill the draw has fallen to the inflow (the level decays
    # onto the dead volume at (10 - 1.85)/0.6 per second).
    end_of_fill = np.searchsorted(t, 4.99)
    assert report.draw[end_of_fill] == pytest.approx(50.0 / 27.0, rel=1e-6)
    assert report.shortfall[end_of_fill] == pytest.approx(10.0 - 50.0 / 27.0, rel=1e-6)
    V = np.asarray(sol.unit_state("sbr")[:, sbr_model.n_species])
    received = V[np.searchsorted(t, 5.0)] - 50.0
    assert received < 40.0
    # The shortfall jumps to zero when the fill ends, which the trapezoids over
    # the 0.01 s saves smear by about a half step (0.4 %).
    assert report.shortfall_volume == pytest.approx(50.0 - received, rel=1e-2)
    assert "ran empty: 11" in report.summary()


def test_the_storage_report_refuses_an_sbr_without_a_holding_volume(make_small_sbr_plant):
    from aquakin.plant import storage_report

    p = make_small_sbr_plant("batch")
    with pytest.raises(ValueError, match="'sbr' is not an SBRUnit with feed='storage'"):
        storage_report(p, None, "sbr")


def test_grad_through_a_storage_fed_cycle_matches_finite_difference(
    sbr_model, make_small_sbr_plant
):
    """d(A + 2B at the end of a cycle)/dk through a storage-fed cycle -- the draw
    from the holding volume, its running empty, and the settle and decant --
    matches a central finite difference. (About 15 s, like the other small-SBR
    gradient tests: compiling the reverse pass through the stiff solve; it is
    the fast tier's only gradient through the holding volume.)"""
    p = make_small_sbr_plant("storage")
    base = p.default_parameters()
    a, b = sbr_model.species_index["A"], sbr_model.species_index["B"]

    def loss(k):
        sol = p.solve(t_span=(0.0, 27.0), t_eval=jnp.array([27.0]),
                      params=base.at[0].set(k), rtol=1e-10, atol=1e-12)
        return sol.state[-1, a] + 2.0 * sol.state[-1, b]

    k0 = float(base[0])
    h = 1e-4 * k0
    g = float(jax.grad(loss)(k0))
    fd = (float(loss(k0 + h)) - float(loss(k0 - h))) / (2.0 * h)
    # Central difference: O(h^2) ~ 1e-8 relative truncation, plus the solver's
    # 1e-10 tolerance over a 1e-4 step, ~1e-6.
    assert g == pytest.approx(fd, rel=1e-5)
    assert abs(g) > 1e-3


def _step_series(model, fill_Q):
    """A sampled fill-only feed over two cycles: ``fill_Q`` during each 5 s
    fill, stepping within 1e-9 s at the boundaries, and a spike of 1 m3/s at
    t = 11 s (inside a react phase) when ``fill_Q`` is negative."""
    spike = fill_Q < 0
    fill_Q = abs(fill_Q)
    eps = 1e-9
    t = [0.0, 5.0, 5.0 + eps, 27.0, 27.0 + eps, 32.0, 32.0 + eps, 54.0]
    Q = [fill_Q, fill_Q, 0.0, 0.0, fill_Q, fill_Q, 0.0, 0.0]
    if spike:
        t[3:3] = [10.5, 11.0, 11.5]
        Q[3:3] = [0.0, 1.0, 0.0]
    C = np.tile([1.0, 0.0], (len(t), 1))
    return InfluentSeries(t=jnp.asarray(t), Q=jnp.asarray(Q), C=jnp.asarray(C), model=model)


def test_a_batch_feed_is_checked_against_a_sampled_influent(sbr_model, make_small_sbr_plant):
    """A sampled feed that steps between 10 m3/s in the fill and nothing else
    passes; one that delivers 9 m3/s, or flows at one sample time inside a
    react phase, is refused, naming the time and the flow."""
    sbr = make_small_sbr_plant("batch").units["sbr"]
    sbr.check_inlet_flows({"feed": _step_series(sbr_model, 10.0)}, 0.0, 54.0)
    with pytest.raises(ValueError, match=r"delivers Q=9 at t=5e-06, in a fill phase, where "
                                         r"the schedule wants 10"):
        sbr.check_inlet_flows({"feed": _step_series(sbr_model, 9.0)}, 0.0, 54.0)
    with pytest.raises(ValueError, match=r"delivers Q=1 at t=11, in a phase that is not a fill"):
        sbr.check_inlet_flows({"feed": _step_series(sbr_model, -10.0)}, 0.0, 54.0)


def test_a_batch_feed_refuses_a_continuous_influent_and_its_advice_runs(
    sbr_model, make_small_sbr_plant
):
    """A continuous influent into a batch-fed SBR is refused before the solve;
    following the message (feed='continuous') solves, with the tank's volume
    after one cycle what the inflow less the decant says."""
    p = make_small_sbr_plant("batch", Q=50.0 / 27.0)
    with pytest.raises(ValueError, match=r"has feed='batch', but its inlet delivers Q=1\.85185 "
                                         r"at t=5e-06, in a fill phase.*use feed='continuous'"):
        p.solve(t_span=(0.0, 27.0))
    p = make_small_sbr_plant("continuous", Q=50.0 / 27.0)
    sol = p.solve(t_span=(0.0, 27.0), t_eval=jnp.asarray([27.0]), rtol=1e-8)
    assert float(sol.state[-1, sbr_model.n_species]) == pytest.approx(50.0, rel=1e-6)


def test_a_batch_feed_from_another_unit_is_refused(sbr_model, make_small_sbr_plant):
    """Fed by another unit, a batch SBR's inlet flow is not known before the
    solve, so it cannot be checked and the solve is refused."""
    from aquakin.plant import MixerUnit, WiringError
    from aquakin.plant.influent import InfluentExpression

    p = Plant("fed", time_unit="s")
    p.add_unit(make_small_sbr_plant("batch").units["sbr"])
    p.add_unit(MixerUnit("mix", ["in"], sbr_model))
    p.add_influent("feed", InfluentExpression(sbr_model, Q=1.0, composition={"A": 1.0}),
                   to="mix.in")
    p.connect("mix.out", "sbr.feed")
    with pytest.raises(WiringError, match=r"needs an influent on 'feed'.*another unit feeds it"):
        p.solve(t_span=(0.0, 27.0))


def test_mass_balance_reads_sbr_inventory(asm1):
    """Regression for #348: mass_balance must read the SBR's [C, V, settling]
    inventory (volume at index n_species, settling state massless) rather than
    dropping the unit or misreading a state entry as the volume."""
    from aquakin import canonical_content
    from aquakin.plant.balance import _unit_inventory
    sbr = _sbr(asm1)
    p = _plant(asm1, sbr)
    p._build_state_layout()
    p._build_parameter_layout()
    params = p.default_parameters()
    n = asm1.n_species
    C = asm1.default_concentrations()
    state = jnp.concatenate([C, jnp.array([700.0]), jnp.array([0.9])])  # V=700, clarity 0.9
    content = {q: canonical_content(asm1, q, electron_acceptor_cod=False, params=params)
               for q in ("COD", "N", "P")}
    inv = _unit_inventory(p, "sbr", state, {asm1.name: content}, params)
    for q in ("COD", "N", "P"):
        expected = 700.0 * float(np.dot(np.asarray(C), np.asarray(content[q])))
        assert inv[q] == pytest.approx(expected, rel=1e-9, abs=0)
    assert inv["COD"] > 0.0                                   # unit is not dropped
    # the dimensionless settling state carries no mass
    inv2 = _unit_inventory(p, "sbr", state.at[n + 1].set(0.1),
                           {asm1.name: content}, params)
    assert inv2["COD"] == pytest.approx(inv["COD"], rel=1e-12)


def test_mass_balance_closes_over_aerobic_anoxic_decant_cycles(asm1):
    """Regression for #852: plant.mass_balance must integrate an SBR's reaction
    and aeration terms, so its COD and N balances close.

    A full tank is never fed; each 0.4 d cycle is aerobic, anoxic, settle and
    decant, so the oxygen transferred in the aerobic phase (COD), the nitrate
    denitrified to N₂ in the anoxic phase (N) and the decanted effluent are the
    only exits, and the volume falls 1000 -> 750 -> 500 m³ over the two
    cycles. Before the fix the SBR took no part in the gas term, so the gas was
    0 and the relative imbalance was 7.4 (COD) and 3.1 (N).

    The gas integrals are trapezoids over the saves, so the closure is set by
    the sampling: on this 321-point grid it is 5.1e-3 (COD) and 2.1e-3 (N),
    against 5.0e-3 and 1.1e-2 for a constantly aerated CSTR of the same
    kinetics and start, and both fall below 3e-4 on 1601 points.
    """
    idx = asm1.species_index
    C0 = np.asarray(asm1.default_concentrations()).copy()
    for sp, v in dict(SS=50.0, SNH=30.0, SNO=20.0, XS=100.0, XB_H=1000.0,
                      XB_A=100.0, SO=2.0, SALK=7.0).items():
        C0[idx[sp]] = v
    phases = [SBRPhase("aerobic", 0.15, kla=120.0),
              SBRPhase("anoxic", 0.15, mixed=True),
              SBRPhase("settle", 0.05, settle=True),
              SBRPhase("decant", 0.05, decant=True)]
    sbr = SBRUnit("sbr", asm1, phases, full_volume=1000.0, feed_flow=None,
                  decant_flow=5000.0,
                  settling=InterfaceSettling(v_settle=400.0, area=200.0),
                  particulate_species=_PARTS, initial_fraction=1.0,
                  conditions={"T": 293.15}, initial_concentrations=jnp.asarray(C0),
                  feed="continuous")
    p = Plant("sbr_batch", time_unit="d")
    p.add_unit(sbr)
    p.add_influent("feed", InfluentSeries.constant(asm1, Q=0.0), to="sbr.feed")
    sol = p.solve(t_span=(0.0, 0.8), t_eval=jnp.linspace(0.0, 0.8, 321))
    V = np.asarray(sol.state[:, asm1.n_species])
    assert V[-1] == pytest.approx(500.0, rel=1e-4)   # the volume did vary

    mb = p.mass_balance(sol)
    assert mb["COD"].inflow == 0.0 and mb["N"].inflow == 0.0
    assert mb.gas_detail["aeration_O2"] > 0.0
    assert mb.gas_detail["denitrification_N2"] > 0.0
    # Tolerance from the sampling (see the docstring): the CSTR reference on the
    # same grid closes to 1.1e-2 in N.
    assert abs(mb["COD"].relative_imbalance) < 1e-2
    assert abs(mb["N"].relative_imbalance) < 1e-2


def test_oxygen_transfer_follows_the_phase_rhs_is_in(asm1):
    """The SBR's mass-balance aeration term is the rhs aeration term: kLa of the
    active phase times (do_sat - SO), and zero in an unaerated phase."""
    sbr = _sbr(asm1)                                  # react (0.1-0.4) aerated
    state = sbr.initial_state()
    o2 = asm1.species_index["SO"]
    so = float(state[o2])
    oxygen, stripped = sbr.oxygen_transfer(0.2, state, {}, None, None)
    assert float(oxygen[o2]) == pytest.approx(240.0 * (sbr.do_sat - so), rel=1e-12)
    assert float(jnp.sum(jnp.abs(oxygen))) == pytest.approx(abs(float(oxygen[o2])), rel=1e-12)
    assert float(jnp.max(jnp.abs(stripped))) == 0.0
    # The fill phase (0-0.1) and the next cycle's settle (0.65+0.45) are unaerated.
    assert float(jnp.max(jnp.abs(sbr.oxygen_transfer(0.05, state, {}, None, None)[0]))) == 0.0
    assert float(jnp.max(jnp.abs(sbr.oxygen_transfer(1.1, state, {}, None, None)[0]))) == 0.0
    assert sbr.required_signals == ()
    # The reaction volume is the state's liquid volume, not the full volume.
    assert np.all(sbr.reaction_volumes(state, asm1.default_parameters()) == 500.0)


@pytest.mark.slow
def test_grad_through_a_cycle_matches_finite_difference(asm1):
    """d(sum of squared end-of-cycle states)/d(parameter scale) through one ASM1
    SBR cycle matches a central finite difference."""
    sbr = _sbr(asm1)
    p = _plant(asm1, sbr)
    base = p.default_parameters()

    def loss(scale):
        sol = p.solve(t_span=(0.0, 0.65), t_eval=jnp.array([0.65]),
                      params=base * scale, rtol=1e-10, atol=1e-8)
        return jnp.sum(sol.state ** 2)

    g = float(jax.grad(loss)(1.0))
    h = 1e-3
    fd = (float(loss(1.0 + h)) - float(loss(1.0 - h))) / (2.0 * h)
    assert np.isfinite(g)
    # The tight tolerances keep the solve's error out of the difference
    # quotient: at these the central difference agrees with itself to ~2e-6
    # for h from 1e-4 to 1e-3, while at the default tolerances (the
    # auto-scaled atol) it moves by ~1e-3.
    assert g == pytest.approx(fd, rel=1e-5)


# --- settling models (unit level) -------------------------------------------

def test_interface_settling_clarifies_particulates_only(asm1):
    m = InterfaceSettling(v_settle=400.0, area=200.0)
    m.bind(asm1, _PARTS)
    C = asm1.default_concentrations()
    # fully clarified (c=1): particulates -> 0, solubles unchanged
    mult = m.decant_multiplier(C, jnp.asarray(1000.0), jnp.asarray([1.0]))
    assert float(mult[asm1.species_index["XB_H"]]) == pytest.approx(0.0, abs=0)
    assert float(mult[asm1.species_index["SS"]]) == pytest.approx(1.0)
    # mixed (c=0): no clarification
    mult0 = m.decant_multiplier(C, jnp.asarray(1000.0), jnp.asarray([0.0]))
    assert float(mult0[asm1.species_index["XB_H"]]) == pytest.approx(1.0)


def test_interface_settling_grows_settles_mixes_and_holds(asm1):
    m = InterfaceSettling(v_settle=400.0, area=200.0)
    m.bind(asm1, _PARTS)
    C = asm1.default_concentrations()
    V = jnp.asarray(1000.0)
    settling, mixing, quiescent = jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(0.0)
    # settling active (not mixed) -> clarity rises
    dc_settle = m.extra_rhs(C, V, jnp.asarray([0.0]), settling, quiescent)
    # actively mixed (not settling) -> clarity relaxes toward 0
    dc_mix = m.extra_rhs(C, V, jnp.asarray([0.5]), quiescent, mixing)
    # quiescent (neither: decant/idle) -> clarity held, so the decant stays clear
    dc_hold = m.extra_rhs(C, V, jnp.asarray([0.5]), quiescent, quiescent)
    assert float(dc_settle[0]) > 0.0
    assert float(dc_mix[0]) < 0.0
    assert float(dc_hold[0]) == pytest.approx(0.0, abs=0)


def test_layered_settling_average_ratio_conserved(asm1):
    """Settling redistributes the particulate profile downward but conserves its
    cross-layer average (mass moves between layers, none leaves the profile)."""
    m = LayeredSettling(n_layers=5, v_settle=400.0, area=200.0)
    m.bind(asm1, _PARTS)
    C = asm1.default_concentrations()
    dr = m.extra_rhs(C, jnp.asarray(1000.0), jnp.ones((5,)),
                     jnp.asarray(1.0), jnp.asarray(0.0))   # settling, not mixed
    assert float(jnp.sum(dr)) == pytest.approx(0.0, abs=1e-9)   # mean ratio held
    assert float(dr[0]) < 0.0 and float(dr[-1]) > 0.0           # top down, bottom up
    # a quiescent (decant) call holds the profile -- no redistribution, no remix
    dr_hold = m.extra_rhs(C, jnp.asarray(1000.0),
                          jnp.asarray([0.2, 0.6, 1.0, 1.4, 1.8]),
                          jnp.asarray(0.0), jnp.asarray(0.0))
    assert float(jnp.max(jnp.abs(dr_hold))) == pytest.approx(0.0, abs=0)
