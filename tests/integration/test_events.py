"""Located event / discontinuity handling on the reactor and plant solves.

The driver is exercised on the first-order-decay fixture (``A -> B`` with rate
``k*[A]``, analytic ``A(t) = A0 e^{-kt}``), so every reset has a closed-form
expectation. One slow plant test checks the same mechanics on BSM1.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import BatchReactor, Event, SpatialConditions, solve_with_events


@pytest.fixture(scope="module")
def decay():
    net = aquakin.load_network_from_file("tests/fixtures/simple_network.yaml")
    return net


@pytest.fixture(scope="module")
def reactor(decay):
    return BatchReactor(decay, SpatialConditions.uniform(T=293.15))


def _A0(net):
    return net.concentrations({"A": 1.0})


# --- the generic driver (no reactor) -----------------------------------------

def _const_rhs(rate):
    return lambda t, y, args: rate * jnp.ones_like(y)


def test_time_event_resets_at_known_time():
    teval = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    ev = Event(at_times=[0.5], apply=lambda t, y, a: y + 10.0, name="bump")
    res = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                            t0=0.0, t1=1.0, t_eval=teval, events=[ev],
                            rtol=1e-9, atol=1e-11)
    # dy/dt = 1: pre-reset 0.5 at the boundary, then +10 -> 10.75, 11.0.
    np.testing.assert_allclose(np.asarray(res.ys[:, 0]),
                               [0.0, 0.25, 0.5, 10.75, 11.0], atol=1e-6)
    assert res.log == [(0.5, "bump")]


def test_state_event_locates_crossing():
    teval = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    ev = Event(cond_fn=lambda t, y, a: y[0] - 0.5, direction=1,
               apply=lambda t, y, a: y + 10.0, name="cross")
    res = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                            t0=0.0, t1=1.0, t_eval=teval, events=[ev],
                            rtol=1e-9, atol=1e-11)
    # Same boundary convention as the time event: pre-reset 0.5 at t=0.5.
    np.testing.assert_allclose(np.asarray(res.ys[:, 0]),
                               [0.0, 0.25, 0.5, 10.75, 11.0], atol=1e-5)
    assert len(res.log) == 1 and res.log[0][1] == "cross"
    assert res.log[0][0] == pytest.approx(0.5, abs=1e-5)


def test_terminal_event_stops_and_pads():
    teval = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    ev = Event(cond_fn=lambda t, y, a: y[0] - 0.5, direction=1, terminal=True)
    res = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                            t0=0.0, t1=1.0, t_eval=teval, events=[ev],
                            rtol=1e-9, atol=1e-11)
    # Stops at t=0.5; later points hold the terminal state.
    np.testing.assert_allclose(np.asarray(res.ys[:, 0]),
                               [0.0, 0.25, 0.5, 0.5, 0.5], atol=1e-5)


def test_direction_filters_crossing():
    # y goes 0 -> up; a downward-only condition never fires.
    ev = Event(cond_fn=lambda t, y, a: y[0] - 0.5, direction=-1,
               apply=lambda t, y, a: y + 10.0)
    res = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                            t0=0.0, t1=1.0, t_eval=jnp.array([1.0]), events=[ev],
                            rtol=1e-9, atol=1e-11)
    assert res.log == []
    assert float(res.ys[-1, 0]) == pytest.approx(1.0, abs=1e-5)


def test_multiple_time_events_in_order():
    teval = jnp.linspace(0.0, 1.0, 11)
    evs = [Event(at_times=[0.3], apply=lambda t, y, a: y + 1.0, name="a"),
           Event(at_times=[0.7], apply=lambda t, y, a: y * 2.0, name="b")]
    res = solve_with_events(_const_rhs(0.0), jnp.array([1.0]), 0.0,
                            t0=0.0, t1=1.0, t_eval=teval, events=evs,
                            rtol=1e-9, atol=1e-11)
    # dy/dt=0 from y0=1: +1 at 0.3 -> 2, *2 at 0.7 -> 4.
    assert float(res.ys[-1, 0]) == pytest.approx(4.0, abs=1e-6)
    assert [n for _, n in res.log] == ["a", "b"]


def test_runaway_state_event_raises():
    # A reset that does not clear the threshold re-fires forever.
    ev = Event(cond_fn=lambda t, y, a: y[0] - 0.5, apply=lambda t, y, a: y)
    with pytest.raises(RuntimeError, match="max_segments"):
        solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                          t0=0.0, t1=1.0, t_eval=None, events=[ev],
                          rtol=1e-9, atol=1e-11, max_segments=5)


def test_event_validation():
    with pytest.raises(ValueError, match="exactly one trigger"):
        Event()
    with pytest.raises(ValueError, match="exactly one trigger"):
        Event(cond_fn=lambda *a: 0.0, at_times=[1.0])
    with pytest.raises(ValueError, match="direction"):
        Event(cond_fn=lambda *a: 0.0, direction=2)
    with pytest.raises(ValueError, match="increasing"):
        Event(at_times=[1.0, 0.5])
    with pytest.raises(ValueError, match="at least one Event"):
        solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                          t0=0.0, t1=1.0, t_eval=None, events=[],
                          rtol=1e-9, atol=1e-11)


# --- AD through the evented solve --------------------------------------------

def test_grad_through_time_event_is_finite_and_correct():
    teval = jnp.array([1.0])

    def loss(scale):
        ev = Event(at_times=[0.5], apply=lambda t, y, a: y * scale)
        res = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                                t0=0.0, t1=1.0, t_eval=teval, events=[ev],
                                rtol=1e-10, atol=1e-12)
        return res.ys[-1, 0]

    # y(1) = scale*0.5 + 0.5, so d/dscale = 0.5.
    g = float(jax.grad(loss)(2.0))
    assert np.isfinite(g)
    assert g == pytest.approx(0.5, abs=1e-5)


# --- BatchReactor integration ------------------------------------------------

def test_batch_reactor_time_event(reactor, decay):
    teval = jnp.linspace(0.0, 2.0, 5)   # [0, 0.5, 1, 1.5, 2]
    # Halve A at t=1.0.
    ev = Event(at_times=[1.0], apply=lambda t, y, a: y.at[0].multiply(0.5),
               name="half")
    sol = reactor.solve(_A0(decay), (0.0, 2.0), teval, events=[ev])
    k = 0.1
    A = np.asarray(sol.C[:, 0])
    # Pre-reset analytic at the boundary, post-reset (halved) after.
    assert A[2] == pytest.approx(np.exp(-k * 1.0), abs=1e-4)          # t=1, pre
    assert A[3] == pytest.approx(0.5 * np.exp(-k * 1.5), abs=1e-4)    # t=1.5, post
    assert sol.events_log == [(1.0, "half")]


def test_batch_reactor_plain_solve_unaffected(reactor, decay):
    """A solve without events still returns the plain (events_log=None) result."""
    sol = reactor.solve(_A0(decay), (0.0, 1.0), jnp.linspace(0.0, 1.0, 3))
    assert sol.events_log is None
    assert float(sol.C[-1, 0]) == pytest.approx(np.exp(-0.1), abs=1e-4)


def test_event_path_matches_plain_solve_identity_reset(reactor, decay):
    """A no-op (identity) reset must reproduce the plain solve point-for-point.

    The drift guard: the event path builds its RHS via the shared
    ``make_chemistry_rhs`` factory and integrates via the shared
    ``_run_diffeqsolve`` kernel -- the same two pieces the plain solve uses. An
    identity reset at an interior time forces the multi-segment + dense-save
    machinery yet must leave the trajectory unchanged, so any divergence in the
    RHS or solver setup between ``solve()`` and ``solve_with_events`` shows up
    here as a numeric mismatch. The tolerance is loose (~1e-4) because splitting
    the span at the event times legitimately changes where the adaptive
    controller places its steps (each segment restarts it), perturbing the
    interpolation/accumulation error at the solver's own rtol -- a real RHS or
    kernel divergence would be orders larger. The companion never-firing
    single-segment test pins the kernel tightly.
    """
    teval = jnp.linspace(0.0, 2.0, 9)
    plain = reactor.solve(_A0(decay), (0.0, 2.0), teval)
    ev = Event(at_times=[0.7, 1.4], apply=lambda t, y, a: y, name="noop")
    evented = reactor.solve(_A0(decay), (0.0, 2.0), teval, events=[ev])
    np.testing.assert_allclose(np.asarray(evented.C), np.asarray(plain.C),
                               rtol=1e-4, atol=1e-7)


def test_event_path_matches_plain_solve_never_firing_state_event(reactor, decay):
    """A state event that never crosses (so a single segment runs under a
    terminating diffrax.Event) must also match the plain solve -- pinning the
    ``has_root`` branch of the driver to the plain kernel."""
    teval = jnp.linspace(0.0, 2.0, 9)
    plain = reactor.solve(_A0(decay), (0.0, 2.0), teval)
    # A decays from 1 toward 0; this threshold at -1 is never reached.
    ev = Event(cond_fn=lambda t, y, a: y[0] + 1.0, direction=-1, name="never")
    evented = reactor.solve(_A0(decay), (0.0, 2.0), teval, events=[ev])
    assert evented.events_log == []
    np.testing.assert_allclose(np.asarray(evented.C), np.asarray(plain.C),
                               rtol=1e-6, atol=1e-9)


def test_batch_reactor_grad_through_event(reactor, decay):
    teval = jnp.array([2.0])

    def loss(scale):
        ev = Event(at_times=[1.0], apply=lambda t, y, a: y * scale)
        sol = reactor.solve(_A0(decay), (0.0, 2.0), teval, events=[ev])
        return sol.C[-1, 0]

    g = float(jax.grad(loss)(0.5))
    # A(2) = scale * e^{-k} * e^{-k} = scale e^{-2k}; d/dscale = e^{-0.2}.
    assert np.isfinite(g)
    assert g == pytest.approx(np.exp(-0.2), abs=1e-4)


# --- Plant integration (slow: a real BSM1 solve) -----------------------------

@pytest.mark.slow
def test_plant_time_event_resets_state():
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start, load_bsm1_influent

    net = aquakin.load_network("asm1")
    plant = build_bsm1(net)
    plant.add_influent("feed", load_bsm1_influent("dry", net))
    y0 = jnp.asarray(bsm1_warm_start(plant))
    teval = jnp.linspace(0.0, 2.0, 9)
    i_ss = net.species_index["SS"]

    def spike(t, y, args):
        units = plant.states_by_unit(y)
        units["tank1"] = units["tank1"].at[i_ss].add(50.0)
        return plant.initial_state(overrides=units)

    ev = Event(at_times=[1.0], apply=spike, name="ss_spike")
    sol = plant.solve((0.0, 2.0), teval, y0=y0, events=[ev], max_steps=300_000)

    assert sol.events_log == [(1.0, "ss_spike")]
    assert sol.state.shape == (9, plant._total_state_size)
    ss = np.asarray(sol.C_named("tank1", "SS"))
    # The spike raises tank1 SS just after t=1 above its pre-event value.
    assert ss[5] > ss[4]
    assert np.all(np.isfinite(sol.state))


@pytest.mark.slow
def test_plant_events_reject_stable_adjoint():
    from aquakin.plant.bsm import build_bsm1

    net = aquakin.load_network("asm1")
    plant = build_bsm1(net)
    ev = Event(at_times=[1.0])
    with pytest.raises(ValueError, match="stable_adjoint"):
        plant.solve((0.0, 2.0), events=[ev], gradient="stable_adjoint")
