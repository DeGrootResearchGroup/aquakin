"""Located event / discontinuity handling on the reactor and plant solves.

The driver is exercised on the first-order-decay fixture (``A -> B`` with rate
``k*[A]``, analytic ``A(t) = A0 e^{-kt}``), so every reset has a closed-form
expectation. The plant tests use the one-tank plant, whose state also has a
closed form.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import BatchReactor, Event, SpatialConditions, solve_with_events


@pytest.fixture(scope="module")
def reactor(simple_model):
    return BatchReactor(simple_model, SpatialConditions.uniform(T=293.15))


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

def test_batch_reactor_time_event(reactor, simple_model):
    teval = jnp.linspace(0.0, 2.0, 5)   # [0, 0.5, 1, 1.5, 2]
    # Halve A at t=1.0.
    ev = Event(at_times=[1.0], apply=lambda t, y, a: y.at[0].multiply(0.5),
               name="half")
    sol = reactor.solve(_A0(simple_model), (0.0, 2.0), teval, events=[ev])
    k = 0.1
    A = np.asarray(sol.C[:, 0])
    # Pre-reset analytic at the boundary, post-reset (halved) after.
    assert A[2] == pytest.approx(np.exp(-k * 1.0), abs=1e-4)          # t=1, pre
    assert A[3] == pytest.approx(0.5 * np.exp(-k * 1.5), abs=1e-4)    # t=1.5, post
    assert sol.events_log == [(1.0, "half")]


def test_batch_reactor_plain_solve_unaffected(reactor, simple_model):
    """A solve without events still returns the plain (events_log=None) result."""
    sol = reactor.solve(_A0(simple_model), (0.0, 1.0), jnp.linspace(0.0, 1.0, 3))
    assert sol.events_log is None
    assert float(sol.C[-1, 0]) == pytest.approx(np.exp(-0.1), abs=1e-4)


def test_event_path_matches_plain_solve_identity_reset(reactor, simple_model):
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
    plain = reactor.solve(_A0(simple_model), (0.0, 2.0), teval)
    ev = Event(at_times=[0.7, 1.4], apply=lambda t, y, a: y, name="noop")
    evented = reactor.solve(_A0(simple_model), (0.0, 2.0), teval, events=[ev])
    np.testing.assert_allclose(np.asarray(evented.C), np.asarray(plain.C),
                               rtol=1e-4, atol=1e-7)


def test_event_path_matches_plain_solve_never_firing_state_event(reactor, simple_model):
    """A state event that never crosses (so a single segment runs under a
    terminating diffrax.Event) must also match the plain solve -- pinning the
    ``has_root`` branch of the driver to the plain kernel."""
    teval = jnp.linspace(0.0, 2.0, 9)
    plain = reactor.solve(_A0(simple_model), (0.0, 2.0), teval)
    # A decays from 1 toward 0; this threshold at -1 is never reached.
    ev = Event(cond_fn=lambda t, y, a: y[0] + 1.0, direction=-1, name="never")
    evented = reactor.solve(_A0(simple_model), (0.0, 2.0), teval, events=[ev])
    assert evented.events_log == []
    np.testing.assert_allclose(np.asarray(evented.C), np.asarray(plain.C),
                               rtol=1e-6, atol=1e-9)


def test_batch_reactor_grad_through_event(reactor, simple_model):
    teval = jnp.array([2.0])

    def loss(scale):
        ev = Event(at_times=[1.0], apply=lambda t, y, a: y * scale)
        sol = reactor.solve(_A0(simple_model), (0.0, 2.0), teval, events=[ev])
        return sol.C[-1, 0]

    g = float(jax.grad(loss)(0.5))
    # A(2) = scale * e^{-k} * e^{-k} = scale e^{-2k}; d/dscale = e^{-0.2}.
    assert np.isfinite(g)
    assert g == pytest.approx(np.exp(-0.2), abs=1e-4)


def test_batch_reactor_time_event_solve_traces_under_jit(reactor, simple_model):
    """A time-event batch solve wrapped in ``jax.jit``, with the parameters
    traced and the save grid built inside the trace, gives the eager result and
    the closed form ``A = A0 exp(-k t)`` halved at the event."""
    ev = Event(at_times=[1.0], apply=lambda t, y, a: y.at[0].multiply(0.5), name="half")

    def trajectory(params):
        return reactor.solve(_A0(simple_model), (0.0, 2.0), jnp.linspace(0.0, 2.0, 5),
                             params=params, events=[ev]).C

    params = simple_model.default_parameters()
    eager = np.asarray(trajectory(params))
    np.testing.assert_allclose(np.asarray(jax.jit(trajectory)(params)), eager,
                               rtol=1e-12, atol=1e-15)
    t = np.linspace(0.0, 2.0, 5)
    expected = np.where(t <= 1.0, np.exp(-0.1 * t), 0.5 * np.exp(-0.1 * t))
    # The default reactor tolerances: the batch suite's 1e-4.
    np.testing.assert_allclose(eager[:, 0], expected, atol=1e-4)


def test_batch_time_unit_converts_event_times(reactor, simple_model):
    """``at_times`` is in the solve's ``time_unit``, and so is the event log."""
    # Model native unit is seconds (k = 0.1 1/s). Reset A to 1 at t = 1 min.
    ev = Event(at_times=[1.0], apply=lambda t, y, a: y.at[0].set(1.0), name="dose")
    sol = reactor.solve(_A0(simple_model), (0.0, 2.0), jnp.array([0.5, 1.0, 1.5, 2.0]),
                        time_unit="min", events=[ev])
    A = np.asarray(sol.C[:, 0])
    np.testing.assert_allclose(A, [np.exp(-3.0), np.exp(-6.0), np.exp(-3.0),
                                   np.exp(-6.0)], rtol=1e-3)
    np.testing.assert_allclose(np.asarray(sol.t), [0.5, 1.0, 1.5, 2.0])
    assert sol.events_log == [(pytest.approx(1.0), "dose")]
    assert sol.time_unit == "min"
    # The caller's Event is not modified by the unit conversion.
    assert ev.at_times == [1.0]


def test_batch_time_unit_state_event_sees_requested_time(reactor, simple_model):
    """``cond_fn`` receives ``t`` in the solve's ``time_unit``."""
    # Fires at t = 0.5 min (30 s), long before A crosses anything.
    ev = Event(cond_fn=lambda t, y, a: t - 0.5, direction=1, terminal=True)
    sol = reactor.solve(_A0(simple_model), (0.0, 2.0), time_unit="min", events=[ev])
    assert float(sol.t[-1]) == pytest.approx(0.5, abs=1e-6)
    assert sol.events_log[0][0] == pytest.approx(0.5, abs=1e-6)
    assert float(sol.C[-1, 0]) == pytest.approx(np.exp(-3.0), rel=1e-4)


def test_grad_through_time_event_with_time_unit(reactor, simple_model):
    """AD through a time event stays finite and correct with ``time_unit``."""

    def loss(scale):
        ev = Event(at_times=[0.5], apply=lambda t, y, a: y.at[0].add(scale))
        sol = reactor.solve(_A0(simple_model), (0.0, 1.0), jnp.array([1.0]),
                            time_unit="min", events=[ev])
        return sol.C[-1, 0]

    # A dose of ``scale`` at 0.5 min (30 s) decays for the remaining 30 s:
    # A(1 min) = e^{-6} + scale * e^{-3}; d/dscale = e^{-3}.
    g = float(jax.grad(loss)(0.5))
    assert np.isfinite(g)
    assert g == pytest.approx(np.exp(-3.0), rel=1e-3)


def test_terminal_event_without_t_eval_reports_stop_time(reactor, simple_model):
    """With ``t_eval=None`` a terminal event reports where the solve stopped."""
    ev = Event(cond_fn=lambda t, y, a: y[0] - 0.5, terminal=True, name="half")
    sol = reactor.solve(_A0(simple_model), (0.0, 100.0), events=[ev])
    t_half = np.log(2.0) / 0.1
    assert float(sol.t[-1]) == pytest.approx(t_half, abs=1e-5)
    assert float(sol.C[-1, 0]) == pytest.approx(0.5, abs=1e-6)


def test_time_event_at_t0_is_applied_and_logged():
    teval = jnp.array([0.0, 1.0])
    ev = Event(at_times=[0.0], apply=lambda t, y, a: y + 10.0, name="start")
    res = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                            t0=0.0, t1=1.0, t_eval=teval, events=[ev],
                            rtol=1e-9, atol=1e-11)
    # t_eval at t0 is the pre-reset initial state; the reset then acts.
    np.testing.assert_allclose(np.asarray(res.ys[:, 0]), [0.0, 11.0], atol=1e-6)
    assert res.log == [(0.0, "start")]
    res = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                            t0=0.0, t1=1.0, t_eval=None, events=[ev],
                            rtol=1e-9, atol=1e-11)
    assert float(res.ys[-1, 0]) == pytest.approx(11.0, abs=1e-6)
    assert res.log == [(0.0, "start")]


def test_time_event_at_t1_same_with_and_without_t_eval():
    """An event at exactly ``t1`` fires and is logged, and both output paths
    report the pre-reset value there."""
    ev = Event(at_times=[1.0], apply=lambda t, y, a: y + 10.0, name="end")
    kw = dict(t0=0.0, t1=1.0, events=[ev], rtol=1e-9, atol=1e-11)
    a = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                          t_eval=None, **kw)
    b = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                          t_eval=jnp.array([1.0]), **kw)
    assert float(a.ts[-1]) == float(b.ts[-1]) == 1.0
    assert float(a.ys[-1, 0]) == pytest.approx(1.0, abs=1e-6)
    assert float(b.ys[-1, 0]) == pytest.approx(1.0, abs=1e-6)
    assert a.log == b.log == [(1.0, "end")]



def test_terminal_time_event_at_t0_stops_before_integrating():
    """A terminal time event at exactly ``t0`` resets the initial state and
    ends the solve there: ``t_eval`` points after it hold the reset state, and
    ``t_eval=None`` reports ``t0`` with the pre-reset state."""
    ev = Event(at_times=[0.0], apply=lambda t, y, a: y + 10.0, terminal=True, name="stop")
    kw = dict(t0=0.0, t1=1.0, events=[ev], rtol=1e-9, atol=1e-11)
    a = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                          t_eval=jnp.array([0.0, 0.5, 1.0]), **kw)
    np.testing.assert_array_equal(np.asarray(a.ys[:, 0]), [0.0, 10.0, 10.0])
    assert a.log == [(0.0, "stop")]
    b = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0, t_eval=None, **kw)
    assert float(b.ts[-1]) == 0.0 and float(b.ys[-1, 0]) == 0.0
    assert b.log == [(0.0, "stop")]

def test_unnamed_event_not_mutated():
    ev = Event(at_times=[0.5])
    res = solve_with_events(_const_rhs(1.0), jnp.array([0.0]), 1.0,
                            t0=0.0, t1=1.0, t_eval=None, events=[ev],
                            rtol=1e-9, atol=1e-11)
    assert res.log == [(0.5, "event0")]
    assert ev.name is None


def test_plant_time_unit_converts_event_times(cstr_plant):
    """``Plant.solve(time_unit=..., events=...)`` reads ``at_times`` in the
    requested unit and reports the event log in it."""
    plant = cstr_plant

    def ev(at):
        return Event(at_times=[at], apply=lambda t, y, a: y.at[0].set(5.0),
                     name="dose")

    native = plant.solve(t_span=(0.0, 120.0), t_eval=jnp.array([90.0, 120.0]),
                         events=[ev(60.0)])
    minutes = plant.solve(t_span=(0.0, 2.0), t_eval=jnp.array([1.5, 2.0]),
                          time_unit="min", events=[ev(1.0)])
    np.testing.assert_allclose(np.asarray(minutes.state),
                               np.asarray(native.state), rtol=1e-6, atol=1e-9)
    assert native.events_log == [(60.0, "dose")]
    assert minutes.events_log == [(pytest.approx(1.0), "dose")]


def _tank_closed_form(t, A0, B0):
    """The one-tank plant's state from ``(A0, B0)`` after a time ``t``.

    With dilution ``Q/V = 0.1`` and ``k = 0.1``, ``dA/dt = 0.1 (1 - A) - 0.1 A``
    gives ``A = 0.5 + c exp(-0.2 t)`` with ``c = A0 - 0.5``, and
    ``dB/dt = 0.1 A - 0.1 B`` gives ``B = 0.5 - c exp(-0.2 t) + d exp(-0.1 t)``
    with ``d = B0 - 0.5 + c``.
    """
    c = A0 - 0.5
    d = B0 - 0.5 + c
    A = 0.5 + c * np.exp(-0.2 * t)
    B = 0.5 - c * np.exp(-0.2 * t) + d * np.exp(-0.1 * t)
    return np.stack([A, B], axis=-1)


def test_plant_time_event_reset_matches_closed_form(cstr_plant):
    """A time event at ``t = 5`` resets the tank's ``A`` to 5. The trajectory
    follows the closed form from ``y0`` up to the event (the save at ``t = 5``
    reports the pre-reset value) and the closed form from the reset state
    after it, with ``B`` carried through the reset unchanged."""
    y0 = jnp.array([0.2, 0.0])
    ev = Event(at_times=[5.0], apply=lambda t, y, a: y.at[0].set(5.0),
               name="dose")
    teval = jnp.linspace(0.0, 10.0, 11)
    sol = cstr_plant.solve((0.0, 10.0), teval, y0=y0, events=[ev],
                           rtol=1e-9, atol=1e-12)

    assert sol.events_log == [(5.0, "dose")]
    t = np.asarray(teval)
    before = _tank_closed_form(t[:6], 0.2, 0.0)
    at_event = _tank_closed_form(5.0, 0.2, 0.0)
    after = _tank_closed_form(t[6:] - 5.0, 5.0, at_event[1])
    expected = np.concatenate([before, after])
    # An adaptive solve at rtol = 1e-9: two decades above it.
    np.testing.assert_allclose(np.asarray(sol.state), expected,
                               rtol=1e-7, atol=1e-10)


def test_plant_events_reject_unsupported_integrator(cstr_plant):
    """The located-event segmented solve manages its own integrator, so an
    explicit colored-Jacobian / custom solver (which it cannot honour) is
    rejected. An explicit default IntegratorConfig is accepted, and the solve
    through a never-resetting event follows the closed form."""
    import diffrax

    plant = cstr_plant
    ev = Event(at_times=[1.0])
    sol = plant.solve((0.0, 2.0), jnp.array([2.0]), y0=jnp.array([0.2, 0.0]),
                      events=[ev], rtol=1e-9, atol=1e-12,
                      integrator=aquakin.IntegratorConfig())
    assert sol.events_log == [(1.0, "event0")]
    # An adaptive solve at rtol = 1e-9: two decades above it.
    np.testing.assert_allclose(np.asarray(sol.state[-1]),
                               _tank_closed_form(2.0, 0.2, 0.0),
                               rtol=1e-7, atol=1e-10)
    with pytest.raises(ValueError, match="not supported"):
        plant.solve((0.0, 2.0), events=[ev],
                    integrator=aquakin.IntegratorConfig(colored_jacobian=True))
    with pytest.raises(ValueError, match="not supported"):
        plant.solve((0.0, 2.0), events=[ev],
                    integrator=aquakin.IntegratorConfig(solver=diffrax.Kvaerno5()))


def test_plant_event_resets_several_units_by_name(simple_model):
    """An event that resets two units at once through ``states_by_unit`` /
    ``initial_state(overrides=)``: in two tanks in series (each ``V = 100``,
    tank ``a`` fed ``Q = 10`` of ``A = 1``) a time event at ``t = 5`` sets
    ``a``'s ``A`` to 5 and ``b``'s ``B`` to 0, leaving the other entries. The
    plant is linear, ``y' = M y + f``, so the closed form on each side of the
    reset is ``y(t) = y* + exp(M t) (y(0) - y*)`` with ``y* = -M^-1 f``."""
    from scipy.linalg import expm

    from aquakin.plant import CSTRUnit, Plant
    from aquakin.plant.influent import InfluentSeries

    plant = Plant("two_tanks", time_unit="s")
    for name in ("a", "b"):
        plant.add_unit(CSTRUnit(name=name, model=simple_model, volume=100.0,
                                input_port_names=["inlet"],
                                conditions={"T": 293.15}))
    plant.add_influent("feed", InfluentSeries(
        t=jnp.asarray([0.0, 100.0]), Q=jnp.asarray([10.0, 10.0]),
        C=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]), model=simple_model),
        to="a.inlet")
    plant.connect("a", "b")

    def reset(t, y, args):
        units = plant.states_by_unit(y)
        units["a"] = units["a"].at[0].set(5.0)
        units["b"] = units["b"].at[1].set(0.0)
        return plant.initial_state(overrides=units)

    y0 = plant.initial_state(overrides={"a": jnp.array([0.2, 0.0]),
                                        "b": jnp.array([0.1, 0.3])})
    ev = Event(at_times=[5.0], apply=reset, name="reset")
    teval = jnp.linspace(0.0, 10.0, 11)
    sol = plant.solve((0.0, 10.0), teval, y0=y0, events=[ev],
                      rtol=1e-9, atol=1e-12)
    assert sol.events_log == [(5.0, "reset")]

    # State order (A_a, B_a, A_b, B_b); dilution Q/V = 0.1 and k = 0.1.
    M = np.array([[-0.2, 0.0, 0.0, 0.0],
                  [0.1, -0.1, 0.0, 0.0],
                  [0.1, 0.0, -0.2, 0.0],
                  [0.0, 0.1, 0.1, -0.1]])
    y_star = -np.linalg.solve(M, np.array([0.1, 0.0, 0.0, 0.0]))

    def closed_form(t, start):
        return y_star + expm(M * t) @ (start - y_star)

    start = np.array([0.2, 0.0, 0.1, 0.3])
    t = np.asarray(teval)
    # The save at t = 5 reports the pre-reset value.
    before = [closed_form(ti, start) for ti in t[:6]]
    after_reset = closed_form(5.0, start)
    after_reset[0], after_reset[3] = 5.0, 0.0
    after = [closed_form(ti - 5.0, after_reset) for ti in t[6:]]
    expected = np.array(before + after)
    got = np.concatenate([np.asarray(sol.unit_state("a")),
                          np.asarray(sol.unit_state("b"))], axis=1)
    # An adaptive solve at rtol = 1e-9: two decades above it.
    np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-10)


def test_state_event_step_budget_raises_friendly(simple_model):
    """A state-event (eager) solve that exhausts the integrator step budget must
    re-raise the friendly remedy message, not the raw equinox MaxStepsReached
    chatter -- the segmented eager solve is now wrapped like the plain solves."""
    fields = SpatialConditions.uniform(T=293.15).fields
    rhs = lambda t, y, p: simple_model.dCdt(y, p, fields, 0)
    C0 = _A0(simple_model)
    params = simple_model.default_parameters()
    # Fires only when A has all but vanished, so the solver keeps stepping over a
    # long span; a tiny max_steps exhausts the budget first.
    ev = Event(cond_fn=lambda t, y, args: y[0] - 1e-12)
    with pytest.raises(RuntimeError, match="integrator step budget"):
        solve_with_events(rhs, C0, params, t0=0.0, t1=1e8,
                          t_eval=jnp.array([1e8]), events=[ev],
                          rtol=1e-6, atol=1e-9, max_steps=4)
