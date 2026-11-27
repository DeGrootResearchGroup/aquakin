"""The compiled located-event solve is kept across calls, and never reused stale.

Every segment of an ``events=`` solve runs one jit-compiled program whose
bounds, save grid, initial state and parameters are runtime arguments. The
plant keeps it in its ``_jit_cache`` and the batch reactor in the shared solver
cache, so a repeat solve compiles nothing. These tests count the builds with a
spy on the builder and the traces with the jitted function's own cache size,
and check a cache hit against a solve by a fresh object, so a hit that returned
a program compiled for different settings would fail.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin.integrate.batch as batch_mod
import aquakin.plant.dynamic_solve as dynamic_mod
from aquakin import BatchReactor, Event, SolverSettings, SpatialConditions


def _spy(monkeypatch, module):
    """Count calls of ``module.build_segment_solver``; return the counter."""
    calls = []
    real = module.build_segment_solver

    def counting(*args, **kwargs):
        calls.append(kwargs.get("has_t_eval"))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "build_segment_solver", counting)
    return calls


def _event_entries(plant):
    """The plant's cached event segment solvers, by key."""
    return {k: v[0] for k, v in plant._jit_cache.items() if k[0] == "events"}


def _dose():
    return Event(at_times=[5.0], apply=lambda t, y, a: y.at[0].set(5.0), name="dose")


def test_repeat_plant_event_solve_reuses_one_compiled_segment(
    make_cstr_plant, monkeypatch, xla_compiles
):
    """A second event solve of the same plant and settings compiles nothing
    (no XLA compilation at all), builds no segment solver and traces nothing
    new, though ``y0``, ``params`` and the ``t_eval`` values change; and every segment of a solve shares the one program. The
    reused program's result equals a fresh plant's bit for bit. (``atol`` is
    given: the default one is scaled off ``y0``, so a new ``y0`` would be a new
    setting.)"""
    calls = _spy(monkeypatch, dynamic_mod)
    plant = make_cstr_plant()
    ev = _dose()
    kw = dict(events=[ev], rtol=1e-8, atol=1e-10)
    plant.solve((0.0, 10.0), jnp.linspace(0.0, 10.0, 11), **kw)
    assert calls == [True]
    (solver,) = _event_entries(plant).values()
    # Two segments (either side of the event), one trace.
    assert solver._cache_size() == 1

    params = plant.default_parameters() * 1.5
    y0 = jnp.array([0.3, 0.1])
    t_eval = jnp.linspace(0.5, 9.5, 11)
    xla_compiles.clear()
    again = plant.solve((0.0, 10.0), t_eval, params=params, y0=y0, **kw)
    assert xla_compiles == []
    assert calls == [True]
    assert solver._cache_size() == 1

    fresh = make_cstr_plant().solve((0.0, 10.0), t_eval, params=params, y0=y0, **kw)
    np.testing.assert_array_equal(np.asarray(again.state), np.asarray(fresh.state))
    assert again.events_log == fresh.events_log == [(5.0, "dose")]


def test_changing_what_the_program_depends_on_misses_the_plant_cache(
    make_cstr_plant, monkeypatch
):
    """A new ``rtol`` builds a new segment solver, and its result equals a
    fresh plant's at that ``rtol`` bit for bit (the two ``rtol`` results differ,
    so a stale hit would show); the first setting then hits its own entry
    again. An added state event builds a new solver too, and
    ``set_temperature`` drops the cached ones."""
    calls = _spy(monkeypatch, dynamic_mod)
    plant = make_cstr_plant()
    y0 = jnp.array([0.2, 0.0])
    grid = jnp.linspace(0.0, 10.0, 6)

    def solve(target, **kw):
        return target.solve((0.0, 10.0), grid, y0=y0, events=kw.pop("events", [_dose()]), **kw)

    loose = solve(plant, rtol=1e-6)
    tight = solve(plant, rtol=1e-10)
    reference = solve(make_cstr_plant(), rtol=1e-10)
    np.testing.assert_array_equal(np.asarray(tight.state), np.asarray(reference.state))
    assert not np.array_equal(np.asarray(tight.state), np.asarray(loose.state))
    assert len(calls) == 3  # one per rtol on this plant, one on the reference
    again = solve(plant, rtol=1e-6)
    assert len(calls) == 3
    np.testing.assert_array_equal(np.asarray(again.state), np.asarray(loose.state))

    never = Event(cond_fn=lambda t, y, a: y[0] - 10.0, name="never")
    with_state_event = solve(plant, events=[_dose(), never], rtol=1e-6)
    assert len(calls) == 4
    np.testing.assert_array_equal(np.asarray(with_state_event.state), np.asarray(loose.state))

    assert len(_event_entries(plant)) == 3
    plant.set_temperature(30.0)
    assert _event_entries(plant) == {}


def test_repeat_batch_event_solve_shares_the_compiled_segment(
    simple_model, monkeypatch, xla_compiles
):
    """A fresh reactor's event solve compiles nothing: it reuses the segment
    solver an earlier reactor of the same model and settings compiled, and its result follows the
    closed form ``A = A0 exp(-k t)`` with ``A`` reset to 1 at the event. A
    different ``rtol`` builds a new one."""
    conditions = SpatialConditions.uniform(T=293.15)
    C0 = simple_model.concentrations({"A": 1.0})
    params = simple_model.default_parameters()
    k = float(params[0])
    ev = Event(at_times=[2.0], apply=lambda t, y, a: y.at[0].set(1.0), name="reset")
    t_eval = jnp.array([1.0, 2.0, 3.0])

    def solve(rtol):
        reactor = BatchReactor(simple_model, conditions, settings=SolverSettings(
            rtol=rtol,
            atol=1e-12,
        ))
        return reactor.solve(C0, (0.0, 3.0), t_eval, params=params, events=[ev])

    solve(1e-9)
    calls = _spy(monkeypatch, batch_mod)
    xla_compiles.clear()
    sol = solve(1e-9)
    assert xla_compiles == []
    assert calls == []
    expected = [np.exp(-k), np.exp(-2.0 * k), np.exp(-k)]
    # An adaptive solve at rtol = 1e-9: two decades above it.
    np.testing.assert_allclose(np.asarray(sol.C[:, 0]), expected, rtol=1e-7)

    solve(1e-8)
    assert calls == [True]


def test_event_time_unit_conversion_keeps_state_event_reusable(make_cstr_plant, monkeypatch):
    """A state event of a solve in another time unit is wrapped by a value
    object, so a repeat solve with the same ``Event`` hits the cache. From
    ``A = 0.2`` the tank follows ``A = 0.5 - 0.3 exp(-0.2 t)`` (seconds), so
    ``A`` rises through 0.3 at ``t = 5 ln 1.5`` s, logged in minutes."""
    calls = _spy(monkeypatch, dynamic_mod)
    plant = make_cstr_plant()
    ev = Event(cond_fn=lambda t, y, a: y[0] - 0.3, direction=1,
               apply=lambda t, y, a: y.at[0].set(0.0), name="high_A")
    y0 = jnp.array([0.2, 0.0])
    kw = dict(y0=y0, events=[ev], time_unit="min", rtol=1e-9, atol=1e-12)
    first = plant.solve((0.0, 0.1), **kw)
    second = plant.solve((0.0, 0.1), **kw)
    assert len(calls) == 1
    assert first.events_log == second.events_log
    (t_fire, name), = first.events_log
    assert name == "high_A"
    # The root find's tolerance (1e-6 relative) bounds the located time.
    assert t_fire == pytest.approx(5.0 * np.log(1.5) / 60.0, rel=1e-5)
