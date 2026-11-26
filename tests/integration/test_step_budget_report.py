"""A solve that runs out of integrator steps raises aquakin's own error, quietly.

The step budget is checked on the host from the solve's ``result``, so hitting it
raises one ``RuntimeError`` naming the remedies, with nothing printed before it:
no JAX callback traceback and no Equinox banner. Under an outer trace (a
gradient) the result cannot be read on the host, and the solve still fails.
"""

import logging

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin import Event

BUDGET = r"step budget \(max_steps=2\)"
TIGHT = aquakin.IntegratorConfig(max_steps=2)


def _batch(simple_model):
    return aquakin.BatchReactor(
        simple_model, aquakin.SpatialConditions.uniform(T=293.15), integrator=TIGHT
    )


def _batch_solve(simple_model, _make_plant):
    _batch(simple_model).solve(jnp.asarray([1.0, 0.0]), t_span=(0.0, 20.0))


def _batch_solve_on_grid(simple_model, _make_plant):
    _batch(simple_model).solve(
        jnp.asarray([1.0, 0.0]), t_span=(0.0, 20.0), t_eval=jnp.linspace(0.0, 20.0, 5)
    )


def _batch_events(simple_model, _make_plant):
    _batch(simple_model).solve(
        jnp.asarray([1.0, 0.0]),
        t_span=(0.0, 20.0),
        events=[Event(at_times=[10.0], apply=lambda t, y, args: y)],
    )


def _pfr(simple_model, _make_plant):
    reactor = aquakin.PlugFlowReactor(
        simple_model,
        aquakin.SpatialConditions.uniform(T=293.15),
        n_points=5,
        length=20.0,
        velocity=1.0,
        integrator=TIGHT,
    )
    reactor.solve(jnp.asarray([1.0, 0.0]))


def _plant(simple_model, make_plant):
    make_plant().solve(t_span=(0.0, 1000.0), integrator=TIGHT)


def _plant_events(simple_model, make_plant):
    make_plant().solve(
        t_span=(0.0, 1000.0),
        integrator=TIGHT,
        events=[Event(at_times=[500.0], apply=lambda t, y, args: y)],
    )


@pytest.mark.parametrize(
    "run",
    [_batch_solve, _batch_solve_on_grid, _batch_events, _pfr, _plant, _plant_events],
)
def test_step_budget_raises_without_a_banner(run, simple_model, make_cstr_plant, capfd, caplog):
    """The error names the budget, and nothing reaches stderr or the log first.
    The absence of output is the property under test."""
    caplog.set_level(logging.DEBUG)
    with pytest.raises(RuntimeError, match=BUDGET):
        run(simple_model, make_cstr_plant)
    err = capfd.readouterr().err
    assert "pure_callback" not in err
    assert "Equinox" not in err and "equinox" not in err
    assert not [r for r in caplog.records if "pure_callback" in r.getMessage()]


def test_step_budget_under_a_gradient_still_fails(simple_model):
    """Under ``jax.grad`` the result is traced, so the solve keeps the runtime
    check: the budget is still enforced rather than returning a gradient of an
    unfinished solve."""
    reactor = _batch(simple_model)

    def final_a(params):
        sol = reactor.solve(jnp.asarray([1.0, 0.0]), params=params, t_span=(0.0, 20.0))
        return sol.C[-1, 0]

    with pytest.raises(RuntimeError, match=BUDGET):
        jax.block_until_ready(jax.grad(final_a)(simple_model.default_parameters()))


def _batched(*results):
    """A batch of diffrax results, as a vmapped solve returns them."""
    import diffrax

    codes = jnp.arange(len(results))
    return jax.vmap(
        lambda i: diffrax.RESULTS.where(
            i == 0,
            results[0],
            results[1] if len(results) > 1 else results[0],
        )
    )(codes)


def test_check_solve_result_reports_each_outcome():
    """A successful result passes the outputs through unchanged; a failure other
    than the step budget raises Equinox's error type (as ``throw=True`` did, so the
    calibration multistart still recognizes it) with Diffrax's reason (scalar) or
    naming a failed batch element; a batch with an element out of steps gives the budget
    message."""
    import diffrax
    import equinox as eqx

    from aquakin.integrate._solver import check_solve_result

    R = diffrax.RESULTS
    out = (jnp.asarray([1.0]), jnp.asarray([2.0]))
    assert check_solve_result(R.successful, out, max_steps=5, what="x solve") is out
    assert check_solve_result(R.event_occurred, out, max_steps=5, what="x solve") is out
    with pytest.raises(eqx.EquinoxRuntimeError, match="x solve stopped before the end of its span: A linear"):
        check_solve_result(R.singular, out, max_steps=5, what="x solve")
    with pytest.raises(RuntimeError, match=r"step budget \(max_steps=5\)"):
        check_solve_result(_batched(R.successful, R.max_steps_reached), out,
                           max_steps=5, what="x solve")
    with pytest.raises(eqx.EquinoxRuntimeError, match="an element of the batch failed"):
        check_solve_result(_batched(R.successful, R.singular), out, max_steps=5, what="x solve")
