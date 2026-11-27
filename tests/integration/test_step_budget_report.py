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
        simple_model, aquakin.SpatialConditions.uniform(T=293.15), settings=aquakin.SolverSettings(
            integrator=TIGHT,
        )
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
        settings=aquakin.SolverSettings(integrator=TIGHT),
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


def _batch_sensitivity(simple_model, _make_plant):
    _batch(simple_model).solve_sensitivity(
        jnp.asarray([1.0, 0.0]),
        (0.0, 20.0),
        params=simple_model.default_parameters(),
        sens_params=["A_to_B.k"],
    )


def _batch_sensitivity_own_tolerance(simple_model, _make_plant):
    """Its own ``sens_atol`` takes the uncached forward-sensitivity path."""
    _batch(simple_model).solve_sensitivity(
        jnp.asarray([1.0, 0.0]),
        (0.0, 20.0),
        params=simple_model.default_parameters(),
        sens_params=["A_to_B.k"],
        sens_atol=1e-8,
    )


def _plant_forward_fast(simple_model, make_plant):
    make_plant().solve(
        t_span=(0.0, 1000.0), t_eval=jnp.asarray([0.0, 1000.0]), integrator=TIGHT,
        forward_fast=True,
    )


def _plant_sensitivity(corrector):
    def run(simple_model, make_plant):
        plant = make_plant()
        plant.solve_sensitivity(
            plant.default_parameters(),
            ["simple_decay.A_to_B.k"],
            t_span=(0.0, 1000.0),
            max_steps=2,
            corrector=corrector,
        )

    run.__name__ = f"_plant_sensitivity_{corrector}"
    return run


@pytest.mark.parametrize(
    "run",
    [
        _batch_solve,
        _batch_solve_on_grid,
        _batch_events,
        _pfr,
        _plant,
        _plant_events,
        _batch_sensitivity,
        _batch_sensitivity_own_tolerance,
        _plant_forward_fast,
        _plant_sensitivity("staggered"),
        _plant_sensitivity("simultaneous"),
    ],
    ids=lambda run: run.__name__.lstrip("_"),
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


def test_sensitivity_step_budget_under_a_gradient_still_fails(make_cstr_plant):
    """The simultaneous plant sensitivity is the differentiable one: under
    ``jax.grad`` its result is traced, so the runtime check stays in the
    program and the budget is still enforced."""
    plant = make_cstr_plant()

    def final_s(params):
        _t, _ys, S = plant.solve_sensitivity(
            params, ["simple_decay.A_to_B.k"], t_span=(0.0, 1000.0), max_steps=2,
            corrector="simultaneous",
        )
        return S[-1].sum()

    with pytest.raises(RuntimeError, match=BUDGET):
        jax.block_until_ready(jax.grad(final_s)(plant.default_parameters()))
