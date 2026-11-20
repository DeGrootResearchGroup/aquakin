"""End-to-end: a model whose rates use the general math functions.

The unit tests in ``tests/unit/test_nodes.py`` pin each node's arithmetic and
its guard. This file pins the thing the functions exist for: a rate law that
could not previously be written inside a model file at all -- a Beer-Lambert
attenuation ``exp(-alpha * z)`` -- now solves, and its attenuation coefficient
is an ordinary calibratable parameter that ``jax.grad`` reaches through the
solve. Before, ``alpha`` had to be folded into a precomputed fluence-rate
condition field, which put it outside the parameter vector and so outside
calibration, sensitivity and identifiability.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import aquakin

FIXTURE = Path(__file__).parent.parent / "fixtures" / "math_functions_model.yaml"


@pytest.fixture(scope="module")
def model():
    return aquakin.load_model_from_file(FIXTURE)


@pytest.fixture(scope="module")
def reactor(model):
    return aquakin.BatchReactor(model, model.default_conditions())


def test_model_with_math_functions_loads_and_is_unit_clean(model):
    """exp / log / sqrt / min survive the load, and the dimensional check has
    nothing to say about a correctly written model using them."""
    assert model.check_units() == []
    assert model.n_species == 3


def test_model_with_math_functions_solves(model, reactor):
    """The rates integrate: the attenuated species decays, the product grows."""
    sol = reactor.solve(
        model.default_concentrations(), t_span=(0.0, 10.0), t_eval=jnp.linspace(0.0, 10.0, 5)
    )
    A = sol.C_named("A")
    C = sol.C_named("C")
    assert jnp.all(jnp.isfinite(sol.C))
    assert A[-1] < A[0]            # photolysed away
    assert C[-1] > C[0]            # product accumulates
    # The half-order species depletes to zero and the adaptive solver
    # undershoots it by ~5e-9 -- within its tolerance, and benign only because
    # ``_safe_sqrt`` returns 0 rather than NaN for a slightly negative
    # argument. Without that guard the solve would fail here, not drift.
    assert jnp.all(sol.C >= -1e-7)


def test_grad_flows_through_solve_to_the_attenuation_coefficient(model, reactor):
    """``alpha`` lives inside ``exp()``. The gradient of a product
    concentration with respect to it must be finite and negative: more
    attenuation, less light, less product."""
    i_C = model.species_index["C"]
    i_alpha = model.param_index["photolysis.alpha"]

    def final_C(p):
        return reactor.solve(
            model.default_concentrations(), params=p, t_span=(0.0, 10.0)
        ).C[-1][i_C]

    g = jax.grad(final_C)(model.default_parameters())
    assert jnp.all(jnp.isfinite(g))
    assert g[i_alpha] < 0.0

    # and it is the real derivative, not an artefact of the guards. The
    # tolerance is set by the central difference and the solver's default
    # rtol/atol, not by the AD, which is exact.
    p = model.default_parameters()
    h = 1e-5
    fd = (final_C(p.at[i_alpha].add(h)) - final_C(p.at[i_alpha].add(-h))) / (2 * h)
    assert float(g[i_alpha]) == pytest.approx(float(fd), rel=1e-3)


def test_grad_is_finite_from_a_fully_depleted_start(model, reactor):
    """The guards earn their keep here: starting every species at zero puts
    ``sqrt`` and ``log`` exactly on their singularities for the whole solve."""
    C0 = jnp.zeros(model.n_species)

    def total(p):
        return jnp.sum(reactor.solve(C0, params=p, t_span=(0.0, 10.0)).C[-1])

    g = jax.grad(total)(model.default_parameters())
    assert jnp.all(jnp.isfinite(g))
