"""End-to-end: a model file whose rate law is not built into aquakin.

The unit tests pin the registration mechanism. This pins the thing it exists
for: a researcher writes a rate law aquakin does not ship, registers it from
their own code, and everything downstream -- loading, solving, and
differentiating through the solve -- works as it does for a built-in, with the
function's own constants as ordinary fitted parameters.
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.core.nodes import ASTNode, _OperatorNode
from aquakin.core.registry import register_rate_function, unregister_rate_function
from aquakin.schema.loader import clear_model_cache
from aquakin.utils.units import DIMENSIONLESS

FIXTURE = Path(__file__).parent.parent / "fixtures" / "registered_function_model.yaml"


@dataclass(frozen=True)
class HillNode(_OperatorNode):
    """Cooperative binding: ``X^n / (K^n + X^n)``."""

    X: ASTNode
    K: ASTNode
    n: ASTNode

    KIND = "integration_hill"
    FUNCTION_NAME = "hill"

    @staticmethod
    def op(o):
        x, k, n = o
        return x**n / (k**n + x**n)

    @staticmethod
    def DIMENSION(args):
        return DIMENSIONLESS


@pytest.fixture
def registered():
    clear_model_cache()
    register_rate_function(HillNode)
    yield
    unregister_rate_function("hill")
    clear_model_cache()


def test_model_naming_an_unregistered_function_fails_clearly():
    """Before registration the file does not load, and the error says what to do."""
    clear_model_cache()
    with pytest.raises(Exception, match="hill"):
        aquakin.load_model_from_file(FIXTURE)


def test_model_with_a_registered_function_loads_and_solves(registered):
    model = aquakin.load_model_from_file(FIXTURE)
    assert model.check_units() == []

    reactor = aquakin.BatchReactor(model, model.default_conditions())
    sol = reactor.solve(
        model.default_concentrations(), t_span=(0.0, 20.0), t_eval=jnp.linspace(0.0, 20.0, 5)
    )
    S, P = sol.C_named("S"), sol.C_named("P")
    assert jnp.all(jnp.isfinite(sol.C))
    assert S[-1] < S[0]          # substrate consumed
    assert P[-1] > P[0]          # product formed
    # S + P is conserved by the stoichiometry
    assert float(S[-1] + P[-1]) == pytest.approx(float(S[0] + P[0]), rel=1e-6)


def test_grad_flows_through_the_solve_to_the_functions_own_parameter(registered):
    """``K_h`` and ``n_h`` are arguments of the *registered* function. They are
    ordinary model parameters, so a gradient reaches them through the solve --
    which is the point: a rate law folded into a condition field instead would
    put them outside the parameter vector entirely."""
    model = aquakin.load_model_from_file(FIXTURE)
    reactor = aquakin.BatchReactor(model, model.default_conditions())
    i_P = model.species_index["P"]
    i_K = model.param_index["cooperative.K_h"]

    def final_P(p):
        return reactor.solve(
            model.default_concentrations(), params=p, t_span=(0.0, 20.0)
        ).C[-1][i_P]

    p = model.default_parameters()
    g = jax.grad(final_P)(p)
    assert jnp.all(jnp.isfinite(g))
    # A larger half-saturation makes the Hill term smaller, so less product.
    assert g[i_K] < 0.0

    h = 1e-6
    fd = (final_P(p.at[i_K].add(h)) - final_P(p.at[i_K].add(-h))) / (2 * h)
    assert float(g[i_K]) == pytest.approx(float(fd), rel=1e-3)


def test_the_model_uses_the_vectorized_kernel(registered):
    """Not the scalar fallback: a registered function must not cost the model
    its batched rate kernel."""
    model = aquakin.load_model_from_file(FIXTURE)
    assert model._rate_kernel is not None
