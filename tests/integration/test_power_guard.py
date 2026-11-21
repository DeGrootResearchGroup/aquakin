"""Half-order kinetics written as ``[X] ** 0.5`` behaves like ``sqrt([X])``.

``d/dx x**e = e * x**(e - 1)`` is infinite at ``x == 0`` whenever ``e < 1``, so
a fractional power carries exactly the singularity ``sqrt`` is guarded for. Two
spellings of the same mathematics must not disagree about it -- which is the
class of defect the sign/power precedence was (#699, #703).

The failure the guard prevents is quiet: the forward solve is unaffected and
only the gradient breaks, surfacing as an exhausted solver step budget that
names the integrator rather than the rate law. So it is pinned here, through a
real solve, rather than by a derivative check on the kernel alone.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import aquakin

FIXTURE = Path(__file__).parent.parent / "fixtures" / "power_guard_model.yaml"


@pytest.fixture(scope="module")
def model():
    return aquakin.load_model_from_file(FIXTURE)


@pytest.fixture(scope="module")
def reactor(model):
    return aquakin.BatchReactor(model, model.default_conditions())


def test_the_model_loads_and_is_unit_clean(model):
    """A fractional power halves the base's dimension, so the declared
    ``mol0.5 L-0.5 s-1`` rate constant is dimensionally right."""
    assert model.check_units() == []


def test_the_two_spellings_give_the_same_trajectory(model, reactor):
    """``k * [A] ** 0.5`` and ``k * sqrt([C])`` are the same rate law. A and C
    start equal, so their whole trajectories must coincide."""
    sol = reactor.solve(
        model.default_concentrations(), t_span=(0.0, 8.0), t_eval=jnp.linspace(0.0, 8.0, 9)
    )
    A, C = sol.C_named("A"), sol.C_named("C")
    assert jnp.allclose(A, C, rtol=0.0, atol=1e-12)
    # And the reaction actually ran, so the agreement is not two flat lines.
    assert float(A[0]) == pytest.approx(1.0)
    assert float(A[-1]) < 0.5


def test_a_gradient_at_a_depleted_reactant_is_finite(model, reactor):
    """Started from zero, both reactants sit on the singularity for the whole
    solve. Unguarded, ``[A] ** 0.5`` makes the adjoint diverge and the solve
    exhausts its step budget -- while the forward pass stays perfectly fine."""
    C0 = jnp.zeros(model.n_species)

    def total(p):
        return jnp.sum(reactor.solve(C0, params=p, t_span=(0.0, 10.0)).C[-1])

    value = float(total(model.default_parameters()))
    assert jnp.isfinite(value)
    grad = jax.grad(total)(model.default_parameters())
    assert jnp.all(jnp.isfinite(grad))


def test_the_gradient_is_finite_as_the_reactant_approaches_zero(model, reactor):
    """The physical case rather than the degenerate one: a reactant that starts
    full and is driven to the edge of depletion inside the window.

    Half-order kinetics reach zero in finite time -- here ``A(t) = (1 - 0.1t)**2``,
    so at ``t = 10`` -- and the extinction itself is non-Lipschitz and genuinely
    hard for an adaptive solver, which is a property of the rate law rather than
    of the guard. The window stops just short of it, at ``A ~ 2.5e-3``, where the
    local derivative of the power is already ~10x the rate constant.

    Unlike the test above, this one passes with or without the guard -- the
    reactant never reaches *exactly* zero, so the derivative stays large but
    finite. It is here to show the guard does not disturb the ordinary
    near-singular case, not to catch the bug.
    """
    i_k = model.param_index["k"]

    def final_B(p):
        sol = reactor.solve(model.default_concentrations(), params=p, t_span=(0.0, 9.5))
        return sol.C[-1, model.species_index["B"]]

    grad = jax.grad(final_B)(model.default_parameters())
    assert jnp.all(jnp.isfinite(grad))
    # More k consumes more A, so more B: the sign is not an accident of the guard.
    assert float(grad[i_k]) > 0.0
