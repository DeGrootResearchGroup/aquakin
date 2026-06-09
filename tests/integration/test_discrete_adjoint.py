"""Tests for the cap-free discrete-adjoint reverse-mode gradient.

``implicit_euler_adjoint_solve`` integrates with a robust adaptive implicit-Euler
forward and a hand-written discrete-adjoint backward (per-step bounded transposed
solves). The point is that it produces a *finite, correct* reverse-mode gradient
of a stiff solve with NO ``dtmax`` cap -- where differentiating through diffrax's
own solve (``RecursiveCheckpointAdjoint``) goes non-finite. Correctness is pinned
two ways: against the closed-form gradient of first-order decay, and against the
(correct but capped) ``RecursiveCheckpointAdjoint`` gradient of the same
implicit-Euler solve.
"""

import math

import jax
import jax.numpy as jnp
import pytest

import diffrax

import aquakin
from aquakin.integrate.discrete_adjoint import implicit_euler_adjoint_solve


def _decay_rhs(simple_network):
    fields = aquakin.SpatialConditions.uniform(1, T=293.15).fields
    return lambda t, y, p: simple_network.dCdt(y, p, fields, 0)


def test_analytic_decay_gradient(simple_network):
    # A -> B, dA/dt = -k A, A(t) = A0 e^{-kt}. Loss = A(T)^2, so
    # dLoss/dk = 2 A(T) (dA/dk) = 2 e^{-kT} (-T e^{-kT}).
    rhs = _decay_rhs(simple_network)
    C0 = jnp.array([1.0, 0.0])
    p = simple_network.default_parameters()
    k = float(p[0])
    T = 15.0

    def loss(pp):
        return implicit_euler_adjoint_solve(
            rhs, C0, pp, (0.0, T), rtol=1e-10, atol=1e-12
        )[0] ** 2

    g = jax.grad(loss)(p)[0]
    exact = 2 * math.exp(-k * T) * (-T * math.exp(-k * T))
    assert jnp.isfinite(g)
    assert abs(float(g) - exact) / abs(exact) < 1e-4


def test_gradient_wrt_y0_finite(simple_network):
    rhs = _decay_rhs(simple_network)
    p = simple_network.default_parameters()

    def loss(C0):
        return jnp.sum(
            implicit_euler_adjoint_solve(rhs, C0, p, (0.0, 5.0)) ** 2
        )

    g = jax.grad(loss)(jnp.array([1.0, 0.0]))
    assert jnp.all(jnp.isfinite(g))
    # more initial A -> more of both species' squared final value
    assert float(g[0]) > 0.0


@pytest.mark.validation
def test_stiff_finite_uncapped_and_matches_capped():
    # The canonical stiff network. Differentiating through diffrax's own solve is
    # non-finite without a dtmax cap; the discrete adjoint is finite uncapped and
    # must match the (correct) capped RecursiveCheckpointAdjoint gradient of the
    # same implicit-Euler solve.
    net = aquakin.load_network("wats_sewer_khalil_paper_balanced")
    cond = net.default_conditions(1)
    C0 = net.default_concentrations()
    p = net.default_parameters()
    fields = cond.fields
    rhs = lambda t, y, pp: net.dCdt(y, pp, fields, 0)
    span = (0.0, 0.5)

    def loss(pp):
        return jnp.sum(
            implicit_euler_adjoint_solve(rhs, C0, pp, span, rtol=1e-6, atol=1e-9) ** 2
        )

    g = jax.jit(jax.grad(loss))(p)
    assert bool(jnp.all(jnp.isfinite(g)))

    # Reference: diffrax ImplicitEuler + RecursiveCheckpointAdjoint, capped so it
    # is finite. Same method => same discrete gradient.
    def loss_ref(pp):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, a: rhs(t, y, a)), diffrax.ImplicitEuler(),
            0.0, span[1], 1e-6, C0, args=pp,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9, dtmax=1e-3),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            saveat=diffrax.SaveAt(t1=True), max_steps=200_000,
        )
        return jnp.sum(sol.ys[0] ** 2)

    g_ref = jax.jit(jax.grad(loss_ref))(p)
    assert bool(jnp.all(jnp.isfinite(g_ref)))
    rel = float(jnp.linalg.norm(g - g_ref) / (jnp.linalg.norm(g_ref) + 1e-30))
    assert rel < 1e-5
