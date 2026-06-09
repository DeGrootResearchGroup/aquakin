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


def test_trajectory_loss_gradient(simple_network):
    # Loss over several observation times: L = sum_i A(t_i)^2, with the closed
    # form dL/dk = sum_i 2 A(t_i) (-t_i A(t_i)), A(t)=e^{-kt}.
    rhs = _decay_rhs(simple_network)
    C0 = jnp.array([1.0, 0.0])
    p = simple_network.default_parameters()
    k = float(p[0])
    t_obs = jnp.array([2.0, 5.0, 9.0, 15.0])

    def loss(pp):
        ys = implicit_euler_adjoint_solve(
            rhs, C0, pp, (0.0, 15.0), t_obs, rtol=1e-10, atol=1e-12
        )
        return jnp.sum(ys[:, 0] ** 2)

    g = jax.grad(loss)(p)[0]
    exact = sum(
        2 * math.exp(-k * ti) * (-ti * math.exp(-k * ti)) for ti in [2.0, 5.0, 9.0, 15.0]
    )
    assert jnp.isfinite(g)
    assert abs(float(g) - exact) / abs(exact) < 1e-4


def test_t_eval_returns_states_at_times(simple_network):
    # With t_eval the solve returns the state at each observation time, matching
    # a plain diffrax solve sampled at those times.
    rhs = _decay_rhs(simple_network)
    C0 = jnp.array([1.0, 0.0])
    p = simple_network.default_parameters()
    t_obs = jnp.array([1.0, 3.0, 7.0])
    ys = implicit_euler_adjoint_solve(rhs, C0, p, (0.0, 7.0), t_obs, rtol=1e-9, atol=1e-11)
    assert ys.shape == (3, 2)
    k = float(p[0])
    assert jnp.allclose(ys[:, 0], jnp.exp(-k * t_obs), atol=1e-5, rtol=1e-4)


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


@pytest.mark.validation
def test_calibrate_discrete_adjoint_matches_capped_ad():
    # End-to-end: a Khalil-model calibration with gradient="discrete_adjoint"
    # (cap-free) must reach the same optimum as the existing capped-Kvaerno5
    # gradient="ad" path. Synthetic recovery; compare the fitted parameters.
    import diffrax

    net = aquakin.load_network("wats_sewer_khalil_paper_balanced")
    cond = net.default_conditions(1)
    C0 = net.default_concentrations()
    p_def = net.default_parameters()
    free = ["mu_h", "q_ferm"]
    obs_species = ["S_SO4", "sumS", "S_VFA", "S_NO"]
    t_obs = jnp.linspace(0.04, 0.2, 5)
    span = (0.0, float(t_obs[-1]))
    rtol, atol = 1e-5, 1e-8

    idx = [net.param_index[n] for n in free]
    p_true = p_def.at[jnp.array(idx)].multiply(jnp.array([1.4, 0.6]))
    gen = aquakin.BatchReactor(net, cond, rtol=1e-9, atol=1e-11)
    obs = gen.solve(C0, p_true, t_span=span, t_eval=t_obs).C[
        :, [net.species_index[s] for s in obs_species]
    ]

    # size the discrete-adjoint trajectory buffer from the actual step count
    ctrl = diffrax.ClipStepSizeController(
        diffrax.PIDController(rtol=rtol, atol=atol), step_ts=t_obs
    )
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, a: net.dCdt(y, a, cond.fields, 0)),
        diffrax.ImplicitEuler(), 0.0, span[1], 1e-6, C0, args=p_def,
        stepsize_controller=ctrl, saveat=diffrax.SaveAt(steps=True), max_steps=300_000,
    )
    max_steps = int(jnp.sum(jnp.isfinite(sol.ts))) * 2 + 50

    common = dict(observed_species=obs_species, loss="mse", laplace=False,
                  max_iter=150, tol=1e-9)
    r_ref = aquakin.calibrate(
        aquakin.BatchReactor(net, cond, rtol=rtol, atol=atol, dtmax=5e-4),
        C0, obs, t_obs, free, gradient="ad", **common,
    )
    r_da = aquakin.calibrate(
        aquakin.BatchReactor(net, cond, rtol=rtol, atol=atol),
        C0, obs, t_obs, free, gradient="discrete_adjoint",
        discrete_adjoint_max_steps=max_steps, **common,
    )
    assert r_ref.converged and r_da.converged
    v_ref = jnp.array([r_ref.params_named[n] for n in free])
    v_da = jnp.array([r_da.params_named[n] for n in free])
    rel = float(jnp.max(jnp.abs(v_da - v_ref) / jnp.abs(v_ref)))
    assert rel < 5e-3


@pytest.mark.validation
def test_stiff_trajectory_loss_matches_capped():
    # A multi-observation (trajectory) loss -- the calibration shape -- must be
    # finite uncapped and match the capped reference using the same forced-step
    # forward solve.
    net = aquakin.load_network("wats_sewer_khalil_paper_balanced")
    cond = net.default_conditions(1)
    C0 = net.default_concentrations()
    p = net.default_parameters()
    fields = cond.fields
    rhs = lambda t, y, pp: net.dCdt(y, pp, fields, 0)
    t_obs = jnp.array([0.05, 0.1, 0.2, 0.35, 0.5])
    si = net.species_index["S_SO4"]

    def loss(pp):
        ys = implicit_euler_adjoint_solve(rhs, C0, pp, (0.0, 0.5), t_obs, rtol=1e-6, atol=1e-9)
        return jnp.sum(ys[:, si] ** 2) + 1e-3 * jnp.sum(ys ** 2)

    g = jax.jit(jax.grad(loss))(p)
    assert bool(jnp.all(jnp.isfinite(g)))

    def loss_ref(pp):
        ctrl = diffrax.ClipStepSizeController(
            diffrax.PIDController(rtol=1e-6, atol=1e-9, dtmax=1e-3), step_ts=t_obs
        )
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, a: rhs(t, y, a)), diffrax.ImplicitEuler(),
            0.0, 0.5, 1e-6, C0, args=pp, stepsize_controller=ctrl,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            saveat=diffrax.SaveAt(ts=t_obs), max_steps=200_000,
        )
        return jnp.sum(sol.ys[:, si] ** 2) + 1e-3 * jnp.sum(sol.ys ** 2)

    g_ref = jax.jit(jax.grad(loss_ref))(p)
    rel = float(jnp.linalg.norm(g - g_ref) / (jnp.linalg.norm(g_ref) + 1e-30))
    assert rel < 1e-5
