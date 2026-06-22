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
import optimistix

import aquakin
from aquakin.integrate.discrete_adjoint import (
    esdirk_adjoint_solve,
    implicit_euler_adjoint_solve,
)


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


def test_equals_autodiff_through_same_solve(simple_network):
    # Permanent guard: the hand-written discrete adjoint computes the SAME
    # gradient that JAX autodiff would, when autodiff can run. On a small
    # (non-stiff) network, differentiate the identical implicit-Euler solve --
    # same forced-step controller, same tolerances -- both ways and require
    # machine-precision agreement. (Only the integrator's adjoint is hand-coded;
    # the model derivatives are autodiff in both. This pins that they match.)
    rhs = _decay_rhs(simple_network)
    C0 = jnp.array([1.0, 0.0])
    p = simple_network.default_parameters()
    t_obs = jnp.array([1.0, 4.0, 8.0])
    rtol, atol = 1e-8, 1e-10

    def loss_stable(pp):
        ys = implicit_euler_adjoint_solve(rhs, C0, pp, (0.0, 8.0), t_obs,
                                          rtol=rtol, atol=atol)
        return jnp.sum(ys ** 2)

    def loss_autodiff(pp):
        # Same forward solve, differentiated by diffrax's RecursiveCheckpointAdjoint.
        ctrl = diffrax.ClipStepSizeController(
            diffrax.PIDController(rtol=rtol, atol=atol), step_ts=t_obs
        )
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, a: rhs(t, y, a)),
            # Explicit root-finder tolerances: older diffrax (py3.10) does not
            # treat ClipStepSizeController (no dtmax) as adaptive and otherwise
            # rejects the implicit solver's unspecified tolerances.
            diffrax.ImplicitEuler(
                root_finder=diffrax.VeryChord(rtol=rtol, atol=atol,
                                              norm=optimistix.max_norm)),
            0.0, 8.0, 1e-6, C0, args=pp, stepsize_controller=ctrl,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            saveat=diffrax.SaveAt(ts=t_obs), max_steps=100_000,
        )
        return jnp.sum(sol.ys ** 2)

    g_stable = jax.grad(loss_stable)(p)
    g_autodiff = jax.grad(loss_autodiff)(p)
    assert jnp.allclose(g_stable, g_autodiff, rtol=1e-7, atol=1e-10)


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


def test_esdirk_analytic_trajectory_gradient(simple_network):
    # The high-order (Kvaerno5) discrete adjoint on the closed-form decay
    # trajectory loss; Kvaerno5's 5th order makes the primal -- and so the
    # gradient -- tighter than implicit Euler at the same tolerance.
    rhs = _decay_rhs(simple_network)
    C0 = jnp.array([1.0, 0.0])
    p = simple_network.default_parameters()
    k = float(p[0])
    t_obs = jnp.array([2.0, 5.0, 9.0, 15.0])

    def loss(pp):
        ys = esdirk_adjoint_solve(rhs, C0, pp, (0.0, 15.0), t_obs)
        return jnp.sum(ys[:, 0] ** 2)

    g = jax.grad(loss)(p)[0]
    exact = sum(
        2 * math.exp(-k * ti) * (-ti * math.exp(-k * ti)) for ti in [2.0, 5.0, 9.0, 15.0]
    )
    assert jnp.isfinite(g)
    assert abs(float(g) - exact) / abs(exact) < 1e-5


@pytest.mark.parametrize("solve", [implicit_euler_adjoint_solve, esdirk_adjoint_solve])
def test_gradient_independent_of_max_steps(simple_network, solve):
    # The backward recurrence is bounded by the real accepted-step count, not the
    # allocated ``max_steps`` buffer: diffrax saves accepted steps contiguously
    # from index 0 and pads the tail, and the bounded loop never visits the
    # padding. So a generously-oversized ``max_steps`` (a loose upper bound) must
    # give the *bit-identical* gradient of a tightly-sized one -- the only effect
    # of the extra capacity is the never-traversed padding.
    rhs = _decay_rhs(simple_network)
    C0 = jnp.array([1.0, 0.0])
    p = simple_network.default_parameters()
    t_obs = jnp.array([2.0, 5.0, 9.0, 15.0])

    def loss(pp, max_steps):
        ys = solve(rhs, C0, pp, (0.0, 15.0), t_obs,
                   rtol=1e-8, atol=1e-10, max_steps=max_steps)
        return jnp.sum(ys[:, 0] ** 2)

    # Both bounds exceed the real step count (implicit Euler, first order, takes
    # the most), so both solves share the identical accepted-step trajectory and
    # differ only in never-traversed padding (~10x more in the loose case).
    g_tight = jax.grad(loss)(p, 20_000)[0]
    g_loose = jax.grad(loss)(p, 200_000)[0]
    assert jnp.isfinite(g_tight)
    assert float(g_tight) == float(g_loose)   # bit-identical, not merely close


def test_esdirk_equals_autodiff_through_same_solve(simple_network):
    # Same machine-precision guard as the implicit-Euler one, but for the
    # Kvaerno5 discrete adjoint: it must equal jax.grad through the identical
    # forced-step Kvaerno5 solve on a small network.
    rhs = _decay_rhs(simple_network)
    C0 = jnp.array([1.0, 0.0])
    p = simple_network.default_parameters()
    t_obs = jnp.array([1.0, 4.0, 8.0])
    rtol, atol = 1e-9, 1e-11

    def loss_stable(pp):
        return jnp.sum(
            esdirk_adjoint_solve(rhs, C0, pp, (0.0, 8.0), t_obs, rtol=rtol, atol=atol) ** 2
        )

    def loss_autodiff(pp):
        ctrl = diffrax.ClipStepSizeController(
            diffrax.PIDController(rtol=rtol, atol=atol), step_ts=t_obs
        )
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, a: rhs(t, y, a)),
            # Explicit root-finder tolerances (see note above) for diffrax 0.7.0.
            diffrax.Kvaerno5(
                root_finder=diffrax.VeryChord(rtol=rtol, atol=atol,
                                              norm=optimistix.max_norm)),
            0.0, 8.0, 1e-6, C0, args=pp, stepsize_controller=ctrl,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            saveat=diffrax.SaveAt(ts=t_obs), max_steps=100_000,
        )
        return jnp.sum(sol.ys ** 2)

    assert jnp.allclose(jax.grad(loss_stable)(p), jax.grad(loss_autodiff)(p),
                        rtol=1e-6, atol=1e-9)


@pytest.mark.validation
def test_esdirk_stiff_trajectory_matches_capped_kvaerno5():
    # The Kvaerno5 discrete adjoint on the stiff network: finite uncapped, and
    # matching the capped-Kvaerno5 jax-adjoint of the same forced-step forward.
    net = aquakin.load_network("wats_sewer_khalil_paper_balanced")
    cond = net.default_conditions(1)
    C0 = net.default_concentrations()
    p = net.default_parameters()
    fields = cond.fields
    rhs = lambda t, y, pp: net.dCdt(y, pp, fields, 0)
    t_obs = jnp.array([0.05, 0.1, 0.2, 0.3])
    si = net.species_index["S_SO4"]

    def loss(pp):
        ys = esdirk_adjoint_solve(rhs, C0, pp, (0.0, 0.3), t_obs,
                                  rtol=1e-7, atol=1e-10, max_steps=50_000)
        return jnp.sum(ys[:, si] ** 2) + 1e-3 * jnp.sum(ys ** 2)

    g = jax.jit(jax.grad(loss))(p)
    assert bool(jnp.all(jnp.isfinite(g)))

    def loss_ref(pp):
        ctrl = diffrax.ClipStepSizeController(
            diffrax.PIDController(rtol=1e-7, atol=1e-10, dtmax=3e-4), step_ts=t_obs
        )
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, a: rhs(t, y, a)), diffrax.Kvaerno5(),
            0.0, 0.3, 1e-6, C0, args=pp, stepsize_controller=ctrl,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            saveat=diffrax.SaveAt(ts=t_obs), max_steps=200_000,
        )
        return jnp.sum(sol.ys[:, si] ** 2) + 1e-3 * jnp.sum(sol.ys ** 2)

    g_ref = jax.jit(jax.grad(loss_ref))(p)
    rel = float(jnp.linalg.norm(g - g_ref) / (jnp.linalg.norm(g_ref) + 1e-30))
    assert rel < 1e-3


@pytest.mark.validation
def test_calibrate_stable_adjoint_matches_jax_adjoint():
    # End-to-end: a Khalil-model calibration with gradient="stable_adjoint"
    # (cap-free) must reach the same optimum as the existing capped-Kvaerno5
    # gradient="jax_adjoint" path. Synthetic recovery; compare the fitted params.
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
    obs = gen.solve(C0, params=p_true, t_span=span, t_eval=t_obs).C[
        :, [net.species_index[s] for s in obs_species]
    ]

    # Generous buffer: the backward skips padded slots (lax.cond), so the cost
    # tracks the actual Kvaerno5 step count, while the buffer stays large enough
    # for the forward across all params the optimiser explores.
    max_steps = 4000
    common = dict(observed_species=obs_species, loss="mse", laplace=False,
                  max_iter=150, tol=1e-9)
    r_ref = aquakin.calibrate(
        aquakin.BatchReactor(net, cond, rtol=rtol, atol=atol, dtmax=5e-4),
        C0, obs, t_obs, free, gradient="jax_adjoint", **common,
    )
    r_da = aquakin.calibrate(
        aquakin.BatchReactor(net, cond, rtol=rtol, atol=atol),
        C0, obs, t_obs, free, gradient="stable_adjoint",
        stable_adjoint_max_steps=max_steps, **common,
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


# --- time-dependent (non-autonomous) right-hand side ------------------------

def _forced_rhs(t, y, p):
    """A non-autonomous, time-forced scalar ODE: dy/dt = -p0 y + p1 sin(2 t).
    The p1 sensitivity is carried entirely by the explicit time dependence."""
    return jnp.array([-p[0] * y[0] + p[1] * jnp.sin(2.0 * t)])


def test_time_dependent_esdirk_gradient_matches_fd():
    """With ``time_dependent=True`` the discrete adjoint is exact for a
    non-autonomous RHS (time carried in the state); it matches central FD."""
    p0 = jnp.array([1.0, 0.5])
    y0 = jnp.array([1.0])
    teval = jnp.linspace(0.5, 4.0, 8)

    def loss(p):
        ys = esdirk_adjoint_solve(_forced_rhs, y0, p, (0.0, 4.0), teval,
                                  max_steps=20_000, time_dependent=True)
        return jnp.sum(ys[:, 0] ** 2)

    g = jax.grad(loss)(p0)
    assert jnp.all(jnp.isfinite(g))
    h = 1e-5
    fd = jnp.array([(loss(p0.at[i].add(h)) - loss(p0.at[i].add(-h))) / (2 * h)
                    for i in range(2)])
    assert g == pytest.approx(fd, rel=1e-3)


def test_time_dependent_implicit_euler_gradient_matches_fd():
    """Same exactness for the first-order implicit-Euler discrete adjoint."""
    p0 = jnp.array([1.0, 0.5])
    y0 = jnp.array([1.0])
    teval = jnp.linspace(0.5, 4.0, 8)

    def loss(p):
        ys = implicit_euler_adjoint_solve(_forced_rhs, y0, p, (0.0, 4.0), teval,
                                          max_steps=20_000, time_dependent=True)
        return jnp.sum(ys[:, 0] ** 2)

    g = jax.grad(loss)(p0)
    assert jnp.all(jnp.isfinite(g))
    h = 1e-5
    fd = jnp.array([(loss(p0.at[i].add(h)) - loss(p0.at[i].add(-h))) / (2 * h)
                    for i in range(2)])
    assert g == pytest.approx(fd, rel=5e-3)


def test_autonomous_default_is_wrong_for_time_forced_parameter():
    """The default autonomous assumption evaluates the field at a fixed time, so
    the gradient of a purely time-coupled parameter is wrong -- it is zeroed here
    because the forcing sin(2 t) is evaluated at t=0. This is the failure that
    ``time_dependent=True`` fixes (compare the test above)."""
    p0 = jnp.array([1.0, 0.5])
    y0 = jnp.array([1.0])
    teval = jnp.linspace(0.5, 4.0, 8)

    def loss(p):
        ys = esdirk_adjoint_solve(_forced_rhs, y0, p, (0.0, 4.0), teval,
                                  max_steps=20_000)   # autonomous (default)
        return jnp.sum(ys[:, 0] ** 2)

    g_auto = jax.grad(loss)(p0)
    assert float(g_auto[1]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("solve", [esdirk_adjoint_solve, implicit_euler_adjoint_solve])
def test_adjoint_solvers_validate_t_eval(solve):
    """Out-of-span or non-ascending save times must raise, not silently return
    inf / wrong values (the backward scan injects cotangents only at landed
    steps). Matches the reactor `solve` contract."""
    def rhs(t, y, p):
        return -p[0] * y

    y0 = jnp.array([1.0])
    p = jnp.array([1.0])
    with pytest.raises(ValueError, match="within t_span"):
        solve(rhs, y0, p, (0.0, 1.0), t_eval=jnp.array([0.5, 2.0]))
    with pytest.raises(ValueError, match="ascending"):
        solve(rhs, y0, p, (0.0, 1.0), t_eval=jnp.array([0.8, 0.3]))
    # A valid t_eval is unaffected (finite states at the requested times).
    ys = solve(rhs, y0, p, (0.0, 1.0), t_eval=jnp.array([0.5, 1.0]))
    assert jnp.all(jnp.isfinite(ys)) and ys.shape == (2, 1)


def test_esdirk_dense_stage_reconstruction_convention():
    """Pin the diffrax dense-output convention the ESDIRK backward relies on.

    ``esdirk_adjoint_solve`` reconstructs each step's stage values from the
    saved dense-output ``k`` via ``Y_i = y_n + sum_j A[i,j]*k_j`` (``k`` is the
    dt-SCALED stage derivative, so no explicit ``dt``). For the stiffly-accurate
    Kvaerno5 the last stage equals the step output, so the reconstruction must
    reproduce ``sol.ys`` exactly. If a diffrax upgrade changes the ``infos["k"]``
    scaling or alignment this fails loudly here rather than silently corrupting
    gradients.
    """
    import numpy as np
    from aquakin.integrate.discrete_adjoint import _esdirk_tableau

    solver = diffrax.Kvaerno5()
    A, b, _diag, _s = _esdirk_tableau(solver)
    A = np.asarray(A)
    assert np.allclose(b, A[-1])   # stiffly accurate: Y_s == y_{n+1}

    y0 = jnp.array([1.0, 2.0])
    term = diffrax.ODETerm(lambda t, y, a: -a * y)
    sol = diffrax.diffeqsolve(
        term, solver, 0.0, 3.0, 0.1, y0, args=0.7,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(steps=True, dense=True), max_steps=64,
    )
    ts = np.asarray(sol.ts)
    ys = np.asarray(sol.ys)
    ks = np.asarray(sol.interpolation.infos["k"])     # (max_steps, s, n)
    nstep = int(np.sum(np.isfinite(ts)))
    y_prev = np.concatenate([np.asarray(y0)[None, :], ys[:-1]], axis=0)
    for m in range(nstep):
        Y_last = (y_prev[m] + (A @ ks[m]))[-1]        # reconstruct, last stage
        assert np.allclose(Y_last, ys[m], rtol=1e-9, atol=1e-12), (
            f"step {m}: reconstructed last stage {Y_last} != step output {ys[m]}"
        )
