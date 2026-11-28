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

import diffrax
import jax
import jax.numpy as jnp
import optimistix
import pytest

import aquakin
from aquakin import OptimizerConfig
from aquakin.integrate.discrete_adjoint import (
    esdirk_adjoint_solve,
    implicit_euler_adjoint_solve,
)


def _decay_rhs(simple_model):
    fields = aquakin.SpatialConditions.uniform(1, T=293.15).fields
    return lambda t, y, p: simple_model.dCdt(y, p, fields, 0)


def test_analytic_decay_gradient(simple_model):
    # A -> B, dA/dt = -k A, A(t) = A0 e^{-kt}. Loss = A(T)^2, so
    # dLoss/dk = 2 A(T) (dA/dk) = 2 e^{-kT} (-T e^{-kT}).
    rhs = _decay_rhs(simple_model)
    C0 = jnp.array([1.0, 0.0])
    p = simple_model.default_parameters()
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


def test_trajectory_loss_gradient(simple_model):
    # Loss over several observation times: L = sum_i A(t_i)^2, with the closed
    # form dL/dk = sum_i 2 A(t_i) (-t_i A(t_i)), A(t)=e^{-kt}.
    rhs = _decay_rhs(simple_model)
    C0 = jnp.array([1.0, 0.0])
    p = simple_model.default_parameters()
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


def test_t_eval_returns_states_at_times(simple_model):
    # With t_eval the solve returns the state at each observation time, matching
    # a plain diffrax solve sampled at those times.
    rhs = _decay_rhs(simple_model)
    C0 = jnp.array([1.0, 0.0])
    p = simple_model.default_parameters()
    t_obs = jnp.array([1.0, 3.0, 7.0])
    ys = implicit_euler_adjoint_solve(rhs, C0, p, (0.0, 7.0), t_obs, rtol=1e-9, atol=1e-11)
    assert ys.shape == (3, 2)
    k = float(p[0])
    assert jnp.allclose(ys[:, 0], jnp.exp(-k * t_obs), atol=1e-5, rtol=1e-4)


def test_equals_autodiff_through_same_solve(simple_model):
    # Permanent guard: the hand-written discrete adjoint computes the SAME
    # gradient that JAX autodiff would, when autodiff can run. On a small
    # (non-stiff) model, differentiate the identical implicit-Euler solve --
    # same forced-step controller, same tolerances -- both ways and require
    # machine-precision agreement. (Only the integrator's adjoint is hand-coded;
    # the model derivatives are autodiff in both. This pins that they match.)
    rhs = _decay_rhs(simple_model)
    C0 = jnp.array([1.0, 0.0])
    p = simple_model.default_parameters()
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


def test_gradient_wrt_y0_matches_closed_form(simple_model):
    # L = A(T)^2 + B(T)^2 with A = A0 e^{-kT}, B = B0 + A0 (1 - e^{-kT}), so at
    # (A0, B0) = (1, 0): dL/dA0 = 2 A0 (e^{-2kT} + (1 - e^{-kT})^2) and
    # dL/dB0 = 2 B(T) = 2 A0 (1 - e^{-kT}).
    rhs = _decay_rhs(simple_model)
    p = simple_model.default_parameters()
    k = float(p[0])
    T = 5.0

    def loss(C0):
        return jnp.sum(
            implicit_euler_adjoint_solve(rhs, C0, p, (0.0, 5.0)) ** 2
        )

    g = jax.grad(loss)(jnp.array([1.0, 0.0]))
    decayed = math.exp(-k * T)
    exact = [2.0 * (decayed**2 + (1.0 - decayed) ** 2), 2.0 * (1.0 - decayed)]
    # First-order implicit Euler at the default tolerances: ~3e-4 relative.
    assert [float(x) for x in g] == pytest.approx(exact, rel=1e-3)


def test_esdirk_analytic_trajectory_gradient(simple_model):
    # The high-order (Kvaerno5) discrete adjoint on the closed-form decay
    # trajectory loss; Kvaerno5's 5th order makes the primal -- and so the
    # gradient -- tighter than implicit Euler at the same tolerance.
    rhs = _decay_rhs(simple_model)
    C0 = jnp.array([1.0, 0.0])
    p = simple_model.default_parameters()
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
def test_gradient_with_t0_in_t_eval(simple_model, solve):
    # A t_eval that INCLUDES t0 exercises the discrete adjoint's idx==0 branch:
    # the observation at t0 is y0 itself, so its cotangent flows straight to y0
    # (via ybar0_obs), not through any backward step. The parameter and y0
    # gradients must both stay finite and match the closed form.
    rhs = _decay_rhs(simple_model)
    C0 = jnp.array([1.0, 0.0])
    p = simple_model.default_parameters()
    k = float(p[0])
    t_obs = jnp.array([0.0, 5.0, 15.0])               # t0 included

    def loss(pp, c0):
        ys = solve(rhs, c0, pp, (0.0, 15.0), t_obs, rtol=1e-9, atol=1e-11)
        return jnp.sum(ys[:, 0] ** 2)

    gp = jax.grad(loss, argnums=0)(p, C0)[0]
    gy = jax.grad(loss, argnums=1)(p, C0)
    # dL/dk = sum_i 2 A(t_i)(-t_i A(t_i)); the t0 term is 0 (t=0), so it matches
    # the no-t0 sum -- but the t0 observation still drives the y0 gradient.
    exact_k = sum(2 * math.exp(-k * ti) * (-ti * math.exp(-k * ti))
                  for ti in [0.0, 5.0, 15.0])
    assert jnp.isfinite(gp) and jnp.all(jnp.isfinite(gy))
    assert abs(float(gp) - exact_k) / abs(exact_k) < 1e-4
    # dL/dA0 = sum_i 2 A0 e^{-2 k t_i}: the t0 term contributes 2 A(0) = 2 on its
    # own, the later observations the rest through the trajectory. B never
    # feeds A, so dL/dB0 is exactly zero.
    exact_y0 = sum(2 * math.exp(-2 * k * ti) for ti in [0.0, 5.0, 15.0])
    assert float(gy[0]) == pytest.approx(exact_y0, rel=1e-4)
    assert float(gy[1]) == 0.0


@pytest.mark.parametrize("solve", [implicit_euler_adjoint_solve, esdirk_adjoint_solve])
def test_gradient_independent_of_max_steps(simple_model, solve):
    # The backward recurrence is bounded by the real accepted-step count, not the
    # allocated ``max_steps`` buffer: diffrax saves accepted steps contiguously
    # from index 0 and pads the tail, and the bounded loop never visits the
    # padding. So a generously-oversized ``max_steps`` (a loose upper bound) must
    # give the *bit-identical* gradient of a tightly-sized one -- the only effect
    # of the extra capacity is the never-traversed padding.
    rhs = _decay_rhs(simple_model)
    C0 = jnp.array([1.0, 0.0])
    p = simple_model.default_parameters()
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


def test_esdirk_equals_autodiff_through_same_solve(simple_model):
    # Same machine-precision guard as the implicit-Euler one, but for the
    # Kvaerno5 discrete adjoint: it must equal jax.grad through the identical
    # forced-step Kvaerno5 solve on a small model.
    rhs = _decay_rhs(simple_model)
    C0 = jnp.array([1.0, 0.0])
    p = simple_model.default_parameters()
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
def test_esdirk_stiff_trajectory_matches_capped_kvaerno5_and_low_memory(stiff_sewer_model):
    # The Kvaerno5 discrete adjoint on the stiff model: finite uncapped, and
    # matching the capped-Kvaerno5 jax-adjoint of the same forced-step forward.
    # The low_memory stage recompute must match the saved-stage gradient here
    # too: the per-stage Newton converges through the same well-conditioned
    # I - dt*gamma*J the forward inverts. The tiny residual vs the saved-stage
    # path is the forward root-finder tolerance vs the machine-precision
    # recompute of the stages. (One test, so the saved-stage gradient is
    # computed once for both comparisons.)
    net = stiff_sewer_model
    cond = net.default_conditions(1)
    C0 = net.default_concentrations()
    p = net.default_parameters()
    fields = cond.fields
    rhs = lambda t, y, pp: net.dCdt(y, pp, fields, 0)
    t_obs = jnp.array([0.05, 0.1, 0.2, 0.3])
    si = net.species_index["S_SO4"]

    def loss(pp, low_memory=False):
        ys = esdirk_adjoint_solve(rhs, C0, pp, (0.0, 0.3), t_obs,
                                  rtol=1e-7, atol=1e-10, max_steps=50_000,
                                  low_memory=low_memory)
        return jnp.sum(ys[:, si] ** 2) + 1e-3 * jnp.sum(ys ** 2)

    g = jax.jit(jax.grad(loss))(p)
    assert bool(jnp.all(jnp.isfinite(g)))

    g_low = jax.jit(jax.grad(lambda pp: loss(pp, True)))(p)
    assert bool(jnp.all(jnp.isfinite(g_low)))
    rel_low = float(jnp.linalg.norm(g_low - g) / (jnp.linalg.norm(g) + 1e-30))
    assert rel_low < 1e-5

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


def _closed_form_decay_fit(t, obs, k0):
    """The least-squares ``k`` of the closed-form decay ``A = e^{-kt}``,
    ``B = 1 - e^{-kt}`` (``A0 = 1``) to ``obs[:, (A, B)]``, by Newton on the
    stationarity condition with the exact first and second ``k`` derivatives."""
    k = k0
    for _ in range(50):
        e = jnp.exp(-k * t)
        r = jnp.stack([e - obs[:, 0], 1.0 - e - obs[:, 1]], axis=1)
        dr = jnp.stack([-t * e, t * e], axis=1)
        d2r = jnp.stack([t**2 * e, -(t**2) * e], axis=1)
        step = jnp.sum(r * dr) / jnp.sum(dr * dr + r * d2r)
        k = k - step
    return float(k)


@pytest.mark.parametrize(
    ("integrator", "solver_cls", "factormax"),
    [
        (None, diffrax.Kvaerno3, 3.0),                                     # the defaults
        (aquakin.IntegratorConfig(order=5, factormax=None), diffrax.Kvaerno5,
         diffrax.PIDController(rtol=1e-3, atol=1e-6).factormax),          # diffrax's own
        (aquakin.IntegratorConfig(solver=diffrax.Kvaerno5(), factormax=7.0),
         diffrax.Kvaerno5, 7.0),                                           # solver= beats order=3
    ],
    ids=["default", "order5_no_factormax", "explicit_kvaerno5_factormax7"],
)
def test_calibrate_stable_adjoint_runs_the_reactor_solver(
        integrator, solver_cls, factormax, simple_model, monkeypatch):
    """``calibrate``'s stable adjoint integrates with the ESDIRK the reactor's
    ``IntegratorConfig`` selects: the solver the discrete adjoint builds (whose
    tableau its backward reads) and the PID step controller it wraps are
    recorded as they are built. The solver is Kvaerno3 by default and Kvaerno5
    for ``order=5`` or an explicit ``solver=``; the controller carries the
    reactor's ``factormax`` (3 by default, diffrax's own when ``None``). The fitted
    rate constant matches the closed-form least-squares optimum of the decay
    model, so the gradient the fit followed is the right one for each solver.

    The fitted value alone cannot tell the solvers apart (each fits to far
    inside the tolerance), so the solver type is what fails if the reactor's
    order or solver is dropped."""
    from aquakin.integrate import discrete_adjoint as da

    built, controllers = [], []
    real, real_controller = da.build_implicit_solver, da.build_step_controller

    def recording(*a, **k):
        built.append(real(*a, **k))
        return built[-1]

    def recording_controller(*a, **k):
        controllers.append(real_controller(*a, **k))
        return controllers[-1]

    monkeypatch.setattr(da, "build_implicit_solver", recording)
    monkeypatch.setattr(da, "build_step_controller", recording_controller)

    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    C0 = jnp.array([1.0, 0.0])
    t_obs = jnp.linspace(1.0, 30.0, 12)
    # Noisy synthetic data (fixed perturbation), so the optimum is not the
    # generating k and the fit has to follow the gradient to find it.
    e = jnp.exp(-0.13 * t_obs)
    wiggle = 0.01 * jnp.sin(1.7 * t_obs)
    obs = jnp.stack([e + wiggle, 1.0 - e - wiggle], axis=1)
    integrator_kw = {} if integrator is None else {"integrator": integrator}
    reactor = aquakin.BatchReactor(simple_model, cond, settings=aquakin.SolverSettings(
        rtol=1e-8, atol=1e-11, **integrator_kw,
    ))

    res = aquakin.calibrate(
        reactor, C0, obs, t_obs, ["A_to_B.k"], observed_species=["A", "B"],
        loss="mse", laplace=False, use_priors=False,
        optimizer=OptimizerConfig(max_iter=200, tol=1e-12),
        diff=aquakin.DifferentiationConfig(method="stable"),
    )

    assert built and all(type(s) is solver_cls for s in built)
    assert controllers and all(c.factormax == factormax for c in controllers)
    # rtol=1e-8 solve: the fitted k agrees with the exact optimum to a decade
    # or two above the solve tolerance.
    k_exact = _closed_form_decay_fit(t_obs, obs, 0.13)
    assert res.params_named["A_to_B.k"] == pytest.approx(k_exact, rel=1e-6)


@pytest.mark.validation
def test_calibrate_stable_adjoint_matches_jax_adjoint(stiff_sewer_model):
    # End-to-end: a stiff-sewer-model calibration with gradient="stable_adjoint"
    # (cap-free) must reach the same optimum as the existing capped
    # gradient="jax_adjoint" path; both run the reactor's default order
    # (Kvaerno3). Synthetic recovery; compare the fitted params.

    net = stiff_sewer_model
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
    gen = aquakin.BatchReactor(net, cond, settings=aquakin.SolverSettings(
        rtol=1e-9,
        atol=1e-11,
    ))
    obs = gen.solve(C0, params=p_true, t_span=span, t_eval=t_obs).C[
        :, [net.species_index[s] for s in obs_species]
    ]

    # Generous buffer: the backward skips padded slots (lax.cond), so the cost
    # tracks the actual step count, while the buffer stays large enough
    # for the forward across all params the optimiser explores.
    max_steps = 4000
    common = dict(observed_species=obs_species, loss="mse", laplace=False,
                  optimizer=OptimizerConfig(max_iter=150, tol=1e-9))
    r_ref = aquakin.calibrate(
        aquakin.BatchReactor(net, cond, settings=aquakin.SolverSettings(
            rtol=rtol,
            atol=atol,
            integrator=aquakin.IntegratorConfig(dtmax=5e-4),
        )),
        C0, obs, t_obs, free,
        diff=aquakin.DifferentiationConfig(mode="reverse", method="through_solve"),
        **common,
    )
    r_da = aquakin.calibrate(
        aquakin.BatchReactor(net, cond, settings=aquakin.SolverSettings(
            rtol=rtol,
            atol=atol,
        )),
        C0, obs, t_obs, free,
        diff=aquakin.DifferentiationConfig(method="stable",
                                           adjoint_max_steps=max_steps),
        **common,
    )
    assert r_ref.converged and r_da.converged
    v_ref = jnp.array([r_ref.params_named[n] for n in free])
    v_da = jnp.array([r_da.params_named[n] for n in free])
    rel = float(jnp.max(jnp.abs(v_da - v_ref) / jnp.abs(v_ref)))
    assert rel < 5e-3


@pytest.mark.validation
def test_stiff_trajectory_loss_matches_capped(stiff_sewer_model):
    # A multi-observation (trajectory) loss -- the calibration shape -- must be
    # finite uncapped and match the capped reference using the same forced-step
    # forward solve.
    net = stiff_sewer_model
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


# --- low_memory: stage-recompute (checkpointing) backward path ---------------
#
# ``low_memory=True`` does not store the forward's dense-output stage buffer
# (~s x the trajectory); it recomputes each step's stages by Newton in the
# backward pass. The reconstruction is exact, so the gradient must match the
# default saved-stage path, and a tableau outside the singly-diagonal ESDIRK
# shape the recompute assumes must fall back (with a warning) rather than silently
# using it.


@pytest.mark.parametrize("solver", [diffrax.Kvaerno5, diffrax.Kvaerno3])
def test_low_memory_matches_saved_stage_gradient(simple_model, solver):
    # On the (linear) decay model the stage equations solve exactly, so the
    # recompute matches the saved-stage path to machine precision -- for both the
    # default Kvaerno5 and the cheaper Kvaerno3.
    rhs = _decay_rhs(simple_model)
    C0 = jnp.array([1.0, 0.0])
    p = simple_model.default_parameters()
    t_obs = jnp.array([1.0, 4.0, 8.0])
    rtol, atol = 1e-9, 1e-11

    def loss(pp, low_memory):
        return jnp.sum(
            esdirk_adjoint_solve(rhs, C0, pp, (0.0, 8.0), t_obs, solver=solver(),
                                 rtol=rtol, atol=atol, low_memory=low_memory) ** 2
        )

    g_saved = jax.grad(lambda pp: loss(pp, False))(p)
    g_low = jax.grad(lambda pp: loss(pp, True))(p)
    assert bool(jnp.all(jnp.isfinite(g_low)))
    assert jnp.allclose(g_low, g_saved, rtol=1e-7, atol=1e-9)


def test_is_singly_diagonal_esdirk_guard():
    # The guard accepts the KV3/KV5 tableaus and rejects shapes the stage
    # recompute does not assume (non-constant gamma; no explicit first stage).
    import numpy as np

    from aquakin.integrate.discrete_adjoint import _esdirk_tableau, _is_singly_diagonal_esdirk

    for solver in (diffrax.Kvaerno5(), diffrax.Kvaerno3()):
        _A, _b, diag, _s = _esdirk_tableau(solver)
        assert _is_singly_diagonal_esdirk(diag)
    assert not _is_singly_diagonal_esdirk(np.array([0.0, 0.3, 0.4]))   # non-constant
    assert not _is_singly_diagonal_esdirk(np.array([0.25, 0.25, 0.25]))  # implicit 1st


def test_low_memory_falls_back_with_warning(simple_model, monkeypatch):
    # A tableau outside the assumed shape must warn and fall back to the
    # saved-stage path (still correct), not silently take the recompute.
    import aquakin.integrate.discrete_adjoint as da
    rhs = _decay_rhs(simple_model)
    C0 = jnp.array([1.0, 0.0])
    p = simple_model.default_parameters()
    t_obs = jnp.array([1.0, 8.0])

    monkeypatch.setattr(da, "_is_singly_diagonal_esdirk", lambda diag: False)
    with pytest.warns(RuntimeWarning, match="singly-diagonal ESDIRK"):
        ys = esdirk_adjoint_solve(rhs, C0, p, (0.0, 8.0), t_obs, low_memory=True)
    ys_ref = esdirk_adjoint_solve(rhs, C0, p, (0.0, 8.0), t_obs, low_memory=False)
    assert jnp.allclose(ys, ys_ref)
