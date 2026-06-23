"""Cap-free reverse-mode gradients through the plant solve (``gradient="stable_adjoint"``).

A stiff plant (the BSM2 water line plus the ADM1 digester) cannot be
differentiated by the default reverse adjoint over a useful horizon: the
through-the-solve adjoint of the stiff implicit method returns non-finite values
unless the integrator step is capped, and capping fails the whole-plant solve.
``plant.solve(gradient="stable_adjoint")`` forms the gradient instead with the
hand-written discrete adjoint (the forward is a robust adaptive ESDIRK solve, the
reverse a per-step transposed solve over the saved trajectory), which is finite
at any step size.

The headline check is a gradient that flows from a *water-line* observation back
through the digester, the activated-sludge to anaerobic-digestion interface, and
the recycle to an ADM1 (digester) parameter -- a cross-network gradient -- and
matches a central finite difference. The cheap API-guard tests do not integrate.
"""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import CSTRUnit, Plant
from aquakin.plant.bsm import (
    bsm1_warm_start,
    bsm2_warm_start,
    build_bsm1,
    load_bsm1_influent,
)
from aquakin.plant.bsm.bsm2 import (
    build_bsm2,
    bsm2_constant_influent,
    bsm2_parameters,
)
from aquakin.plant.influent import InfluentSeries


def _bsm2_plant():
    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    y0 = bsm2_warm_start(plant)
    return asm1, adm1, plant, y0


# --- cheap API guards (no integration) -------------------------------------

def test_invalid_gradient_raises():
    _asm1, _adm1, plant, y0 = _bsm2_plant()
    with pytest.raises(ValueError, match="jax_adjoint.*stable_adjoint"):
        plant.solve(t_span=(0.0, 1.0), y0=y0, gradient="not_a_mode")


def test_stable_adjoint_rejects_adjoint_and_dtmax():
    """``stable_adjoint`` controls its own adjoint and steps; passing the
    diffrax adjoint or a dtmax cap alongside it is a usage error."""
    _asm1, _adm1, plant, y0 = _bsm2_plant()
    with pytest.raises(ValueError, match="do not also pass"):
        plant.solve(t_span=(0.0, 1.0), y0=y0, gradient="stable_adjoint", dtmax=1e-2)
    with pytest.raises(ValueError, match="do not also pass"):
        plant.solve(t_span=(0.0, 1.0), y0=y0, gradient="stable_adjoint",
                    adjoint=diffrax.DirectAdjoint())


# --- fast small-network correctness (a single-CSTR plant) ------------------

def _single_cstr_plant(net):
    """A one-unit plant: a CSTR on the toy decay network fed a constant flow.

    Small and non-stiff, so the stable-adjoint gradient through the *plant*
    solve can be checked against the standard ``jax_adjoint`` gradient and a
    finite difference in the fast gate -- the BSM2 versions of this check are
    validation-marked because the whole plant is expensive to integrate.
    """
    plant = Plant("single_cstr")
    plant.add_unit(CSTRUnit(
        name="tank", network=net, volume=100.0,
        input_port_names=["inlet"], conditions={"T": 293.15},
    ))
    influent = InfluentSeries(
        t=jnp.asarray([0.0, 100.0]), Q=jnp.asarray([10.0, 10.0]),
        C=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]), network=net,
    )
    plant.add_influent("feed", influent, to="tank.inlet")
    return plant


def test_stable_adjoint_plant_gradient_matches_jax_adjoint_and_fd():
    """On a small single-CSTR plant the cap-free stable-adjoint gradient equals
    the standard through-the-solve (jax_adjoint) gradient and a central FD."""
    net = aquakin.load_network_from_file("tests/fixtures/simple_network.yaml")
    plant = _single_cstr_plant(net)
    base = net.default_parameters()
    gidx = plant.parameter_index("simple_decay.A_to_B.k")
    theta0 = float(base[gidx])
    T = 40.0
    teval = jnp.array([T])

    def g(theta, gradient):
        p = base.at[gidx].set(theta)
        sol = plant.solve(t_span=(0.0, T), t_eval=teval, params=p,
                          gradient=gradient)
        return sol.C_named("tank", "B")[-1]   # product at the outlet

    # Forward primal agrees regardless of the gradient backend (the two paths
    # take slightly different adaptive step grids, so this is to solver tol).
    assert float(g(theta0, "stable_adjoint")) == pytest.approx(
        float(g(theta0, "jax_adjoint")), rel=1e-6)

    g_stable = float(jax.grad(lambda th: g(th, "stable_adjoint"))(theta0))
    g_jax = float(jax.grad(lambda th: g(th, "jax_adjoint"))(theta0))
    assert np.isfinite(g_stable)
    assert g_stable != 0.0
    assert g_stable == pytest.approx(g_jax, rel=1e-4)

    h = theta0 * 1e-3
    fd = (float(g(theta0 + h, "stable_adjoint"))
          - float(g(theta0 - h, "stable_adjoint"))) / (2.0 * h)
    assert g_stable == pytest.approx(fd, rel=1e-3)


def test_stable_adjoint_initial_state_gradient_matches_jax_adjoint():
    """The cap-free stable-adjoint gradient w.r.t. the INITIAL STATE y0 (not just
    parameters) is finite and equals the standard jax_adjoint gradient.

    Regression for issue #420: the default per-component absolute tolerance is
    derived from y0 (``default_atol(y0, ...)``); under a gradient w.r.t. y0 that
    traced tolerance was baked into the integrator's step controller and escaped
    the discrete-adjoint custom-VJP forward (which re-runs diffrax.diffeqsolve) as
    a leaked tracer. ``default_atol`` now ``stop_gradient``s the floor (it is a
    non-differentiable solver tolerance), so the initial-condition gradient flows
    -- the direction the standard adjoint already handled."""
    net = aquakin.load_network_from_file("tests/fixtures/simple_network.yaml")
    plant = _single_cstr_plant(net)
    params = plant.default_parameters()
    y0 = plant.initial_state()
    T = 40.0

    def loss(y0_, gradient):
        sol = plant.solve(t_span=(0.0, T), t_eval=jnp.array([T]),
                          params=params, y0=y0_, gradient=gradient)
        return jnp.sum(sol.state ** 2)

    g_stable = np.asarray(jax.grad(lambda y: loss(y, "stable_adjoint"))(y0))
    g_jax = np.asarray(jax.grad(lambda y: loss(y, "jax_adjoint"))(y0))
    assert np.all(np.isfinite(g_stable))
    assert np.any(g_stable != 0.0)
    assert np.allclose(g_stable, g_jax, rtol=1e-3, atol=1e-9)


# --- the cross-interface gradient (slow: integrates the whole plant) -------

def _solve_kwargs():
    # The warm-started forward takes ~205 adaptive steps over a few days, so a
    # small max_steps suffices. Under gradient="stable_adjoint" max_steps also
    # sizes the backward scan's trajectory buffer, so keeping it tight is what
    # keeps the reverse pass cheap.
    #
    # colored_jacobian=False (not the "auto" default): these heavy BSM2 tests
    # check gradient *correctness* (vs FD / jax_adjoint), which the dense backward
    # does; forcing dense keeps them off the (much slower to COMPILE) colored
    # backward and skips the auto build-time measurement. Coloring itself is
    # covered by the fast BSM1 tests (test_stable_adjoint_colored_jacobian_*).
    return dict(rtol=1e-5, atol=1e-3, max_steps=2_000, colored_jacobian=False)


@pytest.mark.validation
@pytest.mark.heavy
def test_stable_adjoint_forward_matches_jax_adjoint():
    """The stable-adjoint forward primal equals the standard (jax_adjoint) solve;
    both integrate the same RHS with Kvaerno5, so they agree closely."""
    _asm1, _adm1, plant, y0 = _bsm2_plant()
    T = 3.0
    teval = jnp.array([T])
    a = plant.solve(t_span=(0.0, T), t_eval=teval, y0=y0, **_solve_kwargs())
    b = plant.solve(t_span=(0.0, T), t_eval=teval, y0=y0, gradient="stable_adjoint",
                    **_solve_kwargs())
    for unit, sp in (("tank1", "SNO"), ("tank5", "SNH"), ("digester", "S_gas_ch4")):
        assert float(b.C_named(unit, sp)[-1]) == pytest.approx(
            float(a.C_named(unit, sp)[-1]), rel=1e-3)


@pytest.mark.validation
@pytest.mark.heavy
def test_stable_adjoint_cross_interface_gradient_matches_fd():
    """Gradient of a water-line output with respect to an ADM1 (digester) rate,
    through the interface and the recycle, is finite and matches central FD."""
    asm1, adm1, plant, y0 = _bsm2_plant()
    base = bsm2_parameters(asm1, adm1)
    gidx = plant.parameter_index("adm1.k_m_ac")   # acetate-uptake max rate
    theta0 = float(base[gidx])
    T = 3.0

    def g(theta):
        p = base.at[gidx].set(theta)
        sol = plant.solve(t_span=(0.0, T), t_eval=jnp.array([T]), params=p, y0=y0,
                          gradient="stable_adjoint", **_solve_kwargs())
        return sol.C_named("tank1", "SNO")[-1]   # water-line nitrate

    grad = float(jax.grad(g)(theta0))
    assert np.isfinite(grad)
    # A digester rate genuinely moves the water line through the reject recycle.
    assert grad != 0.0

    h = theta0 * 1e-3
    fd = (float(g(theta0 + h)) - float(g(theta0 - h))) / (2.0 * h)
    # The discrete adjoint is the exact gradient of the forward solve; it agrees
    # with the central difference to the finite-difference truncation/solver floor.
    # The default adaptive recycle resolution (custom_root) composes with the
    # discrete adjoint exactly (the sibling dM/dtheta test pins it to ~1e-13);
    # the FD floor for this cross-network, recycle-coupled, stiff gradient sits
    # around a few 1e-3, so the tolerance is the FD floor, not a gradient error.
    assert grad == pytest.approx(fd, rel=5e-3)


@pytest.mark.validation
@pytest.mark.heavy
def test_stable_adjoint_flow_setpoint_gradient_preserves_dM_dtheta():
    """The gradient w.r.t. a FLOW-SETPOINT parameter (the RAS recycle flow) is
    unchanged by the cached-recycle-map optimisation -- the ``dM/dtheta`` guard
    for the carve-out in issue #366.

    The recycle concentration map ``M`` depends on the parameters only through
    the flow setpoints. The cap-free path caches ``M`` once and reuses it on the
    *primal* RHS (the forward solve + every backward ``df/dy`` Jacobian, skipping
    the per-call probe), while the ``df/dtheta`` vjp uses the map-recomputing
    ``rhs`` -- so ``dM/dtheta`` is captured exactly. Disabling the cache (the
    ``_recycle_map_constant = False`` path probes ``M`` on every call -- the
    pre-#366 behaviour, the textbook discrete adjoint with no carve-out) must give
    the same gradient. A wrong split would drop the ``dM/dtheta`` term, so the
    cached gradient would differ by O(1). The kinetic-param gradient (above)
    cannot catch this (``dM/dtheta = 0`` there).

    With the default adaptive recycle resolution ``M`` is only the affine
    warm-start seed -- the fixed point the ``custom_root`` converges to is
    M-independent -- so the cached and probed paths agree to floating-point
    rounding (~1e-13, the warm-start arithmetic-order difference) rather than
    bit-for-bit; the tight relative tolerance still catches an O(1) dropped
    ``dM/dtheta`` term."""
    asm1 = aquakin.load_network("asm1")
    plant = build_bsm1(asm1)
    plant.add_influent("influent", load_bsm1_influent("dry", asm1))
    y0 = bsm1_warm_start(plant)
    base = plant.default_parameters()
    gidx = plant.parameter_index("underflow_split.ras")   # RAS recycle flow
    theta0 = float(base[gidx])
    T = 0.3
    kw = dict(rtol=1e-6, atol=1e-3, max_steps=20_000, colored_jacobian=False)

    def g(theta):
        p = base.at[gidx].set(theta)
        sol = plant.solve(t_span=(0.0, T), t_eval=jnp.array([0.15, T]), params=p,
                          y0=y0, gradient="stable_adjoint", **kw)
        return jnp.sum(sol.state ** 2)

    # A concrete solve sets the state-invariant-map flag (BSM1's recycle map is
    # constant), so the cached-map primal is exercised.
    _ = g(theta0)
    assert plant._recycle_map_constant is True

    plant._jit_cache.clear()
    grad_cached = float(jax.grad(g)(theta0))            # cached-map primal (#366)
    assert np.isfinite(grad_cached)
    # RAS genuinely moves the plant through the recycle (so dM/dtheta != 0): if it
    # were 0 the guard would be vacuous.
    assert grad_cached != 0.0

    plant._recycle_map_constant = False                 # probe M every call
    plant._jit_cache.clear()
    grad_probed = float(jax.grad(g)(theta0))            # pre-#366 path
    # Agree to float rounding (adaptive recycle: M only warm-starts, so the
    # result is M-independent); still catches an O(1) dropped dM/dtheta term.
    assert grad_cached == pytest.approx(grad_probed, rel=1e-9)


@pytest.mark.validation
@pytest.mark.heavy
def test_stable_adjoint_colored_jacobian_matches_dense():
    """``colored_jacobian=True`` colors the per-step ``df/dy`` Jacobian build in
    the stable_adjoint **backward** pass (its dominant cost for a large plant).
    The colored Jacobian equals the dense one on its superset sparsity pattern,
    so the gradient must match the dense-Jacobian gradient. Guards the colored
    backward path: a missed pattern entry would change the gradient by O(1) in a
    component, which this catches; the float-order difference of the exact case
    is ~1e-15 here."""
    asm1 = aquakin.load_network("asm1")
    plant = build_bsm1(asm1)
    plant.add_influent("influent", load_bsm1_influent("dry", asm1))
    y0 = bsm1_warm_start(plant)
    base = plant.default_parameters()
    gidx = plant.parameter_index("asm1.muH")
    theta0 = float(base[gidx])
    T = 0.3
    kw = dict(rtol=1e-6, atol=1e-3, max_steps=20_000)

    def g(theta, colored):
        p = base.at[gidx].set(theta)
        sol = plant.solve((0.0, T), t_eval=jnp.array([0.15, T]), params=p, y0=y0,
                          gradient="stable_adjoint", colored_jacobian=colored, **kw)
        return jnp.sum(sol.state ** 2)

    # A concrete solve derives + guards the colored backward Jacobian builder.
    _ = g(theta0, True)
    builder = plant._colored_adjoint_builder
    assert builder is not None and builder[2] is True       # guard passed (ok)
    assert builder[1] < plant._total_state_size             # fewer colors than states

    g_dense = float(jax.grad(lambda th: g(th, False))(theta0))
    g_colored = float(jax.grad(lambda th: g(th, True))(theta0))
    assert np.isfinite(g_colored)
    assert g_colored != 0.0
    # Equal to the dense-Jacobian gradient (the colored Jacobian is exact on the
    # superset pattern; only float summation order differs).
    assert g_colored == pytest.approx(g_dense, rel=1e-8)


@pytest.mark.validation
@pytest.mark.heavy
def test_stable_adjoint_colored_jacobian_auto_off_for_small_plant():
    """``colored_jacobian="auto"`` (the default) measures the colored vs dense
    build time and enables coloring only when it pays. On a small plant (BSM1) the
    colored build is *slower* than dense, so auto picks dense -- and the gradient
    is then the dense gradient. Guards the auto decision and the accessor."""
    asm1 = aquakin.load_network("asm1")
    plant = build_bsm1(asm1)
    plant.add_influent("influent", load_bsm1_influent("dry", asm1))
    y0 = bsm1_warm_start(plant)
    base = plant.default_parameters()
    gidx = plant.parameter_index("asm1.muH")
    theta0 = float(base[gidx])
    T = 0.3
    kw = dict(rtol=1e-6, atol=1e-3, max_steps=20_000)

    def g(theta, colored):
        p = base.at[gidx].set(theta)
        sol = plant.solve((0.0, T), t_eval=jnp.array([0.15, T]), params=p, y0=y0,
                          gradient="stable_adjoint", colored_jacobian=colored, **kw)
        return jnp.sum(sol.state ** 2)

    # A concrete solve under the default "auto" derives the coloring and measures
    # the build speedup; on BSM1 the colored build is the slower one.
    assert plant.colored_jacobian_decision() is None        # not decided yet
    _ = g(theta0, "auto")
    choice, ratio = plant.colored_jacobian_decision()
    assert choice == "dense"
    assert ratio < plant._COLORED_BACKWARD_MARGIN           # colored build not cheaper

    # auto picked dense, so its gradient equals the explicit-dense gradient.
    g_auto = float(jax.grad(lambda th: g(th, "auto"))(theta0))
    g_dense = float(jax.grad(lambda th: g(th, False))(theta0))
    assert np.isfinite(g_auto) and g_auto != 0.0
    assert g_auto == pytest.approx(g_dense, rel=1e-10)


@pytest.mark.validation
@pytest.mark.heavy
def test_auto_gradient_defaults_to_stable_adjoint():
    """With the default ``gradient="auto"`` (nothing passed), a stiff plant
    gradient is finite and matches the explicit stable-adjoint gradient -- the
    auto router sends the differentiated solve down the cap-free path, while a
    plain forward solve still uses the fast jax_adjoint path."""
    asm1, adm1, plant, y0 = _bsm2_plant()
    base = bsm2_parameters(asm1, adm1)
    gidx = plant.parameter_index("adm1.k_m_ac")
    theta0 = float(base[gidx])
    T = 3.0

    def g(theta, gradient):
        p = base.at[gidx].set(theta)
        sol = plant.solve(t_span=(0.0, T), t_eval=jnp.array([T]), params=p, y0=y0,
                          gradient=gradient, **_solve_kwargs())
        return sol.C_named("tank1", "SNO")[-1]

    auto = float(jax.grad(lambda th: g(th, "auto"))(theta0))
    explicit = float(jax.grad(lambda th: g(th, "stable_adjoint"))(theta0))
    assert np.isfinite(auto)
    assert auto == pytest.approx(explicit, rel=1e-6)

    # A concrete forward solve under the same default is unchanged (fast path).
    fwd_auto = float(g(theta0, "auto"))
    fwd_jax = float(g(theta0, "jax_adjoint"))
    assert fwd_auto == pytest.approx(fwd_jax, rel=1e-6)


@pytest.mark.validation
@pytest.mark.heavy
def test_stable_adjoint_transient_influent_gradient_matches_fd():
    """Under a time-varying (diurnal-flow) influent the cross-interface gradient
    is still finite and matches central finite differences. The discrete adjoint
    carries the integration time in the state, so it is exact for the
    non-autonomous plant right-hand side, not only for a constant influent."""
    from aquakin.plant.bsm.bsm2 import BSM2_Q_REF
    from aquakin.plant.influent import InfluentSeries

    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    # A diurnal flow modulation makes the plant RHS explicitly time-dependent.
    c_const = bsm2_constant_influent(asm1).C[0]
    n = 120
    t_inf = jnp.linspace(0.0, 4.0, n)
    q_inf = BSM2_Q_REF * (1.0 + 0.3 * jnp.sin(2.0 * jnp.pi * t_inf))
    plant.add_influent(
        "feed",
        InfluentSeries(t=t_inf, Q=q_inf, C=jnp.tile(c_const, (n, 1)), network=asm1),
    )
    y0 = bsm2_warm_start(plant)
    base = bsm2_parameters(asm1, adm1)
    gidx = plant.parameter_index("adm1.k_m_ac")
    theta0 = float(base[gidx])
    # Two diurnal cycles is enough to make the RHS explicitly time-dependent and
    # exercise the non-autonomous adjoint; a longer horizon only multiplies the
    # (rtol=1e-7) step count -- and so the CI runtime and the backward-scan buffer
    # -- without testing anything new.
    T = 2.0

    # Tighter solver than the shared default, for an *accurate* finite-difference
    # reference. The stable-adjoint gradient is the exact gradient of the discrete
    # solve and is platform-stable; FD is the noisy side. Each g(theta+-h) re-runs
    # an adaptive solve whose step grid shifts discretely with theta, and at the
    # default atol=1e-3 that grid noise puts a ~theta-independent absolute error on
    # the central difference (it landed up to ~8% off across CPU/XLA builds).
    # Dropping to rtol=1e-7 / atol=1e-5 resolves the grid, so FD converges to the
    # gradient to ~0.1% (verified: FD -> grad to 0.00% as h shrinks). rel=2e-2 then
    # covers the residual platform spread with wide margin while still catching a
    # genuinely wrong gradient (sign, magnitude). max_steps also sizes the
    # stable-adjoint backward-scan buffer, so give the tighter solve headroom.
    kw = {**_solve_kwargs(), "rtol": 1e-7, "atol": 1e-5, "max_steps": 8_000}

    def g(theta):
        p = base.at[gidx].set(theta)
        sol = plant.solve(t_span=(0.0, T), t_eval=jnp.array([T]), params=p, y0=y0,
                          gradient="stable_adjoint", **kw)
        return sol.C_named("tank1", "SNO")[-1]

    grad = float(jax.grad(g)(theta0))
    assert np.isfinite(grad)
    assert grad != 0.0
    h = theta0 * 1e-2
    fd = (float(g(theta0 + h)) - float(g(theta0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=2e-2)


@pytest.mark.validation
@pytest.mark.heavy
def test_stable_adjoint_gradient_finite_through_full_param_vector():
    """A full-parameter reverse gradient (the calibration case) is finite, where
    the default through-the-solve adjoint is not without a step cap."""
    asm1, adm1, plant, y0 = _bsm2_plant()
    base = bsm2_parameters(asm1, adm1)
    T = 2.0

    def loss(p):
        sol = plant.solve(t_span=(0.0, T), t_eval=jnp.array([T]), params=p, y0=y0,
                          gradient="stable_adjoint", **_solve_kwargs())
        return jnp.sum(sol.state[-1] ** 2)

    g = jax.grad(loss)(base)
    assert g.shape == base.shape
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0)


@pytest.mark.validation
@pytest.mark.heavy
def test_stable_adjoint_solve_is_jittable():
    """The stable-adjoint plant solve can be wrapped in ``jax.jit``: its ``atol``
    no longer forces concretization. The jitted value and gradient match the
    eager ones. Jitting the calibration loss is what amortizes the (large) solve
    compile across optimizer iterations."""
    asm1, adm1, plant, y0 = _bsm2_plant()
    base = bsm2_parameters(asm1, adm1)
    gidx = plant.parameter_index("adm1.k_m_ac")
    T = 3.0
    teval = jnp.array([T])
    # A tight max_steps keeps the discrete-adjoint trajectory buffer -- and so the
    # peak memory of this jit-plus-gradient test -- small; the warm-started 3-day
    # solve takes far fewer than 600 steps.
    kw = dict(rtol=1e-5, atol=1e-3, max_steps=600, colored_jacobian=False)

    def g(theta):
        p = base.at[gidx].set(theta)
        sol = plant.solve(t_span=(0.0, T), t_eval=teval, params=p, y0=y0,
                          gradient="stable_adjoint", **kw)
        return sol.C_named("tank1", "SNO")[-1]

    theta0 = float(base[gidx])
    # value-and-gradient in one pass, eager and jitted, so the test compiles two
    # programs rather than four. The jitted pass compiling at all exercises the
    # atol concretization fix; its value and gradient must match the eager pass.
    f_e, g_e = jax.value_and_grad(g)(theta0)
    f_j, g_j = jax.jit(jax.value_and_grad(g))(theta0)
    assert np.isfinite(float(f_j)) and np.isfinite(float(g_j))
    assert float(f_j) == pytest.approx(float(f_e), rel=1e-6)
    assert float(g_j) == pytest.approx(float(g_e), rel=1e-6)


@pytest.mark.validation
@pytest.mark.heavy
def test_stable_adjoint_forward_solve_is_cached():
    """A repeat *forward* stable-adjoint solve reuses the compiled closure (the
    parameter-sweep case), while a traced call (a gradient through the solve)
    bypasses the cache: the discrete adjoint's ``custom_vjp`` must be traced
    directly into the outer computation, not routed through an inner ``jax.jit``
    under an outer reverse-mode pass."""
    asm1, adm1, plant, y0 = _bsm2_plant()
    base = bsm2_parameters(asm1, adm1)
    T = 3.0
    # Tight max_steps (3-day solve uses far fewer) to keep the adjoint buffer
    # and so this test's peak memory small.
    kw = dict(t_span=(0.0, T), t_eval=jnp.array([T]), y0=y0,
              gradient="stable_adjoint", rtol=1e-5, atol=1e-3, max_steps=600,
              colored_jacobian=False)

    def _sa_keys():
        return [k for k in plant._jit_cache if k[0] == "stable_adjoint"]

    a = plant.solve(params=base, **kw)
    assert len(_sa_keys()) == 1                       # one compiled stable solve
    cached = plant._jit_cache[_sa_keys()[0]]

    plant.solve(params=base * 1.01, **kw)             # different params, same sig
    assert plant._jit_cache[_sa_keys()[0]] is cached  # reused, not rebuilt
    c = plant.solve(params=base, **kw)                # same params -> same result
    assert float(c.C_named("tank1", "SNO")[-1]) == pytest.approx(
        float(a.C_named("tank1", "SNO")[-1]), rel=1e-10)

    # A traced (gradient) call adds no stable-adjoint cache entry.
    n_before = len(_sa_keys())
    jax.grad(lambda th: plant.solve(
        params=base.at[0].set(th), **kw).C_named("tank1", "SNO")[-1]
    )(float(base[0]))
    assert len(_sa_keys()) == n_before


@pytest.mark.validation
@pytest.mark.heavy
def test_stable_adjoint_colored_jacobian_flow_setpoint_matches_dense():
    """The intersection of the two backward features: ``colored_jacobian`` AND a
    FLOW-SETPOINT parameter (the RAS recycle flow). The primal/``rhs`` split puts
    the cached recycle map on every ``df/dy`` Jacobian build (now colored) while
    the ``df/dtheta`` vjp keeps the map-recomputing ``rhs`` -- so coloring the
    backward Jacobian must NOT drop the ``dM/dtheta`` term. The colored gradient
    w.r.t. RAS (where ``dM/dtheta != 0``) must equal the dense-Jacobian gradient.
    The kinetic-param colored test cannot catch a dropped ``dM/dtheta`` (it is 0
    there); this is the test that does."""
    asm1 = aquakin.load_network("asm1")
    plant = build_bsm1(asm1)
    plant.add_influent("influent", load_bsm1_influent("dry", asm1))
    y0 = bsm1_warm_start(plant)
    base = plant.default_parameters()
    gidx = plant.parameter_index("underflow_split.ras")     # RAS recycle flow
    theta0 = float(base[gidx])
    T = 0.3
    kw = dict(rtol=1e-6, atol=1e-3, max_steps=20_000)

    def g(theta, colored):
        p = base.at[gidx].set(theta)
        sol = plant.solve((0.0, T), t_eval=jnp.array([0.15, T]), params=p, y0=y0,
                          gradient="stable_adjoint", colored_jacobian=colored, **kw)
        return jnp.sum(sol.state ** 2)

    _ = g(theta0, True)                                     # build + guard colored
    builder = plant._colored_adjoint_builder
    assert builder is not None and builder[2] is True       # guard passed

    g_dense = float(jax.grad(lambda th: g(th, False))(theta0))
    g_colored = float(jax.grad(lambda th: g(th, True))(theta0))
    assert np.isfinite(g_colored)
    assert g_colored != 0.0                                 # RAS moves M (dM/dtheta != 0)
    assert g_colored == pytest.approx(g_dense, rel=1e-8)


@pytest.mark.slow
def test_stable_adjoint_accepts_kvaerno3_and_factormax():
    """``solver=``/``factormax=`` are supported on the ``stable_adjoint`` path.

    The discrete adjoint builds its backward from the forward solver's Butcher
    tableau generically, so a cheaper 4-stage ``Kvaerno3`` (and a ``factormax``
    cap on the PID step growth) is a valid forward, matching the optimized
    ``forward_fast`` configuration. The gradient stays finite and agrees with the
    default ``Kvaerno5`` discrete adjoint to the two methods' truncation
    difference (they differ in the realized step sequence, each exact for its
    own)."""
    asm1 = aquakin.load_network("asm1")
    plant = build_bsm1(asm1)
    plant.add_influent("influent", load_bsm1_influent("dry", asm1))
    y0 = bsm1_warm_start(plant)
    base = plant.default_parameters()
    gidx = plant.parameter_index("asm1.muA")
    theta0 = float(base[gidx])
    T = 0.5
    kw = dict(rtol=1e-4, atol=1e-3, max_steps=20_000)

    def g(theta, **solve_kw):
        p = base.at[gidx].set(theta)
        sol = plant.solve((0.0, T), t_eval=jnp.array([T]), params=p, y0=y0,
                          gradient="stable_adjoint", **kw, **solve_kw)
        return sol.C_named("tank5", "SNO")[-1]

    g5 = float(jax.grad(lambda th: g(th))(theta0))                  # Kvaerno5
    g3 = float(jax.grad(lambda th: g(th, solver=diffrax.Kvaerno3()))(theta0))
    g3f = float(jax.grad(
        lambda th: g(th, solver=diffrax.Kvaerno3(), factormax=3.0))(theta0))
    assert np.isfinite(g3) and np.isfinite(g3f)
    assert g3 != 0.0
    # Each is the exact discrete adjoint of its own forward solve, so they agree
    # only to the ESDIRK truncation difference between the realized step
    # sequences -- and the factormax-capped sequence is platform-sensitive (the
    # adaptive controller lands different steps on different hardware/BLAS). So
    # compare to the independent Kvaerno5 reference at 1e-2: tight enough that a
    # broken adjoint (off by tens of percent or sign) still fails, loose enough to
    # absorb the cross-discretization + platform step-sequence spread.
    assert g3 == pytest.approx(g5, rel=1e-2)
    assert g3f == pytest.approx(g5, rel=1e-2)


@pytest.mark.slow
def test_forward_paths_agree_no_config_drift():
    """The three forward integrators -- the ``jax_adjoint`` forward solve, the lean
    ``forward_fast`` solve, and the ``stable_adjoint`` *forward* pass -- all build
    their solver + step controller from the shared single-source-of-truth helpers
    (``build_implicit_solver`` / ``build_step_controller``), so they realize the
    same primal trajectory. If any path's per-step configuration (decoupled
    Newton, colored Jacobian, ESDIRK order, factormax) is changed without the
    others, the primals diverge and this fails -- the regression guard the
    silently-drifted adjoint forward never had.
    """
    asm1 = aquakin.load_network("asm1")
    plant = build_bsm1(asm1)
    plant.add_influent("influent", load_bsm1_influent("dry", asm1))
    y0 = bsm1_warm_start(plant)
    base = plant.default_parameters()
    T = 1.0
    kw = dict(rtol=1e-5, atol=1e-3, max_steps=50_000)
    teval = jnp.array([0.5, T])

    def final(**solve_kw):
        sol = plant.solve((0.0, T), t_eval=teval, params=base, y0=y0, **kw,
                          **solve_kw)
        return np.asarray(sol.state)

    jax_fwd = final(gradient="jax_adjoint")
    fast = final(forward_fast=True)
    stable_fwd = final(gradient="stable_adjoint")
    assert np.all(np.isfinite(jax_fwd))
    # Same primal to the shared integrator tolerance: the adjoint bookkeeping and
    # the lean forward_fast machinery do not change the realized trajectory.
    assert np.allclose(jax_fwd, fast, rtol=1e-3, atol=1e-3)
    assert np.allclose(jax_fwd, stable_fwd, rtol=1e-3, atol=1e-3)
