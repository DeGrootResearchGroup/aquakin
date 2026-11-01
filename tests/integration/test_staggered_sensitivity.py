"""Staggered forward-sensitivity solve and the plant's ``solve_sensitivity``.

``staggered_sensitivity_solve`` integrates the state and its sensitivity
``S = dy/dtheta`` on the lean Kvaerno3 loop: each implicit stage converges the
state, then solves the (linear) sensitivity stage with the same factorization of
``I - h*gamma*J`` and an exact Jacobian-vector product per parameter. These tests
pin the integrator against closed forms and the independent ``diffrax``
augmented solve, check that the Jacobian only steers the iterations and that the
sensitivity error is controlled, pin the step-size controller's contraction-rate
growth cap, and check the plant wiring: both correctors against a closed-form
plant sensitivity, and the compiled-solve cache (a second call does not
retrace).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.integrate.forward_sensitivity import augmented_forward_sensitivity
from aquakin.integrate.forward_solve import (
    StepControl,
    forward_solve,
    staggered_sensitivity_solve,
)
from aquakin.plant import CSTRUnit, Plant
from aquakin.plant.influent import InfluentSeries


def _dense_jac(rhs):
    return lambda t, y, theta: jax.jacfwd(lambda yy: rhs(t, yy, theta))(y)


def _assert_stats_consistent(stats):
    assert int(stats.attempted) == (
        int(stats.accepted) + int(stats.nonconverged) + int(stats.error_rejected)
    )


# --------------------------------------------------------------------------
# The integrator.
# --------------------------------------------------------------------------


def _linear_source(t, y, theta):
    return -theta[0] * y + theta[1]


def _linear_source_closed_form(a, b, y0, t):
    """y(t) and its derivatives w.r.t. a and b for y' = -a y + b."""
    e = np.exp(-a * t)
    y = b / a + (y0 - b / a) * e
    dy_da = -b / a**2 * (1.0 - e) - (y0 - b / a) * t * e
    dy_db = (1.0 - e) / a
    return y, dy_da, dy_db


def test_linear_source_matches_closed_form():
    """Two parameters with different values and effects, saved inside steps (the
    dense output) as well as at the end: a sensitivity column computed for the
    wrong parameter, a wrong stage equation, or a wrong dense output of S would
    all miss the closed form."""
    a, b, y0 = 2.0, 3.0, 1.0
    t_eval = jnp.linspace(0.0, 3.0, 13)
    res = staggered_sensitivity_solve(
        _linear_source,
        _dense_jac(_linear_source),
        jnp.array([y0]),
        jnp.array([a, b]),
        jnp.array([0, 1]),
        0.0,
        3.0,
        t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    y, dy_da, dy_db = _linear_source_closed_form(a, b, y0, np.asarray(t_eval))
    assert res.S.shape == (13, 1, 2)
    assert np.max(np.abs(np.asarray(res.ys[:, 0]) - y)) < 1e-8
    assert np.max(np.abs(np.asarray(res.S[:, 0, 0]) - dy_da)) < 1e-8
    assert np.max(np.abs(np.asarray(res.S[:, 0, 1]) - dy_db)) < 1e-8
    _assert_stats_consistent(res.stats)


def _robertson(t, y, k):
    y1, y2, y3 = y
    return jnp.array(
        [
            -k[0] * y1 + k[1] * y2 * y3,
            k[0] * y1 - k[1] * y2 * y3 - k[2] * y2**2,
            k[2] * y2**2,
        ]
    )


_ROB_K = jnp.array([0.04, 1e4, 3e7])
_ROB_Y0 = jnp.array([1.0, 0.0, 0.0])
_ROB_T = jnp.array([0.0, 1.0, 10.0, 40.0])


@pytest.fixture(scope="module")
def robertson_reference():
    """S of the stiff Robertson problem by the independent diffrax augmented
    [y; S] solve (Kvaerno5, block-arrow corrector), at a tight tolerance."""
    _, _, S = augmented_forward_sensitivity(
        _robertson,
        _ROB_Y0,
        _ROB_K,
        jnp.array([0, 1, 2]),
        t0=0.0,
        t1=40.0,
        t_eval=_ROB_T,
        rtol=1e-10,
        atol_y=jnp.full(3, 1e-14),
        shared_factor=True,
        order=5,
    )
    return np.asarray(S)


def _robertson_staggered(jac):
    return staggered_sensitivity_solve(
        _robertson,
        jac,
        _ROB_Y0,
        _ROB_K,
        jnp.array([0, 1, 2]),
        0.0,
        40.0,
        _ROB_T,
        rtol=1e-8,
        atol=1e-12,
    )


def _column_relative_error(S, ref):
    return np.max(np.abs(S - ref) / np.abs(ref).max(axis=(0, 1), keepdims=True))


def test_stiff_robertson_matches_augmented_solve(robertson_reference):
    """On the stiff, nonlinear Robertson problem the staggered sensitivity of all
    three rate constants matches the diffrax augmented solve."""
    res = _robertson_staggered(_dense_jac(_robertson))
    assert _column_relative_error(np.asarray(res.S), robertson_reference) < 1e-6
    _assert_stats_consistent(res.stats)


def test_jacobian_only_steers_the_iterations(robertson_reference):
    """With the factored Jacobian replaced by its diagonal -- badly wrong for
    Robertson, whose off-diagonal couplings are as large as its diagonal -- the
    sensitivity is unchanged to tolerance: every stage residual is an exact
    Jacobian-vector product at the stage value, so the factor only steers the
    iterations. (A sensitivity stage solved in one shot with the factored J,
    instead of iterated on the exact residual, fails this.)"""

    def diagonal_jac(t, y, k):
        return jnp.diag(jnp.diag(_dense_jac(_robertson)(t, y, k)))

    res = _robertson_staggered(diagonal_jac)
    assert _column_relative_error(np.asarray(res.S), robertson_reference) < 1e-6


_DECAY_A, _DECAY_T = 2.0, jnp.array([0.0, 1.0, 2.0])


def _decay_sensitivity(sens_rtol, sens_atol, free_idx=(0,), sens_error_control=True):
    """``y' = -a y`` at a loose state tolerance, with the sensitivity to ``a``
    (``free_idx=(0,)``) or none (``()``) at the given sensitivity tolerance."""
    return staggered_sensitivity_solve(
        _linear_source,
        _dense_jac(_linear_source),
        jnp.array([1.0]),
        jnp.array([_DECAY_A, 0.0]),
        jnp.asarray(free_idx, dtype=int),
        0.0,
        2.0,
        _DECAY_T,
        rtol=1e-3,
        atol=1e-3,
        sens_rtol=sens_rtol,
        sens_atol=sens_atol,
        sens_error_control=sens_error_control,
    )


def test_sensitivity_error_is_controlled():
    """The step controller bounds the sensitivity error, not only the state's:
    at a loose state tolerance and a tight sensitivity tolerance, S is still
    accurate, and tightening the sensitivity tolerance alone costs steps."""
    tight, loose = _decay_sensitivity(1e-9, 1e-12), _decay_sensitivity(1e-3, 1e-3)
    _, dy_da, _ = _linear_source_closed_form(_DECAY_A, 0.0, 1.0, np.asarray(_DECAY_T))
    assert np.max(np.abs(np.asarray(tight.S[:, 0, 0]) - dy_da)) < 1e-7
    assert int(tight.stats.accepted) > 3 * int(loose.stats.accepted)


def test_sensitivity_left_out_of_the_error_test_keeps_the_state_steps():
    """With ``sens_error_control=False`` (CVODES ``errconS = false``) the steps
    are the state's alone: a tight sensitivity tolerance takes exactly the steps
    and state of the solve with no sensitivity (with it in the test, the same
    tolerance costs over three times the steps -- the test above), and S is
    still the converged sensitivity of those steps."""
    with_S = _decay_sensitivity(1e-9, 1e-12, sens_error_control=False)
    state = _decay_sensitivity(1e-9, 1e-12, ())
    assert [int(v) for v in with_S.stats] == [int(v) for v in state.stats]
    assert np.array_equal(np.asarray(with_S.ys), np.asarray(state.ys))
    _, dy_da, _ = _linear_source_closed_form(_DECAY_A, 0.0, 1.0, np.asarray(_DECAY_T))
    assert np.max(np.abs(np.asarray(with_S.S[:, 0, 0]) - dy_da)) < 1e-2


def test_no_sensitivity_parameters_is_the_state_solve():
    """k = 0 returns an empty sensitivity block and the plain state solve."""
    t_eval = jnp.linspace(0.0, 2.0, 5)
    theta = jnp.array([2.0, 3.0])
    jac = _dense_jac(_linear_source)
    res = staggered_sensitivity_solve(
        _linear_source,
        jac,
        jnp.array([1.0]),
        theta,
        jnp.zeros((0,), int),
        0.0,
        2.0,
        t_eval,
        rtol=1e-6,
        atol=1e-9,
    )
    ys = forward_solve(
        _linear_source, jac, jnp.array([1.0]), theta, 0.0, 2.0, t_eval, rtol=1e-6, atol=1e-9
    )
    assert res.S.shape == (5, 1, 0)
    assert np.array_equal(np.asarray(res.ys), np.asarray(ys))


def _linear_stage(monkeypatch, contraction, rows, depth):
    """Run the stage iteration at Anderson depth ``depth`` on a linear residual
    ``A x - b`` whose factored matrix ``M`` satisfies ``I - M^{-1} A = T``, ``T``
    scaling two directions by ``contraction`` and exact elsewhere -- ``M`` is
    wrong in exactly two directions. Returns the iterate, convergence flag and
    iteration count, and the exact solution. ``rows`` exercises the row-by-row
    (sensitivity block) path."""
    from jax.scipy.linalg import lu_factor

    import aquakin.integrate.forward_solve as fs

    n = 20
    rng = np.random.default_rng(1)
    A = np.eye(n) + np.diag(rng.uniform(0.0, 5.0, n))
    U = np.linalg.qr(rng.standard_normal((n, 2)))[0]
    T = U @ np.diag(contraction) @ U.T
    M = A @ np.linalg.inv(np.eye(n) - T)
    shape = (n,) if rows is None else (rows, n)
    b = rng.standard_normal(shape)
    monkeypatch.setattr(fs, "_ANDERSON_DEPTH", depth)
    x, rate, conv, iterations = fs._stage_solve(
        lambda x: (jnp.asarray(A) @ x.T).T - jnp.asarray(b),
        jnp.zeros(shape),
        lu_factor(jnp.asarray(M)),
        jnp.full(shape, 1e-8),  # tight: KAPPA * 1e-8 absolute accuracy
    )
    return np.asarray(x), bool(conv), int(iterations), np.linalg.solve(A, b.T).T


@pytest.mark.parametrize("rows", [None, 3])
def test_anderson_acceleration_removes_a_few_slow_directions(monkeypatch, rows):
    """With the factored matrix wrong in two directions that contract by 0.7
    and 0.85, the plain iteration stalls at the iteration cap, while the
    Anderson-accelerated one (the default depth) reaches the exact solution in
    a few iterations."""
    import aquakin.integrate.forward_solve as fs

    assert fs._ANDERSON_DEPTH == 2  # the shipped default this test exercises
    *_, plain_iterations, _ = _linear_stage(monkeypatch, [0.7, 0.85], rows, 0)
    x, conv, iterations, exact = _linear_stage(monkeypatch, [0.7, 0.85], rows, 2)
    assert conv
    assert np.max(np.abs(x - exact)) < 1e-8
    assert iterations < plain_iterations  # the plain iteration hits the cap
    assert iterations <= 6


def test_step_factor_caps_growth_by_the_newton_contraction_rate():
    """The controller's growth limiter: a step whose Newton iteration contracted
    at ``rate`` grows by at most ``theta_target / rate`` however small its error
    (and shrinks when ``rate > theta_target``); a slowly varying, tiny error with
    a fast-converging Newton grows by ``maxfac``; a non-converged attempt shrinks
    hard."""
    tiny = jnp.asarray(1e-8)

    def grow(control, rate, conv=True):
        return float(
            control.factor(tiny, tiny, jnp.asarray(rate), jnp.asarray(conv), jnp.asarray(False))
        )

    c = StepControl(theta_target=0.3, maxfac=2.0)
    assert grow(c, 0.6) == pytest.approx(0.5)
    assert grow(c, 0.2) == pytest.approx(1.5)
    assert grow(c, 1e-4) == pytest.approx(2.0)
    assert grow(c, 0.01, conv=False) == pytest.approx(0.25)
    # Without the limiter the same attempt would have grown by maxfac.
    assert grow(StepControl(theta_target=1e9, maxfac=2.0), 0.6) == pytest.approx(2.0)


def _forced_decay_closed_form(a, t):
    """y(t) of y' = -a (y - cos t), y(0) = 1."""
    return (a**2 * jnp.cos(t) + a * jnp.sin(t) + jnp.exp(-a * t)) / (a**2 + 1.0)


def test_limiter_prevents_newton_divergence(monkeypatch):
    """End to end: with a zero Jacobian the plain (unaccelerated) chord
    iteration on the stiff forced decay ``y' = -a (y - cos t)`` is a fixed-point
    iteration contracting at ``h*gamma*a``, while the error controller would let
    the step grow well past ``1 / (gamma*a)`` along the smooth forced solution.
    Without the limiter the step grows until the iteration diverges; the limiter
    holds the contraction near ``theta_target`` and no attempt diverges. Both give
    the right sensitivity (the Jacobian only steers), and every rejection is
    counted once, as either non-converged or error-test. (The default Anderson
    acceleration solves this one-dimensional iteration in two steps whatever the
    step size, so the limiter is pinned on the plain iteration.)"""
    import aquakin.integrate.forward_solve as fs

    monkeypatch.setattr(fs, "_ANDERSON_DEPTH", 0)
    a = 50.0
    t_eval = jnp.array([0.0, 1.0, 5.0, 10.0])

    def solve(theta_target):
        return staggered_sensitivity_solve(
            lambda t, y, th: -th[0] * (y - jnp.cos(t)),
            lambda t, y, th: jnp.zeros((1, 1)),
            jnp.array([1.0]),
            jnp.array([a]),
            jnp.array([0]),
            0.0,
            10.0,
            t_eval,
            rtol=1e-6,
            atol=1e-10,
            control=StepControl(theta_target=theta_target),
        )

    limited, free = solve(0.3), solve(1e9)
    assert int(free.stats.nonconverged) > 0
    assert int(limited.stats.nonconverged) < int(free.stats.nonconverged)
    dy_da = jax.vmap(jax.grad(_forced_decay_closed_form), in_axes=(None, 0))(a, t_eval)
    for res in (limited, free):
        _assert_stats_consistent(res.stats)
        assert np.max(np.abs(np.asarray(res.S[:, 0, 0]) - np.asarray(dy_da))) < 1e-8


def test_a_solve_started_at_its_steady_state_takes_long_steps():
    """Started at its steady state, a stage's starting guess is already
    converged and every correction after the first is round-off, so the ratio
    of two of them is no contraction rate. Accepting the first correction on its
    size keeps that noise from declaring divergence or holding the step down:
    a stiff system (rates 1 to 1000) held at its steady state for 1000 time
    units takes a few dozen steps. Read as rates, the round-off ratios cost it
    thousands. The right-hand side vanishes at ``ystar`` only to round-off --
    its constant is summed in another order than ``A @ y`` -- as a plant's
    does at a computed steady state."""
    A = jnp.array(
        [
            [-1.0, 0.3, 0.0, 0.1],
            [0.2, -10.0, 1.0, 0.0],
            [0.0, 3.0, -100.0, 5.0],
            [0.5, 0.0, 40.0, -1000.0],
        ]
    )
    ystar = jnp.array([1.0 / 3.0, 2.0 / 7.0, 5.0 / 11.0, 3.0 / 13.0])
    c = -jnp.array(
        [sum(float(A[i, j]) * float(ystar[j]) for j in reversed(range(4))) for i in range(4)]
    )
    assert float(jnp.max(jnp.abs(A @ ystar + c))) > 0.0  # round-off, not zero

    def rhs(t, y, th):
        return th[0] * (A @ y + c)

    res = staggered_sensitivity_solve(
        rhs,
        _dense_jac(rhs),
        ystar,
        jnp.array([1.0]),
        jnp.array([0]),
        0.0,
        1000.0,
        jnp.array([0.0, 1000.0]),
        rtol=1e-6,
        atol=1e-9,
    )
    _assert_stats_consistent(res.stats)
    assert int(res.stats.attempted) < 100
    assert int(res.stats.nonconverged) == 0
    np.testing.assert_allclose(np.asarray(res.ys[-1]), np.asarray(ystar), rtol=1e-9)


def test_dtmax_caps_the_step():
    """``dtmax`` bounds every step: a decay that would take a handful of steps
    takes at least ``span / dtmax`` of them."""
    res = staggered_sensitivity_solve(
        _linear_source,
        _dense_jac(_linear_source),
        jnp.array([1.0]),
        jnp.array([1.0, 0.0]),
        jnp.array([0]),
        0.0,
        2.0,
        jnp.array([0.0, 2.0]),
        rtol=1e-3,
        atol=1e-6,
        control=StepControl(dtmax=0.01),
    )
    assert int(res.stats.accepted) >= 200


def test_no_step_growth_right_after_a_rejection():
    """The adaptive loop on a scripted attempt whose error rises as
    ``(h / 0.1)**6`` -- faster than the controller's ``h**3`` model -- from a
    first step of 0.2: it is rejected, and the shrunk retry is accepted with an
    error small enough for the controller's ``maxfac`` growth. That step,
    accepted right after the rejection, does not grow the next one; the next
    one, accepted after an accepted step with the same error, doubles."""
    import aquakin.integrate.forward_solve as fs

    def attempt(t, z, h):
        n = z[0].astype(int)
        zero = jnp.zeros_like(z)
        z1 = z.at[n + 1].set(h).at[0].add(1.0)
        return fs._Attempt(z1, (h / 0.1) ** 6, jnp.asarray(0.0), jnp.asarray(True), zero, zero)

    control = StepControl(h0=0.2, pi_beta=0.0, maxfac=2.0)
    zs, stats = fs._integrate(attempt, jnp.zeros(64), 0.0, 0.5, jnp.array([0.0, 0.5]), control)
    assert int(stats.error_rejected) >= 1
    accepted = np.asarray(zs[-1, 1 : 1 + int(zs[-1, 0])])
    assert accepted[0] == pytest.approx(0.2 * 0.9 * 64.0 ** (-1 / 3))  # the shrunk retry
    assert accepted[1] == pytest.approx(accepted[0])
    assert accepted[2] == pytest.approx(2.0 * accepted[1])


def _robertson_stage_iterations(monkeypatch, time_scale):
    """Total state stage iterations of the Robertson solve with time measured in
    units ``time_scale`` times smaller (every rate divided by it). A power of two
    scales every product in the step exactly, so a test that does not depend on
    the time unit gives the same iterations bit for bit."""
    import aquakin.integrate.forward_solve as fs

    total = []
    stage_solve = fs._stage_solve

    def counted(residual, guess, lu, scale):
        out = stage_solve(residual, guess, lu, scale)
        jax.debug.callback(lambda it: total.append(int(it)), out[3])
        return out

    monkeypatch.setattr(fs, "_stage_solve", counted)

    def rhs(t, y, k):
        return _robertson(t / time_scale, y, k) / time_scale

    ys = forward_solve(
        rhs,
        _dense_jac(rhs),
        _ROB_Y0,
        _ROB_K,
        0.0,
        40.0 * time_scale,
        _ROB_T * time_scale,
        rtol=1e-6,
        atol=1e-10,
        control=StepControl(h0=1e-3 * time_scale),
    )
    jax.effects_barrier()
    return sum(total), np.asarray(ys)


def test_stage_test_does_not_depend_on_the_time_unit(monkeypatch):
    """The stage convergence test reads the stage-value increment ``h*G*dk``, so
    it is the same in any time unit: the Robertson solve in units 1024 times
    smaller takes the same stage iterations and gives the same solution. (A test
    on the slope increment ``dk`` against the state tolerance would be 1024 times
    looser there, and iterate less.)"""
    days, scaled = (_robertson_stage_iterations(monkeypatch, s) for s in (1.0, 1024.0))
    assert days[0] == scaled[0]
    np.testing.assert_allclose(days[1], scaled[1], rtol=1e-12, atol=1e-15)


# --------------------------------------------------------------------------
# The plant wiring (a one-CSTR plant with a closed-form sensitivity).
# --------------------------------------------------------------------------

_OPERATING = [
    {"kind": "influent_concentration", "port": "feed", "species": "A"},
    {"kind": "influent_flow", "port": "feed"},
]
_Y0 = jnp.array([0.2, 0.0])


@pytest.fixture
def cstr_plant():
    """A CSTR (V = 100) on the A -> B decay model fed Q = 10 of A = 1."""
    net = aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")
    plant = Plant("single_cstr")
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    influent = InfluentSeries(
        t=jnp.asarray([0.0, 100.0]),
        Q=jnp.asarray([10.0, 10.0]),
        C=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
        model=net,
    )
    plant.add_influent("feed", influent, to="tank.inlet")
    return plant


def _cstr_A(theta, t):
    """Tank A(t) for rate constant k, influent-A scale s and flow scale q:
    dA/dt = D (s - A) - k A with D = 0.1 q and A(0) = 0.2."""
    k, s, q = theta
    lam = 0.1 * q + k
    return 0.1 * q * s / lam * (1.0 - jnp.exp(-lam * t)) + _Y0[0] * jnp.exp(-lam * t)


@pytest.mark.parametrize("corrector", ["staggered", "simultaneous"])
def test_plant_sensitivity_matches_closed_form(cstr_plant, corrector):
    """Both correctors give the sensitivity of the tank concentration to a
    kinetic parameter and to two operating scales (influent concentration and
    flow), against the closed form differentiated by ``jax.jacfwd``."""
    t_eval = jnp.linspace(0.0, 10.0, 6)
    base = cstr_plant.default_parameters()
    ts, ys, S = cstr_plant.solve_sensitivity(
        base,
        ["simple_decay.A_to_B.k"],
        operating=_OPERATING,
        t_span=(0.0, 10.0),
        t_eval=t_eval,
        y0=_Y0,
        rtol=1e-8,
        atol=1e-10,
        corrector=corrector,
    )
    ref = jax.jacfwd(_cstr_A)(jnp.array([0.1, 1.0, 1.0]), t_eval)  # (n_t, 3)
    assert S.shape == (6, 2, 3)
    assert np.max(np.abs(np.asarray(S[:, 0, :]) - np.asarray(ref))) < 1e-7
    assert np.allclose(np.asarray(ts), np.asarray(t_eval))


def test_plant_sensitivity_compiles_once(cstr_plant):
    """A second call with new parameter values, initial state, ``t_span``
    endpoints, ``t_eval`` values and tolerances reuses the compiled solve: the
    plant right-hand side is not traced again. A new parameter set does build a
    new solve (the counter's positive control)."""
    traces = []
    rhs = cstr_plant._rhs

    def counted(*args, **kwargs):
        traces.append(1)  # runs on tracing only, not per compiled call
        return rhs(*args, **kwargs)

    cstr_plant._rhs = counted
    base = cstr_plant.default_parameters()
    wrt = ["simple_decay.A_to_B.k"]

    def solve(params, y0, t1, rtol, wrt=wrt):
        out = cstr_plant.solve_sensitivity(
            params,
            wrt,
            operating=_OPERATING[:1],
            t_span=(0.0, t1),
            t_eval=jnp.linspace(0.0, t1, 4),
            y0=y0,
            rtol=rtol,
            atol=1e-9,
        )
        return np.asarray(out[2])

    first = solve(base, _Y0, 5.0, 1e-6)
    assert len(traces) > 0
    traces.clear()
    second = solve(base * 1.5, _Y0 * 2.0, 8.0, 1e-7)
    assert traces == []
    assert not np.allclose(first, second)  # the new inputs did reach the solve
    solve(base, _Y0, 5.0, 1e-6, wrt=[])
    assert len(traces) > 0


def test_plant_staggered_sensitivity_rejects_tracing(cstr_plant):
    """The staggered corrector is forward-only: differentiating through it is a
    clear error pointing at the differentiable corrector."""
    base = cstr_plant.default_parameters()

    def loss(scale):
        return cstr_plant.solve_sensitivity(
            base * scale, ["simple_decay.A_to_B.k"], t_span=(0.0, 1.0), y0=_Y0
        )[2].sum()

    with pytest.raises(TypeError, match="corrector='simultaneous'"):
        jax.grad(loss)(1.0)


def test_plant_sensitivity_rejects_unknown_corrector(cstr_plant):
    with pytest.raises(ValueError, match="corrector must be one of"):
        cstr_plant.solve_sensitivity(
            cstr_plant.default_parameters(),
            ["simple_decay.A_to_B.k"],
            t_span=(0.0, 1.0),
            y0=_Y0,
            corrector="dense",
        )


def test_plant_sensitivity_error_control_option(cstr_plant):
    """``sens_error_control=False`` reaches the staggered solve: with a tight
    sensitivity tolerance it takes the steps of the solve with no sensitivity,
    while the default (the same tolerance, called first, so a compile cache that
    ignored the option would hand back its program) takes more. The simultaneous
    corrector refuses it."""
    base = cstr_plant.default_parameters()
    kw = dict(t_span=(0.0, 5.0), y0=_Y0, rtol=1e-4, atol=1e-6, return_stats=True)
    wrt = ["simple_decay.A_to_B.k"]

    def steps(wrt, **extra):
        *_, stats = cstr_plant.solve_sensitivity(base, wrt, sens_rtol=1e-10, **kw, **extra)
        return [int(v) for v in stats]

    controlled = steps(wrt)
    uncontrolled = steps(wrt, sens_error_control=False)
    assert uncontrolled == steps([])
    assert controlled[1] > uncontrolled[1]
    with pytest.raises(ValueError, match="sens_error_control=False is supported"):
        cstr_plant.solve_sensitivity(
            base,
            wrt,
            t_span=(0.0, 1.0),
            y0=_Y0,
            corrector="simultaneous",
            sens_error_control=False,
        )


def test_plant_sensitivity_returns_step_counts(cstr_plant):
    """``return_stats=True`` adds the staggered solve's step counts (and
    ``None`` for the simultaneous corrector, whose counts diffrax does not
    split)."""
    base = cstr_plant.default_parameters()
    kw = dict(t_span=(0.0, 5.0), y0=_Y0, return_stats=True)
    *_, stats = cstr_plant.solve_sensitivity(base, ["simple_decay.A_to_B.k"], **kw)
    assert int(stats.accepted) > 0
    _assert_stats_consistent(stats)
    *_, none = cstr_plant.solve_sensitivity(
        base, ["simple_decay.A_to_B.k"], corrector="simultaneous", **kw
    )
    assert none is None
