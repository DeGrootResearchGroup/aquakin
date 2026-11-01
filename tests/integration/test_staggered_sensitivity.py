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


def test_sensitivity_error_is_controlled():
    """The step controller bounds the sensitivity error, not only the state's:
    at a loose state tolerance and a tight sensitivity tolerance, S is still
    accurate, and tightening the sensitivity tolerance alone costs steps."""
    a, b, y0 = 2.0, 0.0, 1.0
    t_eval = jnp.array([0.0, 1.0, 2.0])

    def solve(sens_rtol, sens_atol):
        return staggered_sensitivity_solve(
            _linear_source,
            _dense_jac(_linear_source),
            jnp.array([y0]),
            jnp.array([a, b]),
            jnp.array([0]),
            0.0,
            2.0,
            t_eval,
            rtol=1e-3,
            atol=1e-3,
            sens_rtol=sens_rtol,
            sens_atol=sens_atol,
        )

    tight, loose = solve(1e-9, 1e-12), solve(1e-3, 1e-3)
    _, dy_da, _ = _linear_source_closed_form(a, b, y0, np.asarray(t_eval))
    assert np.max(np.abs(np.asarray(tight.S[:, 0, 0]) - dy_da)) < 1e-7
    assert int(tight.stats.accepted) > 3 * int(loose.stats.accepted)


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


@pytest.mark.parametrize("rows", [None, 3])
def test_anderson_acceleration_removes_a_few_slow_directions(monkeypatch, rows):
    """The stage iteration on a linear residual ``A x - b`` whose factored matrix
    is wrong in two directions: the plain iteration contracts slowly along them,
    while the Anderson-accelerated one removes them in a few iterations. Both
    must land on the exact solution; the accelerated one in fewer iterations.
    ``rows`` exercises the row-by-row (sensitivity block) path."""
    from jax.scipy.linalg import lu_factor

    import aquakin.integrate.forward_solve as fs

    n = 20
    rng = np.random.default_rng(1)
    A = np.eye(n) + np.diag(rng.uniform(0.0, 5.0, n))
    # Factor M with I - M^{-1} A = T, T contracting by 0.7 and 0.85 along two
    # directions and exact elsewhere: M is wrong in exactly two directions.
    U = np.linalg.qr(rng.standard_normal((n, 2)))[0]
    T = U @ np.diag([0.7, 0.85]) @ U.T
    M = A @ np.linalg.inv(np.eye(n) - T)
    shape = (n,) if rows is None else (rows, n)
    b = rng.standard_normal(shape)
    exact = np.linalg.solve(A, b.T).T
    scale = jnp.full(shape, 1e-8)  # tight: KAPPA * 1e-8 absolute accuracy

    def run(depth):
        monkeypatch.setattr(fs, "_ANDERSON_DEPTH", depth)
        x, rate, conv, iterations = fs._chord(
            lambda x: (jnp.asarray(A) @ x.T).T - jnp.asarray(b),
            jnp.zeros(shape),
            lu_factor(jnp.asarray(M)),
            scale,
        )
        return np.asarray(x), bool(conv), int(iterations)

    plain = run(0)
    accelerated = run(2)
    assert accelerated[1]
    assert np.max(np.abs(accelerated[0] - exact)) < 1e-8
    assert accelerated[2] < plain[2]  # plain hits the cap or needs many more
    assert accelerated[2] <= 6


def test_step_factor_caps_growth_by_the_newton_contraction_rate():
    """The controller's growth limiter: a step whose Newton iteration contracted
    at ``rate`` grows by at most ``theta_target / rate`` however small its error
    (and shrinks when ``rate > theta_target``); a slowly varying, tiny error with
    a fast-converging Newton grows by ``maxfac``; a non-converged attempt shrinks
    hard."""
    c = StepControl(theta_target=0.3, maxfac=2.0)
    tiny = jnp.asarray(1e-8)
    assert float(c.factor(tiny, tiny, jnp.asarray(0.6), jnp.asarray(True))) == pytest.approx(0.5)
    assert float(c.factor(tiny, tiny, jnp.asarray(0.2), jnp.asarray(True))) == pytest.approx(1.5)
    assert float(c.factor(tiny, tiny, jnp.asarray(1e-4), jnp.asarray(True))) == pytest.approx(2.0)
    assert float(c.factor(tiny, tiny, jnp.asarray(0.01), jnp.asarray(False))) == pytest.approx(0.25)
    # Without the limiter the same attempt would have grown by maxfac.
    free = StepControl(theta_target=1e9, maxfac=2.0)
    assert float(free.factor(tiny, tiny, jnp.asarray(0.6), jnp.asarray(True))) == pytest.approx(2.0)


def _forced_decay_closed_form(a, t):
    """y(t) of y' = -a (y - cos t), y(0) = 1."""
    return (a**2 * jnp.cos(t) + a * jnp.sin(t) + jnp.exp(-a * t)) / (a**2 + 1.0)


def test_limiter_prevents_newton_divergence():
    """End to end: with a zero Jacobian the chord iteration on the stiff forced
    decay ``y' = -a (y - cos t)`` is a plain fixed-point iteration contracting at
    ``h*gamma*a``, while the error controller would let the step grow well past
    ``1 / (gamma*a)`` along the smooth forced solution. Without the limiter the
    step grows until the iteration diverges; the limiter holds the contraction
    near ``theta_target`` and no attempt diverges. Both give the right
    sensitivity (the Jacobian only steers), and every rejection is counted once,
    as either non-converged or error-test."""
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
