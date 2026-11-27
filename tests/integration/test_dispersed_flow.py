"""DispersedFlowReactor and TanksInSeries against their closed forms.

Every check is against an analytical solution: the travelling Gaussian of a
tracer pulse, the Wehner--Wilhelm steady state, the plug-flow and stirred-tank
limits, the tanks-in-series cascade and its Erlang RTD, and a central finite
difference for the gradients. The model is the A -> B first-order decay
(``k = 0.1``); ``k = 0`` makes A a non-reactive tracer.
"""

import math
import re

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import DispersedFlowReactor, SolverSettings, TanksInSeries
from aquakin.core.time_expr import TimeExpressionError
from aquakin.utils.rtd import fit_dispersion, fit_tanks_in_series
from tests.conftest import FIXTURES

COND = aquakin.SpatialConditions.uniform(T=293.15)
K = 0.1
LENGTH, VELOCITY = 1.0, 0.05
TAU = LENGTH / VELOCITY  # k tau = 2
TRACER = jnp.array([0.0])
INLET = {"A": 1.0}


def wehner_wilhelm(k_tau, d):
    """Outlet fraction of first-order decay in a closed dispersed vessel."""
    a = math.sqrt(1.0 + 4.0 * k_tau * d)
    numerator = 4.0 * a * math.exp(1.0 / (2.0 * d))
    return numerator / (
        (1 + a) ** 2 * math.exp(a / (2.0 * d)) - (1 - a) ** 2 * math.exp(-a / (2.0 * d))
    )


def steady_outlet(model, d, n_cells=50):
    reactor = DispersedFlowReactor(model, COND, LENGTH, VELOCITY, n_cells, dispersion_number=d)
    return float(reactor.steady_state(INLET).C_named("A")[0])


# --- the advection--dispersion scheme ---------------------------------------


def test_tracer_pulse_matches_the_gaussian_and_converges_at_second_order(simple_model):
    """A Gaussian pulse at Péclet ``u L / D = 1000`` travels and spreads as the exact
    solution ``exp(-(x - x0 - u t)^2 / (4 D t)) / sqrt(4 pi D t)``, and halving the
    cells cuts the mean absolute error about fourfold.

    The pulse stays more than five widths from both ends, so the boundaries do not
    touch it. Time is integrated with the explicit Tsit5 at ``rtol=1e-8``: the problem
    is not stiff at these grids, and the time error then sits orders below the
    spatial error under test. Measured orders: 1.98 (40 -> 80), 2.11 (80 -> 160).
    """
    dispersion, x0, t0, duration = 1e-3, 0.25, 1.0, 0.4

    def gaussian(x, t, centre):
        return np.exp(-((x - centre) ** 2) / (4 * dispersion * t)) / np.sqrt(
            4 * np.pi * dispersion * t
        )

    settings = SolverSettings(
        rtol=1e-8, atol=1e-10, integrator=aquakin.IntegratorConfig(solver=diffrax.Tsit5())
    )
    errors = []
    for n_cells in (40, 80, 160):
        reactor = DispersedFlowReactor(
            simple_model, COND, 1.0, 1.0, n_cells, dispersion=dispersion, settings=settings
        )
        x = np.asarray(reactor.grid.centres)
        C0 = np.zeros((n_cells, 2))
        C0[:, 0] = gaussian(x, t0, x0)
        sol = reactor.solve(C0, (0.0, duration), inlet=jnp.zeros(2), params=TRACER)
        exact = gaussian(x, t0 + duration, x0 + duration)
        errors.append(np.mean(np.abs(np.asarray(sol.profile[-1, :, 0]) - exact)))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert errors[-1] < 0.02  # 0.016 measured against a peak of 7.5
    np.testing.assert_allclose(orders, 2.0, atol=0.2)


@pytest.mark.parametrize("d", [0.1, 1.0])
def test_steady_first_order_decay_matches_wehner_wilhelm(simple_model, d):
    """The steady outlet equals the closed-form Danckwerts solution; the 6e-5 measured
    difference on 50 cells is the second-order discretization error."""
    assert steady_outlet(simple_model, d) == pytest.approx(wehner_wilhelm(K * TAU, d), rel=1e-4)


def test_vanishing_dispersion_reduces_to_the_plug_flow_reactor(simple_model):
    """At ``d = 1e-6`` the outlet equals the PlugFlowReactor's, within the 3e-5 spatial
    error of 200 cells (the PFR integrates the axis adaptively to 1e-8)."""
    pfr = aquakin.PlugFlowReactor(simple_model, COND, 2, LENGTH, VELOCITY, rtol=1e-10, atol=1e-12)
    plug = float(pfr.solve(jnp.array([1.0, 0.0])).C_named("A")[-1])
    assert plug == pytest.approx(math.exp(-K * TAU), rel=1e-8)
    assert steady_outlet(simple_model, 1e-6, n_cells=200) == pytest.approx(plug, rel=1e-4)


def test_large_dispersion_reduces_to_a_stirred_tank(simple_model):
    """At ``d = 1e4`` the outlet is the CSTR's ``C_in / (1 + k tau)`` up to its O(1/d)
    (here 2e-5) departure."""
    assert steady_outlet(simple_model, 1e4) == pytest.approx(1.0 / (1.0 + K * TAU), rel=1e-4)


def test_steady_state_conserves_the_mass_that_enters(simple_model):
    """A -> B converts one to one, so A + B leaves at the inlet's 1."""
    reactor = DispersedFlowReactor(simple_model, COND, LENGTH, VELOCITY, 30, dispersion_number=0.2)
    profile = reactor.steady_state(INLET).profile[0]
    np.testing.assert_allclose(profile.sum(axis=1), 1.0, rtol=1e-9)


# --- tanks in series ---------------------------------------------------------


@pytest.mark.parametrize("n_tanks", [1, 3, 10])
def test_tanks_in_series_steady_decay_matches_the_cascade(simple_model, n_tanks):
    """``C_out = C_in / (1 + k tau / N)^N``."""
    reactor = TanksInSeries(simple_model, COND, n_tanks, TAU)
    outlet = float(reactor.steady_state(INLET).C_named("A")[0])
    assert outlet == pytest.approx((1.0 + K * TAU / n_tanks) ** -n_tanks, rel=1e-7)


@pytest.fixture(scope="module")
def tanks_rtd(simple_model):
    """The F curve of 4 tanks (``tau = 20``) under a unit step at the inlet."""
    reactor = TanksInSeries(simple_model, COND, 4, TAU, settings=SolverSettings(1e-10, 1e-12))
    t = jnp.linspace(0.0, 5.0 * TAU, 201)
    return t, reactor.solve(jnp.zeros(2), (0.0, t[-1]), t, inlet={"A": "step(t)"}, params=TRACER)


def test_tanks_in_series_step_response_is_the_erlang_cdf(tanks_rtd):
    """``F(t) = 1 - exp(-N t / tau) sum_(j<N) (N t / tau)^j / j!``."""
    t, sol = tanks_rtd
    s = 4 * np.asarray(t) / TAU
    exact = 1.0 - np.exp(-s) * sum(s**j / math.factorial(j) for j in range(4))
    np.testing.assert_allclose(sol.C_named("A"), exact, rtol=0, atol=1e-8)


def test_tanks_in_series_pulse_response_is_the_erlang_density(simple_model):
    """A pulse in the first tank leaves as ``E(t) = (N/tau)^N t^(N-1) e^(-N t/tau) / (N-1)!``.

    A Dirac at the inlet puts mass ``M`` in the first tank at ``C = M N / V``, so the
    outlet concentration is ``E(t) * tau / N`` for unit initial concentration.
    """
    reactor = TanksInSeries(simple_model, COND, 3, TAU, settings=SolverSettings(1e-10, 1e-12))
    C0 = jnp.zeros((3, 2)).at[0, 0].set(1.0)
    t = jnp.linspace(0.0, 4.0 * TAU, 81)
    sol = reactor.solve(C0, (0.0, t[-1]), t, inlet=jnp.zeros(2), params=TRACER)
    t_np = np.asarray(t)
    erlang = (3 / TAU) ** 3 * t_np**2 * np.exp(-3 * t_np / TAU) / 2.0
    np.testing.assert_allclose(sol.C_named("A"), erlang * TAU / 3, rtol=0, atol=1e-8)


# --- fitting an RTD ----------------------------------------------------------


@pytest.mark.parametrize("n_tanks", [1, 4, 12])
def test_fit_tanks_in_series_recovers_n_from_an_erlang_curve(n_tanks):
    t = np.linspace(0.0, 15.0 * TAU, 30_001)
    E = t ** (n_tanks - 1) * np.exp(-n_tanks * t / TAU)
    assert fit_tanks_in_series(t, E) == n_tanks


def test_fit_tanks_in_series_from_the_reactors_own_response(tanks_rtd, simple_model):
    """The E curve of the simulated 4-tank step response (its derivative) fits N = 4."""
    t, sol = tanks_rtd
    E = np.gradient(np.asarray(sol.C_named("A")), np.asarray(t))
    assert fit_tanks_in_series(t, E) == 4


@pytest.mark.parametrize("d", [0.01, 0.1, 1.0])
def test_fit_dispersion_inverts_the_closed_vessel_variance(d):
    """A gamma curve with the closed-vessel variance of ``d`` fits back to ``d``.

    The gamma shape ``1 / sigma^2`` is not an integer here, so the curve has an
    infinite slope at ``t = 0``; geometric sampling there keeps the trapezoidal
    moments accurate, and the fit is within 3e-7 of ``d`` (measured, at ``d = 1``).
    """
    shape = 1.0 / float(aquakin.utils.rtd.closed_vessel_variance(d))
    t = np.concatenate(
        [[0.0], np.geomspace(1e-12, 1e-2, 20_000, endpoint=False), np.linspace(1e-2, 40.0, 200_001)]
    )
    E = t ** (shape - 1) * np.exp(-shape * t)
    assert fit_dispersion(t, E) == pytest.approx(d, rel=1e-6)


def test_fit_dispersion_recovers_the_number_a_simulated_pulse_was_made_with(simple_model):
    """The outlet response of a ``d = 0.05`` reactor to a narrow inlet pulse fits back.

    The pulse ``exp(-(t/0.002)^2)`` is not a Dirac: its mean (0.001 against a residence
    time of 1) biases ``sigma^2 / tau^2`` by about 0.2%, and 200 cells add their
    second-order error, so the recovery is checked to 1%.
    """
    settings = SolverSettings(
        rtol=1e-8, atol=1e-12, integrator=aquakin.IntegratorConfig(solver=diffrax.Tsit5())
    )
    reactor = DispersedFlowReactor(
        simple_model, COND, 1.0, 1.0, 200, dispersion_number=0.05, settings=settings
    )
    t = jnp.linspace(0.0, 4.0, 2001)
    sol = reactor.solve(
        jnp.zeros(2), (0.0, 4.0), t, inlet={"A": "exp(-(t/0.002)**2)"}, params=TRACER
    )
    assert fit_dispersion(t, sol.C_named("A")) == pytest.approx(0.05, rel=1e-2)


def test_fit_dispersion_rejects_a_response_wider_than_a_stirred_tank():
    # Two parallel paths, residence times 0.1 and 10: sigma^2 / tau^2 = 2.9.
    t = np.linspace(0.0, 400.0, 400_001)
    E = 0.5 * np.exp(-t / 0.1) / 0.1 + 0.5 * np.exp(-t / 10.0) / 10.0
    with pytest.raises(ValueError, match="wider than a stirred tank"):
        fit_dispersion(t, E)


# --- gradients -----------------------------------------------------------------


def _outlet_after(model, k, d):
    """Outlet A at t = 15 under a rising inlet, as a function of ``k`` and ``d``."""
    reactor = DispersedFlowReactor(
        model,
        COND,
        LENGTH,
        VELOCITY,
        12,
        dispersion_number=d,
        settings=SolverSettings(rtol=1e-10, atol=1e-12),
    )
    sol = reactor.solve(
        jnp.zeros(2), (0.0, 15.0), inlet={"A": "1 - exp(-t/5)"}, params=jnp.array([k])
    )
    return sol.C[-1, 0]


def _steady_outlet(model, k, d):
    reactor = DispersedFlowReactor(model, COND, LENGTH, VELOCITY, 12, dispersion_number=d)
    return reactor.steady_state(INLET, params=jnp.array([k])).C[0, 0]


@pytest.mark.parametrize("fn", [_outlet_after, _steady_outlet], ids=["solve", "steady_state"])
def test_gradient_in_k_and_the_dispersion_number_matches_a_central_difference(simple_model, fn):
    """``jax.grad`` through the solve (and the implicit-function-theorem steady state)
    with respect to ``k`` and ``d`` equals a central difference with step 1e-5, whose
    truncation error is ~1e-10 here; both agree to 1e-6."""

    def loss(k, d):
        return fn(simple_model, k, d)

    grad = jax.grad(loss, argnums=(0, 1))(K, 0.1)
    h = 1e-5
    fd_k = (loss(K + h, 0.1) - loss(K - h, 0.1)) / (2 * h)
    fd_d = (loss(K, 0.1 + h) - loss(K, 0.1 - h)) / (2 * h)
    np.testing.assert_allclose([float(g) for g in grad], [fd_k, fd_d], rtol=1e-6)
    assert float(grad[1]) > 0  # more dispersion, more bypass, more A at the outlet


# --- the solution object and per-cell conditions -------------------------------------


def test_solution_reads_species_and_profiles_by_name(simple_model):
    reactor = TanksInSeries(simple_model, COND, 3, TAU)
    sol = reactor.steady_state(INLET)
    tank = [(1.0 + K * TAU / 3) ** -(i + 1) for i in range(3)]
    np.testing.assert_allclose(sol.profile_named("A")[0], tank, rtol=1e-7)
    np.testing.assert_allclose(sol.x, [1 / 6, 1 / 2, 5 / 6], rtol=1e-15)
    assert reactor.n_tanks == 3
    frame = sol.to_dataframe()
    assert list(frame.columns) == ["A", "B"]
    assert frame["A"].iloc[0] == pytest.approx(tank[-1], rel=1e-7)
    with pytest.raises(KeyError, match="Did you mean: A"):
        sol.profile_named("AA")


def test_per_cell_conditions_reach_their_own_cell():
    """Each tank reads its own depth (the photolysis rate falls with depth).

    Tank 1 sees only the inlet and its own conditions, so with a depth per tank it
    matches a one-tank train at that depth; and tank 2 of a shallow-then-deep train
    differs from tank 2 of an all-shallow one, which a reactor reading the first
    location everywhere would not show.
    """
    model = aquakin.load_model_from_file(FIXTURES / "math_functions_model.yaml")
    shallow, deep = 0.25, 5.0

    def profile(z_per_tank):
        cond = aquakin.SpatialConditions(
            {"z": jnp.asarray(z_per_tank), "T": jnp.full(len(z_per_tank), 293.15)}
        )
        # 5 time units per tank, so the first tank is the same in every train.
        reactor = TanksInSeries(model, cond, len(z_per_tank), 5.0 * len(z_per_tank))
        return reactor.steady_state({"A": 1.0}).profile[0]

    # The steady state converges to a residual of 1e-8 scaled by max(|C|, 1), so
    # concentrations of order 1 agree to about 1e-8 in absolute terms.
    np.testing.assert_allclose(
        profile([shallow, deep])[0], profile([shallow])[0], rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(profile([deep, shallow])[0], profile([deep])[0], rtol=0, atol=1e-7)
    tank_2_deep = profile([shallow, deep])[1, 0]
    tank_2_shallow = profile([shallow, shallow])[1, 0]
    assert float(tank_2_deep - tank_2_shallow) > 0.2  # 0.585 against 0.286


# --- errors ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "exactly one of dispersion="),
        ({"dispersion": 0.1, "dispersion_number": 0.1}, "exactly one of dispersion="),
        ({"dispersion": -1.0}, "dispersion must be finite and >= 0"),
        ({"dispersion_number": float("nan")}, "dispersion_number must be finite and >= 0"),
        ({"dispersion": 0.1, "velocity": 0.0}, "velocity must be positive"),
        ({"dispersion": 0.1, "length": -1.0}, "length must be positive"),
        ({"dispersion": 0.1, "n_cells": 0}, "n_cells must be a positive integer"),
    ],
)
def test_dispersed_flow_reactor_rejects_bad_construction(simple_model, kwargs, message):
    args = {"length": 1.0, "velocity": 1.0, "n_cells": 10, **kwargs}
    with pytest.raises(ValueError, match=re.escape(message)):
        DispersedFlowReactor(simple_model, COND, **args)


def test_dispersion_and_its_number_are_interchangeable(simple_model):
    by_number = DispersedFlowReactor(simple_model, COND, 2.0, 0.5, 10, dispersion_number=0.1)
    by_coefficient = DispersedFlowReactor(simple_model, COND, 2.0, 0.5, 10, dispersion=0.1)
    assert by_number.dispersion == pytest.approx(0.1, rel=1e-15)
    assert by_coefficient.dispersion_number == pytest.approx(0.1, rel=1e-15)


@pytest.mark.parametrize(
    "n_tanks, residence_time, message",
    [
        (0, 1.0, "n_tanks must be a positive integer"),
        (2.5, 1.0, "n_tanks must be a positive integer"),
        (True, 1.0, "n_tanks must be a positive integer"),
        (3, 0.0, "residence_time must be positive"),
    ],
)
def test_tanks_in_series_rejects_bad_construction(simple_model, n_tanks, residence_time, message):
    with pytest.raises(ValueError, match=message):
        TanksInSeries(simple_model, COND, n_tanks, residence_time)


def test_conditions_need_one_location_or_one_per_cell(simple_model):
    cond = aquakin.SpatialConditions({"T": jnp.full(2, 293.15)})
    with pytest.raises(ValueError, match=re.escape("1 location or one per cell (3)")):
        TanksInSeries(simple_model, cond, 3, 1.0)


@pytest.fixture(scope="module")
def tanks(simple_model):
    return TanksInSeries(simple_model, COND, 3, 1.0)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"C0": jnp.zeros(3)}, ValueError, "C0 has shape (3,)"),
        ({"inlet": jnp.zeros(3)}, ValueError, "inlet has shape (3,)"),
        ({"inlet": {"AA": 1.0}}, ValueError, "unknown species 'AA'. Did you mean: A"),
        ({"inlet": {"A": "1 +"}}, TimeExpressionError, "the inlet of 'A'"),
        ({"params": jnp.zeros(2)}, ValueError, "params has shape (2,)"),
        (
            {"conditions": aquakin.SpatialConditions({"T": jnp.full(3, 293.15)})},
            ValueError,
            "conditions override must have n_locations=1",
        ),
    ],
)
def test_solve_rejects_bad_inputs(tanks, kwargs, error, message):
    args = {"C0": jnp.zeros(2), "inlet": INLET, **kwargs}
    C0 = args.pop("C0")
    with pytest.raises(error, match=re.escape(message)):
        tanks.solve(C0, (0.0, 1.0), **args)


def test_steady_state_needs_a_constant_inlet(tanks):
    with pytest.raises(ValueError, match=re.escape("['A'] are given as expressions of time")):
        tanks.steady_state({"A": "1 + t"})


def test_steady_state_reports_non_convergence(tanks):
    with pytest.raises(RuntimeError, match="did not converge in 1 iterations"):
        tanks.steady_state(INLET, guess=jnp.zeros(2), max_iter=1)


def test_steady_state_traces_under_jit(tanks):
    """Under an outer ``jax.jit`` the convergence flag is traced and not read; the
    steady state is the same as the concrete one."""
    concrete = tanks.steady_state(INLET).C[0, 0]
    jitted = jax.jit(lambda p: tanks.steady_state(INLET, params=p).C[0, 0])(jnp.array([K]))
    assert float(jitted) == pytest.approx(float(concrete), rel=1e-12)


def test_inlet_numbers_may_be_written_as_strings(tanks):
    """``"1.0"`` is a number, not an expression, so the steady state accepts it."""
    as_text = tanks.steady_state({"A": "1.0"}).C[0, 0]
    assert float(as_text) == pytest.approx(float(tanks.steady_state(INLET).C[0, 0]), rel=1e-14)
