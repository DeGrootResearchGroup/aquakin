"""Flow setpoints written as expressions of the plant clock.

A pump flow, a clarifier underflow, a splitter threshold or a primary-sludge
fraction may be a string such as ``"5 + 1.5*sin(2*pi*t/20)"`` in place of a
number. Checked here:

- the trajectory against closed forms (a tank fed by a time-varying pump; a
  stateless clarifier loop whose recycle concentrations follow the underflow);
- that a plant with such a setpoint never caches its recycle maps, and that
  caching them would give the wrong answer, while a plant without one keeps
  them;
- the gradient through a solve against a central finite difference;
- the spec round trip and fingerprint, and the errors by message.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import aquakin
from aquakin.plant import CSTRUnit, MixerUnit, Plant, SetpointSplitter, ThresholdSplitter
from aquakin.plant.clarifier import IdealClarifier
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.primary_clarifier import PrimaryClarifier
from aquakin.plant.spec import SpecError, spec_fingerprint
from aquakin.plant.takacs import TakacsClarifier
from aquakin.plant.units import FlowContext

T_TANK = {"T": 293.15}
TIGHT = dict(rtol=1e-10, atol=1e-12)


def _feed(net, Q=10.0, A=1.0):
    return InfluentSeries(
        t=jnp.asarray([0.0, 1000.0]),
        Q=jnp.asarray([Q, Q]),
        C=jnp.asarray([[A, 0.0], [A, 0.0]]),
        model=net,
    )


# -- a tank fed by a time-varying pump: closed form ---------------------------

PUMP = "2 + sin(2*pi*t/50)"


def _pump_integral(t):
    """``∫0^t (2 + sin(2πs/50)) ds``."""
    return 2.0 * t + 50.0 / (2.0 * math.pi) * (1.0 - np.cos(2.0 * math.pi * t / 50.0))


@pytest.fixture(scope="module")
def pump_plant(simple_model):
    """Feed (Q = 10, A = 1) -> a splitter pumping ``2 + sin(2πt/50)`` into a
    100-volume tank, the rest bypassing it."""
    plant = Plant("pump", time_unit="s")
    plant.add_unit(SetpointSplitter("split", simple_model, {"tank": PUMP}, remainder_port="bypass"))
    plant.add_unit(CSTRUnit("tank", simple_model, 100.0, ["inlet"], conditions=T_TANK))
    plant.add_influent("feed", _feed(simple_model), to="split.in")
    plant.connect("split.tank", "tank.inlet")
    return plant


def _pump_solve(plant, k, t_eval):
    index = plant.parameter_index("simple_decay.A_to_B.k")
    params = plant.default_parameters().at[index].set(k)
    y0 = plant.initial_state(overrides={"tank": jnp.zeros(2)})
    return plant.solve(
        t_span=(0.0, float(t_eval[-1])), t_eval=t_eval, y0=y0, params=params, **TIGHT
    )


def test_a_pumped_tank_follows_the_closed_form(pump_plant):
    """With no reaction (k = 0) the tank only dilutes, so
    ``A(t) = 1 - exp(-∫Q/V)`` with ``Q(t)`` the pump expression evaluated on the
    plant's clock."""
    t_eval = jnp.linspace(0.0, 100.0, 11)
    sol = _pump_solve(pump_plant, 0.0, t_eval)
    t = np.asarray(t_eval)
    exact = 1.0 - np.exp(-_pump_integral(t) / 100.0)
    np.testing.assert_allclose(np.asarray(sol.C_named("tank", "A")), exact, rtol=1e-8, atol=1e-10)
    # The stream carries the expression's value at each saved time.
    pumped = np.asarray(pump_plant.stream(sol, "split.tank").Q)
    np.testing.assert_allclose(pumped, 2.0 + np.sin(2.0 * math.pi * t / 50.0), rtol=1e-12)


def test_gradient_through_an_expression_setpoint_matches_finite_difference(pump_plant):
    """``d A(T) / dk`` through a solve driven by the pump expression, by reverse
    mode and by a central difference (step 1e-5 on k = 0.01; the solve's
    rtol 1e-10 leaves the difference good to about 1e-6)."""
    t_eval = jnp.asarray([0.0, 40.0])

    def loss(k):
        return _pump_solve(pump_plant, k, t_eval).C_named("tank", "A")[-1]

    k, h = 0.01, 1e-5
    grad = float(jax.grad(loss)(k))
    fd = (float(loss(k + h)) - float(loss(k - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-6)
    assert grad < 0.0  # decay removes A


# -- a stateless clarifier loop: the recycle concentration map moves in time --

UNDERFLOW = "5 + 1.5*sin(2*pi*t/20)"
CAPTURE, Q_FEED, Q_RAS, V_POLISH = 0.9, 10.0, 3.0, 50.0


def _loop_plant(net, underflow, **options):
    """Feed -> mixer -> ideal clarifier (underflow ``underflow``) -> splitter
    returning a fixed 3 to the mixer, the overflow polished in a 50-volume tank.
    The loop holds no state, so its recycle concentration map ``M`` is set by
    the clarifier's underflow."""
    plant = Plant("loop", time_unit="s", **options)
    plant.add_unit(MixerUnit("mix", ["fresh", "ras"], net))
    plant.add_unit(
        IdealClarifier(
            "clar",
            net,
            underflow_Q=underflow,
            capture_efficiency=CAPTURE,
            particulate_species=["A"],
        )
    )
    plant.add_unit(SetpointSplitter("split", net, {"ras": Q_RAS}, remainder_port="waste"))
    plant.add_unit(CSTRUnit("polish", net, V_POLISH, ["inlet"], conditions=T_TANK))
    plant.add_influent("feed", _feed(net, Q=Q_FEED), to="mix.fresh")
    plant.connect("mix", "clar")
    plant.connect("clar.underflow", "split")
    plant.connect("split.ras", "mix.ras")
    plant.connect("clar.overflow", "polish.inlet")
    return plant


def _loop_overflow(t):
    """The clarifier overflow ``(Q, A)`` at time ``t``, solved by hand.

    Underflow ``A_u = c Q_f A_f / (Q_u - c Q_r)`` (the captured mass of the feed
    plus the recycled underflow, over the underflow); overflow
    ``A_o = (1 - c)(Q_f A_f + Q_r A_u) / Q_o`` with ``Q_o = Q_f + Q_r - Q_u``.
    """
    Q_u = 5.0 + 1.5 * np.sin(2.0 * math.pi * t / 20.0)
    A_u = CAPTURE * Q_FEED / (Q_u - CAPTURE * Q_RAS)
    Q_o = Q_FEED + Q_RAS - Q_u
    return Q_o, (1.0 - CAPTURE) * (Q_FEED + Q_RAS * A_u) / Q_o


def _loop_solve(plant, t_eval):
    params = plant.parameter_values({"simple_decay.A_to_B.k": 0.0})
    y0 = plant.initial_state(overrides={"polish": jnp.zeros(2)})
    return plant.solve(
        t_span=(0.0, float(t_eval[-1])), t_eval=t_eval, y0=y0, params=params, **TIGHT
    )


def _polish_reference(t_eval):
    """The polishing tank by an independent integration of
    ``dA/dt = Q_o/V (A_o - A)`` with the hand-solved overflow."""

    def rhs(t, a):
        Q_o, A_o = _loop_overflow(t)
        return [Q_o / V_POLISH * (A_o - a[0])]

    out = solve_ivp(rhs, (0.0, t_eval[-1]), [0.0], t_eval=t_eval, rtol=1e-12, atol=1e-14)
    return out.y[0]


@pytest.fixture(scope="module")
def loop_solution(simple_model):
    plant = _loop_plant(simple_model, UNDERFLOW)
    t_eval = jnp.linspace(0.0, 30.0, 7)
    return plant, t_eval, _loop_solve(plant, t_eval)


def test_a_time_varying_underflow_keeps_the_recycle_maps_uncached(loop_solution):
    """The plant declares a scheduled setpoint, so neither recycle map is cached."""
    plant, _, _ = loop_solution
    assert plant.units["clar"].scheduled_setpoints() == ("underflow_Q",)
    assert plant._recycle.cache_flags() == (False, False, False)


def test_the_clarifier_loop_follows_its_hand_solution(loop_solution):
    """The reconstructed overflow matches the hand-solved loop at every saved
    time, and the polishing tank an independent integration of it."""
    plant, t_eval, sol = loop_solution
    t = np.asarray(t_eval)
    Q_o, A_o = _loop_overflow(t)
    over = plant.stream(sol, "clar.overflow")
    np.testing.assert_allclose(np.asarray(over.Q), Q_o, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(over.C_named("A")), A_o, rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(sol.C_named("polish", "A")), _polish_reference(t), rtol=1e-7, atol=1e-10
    )


def test_caching_the_recycle_map_would_be_wrong_here(simple_model, loop_solution):
    """The guard is needed. With the fixed recycle passes (``recycle_tol=None``)
    the map is the answer rather than the seed of an iteration, and with ``M``
    cached anyway (built once, at the start of the run) the polishing tank
    drifts away from the hand solution, because the underflow that sets ``M``
    keeps moving. Left to the guard, the same plant follows it."""
    _, t_eval, _ = loop_solution
    reference = _polish_reference(np.asarray(t_eval))
    plant = _loop_plant(simple_model, UNDERFLOW, recycle_tol=None)
    guarded = _loop_solve(plant, t_eval)  # runs the guards: M is not cached
    np.testing.assert_allclose(
        np.asarray(guarded.C_named("polish", "A")), reference, rtol=1e-7, atol=1e-10
    )
    plant._recycle._recycle_map_constant = True  # what a state-only guard concluded
    forced = _loop_solve(plant, t_eval)
    error = np.max(np.abs(np.asarray(forced.C_named("polish", "A")) - reference))
    assert error > 1e-3 * np.max(reference)


def test_a_constant_underflow_keeps_the_maps_cached(simple_model):
    """Without an expression the same loop keeps both cached maps, and its
    overflow is the hand solution at the constant underflow."""
    plant = _loop_plant(simple_model, 5.0)
    t_eval = jnp.asarray([0.0, 5.0])
    sol = _loop_solve(plant, t_eval)
    assert plant.units["clar"].scheduled_setpoints() == ()
    assert plant._recycle.cache_flags() == (True, True, True)
    Q_o, A_o = _loop_overflow(np.zeros(2))  # sin(0) = 0: the constant underflow
    np.testing.assert_allclose(
        np.asarray(plant.stream(sol, "clar.overflow").C_named("A")), A_o, rtol=1e-9
    )


# -- every setpoint unit, clipping and errors ---------------------------------


def _ctx(t):
    return FlowContext(state=None, t=jnp.asarray(t))


def test_every_setpoint_unit_reads_its_expression_and_clips_it(simple_model):
    """Each unit evaluates its expression at the context's time, takes no
    parameter slot for it, and clips it into its range (a flow to >= 0, a
    fraction to [0, 1])."""
    none = jnp.zeros((0,))
    ideal = IdealClarifier("c", simple_model, overflow_Q="4 - t", particulate_species=["A"])
    assert ideal.flow_outputs({"inlet": 10.0}, none, _ctx(1.0))["overflow"] == 3.0
    assert ideal.flow_outputs({"inlet": 10.0}, none, _ctx(6.0))["overflow"] == 0.0
    assert ideal.flow_param_defaults() == []

    takacs = TakacsClarifier(
        "s", simple_model, area=10.0, height=4.0, underflow_Q="2*t", particulate_species=["A"]
    )
    assert takacs.flow_outputs({"inlet": 10.0}, none, _ctx(1.5))["underflow"] == 3.0
    assert takacs.scheduled_setpoints() == ("underflow_Q",)

    primary = PrimaryClarifier("p", simple_model, 100.0, f_PS="0.01*t", settling_species=("A",))
    assert primary.flow_outputs({"inlet": 10.0}, none, _ctx(20.0))["underflow"] == pytest.approx(
        2.0, rel=1e-15
    )
    assert primary.flow_outputs({"inlet": 10.0}, none, _ctx(200.0))["underflow"] == 10.0

    split = SetpointSplitter("sp", simple_model, {"a": 1.0, "b": "t - 1"}, remainder_port="r")
    out = split.flow_outputs({"in": 10.0}, none, _ctx(0.0))
    assert (out["a"], out["b"], out["r"]) == (1.0, 0.0, 9.0)
    # The constant keeps its parameter slot; the expression has none.
    assert split.flow_param_local_names() == ["a"]

    gate = ThresholdSplitter("th", simple_model, threshold="3 + t", threshold_port="hi", remainder_port="lo")
    out = gate.flow_outputs({"in": 10.0}, none, _ctx(2.0))
    assert (out["hi"], out["lo"]) == (5.0, 5.0)
    inlet = aquakin.plant.Stream(Q=10.0, C=jnp.asarray([1.0, 0.5]), model=simple_model)
    streams = gate.compute_outputs(jnp.asarray(4.0), jnp.zeros(0), {"in": inlet}, none)
    assert (float(streams["hi"].Q), float(streams["lo"].Q)) == (3.0, 7.0)


def test_a_settled_takacs_start_reads_an_overflow_expression_at_t0(simple_model):
    """The settled-blanket start sizes the blanket from the overflow at the
    start of the run, so an expression worth 300 at t = 0 seeds exactly what
    the constant 300 does."""
    common = dict(area=100.0, height=4.0, init_underflow_Q=150.0, particulate_species=["A"])
    constant = TakacsClarifier("s", simple_model, overflow_Q=300.0, **common)
    varying = TakacsClarifier("s", simple_model, overflow_Q="300 + 100*sin(2*pi*t)", **common)
    other = TakacsClarifier("s", simple_model, overflow_Q=600.0, **common)
    np.testing.assert_array_equal(
        np.asarray(varying.initial_state()), np.asarray(constant.initial_state())
    )
    assert not np.array_equal(np.asarray(other.initial_state()), np.asarray(constant.initial_state()))


def test_a_fraction_clipped_to_zero_keeps_the_sludge_finite(simple_model):
    """A primary-sludge fraction clipped to 0 sends no flow to the sludge and
    leaves its concentration finite (the thickening factor is guarded)."""
    primary = PrimaryClarifier("p", simple_model, 100.0, f_PS="0.01 - t", settling_species=("A",))
    inlet = aquakin.plant.Stream(Q=10.0, C=jnp.asarray([1.0, 0.5]), model=simple_model)
    out = primary.compute_outputs(
        jnp.asarray(5.0), jnp.asarray([1.0, 0.5]), {"inlet": inlet}, jnp.zeros((0,))
    )
    assert float(out["underflow"].Q) == 0.0
    assert np.all(np.isfinite(np.asarray(out["underflow"].C)))


def test_an_unreadable_expression_names_the_unit_and_setting(simple_model):
    with pytest.raises(ValueError, match=r"IdealClarifier 'c': 'underflow_Q'.*wobble"):
        IdealClarifier("c", simple_model, underflow_Q="5 + wobble(t)", particulate_species=["A"])
    with pytest.raises(ValueError, match="not finite at t = 0"):
        SetpointSplitter("sp", simple_model, {"a": "1/t"}, remainder_port="r")


def test_an_expression_needs_the_time(simple_model):
    ideal = IdealClarifier("c", simple_model, underflow_Q="5 + t", particulate_species=["A"])
    with pytest.raises(ValueError, match="needs the time"):
        ideal.flow_outputs({"inlet": 10.0}, jnp.zeros((0,)))


def test_a_constant_setpoint_is_still_checked(simple_model):
    """Only an expression is clipped; a constant out of range is refused as before."""
    with pytest.raises(ValueError, match="underflow_Q must be non-negative"):
        IdealClarifier("c", simple_model, underflow_Q="-5", particulate_species=["A"])
    with pytest.raises(ValueError, match=r"f_PS must be in \(0, 1\)"):
        PrimaryClarifier("p", simple_model, 100.0, f_PS="1.5", settling_species=("A",))


# -- the spec ------------------------------------------------------------------


@pytest.fixture(scope="module")
def asm1_spec_plant():
    """A shipped-model loop whose underflow and return pump are expressions."""
    asm1 = aquakin.load_model("asm1")
    plant = Plant("spec", time_unit="d")
    plant.add_unit(MixerUnit("mix", ["fresh", "ras"], asm1))
    plant.add_unit(IdealClarifier("clar", asm1, underflow_Q="400 + 75*sin(2*pi*t)"))
    plant.add_unit(
        SetpointSplitter(
            "split", asm1, {"ras": "300 + 100*step(t - 10)"}, remainder_port="waste"
        )
    )
    plant.add_unit(CSTRUnit("tank", asm1, 1000.0, ["inlet"], conditions={"T": 288.15}))
    plant.add_influent("feed", InfluentSeries.constant(asm1, Q=1000.0, SS=50.0), to="mix.fresh")
    plant.connect("mix", "clar")
    plant.connect("clar.underflow", "split")
    plant.connect("split.ras", "mix.ras")
    plant.connect("clar.overflow", "tank.inlet")
    return plant


def test_an_expression_setpoint_round_trips_through_the_spec(asm1_spec_plant):
    """The expressions are written as the text they were given, read back, and
    the rebuilt plant writes the same spec and has the same derivative at a
    time the expressions vary at."""
    spec = asm1_spec_plant.to_spec()
    units = spec["units"]
    assert units["clar"]["underflow_Q"] == "400 + 75*sin(2*pi*t)"
    assert units["split"]["output_port_flows"] == {"ras": "300 + 100*step(t - 10)"}
    rebuilt = Plant.from_spec(spec)
    assert rebuilt.to_spec() == spec
    y0 = asm1_spec_plant.initial_state()
    for t in (0.25, 12.0):
        np.testing.assert_array_equal(
            np.asarray(rebuilt.derivative(y0, t=t)), np.asarray(asm1_spec_plant.derivative(y0, t=t))
        )


def test_the_fingerprint_follows_the_expression_and_not_its_spelling(asm1_spec_plant):
    """A changed expression changes the fingerprint; a constant written as an
    integer, a float or a number in text does not."""
    spec = asm1_spec_plant.to_spec()
    base = spec_fingerprint(spec)
    assert spec_fingerprint(asm1_spec_plant.to_spec()) == base

    changed = {**spec, "units": {**spec["units"], "clar": {**spec["units"]["clar"]}}}
    changed["units"]["clar"]["underflow_Q"] = "400 + 80*sin(2*pi*t)"
    assert spec_fingerprint(changed) != base

    spellings = []
    for value in (400.0, 400, "400", " 4e2"):
        spelled = {**spec, "units": {**spec["units"], "clar": {**spec["units"]["clar"]}}}
        spelled["units"]["clar"]["underflow_Q"] = value
        spellings.append(spec_fingerprint(spelled))
    assert len(set(spellings)) == 1 and spellings[0] != base


def test_a_spec_with_an_unreadable_expression_is_refused(asm1_spec_plant):
    spec = asm1_spec_plant.to_spec()
    bad = {**spec, "units": {**spec["units"], "split": {**spec["units"]["split"]}}}
    bad["units"]["split"]["output_port_flows"] = {"ras": "300 + wobble(t)"}
    with pytest.raises(SpecError, match=r"units\.split: .*'output_port_flows\.ras'.*wobble"):
        Plant.from_spec(bad)
