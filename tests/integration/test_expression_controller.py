"""An expression controller driving a plant through the signal bus.

The single-CSTR plant (A -> B decay, V = 100, Q = 10 of A = 1, ``k = 0.1``) is
aerated on ``A`` from a signal: ``Aeration(kla_signal="kla", species="A",
do_sat=2)``. Two expression controllers set that signal, in a cascade listed
consumer first:

- ``lead`` senses the tank and publishes ``demand = Kp*(sp - [A])``;
- ``follow`` senses nothing and publishes ``kla = clip(g*demand, 0, kmax)``.

So ``dA/dt = (Q/V)(1 - A) - k A + kla(A) (2 - A)``, a scalar ODE integrated here
independently with SciPy at a tight tolerance and compared with the plant
solve. The gradient of ``A(T)`` with respect to ``k`` is checked against a
central finite difference. The dosing unit's ``flow_signal`` mode and the spec
round trip are checked on small plants of the same model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from aquakin.plant import Aeration, CSTRUnit, DosingUnit, ExpressionController, Plant, Reagent
from aquakin.plant.influent import InfluentSeries

KP, SP, GAIN, KMAX = 2.0, 1.0, 0.5, 0.05
T_END = 60.0
T_EVAL = jnp.asarray([0.2, 20.0, 60.0])


def _controlled_plant(make_cstr_plant, simple_model):
    plant = make_cstr_plant(aeration=Aeration(kla_signal="kla", species="A", do_sat=2.0))
    plant.add_unit(
        ExpressionController(
            "follow",
            signal_name="kla",
            law="clip(g*demand, 0, kmax)",
            constants={"g": GAIN, "kmax": KMAX},
        )
    )
    plant.add_unit(
        ExpressionController(
            "lead",
            model=simple_model,
            signal_name="demand",
            law="Kp*(sp - [A])",
            constants={"Kp": KP, "sp": SP},
        )
    )
    plant.connect("tank.out", "lead.measured")
    return plant


def _reference(k=0.1):
    """``A(t)`` at ``T_EVAL`` from SciPy, for the same law written in NumPy."""

    def rhs(_t, y):
        a = y[0]
        kla = np.clip(GAIN * KP * (SP - a), 0.0, KMAX)
        return [0.1 * (1.0 - a) - k * a + kla * (2.0 - a)]

    a0 = 1.0  # the tank starts at the model's default, A = 1
    sol = solve_ivp(rhs, (0.0, T_END), [a0], t_eval=np.asarray(T_EVAL), rtol=1e-11, atol=1e-13)
    return sol.y[0]


@pytest.fixture
def controlled(make_cstr_plant, simple_model):
    return _controlled_plant(make_cstr_plant, simple_model)


def test_the_signal_driven_tank_matches_the_independent_ode(controlled):
    """A(t) of the plant solve equals SciPy's integration of the closed loop, and
    the clip is active over part of the run (so both of its branches are tested)."""
    sol = controlled.solve((0.0, T_END), t_eval=T_EVAL, rtol=1e-9, atol=1e-12)
    a = np.asarray(sol.C_named("tank", "A"))
    np.testing.assert_allclose(a, _reference(), rtol=1e-6)
    signals = [controlled.signals_at(t, sol.state[i]) for i, t in enumerate(T_EVAL)]
    kla = np.asarray([float(s["kla"]) for s in signals])
    np.testing.assert_allclose(kla, np.clip(GAIN * KP * (SP - a), 0.0, KMAX), rtol=1e-12)
    assert kla[-1] == KMAX and kla[0] < KMAX


def test_grad_through_the_controlled_solve_matches_a_central_difference(controlled):
    base = controlled.default_parameters()
    i = controlled.parameter_index("simple_decay.A_to_B.k")

    def a_end(k):
        sol = controlled.solve(
            (0.0, T_END), t_eval=T_EVAL[-1:], params=base.at[i].set(k), rtol=1e-9, atol=1e-12
        )
        return sol.C_named("tank", "A")[-1]

    k0, h = 0.1, 1e-4
    grad = float(jax.grad(a_end)(k0))
    fd = (float(a_end(k0 + h)) - float(a_end(k0 - h))) / (2.0 * h)
    # Central difference of an rtol=1e-9 solve with h = 1e-4: truncation ~h², noise ~1e-9/h.
    assert grad == pytest.approx(fd, rel=1e-4)
    assert grad < 0.0  # faster decay leaves less A
    ref_fd = (_reference(k0 + h)[-1] - _reference(k0 - h)[-1]) / (2.0 * h)
    assert grad == pytest.approx(ref_fd, rel=1e-4)


def test_the_plant_round_trips_through_its_spec(controlled, tmp_path):
    spec = controlled.to_spec(tmp_path / "plant.yaml")
    units = spec["units"]
    assert units["follow"]["kind"] == "expression_controller"
    assert units["follow"]["law"] == "clip(g*demand, 0, kmax)"
    assert units["lead"]["constants"] == {"Kp": KP, "sp": SP}
    assert units["tank"]["aeration"]["kla_signal"] == "kla"
    rebuilt = Plant.from_spec(spec, base_dir=tmp_path)
    assert rebuilt.to_spec(tmp_path / "again.yaml")["units"] == units
    np.testing.assert_array_equal(
        rebuilt.derivative(rebuilt.initial_state()),
        controlled.derivative(controlled.initial_state()),
    )


def test_a_dose_flow_read_from_a_signal(simple_model):
    """``flow_signal``: the dose is the signal times ``gain``, mixed into the
    through-stream: Q_out = Q + q, C_out = (Q C + q C_reagent) / Q_out."""
    plant = Plant("dosed", time_unit="s")
    reagent = Reagent.from_species(simple_model, B=5.0)
    plant.add_unit(DosingUnit("dose", reagent, flow_signal="q_dose", gain=2.0))
    plant.add_unit(
        CSTRUnit("tank", simple_model, 100.0, input_port_names=["inlet"], conditions={"T": 293.15})
    )
    plant.add_unit(ExpressionController("law", signal_name="q_dose", law="0.2 + 0.01*t"))
    feed = InfluentSeries(
        t=jnp.asarray([0.0, 100.0]),
        Q=jnp.asarray([10.0, 10.0]),
        C=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
        model=simple_model,
    )
    plant.add_influent("feed", feed, to="dose.in")
    plant.connect("dose.out", "tank.inlet")
    t = jnp.asarray([0.0, 10.0, 30.0])
    sol = plant.solve((0.0, 30.0), t_eval=t)
    dosed = plant.stream(sol, "dose.out")
    q = 2.0 * (0.2 + 0.01 * np.asarray(t))
    np.testing.assert_allclose(np.asarray(dosed.Q), 10.0 + q, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(dosed.C[:, 1]), 5.0 * q / (10.0 + q), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(dosed.C[:, 0]), 10.0 / (10.0 + q), rtol=1e-10)
