"""``alpha`` and the temperature factor on a signal-driven kLa.

A controller's output is the clean-water kLa, so ``Aeration(alpha=...)`` scales
it exactly as it scales a fixed ``kla``. On the single-CSTR plant (A -> B decay,
``k = 0.1``, V = 100, Q = 10 of A = 1, time in seconds), aerated on ``A`` toward
``do_sat = 2``, the tank balance with an effective kLa ``K`` is::

    dA/dt = 0.1 (1 - A) - 0.1 A + K (2 - A),   A* = (0.1 + 2 K) / (0.2 + K)

so a constant signal gives a closed-form steady state, and the clean-water kLa
the signal bus and the energy readers report is separable from the process kLa
the solve and the oxygen-transfer history apply.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquakin.plant import Aeration, AerationSystem, ExpressionController
from aquakin.plant.aeration_system import blower_power_kw, required_airflow
from aquakin.plant.balance import oxygen_transfer_history
from aquakin.plant.bsm.evaluation import _aeration_energy, _delivered_kla_history, _kla_history

SIGNAL, GAIN, ALPHA, DO_SAT = 0.04, 0.5, 0.5, 2.0
T_END = 200.0  # 40 time constants of the slowest tank below


def _steady_A(K):
    return (0.1 + DO_SAT * K) / (0.2 + K)


def _signal_plant(make_cstr_plant, alpha, law="c", model=None):
    """The single-CSTR plant aerated from the signal ``kla`` that ``law`` sets."""
    plant = make_cstr_plant(
        aeration=Aeration(kla_signal="kla", species="A", do_sat=DO_SAT, gain=GAIN, alpha=alpha)
    )
    plant.add_unit(
        ExpressionController(
            "ctl", model=model, signal_name="kla", law=law, constants={"c": SIGNAL, "sp": 1.6}
        )
    )
    if model is not None:
        plant.connect("tank.out", "ctl.measured")
    return plant


def _solve(plant, **kw):
    return plant.solve((0.0, T_END), t_eval=jnp.asarray([T_END]), rtol=1e-10, atol=1e-12, **kw)


@pytest.fixture
def fouled(make_cstr_plant):
    plant = _signal_plant(make_cstr_plant, ALPHA)
    return plant, _solve(plant)


def test_a_signal_driven_kla_is_scaled_by_alpha(fouled):
    """The steady state is the closed form at ``K = alpha * signal * gain``, and
    the oxygen-transfer history (the mass balance's aeration term) is that kLa
    times the driving force ``(do_sat - A)``."""
    plant, sol = fouled
    a = float(sol.C_named("tank", "A")[-1])
    K = ALPHA * SIGNAL * GAIN
    assert a == pytest.approx(_steady_A(K), rel=1e-8)
    oxygen, stripped = oxygen_transfer_history(plant, sol, None, ["tank"])["tank"]
    assert float(oxygen[-1, 0]) == pytest.approx(K * (DO_SAT - a), rel=1e-12)
    assert float(jnp.max(jnp.abs(stripped))) == 0.0


def test_the_signal_and_energy_kla_stay_clean_water(fouled):
    """The bus reports the controller's output; the energy kLa history is that
    times the gain (clean water), and only ``process_water`` applies ``alpha``."""
    plant, sol = fouled
    assert float(plant.signals_at(T_END, sol.state[-1])["kla"]) == SIGNAL
    params = plant.default_parameters()
    clean = _kla_history(plant, sol, params, ["tank"])
    process = _kla_history(plant, sol, params, ["tank"], process_water=True)
    assert float(clean[-1, 0]) == pytest.approx(SIGNAL * GAIN, rel=1e-15)
    assert float(process[-1, 0]) == pytest.approx(ALPHA * SIGNAL * GAIN, rel=1e-15)


def test_alpha_on_a_signal_matches_a_fixed_kla_of_the_same_effect(make_cstr_plant, fouled):
    """A fixed ``kla`` with the same ``alpha`` gives the same tank."""
    _plant, sol = fouled
    fixed = make_cstr_plant(
        aeration=Aeration(kla=SIGNAL * GAIN, species="A", do_sat=DO_SAT, alpha=ALPHA)
    )
    ref = _solve(fixed)
    np.testing.assert_allclose(sol.C_named("tank", "A"), ref.C_named("tank", "A"), rtol=1e-9)


def _pi_plant(make_cstr_plant, alpha):
    """DO control of ``A`` at 1.2 with the PI output capped at 0.25: a clean
    tank needs ``K = 0.175`` to hold it, a tank at ``alpha = 0.5`` twice that."""
    return make_cstr_plant(
        aeration=Aeration(
            do_setpoint=1.2, species="A", do_sat=DO_SAT, alpha=alpha,
            Kp=1.0, Ti=5.0, Tt=2.0, kla_offset=0.0, kla_min=0.0, kla_max=0.25,
        )
    )


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_a_fouled_closed_loop_tank_saturates_at_kla_max(make_cstr_plant, alpha):
    """Under the same load the clean tank holds the setpoint below ``kla_max``,
    while the fouled one drives its controller to ``kla_max`` and settles at the
    closed form for ``K = alpha * kla_max``."""
    plant = _pi_plant(make_cstr_plant, alpha)
    sol = _solve(plant)
    a = float(sol.C_named("tank", "A")[-1])
    kla = float(plant.signals_at(T_END, sol.state[-1])["_aer_tank_kla"])
    if alpha == 1.0:
        assert a == pytest.approx(1.2, rel=1e-6)
        assert kla == pytest.approx(0.14 / 0.8, rel=1e-6)
    else:
        assert kla == 0.25
        assert a == pytest.approx(_steady_A(alpha * 0.25), rel=1e-8)


def test_grad_through_a_fouled_signal_driven_solve_matches_a_central_difference(
    make_cstr_plant, simple_model
):
    """``d A(T) / d k`` with a proportional law on ``[A]`` and ``alpha = 0.5``."""
    plant = _signal_plant(make_cstr_plant, ALPHA, law="clip(0.2*(sp - [A]), 0, 1)", model=simple_model)
    base = plant.default_parameters()
    i = plant.parameter_index("simple_decay.A_to_B.k")

    def a_end(k):
        sol = plant.solve(
            (0.0, 20.0), t_eval=jnp.asarray([20.0]), params=base.at[i].set(k), rtol=1e-10,
            atol=1e-12,
        )
        return sol.C_named("tank", "A")[-1]

    k0, h = 0.1, 1e-4
    grad = float(jax.grad(a_end)(k0))
    fd = (float(a_end(k0 + h)) - float(a_end(k0 - h))) / (2.0 * h)
    # Central difference of an rtol=1e-10 solve with h = 1e-4: truncation ~h^2.
    assert grad == pytest.approx(fd, rel=1e-5)
    assert grad < 0.0


# --- which kLa each energy figure reads -------------------------------------------

V_TANK, S_SAT = 100.0, 8.0


def _energy_solve(plant):
    return plant.solve(
        (0.0, T_END), t_eval=jnp.asarray([T_END / 2, T_END]), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("mode", ["kla", "kla_signal"])
def test_benchmark_aeration_energy_reads_the_delivered_kla(make_cstr_plant, mode):
    """``AE = S_sat·V·kLa/1800`` on ``alpha × kLa`` for a fixed and a
    signal-driven tank alike (the clean-water kLa would give twice as much)."""
    if mode == "kla":
        plant = make_cstr_plant(
            aeration=Aeration(kla=SIGNAL * GAIN, species="A", do_sat=DO_SAT, alpha=ALPHA)
        )
    else:
        plant = _signal_plant(make_cstr_plant, ALPHA)
    sol = _energy_solve(plant)
    ae, air_flow, clean = _aeration_energy(
        plant, sol, plant.default_parameters(), sol.t, ["tank"], None, S_SAT
    )
    assert air_flow is None
    assert ae == pytest.approx(S_SAT / 1800.0 * ALPHA * SIGNAL * GAIN * V_TANK, rel=1e-12)
    np.testing.assert_allclose(clean, SIGNAL * GAIN, rtol=1e-15)


def test_benchmark_aeration_energy_includes_the_temperature_factor(make_cstr_plant):
    """With the temperature correction on, the delivered kLa of a signal-driven
    tank at 293.15 K specified at ``ref_T = 283.15`` carries ``1.024**10``."""
    aer = Aeration(
        kla_signal="kla", species="A", do_sat=DO_SAT, gain=GAIN, alpha=ALPHA,
        temperature_correction=True, ref_T=283.15,
    )
    plant = make_cstr_plant(aeration=aer)
    plant.add_unit(ExpressionController("ctl", signal_name="kla", law="c", constants={"c": SIGNAL}))
    sol = _energy_solve(plant)
    delivered = _delivered_kla_history(plant, sol, None, ["tank"])
    expected = ALPHA * SIGNAL * GAIN * 1.024**10.0
    np.testing.assert_allclose(delivered, expected, rtol=1e-12)
    ae, _air, _clean = _aeration_energy(plant, sol, None, sol.t, ["tank"], None, S_SAT)
    assert ae == pytest.approx(S_SAT / 1800.0 * expected * V_TANK, rel=1e-12)


def test_blower_energy_and_air_flow_read_the_clean_water_kla(fouled):
    """With an ``AerationSystem`` the air flow is that of the clean-water kLa
    ``signal × gain``, not of ``alpha`` times it."""
    plant, _sol = fouled
    sol = _energy_solve(plant)
    system = AerationSystem(depth=4.0)
    ae, air_flow, _clean = _aeration_energy(
        plant, sol, plant.default_parameters(), sol.t, ["tank"], system, S_SAT
    )
    clean_q = float(required_airflow(SIGNAL * GAIN, V_TANK, system))
    assert air_flow == pytest.approx(clean_q, rel=1e-12)
    assert ae == pytest.approx(float(blower_power_kw(clean_q, system)) * 24.0, rel=1e-12)
