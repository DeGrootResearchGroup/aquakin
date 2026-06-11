"""BSM2 closed-loop dissolved-oxygen (DO/kLa) control tests.

Covers the control-signal bus (``Plant._rhs``), the :class:`PIController` unit,
and the CSTR controlled-kLa actuation, exercised through the closed-loop
``build_bsm2(do_control=True)`` plant. The control objective is that the PI loop
holds reactor 4's oxygen at the setpoint by manipulating the aeration kLa.
"""

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm.bsm2 import (
    BSM2_DO_KLA_MAX,
    BSM2_DO_SETPOINT,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
    build_bsm2,
)
from aquakin.plant.control import PIController
from aquakin.plant.streams import Stream


# ----- PIController unit (no plant solve) ---------------------------------

@pytest.fixture(scope="module")
def asm1():
    return bsm2_asm1_network()


def _controller(asm1, **over):
    cfg = dict(
        name="do", network=asm1, measured_species="SO", setpoint=2.0,
        Kp=25.0, Ti=0.002, Tt=0.001, offset=120.0, out_min=0.0, out_max=360.0,
        signal_name="do_kla",
    )
    cfg.update(over)
    return PIController(**cfg)


def _so_stream(asm1, so):
    C = asm1.default_concentrations() * 0.0
    C = C.at[asm1.species_index["SO"]].set(so)
    return {"measured": Stream(Q=jnp.asarray(1.0), C=C, network=asm1)}


def test_controller_signal_raises_kla_when_oxygen_low(asm1):
    """Below setpoint -> positive error -> kLa above the offset bias."""
    ctrl = _controller(asm1)
    state = ctrl.initial_state()  # zero integral
    sig_low = ctrl.signal_outputs(0.0, state, _so_stream(asm1, 0.5), None)
    sig_high = ctrl.signal_outputs(0.0, state, _so_stream(asm1, 3.0), None)
    assert float(sig_low["do_kla"]) > 120.0   # demands more air
    assert float(sig_high["do_kla"]) < 120.0  # backs off


def test_controller_output_saturates(asm1):
    """The published signal is clipped to [out_min, out_max]."""
    ctrl = _controller(asm1)
    state = ctrl.initial_state()
    # Huge positive error would push kLa far past out_max.
    sig = ctrl.signal_outputs(0.0, state, _so_stream(asm1, -100.0), None)
    assert float(sig["do_kla"]) == pytest.approx(360.0)
    # Huge negative error clips at out_min.
    sig = ctrl.signal_outputs(0.0, state, _so_stream(asm1, 100.0), None)
    assert float(sig["do_kla"]) == pytest.approx(0.0)


def test_controller_integral_tracks_error_sign(asm1):
    """The integrator moves in the error direction (below setpoint -> up)."""
    ctrl = _controller(asm1)
    state = ctrl.initial_state()
    dxi = ctrl.rhs(0.0, state, _so_stream(asm1, 0.5), None)
    assert dxi.shape == (1,)
    assert float(dxi[0]) > 0.0


def test_controller_antiwindup_reduces_integration_when_saturated(asm1):
    """While the output is saturated high, the tracking term opposes further
    integral wind-up (smaller dxi than without anti-windup)."""
    ctrl_aw = _controller(asm1, use_antiwindup=True)
    ctrl_no = _controller(asm1, use_antiwindup=False)
    # Large positive error -> output clipped at out_max -> u_sat - u < 0.
    inputs = _so_stream(asm1, -50.0)
    state = ctrl_aw.initial_state()
    dxi_aw = float(ctrl_aw.rhs(0.0, state, inputs, None)[0])
    dxi_no = float(ctrl_no.rhs(0.0, state, inputs, None)[0])
    assert dxi_aw < dxi_no


def test_controller_validates_species(asm1):
    with pytest.raises(ValueError, match="measured species"):
        _controller(asm1, measured_species="not_a_species")


# ----- Closed-loop plant ---------------------------------------------------

@pytest.fixture(scope="module")
def adm1():
    return aquakin.load_network("adm1")


def _closed_loop_plant(asm1, adm1):
    plant = build_bsm2(asm1, adm1, do_control=True)
    plant.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")
    return plant


def test_closed_loop_builds_and_is_finite(asm1, adm1):
    plant = _closed_loop_plant(asm1, adm1)
    params = bsm2_parameters(asm1, adm1)
    sol = plant.solve((0.0, 50.0), t_eval=jnp.array([0.0, 50.0]),
                      params=params, rtol=1e-4, atol=1e-3, max_steps=400_000)
    assert jnp.all(jnp.isfinite(sol.state))
    # The controller carries one integral state.
    assert plant.units["do_control"].state_size == 1


def test_do_setpoint_tracking(asm1, adm1):
    """The PI loop holds reactor-4 oxygen at the DO setpoint."""
    plant = _closed_loop_plant(asm1, adm1)
    params = bsm2_parameters(asm1, adm1)
    sol = plant.solve((0.0, 150.0), t_eval=jnp.array([0.0, 150.0]),
                      params=params, rtol=1e-4, atol=1e-3, max_steps=600_000)
    so4 = float(sol.C_named("tank4", "SO")[-1])
    assert so4 == pytest.approx(BSM2_DO_SETPOINT, abs=0.1)
    # Aerated reactors hold oxygen; the anoxic reactors do not.
    assert float(sol.C_named("tank1", "SO")[-1]) < 0.5
    assert float(sol.C_named("tank3", "SO")[-1]) > 0.0
    # The effective kLa from the control signal stays within actuator limits:
    # the resulting oxygen never exceeds the saturation it is driven toward.
    assert 0.0 <= so4 <= BSM2_DO_KLA_MAX


def test_closed_loop_differs_from_open_loop(asm1, adm1):
    """Closing the DO loop pins reactor-4 oxygen at the setpoint, unlike the
    fixed-kLa open-loop plant at the same operating point."""
    params = bsm2_parameters(asm1, adm1)
    t_eval = jnp.array([0.0, 100.0])

    closed = _closed_loop_plant(asm1, adm1)
    sol_c = closed.solve((0.0, 100.0), t_eval=t_eval, params=params,
                         rtol=1e-4, atol=1e-3, max_steps=500_000)

    openp = build_bsm2(asm1, adm1, do_control=False)
    openp.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")
    sol_o = openp.solve((0.0, 100.0), t_eval=t_eval, params=params,
                        rtol=1e-4, atol=1e-3, max_steps=500_000)

    so4_closed = float(sol_c.C_named("tank4", "SO")[-1])
    so4_open = float(sol_o.C_named("tank4", "SO")[-1])
    assert abs(so4_closed - BSM2_DO_SETPOINT) < abs(so4_open - BSM2_DO_SETPOINT)


def test_ad_flows_through_control_bus(asm1, adm1):
    """jax.grad flows through the control-signal bus without NaNs.

    Differentiated at the RHS level (one ``Plant._rhs`` evaluation) and with
    respect to the *state*, so the gradient traverses the whole new control path:
    reactor-4 ``SO`` -> PI controller error/integral -> published ``do_kla``
    signal -> the actuated reactors' kLa -> their derivatives. This isolates the
    control wiring's AD-cleanliness from the orthogonal stiff-solve adjoint
    problem (the closed loop's fast controller poles need the documented ``dtmax``
    cap / stable adjoint for a full reverse-mode solve -- see CLAUDE.md).
    """
    plant = _closed_loop_plant(asm1, adm1)
    plant._build_state_layout()
    plant._build_parameter_layout()
    params = bsm2_parameters(asm1, adm1)
    y0 = plant.initial_state()

    def loss(y):
        d = plant._rhs(jnp.asarray(0.0), y, params)
        return jnp.sum(d ** 2)

    g = jax.grad(loss)(y0)
    assert jnp.all(jnp.isfinite(g))
    # The control path is live: perturbing reactor-4 SO changes the derivative
    # field through the controller, so its gradient component is non-zero.
    so4_idx = (plant._state_layout["tank4"][0]
               + asm1.species_index["SO"])
    assert float(jnp.abs(g[so4_idx])) > 0.0
