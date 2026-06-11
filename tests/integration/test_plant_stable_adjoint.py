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
from aquakin.plant.bsm.bsm2 import (
    build_bsm2,
    bsm2_constant_influent,
    bsm2_parameters,
)

# Near-uniform activated-sludge warm start (the slow inerts are ~uniform across
# reactors at the published steady state; the fast variables relax within hours),
# so a short solve starts near steady state instead of from a stiff clean start.
_WARM = {"SI": 28.06, "SS": 2.0, "XI": 1532.3, "XS": 45.0, "XB_H": 2244.0,
         "XB_A": 167.0, "XP": 967.0, "SO": 1.0, "SNO": 7.0, "SNH": 3.0,
         "SND": 0.7, "XND": 3.0, "SALK": 5.0}
_TANKS = ("tank1", "tank2", "tank3", "tank4", "tank5")


def _bsm2_plant():
    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")
    warm = asm1.concentrations(_WARM)
    y0 = plant.initial_state(overrides={tk: warm for tk in _TANKS})
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


# --- the cross-interface gradient (slow: integrates the whole plant) -------

def _solve_kwargs():
    # The warm-started forward takes ~205 adaptive steps over a few days, so a
    # small max_steps suffices. Under gradient="stable_adjoint" max_steps also
    # sizes the backward scan's trajectory buffer, so keeping it tight is what
    # keeps the reverse pass cheap.
    return dict(rtol=1e-5, atol=1e-3, max_steps=2_000)


@pytest.mark.validation
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
def test_stable_adjoint_cross_interface_gradient_matches_fd():
    """Gradient of a water-line output with respect to an ADM1 (digester) rate,
    through the interface and the recycle, is finite and matches central FD."""
    asm1, adm1, plant, y0 = _bsm2_plant()
    base = bsm2_parameters(asm1, adm1)
    gidx = asm1.n_params + adm1.param_index["k_m_ac"]   # acetate-uptake max rate
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
    assert grad == pytest.approx(fd, rel=2e-3)


@pytest.mark.validation
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
