"""Tests for the BSM2 Otterpohl–Freund primary clarifier."""

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.primary_clarifier import PrimaryClarifier
from aquakin.plant.streams import Stream


@pytest.fixture
def asm1():
    return aquakin.load_network("asm1")


def _state(asm1, overrides):
    C = asm1.default_concentrations()
    for sp, v in overrides.items():
        C = C.at[asm1.species_index[sp]].set(v)
    return C


def _outputs(unit, asm1, Q_in, state):
    s_in = Stream(Q=jnp.asarray(float(Q_in)), C=state, network=asm1)
    return unit.compute_outputs(jnp.asarray(0.0), state, {"inlet": s_in},
                                jnp.zeros((0,)))


def test_flow_split_is_fps(asm1):
    """Underflow = f_PS·Q_in; effluent the remainder; flows conserve volume."""
    unit = PrimaryClarifier(name="pc", network=asm1, volume=900.0)
    Q_in = 21000.0
    out = _outputs(unit, asm1, Q_in, asm1.default_concentrations())
    assert float(out["underflow"].Q) == pytest.approx(0.007 * Q_in, rel=1e-9)
    assert float(out["effluent"].Q) == pytest.approx((1 - 0.007) * Q_in, rel=1e-9)
    # flow_outputs (linear pre-solve) agrees.
    fo = unit.flow_outputs({"inlet": jnp.asarray(Q_in)}, jnp.zeros((0,)))
    assert float(fo["underflow"]) == pytest.approx(0.007 * Q_in, rel=1e-9)
    assert float(fo["effluent"] + fo["underflow"]) == pytest.approx(Q_in, rel=1e-9)


def test_mass_balance_on_tank_content(asm1):
    """The two outlets partition the tank content exactly:
    (Q_in-Qu)*C_eff + Qu*C_sludge == Q_in * state, per species."""
    unit = PrimaryClarifier(name="pc", network=asm1, volume=900.0)
    Q_in = 21000.0
    state = _state(asm1, {"XI": 95.0, "XS": 360.0, "XB_H": 50.0, "XB_A": 0.1,
                          "XP": 0.7, "XND": 16.0, "SS": 59.0, "SNH": 35.0})
    out = _outputs(unit, asm1, Q_in, state)
    mass_out = (float(out["effluent"].Q) * out["effluent"].C
                + float(out["underflow"].Q) * out["underflow"].C)
    assert jnp.allclose(mass_out, Q_in * state, rtol=1e-9, atol=1e-6)


def test_solubles_unremoved(asm1):
    """Soluble species are not removed: same concentration in both outlets,
    equal to the tank content."""
    unit = PrimaryClarifier(name="pc", network=asm1, volume=900.0)
    state = _state(asm1, {"SS": 59.0, "SNH": 35.0, "SI": 28.0})
    out = _outputs(unit, asm1, 21000.0, state)
    for sp in ("SI", "SS", "SNH", "SND", "SALK", "SO", "SNO"):
        i = asm1.species_index[sp]
        assert float(out["effluent"].C[i]) == pytest.approx(float(state[i]), rel=1e-9)
        assert float(out["underflow"].C[i]) == pytest.approx(float(state[i]), rel=1e-9)


def test_removal_efficiency_matches_otterpohl(asm1):
    """The particulate-removal fraction matches the Otterpohl formula at the
    BSM2 operating point (~0.48 at HRT ≈ 0.043 d)."""
    unit = PrimaryClarifier(name="pc", network=asm1, volume=900.0)
    n_x = float(unit._removal_fraction(jnp.asarray(21000.0)))
    assert n_x == pytest.approx(0.4775, abs=0.01)
    # Particulate XI effluent fraction = 1 - n_x.
    state = _state(asm1, {"XI": 100.0})
    out = _outputs(unit, asm1, 21000.0, state)
    i = asm1.species_index["XI"]
    assert float(out["effluent"].C[i]) == pytest.approx(100.0 * (1 - n_x), rel=1e-6)


def test_mixing_rhs_relaxes_to_inlet(asm1):
    """At steady state (state == inlet composition) the mixing rhs is zero."""
    unit = PrimaryClarifier(name="pc", network=asm1, volume=900.0)
    C_in = _state(asm1, {"XS": 360.0, "SS": 59.0})
    s_in = Stream(Q=jnp.asarray(21000.0), C=C_in, network=asm1)
    d = unit.rhs(jnp.asarray(0.0), C_in, {"inlet": s_in}, jnp.zeros((0,)))
    assert jnp.allclose(d, 0.0, atol=1e-9)
    # Off steady state, it drives toward the inlet.
    d2 = unit.rhs(jnp.asarray(0.0), C_in * 0.5, {"inlet": s_in}, jnp.zeros((0,)))
    assert jnp.all(d2 >= -1e-9)  # inlet > state everywhere -> non-negative drive


def test_grad_through_primary_clarifier(asm1):
    unit = PrimaryClarifier(name="pc", network=asm1, volume=900.0)
    state = _state(asm1, {"XS": 360.0, "XB_H": 50.0})

    def loss(C):
        out = _outputs(unit, asm1, 21000.0, C)
        return jnp.sum(out["underflow"].C) + jnp.sum(out["effluent"].C)

    g = jax.grad(loss)(state)
    assert jnp.all(jnp.isfinite(g))
