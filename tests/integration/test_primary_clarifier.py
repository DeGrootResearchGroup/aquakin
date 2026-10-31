"""Tests for the BSM2 Otterpohl–Freund primary clarifier."""

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.primary_clarifier import PrimaryClarifier
from aquakin.plant.streams import Stream


@pytest.fixture
def asm1():
    return aquakin.load_model("asm1")


def _state(asm1, overrides):
    return asm1.concentrations(overrides)


def _outputs(unit, asm1, Q_in, state):
    s_in = Stream(Q=jnp.asarray(float(Q_in)), C=state, model=asm1)
    return unit.compute_outputs(jnp.asarray(0.0), state, {"inlet": s_in},
                                jnp.zeros((0,)))


def test_flow_split_is_fps(asm1):
    """Underflow = f_PS·Q_in; effluent the remainder; flows conserve volume."""
    unit = PrimaryClarifier(name="pc", model=asm1, volume=900.0)
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
    unit = PrimaryClarifier(name="pc", model=asm1, volume=900.0)
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
    unit = PrimaryClarifier(name="pc", model=asm1, volume=900.0)
    state = _state(asm1, {"SS": 59.0, "SNH": 35.0, "SI": 28.0})
    out = _outputs(unit, asm1, 21000.0, state)
    for sp in ("SI", "SS", "SNH", "SND", "SALK", "SO", "SNO"):
        i = asm1.species_index[sp]
        assert float(out["effluent"].C[i]) == pytest.approx(float(state[i]), rel=1e-9)
        assert float(out["underflow"].C[i]) == pytest.approx(float(state[i]), rel=1e-9)


def test_removal_efficiency_matches_otterpohl(asm1):
    """The particulate-removal fraction matches the Otterpohl formula at the
    BSM2 operating point (~0.48 at HRT ≈ 0.043 d)."""
    unit = PrimaryClarifier(name="pc", model=asm1, volume=900.0)
    n_x = float(unit._removal_fraction(jnp.asarray(21000.0)))
    assert n_x == pytest.approx(0.4775, abs=0.01)
    # Particulate XI effluent fraction = 1 - n_x.
    state = _state(asm1, {"XI": 100.0})
    out = _outputs(unit, asm1, 21000.0, state)
    i = asm1.species_index["XI"]
    assert float(out["effluent"].C[i]) == pytest.approx(100.0 * (1 - n_x), rel=1e-6)


def test_mixing_rhs_relaxes_to_inlet(asm1):
    """At steady state (state == inlet composition) the mixing rhs is zero."""
    unit = PrimaryClarifier(name="pc", model=asm1, volume=900.0)
    C_in = _state(asm1, {"XS": 360.0, "SS": 59.0})
    s_in = Stream(Q=jnp.asarray(21000.0), C=C_in, model=asm1)
    d = unit.rhs(jnp.asarray(0.0), C_in, {"inlet": s_in}, jnp.zeros((0,)))
    assert jnp.allclose(d, 0.0, atol=1e-9)
    # Off steady state, it drives toward the inlet.
    d2 = unit.rhs(jnp.asarray(0.0), C_in * 0.5, {"inlet": s_in}, jnp.zeros((0,)))
    assert jnp.all(d2 >= -1e-9)  # inlet > state everywhere -> non-negative drive


def test_grad_through_primary_clarifier(asm1):
    unit = PrimaryClarifier(name="pc", model=asm1, volume=900.0)
    state = _state(asm1, {"XS": 360.0, "XB_H": 50.0})

    def loss(C):
        out = _outputs(unit, asm1, 21000.0, C)
        return jnp.sum(out["underflow"].C) + jnp.sum(out["effluent"].C)

    g = jax.grad(loss)(state)
    assert jnp.all(jnp.isfinite(g))


# --- the first-order inflow filter on the HRT (t_m > 0) ----------------------

_T_M = 0.125


def _filtered(asm1, Q_f):
    return PrimaryClarifier(name="pc", model=asm1, volume=900.0, t_m=_T_M, initial_flow=Q_f)


def test_filter_adds_one_trailing_state(asm1):
    """With t_m > 0 the state is [C..., Q_f], seeded with initial_flow; t_m = 0
    keeps the bare concentration vector."""
    plain = PrimaryClarifier(name="pc", model=asm1, volume=900.0)
    assert plain.state_size == asm1.n_species
    assert jnp.array_equal(plain.initial_state(), asm1.default_concentrations())
    unit = _filtered(asm1, 21000.0)
    assert unit.state_size == asm1.n_species + 1
    y0 = unit.initial_state()
    assert y0.shape == (asm1.n_species + 1,)
    assert float(y0[-1]) == 21000.0
    assert jnp.array_equal(y0[:-1], asm1.default_concentrations())


def test_filter_is_identity_at_steady_state(asm1):
    """With Q_f equal to the inflow, the filtered unit's outlets equal the
    unfiltered unit's and its whole rhs is zero at the inlet composition."""
    Q_in = 21000.0
    C = _state(asm1, {"XI": 95.0, "XS": 360.0, "XB_H": 50.0, "SS": 59.0, "SNH": 35.0})
    plain = _outputs(PrimaryClarifier(name="pc", model=asm1, volume=900.0), asm1, Q_in, C)
    unit = _filtered(asm1, Q_in)
    s_in = Stream(Q=jnp.asarray(Q_in), C=C, model=asm1)
    state = jnp.concatenate([C, jnp.asarray([Q_in])])
    out = unit.compute_outputs(jnp.asarray(0.0), state, {"inlet": s_in}, jnp.zeros((0,)))
    for port in ("effluent", "underflow"):
        assert float(out[port].Q) == pytest.approx(float(plain[port].Q), rel=1e-12)
        assert jnp.allclose(out[port].C, plain[port].C, rtol=1e-12, atol=0.0)
    d = unit.rhs(jnp.asarray(0.0), state, {"inlet": s_in}, jnp.zeros((0,)))
    assert jnp.allclose(d, 0.0, atol=1e-9)


def test_removal_reads_the_filtered_flow_and_the_split_the_inflow(asm1):
    """Off steady state the removal fraction follows Q_f, while the underflow is
    still f_PS times the instantaneous inflow (the exact recycle-flow rule)."""
    Q_f, Q_in = 21000.0, 42000.0
    unit = _filtered(asm1, Q_f)
    C = _state(asm1, {"XI": 100.0})
    s_in = Stream(Q=jnp.asarray(Q_in), C=C, model=asm1)
    state = jnp.concatenate([C, jnp.asarray([Q_f])])
    out = unit.compute_outputs(jnp.asarray(0.0), state, {"inlet": s_in}, jnp.zeros((0,)))
    i = asm1.species_index["XI"]
    n_x_filtered = float(unit._removal_fraction(jnp.asarray(Q_f)))
    assert n_x_filtered != pytest.approx(float(unit._removal_fraction(jnp.asarray(Q_in))))
    assert float(out["effluent"].C[i]) == pytest.approx(100.0 * (1 - n_x_filtered), rel=1e-12)
    assert float(out["underflow"].Q) == pytest.approx(0.007 * Q_in, rel=1e-12)
    fo = unit.flow_outputs({"inlet": jnp.asarray(Q_in)}, jnp.zeros((0,)))
    assert float(fo["underflow"]) == pytest.approx(float(out["underflow"].Q), rel=1e-12)


def test_inflow_step_relaxes_the_filtered_flow_with_time_constant_t_m(asm1):
    """After an inflow step Q0 -> Q1, Q_f(t) = Q1 + (Q0 - Q1)·exp(-t/t_m), and the
    removal fraction follows that relaxing flow, not the stepped one."""
    import diffrax

    Q0, Q1 = 21000.0, 42000.0
    unit = _filtered(asm1, Q0)
    C = _state(asm1, {"XI": 100.0, "XS": 300.0})
    s_in = Stream(Q=jnp.asarray(Q1), C=C, model=asm1)
    y0 = jnp.concatenate([C, jnp.asarray([Q0])])

    def f(t, y, args):
        return unit.rhs(t, y, {"inlet": s_in}, jnp.zeros((0,)))

    ts = jnp.asarray([0.5 * _T_M, _T_M, 3.0 * _T_M])
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), diffrax.Kvaerno5(), 0.0, float(ts[-1]), 1e-4, y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-8),
    )
    expected = Q1 + (Q0 - Q1) * jnp.exp(-ts / _T_M)
    assert jnp.allclose(sol.ys[:, -1], expected, rtol=1e-7)
    # One time constant in: removal is set by Q_f(t_m), between the two flows.
    y_tm = sol.ys[1]
    out = unit.compute_outputs(jnp.asarray(_T_M), y_tm, {"inlet": s_in}, jnp.zeros((0,)))
    i = asm1.species_index["XI"]
    n_x = float(unit._removal_fraction(expected[1]))
    assert float(out["effluent"].C[i]) == pytest.approx(float(y_tm[i]) * (1 - n_x), rel=1e-6)


def test_grad_through_the_filtered_flow(asm1):
    """The effluent particulates depend on Q_f through the HRT: a larger filtered
    flow is a shorter HRT, less removal, more particulates in the effluent."""
    unit = _filtered(asm1, 21000.0)
    C = _state(asm1, {"XI": 100.0})
    s_in = Stream(Q=jnp.asarray(21000.0), C=C, model=asm1)
    i = asm1.species_index["XI"]

    def eff_xi(Q_f):
        state = jnp.concatenate([C, jnp.reshape(Q_f, (1,))])
        out = unit.compute_outputs(jnp.asarray(0.0), state, {"inlet": s_in}, jnp.zeros((0,)))
        return out["effluent"].C[i]

    g = float(jax.grad(eff_xi)(jnp.asarray(21000.0)))
    h = 1.0
    fd = float(eff_xi(jnp.asarray(21000.0 + h)) - eff_xi(jnp.asarray(21000.0 - h))) / (2 * h)
    assert g > 0.0
    assert g == pytest.approx(fd, rel=1e-5)
