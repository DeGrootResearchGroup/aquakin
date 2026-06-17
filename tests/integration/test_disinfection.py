"""Disinfection unit ops: UV dose-response and chlorine CT / log-removal (#280).

The UV reactor and the chlorine contact tank pass the process stream through and
reduce the indicator-organism density carried on the stream (``Stream.org``, else
a design ``inlet_density``). These checks pin the credit physics against its
closed form, the units' pass-through + log-removal, the indicator transport
through the flowsheet, and AD-cleanliness.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import (
    ChlorineContactUnit,
    UVUnit,
    ct_log_removal,
    ct_value,
    t10_from_baffling,
    t10_from_rtd,
    uv_dose,
    uv_log_inactivation,
)
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.plant import Plant
from aquakin.plant.streams import Stream, mixed_organism


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


# --- credit physics ---------------------------------------------------------

def test_uv_dose_and_log_inactivation():
    # dose = intensity * exposure; log = dose / d10
    assert float(uv_dose(3.5, 8.64)) == pytest.approx(30.24, rel=1e-9)
    assert float(uv_log_inactivation(30.24, 6.0)) == pytest.approx(5.04, rel=1e-9)
    # UVT correction scales the intensity linearly
    assert float(uv_dose(3.5, 8.64, uvt=65.0, uvt_ref=98.0)) == pytest.approx(
        30.24 * 65.0 / 98.0, rel=1e-9)
    # tailing cap
    assert float(uv_log_inactivation(1000.0, 6.0, max_log=4.0)) == pytest.approx(4.0)


def test_chlorine_ct_and_log_removal():
    # T10 = baffling * V/Q ; CT = residual * T10
    t10 = float(t10_from_baffling(500.0, 1000.0, 0.5))
    assert t10 == pytest.approx(0.25, rel=1e-9)               # 0.5 * 500/1000
    assert float(ct_value(2.0, t10)) == pytest.approx(0.5, rel=1e-9)
    assert float(ct_log_removal(0.5, 0.05)) == pytest.approx(10.0, rel=1e-9)
    assert float(ct_log_removal(100.0, 0.05, max_log=4.0)) == pytest.approx(4.0)


def test_t10_from_rtd_uses_percentile():
    """The non-ideal-contactor T10 is the 10th percentile of the RTD F-curve."""
    from aquakin.utils.rtd import percentile_time
    t = jnp.linspace(0.0, 2.0, 201)
    # a simple CSTR-like pulse response E(t) = exp(-t/tau)/tau
    C = jnp.exp(-t / 0.5)
    assert float(t10_from_rtd(t, C)) == pytest.approx(
        float(percentile_time(t, C, 0.10)), rel=1e-9)


# --- indicator on the stream ------------------------------------------------

def test_stream_org_roundtrip_and_mixing(asm1):
    C = asm1.default_concentrations()
    s = Stream(Q=jnp.asarray(10.0), C=C, network=asm1)
    assert s.org is None
    assert float(s.with_org(jnp.asarray(5.0)).org) == 5.0
    assert s.with_C(C).org is None                            # preserved (was None)
    # flow-weighted mix of two indicator-carrying inlets
    a = Stream(Q=jnp.asarray(10.0), C=C, network=asm1, org=jnp.asarray(100.0))
    b = Stream(Q=jnp.asarray(30.0), C=C, network=asm1, org=jnp.asarray(200.0))
    org = mixed_organism({"a": a, "b": b}, ["a", "b"])
    assert float(org) == pytest.approx((10 * 100 + 30 * 200) / 40)
    # an org-agnostic inlet is ignored, not allowed to poison the mix
    n = Stream(Q=jnp.asarray(5.0), C=C, network=asm1)
    assert float(mixed_organism({"a": a, "n": n}, ["a", "n"])) == pytest.approx(100.0)
    assert mixed_organism({"n": n}, ["n"]) is None            # fully agnostic -> None


# --- UV unit ----------------------------------------------------------------

def test_uv_unit_reduces_indicator_and_passes_process_through(asm1):
    uv = UVUnit("uv", asm1, volume=0.1, intensity=3.5, d10=6.0, inlet_density=1e6)
    C = asm1.default_concentrations()
    flow = jnp.asarray(1000.0)
    s_in = Stream(Q=flow, C=C, network=asm1, T=jnp.asarray(293.0),
                  org=jnp.asarray(1e6))
    out = uv.compute_outputs(0.0, uv.initial_state(), {"in": s_in},
                             asm1.default_parameters())["out"]
    log = float(uv.log_inactivation(flow))
    assert float(out.org) == pytest.approx(1e6 * 10 ** (-log), rel=1e-9)
    assert bool(jnp.allclose(out.C, C))                       # process stream unchanged
    assert float(out.Q) == 1000.0 and float(out.T) == 293.0   # Q / T pass through
    assert uv.state_size == 0
    # falls back to inlet_density when the inlet carries no indicator
    out2 = uv.compute_outputs(0.0, uv.initial_state(),
                              {"in": Stream(Q=flow, C=C, network=asm1)},
                              asm1.default_parameters())["out"]
    assert float(out2.org) == pytest.approx(1e6 * 10 ** (-log), rel=1e-9)


# --- chlorine contact unit --------------------------------------------------

def test_chlorine_residual_dynamics_and_validation(asm1):
    cl = ChlorineContactUnit("cl", asm1, volume=500.0, dose=5.0, ct_per_log=0.05,
                             decay_rate=2.0, baffling_factor=0.5)
    C = asm1.default_concentrations()
    s_in = Stream(Q=jnp.asarray(1000.0), C=C, network=asm1)
    # dCl/dt = (Q/V)(dose - Cl) - k*Cl  ; at Cl=2: (1000/500)(5-2) - 2*2 = 2
    d = cl.rhs(0.0, jnp.asarray([2.0]), {"in": s_in}, asm1.default_parameters())
    assert float(d[0]) == pytest.approx((1000.0 / 500.0) * (5.0 - 2.0) - 2.0 * 2.0)
    assert cl.state_size == 1
    with pytest.raises(ValueError, match="volume must be > 0"):
        ChlorineContactUnit("c", asm1, volume=0.0, dose=5.0, ct_per_log=1.0)
    with pytest.raises(ValueError, match="ct_per_log must be > 0"):
        ChlorineContactUnit("c", asm1, volume=1.0, dose=5.0, ct_per_log=0.0)


def test_chlorine_dechlorination_zeros_discharged_residual(asm1):
    cl = ChlorineContactUnit("cl", asm1, volume=500.0, dose=5.0, ct_per_log=1.0,
                             dechlorinate=True)
    assert cl.discharged_residual(jnp.asarray([2.0])) == 0.0
    cl2 = ChlorineContactUnit("cl", asm1, volume=500.0, dose=5.0, ct_per_log=1.0)
    assert cl2.discharged_residual(jnp.asarray([2.0])) == pytest.approx(2.0)


# --- plant integration ------------------------------------------------------

def test_disinfection_train_in_a_plant(asm1):
    """A chlorine + UV train reduces the indicator through the flowsheet, the
    chlorine residual reaches its steady state, and the effluent indicator is
    surfaced on the reconstructed stream."""
    p = Plant("disinfect")
    p.add_unit(ChlorineContactUnit("cl", asm1, volume=500.0, dose=5.0,
                                   ct_per_log=0.05, decay_rate=2.0,
                                   baffling_factor=0.5, inlet_density=1e6))
    p.add_unit(UVUnit("uv", asm1, volume=0.2, intensity=3.5, d10=6.0))
    p.add_influent("feed", InfluentSeries.constant(asm1, SS=60.0, SNH=20.0, Q=1000.0),
                   to="cl.in")
    p.connect("cl.out", "uv.in")
    sol = p.solve(t_span=(0.0, 2.0), t_eval=jnp.linspace(0.0, 2.0, 11))
    assert np.all(np.isfinite(np.asarray(sol.state)))
    # chlorine residual -> steady state dose/(1 + k*HRT) = 5/(1 + 2*0.5) = 2.5
    res = np.asarray(sol.unit_state("cl"))[:, 0]
    assert res[-1] == pytest.approx(2.5, abs=0.05)
    # effluent indicator is surfaced and monotonically reduced along the train
    cl_out = p.stream(sol, "cl.out")
    uv_out = p.stream(sol, "uv.out")
    assert cl_out.org is not None and uv_out.org is not None
    assert float(cl_out.org[-1]) < 1e6                        # chlorine reduced it
    assert float(uv_out.org[-1]) < float(cl_out.org[-1])      # UV reduced it further


def test_grad_through_disinfection_plant(asm1):
    p = Plant("disinfect")
    p.add_unit(ChlorineContactUnit("cl", asm1, volume=500.0, dose=5.0,
                                   ct_per_log=0.5, decay_rate=2.0))
    p.add_influent("feed", InfluentSeries.constant(asm1, SS=60.0, Q=1000.0),
                   to="cl.in")
    base = p.default_parameters()
    g = jax.grad(lambda s: jnp.sum(
        p.solve(t_span=(0.0, 1.0), t_eval=jnp.array([1.0]), params=base * s).state ** 2))(1.0)
    assert jnp.isfinite(g)


def test_grad_through_credit_physics():
    """jax.grad flows through the dose / CT credit (design optimisation)."""
    g_uv = jax.grad(lambda i: uv_log_inactivation(uv_dose(i, 8.64), 6.0))(3.5)
    g_ct = jax.grad(lambda r: ct_log_removal(ct_value(r, 0.25), 0.05))(2.0)
    assert jnp.isfinite(g_uv) and jnp.isfinite(g_ct)
    assert float(g_uv) == pytest.approx(8.64 / 6.0, rel=1e-9)   # d(dose/d10)/di
    assert float(g_ct) == pytest.approx(0.25 / 0.05, rel=1e-9)  # d(CT/ctpl)/dr
