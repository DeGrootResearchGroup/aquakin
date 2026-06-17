"""Cached recycle affine-map optimization.

The recycle concentration map ``c -> M c + d`` has ``M`` fixed by the recycle
flows + topology (mixer/splitter/clarifier ratios). For a fixed-pump plant ``M``
is invariant to the state and time, so it is precomputed once per solve and
reused -- skipping the per-RHS per-species probe sweeps (the dominant per-RHS
cost). The temperature map ``MT`` is cached too when it is state-invariant
(heat-balance / no-temperature modes) and re-probed per RHS otherwise (algebraic
mode, where T passes through reactors and rides on the concentration-dependent
recycle flows). A non-affine / flow-coupled topology where ``M`` itself moves
with the state falls back to per-RHS probing. The cached and probed paths give a
bit-identical RHS; these tests pin that, the constancy detection, and the
fallback.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.cstr import Aeration, CSTRUnit
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.mixer import MixerUnit, SplitterUnit
from aquakin.plant.plant import Plant


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


def _recycle_plant(asm1, carry_T=False):
    """Mixer (fresh + recycle) -> CSTR -> splitter, splitter.rec back to mixer.
    Fixed-ratio splitter -> fixed recycle flows -> state-invariant M."""
    plant = Plant("recycle")
    plant.add_unit(MixerUnit("mix", ["fresh", "rec"], asm1))
    plant.add_unit(CSTRUnit("tank", asm1, volume=1000.0, input_port_names=["inlet"],
                            conditions={"T": 293.15},
                            aeration=Aeration(kla=120.0, do_sat=8.0)))
    plant.add_unit(SplitterUnit("split", asm1,
                                output_port_ratios={"out": 0.7, "rec": 0.3}))
    T = 291.0 if carry_T else None
    plant.add_influent("feed", InfluentSeries.constant(
        asm1, SS=120.0, SNH=30.0, XS=80.0, XB_H=40.0, Q=1000.0, T=T),
        to="mix.fresh")
    plant.connect("mix", "tank")
    plant.connect("tank", "split")
    plant.connect("split.rec", "mix.rec")
    return plant


def _prime(plant, params=None):
    """A trivial solve to build layouts + run the one-time constancy check."""
    y0 = plant.initial_state()
    plant.solve(t_span=(0.0, 0.05), t_eval=jnp.array([0.05]), params=params,
                y0=y0, rtol=1e-4, atol=1e-3, max_steps=100_000)
    return y0


def test_M_constant_detected(asm1):
    plant = _recycle_plant(asm1, carry_T=False)
    _prime(plant)
    assert plant._recycle_map_constant is True      # fixed-pump recycle
    assert plant._recycle_T_map_constant is True     # no temperature carried


def test_cached_rhs_bit_identical_to_probe(asm1):
    """The core correctness invariant: at a fixed state, the cached-M RHS equals
    the per-RHS-probed RHS to the bit (the cached M IS the probed M)."""
    plant = _recycle_plant(asm1, carry_T=False)
    y0 = _prime(plant)
    pf = plant.default_parameters()
    t = jnp.asarray(2.0)
    states = plant._split_state(y0)
    sig = plant._compute_signals(t, states, pf)
    flows = plant._resolve_flows(t, pf, states)
    rmap = plant._compute_recycle_map(t, states, pf, flows, sig)
    cached = np.asarray(plant._rhs(t, y0, pf, recycle_map=rmap))
    probe = np.asarray(plant._rhs(t, y0, pf, recycle_map=None))
    assert np.array_equal(cached, probe)             # bit-identical


def test_cached_solve_matches_probe(asm1):
    plant = _recycle_plant(asm1, carry_T=False)
    y0 = _prime(plant)
    t_eval = jnp.linspace(0.0, 5.0, 11)
    cached = plant.solve(t_span=(0.0, 5.0), t_eval=t_eval, y0=y0,
                         rtol=1e-6, atol=1e-8, max_steps=1_000_000)
    plant._recycle_map_constant = False              # force the probe path
    plant._jit_cache.clear()
    probe = plant.solve(t_span=(0.0, 5.0), t_eval=t_eval, y0=y0,
                        rtol=1e-6, atol=1e-8, max_steps=1_000_000)
    rel = np.max(np.abs(np.asarray(cached.state) - np.asarray(probe.state))
                 / (np.abs(np.asarray(probe.state)) + 1e-6))
    assert rel < 1e-6
    assert np.all(np.isfinite(np.asarray(cached.state)))


def test_temperature_map_constancy_by_mode(asm1):
    """Algebraic mode: MT rides on the (here fixed) flows -> for this fixed-pump
    plant MT is also constant. The key contrast (MT varies in algebraic mode) is
    exercised at BSM2 scale by the steady-state validation; here we pin that a
    temperature-carrying plant still detects M-constant and solves finite."""
    plant = _recycle_plant(asm1, carry_T=True)
    y0 = _prime(plant)
    assert plant._recycle_map_constant is True
    sol = plant.solve(t_span=(0.0, 3.0), t_eval=jnp.array([3.0]), y0=y0,
                      rtol=1e-5, atol=1e-3, max_steps=1_000_000)
    assert np.all(np.isfinite(np.asarray(sol.state)))


def test_no_recycle_plant_trivially_constant(asm1):
    """A plant with no recycle edges: the maps are trivially constant and the
    resolver returns no seeds -- the optimization is a no-op."""
    plant = Plant("once-through")
    plant.add_unit(CSTRUnit("tank", asm1, volume=1000.0, input_port_names=["inlet"],
                            conditions={"T": 293.15}))
    plant.add_influent("feed", InfluentSeries.constant(asm1, SS=120.0, Q=1000.0),
                       to="tank.inlet")
    y0 = _prime(plant)
    assert plant._recycle_map_constant is True
    assert plant._recycle_T_map_constant is True
    sol = plant.solve(t_span=(0.0, 2.0), t_eval=jnp.array([2.0]), y0=y0,
                      rtol=1e-5, atol=1e-3, max_steps=100_000)
    assert np.all(np.isfinite(np.asarray(sol.state)))


def test_grad_flows_through_cached_path(asm1):
    plant = _recycle_plant(asm1, carry_T=False)
    y0 = _prime(plant)
    snh = asm1.species_index["SNH"]
    base = plant.default_parameters()

    def loss(scale):
        sol = plant.solve(t_span=(0.0, 2.0), t_eval=jnp.array([2.0]),
                          params=base * scale, y0=y0, gradient="jax_adjoint",
                          rtol=1e-5, atol=1e-3, max_steps=1_000_000)
        return jnp.sum(sol.state[-1] ** 2)

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)
