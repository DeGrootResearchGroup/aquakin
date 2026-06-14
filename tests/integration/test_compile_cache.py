"""Compiled-solver caching: load_network, cross-instance reactor cache, plant cache.

Compiling a stiff solve (trace + lower + XLA) dominates its cost; the run is
comparatively free. These tests check the caching that avoids recompiling an
identical solve -- and, crucially, that it never changes results and is bypassed
safely under tracing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.integrate import _common
from aquakin.plant import CSTRUnit, InfluentSeries, Plant


# ----- load_network caching -----------------------------------------------

def test_load_network_returns_cached_object():
    a = aquakin.load_network("asm1")
    b = aquakin.load_network("asm1")
    assert a is b  # same object -> stable id for the solver cache


def test_clear_network_cache_forces_reload():
    from aquakin.schema.loader import clear_network_cache

    a = aquakin.load_network("asm1")
    clear_network_cache()
    b = aquakin.load_network("asm1")
    assert a is not b
    # Re-cache so other tests keep sharing one object.
    assert aquakin.load_network("asm1") is b


# ----- Cross-instance reactor solver cache --------------------------------

def _batch(net):
    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    return aquakin.BatchReactor(net, cond)


def test_fresh_reactors_share_one_compiled_solver():
    """Two fresh reactors for the same network + settings + signature reuse a
    single compiled solve (no second compile entry), with identical results."""
    net = aquakin.load_network("asm1")
    C0, p = net.default_concentrations(), net.default_parameters()
    t_eval = jnp.array([1.0])

    s1 = _batch(net).solve(C0, (0.0, 1.0), t_eval, params=p)
    n_after_first = len(_common._SOLVER_CACHE)
    s2 = _batch(net).solve(C0, (0.0, 1.0), t_eval, params=p)  # fresh reactor, same key
    assert len(_common._SOLVER_CACHE) == n_after_first  # reused, no new compile
    assert np.allclose(np.asarray(s1.C), np.asarray(s2.C))


def test_different_settings_do_not_collide():
    """A different tolerance is a different compile (no false cache hit)."""
    net = aquakin.load_network("asm1")
    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    C0, p = net.default_concentrations(), net.default_parameters()
    s_loose = aquakin.BatchReactor(net, cond, rtol=1e-4).solve(C0, (0.0, 1.0), params=p)
    s_tight = aquakin.BatchReactor(net, cond, rtol=1e-9).solve(C0, (0.0, 1.0), params=p)
    assert np.all(np.isfinite(np.asarray(s_loose.C)))
    assert np.all(np.isfinite(np.asarray(s_tight.C)))


def test_each_compile_affecting_setting_keys_the_cache(simple_network):
    """Every setting that changes the compiled solve -- rtol, atol, adjoint,
    dtmax, max_steps -- must produce a DISTINCT cache entry. A future edit that
    dropped one from ``settings_cache_key`` would otherwise be a silent
    wrong-result cache hit; here it surfaces as a failure. (The toy A->B network
    keeps the compiles cheap; the cache mechanism is network-agnostic.)"""
    import diffrax

    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    C0 = jnp.asarray([1.0, 0.0])
    p = simple_network.default_parameters()
    R = aquakin.BatchReactor

    def adds_entry(reactor_a, reactor_b):
        # Warm A's key, then assert B introduces a NEW compiled entry.
        reactor_a.solve(C0, (0.0, 1.0), params=p)
        n = len(_common._SOLVER_CACHE)
        reactor_b.solve(C0, (0.0, 1.0), params=p)
        return len(_common._SOLVER_CACHE) > n

    # Distinctive values, so the keys cannot pre-exist from another test.
    assert adds_entry(R(simple_network, cond, rtol=3.3e-5),
                      R(simple_network, cond, rtol=7.7e-5)), "rtol not in cache key"
    assert adds_entry(R(simple_network, cond, atol=3.3e-7),
                      R(simple_network, cond, atol=7.7e-7)), "atol not in cache key"
    assert adds_entry(R(simple_network, cond, dtmax=0.37),
                      R(simple_network, cond, dtmax=0.19)), "dtmax not in cache key"
    assert adds_entry(R(simple_network, cond, max_steps=51_234),
                      R(simple_network, cond, max_steps=61_234)), "max_steps not in cache key"
    # A custom adjoint must not reuse the default (None) reactor's compiled solve.
    assert adds_entry(R(simple_network, cond),
                      R(simple_network, cond, adjoint=diffrax.DirectAdjoint())), \
        "adjoint not in cache key"


@pytest.mark.slow  # heavy: jax.grad through ASM1 solve
def test_solver_cache_bypassed_under_tracing():
    """solve() under jax.grad (atol is materialised for the key, which is
    impossible while tracing) must bypass the cache, not crash."""
    net = aquakin.load_network("asm1")
    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    r = aquakin.BatchReactor(net, cond)
    C0, p = net.default_concentrations(), net.default_parameters()

    def loss(params):
        return r.solve(C0, (0.0, 1.0), jnp.array([1.0]), params=params).C.sum()

    g = jax.grad(loss)(p)
    assert jnp.all(jnp.isfinite(g))


# ----- Per-instance plant solve cache -------------------------------------

def _one_cstr_plant(net):
    plant = Plant("cache-test")
    plant.add_unit(CSTRUnit(
        name="r1", network=net, volume=1000.0, input_port_names=["inlet"],
        conditions={n: net._condition_defaults[n] for n in net.conditions_required},
    ))
    C = net.default_concentrations()
    infl = InfluentSeries(t=jnp.array([0.0, 1.0e4]), Q=jnp.full((2,), 1000.0),
                          C=jnp.tile(C, (2, 1)), network=net)
    plant.add_influent("feed", infl, to="r1.inlet")
    return plant


def test_plant_reuses_compiled_solve():
    """Repeat solves of the same plant + signature reuse one compile and match."""
    net = aquakin.load_network("asm1")
    plant = _one_cstr_plant(net)
    p = plant.default_parameters()
    s1 = plant.solve((0.0, 1.0), t_eval=jnp.array([1.0]), params=p)
    assert len(plant._jit_cache) == 1
    s2 = plant.solve((0.0, 1.0), t_eval=jnp.array([1.0]), params=p * 1.0)
    assert len(plant._jit_cache) == 1  # reused -- no second compile
    assert np.allclose(np.asarray(s1.state), np.asarray(s2.state))


def test_plant_different_signature_compiles_separately():
    net = aquakin.load_network("asm1")
    plant = _one_cstr_plant(net)
    p = plant.default_parameters()
    plant.solve((0.0, 1.0), t_eval=jnp.array([1.0]), params=p)
    plant.solve((0.0, 2.0), t_eval=jnp.array([1.0, 2.0]), params=p)  # new sig
    assert len(plant._jit_cache) == 2


@pytest.mark.slow  # heavy: jax.grad through plant solve
def test_plant_solve_grad_bypasses_cache():
    """A traced plant solve (jax.grad) bypasses the cache without crashing."""
    net = aquakin.load_network("asm1")
    plant = _one_cstr_plant(net)
    p = plant.default_parameters()

    def loss(params):
        return plant.solve((0.0, 1.0), t_eval=jnp.array([1.0]),
                           params=params).state.sum()

    g = jax.grad(loss)(p)
    assert jnp.all(jnp.isfinite(g))
