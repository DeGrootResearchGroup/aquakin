"""Compiled-solver caching: load_model, cross-instance reactor cache, plant cache.

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
from tests.conftest import FIXTURES

# ----- load_model caching -----------------------------------------------

def test_load_model_returns_cached_object():
    a = aquakin.load_model("asm1")
    b = aquakin.load_model("asm1")
    assert a is b  # same object -> stable id for the solver cache


def test_clear_model_cache_forces_reload():
    from aquakin.schema.loader import clear_model_cache

    a = aquakin.load_model("asm1")
    clear_model_cache()
    b = aquakin.load_model("asm1")
    assert a is not b
    # Re-cache so other tests keep sharing one object.
    assert aquakin.load_model("asm1") is b


# ----- Cross-instance reactor solver cache --------------------------------

def _batch(net):
    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    return aquakin.BatchReactor(net, cond)


def test_fresh_reactors_share_one_compiled_solver(simple_model):
    """Two fresh reactors for the same model + settings + signature reuse a
    single compiled solve (no second compile entry), with identical results."""
    net = simple_model
    C0, p = net.default_concentrations(), net.default_parameters()
    t_eval = jnp.array([1.0])

    s1 = _batch(net).solve(C0, (0.0, 1.0), t_eval, params=p)
    n_after_first = len(_common._SOLVER_CACHE)
    s2 = _batch(net).solve(C0, (0.0, 1.0), t_eval, params=p)  # fresh reactor, same key
    assert len(_common._SOLVER_CACHE) == n_after_first  # reused, no new compile
    assert np.allclose(np.asarray(s1.C), np.asarray(s2.C))


def test_different_settings_do_not_collide(simple_model):
    """A different tolerance is a different compile (no false cache hit): the
    loose and tight solves differ, and the tight one has the tight solve's
    accuracy against the closed form ``A(1) = exp(-k)``."""
    net = simple_model
    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    C0, p = net.default_concentrations(), net.default_parameters()
    s_loose = aquakin.BatchReactor(net, cond, rtol=1e-4).solve(C0, (0.0, 1.0), params=p)
    s_tight = aquakin.BatchReactor(net, cond, rtol=1e-9).solve(C0, (0.0, 1.0), params=p)
    A_loose = np.asarray(s_loose.C_named("A"))[-1]
    A_tight = np.asarray(s_tight.C_named("A"))[-1]
    exact = np.exp(-float(p[0]))
    # A false hit would hand the tight call the loose call's compiled solve.
    assert A_loose != A_tight
    # An adaptive solve at rtol = 1e-9: two decades above it.
    assert A_tight == pytest.approx(exact, rel=1e-7)


def test_each_compile_affecting_setting_keys_the_cache():
    """Every setting that changes the compiled solve -- rtol, atol, adjoint,
    dtmax, max_steps -- must produce a DISTINCT cache entry. A future edit that
    dropped one from ``settings_cache_key`` would otherwise be a silent
    wrong-result cache hit; here it surfaces as a failure. (The toy A->B model
    keeps the compiles cheap; the cache mechanism is model-agnostic.)"""
    # A model of its own, not the shared ``simple_model`` fixture: the cache is
    # keyed on model identity, so the default-settings keys checked last below
    # could already exist from another test's solve of the shared model.
    simple_model = aquakin.load_model_from_file(FIXTURES / "simple_model.yaml")

    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    C0 = jnp.asarray([1.0, 0.0])
    p = simple_model.default_parameters()
    R = aquakin.BatchReactor

    def adds_entry(reactor_a, reactor_b):
        # Warm A's key, then assert B introduces a NEW compiled entry.
        reactor_a.solve(C0, (0.0, 1.0), params=p)
        n = len(_common._SOLVER_CACHE)
        reactor_b.solve(C0, (0.0, 1.0), params=p)
        return len(_common._SOLVER_CACHE) > n

    # Distinctive values, so the keys cannot pre-exist from another test.
    assert adds_entry(R(simple_model, cond, rtol=3.3e-5),
                      R(simple_model, cond, rtol=7.7e-5)), "rtol not in cache key"
    assert adds_entry(R(simple_model, cond, atol=3.3e-7),
                      R(simple_model, cond, atol=7.7e-7)), "atol not in cache key"
    assert adds_entry(
        R(simple_model, cond, integrator=aquakin.IntegratorConfig(dtmax=0.37)),
        R(simple_model, cond, integrator=aquakin.IntegratorConfig(dtmax=0.19)),
    ), "dtmax not in cache key"
    assert adds_entry(
        R(simple_model, cond,
          integrator=aquakin.IntegratorConfig(max_steps=51_234)),
        R(simple_model, cond,
          integrator=aquakin.IntegratorConfig(max_steps=61_234)),
    ), "max_steps not in cache key"
    # A custom differentiation config must not reuse the default reactor's
    # compiled solve.
    assert adds_entry(
        R(simple_model, cond),
        R(simple_model, cond,
          diff=aquakin.DifferentiationConfig(mode="forward",
                                             method="through_solve")),
    ), "adjoint not in cache key"


def test_solver_cache_bypassed_under_tracing(simple_model):
    """A reactor whose ``atol`` is traced (built inside a jitted loss) cannot
    form a cache key, so its solve bypasses the cache: the traced gradient adds
    no cache entry, and it is the closed form ``dA(t)/dk = -t A0 exp(-k t)``."""
    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    C0 = jnp.asarray([1.0, 0.0])
    p = simple_model.default_parameters()

    def loss(params, atol):
        reactor = aquakin.BatchReactor(simple_model, cond, rtol=1e-10, atol=atol)
        return reactor.solve(C0, (0.0, 1.0), jnp.array([1.0]), params=params).C[-1, 0]

    n_before = len(_common._SOLVER_CACHE)
    g = jax.jit(jax.grad(loss))(p, jnp.asarray(1e-12))
    assert len(_common._SOLVER_CACHE) == n_before
    k = float(p[0])
    # An adaptive solve at rtol = 1e-10, differentiated: within 1e-7.
    assert float(g[0]) == pytest.approx(-1.0 * np.exp(-k * 1.0), rel=1e-7)


# ----- Per-instance plant solve cache -------------------------------------

def test_plant_reuses_compiled_solve(make_cstr_plant):
    """Repeat solves of the same plant + signature reuse one compile and match."""
    plant = make_cstr_plant()
    p = plant.default_parameters()
    s1 = plant.solve((0.0, 1.0), t_eval=jnp.array([1.0]), params=p)
    assert len(plant._jit_cache) == 1
    s2 = plant.solve((0.0, 1.0), t_eval=jnp.array([1.0]), params=p * 1.0)
    assert len(plant._jit_cache) == 1  # reused -- no second compile
    assert np.allclose(np.asarray(s1.state), np.asarray(s2.state))


def test_plant_solve_with_traced_t_eval_bypasses_the_cache(make_cstr_plant):
    """A ``t_eval`` traced by an enclosing ``jax.jit`` cannot be turned into a
    cache key, so the solve runs uncached -- and gives the eager result."""
    plant = make_cstr_plant()
    te = jnp.linspace(0.0, 5.0, 4)
    eager = plant.solve((0.0, 5.0), t_eval=te, rtol=1e-8, atol=1e-10).state
    assert len(plant._jit_cache) == 1
    jitted = jax.jit(lambda t: plant.solve((0.0, 5.0), t_eval=t, rtol=1e-8, atol=1e-10).state)
    np.testing.assert_allclose(np.asarray(jitted(te)), np.asarray(eager), rtol=1e-12, atol=0)
    assert len(plant._jit_cache) == 1  # nothing added for the traced call


def test_plant_different_signature_compiles_separately(make_cstr_plant):
    plant = make_cstr_plant()
    p = plant.default_parameters()
    plant.solve((0.0, 1.0), t_eval=jnp.array([1.0]), params=p)
    plant.solve((0.0, 2.0), t_eval=jnp.array([1.0, 2.0]), params=p)  # new sig
    assert len(plant._jit_cache) == 2


def test_plant_solve_grad_bypasses_cache(make_cstr_plant):
    """A traced plant solve (jax.grad) bypasses the cache: it adds no entry to
    the one the eager solve compiled, and the gradient is the closed form.

    The tank's ``A`` obeys ``dA/dt = 0.1 (1 - A) - k A``, so with
    ``s = 0.1 + k`` and ``a = 0.1 / s``, ``A(t) = a + (A0 - a) exp(-s t)`` and
    ``dA/dk = -(0.1 / s**2) (1 - exp(-s t)) - (A0 - a) t exp(-s t)``.
    """
    plant = make_cstr_plant()
    p = plant.default_parameters()
    A0, t = 0.2, 5.0

    def A_end(params):
        return plant.solve((0.0, t), t_eval=jnp.array([t]), params=params,
                           y0=jnp.array([A0, 0.0]), rtol=1e-10, atol=1e-12).state[-1, 0]

    A_end(p)
    assert len(plant._jit_cache) == 1
    g = jax.grad(A_end)(p)
    assert len(plant._jit_cache) == 1  # nothing added for the traced call
    s = 0.1 + float(p[0])
    a = 0.1 / s
    exact = -(0.1 / s**2) * (1.0 - np.exp(-s * t)) - (A0 - a) * t * np.exp(-s * t)
    # An adaptive solve at rtol = 1e-10, differentiated: within 1e-7.
    assert float(g[0]) == pytest.approx(exact, rel=1e-7)
