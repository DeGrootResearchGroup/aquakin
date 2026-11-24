"""The cross-instance compiled-solver cache and the settings keys it is keyed by.

Not part of the public API; ``aquakin.integrate._common`` re-exports every name.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

# --- Cross-instance compiled-solver cache ------------------------------------
#
# Each ``reactor.solve(...)`` jit-compiles an inner ``_solve`` closure that
# captures the model and the solver settings. Compilation (trace + lower +
# XLA) dominates the cost of a solve -- the run itself is comparatively free --
# so rebuilding that closure for every reactor instance means every fresh
# reactor pays a full from-scratch compile, even for an identical model and
# settings. That is the dominant cost of the test suite (build a reactor, solve
# once) and of any code that constructs many short-lived reactors.
#
# This module-level cache shares the compiled solver across *all* reactor
# instances keyed by everything the compiled computation depends on: the
# model identity, the solver settings, and the call signature. A repeat solve
# of the same (model, settings, signature) then reuses the compiled graph and
# runs in milliseconds. The cache holds a reference to the model (and any
# settings object keyed by identity, e.g. a custom adjoint) so their ``id()``
# cannot be reused by a later object while the entry is live -- keying on ``id``
# is therefore safe.
#
# Built-in models are themselves cached by name (``load_model``), so the
# same model object is reused across calls and the ``id()`` key is stable
# across the whole process. The key NEVER omits anything that changes the
# compiled result, so a hit always returns a solver compiled for the exact same
# computation (no false hits); argument shapes/dtypes (C0, params, conditions,
# t_eval) are handled by JAX's own per-function cache and need not be keyed here.
_SOLVER_CACHE: dict = {}


def atol_cache_key(atol):
    """A hashable, value-based key for an (array or scalar) absolute tolerance.

    ``atol`` is often a per-component array derived from the model, so a fresh
    array object each reactor; key on its *values* so identical tolerances share
    a cache entry. Any other array or scalar a compiled program closes over (a
    steady-state scale) is keyed the same way.
    """
    a = jnp.asarray(atol)
    return (tuple(a.shape), tuple(float(x) for x in np.asarray(a).reshape(-1)))


def _adjoint_key(adjoint):
    """A hashable key for a diffrax adjoint that is equal for equal programs.

    ``None`` (the default ``RecursiveCheckpointAdjoint``) keys as ``None``. An
    adjoint is an immutable dataclass whose type and fields fully determine the
    program it compiles, so one whose fields are all plain scalars (or
    ``None``) keys by its type and field values: a fresh ``DirectAdjoint`` per
    solve (what :meth:`DifferentiationConfig.reactor_adjoint` returns) then
    shares one compiled solve, while a different type or setting does not. Any
    other adjoint (a field holding a solver, a dict, an array) keys by ``id``,
    which never gives a false hit; the caller keeps it alive with the entry.
    """
    if adjoint is None:
        return None
    if dataclasses.is_dataclass(adjoint):
        values = tuple((f.name, getattr(adjoint, f.name)) for f in dataclasses.fields(adjoint))
        if all(v is None or type(v) in (bool, int, float, str) for _, v in values):
            # The value's type is keyed too, so ``True`` and ``1`` stay apart.
            return (type(adjoint), tuple((name, type(v), v) for name, v in values))
    return id(adjoint)


def _assemble_settings_key(rtol, atol_key, adjoint, dtmax, max_steps, order, factormax, solver):
    """Assemble the solver-settings cache-key tuple from an already-materialised
    ``atol_key`` (from :func:`atol_cache_key`).

    The **single** field-ordering of the settings key, shared by
    :func:`settings_cache_key` (the plant path, raw ``atol``) and
    :func:`reactor_settings_key` (the reactor path, memoised ``atol`` key). A new
    compile-affecting setting is added here **once**, so the two entry points
    cannot drift out of lockstep -- a drift would be a silent wrong-result cache
    hit. ``adjoint`` is keyed by :func:`_adjoint_key`, ``solver`` by class name; ``None`` for
    ``order`` / ``factormax`` / ``solver`` keeps the historic key shape.
    """
    return (
        float(rtol),
        atol_key,
        _adjoint_key(adjoint),
        None if dtmax is None else float(dtmax),
        int(max_steps),
        None if order is None else int(order),
        None if factormax is None else float(factormax),
        None if solver is None else type(solver).__name__,
    )


def settings_cache_key(
    rtol, atol, adjoint, dtmax, max_steps, *, order=None, factormax=None, solver=None
):
    """Hashable key for the solver settings that affect the compiled solve.

    ``adjoint`` is keyed by its type and settings (:func:`_adjoint_key`;
    ``None`` for the default), so equal adjoints share a compiled solve; an
    adjoint with non-scalar settings keys by ``id`` -- safe (never a false
    hit), and it shares whenever the same object is reused. ``order`` / ``factormax`` /
    ``solver`` (the integrator-config fields that change the compiled reactor
    solve) are keyed in when supplied -- ``solver`` by class name, matching the
    plant path. ``None`` for any of them keeps the historic key shape.
    """
    return _assemble_settings_key(
        rtol, atol_cache_key(atol), adjoint, dtmax, max_steps, order, factormax, solver
    )


def concrete_settings_key(
    rtol, atol, adjoint, dtmax, max_steps, *, order=None, factormax=None, solver=None
):
    """Return :func:`settings_cache_key`, or ``None`` if a value is traced.

    The key materialises ``atol`` to Python floats, which is impossible when
    ``solve`` runs *under tracing* (e.g. a calibration loss differentiating
    through the solve). In that case the shared cache gives no benefit anyway --
    the solve is being traced into an outer computation that JAX compiles as a
    whole -- so return ``None`` to signal "build without caching".
    """
    try:
        return settings_cache_key(
            rtol, atol, adjoint, dtmax, max_steps, order=order, factormax=factormax, solver=solver
        )
    except jax.errors.TracerArrayConversionError:
        return None


def reactor_settings_key(reactor):
    """Return the reactor's concrete solver-settings cache key.

    Equivalent to ``concrete_settings_key(reactor.rtol, reactor.atol, ...)`` but
    memoises the expensive part. The ``atol`` value-based key materialises the
    per-component tolerance array to Python floats -- a device->host sync that
    otherwise runs *before* the cache lookup on every solve, including hits, on
    exactly the many-short-solve loop the cache exists for. ``atol`` is fixed at
    construction, so its key is computed once and stored on the reactor.

    The cheap parts (``rtol`` / ``adjoint`` key / ``dtmax`` / ``max_steps``) are
    re-read every call rather than cached: :func:`with_adjoint` produces a
    *shallow copy* with a swapped ``adjoint`` (the forward-/reverse-mode variant
    ``calibrate`` / ``sensitivity`` build), and that copy inherits the cached
    ``_atol_key`` (correctly -- same ``atol`` array) while its adjoint differs,
    so the adjoint must not be baked into the cached portion.

    Returns ``None`` under tracing (``atol`` is a tracer); not cached in that
    case, so a later concrete solve still computes and stores a real key.
    """
    atol_key = reactor.__dict__.get("_atol_key")
    if atol_key is None:
        try:
            atol_key = atol_cache_key(reactor.atol)
        except jax.errors.TracerArrayConversionError:
            return None
        reactor._atol_key = atol_key
    return _assemble_settings_key(
        reactor.rtol,
        atol_key,
        reactor.adjoint,
        reactor.dtmax,
        reactor.max_steps,
        getattr(reactor, "order", None),
        getattr(reactor, "factormax", None),
        getattr(reactor, "solver", None),
    )


def cached_jitted_solver(key, build, *keep_alive):
    """Return the cached compiled solver for ``key``, building it once.

    ``build`` is a zero-arg factory returning the jitted ``_solve``; it is called
    only on a cache miss. ``keep_alive`` objects (the model, a custom adjoint)
    are retained by the cache so their ``id()`` stays valid for the lifetime of
    the entry. A ``key`` of ``None`` (an un-cacheable, traced call) bypasses the
    cache and just builds.
    """
    if key is None:
        return build()
    entry = _SOLVER_CACHE.get(key)
    if entry is None:
        fn = build()
        _SOLVER_CACHE[key] = (fn, *keep_alive)
        return fn
    return entry[0]
