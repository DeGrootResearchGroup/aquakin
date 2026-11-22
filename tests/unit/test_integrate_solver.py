"""Direct tests of the shared solve helpers in ``integrate/_solver.py`` and
``integrate/_cache.py`` whose branches no reactor solve in the fast tier takes.
"""

import diffrax
import jax.numpy as jnp
import pytest

from aquakin.integrate._cache import _SOLVER_CACHE, cached_jitted_solver
from aquakin.integrate._solver import build_implicit_solver, to_native_time, validate_t_eval


def test_validate_t_eval_accepts_an_empty_grid_without_raising():
    """An empty ``t_eval`` has no range or order to check, so it passes even
    against a span it could not otherwise be compared with; a non-empty grid
    outside the same span is still refused."""
    assert validate_t_eval(jnp.asarray([]), 0.0, 1.0) is None
    with pytest.raises(ValueError, match="t_eval must lie within t_span"):
        validate_t_eval(jnp.asarray([2.0]), 0.0, 1.0)


def test_to_native_time_scales_t_eval_when_there_is_no_t_span():
    """With ``t_span=None`` only ``t_eval`` is converted: minutes to a native
    unit of seconds multiplies by 60."""
    t_span, t_eval, factor = to_native_time("s", "min", None, [0.5, 2.0])
    assert t_span is None
    assert factor == 60.0
    assert jnp.allclose(t_eval, jnp.asarray([30.0, 120.0]), rtol=1e-12)


def test_to_native_time_scales_t_span_when_there_is_no_t_eval():
    """With ``t_eval=None`` only the span is converted: hours to seconds."""
    t_span, t_eval, factor = to_native_time("s", "h", (0.0, 2.0), None)
    assert t_eval is None
    assert factor == 3600.0
    assert t_span == (0.0, 7200.0)


def test_colored_root_finder_is_used_verbatim_in_the_canonical_solver():
    """A supplied ``colored_root_finder`` replaces the decoupled ``VeryChord``
    that would otherwise be built from ``rtol`` / ``atol``: the canonical solver
    carries that exact object, with its own tolerances."""
    rf = diffrax.VeryChord(rtol=1e-3, atol=1e-3)
    solver = build_implicit_solver(1e-6, 1e-9, order=3, colored_root_finder=rf)
    assert isinstance(solver, diffrax.Kvaerno3)
    assert solver.root_finder is rf
    default = build_implicit_solver(1e-6, 1e-9, order=3)
    assert default.root_finder is not rf
    assert default.root_finder.rtol == pytest.approx(10.0 * 1e-6)


def test_colored_root_finder_is_injected_into_a_supplied_solver():
    """A user-supplied solver is honoured verbatim on the forward path, except
    that a colored root finder is still injected into it."""
    rf = diffrax.VeryChord(rtol=1e-3, atol=1e-3)
    supplied = diffrax.Kvaerno5()
    assert build_implicit_solver(1e-6, 1e-9, solver=supplied) is supplied
    injected = build_implicit_solver(1e-6, 1e-9, solver=supplied, colored_root_finder=rf)
    assert isinstance(injected, diffrax.Kvaerno5)
    # ``eqx.tree_at`` rebuilds the module, so compare the root finder's own
    # tolerances (1e-3), which differ from the decoupled default (10 * rtol).
    assert isinstance(injected.root_finder, diffrax.VeryChord)
    assert injected.root_finder.rtol == pytest.approx(1e-3)
    assert injected.root_finder.atol == pytest.approx(1e-3)


def test_a_none_cache_key_builds_every_time_and_caches_nothing():
    """A ``None`` key (an un-cacheable, traced call) bypasses the shared cache:
    the factory runs on every call and no entry is stored."""
    calls = []

    def build():
        calls.append(1)
        return lambda: len(calls)

    n_before = len(_SOLVER_CACHE)
    first = cached_jitted_solver(None, build)
    second = cached_jitted_solver(None, build)
    assert len(calls) == 2
    assert first is not second
    assert len(_SOLVER_CACHE) == n_before
