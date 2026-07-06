"""Unit tests for the SolveCache collaborator (issue #507)."""

from aquakin.plant.solve_cache import SolveCache


def test_solve_cache_starts_empty():
    sc = SolveCache()
    assert sc.jit == {}
    assert sc.steady_jit == {}
    assert sc.continuation_kernels == {}
    assert sc.arclength_kernels == {}


def test_invalidate_clears_every_cache():
    """One invalidate() drops every compiled artefact -- the single point that
    stops a mutator from clearing one cache and forgetting another."""
    sc = SolveCache()
    for cache in (sc.jit, sc.steady_jit, sc.continuation_kernels, sc.arclength_kernels):
        cache["k"] = object()

    sc.invalidate()

    assert sc.jit == {}
    assert sc.steady_jit == {}
    assert sc.continuation_kernels == {}
    assert sc.arclength_kernels == {}


def test_invalidate_keeps_the_same_dict_objects():
    """invalidate() clears in place, so a holder of a reference (a live solve
    reading the dict) sees the cleared cache rather than a detached stale one."""
    sc = SolveCache()
    jit = sc.jit
    jit["k"] = object()
    sc.invalidate()
    assert sc.jit is jit  # same object, now empty
    assert jit == {}
