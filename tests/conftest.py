"""Shared pytest fixtures."""

import gc
from pathlib import Path

import pytest

import aquakin

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_network():
    """Load the simple A -> B test network."""
    return aquakin.load_network_from_file(FIXTURES / "simple_network.yaml")


@pytest.fixture(autouse=True)
def _bound_slow_test_memory(request):
    """Clear the JAX compilation cache after each ``slow`` test to bound the
    slow-suite shard's memory footprint.

    The slow suite runs a shard's tests in ONE process, and each whole-plant test
    compiles a large stiff-solve program (~5 GB peak) whose XLA executable and live
    JAX buffers accumulate across the shard and OOM the 16 GB CI runner. Sharding
    bounds this only loosely: more shards or better duration balancing just
    *relocate* the overweight shard (the OOM walked 4/6 -> 5/8 across those
    attempts) because the accumulation is per-shard, not per-test. Plant solves are
    cached **per instance** (``Plant._jit_cache``), so a later test builds and
    compiles its own plant regardless; clearing the compilation cache between
    tests therefore frees the accumulated executables (and ``gc.collect`` frees the
    live buffers) without forcing any re-compile. This bounds each shard's peak to
    a single test's footprint. Gated on the ``slow`` marker so the fast suite --
    where lightweight tests share compiled fixtures and clearing would be a net
    cost -- is untouched.
    """
    yield
    if request.node.get_closest_marker("slow") is not None:
        import jax

        jax.clear_caches()
        gc.collect()


# --- Tiering: defer full-plant integration tests to the merge-only `slow` job --
#
# Compiling a stiff solve dominates the test cost; building + integrating the
# full BSM2 plant costs ~30 s to compile, which no caching removes (each is a
# distinct configuration). To keep the PR fast-gate cheap, every test that
# requests one of these "build + solve a full plant" *module fixtures* is marked
# ``slow`` here, so it runs only in the merge-to-main `slow` job. The PR gate
# keeps the cheap coverage of those features -- the unit-level logic tests
# (controller math, storage regimes, schedule, delay, metric kernels), the plant
# *assembly* / flow-resolution checks, and the small-plant smokes in
# test_plant_assembly -- which is where a developer's bug surfaces fastest.
#
# To defer a new plant-solving test, give it (or have it use) one of these
# fixtures, or add its fixture name here. A heavy test that does NOT use a
# fixture (e.g. a parametrized ``jax.grad`` check) carries an explicit
# ``@pytest.mark.slow`` in its own file instead.
_SLOW_PLANT_FIXTURES = frozenset({
    "steady",        # test_bsm2: open-loop steady-state solve
    "evaluated",     # test_bsm2_evaluation: EQI/OCI on a solved plant
    "storage_run",   # test_bsm2_storage
    "closed_sol",    # test_bsm2_control: DO closed-loop solve
    "open_sol",      # test_bsm2_control: open-loop contrast solve
    "control_run",   # test_bsm2_reject_control
    "wastage_run",   # test_bsm2_wastage
    "delay_run",     # test_bsm2_hydraulic_delay
    "bypass_run",    # test_bsm2_bypass
    "growth_setup",  # test_profile: calibrate + profile_likelihood on a growth model
})


def pytest_collection_modifyitems(config, items):
    """Mark any test using a full-plant-solving fixture ``slow`` (see above)."""
    slow = pytest.mark.slow
    for item in items:
        if _SLOW_PLANT_FIXTURES.intersection(getattr(item, "fixturenames", ())):
            item.add_marker(slow)
