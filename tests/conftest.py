"""Shared pytest fixtures."""

from pathlib import Path

import pytest

import aquakin

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_network():
    """Load the simple A -> B test network."""
    return aquakin.load_network_from_file(FIXTURES / "simple_network.yaml")


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
