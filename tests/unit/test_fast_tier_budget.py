"""A fast-tier test stays within its time budget, checked on recorded durations.

The fast tier is the merge gate, sharded by the durations CI records in
``.test_durations``. A test that runs for close to a minute there unbalances its
shard, and several did enough to trip the memory watchdog (``docs/ci.md``).
``.claude/rules/tests.md`` asks for a reason for a fast-tier test over about
10 s; this module turns the clear cases into a ratchet, as
``test_size_limits.py`` does for rule files:

- a fast-tier test recorded over ``BUDGET`` seconds fails, unless it is listed
  in ``RECORDED``;
- a listed test may not grow past ``GROWTH`` times its recorded time;
- a listed test that has dropped under ``RELEASE`` seconds, or that is no
  longer a recorded fast-tier test, must be removed from the list.

The times come from ``.test_durations``, which CI writes (a weekly refresh
opens a PR with it), so this test is deterministic for a given checkout and
never depends on the machine running it. The cost is lag: a new slow test is
caught when its duration is first recorded, on the refresh PR, not on the PR
that added it. The margins (listing tests from 25 s, releasing them only under
20 s, allowing 1.5x growth) keep week-to-week runner noise in the recorded
times from flipping the result.

The fast tier is what pytest collects from ``tests/unit`` and
``tests/integration`` under the merge gate's marker expression, so a test
marked ``slow`` (directly, by module, or through a slow plant fixture in
``tests/conftest.py``) is never counted. ``tests/ui`` runs as its own unsharded
job and is not recorded in ``.test_durations``.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
BUDGET = 30.0
RELEASE = 20.0
GROWTH = 1.5
FAST_TIER = ("tests/unit", "tests/integration")
FAST_MARKERS = "not validation and not slow"

# Fast-tier tests recorded at or near the budget: node id -> recorded seconds.
# Move a test to ``slow``, shrink it, or share its compiled work through a
# fixture, then delete its entry. Never add an entry to make a failure pass
# without saying why in the PR.
RECORDED = {
    "tests/integration/test_plant_stable_adjoint.py::test_stable_adjoint_colored_jacobian_auto_decision_size_gated": 151.2,
    "tests/integration/test_identifiability.py::test_from_calibration_result": 31.8,
    "tests/integration/test_precipitation.py::test_shared_ionic_strength_solves_and_differentiates": 29.2,
    "tests/integration/test_a2o.py::test_asm2d_grad_flows": 29.0,
    "tests/integration/test_forward_fast.py::test_forward_fast_accepts_honored_knobs": 28.2,
    "tests/integration/test_forward_fast.py::test_forward_fast_rejects_events_and_grad": 28.0,
    "tests/integration/test_experiment_design.py::test_rank_experiments_orders_by_information": 26.4,
    "tests/integration/test_asm3_2step_n2o.py::test_grad_through_solve_is_finite": 26.1,
    "tests/integration/test_bsm2_evaluation.py::test_time_unit_override_leaves_bsm2_metrics_unchanged": 25.8,
    "tests/integration/test_calibrate.py::test_compiled_cache_rebuilds_on_every_baked_in_change": 25.8,
    "tests/integration/test_calibrate.py::test_start_list_matches_best_of_separate_calls[gauss_newton]": 25.5,
    "tests/integration/test_calibrate.py::test_declared_bound_stops_the_fit[gauss_newton]": 25.1,
}

ADVICE = (
    "Mark it `slow` (keeping a small fast-tier test of the same code path, "
    ".claude/rules/tests.md), shrink it (the single-CSTR plant, fewer saves), "
    "or share its compiled solve through a module or session fixture."
)


def fast_tier_ids(root=ROOT):
    """The node ids pytest collects for the sharded fast tier."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", *FAST_TIER, "-m", FAST_MARKERS]
        + ["--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {line.strip() for line in out.splitlines() if "::" in line}


def budget_problems(durations, fast_ids, recorded):
    """
    Messages for the fast-tier tests outside the budget, and stale entries.

    Parameters
    ----------
    durations : dict of str to float
        Recorded seconds per node id (``.test_durations``).
    fast_ids : set of str
        The node ids in the fast tier.
    recorded : dict of str to float
        The listed tests and their recorded seconds.

    Returns
    -------
    list of str
    """
    fast = {k: v for k, v in durations.items() if k in fast_ids}
    problems = []
    for node, seconds in sorted(fast.items()):
        if node in recorded:
            ceiling = recorded[node] * GROWTH
            if seconds > ceiling:
                problems.append(
                    f"{node} is recorded at {seconds:.1f} s, past its ceiling of "
                    f"{ceiling:.1f} s ({GROWTH}x its listed {recorded[node]:.1f} s). {ADVICE}"
                )
            elif seconds < RELEASE:
                problems.append(
                    f"{node} is now recorded at {seconds:.1f} s, under {RELEASE:.0f} s: "
                    "delete its RECORDED entry in test_fast_tier_budget.py."
                )
        elif seconds > BUDGET:
            problems.append(
                f"{node} is a fast-tier test recorded at {seconds:.1f} s, over the "
                f"{BUDGET:.0f} s budget. {ADVICE}"
            )
    problems += [
        f"{node} is listed in RECORDED but is not a recorded fast-tier test: "
        "delete its entry."
        for node in sorted(set(recorded) - set(fast))
    ]
    return problems


# --- the check on this repository -------------------------------------------


@pytest.fixture(scope="module")
def collected_fast_ids():
    """The fast tier's node ids, collected once for this module."""
    return fast_tier_ids()


def test_fast_tier_tests_stay_within_the_budget(collected_fast_ids):
    durations = json.loads((ROOT / ".test_durations").read_text())
    problems = budget_problems(durations, collected_fast_ids, RECORDED)
    assert not problems, "\n".join(problems)


# --- the check itself ---------------------------------------------------------


def test_budget_problems_ratchet():
    durations = {
        "a::new_slow": BUDGET + 1,
        "a::new_ok": BUDGET - 1,
        "a::listed_ok": 40.0,
        "a::listed_grew": 61.0,
        "a::listed_fast_now": RELEASE - 1,
        "a::marked_slow": 500.0,
    }
    fast_ids = set(durations) - {"a::marked_slow"}
    recorded = {
        "a::listed_ok": 35.0,
        "a::listed_grew": 40.0,
        "a::listed_fast_now": 26.0,
        "a::gone": 30.0,
    }
    problems = budget_problems(durations, fast_ids, recorded)
    assert [p.split(" ")[0] for p in problems] == [
        "a::listed_fast_now",
        "a::listed_grew",
        "a::new_slow",
        "a::gone",
    ]
    assert "delete its RECORDED entry" in problems[0]
    assert "past its ceiling of 60.0 s" in problems[1]
    assert "over the 30 s budget" in problems[2]
    assert "not a recorded fast-tier test" in problems[3]


def test_fast_tier_ids_follow_the_markers(collected_fast_ids):
    plain = "tests/integration/test_batch_simple.py::test_first_order_decay_matches_analytical"
    marked_slow = "tests/integration/test_calibrate.py::test_gauss_newton_laplace_matches_fd"
    assert plain in collected_fast_ids
    assert marked_slow not in collected_fast_ids
