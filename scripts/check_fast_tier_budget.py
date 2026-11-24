"""Check that no fast-tier test is recorded over its time budget.

The fast tier is the merge gate, sharded by the durations CI records in
``.test_durations``. A test that runs for close to a minute there unbalances its
shard, and several did enough to trip the memory watchdog (``docs/ci.md``).
``.claude/rules/tests.md`` asks for a reason for a fast-tier test over about
10 s; this script turns the clear cases into a ratchet, as
``tests/unit/test_size_limits.py`` does for rule files:

- a fast-tier test recorded over ``BUDGET`` seconds fails, unless it is listed
  in ``RECORDED``;
- a listed test may not grow past ``GROWTH`` times its recorded time;
- a listed test that has dropped under ``RELEASE`` seconds, or that is no
  longer a recorded fast-tier test, must be removed from the list.

**It must run against freshly recorded durations.** A recorded time belongs
to the code that produced it: a test moved from ``slow`` to the fast tier keeps
its slow-tier time until the next refresh, so on an ordinary PR the check would
fail on work that had just made a test faster. CI therefore runs it in the
weekly ``commit-durations`` job, right after the shards' fresh times are merged
and on the same commit they measured, not in the fast gate. The margins (listing
tests from 25 s, releasing them only under 20 s, allowing 1.5x growth) keep
week-to-week runner noise from flipping the result.

The fast tier is what pytest collects from ``tests/unit`` and
``tests/integration`` under the merge gate's marker expression, so a test
marked ``slow`` (directly, by module, or through a slow plant fixture in
``tests/conftest.py``) is never counted. ``tests/ui`` runs as its own unsharded
job and is not recorded in ``.test_durations``.

Usage::

    python scripts/check_fast_tier_budget.py [--durations .test_durations]

Exits 1 and prints one line per problem when the budget is broken.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    "tests/integration/test_precipitation.py::test_shared_ionic_strength_solves_and_differentiates": 29.2,
    "tests/integration/test_a2o.py::test_asm2d_grad_flows": 29.0,
    "tests/integration/test_experiment_design.py::test_rank_experiments_orders_by_information": 26.4,
    "tests/integration/test_bsm2_evaluation.py::test_time_unit_override_leaves_bsm2_metrics_unchanged": 25.8,
    "tests/integration/test_calibrate.py::test_compiled_cache_rebuilds_on_every_baked_in_change": 25.8,
}

ADVICE = (
    "Mark it `slow` (keeping a small fast-tier test of the same code path, "
    ".claude/rules/tests.md), shrink it (the single-CSTR plant, fewer saves), "
    "or share its compiled solve through a module or session fixture."
)


def fast_tier_ids(root: Path = ROOT) -> set[str]:
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


def budget_problems(
    durations: dict[str, float], fast_ids: set[str], recorded: dict[str, float]
) -> list[str]:
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
                    "delete its RECORDED entry in scripts/check_fast_tier_budget.py."
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


def main(argv: list[str] | None = None) -> int:
    """Check ``.test_durations`` against the budget; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--durations", type=Path, default=ROOT / ".test_durations")
    args = parser.parse_args(argv)
    durations = json.loads(args.durations.read_text())
    problems = budget_problems(durations, fast_tier_ids(), RECORDED)
    for line in problems:
        print(line)
    if not problems:
        print(f"Every fast-tier test is within the {BUDGET:.0f} s budget or its recorded ceiling.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
