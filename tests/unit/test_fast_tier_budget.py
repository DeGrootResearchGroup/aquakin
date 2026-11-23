"""The fast-tier time-budget check in ``scripts/check_fast_tier_budget.py``.

The check itself runs in CI's weekly ``commit-durations`` job, against freshly
recorded durations (the script's docstring says why it cannot run in the fast
gate). These tests pin its logic on synthetic data and its fast-tier collection
on the real suite.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_fast_tier_budget.py"


@pytest.fixture(scope="module")
def budget():
    """The script, imported as a module."""
    spec = importlib.util.spec_from_file_location("check_fast_tier_budget", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def collected_fast_ids(budget):
    """The fast tier's node ids, collected once for this module."""
    return budget.fast_tier_ids()


def test_budget_problems_ratchet(budget):
    durations = {
        "a::new_slow": budget.BUDGET + 1,
        "a::new_ok": budget.BUDGET - 1,
        "a::listed_ok": 40.0,
        "a::listed_grew": 61.0,
        "a::listed_fast_now": budget.RELEASE - 1,
        "a::marked_slow": 500.0,
    }
    fast_ids = set(durations) - {"a::marked_slow"}
    recorded = {
        "a::listed_ok": 35.0,
        "a::listed_grew": 40.0,
        "a::listed_fast_now": 26.0,
        "a::gone": 30.0,
    }
    problems = budget.budget_problems(durations, fast_ids, recorded)
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


def test_recorded_entries_are_fast_tier_tests(budget, collected_fast_ids):
    """A RECORDED entry names a test that still exists in the fast tier.

    Unlike the budget itself this does not depend on recorded times, so it is
    safe in the fast gate, and a renamed or removed test is caught on its PR.
    """
    stale = sorted(set(budget.RECORDED) - collected_fast_ids)
    assert not stale, f"delete these RECORDED entries, no such fast-tier test: {stale}"


def test_main_reports_and_exits_nonzero(budget, tmp_path, monkeypatch, capsys):
    node = "tests/x.py::test_y"
    monkeypatch.setattr(budget, "fast_tier_ids", lambda: {node})
    monkeypatch.setattr(budget, "RECORDED", {})
    path = tmp_path / "durations.json"

    path.write_text(json.dumps({node: budget.BUDGET + 5}))
    assert budget.main(["--durations", str(path)]) == 1
    assert "over the 30 s budget" in capsys.readouterr().out

    path.write_text(json.dumps({node: 1.0}))
    assert budget.main(["--durations", str(path)]) == 0
    assert "within the 30 s budget" in capsys.readouterr().out
