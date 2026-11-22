"""The CI `heavy` job must give every heavy test its own shard.

The job runs ``pytest -m "heavy and not xheavy" --splits N --group g`` over a
matrix of N groups, one test per fresh process. The heavy tests have no recorded
durations, so when they outnumber the shards pytest-split's duration-based
chunking pairs them and leaves the trailing shards empty -- and an empty shard
exits 5 ("no tests ran"), failing the run on main. Adding a heavy test therefore
means adding a shard; this pins that in the fast gate.

The heavy tests are counted from the source rather than by a ``pytest
--collect-only`` subprocess, which imports the whole suite. A static count is
exact only for the form every heavy test takes today -- a plain test function
with ``@pytest.mark.heavy`` stacked on it -- so any other way of applying the
mark (a parametrized heavy test, a class or module ``pytestmark``, a mark added
from a hook or by name) fails here instead of being miscounted.
"""

import ast
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

#: Any spelling of the heavy / xheavy mark, decorator or not.
_MARK_TEXT = re.compile(r"""mark\s*\.\s*x?heavy\b|["']x?heavy["']""")


def _heavy_job():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    return workflow["jobs"]["heavy"]


def _mark_name(decorator):
    """``heavy`` for ``@pytest.mark.heavy`` (or ``@mark.heavy``), else None."""
    node = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        if node.value.attr == "mark":
            return node.attr
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "mark":
            return node.attr
    return None


def _heavy_test_count():
    """Count the ``heavy and not xheavy`` tests the suite would collect.

    Raises
    ------
    AssertionError
        If the heavy mark is applied in any form this count cannot size.
    """
    count = 0
    sources = [p for p in (ROOT / "tests").rglob("*.py") if p.resolve() != Path(__file__)]
    for path in sources + [ROOT / "conftest.py"]:
        text = path.read_text()
        mentions = sum(1 for _ in _MARK_TEXT.finditer(text))
        if not mentions:
            continue
        understood = 0
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            marks = [_mark_name(d) for d in node.decorator_list]
            heavy_marks = [m for m in marks if m in ("heavy", "xheavy")]
            if not heavy_marks:
                continue
            understood += len(heavy_marks)
            assert node.name.startswith("test_") and "parametrize" not in marks, (
                f"{path.relative_to(ROOT)}::{node.name} carries the heavy mark in a "
                "form the static count cannot size; extend _heavy_test_count"
            )
            if "heavy" in marks and "xheavy" not in marks:
                count += 1
        assert understood == mentions, (
            f"{path.relative_to(ROOT)} applies the heavy mark other than as a "
            "decorator on a test function; extend _heavy_test_count"
        )
    return count


def test_heavy_job_has_a_shard_for_every_heavy_test():
    job = _heavy_job()
    groups = job["strategy"]["matrix"]["group"]
    run = next(s["run"] for s in job["steps"] if "--splits" in s.get("run", ""))
    splits = int(re.search(r"--splits\s+(\d+)", run).group(1))

    assert groups == list(range(1, splits + 1)), "matrix groups must be 1..--splits"
    n_heavy = _heavy_test_count()
    assert n_heavy > 0
    assert splits >= n_heavy, (
        f"{n_heavy} heavy tests but only {splits} shards: raise --splits and the "
        f"matrix in .github/workflows/ci.yml (and docs/ci.md)"
    )
