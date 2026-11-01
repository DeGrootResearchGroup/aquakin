"""The CI `heavy` job must give every heavy test its own shard.

The job runs ``pytest -m "heavy and not xheavy" --splits N --group g`` over a
matrix of N groups, one test per fresh process. The heavy tests have no recorded
durations, so when they outnumber the shards pytest-split's duration-based
chunking pairs them and leaves the trailing shards empty -- and an empty shard
exits 5 ("no tests ran"), failing the run on main. Adding a heavy test therefore
means adding a shard; this pins that in the fast gate.
"""

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _heavy_job():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    return workflow["jobs"]["heavy"]


def _heavy_test_count():
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", "heavy and not xheavy", "tests"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return sum(1 for line in out.splitlines() if "::" in line)


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
