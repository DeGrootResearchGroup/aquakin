"""Patch coverage of the current branch, measured locally before a push.

Codecov's patch status asks that every line a pull request adds or changes be
covered. Finding out from CI costs a full run per attempt; this reads a local
``coverage.xml`` and the branch's diff and gives the same answer in seconds.

Usage::

    pytest tests/unit tests/integration tests/ui \\
        -m "not validation and not slow" \\
        --cov=aquakin --cov=aquakin_ui --cov-branch --cov-report=xml
    python scripts/patch_coverage.py

Run the tests that exercise the change (the whole fast tier is the honest
measure, since that is what uploads on an ordinary pull request); a narrower
selection reports lines as missed that other tests cover. Pass the package
names to ``--cov``, never a dotted module path.

A changed line counts as **covered** when a test ran it and took every branch
out of it, **partial** when it ran but a branch was not taken, and **missed**
when no test ran it. Lines the coverage report does not list (comments,
docstrings, blank lines) are not counted. The script exits 1 when the covered
share is below ``--fail-under`` (100 by default), printing each line to look
at.

The diff is taken from the merge base with ``--base`` (``origin/main`` by
default) to the working tree, so uncommitted edits are included.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

#: The packages coverage is measured over (``[tool.coverage.run] source``).
PACKAGES = ("aquakin/", "aquakin_ui/")
#: Files left out of the measurement (``[tool.coverage.run] omit``).
OMIT = ("aquakin/models/_make_*.py",)

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass
class FileReport:
    """The changed lines of one file, sorted by what the tests did with them."""

    covered: list[int] = field(default_factory=list)
    partial: list[int] = field(default_factory=list)
    missed: list[int] = field(default_factory=list)


def measured(path: str) -> bool:
    """Whether ``path`` is a Python file coverage is measured over."""
    return (
        path.endswith(".py")
        and path.startswith(PACKAGES)
        and not any(fnmatch.fnmatch(path, pattern) for pattern in OMIT)
    )


def changed_lines(diff: str) -> dict[str, set[int]]:
    """The added or changed line numbers per file in a ``git diff -U0``.

    Parameters
    ----------
    diff : str
        Unified diff text produced with no context lines.

    Returns
    -------
    dict of str to set of int
        New-side line numbers, keyed by the file's path after the change.
        Deleted files and pure deletions contribute nothing.
    """
    lines: dict[str, set[int]] = {}
    path: str | None = None
    for row in diff.splitlines():
        if row.startswith("+++ "):
            target = row[4:].strip()
            path = None if target == "/dev/null" else target.removeprefix("b/")
            continue
        match = _HUNK.match(row)
        if match and path is not None:
            start = int(match.group(1))
            count = 1 if match.group(2) is None else int(match.group(2))
            if count:
                lines.setdefault(path, set()).update(range(start, start + count))
    return lines


def line_states(coverage_xml: str, root: Path | None = None) -> dict[str, dict[int, str]]:
    """Each measured line's state from a Cobertura ``coverage.xml``.

    The report names a file relative to the ``<source>`` folder it was measured
    from (``plant/units.py`` under ``aquakin``), so each name is tried under
    every source and kept where that file exists.

    Parameters
    ----------
    coverage_xml : str
        The report's text.
    root : pathlib.Path, optional
        The repository root the sources are relative to. Default: the current
        folder.

    Returns
    -------
    dict of str to dict of int to str
        ``"covered"``, ``"partial"`` or ``"missed"`` per line number, keyed by
        the file's repository-relative path.
    """
    root = Path.cwd() if root is None else root
    tree = ET.fromstring(coverage_xml)
    sources = [(src.text or "").strip().replace("\\", "/") for src in tree.iter("source")]
    states: dict[str, dict[int, str]] = {}
    for cls in tree.iter("class"):
        name = cls.attrib["filename"].replace("\\", "/")
        candidates = [f"{src}/{name}" for src in sources if src] + [name]
        path = next((c for c in candidates if (root / c).is_file()), name)
        per_line = states.setdefault(path, {})
        for line in cls.iter("line"):
            if int(line.attrib["hits"]) == 0:
                state = "missed"
            elif line.attrib.get("branch") == "true" and not line.attrib.get(
                "condition-coverage", "100%"
            ).startswith("100%"):
                state = "partial"
            else:
                state = "covered"
            per_line[int(line.attrib["number"])] = state
    return states


def patch_report(
    changed: dict[str, set[int]], states: dict[str, dict[int, str]]
) -> tuple[dict[str, FileReport], list[str]]:
    """Sort every changed, measured line by its coverage state.

    Parameters
    ----------
    changed : dict of str to set of int
        The changed lines per file (:func:`changed_lines`).
    states : dict of str to dict of int to str
        The measured lines' states (:func:`line_states`).

    Returns
    -------
    reports : dict of str to FileReport
        One entry per changed file that has measured lines in the diff.
    unmeasured : list of str
        Changed files under the measured packages that the coverage report
        does not mention at all (the report is stale, or the tests that
        produced it never imported the package).
    """
    reports: dict[str, FileReport] = {}
    unmeasured: list[str] = []
    for path in sorted(changed):
        if not measured(path):
            continue
        if path not in states:
            unmeasured.append(path)
            continue
        report = FileReport()
        for number in sorted(changed[path]):
            state = states[path].get(number)
            if state is not None:
                getattr(report, state).append(number)
        if report.covered or report.partial or report.missed:
            reports[path] = report
    return reports, unmeasured


def percent(reports: dict[str, FileReport]) -> float:
    """The covered share of the changed lines, in percent (100 when none)."""
    covered = sum(len(r.covered) for r in reports.values())
    total = covered + sum(len(r.partial) + len(r.missed) for r in reports.values())
    return 100.0 if total == 0 else 100.0 * covered / total


def _ranges(numbers: list[int]) -> str:
    """Line numbers as compact ranges: ``[3, 4, 5, 9]`` gives ``3-5, 9``."""
    spans: list[list[int]] = []
    for n in numbers:
        if spans and n == spans[-1][1] + 1:
            spans[-1][1] = n
        else:
            spans.append([n, n])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def main(argv: list[str] | None = None) -> int:
    """Print the branch's patch coverage; return 1 when it is below the target."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main", help="branch to compare against")
    parser.add_argument("--coverage", default="coverage.xml", help="Cobertura report to read")
    parser.add_argument("--fail-under", type=float, default=100.0, help="required percentage")
    args = parser.parse_args(argv)

    report_path = Path(args.coverage)
    if not report_path.is_file():
        print(f"{report_path} not found: run pytest with --cov-report=xml first.")
        return 2
    base = subprocess.run(
        ["git", "merge-base", args.base, "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "-U0", "--no-color", base], capture_output=True, text=True, check=True
    ).stdout
    reports, unmeasured = patch_report(changed_lines(diff), line_states(report_path.read_text()))

    for path, report in reports.items():
        if report.missed:
            print(f"{path}: missed {_ranges(report.missed)}")
        if report.partial:
            print(f"{path}: partial branch {_ranges(report.partial)}")
    for path in unmeasured:
        print(f"{path}: not in {report_path} (stale report, or the package was not measured)")

    share = percent(reports)
    counted = sum(len(r.covered) + len(r.partial) + len(r.missed) for r in reports.values())
    print(f"patch coverage {share:.2f}% of {counted} changed lines (target {args.fail_under:g}%)")
    return 1 if (share < args.fail_under or unmeasured) else 0


if __name__ == "__main__":
    sys.exit(main())
