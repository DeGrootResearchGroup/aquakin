"""The local patch-coverage helper (``scripts/patch_coverage.py``).

It is pure text processing, so it is tested on a hand-written diff and a
hand-written coverage report whose answer is known by inspection.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "patch_coverage.py"


@pytest.fixture(scope="module")
def pc():
    spec = importlib.util.spec_from_file_location("patch_coverage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered while it loads: its dataclass resolves annotations by module.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


DIFF = """\
diff --git a/aquakin/a.py b/aquakin/a.py
--- a/aquakin/a.py
+++ b/aquakin/a.py
@@ -10,0 +11,3 @@ def f():
+x
+y
+z
@@ -20 +23 @@ def g():
-old
+new
@@ -30,2 +32,0 @@ def h():
-gone
-gone
diff --git a/aquakin/removed.py b/aquakin/removed.py
--- a/aquakin/removed.py
+++ /dev/null
@@ -1,2 +0,0 @@
-a
-b
diff --git a/tests/unit/test_x.py b/tests/unit/test_x.py
--- a/tests/unit/test_x.py
+++ b/tests/unit/test_x.py
@@ -1 +1 @@
-a
+b
diff --git a/aquakin_ui/new.py b/aquakin_ui/new.py
--- /dev/null
+++ b/aquakin_ui/new.py
@@ -0,0 +1,2 @@
+a
+b
"""

REPORT = """\
<coverage><sources><source>aquakin</source><source>aquakin_ui</source></sources>
<packages><package><classes>
<class filename="a.py"><lines>
  <line number="11" hits="3"/>
  <line number="12" hits="0"/>
  <line number="23" hits="1" branch="true" condition-coverage="50% (1/2)"/>
  <line number="40" hits="0"/>
</lines></class>
<class filename="new.py"><lines>
  <line number="1" hits="1" branch="true" condition-coverage="100% (2/2)"/>
  <line number="2" hits="2"/>
</lines></class>
</classes></package></packages></coverage>
"""


@pytest.fixture
def tree(tmp_path):
    """A stand-in repository holding the two files the report names. Both
    packages have an ``a.py``-free layout except the one file each, so the
    report's source-relative names resolve to exactly one path."""
    (tmp_path / "aquakin").mkdir()
    (tmp_path / "aquakin_ui").mkdir()
    (tmp_path / "aquakin" / "a.py").write_text("")
    (tmp_path / "aquakin_ui" / "new.py").write_text("")
    return tmp_path


def test_changed_lines_are_the_new_side_of_each_hunk(pc):
    """Added and replaced lines are listed by their new numbers; a pure deletion
    and a deleted file contribute nothing."""
    assert pc.changed_lines(DIFF) == {
        "aquakin/a.py": {11, 12, 13, 23},
        "tests/unit/test_x.py": {1},
        "aquakin_ui/new.py": {1, 2},
    }


def test_a_line_is_covered_partial_or_missed(pc, tree):
    """Zero hits is missed; a branch line short of 100% is partial; a branch
    line at 100% and a plain line with hits are covered. The report names files
    relative to their source folder; each comes back by its repository path."""
    assert pc.line_states(REPORT, tree) == {
        "aquakin/a.py": {11: "covered", 12: "missed", 23: "partial", 40: "missed"},
        "aquakin_ui/new.py": {1: "covered", 2: "covered"},
    }


def test_the_report_counts_only_changed_measured_lines(pc, tree):
    """Line 13 is changed but not executable (absent from the report), line 40
    is missed but not changed, and the test file is outside the packages: none
    of the three is counted. Of the five counted lines three are covered."""
    reports, unmeasured = pc.patch_report(pc.changed_lines(DIFF), pc.line_states(REPORT, tree))
    assert unmeasured == []
    assert set(reports) == {"aquakin/a.py", "aquakin_ui/new.py"}
    a = reports["aquakin/a.py"]
    assert (a.covered, a.partial, a.missed) == ([11], [23], [12])
    new = reports["aquakin_ui/new.py"]
    assert (new.covered, new.partial, new.missed) == ([1, 2], [], [])
    assert pc.percent(reports) == pytest.approx(60.0)


def test_a_changed_package_file_absent_from_the_report_is_flagged(pc, tree):
    """A changed file under a measured package that the report never mentions
    is reported as unmeasured rather than silently passing."""
    changed = {"aquakin/never_imported.py": {1, 2}}
    reports, unmeasured = pc.patch_report(changed, pc.line_states(REPORT, tree))
    assert reports == {}
    assert unmeasured == ["aquakin/never_imported.py"]


def test_files_outside_the_packages_and_non_python_files_are_not_measured(pc):
    """Files outside the two packages and non-Python files are never counted."""
    assert pc.measured("aquakin/plant/plant.py")
    assert pc.measured("aquakin_ui/app.py")
    assert not pc.measured("aquakin/models/asm1.yaml")
    assert not pc.measured("tests/unit/test_model.py")
    assert not pc.measured("scripts/patch_coverage.py")


def test_no_changed_lines_is_full_coverage(pc):
    """A branch that changes no measured line has nothing to cover."""
    assert pc.percent({}) == 100.0


def test_line_numbers_print_as_ranges(pc):
    assert pc._ranges([3, 4, 5, 9, 11, 12]) == "3-5, 9, 11-12"


def test_a_name_shared_by_both_packages_resolves_under_its_own_source(pc, tmp_path):
    """``units.py`` exists in one package only; the report's bare name resolves
    there and not under the other source."""
    (tmp_path / "aquakin_ui").mkdir()
    (tmp_path / "aquakin_ui" / "units.py").write_text("")
    report = (
        "<coverage><sources><source>aquakin</source><source>aquakin_ui</source></sources>"
        '<packages><package><classes><class filename="units.py"><lines>'
        '<line number="1" hits="1"/></lines></class></classes></package></packages></coverage>'
    )
    assert pc.line_states(report, tmp_path) == {"aquakin_ui/units.py": {1: "covered"}}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A one-commit git repository holding a tracked ``aquakin/a.py`` and an
    empty coverage report, entered as the current folder."""

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "aquakin").mkdir()
    (tmp_path / "aquakin" / "a.py").write_text("x = 1\n")
    (tmp_path / ".gitignore").write_text("coverage.xml\n")
    (tmp_path / "coverage.xml").write_text("<coverage><sources/><packages/></coverage>")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_an_untracked_package_file_is_named_and_fails_the_check(pc, repo, capsys):
    """No ``git diff`` sees an untracked file, so without this it would pass as
    100% with its lines never counted. It is named, the check exits 1, and an
    untracked file outside the packages or not Python is ignored."""
    (repo / "aquakin" / "gases.py").write_text("y = 2\n")
    (repo / "aquakin" / "notes.yaml").write_text("")
    (repo / "scratch.py").write_text("")
    assert pc.untracked() == ["aquakin/gases.py"]
    assert pc.main(["--base", "HEAD"]) == 1
    out = capsys.readouterr().out
    assert "aquakin/gases.py: untracked, so its lines are not measured" in out
    assert "git add -N aquakin/gases.py" in out
    assert "notes.yaml" not in out
    assert "scratch.py" not in out


def test_an_intent_to_add_file_is_no_longer_untracked(pc, repo):
    """``git add -N`` puts the file in the diff, so it stops being flagged."""
    (repo / "aquakin" / "gases.py").write_text("y = 2\n")
    subprocess.run(["git", "add", "-N", "aquakin/gases.py"], cwd=repo, check=True)
    assert pc.untracked() == []
    assert "aquakin/gases.py" in pc._git("diff", "--name-only", "HEAD")


def test_nothing_untracked_gives_no_warning(pc, repo, capsys):
    """A clean tree with no changed lines passes with no untracked warning."""
    assert pc.untracked() == []
    assert pc.main(["--base", "HEAD"]) == 0
    assert "untracked" not in capsys.readouterr().out
