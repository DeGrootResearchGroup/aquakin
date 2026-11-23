"""An error test names the error by its message, as tests.md asks.

``.claude/rules/tests.md``: "Errors are tested by message:
``pytest.raises(SomeError, match="...")`` with the specific exception class and a
fragment of the message that identifies the branch. Never
``pytest.raises(Exception)``". A ``pytest.raises`` without ``match=`` passes for
any error of that class, including one raised by a different branch, or by a
bug before the code under test is reached.

This module turns that into a ratchet over ``tests/``, as
``test_size_limits.py`` does for file sizes:

- a ``pytest.raises(...)`` call without ``match=`` counts against its file;
- so does one whose class is ``Exception`` or ``BaseException`` (even with
  ``match=``, since the class itself then checks nothing);
- a file's count may not exceed its entry in ``RECORDED`` (0 when unlisted), and
  an entry must be lowered when the file improves, so the debt only shrinks.

``with pytest.raises(E) as excinfo:`` followed by a use of ``excinfo`` in the
same test is not counted: the test reads the exception afterwards (its message,
its type, an attribute), which is what ``match=`` would do.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]
BROAD = {"Exception", "BaseException"}

# Pre-existing debt: test file -> number of message-less or broad raises.
# Lower an entry when you fix some; delete it at 0. Never raise one.
RECORDED = {
    "tests/integration/test_observation_units.py": 2,
    "tests/integration/test_plant_calibrate.py": 5,
    "tests/integration/test_plant_identifiability.py": 2,
    "tests/integration/test_study.py": 2,
    "tests/ui/test_main.py": 1,
    "tests/unit/test_conditions.py": 1,
    "tests/unit/test_plant_units_raises.py": 1,
    "tests/unit/test_precedence.py": 2,
    "tests/unit/test_rtd.py": 2,
}


def _is_pytest_raises(node):
    func = node.func if isinstance(node, ast.Call) else None
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "raises"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def _is_weak(call):
    """Whether a ``pytest.raises`` call checks no message, or a broad class."""
    broad = bool(call.args) and isinstance(call.args[0], ast.Name) and call.args[0].id in BROAD
    return broad or not any(k.arg == "match" for k in call.keywords)


def _inspected_calls(tree):
    """The ``pytest.raises`` calls whose ``as`` name is read later in the test."""
    inspected = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        read = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        for node in ast.walk(fn):
            if isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    var = item.optional_vars
                    if _is_pytest_raises(item.context_expr) and isinstance(var, ast.Name):
                        if var.id in read:
                            inspected.add(id(item.context_expr))
    return inspected


def weak_raises(source):
    """
    The line numbers of the weak ``pytest.raises`` calls in ``source``.

    Weak means no ``match=``, or ``Exception`` / ``BaseException`` as the class,
    except a message-less one whose ``as`` name the test reads afterwards (a
    broad class is weak even then).

    Returns
    -------
    list of int
    """
    tree = ast.parse(source)
    inspected = _inspected_calls(tree)
    lines = []
    for node in ast.walk(tree):
        if not (_is_pytest_raises(node) and _is_weak(node)):
            continue
        broad = bool(node.args) and isinstance(node.args[0], ast.Name) and node.args[0].id in BROAD
        if id(node) in inspected and not broad:
            continue
        lines.append(node.lineno)
    return sorted(lines)


def weak_counts(root=ROOT):
    """Test file (repository-relative) -> number of weak raises, nonzero only."""
    counts = {}
    for file in sorted((root / "tests").rglob("*.py")):
        n = len(weak_raises(file.read_text()))
        if n:
            counts[file.relative_to(root).as_posix()] = n
    return counts


def ratchet_problems(counts, recorded):
    """Messages for files over their recorded count, or with a stale entry."""
    problems = []
    for path in sorted(set(counts) | set(recorded)):
        now, allowed = counts.get(path, 0), recorded.get(path, 0)
        if now > allowed:
            problems.append(
                f"{path} has {now} pytest.raises without match= (or with a broad "
                f"Exception class), {now - allowed} more than its recorded {allowed}. "
                "Add match= with a fragment of the message that identifies the "
                "branch, and the specific exception class."
            )
        elif now < allowed:
            fix = f"set its RECORDED entry to {now}" if now else "delete its RECORDED entry"
            problems.append(
                f"{path} now has {now}, below its recorded {allowed}: {fix} in "
                f"{Path(__file__).name} so the debt cannot grow back."
            )
    return problems


# --- the check on this repository -------------------------------------------


def test_error_tests_match_a_message():
    problems = ratchet_problems(weak_counts(), RECORDED)
    assert not problems, "\n".join(problems)


# --- the check itself ---------------------------------------------------------

SAMPLE = '''
import pytest

def test_a():
    with pytest.raises(ValueError, match="bad"):
        f()
    with pytest.raises(ValueError):
        f()
    with pytest.raises(Exception, match="x"):
        f()

def test_b():
    with pytest.raises(KeyError) as excinfo:
        f()
    assert "k" in str(excinfo.value)
    with pytest.raises(KeyError) as unused:
        f()
    with pytest.raises(BaseException) as e:
        f()
    assert e.value
'''


def test_weak_raises_finds_the_message_less_and_broad():
    assert weak_raises(SAMPLE) == [7, 9, 16, 18]


def test_ratchet_problems():
    counts = {"t/a.py": 3, "t/b.py": 1, "t/new.py": 1}
    recorded = {"t/a.py": 3, "t/b.py": 2, "t/gone.py": 4}
    problems = ratchet_problems(counts, recorded)
    assert problems[0].startswith("t/b.py now has 1, below its recorded 2: set its")
    assert problems[1].startswith("t/gone.py now has 0, below its recorded 4: delete")
    assert problems[2].startswith("t/new.py has 1 pytest.raises without match=")
    assert len(problems) == 3
