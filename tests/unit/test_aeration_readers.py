"""Tests read a tank's aeration through its public readers, not its storage.

An aerated unit keeps its per-species aeration vectors in one store, and
``_kla_vec``, ``_sat_vec`` and ``_controlled_kla`` are private views onto that
store. A test that asserts through them pins how the vectors are stored rather
than what the tank does, so a restructure that keeps the behaviour still breaks
it. That happened when ``alpha`` moved out of the stored kLa: tests failed that
were checking only the layout.

The public readers are the contract:

- ``kla_vector(signals)`` and ``oxygen_kla(signals)``: the clean-water kLa
  (``process_water=True`` for the kLa after ``alpha``);
- ``delivered_oxygen_kla(inputs, signals)``: the kLa the solve applies;
- ``aeration_term`` / ``oxygen_transfer``: the transfer itself.

This module turns that into a ratchet over ``tests/``, as
``test_raises_match.py`` does for message-less error tests. An attribute access
to one of the private names counts against its file. A file's count may not
exceed its entry in ``RECORDED`` (0 when unlisted), and an entry must be lowered
when the file improves, so the debt only shrinks. Mentions in docstrings and
comments are not counted.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]
PRIVATE = {"_kla_vec", "_sat_vec", "_controlled_kla"}

# Pre-existing debt: test file -> number of private aeration reads.
# Lower an entry when you fix some; delete it at 0. Never raise one.
RECORDED = {
    "tests/integration/test_aeration.py": 13,
    "tests/unit/test_aeration_stripping.py": 2,
}


def private_reads(source):
    """
    The line numbers of the private aeration-storage reads in ``source``.

    Returns
    -------
    list of int
        One entry per attribute access to a name in :data:`PRIVATE`.
    """
    return sorted(
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute) and node.attr in PRIVATE
    )


def read_counts(root=ROOT):
    """Test file (repository-relative) -> number of private reads, nonzero only."""
    counts = {}
    for file in sorted((root / "tests").rglob("*.py")):
        if file.resolve() == Path(__file__).resolve():
            continue
        n = len(private_reads(file.read_text()))
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
                f"{path} reads a tank's private aeration storage "
                f"({', '.join(sorted(PRIVATE))}) {now} times, {now - allowed} more "
                f"than its recorded {allowed}. Assert through the public readers "
                "instead: kla_vector / oxygen_kla (process_water=True for the kLa "
                "after alpha), delivered_oxygen_kla, or the transfer term "
                "(aeration_term / oxygen_transfer)."
            )
        elif now < allowed:
            fix = f"set its RECORDED entry to {now}" if now else "delete its RECORDED entry"
            problems.append(
                f"{path} now has {now}, below its recorded {allowed}: {fix} in "
                f"{Path(__file__).name} so the debt cannot grow back."
            )
    return problems


# --- the check on this repository -------------------------------------------


def test_tests_read_aeration_through_public_readers():
    problems = ratchet_problems(read_counts(), RECORDED)
    assert not problems, "\n".join(problems)


# --- the check itself ---------------------------------------------------------

SAMPLE = '''
def test_a(tank):
    """Mentions ``_kla_vec`` in a docstring, which does not count."""
    # nor does _controlled_kla in a comment
    assert tank._kla_vec[0] == 1.0
    assert tank.kla_vector({})[0] == 1.0
    assert tank._sat_vec[0] == 8.0 and tank._controlled_kla == {}
'''


def test_private_reads_counts_attribute_access_only():
    assert private_reads(SAMPLE) == [5, 7, 7]


def test_ratchet_problems():
    counts = {"t/a.py": 3, "t/b.py": 1, "t/new.py": 1}
    recorded = {"t/a.py": 3, "t/b.py": 2, "t/gone.py": 4}
    problems = ratchet_problems(counts, recorded)
    assert problems[0].startswith("t/b.py now has 1, below its recorded 2: set its")
    assert problems[1].startswith("t/gone.py now has 0, below its recorded 4: delete")
    assert problems[2].startswith("t/new.py reads a tank's private aeration storage")
    assert "kla_vector / oxygen_kla" in problems[2]
    assert len(problems) == 3
