"""Two written size limits, checked: rule-file length and the large modules.

``CLAUDE.md`` asks for a rule file under about 250 lines ("Maintaining the
Claude-facing files"), because every line of a rule is read on every edit of the
files it matches, and names modules that must not grow (the Engineering
Standards). Both limits used to depend on someone noticing. This module turns
them into ratchets:

- a rule file under ``.claude/rules/`` must be within ``RULE_BUDGET`` lines, or
  be listed in ``RECORDED_LINES`` and stay within its recorded size plus
  ``RULE_ALLOWANCE``;
- each module the Engineering Standards name as not to grow is listed in
  ``RECORDED_LINES`` and stays within its recorded size plus
  ``MODULE_ALLOWANCE``.

Existing debt is recorded and cannot increase. **When a listed file shrinks**
(a rule condensed, a module split), set its entry to the new length in the same
change, a one-line edit; delete a rule file's entry once it is within the
budget. The tests ask for this once a file is more than its allowance below its
entry, so slack cannot pile up unseen. Raising an entry is not the fix for a
failure here: the message says what is.

Each rule file must also be scoped (``paths:`` frontmatter) and have a row in
the ``CLAUDE.md`` Index, unless it is listed in ``LOADED_EVERYWHERE``.

Lengths are counted as ``wc -l`` would, ignoring blank lines at the end of the
file, so trailing whitespace never fails a test. Everything here is text and
AST processing: no JAX, no ``aquakin`` import.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
RULES_DIR = Path(".claude/rules")

RULE_BUDGET = 250
RULE_ALLOWANCE = 10
MODULE_ALLOWANCE = 25

# The one table of ceilings: repository path -> length in lines when recorded.
# A file may grow to its entry plus the allowance for its kind. Lower an entry
# to the file's new length when it shrinks (see the module docstring).
RECORDED_LINES = {
    # Rule files over RULE_BUDGET, still narrative from being split out verbatim.
    ".claude/rules/core.md": 422,
    ".claude/rules/integration-differentiation.md": 395,
    ".claude/rules/plant.md": 401,
    ".claude/rules/plant-solve.md": 267,
    ".claude/rules/schema.md": 301,
    ".claude/rules/ui.md": 286,
    # The modules the Engineering Standards name as not to grow.
    "aquakin/plant/plant.py": 2888,
    "aquakin_ui/app.py": 570,
    "aquakin/integrate/_common.py": 1429,
    "aquakin/core/model.py": 1670,
}

# Rule files meant to load in every session, so deliberately without ``paths:``.
# None today; a file listed here must really have no ``paths:``.
LOADED_EVERYWHERE: set[str] = set()

RULE_ADVICE = (
    "Split it by topic into files with narrower `paths:` (each new file gets a "
    "row in the CLAUDE.md Index), or move measurements, timings and bug history "
    "to `.claude/notes/` and condense the rest to invariants."
)
MODULE_ADVICE = (
    "Do not grow this module: put new behaviour in its own module as free "
    "functions taking the object, bound onto the class as methods (the pattern "
    "of plant/sensitivity.py, plant/calibrate.py and plant/spec.py); new "
    "calibration code goes under integrate/calibration/."
)


def count_lines(text):
    """The number of lines in ``text``, ignoring blank lines at its end."""
    return len(text.rstrip().splitlines())


def rule_paths(text):
    """
    The ``paths:`` globs of a rule file, as Claude Code reads them.

    The frontmatter is YAML between ``---`` markers at the top of the file.
    ``paths`` may be a YAML list or a comma-separated string. Frontmatter that
    does not parse is ignored by Claude Code, which then loads the rule
    everywhere, so it counts as no ``paths:`` here too. Trailing whitespace on
    the markers and CRLF line endings are accepted.

    Returns
    -------
    list of str
        The non-empty globs; empty when the rule is not scoped.
    """
    match = re.match(r"---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(\r?\n|$)", text, re.DOTALL)
    if not match:
        return []
    try:
        front = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return []
    paths = front.get("paths") if isinstance(front, dict) else None
    if isinstance(paths, str):
        paths = paths.split(",")
    if not isinstance(paths, list):
        return []
    return [p.strip() for p in paths if isinstance(p, str) and p.strip()]


def indexed_rules(claude_md):
    """
    The rule files named in the first cell of a table row of ``CLAUDE.md``.

    Any Markdown table row counts, wherever the Index table sits and whatever
    its other columns say, so moving or reshaping the table does not matter.
    """
    found = set()
    for line in claude_md.splitlines():
        cells = line.strip().split("|")
        if len(cells) > 2 and line.strip().startswith("|"):
            found.update(re.findall(r"`(\.claude/rules/[^`]+\.md)`", cells[1]))
    return found


def growth_problems(path, length, recorded, allowance, advice):
    """The messages for a listed file over its ceiling or well under its entry."""
    ceiling = recorded + allowance
    if length > ceiling:
        return [
            f"{path} has {length} lines, {length - ceiling} over its ceiling of "
            f"{ceiling} ({recorded} recorded + {allowance} allowance). {advice}"
        ]
    if length < recorded - allowance:
        return [
            f"{path} has shrunk to {length} lines from its recorded {recorded}: "
            f"set its RECORDED_LINES entry in {Path(__file__).name} to {length} "
            "so the slack cannot be grown back into."
        ]
    return []


def rule_size_problems(root, recorded):
    """Size messages for every rule file under ``root``."""
    problems = []
    for file in sorted((root / RULES_DIR).rglob("*.md")):
        path = file.relative_to(root).as_posix()
        length = count_lines(file.read_text())
        if path in recorded:
            if length <= RULE_BUDGET:
                problems.append(
                    f"{path} is now within the {RULE_BUDGET}-line budget "
                    f"({length} lines): delete its RECORDED_LINES entry in "
                    f"{Path(__file__).name}."
                )
            else:
                problems += growth_problems(
                    path, length, recorded[path], RULE_ALLOWANCE, RULE_ADVICE
                )
        elif length > RULE_BUDGET:
            problems.append(
                f"{path} has {length} lines, {length - RULE_BUDGET} over the "
                f"{RULE_BUDGET}-line budget for a rule file. {RULE_ADVICE}"
            )
    return problems


def rule_scope_problems(root, everywhere):
    """Messages for rule files without ``paths:`` or without an Index row."""
    indexed = indexed_rules((root / "CLAUDE.md").read_text())
    problems = []
    for file in sorted((root / RULES_DIR).rglob("*.md")):
        path = file.relative_to(root).as_posix()
        scoped = bool(rule_paths(file.read_text()))
        if path in everywhere and scoped:
            problems.append(
                f"{path} is listed in LOADED_EVERYWHERE but has `paths:`: remove it from the list."
            )
        elif path not in everywhere and not scoped:
            problems.append(
                f"{path} has no `paths:` frontmatter (or it does not parse), so "
                "Claude Code loads it in every session. Add a `paths:` list of "
                "the files it governs, or, if it really belongs everywhere, add "
                "it to LOADED_EVERYWHERE."
            )
        if path not in indexed:
            problems.append(
                f"{path} has no row in the CLAUDE.md Index: add one whose first "
                f"cell is `{path}`, with the files it loads for and what it covers."
            )
    for path in sorted(indexed):
        if not (root / path).is_file():
            problems.append(f"The CLAUDE.md Index names {path}, which does not exist.")
    return problems


def modules_named_not_to_grow(claude_md):
    """
    The modules the Engineering Standards' "Do not grow" bullet names.

    Returns
    -------
    set of str or None
        Paths as written in the bullet (relative to ``aquakin/`` unless they
        start with another top-level package), or None when there is no such
        bullet.
    """
    match = re.search(r"Do not grow the large modules \(([^)]*)\)", claude_md)
    if match is None:
        return None
    return set(re.findall(r"`([^`]+\.py)`", match.group(1)))


def module_problems(root, recorded):
    """Size messages for the listed modules, and a mismatch with the standards."""
    named = modules_named_not_to_grow((root / "CLAUDE.md").read_text())
    if named is None:
        return [
            "CLAUDE.md no longer has the 'Do not grow the large modules (...)' "
            f"bullet that {Path(__file__).name} reads: update "
            "modules_named_not_to_grow to the new wording."
        ]
    expected = {
        p if p.split("/")[0] in {"aquakin", "aquakin_ui"} else f"aquakin/{p}" for p in named
    }
    listed = {p for p in recorded if p.endswith(".py")}
    problems = [
        f"CLAUDE.md names {p} as not to grow, but RECORDED_LINES has no entry "
        "for it: add one with its current length."
        for p in sorted(expected - listed)
    ]
    problems += [
        f"RECORDED_LINES lists {p}, which CLAUDE.md's 'Do not grow' bullet no "
        "longer names: remove the entry."
        for p in sorted(listed - expected)
    ]
    for path in sorted(listed & expected):
        length = count_lines((root / path).read_text())
        problems += growth_problems(path, length, recorded[path], MODULE_ALLOWANCE, MODULE_ADVICE)
    return problems


# --- the checks on this repository -------------------------------------------


def test_rule_files_are_within_their_budget_or_recorded_ceiling():
    problems = rule_size_problems(ROOT, RECORDED_LINES)
    assert not problems, "\n".join(problems)


def test_rule_files_are_scoped_and_indexed():
    problems = rule_scope_problems(ROOT, LOADED_EVERYWHERE)
    assert not problems, "\n".join(problems)


def test_large_modules_do_not_grow():
    problems = module_problems(ROOT, RECORDED_LINES)
    assert not problems, "\n".join(problems)


def test_every_recorded_file_exists():
    missing = [p for p in RECORDED_LINES if not (ROOT / p).is_file()]
    assert missing == [], f"RECORDED_LINES names files that do not exist: {missing}"


# --- the checks themselves, on small synthetic trees --------------------------

SCOPED = '---\npaths:\n  - "pkg/**"\n---\n\n# Rule\n'
INDEX = "| Rule | Loads for |\n|---|---|\n| `.claude/rules/a.md` | `pkg/**` |\n"
STANDARDS = "- Do not grow the large modules (`plant/big.py`, `aquakin_ui/app.py`). New\n"


def _tree(tmp_path, rules, claude_md=INDEX + STANDARDS, modules=None):
    (tmp_path / RULES_DIR).mkdir(parents=True)
    for name, text in rules.items():
        file = tmp_path / RULES_DIR / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
    (tmp_path / "CLAUDE.md").write_text(claude_md)
    for path, n in (modules or {}).items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("x = 1\n" * n)
    return tmp_path


def test_count_lines_ignores_trailing_blank_lines_and_whitespace():
    assert count_lines("a\nb\n") == 2
    assert count_lines("a\nb  \n\n\n   \n") == 2
    assert count_lines("a\nb") == 2


@pytest.mark.parametrize(
    "text, expected",
    [
        (SCOPED, ["pkg/**"]),
        ('---\npaths: "a/**, b/*.py"\n---\n', ["a/**", "b/*.py"]),
        ('---  \r\npaths:\r\n  - "a/**"  \r\n--- \r\n# Rule\r\n', ["a/**"]),
        ("# no frontmatter\n", []),
        ("---\ntitle: x\n---\n", []),
        ("---\npaths: [unclosed\n---\n", []),
        ("---\npaths: []\n---\n", []),
        ("---\npaths:\n---\n", []),
    ],
    ids=[
        "list",
        "comma-string",
        "trailing-space-crlf",
        "none",
        "no-paths-key",
        "unparseable",
        "empty-list",
        "null",
    ],
)
def test_rule_paths_reads_what_claude_code_accepts(text, expected):
    assert rule_paths(text) == expected


def test_indexed_rules_reads_first_cells_of_any_table_row():
    text = (
        "Prose naming `.claude/rules/prose.md` is not a row.\n"
        "| `.claude/rules/a.md` | x |\n"
        "| x | `.claude/rules/second-cell.md` |\n"
        "  | `.claude/rules/sub/b.md` | moved, indented table |\n"
    )
    assert indexed_rules(text) == {".claude/rules/a.md", ".claude/rules/sub/b.md"}


def test_rule_file_size_ratchet(tmp_path):
    over = SCOPED + "x\n" * RULE_BUDGET
    n = count_lines(over)
    root = _tree(tmp_path, {"a.md": over, "b.md": SCOPED})
    [msg] = rule_size_problems(root, {})
    assert msg.startswith(
        f".claude/rules/a.md has {n} lines, {n - RULE_BUDGET} over the {RULE_BUDGET}"
    )
    assert "`.claude/notes/`" in msg
    # Recorded: passes up to the allowance, fails one line past it.
    assert rule_size_problems(root, {".claude/rules/a.md": n - RULE_ALLOWANCE}) == []
    [msg] = rule_size_problems(root, {".claude/rules/a.md": n - RULE_ALLOWANCE - 1})
    assert "1 over its ceiling" in msg
    # Condensed well below the entry: lower it; within the budget: delete it.
    [msg] = rule_size_problems(root, {".claude/rules/a.md": n + RULE_ALLOWANCE + 1})
    assert f"set its RECORDED_LINES entry in test_size_limits.py to {n}" in msg
    recorded = {".claude/rules/a.md": n, ".claude/rules/b.md": 400}
    [msg] = rule_size_problems(root, recorded)
    assert msg.startswith(".claude/rules/b.md is now within the 250-line budget")
    assert "delete its RECORDED_LINES entry" in msg


def test_rule_scope_and_index(tmp_path):
    root = _tree(tmp_path, {"a.md": SCOPED, "b.md": "# unscoped\n", "sub/c.md": SCOPED})
    problems = rule_scope_problems(root, set())
    assert problems == [
        ".claude/rules/b.md has no `paths:` frontmatter (or it does not parse), so "
        "Claude Code loads it in every session. Add a `paths:` list of the files it "
        "governs, or, if it really belongs everywhere, add it to LOADED_EVERYWHERE.",
        ".claude/rules/b.md has no row in the CLAUDE.md Index: add one whose first "
        "cell is `.claude/rules/b.md`, with the files it loads for and what it covers.",
        ".claude/rules/sub/c.md has no row in the CLAUDE.md Index: add one whose "
        "first cell is `.claude/rules/sub/c.md`, with the files it loads for and "
        "what it covers.",
    ]
    # A rule meant to load everywhere is accepted only while it really has no paths.
    assert all(
        "b.md has no `paths:`" not in p for p in rule_scope_problems(root, {".claude/rules/b.md"})
    )
    [msg] = [p for p in rule_scope_problems(root, {".claude/rules/a.md"}) if "is listed in" in p]
    assert msg.startswith(".claude/rules/a.md is listed in LOADED_EVERYWHERE but has `paths:`")


def test_index_row_for_a_missing_rule_fails(tmp_path):
    root = _tree(tmp_path, {}, claude_md=INDEX + STANDARDS)
    assert rule_scope_problems(root, set()) == [
        "The CLAUDE.md Index names .claude/rules/a.md, which does not exist."
    ]


def test_module_ratchet(tmp_path):
    modules = {"aquakin/plant/big.py": 100, "aquakin_ui/app.py": 50}
    root = _tree(tmp_path, {}, modules=modules)
    assert module_problems(root, modules) == []
    assert module_problems(root, {**modules, "aquakin/plant/big.py": 100 - MODULE_ALLOWANCE}) == []
    [msg] = module_problems(root, {**modules, "aquakin/plant/big.py": 100 - MODULE_ALLOWANCE - 1})
    assert msg.startswith("aquakin/plant/big.py has 100 lines, 1 over its ceiling")
    assert "free functions taking the object" in msg
    [msg] = module_problems(root, {**modules, "aquakin/plant/big.py": 100 + MODULE_ALLOWANCE + 1})
    assert "set its RECORDED_LINES entry in test_size_limits.py to 100" in msg


def test_module_table_matches_the_standards(tmp_path):
    root = _tree(tmp_path, {}, modules={"aquakin/plant/big.py": 1, "aquakin_ui/app.py": 1})
    assert module_problems(root, {"aquakin_ui/app.py": 1, "aquakin/core/gone.py": 1}) == [
        "CLAUDE.md names aquakin/plant/big.py as not to grow, but RECORDED_LINES "
        "has no entry for it: add one with its current length.",
        "RECORDED_LINES lists aquakin/core/gone.py, which CLAUDE.md's 'Do not grow' "
        "bullet no longer names: remove the entry.",
    ]
    (root / "CLAUDE.md").write_text(INDEX)
    [msg] = module_problems(root, {})
    assert "no longer has the 'Do not grow the large modules (...)' bullet" in msg
