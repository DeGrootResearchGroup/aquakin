"""The organization of the published documentation, checked.

``.claude/rules/docs.md`` sets out how the user documentation stays organized.
The parts of it a script can check are checked here, so they do not depend on
someone reading the rule:

- ``index.md``'s root toctree lists exactly the section landing pages, and every
  other published page is in exactly one landing page's toctree;
- each landing page's table links every page of its toctree;
- every published page is within ``PAGE_BUDGET`` lines, or is listed in
  ``RECORDED_PAGE_LINES`` and stays within its recorded length plus
  ``PAGE_ALLOWANCE`` (lower the entry when the page shrinks, and delete it once
  the page is within the budget);
- every published page is in the root ``conftest.py``'s ``GUIDE_PAGES``, so its
  examples run, and in the ``paths:`` of ``.claude/rules/docs.md``, so the rule
  loads when the page is edited.

A published page is a ``docs/*.md`` file not in ``docs/conf.py``'s
``exclude_patterns``, other than the API reference Sphinx generates. Everything
here is text and AST processing: no Sphinx build, no ``aquakin`` import.
"""

import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs"
DOCS_RULE = ROOT / ".claude" / "rules" / "docs.md"

SECTIONS = ["getting_started", "models", "simulation", "analysis", "reference"]
# Getting started is a section of one page, so it has no landing table.
LANDING_PAGES = ["models", "simulation", "analysis", "reference"]
GENERATED_PAGES = {"api"}

PAGE_BUDGET = 600
PAGE_ALLOWANCE = 20
# Published pages over PAGE_BUDGET: page -> length in lines when recorded.
RECORDED_PAGE_LINES = {
    "model_format": 709,
}

SPLIT_ADVICE = (
    "Split it by task into sibling pages of the same section, each with its own "
    "setup block (see 'Page size and splitting' in .claude/rules/docs.md)."
)

_TOCTREE = re.compile(r"```\{toctree\}\n(.*?)```", re.S)
_ENTRY = re.compile(r"^(?:.*<(?P<target>[^>]+)>|(?P<name>\S+))\s*$")
_LINK = re.compile(r"\]\((?P<target>[\w-]+)\.md(?:#[\w-]*)?\)")


def count_lines(text):
    """The number of lines in ``text``, ignoring blank lines at its end."""
    return len(text.rstrip().splitlines())


def module_assignment(path, name):
    """The literal value assigned to ``name`` at the top level of ``path``."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise LookupError(f"{path} assigns no {name}")


def published_pages(docs):
    """The page names (file stems) the Sphinx build publishes."""
    excluded = set(module_assignment(docs / "conf.py", "exclude_patterns"))
    return sorted(
        page.stem
        for page in docs.glob("*.md")
        if page.name not in excluded and page.stem not in GENERATED_PAGES
    )


def toctree_entries(text):
    """The page names in a page's toctrees, in order."""
    entries = []
    for block in _TOCTREE.findall(text):
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            match = _ENTRY.match(line)
            entries.append(match["target"] or match["name"])
    return entries


def structure_problems(docs):
    """Ways the toctrees and landing tables break the section structure."""
    problems = []
    root = toctree_entries((docs / "index.md").read_text())
    if root != SECTIONS:
        problems.append(
            f"index.md's root toctree is {root}; it must list exactly the "
            f"sections {SECTIONS}. Put a new page in a section's toctree."
        )
    owners = {}
    for landing in LANDING_PAGES:
        text = (docs / f"{landing}.md").read_text()
        linked = {m["target"] for m in _LINK.finditer(text.split("```{toctree}")[0])}
        for page in toctree_entries(text):
            owners.setdefault(page, []).append(landing)
            if page not in linked:
                problems.append(
                    f"{landing}.md lists {page} in its toctree but its table "
                    f"does not link {page}.md; add a row for it."
                )
    for page in published_pages(docs):
        if page in SECTIONS or page == "index":
            continue
        found = owners.get(page, [])
        if len(found) != 1:
            problems.append(
                f"{page}.md is in the toctree of {found or 'no landing page'}; "
                f"it must be in exactly one of {LANDING_PAGES}."
            )
    return problems


def size_problems(docs, recorded):
    """Pages over their budget, and recorded entries to lower or delete."""
    problems = []
    for page in published_pages(docs):
        length = count_lines((docs / f"{page}.md").read_text())
        if page in recorded:
            ceiling = recorded[page] + PAGE_ALLOWANCE
            if length <= PAGE_BUDGET:
                problems.append(
                    f"{page}.md is now within the {PAGE_BUDGET}-line budget; "
                    "delete its RECORDED_PAGE_LINES entry."
                )
            elif length > ceiling:
                problems.append(
                    f"{page}.md has {length} lines, over its recorded "
                    f"{recorded[page]} plus {PAGE_ALLOWANCE}. {SPLIT_ADVICE}"
                )
            elif length < recorded[page] - PAGE_ALLOWANCE:
                problems.append(
                    f"{page}.md shrank to {length} lines; set its "
                    "RECORDED_PAGE_LINES entry to that."
                )
        elif length > PAGE_BUDGET:
            problems.append(
                f"{page}.md has {length} lines, over the {PAGE_BUDGET}-line "
                f"budget for a page. {SPLIT_ADVICE}"
            )
    return problems


def coverage_problems(root, docs, rule):
    """Published pages missing from GUIDE_PAGES or from the rule's ``paths:``."""
    guide = set(module_assignment(root / "conftest.py", "GUIDE_PAGES"))
    front = rule.read_text().split("---")[1]
    paths = set(yaml.safe_load(front)["paths"])
    problems = []
    for page in published_pages(docs):
        path = f"docs/{page}.md"
        if path not in guide:
            problems.append(f"{path} is published but not in GUIDE_PAGES.")
        if path not in paths:
            problems.append(f"{path} is published but not in the docs rule's paths:.")
    return problems


def test_sections_and_landing_pages():
    assert structure_problems(DOCS) == []


def test_pages_are_within_their_budget():
    assert size_problems(DOCS, RECORDED_PAGE_LINES) == []


def test_every_page_is_tested_and_governed_by_the_rule():
    assert coverage_problems(ROOT, DOCS, DOCS_RULE) == []


def test_every_recorded_page_exists():
    published = published_pages(DOCS)
    assert [page for page in RECORDED_PAGE_LINES if page not in published] == []


# ---- The checks themselves, on small trees ---------------------------------

CONF = 'exclude_patterns = ["_build", "ci.md"]\n'
INDEX = "# x\n\n```{toctree}\n:hidden:\n\n" + "\n".join(SECTIONS) + "\n```\n"


def landing(entries, linked):
    table = "".join(f"| a | [{p}]({p}.md) |\n" for p in linked)
    return "# L\n\n" + table + "\n```{toctree}\n" + "\n".join(entries) + "\n```\n"


def _docs(tmp_path, pages):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "conf.py").write_text(CONF)
    (docs / "ci.md").write_text("x\n" * (PAGE_BUDGET + 50))  # excluded: never checked
    files = {"index": INDEX, "getting_started": "# G\n"}
    files.update({name: landing([], []) for name in LANDING_PAGES})
    files.update(pages)
    for name, text in files.items():
        (docs / f"{name}.md").write_text(text)
    return docs


def test_a_well_formed_tree_has_no_problems(tmp_path):
    docs = _docs(
        tmp_path,
        {"models": landing(["catalog"], ["catalog"]), "catalog": "# C\n"},
    )
    assert structure_problems(docs) == []
    assert size_problems(docs, {}) == []


def test_a_page_in_the_root_toctree_is_reported(tmp_path):
    docs = _docs(tmp_path, {"index": INDEX.replace("reference", "reference\nextra")})
    assert any("root toctree" in p for p in structure_problems(docs))


def test_a_page_in_no_section_is_reported(tmp_path):
    docs = _docs(tmp_path, {"orphan": "# O\n"})
    assert any("orphan.md" in p and "no landing page" in p for p in structure_problems(docs))


def test_a_page_missing_from_its_landing_table_is_reported(tmp_path):
    docs = _docs(tmp_path, {"models": landing(["catalog"], []), "catalog": "# C\n"})
    assert any("does not link catalog.md" in p for p in structure_problems(docs))


def test_a_titled_toctree_entry_is_read_by_its_target(tmp_path):
    docs = _docs(
        tmp_path,
        {"models": landing(["The catalogue <catalog>"], ["catalog"]), "catalog": "# C\n"},
    )
    assert structure_problems(docs) == []


@pytest.mark.parametrize(
    ("length", "recorded", "expected"),
    [
        (PAGE_BUDGET + 1, {}, "over the"),
        (PAGE_BUDGET + 100, {"long": PAGE_BUDGET + 100}, None),
        (PAGE_BUDGET + 200, {"long": PAGE_BUDGET + 100}, "over its recorded"),
        (PAGE_BUDGET + 10, {"long": PAGE_BUDGET + 100}, "shrank"),
        (PAGE_BUDGET - 1, {"long": PAGE_BUDGET + 100}, "delete its"),
    ],
)
def test_page_sizes_ratchet(tmp_path, length, recorded, expected):
    docs = _docs(
        tmp_path,
        {"models": landing(["long"], ["long"]), "long": "line\n" * length},
    )
    problems = size_problems(docs, recorded)
    if expected is None:
        assert problems == []
    else:
        assert len(problems) == 1 and expected in problems[0]


def test_a_page_missing_from_guide_pages_or_the_rule_is_reported(tmp_path):
    docs = _docs(tmp_path, {})
    (tmp_path / "conftest.py").write_text('GUIDE_PAGES = ["docs/index.md"]\n')
    rule = tmp_path / "rule.md"
    rule.write_text('---\npaths:\n  - "docs/models.md"\n---\n')
    problems = coverage_problems(tmp_path, docs, rule)
    assert "docs/models.md is published but not in GUIDE_PAGES." in problems
    assert "docs/index.md is published but not in the docs rule's paths:." in problems
    assert not any("ci.md" in p for p in problems)
