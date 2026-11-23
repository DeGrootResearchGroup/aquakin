"""The changelog is assembled from fragments in changelog.d/ by towncrier.

A change records itself as one file there rather than as an edit to
CHANGELOG.md, so pull requests do not conflict over the changelog. These checks
keep the fragments in the form towncrier reads at release: a fragment it cannot
name is silently left out of the release notes.
"""

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["towncrier"]
FRAGMENTS = ROOT / CONFIG["directory"]
TYPES = [t["directory"] for t in CONFIG["type"]]
NOT_FRAGMENTS = {"README.md"}

# <issue number> or +<slug>, then the type, then an optional counter for a
# second entry of the same name and type.
NAME = re.compile(rf"^(\d+|\+[a-z0-9][a-z0-9-]*)\.({'|'.join(TYPES)})(\.\d+)?\.md$")


def fragment_paths():
    return sorted(p for p in FRAGMENTS.iterdir() if p.name not in NOT_FRAGMENTS)


def test_the_types_are_keep_a_changelog_groups():
    assert TYPES == ["added", "changed", "deprecated", "removed", "fixed", "security"]


@pytest.mark.parametrize("path", fragment_paths(), ids=lambda p: p.name)
def test_fragment_is_named_and_written_as_towncrier_reads_it(path):
    assert NAME.match(path.name), (
        f"changelog.d/{path.name} is not <issue>.<type>.md or +<slug>.<type>.md with a "
        f"type in {TYPES}; towncrier would leave it out of the release notes"
    )
    text = path.read_text()
    assert text.strip(), f"changelog.d/{path.name} is empty"
    first = text.lstrip().splitlines()[0]
    assert not re.match(r"[-*] ", first), (
        f"changelog.d/{path.name} starts with a list marker; towncrier adds the bullet"
    )


def test_changelog_is_not_edited_for_unreleased_changes():
    """Unreleased changes live in changelog.d/, never in CHANGELOG.md itself."""
    text = (ROOT / CONFIG["filename"]).read_text()
    assert text.count(CONFIG["start_string"]) == 1
    assert not re.search(r"^## \[Unreleased\]", text, flags=re.MULTILINE), (
        "record an unreleased change as a fragment in changelog.d/ (see "
        "changelog.d/README.md), not as an [Unreleased] section in CHANGELOG.md"
    )
