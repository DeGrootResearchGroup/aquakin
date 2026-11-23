"""The Markdown prose uses -ize / -yze, as CLAUDE.md's Spelling Convention asks.

The convention is Canadian English with **-ize** and **-yze** endings
(*optimize*, *analyze*). It used to be prose only, and -ise spellings kept
arriving in PR text, docs and changelog entries. This module checks the
Markdown a reader sees: ``README.md``, ``CLAUDE.md``, ``docs/*.md``, the
``changelog.d/`` fragments, and the rules and notes under ``.claude/``.
``CHANGELOG.md`` is left alone: towncrier writes it from released entries.

Rather than a hand-written list (which would flag the correct *precise*,
*noise* or *otherwise*), the words checked are the ones the repository already
spells with -iz / -yz: every word of the form ``<stem>iz<suffix>`` or
``<stem>yz<suffix>`` in these files and in the package's source. Its -is / -ys
spelling is then an error. A word the repository has never written with a z is
not checked; ``KNOWN_STEMS`` seeds the ones CLAUDE.md names. Stems of three
letters or fewer are skipped, since short identifiers (``fsize``) make them
ambiguous.

Code is not prose: inline code spans and fenced blocks are skipped, so an
identifier or a quoted command keeps its spelling. ``analyses`` is never
flagged, being the plural of *analysis*.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]

# Stems CLAUDE.md names, so they are checked even before the repository
# happens to spell them with a z.
KNOWN_STEMS = {"optimi", "summari", "recogni", "analy", "normali", "parameteri"}

IZE = re.compile(r"\b([A-Za-z]{3,}i)z(?:e|es|ed|ing|ation|ations|er|ers|able)\b")
YZE = re.compile(r"\b([A-Za-z]{3,}y)z(?:e|ed|ing|er|ers)\b")
ISE = re.compile(
    r"\b(([A-Za-z]{3,}i)s(?:e|es|ed|ing|ation|ations|er|ers|able)"
    r"|([A-Za-z]{3,}y)s(?:e|ed|ing|er|ers))\b"
)
FENCE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", re.DOTALL | re.MULTILINE)
CODE_SPAN = re.compile(r"`[^`\n]*`")


def prose_files(root):
    """The Markdown files whose prose is checked, in a stable order."""
    files = [root / "README.md", root / "CLAUDE.md"]
    for folder in ("docs", "changelog.d", ".claude/rules", ".claude/notes"):
        files += sorted((root / folder).glob("*.md"))
    return [f for f in files if f.is_file()]


def prose(text):
    """``text`` with fenced blocks and inline code spans blanked out.

    Line structure is kept (a fenced block becomes as many empty lines), so a
    match can still be reported by line number.
    """
    text = FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    return CODE_SPAN.sub("", text)


def ize_stems(texts):
    """The lower-case stems spelled with -iz / -yz anywhere in ``texts``."""
    stems = set(KNOWN_STEMS)
    for text in texts:
        stems |= {m.lower() for m in IZE.findall(text)}
        stems |= {m.lower() for m in YZE.findall(text)}
    return stems


def spelling_problems(files, stems):
    """
    Every -ise / -yse spelling of a stem in ``stems``, with where it is.

    Returns
    -------
    list of str
        ``"<path>:<line>: <word>"`` for each occurrence.
    """
    problems = []
    for file in files:
        for lineno, line in enumerate(prose(file.read_text()).splitlines(), 1):
            for match in ISE.finditer(line):
                stem = (match.group(2) or match.group(3)).lower()
                if stem in stems:
                    problems.append(f"{file.relative_to(ROOT)}:{lineno}: {match.group(1)}")
    return problems


def repository_stems():
    """The -iz / -yz stems of the checked prose and the package source."""
    texts = [f.read_text() for f in prose_files(ROOT)]
    texts += [f.read_text() for f in sorted((ROOT / "aquakin").rglob("*.py"))]
    return ize_stems(texts)


# --- the check on this repository --------------------------------------------


def test_prose_uses_ize_spellings():
    problems = spelling_problems(prose_files(ROOT), repository_stems())
    assert not problems, (
        "\n".join(problems) + "\nUse the -ize / -yze spelling (CLAUDE.md, "
        "'Spelling Convention'); code spans and fenced blocks are not checked."
    )


# --- the check itself ---------------------------------------------------------


def test_stems_come_from_z_spellings_only():
    stems = ize_stems(["We optimize, then factorization; the catalyzed step.", "prise"])
    assert {"optimi", "factori", "cataly"} <= stems
    assert "pri" not in stems


def test_prose_skips_code_and_keeps_lines():
    text = "a `normalise()` b\n```\noptimise\n```\nnormalised\n"
    assert prose(text) == "a  b\n\n\n\nnormalised\n"


def test_problems_name_file_line_and_word(tmp_path, monkeypatch):
    monkeypatch.setattr(f"{__name__}.ROOT", tmp_path)
    page = tmp_path / "page.md"
    page.write_text(
        "Optimised and normalisation.\n"
        "Precise noise, otherwise fine; analyses are plural.\n"
        "Call `optimise()` here.\n"
        "We analysed it and analyse more.\n"
    )
    stems = {"optimi", "normali", "analy"}
    assert spelling_problems([page], stems) == [
        "page.md:1: Optimised",
        "page.md:1: normalisation",
        "page.md:4: analysed",
        "page.md:4: analyse",
    ]
