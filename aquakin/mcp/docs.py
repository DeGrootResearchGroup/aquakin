"""The published user guide as pages an assistant can list, read and search.

A built package carries the published Markdown pages in ``aquakin/mcp/guide/``
(``setup.py`` copies them there when the wheel is built). A source checkout has
no such folder, and the pages are read from its ``docs/`` folder instead. Which
pages are published is decided in one place: every ``docs/*.md`` except the
``exclude_patterns`` of ``docs/conf.py``, the list the Sphinx build itself uses.

This module imports nothing from aquakin (only the standard library), because
``setup.py`` loads it by path while building the package, before aquakin or its
dependencies are importable.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

#: The folder, beside this module in a built package, that holds the pages.
BUNDLE = "guide"

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def published_pages(docs_dir) -> list[Path]:
    """The Markdown pages the documentation site publishes.

    Parameters
    ----------
    docs_dir : str or pathlib.Path
        A ``docs/`` folder holding ``conf.py`` and the pages.

    Returns
    -------
    list of pathlib.Path
        Every ``*.md`` in ``docs_dir`` that ``conf.py``'s ``exclude_patterns``
        does not name, sorted by name.

    Raises
    ------
    ValueError
        If ``conf.py`` has no ``exclude_patterns`` list of literal strings.
    """
    docs_dir = Path(docs_dir)
    excluded = _excluded(docs_dir / "conf.py")
    return sorted(p for p in docs_dir.glob("*.md") if p.name not in excluded)


def _excluded(conf: Path) -> set[str]:
    """The literal ``exclude_patterns`` of a Sphinx ``conf.py``, read without running it."""
    for node in ast.parse(conf.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "exclude_patterns"
        ):
            return set(ast.literal_eval(node.value))
    raise ValueError(f"{conf} has no exclude_patterns list")


def guide_pages(package_dir=None) -> dict[str, Path]:
    """The published guide pages an installation of aquakin has.

    Parameters
    ----------
    package_dir : str or pathlib.Path, optional
        The ``aquakin/mcp`` folder of an installation (default: this one). A
        built package has the pages in its :data:`BUNDLE` subfolder; a source
        checkout has none, and its ``docs/`` folder (two levels up) is read.

    Returns
    -------
    dict
        ``{page name: file}``, sorted by name.

    Raises
    ------
    FileNotFoundError
        If the installation has neither, saying where to read the guide.
    """
    here = Path(__file__).resolve().parent if package_dir is None else Path(package_dir)
    bundled = here / BUNDLE
    if bundled.is_dir():
        return {p.stem: p for p in sorted(bundled.glob("*.md"))}
    checkout = here.parents[1] / "docs"
    if (checkout / "conf.py").is_file():
        return {p.stem: p for p in published_pages(checkout)}
    raise FileNotFoundError(
        "this installation of aquakin has no user guide pages; read them at "
        "https://aquakin.readthedocs.io"
    )


def _sections(text: str) -> list[tuple[int, str, list[str]]]:
    """A page split at its headings: ``(level, heading, lines)``, code fences respected.

    Text before the first heading is a section of level 0 with an empty heading.
    A ``#`` line inside a fenced code block (a Python comment) is not a heading.
    """
    out: list[tuple[int, str, list[str]]] = [(0, "", [])]
    fence = None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        h = _HEADING.match(line) if fence is None else None
        if h:
            out.append((len(h.group(1)), h.group(2), [line]))
        else:
            out[-1][2].append(line)
    return out


def list_pages(pages: dict[str, Path] | None = None) -> list[dict]:
    """Every page of the user guide with its title.

    Parameters
    ----------
    pages : dict, optional
        ``{page name: file}`` to read (default: :func:`guide_pages`).

    Returns
    -------
    list of dict
        ``{"page", "title"}`` per page, sorted by page name. ``page`` is the
        name :func:`read_page` takes (the file name without ``.md``).
    """
    out = []
    for name, path in (guide_pages() if pages is None else pages).items():
        heads = [h for level, h, _ in _sections(path.read_text(encoding="utf-8")) if level == 1]
        out.append({"page": name, "title": heads[0] if heads else name})
    return out


def read_page(page: str, section: str | None = None, pages: dict[str, Path] | None = None) -> str:
    """One page of the user guide as Markdown, or one section of it.

    Parameters
    ----------
    page : str
        The page name (as :func:`list_pages` gives it; a trailing ``.md`` is
        ignored).
    section : str, optional
        A heading of the page (case ignored). Only that section is returned,
        with its subsections, up to the next heading of the same or a higher
        level.
    pages : dict, optional
        ``{page name: file}`` to read (default: :func:`guide_pages`).

    Returns
    -------
    str

    Raises
    ------
    KeyError
        If there is no such page, or no such heading on it; the message lists
        what there is.
    """
    pages = guide_pages() if pages is None else pages
    name = page.removesuffix(".md")
    if name not in pages:
        raise KeyError(f"no guide page {page!r}; the pages are {sorted(pages)}")
    text = pages[name].read_text(encoding="utf-8")
    if section is None:
        return text
    parts = _sections(text)
    for i, (level, heading, lines) in enumerate(parts):
        if level and heading.lower() == section.strip().lower():
            body = list(lines)
            for deeper, _, more in parts[i + 1 :]:
                if deeper <= level:
                    break
                body += more
            return "\n".join(body).strip() + "\n"
    headings = [h for level, h, _ in parts if level]
    raise KeyError(f"no section {section!r} on page {name!r}; its headings are {headings}")


def search(query: str, limit: int = 10, pages: dict[str, Path] | None = None) -> list[dict]:
    """The sections of the user guide that best match some words.

    A section's score is the number of times the query's words occur in it
    (case ignored), with sections holding every word ranked above those that
    miss one.

    Parameters
    ----------
    query : str
        One or more words.
    limit : int, optional
        The most sections to return (default 10).
    pages : dict, optional
        ``{page name: file}`` to search (default: :func:`guide_pages`).

    Returns
    -------
    list of dict
        ``{"page", "section", "score", "snippet"}`` per match, best first.
        ``page`` and ``section`` are what :func:`read_page` takes; ``section``
        is ``None`` for the text above a page's first heading.

    Raises
    ------
    ValueError
        If ``query`` has no words, or ``limit`` is below 1.
    """
    words = re.findall(r"\w+", query.lower())
    if not words:
        raise ValueError("the query has no words to search for")
    if limit < 1:
        raise ValueError(f"limit must be at least 1, not {limit}")
    hits = []
    for name, path in (guide_pages() if pages is None else pages).items():
        for _, heading, lines in _sections(path.read_text(encoding="utf-8")):
            text = "\n".join(lines)
            low = text.lower()
            counts = [low.count(w) for w in words]
            if not any(counts):
                continue
            first = min(low.find(w) for w, c in zip(words, counts, strict=True) if c)
            snippet = " ".join(text[max(0, first - 120) : first + 180].split())
            hits.append((all(counts), sum(counts), name, heading, snippet))
    hits.sort(key=lambda h: (not h[0], -h[1], h[2], h[3]))
    return [
        {"page": name, "section": heading or None, "score": score, "snippet": snippet}
        for _, score, name, heading, snippet in hits[:limit]
    ]
