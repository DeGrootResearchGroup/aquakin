"""Sphinx configuration for the aquakin documentation.

The docs are Markdown (MyST). The API reference is generated from
the ``__all__`` of ``aquakin``, ``aquakin.plant`` and ``aquakin.study`` at build
time (see ``_write_api_page`` below), so it stays in lock-step with the curated
public surface without a hand-maintained list.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from importlib.metadata import version as _dist_version
from pathlib import Path

from sphinx.errors import PycodeError
from sphinx.pycode import ModuleAnalyzer
from sphinx.util.docstrings import prepare_docstring

# -- Project information ------------------------------------------------------

project = "aquakin"
author = "Christopher DeGroot"
copyright = "2026, Christopher DeGroot"

try:
    release = _dist_version("aquakin")
except Exception:  # pragma: no cover - source checkout without an install
    release = "0.2.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

# Every published page is reachable from a toctree; the dev-internal logs
# (CI internals, the perf log, the reproduction log, the solver spec, the
# annotated file tree, the project reference) are repo references, not user
# docs, so they are kept out of the built site.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "ci.md",
    "package_structure.md",
    "plant_performance.md",
    "khalil_reproduction_log.md",
    "spec_forward_sensitivity_solver.md",
    "project_reference.md",
]

# -- MyST (Markdown) ----------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # ::: fenced directives
    "deflist",
    "dollarmath",  # $...$ / $$...$$ math
    "fieldlist",
    "linkify",  # bare URLs -> links
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 3  # auto-slug headings h1-h3 so cross-page #anchors resolve

# -- autodoc / autosummary ----------------------------------------------------

# Only run *intentional* doctests -- explicit ``.. doctest::`` / ``.. testcode::``
# directives -- not every illustrative ``>>>`` block. Many docstring "Examples"
# sections are illustrative (they reference names built earlier in prose, e.g.
# ``reactor = BatchReactor(model, conditions)``) and are not meant to execute;
# testing them wholesale would fail the gate on documentation that is correct as
# documentation. New executable examples opt in with a ``.. testcode::`` block.
doctest_test_doctest_blocks = ""

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
# Render a docstring `Attributes` section as inline ``:ivar:`` fields rather than
# separate attribute directives -- otherwise a NamedTuple / property documented
# in both the `Attributes` section and by autodoc's member scan collides
# ("duplicate object description").
napoleon_use_ivar = True

# -- intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "diffrax": ("https://docs.kidger.site/diffrax/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}
# Missing cross-refs into third-party APIs should not fail the -W build.
intersphinx_disabled_reftypes = ["*"]
# A sandbox without network access (a Claude Code cloud session) cannot fetch
# the inventories, and each failed fetch is a warning that -W turns into an
# error. AQUAKIN_DOCS_OFFLINE=1 skips them; with the reftypes above disabled,
# no page depends on them to build.
if os.environ.get("AQUAKIN_DOCS_OFFLINE") == "1":
    intersphinx_mapping = {}

# -- HTML output --------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"aquakin {release}"
html_theme_options = {
    "github_url": "https://github.com/DeGrootResearchGroup/aquakin",
    "navigation_with_keys": True,
    "show_prev_next": True,
}


# -- Auto-generated API reference ---------------------------------------------


# The packages the reference covers, in page order, each with its introduction.
# A name a later package shares with an earlier one is documented only under
# the earlier (shorter) one.
_API_PACKAGES = {
    "aquakin": (
        "The top-level API: models, reactors, conditions, the plant assembly "
        "and its common units, calibration and sensitivity."
    ),
    "aquakin.plant": (
        "The rest of the plant-wide API: the unit operations not exported from "
        "`aquakin`, flowsheet files, influents, steady state, plant calibration "
        "and measured channels."
    ),
    "aquakin.study": "Studies: a folder holding a flowsheet, its runs and their results.",
}


def _documented_name(obj):
    """The qualified name the reference documents ``obj`` under, or ``None``.

    That is the first covered package exporting ``obj`` under its own
    ``__name__``.
    """
    own = getattr(obj, "__name__", None)
    for package in _API_PACKAGES:
        module = importlib.import_module(package)
        if own in module.__all__ and getattr(module, own) is obj:
            return f"{package}.{own}"
    return None


def _split_by_kind(module, names):
    """Sort ``names`` exported by ``module`` by the kind of object each names.

    Returns the classes, the functions, the remaining symbols (warnings,
    enums, constants), and the aliases: ``(name, target)`` pairs for a class or
    function exported under a second name, whose page is its own name's.
    """
    classes, functions, other, aliases = [], [], [], []
    for name in sorted(names):
        obj = getattr(module, name)
        if inspect.isclass(obj) or inspect.isroutine(obj):
            target = _documented_name(obj)
            if obj.__name__ != name and target is not None:
                aliases.append((name, target))
            elif inspect.isclass(obj):
                classes.append(name)
            else:
                functions.append(name)
        else:
            other.append(name)
    return classes, functions, other, aliases


def _autosummary_table(package, heading, entries):
    """An ``autosummary`` table of ``package``'s ``entries`` under ``heading``."""
    if not entries:
        return []
    body = "\n".join(f"   {package}.{n}" for n in entries)
    return [
        f"### {heading}",
        "",
        "```{eval-rst}",
        ".. autosummary::",
        "   :toctree: generated",
        "   :nosignatures:",
        "",
        body,
        "```",
        "",
    ]


def _alias_entries(package, aliases):
    """A described object for each alias, linking the name it stands for."""
    if not aliases:
        return []
    lines = ["### Aliases", "", "```{eval-rst}"]
    for name, target in aliases:
        is_class = inspect.isclass(_resolve(target))
        directive, role = ("class", "class") if is_class else ("function", "func")
        lines += [
            f".. py:{directive}:: {name}",
            f"   :module: {package}",
            "",
            f"   Another name for :{role}:`{target}`, documented there.",
            "",
        ]
    return [*lines, "```", ""]


def _resolve(qualified):
    """The object a qualified ``package.name`` refers to."""
    package, _, name = qualified.rpartition(".")
    return getattr(importlib.import_module(package), name)


def _package_section(package, names):
    """The Markdown for one package: a heading, its introduction, its tables."""
    module = importlib.import_module(package)
    classes, functions, other, aliases = _split_by_kind(module, names)
    return [
        f"## {package}",
        "",
        _API_PACKAGES[package],
        "",
        *_autosummary_table(package, "Classes", classes),
        *_autosummary_table(package, "Functions", functions),
        *_autosummary_table(package, "Warnings & other", other),
        *_alias_entries(package, aliases),
    ]


def _write_api_page(app):
    """Generate ``api.md`` from the packages' ``__all__`` before the read phase.

    The page has one section per package in ``_API_PACKAGES``, each listing
    the names of that package's ``__all__`` no earlier package exports, so
    each object is documented once, under the shortest name it is importable
    by. Each section splits its names into classes, functions, the remaining
    exported symbols and the aliases (see ``_split_by_kind``);
    ``autosummary_generate`` then emits a stub page per object under
    ``generated/``.
    """
    seen = set()
    sections = []
    for package in _API_PACKAGES:
        exported = importlib.import_module(package).__all__
        sections += _package_section(package, [n for n in exported if n not in seen])
        seen.update(exported)
    lines = [
        "# API reference",
        "",
        "The full signature and docstring of every public name, generated from "
        "the packages' `__all__`. A name `aquakin` re-exports is listed once, "
        "under `aquakin`.",
        "",
        *sections,
    ]
    (Path(app.srcdir) / "api.md").write_text("\n".join(lines) + "\n")


def _trimmed(lines):
    """``lines`` without its trailing blank lines."""
    end = len(lines)
    while end and not lines[end - 1].strip():
        end -= 1
    return lines[:end]


def _constant_comment(app, what, name, obj, options, lines):
    """Document a re-exported constant by its ``#:`` comment where it is defined.

    autodoc reads a module-level constant's ``#:`` doc-comment only in the
    module the constant is documented under, which for a name a package
    re-exports is the package's ``__init__``, not the module defining it. A
    value without a docstring of its own (a ``dict``, a ``float``) then shows
    its type's docstring, which describes the type rather than the constant.
    This replaces that docstring with the defining module's doc-comment, or
    with nothing when there is none.
    """
    doc = type(obj).__doc__
    if what != "data" or not doc or _trimmed(lines) != _trimmed(prepare_docstring(doc)):
        return
    package, _, attr = name.rpartition(".")
    lines[:] = []
    for modname, module in sorted(sys.modules.items()):
        if not modname.startswith(f"{package}.") or getattr(module, attr, None) is not obj:
            continue
        try:
            analyzer = ModuleAnalyzer.for_module(modname)
            analyzer.analyze()
        except PycodeError:
            continue
        comment = analyzer.attr_docs.get(("", attr))
        if comment:
            lines[:] = list(comment)
            return


def setup(app):
    app.connect("config-inited", lambda app, config: _write_api_page(app))
    app.connect("autodoc-process-docstring", _constant_comment)
