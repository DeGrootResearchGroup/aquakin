"""Sphinx configuration for the aquakin documentation.

The docs are Markdown (MyST). The API reference is generated from
``aquakin.__all__`` at build time (see ``_write_api_page`` below), so it stays
in lock-step with the curated public surface without a hand-maintained list.
"""

from __future__ import annotations

import inspect
from importlib.metadata import version as _dist_version
from pathlib import Path

# -- Project information ------------------------------------------------------

project = "aquakin"
author = "Christopher DeGroot"
copyright = "2026, Christopher DeGroot"

try:
    release = _dist_version("aquakin")
except Exception:  # pragma: no cover - source checkout without an install
    release = "0.1.0"
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
# annotated file tree) are repo references, not user docs, so they are kept
# out of the built site.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "ci.md",
    "package_structure.md",
    "plant_performance.md",
    "khalil_reproduction_log.md",
    "spec_forward_sensitivity_solver.md",
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

# -- HTML output --------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"aquakin {release}"
html_theme_options = {
    "github_url": "https://github.com/DeGrootResearchGroup/aquakin",
    "navigation_with_keys": True,
    "show_prev_next": True,
}


# -- Auto-generated API reference ---------------------------------------------


def _write_api_page(app):
    """Generate ``api.md`` from ``aquakin.__all__`` before the read phase.

    Classes, functions, and the remaining exported symbols (warnings, enums)
    are split into three ``autosummary`` tables; ``autosummary_generate`` then
    emits a stub page per object under ``generated/``.
    """
    import aquakin

    classes, functions, other = [], [], []
    for name in sorted(aquakin.__all__):
        obj = getattr(aquakin, name)
        if inspect.isclass(obj):
            classes.append(name)
        elif inspect.isroutine(obj):
            functions.append(name)
        else:
            other.append(name)

    def table(heading, names):
        if not names:
            return []
        body = "\n".join(f"   aquakin.{n}" for n in names)
        return [
            f"## {heading}",
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

    lines = [
        "# API reference",
        "",
        "The complete curated top-level public API, generated from "
        "`aquakin.__all__`. Each domain also exposes its full surface as a "
        "subpackage (`aquakin.plant`, `aquakin.integrate`, `aquakin.utils`).",
        "",
        *table("Classes", classes),
        *table("Functions", functions),
        *table("Warnings & other", other),
    ]
    (Path(app.srcdir) / "api.md").write_text("\n".join(lines) + "\n")


def setup(app):
    app.connect("config-inited", lambda app, config: _write_api_page(app))
