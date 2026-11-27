"""Run the code examples in the README and the user guide as tests.

Sybil collects every fenced ``python`` block of the pages below and runs each
page's blocks in order in one namespace, so a page is tested exactly as a reader
follows it: a block that raises fails its test. A page states a value it prints
with a hidden check block, which runs but does not render (in the README an HTML
comment, ``<!-- invisible-code-block: python -->``; in a MyST page a ``%``
comment, ``% invisible-code-block: python``). A block too slow to test is
skipped with a reason (``% skip: next "reason"``).

A ``yaml`` block whose first line is a comment naming a file (``# decay.yaml``)
is written to that file, so the Python blocks after it can load it; every other
``yaml`` block is parsed, which catches a fragment that is not valid YAML.

Each page runs in its own temporary directory, so a page that writes a file
(``to_csv``, ``to_spec``) leaves nothing in the repository. Pages whose solves
take minutes are marked ``slow`` and run with the slow suite, not the PR gate.
A page's blocks share one namespace, so under pytest-xdist run them with
``--dist loadfile``, which keeps a page on one worker.
"""

import os
import re
from pathlib import Path

import pytest
import yaml
from sybil import Sybil
from sybil.parsers import markdown, myst

# The published user guide, which docs/conf.py builds; the developer logs in
# docs/ are not user-facing and are not collected.
GUIDE_PAGES = [
    "docs/index.md",
    "docs/getting_started.md",
    "docs/models.md",
    "docs/simulation.md",
    "docs/analysis.md",
    "docs/reference.md",
    "docs/installing_the_editor.md",
    "docs/ai_assistants.md",
    "docs/reactors.md",
    "docs/plants.md",
    "docs/custom_flowsheets.md",
    "docs/influents.md",
    "docs/plant_files.md",
    "docs/analysis_concepts.md",
    "docs/sensitivity_and_calibration.md",
    "docs/calibration.md",
    "docs/fit_uncertainty.md",
    "docs/experiment_design.md",
    "docs/plant_calibration.md",
    "docs/worked_example_design.md",
    "docs/model_format.md",
    "docs/adding_models.md",
    "docs/model_catalog.md",
    "docs/public_api.md",
]

# Pages whose examples take minutes (whole-plant solves, multistart fits).
SLOW_PAGES = {
    "plants.md",
    "custom_flowsheets.md",
    "sensitivity_and_calibration.md",
    "calibration.md",
    "fit_uncertainty.md",
    "experiment_design.md",
}

_NAMED_FILE = re.compile(r"\A#\s*(?P<name>[\w.-]+\.ya?ml)\s*\n")


def _evaluate_yaml(example):
    """Write a ``# <name>.yaml`` block to that file; otherwise check it parses."""
    source = str(example.parsed)
    named = _NAMED_FILE.match(source)
    if named:
        Path(named["name"]).write_text(source[named.end() :])
    else:
        yaml.safe_load(source)


@pytest.fixture(scope="module")
def _page_directory(tmp_path_factory):
    """Run one page in a fresh directory, restoring the working directory after."""
    start = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("page"))
    yield
    os.chdir(start)


_readme = Sybil(
    parsers=[
        markdown.PythonCodeBlockParser(),
        markdown.CodeBlockParser("yaml", _evaluate_yaml),
        markdown.SkipParser(),
    ],
    filenames=["README.md"],
    fixtures=["_page_directory"],
    name="readme",
)

_guide = Sybil(
    parsers=[
        myst.PythonCodeBlockParser(),
        myst.CodeBlockParser("yaml", _evaluate_yaml),
        myst.SkipParser(),
    ],
    patterns=GUIDE_PAGES,
    fixtures=["_page_directory"],
    name="guide",
)

pytest_collect_file = (_readme + _guide).pytest()


def pytest_collection_modifyitems(items):
    """Mark the examples of the slow pages ``slow``."""
    for item in items:
        if item.path.name in SLOW_PAGES and item.path.parent.name == "docs":
            item.add_marker(pytest.mark.slow)
