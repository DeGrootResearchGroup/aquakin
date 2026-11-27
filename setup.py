"""Build step that bundles the published user guide into the package.

All of the project's metadata is in ``pyproject.toml``; this file only adds
one step to building the package. ``aquakin mcp`` lets an AI assistant read
the user guide, so a wheel carries the published Markdown pages in
``aquakin/mcp/guide/``. The pages live in ``docs/`` (outside the package), and
which of them are published is decided by ``aquakin/mcp/docs.py``, loaded here
by path because aquakin itself cannot be imported while it is being built. A
source checkout has no bundle and reads ``docs/`` directly.
"""

import importlib.util
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent


def _guide_module():
    spec = importlib.util.spec_from_file_location(
        "_aquakin_guide", ROOT / "aquakin" / "mcp" / "docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildPyWithGuide(build_py):
    """``build_py``, then copy the published guide pages into the built package."""

    def run(self):
        super().run()
        guide = _guide_module()
        target = Path(self.build_lib) / "aquakin" / "mcp" / guide.BUNDLE
        target.mkdir(parents=True, exist_ok=True)
        for page in guide.published_pages(ROOT / "docs"):
            shutil.copyfile(page, target / page.name)


setup(cmdclass={"build_py": BuildPyWithGuide})
