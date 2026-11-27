"""Stage the Claude Desktop extension (an MCP Bundle) of one aquakin release.

``packaging/claude-desktop/`` holds the extension's template: its
``manifest.json``, the ``pyproject.toml`` Claude Desktop builds the server's
environment from (the MCPB ``uv`` server type), and the entry point. This
script copies it to a folder and stamps it with a release: the manifest's and
the project's version, the ``aquakin[mcp]==<version>`` requirement, and the
manifest's tool list, read from the server itself. The release workflow then
packs the folder with ``mcpb pack`` and attaches ``aquakin.mcpb`` to the GitHub
release.

Usage::

    python scripts/build_mcpb.py 0.4.0 build/mcpb
    npx @anthropic-ai/mcpb pack build/mcpb dist/aquakin.mcpb

``--requirement`` replaces the aquakin requirement (to try a bundle against a
local checkout before a release: ``--requirement "aquakin[mcp] @ file:///path"``).
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "packaging" / "claude-desktop"
#: A release version as ``aquakin.__version__`` spells it.
VERSION = re.compile(r"\d+(\.\d+)*([a-z]+\d+)?(\.(post|dev)\d+)?")


def _tools() -> list[dict]:
    """The server's tools as the manifest lists them: each name and summary."""
    from aquakin.mcp.scripts import run_python
    from aquakin.mcp.server import TOOLS

    out = []
    for fn in [fn for fn, _ in TOOLS] + [run_python]:
        summary = " ".join(inspect.getdoc(fn).split("\n\n")[0].split())
        out.append({"name": fn.__name__, "description": summary})
    return out


def stage(version: str, out: Path, requirement: str | None = None) -> Path:
    """Write the extension's files for ``version`` into ``out``.

    Parameters
    ----------
    version : str
        The release version, e.g. ``"0.4.0"``.
    out : pathlib.Path
        The folder to write; it must not exist.
    requirement : str, optional
        The aquakin requirement to install (default ``aquakin[mcp]==<version>``).

    Returns
    -------
    pathlib.Path
        ``out``.

    Raises
    ------
    ValueError
        If ``version`` is not a release version.
    FileExistsError
        If ``out`` exists.
    """
    if not VERSION.fullmatch(version):
        raise ValueError(f"{version!r} is not a release version")
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    shutil.copytree(TEMPLATE, out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["version"] = version
    # run_python is listed too, though only offered when the setting is on.
    manifest["tools"] = _tools()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    project = (out / "pyproject.toml").read_text(encoding="utf-8")
    project = project.replace('version = "0.0.0"', f'version = "{version}"', 1)
    pin = requirement or f"aquakin[mcp]=={version}"
    project = project.replace(
        'dependencies = ["aquakin[mcp]"]', f"dependencies = [{json.dumps(pin)}]", 1
    )
    (out / "pyproject.toml").write_text(project, encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", help="the release version, e.g. 0.4.0")
    parser.add_argument("out", type=Path, help="the folder to write (must not exist)")
    parser.add_argument("--requirement", help="the aquakin requirement to install instead")
    args = parser.parse_args(argv)
    stage(args.version, args.out, args.requirement)
    print(f"staged the Claude Desktop extension {args.version} in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
