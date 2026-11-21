"""Check that a built wheel carries the editor's widget bundle and icons.

``aquakin_ui`` serves the canvas from ``aquakin_ui/widget/serve/`` and draws
units with ``aquakin_ui/icons/*.svg``; both are package data, so a packaging
change can leave them out of a wheel without any test noticing (the tests run
from the source tree). This compares the wheel against the source tree: every
committed bundle file and icon must be in it, byte for byte.

Usage::

    python scripts/check_ui_package.py dist/aquakin-*.whl
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: The files the editor needs at run time, as globs under the repository.
REQUIRED = ("aquakin_ui/widget/serve/*", "aquakin_ui/icons/*.svg")
#: The bundle's two files, named here so an empty ``serve/`` cannot pass.
BUNDLE = ("aquakin-flowsheet.umd.js", "aquakin-flowsheet.css")


def missing_from(wheel: Path) -> list[str]:
    """The required files that ``wheel`` lacks or holds with other bytes."""
    expected = [p for pattern in REQUIRED for p in sorted(ROOT.glob(pattern))]
    problems = [
        f"aquakin_ui/widget/serve/{name}: not in the source tree"
        for name in BUNDLE
        if not (ROOT / "aquakin_ui" / "widget" / "serve" / name).is_file()
    ]
    if not any(p.parent.name == "icons" for p in expected):
        problems.append("aquakin_ui/icons: no icons in the source tree")
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        for path in expected:
            member = path.relative_to(ROOT).as_posix()
            if member not in names:
                problems.append(f"{member}: missing")
            elif zf.read(member) != path.read_bytes():
                problems.append(f"{member}: differs from the source tree")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    problems = missing_from(Path(argv[0]))
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"{argv[0]}: the widget bundle and every icon are packaged")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
