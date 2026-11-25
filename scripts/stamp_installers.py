"""Write copies of the editor installers that install one release.

``installers/install-aquakin.sh`` and ``installers/install-aquakin.bat`` each
hold an empty ``AQUAKIN_VERSION``, which installs the newest release on PyPI.
The release workflow runs this script to fill in the version being released and
attaches the results to the GitHub release, so a colleague who downloads an
installer gets the release it came with, not whatever is newest later.

Usage::

    python scripts/stamp_installers.py 0.3.0 dist-installers/
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Each installer and the line ending its shell needs: cmd.exe misreads a batch
#: file with bare LF endings, and sh reads a CR as part of each line.
INSTALLERS = {"install-aquakin.sh": "\n", "install-aquakin.bat": "\r\n"}
#: The version line of either installer: ``AQUAKIN_VERSION=""`` (sh) or
#: ``set "AQUAKIN_VERSION="`` (cmd), with the value captured empty.
VERSION_LINE = re.compile(r'^(set "AQUAKIN_VERSION=|AQUAKIN_VERSION=")(")', re.M)
#: A release version as ``aquakin.__version__`` spells it.
VERSION = re.compile(r"\d+(\.\d+)*([a-z]+\d+)?(\.(post|dev)\d+)?")


def stamp(text: str, version: str) -> str:
    """Return installer ``text`` with ``version`` filled in.

    Parameters
    ----------
    text : str
        An installer's text, holding exactly one empty version line.
    version : str
        The release version, e.g. ``"0.3.0"``.

    Returns
    -------
    str
        The text with the version line set to ``version``.

    Raises
    ------
    ValueError
        If ``version`` is not a release version, or ``text`` does not hold
        exactly one empty version line.
    """
    if not VERSION.fullmatch(version):
        raise ValueError(f"{version!r} is not a release version")
    stamped, count = VERSION_LINE.subn(lambda m: m.group(1) + version + m.group(2), text)
    if count != 1:
        raise ValueError(f"expected one empty AQUAKIN_VERSION line, found {count}")
    return stamped


def main(argv: list[str]) -> int:
    """Stamp every installer with ``argv[0]`` and write it under ``argv[1]``."""
    if len(argv) != 2:
        print(__doc__.split("Usage::")[1].strip(), file=sys.stderr)
        return 2
    version, out = argv[0], Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, newline in INSTALLERS.items():
        source = ROOT / "installers" / name
        # Whatever endings the checkout gave it, write the ones its shell needs.
        lines = stamp(source.read_text(encoding="utf-8"), version).splitlines()
        target = out / name
        target.write_bytes("".join(line + newline for line in lines).encode("utf-8"))
        target.chmod(source.stat().st_mode)
        print(f"{target}: aquakin {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
