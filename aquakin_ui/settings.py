"""The editor's preferences, kept for the user between sessions.

A small JSON file in the user's configuration folder (``AQUAKIN_UI_CONFIG``
names another file): ``%APPDATA%\\aquakin\\ui.json`` on Windows,
``~/Library/Application Support/aquakin/ui.json`` on macOS, and
``$XDG_CONFIG_HOME/aquakin/ui.json`` (``~/.config``) elsewhere. A missing or
unreadable file gives the defaults; a preference that cannot be written is
kept for the session only.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

__all__ = ["DEFAULTS", "Settings", "settings_path"]

#: Every preference and its value until the user changes it.
DEFAULTS = {
    # Ask before a run replaces a study's results (with "Copy first" offered).
    "ask_before_replacing": True,
}


def settings_path() -> Path:
    """Where the preferences are kept."""
    given = os.environ.get("AQUAKIN_UI_CONFIG")
    if given:
        return Path(given)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "aquakin" / "ui.json"


class Settings:
    """The user's preferences.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        The file (default :func:`settings_path`).
    """

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else settings_path()
        self._values = dict(DEFAULTS)
        try:
            stored = json.loads(self.path.read_text())
        except (OSError, ValueError):
            stored = {}
        if isinstance(stored, dict):
            for key, value in stored.items():
                if key in DEFAULTS and isinstance(value, type(DEFAULTS[key])):
                    self._values[key] = value

    def __getitem__(self, key: str):
        return self._values[key]

    def set(self, key: str, value) -> bool:
        """Change a preference and keep it; ``False`` when it could not be written
        (it still holds for this session)."""
        if key not in DEFAULTS:
            raise KeyError(f"no preference {key!r}; known: {sorted(DEFAULTS)}")
        self._values[key] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as handle:
                json.dump(self._values, handle, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            return False
        return True
