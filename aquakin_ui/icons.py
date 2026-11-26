"""Icons and colours: the interface's only per-kind presentation table.

Everything else about a kind comes from aquakin's unit catalogue. A kind with
no icon here gets the generic tank outline, so a new unit kind shows up in the
palette before anyone draws it an icon.
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).with_name("icons")

#: The icon file (without ``.svg``) for each unit kind.
ICONS = {
    "influent": "influent",
    "cstr": "cstr",
    "mbr": "mbr",
    "ifas": "ifas",
    "sbr": "sbr",
    "digester": "digester",
    "clarifier": "clarifier-secondary",
    "primary_clarifier": "clarifier-primary",
    "ideal_clarifier": "clarifier-secondary",
    "thickener": "thickener",
    "mixer": "mixer",
    "splitter": "splitter",
    "setpoint_splitter": "splitter-setpoint",
    "threshold_splitter": "splitter-threshold",
    "delay": "delay",
    "storage": "storage",
    "dosing": "dosing",
    "uv": "uv",
    "chlorine_contact": "chlorine-contact",
    "pi_controller": "controller",
    "expression_controller": "controller",
}

#: The accent colour for each catalogue category.
COLOURS = {
    "boundary": "#1a7f37",
    "reactor": "#0969da",
    "separator": "#9a6700",
    "flow": "#8250df",
    "treatment": "#1b7c83",
    "control": "#bf3989",
}

#: The palette heading for each catalogue category.
CATEGORY_LABELS = {
    "boundary": "Boundaries",
    "reactor": "Reactors",
    "separator": "Separators",
    "flow": "Flow handling",
    "treatment": "Treatment",
    "control": "Control",
}

SVG = {p.stem: p.read_text() for p in _DIR.glob("*.svg")}


def icon_name(kind: str, settings: dict) -> str:
    """The icon for a unit: its kind's, or the aerated tank for an aerated CSTR."""
    if kind == "cstr" and settings.get("aeration"):
        return "cstr-aerated"
    return ICONS.get(kind, "cstr")
