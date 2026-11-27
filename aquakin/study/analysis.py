"""What a study's analyses share, whatever they study."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from aquakin.study.study import _Study


def _fit_sigma(study: _Study):
    """The fit's ``sigma``: ``"estimate"``, one error for every observation, or an
    array of one per channel (broadcast over the times; a list would mean one
    per dataset)."""
    import numpy as np

    if study.estimate_sigma:
        return "estimate"
    if isinstance(study.sigma, list):
        return np.asarray(study.sigma, dtype=float)
    return study.sigma


def _flattened(ref, name: str, base_dir: Path) -> str:
    """A model's YAML with its whole ``extends`` chain merged in."""
    from importlib.resources import files

    from aquakin.schema.loader import _resolve_inheritance, parse_model_yaml

    if ref["source"] == "shipped":
        text = (files("aquakin.models") / f"{name}.yaml").read_text()
        base = None
    else:
        path = Path(ref["source"])
        path = path if path.is_absolute() else base_dir / path
        text, base = path.read_text(), path.parent
    data = _resolve_inheritance(parse_model_yaml(text), name, base, seen=())
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
