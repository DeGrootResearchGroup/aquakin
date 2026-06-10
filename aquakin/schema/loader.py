"""YAML -> validated NetworkSpec -> CompiledNetwork."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from aquakin.core.network import CompiledNetwork, compile_network
from aquakin.schema.network_spec import NetworkSpec


def _yaml_to_spec(text: str, source: str) -> NetworkSpec:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML from {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {source} must be a mapping, got {type(data).__name__}")
    try:
        return NetworkSpec.model_validate(data)
    except ValidationError as exc:
        # Only schema-validation failures mean "bad network spec". Other
        # exceptions (e.g. RecursionError) are genuine bugs and must propagate
        # rather than be relabelled as invalid user input.
        raise ValueError(f"Invalid network specification in {source}: {exc}") from exc


def load_network(name: str) -> CompiledNetwork:
    """
    Load a built-in network shipped with ``aquakin``.

    Parameters
    ----------
    name : str
        Network name (e.g. ``"ozone_bromate"``).

    Returns
    -------
    CompiledNetwork
    """
    resource = files("aquakin.networks") / f"{name}.yaml"
    if not resource.is_file():
        available = sorted(
            p.name.removesuffix(".yaml")
            for p in files("aquakin.networks").iterdir()
            if p.name.endswith(".yaml")
        )
        raise FileNotFoundError(
            f"Built-in network '{name}' not found. Available: {available}"
        )
    text = resource.read_text(encoding="utf-8")
    spec = _yaml_to_spec(text, f"built-in network '{name}'")
    return compile_network(spec)


def load_network_from_file(path: Union[str, Path]) -> CompiledNetwork:
    """
    Load a network from a YAML file on disk.

    Parameters
    ----------
    path : str or Path
        Path to a YAML network file.

    Returns
    -------
    CompiledNetwork
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Network file not found: {p}")
    text = p.read_text(encoding="utf-8")
    spec = _yaml_to_spec(text, str(p))
    return compile_network(spec)
