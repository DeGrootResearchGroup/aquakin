"""YAML -> validated NetworkSpec -> CompiledNetwork."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from aquakin.core.network import CompiledNetwork, compile_network
from aquakin.schema.inheritance import (
    apply_remove,
    merge_spec,
    pop_inheritance_keys,
)
from aquakin.schema.network_spec import NetworkSpec


def _read_base(extends: str, base_dir, source: str):
    """Read a base network's YAML text for an ``extends:`` directive.

    ``extends`` is a shipped network *name* (resolved via the package
    resources), unless it contains a path separator or ends in ``.yaml``, in
    which case it is a path relative to the extending file's directory.
    Returns ``(text, base_dir_for_that_file)``.
    """
    if "/" in extends or "\\" in extends or extends.endswith(".yaml"):
        p = Path(extends)
        if not p.is_absolute() and base_dir is not None:
            p = Path(base_dir) / p
        if not p.is_file():
            raise FileNotFoundError(f"{source}: extends: base file not found: {p}")
        return p.read_text(encoding="utf-8"), p.parent
    resource = files("aquakin.networks") / f"{extends}.yaml"
    if not resource.is_file():
        available = sorted(
            q.name.removesuffix(".yaml")
            for q in files("aquakin.networks").iterdir()
            if q.name.endswith(".yaml")
        )
        raise FileNotFoundError(
            f"{source}: extends: base network '{extends}' not found. Available: {available}"
        )
    return resource.read_text(encoding="utf-8"), None


def _resolve_inheritance(data: dict, source: str, base_dir, seen: tuple) -> dict:
    """Resolve ``network.extends`` by merging onto the (recursively resolved)
    base mapping; apply ``remove:``. Returns a plain network mapping ready for
    schema validation. A no-``extends`` mapping is returned unchanged."""
    extends, remove = pop_inheritance_keys(data, source)
    if extends is None:
        if remove is not None:
            raise ValueError(
                f"{source}: 'remove:' has no effect without 'extends:' "
                f"(there is no base network to remove pieces from)."
            )
        return data
    if extends in seen:
        raise ValueError(f"{source}: cyclic 'extends' chain: {' -> '.join(seen + (extends,))}.")
    base_text, base_base_dir = _read_base(extends, base_dir, source)
    try:
        base_data = yaml.safe_load(base_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{source}: failed to parse base '{extends}': {exc}") from exc
    if not isinstance(base_data, dict):
        raise ValueError(f"{source}: base network '{extends}' top-level YAML must be a mapping.")
    base_data = _resolve_inheritance(
        base_data, f"base network '{extends}'", base_base_dir, seen + (extends,)
    )
    merged = merge_spec(base_data, data, source=source)
    if remove is not None:
        merged = apply_remove(merged, remove, source)
    return merged


def _yaml_to_spec(text: str, source: str, *, base_dir=None) -> NetworkSpec:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML from {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {source} must be a mapping, got {type(data).__name__}")
    data = _resolve_inheritance(data, source, base_dir, seen=())
    try:
        return NetworkSpec.model_validate(data)
    except ValidationError as exc:
        # Only schema-validation failures mean "bad network spec". Other
        # exceptions (e.g. RecursionError) are genuine bugs and must propagate
        # rather than be relabelled as invalid user input.
        raise ValueError(f"Invalid network specification in {source}: {exc}") from exc


def _apply_activity_override(spec: NetworkSpec, activity_model: str) -> NetworkSpec:
    """Override the speciation ``activity_model`` on a loaded spec.

    Lets a caller enable an ionic-strength activity correction on a shipped
    network whose YAML ships ``activity_model: none`` (e.g. ``adm1``), without
    editing the YAML. Raises if the network has no ``speciation:`` block (there
    is no pH solver to correct) or the value is invalid.
    """
    from aquakin.core.ph_solver import _ACTIVITY_MODELS

    if activity_model not in _ACTIVITY_MODELS:
        raise ValueError(
            f"activity_model must be one of {_ACTIVITY_MODELS}; got {activity_model!r}"
        )
    if spec.speciation is None:
        raise ValueError(
            "activity_model override requires the network to declare a "
            "speciation: block (a state-derived pH); this network has none."
        )
    spec.speciation = spec.speciation.model_copy(update={"activity_model": activity_model})
    return spec


# Built-in networks are deterministic for a given name and are treated as
# immutable (nothing mutates a CompiledNetwork in place; temperature
# re-referencing etc. go through ``dataclasses.replace``, which copies). Caching
# the compiled object by name therefore (a) skips re-parsing + re-compiling the
# YAML on every call, and (b) -- the bigger win -- returns the *same* object each
# time, so a stable ``id(network)`` lets the cross-instance compiled-solver cache
# (see ``integrate/_common.py``) share compiled solves across every reactor and
# test that loads the same network. Call ``clear_network_cache()`` to reset.
_NETWORK_CACHE: dict[str, CompiledNetwork] = {}


def clear_network_cache() -> None:
    """Clear the built-in-network cache (see :func:`load_network`)."""
    _NETWORK_CACHE.clear()


def load_network(name: str, *, activity_model: Union[str, None] = None) -> CompiledNetwork:
    """
    Load a built-in network shipped with ``aquakin``.

    The compiled network is cached by name and reused on subsequent calls (a
    ``CompiledNetwork`` is immutable in use). Use
    :func:`clear_network_cache` to reset the cache.

    Parameters
    ----------
    name : str
        Network name (e.g. ``"ozone_bromate"``).
    activity_model : str, optional
        Override the speciation ``activity_model`` (ionic-strength activity
        correction in the pH solver) -- e.g. ``"davies"`` to run ``adm1`` with
        activity coefficients instead of the shipped ``"none"`` (which uses molar
        concentrations directly, the ADM1/BSM2 convention). Requires the network
        to declare a ``speciation:`` block. When given, the cache is bypassed (so
        the default and the overridden network coexist); ``None`` (default)
        returns the cached default unchanged.

    Returns
    -------
    CompiledNetwork
    """
    if activity_model is None:
        cached = _NETWORK_CACHE.get(name)
        if cached is not None:
            return cached
    resource = files("aquakin.networks") / f"{name}.yaml"
    if not resource.is_file():
        available = sorted(
            p.name.removesuffix(".yaml")
            for p in files("aquakin.networks").iterdir()
            if p.name.endswith(".yaml")
        )
        raise FileNotFoundError(f"Built-in network '{name}' not found. Available: {available}")
    text = resource.read_text(encoding="utf-8")
    spec = _yaml_to_spec(text, f"built-in network '{name}'")
    if activity_model is not None:
        spec = _apply_activity_override(spec, activity_model)
        return compile_network(spec)
    net = compile_network(spec)
    _NETWORK_CACHE[name] = net
    return net


def load_network_from_file(
    path: Union[str, Path], *, activity_model: Union[str, None] = None
) -> CompiledNetwork:
    """
    Load a network from a YAML file on disk.

    Parameters
    ----------
    path : str or Path
        Path to a YAML network file.
    activity_model : str, optional
        Override the speciation ``activity_model`` (see :func:`load_network`).

    Returns
    -------
    CompiledNetwork
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Network file not found: {p}")
    text = p.read_text(encoding="utf-8")
    spec = _yaml_to_spec(text, str(p), base_dir=p.parent)
    if activity_model is not None:
        spec = _apply_activity_override(spec, activity_model)
    return compile_network(spec)
