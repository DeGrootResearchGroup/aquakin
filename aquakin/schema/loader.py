"""YAML -> validated ModelSpec -> CompiledModel."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from aquakin.core.model import CompiledModel, compile_model
from aquakin.schema.inheritance import (
    apply_remove,
    merge_spec,
    pop_inheritance_keys,
)
from aquakin.schema.model_spec import ModelSpec


def _read_base(extends: str, base_dir, source: str):
    """Read a base model's YAML text for an ``extends:`` directive.

    ``extends`` is a shipped model *name* (resolved via the package
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
    resource = files("aquakin.models") / f"{extends}.yaml"
    if not resource.is_file():
        available = sorted(
            q.name.removesuffix(".yaml")
            for q in files("aquakin.models").iterdir()
            if q.name.endswith(".yaml")
        )
        raise FileNotFoundError(
            f"{source}: extends: base model '{extends}' not found. Available: {available}"
        )
    return resource.read_text(encoding="utf-8"), None


def _resolve_inheritance(data: dict, source: str, base_dir, seen: tuple) -> dict:
    """Resolve ``model.extends`` by merging onto the (recursively resolved)
    base mapping; apply ``remove:``. Returns a plain model mapping ready for
    schema validation. A no-``extends`` mapping is returned unchanged."""
    extends, remove = pop_inheritance_keys(data, source)
    if extends is None:
        if remove is not None:
            raise ValueError(
                f"{source}: 'remove:' has no effect without 'extends:' "
                f"(there is no base model to remove pieces from)."
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
        raise ValueError(f"{source}: base model '{extends}' top-level YAML must be a mapping.")
    base_data = _resolve_inheritance(
        base_data, f"base model '{extends}'", base_base_dir, seen + (extends,)
    )
    merged = merge_spec(base_data, data, source=source)
    if remove is not None:
        merged = apply_remove(merged, remove, source)
    return merged


def _yaml_to_spec(text: str, source: str, *, base_dir=None) -> ModelSpec:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML from {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {source} must be a mapping, got {type(data).__name__}")
    data = _resolve_inheritance(data, source, base_dir, seen=())
    try:
        return ModelSpec.model_validate(data)
    except ValidationError as exc:
        # Only schema-validation failures mean "bad model spec". Other
        # exceptions (e.g. RecursionError) are genuine bugs and must propagate
        # rather than be relabelled as invalid user input.
        raise ValueError(f"Invalid model specification in {source}: {exc}") from exc


def _apply_activity_override(spec: ModelSpec, activity_model: str) -> ModelSpec:
    """Override the speciation ``activity_model`` on a loaded spec.

    Lets a caller enable an ionic-strength activity correction on a shipped
    model whose YAML ships ``activity_model: none`` (e.g. ``adm1``), without
    editing the YAML. Raises if the model has no ``speciation:`` block (there
    is no pH solver to correct) or the value is invalid.
    """
    from aquakin.core.ph_solver import _ACTIVITY_MODELS

    if activity_model not in _ACTIVITY_MODELS:
        raise ValueError(
            f"activity_model must be one of {_ACTIVITY_MODELS}; got {activity_model!r}"
        )
    if spec.speciation is None:
        raise ValueError(
            "activity_model override requires the model to declare a "
            "speciation: block (a state-derived pH); this model has none."
        )
    spec.speciation = spec.speciation.model_copy(update={"activity_model": activity_model})
    return spec


# Built-in models are deterministic for a given name and are treated as
# immutable (nothing mutates a CompiledModel in place; temperature
# re-referencing etc. go through ``dataclasses.replace``, which copies). Caching
# the compiled object by name therefore (a) skips re-parsing + re-compiling the
# YAML on every call, and (b) -- the bigger win -- returns the *same* object each
# time, so a stable ``id(model)`` lets the cross-instance compiled-solver cache
# (see ``integrate/_common.py``) share compiled solves across every reactor and
# test that loads the same model. Call ``clear_model_cache()`` to reset.
_MODEL_CACHE: dict[str, CompiledModel] = {}


def clear_model_cache() -> None:
    """Clear the built-in-model cache (see :func:`load_model`)."""
    _MODEL_CACHE.clear()


def load_model(name: str, *, activity_model: Union[str, None] = None) -> CompiledModel:
    """
    Load a built-in model shipped with ``aquakin``.

    The compiled model is cached by name and reused on subsequent calls (a
    ``CompiledModel`` is immutable in use). Use
    :func:`clear_model_cache` to reset the cache.

    Parameters
    ----------
    name : str
        Model name (e.g. ``"ozone_bromate"``).
    activity_model : str, optional
        Override the speciation ``activity_model`` (ionic-strength activity
        correction in the pH solver) -- e.g. ``"davies"`` to run ``adm1`` with
        activity coefficients instead of the shipped ``"none"`` (which uses molar
        concentrations directly, the ADM1/BSM2 convention). Requires the model
        to declare a ``speciation:`` block. When given, the cache is bypassed (so
        the default and the overridden model coexist); ``None`` (default)
        returns the cached default unchanged.

    Returns
    -------
    CompiledModel
    """
    if activity_model is None:
        cached = _MODEL_CACHE.get(name)
        if cached is not None:
            return cached
    resource = files("aquakin.models") / f"{name}.yaml"
    if not resource.is_file():
        available = sorted(
            p.name.removesuffix(".yaml")
            for p in files("aquakin.models").iterdir()
            if p.name.endswith(".yaml")
        )
        raise FileNotFoundError(f"Built-in model '{name}' not found. Available: {available}")
    text = resource.read_text(encoding="utf-8")
    spec = _yaml_to_spec(text, f"built-in model '{name}'")
    if activity_model is not None:
        spec = _apply_activity_override(spec, activity_model)
        return compile_model(spec)
    net = compile_model(spec)
    _MODEL_CACHE[name] = net
    return net


def load_model_from_file(
    path: Union[str, Path], *, activity_model: Union[str, None] = None
) -> CompiledModel:
    """
    Load a model from a YAML file on disk.

    Parameters
    ----------
    path : str or Path
        Path to a YAML model file.
    activity_model : str, optional
        Override the speciation ``activity_model`` (see :func:`load_model`).

    Returns
    -------
    CompiledModel
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model file not found: {p}")
    text = p.read_text(encoding="utf-8")
    spec = _yaml_to_spec(text, str(p), base_dir=p.parent)
    if activity_model is not None:
        spec = _apply_activity_override(spec, activity_model)
    return compile_model(spec)
