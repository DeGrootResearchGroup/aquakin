"""YAML -> validated ModelSpec -> CompiledModel."""

from __future__ import annotations

from collections.abc import Hashable
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import ValidationError

from aquakin.core.model import CompiledModel, compile_model
from aquakin.schema.inheritance import (
    apply_remove,
    merge_spec,
    pop_inheritance_keys,
)
from aquakin.schema.model_spec import ModelSpec


class _UniqueKeyLoader(yaml.SafeLoader):
    """PyYAML's safe loader, refusing a mapping that repeats a key.

    The safe loader keeps the last of two equal keys without a word, so a
    stoichiometry that names a species twice, or a reaction block pasted twice,
    loads as whichever came last.
    """

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            seen: dict = {}
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    continue  # ``<<:`` merges are overridden by explicit keys by design
                key = self.construct_object(key_node, deep=True)
                if not isinstance(key, Hashable):
                    continue  # the safe loader refuses an unhashable key itself
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r} (first given on line {seen[key].line + 1})",
                        key_node.start_mark,
                    )
                seen[key] = key_node.start_mark
        return super().construct_mapping(node, deep=deep)


def parse_model_yaml(text: str):
    """Parse a model YAML document with the safe loader, refusing duplicate keys.

    Parameters
    ----------
    text : str
        The YAML text.

    Returns
    -------
    object
        The parsed document (a mapping for a model file).

    Raises
    ------
    yaml.YAMLError
        If the text is not valid YAML, or a mapping in it repeats a key (the
        message names the key and both lines).
    """
    return yaml.load(text, Loader=_UniqueKeyLoader)  # a SafeLoader subclass


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
        base_data = parse_model_yaml(base_text)
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


def model_fingerprint(text: str, source: str, base_dir=None) -> str:
    """sha256 of a model YAML's content with its ``extends`` chain merged in.

    Two files that resolve to the same model (formatting and comments aside)
    have the same fingerprint, and a change to a base model reaches every model
    that extends it.

    Parameters
    ----------
    text : str
        The model YAML.
    source : str
        A name for messages.
    base_dir : str or Path, optional
        The directory a relative ``extends:`` path is read from.

    Returns
    -------
    str
        A 64-character hex digest.
    """
    import hashlib
    import json

    data = _resolve_inheritance(parse_model_yaml(text), source, base_dir, seen=())
    canonical = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _yaml_to_spec(text: str, source: str, *, base_dir=None) -> ModelSpec:
    try:
        data = parse_model_yaml(text)
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

    # Validate the value here rather than relying on the speciation schema:
    # ``model_copy(update=...)`` below does NOT re-run Pydantic field validators
    # (Pydantic v2 assigns the updated fields directly), so an invalid
    # activity_model would otherwise slip through silently. This check is load-
    # bearing -- do not remove it as "already validated by the schema".
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
# (see ``integrate/_cache.py``) share compiled solves across every reactor and
# test that loads the same model. Call ``clear_model_cache()`` to reset.
_MODEL_CACHE: dict[str, CompiledModel] = {}


def clear_model_cache() -> None:
    """Clear the built-in-model cache (see :func:`load_model`)."""
    _MODEL_CACHE.clear()


def load_model(name: str, *, activity_model: str | None = None) -> CompiledModel:
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
    # Where the model came from, for writing it into a flowsheet spec. A plain
    # attribute rather than a field, so a modified copy (dataclasses.replace)
    # does not carry it and is not mistaken for the shipped model.
    net._aquakin_source = ("shipped", name, model_fingerprint(text, name))
    _MODEL_CACHE[name] = net
    return net


def load_model_from_file(path: str | Path, *, activity_model: str | None = None) -> CompiledModel:
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
    net = compile_model(spec)
    # See load_model; the fingerprint is the content this model was built from.
    net._aquakin_source = ("file", str(p.resolve()), model_fingerprint(text, str(p), p.parent))
    return net
