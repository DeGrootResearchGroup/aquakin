"""Declarative YAML model inheritance.

A network YAML may declare ``network.extends: <base>`` to inherit a base
network and then **add / modify / remove** pieces, instead of copying the whole
file. The base is resolved and parsed to a raw mapping, the derived mapping is
merged onto it (this module), and the merged whole is handed to the normal
Pydantic schema validation -- so a variant that differs by one parameter and a
rate expression is a few lines, and a fix to the base reaches every variant.

Merge semantics (applied to the raw mappings, before validation):

- **named-list blocks** (``species``, ``conditions``, ``reactions``) are matched
  by ``name``: a derived entry's fields are deep-merged onto the base entry (so
  overriding just ``rate`` keeps the base ``stoichiometry``); a name absent from
  the base is appended in order.
- every other block (``network``, ``parameters``, ``expressions``,
  ``speciation``, ``positivity_limiter``, ``clip_negative_states`` ...) is
  deep-merged: nested mappings merge key-wise, scalars/lists in the derived
  mapping replace the base's.
- a ``remove:`` block then drops named entries / keys from the merged result.

The recursion and base-file resolution live in :mod:`aquakin.schema.loader`
(which knows about ``importlib.resources``); this module is pure dict surgery.
"""

from __future__ import annotations

# Blocks that are lists of ``{name: ..., ...}`` entries, merged by name.
NAMED_LIST_BLOCKS = ("species", "conditions", "reactions")
# Blocks a ``remove:`` mapping may target.
REMOVABLE_BLOCKS = ("species", "conditions", "reactions", "parameters", "expressions")


def pop_inheritance_keys(data: dict, source: str):
    """Extract and strip the inheritance directives from a raw network mapping.

    Returns ``(extends, remove)``. ``extends`` is read from ``network.extends``
    (canonical) or a top-level ``extends:`` (convenience); declaring both is an
    error. Both keys are removed from ``data`` so the remaining mapping is a
    plain network spec.
    """
    remove = data.pop("remove", None)
    top_extends = data.pop("extends", None)
    net = data.get("network")
    net_extends = net.pop("extends", None) if isinstance(net, dict) else None
    if top_extends is not None and net_extends is not None:
        raise ValueError(
            f"{source}: declare 'extends' once -- either top-level or under 'network:', not both."
        )
    return (net_extends if net_extends is not None else top_extends), remove


def _deep_merge(base, derived):
    """Recursively merge ``derived`` onto ``base``: mappings merge key-wise, any
    other value (scalar / list) in ``derived`` replaces the base value."""
    if isinstance(base, dict) and isinstance(derived, dict):
        out = dict(base)
        for key, val in derived.items():
            out[key] = _deep_merge(out[key], val) if key in out else val
        return out
    return derived


def _merge_named_list(base_list, derived_list, *, block: str, source: str):
    """Merge two ``[{name: ..., ...}]`` blocks by ``name`` (field-level)."""
    by_name: dict = {}
    order: list = []
    for entry in base_list or []:
        _require_named(entry, block, source)
        by_name[entry["name"]] = dict(entry)
        order.append(entry["name"])
    for entry in derived_list or []:
        _require_named(entry, block, source)
        name = entry["name"]
        if name in by_name:
            by_name[name] = _deep_merge(by_name[name], entry)
        else:
            by_name[name] = dict(entry)
            order.append(name)
    return [by_name[name] for name in order]


def _require_named(entry, block: str, source: str):
    if not isinstance(entry, dict) or "name" not in entry:
        raise ValueError(
            f"{source}: every '{block}' entry must be a mapping with a 'name' "
            f"to merge by; got {entry!r}."
        )


def merge_spec(base: dict, derived: dict, *, source: str) -> dict:
    """Merge a derived network mapping onto its base, returning a new mapping."""
    out = dict(base)
    for key, val in derived.items():
        if key in NAMED_LIST_BLOCKS:
            out[key] = _merge_named_list(out.get(key), val, block=key, source=source)
        elif isinstance(out.get(key), dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def apply_remove(data: dict, remove, source: str) -> dict:
    """Drop the named entries / keys listed in a ``remove:`` mapping."""
    if not isinstance(remove, dict):
        raise ValueError(
            f"{source}: 'remove:' must be a mapping of {list(REMOVABLE_BLOCKS)} to name lists."
        )
    unknown = set(remove) - set(REMOVABLE_BLOCKS)
    if unknown:
        raise ValueError(
            f"{source}: remove: unknown block(s) {sorted(unknown)};"
            f" valid blocks: {list(REMOVABLE_BLOCKS)}."
        )
    for block in NAMED_LIST_BLOCKS:
        names = set(remove.get(block) or [])
        if not names:
            continue
        # A base entry may legitimately lack a 'name' (malformed/partial); use
        # .get so it is simply un-targetable by remove: rather than a bare KeyError.
        present = {e["name"] for e in data.get(block, []) if isinstance(e, dict) and "name" in e}
        missing = names - present
        if missing:
            raise ValueError(
                f"{source}: remove.{block} names {sorted(missing)} are not in the base network."
            )
        data[block] = [
            e for e in data.get(block, []) if not (isinstance(e, dict) and e.get("name") in names)
        ]
    for block in ("parameters", "expressions"):
        block_map = data.get(block) or {}
        for name in remove.get(block) or []:
            if name not in block_map:
                raise ValueError(f"{source}: remove.{block} '{name}' is not in the base network.")
            del block_map[name]
    return data
