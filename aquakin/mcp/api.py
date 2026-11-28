"""The public API as an assistant reads it: names, signatures and docstrings.

Read from the installed package itself, so it always matches the version that
runs. The public names are each module's ``__all__``: ``aquakin``,
``aquakin.plant`` and ``aquakin.study``, the three modules the API reference
covers. A dotted name below them (``Plant.calibrate``,
``aquakin.plant.bsm.build_bsm1``) can be described too; a private name (one
part starting with ``_``) cannot.
"""

from __future__ import annotations

import importlib
import inspect

#: The modules whose ``__all__`` is the public API.
MODULES = ("aquakin", "aquakin.plant", "aquakin.study")

_MISSING = object()


def _kind(obj) -> str:
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if isinstance(obj, property):
        return "property"
    if inspect.isroutine(obj):
        return "function"
    return "data"


def _summary(obj) -> str:
    """The first paragraph of an object's docstring, on one line."""
    if _kind(obj) == "data":
        return ""
    doc = inspect.getdoc(obj) or ""
    return " ".join(doc.split("\n\n")[0].split())


def _signature(obj) -> str | None:
    target = obj.fget if isinstance(obj, property) else obj
    if not callable(target):
        return None
    try:
        return str(inspect.signature(target))
    except (TypeError, ValueError):  # a builtin or extension object without one
        return None


def list_api(module: str = "aquakin") -> list[dict]:
    """The public names of one API module, each with what it is and its summary.

    Parameters
    ----------
    module : str, optional
        One of :data:`MODULES` (default ``"aquakin"``).

    Returns
    -------
    list of dict
        ``{"name", "kind", "summary"}`` per name of the module's ``__all__``,
        in that order. ``kind`` is ``"class"``, ``"function"``, ``"module"`` or
        ``"data"``.

    Raises
    ------
    KeyError
        If ``module`` is not one of :data:`MODULES`.
    """
    if module not in MODULES:
        raise KeyError(f"no API module {module!r}; the modules are {list(MODULES)}")
    mod = importlib.import_module(module)
    out = []
    for name in mod.__all__:
        obj = getattr(mod, name)
        out.append({"name": name, "kind": _kind(obj), "summary": _summary(obj)})
    return out


def _module(name: str):
    """The module of that name, or ``None`` when there is none."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


def _resolve(name: str):
    """The object a public dotted name refers to, and its full name."""
    parts = name.strip().split(".")
    if any(not p or p.startswith("_") for p in parts):
        raise KeyError(f"{name!r} is not a public name")
    if parts[0] == "aquakin":
        # The longest importable module prefix ("aquakin" itself at least),
        # then attributes below it.
        cut = len(parts)
        while (obj := _module(".".join(parts[:cut]))) is None:
            cut -= 1
        rest = parts[cut:]
    else:
        homes = [m for m in MODULES if parts[0] in importlib.import_module(m).__all__]
        if not homes:
            raise KeyError(
                f"{parts[0]!r} is not in the public API of {list(MODULES)}; "
                f"list_api lists the names, or give a full dotted name"
            )
        obj, rest, parts = importlib.import_module(homes[0]), parts, [*homes[0].split("."), *parts]
    for i, part in enumerate(rest):
        # A property is described as itself, not as the value of a class attribute.
        nxt = inspect.getattr_static(obj, part, None) if inspect.isclass(obj) else None
        nxt = nxt if isinstance(nxt, property) else getattr(obj, part, _MISSING)
        if nxt is _MISSING:
            where = ".".join(parts[: len(parts) - len(rest) + i])
            raise KeyError(f"{where} has no public attribute {part!r}")
        obj = nxt
    return obj, ".".join(parts)


def _members(cls) -> list[dict]:
    """The public methods and properties a class defines or inherits from aquakin."""
    out = []
    for attr in sorted(dir(cls)):
        if attr.startswith("_"):
            continue
        owner = next(c for c in cls.__mro__ if attr in vars(c))
        if not owner.__module__.startswith("aquakin"):
            continue
        member = inspect.getattr_static(cls, attr)
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        kind = _kind(member)
        if kind in ("function", "property"):
            out.append({"name": attr, "kind": kind, "summary": _summary(member)})
    return out


def describe_api(name: str) -> dict:
    """The signature and full docstring of a public name.

    Parameters
    ----------
    name : str
        A name from :func:`list_api` (``"calibrate"``, ``"Plant"``), a member
        of one (``"Plant.calibrate"``), or a full dotted name below ``aquakin``
        (``"aquakin.plant.bsm.build_bsm1"``).

    Returns
    -------
    dict
        ``name`` (the full dotted name), ``kind``, ``signature`` (``None`` for
        what is not callable), ``doc`` (the docstring, NumPy format), and for a
        class ``members`` (its public methods and properties, each with its
        summary; describe one as ``"<class>.<member>"``).

    Raises
    ------
    KeyError
        If the name is private or does not exist.
    """
    obj, full = _resolve(name)
    out = {
        "name": full,
        "kind": _kind(obj),
        "signature": _signature(obj),
        "doc": inspect.getdoc(obj) if _kind(obj) != "data" else repr(obj),
    }
    if inspect.isclass(obj):
        out["members"] = _members(obj)
    return out
