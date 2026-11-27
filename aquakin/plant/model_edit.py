"""Editing a model's reactions and parameters from a front end.

The flowsheet editor shows a model's reactions and parameters and lets a user
edit a reaction's ``rate`` and stoichiometry, and a parameter's ``value``,
``units`` and ``description``, or add a parameter. It parses nothing itself:
these functions, behind the ``aquakin serve`` worker, read the model, check one
expression, check a whole set of edits, and write them.

An **edit set** holds only what changed, every value as the text typed::

    {"reactions": {"<reaction>": {"rate": "<expression>",
                                  "stoichiometry": {"<species>": "<coefficient>"},
                                  "parameters": {"<local name>": {"value": "1.5"}}}},
     "parameters": {"<name>": {"value": "0.5", "units": "1/d", "description": ""}}}

A model-level parameter the model lacks is added (it needs a ``value``);
reactions, species and reaction-local parameters cannot be added. Edits are
written the way ``extends:`` reads them (:mod:`aquakin.schema.inheritance`): a
reaction entry with the same ``name`` overrides only the fields it gives, and a
parameter entry likewise. So a project model is written in place, its own file
holding only what it already held plus the edited fields (overrides, when it
``extends:`` another), and a shipped model is never edited: the edits become a
new project model, ``models/<name>.yaml``, that ``extends:`` it, and every unit
of the flowsheet that used the shipped model switches to the new one.
"""

from __future__ import annotations

import copy
import math
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import ValidationError

from aquakin.core.hints import did_you_mean
from aquakin.core.lexer import tokenize
from aquakin.core.model import compile_model, validate_stoich_ast
from aquakin.core.parser import ParseError, parse_rate_expression
from aquakin.core.stoich_resolve import is_auto
from aquakin.plant.catalog import load_model_file
from aquakin.plant.spec import (
    SpecError,
    _check_identifier_or_error,
    _check_writes,
    _do_writes,
    _file_digest,
    _located,
    _plan_model,
    _shipped_names,
    _within,
    write_spec,
)
from aquakin.schema.inheritance import pop_inheritance_keys
from aquakin.schema.loader import load_model, parse_model_yaml, resolve_model_data
from aquakin.schema.model_spec import ModelSpec

#: The fields of a parameter an edit may set.
PARAMETER_FIELDS = ("value", "units", "description")

#: What an expression may be: a reaction's rate, or a stoichiometric coefficient.
EXPRESSION_KINDS = ("rate", "stoichiometry")


class ModelEditError(SpecError):
    """A set of edits that does not make a valid model; nothing was written.

    Attributes
    ----------
    errors : list of dict
        ``{"field", "error", "position"}`` per problem, as
        :func:`check_model_edits` returns them.
    """

    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(f"{e['field']}: {e['error']}" for e in errors))
        self.errors = errors


# ---------------------------------------------------------------------------
# Reading a model
# ---------------------------------------------------------------------------


def _model_text(model: str, path) -> tuple[str, str, Path | None]:
    """``(YAML text, name for messages, directory its extends: is read from)``."""
    if path is None:
        resource = files("aquakin.models") / f"{model}.yaml"
        if not resource.is_file():
            load_model(model)  # raises the loader's own "not found" with the list
        return resource.read_text(encoding="utf-8"), f"built-in model '{model}'", None
    p = Path(path)
    if not p.is_file():
        raise SpecError(f"model file {p} does not exist")
    return p.read_text(encoding="utf-8"), str(p), p.parent


def _compiled(model: str, path):
    return load_model(model) if path is None else load_model_file(path)


def _text(value) -> str:
    """A number or expression as the text a front end shows (numbers exactly)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return repr(float(value))


def _parameter_rows(block: dict, reaction: str | None) -> list[dict]:
    return [
        {
            "name": name,
            "reaction": reaction,
            "value": _text(entry.get("value")),
            "units": str(entry.get("units", "")),
            "description": str(entry.get("description", "")),
        }
        for name, entry in (block or {}).items()
        if isinstance(entry, dict)
    ]


def model_document(model: str, path=None) -> dict:
    """What a front end shows to edit a model: its reactions and parameters.

    Parameters
    ----------
    model : str
        The model's name (a shipped model's, unless ``path`` is given).
    path : str or pathlib.Path, optional
        The model's YAML file (a project model).

    Returns
    -------
    dict
        ``name``; ``source`` (``"shipped"`` or ``"file"``); ``path`` and
        ``sha256`` of the file (``None`` for a shipped model); ``extends`` (the
        base it names, or ``None``); ``reactions`` (``name``, ``description``,
        ``rate`` and ``stoichiometry``, a list of ``{species, value}``);
        ``parameters`` (``name``, ``reaction`` (``None`` for a model-level one),
        ``value``, ``units``, ``description``), every value as text; and the
        names an expression may use: ``species``, ``conditions`` (derived ones
        such as a speciation's pH included), ``expressions``.
    """
    import hashlib

    text, source, base_dir = _model_text(model, path)
    data = resolve_model_data(text, source, base_dir)
    own = parse_model_yaml(text)
    extends, _ = pop_inheritance_keys(own if isinstance(own, dict) else {}, source)
    context = _context(model, path)
    parameters = _parameter_rows(data.get("parameters"), None)
    reactions = []
    for r in data.get("reactions") or []:
        reactions.append(
            {
                "name": r["name"],
                "description": str(r.get("description", "")),
                "rate": str(r.get("rate", "")),
                "stoichiometry": [
                    {"species": sp, "value": _text(v)}
                    for sp, v in (r.get("stoichiometry") or {}).items()
                ],
            }
        )
        parameters += _parameter_rows(r.get("parameters"), r["name"])
    return {
        "name": model,
        "source": "shipped" if path is None else "file",
        "path": None if path is None else str(Path(path).resolve()),
        "sha256": None if path is None else hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "extends": extends,
        "reactions": reactions,
        "parameters": parameters,
        "species": context["species"],
        "conditions": context["conditions"],
        "expressions": context["expressions"],
    }


def _context(model: str, path) -> dict:
    """The names an expression of ``model`` may use, by what they are."""
    text, source, base_dir = _model_text(model, path)
    data = resolve_model_data(text, source, base_dir)
    compiled = _compiled(model, path)
    declared = [c["name"] for c in data.get("conditions") or [] if isinstance(c, dict)]
    return {
        "species": list(compiled.species),
        "conditions": declared + [f for f in compiled.derived_fields if f not in declared],
        "expressions": list(data.get("expressions") or {}),
        "parameters": list(data.get("parameters") or {}),
        "local": {
            r["name"]: list(r.get("parameters") or {})
            for r in data.get("reactions") or []
            if isinstance(r, dict) and "name" in r
        },
    }


# ---------------------------------------------------------------------------
# One expression
# ---------------------------------------------------------------------------


def _finite_number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def check_expression(
    expression: str,
    *,
    model: str,
    path=None,
    kind: str = "rate",
    reaction: str | None = None,
    parameters=(),
) -> dict:
    """Parse one rate or stoichiometry expression in the context of a model.

    Parameters
    ----------
    expression : str
        The text typed.
    model : str
        The model's name.
    path : str or pathlib.Path, optional
        The model's file (a project model).
    kind : {"rate", "stoichiometry"}
        A reaction's rate, or a stoichiometric coefficient (a number, ``auto``,
        or an expression in parameters and numbers only).
    reaction : str, optional
        The reaction it belongs to, whose local parameters it may use.
    parameters : sequence of str
        Parameters the edit adds, which it may use too.

    Returns
    -------
    dict
        ``ok``; ``error`` (the first problem, or ``None``) and ``position``
        (its character offset in ``expression``, or ``None``); ``references``,
        the known ``species``, ``parameters``, ``conditions`` and ``expressions``
        it uses; and ``unknown``, the ``species``, ``parameters`` and
        ``conditions`` it uses that the model does not define (an unknown
        parameter can be added).

    Raises
    ------
    ValueError
        If ``kind`` is not one of :data:`EXPRESSION_KINDS`.
    """
    if kind not in EXPRESSION_KINDS:
        raise ValueError(f"kind must be one of {list(EXPRESSION_KINDS)}, not {kind!r}")
    references: dict[str, list[str]] = {
        "species": [],
        "parameters": [],
        "conditions": [],
        "expressions": [],
    }
    unknown: dict[str, list[str]] = {"species": [], "parameters": [], "conditions": []}

    def result(error=None, position=None):
        return {
            "ok": error is None,
            "error": error,
            "position": position,
            "references": references,
            "unknown": unknown,
        }

    text = str(expression).strip()
    if not text:
        return result("write an expression" if kind == "rate" else "write a coefficient", 0)
    if kind == "stoichiometry" and (_finite_number(text) is not None or is_auto(text)):
        return result()
    try:
        ast = parse_rate_expression(text)
    except ParseError as exc:
        return result(str(exc), exc.position)
    ctx = _context(model, path)
    known = {
        "species": set(ctx["species"]),
        "conditions": set(ctx["conditions"]),
        "parameters": {*ctx["parameters"], *ctx["local"].get(reaction, []), *parameters},
    }
    expressions = set(ctx["expressions"])
    for what, names in (
        ("species", ast.species()),
        ("conditions", ast.condition_names()),
        ("parameters", ast.param_names()),
    ):
        for name in sorted(names):
            if name in known[what]:
                references[what].append(name)
            elif what == "parameters" and name in expressions:
                references["expressions"].append(name)
            else:
                unknown[what].append(name)
    if kind == "stoichiometry":
        try:
            validate_stoich_ast(ast, reaction or "?", "?")
        except ValueError as exc:
            return result(str(exc))
        if references["expressions"]:
            return result(
                f"'{references['expressions'][0]}' is a named expression; a stoichiometric "
                f"coefficient may use only parameters and numbers"
            )
    for what, noun in (("species", "species"), ("conditions", "condition")):
        if unknown[what]:
            name = unknown[what][0]
            return result(
                f"'{name}' is not a {noun} of {model}.{did_you_mean(name, known[what])}",
                _position_of(text, what, name),
            )
    if unknown["parameters"]:
        name = unknown["parameters"][0]
        either = " or named expression" if kind == "rate" else ""
        return result(
            f"'{name}' is not a parameter{either} of {model}."
            f"{did_you_mean(name, known['parameters'] | expressions)}",
            _position_of(text, "parameters", name),
        )
    return result()


def _position_of(text: str, what: str, name: str) -> int:
    """Where ``text`` (which parses, and names it) first names the species,
    condition or parameter ``name``: the ``[`` or ``{`` before a species or
    condition, the name itself for a parameter."""
    tokens = tokenize(text, error=ParseError)
    kinds = [None, *(t.kind for t in tokens)]  # kinds[i] is the kind before tokens[i]
    if what in ("species", "conditions"):
        opener = "LBRACK" if what == "species" else "LBRACE"
        return next(
            tokens[i - 1].pos
            for i, t in enumerate(tokens)
            if t.kind == "IDENT" and kinds[i] == opener and name.startswith(t.value)
        )
    return next(
        t.pos
        for i, t in enumerate(tokens)
        if t.kind == "IDENT"
        and t.value == name
        and kinds[i] not in ("LBRACK", "LBRACE")
        and tokens[i + 1].kind != "LPAREN"
    )


# ---------------------------------------------------------------------------
# A set of edits
# ---------------------------------------------------------------------------


def _error(field: str, message: str, position=None) -> dict:
    return {"field": field, "error": message, "position": position}


def _parameter_value(fields: dict, where: str, errors: list) -> dict:
    """The parameter fields of an edit as they are written (``value`` a float)."""
    out = {}
    for key, value in fields.items():
        if key not in PARAMETER_FIELDS:
            errors.append(_error(where, f"'{key}' is not a field an edit may set"))
        elif key == "value":
            number = _finite_number(str(value).strip())
            if number is None:
                errors.append(_error(f"{where}.value", f"{str(value).strip()!r} is not a number"))
            else:
                out["value"] = number
        else:
            out[key] = str(value).strip()
    return out


def _coefficient(text: str):
    """A stoichiometric coefficient as it is written: a number stays a number."""
    text = str(text).strip()
    number = _finite_number(text)
    return text if number is None else number


def _merge_edits(data: dict, edits: dict, errors: list) -> dict:
    """``data`` (a model file's own mapping) with ``edits`` written into it."""
    out = copy.deepcopy(data)
    for key in set(edits) - {"reactions", "parameters"}:
        errors.append(_error(key, "only reactions and parameters can be edited here"))
    for name, fields in (edits.get("parameters") or {}).items():
        block = out.get("parameters") or {}
        block[name] = {
            **(block.get(name) or {}),
            **_parameter_value(fields, f"parameters.{name}", errors),
        }
        out["parameters"] = block
    for name, fields in (edits.get("reactions") or {}).items():
        reactions = out.get("reactions") or []
        entry = next((r for r in reactions if isinstance(r, dict) and r.get("name") == name), None)
        if entry is None:
            entry = {"name": name}
            reactions.append(entry)
        out["reactions"] = reactions
        for key in set(fields) - {"rate", "stoichiometry", "parameters"}:
            errors.append(_error(f"reactions.{name}", f"'{key}' is not a field an edit may set"))
        if "rate" in fields:
            entry["rate"] = str(fields["rate"]).strip()
        if fields.get("stoichiometry"):
            entry["stoichiometry"] = {
                **(entry.get("stoichiometry") or {}),
                **{sp: _coefficient(v) for sp, v in fields["stoichiometry"].items()},
            }
        local = entry.get("parameters") or {}
        for p, pfields in (fields.get("parameters") or {}).items():
            where = f"reactions.{name}.parameters.{p}"
            local[p] = {**(local.get(p) or {}), **_parameter_value(pfields, where, errors)}
            entry["parameters"] = local
    return out


def _scope_errors(model: str, path, edits: dict) -> list[dict]:
    """Edits that reach outside what the panel edits, and every edited expression's
    own problem (with its position)."""
    ctx = _context(model, path)
    errors: list[dict] = []
    added = [p for p in edits.get("parameters") or {} if p not in ctx["parameters"]]
    for p in added:
        if not p.isidentifier():
            errors.append(_error(f"parameters.{p}", f"'{p}' cannot be a parameter's name"))
        elif "value" not in (edits["parameters"][p] or {}):
            errors.append(_error(f"parameters.{p}.value", "a new parameter needs a value"))
    for name, fields in (edits.get("reactions") or {}).items():
        where = f"reactions.{name}"
        if name not in ctx["local"]:
            errors.append(
                _error(where, f"{model} has no reaction '{name}'{did_you_mean(name, ctx['local'])}")
            )
            continue
        for p in fields.get("parameters") or {}:
            if p not in ctx["local"][name]:
                errors.append(
                    _error(f"{where}.parameters.{p}", f"'{name}' has no local parameter '{p}'")
                )
        checks = [("rate", fields["rate"], "rate")] if "rate" in fields else []
        checks += [
            (f"stoichiometry.{sp}", v, "stoichiometry")
            for sp, v in (fields.get("stoichiometry") or {}).items()
        ]
        for key, text, kind in checks:
            found = check_expression(
                text, model=model, path=path, kind=kind, reaction=name, parameters=added
            )
            if not found["ok"]:
                errors.append(_error(f"{where}.{key}", found["error"], found["position"]))
    return errors


def _field_of(loc: tuple, data: dict) -> str:
    """A Pydantic error location with each reaction's index replaced by its name."""
    parts = []
    for i, part in enumerate(loc):
        if isinstance(part, int) and i > 0 and loc[i - 1] == "reactions":
            part = data["reactions"][part].get("name", part)
        parts.append(str(part))
    return ".".join(parts) or "model"


def _load_errors(text: str, source: str, base_dir) -> list[dict]:
    """Load a model's text as the loader would, and compile it; the problems by field."""
    data = resolve_model_data(text, source, base_dir)
    try:
        spec = ModelSpec.model_validate(data)
    except ValidationError as exc:
        return [
            _error(_field_of(tuple(e["loc"]), data), e["msg"].removeprefix("Value error, "))
            for e in exc.errors()
        ]
    try:
        compile_model(spec)
    except ValueError as exc:  # what the model says about itself, not about a field
        return [_error("model", str(exc))]
    return []


def _edited_text(model: str, path, edits: dict, name: str, errors: list) -> str:
    """The YAML of the model file that holds ``edits``: the project file with them
    written in, or for a shipped model a new file named ``name`` that extends it."""
    if path is None:
        data: dict = {"model": {"name": name, "extends": model}}
    else:
        data = parse_model_yaml(Path(path).read_text(encoding="utf-8"))
    merged = _merge_edits(data, edits, errors)
    return yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)


def _edit_errors(model: str, path, edits: dict, name: str, dest: Path) -> tuple[str, list]:
    """``(file text, problems)`` for ``edits``: the text is meaningful only without problems."""
    errors = _scope_errors(model, path, edits)
    text = _edited_text(model, path, edits, name, errors)
    if not errors:
        errors = _load_errors(text, str(dest), dest.parent)
    return text, errors


def check_model_edits(model: str, path=None, edits=None) -> dict:
    """Check a set of edits by loading and compiling the model they make.

    The edits are written into the model's YAML exactly as a save would write
    them (into the project file, or into a new file extending a shipped model),
    and that text is loaded through the loader and compiled.

    Parameters
    ----------
    model : str
        The model's name.
    path : str or pathlib.Path, optional
        The model's file (a project model); ``None`` for a shipped model.
    edits : dict
        The edit set (see the module docstring).

    Returns
    -------
    dict
        ``ok``, and ``errors``: ``{"field", "error", "position"}`` per
        problem, ``field`` a dotted path (``reactions.<name>.rate``,
        ``parameters.<name>.value``, or ``model`` for the model as a whole) and
        ``position`` an expression's character offset, or ``None``.
    """
    name = model if path is not None else f"{model}_edited"
    dest = Path(path) if path is not None else Path("models") / f"{name}.yaml"
    _, errors = _edit_errors(model, path, edits or {}, name, dest)
    return {"ok": not errors, "errors": errors}


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def has_comments(text: str) -> bool:
    """Whether YAML ``text`` holds a comment, which writing it back loses.

    A ``#`` is a comment where YAML reads one: outside every scalar (a quoted
    or plain value, or a key) and at the start of a line or after a space.

    Parameters
    ----------
    text : str
        The YAML.

    Returns
    -------
    bool
    """
    scalars = [
        (t.start_mark.index, t.end_mark.index)
        for t in yaml.scan(text)
        if isinstance(t, yaml.ScalarToken)
    ]
    return any(
        (i == 0 or text[i - 1] in " \t\n") and not any(a <= i < b for a, b in scalars)
        for i, c in enumerate(text)
        if c == "#"
    )


def _switched(spec: dict, old: str, new: str) -> tuple[dict, list[str]]:
    """``spec`` with every unit's model (and reagent model) ``old`` named ``new``
    and ``old`` no longer listed, and the units that use it."""
    spec = copy.deepcopy(spec)
    used = []
    for uid, entry in (spec.get("units") or {}).items():
        reagent = entry.get("reagent") if isinstance(entry.get("reagent"), dict) else {}
        if old not in (entry.get("model"), reagent.get("model")):
            continue
        used.append(uid)
        if entry.get("model") == old:
            entry["model"] = new
        if reagent.get("model") == old:
            reagent["model"] = new
    if old != new:
        (spec.get("models") or {}).pop(old, None)  # no unit uses it now
    return spec, used


def save_model_edits(
    spec: dict,
    path,
    *,
    model: str,
    edits: dict,
    name: str | None = None,
    sha256: str | None = None,
    base_dir=None,
    dry_run: bool = False,
) -> dict:
    """Write a model's edits and the flowsheet that names it, together.

    A project model (inside the flowsheet's folder) is edited in place. A
    shipped model is never edited: the edits become ``models/<name>.yaml``,
    which ``extends:`` it, and every unit that used the shipped model uses the
    new one. The edited model is loaded and compiled first, and every write is
    checked, so nothing is written unless all of it can be; the flowsheet is
    then written with :func:`~aquakin.plant.spec.write_spec` (the model's new
    sha256 recorded), and the model file is put back as it was if that fails.

    Parameters
    ----------
    spec : dict
        The flowsheet as the front end holds it.
    path : str or pathlib.Path
        The flowsheet's file.
    model : str
        The model edited (one of ``spec``'s models, or a shipped model).
    edits : dict
        The edit set (see the module docstring).
    name : str, optional
        The new model's name, required for a shipped model.
    sha256 : str, optional
        The project model file's sha256 when it was read for editing: a file
        changed since is refused rather than overwritten.
    base_dir : str or pathlib.Path, optional
        Where ``spec``'s relative paths point from (default: ``path``'s folder).
    dry_run : bool
        Check everything and say what would be written, writing nothing.

    Returns
    -------
    dict
        ``model`` (the edited model's name), ``model_file`` (its file),
        ``units`` (the units that use it), ``warning`` (what rewriting a
        project model's file loses: its comments; ``None`` when nothing is)
        and ``spec`` (the flowsheet as written; ``None`` on a dry run).

    Raises
    ------
    ModelEditError
        If the edits do not make a valid model.
    SpecError
        If the new name is taken or not a name, the model file lies outside
        the project or changed on disk, or a file would overwrite another.
    """
    path = Path(path)
    folder = path.parent
    base = Path(base_dir) if base_dir is not None else folder
    ref = (spec.get("models") or {}).get(model) or {"source": "shipped"}
    if ref.get("source", "shipped") == "shipped":
        if not name:
            raise SpecError(f"give the model made from the shipped '{model}' a name")
        _check_identifier_or_error("model name", name)
        if name in _shipped_names():
            raise SpecError(f"'{name}' is the name of a shipped model; choose another name")
        dest = (folder / "models" / f"{name}.yaml").resolve()
        listed = (spec.get("models") or {}).get(name)
        if listed is not None and _located(listed["source"], base, name).resolve() != dest:
            raise SpecError(
                f"the flowsheet already has a model named '{name}'; choose another name"
            )
        model_path, original = None, None
    else:
        src = _located(ref["source"], base, f"models.{model}").resolve()
        copies: list = []
        _plan_model(model, src, folder, copies)
        if copies:
            raise SpecError(
                f"model '{model}' is read from {src}, outside the flowsheet's folder; save "
                f"the flowsheet first, which copies it into the project"
            )
        if sha256 is not None and _file_digest(src) != sha256:
            raise SpecError(
                f"{src} has changed on disk since it was opened for editing; open the model "
                f"again to edit what it holds now"
            )
        name, dest, model_path, original = model, src, src, src
    text, errors = _edit_errors(model, model_path, edits, name, dest)
    if errors:
        raise ModelEditError(errors)
    content = text.encode()
    writes = [(dest, content, f"model '{name}'", original)]
    _check_writes(writes)
    new_spec, units = _switched(spec, model, name)
    new_spec.setdefault("models", {})[name] = {"source": str(dest)}
    warning = None
    if original is not None and has_comments(original.read_text(encoding="utf-8")):
        shown = _within(original, folder) or original
        warning = f"Comments in {shown.as_posix()} will not be kept when it is saved."
    out = {
        "model": name,
        "model_file": str(dest),
        "units": units,
        "warning": warning,
        "spec": None,
    }
    if dry_run:
        return out
    previous = dest.read_bytes() if dest.exists() else None
    _do_writes(writes)
    try:
        out["spec"] = write_spec(new_spec, path, base_dir=base)
    except BaseException:
        if previous is None:
            dest.unlink()
        else:
            dest.write_bytes(previous)
        raise
    return out
