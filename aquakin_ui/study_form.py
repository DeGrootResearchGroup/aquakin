"""The study dialog's draft: a study's settings in the shape the page edits.

**Pure** (no aquakin, no trame). The worker describes every study setting
(``study_form``: its type, label, help and default, and which kinds use it);
:func:`draft_for` turns a study's settings into editable values (numbers as
text, an even time grid as start / stop / count), and :func:`study_from` turns
the edited values back into settings, leaving out what was left empty. aquakin
checks the result when the study is written.
"""

from __future__ import annotations

import math

__all__ = ["draft_for", "fields_for", "missing", "study_from"]


def fields_for(form: dict, kind: str) -> list[dict]:
    """The fields the dialog shows for ``kind``, in order, each with its name."""
    entry = next(k for k in form["kinds"] if k["kind"] == kind)
    required = set(entry["required"])
    return [
        {"name": name, "required": name in required, **form["fields"][name]}
        for name in entry["fields"]
    ]


# -- settings -> draft --------------------------------------------------------


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return repr(value)
    return str(value)


def _observables(value) -> list[dict]:
    out = []
    for entry in value or []:
        measures = {c["measure"]: c for c in entry.get("channels") or [] if isinstance(c, dict)}
        channels = [c["measure"] if isinstance(c, dict) else c for c in entry.get("channels") or []]
        out.append({"stream": entry["stream"], "channels": channels, "measures": measures})
    return out


def _candidate(value) -> dict:
    value = value or {}
    sigma = value.get("sigma")
    return {
        "name": value.get("name", ""),
        "sigma": ", ".join(_text(s) for s in sigma) if isinstance(sigma, list) else _text(sigma),
        "observables": _observables(value.get("observables")),
    }


def _times(value) -> dict:
    if isinstance(value, dict):
        return {
            "mode": "grid",
            "start": _text(value["start"]),
            "stop": _text(value["stop"]),
            "num": _text(value["num"]),
            "list": "",
        }
    if value:
        return {"mode": "list", "start": "", "stop": "", "num": "", "list": _list(value)}
    return {"mode": "grid", "start": "", "stop": "", "num": "", "list": ""}


def _list(values) -> str:
    return ", ".join(_text(v) for v in values)


def _draft_value(field: dict, value):
    t = field["type"]
    if t in ("text", "file", "parameter"):
        return value or ""
    if t in ("number", "integer"):
        return _text(value)
    if t == "boolean":
        return bool(value)
    if t == "choice":
        return value
    if t == "numbers":
        return _list(value) if isinstance(value, list) else _text(value)
    if t == "span":
        start, stop = value if value else ("", "")
        return {"start": _text(start), "stop": _text(stop)}
    if t == "times":
        return _times(value)
    if t == "run_time":
        value = value or {}
        span = value.get("t_span") or ("", "")
        grid = _times(value.get("t_eval"))
        return {
            "dynamic": bool(value),
            "start": _text(span[0]),
            "stop": _text(span[1]),
            "every": grid,
            "unit": value.get("time_unit") or "",
        }
    if t == "initial_state":
        mode = value if value in ("default", "steady") else "file"
        return {"mode": mode, "file": "" if mode != "file" else value}
    if t in ("streams", "parameters"):
        return list(value or [])
    if t == "parameter_values":
        return [{"name": k, "value": _text(v)} for k, v in (value or {}).items()]
    if t == "transforms":
        return [{"name": k, "transform": v} for k, v in (value or {}).items()]
    if t == "observables":
        return _observables(value)
    if t == "candidates":
        return [_candidate(c) for c in value or []]
    if t == "candidate":
        return {"enabled": value is not None, **_candidate(value)}
    raise ValueError(f"unknown field type {t!r}")


def draft_for(form: dict, kind: str, study: dict | None = None) -> dict:
    """Editable values for every field of ``kind``.

    Parameters
    ----------
    form : dict
        The worker's ``study_form`` reply.
    kind : str
        The study kind.
    study : dict, optional
        Settings to start from (a study being edited, or the draft's earlier
        settings when the kind changes); the fields' defaults otherwise.

    Returns
    -------
    dict
        ``{field name: editable value}``.
    """
    study = study or {}
    return {
        f["name"]: _draft_value(f, study.get(f["name"], f["default"]))
        for f in fields_for(form, kind)
    }


# -- draft -> settings --------------------------------------------------------


def _number(label: str, text, *, integer: bool = False):
    text = str(text).strip()
    try:
        value = int(text) if integer else float(text)
    except ValueError:
        kind = "a whole number" if integer else "a number"
        raise ValueError(f"{label}: {text!r} is not {kind}") from None
    if not math.isfinite(value):
        raise ValueError(f"{label}: {text!r} is not a finite number")
    return value


def _numbers(label: str, text) -> list[float]:
    parts = [p for p in str(text).replace(";", ",").split(",") if p.strip()]
    return [_number(label, p) for p in parts]


def _read_times(label: str, value: dict):
    if value["mode"] == "list":
        values = _numbers(label, value["list"])
        return values or None
    given = [str(value[k]).strip() for k in ("start", "stop", "num")]
    if not any(given):
        return None
    if not all(given):
        raise ValueError(f"{label}: give the start, the end and the number of times")
    return {
        "start": _number(label, value["start"]),
        "stop": _number(label, value["stop"]),
        "num": _number(label, value["num"], integer=True),
    }


def _read_observables(label: str, rows) -> list[dict]:
    out = []
    for row in rows:
        stream = (row.get("stream") or "").strip()
        if not stream:
            continue
        measures = row.get("measures") or {}
        channels = [measures.get(c, c) for c in row.get("channels") or []]
        out.append({"stream": stream, **({"channels": channels} if channels else {})})
    return out


def _read_candidate(label: str, value: dict) -> dict | None:
    name = (value.get("name") or "").strip()
    observables = _read_observables(label, value.get("observables") or [])
    if not name and not observables:
        return None
    if not name:
        raise ValueError(f"{label}: give each sensor set a name")
    sigma = _numbers(f"{label} {name}", value.get("sigma", ""))
    if not sigma:
        raise ValueError(f"{label} {name}: give its measurement error")
    return {
        "name": name,
        "observables": observables,
        "sigma": sigma[0] if len(sigma) == 1 else sigma,
    }


def _read(field: dict, value):
    """A field's settings value from its draft, or ``None`` when left empty."""
    t, label = field["type"], field["label"]
    if t in ("text", "file", "parameter", "choice"):
        value = (value or "").strip() if isinstance(value, str) else value
        return value or None
    if t in ("number", "integer"):
        if not str(value).strip():
            return None
        return _number(label, value, integer=t == "integer")
    if t == "boolean":
        return bool(value)
    if t == "numbers":
        values = _numbers(label, value)
        if not values:
            return None
        return values[0] if len(values) == 1 else values
    if t == "span":
        if not str(value["start"]).strip() and not str(value["stop"]).strip():
            return None
        return [_number(label, value["start"]), _number(label, value["stop"])]
    if t == "times":
        return _read_times(label, value)
    if t == "run_time":
        if not value["dynamic"]:
            return None
        out = {"t_span": [_number(label, value["start"]), _number(label, value["stop"])]}
        every = _read_times(label, value["every"])
        if every is not None:
            out["t_eval"] = every
        if str(value.get("unit") or "").strip():
            out["time_unit"] = value["unit"].strip()
        return out
    if t == "initial_state":
        if value["mode"] != "file":
            return value["mode"]
        path = (value.get("file") or "").strip()
        if not path:
            raise ValueError(f"{label}: give the file of the state to start from")
        return path
    if t in ("streams", "parameters"):
        return [v for v in value if str(v).strip()] or None
    if t == "parameter_values":
        out = {}
        for row in value:
            name = (row.get("name") or "").strip()
            if name:
                out[name] = _number(f"{label} {name}", row.get("value", ""))
        return out or None
    if t == "transforms":
        out = {r["name"].strip(): r["transform"] for r in value if (r.get("name") or "").strip()}
        return out or None
    if t == "observables":
        return _read_observables(label, value) or None
    if t == "candidates":
        out = [c for c in (_read_candidate(label, v) for v in value) if c is not None]
        return out or None
    if t == "candidate":
        return _read_candidate(label, value) if value.get("enabled") else None
    raise ValueError(f"unknown field type {t!r}")


def study_from(form: dict, kind: str, draft: dict) -> dict:
    """The study settings an edited draft gives (empty fields and defaults left out).

    Raises
    ------
    ValueError
        Naming the field whose text is not a number where one belongs.
    """
    study: dict = {"kind": kind}
    for field in fields_for(form, kind):
        name = field["name"]
        value = _read(
            field, draft[name] if name in draft else _draft_value(field, field["default"])
        )
        if value is None or value == field["default"]:
            continue
        study[field["name"]] = value
    return study


def missing(form: dict, kind: str, study: dict) -> str | None:
    """What a study of ``kind`` still needs, by the fields' labels, or ``None``.

    aquakin refuses such a study too, but by its setting names; the dialog asks
    first so the message names what the page shows.
    """
    labels = [
        f["label"] for f in fields_for(form, kind) if f["required"] and f["name"] not in study
    ]
    if not labels:
        return None
    return f"Still needed: {', '.join(labels)}."
