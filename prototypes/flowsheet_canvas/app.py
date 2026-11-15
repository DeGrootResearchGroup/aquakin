"""Prototype: a drag-and-drop flowsheet editor on the Vue Flow trame widget.

Every unit kind, setting, default, help text and port comes from aquakin's
unit catalogue, asked of an ``aquakin serve`` worker process (the interface
imports neither aquakin nor JAX). Each unit is built by the worker whenever its
settings change, so the Checks list carries aquakin's own error messages.

Run: ``python app.py --port 8080``.
"""

from pathlib import Path

import yaml
from aquakin_worker import AquakinWorker, WorkerError
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import client, html
from trame.widgets import vuetify3 as v3
from trame_flowsheet import Flowsheet

DRAG_TYPE = "application/x-aquakin-unit"
HERE = Path(__file__).parent

# Presentation only: the icon per kind and the colour per category. Everything
# else about a kind comes from the catalogue.
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
}
COLOURS = {
    "boundary": "#1a7f37",
    "reactor": "#0969da",
    "separator": "#9a6700",
    "flow": "#8250df",
    "treatment": "#1b7c83",
    "control": "#bf3989",
}
CATEGORY_LABELS = {
    "boundary": "Boundaries",
    "reactor": "Reactors",
    "separator": "Separators",
    "flow": "Flow handling",
    "treatment": "Treatment",
    "control": "Control",
}
SVG = {p.stem: p.read_text() for p in (HERE / "icons").glob("*.svg")}

server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

# ---------------------------------------------------------------------------
# What aquakin knows (asked once of the worker)
# ---------------------------------------------------------------------------
worker = AquakinWorker()
CATALOG = worker.request("catalog")
KINDS = {k["kind"]: k for k in CATALOG["kinds"]}
MODELS = [m["name"] for m in worker.request("models")]
_model_info: dict[str, dict] = {}


def model_info(name):
    if name not in _model_info:
        _model_info[name] = worker.request("model", name=name)
    return _model_info[name]


def setting(kind, name):
    return next(s for s in KINDS[kind]["settings"] if s["name"] == name)


def ports_for(kind, settings):
    """The unit's ports from the catalogue's port rules (no worker round trip)."""
    k = KINDS[kind]
    full = {s["name"]: s["default"] for s in k["settings"]} | settings
    out = {"input": [], "output": []}
    for r in k["ports"]:
        if r["form"] == "fixed":
            names = r["names"]
        else:
            value = full.get(r["field"])
            if value is None:
                names = []
            elif r["form"] == "name":
                names = [value]
            else:
                names = list(value)
        out[r["side"]] += [str(n) for n in names]
    return out["input"], out["output"]


# ---------------------------------------------------------------------------
# The flowsheet document (what would be saved as flowsheet.yaml)
# ---------------------------------------------------------------------------
units: dict[str, dict] = {}
connections: list[tuple[str, str, str, str]] = []
build_errors: dict[str, str] = {}
# Connection settings the canvas does not edit (a translator, a recycle seed),
# kept by edge id so a spec opened and saved again keeps them.
edge_extras: dict[str, dict] = {}
# The spec's plant-level entries, kept as opened.
plant_meta: dict = {"name": "BSM1-like"}
_plant_check: dict = {"key": None, "result": None}


def check(name):
    """Build the unit in aquakin; record its error, if any."""
    u = units[name]
    try:
        out = worker.request("check_unit", kind=u["kind"], name=name, settings=u["settings"])
    except WorkerError as exc:
        build_errors[name] = str(exc)
        return
    if out["ok"]:
        build_errors.pop(name, None)
        if (out["inputs"], out["outputs"]) != ports_for(u["kind"], u["settings"]):
            build_errors[name] = (
                f"aquakin built ports {out['inputs']} -> {out['outputs']}, not the drawn ones"
            )
    else:
        build_errors[name] = out["error"]


def add_unit(name, kind, x, y, **settings):
    units[name] = dict(kind=kind, settings=settings, position=dict(x=x, y=y))
    check(name)


def bsm1_like():
    T = {"T": 288.15}
    add_unit(
        "influent",
        "influent",
        0,
        140,
        model="asm1",
        Q=18446.0,
        composition=dict(
            SI=30.0,
            SS=69.5,
            XI=51.2,
            XS=202.32,
            XB_H=28.17,
            SNH=31.56,
            SND=6.95,
            XND=10.59,
            SALK=7.0,
        ),
    )
    add_unit(
        "front_mix",
        "mixer",
        230,
        120,
        model="asm1",
        input_port_names=["upstream", "internal_recycle", "ras"],
    )
    for i, x in enumerate((460, 690)):
        add_unit(f"anox{i + 1}", "cstr", x, 140, model="asm1", volume=1000.0, conditions=T)
    for i, (x, kla) in enumerate(zip((920, 1150, 1380), (240.0, 240.0, 84.0))):
        add_unit(
            f"aer{i + 1}",
            "cstr",
            x,
            140,
            model="asm1",
            volume=1333.0,
            conditions=T,
            aeration={"kla": kla},
        )
    add_unit(
        "aer_split",
        "setpoint_splitter",
        1610,
        130,
        model="asm1",
        output_port_flows={"internal_recycle": 55338.0},
        remainder_port="to_clarifier",
    )
    add_unit(
        "clarifier",
        "clarifier",
        1840,
        120,
        model="asm1",
        area=1500.0,
        height=4.0,
        underflow_Q=18831.0,
    )
    add_unit(
        "underflow_split",
        "setpoint_splitter",
        2070,
        250,
        model="asm1",
        output_port_flows={"waste": 385.0},
        remainder_port="ras",
    )
    chain = [
        "influent.out", "front_mix.upstream", "front_mix.out", "anox1.in", "anox1.out", "anox2.in",
        "anox2.out", "aer1.in", "aer1.out", "aer2.in", "aer2.out", "aer3.in", "aer3.out", "aer_split.in",
        "aer_split.to_clarifier", "clarifier.inlet", "clarifier.underflow", "underflow_split.in",
        "aer_split.internal_recycle", "front_mix.internal_recycle", "underflow_split.ras", "front_mix.ras",
    ]  # fmt: skip
    for a, b in zip(chain[::2], chain[1::2]):
        connections.append((*a.split("."), *b.split(".")))


# ---------------------------------------------------------------------------
# Checks, as aquakin's PlantCheck reports them
# ---------------------------------------------------------------------------


def port_status():
    """Unfed inputs (errors) and unconsumed outputs (the plant's product streams)."""
    fed = {(c[2], c[3]) for c in connections}
    used = {(c[0], c[1]) for c in connections}
    unfed, dangling = {}, {}
    for name, u in units.items():
        ins, outs = ports_for(u["kind"], u["settings"])
        unfed[name] = [p for p in ins if (name, p) not in fed]
        dangling[name] = [p for p in outs if (name, p) not in used]
    return unfed, dangling


def recycle_edges():
    """Back edges of a depth-first walk from the influents: the recycle loops."""
    adj: dict[str, list] = {n: [] for n in units}
    for c in connections:
        adj[c[0]].append(c)
    colour, back = {}, set()

    def visit(n):
        colour[n] = 1
        for c in adj[n]:
            if colour.get(c[2]) == 1:
                back.add(c)
            elif c[2] not in colour:
                visit(c[2])
        colour[n] = 2

    for n, u in units.items():
        if u["kind"] == "influent" and n not in colour:
            visit(n)
    for n in units:
        if n not in colour:
            visit(n)
    return back


NODE_HEIGHT = 120  # a generous unit height for lane placement (flow units)


def recycle_lanes(back):
    """A lane below the units each loop spans, stacked so loops do not overlap."""
    lanes = {}
    span = lambda c: units[c[0]]["position"]["x"] - units[c[2]]["position"]["x"]
    for c in sorted(back, key=span):
        x0, x1 = units[c[2]]["position"]["x"], units[c[0]]["position"]["x"]
        spanned = [
            u["position"]["y"] for u in units.values() if x0 - 10 <= u["position"]["x"] <= x1 + 10
        ]
        floor = max(spanned) + NODE_HEIGHT + 20
        lanes[c] = max([floor] + [v + 26 for v in lanes.values()])
    return lanes


def edge_id(c):
    return f"{c[0]}.{c[1]}->{c[2]}.{c[3]}"


def summary(u):
    """A one-line reminder of a unit's main settings."""
    k, s, parts = KINDS[u["kind"]], u["settings"], []
    for st in k["settings"]:
        v = s.get(st["name"])
        if v is None:
            continue
        if st["type"] in ("number", "integer") and st["required"] and isinstance(v, (int, float)):
            parts.append(f"{st['name']} {v:g}")
        elif st["type"] == "aeration" and v:
            parts.append(f"kLa {v['kla']:g}" if "kla" in v else f"DO {v.get('do_setpoint', 0):g}")
        elif st["type"] == "port_map":
            parts += [f"{p} {q:g}" for p, q in v.items()]
    return " · ".join(parts[:3])


def icon_for(u):
    if u["kind"] == "cstr" and u["settings"].get("aeration"):
        return SVG["cstr-aerated"]
    return SVG[ICONS[u["kind"]]]


def spec(layout=True):
    """The flowsheet as an aquakin spec (``Plant.from_spec`` reads it as is)."""
    out = {"format": 1, "name": plant_meta.get("name", "plant")}
    if plant_meta.get("options"):
        out["options"] = plant_meta["options"]
    out["units"] = {n: {"kind": u["kind"], **u["settings"]} for n, u in units.items()}
    out["connections"] = [
        {"from": f"{c[0]}.{c[1]}", "to": f"{c[2]}.{c[3]}", **edge_extras[edge_id(c)]}
        if edge_extras.get(edge_id(c))
        else f"{c[0]}.{c[1]} -> {c[2]}.{c[3]}"
        for c in connections
    ]
    for key in ("streams", "influent_endpoint", "effluent_endpoint"):
        if plant_meta.get(key):
            out[key] = plant_meta[key]
    if layout:
        out["layout"] = {n: [u["position"]["x"], u["position"]["y"]] for n, u in units.items()}
    return out


def plant_check():
    """aquakin's ``Plant.check`` on the whole flowsheet, asked again only when
    the flowsheet (not its layout) changed."""
    s = spec(layout=False)
    key = yaml.safe_dump(s, sort_keys=True)
    if key != _plant_check["key"]:
        try:
            _plant_check["result"] = worker.request("check_spec", spec=s)
        except WorkerError as exc:
            _plant_check["result"] = {"ok": False, "error": str(exc)}
        _plant_check["key"] = key
    return _plant_check["result"]


def _endpoint(text, side):
    """``(unit, port)`` of ``"unit.port"`` or of a bare unit with one port on ``side``."""
    unit, _, port = text.partition(".")
    if not port:
        ins, outs = ports_for(units[unit]["kind"], units[unit]["settings"])
        ports = outs if side == "output" else ins
        if len(ports) != 1:
            raise ValueError(f"'{unit}' has {len(ports)} {side} ports; name one")
        port = ports[0]
    return unit, port


def auto_layout():
    """Left-to-right ranks along the forward (non-recycle) connections."""
    back = recycle_edges()
    rank = dict.fromkeys(units, 0)
    for _ in units:
        for c in connections:
            if c not in back:
                rank[c[2]] = max(rank[c[2]], rank[c[0]] + 1)
    slots: dict[int, int] = {}
    for n in units:
        row = slots.get(rank[n], 0)
        slots[rank[n]] = row + 1
        units[n]["position"] = dict(x=rank[n] * 240, y=row * 170)


def load_spec(data):
    """Replace the flowsheet with an aquakin spec (``Plant.to_spec`` output or a saved file)."""
    units.clear()
    connections.clear()
    build_errors.clear()
    edge_extras.clear()
    plant_meta.clear()
    plant_meta.update(
        {k: v for k, v in data.items() if k not in ("format", "units", "connections", "layout")}
    )
    layout = data.get("layout") or {}
    for name, entry in (data.get("units") or {}).items():
        x, y = layout.get(name, (0, 0))
        units[name] = dict(
            kind=entry["kind"],
            settings={k: v for k, v in entry.items() if k != "kind"},
            position=dict(x=x, y=y),
        )
    for entry in data.get("connections") or []:
        if isinstance(entry, str):
            src, dst, extra = *[p.strip() for p in entry.split("->")], {}
        else:
            src, dst = entry["from"], entry["to"]
            extra = {k: v for k, v in entry.items() if k not in ("from", "to")}
        c = (*_endpoint(src, "output"), *_endpoint(dst, "input"))
        connections.append(c)
        if extra:
            edge_extras[edge_id(c)] = extra
    if not layout:
        auto_layout()
    for name in units:
        check(name)
    state.selected = None
    state.fit_key += 1


@ctrl.add("open_spec")
def open_spec():
    try:
        load_spec(yaml.safe_load(Path(state.spec_path).read_text()))
    except Exception as exc:
        say(f"could not open {state.spec_path}: {type(exc).__name__}: {exc}")
        return
    publish()
    say(f"opened {state.spec_path}", error=False)


@ctrl.add("save_spec")
def save_spec():
    try:
        Path(state.spec_path).write_text(
            yaml.safe_dump(spec(), sort_keys=False, allow_unicode=True)
        )
    except OSError as exc:
        say(f"could not save {state.spec_path}: {exc}")
        return
    say(f"saved {state.spec_path}", error=False)


def publish():
    """Push the document to the page: canvas nodes/edges, checks and the spec."""
    unfed, dangling = port_status()
    back = recycle_edges()
    lanes = recycle_lanes(back)
    nodes = []
    for n, u in units.items():
        k = KINDS[u["kind"]]
        ins, outs = ports_for(u["kind"], u["settings"])
        nodes.append(
            dict(
                id=n,
                position=u["position"],
                selected=n == state.selected,
                data=dict(
                    label=n,
                    kindLabel=k["label"],
                    colour=COLOURS[k["category"]],
                    icon=icon_for(u),
                    inputs=ins,
                    outputs=outs,
                    open=unfed[n],
                    dangling=dangling[n],
                    error=build_errors.get(n),
                    summary=summary(u),
                ),
            )
        )
    state.nodes = nodes
    state.edges = [
        dict(
            id=edge_id(c),
            source=c[0],
            sourceHandle=c[1],
            target=c[2],
            targetHandle=c[3],
            data=dict(recycle=c in back, lane=lanes.get(c)),
        )
        for c in connections
    ]
    errors = [f"{n}: {e}" for n, e in build_errors.items()]
    errors += [f"{n}.{p} is not fed" for n, ps in unfed.items() for p in ps]
    state.n_recycles = len(back)
    if not build_errors:
        # Every unit builds: ask aquakin to build the whole plant and check it.
        result = plant_check()
        if not result["ok"]:
            errors.append(f"plant: {result['error']}")
        else:
            state.n_recycles = len(result["recycles"])
    state.errors = errors
    state.products = [f"{n}.{p}" for n, ps in dangling.items() for p in ps]
    state.spec_text = yaml.safe_dump(spec(), sort_keys=False, allow_unicode=True)
    refresh_selection()


def say(msg, error=True):
    state.message = msg
    state.message_colour = "error" if error else "success"
    state.show_message = True


# ---------------------------------------------------------------------------
# Canvas events (finished edits only)
# ---------------------------------------------------------------------------


def on_connect(c):
    src, sp, dst, dp = c["source"], c["sourceHandle"], c["target"], c["targetHandle"]
    if any(x[2] == dst and x[3] == dp for x in connections):
        say(f"{dst}.{dp} is already fed")
    elif any(x[0] == src and x[1] == sp for x in connections):
        say(f"{src}.{sp} already feeds a unit: add a splitter to divide a stream")
    else:
        connections.append((src, sp, dst, dp))
    publish()


def on_move(moved):
    for m in moved:
        if m["id"] in units:
            units[m["id"]]["position"] = m["position"]
    publish()


def on_remove(r):
    for n in r["nodes"]:
        units.pop(n, None)
        build_errors.pop(n, None)
    gone = set(r["edges"])
    connections[:] = [
        c for c in connections if edge_id(c) not in gone and c[0] in units and c[2] in units
    ]
    if state.selected not in units:
        state.selected = None
    publish()


def default_model():
    used = [u["settings"].get("model") for u in units.values() if u["settings"].get("model")]
    return max(set(used), key=used.count) if used else "asm1"


def on_dropped(d):
    kind = d["kind"]
    i = 1
    while f"{kind}{i}" in units:
        i += 1
    name = f"{kind}{i}"
    settings = {}
    names = {s["name"] for s in KINDS[kind]["settings"]}
    if "model" in names:
        settings["model"] = "adm1" if kind == "digester" else default_model()
        if "conditions" in names:
            conds = model_info(settings["model"])["conditions"]
            settings["conditions"] = {
                c["name"]: c["default"] for c in conds if c["default"] is not None
            }
    add_unit(name, kind, d["position"]["x"], d["position"]["y"], **settings)
    state.selected = name
    state.panel = "properties"
    publish()


def on_select(s):
    state.selected = s["nodes"][0] if len(s["nodes"]) == 1 else None
    refresh_selection()


# ---------------------------------------------------------------------------
# Properties: a form generated from the catalogue's settings
# ---------------------------------------------------------------------------


def _text(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        return ", ".join(
            f"{k}: {v:g}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in value.items()
        )
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _placeholder(s):
    if s["required"]:
        return "required"
    if s["default"] in (None, {}, []):
        return "not set"
    return f"{_text(s['default'])} (default)"


def _row(key, label, widget="text", value="", **extra):
    return (
        dict(key=key, label=label, widget=widget, value=value, items=[], hint="", placeholder="")
        | extra
    )


def rows_for(name):
    u = units[name]
    settings = u["settings"]
    model = settings.get("model")
    info = model_info(model) if model in MODELS else None
    species = [s["name"] for s in info["species"]] if info else []
    rows = [_row("name", "name", value=name)]
    for s in KINDS[u["kind"]]["settings"]:
        key, value = s["name"], settings.get(s["name"])
        label = key + (" *" if s["required"] else "")
        common = dict(hint=s["help"], placeholder=_placeholder(s))
        t = s["type"]
        if t == "model":
            rows.append(_row(key, label, "select", value or "", **common) | {"items": MODELS})
        elif t == "conditions":
            for c in info["conditions"] if info else []:
                v = (value or {}).get(c["name"])
                rows.append(
                    _row(
                        f"conditions.{c['name']}",
                        f"condition {c['name']}",
                        value=_text(v),
                        hint=s["help"],
                        placeholder=f"{c['default']:g} (model default)"
                        if c["default"] is not None
                        else "required",
                    )
                )
        elif t == "aeration":
            mode = next((m for m in ("kla", "do_setpoint") if value and m in value), "none")
            rows.append(
                _row("aeration.mode", "aeration", "select", mode, hint=s["help"])
                | {"items": ["none", "kla", "do_setpoint"]}
            )
            if mode != "none":
                rows.append(_row("aeration.value", mode, value=_text(value[mode]), hint=s["help"]))
        elif t == "species" and species:
            rows.append(_row(key, label, "select", value or "", **common) | {"items": species})
        elif t == "species_list" and species:
            rows.append(_row(key, label, "multi", list(value or []), **common) | {"items": species})
        elif t == "boolean":
            rows.append(
                _row(key, label, "select", "" if value is None else str(value).lower(), **common)
                | {"items": ["true", "false"]}
            )
        elif t == "reagent":
            rows.append(
                _row(
                    key,
                    label,
                    "readonly",
                    "",
                    hint=s["help"],
                    placeholder=(
                        f"{value['label']}: {_text(value.get('composition', {}))}"
                        if value
                        else "set in the spec file"
                    ),
                )
            )
        elif t == "unsupported":
            rows.append(
                _row(
                    key,
                    label,
                    "readonly",
                    "",
                    hint=s["help"],
                    placeholder=f"set from Python ({s['python_type']})",
                )
            )
        else:
            rows.append(_row(key, label, value=_text(value), **common))
    return rows


def refresh_selection():
    if state.selected not in units:
        state.sel_rows = []
        return
    u = units[state.selected]
    k = KINDS[u["kind"]]
    state.sel_kind = k["label"]
    state.sel_summary = k["summary"]
    state.sel_icon = icon_for(u)
    state.sel_error = build_errors.get(state.selected)
    state.sel_rows = rows_for(state.selected)
    state.focus_help = ""


def _parse(s, text):
    """A form value back to plain data, by setting type (empty -> unset)."""
    t = s["type"]
    text = text.strip() if isinstance(text, str) else text
    if text in ("", None, []):
        return None
    if t == "number":
        return text if text.lower() in ("inf", "-inf") else float(text)
    if t == "integer":
        return int(text)
    if t == "boolean":
        return text == "true"
    if t == "port_names":
        return [p.strip() for p in text.split(",") if p.strip()]
    if t in ("port_map", "species_values", "number_map"):
        pairs = [p.split(":") for p in text.split(",") if p.strip()]
        return {k.strip(): float(v) for k, v in pairs}
    return text


def apply_properties(rows=None):
    """Read the form's rows back into the selected unit's settings.

    The form is edited in the browser and sent whole (``rows``) on Apply, so no
    keystroke waits on the server or is overwritten by it.
    """
    old = state.selected
    if old not in units:
        return
    u = units[old]
    values = {r["key"]: r["value"] for r in (rows if rows is not None else state.sel_rows)}
    new_settings = {}
    try:
        for s in KINDS[u["kind"]]["settings"]:
            t = s["type"]
            if t == "conditions":
                conds = {
                    k.split(".", 1)[1]: float(v)
                    for k, v in values.items()
                    if k.startswith("conditions.") and str(v).strip()
                }
                if conds:
                    new_settings["conditions"] = conds
            elif t == "aeration":
                mode = values.get("aeration.mode", "none")
                if mode != "none":
                    raw = values.get("aeration.value", "")
                    old_aer = u["settings"].get("aeration") or {}
                    num = float(raw) if str(raw).strip() else old_aer.get(mode, 0.0)
                    # Keep the aeration settings the form does not edit (gains, ...).
                    kept = {k: v for k, v in old_aer.items() if k not in ("kla", "do_setpoint")}
                    new_settings["aeration"] = {**kept, mode: num}
            elif t in ("unsupported", "reagent"):
                # Shown read-only: keep what the spec gave.
                if s["name"] in u["settings"]:
                    new_settings[s["name"]] = u["settings"][s["name"]]
            elif s["name"] in values:
                v = _parse(s, values[s["name"]])
                if v is not None:
                    new_settings[s["name"]] = v
    except ValueError as exc:
        say(f"could not read a value: {exc}")
        return
    # Connections to ports that no longer exist go with them.
    ins, outs = ports_for(u["kind"], new_settings)
    connections[:] = [
        c
        for c in connections
        if not ((c[0] == old and c[1] not in outs) or (c[2] == old and c[3] not in ins))
    ]
    u["settings"] = new_settings
    new = str(values["name"]).strip()
    if new and new != old:
        if new in units:
            say(f"a unit named {new} already exists")
        else:
            units[new] = units.pop(old)
            build_errors.pop(old, None)
            connections[:] = [
                (new if a == old else a, b, new if c == old else c, d) for a, b, c, d in connections
            ]
            state.selected = new
    check(state.selected)
    publish()


def reshape_form(rows, i, value):
    """A choice that changes which rows the form has (aeration mode, model)."""
    rows = [dict(r) for r in rows]
    rows[i]["value"] = value
    apply_properties(rows)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

state.trame__title = "aquakin flowsheet"
state.selected = None
state.show_message = False
state.message = ""
state.panel = "properties"
state.dark = False
state.focus_help = ""
state.focus_label = ""
state.spec_path = "flowsheet.yaml"
state.message_colour = "error"
state.fit_key = 0
state.palette = [
    dict(
        category=CATEGORY_LABELS[cat],
        kinds=[
            dict(
                kind=k["kind"],
                label=k["label"],
                summary=k["summary"],
                icon=SVG[ICONS[k["kind"]]],
                colour=COLOURS[cat],
            )
            for k in CATALOG["kinds"]
            if k["category"] == cat
        ],
    )
    for cat in CATALOG["categories"]
]
bsm1_like()
publish()

PAGE_CSS = """
.ak-pal-item { cursor: grab; user-select: none; display: flex; align-items: center; gap: 10px;
  padding: 5px 10px; margin: 1px 8px; border-radius: 6px; font-size: 13px; }
.ak-pal-item:hover { background: rgba(127,127,127,.12); }
.ak-pal-icon { width: 20px; height: 20px; flex: none; display: inline-flex; }
.ak-pal-icon svg { width: 20px; height: 20px; }
.ak-sel-icon svg { width: 28px; height: 28px; }
"""

with SinglePageWithDrawerLayout(server, theme=("dark ? 'dark' : 'light'",)) as layout:
    layout.title.set_text("aquakin · flowsheet")
    layout.footer.hide()
    with layout.toolbar:
        v3.VSpacer()
        v3.VTextField(
            v_model=("spec_path",),
            density="compact",
            variant="outlined",
            hide_details=True,
            prepend_inner_icon="mdi-file-outline",
            style="max-width: 260px;",
            classes="mr-2",
        )
        v3.VBtn("Open", click=ctrl.open_spec, variant="tonal", classes="mr-1 text-none")
        v3.VBtn("Save", click=ctrl.save_spec, variant="tonal", classes="mr-4 text-none")
        v3.VChip(
            "{{ nodes.length }} units · {{ edges.length }} connections · {{ n_recycles }} recycles",
            size="small",
            variant="tonal",
            classes="mr-2",
        )
        v3.VBtn(icon="mdi-theme-light-dark", click="dark = !dark", variant="text")

    with layout.drawer as drawer:
        drawer.width = 250
        client.Style(PAGE_CSS)
        html.Div(
            "Drag a unit onto the canvas",
            classes="text-caption text-medium-emphasis px-4 pt-3 pb-1",
        )
        with html.Div(v_for="group in palette", key="group.category", classes="pb-1"):
            html.Div(
                "{{ group.category }}", classes="text-overline px-4", style="line-height: 2rem;"
            )
            with html.Div(
                v_for="k in group.kinds",
                key="k.kind",
                classes="ak-pal-item",
                draggable="true",
                title=("k.summary",),
                __events=["dragstart"],
                dragstart=f"$event.dataTransfer.setData('{DRAG_TYPE}', k.kind); "
                "$event.dataTransfer.effectAllowed = 'copy'",
            ):
                html.Span(v_html="k.icon", classes="ak-pal-icon", style=("{ color: k.colour }",))
                html.Span("{{ k.label }}")

    with layout.content:
        with html.Div(style="display: flex; height: calc(100vh - 64px);"):
            with html.Div(style="flex: 1; position: relative;"):
                Flowsheet(
                    nodes=("nodes",),
                    edges=("edges",),
                    drag_type=DRAG_TYPE,
                    fit_key=("fit_key",),
                    connect=(on_connect, "[$event]"),
                    move=(on_move, "[$event]"),
                    remove=(on_remove, "[$event]"),
                    dropped=(on_dropped, "[$event]"),
                    select=(on_select, "[$event]"),
                    style="position: absolute; inset: 0;",
                )
            with v3.VSheet(width=340, border="s", style="overflow: auto; flex: none;"):
                with v3.VTabs(v_model=("panel",), density="compact", grow=True):
                    v3.VTab("Unit", value="properties", classes="text-none", min_width=0)
                    v3.VTab(
                        "Checks ({{ errors.length }})",
                        value="checks",
                        classes="text-none",
                        min_width=0,
                    )
                    v3.VTab("Spec", value="spec", classes="text-none", min_width=0)
                with v3.VWindow(v_model=("panel",)):
                    with v3.VWindowItem(value="properties", classes="pa-3"):
                        html.Div(
                            "Select a unit to edit it.",
                            v_if="!selected",
                            classes="text-body-2 text-medium-emphasis",
                        )
                        with html.Div(v_else=True):
                            with html.Div(classes="d-flex align-center ga-2 mb-1"):
                                html.Span(v_html="sel_icon", classes="ak-sel-icon d-inline-flex")
                                html.Div(
                                    "{{ sel_kind }}", classes="text-subtitle-1 font-weight-medium"
                                )
                            html.Div(
                                "{{ sel_summary }}",
                                classes="text-caption text-medium-emphasis mb-3",
                            )
                            v3.VBtn(
                                "Apply",
                                click=(apply_properties, "[sel_rows]"),
                                color="primary",
                                block=True,
                                variant="flat",
                                classes="mb-3",
                            )
                            v3.VAlert(
                                "{{ sel_error }}",
                                v_if="sel_error",
                                type="error",
                                variant="tonal",
                                density="compact",
                                classes="mb-3 text-caption",
                            )
                            with html.Div(v_for="(row, i) in sel_rows", key="row.key"):
                                common = dict(
                                    label=("row.label",),
                                    hide_details=True,
                                    update_focused="if ($event) { focus_label = row.label; focus_help = row.hint }",
                                    placeholder=("row.placeholder",),
                                    persistent_placeholder=True,
                                    density="compact",
                                    variant="outlined",
                                    classes="mb-2",
                                )
                                v3.VSelect(
                                    v_if="row.widget === 'select' && ['aeration.mode', 'model'].includes(row.key)",
                                    items=("row.items",),
                                    model_value=("row.value",),
                                    update_modelValue=(reshape_form, "[sel_rows, i, $event]"),
                                    **common,
                                )
                                v3.VSelect(
                                    v_else_if="row.widget === 'select'",
                                    items=("row.items",),
                                    v_model="row.value",
                                    **common,
                                )
                                v3.VSelect(
                                    v_else_if="row.widget === 'multi'",
                                    items=("row.items",),
                                    multiple=True,
                                    chips=True,
                                    closable_chips=True,
                                    v_model="row.value",
                                    **common,
                                )
                                v3.VTextField(
                                    v_else_if="row.widget === 'readonly'",
                                    disabled=True,
                                    model_value="",
                                    **common,
                                )
                                v3.VTextField(v_else=True, v_model="row.value", **common)
                            with v3.VSheet(
                                v_if="focus_help",
                                rounded=True,
                                border=True,
                                classes="pa-2 mt-1 text-caption",
                            ):
                                html.Div("{{ focus_label }}", classes="font-weight-medium mb-1")
                                html.Div("{{ focus_help }}", classes="text-medium-emphasis")
                    with v3.VWindowItem(value="checks", classes="pa-3"):
                        html.Div("Errors", classes="text-overline")
                        html.Div(
                            "None: aquakin built every unit.",
                            v_if="!errors.length",
                            classes="text-body-2 mb-2",
                        )
                        with v3.VList(density="compact", v_else=True):
                            v3.VListItem(
                                v_for="c in errors",
                                key="c",
                                subtitle=("c",),
                                prepend_icon="mdi-alert-circle-outline",
                                base_color="error",
                                lines="three",
                            )
                        html.Div(
                            "Product streams (unconnected outputs)", classes="text-overline mt-2"
                        )
                        with v3.VList(density="compact"):
                            v3.VListItem(
                                v_for="p in products",
                                key="p",
                                title=("p",),
                                prepend_icon="mdi-arrow-right-bold-outline",
                            )
                    with v3.VWindowItem(value="spec", classes="pa-3"):
                        html.Div(
                            "flowsheet.yaml (as it would be saved)", classes="text-caption mb-1"
                        )
                        html.Pre(
                            "{{ spec_text }}",
                            style="font-size: 11px; white-space: pre; overflow-x: auto;",
                        )
        v3.VSnackbar(
            "{{ message }}", v_model=("show_message",), timeout=4000, color=("message_colour",)
        )

if __name__ == "__main__":
    server.start(open_browser=False)
    worker.stop()
