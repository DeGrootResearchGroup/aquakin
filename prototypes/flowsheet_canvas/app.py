"""Prototype: drag-and-drop flowsheet editor on the Vue Flow trame widget.

Pure Python state; aquakin is not imported (the UI talks to aquakin through a
worker process in the real design). Run: ``python app.py --port 8080``.
"""

import yaml
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import html
from trame.widgets import vuetify3 as v3
from trame_flowsheet import Flowsheet

DRAG_TYPE = "application/x-aquakin-unit"

# What the aquakin worker's unit catalogue would answer. Port lists marked
# configurable come from the unit's own settings, as on aquakin's mixers and
# splitters.
CATALOG = {
    "influent": dict(
        label="Influent",
        colour="#1a7f37",
        inputs=[],
        outputs=["out"],
        fields={"series": "constant"},
    ),
    "cstr": dict(
        label="CSTR",
        colour="#0969da",
        inputs=["in"],
        outputs=["out"],
        fields={"volume_m3": 1000.0, "aeration": "none"},
    ),
    "mixer": dict(
        label="Mixer",
        colour="#8250df",
        inputs=["upstream", "recycle"],
        outputs=["out"],
        fields={},
        configurable="inputs",
    ),
    "splitter": dict(
        label="Splitter",
        colour="#bf3989",
        inputs=["in"],
        outputs=["a", "b"],
        fields={"ratios": "0.5, 0.5"},
        configurable="outputs",
    ),
    "clarifier": dict(
        label="Clarifier (Takács)",
        colour="#9a6700",
        inputs=["in"],
        outputs=["overflow", "underflow"],
        fields={"area_m2": 1500.0, "layers": 10},
    ),
    "sink": dict(label="Product stream", colour="#57606a", inputs=["in"], outputs=[], fields={}),
}

server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

# ---------------------------------------------------------------------------
# The flowsheet document (what would be saved as flowsheet.yaml)
# ---------------------------------------------------------------------------
units: dict[str, dict] = {}
connections: list[tuple[str, str, str, str]] = []


def add_unit(name, kind, x, y, **overrides):
    spec = CATALOG[kind]
    units[name] = dict(
        kind=kind,
        inputs=list(overrides.pop("inputs", spec["inputs"])),
        outputs=list(overrides.pop("outputs", spec["outputs"])),
        settings={**spec["fields"], **overrides},
        position=dict(x=x, y=y),
    )


def bsm1_like():
    add_unit("influent", "influent", 0, 140)
    add_unit("front_mix", "mixer", 220, 120, inputs=["upstream", "internal_recycle", "ras"])
    for i, x in enumerate((440, 660)):
        add_unit(f"anox{i + 1}", "cstr", x, 140, volume_m3=1000.0)
    for i, x in enumerate((880, 1100, 1320)):
        add_unit(f"aer{i + 1}", "cstr", x, 140, volume_m3=1333.0, aeration="KLa")
    add_unit(
        "aer_split",
        "splitter",
        1540,
        130,
        outputs=["to_clarifier", "internal_recycle"],
        ratios="0.25, 0.75",
    )
    add_unit("clarifier", "clarifier", 1760, 120)
    add_unit("effluent", "sink", 2000, 90)
    add_unit(
        "underflow_split", "splitter", 1760, 330, outputs=["ras", "waste"], ratios="0.98, 0.02"
    )
    add_unit("wastage", "sink", 2000, 360)
    chain = [
        "influent.out",
        "front_mix.upstream",
        "front_mix.out",
        "anox1.in",
        "anox1.out",
        "anox2.in",
        "anox2.out",
        "aer1.in",
        "aer1.out",
        "aer2.in",
        "aer2.out",
        "aer3.in",
        "aer3.out",
        "aer_split.in",
        "aer_split.to_clarifier",
        "clarifier.in",
        "clarifier.overflow",
        "effluent.in",
        "clarifier.underflow",
        "underflow_split.in",
        "underflow_split.waste",
        "wastage.in",
        "aer_split.internal_recycle",
        "front_mix.internal_recycle",
        "underflow_split.ras",
        "front_mix.ras",
    ]
    for a, b in zip(chain[::2], chain[1::2]):
        connections.append((*a.split("."), *b.split(".")))


# Sensors are a study overlay, shown on the canvas but not part of the flowsheet.
SENSORS = {("clarifier", "overflow"): "sensor · S_NH, S_NO · σ 0.5"}  # noqa: RUF001

# ---------------------------------------------------------------------------
# Checks: what the canvas shows that the flowsheet itself does not store
# ---------------------------------------------------------------------------


def open_ports():
    fed = {(c[2], c[3]) for c in connections}
    used = {(c[0], c[1]) for c in connections}
    out = {}
    for name, u in units.items():
        out[name] = [p for p in u["inputs"] if (name, p) not in fed] + [
            p for p in u["outputs"] if (name, p) not in used
        ]
    return out


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


NODE_HEIGHT = 110  # a generous unit height for lane placement (flow units)


def recycle_lanes(back):
    """A lane below the units each loop spans, stacked so loops do not overlap."""
    lanes = {}
    for k, c in enumerate(
        sorted(back, key=lambda c: units[c[0]]["position"]["x"] - units[c[2]]["position"]["x"])
    ):
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
    s = u["settings"]
    if u["kind"] == "cstr":
        return f"{s['volume_m3']:g} m³ · {'aerated' if s['aeration'] != 'none' else 'unaerated'}"
    if u["kind"] == "splitter":
        return f"ratios {s['ratios']}"
    if u["kind"] == "clarifier":
        return f"{s['area_m2']:g} m² · {s['layers']} layers"
    return ""


def publish():
    """Push the document to the page: canvas nodes/edges, checks and the spec."""
    opens, back = open_ports(), recycle_edges()
    lanes = recycle_lanes(back)
    state.nodes = [
        dict(
            id=n,
            position=u["position"],
            data=dict(
                label=n,
                kindLabel=CATALOG[u["kind"]]["label"],
                colour=CATALOG[u["kind"]]["colour"],
                inputs=u["inputs"],
                outputs=u["outputs"],
                open=opens[n],
                summary=summary(u),
            ),
        )
        for n, u in units.items()
    ]
    state.edges = [
        dict(
            id=edge_id(c),
            source=c[0],
            sourceHandle=c[1],
            target=c[2],
            targetHandle=c[3],
            data=dict(recycle=c in back, lane=lanes.get(c), label=SENSORS.get((c[0], c[1]))),
        )
        for c in connections
    ]
    state.checks = [f"{n}.{p} is not connected" for n, ps in opens.items() for p in ps]
    state.n_recycles = len(back)
    state.spec_text = yaml.safe_dump(
        dict(
            units={
                n: dict(
                    kind=u["kind"],
                    **(
                        {"inputs": u["inputs"]}
                        if CATALOG[u["kind"]].get("configurable") == "inputs"
                        else {}
                    ),
                    **(
                        {"outputs": u["outputs"]}
                        if CATALOG[u["kind"]].get("configurable") == "outputs"
                        else {}
                    ),
                    **u["settings"],
                )
                for n, u in units.items()
            },
            connections=[f"{c[0]}.{c[1]} -> {c[2]}.{c[3]}" for c in connections],
            layout={n: [u["position"]["x"], u["position"]["y"]] for n, u in units.items()},
        ),
        sort_keys=False,
        allow_unicode=True,
    )
    refresh_selection()


def say(msg):
    state.message = msg
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
    gone = set(r["edges"])
    connections[:] = [
        c for c in connections if edge_id(c) not in gone and c[0] in units and c[2] in units
    ]
    if state.selected not in units:
        state.selected = None
    publish()


def on_dropped(d):
    kind = d["kind"]
    i = 1
    while f"{kind}{i}" in units:
        i += 1
    add_unit(f"{kind}{i}", kind, d["position"]["x"], d["position"]["y"])
    state.selected = f"{kind}{i}"
    publish()


def on_select(s):
    state.selected = s["nodes"][0] if len(s["nodes"]) == 1 else None
    refresh_selection()


# ---------------------------------------------------------------------------
# Properties panel
# ---------------------------------------------------------------------------


def refresh_selection():
    u = units.get(state.selected)
    if u is None:
        state.sel_rows = []
        return
    rows = [dict(key="name", value=state.selected)]
    conf = CATALOG[u["kind"]].get("configurable")
    if conf:
        rows.append(dict(key=conf, value=", ".join(u[conf])))
    rows += [dict(key=k, value=str(v)) for k, v in u["settings"].items()]
    state.sel_kind = CATALOG[u["kind"]]["label"]
    state.sel_rows = rows


@ctrl.add("apply_properties")
def apply_properties():
    old = state.selected
    u = units[old]
    for row in state.sel_rows:
        k, val = row["key"], row["value"]
        if k in ("inputs", "outputs"):
            ports = [p.strip() for p in val.split(",") if p.strip()]
            removed = set(u[k]) - set(ports)
            u[k] = ports
            side = 1 if k == "outputs" else 3
            unit_at = 0 if k == "outputs" else 2
            connections[:] = [
                c for c in connections if not (c[unit_at] == old and c[side] in removed)
            ]
        elif k in u["settings"]:
            cur = u["settings"][k]
            try:
                u["settings"][k] = type(cur)(val) if not isinstance(cur, str) else val
            except ValueError:
                say(f"{k}: '{val}' is not a {type(cur).__name__}")
    new = next(r["value"] for r in state.sel_rows if r["key"] == "name").strip()
    if new and new != old:
        if new in units:
            say(f"a unit named {new} already exists")
        else:
            units[new] = units.pop(old)
            connections[:] = [
                (new if a == old else a, b, new if c == old else c, d) for a, b, c, d in connections
            ]
            state.selected = new
    publish()


@ctrl.add("set_row")
def set_row(i, value):
    rows = list(state.sel_rows)
    rows[i] = dict(rows[i], value=value)
    state.sel_rows = rows


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

state.trame__title = "aquakin flowsheet"
state.selected = None
state.show_message = False
state.message = ""
state.panel = "properties"
state.dark = False
bsm1_like()
publish()

with SinglePageWithDrawerLayout(server, theme=("dark ? 'dark' : 'light'",)) as layout:
    layout.title.set_text("aquakin · flowsheet")
    layout.footer.hide()
    with layout.toolbar:
        v3.VSpacer()
        v3.VChip(
            "{{ nodes.length }} units · {{ edges.length }} connections · {{ n_recycles }} recycles",
            size="small",
            variant="tonal",
            classes="mr-2",
        )
        v3.VBtn(icon="mdi-theme-light-dark", click="dark = !dark", variant="text")

    with layout.drawer as drawer:
        drawer.width = 230
        html.Div("Units", classes="text-overline px-4 pt-3")
        html.Div("Drag onto the canvas", classes="text-caption text-medium-emphasis px-4 pb-2")
        for kind, spec in CATALOG.items():
            with html.Div(
                draggable="true",
                __events=["dragstart"],
                dragstart=f"$event.dataTransfer.setData('{DRAG_TYPE}', '{kind}'); "
                f"$event.dataTransfer.effectAllowed = 'copy'",
                classes="mx-3 my-1 px-3 py-2 rounded border d-flex align-center",
                style=f"cursor: grab; border-left: 4px solid {spec['colour']} !important; user-select: none;",
            ):
                html.Span(spec["label"], classes="text-body-2")

    with layout.content:
        with html.Div(style="display: flex; height: calc(100vh - 64px);"):
            with html.Div(style="flex: 1; position: relative;"):
                Flowsheet(
                    nodes=("nodes",),
                    edges=("edges",),
                    drag_type=DRAG_TYPE,
                    connect=(on_connect, "[$event]"),
                    move=(on_move, "[$event]"),
                    remove=(on_remove, "[$event]"),
                    dropped=(on_dropped, "[$event]"),
                    select=(on_select, "[$event]"),
                    style="position: absolute; inset: 0;",
                )
            with v3.VSheet(width=320, border="s", style="overflow: auto; flex: none;"):
                with v3.VTabs(v_model=("panel",), density="compact", grow=True):
                    v3.VTab("Unit", value="properties", classes="text-none", min_width=0)
                    v3.VTab(
                        "Checks ({{ checks.length }})",
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
                            html.Div("{{ sel_kind }}", classes="text-overline")
                            v3.VTextField(
                                v_for="(row, i) in sel_rows",
                                key="row.key",
                                label=("row.key",),
                                model_value=("row.value",),
                                update_modelValue=(set_row, "[i, $event]"),
                                density="compact",
                                variant="outlined",
                                hide_details=True,
                                classes="mb-2",
                            )
                            v3.VBtn(
                                "Apply",
                                click=ctrl.apply_properties,
                                color="primary",
                                block=True,
                                variant="flat",
                            )
                    with v3.VWindowItem(value="checks", classes="pa-3"):
                        html.Div("No problems found.", v_if="!checks.length", classes="text-body-2")
                        with v3.VList(density="compact", v_else=True):
                            v3.VListItem(
                                v_for="c in checks",
                                key="c",
                                title=("c",),
                                prepend_icon="mdi-alert-circle-outline",
                                base_color="error",
                            )
                    with v3.VWindowItem(value="spec", classes="pa-3"):
                        html.Div(
                            "flowsheet.yaml (as it would be saved)", classes="text-caption mb-1"
                        )
                        html.Pre(
                            "{{ spec_text }}",
                            style="font-size: 11px; white-space: pre; overflow-x: auto;",
                        )
        v3.VSnackbar("{{ message }}", v_model=("show_message",), timeout=3500, color="error")

if __name__ == "__main__":
    server.start(open_browser=False)
