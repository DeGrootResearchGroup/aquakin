"""The flowsheet editor page: palette, canvas, unit form, checks and spec.

Every unit kind, setting, default, help text and port comes from aquakin's unit
catalogue, asked of an ``aquakin serve`` worker (:mod:`aquakin_ui.worker`); the
page imports neither aquakin nor JAX. The flowsheet itself is a
:class:`~aquakin_ui.document.Flowsheet`; the page draws it and turns finished
canvas edits into document changes. aquakin's checks (each unit built from its
settings, then the whole plant from the spec) run in the background after an
edit, so the canvas never waits on them.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yaml
from trame.app import asynchronous
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import client, html
from trame.widgets import vuetify3 as v3

from aquakin_ui import forms
from aquakin_ui.catalog import Catalog
from aquakin_ui.document import Flowsheet, edge_id
from aquakin_ui.icons import CATEGORY_LABELS, COLOURS, ICONS, SVG, icon_name
from aquakin_ui.widget import Flowsheet as FlowsheetCanvas
from aquakin_ui.worker import AquakinWorker, WorkerError

DRAG_TYPE = "application/x-aquakin-unit"
CHECK_DELAY = 0.3  # seconds of quiet before aquakin's checks run

PAGE_CSS = """
.ak-pal-item { cursor: grab; user-select: none; display: flex; align-items: center; gap: 10px;
  padding: 5px 10px; margin: 1px 8px; border-radius: 6px; font-size: 13px; }
.ak-pal-item:hover { background: rgba(127,127,127,.12); }
.ak-pal-icon { width: 20px; height: 20px; flex: none; display: inline-flex; }
.ak-pal-icon svg { width: 20px; height: 20px; }
.ak-sel-icon svg { width: 28px; height: 28px; }
"""


class Editor:
    """The editor on one trame server.

    Parameters
    ----------
    server : trame_server.Server
        The server to build the page on (``client_type="vue3"``).
    worker : AquakinWorker, optional
        The line to aquakin (default: a new ``aquakin serve`` process).
    path : str or pathlib.Path, optional
        A spec file to open; the editor starts empty otherwise.
    """

    def __init__(self, server, worker: AquakinWorker | None = None, path=None):
        self.server = server
        self.state = server.state
        self.worker = worker or AquakinWorker()
        self.catalog = Catalog(self.worker.request("catalog"), self.worker.request("models"))
        self.doc = Flowsheet(self.catalog)
        self._model_info: dict = {}
        self._unit_checks: dict[str, tuple[str, str | None]] = {}
        self._plant_check: tuple[str | None, dict | None] = (None, None)
        self._check_task = None
        self._pending_rows = None
        self._init_state(path)
        if path is not None:
            self.open(path, quiet=True)
        self.publish(checks=False)
        self.server.controller.on_server_ready.add(lambda **_: self.schedule_checks())
        self._build()

    # -- aquakin ------------------------------------------------------------

    def model_info(self, name: str) -> dict | None:
        """The worker's description of a model (cached), or ``None``."""
        path = self.doc.model_files().get(name)
        key = (name, path)
        if key not in self._model_info:
            try:
                self._model_info[key] = self.worker.request("model", name=name, path=path)
            except WorkerError:
                self._model_info[key] = None
        return self._model_info[key]

    def _check_unit(self, uid: str) -> str | None:
        u = self.doc.units[uid]
        key = json.dumps([u.kind, u.settings, self.doc.model_files()], sort_keys=True)
        if self._unit_checks.get(uid, (None,))[0] == key:
            return self._unit_checks[uid][1]
        try:
            out = self.worker.request(
                "check_unit",
                kind=u.kind,
                name=uid,
                settings=u.settings,
                model_files=self.doc.model_files(),
                base_dir=str(self.doc.base_dir) if self.doc.base_dir else None,
            )
            error = None if out["ok"] else out["error"]
            if out["ok"] and (out["inputs"], out["outputs"]) != self.doc.ports(uid):
                error = (
                    f"aquakin built ports {out['inputs']} -> {out['outputs']}, not the drawn ones"
                )
        except WorkerError as exc:
            error = str(exc)
        self._unit_checks[uid] = (key, error)
        return error

    def _check_plant(self) -> dict | None:
        spec = self.doc.to_spec(layout=False)
        key = json.dumps(spec, sort_keys=True)
        if self._plant_check[0] != key:
            base = str(self.doc.base_dir) if self.doc.base_dir else None
            try:
                result = self.worker.request("check_spec", spec=spec, base_dir=base)
            except WorkerError as exc:
                result = {"ok": False, "error": str(exc)}
            self._plant_check = (key, result)
        return self._plant_check[1]

    def unit_errors(self) -> dict[str, str]:
        """The latest aquakin error of each unit that failed to build."""
        return {uid: e for uid, (_, e) in self._unit_checks.items() if e and uid in self.doc.units}

    def schedule_checks(self) -> None:
        """Run aquakin's checks after a short quiet spell, off the page's thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no page served yet: the checks run once the server is ready
        if self._check_task is not None:
            self._check_task.cancel()
        self.state.checking = True
        self._check_task = asynchronous.create_task(self._run_checks())

    async def _run_checks(self) -> None:
        await asyncio.sleep(CHECK_DELAY)
        loop = asyncio.get_running_loop()
        for uid in list(self.doc.units):
            if uid in self.doc.units:
                await loop.run_in_executor(None, self._check_unit, uid)
        if not self.unit_errors():
            await loop.run_in_executor(None, self._check_plant)
        with self.state:
            self.state.checking = False
            self.state.checked = True
            self.publish(checks=False)

    # -- drawing ------------------------------------------------------------

    def publish(self, *, checks: bool = True) -> None:
        """Push the document to the page; with ``checks``, rerun aquakin's checks."""
        unit_errors = self.unit_errors()
        plant = self._plant_check[1] if not unit_errors else None
        recycles = plant["recycles"] if plant and plant.get("ok") else None
        back = self.doc.back_edges(recycles)
        lanes = self.doc.lanes(back)
        unfed, dangling = self.doc.port_status()
        nodes = []
        for uid, u in self.doc.units.items():
            k = self.catalog.kind(u.kind)
            ins, outs = self.doc.ports(uid)
            nodes.append(
                {
                    "id": uid,
                    "position": u.position,
                    "selected": uid == self.state.selected,
                    "data": {
                        "label": u.label or uid,
                        "kindLabel": k["label"] if not u.label else f"{uid} · {k['label']}",
                        "colour": COLOURS.get(k["category"], "#57606a"),
                        "icon": icon_name(u.kind, u.settings),
                        "inputs": ins,
                        "outputs": outs,
                        "open": unfed[uid],
                        "dangling": dangling[uid],
                        "error": unit_errors.get(uid),
                        "summary": self._summary(uid),
                    },
                }
            )
        self.state.nodes = nodes
        self.state.edges = [
            {
                "id": edge_id(c),
                "source": c[0],
                "sourceHandle": c[1],
                "target": c[2],
                "targetHandle": c[3],
                "data": {"recycle": c in back, "lane": lanes.get(c)},
            }
            for c in self.doc.connections
        ]
        errors = [f"{uid}: {e}" for uid, e in unit_errors.items()]
        errors += [f"{uid}.{p} is not fed" for uid, ps in unfed.items() for p in ps]
        if plant and not plant.get("ok"):
            errors.append(f"plant: {plant['error']}")
        self.state.errors = errors
        self.state.products = [f"{uid}.{p}" for uid, ps in dangling.items() for p in ps]
        self.state.n_recycles = len(back)
        self.state.plant_name = self.doc.meta.get("name", "plant")
        self._publish_spec()
        self.refresh_selection()
        if checks:
            self.schedule_checks()

    def _publish_spec(self) -> None:
        if self.state.panel == "spec":
            spec = self.doc.to_spec()
            self.state.spec_text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)

    def _summary(self, uid: str) -> str:
        """A one-line reminder of a unit's main settings."""
        u = self.doc.units[uid]
        parts = []
        for s in self.catalog.settings(u.kind):
            v = u.settings.get(s["name"])
            if v is None:
                continue
            if s["type"] in ("number", "integer") and s["required"] and isinstance(v, (int, float)):
                parts.append(f"{s['name']} {v:g}")
            elif s["type"] == "aeration" and v:
                if v.get("kla") is not None:
                    parts.append(f"kLa {v['kla']:g}")
                elif v.get("do_setpoint") is not None:
                    parts.append(f"DO {v['do_setpoint']:g}")
            elif s["type"] == "port_map":
                parts += [f"{p} {q:g}" for p, q in v.items() if isinstance(q, (int, float))]
        return " · ".join(parts[:3])

    def say(self, message: str, error: bool = True) -> None:
        """Show a message in the snackbar."""
        self.state.message = message
        self.state.message_colour = "error" if error else "success"
        self.state.show_message = True

    # -- canvas events (finished edits only) --------------------------------

    def on_connect(self, c: dict) -> None:
        why = self.doc.connect(c["source"], c["sourceHandle"], c["target"], c["targetHandle"])
        if why:
            self.say(why)
        self.publish()

    def on_move(self, moved: list[dict]) -> None:
        for m in moved:
            if m["id"] in self.doc.units:
                self.doc.units[m["id"]].position = m["position"]
        # Only the drawing changed: no aquakin check is needed.
        self.publish(checks=False)

    def on_remove(self, r: dict) -> None:
        self.doc.remove(units=r["nodes"], edges=r["edges"])
        if self.state.selected not in self.doc.units:
            self.state.selected = None
        self.publish()

    def on_dropped(self, d: dict) -> None:
        kind = d["kind"]
        names = {s["name"] for s in self.catalog.settings(kind)}
        settings = {}
        if "model" in names:
            settings["model"] = "adm1" if kind == "digester" else self.doc.default_model()
            info = self.model_info(settings["model"])
            if "conditions" in names and info:
                defaults = {c["name"]: c["default"] for c in info["conditions"]}
                settings["conditions"] = {k: v for k, v in defaults.items() if v is not None}
        uid = self.doc.add(kind, d["position"], settings)
        self.state.selected = uid
        self.state.panel = "properties"
        self.publish()

    def on_select(self, s: dict) -> None:
        self.state.selected = s["nodes"][0] if len(s["nodes"]) == 1 else None
        self.refresh_selection()

    # -- the unit form ------------------------------------------------------

    def refresh_selection(self) -> None:
        uid = self.state.selected
        if uid not in self.doc.units:
            self.state.sel_rows = []
            return
        u = self.doc.units[uid]
        k = self.catalog.kind(u.kind)
        self.state.sel_kind = k["label"]
        self.state.sel_summary = k["summary"]
        self.state.sel_icon = SVG[icon_name(u.kind, u.settings)]
        self.state.sel_error = self.unit_errors().get(uid)
        self.state.sel_rows = forms.rows_for(self.doc, uid, self.model_info)
        self.state.focus_help = ""

    def apply_properties(self, rows: list[dict], confirmed: bool = False) -> None:
        """Read the form (sent whole, so no keystroke waits on the server) into the unit.

        When the new settings remove ports that have connections, nothing
        changes until the user confirms the removal (``confirmed``).
        """
        uid = self.state.selected
        if uid not in self.doc.units:
            return
        try:
            label, settings = forms.read_rows(self.doc, uid, rows)
        except ValueError as exc:
            self.say(f"could not read a value: {exc}")
            return
        dropped = self.doc.dropped_by(uid, settings)
        if dropped and not confirmed:
            self._pending_rows = rows
            self.state.confirm_lines = dropped
            self.state.confirm_open = True
            return
        self.state.confirm_open = False
        self.doc.apply(uid, settings, label)
        if dropped:
            self.say("removed " + "; ".join(dropped), error=False)
        self.publish()

    def confirm_apply(self) -> None:
        """Apply the form the user confirmed, removing the listed connections."""
        rows, self._pending_rows = self._pending_rows, None
        if rows is not None:
            self.apply_properties(rows, confirmed=True)

    def cancel_apply(self) -> None:
        """Keep the unit as it was; the form shows its settings again."""
        self._pending_rows = None
        self.state.confirm_open = False
        self.refresh_selection()

    def reshape_form(self, rows: list[dict], i: int, value) -> None:
        """A choice that changes which rows the form has (aeration mode, model)."""
        rows = [dict(r) for r in rows]
        rows[i]["value"] = value
        self.apply_properties(rows)

    # -- files --------------------------------------------------------------

    def open(self, path=None, *, quiet: bool = False) -> None:
        path = Path(path or self.state.spec_path)
        try:
            data = yaml.safe_load(path.read_text())
            self.doc = Flowsheet.from_spec(self.catalog, data, base_dir=path.parent.resolve())
        except Exception as exc:
            self.say(f"could not open {path}: {type(exc).__name__}: {exc}")
            return
        self.state.spec_path = str(path)
        self.state.selected = None
        self.state.fit_key += 1
        self._unit_checks.clear()
        self._plant_check = (None, None)
        if not quiet:
            self.publish()
            self.say(f"opened {path}", error=False)

    def save(self) -> None:
        path = Path(self.state.spec_path)
        try:
            path.write_text(yaml.safe_dump(self.doc.to_spec(), sort_keys=False, allow_unicode=True))
        except OSError as exc:
            self.say(f"could not save {path}: {exc}")
            return
        self.say(f"saved {path}", error=False)

    # -- the page -----------------------------------------------------------

    def _init_state(self, path) -> None:
        s = self.state
        s.trame__title = "aquakin flowsheet"
        s.selected = None
        s.show_message = False
        s.message = ""
        s.message_colour = "error"
        s.panel = "properties"
        s.dark = False
        s.focus_help = ""
        s.focus_label = ""
        s.spec_path = str(path) if path else "flowsheet.yaml"
        s.fit_key = 0
        s.spec_text = ""
        s.checking = False
        s.confirm_open = False
        s.confirm_lines = []
        s.checked = False
        s.icons = SVG
        s.palette = [
            {
                "category": CATEGORY_LABELS.get(cat, cat),
                "kinds": [
                    {
                        "kind": k["kind"],
                        "label": k["label"],
                        "summary": k["summary"],
                        "icon": SVG[ICONS.get(k["kind"], "cstr")],
                        "colour": COLOURS.get(cat, "#57606a"),
                    }
                    for k in self.catalog.kinds.values()
                    if k["category"] == cat
                ],
            }
            for cat in self.catalog.categories
        ]
        s.change("panel")(lambda **_: self._publish_spec())

    def _build(self) -> None:
        with SinglePageWithDrawerLayout(self.server, theme=("dark ? 'dark' : 'light'",)) as layout:
            layout.title.set_text("aquakin · {{ plant_name }}")
            layout.footer.hide()
            self._toolbar(layout)
            self._palette(layout)
            with layout.content:
                with html.Div(style="display: flex; height: calc(100vh - 64px);"):
                    with html.Div(style="flex: 1; position: relative;"):
                        FlowsheetCanvas(
                            nodes=("nodes",),
                            edges=("edges",),
                            icons=("icons",),
                            drag_type=DRAG_TYPE,
                            fit_key=("fit_key",),
                            connect=(self.on_connect, "[$event]"),
                            move=(self.on_move, "[$event]"),
                            remove=(self.on_remove, "[$event]"),
                            dropped=(self.on_dropped, "[$event]"),
                            select=(self.on_select, "[$event]"),
                            style="position: absolute; inset: 0;",
                        )
                    self._side_panel()
                self._confirm_dialog()
                v3.VSnackbar(
                    "{{ message }}",
                    v_model=("show_message",),
                    timeout=4000,
                    color=("message_colour",),
                )

    def _toolbar(self, layout) -> None:
        with layout.toolbar:
            v3.VSpacer()
            v3.VTextField(
                v_model=("spec_path",),
                density="compact",
                variant="outlined",
                hide_details=True,
                prepend_inner_icon="mdi-file-outline",
                style="max-width: 300px;",
                classes="mr-2",
            )
            v3.VBtn("Open", click=lambda: self.open(), variant="tonal", classes="mr-1 text-none")
            v3.VBtn("Save", click=self.save, variant="tonal", classes="mr-4 text-none")
            v3.VChip(
                "{{ nodes.length }} units · {{ edges.length }} connections · {{ n_recycles }} recycles",
                size="small",
                variant="tonal",
                classes="mr-2",
            )
            v3.VBtn(icon="mdi-theme-light-dark", click="dark = !dark", variant="text")

    def _palette(self, layout) -> None:
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
                    html.Span(
                        v_html="k.icon", classes="ak-pal-icon", style=("{ color: k.colour }",)
                    )
                    html.Span("{{ k.label }}")

    def _side_panel(self) -> None:
        with v3.VSheet(width=340, border="s", style="overflow: auto; flex: none;"):
            with v3.VTabs(v_model=("panel",), density="compact", grow=True):
                v3.VTab("Unit", value="properties", classes="text-none", min_width=0)
                v3.VTab(
                    "Checks ({{ errors.length }})", value="checks", classes="text-none", min_width=0
                )
                v3.VTab("Spec", value="spec", classes="text-none", min_width=0)
            with v3.VWindow(v_model=("panel",)):
                with v3.VWindowItem(value="properties", classes="pa-3"):
                    self._unit_form()
                with v3.VWindowItem(value="checks", classes="pa-3"):
                    html.Div("Errors", classes="text-overline")
                    with html.Div(v_if="checking", classes="d-flex align-center ga-2 mb-2"):
                        v3.VProgressCircular(indeterminate=True, size=16, width=2)
                        html.Span("aquakin is checking the flowsheet…", classes="text-body-2")
                    html.Div(
                        "None: aquakin built every unit and the plant.",
                        v_if="!checking && checked && !errors.length",
                        classes="text-body-2 mb-2",
                    )
                    with v3.VList(density="compact", v_if="errors.length"):
                        v3.VListItem(
                            v_for="c in errors",
                            key="c",
                            subtitle=("c",),
                            prepend_icon="mdi-alert-circle-outline",
                            base_color="error",
                            lines="three",
                        )
                    html.Div("Product streams (unconnected outputs)", classes="text-overline mt-2")
                    with v3.VList(density="compact"):
                        v3.VListItem(
                            v_for="p in products",
                            key="p",
                            title=("p",),
                            prepend_icon="mdi-arrow-right-bold-outline",
                        )
                with v3.VWindowItem(value="spec", classes="pa-3"):
                    html.Div("The spec as it will be saved", classes="text-caption mb-1")
                    html.Pre(
                        "{{ spec_text }}",
                        style="font-size: 11px; white-space: pre; overflow-x: auto;",
                    )

    def _confirm_dialog(self) -> None:
        with v3.VDialog(v_model=("confirm_open",), max_width=480, persistent=True):
            with v3.VCard(title="Remove connections?"):
                with v3.VCardText():
                    html.Div(
                        "These settings remove ports that are connected. Applying them "
                        "removes these connections:",
                        classes="mb-2",
                    )
                    html.Div("{{ c }}", v_for="c in confirm_lines", key="c", classes="text-mono")
                with v3.VCardActions():
                    v3.VSpacer()
                    v3.VBtn("Cancel", click=self.cancel_apply, variant="text")
                    v3.VBtn(
                        "Apply and remove",
                        click=self.confirm_apply,
                        color="error",
                        variant="flat",
                    )

    def _unit_form(self) -> None:
        html.Div(
            "Select a unit to edit it.",
            v_if="!selected",
            classes="text-body-2 text-medium-emphasis",
        )
        with html.Div(v_else=True):
            with html.Div(classes="d-flex align-center ga-2 mb-1"):
                html.Span(v_html="sel_icon", classes="ak-sel-icon d-inline-flex")
                html.Div("{{ sel_kind }}", classes="text-subtitle-1 font-weight-medium")
            html.Div("{{ sel_summary }}", classes="text-caption text-medium-emphasis mb-3")
            v3.VBtn(
                "Apply",
                click=(self.apply_properties, "[sel_rows]"),
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
                    update_modelValue=(self.reshape_form, "[sel_rows, i, $event]"),
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
                    model_value=("row.value",),
                    **common,
                )
                v3.VTextField(v_else=True, v_model="row.value", **common)
            with v3.VSheet(
                v_if="focus_help", rounded=True, border=True, classes="pa-2 mt-1 text-caption"
            ):
                html.Div("{{ focus_label }}", classes="font-weight-medium mb-1")
                html.Div("{{ focus_help }}", classes="text-medium-emphasis")
