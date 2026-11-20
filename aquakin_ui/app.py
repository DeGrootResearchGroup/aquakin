"""The flowsheet editor page: palette, canvas, unit form, checks and spec.

Every unit kind, setting, default, help text and port comes from aquakin's unit
catalogue, asked of an ``aquakin serve`` worker (:mod:`aquakin_ui.worker`); the
page imports neither aquakin nor JAX. The flowsheet itself is a
:class:`~aquakin_ui.document.Flowsheet`; the page draws it and turns finished
canvas edits into document changes. aquakin's checks (each unit built from its
settings, then the whole plant from the spec) run in the background after an
edit, so the canvas never waits on them. The project's studies are listed with
where each stands, and run in processes of their own
(:class:`~aquakin_ui.jobs.StudyRun`) with progress and cancel.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
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
from aquakin_ui.jobs import StudyRun
from aquakin_ui.results import channels_of, line_chart, steady_table
from aquakin_ui.settings import Settings
from aquakin_ui.study_form import draft_for, fields_for, missing, study_from
from aquakin_ui.widget import Flowsheet as FlowsheetCanvas
from aquakin_ui.worker import AquakinWorker, WorkerError

DRAG_TYPE = "application/x-aquakin-unit"
CHECK_DELAY = 0.3  # seconds of quiet before aquakin's checks run
UNDO_DEPTH = 50  # edits of the flowsheet Undo can take back

PAGE_CSS = """
.ak-chart { display: block; font: 11px Roboto, sans-serif; }
.ak-chart-title { font-size: 13px; font-weight: 500; fill: rgba(var(--v-theme-on-surface), .87); }
.ak-chart-axis { fill: rgba(var(--v-theme-on-surface), .62); }
.ak-chart-grid { stroke: rgba(var(--v-theme-on-surface), .12); stroke-width: 1; }
.ak-chart-line { fill: none; stroke: #2a78d6; stroke-width: 2; stroke-linejoin: round;
  stroke-linecap: round; }
.ak-chart-hit rect { fill: transparent; }
.ak-chart-hit line, .ak-chart-hit circle { opacity: 0; }
.ak-chart-hit line { stroke: rgba(var(--v-theme-on-surface), .35); stroke-width: 1; }
.ak-chart-hit circle { fill: #2a78d6; stroke: rgb(var(--v-theme-surface)); stroke-width: 2; }
.ak-chart-hit:hover line, .ak-chart-hit:hover circle { opacity: 1; }
.v-theme--dark .ak-chart-line { stroke: #3987e5; }
.v-theme--dark .ak-chart-hit circle { fill: #3987e5; }
.ak-num { font-variant-numeric: tabular-nums; text-align: right !important; white-space: nowrap; }
.ak-charts { display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); }
.ak-pal-item { cursor: grab; user-select: none; display: flex; align-items: center; gap: 10px;
  padding: 5px 10px; margin: 1px 8px; border-radius: 6px; font-size: 13px; }
.ak-pal-item:hover { background: rgba(127,127,127,.12); }
.ak-pal-icon { width: 20px; height: 20px; flex: none; display: inline-flex; }
.ak-pal-icon svg { width: 20px; height: 20px; }
.ak-sel-icon svg { width: 28px; height: 28px; }
"""


#: The study the Simulate button sets up and runs for a project's flowsheet.yaml;
#: another spec in the folder gets its own, ``simulation_<file name>``.
SIMULATION = "simulation"
#: The spec a study follows unless its ``flowsheet:`` says otherwise.
DEFAULT_SPEC = "flowsheet.yaml"


def simulation_name(spec: Path) -> str:
    """The name of the Simulate study of the spec file ``spec``."""
    if spec.name == DEFAULT_SPEC:
        return SIMULATION
    return SIMULATION + "_" + re.sub(r"[^A-Za-z0-9_-]", "_", spec.stem)


def flowsheet_entry(spec: Path) -> dict:
    """The ``flowsheet:`` setting a study of ``spec`` needs (none for flowsheet.yaml)."""
    return {} if spec.name == DEFAULT_SPEC else {"flowsheet": f"../../{spec.name}"}


def _plain_error(exc: Exception) -> str:
    """A worker's error without the exception's class name in front."""
    return str(exc).removeprefix("StudyError: ")


# Rows are added to and removed from the study draft in place, which the page
# does not see; reassigning the draft redraws them.
_REDRAW = "; study_draft = {...study_draft}"


def _bound(expr: str) -> dict:
    """Bind a widget to a value inside the study draft, redrawing after each change."""
    return {
        "v_model": expr,
        "update_modelValue": "$nextTick(() => { study_draft = {...study_draft} })",
    }


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

    def __init__(self, server, worker: AquakinWorker | None = None, path=None, settings=None):
        self.server = server
        self.settings = settings if settings is not None else Settings()
        self.state = server.state
        self.worker = worker or AquakinWorker()
        self.catalog = Catalog(self.worker.request("catalog"), self.worker.request("models"))
        self.doc = Flowsheet(self.catalog)
        self._model_info: dict = {}
        self._unit_checks: dict[str, tuple[str, str | None]] = {}
        self._plant_check: tuple[str | None, dict | None] = (None, None)
        self._check_task = None
        self._pending_rows = None
        self._runs: dict[str, StudyRun] = {}
        self._loop = None
        self._opened: tuple[Path, str] | None = None  # the file opened, and its sha256
        self._study_action = None
        self._study_form: dict | None = None  # the worker's study_form, asked once
        self._saved: str | None = None  # the document as last opened or saved
        self._show_when_done: set[str] = set()  # runs whose results open when they finish
        self._results: dict = {}  # the results on show
        self._study_flowsheet: str | None = None  # the spec the study being edited follows
        self._history: list[tuple] = []  # document snapshots to undo to, oldest first
        self._after_unsaved = None  # what the unsaved-changes question goes on to do
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

    def _unit_check(self, uid: str):
        """``(key, request)`` for checking unit ``uid``, read from the document now;
        ``request`` is ``None`` when the latest check of the same input stands."""
        u = self.doc.units[uid]
        key = json.dumps([u.kind, u.settings, self.doc.model_files()], sort_keys=True)
        if self._unit_checks.get(uid, (None,))[0] == key:
            return key, None
        request = dict(
            kind=u.kind,
            name=uid,
            settings=json.loads(json.dumps(u.settings)),
            model_files=self.doc.model_files(),
            base_dir=str(self.doc.base_dir) if self.doc.base_dir else None,
        )
        return key, (request, self.doc.ports(uid))

    def _ask_unit(self, request, ports) -> str | None:
        """Run one unit check in the worker (safe off the page's thread: it
        reads only its arguments)."""
        try:
            out = self.worker.request("check_unit", **request)
        except WorkerError as exc:
            return str(exc)
        if not out["ok"]:
            return out["error"]
        if (out["inputs"], out["outputs"]) != ports:
            return f"aquakin built ports {out['inputs']} -> {out['outputs']}, not the drawn ones"
        return None

    def _check_unit(self, uid: str) -> str | None:
        key, pending = self._unit_check(uid)
        if pending is not None:
            self._unit_checks[uid] = (key, self._ask_unit(*pending))
        return self._unit_checks[uid][1]

    def _plant_request(self):
        """``(key, spec, base_dir)`` for checking the whole plant, read now."""
        spec = self.doc.to_spec(layout=False)
        base = str(self.doc.base_dir) if self.doc.base_dir else None
        return json.dumps([spec, base], sort_keys=True), spec, base

    def _ask_plant(self, spec, base) -> dict:
        try:
            return self.worker.request("check_spec", spec=spec, base_dir=base)
        except WorkerError as exc:
            return {"ok": False, "error": str(exc)}

    def _check_plant(self) -> dict | None:
        key, spec, base = self._plant_request()
        if self._plant_check[0] != key:
            self._plant_check = (key, self._ask_plant(spec, base))
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
        # The document is read here, on the page's thread; only the worker's
        # answers are awaited in the executor, so an edit made meanwhile cannot
        # be half-read. An edit also cancels this task and schedules a new one.
        for uid in list(self.doc.units):
            if uid not in self.doc.units:
                continue
            key, pending = self._unit_check(uid)
            if pending is not None:
                error = await loop.run_in_executor(None, self._ask_unit, *pending)
                self._unit_checks[uid] = (key, error)
        if not self.unit_errors():
            key, spec, base = self._plant_request()
            if self._plant_check[0] != key:
                result = await loop.run_in_executor(None, self._ask_plant, spec, base)
                self._plant_check = (key, result)
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

    # -- undo -----------------------------------------------------------------

    def _remember(self) -> None:
        """Keep the document as it is now, to undo the edit about to be made."""
        self._history = [*self._history[-(UNDO_DEPTH - 1) :], self.doc.snapshot()]
        self.state.can_undo = True

    def undo(self) -> None:
        """Take back the last edit of the flowsheet (a removal, a wire, a move, Apply)."""
        if not self._history:
            return
        self.doc.restore(self._history.pop())
        self.state.can_undo = bool(self._history)
        if self.state.selected not in self.doc.units:
            self.state.selected = None
        self.publish()

    # -- canvas events (finished edits only) --------------------------------

    def on_connect(self, c: dict) -> None:
        before = self.doc.snapshot()
        why = self.doc.connect(c["source"], c["sourceHandle"], c["target"], c["targetHandle"])
        if why:
            self.say(why)
        else:
            self._history = [*self._history[-(UNDO_DEPTH - 1) :], before]
            self.state.can_undo = True
        self.publish()

    def on_move(self, moved: list[dict]) -> None:
        self._remember()
        for m in moved:
            if m["id"] in self.doc.units:
                self.doc.units[m["id"]].position = m["position"]
        # Only the drawing changed: no aquakin check is needed.
        self.publish(checks=False)

    def on_remove(self, r: dict) -> None:
        self._remember()
        self.doc.remove(units=r["nodes"], edges=r["edges"])
        if self.state.selected not in self.doc.units:
            self.state.selected = None
        gone = [*r["nodes"], *(e for e in r["edges"] if not r["nodes"])]
        self.say(f"removed {', '.join(gone)}: Undo brings it back", error=False)
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
        self._remember()
        uid = self.doc.add(kind, d["position"], settings)
        self.state.selected = uid
        self.state.panel = "properties"
        self.publish()

    def on_select(self, s: dict, pending: list[dict] | None = None) -> None:
        """A new selection on the canvas; ``pending`` is the form's unapplied
        edits of the unit selected until now, which are applied first (or, when
        they cannot be, that unit stays selected so nothing typed is lost)."""
        if pending is not None and self.state.selected in self.doc.units:
            if not self.apply_properties(pending):
                self.state.dirty("nodes")  # the canvas shows the selection Python keeps
                return
        self.state.selected = s["nodes"][0] if len(s["nodes"]) == 1 else None
        self.refresh_selection()

    # -- the unit form ------------------------------------------------------

    def refresh_selection(self) -> None:
        uid = self.state.selected
        if uid not in self.doc.units:
            self.state.sel_rows = []
            self.state.sel_dirty = False
            return
        u = self.doc.units[uid]
        k = self.catalog.kind(u.kind)
        self.state.sel_kind = k["label"]
        self.state.sel_summary = k["summary"]
        self.state.sel_icon = SVG[icon_name(u.kind, u.settings)]
        self.state.sel_error = self.unit_errors().get(uid)
        self.state.sel_rows = forms.rows_for(self.doc, uid, self.model_info)
        self.state.sel_dirty = False
        self.state.focus_help = ""

    def apply_properties(self, rows: list[dict], confirmed: bool = False) -> bool:
        """Read the form (sent whole, so no keystroke waits on the server) into the unit.

        When the new settings remove ports that have connections, nothing
        changes until the user confirms the removal (``confirmed``).

        Returns
        -------
        bool
            Whether the unit now has the form's settings.
        """
        uid = self.state.selected
        if uid not in self.doc.units:
            return False
        try:
            label, settings = forms.read_rows(self.doc, uid, rows)
        except ValueError as exc:
            self.say(f"{uid}: {exc}")
            return False
        dropped = self.doc.dropped_by(uid, settings)
        if dropped and not confirmed:
            self._pending_rows = rows
            self.state.confirm_lines = dropped
            self.state.confirm_open = True
            return False
        self.state.confirm_open = False
        self._remember()
        self.doc.apply(uid, settings, label)
        if dropped:
            self.say("removed " + "; ".join(dropped), error=False)
        self.publish()
        return True

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
        # The rows equal the last ones sent, so they would not be sent again
        # and the page would keep the edits just declined.
        self.state.dirty("sel_rows")

    def reshape_form(self, rows: list[dict], i: int, value) -> None:
        """A choice that changes which rows the form has (aeration mode, model)."""
        rows = [dict(r) for r in rows]
        rows[i]["value"] = value
        self.apply_properties(rows)

    # -- files --------------------------------------------------------------

    def _off_page(self, work, done) -> None:
        """Run ``work()`` (a worker call) off the page's thread, then ``done(result)``
        on it; ``done`` gets the exception instead when ``work`` raised. Without a
        served page (at start-up, in tests) both run here and now."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                result = work()
            except Exception as exc:
                result = exc
            done(result)
            return

        async def run():
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(None, work)
            except Exception as exc:
                result = exc
            with self.state:
                self.state.busy = False
                done(result)

        self.state.busy = True
        asynchronous.create_task(run())

    def open(self, path=None, *, quiet: bool = False) -> None:
        """Read a spec with aquakin's reader (YAML 1.2) and show it."""
        path = Path(path or self.state.spec_path).resolve()

        def done(data):
            try:
                if isinstance(data, Exception):
                    raise data
                doc = Flowsheet.from_spec(self.catalog, data, base_dir=path.parent)
            except Exception as exc:
                self.say(f"could not open {path}: {type(exc).__name__}: {exc}")
                return
            self.doc = doc
            self._opened = (path, _digest(path))
            self._saved = self._snapshot()
            self.state.spec_path = str(path)
            self.state.selected = None
            self.state.fit_key += 1
            self._unit_checks.clear()
            self._plant_check = (None, None)
            self._history = []
            self.state.can_undo = False
            if not quiet:
                self.publish()
                self.say(f"opened {path}", error=False)
            self.refresh_studies()

        self._off_page(lambda: self.worker.request("read_spec", path=str(path)), done)

    def _snapshot(self) -> str:
        return json.dumps(self.doc.to_spec(), sort_keys=True)

    def _unsaved(self) -> bool:
        """Whether the flowsheet has edits its file does not hold (an empty new
        flowsheet has none)."""
        if self._saved is None:
            return bool(self.doc.units)
        return self._snapshot() != self._saved

    def _spec_file(self) -> Path | None:
        """The flowsheet's file: the one last opened or saved here, else the path
        typed in the toolbar, else ``None`` (a new flowsheet with no name yet).

        The toolbar's path is also where Open reads and Save writes next, so it
        only becomes the flowsheet's file once one of them has gone through.
        """
        if self._opened is not None:
            return self._opened[0]
        typed = str(self.state.spec_path or "").strip()
        return Path(typed).resolve() if typed else None

    def request_open(self) -> None:
        """Open the path in the toolbar, asking first when there are unsaved edits."""
        target = str(self.state.spec_path or "").strip()
        if not target:
            self.say("type the path of a flowsheet file to open")
            return
        self._unless_unsaved(
            lambda: self.open(target),
            f"Open {Path(target).name} and leave the unsaved edits of this flowsheet?",
        )

    def _unless_unsaved(self, then, question: str) -> None:
        """``then()`` at once when nothing is unsaved; otherwise ask, offering to
        save to the flowsheet's file first or to discard the edits."""
        if not self._unsaved():
            then()
            return
        self._after_unsaved = then
        self.state.unsaved_text = question
        self.state.unsaved_can_save = self._opened is not None
        self.state.unsaved_open = True

    def answer_unsaved(self, choice: str) -> None:
        """The answer to the unsaved-edits question: ``"save"``, ``"discard"`` or
        ``"cancel"``."""
        self.state.unsaved_open = False
        then, self._after_unsaved = self._after_unsaved, None
        if then is None or choice == "cancel":
            return
        if choice == "save":
            self.save(then=then, path=self._opened[0])
        else:
            then()

    def save(self, then=None, path=None) -> None:
        """Save through aquakin's own spec writer: settings checked and filled in,
        the aquakin version and file hashes recorded, outside model and influent
        files copied into the project, and the file replaced atomically; then
        ``then()``, when given and the save went through.

        ``path`` defaults to the path in the toolbar (so Save can save as)."""
        if path is None:
            typed = str(self.state.spec_path or "").strip()
            if not typed:
                self.say(
                    "type where to keep the flowsheet in the path field (a .yaml file, "
                    "in the folder of its project), then Save"
                )
                return
            path = typed
        path = Path(path).resolve()
        refusal = self._save_refusal(path)
        if refusal:
            self.say(refusal)
            return
        base = str(self.doc.base_dir) if self.doc.base_dir else str(path.parent)
        spec = self.doc.to_spec()  # read now, on the page's thread
        snapshot = json.dumps(spec, sort_keys=True)

        def done(written):
            if isinstance(written, Exception):
                self.say(f"could not save {path}: {type(written).__name__}: {written}")
                return
            # Copied files are now referenced inside the project; an edit made
            # while saving is kept rather than replaced by what was written.
            if json.dumps(self.doc.to_spec(), sort_keys=True) == snapshot:
                self.doc = Flowsheet.from_spec(self.catalog, written, base_dir=path.parent)
                self._saved = self._snapshot()
            self._opened = (path, _digest(path))
            self.state.spec_path = str(path)
            self._plant_check = (None, None)
            self.publish()
            self.say(f"saved {path}", error=False)
            if then is not None:
                then()

        self._off_page(
            lambda: self.worker.request("write_spec", spec=spec, path=str(path), base_dir=base),
            done,
        )

    def _save_refusal(self, path: Path) -> str | None:
        """Why saving to ``path`` would lose someone else's work, or ``None``.

        Each window is a document of its own, so two can hold the same file:
        saving over a file that changed since this window opened it (or over
        another existing file) would silently replace that work."""
        if not path.exists():
            return None
        if self._opened is not None and self._opened[0] == path:
            if _digest(path) != self._opened[1]:
                return (
                    f"{path.name} has changed on disk since it was opened here (another "
                    f"window or program saved it); open it again, or save under another name"
                )
            return None
        return f"{path} already exists; open it, or save under another name"

    # -- studies ------------------------------------------------------------

    def _project(self) -> Path | None:
        """The project folder: the flowsheet file's (``None`` before it has one)."""
        spec = self._spec_file()
        return spec.parent if spec is not None else None

    def refresh_studies(self) -> None:
        """List the project's studies with where each stands (off the page's thread)."""
        project = self._project()
        if project is None:
            self.state.studies = []
            return

        def done(studies):
            if isinstance(studies, Exception):
                self.say(f"could not list the studies: {studies}")
                return
            self.state.studies = [{**st, "running": st["name"] in self._runs} for st in studies]

        self._off_page(lambda: self.worker.request("list_studies", project=str(project)), done)

    def _study(self, name: str) -> dict | None:
        return next((st for st in self.state.studies if st["name"] == name), None)

    def run_study(self, name: str) -> None:
        """Run a study; ask first when that would replace its results."""
        study = self._study(name)
        if study is None or name in self._runs:
            return
        if not study.get("frozen") and self._unsaved():
            # A live study runs the flowsheet's file: bring it up to the drawing.
            self.save(then=lambda: self.run_study(name), path=self._spec_file())
            return
        if study["state"] in ("not run",):
            self._start_run(study, replace=False)
            return
        if not self.state.ask_replace:
            self._start_run(study, replace=True)
            return
        self._ask(
            "run",
            name,
            f"Replace the results of {name}?",
            "Running the study again replaces its results. Copy the study first to keep them.",
            "Replace results",
        )

    def unfreeze_study(self, name: str) -> None:
        """Unfreeze a study; ask first when that deletes its results."""
        study = self._study(name)
        if study is None:
            return
        if study["state"] == "not run":
            self._worker_study("unfreeze_study", study, discard_results=True)
            return
        self._ask(
            "unfreeze",
            name,
            f"Unfreeze {name} and delete its results?",
            "Unfreezing makes the study follow the flowsheet again. Its results were "
            "computed on its frozen snapshot, so they are deleted. Copy the study first to "
            "keep them.",
            "Unfreeze and delete",
        )

    def freeze_study(self, name: str) -> None:
        study = self._study(name)
        if study is not None:
            self._worker_study("freeze_study", study)

    def copy_study(self, name: str, then=None) -> None:
        """Copy a study beside it as ``<name>_copy`` (``_copy2``, ... if taken)."""
        study = self._study(name)
        if study is None:
            return
        taken = {st["name"] for st in self.state.studies}
        new, i = f"{name}_copy", 2
        while new in taken:
            new, i = f"{name}_copy{i}", i + 1
        self._worker_study("copy_study", study, then=then, name=new)

    # -- simulating the plant -------------------------------------------------

    def simulate(self) -> None:
        """Run the plant: save the flowsheet if it changed, set up the project's
        ``simulation`` study the first time (the plant at steady state, saving the
        streams that leave it), run it, and show its results when it finishes."""
        if self._simulation() in self._runs:
            self.state.panel = "studies"
            return
        if self._spec_file() is None:
            self.say(
                "the flowsheet has no file yet: type where to keep it in the path field "
                "(a .yaml file in the folder of its project), then Simulate"
            )
            return
        if self._unsaved() or not self._spec_file().exists():
            self.save(then=self._run_simulation, path=self._spec_file())
        else:
            self._run_simulation()

    def _simulation(self) -> str:
        """The name of the open spec's Simulate study."""
        spec = self._spec_file()
        return simulation_name(spec) if spec is not None else SIMULATION

    def _run_simulation(self) -> None:
        spec = self._spec_file()
        project = spec.parent
        name = simulation_name(spec)
        folder = project / "studies" / name

        def work():
            if not (folder / "study.yaml").is_file():
                settings = self.worker.request("simulation_study", spec=str(spec))
                self.worker.request(
                    "create_study",
                    project=str(project),
                    name=name,
                    study={**settings, **flowsheet_entry(spec)},
                )
            else:
                # The study must follow this spec: a copy or a hand edit may point
                # it at another file of the folder.
                settings = self.worker.request("read_study", path=str(folder))
                follows = folder / settings.get("flowsheet", f"../../{DEFAULT_SPEC}")
                if follows.resolve() != spec:
                    settings = {k: v for k, v in settings.items() if k != "flowsheet"}
                    self.worker.request(
                        "update_study",
                        path=str(folder),
                        study={**settings, **flowsheet_entry(spec)},
                    )
            return self.worker.request("list_studies", project=str(project))

        def done(studies):
            if isinstance(studies, Exception):
                self.say(f"could not set up the simulation: {_plain_error(studies)}")
                return
            self.state.studies = [{**st, "running": st["name"] in self._runs} for st in studies]
            self.state.panel = "studies"
            if (self._study(name) or {}).get("frozen"):
                self.say(
                    f"{name} is frozen on a snapshot of the plant, so it would not simulate "
                    f"the flowsheet as drawn; unfreeze it (or remove it) to simulate"
                )
                return
            self._show_when_done.add(name)
            self.run_study(name)

        self._off_page(work, done)

    def show_results(self, name: str, state: str | None = None) -> None:
        """Open a study's results: a steady state as tables, a run in time as a
        chart per channel, any other kind as the list of its result files.

        ``state`` is where the study stands when the caller knows better than
        the listing (a run that just finished, before the list is refreshed).
        """
        study = self._study(name)
        if study is None:
            return
        state = state or study["state"]

        def done(results):
            if isinstance(results, Exception):
                self.say(f"could not read the results of {name}: {_plain_error(results)}")
                return
            s = self.state
            self._results = results
            s.results_title = f"Results of {name}"
            s.results_folder = str(Path(study["path"]) / "results")
            s.results_files = results["files"]
            s.results_note = (
                "The flowsheet has changed since this ran; run it again for the plant as drawn."
                if state == "stale"
                else ""
            )
            units = results.get("units") or {}
            if results["kind"] != "simulate":
                s.results_mode = "files"
            elif results["steady"]:
                s.results_mode = "steady"
                s.results_measured = steady_table(results.get("measured") or {}, units)
                s.results_streams = steady_table(results.get("streams") or {}, units)
            else:
                s.results_mode = "time"
                streams = [*(results.get("measured") or {}), *(results.get("streams") or {})]
                s.results_stream_choices = list(dict.fromkeys(streams))
                self.choose_results_stream(streams[0] if streams else "")
            s.results_open = True

        self._off_page(lambda: self.worker.request("study_results", path=study["path"]), done)

    def choose_results_stream(self, stream: str) -> None:
        """Show ``stream``'s charts: its measurements, or its flow and first species."""
        self.state.results_stream = stream
        channels = channels_of(self._results, stream) if stream else []
        self.state.results_channel_choices = channels
        measured = list((self._results.get("measured") or {}).get(stream, {}))
        self.show_charts(measured[:6] or channels[:4])

    def show_charts(self, channels: list[str]) -> None:
        """Draw a chart per channel of the chosen stream (each on its own axis)."""
        stream = self.state.results_stream
        self.state.results_channels = list(channels)
        self.state.results_charts = [
            {"channel": c, "svg": line_chart(self._results, stream, c)} for c in channels
        ]

    # -- the study dialog ---------------------------------------------------

    def _plant_choices(self) -> dict:
        """What a study of the flowsheet as drawn may name; aquakin's answer when
        the plant builds, else the drawn units' output ports (names typed into
        the dialog are checked when the study is written)."""
        spec = self.doc.to_spec(layout=False)
        base = str(self.doc.base_dir) if self.doc.base_dir else str(self._project())
        try:
            return self.worker.request("plant_choices", spec=spec, base_dir=base)
        except WorkerError:
            ports = [f"{uid}.{port}" for uid in self.doc.units for port in self.doc.ports(uid)[1]]
            return {"streams": ports, "parameters": [], "channels": {}}

    def _open_study_dialog(self, *, title: str, kind: str, study: dict | None, editing: str):
        form = self._study_form

        def done(choices):
            if isinstance(choices, Exception):
                choices = {"streams": [], "parameters": [], "channels": {}}
            s = self.state
            s.study_kinds = [
                {"title": k["label"], "value": k["kind"], "summary": k["summary"]}
                for k in form["kinds"]
            ]
            s.study_choices = choices
            s.study_kind = kind
            s.study_fields = fields_for(form, kind)
            s.study_draft = draft_for(form, kind, study)
            s.study_editing = editing
            s.study_error = ""
            s.study_dialog_title = title
            s.study_dialog_open = True

        self._off_page(self._plant_choices, done)

    def _with_form(self, then) -> None:
        """Run ``then()`` once the worker's study form is known."""
        if self._study_form is not None:
            then()
            return

        def done(form):
            if isinstance(form, Exception):
                self.say(f"could not read the study settings: {form}")
                return
            self._study_form = form
            then()

        self._off_page(lambda: self.worker.request("study_form"), done)

    def new_study(self) -> None:
        """Open the dialog for a new study (a simulation to start with)."""
        if self._project() is None:
            self.say("save the flowsheet first: its studies are kept in its folder")
            return
        self.state.study_name = ""
        self._with_form(
            lambda: self._open_study_dialog(
                title="New study", kind="simulate", study=None, editing=""
            )
        )

    def edit_study(self, name: str) -> None:
        """Open the dialog on a study's settings."""
        study = self._study(name)
        if study is None:
            return

        def opened(settings):
            if isinstance(settings, Exception):
                self.say(f"could not read study {name}: {settings}")
                return
            self.state.study_name = name
            # The form has no flowsheet field: the study keeps the spec it follows.
            self._study_flowsheet = settings.get("flowsheet")
            self._with_form(
                lambda: self._open_study_dialog(
                    title=f"Edit {name}",
                    kind=settings["kind"],
                    study=settings,
                    editing=study["path"],
                )
            )

        self._off_page(lambda: self.worker.request("read_study", path=study["path"]), opened)

    def set_study_kind(self, kind: str, draft: dict) -> None:
        """Change the dialog's kind, keeping the values of the fields both kinds have."""
        fresh = draft_for(self._study_form, kind)
        for name in fresh:
            if name in draft:
                fresh[name] = draft[name]
        self.state.study_kind = kind
        self.state.study_fields = fields_for(self._study_form, kind)
        self.state.study_draft = fresh

    def save_study_dialog(self, name: str, kind: str, draft: dict) -> None:
        """Write the study the dialog describes (create it, or replace its settings)."""
        try:
            study = study_from(self._study_form, kind, draft)
        except ValueError as exc:
            self.state.study_error = str(exc)
            return
        editing = self.state.study_editing
        name = (name or "").strip()
        if not editing and not name:
            self.state.study_error = "Give the study a name."
            return
        needed = missing(self._study_form, kind, study)
        if needed:
            self.state.study_error = needed
            return
        if editing:
            if self._study_flowsheet:
                study["flowsheet"] = self._study_flowsheet
            work = lambda: self.worker.request("update_study", path=editing, study=study)
        else:
            study.update(flowsheet_entry(self._spec_file()))
            project = str(self._project())
            work = lambda: self.worker.request(
                "create_study", project=project, name=name, study=study
            )

        def done(result):
            if isinstance(result, Exception):
                self.state.study_error = _plain_error(result)
                return
            self.state.study_dialog_open = False
            self.say(f"{'updated' if editing else 'created'} study {name}", error=False)
            self.refresh_studies()

        self._off_page(work, done)

    def _worker_study(self, command: str, study: dict, then=None, **fields) -> None:
        def done(result):
            if isinstance(result, Exception):
                self.say(f"{command.replace('_', ' ')} failed: {result}")
                return
            if then is not None:
                then()
            self.refresh_studies()

        self._off_page(lambda: self.worker.request(command, path=study["path"], **fields), done)

    def _ask(self, action: str, name: str, title: str, text: str, confirm: str) -> None:
        self._study_action = (action, name)
        self.state.study_confirm_kind = action
        self.state.study_confirm_dont_ask = False
        self.state.study_confirm_title = title
        self.state.study_confirm_text = text
        self.state.study_confirm_label = confirm
        self.state.study_confirm_open = True

    def set_ask_replace(self, ask: bool) -> None:
        """Whether a run asks before replacing a study's results; kept as the
        user's preference (the Studies tab's switch, or "Don't ask again")."""
        self.state.ask_replace = ask
        if ask != self.settings["ask_before_replacing"]:
            if not self.settings.set("ask_before_replacing", ask):
                self.say(f"could not save the preference to {self.settings.path}")

    def answer_study(self, choice: str) -> None:
        """The answer to a study question: ``"cancel"``, ``"copy"`` (copy, then go
        ahead) or ``"go"``."""
        self.state.study_confirm_open = False
        action, self._study_action = self._study_action, None
        if action is None or choice == "cancel":
            if action is not None:
                self._show_when_done.discard(action[1])  # it will not run now
            return
        kind, name = action
        if kind == "run" and self.state.study_confirm_dont_ask:
            self.set_ask_replace(False)

        def go():
            study = self._study(name)
            if study is None:
                return
            if kind == "run":
                self._start_run(study, replace=True)
            else:
                self._worker_study("unfreeze_study", study, discard_results=True)

        if choice == "copy":  # keep a copy of the study and its results, then go ahead
            self.copy_study(name, then=go)
        else:
            go()

    def _start_run(self, study: dict, *, replace: bool) -> None:
        name = study["name"]
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self.state.study_progress = {
            **self.state.study_progress,
            name: {"fraction": None, "message": "starting"},
        }
        self._runs[name] = StudyRun(
            study["path"], lambda event: self._on_run_event(name, event), replace=replace
        )
        self._mark_running()

    def cancel_run(self, name: str) -> None:
        run = self._runs.get(name)
        if run is not None:
            run.cancel()
            self.state.study_progress = {
                **self.state.study_progress,
                name: {**self.state.study_progress.get(name, {}), "message": "cancelling"},
            }

    def _on_run_event(self, name: str, event: dict) -> None:
        # Called on the run's reader thread; the page's state is changed on its loop.
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._apply_run_event, name, event)
        else:
            self._apply_run_event(name, event)

    def _apply_run_event(self, name: str, event: dict) -> None:
        with self.state:
            progress = dict(self.state.study_progress)
            if event["event"] == "progress":
                progress[name] = {"fraction": event["fraction"], "message": event["message"]}
                self.state.study_progress = progress
                return
            progress.pop(name, None)
            self.state.study_progress = progress
            self._runs.pop(name, None)
            show = name in self._show_when_done
            self._show_when_done.discard(name)
            if event["event"] == "done":
                self.say(f"{name}: {event['state']}", error=event["state"] != "fresh")
                if show and event["state"] in ("fresh", "stale"):
                    self.show_results(name, state=event["state"])
            elif event["event"] == "cancelled":
                self.say(f"{name}: cancelled", error=False)
            else:
                self.say(f"{name} failed: {event['error']}")
            self._mark_running()
            self.refresh_studies()

    def _mark_running(self) -> None:
        self.state.studies = [
            {**st, "running": st["name"] in self._runs} for st in self.state.studies
        ]
        self.state.simulating = self._simulation() in self._runs

    def stop_runs(self) -> None:
        """Cancel every run (when the editor closes)."""
        for run in list(self._runs.values()):
            run.cancel()

    # -- the page -----------------------------------------------------------

    def _init_state(self, path) -> None:
        s = self.state
        s.trame__title = "aquakin flowsheet"
        s.selected = None
        s.show_message = False
        s.message = ""
        s.message_colour = "error"
        s.busy = False
        s.studies = []
        s.study_progress = {}
        s.study_confirm_open = False
        s.study_confirm_title = ""
        s.study_confirm_text = ""
        s.study_confirm_label = ""
        s.study_dialog_open = False
        s.study_dialog_title = ""
        s.study_name = ""
        s.study_kind = "simulate"
        s.study_kinds = []
        s.study_fields = []
        s.study_draft = {}
        s.study_choices = {"streams": [], "parameters": [], "channels": {}}
        s.study_editing = ""
        s.study_error = ""
        s.simulating = False
        s.ask_replace = self.settings["ask_before_replacing"]
        s.study_confirm_kind = ""
        s.study_confirm_dont_ask = False
        s.results_open = False
        s.results_title = ""
        s.results_folder = ""
        s.results_files = []
        s.results_note = ""
        s.results_mode = "steady"
        s.results_measured = {"columns": [], "rows": []}
        s.results_streams = {"columns": [], "rows": []}
        s.results_stream = ""
        s.results_stream_choices = []
        s.results_channel_choices = []
        s.results_channels = []
        s.results_charts = []
        s.panel = "properties"
        s.dark = False
        s.focus_help = ""
        s.focus_label = ""
        s.spec_path = str(path) if path else ""
        s.can_undo = False
        s.sel_dirty = False
        s.unsaved_open = False
        s.unsaved_text = ""
        s.unsaved_can_save = False
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

        @s.change("panel")
        def _studies_tab(panel, **_):
            if panel == "studies":
                self.refresh_studies()

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
                            select=(self.on_select, "[$event, sel_dirty ? sel_rows : null]"),
                            style="position: absolute; inset: 0;",
                        )
                    self._side_panel()
                self._confirm_dialog()
                self._study_dialog()
                self._study_settings_dialog()
                self._results_dialog()
                self._unsaved_dialog()
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
                placeholder="path/to/flowsheet.yaml",
                title="The flowsheet's file: Open reads it, Save writes it",
                density="compact",
                variant="outlined",
                hide_details=True,
                prepend_inner_icon="mdi-file-outline",
                style="max-width: 300px;",
                classes="mr-2",
            )
            v3.VBtn(
                icon="mdi-undo",
                click=self.undo,
                disabled=("!can_undo",),
                variant="text",
                title="Undo the last edit of the flowsheet",
                classes="mr-1",
            )
            v3.VBtn(
                "Open",
                click=self.request_open,
                disabled=("busy",),
                variant="tonal",
                classes="mr-1 text-none",
            )
            v3.VBtn(
                "Save",
                click=self.save,
                loading=("busy",),
                disabled=("busy",),
                variant="tonal",
                classes="mr-2 text-none",
            )
            v3.VBtn(
                "Simulate",
                prepend_icon="mdi-play",
                click=self.simulate,
                loading=("simulating",),
                disabled=("busy",),
                color="primary",
                variant="flat",
                classes="mr-4 text-none",
                title="Run the plant to steady state and show what leaves it",
            )
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
                v3.VTab("Studies", value="studies", classes="text-none", min_width=0)
                v3.VTab("Spec", value="spec", classes="text-none", min_width=0)
            with v3.VWindow(v_model=("panel",)):
                with v3.VWindowItem(value="studies", classes="pa-3"):
                    self._studies_panel()
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

    def _studies_panel(self) -> None:
        with html.Div(classes="d-flex align-center"):
            html.Div("Studies of this project", classes="text-overline")
            v3.VSpacer()
            v3.VBtn(
                icon="mdi-refresh",
                size="small",
                variant="text",
                click=self.refresh_studies,
                title="Check every study again",
            )
        v3.VBtn(
            "New study",
            prepend_icon="mdi-plus",
            block=True,
            variant="tonal",
            classes="text-none mb-3",
            click=self.new_study,
            disabled=("busy",),
        )
        v3.VSwitch(
            model_value=("ask_replace",),
            update_modelValue=(self.set_ask_replace, "[!!$event]"),
            label="Ask before replacing results",
            color="primary",
            density="compact",
            hide_details=True,
            classes="mb-2 ml-2",
            title="When off, running a study again replaces its results without asking",
        )
        html.Div(
            "No studies yet: set one up with New study (once the flowsheet has a file).",
            v_if="!studies.length",
            classes="text-body-2 text-medium-emphasis",
        )
        with v3.VCard(v_for="st in studies", key="st.name", variant="outlined", classes="mb-2"):
            with v3.VCardItem():
                with html.Div(classes="d-flex align-center ga-2"):
                    html.Span("{{ st.name }}", classes="text-subtitle-2")
                    v3.VChip("{{ st.kind }}", size="x-small", variant="tonal")
                    v3.VIcon("mdi-snowflake", v_if="st.frozen", size="small", title="frozen")
                    v3.VSpacer()
                    v3.VChip(
                        "{{ st.running ? 'running' : st.state }}",
                        size="x-small",
                        color=(
                            "st.running ? 'primary' : ({fresh: 'success', stale: 'warning', "
                            "'not run': undefined, cancelled: undefined}[st.state] ?? 'error')",
                        ),
                    )
                html.Div(
                    "{{ st.description }}",
                    v_if="st.description",
                    classes="text-caption text-medium-emphasis",
                )
                html.Div(
                    "{{ r }}",
                    v_for="r in st.reasons",
                    key="r",
                    classes="text-caption text-warning",
                )
                html.Div(
                    "{{ n }}",
                    v_for="n in st.notes",
                    key="n",
                    classes="text-caption text-medium-emphasis",
                )
                with html.Div(v_if="st.running && study_progress[st.name]", classes="mt-1"):
                    v3.VProgressLinear(
                        model_value=(
                            "study_progress[st.name].fraction == null ? 0 : "
                            "100 * study_progress[st.name].fraction",
                        ),
                        indeterminate=("study_progress[st.name].fraction == null",),
                        height=4,
                        rounded=True,
                    )
                    html.Div(
                        "{{ study_progress[st.name].message }}",
                        classes="text-caption text-medium-emphasis",
                    )
            with v3.VCardActions(classes="py-0"):
                v3.VBtn(
                    "Run",
                    v_if="!st.running",
                    size="small",
                    disabled=("st.state === 'unreadable'",),
                    click=(self.run_study, "[st.name]"),
                )
                v3.VBtn(
                    "Cancel",
                    v_else=True,
                    size="small",
                    color="error",
                    click=(self.cancel_run, "[st.name]"),
                )
                v3.VBtn(
                    "Results",
                    v_if="!st.running && ['fresh', 'stale'].includes(st.state)",
                    size="small",
                    color="primary",
                    click=(self.show_results, "[st.name]"),
                )
                v3.VBtn(
                    "Freeze",
                    v_if="!st.frozen",
                    size="small",
                    disabled=("st.running",),
                    click=(self.freeze_study, "[st.name]"),
                )
                v3.VBtn(
                    "Unfreeze",
                    v_else=True,
                    size="small",
                    disabled=("st.running",),
                    click=(self.unfreeze_study, "[st.name]"),
                )
                v3.VBtn(
                    "Edit",
                    size="small",
                    disabled=("st.running || st.state === 'unreadable'",),
                    click=(self.edit_study, "[st.name]"),
                )
                v3.VBtn("Copy", size="small", click=(self.copy_study, "[st.name]"))

    @staticmethod
    def _observables_editor(rows: str) -> None:
        """Rows of ``{stream, channels}`` bound to the client expression ``rows``."""
        with html.Div(v_for=f"(o, oi) in {rows}", key="oi", classes="d-flex ga-2 mb-1"):
            v3.VCombobox(
                **_bound("o.stream"),
                items=("study_choices.streams",),
                label="Stream",
                density="compact",
                variant="outlined",
                hide_details=True,
                style="max-width: 220px;",
            )
            v3.VCombobox(
                **_bound("o.channels"),
                items=("study_choices.channels[o.stream] || []",),
                label="Channels (all species when empty)",
                multiple=True,
                chips=True,
                closable_chips=True,
                density="compact",
                variant="outlined",
                hide_details=True,
            )
            v3.VBtn(
                icon="mdi-close",
                size="small",
                variant="text",
                click=f"{rows}.splice(oi, 1){_REDRAW}",
            )
        v3.VBtn(
            "Add a stream",
            prepend_icon="mdi-plus",
            size="small",
            variant="text",
            classes="text-none",
            click=f"{rows}.push({{stream: '', channels: [], measures: {{}}}}){_REDRAW}",
        )

    def _candidate_fields(self, c: str) -> None:
        with html.Div(classes="d-flex ga-2 mb-1"):
            v3.VTextField(
                **_bound(f"{c}.name"),
                label="Name",
                density="compact",
                variant="outlined",
                hide_details=True,
            )
            v3.VTextField(
                **_bound(f"{c}.sigma"),
                label="Measurement error (one, or one per channel)",
                density="compact",
                variant="outlined",
                hide_details=True,
            )
        self._observables_editor(f"{c}.observables")

    def _study_settings_dialog(self) -> None:
        """The New / Edit study dialog, generated from aquakin's study form."""
        common = dict(density="compact", variant="outlined", hide_details=True)
        draft = "study_draft[f.name]"
        # Persistent: a click beside it would lose a long form; Cancel closes it.
        with v3.VDialog(
            v_model=("study_dialog_open",), max_width=760, scrollable=True, persistent=True
        ):
            with v3.VCard(title=("study_dialog_title",)):
                with v3.VCardText(style="max-height: 70vh;"):
                    with html.Div(classes="d-flex ga-2 mb-1 pt-2"):
                        v3.VTextField(
                            v_model=("study_name",),
                            label="Name",
                            disabled=("!!study_editing",),
                            **common,
                        )
                        v3.VSelect(
                            model_value=("study_kind",),
                            items=("study_kinds",),
                            label="Kind",
                            disabled=("!!study_editing",),
                            update_modelValue=(self.set_study_kind, "[$event, study_draft]"),
                            **common,
                        )
                    html.Div(
                        "{{ (study_kinds.find(k => k.value === study_kind) || {}).summary }}",
                        classes="text-caption text-medium-emphasis mb-3",
                    )
                    with html.Div(v_for="f in study_fields", key="f.name", classes="mb-3"):
                        html.Div(
                            "{{ f.label }}{{ f.required ? ' *' : '' }}",
                            id=("'study-' + f.name + '-label'",),
                            classes="text-subtitle-2",
                        )
                        html.Div("{{ f.help }}", classes="text-caption text-medium-emphasis mb-1")
                        v3.VTextField(
                            v_if="['text', 'file', 'number', 'integer', 'numbers'].includes(f.type)",
                            **_bound(draft),
                            id=("'study-' + f.name",),
                            **common,
                        )
                        v3.VCheckbox(
                            v_if="f.type === 'boolean'",
                            **_bound(draft),
                            id=("'study-' + f.name",),
                            label=("f.label",),
                            **common,
                        )
                        v3.VSelect(
                            v_if="f.type === 'choice'",
                            **_bound(draft),
                            id=("'study-' + f.name",),
                            items=("f.choices",),
                            **common,
                        )
                        v3.VCombobox(
                            v_if="f.type === 'parameter'",
                            **_bound(draft),
                            id=("'study-' + f.name",),
                            items=("study_choices.parameters",),
                            **common,
                        )
                        v3.VCombobox(
                            v_if="['streams', 'parameters'].includes(f.type)",
                            **_bound(draft),
                            id=("'study-' + f.name",),
                            items=(
                                "f.type === 'streams' ? study_choices.streams "
                                ": study_choices.parameters",
                            ),
                            multiple=True,
                            chips=True,
                            closable_chips=True,
                            **common,
                        )
                        with html.Div(v_if="f.type === 'span'", classes="d-flex ga-2"):
                            v3.VTextField(**_bound(f"{draft}.start"), label="From", **common)
                            v3.VTextField(**_bound(f"{draft}.stop"), label="To", **common)
                        with html.Div(v_if="f.type === 'times'"):
                            with v3.VBtnToggle(
                                **_bound(f"{draft}.mode"), density="compact", mandatory=True
                            ):
                                v3.VBtn("Evenly spaced", value="grid", classes="text-none")
                                v3.VBtn("A list", value="list", classes="text-none")
                            with html.Div(
                                v_if=f"{draft}.mode === 'grid'", classes="d-flex ga-2 mt-1"
                            ):
                                v3.VTextField(**_bound(f"{draft}.start"), label="From", **common)
                                v3.VTextField(**_bound(f"{draft}.stop"), label="To", **common)
                                v3.VTextField(**_bound(f"{draft}.num"), label="How many", **common)
                            v3.VTextField(
                                v_else=True,
                                **_bound(f"{draft}.list"),
                                label="Values, separated by commas",
                                classes="mt-1",
                                **common,
                            )
                        with html.Div(v_if="f.type === 'run_time'"):
                            v3.VCheckbox(
                                **_bound(f"{draft}.dynamic"),
                                label="A dynamic run (otherwise the steady state)",
                                **common,
                            )
                            with html.Div(v_if=f"{draft}.dynamic"):
                                with html.Div(classes="d-flex ga-2 mb-1"):
                                    v3.VTextField(
                                        **_bound(f"{draft}.start"), label="From", **common
                                    )
                                    v3.VTextField(**_bound(f"{draft}.stop"), label="To", **common)
                                    v3.VTextField(
                                        **_bound(f"{draft}.unit"), label="Time unit", **common
                                    )
                                with html.Div(classes="d-flex ga-2"):
                                    v3.VTextField(
                                        **_bound(f"{draft}.every.start"),
                                        label="Save from",
                                        **common,
                                    )
                                    v3.VTextField(
                                        **_bound(f"{draft}.every.stop"), label="to", **common
                                    )
                                    v3.VTextField(
                                        **_bound(f"{draft}.every.num"), label="times", **common
                                    )
                        with html.Div(v_if="f.type === 'initial_state'", classes="d-flex ga-2"):
                            v3.VSelect(
                                **_bound(f"{draft}.mode"),
                                items=(
                                    "[{title: 'the default of each unit', value: 'default'}, "
                                    "{title: 'the steady state', value: 'steady'}, "
                                    "{title: 'a saved state', value: 'file'}]",
                                ),
                                **common,
                            )
                            v3.VTextField(
                                v_if=f"{draft}.mode === 'file'",
                                **_bound(f"{draft}.file"),
                                label="File (from the project folder)",
                                **common,
                            )
                        with html.Div(v_if="f.type === 'parameter_values'"):
                            with html.Div(
                                v_for=f"(r, ri) in {draft}", key="ri", classes="d-flex ga-2 mb-1"
                            ):
                                v3.VCombobox(
                                    **_bound("r.name"),
                                    items=("study_choices.parameters",),
                                    label="Parameter",
                                    **common,
                                )
                                v3.VTextField(
                                    **_bound("r.value"),
                                    label="Value",
                                    style="max-width: 160px;",
                                    **common,
                                )
                                v3.VBtn(
                                    icon="mdi-close",
                                    size="small",
                                    variant="text",
                                    click=f"{draft}.splice(ri, 1){_REDRAW}",
                                )
                            v3.VBtn(
                                "Add a parameter",
                                prepend_icon="mdi-plus",
                                size="small",
                                variant="text",
                                classes="text-none",
                                click=f"{draft}.push({{name: '', value: ''}}){_REDRAW}",
                            )
                        with html.Div(v_if="f.type === 'transforms'"):
                            with html.Div(
                                v_for=f"(r, ri) in {draft}", key="ri", classes="d-flex ga-2 mb-1"
                            ):
                                v3.VCombobox(
                                    **_bound("r.name"),
                                    items=("study_choices.parameters",),
                                    label="Parameter",
                                    **common,
                                )
                                v3.VSelect(
                                    **_bound("r.transform"),
                                    items=("f.choices",),
                                    style="max-width: 180px;",
                                    **common,
                                )
                                v3.VBtn(
                                    icon="mdi-close",
                                    size="small",
                                    variant="text",
                                    click=f"{draft}.splice(ri, 1){_REDRAW}",
                                )
                            v3.VBtn(
                                "Add a transform",
                                prepend_icon="mdi-plus",
                                size="small",
                                variant="text",
                                classes="text-none",
                                click=f"{draft}.push({{name: '', transform: 'positive_log'}}){_REDRAW}",
                            )
                        with html.Div(v_if="f.type === 'observables'"):
                            self._observables_editor(draft)
                        with html.Div(v_if="f.type === 'candidates'"):
                            with v3.VSheet(
                                v_for=f"(c, ci) in {draft}",
                                key="ci",
                                border=True,
                                rounded=True,
                                classes="pa-2 mb-2",
                            ):
                                self._candidate_fields("c")
                                v3.VBtn(
                                    "Remove this set",
                                    size="small",
                                    variant="text",
                                    color="error",
                                    classes="text-none",
                                    click=f"{draft}.splice(ci, 1){_REDRAW}",
                                )
                            v3.VBtn(
                                "Add a sensor set",
                                prepend_icon="mdi-plus",
                                size="small",
                                variant="tonal",
                                classes="text-none",
                                click=(
                                    f"{draft}.push({{name: '', sigma: '', "
                                    "observables: [{stream: '', channels: [], measures: {}}]})"
                                    f"{_REDRAW}"
                                ),
                            )
                        with html.Div(v_if="f.type === 'candidate'"):
                            v3.VCheckbox(
                                **_bound(f"{draft}.enabled"),
                                label="There are sensors in place",
                                **common,
                            )
                            with html.Div(v_if=f"{draft}.enabled"):
                                self._candidate_fields(draft)
                v3.VAlert(
                    "{{ study_error }}",
                    v_if="study_error",
                    type="error",
                    variant="tonal",
                    density="compact",
                    classes="text-caption mx-4 flex-grow-0",
                    style="white-space: pre-wrap;",
                )
                with v3.VCardActions():
                    v3.VSpacer()
                    v3.VBtn(
                        "Cancel",
                        variant="text",
                        classes="text-none",
                        click="study_dialog_open = false",
                    )
                    v3.VBtn(
                        "{{ study_editing ? 'Save' : 'Create' }}",
                        color="primary",
                        variant="flat",
                        classes="text-none",
                        loading=("busy",),
                        click=(
                            self.save_study_dialog,
                            "[study_name, study_kind, study_draft]",
                        ),
                    )

    @staticmethod
    def _table(table: str) -> None:
        """A results table bound to the state ``table`` (``steady_table``'s shape)."""
        with v3.VTable(density="compact", classes="mb-2"):
            with html.Thead():
                with html.Tr():
                    html.Th("")
                    html.Th("Unit")
                    html.Th("{{ c }}", v_for=f"c in {table}.columns", key="c", classes="ak-num")
            with html.Tbody():
                with html.Tr(v_for=f"r in {table}.rows", key="r.channel"):
                    html.Td("{{ r.channel }}", classes="font-weight-medium")
                    html.Td("{{ r.unit }}", classes="text-medium-emphasis text-caption")
                    html.Td("{{ v }}", v_for="(v, vi) in r.values", key="vi", classes="ak-num")

    def _results_dialog(self) -> None:
        with v3.VDialog(v_model=("results_open",), max_width=980, scrollable=True):
            with v3.VCard():
                v3.VCardTitle("{{ results_title }}", classes="pt-4")
                with v3.VCardText():
                    v3.VAlert(
                        "{{ results_note }}",
                        v_if="results_note",
                        type="warning",
                        variant="tonal",
                        density="compact",
                        classes="mb-3 text-body-2",
                    )
                    with html.Div(v_if="results_mode === 'steady'"):
                        with html.Div(v_if="results_measured.rows.length"):
                            html.Div("Measured", classes="text-subtitle-2")
                            self._table("results_measured")
                            with v3.VExpansionPanels(variant="accordion", classes="mb-2"):
                                with v3.VExpansionPanel(title="Flow and species"):
                                    with v3.VExpansionPanelText():
                                        self._table("results_streams")
                        with html.Div(v_else=True):
                            html.Div("Flow and species", classes="text-subtitle-2")
                            self._table("results_streams")
                    with html.Div(v_if="results_mode === 'time'"):
                        with html.Div(classes="d-flex ga-2 mb-3"):
                            v3.VSelect(
                                label="Stream",
                                model_value=("results_stream",),
                                items=("results_stream_choices",),
                                update_modelValue=(self.choose_results_stream, "[$event]"),
                                density="compact",
                                variant="outlined",
                                hide_details=True,
                                style="max-width: 240px;",
                            )
                            v3.VSelect(
                                label="Channels",
                                model_value=("results_channels",),
                                items=("results_channel_choices",),
                                update_modelValue=(self.show_charts, "[$event]"),
                                multiple=True,
                                chips=True,
                                closable_chips=True,
                                density="compact",
                                variant="outlined",
                                hide_details=True,
                            )
                        with html.Div(classes="ak-charts"):
                            html.Div(v_for="c in results_charts", key="c.channel", v_html="c.svg")
                    with html.Div(v_if="results_mode === 'files'"):
                        html.Div(
                            "This kind of study writes its results as files:",
                            classes="text-body-2 mb-1",
                        )
                        with v3.VList(density="compact"):
                            v3.VListItem(
                                v_for="f in results_files",
                                key="f",
                                title=("f",),
                                prepend_icon="mdi-file-outline",
                            )
                    html.Div(
                        "Saved in {{ results_folder }}",
                        classes="text-caption text-medium-emphasis mt-2",
                    )
                with v3.VCardActions():
                    v3.VSpacer()
                    v3.VBtn(
                        "Close", variant="text", classes="text-none", click="results_open = false"
                    )

    def _study_dialog(self) -> None:
        with v3.VDialog(v_model=("study_confirm_open",), max_width=560, persistent=True):
            with v3.VCard(title=("study_confirm_title",)):
                with v3.VCardText():
                    html.Div("{{ study_confirm_text }}")
                    v3.VCheckbox(
                        v_if="study_confirm_kind === 'run'",
                        v_model=("study_confirm_dont_ask",),
                        label="Don't ask again before replacing results",
                        hint="You can turn this back on in the Studies tab.",
                        persistent_hint=True,
                        density="compact",
                        classes="mt-2",
                    )
                with v3.VCardActions(classes="flex-wrap ga-1"):
                    v3.VSpacer()
                    v3.VBtn(
                        "Cancel",
                        click=(self.answer_study, "['cancel']"),
                        variant="text",
                        classes="text-none",
                    )
                    v3.VBtn(
                        "Copy first",
                        click=(self.answer_study, "['copy']"),
                        variant="tonal",
                        classes="text-none",
                    )
                    v3.VBtn(
                        "{{ study_confirm_label }}",
                        click=(self.answer_study, "['go']"),
                        color="error",
                        variant="flat",
                        classes="text-none",
                    )

    def _unsaved_dialog(self) -> None:
        with v3.VDialog(v_model=("unsaved_open",), max_width=520, persistent=True):
            with v3.VCard(title="Unsaved edits"):
                v3.VCardText("{{ unsaved_text }}")
                with v3.VCardActions(classes="flex-wrap ga-1"):
                    v3.VSpacer()
                    v3.VBtn(
                        "Cancel",
                        click=(self.answer_unsaved, "['cancel']"),
                        variant="text",
                        classes="text-none",
                    )
                    v3.VBtn(
                        "Discard edits",
                        click=(self.answer_unsaved, "['discard']"),
                        color="error",
                        variant="tonal",
                        classes="text-none",
                    )
                    v3.VBtn(
                        "Save first",
                        v_if="unsaved_can_save",
                        click=(self.answer_unsaved, "['save']"),
                        color="primary",
                        variant="flat",
                        classes="text-none",
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
                "{{ sel_dirty ? 'Apply changes' : 'Apply' }}",
                click=(self.apply_properties, "[sel_rows]"),
                color="primary",
                block=True,
                variant=("sel_dirty ? 'flat' : 'tonal'",),
                classes="mb-3 text-none",
            )
            html.Div(
                "Selecting another unit applies these changes.",
                v_if="sel_dirty",
                classes="text-caption text-medium-emphasis mb-2",
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
                    update_modelValue="sel_dirty = true",
                    **common,
                )
                v3.VSelect(
                    v_else_if="row.widget === 'multi'",
                    items=("row.items",),
                    multiple=True,
                    chips=True,
                    closable_chips=True,
                    v_model="row.value",
                    update_modelValue="sel_dirty = true",
                    **common,
                )
                v3.VTextField(
                    v_else_if="row.widget === 'readonly'",
                    disabled=True,
                    model_value=("row.value",),
                    **common,
                )
                v3.VTextField(
                    v_else=True, v_model="row.value", update_modelValue="sel_dirty = true", **common
                )
            with v3.VSheet(
                v_if="focus_help", rounded=True, border=True, classes="pa-2 mt-1 text-caption"
            ):
                html.Div("{{ focus_label }}", classes="font-weight-medium mb-1")
                html.Div("{{ focus_help }}", classes="text-medium-emphasis")


def _digest(path: Path) -> str | None:
    """sha256 of a file's bytes (``None`` when it does not exist)."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None
