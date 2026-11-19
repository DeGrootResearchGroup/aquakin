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
        self._runs: dict[str, StudyRun] = {}
        self._loop = None
        self._opened: tuple[Path, str] | None = None  # the file opened, and its sha256
        self._study_action = None
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
            self.state.spec_path = str(path)
            self.state.selected = None
            self.state.fit_key += 1
            self._unit_checks.clear()
            self._plant_check = (None, None)
            if not quiet:
                self.publish()
                self.say(f"opened {path}", error=False)
            self.refresh_studies()

        self._off_page(lambda: self.worker.request("read_spec", path=str(path)), done)

    def save(self) -> None:
        """Save through aquakin's own spec writer: settings checked and filled in,
        the aquakin version and file hashes recorded, outside model and influent
        files copied into the project, and the file replaced atomically."""
        path = Path(self.state.spec_path).resolve()
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
            self._opened = (path, _digest(path))
            self._plant_check = (None, None)
            self.publish()
            self.say(f"saved {path}", error=False)

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

    def _project(self) -> Path:
        return Path(self.state.spec_path).resolve().parent

    def refresh_studies(self) -> None:
        """List the project's studies with where each stands (off the page's thread)."""
        project = self._project()

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
        if study["state"] in ("not run",):
            self._start_run(study, replace=False)
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
        self.state.study_confirm_title = title
        self.state.study_confirm_text = text
        self.state.study_confirm_label = confirm
        self.state.study_confirm_open = True

    def answer_study(self, choice: str) -> None:
        """The answer to a study question: ``"cancel"``, ``"copy"`` (copy, then go
        ahead) or ``"go"``."""
        self.state.study_confirm_open = False
        action, self._study_action = self._study_action, None
        if action is None or choice == "cancel":
            return
        kind, name = action

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
            if event["event"] == "done":
                self.say(f"{name}: {event['state']}", error=event["state"] != "fresh")
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
                            select=(self.on_select, "[$event]"),
                            style="position: absolute; inset: 0;",
                        )
                    self._side_panel()
                self._confirm_dialog()
                self._study_dialog()
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
            v3.VBtn(
                "Open",
                click=lambda: self.open(),
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
                classes="mr-4 text-none",
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
        with html.Div(classes="d-flex align-center mb-2"):
            html.Div("Studies of this project", classes="text-overline")
            v3.VSpacer()
            v3.VBtn(
                icon="mdi-refresh",
                size="small",
                variant="text",
                click=self.refresh_studies,
                title="Check every study again",
            )
        html.Div(
            "No studies yet: add a folder with a study.yaml under studies/.",
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
                v3.VBtn("Copy", size="small", click=(self.copy_study, "[st.name]"))

    def _study_dialog(self) -> None:
        with v3.VDialog(v_model=("study_confirm_open",), max_width=560, persistent=True):
            with v3.VCard(title=("study_confirm_title",)):
                v3.VCardText("{{ study_confirm_text }}")
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


def _digest(path: Path) -> str | None:
    """sha256 of a file's bytes (``None`` when it does not exist)."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None
