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
from pathlib import Path

import yaml
from trame.app import asynchronous

from aquakin_ui import checks, disk, files, forms, page, studies, study_page, unit_form
from aquakin_ui.catalog import Catalog
from aquakin_ui.document import Flowsheet, edge_id
from aquakin_ui.icons import CATEGORY_LABELS, COLOURS, ICONS, SVG, icon_name
from aquakin_ui.jobs import StudyRun
from aquakin_ui.settings import Settings
from aquakin_ui.worker import AquakinWorker

UNDO_DEPTH = 50  # edits of the flowsheet Undo can take back


def _needs_python(kind: dict) -> bool:
    """Whether a kind has a required setting the form cannot edit (an SBR's
    phases and settling model)."""
    return any(s["required"] and s["type"] == "unsupported" for s in kind["settings"])


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
        self._check_task: asyncio.Task | None = None
        self._pending_rows = None
        self._runs: dict[str, StudyRun] = {}
        self._listed: list[dict] = []  # the worker's list_studies, as last asked
        self._loop: asyncio.AbstractEventLoop | None = None
        # The file opened, and its sha256 (None when it had gone by the time it was read).
        self._opened: tuple[Path, str | None] | None = None
        self._declined_digest: str | None = None  # a version on disk the user kept edits over
        self._project_seen: tuple | None = None  # the project folder and its files, last look
        self._study_action: tuple[str, str] | None = None
        self._study_form: dict | None = None  # the worker's study_form, asked once
        self._saved: str | None = None  # the document as last opened or saved
        self._show_when_done: set[str] = set()  # runs whose results open when they finish
        self._results: dict = {}  # the results on show
        self._study_flowsheet: str | None = None  # the spec the study being edited follows
        self._form_revision = 0  # makes a reset of the unit form reach the page
        self._selection_revision = 0  # makes a kept selection reach the canvas
        self._history: list[tuple] = []  # document snapshots to undo to, oldest first
        self._after_unsaved = None  # what the unsaved-changes question goes on to do
        self._init_state(path)
        if path is not None:
            self.open(path, quiet=True)
        self.publish(checks=False)
        self.server.controller.on_server_ready.add(lambda **_: self.schedule_checks())
        self.server.controller.on_server_ready.add(
            lambda **_: asynchronous.create_task(self.watch_disk())
        )
        self._build()

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
            elif s["type"] in ("port_map", "scheduled_port_map"):
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
            if info:
                species = [s["name"] for s in info["species"]]
                for s in self.catalog.settings(kind):
                    suggested = forms.suggest_species(s, None, species)
                    if suggested is not None:
                        settings[s["name"]] = suggested
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
                self._show_kept_selection()
                return
        self.state.selected = s["nodes"][0] if len(s["nodes"]) == 1 else None
        if self.state.selected is not None:
            self.state.panel = "properties"  # its form, from whichever tab was open
        self.refresh_selection()

    def _show_kept_selection(self) -> None:
        """Show the canvas the unit Python keeps selected, over the one the canvas
        has just selected itself. trame sends a value only when it differs from
        the one it last pushed (``state.dirty`` included), and the nodes may not
        have changed, so the kept node carries a new revision."""
        self._selection_revision += 1
        uid = self.state.selected
        self.state.nodes = [
            {
                **{k: v for k, v in n.items() if k != "revision"},
                "selected": n["id"] == uid,
                **({"revision": self._selection_revision} if n["id"] == uid else {}),
            }
            for n in self.state.nodes
        ]

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
        self.state.sel_errors = {}
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
            # Shown under the field too, where it stays until the form is applied.
            key = getattr(exc, "key", None)
            self.state.sel_errors = {key: str(exc).split(": ", 1)[-1]} if key else {}
            return False
        self.state.sel_errors = {}
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
        self._resend_rows(self.state.sel_rows)

    # -- work off the page's thread -----------------------------------------

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

    def _worker_study(self, command: str, study: dict, then=None, **fields) -> None:
        def done(result):
            if isinstance(result, Exception):
                self.say(f"{command.replace('_', ' ')} failed: {result}")
                return
            if then is not None:
                then()
            self.refresh_studies()

        self._off_page(lambda: self.worker.request(command, path=study["path"], **fields), done)

    # -- the page's state ---------------------------------------------------

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
        s.results_lines = []
        s.results_tables = []
        s.panel = "properties"
        s.dark = False
        s.focus_help = ""
        s.focus_label = ""
        s.spec_path = str(path) if path else ""
        s.can_undo = False
        s.sel_dirty = False
        s.sel_errors = {}
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
                        "summary": k["summary"]
                        + (
                            " Some of its settings can only be set from Python."
                            if _needs_python(k)
                            else ""
                        ),
                        "python": _needs_python(k),
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

    # -- the page (page.py) ------------------------------------------------

    _build = page._build
    _toolbar = page._toolbar
    _palette = page._palette
    _side_panel = page._side_panel
    _unsaved_dialog = page._unsaved_dialog
    _confirm_dialog = page._confirm_dialog
    _unit_form = page._unit_form

    # -- the unit form's redrawing choices (unit_form.py) ------------------

    reshape_form = unit_form.reshape_form
    _resend_rows = unit_form.resend_rows

    # -- the Studies tab and dialogs (study_page.py) -----------------------

    _studies_panel = study_page._studies_panel
    _study_settings_dialog = study_page._study_settings_dialog
    _results_dialog = study_page._results_dialog
    _study_dialog = study_page._study_dialog

    # -- aquakin's checks (checks.py) --------------------------------------

    model_info = checks.model_info
    _unit_check = checks._unit_check
    _ask_unit = checks._ask_unit
    _check_unit = checks._check_unit
    _plant_request = checks._plant_request
    _ask_plant = checks._ask_plant
    _check_plant = checks._check_plant
    unit_errors = checks.unit_errors
    schedule_checks = checks.schedule_checks
    _run_checks = checks._run_checks

    # -- the flowsheet's file (files.py) -----------------------------------

    open = files.open_spec
    _snapshot = files._snapshot
    _unsaved = files._unsaved
    _spec_file = files._spec_file
    _apply_pending = files._apply_pending
    save_button = files.save_button
    request_open = files.request_open
    _unless_unsaved = files._unless_unsaved
    answer_unsaved = files.answer_unsaved
    save = files.save
    _save_refusal = files._save_refusal
    reload_if_changed = files.reload_if_changed

    # -- other programs' changes to the project (disk.py) -------------------

    follow_disk = disk.follow_disk
    watch_disk = disk.watch_disk

    # -- the project's studies (studies.py) --------------------------------

    _project = studies._project
    refresh_studies = studies.refresh_studies
    _study = studies._study
    run_study = studies.run_study
    unfreeze_study = studies.unfreeze_study
    freeze_study = studies.freeze_study
    copy_study = studies.copy_study
    simulate = studies.simulate
    _simulation = studies._simulation
    _run_simulation = studies._run_simulation
    show_results = studies.show_results
    choose_results_stream = studies.choose_results_stream
    show_charts = studies.show_charts
    _plant_choices = studies._plant_choices
    _open_study_dialog = studies._open_study_dialog
    _with_form = studies._with_form
    new_study = studies.new_study
    edit_study = studies.edit_study
    set_study_kind = studies.set_study_kind
    save_study_dialog = studies.save_study_dialog
    _write_study = studies._write_study
    _ask = studies._ask
    set_ask_replace = studies.set_ask_replace
    answer_study = studies.answer_study
    _start_run = studies._start_run
    cancel_run = studies.cancel_run
    _on_run_event = studies._on_run_event
    _apply_run_event = studies._apply_run_event
    _list = studies._list
    _mark_running = studies._mark_running
    stop_runs = studies.stop_runs
