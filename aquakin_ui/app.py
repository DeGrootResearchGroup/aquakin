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
import re
from collections.abc import Callable
from pathlib import Path

import yaml
from trame.app import asynchronous

from aquakin_ui import checks, files, forms, page, study_page
from aquakin_ui.catalog import Catalog
from aquakin_ui.document import Flowsheet, edge_id
from aquakin_ui.icons import CATEGORY_LABELS, COLOURS, ICONS, SVG, icon_name
from aquakin_ui.jobs import StudyRun
from aquakin_ui.results import SIMULATE, analysis_view, channels_of, line_chart, steady_table
from aquakin_ui.settings import Settings
from aquakin_ui.study_form import draft_for, fields_for, missing, study_from
from aquakin_ui.worker import AquakinWorker, WorkerError

UNDO_DEPTH = 50  # edits of the flowsheet Undo can take back


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


def _needs_python(kind: dict) -> bool:
    """Whether a kind has a required setting the form cannot edit (an SBR's
    phases, a dosing unit's reagent)."""
    return any(s["required"] and s["type"] in ("unsupported", "reagent") for s in kind["settings"])


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
        self._loop = None
        # The file opened, and its sha256 (None when it had gone by the time it was read).
        self._opened: tuple[Path, str | None] | None = None
        self._study_action = None
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
        self._build()

    # -- aquakin ------------------------------------------------------------

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
        # The rows equal the last ones sent, and the page keeps the edits just
        # declined unless it is sent something new: a revision mark makes them so.
        self._form_revision += 1
        rows = list(self.state.sel_rows)
        if rows:
            rows[0] = {**rows[0], "revision": self._form_revision}
            self.state.sel_rows = rows

    def reshape_form(self, rows: list[dict], i: int, value) -> None:
        """A choice that changes which rows the form has (aeration mode, model)."""
        rows = [dict(r) for r in rows]
        rows[i]["value"] = value
        if not self.apply_properties(rows):
            return
        # The new rows may suggest values the old ones could not (the aerated
        # species of the unit's model): take them, so the unit builds at once.
        uid = self.state.selected
        try:
            _, settings = forms.read_rows(self.doc, uid, self.state.sel_rows)
        except ValueError:
            return
        if settings != self.doc.units[uid].settings:
            self.doc.apply(uid, settings, self.doc.units[uid].label)
            self.publish()

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

    def run_study(self, name: str, pending: list[dict] | None = None) -> None:
        """Run a study; ask first when that would replace its results."""
        if not self._apply_pending(pending):
            return
        study = self._study(name)
        if study is None or name in self._runs:
            return
        if not study.get("frozen") and self._unsaved():
            # A live study runs the flowsheet's file: bring it up to the drawing.
            self.save(then=lambda: self.run_study(name), path=self._spec_file())
            return
        if not study["has_results"]:
            # Not run, failed or cancelled: no results to replace.
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
        if not study["has_results"]:  # nothing to delete
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

    def simulate(self, pending: list[dict] | None = None) -> None:
        """Run the plant: save the flowsheet if it changed, set up the project's
        ``simulation`` study the first time (the plant at steady state, saving the
        streams that leave it), run it, and show its results when it finishes."""
        if not self._apply_pending(pending):
            return
        if self._simulation() in self._runs:
            self.state.panel = "studies"
            return
        spec = self._spec_file()
        if spec is None:
            self.say(
                "the flowsheet has no file yet: type where to keep it in the path field "
                "(a .yaml file in the folder of its project), then Simulate"
            )
            return
        if self._unsaved() or not spec.exists():
            self.save(then=lambda: self._run_simulation(spec), path=spec)
        else:
            self._run_simulation(spec)

    def _simulation(self) -> str:
        """The name of the open spec's Simulate study."""
        spec = self._spec_file()
        return simulation_name(spec) if spec is not None else SIMULATION

    def _run_simulation(self, spec: Path) -> None:
        project = spec.parent
        name = simulation_name(spec)

        def work():
            listed = self.worker.request("list_studies", project=str(project))
            study = next((st for st in listed if st["name"] == name), None)
            if study is None:
                settings = self.worker.request("simulation_study", spec=str(spec))
                self.worker.request(
                    "create_study",
                    project=str(project),
                    name=name,
                    study={**settings, "flowsheet": str(spec)},
                )
            elif study["flowsheet"] is None:  # its study.yaml cannot be read
                raise ValueError("; ".join(study["reasons"]))
            elif not study["frozen"] and Path(study["flowsheet"]).resolve() != spec.resolve():
                # The study must follow this spec: a copy or a hand edit may point
                # it at another file of the folder. (A frozen one is refused below.)
                settings = self.worker.request("read_study", path=study["path"])
                self.worker.request(
                    "update_study", path=study["path"], study={**settings, "flowsheet": str(spec)}
                )
            return self.worker.request("list_studies", project=str(project))

        def done(studies):
            if isinstance(studies, Exception):
                self.say(f"could not set up the simulation: {studies}")
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

    def show_results(self, name: str) -> None:
        """Open a study's results: a simulation at steady state as tables, a run
        in time as a chart per channel, any other kind as its own view
        (``results.ANALYSIS_VIEWS``). Where the study stands comes with the
        results, so a run that just finished is shown as it ended."""
        study = self._study(name)
        if study is None:
            return

        def done(results):
            if isinstance(results, Exception):
                self.say(f"could not read the results of {name}: {results}")
                return
            s = self.state
            self._results = results
            s.results_title = f"Results of {name}"
            s.results_folder = str(Path(study["path"]) / "results")
            s.results_files = results["files"]
            reasons = results["reasons"]
            s.results_note = (
                f"Run the study again for up-to-date results: {'; '.join(reasons)}."
                if reasons
                else ""
            )
            units = results.get("units") or {}
            if results["kind"] != SIMULATE:
                s.results_mode = "analysis"
                view = analysis_view(results)
                s.results_lines = view["lines"]
                s.results_tables = view["tables"]
                s.results_charts = view["charts"]
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
        self.show_charts(measured or channels[:4])

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

    def _open_study_dialog(
        self, form: dict, *, title: str, kind: str, study: dict | None, editing: str
    ):
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

    def _with_form(self, then: Callable[[dict], None]) -> None:
        """Run ``then(form)`` once the worker's study form is known."""
        if self._study_form is not None:
            then(self._study_form)
            return

        def done(form):
            if isinstance(form, Exception):
                self.say(f"could not read the study settings: {form}")
                return
            self._study_form = form
            then(form)

        self._off_page(lambda: self.worker.request("study_form"), done)

    def new_study(self) -> None:
        """Open the dialog for a new study (a simulation to start with)."""
        if self._project() is None:
            self.say("save the flowsheet first: its studies are kept in its folder")
            return
        self.state.study_name = ""
        self._with_form(
            lambda form: self._open_study_dialog(
                form, title="New study", kind="simulate", study=None, editing=""
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
                lambda form: self._open_study_dialog(
                    form,
                    title=f"Edit {name}",
                    kind=settings["kind"],
                    study=settings,
                    editing=study["path"],
                )
            )

        self._off_page(lambda: self.worker.request("read_study", path=study["path"]), opened)

    def set_study_kind(self, kind: str, draft: dict) -> None:
        """Change the dialog's kind, keeping the values of the fields both kinds have."""

        def change(form: dict) -> None:
            fresh = draft_for(form, kind)
            for name in fresh:
                if name in draft:
                    fresh[name] = draft[name]
            self.state.study_kind = kind
            self.state.study_fields = fields_for(form, kind)
            self.state.study_draft = fresh

        self._with_form(change)

    def save_study_dialog(self, name: str, kind: str, draft: dict) -> None:
        """Write the study the dialog describes (create it, or replace its settings)."""
        self._with_form(lambda form: self._write_study(form, name, kind, draft))

    def _write_study(self, form: dict, name: str, kind: str, draft: dict) -> None:
        try:
            study = study_from(form, kind, draft)
        except ValueError as exc:
            self.state.study_error = str(exc)
            return
        editing = self.state.study_editing
        name = (name or "").strip()
        if not editing and not name:
            self.state.study_error = "Give the study a name."
            return
        needed = missing(form, kind, study)
        if needed:
            self.state.study_error = needed
            return
        if editing:
            if self._study_flowsheet:
                study["flowsheet"] = self._study_flowsheet
            work = lambda: self.worker.request("update_study", path=editing, study=study)
        else:
            study["flowsheet"] = str(self._spec_file())
            project = str(self._project())
            work = lambda: self.worker.request(
                "create_study", project=project, name=name, study=study
            )

        def done(result):
            if isinstance(result, Exception):
                self.state.study_error = str(result)
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
                self.say(f"{name}: {event['state']}", error=bool(event["reasons"]))
                if show and event["has_results"]:
                    self.show_results(name)
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
                            " Some of its settings can only be set from Python or in the spec file."
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

    # -- the Studies tab and dialogs (study_page.py) -----------------------

    _studies_panel = study_page._studies_panel
    _observables_editor = staticmethod(study_page._observables_editor)
    _candidate_fields = study_page._candidate_fields
    _study_settings_dialog = study_page._study_settings_dialog
    _table = staticmethod(study_page._table)
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

    open = files.open
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
