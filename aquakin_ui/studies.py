"""The project's studies: listing them through the worker, running them (each run
a :class:`~aquakin_ui.jobs.StudyRun` process of its own, never a worker
request), freezing, copying, Simulate, the results view and the study dialog's
actions, and the questions asked before results are replaced or deleted.

Each function is bound onto :class:`~aquakin_ui.app.Editor` as a method
(``self`` is the editor); the Studies tab and its dialogs are drawn by
:mod:`aquakin_ui.study_page`.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from aquakin_ui.jobs import StudyRun
from aquakin_ui.results import SIMULATE, analysis_view, channels_of, line_chart, steady_table
from aquakin_ui.study_form import draft_for, fields_for, missing, study_from
from aquakin_ui.worker import WorkerError

if TYPE_CHECKING:
    from aquakin_ui.app import Editor


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


def _project(self: Editor) -> Path | None:
    """The project folder: the flowsheet file's (``None`` before it has one)."""
    spec = self._spec_file()
    return spec.parent if spec is not None else None


def refresh_studies(self: Editor) -> None:
    """List the project's studies with where each stands (off the page's thread)."""
    project = self._project()
    if project is None:
        self._list([])
        return

    def done(studies):
        if isinstance(studies, Exception):
            self.say(f"could not list the studies: {studies}")
            return
        self._list(studies)

    self._off_page(lambda: self.worker.request("list_studies", project=str(project)), done)


def _study(self: Editor, name: str) -> dict | None:
    return next((st for st in self.state.studies if st["name"] == name), None)


def run_study(self: Editor, name: str, pending: list[dict] | None = None) -> None:
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


def unfreeze_study(self: Editor, name: str) -> None:
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


def freeze_study(self: Editor, name: str) -> None:
    study = self._study(name)
    if study is not None:
        self._worker_study("freeze_study", study)


def copy_study(self: Editor, name: str, then=None) -> None:
    """Copy a study beside it as ``<name>_copy`` (``_copy2``, ... if taken)."""
    study = self._study(name)
    if study is None:
        return
    taken = {st["name"] for st in self.state.studies}
    new, i = f"{name}_copy", 2
    while new in taken:
        new, i = f"{name}_copy{i}", i + 1
    self._worker_study("copy_study", study, then=then, name=new)


def simulate(self: Editor, pending: list[dict] | None = None) -> None:
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


def _simulation(self: Editor) -> str:
    """The name of the open spec's Simulate study."""
    spec = self._spec_file()
    return simulation_name(spec) if spec is not None else SIMULATION


def _run_simulation(self: Editor, spec: Path) -> None:
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
        elif study.get("reactor"):
            raise ValueError(f"study {name} studies a reactor, not the flowsheet")
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
        self._list(studies)
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


def show_results(self: Editor, name: str) -> None:
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
            f"Run the study again for up-to-date results: {'; '.join(reasons)}." if reasons else ""
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


def choose_results_stream(self: Editor, stream: str) -> None:
    """Show ``stream``'s charts: its measurements, or its flow and first species."""
    self.state.results_stream = stream
    channels = channels_of(self._results, stream) if stream else []
    self.state.results_channel_choices = channels
    measured = list((self._results.get("measured") or {}).get(stream, {}))
    self.show_charts(measured or channels[:4])


def show_charts(self: Editor, channels: list[str]) -> None:
    """Draw a chart per channel of the chosen stream (each on its own axis)."""
    stream = self.state.results_stream
    self.state.results_channels = list(channels)
    self.state.results_charts = [
        {"channel": c, "svg": line_chart(self._results, stream, c)} for c in channels
    ]


def _plant_choices(self: Editor) -> dict:
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
    self: Editor, form: dict, *, title: str, kind: str, study: dict | None, editing: str
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


def _with_form(self: Editor, then: Callable[[dict], None]) -> None:
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


def new_study(self: Editor) -> None:
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


def edit_study(self: Editor, name: str) -> None:
    """Open the dialog on a study's settings."""
    study = self._study(name)
    if study is None:
        return
    if study.get("reactor"):
        # The dialog shows a flowsheet study's fields; saving would drop the reactor's.
        self.say(f"{name} studies a reactor: edit its study.yaml to change it")
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


def set_study_kind(self: Editor, kind: str, draft: dict) -> None:
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


def save_study_dialog(self: Editor, name: str, kind: str, draft: dict) -> None:
    """Write the study the dialog describes (create it, or replace its settings)."""
    self._with_form(lambda form: self._write_study(form, name, kind, draft))


def _write_study(self: Editor, form: dict, name: str, kind: str, draft: dict) -> None:
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
        work = lambda: self.worker.request("create_study", project=project, name=name, study=study)

    def done(result):
        if isinstance(result, Exception):
            self.state.study_error = str(result)
            return
        self.state.study_dialog_open = False
        self.say(f"{'updated' if editing else 'created'} study {name}", error=False)
        self.refresh_studies()

    self._off_page(work, done)


def _ask(self: Editor, action: str, name: str, title: str, text: str, confirm: str) -> None:
    self._study_action = (action, name)
    self.state.study_confirm_kind = action
    self.state.study_confirm_dont_ask = False
    self.state.study_confirm_title = title
    self.state.study_confirm_text = text
    self.state.study_confirm_label = confirm
    self.state.study_confirm_open = True


def set_ask_replace(self: Editor, ask: bool) -> None:
    """Whether a run asks before replacing a study's results; kept as the
    user's preference (the Studies tab's switch, or "Don't ask again")."""
    self.state.ask_replace = ask
    if ask != self.settings["ask_before_replacing"]:
        if not self.settings.set("ask_before_replacing", ask):
            self.say(f"could not save the preference to {self.settings.path}")


def answer_study(self: Editor, choice: str) -> None:
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


def _start_run(self: Editor, study: dict, *, replace: bool) -> None:
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


def cancel_run(self: Editor, name: str) -> None:
    """Stop a study's run: this editor's own through its process, one another
    program started through the worker (it stops at its next checkpoint)."""
    run = self._runs.get(name)
    if run is not None:
        run.cancel()
        self.state.study_progress = {
            **self.state.study_progress,
            name: {**self.state.study_progress.get(name, {}), "message": "cancelling"},
        }
        return
    study = self._study(name)
    if study is not None and study.get("run"):
        self._worker_study(
            "cancel_study", study, then=lambda: self.say(f"{name}: cancelling", error=False)
        )


def _on_run_event(self: Editor, name: str, event: dict) -> None:
    # Called on the run's reader thread; the page's state is changed on its loop.
    if self._loop is not None and self._loop.is_running():
        self._loop.call_soon_threadsafe(self._apply_run_event, name, event)
    else:
        self._apply_run_event(name, event)


def _apply_run_event(self: Editor, name: str, event: dict) -> None:
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


def _list(self: Editor, studies: list[dict]) -> None:
    """Show the worker's ``list_studies``: each study ``running`` when this
    editor runs it or the worker reports a run in progress (``run``: when it
    started and its progress, from whichever program)."""
    self._listed = studies
    self._mark_running()


def _mark_running(self: Editor) -> None:
    self.state.studies = [
        {**st, "run": st["running"], "running": st["name"] in self._runs or bool(st["running"])}
        for st in self._listed
    ]
    self.state.simulating = self._simulation() in self._runs


def stop_runs(self: Editor) -> None:
    """Cancel every run (when the editor closes)."""
    for run in list(self._runs.values()):
        run.cancel()
