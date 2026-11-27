"""The model panel: a model's reactions and parameters, edited and saved.

The page edits a draft of the model (:mod:`aquakin_ui.model_form`) and sends it
whole when a field loses focus; the edits are then checked in the background
with the flowsheet's other checks (``schedule_checks``): each edited expression
by itself (``check_expression``: its error and position, the names it uses, the
names the model lacks), then the whole edited model (``check_model_edits``),
each answer cached by what it checked. Apply checks the edits once more and
saves them through the worker (``save_model_edits``), which writes the model
file and the flowsheet together: a project model in place; a shipped model as a
new project model, named in a dialog that lists the units that will use it.
Each function is bound onto :class:`~aquakin_ui.app.Editor` as a method
(``self`` is the editor).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from aquakin_ui import model_form
from aquakin_ui.document import Flowsheet
from aquakin_ui.files import _digest
from aquakin_ui.worker import WorkerError

if TYPE_CHECKING:
    import asyncio

    from aquakin_ui.app import Editor


@dataclass
class ModelPanel:
    """What the panel holds on the Python side.

    Attributes
    ----------
    document : dict or None
        The worker's ``model_document`` of the model on show.
    draft : dict or None
        The draft as the page last sent it (or as drawn).
    expressions : dict
        ``check_expression`` answers, keyed by everything they depend on.
    checked : tuple
        ``(edits as JSON, check_model_edits answer)`` of the latest whole check.
    plan : dict or None
        The save the new-model dialog confirms.
    """

    document: dict | None = None
    draft: dict | None = None
    expressions: dict = field(default_factory=dict)
    checked: tuple[str | None, dict | None] = (None, None)
    plan: dict | None = None


def init_model_state(self: Editor) -> None:
    """The page state the panel and its dialog read."""
    s = self.state
    s.model_choices = []
    s.model_name = None
    s.model_source = ""
    s.model_draft = {"reactions": [], "parameters": []}
    s.model_names = {}
    s.model_dirty = False
    s.model_errors = {}
    s.model_pointers = {}
    s.model_uses = {}
    s.model_offers = []
    s.model_general = []
    s.model_valid = False
    s.model_dialog_open = False
    s.model_new_name = ""
    s.model_dialog_units = []
    s.model_dialog_error = ""
    s.model_dialog_warning = ""
    s.model_dialog_new = True  # naming a model made from a shipped one

    @s.change("panel")
    def _model_tab(panel, **_):
        if panel == "model" and self._model.document is None and s.model_choices:
            self.open_model(_default_model(self))


def _default_model(self: Editor) -> str:
    """The selected unit's model, else the first the flowsheet uses."""
    choices = list(self.state.model_choices)
    uid = self.state.selected
    model = self.doc.units[uid].settings.get("model") if uid in self.doc.units else None
    return choices[choices.index(model)] if model in choices else choices[0]


def refresh_model_choices(self: Editor) -> None:
    """The models the panel offers: those the flowsheet's units use. A model on
    show that the flowsheet no longer uses, or now reads from another file (a
    flowsheet opened, an edit undone), leaves the panel."""
    used = list(self.doc.to_spec(layout=False)["models"])
    self.state.model_choices = used
    shown = self._model.document
    if shown is not None and (
        shown["name"] not in used or shown["path"] != _path_of(self, shown["name"])
    ):
        self._model = ModelPanel()
        self.state.model_name = None
        self.state.model_draft = {"reactions": [], "parameters": []}
        self.state.model_dirty = False


def _path_of(self: Editor, name: str) -> str | None:
    """The absolute file of a project model (``None`` for a shipped one), as
    ``model_document`` reports it."""
    path = self.doc.model_files().get(name)
    return None if path is None else str(Path(path).resolve())


def open_model(self: Editor, name: str) -> None:
    """Show model ``name`` in the panel (read by the worker, off the page's thread)."""
    path = _path_of(self, name)

    def done(document):
        if isinstance(document, Exception):
            self.say(f"could not read model {name}: {document}")
            return
        self._model = ModelPanel(document=document, draft=model_form.draft_of(document))
        self.state.model_name = name
        self.state.model_source = (
            f"{name} is a shipped model: Apply saves the edits as a new model of the "
            f"project, which extends it."
            if document["source"] == "shipped"
            else f"Apply writes the edits into {document['path']}."
        )
        self.state.model_names = {
            "species": document["species"],
            "parameters": sorted({p["name"] for p in document["parameters"]}),
            "conditions": document["conditions"],
            "expressions": document["expressions"],
        }
        _show_draft(self)

    self._off_page(lambda: self.worker.request("model_document", name=name, path=path), done)


def _show_draft(self: Editor) -> None:
    """Draw the draft afresh. It carries a new revision: trame sends a value
    only when it differs from the one it last pushed, and the page may hold
    edits of the draft it was sent last."""
    m = self._model
    if m.document is None or m.draft is None:
        return  # no model open
    self._form_revision += 1
    self.state.model_draft = {**m.draft, "revision": self._form_revision}
    self.state.model_dirty = bool(model_form.edits(m.document, m.draft))
    _show_checks(self, m.document, m.draft)


def _plain(draft: dict) -> dict:
    return {k: v for k, v in draft.items() if k != "revision"}


def discard_model_edits(self: Editor) -> None:
    """Draw the model as it is, dropping the draft's edits."""
    if self._model.document is not None:
        self._model.draft = model_form.draft_of(self._model.document)
    _show_draft(self)


def model_edited(self: Editor, draft: dict) -> None:
    """The page's draft after an edit (sent when a field loses focus): check it
    in the background."""
    self._model.draft = _plain(draft)
    self.schedule_checks()


def add_model_parameter(self: Editor, draft: dict, name: str) -> None:
    """Add a parameter row named ``name`` (an unknown name a rate uses) to the
    page's draft, keeping what was typed."""
    self._model.draft = model_form.add_parameter(_plain(draft), name)
    _show_draft(self)
    self.schedule_checks()


# -- the background checks ---------------------------------------------------


def _expression_key(document: dict, draft: dict, request: dict) -> str:
    """What a ``check_expression`` answer depends on: the model (and its file's
    content), the expression and the parameters the draft adds."""
    return json.dumps(
        [document["name"], document["sha256"], request, model_form.added(draft)], sort_keys=True
    )


def _ask(request) -> dict:
    """The answer of one check in the worker, ``request()`` (safe off the page's
    thread); a refused request becomes the error it reports."""
    try:
        return request()
    except WorkerError as exc:
        return {"ok": False, "error": str(exc), "position": None, "errors": []}


async def run_model_checks(self: Editor, loop: asyncio.AbstractEventLoop) -> None:
    """Check the panel's edits: each edited expression, then, when every one
    reads, the whole model. Called by ``_run_checks``; the draft is read here,
    on the page's thread, and only the worker's answers are awaited."""
    m = self._model
    if m.document is None or m.draft is None:
        return
    document, draft = m.document, m.draft
    model, path = document["name"], document["path"]
    edits = model_form.edits(document, draft)
    added = model_form.added(draft)
    requests = model_form.expressions(edits)
    for request in requests:
        key = _expression_key(document, draft, request)
        if key not in m.expressions:
            fields = {k: v for k, v in request.items() if k != "field"}
            m.expressions[key] = await loop.run_in_executor(
                None,
                _ask,
                lambda f=fields: self.worker.request(
                    "check_expression", model=model, path=path, parameters=added, **f
                ),
            )
    if self._model is not m:
        return  # another model was opened meanwhile
    key = json.dumps(edits, sort_keys=True)
    readable = all(m.expressions[_expression_key(document, draft, r)]["ok"] for r in requests)
    if edits and readable and m.checked[0] != key:
        answer = await loop.run_in_executor(
            None,
            _ask,
            lambda: self.worker.request("check_model_edits", model=model, path=path, edits=edits),
        )
        m.checked = (key, answer)
    with self.state:
        _show_checks(self, document, draft)


def _show_checks(self: Editor, document: dict, draft: dict) -> None:
    """Show the latest answers for the draft as it stands: errors under their
    fields (with a caret at the position), what each edited expression uses,
    the unknown parameters a rate names (offered as new rows), and whether the
    edits make a valid model."""
    m = self._model
    s = self.state
    errors, pointers, uses, offers = [], {}, {}, []
    edits = model_form.edits(document, draft)
    have = set(model_form.added(draft))
    keys = model_form.page_keys(draft)
    for request in model_form.expressions(edits):
        answer = m.expressions.get(_expression_key(document, draft, request))
        if answer is None:
            continue
        key = keys[request["field"]]
        if answer.get("references"):
            uses[key] = model_form.uses(answer["references"])
        if not answer["ok"]:
            errors.append({"field": request["field"], "error": answer["error"]})
            pointers[key] = model_form.pointer(request["expression"], answer["position"])
        unknown = (answer.get("unknown") or {}).get("parameters", [])
        offers += [n for n in unknown if n not in have and n not in offers]
    whole = m.checked[1] if m.checked[0] == json.dumps(edits, sort_keys=True) else None
    if whole is not None and not whole["ok"]:
        errors += whole["errors"] or [{"field": "model", "error": whole["error"]}]
    shown, general = model_form.field_errors(errors, draft)
    s.model_errors = shown
    s.model_pointers = pointers
    s.model_uses = uses
    s.model_offers = offers
    s.model_general = general
    s.model_valid = bool(edits) and whole is not None and whole["ok"]


# -- Apply -------------------------------------------------------------------


def apply_model(self: Editor, draft: dict, pending: list[dict] | None = None) -> None:
    """Apply the panel's edits: check the whole edited model, then save it with
    the flowsheet. A shipped model asks for the new model's name first, and a
    project model asks first when rewriting its file loses something (its
    comments)."""
    m = self._model
    if m.document is None or not self._apply_pending(pending):
        return
    m.draft = draft = _plain(draft)
    edits = model_form.edits(m.document, draft)
    if not edits:
        self.say("the model has no edits to apply")
        return
    path = self._spec_file()
    if path is None:
        self.say(
            "type where to keep the flowsheet in the path field, then Apply: an edited "
            "model is saved in the flowsheet's project"
        )
        return
    refusal = self._save_refusal(path)
    if refusal:
        self.say(refusal)
        return
    document = m.document
    shipped = document["source"] == "shipped"
    request = {
        "spec": self.doc.to_spec(),
        "path": str(path),
        "base_dir": str(self.doc.base_dir or path.parent),
        "model": document["name"],
        "edits": edits,
        "sha256": document["sha256"],
    }
    if shipped:
        request["name"] = model_form.proposed_name(
            document["name"], self.doc.meta.get("name", "plant")
        )

    def work():
        checked = self.worker.request(
            "check_model_edits", model=document["name"], path=document["path"], edits=edits
        )
        if not checked["ok"]:
            return {"checked": checked}
        planned = _save(self, request, dry_run=True)
        if shipped or isinstance(planned, str) or planned["warning"]:
            return {"checked": checked, "planned": planned}
        return {"checked": checked, "saved": _save(self, request, dry_run=False)}

    def done(result):
        if isinstance(result, Exception):
            self.say(f"could not save model {document['name']}: {result}")
            return
        m.checked = (json.dumps(edits, sort_keys=True), result["checked"])
        _show_checks(self, document, draft)
        if not result["checked"]["ok"]:
            self.say("the edits do not make a valid model; the panel shows why")
        elif "saved" in result:
            _saved(self, path, request["spec"], result["saved"])
        elif shipped or not isinstance(result["planned"], str):
            m.plan = request
            self.state.model_dialog_new = shipped
            _show_plan(self, request.get("name", ""), result["planned"])
            self.state.model_dialog_open = True
        else:
            _saved(self, path, request["spec"], result["planned"])

    self._off_page(work, done)


def _save(self: Editor, request: dict, dry_run: bool) -> dict | str:
    """``save_model_edits``'s answer, or its refusal as the message to show."""
    try:
        return self.worker.request("save_model_edits", **request, dry_run=dry_run)
    except WorkerError as exc:
        return str(exc)


def _show_plan(self: Editor, name: str, planned: dict | str) -> None:
    """The dialog's name, the units that will use the saved model, and what
    saving loses (or why it cannot be saved under that name)."""
    self.state.model_new_name = name
    if isinstance(planned, str):
        self.state.model_dialog_error = planned
        self.state.model_dialog_units = []
        self.state.model_dialog_warning = ""
    else:
        self.state.model_dialog_error = ""
        self.state.model_dialog_units = planned["units"]
        self.state.model_dialog_warning = planned["warning"] or ""


def plan_model_name(self: Editor, name: str) -> None:
    """The dialog's name changed: say again what saving under it would do."""
    plan = self._model.plan
    if plan is None:
        return
    request = {**plan, "name": str(name).strip()}
    self._off_page(
        lambda: _save(self, request, dry_run=True),
        lambda planned: _show_plan(self, request["name"], planned),
    )


def answer_model_dialog(self: Editor, choice: str, name: str = "") -> None:
    """The save dialog's answer: ``"save"`` (a model made from a shipped one
    under ``name``), or ``"cancel"``."""
    plan, self._model.plan = self._model.plan, None
    if choice != "save" or plan is None:
        self.state.model_dialog_open = False
        return
    request = {**plan, "name": str(name).strip()} if "name" in plan else plan

    def done(saved):
        if isinstance(saved, str):
            self._model.plan = plan  # the dialog stays, with the refusal
            _show_plan(self, request["name"], saved)
            return
        self.state.model_dialog_open = False
        _saved(self, Path(request["path"]), request["spec"], saved)

    self._off_page(lambda: _save(self, request, dry_run=False), done)


def _saved(self: Editor, path: Path, spec: dict, saved: dict | str) -> None:
    """Show what a save wrote: the flowsheet as written (its units naming the
    saved model), every check run again against the new model, and the panel
    on that model."""
    if isinstance(saved, str):
        self.say(f"could not save the model: {saved}")
        return
    changed = json.dumps(self.doc.to_spec(), sort_keys=True) != json.dumps(spec, sort_keys=True)
    self.doc = Flowsheet.from_spec(self.catalog, saved["spec"], base_dir=path.parent)
    self._opened = (path, _digest(path))
    self._saved = self._snapshot()
    self.state.spec_path = str(path)
    self._model_info.clear()
    self._unit_checks.clear()
    self._plant_check = (None, None)
    self.publish()
    units = ", ".join(saved["units"]) or "no unit"
    self.say(
        f"saved model {saved['model']} ({saved['model_file']}), used by {units}"
        + ("; edits made to the flowsheet while it saved were not kept" if changed else ""),
        error=changed,
    )
    self.open_model(saved["model"])
