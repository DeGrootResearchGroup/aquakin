"""aquakin's checks of the flowsheet, run in the background.

Each unit is built from its settings, then the whole plant from the spec, by
the worker; each answer is cached by the input it checked. The document is read
on the page's thread and only the worker call runs in an executor, so an edit
made during a check cannot be half-read. Each function is bound onto
:class:`~aquakin_ui.app.Editor` as a method (``self`` is the editor).
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from trame.app import asynchronous

from aquakin_ui.worker import WorkerError

if TYPE_CHECKING:
    from aquakin_ui.app import Editor


CHECK_DELAY = 0.3  # seconds of quiet before aquakin's checks run


def model_info(self: Editor, name: str) -> dict | None:
    """The worker's description of a model (cached), or ``None``."""
    path = self.doc.model_files().get(name)
    key = (name, path)
    if key not in self._model_info:
        try:
            self._model_info[key] = self.worker.request("model", name=name, path=path)
        except WorkerError:
            self._model_info[key] = None
    return self._model_info[key]


def _unit_check(self: Editor, uid: str):
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


def _ask_unit(self: Editor, request, ports) -> str | None:
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


def _check_unit(self: Editor, uid: str) -> str | None:
    key, pending = self._unit_check(uid)
    if pending is not None:
        self._unit_checks[uid] = (key, self._ask_unit(*pending))
    return self._unit_checks[uid][1]


def _plant_request(self: Editor):
    """``(key, spec, base_dir)`` for checking the whole plant, read now."""
    spec = self.doc.to_spec(layout=False)
    base = str(self.doc.base_dir) if self.doc.base_dir else None
    return json.dumps([spec, base], sort_keys=True), spec, base


def _ask_plant(self: Editor, spec, base) -> dict:
    try:
        return self.worker.request("check_spec", spec=spec, base_dir=base)
    except WorkerError as exc:
        return {"ok": False, "error": str(exc)}


def _check_plant(self: Editor) -> dict | None:
    key, spec, base = self._plant_request()
    if self._plant_check[0] != key:
        self._plant_check = (key, self._ask_plant(spec, base))
    return self._plant_check[1]


def unit_errors(self: Editor) -> dict[str, str]:
    """The latest aquakin error of each unit that failed to build."""
    return {uid: e for uid, (_, e) in self._unit_checks.items() if e and uid in self.doc.units}


def schedule_checks(self: Editor) -> None:
    """Run aquakin's checks after a short quiet spell, off the page's thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return  # no page served yet: the checks run once the server is ready
    if self._check_task is not None:
        self._check_task.cancel()
    # Pushed now: trame sends a value only when it differs from the one it
    # last pushed. Left for the next flush, a `checking` set at start-up
    # reaches a page only in its first state, the push still holds `False`,
    # and the `False` that ends the round would never be sent.
    with self.state:
        self.state.checking = True
    self._check_task = asynchronous.create_task(self._run_checks())


async def _run_checks(self: Editor) -> None:
    await asyncio.sleep(CHECK_DELAY)
    loop = asyncio.get_running_loop()
    await self.run_model_checks(loop)  # the model panel's edits, which are quick
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
        # The redraw leaves the form's unapplied edits marked as such: the rows
        # stay the page's own until Apply, and a cleared mark would let the next
        # selection drop them.
        dirty = self.state.sel_dirty
        self.publish(checks=False)
        self.state.sel_dirty = dirty
