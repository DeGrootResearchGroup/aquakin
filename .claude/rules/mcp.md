---
paths:
  - "aquakin/mcp/**"
  - "setup.py"
  - "MANIFEST.in"
---

# Rules — `aquakin/mcp/` (`aquakin mcp`)

The MCP server an AI assistant starts on the user's computer. The user guide
page is `docs/ai_assistants.md`; the editor shows the assistant's changes by
following the project folder (`aquakin_ui/disk.py`, `.claude/rules/ui.md`).

## A tool is a thin wrapper

- **Each tool calls the function the editor's `aquakin serve` worker calls**
  (`aquakin.plant.catalog`, `aquakin.plant.spec`, `aquakin.study`), never a
  copy of its logic, so the assistant and the editor cannot disagree about a
  flowsheet. If a tool needs behaviour the worker lacks, add it to the shared
  function, not to `server.py`.
- **A tool is a plain function in `server.py` listed in `TOOLS`**, with a NumPy
  docstring: the whole docstring is the description the assistant reads, so its
  Parameters section is how the model learns each argument.
- **Every tool is registered through `_reported`**. The SDK replaces any
  exception other than its `ToolError` with a bare "Error executing tool", so
  without it the assistant never sees aquakin's message.
- **Return plain data** (dicts, lists, strings of JSON-able values). The SDK
  serializes it; a JAX array or a `Plant` does not cross the protocol.
- **Keep tools small in what they return.** A whole catalogue or every model's
  full description in one answer costs the user's context window: list names
  and summaries, and describe one item on request.

## Reading versus writing

- **Every tool declares its effect in `TOOLS`** (`server.py` for the readers,
  `projects.py` for the rest): `"read"`, `"write"` (adds files or records) or
  `"overwrite"` (may replace or delete, given its `replace` /
  `discard_results`). The effect sets the MCP hints the user's app shows and
  asks on; a tool that can replace anything is `"overwrite"` even when the
  default call cannot.
- **Keep aquakin's refusals.** Writing goes through the functions the editor
  uses (`write_spec`, `aquakin.study`) and never past their guards: results are
  replaced only with `replace`, a frozen study with results unfrozen only with
  `discard_results`, and an existing flowsheet file written over only with
  `replace` (the editor's Save refuses it outright).
- **A study run is a detached `aquakin study run` process** (`runs.py`,
  `start_run`), never a thread of the server: it must outlive the assistant's
  app. Follow and cancel it through the study folder (`aquakin.study.running`:
  `check_study`'s `running`, `cancel_study`), never through the process, so
  the editor and the command line see the same run. `run_study` returns once
  the run holds its study (its own pid in `running.yaml`); the assistant polls
  `study_status`.

## Imports and packaging

- **Only `server.py` imports `mcp`, and only inside `build_server`**, so
  `aquakin.mcp.docs` and `aquakin.mcp.api` work and are tested without the
  `mcp` extra.
- **`docs.py` imports only the standard library**: `setup.py` loads it by path
  while building, before aquakin is importable, to bundle the published pages
  into the wheel (the `BUNDLE` folder beside `docs.py`). Which pages are
  published is decided only by `docs/conf.py`'s `exclude_patterns`; do not
  keep a second list.
- **A new published page needs nothing here**; a new docs *file type* the
  guide tools should serve needs `MANIFEST.in` (the sdist) and `setup.py`
  (the wheel) together, because `python -m build` builds the wheel from the
  sdist.
