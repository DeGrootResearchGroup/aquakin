"""The ``aquakin mcp`` server: aquakin's tools for an AI assistant.

Each tool is a plain function, calling the same aquakin function the ``aquakin
serve`` worker calls for the editor (the unit catalogue,
:func:`~aquakin.plant.spec.check_spec`, :mod:`aquakin.study`, ...), so the
assistant and the editor see one implementation. The tools that only read are
here; those that write a flowsheet or create, edit and run studies are in
:mod:`aquakin.mcp.projects`. :func:`build_server` registers them all on an MCP
server; only it needs the ``mcp`` package.
"""

from __future__ import annotations

import functools
import inspect
import re
from importlib.resources import files

from aquakin.mcp import api, docs, projects

#: What the assistant is told about the server when it connects.
INSTRUCTIONS = """\
aquakin models reactive transport in water and wastewater treatment: kinetic
models written in YAML (activated sludge, anaerobic digestion, ozonation, ...),
reactors and whole-plant flowsheets built from them, and analyses on those:
sensitivity, calibration, Fisher information, identifiability and experiment
design. Every solve is differentiable (JAX).

Before writing a model file, a flowsheet or Python that uses aquakin, read the
relevant guide page (list_doc_pages, search_docs, read_doc_page) and look up
the functions you use (list_api, describe_api) rather than guessing names or
arguments. Check a flowsheet with check_flowsheet, and a single unit's
settings with check_unit, before relying on it. Model and unit names come from
list_models and list_unit_kinds.

A project is a folder holding a flowsheet file and its studies (studies/<name>/).
write_flowsheet writes a flowsheet; a study is one analysis of it (simulate,
fisher, calibrate, profile, rank: study_settings lists each kind's settings,
study_choices what a study may name). run_study starts a run in the background
and returns at once: poll study_status until it is no longer running, then read
study_results. A user may have the project open in the aquakin editor, which
shows each change as it is written. Never replace a file or results
(replace=true, discard_results=true) unless the user asked for it.
"""


#: The end of a description's first sentence: a full stop before a capital
#: letter, so "Model No. 1" or "et al. 2002" does not end it.
_SENTENCE = re.compile(r"^(.*?\.)\s+(?=[A-Z])")


def _summary(text: str) -> str:
    """The first sentence of a description (all of it when there is one sentence)."""
    m = _SENTENCE.match(text)
    return m.group(1) if m else text


def list_models() -> list[dict]:
    """List the kinetic models shipped with aquakin, each with a one-sentence summary.

    Returns
    -------
    list of dict
        ``{"name", "summary"}`` per model, sorted by name.
    """
    from aquakin.plant.catalog import shipped_models

    return [{"name": m["name"], "summary": _summary(m["description"])} for m in shipped_models()]


def _model(name: str | None, path: str | None):
    from aquakin.plant.catalog import load_model_file
    from aquakin.schema.loader import load_model

    if (name is None) == (path is None):
        raise ValueError("give either the name of a shipped model or the path of a model file")
    return load_model(name) if path is None else load_model_file(path)


def describe_model(name: str | None = None, path: str | None = None) -> str:
    """Describe a kinetic model: species, conditions, reactions with their stoichiometry,
    and parameters with their values, bounds and temperature corrections.

    Parameters
    ----------
    name : str, optional
        A shipped model's name (see ``list_models``).
    path : str, optional
        The path of a model YAML file instead (a project's own model).

    Returns
    -------
    str
        The model's summary table.
    """
    return _model(name, path).summary()


def read_model_file(name: str | None = None, path: str | None = None) -> str:
    """Read a kinetic model's YAML source, the format a new model is written in.

    Parameters
    ----------
    name : str, optional
        A shipped model's name (see ``list_models``).
    path : str, optional
        The path of a model YAML file instead.

    Returns
    -------
    str
        The file's text.
    """
    from pathlib import Path

    if (name is None) == (path is None):
        raise ValueError("give either the name of a shipped model or the path of a model file")
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    resource = files("aquakin.models") / f"{name}.yaml"
    if not resource.is_file():
        raise KeyError(f"no shipped model {name!r}; list_models lists them")
    return resource.read_text(encoding="utf-8")


def list_unit_kinds() -> dict:
    """List the kinds of unit a plant flowsheet can contain (reactors, clarifiers,
    splitters, controllers, ...), each with its category and summary.

    Returns
    -------
    dict
        ``{"categories": [...], "kinds": [{"kind", "label", "category",
        "summary"}, ...]}``.
    """
    from aquakin.plant.catalog import catalog

    cat = catalog()
    keep = ("kind", "label", "category", "summary")
    return {
        "categories": cat["categories"],
        "kinds": [{k: entry[k] for k in keep} for entry in cat["kinds"]],
    }


def describe_unit_kind(kind: str) -> dict:
    """Describe one kind of unit: every setting (type, default, help), its ports,
    and the settings of which exactly one must be given.

    Parameters
    ----------
    kind : str
        A unit kind from ``list_unit_kinds`` (``"cstr"``, ``"clarifier"``, ...).

    Returns
    -------
    dict
        The kind's catalogue entry, with ``setting_types`` explaining each
        setting type.
    """
    from aquakin.plant.catalog import catalog

    cat = catalog()
    for entry in cat["kinds"]:
        if entry["kind"] == kind:
            return {**entry, "setting_types": cat["setting_types"]}
    known = [e["kind"] for e in cat["kinds"]]
    raise KeyError(f"no unit kind {kind!r}; the kinds are {known}")


def check_unit(
    kind: str,
    settings: dict,
    name: str = "unit",
    model_files: dict[str, str] | None = None,
    base_dir: str | None = None,
) -> dict:
    """Build one unit from its settings and report its ports, or why it cannot be built.

    Parameters
    ----------
    kind : str
        A unit kind from ``list_unit_kinds``.
    settings : dict
        The unit's settings, as a flowsheet file holds them.
    name : str, optional
        The unit's name.
    model_files : dict, optional
        ``{model name: YAML path}`` for project models the settings name.
    base_dir : str, optional
        The folder a relative file path in the settings is relative to.

    Returns
    -------
    dict
        ``{"ok": true, "inputs", "outputs"}``, or ``{"ok": false, "error",
        "type"}``.
    """
    from aquakin.plant.catalog import check_unit as check

    return check(kind, name, settings, model_files, base_dir)


def read_flowsheet(path: str) -> dict:
    """Read a plant flowsheet file (YAML or JSON) as data.

    Parameters
    ----------
    path : str
        The flowsheet file.

    Returns
    -------
    dict
        The flowsheet spec: its units, wiring and settings.
    """
    from aquakin.plant.spec import load_spec_file

    return load_spec_file(path)


def check_flowsheet(
    path: str | None = None, spec: dict | None = None, base_dir: str | None = None
) -> dict:
    """Build a plant from a flowsheet and report every problem found in it
    (unwired ports, units that cannot be built, ...).

    Parameters
    ----------
    path : str, optional
        A flowsheet file.
    spec : dict, optional
        A flowsheet spec as data instead (as ``read_flowsheet`` returns it).
    base_dir : str, optional
        The folder relative file paths in ``spec`` are relative to (a file's
        own folder when ``path`` is given).

    Returns
    -------
    dict
        The plant's check report.
    """
    from aquakin.plant.spec import check_spec

    if (path is None) == (spec is None):
        raise ValueError("give either the path of a flowsheet file or a flowsheet spec")
    return check_spec(path if spec is None else spec, base_dir=base_dir)


def list_doc_pages() -> list[dict]:
    """List the pages of the aquakin user guide.

    Returns
    -------
    list of dict
        ``{"page", "title"}`` per page.
    """
    return docs.list_pages()


def read_doc_page(page: str, section: str | None = None) -> str:
    """Read one page of the aquakin user guide (Markdown), or one section of it.

    Parameters
    ----------
    page : str
        A page name from ``list_doc_pages`` or ``search_docs``.
    section : str, optional
        One heading of the page; only that section is returned.

    Returns
    -------
    str
    """
    return docs.read_page(page, section)


def search_docs(query: str, limit: int = 10) -> list[dict]:
    """Search the aquakin user guide for sections matching some words.

    Parameters
    ----------
    query : str
        One or more words.
    limit : int, optional
        The most sections to return.

    Returns
    -------
    list of dict
        ``{"page", "section", "score", "snippet"}`` per match, best first.
    """
    return docs.search(query, limit)


def list_api(module: str = "aquakin") -> list[dict]:
    """List the public names of an aquakin API module with their one-line summaries.

    Parameters
    ----------
    module : str, optional
        ``"aquakin"`` (the default), ``"aquakin.plant"`` or ``"aquakin.study"``.

    Returns
    -------
    list of dict
        ``{"name", "kind", "summary"}`` per name.
    """
    return api.list_api(module)


def describe_api(name: str) -> dict:
    """Give the signature and full docstring of an aquakin function, class or method.

    Parameters
    ----------
    name : str
        A name from ``list_api`` (``"calibrate"``), a member of a class
        (``"Plant.calibrate"``) or a full dotted name
        (``"aquakin.plant.bsm.build_bsm1"``).

    Returns
    -------
    dict
        ``name``, ``kind``, ``signature``, ``doc``, and ``members`` for a class.
    """
    return api.describe_api(name)


#: Every tool, in the order an assistant is shown them, with what it may do to
#: the user's files (see :data:`aquakin.mcp.projects.TOOLS`).
TOOLS = (
    *(
        (fn, "read")
        for fn in (
            list_models,
            describe_model,
            read_model_file,
            list_unit_kinds,
            describe_unit_kind,
            check_unit,
            read_flowsheet,
            check_flowsheet,
            list_doc_pages,
            search_docs,
            read_doc_page,
            list_api,
            describe_api,
        )
    ),
    *projects.TOOLS,
)


def _annotations(effect: str) -> dict:
    """The MCP hints for a tool's effect, so the user's app knows which calls
    only read and which may replace or delete what is there."""
    return {
        "read": dict(read_only_hint=True, destructive_hint=False, idempotent_hint=True),
        "write": dict(read_only_hint=False, destructive_hint=False, idempotent_hint=False),
        "overwrite": dict(read_only_hint=False, destructive_hint=True, idempotent_hint=False),
    }[effect]


def _reported(fn, tool_error):
    """``fn``, with any exception it raises turned into ``tool_error(message)``.

    The MCP SDK answers an exception other than its ``ToolError`` with only
    "Error executing tool", so without this the assistant would never see why
    a call failed (a wrong model name, a flowsheet that does not build).
    """
    from aquakin.plant.errors import error_reply

    @functools.wraps(fn)
    def tool(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # the protocol boundary: every failure is an answer
            reply = error_reply(exc)
            raise tool_error(f"{reply['type']}: {reply['error']}") from exc

    return tool


def build_server():
    """The MCP server with every tool of :data:`TOOLS` registered.

    Returns
    -------
    mcp.server.MCPServer
        Not yet running; :func:`main` runs it over standard input and output.

    Raises
    ------
    ImportError
        If the ``mcp`` package is not installed (the ``mcp`` extra).
    """
    try:
        from mcp.server import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp_types import ToolAnnotations
    except ImportError as exc:
        raise ImportError('aquakin mcp needs the "mcp" extra: pip install "aquakin[mcp]"') from exc
    from aquakin import __version__

    server = MCPServer(name="aquakin", version=__version__, instructions=INSTRUCTIONS)
    for fn, effect in TOOLS:
        # The whole docstring: its Parameters section is how the assistant
        # learns what each argument means.
        server.add_tool(
            _reported(fn, ToolError),
            description=inspect.getdoc(fn),
            annotations=ToolAnnotations(open_world_hint=False, **_annotations(effect)),
        )
    return server


def main() -> None:
    """Run the server over standard input and output until the client closes it."""
    build_server().run("stdio")
