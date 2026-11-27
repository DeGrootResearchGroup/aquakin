"""``aquakin mcp`` driven by a real MCP client: the tools it lists, their answers
compared with the aquakin functions they wrap, and errors that reach the
assistant with aquakin's own message."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed")

from mcp import Client, StdioServerParameters

import aquakin
from aquakin.mcp import server as mcp_server
from aquakin.plant.catalog import catalog, shipped_models
from aquakin.plant.spec import check_spec, load_spec_file
from tests.conftest import EXAMPLES, FIXTURES
from tests.integration.mcp_client import call as _call
from tests.integration.mcp_client import session as _session
from tests.integration.mcp_client import value as _value

BSM1 = EXAMPLES / "projects" / "bsm1" / "flowsheet.yaml"


#: What each effect tells the user's app: (read only, may destroy, idempotent).
_HINTS = {
    "read": (True, False, True),
    "write": (False, False, False),
    "overwrite": (False, True, False),
}


def test_every_tool_is_listed_with_its_whole_docstring_and_its_effect():
    async def calls(client):
        return (await client.list_tools()).tools

    tools = _session(calls)
    assert [t.name for t in tools] == [fn.__name__ for fn, _ in mcp_server.TOOLS]
    for tool, (fn, effect) in zip(tools, mcp_server.TOOLS, strict=True):
        assert tool.description == mcp_server.inspect.getdoc(fn)
        a = tool.annotations
        assert (a.read_only_hint, a.destructive_hint, a.idempotent_hint) == _HINTS[effect]
        assert a.open_world_hint is False


def test_tools_that_can_replace_or_delete_are_marked_destructive():
    effects = {fn.__name__: effect for fn, effect in mcp_server.TOOLS}
    assert {name for name, e in effects.items() if e == "overwrite"} == {
        "write_flowsheet",
        "update_study",
        "run_study",
        "unfreeze_study",
    }
    assert effects["check_flowsheet"] == "read" and effects["create_study"] == "write"


def test_the_server_names_itself_and_says_how_to_use_its_tools():
    built = mcp_server.build_server()
    assert built.name == "aquakin"
    assert "read the\nrelevant guide page" in built.instructions


def test_list_models_gives_each_shipped_model_with_its_first_sentence():
    listed = _value("list_models")
    shipped = shipped_models()
    assert [m["name"] for m in listed] == [m["name"] for m in shipped]
    asm1 = next(m for m in listed if m["name"] == "asm1")
    # "No. 1" does not end the sentence; the full stop before a capital does.
    assert asm1["summary"].startswith("Activated Sludge Model No. 1 (ASM1), the IAWQ")
    assert asm1["summary"].endswith("ammonification.")


@pytest.mark.parametrize(
    "text, first",
    [
        ("Model No. 1 here. Next one.", "Model No. 1 here."),
        ("After Smith et al. 2002. More.", "After Smith et al. 2002."),
        ("Uses e.g. ozone here. More.", "Uses e.g. ozone here."),
        ("A single sentence", "A single sentence"),
    ],
)
def test_a_summary_is_the_first_sentence(text, first):
    assert mcp_server._summary(text) == first


def test_describe_model_is_the_models_summary_for_a_name_or_a_file():
    assert _call("describe_model", name="asm1") == (False, aquakin.load_model("asm1").summary())
    is_error, text = _call("describe_model", path=str(FIXTURES / "simple_model.yaml"))
    assert not is_error
    assert text.startswith("Model: simple_decay")


@pytest.mark.parametrize("tool", ["describe_model", "read_model_file"])
@pytest.mark.parametrize("arguments", [{}, {"name": "asm1", "path": "x.yaml"}])
def test_a_model_is_named_or_given_by_path_not_both(tool, arguments):
    assert _call(tool, **arguments) == (
        True,
        f"Error executing tool {tool}: ValueError: give either the name of a shipped "
        "model or the path of a model file",
    )


def test_read_model_file_returns_the_yaml_source():
    shipped = Path(aquakin.__file__).parent / "models" / "asm1.yaml"
    assert _call("read_model_file", name="asm1") == (False, shipped.read_text(encoding="utf-8"))
    fixture = FIXTURES / "simple_model.yaml"
    assert _call("read_model_file", path=str(fixture)) == (False, fixture.read_text())


def test_an_unknown_model_name_points_at_list_models():
    is_error, text = _call("read_model_file", name="asm99")
    assert is_error
    assert text.endswith("KeyError: no shipped model 'asm99'; list_models lists them")


def test_unit_kinds_are_the_catalogues():
    cat = catalog()
    listed = _value("list_unit_kinds")
    assert listed["categories"] == cat["categories"]
    assert [k["kind"] for k in listed["kinds"]] == [k["kind"] for k in cat["kinds"]]
    assert set(listed["kinds"][0]) == {"kind", "label", "category", "summary"}
    cstr = next(k for k in cat["kinds"] if k["kind"] == "cstr")
    assert _value("describe_unit_kind", kind="cstr") == {
        **cstr,
        "setting_types": cat["setting_types"],
    }


def test_an_unknown_unit_kind_lists_the_kinds():
    is_error, text = _call("describe_unit_kind", kind="lagoon")
    assert is_error
    assert "KeyError: no unit kind 'lagoon'; the kinds are ['influent', 'cstr'," in text


def test_check_unit_reports_ports_or_the_constructors_reason():
    settings = {"model": "asm1", "input_port_names": ["a", "b", "c"]}
    assert _value("check_unit", kind="mixer", settings=settings) == {
        "ok": True,
        "inputs": ["a", "b", "c"],
        "outputs": ["out"],
    }
    assert _value("check_unit", kind="cstr", settings={"volume": 1000.0}) == {
        "ok": False,
        "error": "'model' is required",
        "type": "ValueError",
    }


def test_a_flowsheet_is_read_and_checked_as_the_editor_does():
    assert _value("read_flowsheet", path=str(BSM1)) == load_spec_file(BSM1)
    expected = check_spec(BSM1)
    assert _value("check_flowsheet", path=str(BSM1)) == expected
    spec = load_spec_file(BSM1)
    assert _value("check_flowsheet", spec=spec, base_dir=str(BSM1.parent)) == expected


@pytest.mark.parametrize("arguments", [{}, {"path": "a.yaml", "spec": {}}])
def test_a_flowsheet_is_a_path_or_a_spec_not_both(arguments):
    is_error, text = _call("check_flowsheet", **arguments)
    assert is_error
    assert text.endswith("give either the path of a flowsheet file or a flowsheet spec")


def test_the_guide_and_the_api_reference_are_served():
    from aquakin.mcp import api, docs

    assert _value("list_doc_pages") == docs.list_pages()
    assert _call("read_doc_page", page="calibration") == (False, docs.read_page("calibration"))
    hits = _value("search_docs", query="Fisher information", limit=3)
    assert hits == docs.search("Fisher information", 3)
    assert _value("list_api", module="aquakin.study") == api.list_api("aquakin.study")
    assert _value("describe_api", name="Plant.calibrate") == api.describe_api("Plant.calibrate")


def test_without_the_mcp_extra_the_error_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp.server", None)  # as if not installed
    with pytest.raises(ImportError, match=r'needs the "mcp" extra: pip install "aquakin\[mcp\]"'):
        mcp_server.build_server()


def test_the_mcp_command_runs_the_server_over_stdio(monkeypatch):
    from mcp.server import MCPServer

    from aquakin.__main__ import main

    ran = []
    monkeypatch.setattr(MCPServer, "run", lambda self, transport="stdio": ran.append(transport))
    assert main(["mcp"]) == 0
    assert ran == ["stdio"]


def test_the_aquakin_mcp_command_answers_over_standard_input_and_output():
    """The real entry point, as an assistant's app starts it: a subprocess
    speaking MCP on its standard streams."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "aquakin", "mcp"])

    async def run():
        async with Client(params) as client:
            tools = (await client.list_tools()).tools
            result = await client.call_tool("describe_unit_kind", {"kind": "mixer"})
            return [t.name for t in tools], result

    names, result = asyncio.run(run())
    assert names == [fn.__name__ for fn, _ in mcp_server.TOOLS]
    assert not result.is_error
    assert json.loads(result.content[0].text)["kind"] == "mixer"
