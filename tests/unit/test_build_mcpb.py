"""``scripts/build_mcpb.py``: the Claude Desktop extension of a release, staged
from ``packaging/claude-desktop/`` with that release's version, aquakin pin and
the server's own tool list."""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_mcpb.py"


@pytest.fixture(scope="module")
def build():
    spec = importlib.util.spec_from_file_location("build_mcpb", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_release_is_stamped_into_the_manifest_and_the_environment(build, tmp_path):
    out = build.stage("1.2.3", tmp_path / "mcpb")
    manifest = json.loads((out / "manifest.json").read_text())
    project = tomllib.loads((out / "pyproject.toml").read_text())
    assert manifest["version"] == project["project"]["version"] == "1.2.3"
    assert project["project"]["dependencies"] == ["aquakin[mcp]==1.2.3"]
    assert project["project"]["description"] == "aquakin's MCP server for Claude Desktop"
    assert (out / "src" / "server.py").read_text() == (
        ROOT / "packaging" / "claude-desktop" / "src" / "server.py"
    ).read_text()
    # The template itself is left as it is.
    template = json.loads((ROOT / "packaging" / "claude-desktop" / "manifest.json").read_text())
    assert template["version"] == "0.0.0" and template["tools"] == []


def test_the_manifest_lists_every_tool_the_server_can_offer(build, tmp_path):
    import inspect

    from aquakin.mcp.scripts import run_python
    from aquakin.mcp.server import TOOLS

    out = build.stage("1.2.3", tmp_path / "mcpb")
    listed = json.loads((out / "manifest.json").read_text())["tools"]
    expected = [fn for fn, _ in TOOLS] + [run_python]
    assert [t["name"] for t in listed] == [fn.__name__ for fn in expected]
    first = inspect.getdoc(expected[0]).split("\n\n")[0]
    assert listed[0]["description"] == " ".join(first.split())


def test_the_setting_turns_on_the_script_tool(build):
    manifest = json.loads((ROOT / "packaging" / "claude-desktop" / "manifest.json").read_text())
    setting = manifest["user_config"]["allow_python"]
    assert setting["type"] == "boolean" and setting["default"] is False
    assert manifest["server"]["mcp_config"]["args"][-1] == "${user_config.allow_python}"
    entry = (ROOT / "packaging" / "claude-desktop" / "src" / "server.py").read_text()
    assert '["--allow-python"] if sys.argv[1:] == ["true"]' in entry


def test_another_requirement_can_be_tried_before_a_release(build, tmp_path):
    out = build.stage("1.2.3", tmp_path / "mcpb", requirement="aquakin[mcp] @ file:///src")
    project = tomllib.loads((out / "pyproject.toml").read_text())
    assert project["project"]["dependencies"] == ["aquakin[mcp] @ file:///src"]


def test_a_bad_version_or_an_existing_folder_is_refused(build, tmp_path):
    with pytest.raises(ValueError, match=r"'v1\.2' is not a release version"):
        build.stage("v1.2", tmp_path / "a")
    (tmp_path / "b").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build.stage("1.2.3", tmp_path / "b")


def test_the_command_line_stages_a_folder(build, tmp_path, capsys):
    assert build.main(["1.2.3", str(tmp_path / "c")]) == 0
    assert "staged the Claude Desktop extension 1.2.3" in capsys.readouterr().out
    assert (tmp_path / "c" / "manifest.json").is_file()
