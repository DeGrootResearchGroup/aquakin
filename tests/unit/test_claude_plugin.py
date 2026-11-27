"""The Claude Code plugin and its marketplace (``.claude-plugin/``,
``plugins/aquakin/``): well formed, and pinned to this version of aquakin, so a
release cannot ship a plugin that starts a different aquakin."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

import aquakin

ROOT = Path(__file__).resolve().parents[2]
MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
PLUGIN = ROOT / "plugins" / "aquakin"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_marketplace_lists_the_plugin_by_its_own_name():
    market = _json(MARKETPLACE)
    (entry,) = market["plugins"]
    assert (ROOT / entry["source"]).resolve() == PLUGIN
    assert entry["name"] == _json(PLUGIN / ".claude-plugin" / "plugin.json")["name"] == "aquakin"
    # A plugin.json at the repository root would make the repository itself a plugin.
    assert not (ROOT / ".claude-plugin" / "plugin.json").exists()


def test_the_plugin_is_this_aquakin_s_version_and_starts_this_version():
    assert _json(PLUGIN / ".claude-plugin" / "plugin.json")["version"] == aquakin.__version__
    server = _json(PLUGIN / ".mcp.json")["mcpServers"]["aquakin"]
    assert server["command"] == "uvx"
    assert server["args"] == ["--from", f"aquakin[mcp]=={aquakin.__version__}", "aquakin", "mcp"]


@pytest.mark.parametrize(
    "skill", sorted(p.name for p in (PLUGIN / "skills").iterdir() if p.is_dir())
)
def test_each_skill_names_itself_and_says_when_to_use_it(skill):
    text = (PLUGIN / "skills" / skill / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    assert match, "SKILL.md must open with --- frontmatter ---"
    front = yaml.safe_load(match.group(1))
    assert front["name"] == skill
    assert "Use when" in front["description"]


def test_skills_name_only_tools_the_server_has():
    """A skill that names a tool the server lacks sends the assistant looking
    for something that is not there."""
    from aquakin.mcp.scripts import run_python
    from aquakin.mcp.server import TOOLS

    tools = {fn.__name__ for fn, _ in TOOLS} | {run_python.__name__}
    for skill in (PLUGIN / "skills").glob("*/SKILL.md"):
        named = set(re.findall(r"`([a-z_]+)`", skill.read_text(encoding="utf-8")))
        verbs = r"(list|describe|read|check|write|create|update|run|study|cancel|copy|freeze|unfreeze|search)"
        called = {n for n in named if re.fullmatch(verbs + r"_[a-z_]+", n)}
        assert called <= tools, f"{skill.parent.name} names {sorted(called - tools)}"


def test_the_guide_pages_the_skills_read_exist():
    pages = {p.stem for p in (ROOT / "docs").glob("*.md")}
    for skill in (PLUGIN / "skills").glob("*/SKILL.md"):
        text = skill.read_text(encoding="utf-8")
        named = set(re.findall(r"`read_doc_page` on `([a-z_]+)`", text))
        named |= set(re.findall(r"`read_doc_page`: `([a-z_]+)`", text))
        for listed in re.findall(r"`read_doc_page`[^.]*?:((?: `[a-z_]+`,?)+)", text):
            named |= set(re.findall(r"`([a-z_]+)`", listed))
        assert named <= pages, f"{skill.parent.name} names {sorted(named - pages)}"
