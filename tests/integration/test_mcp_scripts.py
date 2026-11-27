"""The opt-in script tool, ``run_python`` (``aquakin mcp --allow-python``): what
a script sees, what comes back, and the server offering it only when asked."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed")

from mcp import Client

import aquakin
from aquakin.mcp import scripts, server
from aquakin.mcp.scripts import OUTPUT_LIMIT, run_python


def test_a_script_runs_in_the_project_with_aquakin_and_reports_what_it_wrote(tmp_path):
    (tmp_path / ".git").mkdir()
    code = (
        "import os, aquakin\n"
        "print(os.getcwd())\n"
        "print(aquakin.__version__)\n"
        "print(os.environ['MPLBACKEND'])\n"
        "os.makedirs('figures', exist_ok=True)\n"
        "open('figures/out.txt', 'w').write('done')\n"
        "open('.git/ignored', 'w').write('x')\n"
    )
    result = run_python(str(tmp_path), code)
    assert result["exit_code"] == 0 and not result["timed_out"]
    assert result["stdout"].splitlines() == [str(tmp_path), aquakin.__version__, "Agg"]
    assert result["stderr"] == "" and not result["truncated"]
    assert result["files"] == ["figures/out.txt"]  # not .git/
    assert (tmp_path / "figures" / "out.txt").read_text() == "done"
    assert "saved_as" not in result


def test_a_failing_script_reports_its_exit_code_and_traceback(tmp_path):
    result = run_python(str(tmp_path), "1 / 0\n")
    assert result["exit_code"] == 1
    assert result["stderr"].rstrip().endswith("ZeroDivisionError: division by zero")
    assert result["files"] == []


def test_a_script_past_its_timeout_is_stopped_with_what_it_printed(tmp_path):
    code = "import time\nprint('started', flush=True)\ntime.sleep(60)\n"
    result = run_python(str(tmp_path), code, timeout=2)
    assert result["timed_out"] and result["exit_code"] is None
    assert result["stdout"] == "started\n"
    assert result["seconds"] < 60


def test_long_output_keeps_its_end(tmp_path):
    result = run_python(str(tmp_path), f"print('a' * {OUTPUT_LIMIT + 10}, end='')\nprint('END')\n")
    assert result["truncated"]
    assert len(result["stdout"]) == OUTPUT_LIMIT and result["stdout"].endswith("aEND\n")


def test_a_kept_script_is_saved_in_the_project_and_never_written_over(tmp_path):
    code = "open('result.txt', 'w').write('1')\n"
    result = run_python(str(tmp_path), code, save_as="scripts/fit.py")
    assert result["saved_as"] == "scripts/fit.py"
    assert (tmp_path / "scripts" / "fit.py").read_text() == code
    assert result["files"] == ["result.txt"]  # the script itself is not "written by the run"
    with pytest.raises(FileExistsError, match=r"fit\.py already exists; choose another save_as"):
        run_python(str(tmp_path), "print(2)\n", save_as="scripts/fit.py")
    assert (tmp_path / "scripts" / "fit.py").read_text() == code
    with pytest.raises(ValueError, match=r"save_as must be a path inside the project, not '\.\./x\.py'"):
        run_python(str(tmp_path), "print(3)\n", save_as="../x.py")
    assert not (tmp_path.parent / "x.py").exists()


@pytest.mark.parametrize("timeout", [0, -1, scripts.MAX_TIMEOUT + 1])
def test_a_timeout_outside_its_range_is_refused(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout must be more than 0 and at most 3600 seconds"):
        run_python(str(tmp_path), "print(1)\n", timeout=timeout)


def test_a_project_that_is_not_a_folder_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="is not a folder"):
        run_python(str(tmp_path / "nowhere"), "print(1)\n")


def test_a_file_removed_while_the_folder_is_read_is_left_out(tmp_path, monkeypatch):
    (tmp_path / "gone.txt").write_text("x")
    (tmp_path / "kept.txt").write_text("y")
    real = scripts.Path.stat

    def stat(path, *args, **kwargs):
        if path.name == "gone.txt":
            raise FileNotFoundError(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(scripts.Path, "stat", stat)
    assert scripts._written_since(tmp_path, 0) == ["kept.txt"]


def test_the_server_offers_the_tool_only_when_allowed(tmp_path):
    async def tools(allow):
        async with Client(server.build_server(allow_python=allow)) as client:
            listed = {t.name: t for t in (await client.list_tools()).tools}
            result = None
            if allow:
                result = await client.call_tool(
                    "run_python", {"project": str(tmp_path), "code": "print(6 * 7)"}
                )
            return listed, result

    listed, _ = asyncio.run(tools(False))
    assert "run_python" not in listed
    assert "run_python" not in server.build_server().instructions
    listed, result = asyncio.run(tools(True))
    hints = listed["run_python"].annotations
    assert (hints.read_only_hint, hints.destructive_hint) == (False, True)
    assert "run_python runs a Python script" in server.build_server(allow_python=True).instructions
    assert not result.is_error
    assert '"stdout": "42\\n"' in result.content[0].text


def test_the_command_line_flag_turns_the_tool_on(monkeypatch):
    from aquakin.__main__ import main

    seen = []
    monkeypatch.setattr(server, "main", lambda *, allow_python: seen.append(allow_python))
    assert main(["mcp"]) == 0 and main(["mcp", "--allow-python"]) == 0
    assert seen == [False, True]
