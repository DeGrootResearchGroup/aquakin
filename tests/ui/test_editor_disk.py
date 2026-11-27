"""The editor follows the project's files as another program (``aquakin mcp``,
a text editor) changes them: the open flowsheet is shown again, unsaved edits
are not dropped without asking, and the studies are listed again."""

import asyncio
import itertools
import os

import pytest
from trame.app import get_server

from aquakin_ui import disk
from aquakin_ui.app import Editor
from tests.conftest import study_project

from .inprocess import InProcessWorker

_names = itertools.count()


@pytest.fixture
def project_editor(tmp_path):
    """``(editor, project)``: an editor on a fresh one-tank project."""
    project = study_project(tmp_path)
    server = get_server(name=f"aquakin-ui-disk-{next(_names)}", client_type="vue3")
    ed = Editor(server, worker=InProcessWorker(), path=project / "flowsheet.yaml")
    yield ed, project
    ed.worker.stop()


def _rewrite_volume(project, volume: float) -> None:
    """Another program writes the flowsheet with the tank's volume changed."""
    spec = project / "flowsheet.yaml"
    text = spec.read_text()
    assert "volume: 1000.0" in text
    spec.write_text(text.replace("volume: 1000.0", f"volume: {volume}"))


def test_a_flowsheet_changed_on_disk_is_shown_again(project_editor):
    ed, project = project_editor
    ed.follow_disk()  # the first look only records the folder
    fit = ed.state.fit_key
    _rewrite_volume(project, 1500.0)
    ed.follow_disk()
    assert ed.doc.units["tank"].settings["volume"] == 1500.0
    assert ed.state.message == "reloaded flowsheet.yaml: it changed on disk"
    assert ed.state.fit_key == fit  # the view is kept
    assert not ed._unsaved()
    # Nothing changed since: no second reload.
    ed.state.message = ""
    ed.follow_disk()
    assert ed.state.message == ""


def test_unsaved_edits_are_kept_unless_the_user_discards_them(project_editor):
    ed, project = project_editor
    ed.on_move([{"id": "tank", "position": {"x": 999.0, "y": 0.0}}])
    _rewrite_volume(project, 1500.0)
    ed.follow_disk()
    assert ed.state.unsaved_open
    assert "flowsheet.yaml has changed on disk" in ed.state.unsaved_text
    assert not ed.state.unsaved_can_save  # saving would write over the new version
    assert ed.doc.units["tank"].settings["volume"] == 1000.0

    ed.answer_unsaved("cancel")  # keep the edits
    ed.follow_disk()
    assert not ed.state.unsaved_open  # asked once per change of the file
    assert ed.doc.units["tank"].position["x"] == 999.0
    ed.save()
    assert "has changed on disk" in ed.state.message  # Save still refuses

    _rewrite_volume_again = (project / "flowsheet.yaml").read_text().replace(
        "volume: 1500.0", "volume: 2000.0"
    )
    (project / "flowsheet.yaml").write_text(_rewrite_volume_again)
    ed.follow_disk()  # a new change: asked again
    assert ed.state.unsaved_open
    ed.answer_unsaved("discard")
    assert ed.doc.units["tank"].settings["volume"] == 2000.0
    assert ed.doc.units["tank"].position["x"] != 999.0
    assert not ed._unsaved()


def test_a_save_here_is_not_taken_for_another_programs_change(project_editor):
    ed, _ = project_editor
    ed.follow_disk()
    ed.on_move([{"id": "tank", "position": {"x": 999.0, "y": 0.0}}])
    ed.save()
    assert ed.state.message.startswith("saved ")
    ed.follow_disk()
    assert not ed.state.unsaved_open
    assert not ed.state.message.startswith("reloaded")


def test_a_study_written_by_another_program_is_listed(project_editor):
    from aquakin.study import create_study

    ed, project = project_editor
    ed.follow_disk()
    assert [st["name"] for st in ed.state.studies] == ["short"]
    create_study(project, "longer", {"kind": "simulate", "streams": ["tank"]})
    ed.follow_disk()
    assert [st["name"] for st in ed.state.studies] == ["longer", "short"]


def test_without_a_file_nothing_is_followed(tmp_path):
    server = get_server(name=f"aquakin-ui-disk-{next(_names)}", client_type="vue3")
    ed = Editor(server, worker=InProcessWorker())
    try:
        ed.follow_disk()
        assert ed._project_seen is None and ed.state.studies == []
    finally:
        ed.worker.stop()


def test_project_files_lists_each_file_once_and_skips_hidden_folders(tmp_path):
    (tmp_path / "studies" / "a").mkdir(parents=True)
    (tmp_path / "studies" / "a" / "study.yaml").write_text("x")
    (tmp_path / "flowsheet.yaml").write_text("yy")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "index").write_text("zzz")
    os.utime(tmp_path / "flowsheet.yaml", ns=(1, 2_000_000_000))
    files = disk.project_files(tmp_path)
    assert [(name, size) for name, _, size in files] == [
        ("flowsheet.yaml", 2),
        (os.path.join("studies", "a", "study.yaml"), 1),
    ]
    assert files[0][1] == 2_000_000_000
    (tmp_path / "flowsheet.yaml").write_text("yyy")
    assert disk.project_files(tmp_path) != files


def test_a_file_removed_while_listed_is_left_out(tmp_path, monkeypatch):
    (tmp_path / "gone.yaml").write_text("x")
    (tmp_path / "kept.yaml").write_text("y")
    real_stat = disk.Path.stat

    def stat(path, *args, **kwargs):
        if path.name == "gone.yaml":
            raise FileNotFoundError(path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(disk.Path, "stat", stat)
    assert [name for name, _, _ in disk.project_files(tmp_path)] == ["kept.yaml"]


def test_the_page_looks_at_the_disk_while_it_is_served(project_editor, monkeypatch):
    ed, project = project_editor
    monkeypatch.setattr(disk, "WATCH_SECONDS", 0.01)

    async def go():
        task = asyncio.get_running_loop().create_task(ed.watch_disk())
        _rewrite_volume(project, 1500.0)
        loop = asyncio.get_running_loop()
        end = loop.time() + 60.0
        while ed.doc.units["tank"].settings["volume"] != 1500.0:
            assert loop.time() < end, "the change was not picked up"
            await asyncio.sleep(0.01)
        task.cancel()

    asyncio.run(go())


def test_a_run_another_program_started_is_shown_and_can_be_cancelled(project_editor, hold_run):
    ed, project = project_editor
    ed.follow_disk()
    held = hold_run(project / "studies" / "short")
    ed.follow_disk()  # running.yaml appeared: the studies are listed again
    (card,) = ed.state.studies
    assert card["running"] is True and card["state"] == "not run"
    assert card["run"]["progress"]["message"] == "step 3"
    assert card["run"]["pid"] == held.pid

    ed.cancel_run("short")  # not this editor's run: asked through the worker
    assert ed.state.message == "short: cancelling"
    held.stdin.write("next\n")
    held.stdin.flush()
    assert held.stdout.readline() == "cancelled\n"
    assert held.wait(60) == 0
    ed.follow_disk()
    (card,) = ed.state.studies
    assert card["running"] is False and card["run"] is None


def test_cancelling_a_study_that_is_not_running_does_nothing(project_editor):
    ed, _ = project_editor
    ed.refresh_studies()
    ed.state.message = ""
    ed.cancel_run("short")
    ed.cancel_run("no such study")
    assert ed.state.message == ""
