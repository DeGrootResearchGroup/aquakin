"""The editor's preferences (``aquakin_ui.settings``)."""

import json

import pytest

from aquakin_ui.settings import DEFAULTS, Settings, settings_path


def test_preferences_start_at_their_defaults_and_are_kept(tmp_path):
    path = tmp_path / "config" / "ui.json"
    settings = Settings(path)
    assert settings["ask_before_replacing"] is True and not path.exists()
    assert settings.set("ask_before_replacing", False)
    assert json.loads(path.read_text()) == {**DEFAULTS, "ask_before_replacing": False}
    assert Settings(path)["ask_before_replacing"] is False
    with pytest.raises(KeyError, match="no preference 'colour'"):
        settings.set("colour", "blue")


def test_a_bad_file_gives_the_defaults_and_an_unwritable_one_the_session(tmp_path):
    path = tmp_path / "ui.json"
    for text in ("not json", "[1, 2]", '{"ask_before_replacing": "no", "other": 1}'):
        path.write_text(text)
        assert Settings(path)["ask_before_replacing"] is True
    blocked = tmp_path / "a file" / "ui.json"
    (tmp_path / "a file").write_text("")
    settings = Settings(blocked)
    assert settings.set("ask_before_replacing", False) is False
    assert settings["ask_before_replacing"] is False


def test_the_file_is_in_the_users_configuration_folder(monkeypatch, tmp_path):
    assert settings_path() == tmp_path / "preferences" / "ui.json"  # the tests' own
    monkeypatch.delenv("AQUAKIN_UI_CONFIG")
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert settings_path() == tmp_path / "xdg" / "aquakin" / "ui.json"
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    assert settings_path() == tmp_path / "roaming" / "aquakin" / "ui.json"
    monkeypatch.setattr("sys.platform", "darwin")
    assert settings_path().parts[-3:] == ("Application Support", "aquakin", "ui.json")
