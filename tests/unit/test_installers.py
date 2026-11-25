"""The editor installers (``installers/``) and their release stamping.

The shell installer is run for real against a stand-in ``uv`` that records its
arguments, in a temporary home, so the test sees what it would install and the
launcher it writes without downloading anything. The batch installer cannot run
off Windows; it is held to the shell one by comparing what both declare.
"""

import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SH = ROOT / "installers" / "install-aquakin.sh"
BAT = ROOT / "installers" / "install-aquakin.bat"
SCRIPT = ROOT / "scripts" / "stamp_installers.py"


@pytest.fixture(scope="module")
def stamper():
    spec = importlib.util.spec_from_file_location("stamp_installers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _declared(path, name):
    """The value ``path`` assigns to ``name`` (sh ``X="v"`` or cmd ``set "X=v"``)."""
    text = path.read_text(encoding="utf-8")
    found = re.findall(rf'^(?:set ")?{name}="?([^"\r\n]*)"?\r?$', text, re.M)
    assert len(found) == 1, (path.name, name, found)
    return found[0]


def test_both_installers_declare_the_same_python_and_an_unstamped_version():
    assert _declared(SH, "PYTHON_VERSION") == _declared(BAT, "PYTHON_VERSION") == "3.12"
    assert _declared(SH, "AQUAKIN_VERSION") == _declared(BAT, "AQUAKIN_VERSION") == ""


def test_both_installers_install_the_same_package_with_the_same_uv_options():
    for path in (SH, BAT):
        text = path.read_text(encoding="utf-8")
        assert "aquakin[ui]==" in text and "uv tool install --python" in text, path.name
        assert "--python-preference only-managed --upgrade" in text, path.name
        assert "https://raw.githubusercontent.com/DeGrootResearchGroup/aquakin/v" in text.replace(
            "%AQUAKIN_VERSION%", "$AQUAKIN_VERSION"
        ), path.name
        assert "/constraints.txt" in text, path.name


@pytest.mark.parametrize("path", [SH, BAT], ids=["sh", "bat"])
def test_stamp_fills_in_the_version(stamper, path):
    stamped = stamper.stamp(path.read_text(encoding="utf-8"), "1.2.3")
    assert re.findall(r'^(?:set ")?AQUAKIN_VERSION="?([^"\r\n]*)"', stamped, re.M) == ["1.2.3"]


def test_stamp_refuses_a_stamped_installer_and_a_non_version(stamper):
    text = SH.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="found 0"):
        stamper.stamp(stamper.stamp(text, "1.2.3"), "1.2.4")
    with pytest.raises(ValueError, match="not a release version"):
        stamper.stamp(text, "v1.2.3")


def test_main_writes_both_installers_with_their_shells_line_endings_and_mode(stamper, tmp_path):
    out = tmp_path / "out"
    assert stamper.main(["2.0.0rc1", str(out)]) == 0
    sh, bat = (out / SH.name).read_bytes(), (out / BAT.name).read_bytes()
    assert b"\r" not in sh and sh.count(b"\n") == len(SH.read_text().splitlines())
    assert bat.count(b"\r\n") == bat.count(b"\n") == len(BAT.read_text().splitlines())
    for source in (SH, BAT):
        written = (out / source.name).read_bytes()
        assert b"2.0.0rc1" in written
        assert stat.S_IMODE((out / source.name).stat().st_mode) == stat.S_IMODE(
            source.stat().st_mode
        )


def test_main_explains_its_usage(stamper, capsys):
    assert stamper.main([]) == 2
    assert "stamp_installers.py" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("sh") is None, reason="the shell installer needs a POSIX sh")
def test_shell_installer_installs_the_stamped_release_and_writes_a_launcher(stamper, tmp_path):
    home, fake, tools = tmp_path / "home", tmp_path / "fake", tmp_path / "tools"
    for d in (home, fake, tools):
        d.mkdir()
    (tools / "aquakin-ui").write_text("#!/bin/sh\n")
    (tools / "aquakin-ui").chmod(0o755)
    log = tmp_path / "uv.log"
    # A stand-in uv: answers `uv tool dir --bin`, records every other call and
    # the constraints file it was given.
    (fake / "uv").write_text(
        "#!/bin/sh\n"
        f'if [ "$1 $2 $3" = "tool dir --bin" ]; then echo "{tools}"; exit 0; fi\n'
        f'printf "%s\\n" "$@" "constraints=$UV_CONSTRAINT" >> "{log}"\n'
    )
    (fake / "uv").chmod(0o755)
    installer = tmp_path / "install-aquakin.sh"
    installer.write_text(stamper.stamp(SH.read_text(encoding="utf-8"), "9.8.7"))
    env = {"HOME": str(home), "PATH": f"{fake}{os.pathsep}/usr/bin{os.pathsep}/bin"}

    result = subprocess.run(["sh", str(installer)], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert log.read_text().split("\n")[:9] == [
        "tool",
        "install",
        "--python",
        "3.12",
        "--python-preference",
        "only-managed",
        "--upgrade",
        "aquakin[ui]==9.8.7",
        "constraints=https://raw.githubusercontent.com/DeGrootResearchGroup/aquakin/v9.8.7/constraints.txt",
    ]
    if sys.platform == "darwin":
        launcher = home / "Desktop" / "aquakin.command"
        assert os.access(launcher, os.X_OK)
        assert f'exec "{tools}/aquakin-ui"' in launcher.read_text()
    else:
        launcher = home / ".local" / "share" / "applications" / "aquakin.desktop"
        assert f'Exec="{tools}/aquakin-ui"' in launcher.read_text()
    assert "aquakin is installed." in result.stdout


@pytest.mark.skipif(shutil.which("sh") is None, reason="the shell installer needs a POSIX sh")
def test_shell_installer_stops_when_the_editor_is_missing(tmp_path):
    home, fake = tmp_path / "home", tmp_path / "fake"
    home.mkdir()
    fake.mkdir()
    (fake / "uv").write_text(f'#!/bin/sh\n[ "$2" = dir ] && echo "{tmp_path / "empty"}"\nexit 0\n')
    (fake / "uv").chmod(0o755)
    env = {"HOME": str(home), "PATH": f"{fake}{os.pathsep}/usr/bin{os.pathsep}/bin"}

    result = subprocess.run(["sh", str(SH)], env=env, capture_output=True, text=True)

    assert result.returncode == 1
    assert "the editor was not found" in result.stdout
    assert not (home / "Desktop" / "aquakin.command").exists()
