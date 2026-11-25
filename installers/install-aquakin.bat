@echo off
rem Install the aquakin flowsheet editor on Windows, with no Python needed.
rem
rem Download this file and double-click it. It installs uv (a small program
rem that fetches Python and aquakin into a folder of their own, leaving any
rem other Python on the computer alone), installs aquakin with its editor, and
rem puts an "aquakin" shortcut on the Desktop and in the Start menu. Run it
rem again to update.
rem
rem A release's installer installs that release with the dependency versions
rem it was tested with (its constraints.txt). For trying a build before
rem release, AQUAKIN_SPEC is installed instead when set (a requirement such as
rem "aquakin[ui] @ file:///C:/path/to/aquakin"), and AQUAKIN_CONSTRAINTS names
rem the constraints file to use (a path or URL).

setlocal

rem The release this installer installs; the release workflow fills it in.
rem Empty means the newest release on PyPI.
set "AQUAKIN_VERSION="
rem The Python the editor runs on: one every dependency publishes wheels for.
set "PYTHON_VERSION=3.12"

if defined AQUAKIN_SPEC (
    set "SPEC=%AQUAKIN_SPEC%"
) else if defined AQUAKIN_VERSION (
    set "SPEC=aquakin[ui]==%AQUAKIN_VERSION%"
) else (
    set "SPEC=aquakin[ui]"
)
if defined AQUAKIN_CONSTRAINTS (
    set "UV_CONSTRAINT=%AQUAKIN_CONSTRAINTS%"
) else if defined AQUAKIN_VERSION (
    set "UV_CONSTRAINT=https://raw.githubusercontent.com/DeGrootResearchGroup/aquakin/v%AQUAKIN_VERSION%/constraints.txt"
)

rem uv's own installer puts it here; add it so a uv installed just now (or
rem earlier, before this window's PATH was updated) is found.
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>nul
if errorlevel 1 (
    echo Installing uv ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto failed
)
where uv >nul 2>nul
if errorlevel 1 goto failed

echo Installing %SPEC% - the first time downloads about 300 MB ...
uv tool install --python %PYTHON_VERSION% --python-preference only-managed --upgrade "%SPEC%"
if errorlevel 1 goto failed

for /f "usebackq delims=" %%i in (`uv tool dir --bin`) do set "AQUAKIN_BIN=%%i"
if not exist "%AQUAKIN_BIN%\aquakin-ui.exe" goto failed

rem The shortcuts start in Documents, so the editor opens and saves files there.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$shell = New-Object -ComObject WScript.Shell;" ^
    "foreach ($folder in 'Desktop', 'Programs') {" ^
    "  $link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath($folder)) 'aquakin.lnk'));" ^
    "  $link.TargetPath = Join-Path $env:AQUAKIN_BIN 'aquakin-ui.exe';" ^
    "  $link.WorkingDirectory = [Environment]::GetFolderPath('MyDocuments');" ^
    "  $link.Description = 'aquakin flowsheet editor';" ^
    "  $link.Save()" ^
    "}"
if errorlevel 1 goto failed

echo.
echo aquakin is installed.
echo Double-click "aquakin" on your Desktop or in the Start menu to open the flowsheet editor in your browser.
echo It opens a small window too: keep it open while you work, and close it to stop aquakin.
echo Run this installer again at any time to update.
echo.
pause
exit /b 0

:failed
echo.
echo aquakin was not installed: the step above stopped with an error.
echo.
pause
exit /b 1
