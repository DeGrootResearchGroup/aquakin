# Installing the editor without Python

The flowsheet editor can be installed by someone who has never used Python or
a terminal. An installer fetches everything it needs, including Python, into a
folder of its own. It leaves any other Python on the computer alone, and it
needs no administrator rights.

The installer installs the release it came with. It also installs the exact
versions of aquakin's dependencies that the release was tested with. The first
installation downloads about 300 MB.

## Windows

1. Download
   [install-aquakin.bat](https://github.com/DeGrootResearchGroup/aquakin/releases/latest/download/install-aquakin.bat).
2. Double-click it in your Downloads folder. If Windows shows "Windows protected
   your PC", click **More info**, then **Run anyway**.
3. Wait for the window to say "aquakin is installed", then press any key to
   close it.

An **aquakin** shortcut is now on your Desktop and in the Start menu.

## macOS and Linux

1. Open **Terminal** (on a Mac, press ⌘-Space, type *Terminal* and press Return).
2. Paste this line and press Return:

   ```bash
   curl -LsSf https://github.com/DeGrootResearchGroup/aquakin/releases/latest/download/install-aquakin.sh | sh
   ```

3. Wait for it to say "aquakin is installed", then close Terminal.

On a Mac, an **aquakin** launcher is now on your Desktop. On Linux, **aquakin**
is in your applications menu.

## Using the editor

Double-click **aquakin**. A small window opens and starts the editor, and the
editor then opens in your web browser. The first start can take a minute.

Keep the small window open while you work. Close it when you are done to stop
aquakin. The editor starts in your Documents folder.

The editor itself is described in [Plant-wide simulation](plants.md).

## Updating and removing

To update, download and run the installer again, the same way as the first
time. It replaces the installed version with the newest release.

To remove aquakin, delete the **aquakin** shortcut or launcher and run this in
a terminal (Command Prompt on Windows):

```bash
uv tool uninstall aquakin
```

## If you already use Python

The installer is a convenience. If you manage Python environments yourself,
install the editor as described in [Getting started](getting_started.md):

```bash
pip install "aquakin[ui]"
aquakin-ui
```
