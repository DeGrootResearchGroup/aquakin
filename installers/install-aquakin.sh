#!/bin/sh
# Install the aquakin flowsheet editor on macOS or Linux, with no Python needed.
#
# Paste this line into Terminal:
#
#     curl -LsSf https://github.com/DeGrootResearchGroup/aquakin/releases/latest/download/install-aquakin.sh | sh
#
# It installs uv (a small program that fetches Python and aquakin into a folder
# of their own, leaving any other Python on the computer alone), installs
# aquakin with its editor, and puts an "aquakin" launcher on the Desktop
# (macOS) or in the applications menu (Linux). Run it again to update.
#
# A release's installer installs that release with the dependency versions it
# was tested with (its constraints.txt). For trying a build before release,
# AQUAKIN_SPEC is installed instead when set (a requirement such as
# "aquakin[ui] @ file:///path/to/aquakin"), and AQUAKIN_CONSTRAINTS names the
# constraints file to use (a path or URL).

set -eu

# The release this installer installs; the release workflow fills it in.
# Empty means the newest release on PyPI.
AQUAKIN_VERSION=""
# The Python the editor runs on: one every dependency publishes wheels for.
PYTHON_VERSION="3.12"

say() {
    printf '%s\n' "$*"
}

fail() {
    say ""
    say "aquakin was not installed: $*"
    exit 1
}

if [ -n "${AQUAKIN_SPEC:-}" ]; then
    spec="$AQUAKIN_SPEC"
elif [ -n "$AQUAKIN_VERSION" ]; then
    spec="aquakin[ui]==$AQUAKIN_VERSION"
else
    spec="aquakin[ui]"
fi
if [ -n "${AQUAKIN_CONSTRAINTS:-}" ]; then
    UV_CONSTRAINT="$AQUAKIN_CONSTRAINTS"
    export UV_CONSTRAINT
elif [ -n "$AQUAKIN_VERSION" ]; then
    UV_CONSTRAINT="https://raw.githubusercontent.com/DeGrootResearchGroup/aquakin/v$AQUAKIN_VERSION/constraints.txt"
    export UV_CONSTRAINT
fi

# uv's own installer puts it in the first of these; add them all so a uv
# installed just now (or earlier, before the shell's PATH was updated) is found.
PATH="${UV_INSTALL_DIR:-$HOME/.local/bin}:${XDG_BIN_HOME:-$HOME/.local/bin}:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PATH

if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv ..."
    command -v curl >/dev/null 2>&1 || fail "curl is needed to download uv."
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv could not be downloaded."
    command -v uv >/dev/null 2>&1 || fail "uv was installed but cannot be found."
fi

say "Installing $spec (the first time downloads about 300 MB) ..."
uv tool install --python "$PYTHON_VERSION" --python-preference only-managed --upgrade "$spec" \
    || fail "uv tool install stopped with an error (see above)."

bin="$(uv tool dir --bin)"
editor="$bin/aquakin-ui"
[ -x "$editor" ] || fail "the editor was not found at $editor."

case "$(uname -s)" in
    Darwin)
        launcher="$HOME/Desktop/aquakin.command"
        mkdir -p "$HOME/Desktop"
        cat >"$launcher" <<EOF
#!/bin/sh
# Opens the aquakin flowsheet editor in the browser. Closing this window stops it.
cd "\$HOME/Documents" 2>/dev/null || cd "\$HOME"
exec "$editor"
EOF
        chmod +x "$launcher"
        where="Double-click \"aquakin\" on your Desktop"
        ;;
    *)
        apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
        launcher="$apps/aquakin.desktop"
        mkdir -p "$apps"
        cat >"$launcher" <<EOF
[Desktop Entry]
Type=Application
Name=aquakin
Comment=aquakin flowsheet editor
Exec="$editor"
Terminal=true
Categories=Science;Education;
EOF
        where="Open \"aquakin\" from your applications menu"
        ;;
esac

say ""
say "aquakin is installed."
say "$where to open the flowsheet editor in your browser."
say "It opens a small window too: keep it open while you work, and close it to stop aquakin."
say "Run this installer again at any time to update."
