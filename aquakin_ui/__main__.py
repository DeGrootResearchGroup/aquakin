"""``aquakin-ui``: open the flowsheet editor in the browser.

Usage::

    aquakin-ui [flowsheet.yaml] [--port PORT] [--no-browser]

The page is served on this machine only. Stop it with Ctrl-C.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    """Run the editor."""
    parser = argparse.ArgumentParser(prog="aquakin-ui", description="aquakin flowsheet editor")
    parser.add_argument("flowsheet", nargs="?", help="a flowsheet spec file to open")
    parser.add_argument("--port", type=int, default=0, help="port (default: a free one)")
    parser.add_argument("--no-browser", action="store_true", help="print the address only")
    args = parser.parse_args(argv)

    from trame.app import get_server

    from aquakin_ui.app import Editor

    server = get_server(client_type="vue3")
    editor = Editor(server, path=args.flowsheet)
    try:
        server.start(port=args.port, host="127.0.0.1", open_browser=not args.no_browser)
    finally:
        editor.stop_runs()
        editor.worker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
