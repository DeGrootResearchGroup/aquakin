"""Start aquakin's MCP server for Claude Desktop: the extension's entry point.

Its one argument is the extension's "Let Claude run Python scripts" setting,
which Claude Desktop passes as ``true`` or ``false``.
"""

import sys

from aquakin.__main__ import main

sys.exit(main(["mcp", *(["--allow-python"] if sys.argv[1:] == ["true"] else [])]))
