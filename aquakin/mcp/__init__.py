"""aquakin as a Model Context Protocol (MCP) server: ``aquakin mcp``.

An AI assistant (Claude Desktop, Claude Code, or any other MCP client) starts
``aquakin mcp`` on the user's own computer and talks to it over standard input
and output. The server offers *tools* the assistant may call: list and describe
the shipped kinetic models and unit kinds, read and check a flowsheet, and read
or search the user guide and the API reference. Every computation runs locally,
under the assistant's own account; aquakin sends nothing anywhere.

The server needs the optional ``mcp`` extra (``pip install "aquakin[mcp]"``).
:mod:`aquakin.mcp.docs` and :mod:`aquakin.mcp.api` do not, so the guide pages
and the API reference can be read and tested without it.
"""
