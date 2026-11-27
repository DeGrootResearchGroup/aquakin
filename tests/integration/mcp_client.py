"""Calling ``aquakin mcp``'s tools through a real MCP client, in this process."""

from __future__ import annotations

import asyncio
import json

from mcp import Client

from aquakin.mcp import server as mcp_server


def session(calls):
    """Run ``await calls(client)`` against a new server; return its result."""

    async def run():
        async with Client(mcp_server.build_server()) as client:
            return await calls(client)

    return asyncio.run(run())


def call(tool: str, **arguments):
    """One tool call: ``(is_error, text of the first content block)``."""

    async def calls(client):
        result = await client.call_tool(tool, arguments)
        return result.is_error, result.content[0].text

    return session(calls)


def value(tool: str, **arguments):
    """A tool call that must succeed, its answer as data.

    The SDK sends a list or a string (one content block per item) with a
    structured copy under ``"result"``, and a dict as one JSON text block."""

    async def calls(client):
        return await client.call_tool(tool, arguments)

    result = session(calls)
    assert not result.is_error, result.content[0].text
    if result.structured_content is not None:
        return result.structured_content["result"]
    return json.loads(result.content[0].text)


def error(tool: str, **arguments) -> str:
    """A tool call that must fail: the message the assistant sees."""
    is_error, text = call(tool, **arguments)
    assert is_error, text
    return text
