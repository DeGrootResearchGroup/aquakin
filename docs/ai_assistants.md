# Using aquakin from an AI assistant

`aquakin mcp` connects aquakin to an AI assistant, such as Claude Desktop,
Claude Code or any other app that supports the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP). You then ask
the assistant in plain language ("which shipped model has two-step
nitrification?", "check my flowsheet", "how do I calibrate a batch test?"), and
it answers by calling aquakin's tools.

The assistant runs under your own account and plan with its provider. aquakin
itself needs no account or key and sends nothing anywhere: the assistant's app
starts `aquakin mcp` on your computer, and every tool runs there. What a tool
returns, such as a model's parameters or your flowsheet, becomes part of your
conversation with the assistant, so it is sent to the assistant's provider like
anything else you type.

## Installing

Install aquakin with the `mcp` extra:

```bash
pip install "aquakin[mcp]"
```

Then tell your assistant's app how to start the server. The command is
`aquakin mcp`. If aquakin is installed in a virtual environment, give the full
path of the `aquakin` program in that environment (`which aquakin` on macOS or
Linux, `where aquakin` on Windows), because the app does not activate the
environment for you.

**Claude Code.** Run this once, in a terminal:

```bash
claude mcp add aquakin -- aquakin mcp
```

**Claude Desktop.** Open **Settings → Developer → Edit Config**, add aquakin to
`mcpServers` in the file that opens, save it, and restart Claude Desktop:

```json
{
  "mcpServers": {
    "aquakin": {"command": "aquakin", "args": ["mcp"]}
  }
}
```

**Other apps.** Any MCP app takes the same command: `aquakin` with the
argument `mcp`, speaking over standard input and output.

## What the assistant can do

| Tool | What it does |
|---|---|
| `list_models`, `describe_model`, `read_model_file` | list the shipped kinetic models; show a model's species, reactions, stoichiometry and parameters; read a model's YAML source, the shipped ones or your own |
| `list_unit_kinds`, `describe_unit_kind`, `check_unit` | list the units a flowsheet can contain; show every setting of one kind; build one unit from settings and report its ports or why it cannot be built |
| `read_flowsheet`, `check_flowsheet` | read a flowsheet file; build the plant and report every problem in it, as the editor does |
| `list_doc_pages`, `search_docs`, `read_doc_page` | list, search and read this user guide, for the version of aquakin you have installed |
| `list_api`, `describe_api` | list the public functions and classes; give one's signature and full docstring |

Every tool only reads: none writes a file or runs a simulation. When the
assistant writes a model file, a flowsheet or a Python script for you, it can
check its work against these tools and the guide, and you save and run the
result. In Claude Code, the assistant can also run Python itself, so it can use
the whole [public API](public_api.md) directly.

## Things to ask

- "Which shipped models include phosphorus removal, and how do they differ?"
- "Describe the ASM1 parameters that control nitrification."
- "Check `flowsheet.yaml` in this folder and explain each problem it reports."
- "Write a model file for ozone decay with a fast and a slow phase, and check
  it against the model format."
- "How do I find which parameters my batch data can identify?"
