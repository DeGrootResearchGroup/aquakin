# Using aquakin from an AI assistant

`aquakin mcp` connects aquakin to an AI assistant, such as Claude Desktop,
Claude Code or any other app that supports the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP). You then ask
the assistant in plain language ("which shipped model has two-step
nitrification?", "add a second aerobic tank to my plant", "calibrate the
nitrifier growth rate against these data"), and it answers by calling
aquakin's tools: it can read and check your project, write flowsheets, and
create, run and read [studies](plant_files.md).

The assistant runs under your own account and plan with its provider. aquakin
itself needs no account or key and sends nothing anywhere: the assistant's app
starts `aquakin mcp` on your computer, and every tool runs there. What a tool
returns, such as a model's parameters or your flowsheet, becomes part of your
conversation with the assistant, so it is sent to the assistant's provider like
anything else you type.

## Installing

**Claude Code.** Install the aquakin plugin, in a Claude Code session:

```text
/plugin install aquakin --marketplace DeGrootResearchGroup/aquakin
```

It starts the server for you and adds skills for the common tasks (building a
flowsheet, running a study, writing a model, analyzing in Python), which you
can also call as `/aquakin:build-flowsheet` and so on. The plugin runs the
aquakin release it was published with through
[uv](https://docs.astral.sh/uv/), which must be installed; the first start
downloads aquakin and its dependencies (a few hundred MB), once.

**Claude Desktop.** Download
[aquakin.mcpb](https://github.com/DeGrootResearchGroup/aquakin/releases/latest/download/aquakin.mcpb)
and open it, or drag it onto Claude Desktop's **Settings → Extensions**
window, then select **Install**. Claude Desktop installs aquakin and its
dependencies for the extension the first time it starts (a few hundred MB,
once). The extension's settings hold the switch that lets Claude run Python
scripts (see [below](#letting-the-assistant-run-python)).

**Any app, from your own installation.** Install aquakin with the `mcp` extra:

```bash
pip install "aquakin[mcp]"
```

Then tell your assistant's app how to start the server. The command is
`aquakin mcp`. If aquakin is installed in a virtual environment, give the full
path of the `aquakin` program in that environment (`which aquakin` on macOS or
Linux, `where aquakin` on Windows), because the app does not activate the
environment for you.

**Claude Code without the plugin.** Run this once, in a terminal:

```bash
claude mcp add aquakin -- aquakin mcp
```

**Claude Desktop without the extension.** Open **Settings → Developer → Edit Config**, add aquakin to
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
| `read_flowsheet`, `check_flowsheet`, `write_flowsheet` | read a flowsheet file; build the plant and report every problem in it, as the editor does; write a flowsheet file, as the editor's Save does |
| `list_studies`, `study_settings`, `study_choices`, `read_study`, `create_study`, `update_study` | list a project's studies and where each stands; describe every kind of study and its settings; list the streams, parameters and channels a study may name; read, create and edit a study |
| `run_study`, `study_status`, `cancel_study`, `study_results` | start a run in the background, follow its progress, stop it, and read the results |
| `copy_study`, `freeze_study`, `unfreeze_study` | copy a study with its results; freeze it on a snapshot of its plant, or make it follow the flowsheet again |
| `list_doc_pages`, `search_docs`, `read_doc_page` | list, search and read this user guide, for the version of aquakin you have installed |
| `list_api`, `describe_api` | list the public functions and classes; give one's signature and full docstring |

The tools write the same files the editor and `aquakin study` write, and they
keep the same safeguards. A flowsheet file is not written over, a study's
results are not replaced, and a frozen study with results is not unfrozen,
unless the assistant passes `replace` or `discard_results`. It is told to do
that only when you ask. Your assistant's app also shows which tools change
files, and may ask you before each such call.

A study runs in the background, so a long calibration does not hold up the
conversation: the assistant starts it, checks on it, and reads the results when
it has finished. Each run is a process of its own, so it goes on if you close
the assistant's app; ask again later, or watch it in the editor. A study runs
once at a time, whichever program started it (see
[Studies](plant_files.md#studies)).

In Claude Code, the assistant can also run Python itself, so it can use the
whole [public API](public_api.md) directly, including analyses no study kind
covers.

## Letting the assistant run Python

Other apps, Claude Desktop among them, cannot run code themselves. Start the
server with `aquakin mcp --allow-python` (in the Claude Desktop extension,
turn on **Let Claude run Python scripts** in its settings; in Claude Desktop's
config file, `"args": ["mcp", "--allow-python"]`) to add one more tool, `run_python`. With
it the assistant writes a Python script and runs it in your project folder, so
it can use any part of aquakin: a reactor fit, a sensitivity analysis, a plot
saved to a file. The tool returns what the script printed and the files it
wrote, and can keep the script in the project so you can run it again.

A script can do anything Python can on your computer, so the tool is off
unless you turn it on, and your app asks before each run. Read the script
before you approve it. A script is stopped after 5 minutes unless the
assistant asks for longer (up to an hour); a long analysis is better run as a
study.

## Watching in the editor

Keep the [flowsheet editor](plant_files.md#the-flowsheet-editor) open on the
same project while you work with the assistant: it shows each flowsheet the
assistant writes, and the studies it creates and runs, within a second or two.

## Things to ask

- "Which shipped models include phosphorus removal, and how do they differ?"
- "Describe the ASM1 parameters that control nitrification."
- "Check `flowsheet.yaml` in this folder and explain each problem it reports."
- "Add a 500 m³ anoxic tank before the first aerobic tank, then simulate a
  week of dynamic influent and compare effluent nitrate."
- "Set up a calibration of `muA` and `KNH_A` against `data/nitrification.csv`,
  run it, and tell me how well the data determine them."
- "Write a model file for ozone decay with a fast and a slow phase, and check
  it against the model format."
- "How do I find which parameters my batch data can identify?"
