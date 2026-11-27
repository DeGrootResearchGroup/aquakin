# aquakin for Claude Code

[aquakin](https://github.com/DeGrootResearchGroup/aquakin) is an open-source
Python library for modelling reactive transport in water and wastewater
treatment: kinetic models such as the activated sludge and anaerobic digestion
models, whole-plant flowsheets built from them, and analyses on those plants
(simulation, calibration, Fisher information, identifiability and experiment
design). This plugin lets Claude work with it.

## What it adds

**aquakin's tools**, served by `aquakin mcp` on your computer. Claude can:

- look up the shipped kinetic models and the unit operations a flowsheet can
  contain, and read model files;
- read, check and write plant flowsheets, as the aquakin flowsheet editor does;
- create, edit, run and read studies (simulations, calibrations, Fisher
  information, profile likelihoods, sensor rankings, posterior sampling of a
  calibration), saved in your project
  folder with their results; a run goes on in the background;
- search and read the aquakin user guide and API reference of the installed
  version.

**Skills** for the common tasks, which Claude uses when a request calls for
them, or you can call by name:

| Skill | For |
|---|---|
| `/aquakin:build-flowsheet` | adding, removing or rewiring units, and checking a plant |
| `/aquakin:run-study` | setting up, running and reading studies |
| `/aquakin:write-model` | writing or extending a kinetic model file |
| `/aquakin:analyze-with-python` | analyses in Python that no study covers |

## Installing

In a Claude Code session:

```text
/plugin install aquakin --marketplace DeGrootResearchGroup/aquakin
```

The plugin needs [uv](https://docs.astral.sh/uv/getting-started/installation/).
It runs the aquakin release it was published with; the first start downloads
aquakin and its dependencies from PyPI (a few hundred MB), once.

## Your files and data

Everything runs on your computer, in the folders you point Claude at. The
tools refuse to write over a flowsheet file or a study's results unless Claude
asks for that explicitly, and Claude Code asks for your approval before tools
that change files. What a tool returns, such as a model's parameters or your flowsheet,
becomes part of your conversation with Claude, like anything else you share
with it. The plugin itself sends nothing anywhere else.

## More

- [Using aquakin from an AI assistant](https://aquakin.readthedocs.io/en/latest/ai_assistants.html)
- [aquakin documentation](https://aquakin.readthedocs.io/en/latest/)
- [Issues](https://github.com/DeGrootResearchGroup/aquakin/issues)

aquakin and this plugin are released under the MIT licence.
