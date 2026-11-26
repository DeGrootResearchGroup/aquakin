# Simulation

Once you have a model, run it in a **reactor** (one volume or one flow path) or
in a whole **plant** (a flowsheet of units joined by streams and recycles).

| If you want to… | Read |
|---|---|
| simulate a batch test, a plug-flow reactor, a biofilm or a particle track; dose or switch during a run; set the integrator | [Reactors](reactors.md) |
| run the BSM1 and BSM2 benchmarks: steady state, dynamic runs, performance indices, cost and greenhouse gases; check a plant's units and mass balance | [Plant-wide simulation](plants.md) |
| build a treatment plant of your own from the unit operations: tanks, biofilm carriers, SBRs, clarifiers, disinfection | [Building a flowsheet](custom_flowsheets.md) |
| drive a plant with a feed that follows a pattern, a pump schedule, or laboratory and SCADA records | [Influents](influents.md) |
| save a plant as a file, run reproducible studies on it, or draw and edit it in the browser | [Flowsheet files, studies and the editor](plant_files.md) |
| install the flowsheet editor on a computer without Python | [Installing the editor without Python](installing_the_editor.md) |

```{toctree}
:maxdepth: 2

reactors
plants
custom_flowsheets
influents
plant_files
installing_the_editor
```
