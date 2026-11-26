# Analysis

Every solve in `aquakin` is differentiable, so the same reactor or plant you
simulate can be analyzed: how its outputs depend on the parameters, which
parameter values fit your data, how well the data determine them, and which
experiment would determine them best. If Fisher information, posteriors or
identifiability are new to you, start with the key ideas.

| If you want to… | Read |
|---|---|
| learn the ideas the other pages use: Fisher information, the Laplace posterior, profile likelihood, identifiability | [Key ideas: information, uncertainty and identifiability](analysis_concepts.md) |
| find the tool for a question; take a gradient through a solve; see how outputs depend on parameters; propagate uncertainty | [Sensitivity analysis](sensitivity_and_calibration.md) |
| fit parameters to your measurements | [Calibration](calibration.md) |
| say how well the data determine the fitted values, report them, compare model structures | [Uncertainty and model comparison](fit_uncertainty.md) |
| choose sampling times, doses or measurements that determine the parameters best | [Experiment design](experiment_design.md) |
| fit and analyze the parameters of a whole plant against measured streams | [Calibrating a plant](plant_calibration.md) |
| follow one study from a first experiment design to a calibrated model | [Worked example: designing an experiment](worked_example_design.md) |

```{toctree}
:maxdepth: 2

analysis_concepts
sensitivity_and_calibration
calibration
fit_uncertainty
experiment_design
plant_calibration
worked_example_design
```
