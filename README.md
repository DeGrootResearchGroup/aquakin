# aquakin

[![CI](https://github.com/DeGrootResearchGroup/aquakin/actions/workflows/ci.yml/badge.svg)](https://github.com/DeGrootResearchGroup/aquakin/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DeGrootResearchGroup/aquakin/branch/main/graph/badge.svg)](https://codecov.io/gh/DeGrootResearchGroup/aquakin)
[![Docs](https://readthedocs.org/projects/aquakin/badge/?version=latest)](https://aquakin.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/aquakin.svg)](https://pypi.org/project/aquakin/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23784521.svg)](https://doi.org/10.5281/zenodo.23784521)

**Differentiable reaction kinetics for water and wastewater treatment.**

`aquakin` simulates the chemistry and biology of aqueous systems — from a
single ozonation batch test to a whole wastewater treatment plant. You write a
reaction model in a short YAML file (or pick one of the shipped models), and
aquakin compiles it to fast [JAX](https://github.com/jax-ml/jax) code solved by
the stiff integrators of [Diffrax](https://github.com/patrick-kidger/diffrax).
Because every solve is differentiable, the same model can be fitted to data,
screened for sensitivity, and used to design the next experiment.

## Highlights

- **Models as data.** Species, reactions, rate laws and parameters live in a
  readable YAML file with its literature references — no code to write and
  nothing to recompile. Rate expressions are parsed safely (no `eval`) and
  checked for unit consistency and mass / COD / nitrogen conservation.
- **A library of standard models.** Ozonation / bromate and UV/H₂O₂ chemistry;
  ASM1, ASM2d and ASM3 with two-step nitrification, N₂O, anammox and comammox
  variants; ADM1 anaerobic digestion; WATS sewer processes; and pH speciation
  with mineral precipitation (struvite, calcite, chemical phosphorus removal).
- **Reactors and whole plants.** Batch, plug-flow, biofilm and particle-track
  reactors, and plant-wide flowsheets: the IWA benchmarks BSM1 and BSM2 build
  in one line, with clarifiers, IFAS/MBBR, digestion, disinfection, events
  (pumps, dosing, SBR phases), steady-state solvers and EQI / OCI, carbon
  footprint and cost reporting.
- **Gradients through everything.** `jax.grad` works through a stiff solve —
  including a full BSM2 plant — with exact forward sensitivities and a stable
  adjoint, so derivatives stay finite where naive autodiff breaks down.
- **Calibration and experiment design.** Fit parameters with uncertainty,
  check identifiability before running an experiment, choose sampling times
  and sensors, and propagate uncertainty to design decisions.
- **Projects, studies and a browser editor.** Save a flowsheet as a file, keep
  analyses as reproducible *studies* that know when they are out of date, and
  draw plants in the `aquakin-ui` flowsheet editor.

## Installation

Requires Python 3.11 or later.

```bash
pip install aquakin
```

Optional extras: `aquakin[plot]` (`solution.plot(...)` via matplotlib),
`aquakin[dataframe]` (export to pandas / CSV) and `aquakin[ui]` (the flowsheet
editor).

> **Note.** `import aquakin` turns on JAX 64-bit mode for the whole process,
> because stiff solves need double precision. Other JAX code in the same process
> will then use float64; run it in a separate process if it needs float32.

## Quickstart

### 1. Write a model

A model is a YAML file. Here is first-order chlorine decay:

```yaml
# decay.yaml
model:
  name: chlorine_decay
  version: "1.0"
  description: "First-order chlorine decay"

species:
  - {name: HOCl, units: mg/L, default_concentration: 2.0}

conditions:
  - {name: T, default: 293.15}

reactions:
  - name: decay
    rate: "k * [HOCl]"
    parameters:
      k: {value: 0.05, units: h-1}
    stoichiometry: {HOCl: -1}
```

### 2. Solve it, differentiate it, fit it

```python
import aquakin
import jax
import jax.numpy as jnp

model = aquakin.load_model_from_file("decay.yaml")   # or aquakin.load_model("asm1")
reactor = aquakin.BatchReactor(model, model.default_conditions())
C0 = model.default_concentrations()

# Solve over a day (the model's time unit, hours, comes from its rate constant).
sol = reactor.solve(C0, t_span=(0.0, 24.0), t_eval=jnp.linspace(0.0, 24.0, 25))
sol.final_named(["HOCl"])              # {'HOCl': 0.602}

# Differentiate the result with respect to the parameters.
def final_chlorine(params):
    return reactor.solve(C0, t_span=(0.0, 24.0), params=params).C_named("HOCl")[-1]

jax.grad(final_chlorine)(model.default_parameters())   # d[HOCl]/dk = -14.46

# Fit the rate constant to measurements, with its uncertainty.
t_obs = jnp.array([0.0, 4.0, 8.0, 12.0, 24.0])
measured = jnp.array([[2.00], [1.62], [1.36], [1.08], [0.61]])   # HOCl, mg/L
fit = aquakin.calibrate(reactor, C0, measured, t_obs, ["decay.k"],
                        loss="nll", sigma=0.02)
fit.params_named, fit.params_named_std  # k = 0.0501 ± 0.0009 h⁻¹
```

<!-- invisible-code-block: python
assert round(float(sol.final_named(["HOCl"])["HOCl"]), 3) == 0.602
assert round(float(jax.grad(final_chlorine)(model.default_parameters())[0]), 2) == -14.46
assert round(fit.params_named["decay.k"], 4) == 0.0501
assert round(fit.params_named_std["decay.k"], 4) == 0.0009
-->

Solutions carry species names and units through to results:
`sol.C_named("HOCl")`, `sol.to_dataframe()`, `sol.plot("HOCl")`.

## Plant-wide simulation

`aquakin.plant` connects reactors, clarifiers, digesters and other unit
operations into a flowsheet and integrates the whole plant in one solve. The IWA
benchmark plants are ready to build:

```python
import aquakin
from aquakin.plant.bsm import build_bsm1

model = aquakin.load_model("asm1")
plant = build_bsm1(model)            # 5 reactors + secondary clarifier + recycles
plant.add_influent("feed", model.influent(
    {"SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
     "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0}, Q=18446.0))

ss = plant.run_to_steady_state()     # integrate until the plant settles
eff = plant.stream(ss.solution, "effluent")
eff.final_named(["SNH", "SNO"])      # {'SNH': 0.47, 'SNO': 11.6} g N/m³
```

<!-- invisible-code-block: python
effluent = eff.final_named(["SNH", "SNO"])
assert round(effluent["SNH"], 2) == 0.47 and round(effluent["SNO"], 1) == 11.6
-->

From there you can:

- run **dynamic** simulations with the benchmark dry / rain / storm influents,
  a measured CSV, or an influent written as a formula of time;
- jump straight to a **differentiable steady state** (`plant.steady_state()`),
  several times faster than integrating to it;
- score **EQI / OCI**, a **carbon footprint** (N₂O, energy, fugitive methane) and
  an **operating cost**, and compare scenarios in one KPI table;
- build **BSM2** (`build_bsm2`) with primary clarification, thickening, ADM1
  digestion and dewatering, or your own flowsheet from the unit catalogue —
  including IFAS/MBBR biofilm, UV and chlorine disinfection, and aeration
  systems sized from blower and diffuser physics;
- **calibrate** plant parameters against measured streams and **rank sensor
  sets** by what they can identify, with gradients through the whole plant.

## Projects, studies and the flowsheet editor

A plant can be saved as a YAML flowsheet and rebuilt from it
(`plant.to_spec("plant.yaml")`, `aquakin.Plant.from_spec("plant.yaml")`). A
**study** stores one analysis of that flowsheet — a simulation, a calibration,
a profile likelihood, a sensor ranking — beside its results, and knows when they
are out of date:

```bash
aquakin study run studies/effluent      # writes the results
aquakin study check studies/effluent    # fresh, or stale and why
```

To draw and edit flowsheets in your browser:

```bash
pip install "aquakin[ui]"
aquakin-ui examples/projects/bsm2/flowsheet.yaml
```

The editor lets you drag units from a palette, wire their ports, edit each unit
in a form built from its documentation, see errors as you edit, and set up and
run studies with their results plotted. Colleagues who do not use Python can
install it with one download (Windows) or one pasted line (macOS, Linux): see
[Installing the editor without Python](https://aquakin.readthedocs.io/en/latest/installing_the_editor.html).

## Documentation

The full documentation is at **[aquakin.readthedocs.io](https://aquakin.readthedocs.io/en/latest/)**:

| Page | |
|---|---|
| [Getting started](https://aquakin.readthedocs.io/en/latest/getting_started.html) | installation, a first solve, building inputs, reading results |
| [Model catalogue](https://aquakin.readthedocs.io/en/latest/model_catalog.html) | every shipped model, its provenance and how to load it |
| [Your first model](https://aquakin.readthedocs.io/en/latest/adding_models.html) | writing a model of your own, step by step |
| [YAML model format](https://aquakin.readthedocs.io/en/latest/model_format.html) | every key, function and rule of a model file |
| [Reactors](https://aquakin.readthedocs.io/en/latest/reactors.html) | batch, plug-flow, biofilm and particle-track reactors; events; integrator settings |
| [Plant-wide simulation](https://aquakin.readthedocs.io/en/latest/plants.html) | the BSM1 / BSM2 benchmarks, performance indices, reporting, unit and mass-balance checks |
| [Building a flowsheet](https://aquakin.readthedocs.io/en/latest/custom_flowsheets.html) | a plant of your own and every unit operation |
| [Influents](https://aquakin.readthedocs.io/en/latest/influents.html) | patterned feeds, pump schedules, laboratory and SCADA records |
| [Flowsheet files, studies and the editor](https://aquakin.readthedocs.io/en/latest/plant_files.html) | saving a plant, reproducible studies, the browser editor |
| [Sensitivity analysis](https://aquakin.readthedocs.io/en/latest/sensitivity_and_calibration.html) | which tool answers which question, gradients, local and global sensitivity, uncertainty propagation |
| [Calibration](https://aquakin.readthedocs.io/en/latest/calibration.html) | fitting parameters to data, posteriors, profile likelihood, identifiability, model comparison |
| [Experiment design](https://aquakin.readthedocs.io/en/latest/experiment_design.html) | Fisher information, scoring and optimizing a design, choosing measurements |
| [Calibrating a plant](https://aquakin.readthedocs.io/en/latest/plant_calibration.html) | plant fits, measured channels, plant identifiability and sensor ranking |
| [Worked example: designing an experiment](https://aquakin.readthedocs.io/en/latest/worked_example_design.html) | an ozone-demand study from first design to calibrated model |
| [Public API](https://aquakin.readthedocs.io/en/latest/public_api.html) | the API grouped by task |

Runnable scripts are in [`examples/`](https://github.com/DeGrootResearchGroup/aquakin/tree/main/examples), and the
[changelog](https://github.com/DeGrootResearchGroup/aquakin/blob/main/CHANGELOG.md) lists what is new in each release.

## Development

```bash
git clone https://github.com/DeGrootResearchGroup/aquakin.git
cd aquakin
pip install -e ".[test]"
pytest -m "not validation and not slow"   # fast unit + integration tests
pytest -m validation                      # validation against published data
```

## Citing

If you use aquakin in your work, please cite it using the metadata in
[`CITATION.cff`](https://github.com/DeGrootResearchGroup/aquakin/blob/main/CITATION.cff) (DOI
[10.5281/zenodo.23784521](https://doi.org/10.5281/zenodo.23784521)).

## License

MIT — see [LICENSE](https://github.com/DeGrootResearchGroup/aquakin/blob/main/LICENSE).
