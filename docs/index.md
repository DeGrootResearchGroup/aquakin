# aquakin

`aquakin` models reactive scalar transport in aqueous environmental systems.
You declare a reaction model in YAML, and `aquakin` compiles it into rate
functions that JAX can differentiate, integrated by
[Diffrax](https://github.com/patrick-kidger/diffrax).

It ships a library of ready-to-use models and the tools to work with them. The
documentation follows the order you will need it in:

1. **[Getting started](getting_started.md)** — install `aquakin`, solve a
   shipped model, and read the results.
2. **[Models](models.md)** — the shipped models (ozonation and UV/H₂O₂
   chemistry, the ASM activated-sludge family, ADM1 anaerobic digestion, WATS
   sewer processes and mineral precipitation), and how to write your own in
   YAML.
3. **[Simulation](simulation.md)** — batch, plug-flow, particle-track and
   biofilm reactors with pumps, dosing and phase switches; plant-wide
   flowsheets such as the BSM1 and BSM2 benchmarks, saved as files and analyzed
   in reproducible studies, with a browser flowsheet editor.
4. **[Analysis](analysis.md)** — every solve is differentiable, so you can
   compute sensitivities, fit parameters to data, check which parameters the
   data can determine, and choose the experiment that determines them best.
5. **[Reference](reference.md)** — the API grouped by task, and every public
   function and class.

```{toctree}
:hidden:
:maxdepth: 2

getting_started
models
simulation
analysis
reference
```

## Installation

`aquakin` requires Python 3.11 or later.

```bash
pip install aquakin
```

`import aquakin` switches JAX to 64-bit floats for the whole process; see
[Getting started](getting_started.md#installation) for what that means and for
the optional extras.

## Quickstart

Bromate formation during ozonation, in a well-mixed batch reactor at pH 8:

```python
import aquakin
import jax.numpy as jnp

model = aquakin.load_model("ozone_bromate")
conditions = model.default_conditions().with_(pH=8.0)
reactor = aquakin.BatchReactor(model, conditions)
sol = reactor.solve(
    model.default_concentrations(),
    t_span=(0.0, 600.0),                    # seconds, this model's time unit
    t_eval=jnp.linspace(0.0, 600.0, 121),
)
print(f"[BrO3-] at 10 min: {float(sol.C_named('BrO3-')[-1]):.3g} mol/L")
# [BrO3-] at 10 min: 5.7e-07 mol/L
```

[Getting started](getting_started.md) walks through each of these steps.

## How it works

`aquakin` reads a model in two stages:

1. **Loading** — the YAML file is checked against a schema, so a malformed
   model fails with an error naming the problem.
2. **Compiling** — the checked model becomes a `CompiledModel`, which holds the
   stoichiometry matrix and the compiled rate expressions. This is the object
   the reactors integrate.

`model.rates(C, params, condition_arrays, loc_idx)` returns the
`(n_reactions,)` rate vector, and `model.dCdt(...)` the right-hand side,
`stoich.T @ rates` with a guard against negative concentrations. Rate
constants always come in through `params` rather than being fixed inside the
compiled expressions, which is what lets JAX differentiate a whole solve with
respect to them for sensitivity analysis and calibration.
