# Getting started

This page takes you from installation to your first solved model and shows how
to build inputs, run a solve, and read the results. It assumes only that you
know a little Python and NumPy. The code blocks continue from one another, so
the page can be run top to bottom as one script.

## Installation

`aquakin` requires Python 3.11 or later.

```bash
pip install aquakin
```

Optional extras add convenience features:

```bash
pip install "aquakin[dataframe]"   # export solutions to pandas DataFrames / CSV
pip install "aquakin[plot]"        # solution.plot(...) via matplotlib
pip install "aquakin[ui]"          # the browser flowsheet editor
```

The flowsheet editor is described in
[Flowsheet files, studies and the editor](plant_files.md#the-flowsheet-editor). To
install the editor on a computer without Python, for a colleague who does not
use Python, see [Installing the editor without Python](installing_the_editor.md).

```{note}
**`import aquakin` enables JAX 64-bit (x64) mode process-wide.** The stiff
implicit ODE solves require double precision, so at import time aquakin runs
`jax.config.update("jax_enable_x64", True)`. This is **global** JAX state: any
other JAX code in the same process will use float64 afterward. If you are
co-running JAX code that needs float32, run aquakin in a separate process.

aquakin warns once if JAX was already imported before it, or if
`JAX_ENABLE_X64` is set to false, since either may mean another part of the
program expects float32. Import `aquakin` before `jax` (as the examples here
do) and the warning does not appear.
```

## Your first solve

Every simulation follows the same four steps: **load** a model, set the
**operating conditions**, build a **reactor**, and **solve**. Here is a complete
example — bromate formation during ozonation:

```python
import aquakin
import jax.numpy as jnp

# 1. Load a shipped model (the Model catalogue lists them all).
model = aquakin.load_model("ozone_bromate")

# 2. Operating conditions. Start from the model's declared defaults — they
#    always supply every field the model needs (here the pH and a
#    hydroxyl-radical scavenging rate) — and change only what differs with
#    .with_(...). The default pH is 7.5.
conditions = model.default_conditions().with_(pH=8.0)

# 3. A batch (0-D, well-mixed) reactor.
reactor = aquakin.BatchReactor(model, conditions)

# 4. Solve. Start from the model's declared reference concentrations and
#    integrate for 600 s, saving 121 evenly spaced points.
sol = reactor.solve(
    model.default_concentrations(),
    params=model.default_parameters(),
    t_span=(0.0, 600.0),
    t_eval=jnp.linspace(0.0, 600.0, 121),
)

print(f"[BrO3-] at 10 min: {float(sol.C_named('BrO3-')[-1]):.3g} mol/L")
# [BrO3-] at 10 min: 5.7e-07 mol/L
```

% invisible-code-block: python
%
%     assert f"{float(sol.C_named('BrO3-')[-1]):.3g}" == "5.7e-07", "the BrO3- value stated above changed"

`params` is optional — it defaults to `model.default_parameters()` — so the
minimal call is `reactor.solve(C0, t_span=(0.0, 600.0))`. Without `t_eval`
the solution holds only the final state.

To see what a model contains before you run it — its species, parameters,
conditions and literature — `print(model.summary())`; more in
[Inspecting a model](#inspecting-a-model) below.

## Setting the initial state by name

`model.default_concentrations()` returns the model's declared reference state
(for the activated-sludge models, a typical aerated mixed liquor). It is a
starting point for trying a model, not a description of your system. For a real
run you usually want to set a few species and leave the rest at their
defaults. Build the concentration vector **by name** rather than juggling array
indices. The rest of this page uses an activated-sludge model, so it builds a
new reactor for it:

```python
model = aquakin.load_model("asm1")   # Activated Sludge Model No. 1
reactor = aquakin.BatchReactor(model, model.default_conditions())

# A simple aerobic batch: biomass + substrate + ammonia + oxygen.
# Unspecified species keep their YAML default; unknown names raise a helpful
# "did you mean?" error.
C0 = model.concentrations({
    "SS": 60.0, "SNH": 25.0, "XB_H": 500.0, "XB_A": 80.0, "SO": 2.0})
```

For a **feed / influent** composition, use `base="zero"` so that species you do
not list are treated as *absent* rather than silently left at their reference
value:

```python
feed = model.concentrations({"SS": 60.0, "SNH": 25.0}, base="zero")
```

A plant's influent is built the same way, with its flow, by
`model.influent({...}, Q=...)`, which also starts from zero
([Influents](influents.md)).

You can also set parameters and per-species solver tolerances by name:

```python
params = model.parameter_values({"muH": 6.0})     # override one rate constant
atol   = model.atol({"SO": 1e-8}, default=1e-6)   # tighter tolerance on oxygen
```

## Time units

There is **no global time unit**. `t_span` and `t_eval` are in whatever unit the
model's rate constants use, and it differs by model — the ozone and UV models
are in **seconds**, the biological models (ASM, ADM, WATS) and the mineral
precipitation models in **days**. Check
before choosing an integration window:

```python
model.time_unit          # "d" for asm1; "s" for ozone_bromate / uv_h2o2
```

If you would rather work in another unit, pass `time_unit=` to `solve`: the
input times are converted to the model's native unit for the integration, and
`solution.t` comes back in the unit you asked for (with `solution.time_unit`
reporting it). Here, 48 hours of the aerobic batch:

```python
sol = reactor.solve(C0, t_span=(0.0, 48.0),
                    t_eval=jnp.linspace(0.0, 48.0, 49), time_unit="h")
sol.time_unit            # "h"
```

## Reading results

A solution carries the trajectory as `sol.t` (shape `(n_t,)`) and `sol.C`
(shape `(n_t, n_species)`), plus by-name accessors so you never index by column:

```python
sol.C_named("SNH")                 # one species' trajectory, shape (n_t,)
sol.C_named_many(["SNH", "SNO"])   # several at once -> {name: trajectory}
sol.final_named(["SNH", "SNO"])    # last-time values -> {name: float}
sol.final                          # every species' last value -> {name: float}
sol.final_named("SNH")             # a single name -> the same one-entry dict
```

Species units and descriptions are carried from the model to the solution, so
results can be labelled without looking anything up:

```python
sol.units_named("SNH")             # "g_N/m³"
model.units_of("SNH")              # the same, from the model
model.description_of("SNH")        # "Ammonia + ammonium nitrogen"
```

To read results in another unit, pass `units=`. It takes one unit for every
species read, or a `{species: unit}` dict for some of them; the rest stay in
the model's units:

```python
snh_mg = sol.C_named("SNH", units="mg/L")                  # g/m³ and mg/L agree
sol.final_named(["SS", "SNH"], units={"SS": "kgCOD/m3"})   # SS in kg/m³, SNH as declared
```

% invisible-code-block: python
%     import numpy as np
%     assert np.allclose(snh_mg, sol.C_named("SNH"))
%     _f = sol.final_named(["SS", "SNH"], units={"SS": "kgCOD/m3"})
%     assert np.isclose(_f["SS"], 1e-3 * sol.final["SS"])
%     assert np.isclose(_f["SNH"], sol.final["SNH"])

Only the scale changes, so a unit of another quantity (moles where the model
has grams, g N where it has g COD) raises. `C_named_many`, `to_dataframe` and
`to_csv` take `units=` too, as do a plant solution's readers and a plant
stream's (`sol.C_named("tank5", "SNH", units="mg/L")`).

With the optional extras installed you can export or plot directly:

```python
df = sol.to_dataframe()            # time-indexed pandas DataFrame  [dataframe extra]
sol.to_csv("run.csv")              # units embedded in the header   [dataframe extra]
ax = sol.plot(["SNH", "SNO"])      # matplotlib Axes with a legend  [plot extra]
```

The plot labels the time axis with its unit and gives each species a legend
entry; plot a single species and the vertical axis carries its unit too.

## Inspecting a model

Before running an unfamiliar model, inspect what it expects:

```python
print(model.summary())      # species, reactions, parameters and references
model.species               # ordered species names (the columns of sol.C)
model.parameters            # ordered parameter names (the entries of params)
model.conditions_required   # condition fields the model needs: ["T"] for asm1
model.references            # literature the model is built from
```

Two **opt-in, advisory** checks help validate a model. Each returns a list of
findings rather than raising on one:

```python
model.check_units()          # dimensional consistency of the rate expressions: []
model.check_conservation()   # mass / electron balance of the stoichiometry
```

`check_units()` checks that every rate expression has consistent dimensions,
and that a rate is in the concentration-per-time units of a species whose
stoichiometric coefficient is +1 or −1 (turn that part off with
`check_units(check_stoichiometry=False)`).

`check_conservation()` needs a per-species `composition:` table and raises
`ValueError` for a model without one (`ozone_bromate` has none). A finding is
not always a mistake: for ASM1 it reports a COD and a nitrogen residual on
anoxic heterotroph growth, because the N₂ gas that denitrification produces is
not one of the model's species.

## Where to go next

- [Model catalogue](model_catalog.md) — every shipped model, to find the one
  for your process.
- [Your first model](adding_models.md) and the
  [YAML model format](model_format.md) — write your own model, including the
  built-in math functions (`exp`, `log`, `sqrt`, `min`, ...) and rate functions
  of your own registered with `aquakin.register_rate_function`.
- [Reactors](reactors.md) — batch, plug-flow, particle-track, and biofilm
  reactors; operating conditions; discontinuous events; and the integrator
  settings.
- [Plant-wide simulation](plants.md) — run the BSM1 and BSM2 benchmarks, and
  [build a flowsheet](custom_flowsheets.md) of your own.
- [Sensitivity analysis](sensitivity_and_calibration.md),
  [Calibration](calibration.md) and [Experiment design](experiment_design.md) —
  differentiate a solve, see which parameters matter, fit them to data, and
  plan the experiment that determines them best.
