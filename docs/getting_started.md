# Getting started

This page takes you from installation to your first solved model and shows how
to build inputs, run a solve, and read the results. It assumes only that you
know a little Python and NumPy.

## Installation

```bash
pip install aquakin
```

Two optional extras unlock result convenience features:

```bash
pip install "aquakin[dataframe]"   # export solutions to pandas DataFrames / CSV
pip install "aquakin[plot]"        # solution.plot(...) via matplotlib
```

```{note}
**`import aquakin` enables JAX 64-bit (x64) mode process-wide.** The stiff
implicit ODE solves require double precision, so at import time aquakin runs
`jax.config.update("jax_enable_x64", True)`. This is **global** JAX state: any
other JAX code in the same process will use float64 afterward. If you are
co-running JAX code that needs float32, run aquakin in a separate process.
aquakin emits a one-time warning if it overrides an explicit float32
preference, so the side effect is never silent.
```

## Your first solve

Every simulation follows the same four steps: **load** a model, set the
**operating conditions**, build a **reactor**, and **solve**. Here is a complete
example — bromate formation during ozonation:

```python
import jax.numpy as jnp
import aquakin

# 1. Load a shipped model (see the Model catalog for the full list).
model = aquakin.load_model("ozone_bromate")

# 2. Operating conditions. Start from the model's declared defaults — this
#    always supplies every field the model needs (here pH, T, and a hydroxyl-
#    radical scavenging rate) — and change only what differs with .with_(...).
conditions = model.default_conditions().with_(pH=7.5, T=293.15)   # 20 C

# 3. A batch (0-D) reactor.
reactor = aquakin.BatchReactor(model, conditions)

# 4. Solve. Start from the model's declared reference concentrations and
#    integrate for 600 s, saving 121 evenly spaced points.
sol = reactor.solve(
    model.default_concentrations(),
    params=model.default_parameters(),
    t_span=(0.0, 600.0),
    t_eval=jnp.linspace(0.0, 600.0, 121),
)

print("[BrO3-] at 10 min:", float(sol.C_named("BrO3-")[-1]))
```

`params` is optional — it defaults to `model.default_parameters()` — so the
minimal call is `reactor.solve(C0, t_span=(0.0, 600.0))`.

## Setting the initial state by name

`model.default_concentrations()` returns the model's declared reference state.
For a real run you usually want to set a few species and leave the rest at their
defaults. Build the concentration vector **by name** rather than juggling array
indices:

```python
model = aquakin.load_model("asm1")   # Activated Sludge Model No. 1

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

You can also set parameters and per-species solver tolerances by name:

```python
params = model.parameter_values({"muH": 6.0})     # override one rate constant
atol   = model.atol({"SO": 1e-8}, default=1e-6)   # tighter tolerance on oxygen
```

## Time units

There is **no global time unit**. `t_span` and `t_eval` are in whatever unit the
model's rate constants use, and it differs by model — the ozone and UV models
are in **seconds**, the biological models (ASM, ADM, WATS) in **days**. Check
before choosing an integration window:

```python
model.time_unit          # "d" for asm1; "s" for ozone_bromate / uv_h2o2
```

If you would rather work in another unit, pass `time_unit=` to `solve`: the
input times are converted to the model's native unit for the integration, and
`solution.t` comes back in the unit you asked for (with `solution.time_unit`
reporting it):

```python
sol = reactor.solve(C0, t_span=(0.0, 48.0),
                    t_eval=jnp.linspace(0.0, 48.0, 49), time_unit="h")
```

## Reading results

A solution carries the trajectory as `sol.t` (shape `(n_t,)`) and `sol.C`
(shape `(n_t, n_species)`), plus by-name accessors so you never index by column:

```python
sol.C_named("SNH")                 # one species' trajectory, shape (n_t,)
sol.C_named_many(["SNH", "SNO"])   # several at once -> {name: trajectory}
sol.final_named(["SNH", "SNO"])    # last-time values -> {name: float}
sol.final                          # every species' last value
```

Species units and descriptions are carried from the model to the solution, so
labels are automatic:

```python
model.units_of("SNH")              # e.g. "g_N/m³"
model.description_of("SNH")        # "Ammonia + ammonium nitrogen"
```

With the optional extras installed you can export or plot directly:

```python
df = sol.to_dataframe()            # time-indexed pandas DataFrame  [dataframe extra]
sol.to_csv("run.csv")              # units embedded in the header
ax = sol.plot(["SNH", "SNO"])      # matplotlib Axes, axes auto-labelled  [plot extra]
```

## Inspecting a model

Before running an unfamiliar model, inspect what it expects:

```python
model.summary()             # human-readable table: species, reactions, params, refs
model.species               # ordered species names (the columns of sol.C)
model.parameters            # ordered, namespaced parameter names
model.conditions_required   # condition fields the model needs (e.g. "pH", "T")
model.references            # literature the model is built from
```

Two **opt-in, advisory** checks help validate a model (they never raise; they
return a list of findings):

```python
model.check_units()          # dimensional consistency of the rate expressions
model.check_conservation()   # mass / electron balance of the stoichiometry
```

## Where to go next

- [Reactors](reactors.md) — batch, plug-flow, particle-track, and biofilm
  reactors; operating conditions; and discontinuous events.
- [Plant-wide simulation](plants.md) — assemble full treatment-plant flowsheets
  such as the BSM1 and BSM2 benchmarks.
- [Sensitivity & calibration](sensitivity_and_calibration.md) — differentiate a
  solve, run sensitivity analysis, and fit parameters to data.
- [Model catalog](model_catalog.md) — every shipped model.
- [Model file format](model_format.md) — write your own model in YAML.
