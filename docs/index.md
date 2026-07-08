# aquakin

`aquakin` models reactive scalar transport in aqueous environmental systems.
Reaction models are declared at runtime in YAML, parsed into an AST, and
compiled to JAX-native, automatically-differentiable rate functions integrated
by [Diffrax](https://github.com/patrick-kidger/diffrax).

It ships a library of ready-to-use models — ozonation and UV/H₂O₂ chemistry,
the ASM activated-sludge family, ADM1 anaerobic digestion, WATS sewer processes,
and mineral precipitation — and the tools to simulate them in batch, plug-flow,
biofilm, and full plant-wide flowsheets, with automatic differentiation
throughout for sensitivity analysis and parameter calibration.

New here? Start with [Getting started](getting_started.md).

```{toctree}
:maxdepth: 2
:caption: Guide

getting_started
reactors
plants
sensitivity_and_calibration
```

```{toctree}
:maxdepth: 2
:caption: Authoring models

model_format
adding_models
```

```{toctree}
:maxdepth: 2
:caption: Reference

model_catalog
public_api
api
```

## Installation

```bash
pip install aquakin
```

```{note}
`import aquakin` enables JAX 64-bit (x64) mode process-wide — the stiff implicit
ODE solves require double precision. This is global JAX state: other JAX code in
the same process will use float64 afterward. aquakin emits a one-time warning if
it overrides an explicit float32 preference, so the side effect is never silent.
```

## Quickstart

```python
import jax.numpy as jnp
import aquakin

model = aquakin.load_model("ozone_bromate")
conditions = model.default_conditions().with_(pH=7.5, T=293.15)   # 0-D batch case
reactor = aquakin.BatchReactor(model, conditions)
sol = reactor.solve(
    model.default_concentrations(),
    params=model.default_parameters(),
    t_span=(0.0, 600.0),
    t_eval=jnp.linspace(0.0, 600.0, 121),
)
print("[BrO3-] at 10 min:", float(sol.C_named("BrO3-")[-1]))
```

See [Getting started](getting_started.md) for a step-by-step walkthrough.

## Architecture

`aquakin` uses a two-layer data model:

1. **Schema layer (load time)** — Pydantic models validate the YAML and produce
   a clean spec object. Pydantic never appears on the hot path.
2. **Runtime layer** — a `CompiledModel` dataclass, built once from the spec,
   holds the stoichiometry matrix, the per-reaction compiled rate callables, and
   the parameter index map. This is what the integrators operate on.

Each rate callable has the signature `rates(C, params, condition_arrays,
loc_idx)` returning a `(n_reactions,)` vector, and the reaction right-hand side
is `stoich.T @ rates(...)`. Rate constants are always passed in via `params`,
never baked in — which is what makes the whole solve differentiable for
sensitivity analysis and calibration.
