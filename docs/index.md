# aquakin documentation

`aquakin` models reactive scalar transport in aqueous environmental systems.
Reaction models are declared at runtime in YAML, parsed into an AST, and
compiled to JAX-native rate functions integrated by Diffrax.

## Contents

- [Model file format](model_format.md) — schema for YAML model files.
- [Adding models](adding_models.md) — how to ship a new built-in model.

## Quickstart

```python
import jax.numpy as jnp
import aquakin

model = aquakin.load_model("ozone_bromate")
conditions = aquakin.OperatingConditions(pH=7.5, T=293.15)   # 0-D batch case
reactor = aquakin.BatchReactor(model, conditions)
sol = reactor.solve(
    model.default_concentrations(),
    model.default_parameters(),
    t_span=(0.0, 600.0),
    t_eval=jnp.linspace(0.0, 600.0, 121),
)
print("[BrO3-] at 10 min:", float(sol.C_named("BrO3-")[-1]))
```

## Architecture

Two-layer data model:

1. **Schema layer (load time)** — Pydantic models in `aquakin.schema`. Validates
   YAML, produces a clean spec object. No Pydantic dependency on the hot path.
2. **Runtime layer** — `CompiledModel` dataclass in `aquakin.core.model`,
   built once via `compile_model(spec)`. Holds the stoichiometry matrix, the
   per-reaction compiled rate callables, and the parameter index map.

The rate callable signature is

```python
rates(C, params, condition_arrays, loc_idx) -> jnp.ndarray  # shape (n_reactions,)
```

and the chemistry RHS is `stoich.T @ rates(...)`.
