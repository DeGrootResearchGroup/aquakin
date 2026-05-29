# aquakin

`aquakin` is a Python library for modelling reactive scalar transport in
aqueous environmental systems. Reaction networks are declared at runtime in
YAML and compiled to JAX-native, automatic-differentiable rate functions
integrated with [Diffrax](https://github.com/patrick-kidger/diffrax).

The first demonstration network is ozone/bromate formation after Acero & von
Gunten (2001). Future networks include UV/H₂O₂ and chlorine decay.

## Features

- Reaction networks declared in YAML — no recompilation required.
- Full automatic differentiation through `solve()` via JAX.
- JAX-native stiff ODE integration via Diffrax (`Kvaerno5` by default).
- Safe rate expression evaluation via a custom AST (no `eval()`).
- Decoupled transport / reaction operator splitting at all scales (0D, 1D, 3D).

## Installation

```bash
pip install -e ".[test]"
```

`aquakin` enables JAX 64-bit mode automatically at import — stiff ODE
integration requires it.

## Quickstart

```python
import jax.numpy as jnp
import aquakin

network = aquakin.load_network("ozone_bromate")
conditions = aquakin.SpatialConditions.uniform(n_locations=1, pH=7.5, T=293.15)

reactor = aquakin.BatchReactor(network, conditions)
solution = reactor.solve(
    network.default_concentrations(),
    network.default_parameters(),
    t_span=(0.0, 600.0),
    t_eval=jnp.linspace(0.0, 600.0, 121),
)

print("[BrO3-] at t=600s:", float(solution.C_named("BrO3-")[-1]))
```

## Testing

```bash
pytest -m "not validation"   # unit + integration (fast)
pytest -m validation          # scientific validation against published data
pytest                        # everything
```

## License

MIT.
