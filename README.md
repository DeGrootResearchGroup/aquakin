# aquakin

[![CI](https://github.com/ctdegroot/aquakin/actions/workflows/ci.yml/badge.svg)](https://github.com/ctdegroot/aquakin/actions/workflows/ci.yml)

`aquakin` is a Python library for modelling reactive scalar transport in
aqueous environmental systems. Reaction networks are declared at runtime in
YAML and compiled to JAX-native, automatic-differentiable rate functions
integrated with [Diffrax](https://github.com/patrick-kidger/diffrax).

Shipped networks span chemistry (ozonation/bromate after Acero & von Gunten,
2001; UV/H₂O₂) and biology (the ASM activated-sludge family; the WATS
sewer-process models `wats_sewer_extended` and the paper-faithful `wats_sewer_khalil_paper`,
the latter with structural variants for model-structure studies). The network
YAML files live under `aquakin/networks/`; see `CLAUDE.md` for the full list.
Future networks include UV/TiO₂, chlorine decay, and ADM1.

## Features

- Reaction networks declared in YAML — no recompilation required.
- Full automatic differentiation through `solve()` via JAX.
- JAX-native stiff ODE integration via Diffrax (`Kvaerno5` by default).
- Safe rate expression evaluation via a custom AST (no `eval()`).
- Decoupled transport / reaction operator splitting at all scales (0D, 1D, 3D).
- Reactors for batch (0D), plug flow (1D), Lagrangian particle tracks, and a
  layered biofilm (`BiofilmReactor`: 1-D diffusion-reaction over biofilm depth,
  for penetration-controlled processes).
- Forward (variational) sensitivity solve (`solve_sensitivity` /
  `forward_sensitivity`): integrate `dC/dθ` alongside the state, giving exact
  parameter sensitivities of stiff networks with no integrator-step cap.

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

## Forward sensitivity (cap-free stiff gradients)

Differentiating *through* a stiff reaction-network solve with ordinary AD goes
non-finite above an integrator-step threshold, and the usual workaround — a
global `dtmax` cap — forces tiny steps over the whole solve. `solve_sensitivity`
avoids both: it integrates the sensitivity `S = dC/dθ` *alongside* the state and
lets the adaptive step controller bound the sensitivity error too, so the step
tightens only where the sensitivity is stiff and the result is exact with no cap.

```python
import jax.numpy as jnp
import aquakin

network = aquakin.load_network("uv_h2o2")
conditions = network.default_conditions(1)
reactor = aquakin.BatchReactor(network, conditions)

C0 = network.default_concentrations()
params = network.default_parameters()
t_eval = jnp.linspace(0.0, 5.0, 6)

sol, S = reactor.solve_sensitivity(
    C0, params, t_span=(0.0, 5.0), t_eval=t_eval,
    sens_params=["H2O2_photolysis.k_photo", "OH_target.k_OH_target"],
)
# sol : the usual solution; S : dC/dθ, shape (n_t, n_species, n_sens_params)

# A richer wrapper with by-name accessors:
res = aquakin.forward_sensitivity(
    reactor, C0, params, t_span=(0.0, 5.0), t_eval=t_eval,
    sens_params=["H2O2_photolysis.k_photo"],
)
res.dC_dparam("target", "H2O2_photolysis.k_photo")   # (n_t,)
```

`solve_sensitivity` is available on `BatchReactor`, `PlugFlowReactor` and
`BiofilmReactor`.

## Testing

```bash
pytest -m "not validation"   # unit + integration (fast)
pytest -m validation          # scientific validation against published data
pytest                        # everything
```

## License

MIT.
