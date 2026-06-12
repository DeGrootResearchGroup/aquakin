# aquakin

[![CI](https://github.com/ctdegroot/aquakin/actions/workflows/ci.yml/badge.svg)](https://github.com/ctdegroot/aquakin/actions/workflows/ci.yml)

`aquakin` is a Python library for modelling reactive scalar transport in
aqueous environmental systems. Reaction networks are declared at runtime in
YAML and compiled to JAX-native, automatic-differentiable rate functions
integrated with [Diffrax](https://github.com/patrick-kidger/diffrax).

Shipped networks span chemistry (ozonation/bromate after Acero & von Gunten,
2001; UV/H₂O₂) and biology (the ASM activated-sludge family; ADM1 anaerobic
digestion in its BSM2 form, with gas headspace; the WATS sewer-process models
`wats_sewer_extended` and the paper-faithful `wats_sewer_khalil_paper`, the
latter with structural variants for model-structure studies). The network
YAML files live under `aquakin/networks/`; see `CLAUDE.md` for the full list.
Future networks include UV/TiO₂ and chlorine decay.

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
conditions = aquakin.SpatialConditions.uniform(pH=7.5, T=293.15)   # n_locations=1 default

reactor = aquakin.BatchReactor(network, conditions)

# Build the initial state by name -- no .at[species_index[...]].set() chains.
# Use a dict (species names like "Br-" aren't valid kwargs); rest = YAML defaults.
C0 = network.concentrations({"O3": 1.0e-4, "Br-": 1.0e-5})

# params is optional and defaults to network.default_parameters().
solution = reactor.solve(
    C0, t_span=(0.0, 600.0), t_eval=jnp.linspace(0.0, 600.0, 121),
)

print("[BrO3-] at t=600s:", float(solution.C_named("BrO3-")[-1]))

# Species units and descriptions are carried from the YAML to results, so you
# never have to re-derive units by string-matching names.
network.units_of("BrO3-")            # e.g. "mol/L"
network.description_of("BrO3-")
solution.units_named("BrO3-")        # same, for axis/column labels
network.summary()                    # tabulates every species with its units
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
`BiofilmReactor`. For more than one parameter it defaults to a CVODES-style
*simultaneous corrector* (`shared_factor=True`): the augmented Jacobian is
block-lower-triangular with one shared diagonal block, so that block is
factorised once per step and reused across the sensitivity columns instead of
factorising the full augmented system. This is several times faster than the
dense augmented solve on large stiff systems (e.g. the layered biofilm) and
gives bit-identical results.

## Cap-free reverse-mode gradients (stable adjoint)

`solve_sensitivity` scales with the parameter count, so for a scalar-loss
gradient over many parameters — the calibration case — reverse mode is wanted,
and that is the mode the `dtmax` cap exists for. The cap-free alternative there is
a hand-written discrete adjoint: the forward is an ordinary robust adaptive ESDIRK
(Kvaerno5) solve and the reverse is a per-step transposed solve over the saved
trajectory, finite at any step size with no cap. It is exposed as
`gradient="stable_adjoint"` on `aquakin.calibrate(...)` and, for whole plants, on
`Plant.solve(...)`:

```python
sol = plant.solve(
    t_span=(0.0, T), t_eval=t_eval, params=params, y0=y0,
    gradient="stable_adjoint",
)
g = jax.grad(loss)(params)   # finite through the stiff, coupled BSM2 plant
```

This is what lets a reverse-mode gradient flow through the whole monolithic BSM2
solve — across the ASM↔ADM interface and the recycle loops — where differentiating
*through* the stiff solve is non-finite. It is exact through a transient solve:
`plant.solve` carries the integration time in the state, so the explicit time
dependence of a time-varying influent is captured exactly in the gradient.

## Testing

```bash
pytest -m "not validation"   # unit + integration (fast)
pytest -m validation          # scientific validation against published data
pytest                        # everything
```

## License

MIT.
