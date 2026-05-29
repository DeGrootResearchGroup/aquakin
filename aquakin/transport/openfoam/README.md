# OpenFOAM coupling

`aquakin` is decoupled from any specific flow solver. The
`transport/openfoam/` submodule is the Python side of an OpenFOAM coupling.
Both Option A (offline) and the Python seam for Option C (runtime) are
implemented here; the C++ `fvOptions` plugin for Option C lives in a
separate repository.

## Option A (implemented) — offline coupling via Lagrangian tracks

Run OpenFOAM for flow / residence-time distribution. Export, per particle, a
time series of cell-field values along the particle path. Load those tracks
into `aquakin` and integrate kinetics along each.

### CSV track format

The user-facing contract is a single CSV file:

```
particle_id,t,<field1>,<field2>,...
0,0.0,7.5,293.15,5.0e4
0,0.5,7.5,293.15,5.0e4
...
1,0.0,7.5,293.15,5.0e4
...
```

- `particle_id` (integer): groups rows by particle.
- `t` (float, seconds): time at which this sample is taken; must be strictly
  ascending within each particle.
- All remaining columns are condition field values; their names become
  `Track.fields` keys and must cover every field declared in the network's
  `conditions:` block.

Read / write through `aquakin.transport.openfoam.read_tracks_csv` and
`write_tracks_csv`, or via the package re-exports.

### Driving an offline integration

```python
import aquakin
from aquakin.transport.openfoam import read_tracks_csv

network = aquakin.load_network("ozone_bromate")
tracks = read_tracks_csv("particles.csv")

solutions = aquakin.integrate_ensemble(
    network,
    tracks,
    C0_fn=lambda pid: network.default_concentrations(),
    params=network.default_parameters(),
)

for pid, sol in solutions.items():
    print(pid, float(sol.C_named("BrO3-")[-1]))
```

## Option C (Python seam implemented) — runtime coupling via pybind11

The Python entry point is `aquakin.CFDReactor`. A C++ `fvOptions` plugin
(separate repository) embeds Python, imports `aquakin`, constructs one
`CFDReactor` per MPI rank at simulation start, and calls
`reactor.step(...)` each transport sub-step.

### The contract

```python
reactor = aquakin.CFDReactor(
    network,        # CompiledNetwork loaded via aquakin.load_network(...)
    rtol=1e-6,
    atol=1e-9,      # scalar or (n_species,) array for per-species tolerance
    adjoint=None,   # diffrax adjoint strategy; defaults to RecursiveCheckpoint
    on_nan="raise", # or "ignore"; raises with offending cell indices
)

# Once per simulation start (sanity check / fail-fast):
assert reactor.species_field_order == [...]       # what the C++ side must deliver
assert reactor.condition_field_names == [...]     # required dict keys

# Once per transport timestep:
C_new = reactor.step(
    C,             # (n_cells, n_species) float64 NumPy
    conditions,    # {name: (n_cells,) float64 NumPy}
    dt,            # scalar float (seconds)
    params,        # optional (n_params,) NumPy; defaults to network.default_parameters()
)                  # returns (n_cells, n_species) float64 NumPy
```

### What the C++ plugin must do

1. **At simulation start (once per MPI rank):**
   - Start an embedded Python interpreter via pybind11.
   - Import `aquakin`, load the network YAML, construct one `CFDReactor`.
   - Read `reactor.species_field_order` and `reactor.condition_field_names`
     to discover what `volScalarField`s the OpenFOAM case must provide.

2. **Each transport sub-step:**
   - Assemble a contiguous `(n_cells, n_species)` `double[]` buffer of
     post-transport concentrations, with columns ordered per
     `reactor.species_field_order`. The cell index is the OpenFOAM
     internal cell label.
   - Assemble a `dict[str, double[]]` of per-cell condition values, one
     `(n_cells,)` buffer per name in `reactor.condition_field_names`.
   - Call `reactor.step(C, conditions, dt)`.
   - Copy the returned `(n_cells, n_species)` array back into the
     `volScalarField`s in the same column order.

3. **At simulation end:**
   - Release the reactor reference; Python finalisation handles the rest.

### Performance notes

- The first call with a given `n_cells` triggers a JAX trace; subsequent
  calls (same mesh, varying inputs) hit the jit cache and run at native
  XLA speed.
- The cache is keyed by `n_cells`; mesh refinement mid-simulation will
  invalidate it. Acceptable in practice — refinement is rare.
- Per-species `atol` arrays (passed at construction) are the right knob
  for tracking transient radicals like OH (~10⁻¹² M) without losing them
  in the noise floor of bulk species.

### Error semantics

`reactor.step` may raise:

- `ValueError` — caller-side shape / argument errors. Should be treated as
  unrecoverable plugin bugs.
- `RuntimeError` — non-finite output from the chemistry sub-step; the
  message includes offending cell indices. The C++ side may retry with a
  smaller `dt` or abort.
- A Diffrax / Equinox runtime error — the stiff solver could not converge
  (e.g. `max_steps` reached). Same recovery semantics as the `RuntimeError`
  case.

The Python seam guarantees that any failure surfaces as an exception —
silent corruption of `volScalarField` data is not a possible outcome.

### AD-clean

`jax.grad` through the internal jit'd inner solver is well-defined; this
enables future inverse-design / parameter-fitting workflows that close the
loop through the CFD chemistry sub-step. (The NumPy boundary on `step()`
itself is opaque to JAX — for AD purposes the underlying vmapped jit is
the entry point.)

### Contract test

`tests/integration/test_cfd_fake_caller.py` exercises the full timestep
loop with NumPy inputs/outputs against the ozone/bromate network, and
asserts that one `CFDReactor.step` call is numerically identical to running
a `BatchReactor` independently on each cell. If the C++ plugin matches the
calling convention documented above, that test is sufficient to guarantee
identical chemistry behaviour.

## Operator splitting

Transport and reaction are split:

| Scale | Transport step | Reaction step |
| --- | --- | --- |
| 3D CFD (Option A) | OpenFOAM exports tracks | `ParticleTrackReactor` along each |
| 3D CFD (Option C) | OpenFOAM advection / diffusion | `CFDReactor.step` per timestep |

The reaction sub-step is pure chemistry at fixed location / along a
trajectory, so the existing `network.dCdt(...)` is sufficient. No special
CFD-aware reactor is needed beyond the vmapped wrapper.
