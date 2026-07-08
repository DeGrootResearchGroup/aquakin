# Reactors

A **reactor** couples a compiled model to a spatial context and integrates it.
`aquakin` ships reactors for the well-mixed batch (0-D), plug flow (1-D),
Lagrangian particle tracks, and a depth-resolved biofilm. They share one
contract:

- Construct once from a model and its operating conditions.
- Reactors are **stateless after construction** — `solve()` takes every variable
  input (initial state, parameters, time span) as an argument. This is what lets
  you `jax.vmap` a reactor over an ensemble of initial conditions or parameters,
  and it is why the same reactor object can be reused across many solves.
- `solve()` returns a solution object with a common set of by-name accessors.

## Operating conditions

Reactions depend on operating conditions — pH, temperature, an irradiation
rate, a scavenging term — that a model declares in its `conditions:` block. You
supply them as a conditions object.

The robust way is to start from the model's declared defaults, which always
include every field the model requires, and edit what differs:

```python
conditions = model.default_conditions()                    # all declared defaults
conditions = model.default_conditions().with_(T=283.15)    # ...run it at 10 C
```

For a simple case you can also build one from scratch with
`OperatingConditions`, the 0-D (single-tank) shorthand — but then you must
supply **every** field the model needs:

```python
conditions = aquakin.OperatingConditions(pH=7.5, T=293.15)
```

Both are single-location conditions. For a spatially varying case (a plug-flow
reactor or a CFD field) use `SpatialConditions`, which holds one array per field
over `n_locations`:

```python
conditions = aquakin.SpatialConditions.uniform(pH=7.5, T=293.15)          # constant
conditions = aquakin.SpatialConditions(fields={"pH": jnp.array([...]),    # varying
                                               "T": jnp.array([...])})
```

`OperatingConditions` *is* a one-location `SpatialConditions`, so it works
unchanged in every reactor.

## Batch reactor (0-D)

The batch reactor integrates a single well-mixed volume in time — the workhorse
for kinetics, batch experiments, and calibration:

```python
reactor = aquakin.BatchReactor(model, conditions)
sol = reactor.solve(C0, t_span=(0.0, 600.0), t_eval=t_eval)   # params optional
```

Useful construction options (all keyword-only):

- `rtol`, `atol` — solver tolerances. `atol` may be a per-species array (use
  `model.atol({...})` to build one) when trace intermediates sit orders of
  magnitude below the bulk species.
- `integrator=IntegratorConfig(...)` — the ESDIRK order, step caps, and Jacobian
  strategy. See [Choosing the integrator](sensitivity_and_calibration.md#choosing-the-integrator).
- `diff=DifferentiationConfig(...)` — how a solve under `jax.grad` is
  differentiated. The default is finite through stiff models; see
  [Sensitivity & calibration](sensitivity_and_calibration.md).

## Plug-flow reactor (1-D)

The plug-flow reactor integrates the steady-state concentration profile along a
reactor of a given length and flow velocity. The independent axis is position,
not time:

```python
reactor = aquakin.PlugFlowReactor(model, conditions,
                                  n_points=101, length=10.0, velocity=0.05)
sol = reactor.solve(C0, params=params)
sol.x        # (n_points,) axial positions
sol.C        # (n_points, n_species) profile
```

Because conditions can vary along the reactor, pass a `SpatialConditions` with
`n_points` locations to model, for example, a temperature or pH gradient down
the channel.

## Biofilm reactor (1-D over depth)

The biofilm reactor resolves a biofilm into `n_layers` between a well-mixed bulk
and a no-flux wall, so **penetration-controlled** processes are captured — an
electron acceptor consumed in the outer layers never reaches the deep organisms,
and deep uptake is diffusion-limited. A lumped area-to-volume reactor cannot
represent this. Solubles diffuse (Fick's law with an effective diffusivity) and
exchange with the bulk across a boundary layer; particulates are held in place.

```python
reactor = aquakin.BiofilmReactor(
    model, conditions,
    n_layers=6, thickness=8e-4, area_per_volume=50.0,
    diffusivity=1e-4, boundary_layer=1e-4,
)
sol = reactor.solve(C0, t_span, t_eval, params=params)
sol.C                    # (n_t, n_species) — the bulk (measurable) trajectory
sol.profile              # (n_t, n_layers+1, n_species) — depth-resolved (0 = bulk)
sol.depth                # (n_layers,) layer mid-depths from the surface
sol.profile_named("S_NO")  # (n_t, n_layers+1) one species' depth profile over time
```

The reactor also supports attachment/detachment, a bulk feed (CSTR-style),
biomass-density limits, and a steady-state solve for maturing a multispecies
biofilm to its operating state — see the API reference for `BiofilmReactor`.

## Particle-track and CFD reactors

- `ParticleTrackReactor` integrates the chemistry along a single Lagrangian
  particle `Track` (e.g. a pathline exported from a flow solver), for
  offline coupling to a CFD residence-time field.
- `CFDReactor` is a vectorised batch reactor used as the reaction operator in a
  transport/reaction operator split at the cell level.

Both are documented in the [API reference](api.md).

## Solutions

Every single-vector solution (batch, plug-flow, particle-track, biofilm bulk)
shares the same accessors, so you read results by name rather than by column
index:

```python
sol.t / sol.x                      # independent axis (time or position)
sol.C                              # (n, n_species) trajectory / profile
sol.C_named("SNH")                 # one species
sol.C_named_many(["SNH", "SNO"])   # several -> {name: array}
sol.final_named(["SNH"])           # last-point values -> {name: float}
sol.to_dataframe()                 # pandas DataFrame          [dataframe extra]
sol.plot(["SNH", "SNO"])           # matplotlib Axes           [plot extra]
```

```{note}
`final_named` and `final` return plain Python floats for reporting. For a
*differentiable* final value (inside a loss or sensitivity function) use
`sol.C_named("SNH")[-1]`, which stays a JAX array.
```

## Events: pumps, dosing, phase switches

Discontinuous operations — on/off pumps, dosing, SBR phases, level limits — are
handled with **located events**, available on every reactor and the plant. An
event has a trigger and an optional state reset:

- `at_times=[...]` fires at known times. This is **AD-safe**: the segment
  boundaries are static, so `jax.grad` stays finite through the solve.
- `cond_fn=lambda t, C, p: ...` fires when the scalar condition crosses zero (in
  an optional `direction`). This is located by a root find, so it is
  forward-simulation only.

Set `terminal=True` to stop the solve when the event fires, and `apply=...` to
reset the state:

```python
i_ss  = model.species_index["SS"]
i_sno = model.species_index["SNO"]

events = [
    # Dose external carbon at t = 0.1 d to drive denitrification.
    aquakin.Event(at_times=[0.1],
                  apply=lambda t, C, p: C.at[i_ss].add(60.0)),
    # Stop once nitrate is essentially gone.
    aquakin.Event(cond_fn=lambda t, C, p: C[i_sno] - 0.5,
                  direction=-1, terminal=True, name="denitrified"),
]

sol = reactor.solve(C0, t_span=(0.0, 0.5),
                    t_eval=jnp.linspace(0.0, 0.5, 101), events=events)
sol.events_log     # audit trail: [(0.1, 'event0'), (~0.13, 'denitrified')]
```
