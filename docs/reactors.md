# Reactors

A **reactor** couples a compiled model to a spatial context and integrates it.
`aquakin` ships reactors for the well-mixed batch (0-D), plug flow (1-D),
Lagrangian particle tracks, and a depth-resolved biofilm. They share one
contract:

- Construct once from a model and its spatial context: operating conditions
  for the batch, plug-flow and biofilm reactors, a particle `Track` for the
  particle-track reactor.
- Reactors are **stateless after construction** — `solve()` takes every variable
  input (initial state, parameters, time span) as an argument. This is what lets
  you `jax.vmap` a reactor over an ensemble of initial conditions or parameters,
  and it is why the same reactor object can be reused across many solves.
- `solve()` returns a solution object with a common set of by-name accessors.

The examples on this page continue from one another. They start from the
ozonation model used in [Getting started](getting_started.md):

```python
import aquakin
import jax.numpy as jnp

model = aquakin.load_model("ozone_bromate")   # time unit: seconds
C0 = model.default_concentrations()
params = model.default_parameters()
t_eval = jnp.linspace(0.0, 600.0, 121)
```

## Operating conditions

Reactions depend on operating conditions — pH, temperature, an irradiation
rate, a scavenging term — that a model declares in its `conditions:` block. You
supply them as a conditions object.

The robust way is to start from the model's declared defaults, which always
include every field the model requires, and edit what differs:

```python
conditions = model.default_conditions()                    # all declared defaults
conditions = model.default_conditions().with_(pH=8.0)      # ...run it at pH 8
```

For a simple case you can also build one from scratch with
`OperatingConditions`, the 0-D (single-tank) shorthand — but then you must
supply **every** field the model needs, or building a reactor from it raises a
`ValueError` naming the missing ones:

```python
conditions = aquakin.OperatingConditions(pH=7.5, OH_scavenging=5.0e4)
```

Both are single-location conditions. For a spatially varying case (a plug-flow
reactor or a CFD field) use `SpatialConditions`, which holds one array per field
over `n_locations`:

```python
# The same value at every one of 101 locations:
uniform = aquakin.SpatialConditions.uniform(
    n_locations=101, pH=7.5, OH_scavenging=5.0e4)

# A pH that rises from 7.0 to 8.0 along those locations:
gradient = aquakin.SpatialConditions(fields={
    "pH": jnp.linspace(7.0, 8.0, 101),
    "OH_scavenging": jnp.full(101, 5.0e4),
})
```

`OperatingConditions` *is* a one-location `SpatialConditions`, so it works
unchanged in every reactor that takes conditions; a reactor with several
locations applies a one-location value at all of them.

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
  strategy. See [Choosing the integrator](#choosing-the-integrator).
- `diff=DifferentiationConfig(...)` — how a solve under `jax.grad` is
  differentiated. For a stiff model (ASM, ADM, WATS) a reverse-mode gradient
  through a reactor solve needs a step cap; see
  [Differentiating a solve](sensitivity_and_calibration.md#differentiating-a-solve).

## Plug-flow reactor (1-D)

The plug-flow reactor integrates the steady-state concentration profile along a
reactor of a given length and flow velocity. The independent axis is position,
not time. `length` and `velocity` share a length unit, and `velocity` is per
the model's time unit (here metres and metres per second):

```python
reactor = aquakin.PlugFlowReactor(model, gradient,
                                  n_points=101, length=10.0, velocity=0.05)
sol = reactor.solve(C0, params=params)
sol.x        # (n_points,) axial positions, 0 to 10 m
sol.C        # (n_points, n_species) profile
```

The conditions passed here are the pH gradient built above: a
`SpatialConditions` with `n_points` locations describes a temperature or pH
that changes down the channel.

## Biofilm reactor (1-D over depth)

The biofilm reactor resolves a biofilm into `n_layers` between a well-mixed bulk
and a no-flux wall, so **penetration-controlled** processes are captured — an
electron acceptor consumed in the outer layers never reaches the deep organisms,
and deep uptake is diffusion-limited. A lumped area-to-volume reactor cannot
represent this. Solubles diffuse (Fick's law with an effective diffusivity) and
exchange with the bulk across a boundary layer; particulates stay in the layer
they are in.

By default the reactor also **freezes** every particulate: its reactions are
switched off, and it warns when that includes reactive biomass. Pass
`fixed_mask`, marking only the genuinely inert solids, so the biomass can grow
and decay. This example uses ASM1, whose time unit is days:

```python
asm1 = aquakin.load_model("asm1")
reactor = aquakin.BiofilmReactor(
    asm1, asm1.default_conditions(),
    n_layers=6, thickness=8e-4, area_per_volume=50.0,   # m, m²/m³
    diffusivity=1e-4, boundary_layer=1e-4,              # m²/d, m
    fixed_mask=jnp.array([s == "XI" for s in asm1.species]),
)
C0_bulk = asm1.concentrations({"SS": 60.0, "SNH": 25.0, "SO": 2.0})
sol = reactor.solve(C0_bulk, (0.0, 1.0), jnp.linspace(0.0, 1.0, 11))
sol.C                    # (n_t, n_species) — the bulk (measurable) trajectory
sol.profile              # (n_t, n_layers+1, n_species) — depth-resolved (0 = bulk)
sol.depth                # (n_layers,) layer mid-depths from the surface
sol.profile_named("SNO") # (n_t, n_layers+1) one species' depth profile over time
```

`solve` also takes `time_unit=`, as the batch reactor does. The reactor
supports attachment/detachment, a bulk feed (CSTR-style) and biomass-density
limits, and `reactor.steady_state(...)` matures a multispecies biofilm to its
operating state — see the [API reference](api.md) for `BiofilmReactor`.

## Particle-track and CFD reactors

- `ParticleTrackReactor(model, track)` integrates the chemistry along a single
  Lagrangian particle `Track` (e.g. a pathline exported from a flow solver), for
  offline coupling to a CFD residence-time field.
- `CFDReactor(model)` is a vectorized batch reactor used as the reaction
  operator in a transport/reaction operator split at the cell level.

Both are documented in the [API reference](api.md).

## Solutions

Every single-vector solution (batch, plug-flow, particle-track, biofilm bulk)
shares the by-name accessors listed in
[Reading results](getting_started.md#reading-results): `C_named`,
`C_named_many`, `final_named`, `final`, `units_named`, `to_dataframe`, `to_csv`
and `plot`; the readers take `units=` to return species in another unit. The
independent axis is `sol.t` for a solve in time and `sol.x`
for the plug-flow reactor's position.

```{note}
`final_named` and `final` return plain Python floats for reporting. For a
*differentiable* final value (inside a loss or sensitivity function) use
`sol.C_named("BrO3-")[-1]`, which stays a JAX array.
```

## Events: pumps, dosing, phase switches

Discontinuous operations — on/off pumps, dosing, SBR phases, level limits — are
handled with **located events**, which `BatchReactor.solve` and the plant's
`Plant.solve` accept as `events=`. An event has a trigger and an optional state
reset:

- `at_times=[...]` fires at known times. This is **AD-safe**: the segment
  boundaries are static, so `jax.grad` stays finite through the solve, and the
  whole solve can be wrapped in `jax.jit`.
- `cond_fn=lambda t, C, p: ...` fires when the scalar condition crosses zero (in
  an optional `direction`: `-1` for downward only, `+1` for upward only). This
  is located by a root find, so it is forward-simulation only.

Set `terminal=True` to stop the solve when the event fires, and `apply=...` to
reset the state. Event times (`at_times`, the `t` passed to `cond_fn` and
`apply`, and the times in `events_log`) use the same unit as `t_span`, so they
follow `time_unit=` when you pass it. An output time that coincides with an
event reports the state just before the reset. The solve between events is
compiled once and reused, so solving again with the same settings (new
parameters or output times) does not compile anything.

This example doses external carbon into an anoxic ASM1 batch and stops once the
nitrate is used up:

```python
reactor = aquakin.BatchReactor(asm1, asm1.default_conditions())
C0_anoxic = asm1.concentrations({"SS": 5.0, "SNO": 20.0, "XB_H": 500.0, "SO": 0.0})
i_ss  = asm1.species_index["SS"]
i_sno = asm1.species_index["SNO"]

events = [
    # Dose 60 g COD/m³ of external carbon at t = 0.1 d.
    aquakin.Event(at_times=[0.1],
                  apply=lambda t, C, p: C.at[i_ss].add(60.0)),
    # Stop once nitrate falls below 0.5 g N/m³.
    aquakin.Event(cond_fn=lambda t, C, p: C[i_sno] - 0.5,
                  direction=-1, terminal=True, name="denitrified"),
]

sol = reactor.solve(C0_anoxic, t_span=(0.0, 0.5),
                    t_eval=jnp.linspace(0.0, 0.5, 101), events=events)
sol.events_log     # audit trail: [(0.1, 'event0'), (0.208..., 'denitrified')]
```

After a terminal event, the remaining `t_eval` points repeat the state the
solve stopped in, so `sol.t` still runs to 0.5 here. With `t_eval=None` the
solution is the single point where the solve stopped.

## Choosing the integrator

The integrator is configured with one value object, `aquakin.IntegratorConfig`,
passed as `integrator=` to a reactor or to `plant.solve`. Its defaults are:

```python
aquakin.IntegratorConfig(order=3, factormax=3.0, colored_jacobian="auto",
                         dtmax=None, max_steps=100_000, solver=None)
```

- `order`: the ESDIRK method, `3` (`Kvaerno3`, the default) or `5` (`Kvaerno5`,
  higher order and more robust). Raise it to 5 if a solve struggles to
  converge.
- `factormax`: caps how fast the step size may grow from one step to the next.
- `dtmax`: caps the step. Set it **only** for a reverse-mode gradient taken
  *through* a stiff solve (`method="through_solve"`); it is in the model's
  time unit. The cap-free gradient methods in
  [Differentiating a solve](sensitivity_and_calibration.md#differentiating-a-solve)
  do not need it.
- `colored_jacobian`: builds the implicit-step Jacobian sparsely. It applies to
  plants only; reactors ignore it.
- `max_steps`: the step budget; `solver` overrides `order` with an explicit
  Diffrax solver.

`rtol` and `atol` stay separate arguments of the reactor and of `plant.solve`.
For faster plant solves, including `forward_fast`, see
[Plant-wide simulation](plants.md#faster-dynamic-solves).
