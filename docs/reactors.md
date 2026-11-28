# Reactors

A **reactor** couples a compiled model to a spatial context and integrates it.
`aquakin` ships reactors for the well-mixed batch (0-D), plug flow (1-D),
dispersed flow and tanks in series (non-ideal 1-D), Lagrangian particle
tracks, and a depth-resolved biofilm. They share one
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
conditions = aquakin.OperatingConditions(pH=7.5, OH_scavenging=5.0e4)   # OH_scavenging in 1/s
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

`t_eval` sets the times the solution keeps. Without it the solution holds only
the end of the solve, which is enough for a final value:

```python
end = reactor.solve(C0, t_span=(0.0, 600.0))
end.t        # [600.]
```

% invisible-code-block: python
%     assert end.t.tolist() == [600.0]

Every reactor takes its solver configuration as one keyword-only value,
`settings=aquakin.SolverSettings(...)`, with four fields:

- `rtol`, `atol` — solver tolerances (defaults `1e-6` and a per-species floor
  scaled to the model's concentrations). `atol` may be a per-species array (use
  `model.atol({...})` to build one) when trace intermediates sit orders of
  magnitude below the bulk species.
- `integrator=IntegratorConfig(...)` — the ESDIRK order, step caps, and Jacobian
  strategy. See [Choosing the integrator](#choosing-the-integrator).
- `diff=DifferentiationConfig(...)` — how a solve under `jax.grad` is
  differentiated. For a stiff model (the activated-sludge models ASM1–ASM3, the
  anaerobic digestion model ADM1, the WATS sewer models) a reverse-mode gradient
  through a reactor solve needs a step cap; see
  [Differentiating a solve](sensitivity_and_calibration.md#differentiating-a-solve).

```python
settings = aquakin.SolverSettings(rtol=1e-8, atol=model.atol({"OH": 1e-20}, default=1e-12))
tight = aquakin.BatchReactor(model, conditions, settings=settings)
```

The same value works for every reactor, so one configuration can be built once
and passed to several.

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

## Dispersed flow and tanks in series (non-ideal 1-D)

Real contactors, columns and channels fall between plug flow and a stirred
tank: the flow carries the water forward, and eddies and velocity differences
spread it along the axis. Two reactors model this, solving in time on a line of
cells, with the reaction in every cell.

- `DispersedFlowReactor` adds axial dispersion to plug flow,
  `dC/dt = -u dC/dx + D d²C/dx² + R(C)`. Give the dispersion as a coefficient,
  `dispersion=D`, or as the dispersion number `dispersion_number=D/(uL)` (the
  reciprocal of the Péclet number), not both. A dispersion number near 0 is plug
  flow; a large one is a stirred tank.
- `TanksInSeries` is `N` equal stirred tanks, each feeding the next, with a
  residence time for the whole train. One tank is a CSTR; many approach plug
  flow.

The examples use a contactor model in minutes: free chlorine decaying at first
order, and an inert tracer.

```yaml
# contactor.yaml
model:
  name: contactor
  version: "1.0"
  description: "First-order free-chlorine decay, and an inert tracer."
  references: []

species:
  - name: HOCl
    description: "Free chlorine"
    units: mg/L
    default_concentration: 0.0
  - name: Tracer
    description: "Inert tracer"
    units: mg/L
    default_concentration: 0.0

reactions:
  - name: decay
    description: "First-order chlorine decay."
    rate: "k * [HOCl]"
    parameters:
      k:
        value: 0.02
        units: 1/min
    stoichiometry:
      HOCl: -1
```

A 30 m contactor with a mean velocity of 0.5 m/min holds the water for 60
minutes. With a dispersion number of 0.05 its steady outlet chlorine is:

```python
contactor = aquakin.load_model_from_file("contactor.yaml")
basin = aquakin.DispersedFlowReactor(
    contactor, aquakin.SpatialConditions.uniform(),
    length=30.0, velocity=0.5, n_cells=40, dispersion_number=0.05,
)
steady = basin.steady_state({"HOCl": 2.0})
steady.C_named("HOCl")[0]      # 0.64 mg/L at the outlet
steady.profile_named("HOCl")   # (1, n_cells): the profile along the basin
```

% invisible-code-block: python
%     assert round(float(steady.C_named("HOCl")[0]), 2) == 0.64

The inlet is a mapping of species to values, and a species left out enters at
zero; a full vector of every species works too. A value may also be an
expression of time `t`, in the model's time unit, written in the language of
[influent expressions](influents.md). `solve` integrates in time from an initial
profile `C0` (one value per species, or one row per cell). Here a 5-minute tracer
dose enters a basin full of clean water:

```python
t = jnp.linspace(0.0, 240.0, 241)
dose = basin.solve(
    jnp.zeros(2), (0.0, 240.0), t,
    inlet={"Tracer": "step(t) - step(t - 5)"},
)
dose.C_named("Tracer")         # the outlet response, one value per time in t
dose.profile.shape             # (241, 40, 2): every cell at every time
```

`sol.C` and the by-name readers give the **outlet**, which is what a tracer
test measures; `sol.profile` holds every cell, and `sol.x` the cell centres.

The scheme is cell-centred finite volume, second order in space: advection is
upwinded from a limited reconstruction (van Albada), so a sharp front does not
oscillate, and the dispersion is a central difference. The inlet is the
Danckwerts condition and the outlet has zero gradient. Use more cells
(`n_cells`, 50 by default) to resolve sharp fronts at a high Péclet number.

`TanksInSeries` takes the number of tanks and the residence time of the train,
and the same `solve` and `steady_state`:

```python
train = aquakin.TanksInSeries(
    contactor, aquakin.SpatialConditions.uniform(), n_tanks=4, residence_time=60.0,
)
train.steady_state({"HOCl": 2.0}).C_named("HOCl")[0]   # 2 / (1 + 0.02*60/4)**4 = 0.70
```

% invisible-code-block: python
%     out = float(train.steady_state({"HOCl": 2.0}).C_named("HOCl")[0])
%     assert abs(out - 2.0 / (1 + 0.02 * 60 / 4) ** 4) < 1e-6 and round(out, 2) == 0.70

### Fitting a tracer test

`fit_dispersion(t, C)` and `fit_tanks_in_series(t, C)` turn the outlet
response to a short tracer pulse into the dispersion number or the number of
tanks, which go straight into the reactors above. Both match the spread of the
curve, its variance over the squared mean residence time: for tanks in series
it is `1/N`, and for a dispersed vessel with these inlet and outlet conditions
`2d - 2d²(1 - exp(-1/d))`. The curve must return to baseline before the record
ends (a cut-off tail is warned about), and a noisy tail should be
baseline-corrected first, since the variance weighs it heavily. A pulse
response simulated in the basin above fits back close to its 0.05:

```python
pulse = basin.solve(
    jnp.zeros(2), (0.0, 240.0), t, inlet={"Tracer": "exp(-(t/0.5)**2)"},
)
aquakin.fit_dispersion(t, pulse.C_named("Tracer"))       # 0.05
aquakin.fit_tanks_in_series(t, pulse.C_named("Tracer"))  # 11
```

% invisible-code-block: python
%     assert round(aquakin.fit_dispersion(t, pulse.C_named("Tracer")), 2) == 0.05
%     assert aquakin.fit_tanks_in_series(t, pulse.C_named("Tracer")) == 11

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
passed as `integrator=` to `plant.solve`, or to a reactor inside its
`SolverSettings`. Its defaults are:

```python
aquakin.IntegratorConfig(order=3, factormax=3.0, colored_jacobian="auto",
                         dtmax=None, max_steps=100_000, solver=None)
```

- `order`: the ESDIRK method, `3` (`Kvaerno3`, the default, with fewer stages
  per step) or `5` (`Kvaerno5`, fifth order: more work per step, but fewer steps
  at tight tolerances and more robust on a hard transient). Try `5` when a solve
  struggles to converge.
- `factormax`: caps how fast the step size may grow from one step to the next.
- `dtmax`: caps the step. Set it **only** for a reverse-mode gradient taken
  *through* a stiff solve (`method="through_solve"`); it is in the model's
  time unit. The cap-free gradient methods in
  [Differentiating a solve](sensitivity_and_calibration.md#differentiating-a-solve)
  do not need it.
- `colored_jacobian`: builds the Jacobian of each implicit step sparsely, by
  graph colouring, which pays on plants with many states. It applies to plants
  only; reactors ignore it. The default `"auto"` uses it for the reverse-mode
  gradient of a large plant (BSM1 and BSM2, but not BSM1 with
  `use_takacs=False`) and leaves the forward solve dense; `True` forces it on,
  `False` off.
- `max_steps`: the step budget; `solver` overrides `order` with an explicit
  Diffrax solver.

`rtol` and `atol` are separate fields of `SolverSettings` and separate arguments of `plant.solve`.
For faster plant solves, including `forward_fast`, see
[Plant-wide simulation](plants.md#faster-dynamic-solves).

## When a solve fails

A solve that cannot finish within its step budget raises a `RuntimeError`
saying that it hit `max_steps`. That is almost always a stiff transient rather
than a bug. Try, in order:

1. **Start closer to where the system operates.** For a plant, pass `y0` from a
   steady state or an earlier run
   ([Initial states and warm starts](plants.md#initial-states-and-warm-starts)).
2. **Loosen `rtol`** (default `1e-6`). The default `atol` scales itself to the
   magnitude of each state; pass a per-species `atol` (`model.atol({...})`) when
   a trace species needs a finer one.
3. **Use the fifth-order method**, `IntegratorConfig(order=5)`.
4. **Raise the budget**, `IntegratorConfig(max_steps=...)` (default 100 000).

If none of these help, the model may be unstable at those parameters or
inputs. A gradient that comes out non-finite is a different problem, a stiff
solve differentiated in reverse mode; `dtmax` and the cap-free methods that
avoid it are in
[Differentiating a solve](sensitivity_and_calibration.md#differentiating-a-solve).
A plant solve that finishes but slowly can use `forward_fast=True`
([Faster dynamic solves](plants.md#faster-dynamic-solves)).
