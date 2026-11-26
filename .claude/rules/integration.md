---
paths:
  - "aquakin/integrate/**"
---

# Rules — `aquakin/integrate/`

ODE integration strategy: solver choice, differentiating stiff models
(`dtmax`, forward/reverse adjoints, the discrete adjoint), operator splitting,
located events, and compiled-solve caching. Loaded automatically when editing
files under `aquakin/integrate/`.

**Topic rules split out of this file** (each auto-loads for its own files; read
it on demand when a change elsewhere touches the topic):

| Rule | Loads for | Covers |
|---|---|---|
| `integration-calibrate.md` | `calibrate.py`, the `calibration/` subpackage, `seam.py`, `_linalg.py`, `fit.py`, `fisher.py`, `profile.py`, `identifiability.py`, the experiment-design modules, and the plant calibration / identifiability modules that reuse them | `calibrate()`'s internal structure and the helpers `Plant.calibrate` shares |
| `integration-differentiation.md` | `_common.py` / `_config.py` / `_solver.py`, `discrete_adjoint.py`, `forward_sensitivity.py`, `forward_solve.py`, `_simultaneous_corrector.py`, `sensitivity.py`, `colored_jacobian.py`, and `plant/sensitivity.py` / `plant/dynamic_solve.py` / `plant/colored.py` | differentiating stiff models: `dtmax`, the adjoints, the discrete adjoint, forward sensitivities |

## Layering

`integrate` must not import `plant` or `study`, not even inside a function
body (`tests/unit/test_layering.py` checks every import). Solvers generic in
the right-hand side that the plant also uses live here: the PTC steady-state
kernel and the continuation drivers are in `integrate/steady.py`, which
`plant/steady.py` re-exports. Within `integrate`, the shared internals
`_config.py`, `_solution.py`, `_cache.py` and `_solver.py` import no other
`integrate` module, and `_common.py` only re-exports them (its `__all__` is
pinned by `tests/unit/test_common_exports.py`; do not grow it, and put new
shared code in the module for its concern). The private names the plant
layer reuses are listed in `integrate/seam.py`, the only route `plant` and
`study` may take to them (`tests/unit/test_integrate_seam.py`), so renaming one
means updating the seam and its plant caller in the same change. The
sensitivity preamble (`prepare_sensitivity`, `resolve_sens_indices`) lives in
`forward_sensitivity.py`.

## ODE Integration Strategy

### Solver

**Diffrax** is the JAX-native ODE library used throughout. Default stiff solver:
`Kvaerno3` (`IntegratorConfig(order=3)`; `order=5` selects `Kvaerno5`). Adjoint
sensitivity (`diffrax.RecursiveCheckpointAdjoint`) is the default for parameter
estimation — memory-efficient for the long integration times of reactors.

**Adjoint choice and forward-mode AD.** A reactor takes no `adjoint=`
argument: its diffrax adjoint comes from `diff=DifferentiationConfig(...)`
(`DifferentiationConfig.reactor_adjoint`). The default `RecursiveCheckpointAdjoint`
is **reverse-mode only** (it registers a `custom_vjp`, so `jax.jvp`/`jacfwd` are
rejected); `mode="forward", method="through_solve"` gives the reactor
`DirectAdjoint` (`aquakin.forward_adjoint()`), differentiable in both modes at
the cost of memory. `calibrate` and `sensitivity` build the forward adjoint
internally from `diff.mode`; `dgsm` cannot (its `fn` builds the reactor).
**Advice to users names the real keyword arguments.** Every error message or
docstring that tells a user how to switch AD direction or adjoint uses the
constants in `integrate/_config.py` (`FORWARD_DIFF`, `FORWARD_THROUGH_SOLVE_DIFF`,
`STABLE_REVERSE_DIFF`, `FORWARD_REACTOR_REMEDY`); do not write the spelling out
again, and do not name removed keywords (`ad_mode=`, `gradient=`, a reactor
`adjoint=`), which once survived the API change in every remedy.
`tests/unit/test_user_hints.py` fails on a message or public docstring that names
a keyword the API does not take. A non-finite
reverse gradient a user takes by hand through `reactor.solve` is guarded by
`reactor.check_gradient_finite(g)` (`GradientCheckMixin`) or the free
`aquakin.check_finite_gradient`. `Plant.solve` with the default `diff`
(reverse, `method="stable"`) routes a differentiated stiff plant to the cap-free
discrete adjoint and keeps a plain forward solve on the fast cached path, so a
plant gradient is finite by default with no `dtmax` to tune.
`dgsm(..., diff=DifferentiationConfig(mode="forward"))` is the first-class consumer of this (see the DGSM API
below): for a **multi-output sensitivity screen of a stiff model** forward mode
can be *faster and lighter* than reverse — the reverse adjoint is paid once per
output and is inflated by the `dtmax` step cap, whereas forward pushes all `d`
tangents through one solve, independent of the output count (both are exposed;
measured in `.claude/notes/differentiation-log.md`). The **full** second-order
AD Hessian (`jax.hessian`) is best avoided: with the default adjoint it hits
the `custom_vjp` wall, and even with `DirectAdjoint` the second derivatives
through the stiff solve disagree with finite differences. Use a first-order Gauss–Newton
`H = JᵀJ` instead (see `calibrate(laplace=LaplaceConfig(method="gauss_newton"))`).

**A failed solve is checked on the host.** `_run_diffeqsolve`, the forward
sensitivity and the lean `forward_solve` loop return a failed solve with its
diffrax `result`, so every caller must pass it to `check_solve_result` in
`integrate/_solver.py`: raising inside the program makes JAX log Equinox's
banner first. Under a trace it restores the runtime check (gradients still fail).

### Operator Splitting

Transport and reaction are decoupled at all scales:

| Scale | Transport | Reaction |
|---|---|---|
| 0D batch | n/a | Diffrax directly |
| 1D PFR | advection/diffusion step | Diffrax reaction sub-step |
| 3D CFD | OpenFOAM transport step | Diffrax (or C++ stiff solver) reaction sub-step |

The ODE integrator only ever sees the reaction sub-problem — a pure chemistry
integration over one transport timestep at a fixed spatial location.

### Located events / discontinuities ([`integrate/events.py`](aquakin/integrate/events.py))

A plain solve is continuous; on/off pumps, SBR fill/react/settle/decant phase
switches, relay/saturating control, dosing on/off and tank-level limits are
**discontinuous**. `aquakin.Event` + `solve_with_events` locate the switch
exactly and apply a **state reset / mode switch** there, then continue —
instead of smoothing it or grid-snapping with `searchsorted`. Exposed as an
`events=` argument on `BatchReactor.solve` and `Plant.solve`; both build their
RHS and hand it to the shared driver, which returns the trajectory on the
requested `t_eval` grid plus a `solution.events_log` of `(time, name)` firings.
**No drift from the plain solve:** the event path reuses the *same* two pieces
the plain solve uses — the reaction RHS comes from the shared
`make_chemistry_rhs` factory (batch) or `self._rhs` (plant), and each segment is
integrated by the canonical `_run_diffeqsolve` (the configured Kvaerno +
`PIDController` + adjoint), so the per-step integration and the RHS cannot diverge between
`solve()` and `solve_with_events`. A parity test pins this: an identity reset (or
a never-firing state event) reproduces the plain `solve()` trajectory, so any
future change to the RHS/kernel that reaches only one path fails the test
(`tests/integration/test_events.py`).

An `Event` carries exactly one trigger — `at_times=[...]` (a **time event**) or
`cond_fn(t, y, args)` (a **state event**, located by an optimistix root find on
the zero crossing, filtered by `direction` ±1) — plus an optional
`apply(t, y, args) -> y` reset and a `terminal` flag. The driver splits the
solve into segments at the firings; the boundary convention is that a `t_eval`
point coinciding with a firing reports its **pre-reset** value (it belongs to
the segment ending at the event), so the reset defines the next segment's
initial condition.
The same holds for the single final point returned when `t_eval` is `None`
(an event at exactly `t1`, or a terminal event, is applied and logged, but the
reported state is pre-reset, and its time is where the solve actually stopped),
so `t_eval=None` and `t_eval=[t_end]` agree. A time event at exactly `t0` is
applied to `y0` before the first segment and logged. With
`time_unit=` (`BatchReactor.solve` and `Plant.solve`), events are in the
requested unit: `events_to_native_time` converts copies (`at_times` scaled,
`cond_fn`/`apply` wrapped to receive the requested-unit `t`) and the log is
scaled back. In the plant only the user's events are converted; the SBR unit
cycle events are built natively and merged after the conversion.

Two paths, one driver (`_drive`), chosen by whether any state event is present:
- **Time events only** — the segment boundaries are static Python constants and
  no branch depends on traced state, so the whole solve is a fixed sequence of
  differentiable diffrax sub-solves: **`jax.grad` flows through it** (the SBR /
  scheduled-dosing / AD-safe case). It still needs the `dtmax` cap for a stiff
  reverse-mode gradient, exactly like the plain solve.
  With time events only the drive also traces under `jax.jit`: the `t_eval`
  points are assigned to segments by mask, never by reading their values, so
  a traced grid works. Do not add a concrete conversion (`np.asarray`,
  `float()`, a Python `if`) of `t_eval`, `y` or `args` to that path.
- **Any state event** — the firing time/count is discovered at runtime (located
  via a terminating `diffrax.Event` whose `event_mask` says which fired), so the
  loop is an **eager forward simulation**, not differentiable through the switch
  (use a smoothed `cond_fn` where a gradient through the threshold is needed). A
  `max_segments` guard raises a clear error if a reset fails to clear the
  threshold and the event re-fires without advancing.

This is distinct from the low-level `Plant.solve(event=<diffrax.Event>)` used
internally by `run_to_steady_state` (a single terminating event); the
user-facing API is `events=[Event(...)]`. `events=` is rejected with
`gradient="stable_adjoint"` (it runs its own segmented solve). It is the
prerequisite for the SBR unit (#273) and relay/on-off control studies.
Demonstrated in `examples/event_handling.py` (scheduled ozone re-dosing + a
bromate-limit terminal cut-off); tested in `tests/integration/test_events.py`
(reset/terminal/direction/multi-event, AD through a time event, the runaway
guard, and plant resets checked against the closed form).


### Compiled-solve caching

Compiling a stiff solve (JAX trace + lower + XLA) dominates its cost, so code
that solves repeatedly pays in *recompilation*; the integrators cache the
compiled solve (timings: `docs/plant_performance.md`):

- **Models** (`load_model`) are cached by name, so repeated
  `load_model("asm1")` returns the **same** object (and skips re-parsing the
  YAML). A `CompiledModel` is immutable in use; `clear_model_cache()` resets
  the cache. The stable identity is what lets the solver caches key on the
  model across calls.
- **Reactor solves** are cached **across instances** in a module-level cache
  (`integrate/_cache.py`) keyed by `(model identity, solver settings, call
  signature)`. Two *fresh* reactors for the same model + settings + signature
  reuse one compiled solve — so building many short-lived reactors (ensembles,
  library code that constructs reactors internally) no longer recompiles each
  time. (The `_build_jitted_solve` closure captures only the model and the
  scalar settings, so the key is complete; argument shapes/dtypes are handled by
  JAX's own per-function cache.) `BatchReactor`, `PlugFlowReactor` and
  `ParticleTrackReactor` all route through this cache: the batch key carries the
  `(t0, t1, t_eval shape)` call signature, the PFR key the fixed geometry
  (`velocity`/`length`/`n_points`/`n_locations`), and the particle key only the
  `(model, settings)` — the particle reactor passes the track's sample times
  and condition fields as **runtime arguments** (not baked into the closure), so
  an `integrate_ensemble` over same-shape tracks compiles **once** and JAX's
  per-shape cache covers tracks of differing length. (`BiofilmReactor` keeps a
  per-instance cache — its multi-compartment geometry makes a complete
  cross-instance key less clear-cut.)
- **Plant solves** are cached **per instance** (`Plant._jit_cache`), keyed by
  signature + settings. The plant RHS closes over the (static) unit graph, so
  the first solve compiles and every later solve of that plant reuses it —
  e.g. a parameter sweep / Monte Carlo that builds the plant once and solves
  many times, or a warm-started steady-state-then-dynamic run. Cross-*instance*
  sharing is only between plants built by `Plant.from_spec` from specs of the
  same fingerprint (as built, model and table content included) and unchanged
  since: a structural key computed from an arbitrary plant would be fragile,
  but the spec fingerprint already defines what is simulated (the study
  staleness check relies on it), and any mutator ends the sharing
  (`.claude/rules/plant-solve.md`). **Located-event solves** (`events=`) are cached too: every
  segment runs one jitted program from `events.build_segment_solver` whose
  segment bounds, save grid, `y0`, `args` and RHS context (the plant's cached
  recycle maps; the batch reactor's `params` and conditions) are runtime
  arguments, so all segments of a solve share one compile and a repeat solve
  compiles nothing. The key is `SegmentSettings.cache_key()` (the settings key
  plus the root-find tolerances), whether a grid is reported, and
  `events.segment_signature` (the state events' `cond_fn` / `direction`: time
  events, resets and names stay in the eager driver and are not compiled);
  the plant adds its recycle `cache_flags()` and keeps the entry in
  `_jit_cache` under `"events"`, the batch reactor uses the shared
  `cached_jitted_solver` keyed on the model. A `cond_fn` keys by its own
  equality, so `events_to_native_time` wraps it in a value object
  (`_InRequestedTime`) rather than a lambda. `Plant.solve_sensitivity` is cached the same way (keyed
  by its static structure, everything else a runtime argument — see its bullet
  above). The event path (`run_to_steady_state`) is not cached
  (run-once). The `stable_adjoint` path is **not** cached: `solve` routes to it
  only under reverse-mode differentiation (traced `params` / `y0`), and there the
  adjoint's `custom_vjp` is traced directly into the outer computation — routing
  it through an inner `jax.jit` does not compose with an outer reverse-mode pass.
  A jitted calibration loss amortizes the (large) plant compile across optimizer
  iterations through the *outer* jit. Do not add a concrete-input cache to that
  path: no public call reaches it with concrete inputs. Jitting that loss is possible because
  `_coerce_atol` returns the 0-d `atol` array unchanged under tracing instead of
  forcing it to a Python `float` (a `float()` on a tracer raises a
  concretization error).

**Correctness guarantees.** A cache key never omits anything that changes the
compiled result, so a hit always returns a solver compiled for the exact same
computation. The key materializes `atol` values, which is impossible **under
tracing** (a calibration loss differentiating through `solve`); in that case the
key is `None` and the cache is bypassed (the solve is traced into the outer
computation, which JAX compiles as a whole, so caching gives nothing there
anyway). The reactor cache assumes the model is not mutated after the first
solve; the plant's is dropped by `Plant._structure_changed()` whenever a unit,
connection or influent is added (`.claude/rules/plant-solve.md`).

**What this is and isn't.** It removes *duplicate* compiles; it does not remove
the first compile of each distinct `(model/plant, settings, signature)`, and
the JAX **persistent** (cross-process / cross-run) compilation cache does *not*
help these Diffrax solves (verified: no reuse across processes for either an
ASM1 reactor or the BSM2 plant — it caches only the XLA step, not tracing, and
Diffrax programs miss it across processes). So this speeds repeated solving
within a process; it does not by itself shrink a cold test suite where each test
compiles a distinct configuration once.

---
