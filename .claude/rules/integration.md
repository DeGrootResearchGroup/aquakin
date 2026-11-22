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

**Diffrax** is the JAX-native ODE library used throughout. Default solver for
stiff systems: `Kvaerno5`. Adjoint sensitivity (`diffrax.RecursiveCheckpointAdjoint`)
is the default for parameter estimation — memory-efficient for long integration
times typical of reactor contact times.

**Adjoint choice and forward-mode AD.** Every reactor takes an `adjoint=`
argument. The default `RecursiveCheckpointAdjoint` is **reverse-mode only**: it
registers a `custom_vjp`, so `jax.grad`/`jacrev` work but `jax.jvp`/`jacfwd`
(forward mode) are rejected with *"can't apply forward-mode autodiff (jvp) to a
custom_vjp function"*. When you need forward-mode AD through the solve — e.g. a
forward-mode sensitivity Jacobian or a Gauss–Newton/Fisher matrix — construct the
reactor with `adjoint=diffrax.DirectAdjoint()` (or the dependency-free alias
`aquakin.forward_adjoint()`), which is plainly differentiable
in both modes. Its drawback is *usually* memory (it stores/unrolls the whole
solve, cost growing with step count), so keep `RecursiveCheckpointAdjoint` as the
default and switch to `DirectAdjoint` only when forward-mode is actually required.
**The plumbing is now hidden for the common consumers:** `calibrate` and
`sensitivity` take `ad_mode="forward"|"reverse"` and build the right reactor
adjoint internally (no `diffrax` / `adjoint=` in user code); `calibrate` also has
`check_finite=True` to turn a non-finite stiff gradient into a friendly error with
the remedy. **For a user who rolls their own loss + optimizer through
`reactor.solve` (outside `calibrate`/`sensitivity`),** the same silent-non-finite
reverse-gradient footgun is exposed and nothing raises. Every reactor therefore
carries `check_gradient_finite(grad_value, what=...)` (the `GradientCheckMixin` in
`integrate/_config.py`): wrap a freshly computed `jax.grad` in it
(`g = reactor.check_gradient_finite(jax.grad(loss)(p))`) to convert a silent
`NaN`/`Inf` into an actionable error whose remedy is tailored to whether the
reactor already caps `dtmax`. The underlying free checker `check_finite_gradient`
is exported at the package level too (`aquakin.check_finite_gradient`) for guarding
a gradient computed without a reactor handle. `dgsm` takes `ad_mode=` too but cannot set the adjoint for you (your
`fn` constructs the reactor), so its `fn` still needs
`adjoint=aquakin.forward_adjoint()`. And `Plant.solve(gradient="auto")` (the
default) auto-routes a differentiated stiff plant to the cap-free `stable_adjoint`
while keeping a plain forward solve on the fast cached path — so a plant gradient
is finite by default with no `dtmax` to tune.
`dgsm(..., ad_mode="forward")` is the first-class consumer of this (see the DGSM API
below): for a **multi-output sensitivity screen of a stiff model** forward mode
can be *faster and lighter* than reverse — the reverse adjoint is paid once per
output and is inflated by the `dtmax` step cap, whereas forward pushes all `d`
tangents through one solve, independent of the output count. (Benchmarked at
~2× faster, less memory, for the 4-output/17-input Khalil batch screen; for a
single scalar output reverse still wins. Forward and reverse agree to machine
precision — the choice is purely performance, and the right one depends on the
output/input counts and the adjoint stiffness, so both are exposed.) The
**full** second-order AD Hessian (`jax.hessian`) is best avoided entirely: with
the default adjoint it hits the `custom_vjp` wall, and even with `DirectAdjoint`
the second derivatives through the stiff implicit solve are unreliable
(they disagree with finite differences). Use a first-order Gauss–Newton
`H = JᵀJ` instead (see `calibrate(laplace=LaplaceConfig(method="gauss_newton"))`).

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
integrated by the canonical `_run_diffeqsolve` (Kvaerno5 + `PIDController` +
adjoint), so the per-step integration and the RHS cannot diverge between
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

Compiling a stiff solve (JAX trace + lower + XLA) dominates its cost — the run
itself is comparatively free (measured ~1.6 s compile vs ~0.02 s run for an
ASM1 batch solve; ~34 s vs ~4 s for the full BSM2 plant). So the cost of code
that solves repeatedly is *recompilation*, and the integrators cache the
compiled solve to avoid it:

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
  many times, or a warm-started steady-state-then-dynamic run. (Cross-*instance*
  plant caching is deliberately **not** done: a fresh plant's compiled RHS
  depends on the entire unit-config + connection graph, and a structural key
  complete enough to never false-hit would be fragile — a miss there would
  silently return a solve compiled for a *different* plant. Per-instance keying
  cannot false-hit.) `Plant.solve_sensitivity` is cached the same way (keyed
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
computation. The key materialises `atol` values, which is impossible **under
tracing** (a calibration loss differentiating through `solve`); in that case the
key is `None` and the cache is bypassed (the solve is traced into the outer
computation, which JAX compiles as a whole, so caching gives nothing there
anyway). Both caches assume the model / plant is not structurally mutated
after the first solve — the same assumption reactors already make about their
fixed model and conditions.

**What this is and isn't.** It removes *duplicate* compiles; it does not remove
the first compile of each distinct `(model/plant, settings, signature)`, and
the JAX **persistent** (cross-process / cross-run) compilation cache does *not*
help these Diffrax solves (verified: no reuse across processes for either an
ASM1 reactor or the BSM2 plant — it caches only the XLA step, not tracing, and
Diffrax programs miss it across processes). So this speeds repeated solving
within a process; it does not by itself shrink a cold test suite where each test
compiles a distinct configuration once.

---
