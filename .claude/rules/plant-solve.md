---
paths:
  - "aquakin/plant/plant.py"
  - "aquakin/plant/dynamic_solve.py"
  - "aquakin/plant/steady_solve.py"
  - "aquakin/plant/recycle.py"
  - "aquakin/plant/colored.py"
  - "aquakin/plant/solve_cache.py"
  - "aquakin/plant/steady.py"
  - "aquakin/integrate/steady.py"
---

# Rules — plant solve internals (`aquakin/plant/`)

The coloured Jacobian, the parameter-slice and compiled-solve caches, and
recycle resolution (flows, concentrations, the cached maps). The plant overview
and the `Unit` contract are in `.claude/rules/plant.md`. Performance numbers are
in `docs/plant_performance.md`; accuracy measurements behind the rules below are
in `.claude/notes/plant-solve-log.md`. What each class does is in the module
docstrings of `aquakin/plant/colored.py`, `aquakin/plant/solve_cache.py` and
`aquakin/plant/recycle.py`.

## Coloured Jacobian

- **Coloured-Jacobian machinery lives in `ColoredJacobianManager`**
  (`plant._colored`), not on `Plant`. A new coloured builder must go through the
  shared probe → structural union → build → guard scaffold
  (`_build_and_guard` / `_colored_from_probe`): the guard against the dense
  Jacobian, with a fall-back to dense on a mismatch, is what keeps a coloured
  build exact.
- **The size-based decision stays on the plant's solve routing**
  (`Plant._COLORED_BACKWARD_MIN_STATES` / `colored_jacobian_decision`); the
  manager only builds when asked. The report and the stable-adjoint solve must
  both take it from `aquakin.plant.dynamic_solve._auto_colored_backward`, so the
  reported choice is the one the solve makes.
- **A state-layout change must call `plant._colored.reset()`** (a temperature
  block appended changes the state length), because the builders cache patterns
  keyed to the layout.
- **The structural pattern covers unit-state couplings only.** It places a
  source block for every stateful unit that emits a stream, composed with that
  unit's `CouplingPattern.output_pattern` (which state entries its outlets
  read); a unit without one gets the `_output_state_pattern` default (identity,
  head + trailing scalars, or dense by layout). A unit whose outlets read its
  state differently must declare `output_pattern`. Flow-path couplings
  (TSS-dependent thickener / dewatering flows, the interface pH feedback) are
  **not** modelled structurally; they come only from the numerical probe, whose
  threshold is per row.
- **The `stable_adjoint` backward builder must fail closed.** A missed entry
  corrupts gradients there, so `pattern_coverage_misses` must find no uncovered
  entry at the default state, three perturbed copies and the concrete solve
  `y0`, or the backward falls back to dense. Do not relax this check to keep the
  coloured path on.

## Parameter-slice cache

- **`_params_for_unit` stores only concrete slices.** A concrete parameter
  vector closed over by `jax.jit` still yields traced slices; storing one on the
  plant leaks it into another trace.

## Compiled-solve caches and structure changes

- **All solver-run memoization that bakes in reactor conditions lives in
  `SolveCache`** (`plant._solve_cache`: the jitted forward and
  `solve_sensitivity` solves, the PTC steady solves, the continuation and
  arclength kernels). A new compiled artefact of that kind is stored there, so
  one `invalidate()` drops it.
- **A condition mutator calls `plant._solve_cache.invalidate()`**, never
  hand-clears named dicts: clearing some caches and not others silently reuses
  stale conditions. A mutator that also changes the state length
  (`set_temperature_model`) calls `plant._colored.reset()` as well.
- **A structure change goes through `plant._structure_changed()`**
  (`solve_cache.structure_changed`). Every mutator that adds a unit, a
  connection or an influent calls it, and every new structure-derived memo is
  cleared in it (not by each mutator), so an edited plant solves as if it had
  been built that way.

## Arclength continuation must not jump branches

- In `arclength_continuation_solve` (`integrate/steady.py`), a converged
  corrector step farther than `max_correction * dsigma` from its predictor must
  be rejected and retried shorter. Near a fold the step's hyperplane also cuts
  the other branch; accepting that corrector hides the fold and reports the
  other branch as the operating point. Keep this check when changing the step
  control or the corrector.

## Recycle resolution

- **Recycle resolution lives in `RecycleResolver`** (`plant._recycle`), with
  its tri-state map-constancy caches; keep it out of the plant's assembly code.
- **Recycles are resolved exactly and gain-independently per RHS, never by
  iterating to a tolerance inside the jitted, differentiated RHS.** Flows
  (`_resolve_flows`) and concentrations (`_resolve_recycle_concentrations`) are
  each an affine map, probed once at zero plus once per back edge and solved as
  `(I − A)x = b` / `(I − M)c = d`. This relies on the in-cycle units (mixers,
  splitters, clarifiers) being linear in concentration at fixed flows, on the
  map being species-decoupled (the ASM↔ADM translator is fed by a digester state
  and so never enters the cyclic map), and on edges of different models not
  coupling. A new unit that breaks one of these inside a recycle loop must be
  handled by the refinement below, not by more probe passes.
- **`recycle_tol` (default `1e-8`) closes the non-affine residual adaptively.**
  `_adaptive_recycle_refine` iterates the full flow-and-concentration (and
  temperature) fixed point from the affine seed with a `jax.lax.while_loop`
  (capped at `recycle_max_passes`, default 100) inside `jax.lax.custom_root`, so
  the derivative is the implicit-function-theorem tangent and AD cost does not
  grow with the iteration count. It must iterate `Q` together with `C`:
  iterating `C` alone solves the wrong problem when an in-loop flow depends on
  concentration. Do not replace it with a fixed pass count: the fixed
  `recycle_passes` mop-up (default 3, used when `recycle_tol=None`) leaves a
  residual of order `rho**passes`, and `rho` (the loop's spectral radius) is
  topology-dependent and not bounded away from 1, so a fixed count can give a
  silently wrong steady state (`.claude/notes/plant-solve-log.md`).
  `recycle_tol` is read inside `_resolve_recycle_concentrations`, so it reaches
  every solve path without threading.
- **The one-time concrete guards run before the `events=` branch** of the
  dynamic solve (`_run_one_time_guards`): the flow-affinity check,
  `_check_recycle_convergence` (warns, never blocks; skipped under tracing and
  without recycle edges) and the map-constancy checks. An events-only plant
  (SBR, control study) must still get the diagnostics and the cached maps.

### Cached recycle and flow maps

- **`M` (concentration), `MT` (temperature) and `A` (flow) are cached once per
  solve only when verified state-invariant.** `_check_recycle_map_constant` and
  `_check_flow_map_constant` compare each map at three materially different
  states (concrete only, once per plant); a map that varies is re-probed every
  RHS. `MT` is constant in heat-balance mode and when no temperature is carried,
  but not in algebraic mode, where temperature passes through reactors on the
  concentration-dependent flows. Do not cache a map without this check: there is
  no mid-solve re-guard.
- **The maps are built from the runtime `params`**, so the gradient flows
  through them and a parameter sweep stays correct. `_compute_flow_map` coerces
  `params` first so the flow-setpoint block is present, matching the hot path.
- **Every path builds the cached maps through `RecycleResolver.cached_rhs`**
  (the plant RHS closed over both maps) **or `RecycleResolver.cached_maps`**
  (the maps alone, for a path that calls `_resolve_streams` directly). Do not
  call `_maybe_recycle_map` / `_maybe_flow_map` and close over `_rhs` by hand:
  the cached maps are the exactness-critical surface, so one helper wires them.
  A new solve or reconstruction path uses one of these two.
- **Pass `stop_gradient=True` only where the maps are closed over inside a
  custom VJP that accounts for their parameter dependence itself** — the
  discrete-adjoint primal in `Plant._esdirk_stable_adjoint` (the internal
  `"stable_adjoint"` route, reached through
  `diff=DifferentiationConfig(mode="reverse", method="stable")` on a
  differentiated solve). Everywhere else the gradient must flow through them.
- **The discrete adjoint splits its RHS.** `esdirk_adjoint_solve` takes the
  cached-map `primal_rhs=` for the forward solve and the backward `∂f/∂y` stage
  Jacobians, and keeps the map-recomputing `rhs` for the `∂f/∂θ` vjp. The vjp
  supplies the whole parameter gradient, so this split is what captures
  `∂M/∂θ` and `∂A/∂θ` for a flow-setpoint parameter (RAS, `Qw`, `f_PS`); a
  cached map closed over as a constant in the vjp would silently drop that term.
  The cached primal is used only when `M` is cached (`_recycle_map_constant`
  True); a constant `A` alone does not switch it on.
- **The cached and probed paths must give a bit-identical RHS and Jacobian**
  (the cached map *is* the probed map, and `dM/dy = dA/dy = 0`), hence
  bit-identical steady states. A long dynamic run may drift by floating-point
  operation order within the solver tolerance; that is expected, not a bug.
