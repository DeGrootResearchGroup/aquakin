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
recycle resolution (flows, concentrations, the cached maps). Split out of
`.claude/rules/plant.md`, which holds the plant overview and the `Unit` contract.
The long performance log is `docs/plant_performance.md`.

**Colored Jacobian** — the graph-coloring / Jacobian-sparsity subsystem
(`_structural_plant_pattern` + the forward / adjoint / steady builders and their
caches) lives in a [`ColoredJacobianManager`](aquakin/plant/colored.py)
collaborator (`plant._colored`), mirroring `RecycleResolver`: it back-references
the `Plant` for the state layout, unit couplings and RHS, and exposes
`forward_root_finder` (the guarded forward colored root finder, or `None`;
`forward_fast` and the staggered `solve_sensitivity` materialize their per-step
`J` from it via `colored_jacobian.state_jacobian`) / `jacobian_solver` (a diffrax
solver configured with it), `adjoint_jacobian_builder`
(`stable_adjoint` backward `df/dy`) and `steady_jacobian_builder` (PTC steady
`dF/dy`). All three share one probe→structural-union→build→guard scaffold
(`_build_and_guard` / `_colored_from_probe`); each caches its result and
`plant._colored.reset()` clears them when the state length changes (a
temperature block). The size-based *decision* to use the colored backward
(`Plant._COLORED_BACKWARD_MIN_STATES` / `colored_jacobian_decision`) stays on the
plant's solve routing. The performance rationale is in `docs/plant_performance.md`.
The structural assembly places a source block for **every** stateful unit that
emits a stream, composed with that unit's `CouplingPattern.output_pattern` (which
state entries its outlets read: Takács by AD, IFAS the bulk row; undeclared →
identity / head + trailing scalars / dense by layout, `_output_state_pattern`).
It does **not** model flow-path couplings (TSS-dependent thickener/dewatering
flows, the interface pH feedback); those come from the probe, whose threshold is
**per row**. Because a miss corrupts gradients there, the `stable_adjoint` builder
fails closed: `pattern_coverage_misses` must find no uncovered entry at the
default state, three perturbed copies and the concrete solve `y0`, or it falls
back to dense (#647).
**Parameter-slice cache** — `_params_for_unit` retains only concrete slices.
A concrete parameter vector closed over by `jax.jit` can still produce traced
slices; those must not be stored on the plant or reused by another trace.
**Compiled-solve caches** — the jitted forward + PTC steady solves (the
`solve_sensitivity` solves sit in `.jit` under a `"solve_sensitivity"` key) and the
continuation / arclength kernels (all solver-run memoization, keyed by
settings, that bake in the reactor conditions) live in a
[`SolveCache`](aquakin/plant/solve_cache.py) value object (`plant._solve_cache`,
`.jit` / `.steady_jit` / `.continuation_kernels` / `.arclength_kernels`; the
`plant._jit_cache` / `_steady_jit_cache` properties are thin views). A
condition mutator calls the single **`plant._solve_cache.invalidate()`**, never
hand-clears named dicts (`set_temperature`); `set_temperature_model` changes the
state length, so it also calls `plant._colored.reset()`.
**Structure changes go through `plant._structure_changed()`**
(`solve_cache.structure_changed`): `add_unit`, `connect` and `add_influent`
call it; it clears the `SolveCache`, the colored builders,
`RecycleResolver.reset()`, the parameter slices, the one-time recycle
diagnostics and the controller-materialization guards (`_materialize_*` skip
controllers already in `_derived_units`). A new mutator calls it; a new
structure-derived memo is cleared in it, so an edited plant solves as if built so.
**Arclength continuation must not jump branches** (`integrate/steady.py`,
`arclength_continuation_solve`). A converged corrector step farther than
`max_correction * dsigma` from its predictor is rejected and retried shorter:
near a fold the step's hyperplane also cuts the other branch, and accepting a
corrector that lands there hides the fold and reports that branch as the
operating point. Keep this check when changing the step control or the
corrector.
**Recycle resolution** — the methods named below (`_resolve_flows`,
`_resolve_recycle_concentrations`, `_adaptive_recycle_refine`,
`_recycle_context`, `_compute_recycle_map`, `_check_recycle_map_constant`, …)
and the tri-state map-constant caches — lives in a
[`RecycleResolver`](aquakin/plant/recycle.py) collaborator
(`plant._recycle`), which keeps a back-reference to its `Plant` for the
topology, state layout, signal bus and output sweep. Both layers are kept out of
the assembly code below. Recycles are resolved **exactly and
gain-independently** per RHS, in two decoupled steps that both use the same
affine-probe + linear-solve trick (no iterate-to-tolerance — the RHS is
jitted/differentiated):
- **Flows** — `_resolve_flows` probes the (affine) recycle-flow map and solves
  `(I − A)x = b` for the back-edge flows.
- **Concentrations** — `_resolve_recycle_concentrations` does the same for the
  recycle-edge *concentrations*. One forward output sweep at fixed flows is an
  affine map `c → M·c + d` (mixers/splitters/clarifiers are linear in
  concentration; stateful units output their state, a constant), so it probes
  `M`/`d` (one pass at `c=0`, one per recycle edge set to a unit concentration)
  and solves `(I − M)c = d`. The map is **species-decoupled** (the only
  species-coupling unit, an ASM↔ADM translator, is fed by a digester *state* so
  never enters the cyclic map), so one probe per edge yields its whole column
  across all species — `n_recycle_edges + 1` cheap passes, like the flow probe.
  Edges of **different models** don't couple (the translator that would couple
  them is broken by the digester state), so the solve is grouped by model;
  **temperature**, when an influent carries it, is one more decoupled scalar
  channel. Exact and gain-independent: a recycle loop whose bare Gauss-Seidel
  would need thousands of passes (a clarifier in a high-capture stateless loop)
  is solved in one linear solve. Validated as a fixed point on BSM1, the
  multi-model BSM2, and a temperature-carrying loop (residual ~1e-12).
The exact concentration solve **seeds** the `recycle_passes` Gauss-Seidel
mop-up (default 3), which therefore does no work for any linear topology (every
shipped plant) and only refines a genuinely *non-affine* in-cycle unit (a
translator inside a pure-stateless loop — not constructible from the shipped
units). A one-time `_check_recycle_convergence` diagnostic (concrete-only,
skipped under tracing, skipped without recycle edges) warns if even that has
not converged — the backstop for the non-affine case.
- **Adaptive AD-safe recycle convergence (`recycle_tol`).** The fixed
  `recycle_passes` count is a *diagnostic-backed default*, not a general
  convergence guarantee: the mop-up converges geometrically in
  `log(tol)/log(rho)` passes, where `rho` is the spectral radius of the
  nonlinear flow↔concentration coupling Jacobian (the reject loop). For BSM2 the
  only iterating stream is the front mixer (influent + reject recycle) and
  `rho ≈ 0.0066`, so 3 passes leaves ~1e-6 residual — but `rho` is
  topology-dependent and **not bounded below 1**: a recycle-heavy plant with a
  strong concentration-dependent in-loop flow (e.g. a high-capture
  thickener/dewatering `%TSS` underflow on a tight reject loop) can have `rho`
  near 1, where the fixed 3 passes leaves residual `rho³` — a **silently wrong
  steady state** (measured on a synthetic map: 13% error at `rho=0.5`, 73% at
  `rho=0.9`, 97% at `rho=0.99`). `Plant(..., recycle_tol=...)` (**on by
  default, `1e-8`**) replaces the fixed mop-up with an **adaptive solve to that
  relative tolerance**, mirroring the charge-balance pH solver
  ([`core/ph_solver.py`](aquakin/core/ph_solver.py)): the recycle back-edge
  **streams** — flow `Q`, concentration `C`, and (when carried) temperature `T`
  — are the fixed point `x = G(x)` of one forward output sweep
  (`RecycleResolver._recycle_context`'s `forward_full`, which lets `Q` vary so the true
  `Q↔C` reject-loop coupling is captured; iterating `C` alone solves the wrong
  problem), warm-started from the exact affine seed and iterated by a
  `jax.lax.while_loop` that **stops once the actual residual clears** (capped at
  `recycle_max_passes`, default 100), wrapped in `jax.lax.custom_root` so the
  sensitivity is the exact **implicit-function-theorem tangent** — a small dense
  solve of the linearised recycle-edge operator (the recycle edges are few ×
  ~tens of channels), the vector generalisation of the pH solver's scalar
  `y / g(1)`. AD (forward and reverse) is therefore **O(1) in the iteration
  count** rather than differentiating through every sweep
  (`RecycleResolver._adaptive_recycle_refine`). It converges for any `rho < 1`, stops
  early on a low-gain plant, and is **on by default** at `1e-8` (well below the
  typical solver `rtol`, a strict improvement on the old fixed-3-pass ~1e-6 at
  ~neutral cost — ~3 iterations from the affine seed for BSM); `recycle_tol=None`
  falls back to the fixed `recycle_passes` mop-up (the bit-identical historic
  behaviour). **The validated BSM steady states are reproduced** with the
  default on (the published BSM2 / ADM1 / Takács validations pass unchanged) —
  the adaptive path converges to the *same* recycle fixed point the fixed-pass
  mop-up approximates, only tighter. **Verified** (`tests/integration/
  test_adaptive_recycle.py`): on a synthetic tunable-`rho` map the adaptive
  solve reaches tolerance for `rho` up to 0.99 where the fixed 3-pass leaves
  13–97%, and its IFT gradient matches central finite differences to ~2e-10; on
  the real BSM2 reject loop the adaptive forward fixed point matches a 14-pass
  deep sweep to ~4e-15, and its IFT tangent matches the deep-sweep gradient to
  ~6e-19. `recycle_tol` is read inside `_resolve_recycle_concentrations`, so it
  reaches every solve path automatically (no per-path threading); the cached
  affine `recycle_map` still supplies the warm-start seed. Because the default
  routes every plant solve through `jax.lax.custom_root`, the
  `gradient="stable_adjoint"` discrete adjoint composes with the recycle IFT
  tangent (verified exact: the cached/probed `dM/dθ` gradient agrees to ~1e-13,
  and the cross-interface gradient matches finite differences to the FD floor);
  with the adaptive default `M` is only the warm-start seed (the fixed point is
  M-independent), so the `#366` cached/probed-map gradient distinction now agrees
  to float rounding rather than bit-for-bit.
- **Cached recycle map (per-RHS speedup).** The concentration map `M` is fixed
  by the recycle flows + topology, so for a **fixed-pump** plant (every BSM
  plant — the recycle pumps are constant) it is **invariant to the state and
  time**; only `d = forward(0)` varies. The `n_recycle_edges` per-species
  `M`-probe sweeps are therefore recomputing a constant on every one of the
  ~17 RHS calls per implicit step. `_compute_recycle_map` precomputes `M`
  **once per solve** (from the runtime `params`, so the gradient still flows
  and a parameter sweep stays correct) and `_build_jitted_solve` threads it into
  every RHS as `recycle_map=`; the per-RHS recycle resolution then computes only
  `d` (one sweep) + the cached `(I−M)` solve. Profiling located the per-RHS cost
  as ~88% the recycle resolution and the per-step cost as ~RHS-evaluation-bound,
  so this is a real dynamic-solve win. The **temperature** map `MT` is cached
  too **when it is state-invariant**, which depends on the temperature model:
  in **heat-balance** mode the reactor temperature is a *state* (it breaks the
  loop coupling at reactors, exactly as concentration does) so `MT` is constant
  and cached → full win; in **algebraic** mode temperature *passes through*
  reactors (no thermal mass) so `MT` rides on the concentration-dependent
  recycle flows and is **not** constant → it is re-probed every RHS (a cheap
  scalar T-only sweep, its per-species part CSE-shared with `d`), while `M`
  stays cached. Net measured speedup on the dynamic BSM2 (algebraic):
  ~recycle-resolution 2.4× → ~1.18× wall; larger (full ~5.6× recycle) for
  heat-balance / no-temperature plants. **Exactness:** the cached and probed
  paths produce a **bit-identical RHS** (the cached `M` *is* the probed `M`);
  the dynamic trajectory shows only ~1e-3 floating-point operation-order drift
  over multi-day runs (the cached `M` is formed once outside the integration
  loop vs per-call inside), within the solver tolerance, and the validated
  steady states are preserved. A one-time concrete guard
  (`_check_recycle_map_constant`, set per instance) compares each map at three
  materially-different states and **falls back to per-RHS probing** for any
  topology whose `M` is genuinely state-coupled — so the optimization is safe
  for arbitrary plants.
  **Every path builds the cached maps through `RecycleResolver.cached_rhs`**
  (the plant RHS closed over both maps) **or `cached_maps`** (the maps alone,
  for a path that calls `_resolve_streams` directly). Do not call
  `_maybe_recycle_map` / `_maybe_flow_map` and close over `_rhs` by hand: the
  cached maps are the exactness-critical surface, so one helper wires them.
  Pass `stop_gradient=True` only where the maps are closed over inside a custom
  VJP that accounts for their parameter dependence itself (the
  `stable_adjoint` primal); everywhere else the gradient flows through them.
  The cached map is built once from `params` and reused by **four** paths: the forward `jax_adjoint` solve,
  the located-event segmented solve (`events=`, reused across every segment),
  the single-instant `outputs_at`, and the whole-trajectory stream
  reconstruction (`_cached_streams` / `plant.stream`, the `evaluate_bsm*`
  evaluation path — measured ~1.16×, bit-identical). The events path needed the
  one-time constancy check hoisted *above* the events branch in `solve` so an
  events-only plant (SBR / control study) still gets the cached map (and the
  affinity/convergence diagnostics). The reconstruction win confirms its
  per-time cost was also recycle-resolution-dominated. **`gradient=
  "stable_adjoint"` also uses the cached map (#366)**, via a *primal/param RHS
  split* of the discrete-adjoint kernel. `esdirk_adjoint_solve` forms `∂f/∂θ` by
  differentiating the per-call `rhs(t, y, params)`, so a precomputed `M` closed
  over as a constant would be invisible to that vjp and a gradient w.r.t. a
  **flow-setpoint param** (RAS/`Qw`/`f_PS`, the only params `M` depends on) would
  silently drop its `∂M/∂θ` term. The kernel therefore takes an optional
  `primal_rhs=`: the forward solve and the backward **`∂f/∂y`** stage Jacobians
  use the cached-`M` `primal_rhs` (the recycle probe hoisted out of the hot
  loop), while the **`∂f/∂θ`** vjp keeps the map-recomputing `rhs` — so `∂M/∂θ`
  is captured exactly. Because the discrete adjoint draws its *entire* parameter
  gradient from that vjp and uses the stages/Jacobians only to propagate the
  *state* cotangent, and the cached `M` *is* the probed `M`, the result is
  **bit-identical** to probing every call, just faster (the gradient w.r.t. a
  kinetic *and* a flow-setpoint param both match the per-call-probe gradient
  bit-for-bit). The cached map must be `stop_gradient`'d before being closed over
  (it is a params-derived value inside the `custom_vjp`; its parameter dependence
  is the vjp's job). The path runs through `Plant._esdirk_stable_adjoint`.
  Falls back to per-call probing when `M` is not state-invariant
  (`_recycle_map_constant` not True). **Measured (clean serial min-of-8 timing):
  a modest reverse-gradient win where the recycle probe is non-trivial — BSM2
  `value+grad` ~1.15× (14.7→12.8 s, the ASM↔ADM-interface probe hoisted out of
  the backward) — and neutral on BSM1** (whose probe is cheap, so its gradient is
  unchanged). The one-time map build makes the *forward* marginally slower (BSM2
  ~0.95→1.09 s), but `stable_adjoint` exists for gradients, where the net is
  positive. *(Historical note: when this was measured the backward's dominant
  cost was the per-step `n≈167` **dense stage Jacobian builds** — ~82% of the
  backward — because it recomputed every ESDIRK stage by Newton; the saved-stage
  backward later removed that recompute, dropping the builds to ~7/step (~24%),
  so the cached-map win and the colored-build win are both proportionally
  different now — see the stage-saving and colored-backward bullets for the
  re-measured numbers.)*
  Covered by
  `tests/integration/test_recycle_cached_map.py` and the bit-identical
  flow-setpoint `∂M/∂θ` guard
  `test_plant_stable_adjoint.py::test_stable_adjoint_flow_setpoint_gradient_preserves_dM_dtheta`.
- **Cached recycle *flow* map (the analogue, #397).** The recycle *flow* solve
  `_resolve_flows` is the same `(I−A)x=b` affine structure as the concentration
  solve: the `n×n` back-edge flow-response `A` is fixed by the recycle flows +
  topology (so constant for a fixed-pump plant), while only `b` (the
  influent-driven constant) varies per RHS. The per-RHS flow probe therefore
  re-derives a constant `A` (the `n` per-back-edge `one_pass(eye[i])` column
  passes) on every RHS. `_compute_flow_map` precomputes `A` **once per solve**
  (from `params`, gradient-preserving — `A` depends on the flow-setpoint block,
  so it is coerced in and `stop_gradient`'d on the `stable_adjoint` primal like
  `M`), threaded into every RHS as `flow_map=` by the same helper; the flow
  resolution then computes only `b` (one pass) + the cached `(I−A)` solve,
  skipping the `n` column probes. State-invariance is detected once by
  `_check_flow_map_constant` (compares `A` at three states; falls back to per-RHS
  probing for a state-coupled split like a level-gated storage bypass), wired
  into the same four paths as `M` (`make_rhs` forward, events, `outputs_at`,
  `_cached_streams`) plus the `forward_fast`, steady-state-PTC, and
  `stable_adjoint` builders. **Correctness:** the cached-`A` RHS *and Jacobian*
  are **bit-identical** to the probe (`A` is state-invariant, so `dA/dy=0` in
  both), and every steady state (constant influent → convergent dynamics) is
  bit-identical — so all validations are preserved. On a **sensitive
  time-varying** run the cached and probe solves can separate (measured ~9.5e-3
  rel over a dynamic BSM2), because the probe recomputes
  `A = one_pass(eye)−one_pass(0)` each step and the varying influent perturbs
  that cancellation's rounding by ~1e-16 each step, while the cache holds `A` at
  its exact constant value — the cached path is the **cleaner** of the two valid
  solves, and the divergence cannot amplify at a fixed point (hence the
  bit-identical steady state). **Measured 1.076× (7.6%)** on the 60-day dynamic
  BSM2 forward solve. Covered by the flow-map tests in
  `tests/integration/test_recycle_cached_map.py` (constancy detection,
  bit-identical RHS + Jacobian, bit-identical steady state, the no-recycle case,
  and a gradient through the cached path).
