---
paths:
  - "aquakin/plant/**"
---

# Rules — `aquakin/plant/`

Plant-wide flowsheet simulation: units, recycle resolution, BSM1/BSM2/A2O
builders, control, dosing, design, evaluation. Loaded automatically when
editing files under `aquakin/plant/`.

> The long **dynamic-solve performance** log (the stiffness-bound regime and its
> levers — colored Jacobian, cached recycle/flow maps, forward_fast, PTC steady
> state, etc.) lives in `docs/plant_performance.md` (read on demand).

## Plant-Wide Simulation

`aquakin.plant` composes kinetic reactors with non-reactive unit ops into
a full plant flowsheet. The plant assembles each unit's internal state
into one flat vector and integrates the whole thing under a monolithic
Diffrax solve — so `jax.grad` flows end-to-end across the plant, and
`aquakin.calibrate()` works on plant-level parameter vectors.

**By-name plant parameters.** A `Plant` concatenates its models' parameter
vectors into one flat `default_parameters()`. `Plant.parameter_values(overrides)`
gives that flat vector the same friendly by-name API as
`CompiledModel.parameter_values`, keyed by `"<model>.<param>"` (the model
name plus the model's own namespaced parameter name) — e.g.
`plant.parameter_values({"asm1.muH": 4.0, "adm1.k_hyd_ch": 10.0})` to bump one
rate in a multi-model plant (BSM2's ASM1 water line + ADM1 digester) without
hand-computing the block offset. `parameter_names()` lists the valid keys;
`parameter_index(name)` returns the flat index (the companion for `jax.grad`
w.r.t. one parameter, which can't go through `parameter_values` — that
materialises concrete values). All three reuse the existing
`model_param_blocks` layout. Unknown names raise a `KeyError` with a
close-match hint.

Key types:

- `Stream(Q, C, model)` — the bulk-flow + concentration record passed
  between units.
- `Unit` Protocol — every unit declares `state_size`, `input_ports`,
  `output_ports`, and implements `initial_state()`, `compute_outputs()`,
  `rhs(t, state, inputs, params, signals)`, and `flow_outputs(input_flows,
  params, ctx)`. **Every unit has the *same fixed signature* for each method**
  — the plant never branches its call on a per-unit capability flag. The
  control-signal bus is threaded into every `rhs` as `signals` (an uncontrolled
  unit ignores it), and `flow_outputs` always receives a `FlowContext` carrying
  the unit's own state and the time (a fixed-split unit ignores it). The one
  optional, duck-typed hook is `signal_outputs(...)`, implemented only by units
  that *produce* control signals (e.g. `PIController`). A **stateless** unit
  (`state_size == 0`: mixers, splitters, ideal separators) inherits the
  `StatelessUnit` mixin (`plant/units.py`), which supplies the three trivial
  state members (`state_size → 0`, empty `initial_state`, no-op `rhs`), so it
  only writes `compute_outputs` / `flow_outputs`. It is a plain mixin, not part
  of the Protocol, so it composes with the `@dataclass` units.
- `StateTranslator` Protocol — converts streams between models.
  `IdentityTranslator` covers single-model plants (BSM1).
- `Plant` — assembles units and connections, drives the monolithic
  integration. Its **sensitivity / uncertainty-quantification surface** —
  `steady_state_sensitivity` / `steady_state_dgsm` / `solve_sensitivity` /
  `dynamic_sensitivity` / `dynamic_dgsm` and their result dataclasses — lives in
  [`plant/sensitivity.py`](aquakin/plant/sensitivity.py) as free functions taking
  the plant, bound onto `Plant` as methods (the public `plant.steady_state_dgsm(...)`
  API is unchanged); that layer only *consumes* the public solve API.
  **Calibration** — `plant.calibrate(observations, t_obs, free_params, target=,
  observed_channels=, y0=, ...)` ([`plant/calibrate.py`](aquakin/plant/calibrate.py),
  bound as a method) MAP-fits by-name plant parameters (`"<model>.<param>"`)
  against an output stream. It **reuses the reactor-calibration machinery**
  (`integrate/calibrate.py`: `_CalibrationProblem`, `_build_objective`,
  `_run_multistart`, `_laplace_posterior`) through the forward-model seam — the
  plant supplies a `_PlantForwardModel` (`plant.solve` + stream reconstruction; the
  cap-free stable adjoint by default, so the reverse gradient is finite) and a
  `_PlantParamNamespace` (adapts `plant.parameter_index` / per-model transforms &
  priors to the `model` interface the generic layer expects). Fits kinetic
  parameters against **one or more output streams** — `observables=[PlantObservable(
  stream, channels), ...]` (exported `aquakin.plant.PlantObservable`; the
  `_PlantForwardModel` reconstructs each stream and concatenates the observed
  channels, so the observation columns run in observable order), or the single
  `target=`/`observed_channels=` sugar. **Assembled-state initial conditions** can
  be fit alongside the parameters via `free_ic=FreeICConfig(species=["unit.species",
  ...])` (species also accept `(unit, species)` pairs) — each names a slot of `y0`
  on a concentration unit
  (CSTR / digester), resolved to a flat-state index (`_state_layout[unit][0] +
  species_index`) and fit in log space through the *same* reactor free-IC machinery
  (`m_ic`/`ic_species_idx`/`ic_center_full` on `_CalibrationProblem`); the fitted
  state comes back as `result.C0_fitted[0]` / `result.ic_named[0]`, and the
  `_PlantForwardModel` uses the per-dataset `C0` as `y0` when `use_c0_as_y0` (free
  ICs *or* multi-batch). A **joint multi-batch fit** — several plant runs from
  different initial states sharing the parameters, list-valued `observations` /
  `t_obs` / `y0`, data terms summed — is the plant analogue of the reactor
  multi-batch and reuses the generic multi-dataset machinery (n_datasets > 1,
  per-dataset `C0_base`); `free_ic` + multi-batch are not yet combinable, and
  `predictive_band` remains reactor-only. Covered by
  `tests/integration/test_plant_calibrate.py` (single- and multi-stream synthetic
  recovery, a joint rate + initial-condition recovery, a multi-batch shared-rate
  recovery, a full **BSM1** muH recovery through the stiff recycled plant, and the
  finite-through-the-plant gradient in the fast gate).
  **Colored Jacobian** — the graph-coloring / Jacobian-sparsity subsystem
  (`_structural_plant_pattern` + the forward / adjoint / steady builders and their
  caches) lives in a [`ColoredJacobianManager`](aquakin/plant/colored.py)
  collaborator (`plant._colored`), mirroring `RecycleResolver`: it back-references
  the `Plant` for the state layout, unit couplings and RHS, and exposes
  `jacobian_solver` (forward implicit solve), `adjoint_jacobian_builder`
  (`stable_adjoint` backward `df/dy`) and `steady_jacobian_builder` (PTC steady
  `dF/dy`). All three share one probe→structural-union→build→guard scaffold
  (`_build_and_guard` / `_colored_from_probe`); each caches its result and
  `plant._colored.reset()` clears them when the state length changes (a
  temperature block). The size-based *decision* to use the colored backward
  (`Plant._COLORED_BACKWARD_MIN_STATES` / `colored_jacobian_decision`) stays on the
  plant's solve routing. The performance rationale is in `docs/plant_performance.md`.
  **Compiled-solve caches** — the jitted forward + PTC steady solves and the
  continuation / arclength kernels (all solver-run memoization, keyed by
  settings, that bake in the reactor conditions) live in a
  [`SolveCache`](aquakin/plant/solve_cache.py) value object (`plant._solve_cache`,
  `.jit` / `.steady_jit` / `.continuation_kernels` / `.arclength_kernels`; the
  `plant._jit_cache` / `_steady_jit_cache` properties are thin views). Every
  condition/topology mutator calls the single **`plant._solve_cache.invalidate()`**
  rather than hand-clearing named dicts — so a mutator can't clear one cache and
  leave another stale (`set_temperature` clears all four; the old code left the
  continuation/arclength kernels stale). `set_temperature_model` also changes the
  state length, so it additionally calls `plant._colored.reset()` (the structural
  colored builders), which `set_temperature` deliberately does not (its pattern
  stays valid).
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
    deep sweep to ~4e-15, its IFT tangent matches the deep-sweep gradient to
    ~6e-19, and a short BSM2 solve matches the fixed-pass trajectory to ~3e-8
    (within the solver tolerance — the adaptive path is the more-converged of the
    two). `recycle_tol` is read inside `_resolve_recycle_concentrations`, so it
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
    (`_check_recycle_map_constant`, set per instance) compares each map at two
    states and **falls back to per-RHS probing** for any topology whose `M` is
    genuinely state-coupled — so the optimization is safe for arbitrary plants.
    The cached map is built once from `params` by `_maybe_recycle_map` (the
    shared helper) and reused by **four** paths: the forward `jax_adjoint` solve,
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
    is the vjp's job). Both the cached jitted forward and the under-trace
    calibration-gradient path go through the shared `Plant._esdirk_stable_adjoint`.
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
    `M`), threaded into every RHS as `flow_map=` by `_maybe_flow_map`; the flow
    resolution then computes only `b` (one pass) + the cached `(I−A)` solve,
    skipping the `n` column probes. State-invariance is detected once by
    `_check_flow_map_constant` (compares `A` at two states; falls back to per-RHS
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
  - **Wiring API.** `plant.connect(source, dest)` takes two `"unit.port"`
    endpoint strings, read as `source -> dest`. The port may be omitted
    (bare `"unit"`) when the unit has exactly one port for that role — a
    single output (source) or single input (dest) — so only multi-port
    units (mixers/splitters/clarifiers) name a port.
    External influents are wired through
    `plant.add_influent(name, series, to="unit.port")` — they are *not*
    valid `connect` sources (a clear error redirects you). `connect`
    resolves the default `IdentityTranslator` when the two ends share a
    model and requires an explicit `translator=` across models (e.g.
    the BSM2 ASM1↔ADM1 digester edges). The endpoint parsing lives in
    `Plant._parse_endpoint`.
  - **Arbitrary add order; topological sort.** Units may be `add_unit`-ed in
    **any order**: `Plant._finalize_topology` (run from `_build_state_layout`,
    so at every solve) topologically sorts the feed-forward connection graph
    into the RHS evaluation order `_unit_order`, and the **recycles are the graph
    back-edges, detected automatically** — you no longer add the downstream unit
    before its upstream consumer or mark recycles by ordering. The sort is Kahn's
    algorithm with an insertion-order tie-break (deterministic, so the
    state/parameter layouts are stable): a connection carrying an explicit
    `initial_value` is a declared recycle and cut first (its seed rides the cut
    edge); any remaining cycle is broken by cutting the earliest-added remaining
    unit's still-active incoming edges, which become auto-detected recycles
    (zero-flow seeded via `_recycle_seeds`). For a plant already added in a valid
    order it **reproduces that order and recycle set exactly** (BSM1/BSM2
    unchanged); any valid feedback-arc cut gives the same converged solve, so the
    result is add-order-independent. `add_unit` records the raw add order in
    `_insertion_order` (the parameter-block order and the tie-break read it);
    `_unit_order` is the computed eval order. Recycles are then resolved by
    iterating the per-RHS stream computation 3 times (sufficient for typical BSM
    topologies). `initial_value=` on `connect` overrides a recycle's zero-flow
    seed with a non-zero warm start (e.g. the BSM2 temperature-carrying seed).
  - **Pre-solve wiring check.** `plant.check()` → `PlantCheck` reports **unfed
    input ports** (`.unfed_ports`, an error — the RHS sweep has no source for
    them) and **unconsumed outputs** (`.dangling_outputs`, info — a terminal
    stream like the final effluent / wasted sludge / disposal cake / biogas
    legitimately leaves the plant), plus the detected `.recycles`; `.ok` is true
    when nothing is unfed and `.summary()` prints it. `check(raise_on_error=True)`
    raises on an unfed port. Exported as `aquakin.PlantCheck`.
  - **Operating temperature.** `plant.set_temperature(celsius)` sets the static
    `T` condition of every temperature-bearing reactor in one call (°C → K),
    leaving a heated fixed-`T` unit like the digester untouched; clears the
    compiled-solve cache and returns `self`. See the seasonal-temperature notes
    below.
  - **Warm-starting.** `plant.initial_state(overrides={"tank1": vec, ...})`
    builds the flat initial-state vector with selected units' states replaced
    by name (each vector must match the unit's `state_size`) — the supported
    way to seed a plant (e.g. a healthy activated-sludge biomass before a slow
    digester settle) instead of reaching into the private `_state_layout`. Pass
    the result as `solve(y0=...)`. For the BSM plants the reference seed is
    shipped — **`bsm2_warm_start(plant)`** / **`bsm1_warm_start(plant)`**
    (`aquakin.plant.bsm`) return a ready flat `y0` with the five AS reactors
    seeded from the reference reactor composition (the dict constants
    `BSM2_WARM_REACTOR_COMPOSITION` / `BSM1_WARM_REACTOR_COMPOSITION`) and every
    other unit at its default. **BSM2 should always be warm-started**: the
    digester's ~19-day retention makes a cold start slow and stiff (the
    near-empty AS basin filling against the recycle loops can crawl or hit the
    step ceiling), and the warm seed removes that transient so only the digester
    has to settle. The reactor set and water-line model are auto-detected from
    the plant, so a single `bsm2_warm_start(plant)` replaces the
    seed-composition dict + tank list + `initial_state(overrides=…)` boilerplate
    the BSM2 scripts used to copy-paste. (The BSM2 composition is the validated
    reference reactor state; the BSM1 one is ~aquakin's BSM1 steady state. Both
    are *seeds* — the solve relaxes them — so the values affect settling speed,
    not the steady state.)
  - **Introspection — discover names instead of reading the builder source.**
    `plant.list_units()` lists the unit names (in add order); `plant.list_ports()`
    lists every `"unit.port"` **output** endpoint — the exact strings
    `plant.stream(sol, …)` accepts (pass `unit=` to scope, `role="input"` for the
    `connect`-destination endpoints); `plant.list_species(unit)` lists a
    concentration-vector unit's species (the valid `C_named` / `to_dataframe`
    columns). All three work **before** solving (plant structure) and raise a
    `KeyError` with a `difflib` "did you mean?" hint for an unknown name.
    `list_species` / `C_named` are restricted to units whose *state is a
    concentration vector* (`state_size == model.n_species`: the CSTRs, the
    primary clarifier holding tank, the digester) via `Plant._is_concentration_unit`
    — a stateless mixer/splitter/ideal-clarifier or the **layered Takács settler**
    (which carries a model but a non-species state) is rejected with a clear
    "read it as a stream with `plant.stream(...)`" message rather than an
    `IndexError`. `PlantSolution.available_streams()` is a convenience alias for
    `plant.list_ports()`, and `solution.C_named(unit, species)` now gives the same
    hinted errors (unknown unit, unknown species, non-concentration unit).
    `plant.activated_sludge_reactors(require_volume=True)` lists the AS reactor
    units (the CSTR/MBR `aeration`-carrying units, digester excluded; in plant
    order) — the single source of truth behind the warm-start / design-sizing /
    evaluation reactor heuristics (`require_volume=False` keeps every mechanically
    mixed reactor, e.g. for the mixing-energy term).
  - **Reading state back by unit.** `plant.states_by_unit(vec)` splits any flat
    plant vector into a `{unit_name: sub-vector}` map — the exact inverse of
    `initial_state(overrides=...)`. It works on a `y0`, a `PlantSolution.final_state`
    (the last save row, shape `(total_state_size,)`, so no opaque `[-1]` on the
    2-D `state` trajectory), or a `derivative` result. For a *trajectory* of one
    unit, `PlantSolution.unit_state(name)` returns `(n_t, unit.state_size)`.
  - **Evaluating the RHS once.** `plant.derivative(state, params=None, *, t=0.0)`
    is the public single evaluation of the assembled flowsheet RHS (`dstate/dt`,
    recycles resolved) — for inspecting the dynamics without a full solve. Same
    layout as `state`; split it with `states_by_unit`. (Wraps the private `_rhs`,
    building the layouts internally.)
  - **Effluent reconstruction (streams are recomputed, not stored).** The plant
    integrates unit *states*, not the inter-unit streams, so a stream such as the
    secondary-clarifier effluent is **recomputed on demand** from the saved
    states — it is *not* in the solution. `plant.stream(solution,
    "clarifier.overflow")` (or the convenience `solution.stream("effluent")`,
    plant carried on the solution) returns a `StreamSeries` (`t`, `Q`, `C` shape
    `(n_t, n_species)`, `model`, with a `C_named(species)` accessor) — feed it
    straight to `effluent_averages`. **The whole output sweep (every `(unit,
    port)`) is reconstructed in one `jax.vmap` pass over the saved times and
    cached on the solution** (`Plant._cached_streams`, keyed by the parameter
    vector via `_concrete_teval_key`; skipped under tracing), so a sequence of
    `stream` calls for different ports — or `evaluate_bsm*` reading ~8 streams —
    costs one reconstruction, not one per stream. The reconstruction is
    **vectorised**: each saved time's `_resolve_streams` sweep (a recycle-flow +
    concentration solve) is batched by `vmap` into a single XLA program rather
    than a Python loop of per-step sweeps — turning a long dynamic run's
    evaluation from minutes into seconds (a 609-day hourly evaluation drops from
    ~20 min to a few seconds). `evaluate_bsm2`'s digester-feed-temperature and
    closed-loop kLa histories are vmapped the same way. `plant.outputs_at(t, state, params=None)`
    is the single-instant primitive (returns `{(unit, port): Stream}`,
    uncached); both reuse the same `_resolve_streams` helper the RHS uses, so the
    reconstruction matches the integrated wiring exactly (including resolved
    recycle flows).
  - **Semantic stream shortcuts.** `plant.stream(sol, …)` also accepts an
    engineering **name** instead of a `"unit.port"` — the builders register a
    `named_streams` map (`plant.register_stream(name, endpoint)`,
    `plant.list_streams()`) so `plant.stream(sol, "effluent")` reads the right
    port without the user knowing it is `"tank5_split.internal_recycle"`. BSM1
    registers `effluent`/`ras`/`wastage`/`internal_recycle`; BSM2 adds
    `primary_effluent`/`primary_sludge`/`thickener_overflow`/`reject`/
    `dewatering_reject`/`disposal_sludge`, with `effluent` tracking the
    option-dependent `effluent_endpoint`. A misspelled name gives a hinted error
    listing `list_streams()`. `plant.effluent_stream(sol)` is the first-class
    shortcut for the most-read one (reads `effluent_endpoint`). The digester
    **biogas** is a *derived* output (computed from the ADM1 headspace state, not
    a material port), so it has its own accessor: `plant.digester_gas(sol)` →
    `DigesterGas` (`t`, `Q` m³/d the biogas flow **normalized to atmospheric
    pressure**, `p_ch4`/`p_co2`/`p_h2` bar, `ch4` kg/d, and
    `.methane_production()` time-averaged kg CH₄/d), reusing the OCI biogas
    formula (`evaluate_bsm2`'s `_methane_production` now delegates to it). Raises
    if the plant has no ADM1 digester.
  - **Results-level mass-balance closure — `plant.mass_balance(sol, …)` (#150).**
    The first thing an engineer does with a result: *does what went in equal what
    came out + what left as gas + what accumulated?* Returns a `MassBalance`
    (`aquakin.plant.balance`, exported as `aquakin.MassBalance` /
    `aquakin.ComponentBalance` / `aquakin.mass_balance`) with, per component (COD
    / N / P), the **inflow** (influents), **outflow** (terminal/dangling material
    streams — effluent, wasted sludge, disposal cake), **gas** (O₂ transferred in
    by aeration, the digester biogas, denitrification N₂ — computed from the
    aeration term and a reaction-production integral over the reactive units, with
    the digester deliberately excluded from the N gas term since it has no N gas
    phase) and **accumulation** (ΔInventory across every unit — reactor / clarifier
    / digester liquid+headspace at `V_liq`/`V_gas` / storage / Takács blanket).
    `_unit_inventory` **dispatches on unit-implemented contracts, never on private
    state layout** (#505): a unit whose state layout is non-trivial declares
    **`component_inventory(state, content, params)`** returning its own
    `{component: grams}` — the layered **`TakacsClarifier`** (blanket summed over
    layers, both `composition_mode`s + `soluble_holdup`) and the
    **`ADM1DigesterUnit`** (liquid states at `V_liq`, the three gas-headspace states
    at `V_gas`, via its own `_state_volume_vector(params)` which `_reaction_volume`
    also reuses for the reaction integral). A unit holding a single well-mixed
    liquid volume (`StorageTank` / `MBRUnit` / `SBRUnit`, whose states are
    `[C…, scalar(s)]`) instead declares that volume through **`liquid_volume(state)`**,
    so its inventory is `V·C`; a plain concentration-vector unit (`CSTRUnit` /
    `PrimaryClarifier`) falls through to the generic `volume·C`. This completes the
    inversion `liquid_volume` started — the balance no longer reaches into
    `_part_block_size` / `_n_part` / `param_index["V_gas"]` etc., and a unit with a
    new state representation participates by implementing `component_inventory`
    rather than editing the balance helper. Contract covered by
    `test_mass_balance_plant.py::test_stateful_units_own_their_component_inventory`.
    `imbalance = in − out − gas − accumulation` is the closure; `mb["N"]`,
    `mb.closed(rtol)`, `mb.summary()`, `mb[q].relative_imbalance`. Everything is on
    one canonical g basis (g COD / g N / g P), so the ASM water line (g/m³) and the
    ADM digester (kg/m³, kmol/m³) sum via `aquakin.composition_table` /
    `aquakin.canonical_content` (the shipped per-species COD/N/P content tables;
    `composition_table(net, electron_acceptor_cod=False)` = lab COD, the default
    `True` = the electron-equivalent convention `check_conservation` wants;
    `params=` reads a calibrated/BSM-specific composition such as `i_XB`). Closes
    BSM1 to ~1e-7 and BSM2 (two models, biogas, recycles) to COD ~0.08% / N
    ~0.03% at steady state; the gas integrals are exact at steady state and
    otherwise accurate to the `t_eval` sampling. **This is the tool that found the
    ADM1 nitrogen transcription error** (see the `adm1` model note).

Shipped units: `CSTRUnit` (kinetics + aeration), `IFASUnit` / `MBBRUnit`
(an IFAS/MBBR tank: a CSTR bulk coupled to a depth-resolved attached biofilm —
see below), `MBRUnit` (membrane bioreactor: a high-MLSS aerated reactor whose
membrane retains the solids into a near-solids-free permeate, with fouling/TMP —
see *Membrane bioreactor* below), `MixerUnit`,
`SplitterUnit`, `IdealClarifier` (fast, stateless separator),
`PrimaryClarifier` (BSM2 Otterpohl–Freund: a well-mixed holding tank split by
an HRT-dependent particulate-removal efficiency, fixed underflow `f_PS·Q`),
`IdealThickener` (BSM2 thickener / dewatering — a stateless ideal `%TSS`
separator, concentration-dependent underflow flow), `ADM1DigesterUnit`
(continuously-fed ADM1 CSTR with gas headspace, dilution masked to the liquid
states), `DosingUnit` (chemical dosing: injects a `Reagent` — a fixed
composition, e.g. metal salt / acid-base / external carbon — into a stream at a
fixed or feedback-controlled flow; see *Chemical dosing* below), `UVUnit` /
`ChlorineContactUnit` (disinfection: UV dose-response and chlorine CT /
log-removal — see *Disinfection* below), `SBRUnit`
(sequencing batch reactor: one tank cycling fill/react/settle/decant/idle with
variable volume and a pluggable settling model — see *Sequencing batch reactor*
below), and `TakacsClarifier` (10-layer 1-D Takács 1991 model). Its settling physics
are correct and verified in isolation at BSM1 solids loading: the
clarification-zone flux limiting (above the feed, the downward flux is
limited by the layer below only when that layer exceeds `X_threshold`) and
the per-species flux apportioning (each species settles at the bulk
velocity, `flux_tss · X_k/TSS`, conserving total settleable solids) produce
a monotone sludge blanket, a strongly clarified effluent, a thickened
underflow, and tight solids mass balance (verified to machine precision
against an independent port of the reference BSM1 settler derivative in
`tests/validation/test_takacs_vs_bsm1_reference.py`). `build_bsm1(use_takacs=
True)` selects it in the full plant (both clarifiers expose the same ports),
and `Plant.solve` takes `max_steps`. By default the **soluble** species are not
held in the settler — they pass straight through (overflow = underflow = feed,
no holdup), the common simplification. The opt-in **`soluble_holdup=True`** makes
each soluble a per-layer well-mixed state advected by the bulk flow (convection
only, no settling), so the clarifier's liquid volume (~`area·height`) damps the
soluble effluent signal — the BSM2 `settler1dv5` behaviour, which carries
`SNH_1..SNH_10` etc. per layer. The soluble holdup is a tail block of shape
`(n_layers, n_soluble)` appended to the state (so the particulate layout /
`state_size` are unchanged when off), orthogonal to `composition_mode`. **It
leaves every steady state unchanged** — a non-reacting soluble's only transport
is convection, whose fixed point is the uniform feed concentration (overflow =
underflow = feed), verified in `tests/integration/test_takacs.py` — so it only
matters under a dynamic influent, where it smooths the effluent ammonia
peaks/troughs. This is the structural cause of aquakin's wider dynamic-BSM2
effluent-NH4 distribution vs the reference: the reactors agree to corr 0.99 but
the pass-through settler does not damp the soluble signal the way BSM2's
soluble-carrying settler does (the JRN-056 dynamic validation). `build_bsm2(
settler_soluble_holdup=True)` enables it plant-wide.

