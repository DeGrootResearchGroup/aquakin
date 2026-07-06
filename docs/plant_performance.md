# Plant dynamic-solve performance

The stiffness-bound dynamic-solve regime and its levers: decoupled-Newton root
finder, `Kvaerno3`/`factormax`, colored Jacobian, cached recycle/flow maps,
`forward_fast`, PTC algebraic steady state, and the experiments that were tested
and rejected. Loaded on demand; the actionable plant guidance is in
`.claude/rules/plant.md`.


Profiling the long dynamic BSM2 run (the JRN-056 609-day simulation) established
that it is **stiffness-bound, not wasted-work-bound** — a finding worth not
re-discovering. The signature: ~750–1000 accepted steps/day, **step count nearly
invariant to `rtol`** (1e-4 vs 1e-3 → <2%), **~50% step rejection** with the
default solver, and per-step cost dominated by the **implicit Jacobian
factorisation of the 167-state plant** across the solver's stages (the raw RHS is
~4% of per-step cost). It extrapolates to ~38 min run-only for 609 days.

**Levers that do NOT help (measured — do not re-try):** `jump_ts` at the 15-min
influent kinks (the kinks aren't the bottleneck; a state-triggered clamp kink is
— see below); looser `rtol` (step count is tolerance-independent); PI-controller
tuning (`pcoeff`/`icoeff`) and `factormax` *alone* (they cut the rejection *rate*
but trade rejections for accepted steps → wall flat — reducing the rejection rate
is **not** itself the win); `recycle_passes` 3→1 (changes the answer ~2% via the
concentration-dependent reject loop — unsafe). The wall-time wins come from
**cheaper steps**, **fewer stages**, and **less stiffness**, not from chasing the
rejection rate.

**Verified speedups (20-day proxy, final-state agreement ≤ 6e-5 vs the old
default):** decoupled Newton tolerance **~18%**; `Kvaerno3` **~16%**; the two plus
`factormax=3` together **~42%**. These are exposed as two `Plant.solve` knobs and
one new default:

- **Default decoupled root finder (no opt-in).** `_run_diffeqsolve` now builds
  the default `Kvaerno5` with `root_finder=VeryChord(rtol=10·rtol, atol=10·atol)`
  — the per-stage **Newton** tolerance loosened 10× from the step tolerance.
  diffrax's stock Kvaerno root finder *copies* the controller tolerances, driving
  each stage solve to the full step accuracy (more Newton iterations than the
  embedded error estimate needs); the step controller still enforces the solution
  accuracy through `rtol`/`atol`, so this only ends each stage solve sooner —
  ~15–20% faster everywhere at preserved accuracy. Applies to **every** reactor
  and the forward `jax_adjoint` plant path (the shared `_run_diffeqsolve`); a
  user-supplied `solver=` is honoured verbatim (opts out of the loosening). The
  10× scale is off the *actual* `rtol`/`atol`, so it is correct for any model
  scale (mol/L ozone as well as g/m³ ASM/ADM); validated steady states are
  unchanged within their tolerances.
- **`Plant.solve(solver=...)`** overrides the integrator (`None` keeps the
  decoupled `Kvaerno5`). `diffrax.Kvaerno3` (4 stages vs 7) does less linear
  algebra per step. To keep the Newton decoupling with a custom order, pass it on
  the solver: `Kvaerno3(root_finder=VeryChord(rtol=10*rtol, atol=10*atol))`.
- **`Plant.solve(factormax=...)`** caps the `PIDController` per-step growth factor
  (diffrax default 10). On `Kvaerno3` the levers **stack** (unlike on `Kvaerno5`,
  where `factormax` cancels the Newton saving): `solver=Kvaerno3(...)` +
  `factormax=3` is the **~42%** config.

Both are threaded `Plant.solve` → `_build_jitted_solve` → `_run_diffeqsolve` and
keyed into the per-instance compiled-solve cache (`solver` **by class** — a fresh
stock instance shares the entry, a different class keys separately, a
custom-*configured* instance of an otherwise-default class shares the default's
entry; `factormax` by value). `events=` (the segmented solve) rejects them. They
are **also supported on `gradient="stable_adjoint"`**: the discrete adjoint builds
its backward from the forward solver's Butcher tableau generically, so a cheaper
4-stage `Kvaerno3` forward (with the matching backward recurrence) and a
`factormax` cap apply there too — the same optimized configuration the
`forward_fast` path uses, keyed into the stable-adjoint cache by solver class +
`factormax`. Like `dtmax=`, they do not change the `gradient="auto"` routing.
Covered by `tests/integration/test_plant_solver_option.py` and (the stable-adjoint
path) `tests/integration/test_plant_stable_adjoint.py`.

**Single source of truth for the integrator config — no drift between modes.**
The forward solve and the stable-adjoint *forward* pass used to construct their
solver + step controller independently, so the forward path's accumulated per-step
optimizations (the decoupled-Newton root finder, the colored Jacobian, the
`Kvaerno3`/`factormax` knobs) silently failed to reach the adjoint's forward pass —
it kept paying dense, full-Newton, 7-stage costs. Both now build from one pair of
helpers in [`integrate/_common.py`](aquakin/integrate/_common.py):
`build_implicit_solver(rtol, atol, order=, solver=, colored_root_finder=, linear_solver=, force_root_finder=)`
(the decoupled-Newton `Kvaerno` of the requested `order` — `5` default, `3` for the
lean / forward-sensitivity paths — built from the `_CANONICAL_SOLVERS` table, with
the colored `ColoredVeryChord` or the block-arrow `SimultaneousCorrector`
`linear_solver` injected when given) and
`build_step_controller(rtol, atol, factormax=, dtmax=)` (the PID core the forward
uses directly and the adjoint wraps in a `ClipStepSizeController`). So a future
per-step optimization lands in one place and reaches **both** modes. Specifically
the stable-adjoint forward pass now gets the **decoupled Newton** — previously only
the forward `jax_adjoint`/`forward_fast` paths had it. (The helpers also carry a
`colored_root_finder` so the adjoint *forward* chord could color its per-step
Jacobian too, and `esdirk_adjoint_solve` accepts it — but it is **not auto-enabled**:
the colored *backward* feeds J straight into the transposed solve and is exact on a
superset pattern, whereas the colored *forward* feeds J into an iterative chord
whose decoupled-Newton convergence point depends on the J approximation, so a
colored-vs-dense difference shifts the forward trajectory at the
~Newton-tolerance level (~1e-4) — which would break the bit-identical
`colored_jacobian=True` == dense invariant. It awaits the structural pattern being
reconciled so the colored and dense forward chords converge identically.) The
forward path's construction is unchanged (the helper reproduces it). A regression
guard,
`test_plant_stable_adjoint.py::test_forward_paths_agree_no_config_drift`, asserts
the `jax_adjoint`, `forward_fast`, and `stable_adjoint`-forward integrators realize
the **same primal trajectory** — so a future divergence in any one path's
configuration fails loudly.

**Every diffrax solve funnels through these helpers, structurally enforced.** The
forward-sensitivity reactor path and the plant's colored-solver method were the
last two paths still constructing a `Kvaerno5` / `PIDController` *directly* (the
former with a *tight*, controller-tied Newton — a real divergence from the
decoupled Newton everything else used; the latter hand-injecting the colored root
finder into a bare `Kvaerno5`). Both now build through `build_implicit_solver` /
`build_step_controller`: the augmented `[y; S]` forward-sensitivity solve passes
`order=` (5 reactors / 3 plant) + the `SimultaneousCorrector` `linear_solver`, and
the colored method passes `colored_root_finder=`. So the diffrax ESDIRK object is
constructed in exactly **one** place — the `_CANONICAL_SOLVERS` table inside
`build_implicit_solver` — the only legitimate variation being the explicit axes the
helper exposes (order, colored, `linear_solver`, factormax, dtmax) plus the one
escape hatch (`Plant.solve(solver=...)`, a user object honoured verbatim). The lone
*conceptual* exception is `forward_solve.py` (the `forward_fast` lean
`lax.while_loop`), which builds no diffrax solver at all — its Kvaerno3 tableau is
hand-rolled and must track diffrax's by hand. **A drift guard
([`tests/unit/test_solver_config_single_source.py`](tests/unit/test_solver_config_single_source.py))
AST-scans the package for any direct `Kvaerno*` / `PIDController` / `VeryChord` /
`with_stepsize_controller_tols` *call* outside an explicit allowlist (just
`_common.py`) and fails on a new one**, so a future solve path cannot silently
re-introduce the drift — it must route through the helpers or add an audited
allowlist entry (Python has no real access control, so this static AST lint is the
enforcement, backed by the runtime trajectory-agreement guard above). The
unification also switched the reactor forward-sensitivity from its tight Newton to
the decoupled Newton; the reactor forward-sensitivity suite passes unchanged at its
~1e-8 `jacfwd` tolerance, and the forward / discrete-adjoint paths are bit-identical
(`order=5` reproduces the previous `diffrax.Kvaerno5(root_finder=rf)`).

**`Plant.solve(colored_jacobian=True)` — sparse (colored-AD) Jacobian
materialisation ([`integrate/colored_jacobian.py`](aquakin/integrate/colored_jacobian.py)).**
Profiling the per-step *linear algebra* (after the decoupled-Newton + cached-map
wins) found the dominant cost is **forming the implicit Jacobian, not factorising
it**: diffrax's `VeryChord` materialises + factorises `I − γ·dt·J` **once per
step** (reused across stages/iterations), and for the 167-state plant the dense
materialisation (`jacfwd`, ~33% of a step-attempt) dwarfs the LU factor (~4%,
`cond ~10³` — well-conditioned, fast). But the plant Jacobian is **5–15%
nonzero** — dense per-unit kinetic blocks (the model stoichiometry × rate
dependencies) plus sparse inter-unit flow coupling — so it can be formed by
**column compression** (Curtis–Powell–Reid 1974): group structurally-orthogonal
columns (sharing no nonzero row) into *colors*, push one seed per color through a
single forward linearisation (`jax.linearize` once + `vmap` the tangent — **not**
`jax.jvp` per color, which redoes the expensive nonlinear primal — the recycle +
pH solves — every color), and scatter each color's JVP back to its columns via
the pattern. For BSM2 that is **~45 colors vs 167 columns**, set by the widest
dense block (the digester). The reconstructed matrix **equals the dense Jacobian**
when the pattern is a superset of the real nonzeros, so the chord iteration — the
step sequence, the trajectory, the gradient — is numerically unchanged; only the
formation cost drops. **Measured ~1.43× on the 14-day BSM2 solve** (trajectory
within integration tolerance, ~5e-3, the within-tolerance `LU`-vs-`AutoLinearSolver`
step-path drift; gradient finite and matching the dense path to ~1e-8). It
**stacks** with `Kvaerno3`/`factormax` (it helps any ESDIRK) and the cached map.
- `ColoredVeryChord(VeryChord)` overrides only `init` (materialise via colored
  forward AD into an `lx.MatrixLinearOperator` with an explicit `lx.LU()`,
  avoiding the `AutoLinearSolver(well_posed=None)` least-squares fallback a bare
  matrix would trigger); `step`/`terminate` are inherited, so the chord is
  identical.
- **Sparsity pattern** (`jacobian_sparsity_pattern`): the union of `|J|>tol` over
  **strictly-positive** probe states drawn at **two scales**, plus `y0` itself and
  the full diagonal. Two failure modes must both be covered and a single scale
  covers only one: (1) a *depleted* (zero-at-`y0`) component zeroes the entries
  that couple through it, so the probe lifts every component to `|y0|+1` and
  jitters to reveal those couplings (the *lifted* scale; missing it made an early
  prototype 6× *slower* via an 8× step explosion); (2) a *small-natural-scale*
  component — the ADM1 dissolved hydrogen `S_h2` sits at ~`1e-7` at its inhibition
  knee, where its Jacobian column is enormous — is pushed by that same `|y0|+1`
  lift into a **saturated**, flat regime where the steep column collapses below
  the relative threshold (set by the large biomass/settling entries) and is
  dropped, so the probe also jitters each component around its **own** magnitude
  (the *own* scale) to keep it in its physical regime. Including the Jacobian at
  `y0` makes the start-state guard pass by construction. *(This two-scale probe
  fixed a real fall-back: the BSM2 settler `soluble_holdup` states settle the
  digester to its operating point, surfacing the steep `S_h2` column that the
  lifted-only probe dropped — the colored path then fell back to dense; it now
  stays colored, ~2.6× over dense, matching to round-off.)*
- **Correctness model:** a pattern *miss* does **not** corrupt the result — the
  chord still converges to the stage residual's root — it only degrades
  convergence (costs steps, not accuracy). The pattern is therefore conservative,
  and **guarded once per plant** (`colored_jacobian_max_error`): the colored and
  dense Jacobians are compared at the start state and the solve **falls back to
  the dense solver with a warning** on any mismatch. Built concretely once and
  reused; a first solve under reverse-mode tracing also falls back (the probe
  needs concrete arrays — run one concrete solve to build it, then differentiate).
- Wired like `solver=`/`factormax=` on the forward `jax_adjoint` path (rejected
  with `events=`), keyed into the compiled-solve cache by a `colored_active` flag
  so it never collides with a plain solve. **Most worthwhile for a large stiff
  plant (BSM2); on small BSM1 the materialisation is not the bottleneck (≈1×, but
  still numerically matches to ~3e-13).** Covered by
  `tests/integration/test_colored_jacobian.py` (coloring/reconstruction math,
  positive-probe superset over the trajectory, colored==dense J, full-solve
  trajectory + gradient match, the guard/fallback on a truncated pattern, BSM2).
- **The IC probe goes STALE on a wide dynamic run — fixed by per-component
  structural couplings (issue #388, [`plant/coupling.py`](aquakin/plant/coupling.py)).**
  `jacobian_sparsity_pattern` probes `|J|` at the *start state*, so on a long
  dynamic BSM2 run it drops every coupling that is numerically tiny at the
  warm-start operating point but switches on once the influent drives the plant
  off it — saturated Monod kinetics, the Takacs settling velocity, the ASM<->ADM
  interface's nitrogen-budget branches. Those are the **stiff** couplings, so the
  stale pattern collapses the chord-Newton convergence: colored ran **~6×
  *slower* than no-colored** on the validated 244-state JRN-056 dynamic BSM2 (a
  convergence-rejection explosion — 17× the step attempts, 100% Newton-failure
  rejections). The fix is to build the pattern from the **equations, not a probe**:
  every stateful unit emits its structural Jacobian sparsity via the
  **`CouplingAware`** ABC's **`coupling_pattern()`** — a `self` block
  (`d rhs / d own state`) and an `inlet` block (`d rhs / d inlet concentration`).
  Reactors (`CSTRUnit`, `ADM1DigesterUnit`) derive `self` from the rate AST
  (`structural_sparsity_pattern`; a saturated Monod term is numerically invisible
  to a probe, so the *syntactic* dependency is needed) and `inlet` from the
  dilution diagonal; the `TakacsClarifier` derives both by AD over diverse solids
  profiles (`ad_union` — the settling law is a smooth nonlinearity whose branches
  a sample exercises, unlike Monod saturation); cross-model translators emit
  their own `coupling_pattern()` (`translator_coupling_pattern`, AD over the
  interface branches); stateless units are empty (the `StatelessUnit` default).
  `ColoredJacobianManager._structural_plant_pattern` (the plant's `plant._colored`
  collaborator, `aquakin/plant/colored.py`) assembles these — `self` blocks on the
  diagonal, each `inlet` block composed with the feeding stream's translator
  coupling on the off-diagonal — **unioned with the IC probe**, which supplies the
  linear, always-on couplings and the recycle's real block structure (so the
  off-diagonal placement is restricted to genuinely-coupled unit pairs, keeping
  the coloring tight). The result is a structural superset that **cannot go stale
  for any influent**, with **no trajectory sampling** (only the single
  always-available IC probe). On the JRN-056 dynamic BSM2 it turns colored from
  ~6× slower into **1.71× faster than no-colored** (49.8 s vs 85.3 s, 37 colors —
  tighter than even a trajectory-sampled pattern), with 0 within-unit couplings
  missing and the residual misses all `|J| ≲ 0.3` (negligible for convergence).
  Covered by `tests/integration/test_coupling_pattern.py` (the contract shapes,
  ABC enforcement, the settler/`ad_union` superset, and the assembled BSM1 plant
  pattern leaving no within-unit coupling missing along a trajectory). The
  reactive units present only in non-BSM plants — `MBRUnit`, `SBRUnit` and
  `IFASUnit`/`MBBRUnit` — **also emit `coupling_pattern()`** (issues #390–#392):
  the MBR is the CSTR's AST kinetics block plus a decoupled fouling-resistance
  diagonal; the SBR unions the AD-derived state couplings (the `1/V` convection
  and the settling-clarity dynamics) over its phases with the rate AST for the
  kinetics; the IFAS unions the (linear) soluble inter-layer diffusion +
  bulk-convection couplings from AD with the per-compartment rate AST (frozen
  layer-biomass rows dropped in the layers). So colored is staleness-free on
  those flowsheets too; the single-unit assembled-pattern superset is checked per
  unit in the same test.
- **Also colors the `gradient="stable_adjoint"` BACKWARD pass.** Since the
  saved-stage backward (above) reconstructs the stages instead of re-solving them
  by a 12-iteration Newton scan, the per-step `df/dy` builds dropped from ~79 to
  the **~7 stage `Js` only**. So the Jacobian builds — once **~82%** of the
  backward (with the dense solves ~17% and the parameter vjp ~1%, when the Newton
  recompute dominated) — are now only **~24%** (BSM2, estimated from the
  single-build colored ratio below); the rest is the transposed solves + parameter
  vjps + the stage reconstruction. **The bottleneck has shifted off the builds**,
  so coloring them — which still cuts each build's cost — now moves the total far
  less (the measured numbers below). `colored_jacobian=True` passes a colored
  builder into
  `esdirk_adjoint_solve` (`jacobian_builder=`), which builds each stage Jacobian
  in one JVP per *color* instead of per state. The coloring is derived once,
  concretely, for the **augmented** (`n+1`, time-carrying) primal rhs the discrete
  adjoint differentiates — `ColoredJacobianManager.adjoint_jacobian_builder`, the
  backward analogue of `jacobian_solver`, guarded by `colored_jacobian_max_error`
  with a dense fallback, cached in `plant._colored._adjoint_builder`. **Its sparsity
  pattern is the per-component structural pattern (issue #381).** The backward
  feeds `J` directly into `I − dt·γ·Jᵀ` and the transposed solve, so a missed
  coupling does **not** cost steps (as it does for the self-correcting forward
  chord) — it **silently corrupts the gradient**, undetected mid-transient by the
  start-state guard, and a start-state-only or trajectory-*sampled* pattern can
  miss a coupling that only activates at an unvisited correlated operating point.
  So the builder unions the IC probe with **`_structural_plant_pattern`** (each
  unit's equation-derived `coupling_pattern()` — the same complete assembly the
  forward path uses), embedded in the augmented `[y; τ]` layout's `df/dy` block
  (the probe supplies the always-on `τ` time-dependence column). A *complete*
  structural superset closes the silent-corruption risk a sampled pattern only
  reduces. The PTC steady-state builder (`ColoredJacobianManager.steady_jacobian_builder`)
  uses the same structural pattern (it marches in a narrow neighbourhood so the probe
  usually suffices, but the superset is complete regardless). Validated: the
  colored backward gradient w.r.t. a kinetic param **and** a flow-setpoint param
  (`underflow_split.ras`, where `dM/dθ ≠ 0`) matches the dense-Jacobian gradient,
  and the backward / PTC guards fall back to dense on a truncated pattern
  (`tests/integration/test_plant_stable_adjoint.py`, `test_colored_jacobian.py`).
  The dense default
  (`jacobian_builder=None`) is a trace-time branch, so it is bit-identical to the
  historic backward; the colored gradient equals the dense one (exact on the
  superset pattern — only float summation order differs: ~1e-15 on BSM1, ~6e-7 on
  BSM2 through the ADM1 pH-solver linearization, well inside the FD/`jax_adjoint`
  match envelope). **Re-measured on the saved-stage backward (BSM2/BSM1
  `value+grad`, 3-day warm-started span, dense vs colored backward): BSM2 backward
  `615 → 558 ms` (colored `0.91×`, ~9% faster — down from the ~1.95× of the
  Newton-recompute era); BSM1 backward `1105 → 1328 ms` (colored `1.20×`, now
  *slower* — its build overhead exceeds the saving at `n_colors=14` vs 65). With
  the builds no longer dominant, the colored win is marginal on BSM2 and negative
  on BSM1 — which is exactly what the `"auto"` decision (below) picks.**
- **`colored_jacobian="auto"` is the default — it measures whether the backward
  coloring pays and turns it on only then.** The GO/NO-GO reduces to "is the
  colored `df/dy` build cheaper than dense?" (the backward rebuilds it ~7×/step —
  ~24% of the backward — so the *sign* of the build-time difference is still the
  sign of the overall speedup, just a smaller magnitude than in the
  Newton-recompute era; the heuristic still picks correctly, enabling colored only
  when it helps). The
  decision **must** use *jitted* build times — eager timing is misleading because
  XLA fuses the colored build's scatter away (eager shows colored slower for both
  plants; jitted shows BSM1 `ratio=0.50`, BSM2 `ratio=1.65`). So on the first
  concrete `stable_adjoint` solve `_colored_build_speedup` jit-compiles the dense
  and colored builds once (a few-seconds one-time cost, cached, amortized over a
  calibration) and stores `ratio = t_dense/t_colored`; `"auto"` enables coloring
  iff `ratio > _COLORED_BACKWARD_MARGIN` (1.05). The `ratio` is a host-specific
  wall-clock measurement, so the *outcome* is not a portable invariant: a large
  stiff plant (BSM2) clears the margin robustly (colored ~9% faster on the
  backward), but a borderline-size plant (BSM1, whose dense and colored builds are
  close) lands on either side by machine — the dev box measures `ratio≈0.5`
  (dense), some CI hosts measure `>1` (colored), and **both are correct**, since
  the colored and dense gradients are equal on the superset pattern. So the auto
  decision is a per-host *performance* choice, never a tested outcome — the tests
  assert its self-consistency and gradient correctness, not which branch it picks.
  `plant.colored_jacobian_decision()` returns `("colored"|"dense", ratio)`. **`"auto"` governs only the
  `stable_adjoint` backward** — it leaves the **forward** `jax_adjoint` solve dense
  (forward coloring swaps the implicit linear solver and so is not guaranteed
  bit-identical; making it the all-solves default is a separate, full-suite-
  validated change). `colored_jacobian=True` forces coloring on **both** paths
  (skipping the measurement); `False` disables it. Covered by
  `test_plant_stable_adjoint.py` (`test_stable_adjoint_colored_jacobian_matches_dense`,
  the forced path; `test_stable_adjoint_colored_jacobian_auto_decision_is_consistent`,
  which asserts the auto decision's self-consistency and gradient correctness —
  machine-independent invariants — not the wall-clock outcome). These BSM1 colored
  tests are `@slow` (free runner, not the paid `heavy` larger runner): they are
  small-plant correctness checks, not the OOM-prone whole-plant BSM2 solves the
  `heavy` marker is for.

**`Plant.solve(forward_fast=True)` — lean non-AD forward integrator
([`integrate/forward_solve.py`](aquakin/integrate/forward_solve.py)).** A stiff
diffrax solve carries machinery whose purpose is to make the *whole solve*
differentiable — an optimistix root finder, a lineax linear-solve abstraction, and
a checkpointing reverse-mode adjoint (`custom_vjp`). **Tracing all of that
dominates compile time** (the implicit scaffolding traces ~10× slower than the
bare ODE loop — an explicit-solver solve of the same plant RHS traces in ~2 s vs
~30 s for the diffrax implicit solve; the RHS itself is ~0.2 s). A **forward-only**
plant solve — one that never needs `jax.grad`/`calibrate`/`sensitivity` of the
result — can skip all of it: `forward_solve` is a plain `lax.while_loop` running
the Kvaerno3 ESDIRK stages with a simplified Newton (Hairer–Wanner contraction
test) + a direct dense `lu_factor`/`lu_solve`, an embedded-error PI controller, and
a **convergence-aware step-growth limiter** (the Newton contraction rate caps the
step growth, so it rarely grows into nonlinear-divergence). Output at `t_eval` is
*exact* — the step is clipped to land on each save time (no dense-output
interpolation; for a dense `t_eval` this adds ~clip-per-save-interval steps, the
one cost vs diffrax's interpolation — a future dense-output refinement).
- **The per-step Jacobian `J = df/dy` is STILL colored forward-mode AD** (the same
  exact matrix the differentiable path forms, so the step behaviour matches). What
  is dropped is only the *adjoint over the whole solve*: the result is **not**
  differentiable w.r.t. parameters / initial conditions. So `J` uses AD locally;
  the solve is just not wrapped to be differentiable globally.
- **Opt-in, forward-only, concrete-only.** Rejected with `events=` and
  `gradient="stable_adjoint"`. It threads only `rtol` / `atol` /
  `integrator.max_steps`; a **non-default** integrator `solver` / `order` /
  `factormax` / `dtmax` / `colored_jacobian` is **rejected** (it runs its own
  hand-rolled Kvaerno3 with an internal colored Jacobian and the diffrax default
  step growth, so those knobs cannot reach it — rejecting beats silently dropping,
  #446). And it **requires concrete `params`/`y0`** (a `jax.grad`
  / `jax.jit` of a `forward_fast` solve raises a clear error — the `lax.while_loop`
  is not reverse-mode differentiable and the colored pattern needs concrete arrays
  to build). It needs the colored-Jacobian pattern; it builds + guards it like
  `colored_jacobian=True` and **falls back to the diffrax forward path (with a
  warning) if the guard fails**. Composes with the cached recycle map; its compiled
  solve is cached per instance (so a parameter sweep at fixed signature reuses it).
- **Measured: ~3× faster compile** (the implicit-machinery tracing collapses — the
  part that *cannot* be file-cached, since tracing is Python; standalone 14 s vs
  42–48 s) — the robust win, and the main reason to use it for long one-off dynamic
  runs (the 7-min full-BSM2 compile). **Run is ~1.3–1.9× faster** on the
  validated 244-state JRN-056 dynamic BSM2 (real 609-day influent, `HeatBalance` +
  `settler_soluble_holdup`): a 60-day window is 1.91× and the full 609-day run
  1.26× — the run gain narrows over a long run because the `t_eval` step-clipping
  adds a boundary per save point (8737 over days 245–609) where diffrax interpolates
  (a future dense-output refinement would recover it). Same accuracy (a valid
  solution to the same `rtol` — it differs from the diffrax trajectory only by the
  step-sequence variation between two valid adaptive solves, ~2e-2 on the dynamic
  BSM2). Covered by
  `tests/integration/test_forward_fast.py` (analytic decay + order, exact `t_eval`,
  BSM1/BSM2 agreement with diffrax, the guards). NOTE the lean integrator (Kvaerno3,
  3rd order) vs the diffrax default (Kvaerno5, 5th order) — both controlled to
  `rtol`, so equally `rtol`-accurate, K3 just takes more (cheaper) steps; a Kvaerno5
  forward_fast is a future option if a tighter match to the validated steady states
  is wanted.

**`S_h2` quasi-steady-state — TESTED AND REJECTED for our solver (issue #361).**
Every production WWTP simulator (BSM2 reference, GPS-X, WEST) makes the two
fastest ADM1 states — pH **and dissolved hydrogen `S_h2`** — algebraic, reporting
~18–28× (Rosen et al. 2006, Table 4.5). **But that win is an *explicit*-solver
(ODE45) benefit and does NOT transfer to our L-stable implicit `Kvaerno5`.** A
proof-of-concept confirmed it: the QSS equation is sound (monotone residual,
unique root, algebraic `S_h2` = 2.508e-7 vs the reference 2.506e-7, slow states
reproduced to 1e-10), but freezing `S_h2` at its exact QSS value via a smooth
Newton solver left the digester step count **unchanged** (801 vs 812 steps; 384
vs 389 rejections). The reason is fundamental: an L-stable implicit method
*already* performs the QSS implicitly — it damps the `S_h2` fast mode (eigenvalue
~1.4e6 d⁻¹) to its quasi-steady value at any step size, so removing it by hand is
redundant. (pH is different: it is not a fast *mode* but a state-derived
algebraic condition, which is why we solve it directly.) **Do not build the
`S_h2` DAE machinery** — it is multi-day, fragile, and zero-benefit here.

**The `clip_negative_states` `max(x,0)` kink — ALSO tested and rejected (issue
#361).** The hypothesis was that the hard clamp is a *state-triggered moving
derivative kink* near depleted species (DO in anoxic zones, depleted substrates)
that the embedded error estimator rejects at (and that, being state-triggered,
can't be a `jump_ts` breakpoint — why `jump_ts` did nothing). A direct test
replaced it with a smooth clamp `½(x+√(x²+ε²))` and swept `ε` over three orders
of magnitude on the BSM2 dynamic solve: the rejection rate stayed **pinned at
~50.7%** at every `ε` (even `ε=0.5`, which rewrites the entire sub-0.5 region of
the RHS), while the smooth clamp *biased* depleted species toward `ε/2`. So the
clip kink is **not** the rejection source either, and the smooth clamp is a net
negative (no speedup, worse accuracy).

**Diagnosis of the ~50% rejection — a controller property, and why it is not
worth chasing.** Across all experiments the rejection rate moved *only* for
*step-size-controller* changes — `factormax=3` (→39%) and PI coefficients (→31%)
— and never for RHS changes (tolerance, `jump_ts`, `S_h2` QSS, the clip kink).
It is the classic deadbeat-I-controller overshoot→reject→shrink oscillation on a
stiff forced system: diffrax's `PIDController` is a pure Söderlind error filter
with **no Gustafsson iteration-count-aware predictive control** (the standard
cure in RADAU/SDIRK codes, which diffrax lacks). Crucially, **lowering the
rejection rate this way does not lower wall time** — it trades rejected steps for
accepted ones. The wall-time wins are therefore *cheaper steps* (the decoupled
Newton default) and *fewer stages* (`Kvaerno3`), banked in the dynamic-solve
knobs above; the residual rejection rate is left as-is. A genuinely different
integrator (a Gustafsson-predictive controller, or an exponential/QSS-reduced
non-stiff formulation) is the only thing that would cut it further, and is a
large effort not currently justified by the already-shipped ~42%. See issue #361
for the full experiment log.

**IFAS / MBBR unit ([`plant/ifas.py`](aquakin/plant/ifas.py)).** `IFASUnit`
(alias `MBBRUnit`) places carrier-media biofilm in the flowsheet by **wiring the
existing depth-resolved `BiofilmReactor`** (1-D diffusion–reaction over biofilm
depth) into a plant unit, alongside the suspended (CSTR) fraction — the
intensification retrofit the BSM palette lacked. Its state is the bulk
concentration **plus** the biofilm layer profile (`(n_layers+1)·n_species`); its
`rhs` is `BiofilmReactor._make_rhs` (finite-volume bulk↔surface↔…↔wall soluble
diffusion + per-compartment reaction) with the **plant's bulk convection +
aeration added on the bulk row**, replacing the biofilm reactor's own
stand-alone CSTR feed (built with `feed=None`). Carrier geometry is the
designer's `specific_surface_area` (media SSA, m²/m³) × `fill_fraction` →
`area_per_volume`; oxygen enters the **bulk** and reaches the biofilm only by
diffusion (so deep layers can be O₂-limited — the reason for depth resolution).
The effluent is the well-mixed bulk; the biofilm stays on the carrier. Aeration
reuses the **same `Aeration` spec as `CSTRUnit`** (open- or closed-loop; the
plant's generic `_materialize_aeration` auto-wires a DO controller from the
spec), via aeration helpers (`build_aeration_vectors` / `aeration_transfer`)
factored out of `CSTRUnit` and shared by both (CSTR behaviour is bit-unchanged).
The biofilm is a **mature, fixed attached-biomass** model: the layers' biomass +
inert structure is held as a sustained reservoir while the substrate pools and
solubles react and diffuse and the suspended bulk fraction evolves fully. The
default freeze mask is **stoichiometry-derived** (`_default_biofilm_fixed_mask`):
freeze every particulate **except** a hydrolysis substrate (one consumed while a
soluble is produced, `XS→SS`) — freezing such a pool would make it a
non-depleting soluble source (the biofilm footgun), whereas biomass/inerts are
the intended structure. For ASM1 that freezes `XI`/`XB_H`/`XB_A`/`XP` and leaves
`XS`/`XND` dynamic. **Validated:** at equal volume + aeration an IFAS tank
removes more soluble COD (effluent SS 1.8 vs 2.7) and nitrifies markedly more
(SNH 0.4 vs 2.6) than a plain CSTR, converges to steady state, and `jax.grad`
flows end-to-end through the biofilm core (`tests/integration/test_ifas.py`). A
fully dynamic biofilm (growth with attachment/detachment/a density cap) is the
underlying `BiofilmReactor`'s domain and a follow-up for the unit.

**Coupled BSM1 — steady state now works.** The *coupled* BSM1 plant reaches the
correct steady state for **both** clarifiers (Takács and Ideal agree: tank-5
XB_H ≈ 1.7e3, SNH ≈ 0.5, healthy nitrification) in ~10 s. Getting there took
three fixes, all diagnosed against the official BSM1 reference code:
- **Decoupled recycle-flow resolution** (`Plant._resolve_flows`): the recycle
  *flow* model is linear and concentration-independent, but BSM1's loop gain
  is ≈0.99 (3× internal + 1× RAS), so the old 3-pass Gauss–Seidel left the
  flows at ~40% of steady → starved underflow → washout. Each unit now exposes a
  `flow_outputs` rule; `Plant` solves the small flow fixed point **exactly**
  (probe the affine map, one `lineax`/`jnp.linalg.solve`), then runs the
  concentration sweep on the fixed flows. This was the keystone. **Affinity is
  checked:** the probe is exact only if every `flow_outputs` is affine in the
  recycle flows, which a threshold-mode `SplitterUnit` / `StorageTank` bypass is
  *not* (piecewise-linear, a kink). On the first non-traced `solve` the plant
  re-evaluates the forward pass at the solved recycle flows and `warnings.warn`s
  if it does not reproduce them (`Plant._warn_if_flow_nonaffine`) — the residual
  is exactly the affine-violation indicator, so it fires only when a
  recycle-dependent inlet actually crosses a kink (no false positives: the shipped
  BSM2 bypass/storage plants, whose such units are fed by the influent / a fixed
  pump, never warn). It is a warning, not an error: a kink the flow never crosses
  in operation is still correct.
- **Non-negative flow split** (the remainder outflow clamped into `[0, Q_in]` in
  both clarifiers): guards against a negative underflow when the feed dips below
  the design split — closes issue #17; inactive at steady state.
- **`clip_negative_states`** on ASM1 (the reference `xtemp = max(x,0)` clamp).

`Plant.solve` takes an optional `y0=` for warm-starting (e.g. a dynamic run
from a precomputed steady state).

**Reaching steady state — `plant.run_to_steady_state(...)`.** A single continuous
adaptive solve that **self-terminates** at steady state via diffrax's
`steady_state_event` (halts when `||dstate/dt|| <= ss_atol + ss_rtol*||state||`,
the standard march-to-steady-state criterion) — no fixed horizon to guess and no
chunked re-integration; `max_time` is only a safety cap (reached ⇒
`converged=False`). Returns a `SteadyStateResult(state, converged, time,
solution)`. Implemented by threading an `event=` argument through
`Plant.solve` → `_run_diffeqsolve` → `diffeqsolve` (forward `jax_adjoint` path
only; rejected under `stable_adjoint`). Warm-started BSM2 settles in ~51 d /
~25 s reproducing the validated steady state.

**Algebraic steady state — `plant.steady_state(...)` (pseudo-transient continuation).**
A fast, robust, *differentiable* alternative to the forward solve: it finds the
root of the plant RHS `F(y)=dy/dt=0` directly by **pseudo-transient continuation
(PTC)** rather than integrating until the dynamics die out. The core lives in
[`plant/steady.py`](aquakin/plant/steady.py) (`solve_steady_state` / `ptc_forward`)
and is reusable on any `rhs(y, params)` — `BiofilmReactor.steady_state` is also
routed through it (replacing the Levenberg–Marquardt root-find that stalled). PTC takes damped-Newton steps
`(V/δ − J)·Δy = F(y)` with the exact AD Jacobian `J = ∂F/∂y` (forward-mode) and a
**per-state** pseudo-time `V/δ`, `V = diag(max(|y|, floor))`: at small `δ` the
step is a stable backward-Euler move along the physical transient (globally
convergent, like time-stepping — the regime where a plain Newton root-find
stalls), and as the Switched-Evolution-Relaxation ramp `δ ← δ·min(cap,
‖F_old‖/‖F_new‖)` grows `δ` the term vanishes and it becomes Newton (quadratic
terminal convergence). The per-state `V` is essential — plant states span orders
of magnitude (DO ~2, heterotrophs ~2000, gas ~1e-3) and a scalar `I/δ` thrashes.
This is the standard method for "forward integration converges but Newton stalls"
stiff systems (Kelley–Keyes 1998; the flowsheet form is Pattison–Baldea 2014) and
is what production simulators use to snap to steady state on any topology.
- **Validated:** BSM1 (75 iters) and BSM2 (the 167-state plant with the long-SRT
  digester — the stiff case a plain root-find stalls on; 85 iters) both reach the
  forward-integration steady state to within ~1–3% on every key state, **~10×
  faster** than `run_to_steady_state`, to a tighter residual.
- **Differentiable in BOTH AD directions** for design sweeps and sensitivity: the
  returned `state` carries the **implicit-function-theorem** parameter
  sensitivity (the iteration — a `while_loop` — is gradient-blocked; the
  sensitivity is re-attached by a **`custom_jvp`** that gives the forward tangent
  `dy = −J⁻¹(∂F/∂params)·dθ`). Because that map is *linear in the tangent*, JAX
  transposes it automatically to the reverse gradient `−(∂F/∂params)ᵀJ⁻ᵀḡ`, so the
  one rule serves **forward** (`jax.jvp`/`jacfwd` — the many-output
  sensitivity-screen direction) and **reverse** (`jax.grad`/`jacrev` — the
  calibration-gradient direction) alike. (It was a reverse-only `custom_vjp`;
  the `custom_jvp` is what unblocks forward-mode AD and `dgsm(ad_mode="forward")`
  through `plant.steady_state`.) `J = ∂F/∂y` is full rank for the shipped models
  at their operating point, where the `jnp.linalg.solve` is exact; a rank-deficient
  `J` (a fully dormant species) leaves the IFT sensitivity undefined along that
  null direction (the old `lstsq` returned an arbitrary min-norm cotangent there
  rather than the exact gradient, so it is not used). Verified: forward == reverse
  to machine precision and both match finite differences (`tests/integration/test_steady_state.py`).
- **`plant.steady_state_sensitivity(params, *, output_fn=, wrt=, mode=, elasticity=)`** —
  the exact steady-state output sensitivity `d(output)/dθ` from the IFT, **far
  cheaper than `jacfwd`/`jacrev` through `steady_state`** (which re-solves per
  call): it solves the steady state once and reuses a single `∂F/∂y` factorisation
  for every output and parameter. `output_fn` maps the flat plant state to a
  length-`m` output vector (default: the full state, giving `dy*/dθ`). `wrt`
  selects the parameters to differentiate (flat indices or `"<model>.<param>"`
  names; default all) — restricting to `k` parameters makes forward mode cost `k`
  solves rather than `n_params`. `mode` selects the AD direction — `"forward"`
  (one solve per parameter, all outputs follow; efficient when outputs outnumber
  parameters), `"reverse"` (one transposed solve + VJP per output, all parameters
  follow; efficient when parameters outnumber outputs), or `"auto"` (forward iff
  `k ≤ m`). Both give the same exact sensitivity; `elasticity=True` returns the
  dimensionless `(dg/dθ)(θ/g)`. This is the general form of the plant-scale
  sensitivity screen.
- **`plant.steady_state_dgsm(ranges, *, output_fn=, wrt=, mode=, n_samples=, seed=, cond_factor=)`**
  — **global** sensitivity (DGSM) of the steady state: samples the screened
  parameters over their ranges (scrambled-Sobol QMC), solves the steady state at
  each sample, and reads each output's sensitivity through
  `steady_state_sensitivity` — reusing **one** `∂F/∂y` factorisation per sample, so
  it is far cheaper than the generic `aquakin.dgsm` over `steady_state` (whose
  `jacfwd`/`jacrev` recompute the steady-state structure per input tangent /
  output). Aggregates to the Sobol total-index upper bound
  `S_ij^tot ≤ ν_ij(b_j−a_j)²/(π²Var(g_i))`, `ν_ij = E[(∂g_i/∂z_j)²]`, returning a
  `SteadyStateDGSMResult` (`sobol_total_bound`/`std_error` shape `(m, k)`,
  `.ranked(output)`). Non-finite samples are dropped per output exactly as
  `aquakin.dgsm` does, so with `cond_factor=None` (default) the bounds are
  **bit-identical to `aquakin.dgsm`** (same Sobol seed → same points → same
  formula), just computed more cheaply. **`cond_factor`** adds the
  heavy-tail robustification a stiff plant needs: a steady-state sensitivity
  `−J⁻¹(…)` is only well-defined at a hyperbolic operating point, but over a wide
  parameter screen many Sobol samples land near plant bifurcations (washout,
  nitrification collapse) where `∂F/∂y` is near-singular and the sensitivity blows
  up (finite but huge), giving the DGSM a heavy tail the Monte-Carlo mean cannot
  resolve (it spikes and fails to reach `1/√N`); `cond_factor` drops any sample
  whose Jacobian condition number exceeds `cond_factor ×` the sample median — a
  near-singular operating point — restoring finite variance and clean `1/√N`
  convergence (the condition number is recorded per sample in `result.cond`).
  It **retains the per-sample data**, so `result.convergence()` returns the running
  bound + MC standard error versus sample count — the **sample-size convergence
  study** with no re-solving — and `result.with_cond_factor(c)` re-applies a
  different threshold (re-aggregating from the retained data, no re-solve).
  (`tests/integration/test_steady_state.py`.)
- **Dynamic (transient) sensitivity — `plant.dynamic_sensitivity(params, *, output_fn=, t_span=, t_eval=, wrt=, mode=, elasticity=)` and `plant.dynamic_dgsm(ranges, *, output_fn=, t_span=, t_eval=, wrt=, mode=, n_samples=, seed=)`.**
  The dynamic counterparts of the steady-state pair, for an output that depends on
  the *trajectory* (an effluent time series, a window average, a peak) rather than
  the operating point. There is no implicit-function-theorem shortcut here, so the
  cost is one stiff solve per direction (sensitivity) or per sample (DGSM), far
  heavier than the steady-state IFT. The wrapper's value is **using the stable
  method for each AD direction**, the easy thing to get wrong by hand — a naive
  differentiation of `plant.solve` is non-finite on a stiff plant. `mode="reverse"`
  differentiates the solve through the cap-free `gradient="stable_adjoint"` (one
  `jax.vjp`). `mode="forward"` integrates the augmented `[y; S]` variational system
  (`plant.solve_sensitivity`), whose step controller bounds `S` so it stays
  **finite over long horizons** where forward-mode `jacfwd` through the stiff solve
  goes non-finite (a numerical, not genuine, blow-up — the true sensitivity stays
  bounded), then chains the full-state sensitivity through `output_fn` with one
  `jax.linearize` of the output map over the saved trajectory (no extra solve).
  `output_fn` maps the `PlantSolution` to a length-`m` output vector. The reverse
  solve is differentiated directly (no enclosing jit — the primal must run with
  concrete params, since some plant setup, e.g. a unit `initial_state`,
  concretizes; an outer jit makes the BSM2 dynamic plant fail with a
  `ConcretizationTypeError`); the solve's own compiled-solve cache still reuses the
  integrator compile. **Forward is the memory-light direction over a long horizon**
  — `solve_sensitivity` carries the parameter tangents in lockstep with the state,
  so memory is independent of the integration length, whereas reverse stores the
  whole trajectory to replay it (prohibitive over a 609-day horizon). Both
  directions go through the shared `Plant._dynamic_value_jac` helper, so
  `dynamic_dgsm`'s per-sample screen inherits the same stable forward/reverse.
  `dynamic_dgsm` reuses the per-sample sensitivity into a Sobol total-index screen
  returning a `DynamicDGSMResult`
  (mirroring `SteadyStateDGSMResult`: `.ranked()`, `.convergence()`); verified
  forward == reverse, the reverse sensitivity matches a manual `stable_adjoint`
  gradient to machine precision, `dynamic_dgsm` matches `aquakin.dgsm` over the
  same transient solve, and `solve_sensitivity` matches `jacfwd` where both are
  finite (`tests/integration/test_dynamic_sensitivity.py`). The steady-state pair
  stays the cheap, both-directions-free path (the IFT); the dynamic pair is the
  convenience layer over the stable differentiation of `plant.solve`.
- **Design variables** (`steady_state(..., design=...)`): because the IFT
  differentiates w.r.t. *whatever pytree the residual consumes*, the steady state
  is differentiable w.r.t. design variables, not only kinetic parameters, by
  folding them into `θ = (params, design)`. **Influent load** is wired:
  `design={"influent": {port: {"Q": ..., "C": ..., "T": ...}}}` (plain arrays —
  a `Stream` can't be a θ leaf, it carries the non-JAX `model`) overrides the
  recorded influent at `influent_time` inside `_resolve_streams`/`_resolve_flows`,
  so `jax.grad` of a steady-state output w.r.t. the influent composition/flow
  works (BSM1 `d(effluent NH)/d(influent NH)` matches FD).
- **Flow setpoints as first-class parameters (the SRT / recycle knobs).** A flow
  setpoint — a recycle / wastage pump flow, a clarifier underflow, the primary
  sludge fraction — is consumed in **two** code paths (`_resolve_flows` →
  `flow_outputs` and `_sweep_outputs` → `compute_outputs`, which recompute the
  split). [`plant/flow_setpoint.py`](aquakin/plant/flow_setpoint.py)'s
  `FlowSetpoint` is the single source of truth: both paths call
  `resolve(flow_params)` on the same object, so they cannot desync, and the value
  is read from the unit's slice of the **parameter vector** (which both
  `flow_outputs` and `compute_outputs` already receive as `params_unit`) — making
  it differentiable everywhere (steady-state IFT *and* dynamic solves) with no
  Protocol change. `_build_parameter_layout` **appends** a per-unit flow-setpoint
  block after the kinetic model blocks (so kinetic indices are unchanged); the
  setpoints are addressed by name `"<unit>.<setpoint>"` (e.g.
  `"underflow_split.ras"`, `"clarifier.underflow_Q"`, `"primary.f_PS"`). The
  `FlowParameterized` mixin (on `SplitterUnit`, `IdealClarifier`,
  `TakacsClarifier`, `PrimaryClarifier`) provides the resolution; a unit used
  standalone (no plant) resolves the default, so it is unchanged. **Backward
  compatible:** a kinetic-only parameter vector (the pre-flow convention, e.g.
  `bsm2_parameters`) is padded with the default flow setpoints by
  `Plant._coerce_params`. Validated: BSM1 `d(effluent NH)/d(RAS flow)` matches FD
  (and is negative — more recycle retains biomass, lowering effluent ammonia).
  *Not* a flow setpoint: the thickener/dewatering underflow is
  concentration-derived (`%TSS` target), so `IdealThickener` is left as-is.
- Returns the same `SteadyStateResult` (now `method="ptc"`, with `iterations`
  and the scaled `residual`; `time`/`solution` are `None`). Eager calls get
  concrete diagnostics and, if PTC fails to converge within `max_iter`, an
  automatic **fallback** to `run_to_steady_state` (`method="ptc->forward"`);
  under a `jit`/`grad` trace the diagnostics are traced values and the fallback
  is skipped (only the differentiable `state` is used there). Constant influent
  is assumed (the residual samples the influent at `influent_time`, default 0).
- **Layered algebraic fallback — continuation and pseudo-arclength before the
  forward backstop (`continuation_from=`, `arclength=True`).** A direct PTC solve
  from a fixed warm start fails for a parameter set far enough that the start is
  out of basin, or whose operating point is near-singular (close to a
  bifurcation). Rather than fall straight to the slow time-integration backstop,
  `steady_state` tries two cheap *algebraic* fallbacks that deform from a **known
  nearby solution** `continuation_from=(params_known, y_known)` (for a sweep, the
  nominal operating point): (1) **natural-parameter continuation**
  ([`continuation_solve`](aquakin/plant/steady.py)) — a predictor-corrector that
  steps the parameters from the known set to the target, the IFT tangent
  (`−J⁻¹∂F/∂θ`, free from AD) as the Euler predictor and PTC as the corrector, with
  an adaptive step (`method="continuation"`); then, if that stalls, (2)
  **pseudo-arclength continuation**
  ([`arclength_continuation_solve`](aquakin/plant/steady.py)) — Keller's *scaled*
  arclength tracking (in `z = y/scale`, so the large biomass and tiny gas states
  contribute comparably) with a **fold-regularizing augmented corrector**
  `A = [[∂F/∂y·diag(scale), ∂F/∂s], [tangentᵀ]]`, which is non-singular even where
  `∂F/∂y` is singular (the fold / soft direction), so it reaches operating points
  behind a near-singular Jacobian where PTC overshoots (`method="arclength"`). The
  chain is **PTC → continuation → arclength → forward**: a permissive, wide-basin
  PTC for the common case (the `divergence_factor` bound is deliberately wide
  *because* these cheap fallbacks exist), algebraic deformation for the
  out-of-basin / near-fold cases, and time-integration only for the irreducible
  few. Validated on the BSM2 off-nominal Sobol screen: pseudo-arclength solves
  near-washout operating points to machine precision (1e-10) that PTC, a
  `dt`-capped PTC, **and** a 108-day forward solve all wall at ~5e-2.
- **Existence classification — operating point vs "past the fold"
  (`operating_point_exists`).** Tracking the branch by arclength does more than
  solve: it **detects when the operating branch folds before the target** — the
  continuation parameter `s` stops increasing and the tangent's `s`-velocity
  *reverses sign* (a saddle-node bifurcation). There, **no operating-branch steady
  state exists** at `params_target`: the parameters are past the survival limit
  (e.g. digester acetoclastic-methanogen washout) and only a different branch
  (washout) exists. That returns `method="past_fold"`, `converged=False`, and the
  new `SteadyStateResult.operating_point_exists=False`. The test is a *true* `ts`
  sign reversal, not merely a small `ts` — `ts` dips and **recovers** in the
  high-sensitivity regions of a perfectly reachable (marginally-stable) branch
  whose operating point *does* exist, so a small-`ts` heuristic would wrongly
  exclude real operating points. The few genuinely-near-fold cases the arclength
  cannot resolve within its step budget fall through to the forward backstop
  (`method="ptc->forward"`, `operating_point_exists=None`) — conservatively
  *included*, since they cannot be *confirmed* past-fold.
- **Fold-based operating-regime exclusion in the screen
  (`steady_state_dgsm`).** The DGSM solves each sample through the layered chain
  and **excludes the `past_fold` samples** (no operating point) from the
  aggregation, `convergence()`, and `with_cond_factor()`, recording the per-sample
  `solve_method` and `operating_point_exists` on the result. This is the
  *physical* operating-regime boundary — does a stable operating point exist? —
  replacing the conditioning heuristic for those samples: a near-bifurcation
  sample whose operating point genuinely exists (marginally stable, large *real*
  sensitivity) is **kept**; only the non-existent (past-fold) ones are dropped.
  This refines the heavy-tail diagnosis: a forward-backstop sample with a huge
  sensitivity was either a real marginally-stable operating point (keep) or a
  washout state past the fold (exclude), which the arclength distinguishes where
  `cond(J)` — baseline ~1e10 from state scaling, not a fold signal — could not.
- **Step-acceptance guard (the robustness lever, `divergence_factor`).** PTC is
  legitimately **non-monotone** — a healthy step can spike the scaled residual
  (~20–30× on the BSM plants) and recover, so the ramp must accept those. But a
  Newton step from far off the solution (a cold start) can overshoot into a bad
  region where the residual blows up by orders of magnitude and then goes
  non-finite — the accept-always iteration runs to **NaN**. `ptc_forward` now
  **rejects** a non-finite or grossly-diverging step (scaled-residual growth past
  the generous `divergence_factor`, default `1000`): it **holds the iterate and
  hard-shrinks `dt`** (×0.1, floored) so the retry is a stabler backward-Euler
  step. The threshold sits in the **wide gap between benign (~30×) and
  catastrophic (~1e5×) growth**, so a converging run **never rejects and is
  bit-identical** to the unguarded iteration (BSM1/BSM2 warm: same 23/38
  iterations), while a divergent one is pulled back. **Measured:** BSM2 from a
  *cold* `initial_state` went to NaN before; it now stays finite and **converges**
  (~450 PTC iterations on the dev machine — the cold-start *count* is numerically
  platform-sensitive, so it is not asserted in CI). The growth guard is what
  rescues it, not merely catching NaN: a non-finite-only guard
  (`divergence_factor=inf`) accepts the finite blow-ups and stalls. The guard is
  in the core `ptc_forward`, so it also hardens `BiofilmReactor.steady_state` and
  direct callers, identity on the happy path. Regressions (both fast +
  deterministic — no brittle convergence count):
  `test_steady_state.py::test_ptc_step_guard_keeps_overshoot_finite` (an overshoot
  to a non-finite region is rescued to convergence) and
  `::test_ptc_step_guard_rejects_finite_blowup` (a large *finite* residual blow-up
  is rejected with the default `divergence_factor` but accepted with `inf`,
  checking the acceptance logic directly).
- **Per-state pseudo-time / residual scaling (the iteration-count lever).** PTC's
  step damping `V` and convergence criterion both use `max(|y|, scale_floor)`. A
  flat scalar floor (the old default `1.0`) **over-damps the small-magnitude
  states** (gas fractions ~1e-3, dissolved hydrogen ~1e-7): their relative rate is
  throttled, which throttles the SER `dt`-ramp and roughly **doubles** the
  iteration count. `plant.steady_state` now defaults `scale_floor` to a
  **per-state** floor `max(|y0|, 1e-6)` — each state scaled by its own warm-start
  magnitude — so every state has a magnitude-consistent pseudo-time. Measured on
  **BSM2: 80 → 38 PTC iterations (run-only 39.4 → 18.9 ms, ~2.1×), same root
  (rel ≤ 7e-6)**; neutral on BSM1 (24 → 23). The small `1e-6` absolute floor
  anchors near-zero states — a *pure* `|y|` relative scale (no floor) is faster
  still on BSM2 (~44 it) but **destabilises BSM1** (~280 it), so the `|y0|`-anchored
  floor is the robust choice. The win is in the **run/amortized** time (a jitted
  design sweep, calibration); the un-jitted one-shot `steady_state` wall is
  compile-bound, so its time is roughly unchanged. The change is confined to
  `plant.steady_state`'s default — `ptc_forward` / `solve_steady_state` keep the
  scalar `scale_floor=1.0` default (so `BiofilmReactor.steady_state` and direct
  callers are unchanged), and an explicit `scale_floor` (scalar or per-state
  array) is always honoured. `scale_floor` only affects the path and the
  convergence criterion, never the root. (The backtracking line-search PTC below
  now does much of the per-step size control the per-state floor used to dominate,
  so the per-state *iteration* advantage is narrower than the original 80 → 38 and
  platform-sensitive at the few-iteration level; the regression therefore guards
  the stable invariants — both floors converge to the *same* root and per-state
  scaling is not a material iteration regression — not a strict iteration win.)
  Regression: `test_steady_state.py::
  test_bsm2_steady_state_per_state_scaling_converges_competitively`.
- **Compiled-solve cache — the single-run-compile lever (`Plant._steady_jit_cache`).**
  A one-shot `steady_state` is **~99% compilation** (BSM2: ~12 s compile of the
  plant-RHS `jacfwd` inside the PTC `while_loop`, vs ~40 ms of actual solving),
  and the eager `jax.lax.while_loop` in `ptc_forward` **re-traces and recompiles
  on every call** — so before this, a *repeated* `steady_state` (a temperature /
  SRT sweep, multistart, regenerating a figure) paid the full ~12–17 s each time.
  `plant.steady_state` now **persists a jitted forward solver** keyed by the PTC
  settings (`dt0`/`dt_max`/`growth_cap`/`max_iter`/`tol`/`nonneg`/`influent_time`)
  and reuses it, so JAX skips the recompile: **BSM2 call 1 ≈ 15.6 s, call 2 ≈
  0.02 s (~780×), and a swept-`params` call is also ~0.02 s** (the `rhs` reads
  `params` as a jit *argument* and recomputes the recycle map inside, so one
  compiled solver is correct for any params — and `y0`-derived `scale_floor` is an
  argument too, so a varying warm start does not recompile). Cached **only on the
  dense, design-free, concrete path**: the colored primal bakes a params-derived
  recycle map, the `design=` path differentiates a pytree, and a traced (gradient)
  call needs the IFT `custom_vjp` — those keep `solve_steady_state` (the gradient
  is amortized by the caller's own `jit`). The cached path returns the converged
  state directly (no IFT wrapper — a concrete call takes no gradient) and still
  honours the non-convergence fallback to `run_to_steady_state`. The cache is
  cleared by `set_temperature` / `set_temperature_model` (they change the RHS /
  state size). **This does NOT speed up the *first* (one-shot) call — that compile
  is irreducible here — only repeated calls.** Regression: `test_steady_state.py::
  test_bsm1_steady_state_solve_is_cached` (one entry, reused across params,
  bit-identical re-call, gradient path bypasses it and stays finite).
- **`steady_state(..., colored_jacobian=True)` — sparse (colored-AD) PTC
  Jacobian.** PTC forms the full plant `dF/dy` every Newton step (~tens of times
  for BSM2), the *same* block-sparse object the integrator's implicit-stage
  (`Plant.solve(colored_jacobian=True)`) and the `stable_adjoint` backward color.
  This flag materializes it by column compression (one Jacobian-vector product
  per color — BSM2 46 colors vs 167 states) instead of dense `jax.jacfwd`,
  reconstructing the same matrix on the sparsity-pattern support: **bit-identical
  to dense on a single-model plant** (BSM1, the recycle reconstruction is
  exact) and **identical to PTC tolerance (~1e-7) on a multi-model plant**
  (BSM2 — the colored `linearize`+vmap materialization orders the recycle
  linear-solve arithmetic differently from dense `jacfwd`, a round-off difference
  well inside the 1e-6 convergence tolerance; same 83 iterations). The injection
  point is `ptc_forward`/`solve_steady_state`'s new `jac_fn=(F, y) -> dF/dy`
  argument; `Plant.steady_state` builds the colored materializer once concretely
  (`ColoredJacobianManager.steady_jacobian_builder`, reusing the `colored_jacobian` module's
  pattern/coloring) and **guards** it against the dense Jacobian at the warm
  start, **falling back to dense** on a mismatch or under a `jit`/`grad` trace
  (the probe needs concrete arrays). To stay leak-free it builds the pattern from
  a **cached-recycle-map** forward rhs (the per-call recycle probing in
  `_rhs(recycle_map=None)` leaks a traced intermediate under the pattern-probe
  `jit` on a multi-model plant); this cached-map rhs (`primal_rhs`) is also used
  for the **forward iteration** (identical result, faster), while the one-shot
  implicit-function-theorem *gradient* keeps the **map-recomputing** rhs so a
  flow-setpoint parameter retains its `d(map)/d(param)` term (the #366 split).
  **PTC is a better fit for coloring than the dynamic solve**: it marches to a
  single operating point in a narrow neighbourhood, so the start-state sparsity
  pattern stays valid throughout — unlike the 609-day dynamic run's wide load
  excursion. **Measured (BSM2, 167 states / 46 colors):** the per-iteration
  Jacobian build is **2.4× cheaper** (0.62 → 0.26 ms) and the whole PTC solve,
  **run-only under `jit`, is 1.87× faster** (58 → 31 ms). **But the un-jitted
  one-shot `steady_state` call is compile/trace-bound, not Jacobian-build-bound**
  (the `while_loop` re-traces per call), so the run-phase saving is invisible
  there and the one-time pattern build (~49 dense probes) makes a single
  `steady_state(colored_jacobian=True)` call *slower* (~0.8×). The win therefore
  materializes only when the solve is run **repeatedly under `jit`** (differentiable
  design sweeps / optimization loops, where compilation is amortized) or for a
  much larger plant. The implicit-function-theorem *gradient* Jacobian stays
  dense (a single evaluation). **Default off; opt in for the jitted/amortized
  regime, not for a one-shot steady state.**
- **Carrying the RHS across PTC iterations — TESTED AND REJECTED (no benefit).**
  The PTC `step` (`plant/steady.py`) evaluates the RHS twice per iteration —
  `Fy = F(y)` at the top (for the linear solve) and `F(y_new)` for the residual —
  and the next iteration's `F(y)` *is* the previous iteration's `F(y_new)`. The
  obvious optimization is to carry the `F` vector in the `while_loop` carry so the
  RHS is evaluated once per iteration (each eval includes the recycle and pH
  solves, so on paper the saving looks real). **It does not help:** the result is
  bit-identical but the BSM2 run-only (jitted) time is **neutral-to-slightly-worse**
  (~43 → ~44 ms, measured min-of-8). The reason is that `jac = jax.jacfwd(F)`
  computes `F(y)` as its forward-mode **primal** on the same `y`, so **XLA already
  CSE-eliminates the redundant top-level `F(y)`** against the Jacobian's primal
  pass; removing it by hand saves nothing and threading the extra `n`-vector
  through the loop carry adds a hair of overhead. **Lesson:** redundant RHS
  *evaluations* in the PTC loop are the compiler's job — it fuses them. Real PTC
  speedups must cut *distinct* work the compiler cannot share across iterations:
  the per-iteration Jacobian materialization (the `colored_jacobian` builder, or
  freezing/reusing `J` for several steps) or the **iteration count** itself
  (better pseudo-time / residual scaling, a line search). Do not re-attempt the
  carry-`F` micro-optimization.

**Default `atol` is now per-component, scaled to the state magnitudes.** When
`atol` is omitted, **every single-concentration-vector reactor**
(`BatchReactor`/`PlugFlowReactor`/`ParticleTrackReactor`/`CFDReactor`, via the
shared `integrate/_common.resolve_state_atol`) and `Plant.solve` build a
per-species noise floor
`atol_i = atol_factor·max(|operating_i|, |reference_i|, floor_frac·char)`
(`atol_factor=floor_frac=1e-6`) via `integrate/_common.default_atol` — the
SUNDIALS "vector atol" / Hairer "atol ∝ typical value" rule. The reactors scale
off the model's `default_concentrations` (at construction); the plant scales
off `y0` (at solve time). **`default_atol` `stop_gradient`s its result** — the
tolerance is a solver noise floor, never a differentiated quantity. This matters
because the plant scales off `y0`: under a gradient **with respect to `y0`** the
floor would otherwise be a traced array, get baked into the integrator's step
controller, and — inside the discrete-adjoint (`gradient="stable_adjoint"`)
custom-VJP forward (which re-runs `diffrax.diffeqsolve`) — escape that inner solve
as a leaked tracer (`UnexpectedTracerError`, issue #420). Detaching it is the
identity for the value (so every steady state is unchanged) and lets the
**initial-state gradient** flow through `stable_adjoint`, the one direction the
standard `jax_adjoint` already handled. (The leak was specific to the
state-derived tolerance; the param-gradient direction was always fine, since the
tolerance does not depend on the parameters.) When **every** magnitude is zero (an all-zero
`scale_like` with no reference) the relative floor `floor_frac·char` would
itself be 0, so `char` falls back to unit scale — keeping every `atol_i`
strictly positive rather than 0 (the very invariant this floor upholds). This
fallback is identity for any input with a nonzero magnitude (the common path). `BiofilmReactor` is the exception — its multi-
compartment `(n_layers+1, n_species)` state does not match the per-species
vector, so it keeps an explicit scalar `atol` (default `1e-9`). This replaces
the old fixed `atol=1e-9`, which was ~9 orders too tight for g/m³ ASM/ADM states
and forced the integrator step ceiling — so a warm-started BSM2 now solves with
**nothing passed** (no `atol=1e-3, max_steps=500_000` magic). An explicit scalar
or `(n_species,)` array still overrides it verbatim (e.g. the ozone `OH→1e-20`
per-species atol), so existing calls are unchanged. Verified to reproduce every validated steady
state (691 non-validation + 23 validation tests). Any solve that hits the
integrator step budget -- `Plant.solve` **and every reactor**
(`BatchReactor`/`PlugFlowReactor`/`BiofilmReactor`/`ParticleTrackReactor`) --
re-raises the Diffrax/Equinox failure as a domain `RuntimeError` naming the
remedies (warm-start via `run_to_steady_state`, loosen `rtol`, raise
`max_steps`), with the noisy equinox exception chain suppressed (`from None`).
This is the shared `integrate/_common.friendly_solve_errors(max_steps, what=...)`
context manager wrapped around each solve's *execution* (the call to the jitted
solve / `diffeqsolve`, where the runtime error surfaces -- not the traced
`_run_diffeqsolve`). The jitted reactors emit one extra equinox *stderr* line
about `filter_jit` that the exception machinery cannot suppress; the raised
exception itself is clean. The **same** context manager also catches the other
opaque solve-time failure — a `jax.jacfwd`/`jax.jvp` (forward-mode AD) through
the default reverse-only adjoint, which JAX rejects with "can't apply
forward-mode autodiff (jvp) to a custom_vjp function". That is re-raised as a
`RuntimeError` naming the cure, `aquakin.forward_adjoint()` (build the reactor
with that adjoint, or take a reverse-mode gradient). `sensitivity`/`dgsm` with
`ad_mode="forward"` set the forward-capable adjoint for you and so never hit it.

**Dynamic influent now works too — flow-controlled recycle pumps (issue #30).**
The dynamic (time-varying-influent) run *used* to hit the step ceiling, which
was attributed to diurnal-forcing stiffness. The real cause was a **flow-model
bug**: the recycle streams (internal recycle `Qa`, RAS `Qr`, wastage `Qw`) were
modelled as fixed-*fraction* `SplitterUnit`s and the clarifier effluent as a
fixed flow — constants calibrated only at the design influent `Q_avg`. The
recycle-flow algebra then has a near-singular gain
(`tank5_throughput = (Q_fresh − 17693)/0.00816`), so a ±10% influent swing whips
the throughput from 5× to ~23× `Q_in`; that violently amplified, fast-varying
flow field is what made the monolithic solve crawl. `_resolve_flows` was exact —
it was faithfully resolving the *wrong* flow model — which is why it stayed
hidden at steady state (sitting exactly at `Q_avg`, where the fractions are
correct). The BSM1/BSM2 reference (`asm1init_bsm2.m`) settles it: `Qa = 3·Qin0`,
`Qr = Qin0`, `Qw = 300` are **constant pumped flows** off a fixed reference flow,
and the settler computes the effluent as the *free remainder*
(`Q_e = Q_f − (Q_r + Q_w)`). The fix mirrors this: `SplitterUnit` gains a
fixed-setpoint *flow mode* (`output_port_flows` + `remainder_port`); the
clarifiers gain a fixed `underflow_Q` (= `Qr + Qw`) with the overflow as the
remainder; `build_bsm1` wires the recycles as constant pumps. Throughput now
holds ~5× `Q_in` under any influent, and the 14-day dry run integrates in ~5k
steps (Ideal, ~10 s) / ~18k steps (Takács, ~30 s) to a healthy state —
**~1000× fewer steps**, steady state unchanged. Regression-guarded by
`test_bsm1_dry_weather_runs` and `test_bsm1_takacs_dry_weather_runs`.

The first plant-wide demonstration target is **BSM1** (Copp 2002 / Alex
2008) — built by `aquakin.plant.bsm.build_bsm1()`. Three synthesised
influent CSVs (dry / rain / storm) ship under
`aquakin/plant/bsm/data/` and load via `load_bsm1_influent()`. The
synthesised files match BSM1's *statistical* profile but are not the
canonical IWA files; for quantitative comparison to Alex 2008's
published EQI / OCI values, users should replace them with the
official files.

**A²O biological-nutrient-removal plant (`aquakin.plant.build_a2o`,
[`plant/a2o.py`](aquakin/plant/a2o.py)).** The first phosphorus-capable
flowsheet — the BSM plants run the P-free ASM1, so they cannot host bio-P or the
chemical-P (metal-salt dosing) demonstration. `build_a2o` is the canonical
**Anaerobic–Anoxic–Oxic** layout on the shipped `asm2d` model: an anaerobic
selector (where PAOs release phosphate and store fermentation products) → anoxic
denitrification → aerated nitrification + luxury P uptake → secondary clarifier,
with the mixed-liquor internal recycle (aerobic→anoxic) and RAS
(underflow→anaerobic) closing the loop, so it removes carbon, nitrogen **and**
phosphorus in one plant. `a2o_influent(net)` is a matching constant municipal
(VFA-bearing) influent and `a2o_warm_start(plant)` seeds the AS reactors with an
established EBPR mixed liquor (a large PAO population + stored poly-P), so a solve
starts from healthy bio-P sludge rather than the slow, seed-sensitive cold-start
PAO establishment. The default config reaches a feasible steady state with
**complete biological P removal** (effluent SPO4 ≈ 0) and ~80% N removal (full
nitrification of the influent ammonia + denitrification), with no recirculating
negative soluble pools (it relies on the `asm2d` `positivity_limiter`, now
honoured inside `CSTRUnit`). It is **not** a standardised benchmark — the sizing
is a representative municipal design, not a published reference set, so it is a
worked nutrient-removal flowsheet, not a validation target. Building it is what
surfaced the `asm2d` process-matrix import errors (see the `asm2d` model note);
the A²O viability test (`tests/integration/test_a2o.py`) is the regression guard
the COD/N/P continuity suite could not be (each broken coefficient still
conserves mass). It is the substrate for the chemical-P (ferric/alum dosing)
demonstration.

**BSM2 — open-loop plant (Gernaey et al. 2014 / Jeppsson et al. 2007).**
`aquakin.plant.bsm.build_bsm2()` wraps the BSM1 activated-sludge core with the
full sludge train: a **primary clarifier** ahead of the reactors, and
downstream a **thickener**, an **ADM1 anaerobic digester** (35 °C, 3400 m³
liquid + headspace) with the **ASM1↔ADM1 interfaces**, and a **dewatering**
unit, with the two reject-water streams (thickener overflow + dewatering reject)
recycled to the plant front. This is a genuinely **two-model** plant (ASM1
water line + ADM1 digester); the interfaces ride on the cross-model
connections as `StateTranslator`s, so the whole thing still integrates under one
monolithic Diffrax solve with `jax.grad` flowing end to end. All controlled
flows (internal recycle `Qintr=3·Q_ref`, RAS `Qr=Q_ref`, wastage `Qw=300`,
primary sludge `f_PS·Q`) are fixed-flow pumps (the BSM1 flow-control fix carries
over); the thickener/dewatering underflows are concentration-dependent but sit
on the low-gain reject loop, which the concentration sweep resolves (their
`flow_outputs` seed the linear pre-solve with a nominal fraction). A constant
**external carbon dose** (2 m³/d of readily-biodegradable COD to reactor 1,
`carbon_flow`/`carbon_conc`, BSM2 default on) feeds denitrification in the
anoxic tanks. **It reaches a healthy open-loop steady state in ~20 s** —
nitrifying AS, biomass sustained, and a methanogenic digester.

**Optional features are configured with option objects, and the entry/exit
endpoints are exposed (so callers never hard-code a port).** The optional BSM2
features used to be a dozen cross-coupled boolean/float flags; they are now small
**frozen option objects** (`aquakin.plant.bsm`), one per feature, passed to
`build_bsm2` — present ⇒ enabled, `None` ⇒ off: `ExternalCarbon(flow, conc)`
(`carbon=`, default-on; `carbon=None` disables), `RejectStorage(volume,
output_flow, control)` (`reject=`, with `control=True` for the closed-loop level
controller), `InfluentBypass(threshold)` (`bypass=`), `HydraulicDelay(tau)`
(`hydraulic_delay=`). `do_control: bool` and `wastage_schedule` stay as they were
(a lone toggle / an already-an-object schedule). Some features move the front
ports (the bypass relocates the entry to `bypass_split.in` and the effluent to
`effluent_mix.out`; the hydraulic delay relocates the entry to
`influent_delay.in`) — so `build_bsm2`/`build_bsm1` record the canonical ports on
the generic **`Plant.influent_endpoint`** / **`Plant.effluent_endpoint`**
attributes. `plant.add_influent("feed", series)` defaults its `to=` to
`plant.influent_endpoint`, and `evaluate_bsm2` / `sludge_metrics` default their
effluent port to `plant.effluent_endpoint`, so feature flags can no longer
silently mis-wire the influent or score the wrong effluent. (The endpoints are
plain optional attributes on every `Plant`, `None` unless a builder sets them.)

**Quantitatively validated** against the published BSM2 open-loop steady state
(`tests/validation/test_bsm2_steadystate.py`): run with the published constant
influent (`bsm2_constant_influent`) and the BSM2 (15 °C) ASM1 parameter set
(`bsm2_parameters`), the whole multi-model plant — the 5 AS reactors, the
secondary settler, the primary clarifier, both ASM1↔ADM1 interfaces, the
digester, and all recycle loops including the reject water — reproduces the
reference reactor states (`asm1init_bsm2` `XINIT`: XB_H ≈ 2245, XB_A ≈ 167,
XP ≈ 967, XI ≈ 1532, the SNH/SNO/SO profiles) **to round-off (≤0.06% on every AS
state — the level at which the reference ring-test simulators agree with one
another)** and the digester (`DIGESTERINIT`: headspace methane to ~0.2%) **to
within ~1.3% (worst: headspace CO₂, the charge-balance-pH vs algebraic-pH
difference in the gas phase)**. **Reaching the round-off AS match needs the
benchmark operating temperature, not just the 15 °C parameters.** The ASM1 rates
are defined at 15 °C and Arrhenius-corrected to each reactor's (flow-weighted)
*inlet* temperature; the BSM2 constant influent enters at **14.858 °C**
(`BSM2_CONSTANT_INFLUENT_T`, the `constinfluent` T column = the annual mean), so
the AS line operates 0.14 °C below the reference and every rate is slowed ~1.4%.
Omitting this (running the line at the bare 15 °C reference) over-predicts
nitrification by ~1.4% — the entire otherwise-residual deviation (SNH/SNO drift
~1–1.5%). `bsm2_constant_influent` therefore takes a `T=` argument: pass
`T=BSM2_CONSTANT_INFLUENT_T` **together with `bsm2_asm1_model()`** (the 15 °C-
referenced corrections) for the faithful match. The default `T=None` keeps the
historic temperature-agnostic behaviour (reactors fall back to their static 15 °C
condition); do **not** pass `T` with the plain 20 °C `load_model("asm1")` — a
14.858 °C inlet on a 20 °C-referenced model applies a large spurious slowdown
(~40% on nitrification). `bsm2_constant_influent` guards this footgun: a `T`
more than `BSM2_INFLUENT_REF_T_TOL` (1 K) from the model's Arrhenius `ref_T`
warns, naming both values (the benchmark pairing is 0.14 K, well inside; the
20 °C-model mismatch is ~5 K, caught). aquakin carries the reactor temperature *algebraically*
(the flow-weighted inlet each RHS, resolved with the recycle solve), not as a
BSM2-style heat-balance state `dT/dt=(Q/V)(T_in−T)`; the two agree at steady
state (both give T=T_in) and differ only by the (sub-hour) thermal lag in
transient. Two parameter
reconciliations were needed: the BSM2 ASM1 values are the 15 °C set
(`muH=4, KS=10, muA=0.5, bH=0.3, KX=0.1, etah=0.8`). (The shipped `asm1` is the
textbook Gujer matrix with no heterotroph ammonia-limitation term, so — unlike
earlier versions — no neutralising override is needed; for the BioWin/SUMO
nutrient switch use the `asm1_ammonia_limitation` model, where that term
suppresses tank-5 growth ~24% and roughly halves XB_H.) ASM1 has no Arrhenius T-dependence
(the `T` condition is declared but unused), so only the parameter *values*
matter, not the 15 °C operating temperature.

**Dynamic influent runs too.** Synthesised BSM2 dry / rain / storm influent
files (`scripts/generate_bsm2_influent.py` → `aquakin/plant/bsm/data/BSM2_*.csv`,
loaded by `load_bsm2_influent()`) drive the plant under diurnal + wet-weather
forcing. The fixed-flow-pump fix carries straight over to BSM2 scale: warm-started
from steady state, the 167-state two-model plant integrates a 14-day dynamic
run **efficiently** (~140 steps/day, not a step-ceiling blow-up) to a finite,
healthy trajectory, and a rain event doubling the influent stays bounded because
the recycle pumps hold throughput at `Q_in + Qintr + Qr`
(`tests/integration/test_bsm2_dynamic.py`). The shipped influent CSVs are
**synthesised**, not the canonical 609-day IWA series, so the dynamic tests
assert qualitative stability, not published dynamic metrics.

**Temperature handling is a selectable `TemperatureModel`
([`plant/temperature.py`](aquakin/plant/temperature.py)).** Two strategies, set on
the plant (`plant.set_temperature_model(...)`, or `build_bsm2(temperature_model=
...)`); exported at the top level (`aquakin.TemperatureModel` /
`AlgebraicTemperature` / `HeatBalanceTemperature`):
- **`AlgebraicTemperature`** (default) — temperature is *instantaneous*: each unit
  flow-weights its inlet `T` (a heat balance) and passes it through, so a reactor
  runs its kinetics at its flow-weighted inlet temperature, with **no thermal
  storage**. Carries **zero** extra state and is a pure no-op (every existing
  plant and validated steady state is byte-for-byte unchanged). This is the
  historic behaviour, described in the rest of this section.
- **`HeatBalanceTemperature`** — every finite-volume liquid unit (one exposing a
  positive `volume`) that is not temperature-fixed carries its temperature as a
  **dynamic state** with the completely-mixed first-order balance
  `V dT/dt = Q_in (T_in − T)`; the heated digester sets `temperature_fixed = True`
  and stays pinned (the BSM2-protocol treatment, Jeppsson et al. 2007). The
  reactor then runs at this **lagged tank temperature**, so it damps/lags the
  influent (important because recycles trap heat — the effective AS time constant
  `V_total/Q_fresh` is hours, comparable to diurnal forcing — which the algebraic
  model cannot represent). For BSM2 it tracks the 5 reactors + primary clarifier +
  settler (the `TakacsClarifier` exposes a `volume = area·height` for this). The
  temperature states are appended as one block at the **tail** of the flat plant
  state vector (the `FlowSetpoint` tail-append pattern, but for state), so every
  per-unit state slice keeps its index (warm-starts / `states_by_unit` unaffected);
  `Plant._split_state` exposes the block under a reserved key, `_sweep_outputs`
  overrides each tracked unit's outlet `T` with its state (so the lag propagates
  through the exact recycle-temperature solve), and the reactor reads its operating
  temperature from a reserved control-signal key (`OPERATING_T_SIGNAL`), falling
  back to the flow-weighted inlet T when absent. At a constant influent temperature
  the heat-balance fixed point IS the influent temperature, so it reproduces the
  algebraic steady state. Tested in
  `tests/integration/test_temperature_model.py` (tracked set, the first-order
  balance + `V/Q` time constant, the constant-influent fixed point, AD through the
  state). *(Motivation: investigating the ~16% effluent-S_NH gap in the dynamic
  BSM2 vs the ring-test consensus — the algebraic and heat-balance reactor
  temperatures are equal to ≤0.1 °C across the AS line because the lag averages
  out over a seasonal window, so this is for transient-temperature fidelity, not a
  fix for that gap.)*

The default-model behaviour: temperature is carried *algebraically* through the
flowsheet: `Stream` and `InfluentSeries` have an optional `T` (Kelvin); mixers
flow-weight it (a heat balance) and every other unit passes it through, so a
reactor reads its (flow-weighted) inlet temperature and feeds it to the ASM1
temperature corrections. `T=None` is the default and a static structural
property — a temperature-agnostic influent leaves every stream `T=None` and the
reactors fall back to their static condition, so existing plants are unchanged.
The single heat-balance rule every multi-inlet unit (mixer, CSTR, clarifier,
digester) uses is `streams.mixed_temperature(inputs, names)`: it flow-weights
only the inlets that carry a temperature and *ignores* a `T=None` inlet rather
than letting one collapse the whole mix to `None`. This is what lets a
temperature-carrying influent propagate around a recycle loop whose back-edge is
auto-seeded with a zero-flow, temperature-agnostic stream (the seed contributes
nothing and is ignored); earlier the `all(inlet.T is not None)` gate meant one
agnostic seed disabled temperature around the loop, so `build_bsm2` had to
hand-seed its recycles with a nominal `T` (now redundant — kept only as an
explicit warm start). The helper is also zero-flow-safe: if every
temperature-carrying inlet is momentarily at zero flow it returns their mean
rather than dividing by the flow epsilon (which would drive the result toward
0 K and feed a garbage value into the Arrhenius correction). For BSM2 the AS
reactors run at 15 °C:
`bsm2_asm1_model()` re-references the ASM1 temperature corrections from 20 °C
to 15 °C (keeping the BSM2 slopes), so with `bsm2_parameters` (the 15 °C values)
the correction is unity at 15 °C — a constant-15 °C run reproduces the validated
steady state exactly — and a temperature-carrying influent drives it away:
colder water nitrifies more slowly (higher residual ammonia), warmer faster
(`tests/integration/test_bsm2_seasonal.py`). `build_bsm2()` now **defaults** its
ASM1 model to `bsm2_asm1_model()` (the 15 °C reference), so the out-of-the-box
plant is the BSM2 calibration; pass the plain `load_model("asm1")` explicitly to
get the 20 °C reference. When you build the influent yourself, reuse the **same
model instance** for both `build_bsm2` and the influent so their identities match
(a clear error fires otherwise). The
synthesised BSM2 influent CSVs carry a time-varying temperature column (`T`, in
°C; a shoulder-season ~12→18 °C ramp + diurnal ripple), which
`load_bsm2_influent` returns as `InfluentSeries.T` in **Kelvin** — so a dynamic
run on `load_bsm2_influent(...)` is seasonally temperature-driven out of the box.
(The generic `read_influent_csv` / `_influent_from_text` capture a `T` column
when present, in the file's own units; only the BSM2 loader converts °C→K.)

**Temperature-dependent oxygen transfer (issue #206).** By default the aeration
term is `kLa·(C_sat − C)` with `C_sat` a fixed constant (8.0 gO₂/m³) and `kLa`
constant — the literal IWA benchmark definition. That left a seasonal-run
inconsistency: a warm influent already speeds the (Arrhenius) biology while the
oxygen driving force stayed pinned. `Aeration` now carries **opt-in** transfer
corrections, all identity by default so the benchmark stays bit-faithful:
`temperature_correction=True` scales the saturation by the clean-water ratio
`C_s(T)/C_s(ref_T)` (the Benson–Krause `aquakin.plant.oxygen_saturation`,
~9.09 mg/L at 20 °C → ~7.56 at 30 °C) and the **open-loop** `kLa` by
`kla_theta**(T−ref_T)` (default θ=1.024), using the same flow-weighted inlet `T`
the kinetics use (falling back to the static `T` condition); a closed-loop
controlled `kLa` is **not** θ-scaled (the controller already manipulates it) but
its driving-force saturation still gets the `C_s(T)` correction. Constant factors
`alpha` (kLa transfer fouling), `beta` (salinity) and `pressure_factor`
(elevation) fold into the precomputed vectors at construction (defaults 1.0). All
AD-clean (the correction is a smooth function of `T` inside the monolithic plant
solve). `build_bsm2(do_temperature_correction=True)` turns it on plant-wide with
`ref_T` = the reactors' static temperature (so it is unity at the benchmark
operating point and only a temperature-carrying influent drives it); default off
reproduces the validated steady state exactly. The saturation curve used for the
`C_s(T)/C_s(ref_T)` ratio is selectable via `Aeration(saturation_model=...)`:
`"benson_krause"` (default, the APHA `oxygen_saturation`) or `"bsm2"` (the IWA
benchmark van't Hoff `oxygen_saturation_bsm2`, normalised to 8.0 mg/L at 15 °C);
the two differ by ~0.5 % in shape. `build_bsm2(do_temperature_correction=True)`
uses `"bsm2"` so the seasonal oxygen driving force matches the benchmark exactly.

**Diffuser / blower aeration-design physics (issue #279,
[`plant/aeration_system.py`](aquakin/plant/aeration_system.py)).** The kinetic
model aerates through a per-species `kLa`, and the Copp-2002 OCI scores aeration
*energy* with the fixed correlation `AE ∝ Σ V_i·kLa_i`. `AerationSystem` is the
blower/diffuser physics behind that `kLa` — how much **air** must be blown and the
**power** to compress it — kept **standalone** (it does **not** change the `kLa`
interface). From the `kLa` a solve produced it computes the standard oxygen
transfer rate `SOTR = kLa·C_s,std·V` (the clean-water transfer the airflow must
deliver — a given `kLa` needs a given airflow, independent of the operating DO
deficit), the **air flow** `Q_air = SOTR/(SOTE·o2_per_air)` from the diffuser's
standard transfer efficiency `SOTE` (rising with submergence, default `6 %/m`,
reduced by a fouling factor `F`), the blower **discharge pressure**
`p_atm + ρ_w·g·depth + headloss`, and the blower **power** by adiabatic
compression `P = (Q·p1/η)·(γ/(γ−1))·[(p2/p1)^((γ−1)/γ) − 1]`. Because the power is
linear in airflow and airflow is linear in `kLa`, `blower_energy(t, kla_history,
volumes, system)` has the same form as the Copp kernel but with a mechanistic
coefficient (SOTE/depth/blower curve) in place of the fixed one, and stays
`jit`/`grad`-clean (the differentiable primitives are `required_airflow` /
`blower_power_kw`; the float-returning `blower_energy` is the reporting kernel, the
drop-in for `aeration_energy`). The α/β/temperature *field* corrections stay on
`Aeration` (they shape the `kLa` and driving force in the solve); `AerationSystem`
adds the diffuser-fouling `F` and the blower curve. **Wired into the evaluators:**
`evaluate_bsm1(..., aeration_system=AerationSystem(...))` and `evaluate_bsm2(...,
aeration_system=...)` **replace** the correlation AE with the mechanistic blower
energy (flowing into the OCI and, via `total_energy()`, the GHG/cost report) and
expose `air_flow` (m³/d) on the evaluation; `aeration_system=None` (default) keeps
the validated Copp AE, so the benchmark numbers are unchanged. `design_summary(kla,
volume, system)` is the standalone sizing entry point → an `AerationDesignPoint`
(SOTE / SOTR / airflow / discharge pressure / power) with a labeled `report()`.
Covered by `tests/integration/test_aeration_system.py` (physics vs closed form,
SOTE/depth/fouling, validation, AD) and the evaluator wiring in
`tests/integration/test_bsm2_evaluation.py`.

**Influent characterization + CSV `column_map` (issue #136).** Real influent is
measured as aggregates (total COD, TKN, ammonia, alkalinity, optionally
filtered/flocculated COD), not as the 13 ASM1 states. `aquakin/plant/characterize.py`
maps them: `fractionate(total_cod=, tkn=, ...) -> {ASM1 state: value}` follows the
**SUMO Sumo1 raw-influent fractionation reduced to ASM1** — COD split by
filtration (soluble/colloidal/particulate) then biodegradability, reduced to ASM1
by lumping colloidal-biodegradable into `XS` and colloidal/soluble-inert into
`XI`/`SI` (`SI=SU, SS=SB, XI=CU+XU, XS=CB+XB, XB_H=XOHO, XP=XE, XB_A=0`); N gives
`SNH` (ammonia or `f_snh·TKN`), `SND` (soluble-biodeg N), `XND` (TKN-balance
remainder using ASM1's `i_XB`/`i_XP`); alkalinity mg CaCO₃/L → `SALK` mol/m³ via
`/50`. A measured `filtered_cod`/`flocculated_filtered_cod`/`soluble_inert_cod`
drives its split; absent, the SUMO default fraction (`InfluentFractions`, the
Sumo1 tool's municipal values) is used. The reduction **conserves total COD**
(`Σ COD states = total_cod`) and closes the ASM1 TKN balance. `fractionate` is
plain arithmetic, so it runs element-wise on scalars **or arrays** — the per-row
path. `characterize_influent(model, flow=, total_cod=, ...)` wraps it into a
constant `InfluentSeries`. `read_influent_csv(..., column_map={role: header})`
loads an **arbitrary-header** CSV (a lab/SCADA export — no renaming): roles are
`t`/`Q`/`T`, any ASM species (mapped directly), and the aggregate names; mapped
aggregates are fractionated **per row** (a directly-mapped species overrides its
fractionated value; unmapped species default to zero). Validated against the
spreadsheet's worked example (`tests/integration/test_characterize.py`). Exported
as `aquakin.characterize_influent` / `fractionate` / `InfluentFractions` /
`read_influent_csv`.

**`Plant.set_temperature(celsius)` — one knob for the operating temperature.**
Setting a plant's temperature used to mean writing the static `T` condition of
every reactor by hand (in Kelvin, at the correction `ref_T`). `set_temperature`
takes **°C**, converts to Kelvin, and writes the static `T` of every
temperature-bearing reactor — so a re-solve runs the Arrhenius
`temperature_corrections` at that temperature (`build_bsm2(...)` then
`plant.set_temperature(15)` is the BSM2 15 °C operating point; `set_temperature(10)`
drives nitrification down — verified in
`tests/integration/test_plant_temperature.py`). It targets the activated-sludge
reactors (`CSTRUnit`s exposing `set_temperature` with a `T` condition) and
**leaves the heated anaerobic digester untouched** (a fixed-`T` ADM1 unit without
the method); pass `units=[...]` to target a specific set. It invalidates the
plant's compiled-solve caches (`plant._solve_cache.invalidate()` — the jitted
forward + PTC steady solves *and* the continuation / arclength kernels, all of
which bake in the conditions) so the next solve recompiles at the new
temperature, and returns `self` for chaining after `build_*`. The per-unit
mechanic is `CSTRUnit.set_temperature(temperature_K)` (updates `conditions["T"]`
and its precomputed condition array).

**Clear error on an influent/plant model-instance mismatch.** The seasonal
footgun was using *different instances* of the same ASM1 model for the plant and
the influent (e.g. calling `bsm2_asm1_model()` twice): their temperature
corrections / parameters then silently disagree. `Plant._default_translator` now
distinguishes this from a genuine cross-model edge — when the two models have
the same `name` and `species` but are different objects, the error says to *build
the model once and pass that same object to both* (rather than the old, here
misleading, "supply an explicit translator"); a truly different model still gets
the translator message.

**Closed-loop DO/kLa control (`build_bsm2(do_control=True)`).** The first
closed-loop element is the BSM2 dissolved-oxygen controller: a PI loop senses
`SO` in reactor 4 and manipulates its aeration `kLa` (reactors 3 and 5 scale off
the same signal at gains 1.0/0.5), driving the oxygen to the `SO=2` gO₂/m³
setpoint instead of the fixed open-loop `kLa`. Tuning is the reference DO loop
(`Kp=25`, integral time `Ti=0.002` d, anti-windup tracking `Tt=0.001` d, `kLa`
offset 120 d⁻¹, bounded `[0, 360]`). It is built on a small, general
**control-signal bus** layered on the material flowsheet (so the loop closes
inside the one monolithic Diffrax solve and `jax.grad` still flows end to end):
- `PIController` ([`plant/control.py`](aquakin/plant/control.py)) is a Unit with
  one integral state. It reads its measured variable from a *sensed input
  stream* (wired like any other connection, `tank4 → do_control.measured`, but it
  produces no material output), and publishes a named scalar **signal**
  `u_sat = clip(offset + Kp·e + x_i, out_min, out_max)` via `signal_outputs(...)`;
  its `rhs` integrates `dx_i/dt = (Kp/Ti)·e + (1/Tt)·(u_sat − u)` (back-calculation
  anti-windup). `x_i` is the integral *contribution to the output* (already
  scaled), so the tracking term has consistent units.
- `Plant._rhs` evaluates `signal_outputs` on every controller each RHS call,
  gathers the results into a `signals` dict, and threads it into **every** unit's
  `compute_outputs` *and* `rhs` as the trailing `signals` argument (a unit that
  reads no signals simply ignores it). Producing signals is the one optional,
  class-level/duck-typed hook (`hasattr(unit, "signal_outputs")`), so the branch
  is static and jit/AD-safe. **The bus is computed from the reactor states
  *before* the stream sweep** (`_compute_signals`): a controller senses a
  reactor's concentration, which *is* that unit's state, so the sensed value is
  read directly from `states` (the controller's sensed `inputs` stream is
  reconstructed as `C = states[sensor]`, so the sensor must be a reactor whose
  output concentration is its state, i.e. a `CSTRUnit`). Computing signals first
  is what lets a unit whose *output stream* depends on a signal — a
  feedback-`DosingUnit` — read it in `compute_outputs` (the sweep), which runs
  before any post-sweep quantity. For a DO controller sensing a CSTR the value is
  identical to the old post-sweep read (state == output `C`), so aeration is
  unchanged.
- **`Aeration` on `CSTRUnit` (issue #137).** A tank's aeration is set with one
  `aeration=Aeration(...)` object, not raw per-species `kla`/`C_sat`/`controlled_kla`
  dicts (those fields are gone — pre-release, single interface). `Aeration` has two
  modes: open loop `Aeration(kla=120, do_sat=8)` (a fixed mass-transfer
  coefficient; `do_sat` defaults to 8.0), and closed loop `Aeration(do_setpoint=2.0)`
  (a DO target). `CSTRUnit.__post_init__` translates the spec into the internal
  `_kla_vec`/`_sat_vec`/`_controlled_kla` the `rhs` uses, so the aeration term
  `kLa·(do_sat - C)` is unchanged. For closed loop the species' `kLa` is taken from
  `signals[name]·gain` each step (overriding the fixed `kLa`); the signal name is
  derived deterministically from the controller id.
- **Auto-wired DO controllers (`Plant._materialize_aeration`).** A closed-loop
  `Aeration` consumes a kLa signal but does not itself add the controller. The
  plant materialises it: at topology setup (once, before the state layout) it
  groups the closed-loop tanks by their aeration `controller` id — the shared-
  controller case (BSM2: one sensor on `tank4`, per-tank `gain`s) — or, when no id
  is given, gives each tank its own controller (per-tank DO control, `sensor`
  defaults to the tank). One `PIController` per group is added (named after the
  shared id, or `<tank>_aeration`), sensing the group's `sensor`, with the
  setpoint/PI tuning/bounds from the `Aeration` (defaults are the BSM2 DO loop).
  Tanks sharing a controller must agree on its setpoint/sensor/tuning; only `gain`
  differs. `build_bsm2(do_control=True)` now expresses the loop purely as
  `Aeration(do_setpoint=2.0, controller="do_control", sensor="tank4", gain=...)`
  on the reactors — the controller and its sensor tap are auto-wired (the manual
  `PIController` + `connect` are gone). `build_bsm1`/`build_bsm2` open-loop tanks
  use `Aeration(kla=..., do_sat=8)`.
- **Assembly-time signal validation.** A unit declares the bus names it reads via
  `required_signals` (`CSTRUnit` derives it from its closed-loop aeration) and the
  names it publishes via `signal_names` (`PIController` -> its `signal_name`).
  `Plant._validate_control_signals` (run from `_build_state_layout`, before the
  RHS is traced) checks every consumed name is published, so a forgotten/mistyped
  controller signal raises a clear `ValueError` naming the unit and the available
  signals -- not a bare `KeyError` from deep in the first jitted solve. It is
  conservative: if any producer (a unit exposing `signal_outputs`) does not
  declare `signal_names`, the published set is unknown and validation is skipped.
Covered by `tests/integration/test_bsm2_control.py` (controller-unit behaviour:
signal sign, saturation, integral direction, anti-windup; closed-loop setpoint
tracking; closed-vs-open contrast; `jax.grad` through the closed loop). The
digester is additionally validated at the unit level in
`tests/validation/test_bsm2_digester_unit.py`.

**Chemical dosing (`DosingUnit` / `Reagent`, issue #278).** A general inline
dosing unit (`aquakin/plant/dosing.py`): `in` stream → `out` = inlet + reagent
dose, flow-mixing the compositions. A `Reagent` is a value object — a fixed
composition vector built by name (`Reagent.from_species(asm1, SS=4e5,
label="methanol")`, base zero, so the neat reagent contains only what you name) —
covering metal salts, acid/base, and external carbon. The dose flow is either
**fixed** (`DosingUnit(name, reagent, flow=2.0)`) or **feedback-controlled**
(`DosingUnit(name, reagent, setpoint=1.0, measured_species="SNO", sensor="anoxic",
flow_max=...)`): a feedback dose declares a sensed reactor + species + setpoint,
and the plant auto-wires a `PIController` (`Plant._materialize_dosing`, the dosing
analogue of `_materialize_aeration`, reusing the same controller) that
manipulates the dose flow to hold the setpoint, publishing a `_dose_<id>_flow`
signal the unit consumes. The unit is **stateless** — a fixed dose needs no
state, a feedback dose's PI integral lives in the controller — and reads its
dose-flow signal in `compute_outputs` (the dose changes the *output stream*),
which is why the signal bus is computed before the sweep (above); `flow_outputs`
seeds the recycle-flow solve with the nominal `flow_offset` for a feedback dose
(the exact, concentration-dependent flow is applied in `compute_outputs`, the
same convention the separators' concentration-dependent flows use). `build_bsm2`
now expresses its external-carbon feed as a fixed `DosingUnit` on the
`as_mix → tank1` line (the former hard-coded carbon influent is gone); the
validated steady state is unchanged (same carbon mass, same tank-1 inlet). The
dose only adds the reagent's *mass*; the **reactive** response — an acid/base's
pH shift, metal-phosphate precipitation, the added COD's oxygen demand — is the
downstream reactor's chemistry (the precipitation/pH engine, issue #271), not
this unit's job. Covered by `tests/integration/test_dosing.py`.

**Disinfection unit ops (`UVUnit` / `ChlorineContactUnit`, issue #280,
[`plant/disinfection.py`](aquakin/plant/disinfection.py)).** The `uv_h2o2` /
`ozone_bromate` *models* model the oxidation chemistry, but neither is a
disinfection *unit op* that reduces a pathogen indicator in the flowsheet. These
two add that, matching the commercial simulators (GPS-X / SUMO track an indicator
organism + the disinfectant residual and apply a dose/CT log-removal). Both
**pass the process (ASM) stream through unchanged** (disinfection does not
materially change COD/N/P at this fidelity) and reduce an **indicator-organism
density carried on the stream** — a new optional `Stream.org` scalar, the
disinfection analogue of the temperature `T` scalar: mixers flow-weight it (the
shared `streams._flow_weighted_scalar` behind both `mixed_temperature` and the
new `mixed_organism`) and pass-through units propagate it, and a disinfection unit
applies `N = N0·10^(−log)`. When the inlet carries no indicator (`org is None`)
the unit falls back to its design `inlet_density`, so a terminal disinfection
train works without wiring an indicator influent. The reconstructed effluent
surfaces it: `Plant.stream(...)` returns a `StreamSeries` with an `org` trajectory
(reconstructed on demand via `Plant._reconstruct_stream_org`; `None` for an
indicator-agnostic stream, so every BSM stream is unaffected and does no extra
work). `UVUnit` is **stateless**: the dose is `intensity · exposure · UVT-factor`
with the exposure the baffling-scaled residence `V/Q` converted to seconds
(fluence rate mW/cm², dose mJ/cm²), and a log-linear dose-response
`log = dose/d10` (optional `max_log` tailing). `ChlorineContactUnit` carries a
**one-state chlorine residual** (a completely-mixed tank with first-order decay,
`dCl/dt = (Q/V)(dose − Cl) − k_decay·Cl`); the CT credit `residual · T10` drives
`log = CT/ct_per_log`, with `T10 = baffling·V/Q` (or `t10_from_rtd`, the
non-ideal-contactor 10th-percentile of a residence-time distribution, reusing
`utils/rtd.percentile_time`); `dechlorinate=True` reports the discharged residual
as zero. The credit physics is exposed as pure, AD-clean functions (`uv_dose`,
`uv_log_inactivation`, `ct_value`, `ct_log_removal`, `t10_from_baffling`,
`t10_from_rtd`) for standalone sizing/credit, and `jax.grad` flows through both the
credit and a plant solve (design optimisation). Covered by
`tests/integration/test_disinfection.py`. **Scope/fidelity notes:** the UVT
correction is a first-order linear `uvt/uvt_ref` (a full UVDGM dose-distribution is
future work); chlorine is modelled as a residual-decay + CT credit (no breakpoint
/ chloramine speciation); the indicator transports through mixers + the
disinfection units (the biological train does not yet copy `org`, which is right
for a terminal disinfection step).

**Sequencing batch reactor (`SBRUnit`, issue #273).** A single tank that treats
in batches, cycling through timed phases (fill → react → settle → decant → idle)
defined by a list of `SBRPhase(name, duration, feed=, decant=, kla=, settle=,
mixed=)`. Variable-volume state `[C, V]` (volume rises at `feed_flow` during fill,
falls at `decant_flow` during decant; the `StorageTank` `dV/dt = Q_in − Q_out`
pattern) plus the internal state of a pluggable `SettlingModel`. The biology reacts
every phase; aeration is the per-phase `kla` on the oxygen species; the settle phase
clarifies the supernatant the decant draws as the treated effluent. **Phase transitions are
located events:** `SBRUnit.cycle_events(t0, t1)` returns the phase-boundary times
as a time `Event`, and `Plant.solve` **auto-collects** every unit's `cycle_events`
(merged with any user `events=`) so the integrator lands exactly on each switch —
the flow/aeration discontinuities are resolved at the boundary, not stepped across,
while within a phase the ODE is smooth and differentiable (`jax.grad` flows through
a cycle). Feed is drawn at the unit's own `feed_flow` (a fill pump) taking the
connected stream's composition; a standalone SBR plant is just the `SBRUnit` + an
influent on `sbr.feed`. **Modular settling** ([`plant/settling.py`](aquakin/plant/settling.py)):
a `SettlingModel` strategy reports, each step, how its internal clarity state
evolves and a per-species multiplier the decant draw is scaled by (1 for solubles,
< 1 for settled particulates); mass is conserved by the SBR (a clarified decant
concentrates the retained solids). Two ship: `InterfaceSettling` (one state — a
clarified fraction growing at a settling velocity while the tank settles) and
`LayeredSettling` (a Takács-style vertical profile of the particulate distribution;
the decant draws the top layer). New models slot in by implementing `SettlingModel`.
**Clarity is driven by three regimes**, so the decant actually draws a clarified
effluent: the model's clarity *grows* while settling, *relaxes* back to mixed only
while the tank is **actively mixed** (fed or aerated), and is **held** during a
quiescent phase (decant/idle) — relaxing whenever `settle=False` would otherwise
wash the clarity out during the decant draw itself. A phase's mixing is derived as
`feed or kla>0` unless `SBRPhase(..., mixed=)` is set explicitly (e.g. an unaerated
but mechanically mixed anoxic react). (Settling is well-mixed for the biology — the
bulk `C` is the average; the model affects only the decant clarity.) `plant.mass_balance`
reads the SBR's `[C, V, settling]` inventory (volume at index `n_species`, the
settling state massless); the reaction/aeration *gas* term does not yet cover the
SBR's variable volume and per-phase aeration, so end-to-end closure of an aerated
SBR plant is a follow-up. Note: sludge wasting is not yet a phase, so over many
cycles solids concentrate (a clarified decant retains them); add a waste draw for a
closed long-run solids balance. The
located-event machinery also gained a fix here: a `t_eval` point landing exactly on
an event boundary now emits the segment-endpoint state rather than a dense-output
edge evaluation, which could return NaN for a stiff segment
([`integrate/events.py`](aquakin/integrate/events.py)). Covered by
`tests/integration/test_sbr.py`.

**Membrane bioreactor (`MBRUnit`, issue #274).** A high-MLSS aerated reactor
([`plant/mbr.py`](aquakin/plant/mbr.py)) whose membrane retains the solids,
replacing the secondary clarifier. Fixed-volume reactor state `[C, R_f]` (the bulk
concentrations + a membrane-fouling resistance), reusing the CSTR kinetics and the
`Aeration` machinery — so it takes an open-loop `kla` or a `do_setpoint` the plant
**auto-wires** a DO controller for, exactly like a CSTR. Two outlets: `permeate`
(the filtrate — solubles pass, particulates carried at `(1 − rejection)`, so the
effluent is near solids-free) and `waste` (mixed liquor at the full reactor MLSS,
drawn at the `waste_flow` setpoint). The volume is held constant
(`Q_permeate = Q_in − Q_waste`), and because solids leave **only** via the waste
draw the MLSS concentrates: at a 1-day HRT the biomass is retained where a
clarifier-less CSTR would wash out, and the **SRT = V / Q_waste decouples from the
HRT** (the defining MBR behaviour). A simple membrane-fouling state grows with the
permeate flux and relaxes (`dR_f/dt = fouling_rate·J − fouling_relax·R_f`,
`J = Q_permeate / membrane_area`), reaching a quasi-steady fouled state;
`MBRUnit.tmp(R_f, Q_permeate)` reports the trans-membrane pressure
`tmp_viscosity·J·(R_m + R_f)`. The permeate particulate split uses a per-species
mask built from `particulate_species` (solubles pass unhindered). Scour-air energy
couples to the aeration/blower accounting (the membrane needs continuous coarse-
bubble aeration); for now the biological aeration is the modelled term. The
`_materialize_aeration` sensor tap now names the sensor's first output port
explicitly (a bare endpoint is ambiguous for a multi-output unit like the MBR);
the controller reads the sensed value from the sensor's *state*, so any output
port carries it (unchanged for single-output CSTRs). Modelling choices for the
MVP — fixed volume with the permeate following the feed (vs a flux-controlled
variable-volume membrane) and reversible-fouling TMP (vs explicit backwash/cleaning
events) — are the natural simple forms; both are extension points. Like the CSTR,
the MBR carries the **flow-weighted inlet temperature** onto its outlet streams and
into the Arrhenius kinetics/aeration (via `streams.mixed_temperature`), so a
seasonal influent drives it; and `plant.mass_balance` treats it as a first-class
reactive aerated unit — its `[C, R_f]` inventory reads the fixed reactor volume (the
fouling resistance `R_f` is massless), and it exposes the CSTR `_kla_vec`/`_sat_vec`/
`_controlled_kla` accessors so its aeration-O2 and reaction-gas terms are counted (a
single-MBR plant closes COD/N to a few %). Covered by
`tests/integration/test_mbr.py`.

**Reject storage tank (`build_bsm2(reject=RejectStorage())`).** A variable-volume
equalisation tank on the reject-recycle line: a completely-mixed CSTR with **no
reactions** (`StorageTank`, [`plant/storage.py`](aquakin/plant/storage.py))
whose liquid volume `V` is a state (`dV/dt = Q_in_stored − Q_out`,
`dC_i/dt = Q_in_stored/V·(C_in,i − C_i)`). It releases at a controlled rate
`storage_output_flow` (default 0) with a **level-gated automatic bypass**: full
and filling → divert the whole inflow (don't overfill); full and draining →
release normally; empty → stop releasing and just fill. The two outlets
(`out`, the released stream at tank concentration; `bypass`, the diverted inflow
at inlet concentration) recombine at the front mixer. With the default zero
release the open-loop tank fills to its upper limit (`0.9·Vmax`) and bypasses
**all** reject, so it is a faithful pass-through and the steady state is
unchanged from the no-storage plant (verified: tank5 XB_H identical).
*Architecture note:* the bypass split is gated by the tank's own volume state,
which `StorageTank.flow_outputs` reads from the `FlowContext` `Plant._resolve_flows`
passes into every unit's `flow_outputs`. The exact affine flow solve stays valid
because the tank's *inlet* comes from the fixed-pump sludge line (the wastage
`Qw` is a constant pump), so at fixed volume its outputs are constant in the
recycle flows — the state-dependence does not couple to the recycle variables.
(In this benchmark the reject flow is nearly constant, so a *fixed* release just
fills or drains the tank; genuine equalisation needs the level-based release
controller below.) Wired into `build_bsm2` behind `reject_storage`; demonstrated
in `examples/bsm2_reject_storage.py` (level-gated behaviour by release rate) and
tested in `tests/integration/test_bsm2_storage.py` (the four regimes + flow/
volume conservation, no-solve; wired plant fills-and-bypasses, steady state
healthy).

**Scheduled (timed) wastage (`build_bsm2(wastage_schedule=...)`).** The
waste-sludge pump can follow a time schedule instead of the constant `Qw=300`:
the BSM2 strategy steps the wastage between a low (300) and a high (450) rate at
~182-day half-year blocks over the 609-day evaluation, managing the sludge
inventory (wasting more sludge shortens the solids retention time and draws the
reactor biomass down — verified: tank5 XB_H falls after the step). Built on a
reusable **`PiecewiseConstantSchedule`**
([`plant/schedule.py`](aquakin/plant/schedule.py)): `values[i]` holds on
`[t_breaks[i-1], t_breaks[i])`, evaluated by a `jit`/AD-safe `searchsorted`
gather. `bsm2_wastage_schedule()` returns the BSM2 `Qw(t)`; `build_bsm2` makes
the secondary-clarifier underflow the schedule `Qr + Qw(t)` (via
`schedule.shifted(Qr)`), so the `underflow_split` sends `Qr` to RAS and the
scheduled remainder to wastage. **Time-threaded flow solve:** the settler's
underflow is now time-dependent, so `TakacsClarifier.flow_outputs` reads the time
from the `FlowContext` `Plant._resolve_flows` passes into every unit's
`flow_outputs` (a scheduled setpoint uses `ctx.t`; a constant setpoint ignores
it). The schedule value is a constant at a given `t`, so the affine recycle-flow
probe stays exact; constant-setpoint clarifiers are unaffected.
`split_controlled_flows` drops its `float()` cast so a traced (scheduled)
setpoint flows through. Demonstrated in `examples/bsm2_wastage_schedule.py` and
tested in `tests/integration/test_bsm2_wastage.py` (the schedule's step/validation/
shift/jit behaviour, no-solve; wired plant steps the waste flow on schedule with
RAS held fixed, and higher wastage lowers the biomass).

**Closed-loop reject control (`build_bsm2(reject=RejectStorage(control=True))`).** The storage
tank's release runs a **proportional level controller** instead of a fixed
`Q_out`: `Q_out = clip(bias + gain·(V − V_set), 0, Q_max)` (BSM2: setpoint
`0.5·Vmax`, gain 30 m³/d per m³, pump cap `Q_max = 1500` m³/d = the reference
`Qstorage_max`). The release rises with the level, so the tank self-regulates to
a steady mid-level and releases the reject *smoothly through the controlled pump
with no overflow bypass* — a functioning equalisation tank, versus the open-loop
fill-and-bypass. The net reject returned is the same, so the activated-sludge
steady state is unchanged (XB_H ≈ 2224); only the path differs (controlled
release vs bypass spill), and under a varying reject load the controlled tank
buffers (a level step up → smoothly higher release, no bypass, no chatter — the
proportional law is continuous, so unlike a fixed release > inflow it does *not*
chatter at the empty limit). **Architecture:** the controller lives *inside*
`StorageTank` (`level_setpoint`/`level_gain`/`output_flow_bias`/`output_flow_max`),
not on the signal bus, because the release feeds back into the flow network and
must be resolved *during* the flow solve — but the signal bus is computed
*after* it (`Plant._compute_signals` follows the stream sweep). Since the
release is a pure function of the volume *state*, the in-tank law resolves
exactly via the state the `flow_outputs` `FlowContext` carries. (The signal bus
remains the
right home for a non-flow actuator like the DO `kLa`, which senses a
concentration the sweep must produce first.) Demonstrated in
`examples/bsm2_reject_control.py` (open-loop bypass vs closed-loop control) and
tested in `tests/integration/test_bsm2_reject_control.py` (the release law +
flow/volume conservation, no-solve; wired plant holds a mid-level and releases
the reject with zero bypass).

**Influent hydraulic delay (`build_bsm2(hydraulic_delay=HydraulicDelay())`).** A first-order
lag on the raw influent's flow and load, modelling the transport delay of the
sewer/channel ahead of the works. `HydraulicDelayUnit`
([`plant/delay.py`](aquakin/plant/delay.py)) carries the **load** (`Q·C`) and
the **flow** `Q` as state, each relaxing to the inlet with time constant `tau`
(`d(Q·Cᵢ)/dt = (Q_in·C_in,i − Q·Cᵢ)/tau`, `dQ/dt = (Q_in − Q)/tau`); the outlet
concentration is the lagged load over the lagged flow. This is the BSM2
`hyddelay` structure (a fixed-`tau` lag on load, *not* a fixed-volume tank whose
residence time varies with flow). A flow/load pulse emerges delayed and rounded
(first-order, ~63% of a step after one `tau`); at steady state `Q→Q_in`,
`C→C_in` (a pass-through, so the operating point is unchanged). The **outlet
flow is the held-flow state**, which `flow_outputs` reads from the `FlowContext`
the plant passes into every unit (the same mechanism the storage tank uses).
Wired front-most: `build_bsm2(hydraulic_delay=HydraulicDelay())` puts it on the
influent (entry point becomes `"influent_delay.in"`, read off
`plant.influent_endpoint`), composing with the bypass
(influent → delay → bypass_split → front). **Faithfulness note:** the BSM2
reference `tau≈1e-4` d is a near-instantaneous lag whose role is to break
algebraic loops in a sequential-modular solver — aquakin resolves recycles
directly in one monolithic solve and does not need it, so the unit is here to
model a *physical* delay (`hydraulic_delay_tau`, default ~0.02 d) and to complete
the BSM2 element set. Demonstrated in `examples/bsm2_hydraulic_delay.py` (a flow
pulse emerges lagged) and tested in
`tests/integration/test_bsm2_hydraulic_delay.py` (the lag's fixed-point /
load-over-flow / first-order-response behaviour, no-solve; wired plant builds
front-most, steady state unchanged).

**Hydraulic influent bypass (`build_bsm2(bypass=InfluentBypass())`).** The BSM2
wet-weather bypass: raw influent flow above `bypass_threshold` (default 60000
m³/d) is diverted around the whole treatment train (primary, AS, secondary
clarifier) and rejoined with the clarified effluent — protecting the plant
hydraulics at the cost of releasing untreated wastewater. Built on a new
`SplitterUnit` **threshold mode** (`threshold` + `threshold_port` +
`remainder_port`): `above = max(Q_in − threshold, 0)` to the threshold port,
`min(Q_in, threshold)` to the remainder. The split is on the **raw influent**
flow (an external input), so it stays a constant within the exact recycle-flow
solve (`_resolve_flows`) and doesn't break its affine assumption — important
because the split is piecewise-linear (a kink at the threshold) and would
otherwise be non-affine in the recycle flows. The diverted flow skips the
clarifier too (matching the reference `Qbypassplant=1`: it bypasses the *plant*,
not just the AS) and joins the final effluent through a new `effluent_mix`
combiner, so the final effluent is `effluent_mix.out` (treated + bypassed) —
`evaluate_bsm2` auto-detects it. When a bypass is present `evaluate_bsm2` also
applies the BSM2 **split BOD weighting** — the benchmark's `0.65` raw-sewage
BOD₅/BODu coefficient on the *bypassed* (untreated) BOD vs `0.25` on the treated
effluent — both to the reported BOD *average* (a load-weighted average over the two
source streams `settler.overflow` + `bypass_split.bypass`) **and to the scored
`effluent_quality_index`** (the flat-weight EQI runs on the combined effluent, so
the extra `0.65 − 0.25` weight on the bypass BOD load is added back); the no-bypass
path keeps the flat `0.25` and is untouched. **This changes the influent entry point**: with
the bypass, the influent entry moves to `bypass_split.in` and the effluent to
`effluent_mix.out` -- both reported on `plant.influent_endpoint` /
`plant.effluent_endpoint`, so example/user code reads those instead of a literal.
Default `influent_bypass=False` leaves the plant and its entry point unchanged.
Demonstrated in `examples/bsm2_influent_bypass.py` (storm flow degrades the
effluent) and tested in `tests/integration/test_bsm2_bypass.py` (threshold-mode
flow split + validation; wired-plant flow balance, effluent = treated + bypass,
bypass degrades effluent, evaluation auto-detects the combined effluent).

**BSM2 performance evaluation — EQI / full OCI (`evaluate_bsm2`).** The generic
metric kernels (`aquakin/plant/metrics.py`) are wired to a concrete BSM2
flowsheet by `aquakin.plant.bsm.evaluate_bsm2(plant, solution, params)`,
returning a `BSM2Evaluation` with the EQI, the **full BSM2 OCI** and every
component term. The OCI is the Gernaey et al. 2014 index:
`AE + PE + ME + 3·sludge + 3·carbon − 6·methane + max(0, HE − 7·methane)`:
- **AE** aeration + **ME** mixing energy from the actual kLa over the run
  (`aeration_energy`, `mixing_energy`). Mixing counts the *unaerated* reactors
  (anoxic tanks need mechanical mixing; an aerated tank is mixed by its aeration)
  plus the always-mixed digester, so it spans **all** AS reactors, not just the
  aerated subset.
- **PE** pumping over the full BSM2 pump set (`pumping_energy_bsm2`): AS internal
  recycle / RAS / wastage + the primary / thickener / dewatering underflows, each
  with its own per-m³ factor.
- **sludge** disposal TSS mass flow (factor 3, not the BSM1 5); **carbon** the
  external dose `Q·conc` (`carbon_mass`, kg COD/d).
- **methane** the digester biogas credit — reconstructed from the ADM1 headspace
  gas state and parameters (`_methane_production`: the raw overpressure outflow
  `k_P·(P_gas−P_atm)` **renormalized to atmospheric pressure** by `·P_gas/P_atm`,
  `Q_gas = k_P·(P_gas−P_atm)·P_gas/P_atm`, then `CH4 = (p_ch4/P_gas)·P_atm·16/R_T
  · Q_gas`); ~1065 kg CH₄/d at the BSM2 steady state, matching the reference
  (~1065). *(The `·P_gas/P_atm` normalization is the BSM2 ADM1 convention — the
  gas-phase ODE uses the un-normalized `k_P·(P_gas−P_atm)`, but the **reported**
  gas flow, and hence the methane-production / OCI credit, is recalculated to
  atmospheric pressure. Omitting it understated the reported biogas flow, the
  methane production, and its OCI credit by `P_gas/P_atm ≈ 1.05` — about 5% — while
  leaving the gas-phase **concentrations** exactly matched, so the
  concentration-level digester validation could not see it. With the
  normalization the dynamic BSM2 methane matches the ring-test consensus to
  ~0.2% (was ~5% low) and the OCI to ~0.7% (was ~4% high). Pinned to the published
  steady-state biogas flow / methane in `tests/integration/test_plant_assembly.py::
  test_digester_gas_normalized_to_published_steady_state`.)*
- **HE** sludge-heating energy (`heating_energy`): raise the digester feed from
  its temperature (the carried stream T, else a 15 °C default) to 35 °C. At the
  BSM2 operating point methane more than covers it, so `max(0, HE − 7·methane)`
  contributes **0** — the biogas self-sufficiency the index rewards.

The aeration kLa **reads the actual value over the run** — a fixed `kla`
open-loop, or under closed-loop DO control the controller's manipulated signal
recovered per saved state by **`Plant.signals_at(t, state, params)`** (the
signal-bus analogue of `outputs_at`; `_rhs`'s signal step is the shared
`Plant._compute_signals`). All output streams are reconstructed in **one
`outputs_at` pass per saved time** (`_reconstruct`), since the indices need ~8
streams. The BSM1-form kernels (`operational_cost_index`, `pumping_energy`) are
kept for BSM1, wrapped by **`evaluate_bsm1(plant, solution, params)`** →
`BSM1Evaluation` (the BSM1 analogue of `evaluate_bsm2`): EQI + the BSM1 OCI
`AE + PE + 5·sludge`, with sludge the wastage TSS mass flow and PE over the
internal-recycle / RAS / wastage pumps. Demonstrated in
`examples/bsm1_dry_weather.py`.

**Labeled report (`str(eval)` / `eval.report()`).** Both `BSM1Evaluation` and
`BSM2Evaluation` render a units-annotated breakdown when printed: the EQI
(`kg poll.-units/d`) and OCI, then each OCI term with its physical value, units
(`kWh/d`, `kg TSS/d`, `kg COD/d`, `kg CH4/d`) and **signed OCI contribution**
(so the methane credit shows as `−6·CH4` and the BSM2 heating enters via
`max(0, HE − 7·methane)`), the effluent averages with currency-specific units,
the aerated reactors counted, and the `oci_note` caveat (always shown, wrapped).
The raw float fields stay available for programmatic use; `str` delegates to
`report()`. So the headline indices are not bare floats to misread against
published Alex 2008 / Gernaey 2014 values (issue #153).

**Top-level exports + `StreamSeries`-friendly kernels.** The metric kernels
(`effluent_quality_index`, `effluent_averages`, `derived_TSS`/`COD`/`BOD`/`TKN`,
`aeration_energy`, `pumping_energy`, `mixing_energy`, `carbon_mass`,
`heating_energy`, `operational_cost_index`, `operational_cost_index_bsm2`,
`pumping_energy_bsm2`), both evaluators (`evaluate_bsm1`/`evaluate_bsm2`) and
`check_conservation` are exported at the top level (`aquakin.…`), not only via the
deep `aquakin.plant.metrics` path. The effluent kernels and the `derived_*`
functions accept a **`StreamSeries` directly** — `effluent_quality_index(eff)` /
`derived_TSS(eff)` (model taken from the stream) — as well as the original
explicit `(t, C, Q, model)` / `(C, model)` forms; a `StreamSeries` is
duck-typed (`.t`/`.C`/`.model`), so a plain concentration array is unaffected.
Demonstrated in `examples/bsm2_evaluation.py` (open- vs
closed-loop table with the full term breakdown) and tested in
`tests/integration/test_bsm2_evaluation.py` (plant terms finite/positive,
aerated-tank detection, AE/ME/carbon match their closed forms, OCI equals the
full-formula sum; plus fast no-solve kernel tests). Note the shipped influent is
synthesised, so these are method-validated numbers, not the published EQI/OCI over
the canonical days-245–609 window (that needs the official IWA influent file).

**Single-point (steady-state) evaluation.** Every time-averaged kernel runs
through one `metrics._time_average(integrand, t)` (and the evaluator's own
`_time_average`): over a multi-point window it is the trapezoidal mean, but for a
**single saved point** — exactly what `plant.run_to_steady_state()` returns (the
terminal state only) — the average of a constant is that sample, so it returns
the **instantaneous steady-state value** instead of dividing by a zero-width
window. So the natural "run to steady state, then `evaluate_bsm1(plant,
ss.solution)`" flow returns finite, meaningful indices rather than raising
`ZeroDivisionError` (the old `aeration_energy` divided by the bare window) or a
spurious zero (the other kernels' `+1e-12` guard). Multi-point results are
unchanged.

**GHG / cost reporting + standardized scenario KPI tables.** On top of the
EQI/OCI evaluation, two presentation layers turn the physical flows a
`BSM2Evaluation`/`BSM1Evaluation` already carries into the carbon-footprint and
cost-OPEX deliverables, plus a standardized side-by-side KPI table:
- **Carbon footprint** ([`aquakin/plant/ghg.py`](aquakin/plant/ghg.py)):
  generic CO₂e kernels (`co2e_from_energy`, `n2o_n_to_co2e` — N₂O-N → N₂O via
  44/28 then ×GWP, `methane_to_co2e`) plus `stripped_n2o` (the aeration-rate
  stripping `Σ kLa_N2O·(S_N2O−S*)·V`, so only aerated tanks emit), assembled by
  `carbon_footprint(energy_kwh, *, grid_factor, n2o_emission, methane_production,
  ch4_fugitive_fraction, biogas_recovered_kwh, ...)` into a `CarbonFootprint`
  (direct N₂O + grid-energy CO₂e + fugitive CH₄ − biogas-energy credit). IPCC
  AR6 100-yr GWP defaults (N₂O 273, biogenic CH₄ 27) and a representative grid
  factor, all overridable. The plant-coupled `direct_n2o_emission(plant, solution,
  params)` (in [`bsm/evaluation.py`](aquakin/plant/bsm/evaluation.py)) reconstructs
  the stripped N₂O from a solved plant (reusing the control-aware `_kla_history`
  and reading the dissolved `SN2O` per reactor); it returns **0** when the AS
  model has no `SN2O` state (the standard ASM1 BSM2 plant — only an N₂O-capable
  model such as `asm3_2step_n2o` gives a non-zero direct term).
- **Operating cost** ([`aquakin/plant/cost.py`](aquakin/plant/cost.py)):
  `operating_cost(*, energy_kwh_per_d, carbon_kg_cod_per_d, sludge_kg_tss_per_d,
  methane_kg_per_d, factors, co2e_per_d)` prices energy / external carbon /
  sludge disposal / biogas credit (`CostFactors`, currency/d) with an optional
  annualised CAPEX and a CO₂e carbon charge → `OperatingCost` (per-day +
  annual).
- **Standardized KPI comparison** (`kpi_comparison` in
  [`integrate/experiments.py`](aquakin/integrate/experiments.py)): tabulates
  heterogeneous report objects (`BSM2Evaluation`, `CarbonFootprint`,
  `OperatingCost` — anything exposing `.kpis()`, or a plain dict) side by side
  into a `KPIComparison` (union of KPI columns, `.best(kpi, minimize=)`). The
  report-object companion to `compare_scenarios` (which runs a model and
  tabulates a fixed output vector). The four evaluation/report dataclasses each
  expose `.kpis()`, and the evaluators a `total_energy()` (AE+PE[+ME]) — the
  energy basis for the GHG/cost layers.
  Demonstrated in `examples/bsm2_ghg_cost_report.py`; kernels + KPI logic tested
  fast in `tests/unit/test_ghg_cost.py`, the plant-coupled path on the shared
  BSM2 solve in `tests/integration/test_bsm2_evaluation.py`.

**Activated-sludge design layer — SRT / HRT / F:M (`aquakin/plant/design.py`).**
Plants are specified in the quantities the solver integrates (tank `volume`,
fixed pump flows, per-species `kLa`), but engineers design in the quantities
those derive *from*: the solids retention time (SRT / sludge age), the hydraulic
retention time (HRT) and the food-to-microorganism ratio (F:M). The design layer
bridges both directions, exported at top level (`aquakin.size_activated_sludge`,
`aquakin.sludge_metrics`, `ActivatedSludgeSizing`, `SludgeMetrics`):
- **Forward sizing** — `size_activated_sludge(SRT=…, HRT_h=…, Q=…, …)` →
  `ActivatedSludgeSizing`. `V = Q·HRT`; the wastage `Qw` from the SRT under a
  stated wasting model: `wastage_from="mixed_liquor"` (hydraulic/Garrett control,
  `Qw = V/SRT`, concentration-independent) or `"underflow"`
  (`Qw = V/(SRT·thickening_ratio)`). Optional `n_tanks`/`volume_fractions` split
  the basin into a CSTR cascade; `internal_recycle_ratio`/`ras_ratio` report the
  pump flows.
- **Achieved metrics (closing the loop)** — `sludge_metrics(plant, solution, …)`,
  also reachable as **`plant.sludge_age(solution)`** (a thin `Plant` method with a
  lazy import to avoid the plant↔design circular). SRT is an *emergent* property of
  `Qw`, so rather than guessing it this reports what the solved model achieved,
  time-averaged over the window: **SRT** = system solids inventory / (wastage +
  effluent solids loss); **HRT** = total reactor volume / influent flow; **F:M** =
  influent BOD load / reactor TSS mass. Reactors are auto-detected (the
  `aeration`-carrying `CSTRUnit`s, so the ADM1 digester is excluded); the
  effluent/wastage ports auto-detect for BSM1/BSM2; the secondary-clarifier sludge
  blanket is included via a new **`TakacsClarifier.solids_mass(state)`** accessor
  (the stateless `IdealClarifier` holds ~0), so the Takács plant correctly reports
  a larger system SRT than the ideal clarifier at the same `Qw`.
- `build_bsm1` gains a `wastage_flow=` argument so `Qw` can be varied without
  reaching into the wiring. Worked end-to-end in
  `examples/bsm1_target_srt.py` — a secant iteration on `achieved_SRT(Qw) − target`
  lands the wastage flow that hits a target sludge age (the by-hand iteration the
  layer replaces; it converges to `Qw ≈ 269` m³/d for a 10-day SRT and shows the
  mixed-liquor forward guess `Qw ≈ 599` differs because BSM1 wastes from the
  thickened underflow). Tested in `tests/integration/test_design.py` (fast
  sizing-relation + validation + `solids_mass` tests in the PR gate; slow
  plant-solve tests for the achieved metrics: SRT/HRT/F:M sensible, HRT = V/Q,
  `plant.sludge_age` delegation, SRT monotone-decreasing in `Qw`, Takács inventory
  raising SRT).

The **ASM1↔ADM1 interfaces** (`aquakin/plant/interfaces.py`, `ASM1toADM1` /
`ADM1toASM1`) are the continuity-based BSM2 interfaces (Nopens et al. 2009 /
Rosen & Jeppsson 2006). `asm2adm` removes the COD demand of O₂/NO₃, then
partitions the remaining ASM COD into ADM substrates under a nitrogen budget
drawn greedily from a priority-ordered list of N pools, with inorganic carbon
and the strong-ion difference (`S_cat`/`S_an`) from a charge balance at the
digester pH; `adm2asm` maps biomass→XS+XP, solubles→SS (H₂/CH₄ stripped),
inerts→XI/SI and inorganic N→SNH. The reference's deeply nested `if/else`
nitrogen cascades are written here **branch-free with `jnp.minimum`** (the
unrolled conditionals are mathematically a greedy allocation), so the maps are
AD-clean. Both **conserve total COD** (`asm2adm` minus the electron-acceptor
demand; `adm2asm` minus the stripped `S_h2`+`S_ch4`) **and total nitrogen** —
verified to `rel 1e-6` in `tests/integration/test_interfaces.py`. The COD
conservation holds whenever the `asm2adm` electron-acceptor (O₂+NO₃) demand does
not exceed the degradable COD it draws from (`SS+XS+XB_H+XB_A`) — always true for
a real near-anoxic digester feed; in the pathological case (recycled nitrate far
exceeding the substrate) the surplus demand is dropped and COD is over-conserved,
mirroring the reference. Construct `ASM1toADM1(strict=True)` to instead raise
(jit/AD-safe, via `eqx.error_if`) when the demand is not fully absorbed, asserting
the feed stays in the intended regime (default `False` is bit-faithful to the
reference). Only the BSM2 `fdegrade = 0` case is implemented (other values raise
`NotImplementedError`).
The charge balances (inorganic carbon + `S_cat`/`S_an` in `asm2adm`, alkalinity
`SALK` in `adm2asm`) are evaluated at the **digester pH**, fed back from the
digester's own state-derived (charge-balance speciation) pH each RHS — as in the
benchmark, where the interface pH is the digester's. The plumbing: each interface
declares `needs_dest_pH` (`asm2adm`, whose destination is the digester) or
`needs_src_pH` (`adm2asm`, whose source is the digester), and
`Plant._collect_inputs` reads that unit's `operating_pH(state, params)` and passes
it as `translate(..., digester_pH=...)`. The `pH_adm` parameter (default 7.0) is
only the fallback for a standalone `translate` call with no plant to supply it.
This is the **only pH-dependent part of the maps** (the inorganic-carbon and
alkalinity charge balances); the COD/N partition is pH-independent, so feeding the
real digester pH (~7.27) instead of the fixed 7.0 leaves every substrate pool
unchanged and only corrects the charge-balance pools: the post-interface digester
**`S_IC`** matches the published BSM2 feed to **0.13%** (was 7.6% at the fixed 7.0)
and the strong-ion `S_an` to 0.015% (was 1.17%), eliminating what had been the
digester's largest steady-state residual. The validated BSM2 reactor steady state
is unchanged (≤0.06%); the digester's remaining ~1.3% is the headspace CO₂
(the charge-balance-pH vs reference-algebraic-pH difference, not the interface).

---

