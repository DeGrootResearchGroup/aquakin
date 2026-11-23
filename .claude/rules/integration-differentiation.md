---
paths:
  - "aquakin/integrate/_common.py"
  - "aquakin/integrate/_config.py"
  - "aquakin/integrate/_solver.py"
  - "aquakin/integrate/discrete_adjoint.py"
  - "aquakin/integrate/forward_sensitivity.py"
  - "aquakin/integrate/forward_solve.py"
  - "aquakin/integrate/_simultaneous_corrector.py"
  - "aquakin/integrate/sensitivity.py"
  - "aquakin/integrate/colored_jacobian.py"
  - "aquakin/plant/sensitivity.py"
  - "aquakin/plant/dynamic_solve.py"
  - "aquakin/plant/colored.py"
---

# Rules — differentiating stiff models

`dtmax`, forward sensitivities and the hand-written discrete adjoint. The
measurements behind these rules are in `.claude/notes/differentiation-log.md`,
`.claude/notes/plant-sensitivity-log.md` and `docs/plant_performance.md`; the
mechanics are in the module docstrings of `aquakin/integrate/forward_sensitivity.py`,
`aquakin/integrate/forward_solve.py`, `aquakin/integrate/_simultaneous_corrector.py`
and `aquakin/integrate/discrete_adjoint.py`.

### Selecting the differentiation route

- **The public selector is `diff=DifferentiationConfig(mode=..., method=...)`**
  (`aquakin/integrate/_config.py`): `mode` is the AD direction (`"reverse"` /
  `"forward"`), `method` how the adjoint or sensitivity is formed (`"stable"` /
  `"through_solve"`). `IntegratorConfig(dtmax=...)` carries the step cap.
- **Decode `(mode, method)` through the class, not per entry point**
  (`gradient_backend()`, `reactor_adjoint()`; the plant's
  `aquakin/plant/dynamic_solve.py::_resolve_diff_config`). The strings
  `"auto"` / `"jax_adjoint"` / `"stable_adjoint"` are internal routing values, not
  public keywords: do not document them or ask users for them.
- **An error message that advises a differentiation setting names one of the
  `*_DIFF` constants in `_config.py`**, so the advice cannot drift from the real
  keywords.

### Differentiating stiff models (`dtmax`)

- **Reverse mode *through* a stiff implicit solve goes non-finite above a
  step-size threshold; forward mode through it stays finite at any step** (it only
  loses accuracy). The primal is fine at any step (L-stable ESDIRK).
- **The cause is an overflow in the reverse accumulation**, controlled by the
  per-step stiffness `γ·dt·‖J‖`: reverse AD forms cotangents of the large stored
  stage values `f_i ∼ ‖J‖·y`. It is *not* ill-conditioning of `I − γ·dt·J` (which
  stays well-conditioned), not the checkpointing (`RecursiveCheckpointAdjoint` and
  `DirectAdjoint` reverse fail alike), not the positivity limiter. Do not try to fix
  it there, nor with a stiffness-aware step controller or k-space stage storage:
  both were tried and do not remove it (evidence in the differentiation note).
- **`dtmax` defaults to `None` (uncapped)**, the fastest plain forward solve; keep
  it so. It reaches the `PIDController` through `build_step_controller`. Set it only
  for a reverse gradient with `method="through_solve"` on a stiff model, to a small
  multiple of the fastest reaction timescale; the threshold is model-dependent.
- **Use a cap-free route instead where one exists:**
  - a sensitivity `dC/dθ` or a screen: the forward sensitivity
    (`solve_sensitivity`, `aquakin.forward_sensitivity`) or `mode="forward"`
    (`sensitivity`, `dgsm`), which is also the faster screen;
  - a scalar-loss gradient of many parameters (calibration): the reverse
    `method="stable"` discrete adjoint (below).
- **The cap is still required** for `calibrate`'s default
  `method="through_solve"`; for `calibrate` on a reactor without a single-location
  `conditions` (`method="stable"` rejects it); and for `jax.grad` through a reactor's
  `solve` (reactors have no stable reverse adjoint). `calibrate` has no
  forward-sensitivity residual Jacobian (`mode="forward", method="stable"` is
  rejected).

### Forward sensitivity of a reactor (`solve_sensitivity`, `shared_factor`)

`BatchReactor`, `PlugFlowReactor` and `BiofilmReactor` integrate the variational
equation `dS/dt = J·S + f_θ` alongside the state as one augmented `[y; S]` system
whose controller bounds `S`, with the primal uncapped.

- **The augmented right-hand side is `f(y)` plus one JVP of `f` per sensitivity
  parameter; do not form `J` or `f_θ` explicitly.** The JVP differentiates through
  the state-derived-pH speciation solver, the positivity limiter and the
  density-cap throttle, so none of them needs special-casing; keep them
  JVP-differentiable.
- **The solver and controller come from `build_implicit_solver` /
  `build_step_controller`**, the same configuration as the forward and
  discrete-adjoint paths.
- **The default `atol_S` is `forward_sensitivity.sensitivity_atol`** (CVODES
  `atol_y/|θ_j|`), shared with the plant's staggered corrector: one default, both
  paths.
- **`shared_factor=True` must give the same Newton step as the dense augmented
  solve** (results bit-equivalent): the `SimultaneousCorrector` changes only the
  linear-algebra cost (factor `D = I − γ·dt·J` once per step, forward-substitute
  across the `S` columns). `shared_factor=None` (default) selects it for more than
  one parameter and the dense solve for one, where it has no advantage.
- **Known cost, not a bug:** as a custom *solver* it sees only the augmented
  operator, so it materializes `D` and the coupling blocks by `n` probes of the
  augmented `M.mv`, each a second-order JVP of `f`. The plant avoids it with the
  staggered corrector; moving the reactors off it needs a lean Kvaerno5 loop, not a
  change to the solver. Whether it beats a capped `jacfwd` depends on the span
  (the differentiation note).

### Forward sensitivity of the plant (`Plant.solve_sensitivity`)

- **Use `Plant.solve_sensitivity`, not `jacfwd` through `plant.solve`, for a
  dynamic plant over a long horizon:** `jacfwd` goes non-finite numerically while
  the true sensitivity stays bounded (the plant-sensitivity note).
- **It runs on the plant's cached recycle / flow maps**, built once per solve from
  the runtime `params` and closed over, so the linearization drops `∂M/∂θ`. This
  is exact for kinetic `wrt` parameters and for `operating=` influent scales; a
  flow-setpoint sensitivity needs the per-call map and is not wired, so do not
  offer one without it.
- **Operating parameters go through the influent `design=` override**
  (`Q_scale` / `C_scale` multipliers on the sampled series), the same path the
  steady-state IFT uses: do not add a second one. Unit-level setpoints such as
  `kLa` have no per-RHS `design` hook (they are baked into precomputed aeration
  vectors), so they are not supported.
- **Two correctors, one builder:** `_CORRECTORS` in `plant/sensitivity.py`, both
  built by `_build_sensitivity_solve`. `"staggered"` (default) is forward-only and
  needs concrete `params` / `y0` (a tracer raises `TypeError` naming
  `corrector="simultaneous"`). Keep `"simultaneous"` (the diffrax augmented solve,
  `order=3`): it is the differentiable one and the independent check of the
  staggered result.

#### The staggered corrector and the lean loop

`integrate/forward_solve.py::staggered_sensitivity_solve` runs on the lean
`lax.while_loop` Kvaerno3 loop it shares with `forward_fast`.

- **One loop for both:** one `_integrate` driver, one `_stage_solve` stage
  iteration, one `StepControl`. Change them together; do not fork a sensitivity
  copy.
- **`J` only steers the iteration; every residual is exact.** `J` is the colored
  Jacobian from `ColoredJacobianManager.forward_root_finder`, dense `jacfwd` if its
  guard fails, so a pattern miss costs iterations, never accuracy.
- **The sensitivity stage is linear in `K` and reuses the state's factor:** its
  residual is one `jax.linearize` of `f` at `(Y_i, θ)` pushed along the `k` columns.
  Keep it free of second derivatives and of any `n(1+k)` factorization.
- **The state error test runs first;** a step rejected on the state skips the
  sensitivity work (`lax.cond`).
- **The error norm is CVODES', `max(WRMS_y, max_j WRMS_{S_j})`.** Do not replace
  it with one RMS over the `n(1+k)` components: the `S` block would dilute the
  state's error as `k` grows. `sens_error_control=False` (CVODES `errconS = false`)
  drops `S` from it: staggered only (`ValueError` on the simultaneous corrector),
  part of the compile-cache key, and passed through by forward-mode
  `dynamic_sensitivity` / `dynamic_dgsm`. With it the error of `S` is unbounded;
  it suits screening and very long runs.
- **Step control:** the PI controller plus the contraction-rate growth limiter,
  and no growth on an attempt accepted right after a failed one; `factormax` maps
  to `StepControl.maxfac` (default 2, against diffrax's 10).
- **The stage iteration is Anderson-accelerated** (`_ANDERSON_DEPTH = 2`; `0` is
  the plain chord iteration), preconditioned by the step's factorization. **Its
  contraction / divergence tests must read the plain step `−M⁻¹R(x)`, not the
  accelerated update:** accelerated updates are non-monotone and testing them
  stopped converging solves as "diverged". Newton-Krylov (GMRES) stage updates were
  measured slower than Anderson and deleted; do not reintroduce them without new
  measurements.
- **The stage test reads the stage-value increment `γh·Δk`** (`_stage_weight`,
  the same in any time unit) at `_KAPPA = 0.01`. Do not go back to the slope
  increment `Δk`, which is unit-dependent. Retune κ only on a multi-day BSM2 run
  and the settler-layer error, not on a 1-day window (the plant-sensitivity note).
- **A stage accepts its first correction on its size** (`η = 1`, rate reported 0).
  Do not remove it: without it a solve started at a steady state stalls, because
  the round-off corrections after a converged guess read as divergence and feed the
  growth limiter.
- **At equal `rtol` the lean loop is less accurate than diffrax on the Takács
  settler blanket layers**, because diffrax's failed root finds force it to take
  twice the steps there. This is a step-count effect, not a lean-loop bug; compare
  at matched step counts (the plant-sensitivity note, which also lists the
  profiling and probe scripts).

#### Compiled once per structure

The jitted solve lives in `plant._jit_cache` (the `SolveCache.jit` dict, so
`invalidate()` drops it). **Every option that changes the trace must be in its
key** (corrector, `wrt` and operating indices, the `t_eval` shape, the colored and
map-constant flags, `factormax`, `dtmax`, `max_steps`, `sens_error_control`), and
**the parameter values, `y0`, the `t_span` endpoints, the `t_eval` values and the
tolerances must stay runtime arguments**, so repeated calls compile once.

### Reverse mode without the cap — the discrete adjoint

`aquakin.implicit_euler_adjoint_solve` (first order, the baseline) and
`aquakin.esdirk_adjoint_solve` (a general ESDIRK, default `Kvaerno5`), in
`aquakin/integrate/discrete_adjoint.py`. The forward is an ordinary adaptive diffrax
solve; the backward is the discrete adjoint written by hand, one transposed solve
through `I − γ·dt·J` per stage (Sandu's construction), so no large cotangent is
ever formed.

- **Do not differentiate through the solve inside these functions.** The overflow
  lives in AD of the stage arithmetic; the hand-written transposed solve is what
  keeps the gradient finite.
- **The result must stay the exact gradient of the discrete solve:** any change is
  checked against closed-form gradients and against the capped through-solve
  gradient of the same method.
- **One driver:** both solvers go through `_discrete_adjoint_solve`; the only
  per-method piece is the single-step `step_adjoint`. Add a method there, not as a
  copy of the driver.
- **Observation times are step boundaries.** With `t_eval` the forward is forced
  to land on every observation (`diffrax.ClipStepSizeController(step_ts=t_eval)`), so
  each cotangent is injected at a step without differentiating through dense
  interpolation. Keep that rather than interpolating.
- **The ESDIRK stages are saved by the forward, not recomputed by default.** The
  forward runs `SaveAt(steps=True, dense=True)` and the backward rebuilds each stage
  as `Yᵢ = yₙ + Σⱼ A[i,j]·kⱼ` (the dense-output `kⱼ` already carry `dt`); ESDIRK sets
  `save_stages=`, implicit Euler leaves it off. Do not go back to a Newton recompute
  on the default path: it was the dominant backward cost (`docs/plant_performance.md`,
  "Saved-stage discrete-adjoint backward").
- **`low_memory=True` is the one exception:** it drops the ~`n_stages`× dense buffer
  and recomputes the stages by a fixed Newton scan (`newton_iters`). It is guarded to
  the singly-diagonal ESDIRK shape (explicit first stage, constant γ — Kvaerno3 /
  Kvaerno5); any other tableau falls back to the saved-stage path with a
  `RuntimeWarning`. Users reach it through
  `DifferentiationConfig(adjoint_low_memory=True)`.
- **Explicit time dependence needs `time_dependent=True`** (time carried in the
  state). The default assumes an autonomous right-hand side (the stage times do not
  enter) and silently gives a wrong, possibly zero, gradient for a time-coupled
  parameter. `plant.solve` passes `time_dependent=True`; any new caller whose field
  depends on `t` must too.
- **The saved-trajectory buffer is `DifferentiationConfig.adjoint_max_steps`**; it
  must exceed the forward step count. The backward loop runs over the accepted
  steps only, but the buffer (×`n_stages` with saved stages) is allocated in full,
  so keep it a tight bound.

#### Where the discrete adjoint is wired

- **`calibrate(diff=DifferentiationConfig(method="stable"))`** uses
  `esdirk_adjoint_solve` on the reactor's `model.dCdt`, for batch reactors only. It
  is reverse-only, so `mode="forward"` with `method="stable"` is rejected. Its
  forward is the function's default `Kvaerno5`, not the reactor's integrator, so it
  agrees with the capped `through_solve` path to the optimizer tolerance, not
  bit-for-bit.
- **`plant.solve` with the default `diff` (`mode="reverse", method="stable"`)**
  routes a concrete forward solve to the fast cached through-solve path and a solve
  whose `params` / `y0` are tracers to `esdirk_adjoint_solve`
  (`time_dependent=True`), so a stiff plant gradient is finite with nothing to set.
  `event=` pins the through-solve path (so `run_to_steady_state` is unaffected);
  `integrator=IntegratorConfig(dtmax=...)` with `method="stable"` raises
  `ValueError`. A `jax.jit`-wrapped forward solve looks traced and so takes the
  discrete adjoint (correct but uncached); pass `method="through_solve"` for the
  cached path.
- **The plant's cached recycle map is used only on the primal right-hand side and
  must be `stop_gradient`'d**; the map-recomputing `rhs` drives the `∂f/∂θ` VJP, so
  a flow-setpoint gradient keeps `∂M/∂θ` exact.
