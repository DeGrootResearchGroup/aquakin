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

`dtmax`, the adjoints, the discrete adjoint and forward sensitivities. Split out
of `.claude/rules/integration.md`. Measurements live in
`.claude/notes/plant-sensitivity-log.md` and `docs/plant_performance.md`.

### Differentiating stiff models (`dtmax`)

Every reactor takes an optional `dtmax` (maximum integrator step), threaded
into the `PIDController`. Default `None` (uncapped) — fastest for plain forward
solves.

**Set `dtmax` when taking a *reverse-mode* gradient of a very stiff model.**
`Kvaerno5` is L-stable, so a solve can take steps far larger than the fastest
reaction timescale and simply damp the unresolved fast modes — the primal is
fine at any step. Differentiation splits by mode: **forward mode**
(`jax.jvp`/`jax.jacfwd` via `DirectAdjoint`) stays **finite at any step**,
losing only accuracy when the fast modes are unresolved; **reverse mode**
(`jax.grad`, the discrete adjoint) returns **non-finite** values above a
step-size threshold. The reverse failure is **not** a near-singular
per-step solve, despite the bare reaction Jacobian being stiff *and*
ill-conditioned (condition number ~1e20 at t=0 for the Khalil/extended sewer
models, |eig| up to ~1e5 d⁻¹). That cond(J) is dominated by structurally
near-zero eigenvalues of depleted/dormant species; the operator the implicit
step and its adjoint actually invert, `I − γ·dt·J` (γ≈0.26 for Kvaerno5),
regularizes those directions and stays **well-conditioned** across the failing
step range (cond ~650 at dt=1e-2, ≤~3e4 at dt=1e-1). The failure is instead an
**overflow in the reverse (backward) accumulation**, controlled by the per-step
stiffness `γ·dt·‖J‖` (fast-mode timescales per step), not by operator
conditioning — consistent with the threshold scaling below (the steeper-Jacobian
half-order variants need a tighter cap). Forward-mode tangent propagation, in the
same direction as the primal and through the same well-conditioned operator,
stays finite. Capping `dtmax` to a small multiple of the fastest reaction
timescale bounds `γ·dt·‖J‖`; the resulting reverse gradient is finite and matches
both forward mode and finite differences. This is **reverse-mode-specific and
independent of the adjoint flavour** — `RecursiveCheckpointAdjoint` and
`DirectAdjoint` reverse both fail identically, so it is not the checkpointing.
(It is also *not* the positivity limiter, *not* the zero-valued initial
species, and *not* stiffness alone: a 2–3 species stiff toy differentiates
finitely in reverse even at 1e7 d⁻¹ — the failure needs the full coupled,
many-species system.) The threshold is model-dependent: ~5e-3 d for the
Khalil Monod biofilm, ~10× tighter (~5e-4 d) for the half-order variants whose
√C kinetics steepen the Jacobian; the study uses `dtmax = 1e-4` d (3e-5 d for
the stiffer balanced base) — inside both. Because calibration needs the reverse
adjoint (one pass for the whole parameter gradient), the cap matters there; a
forward-mode sensitivity screen is unaffected, and is also faster (see
`dgsm(diff=DifferentiationConfig(mode="forward"))`). A future alternative is a quasi-steady-state (QSS)
reduction of the near-instantaneous fast reactions, which would remove the
stiff modes entirely and avoid needing the cap.

**Cap-free alternative — forward-sensitivity solve.** For the sensitivity
`dC/dθ` itself there is now a way that needs **no cap at all**:
`reactor.solve_sensitivity(...)` (and the free function
`aquakin.forward_sensitivity(...)`). Instead of differentiating *through* the
stiff solve, it integrates the variational equation `dS/dt = J·S + f_θ`
*alongside* the state — one augmented `[y; S]` system, stock `Kvaerno5` +
`PIDController` whose error norm now also bounds `S` — so the adaptive
controller tightens the step only where the *sensitivity* is stiff and runs
free elsewhere. The primal stays uncapped and the returned `S` is finite and
exact (it matches a tightly-capped `jacfwd` to ~1e-8; validated against the
closed-form sensitivity of first-order decay and against capped `jacfwd` on the
stiff Khalil biofilm in `tests/integration/test_forward_sensitivity.py`). The
augmented RHS is `f(y)` plus one JVP of `f` per sensitivity parameter (the
JVP's primal gives `f(y)` for free; the JVP also differentiates through the
state-derived-pH speciation solver, the positivity limiter and the density-cap
throttle, so no special-casing). Implemented in
[`integrate/forward_sensitivity.py`](aquakin/integrate/forward_sensitivity.py)
on `BatchReactor`, `PlugFlowReactor` and `BiofilmReactor`.

**`Plant.solve_sensitivity` — the stable forward mode on the plant.** A forward
sensitivity of a stiff dynamic plant by `jacfwd` through `plant.solve` is finite
per step but **non-finite over a long horizon**. That failure is **numerical, not
a genuine tangent divergence**: the true sensitivity stays bounded (`‖S‖`
*oscillates* with the diurnal load over 17 days — 45→196 — rather than growing),
and a stock-`Kvaerno5` augmented solve blows up on the 20-day BSM2 window too.
`Plant.solve_sensitivity(params, wrt, *, operating=, t_span, t_eval=, y0=, rtol=,
atol=, sens_rtol=, factormax=, dtmax=, max_steps=, corrector=, return_stats=)`
integrates `[y; S]` with the controller bounding `S`, on the plant's **cached
recycle/flow maps** (built once per solve inside the compiled function, from the
runtime `params`, and closed over — so the linearisation w.r.t. the sensitivity
parameters drops `∂M/∂θ`: **exact for kinetic parameters**, `M` depending only on
the flow setpoints; a flow-setpoint sensitivity needs the per-call map, not
wired). Returns `(ts, ys, S)`, `S` shape `(n_t, ndof, k)` (plus a `SolveStats` for
`return_stats=True`). Two correctors (`_CORRECTORS` in `plant/sensitivity.py`),
built by the one `_build_sensitivity_solve`:
- **`corrector="staggered"` (default, #654)** —
  `integrate/forward_solve.py::staggered_sensitivity_solve` on the lean
  `lax.while_loop` Kvaerno3 loop it shares with `forward_fast` (one `_integrate`
  driver over a pytree state, one `_stage_solve` stage iteration, one `StepControl`).
  Per implicit stage: converge the state (chord on `LU(I − γhJ)`, `J` the
  **colored** Jacobian from `ColoredJacobianManager.forward_root_finder`, dense
  `jacfwd` if its guard fails — `J` only steers, so a pattern miss costs
  iterations, never accuracy), then solve the sensitivity stage, **linear in
  `K`**: `K = J(Y_i)(S + h Σa_ij K_j + γhK) + f_θ(Y_i)`, by the same chord on the
  same factor, its residual one `jax.linearize` of `f` at `(Y_i, θ)` pushed along
  the `k` columns (`vmap`). It is the augmented-system stage to Newton tolerance,
  with **no second derivatives** (the simultaneous corrector's `n·k`
  second-order probes per Jacobian are gone) and no `n(1+k)` factorization. The
  state error test runs first; a step rejected on the state skips the sensitivity
  work (`lax.cond`). Error norm: **CVODES** — `max(WRMS_y, max_j WRMS_{S_j})` (the
  augmented diffrax path's single RMS over `n(1+k)` components lets the `S` block
  dilute the state's error as `k` grows); `sens_error_control=False` (CVODES
  `errconS = false`, staggered only, `ValueError` on the simultaneous corrector,
  in the compile-cache key, passed through forward-mode `dynamic_sensitivity` /
  `dynamic_dgsm`) drops `S` from it. Step control: the PI controller plus
  the **contraction-rate growth limiter** (`StepControl.factor`; the rate is the
  max over state and sensitivity iterations) and **no growth on an attempt
  accepted right after a failed one** (Hairer's rule; `factor`'s
  `after_failure`, carried by `_integrate`); `factormax` maps to
  `StepControl.maxfac` (default 2, vs diffrax's 10). **Forward-only**: concrete
  `params`/`y0` required (a tracer raises `TypeError` naming
  `corrector="simultaneous"`). Default `atol_S` shared with the augmented path
  (`forward_sensitivity.sensitivity_atol`, CVODES `atol_y/|θ_j|`). Measured on a
  synthesised-influent BSM2 day (`.claude/notes/plant-sensitivity-log.md`, which
  carries the configuration and the container's ~2× timing noise): several times
  faster than the simultaneous corrector at `k=16`, but the per-attempt cost is
  **not flat in `k`** (~2.7× from `k=0` to `k=16`: the `k`-column Jacobian-vector
  products, ~10 per accepted step). Before Anderson acceleration ~20% of attempts
  failed Newton — the step-start `J` is wrong in one or two directions at those
  stages (mostly the Takács settler), not grown into divergence; a dense `J` fails
  the same. With the current defaults (Anderson, the stage-value test, the hold
  rule) it is ~9% at `k=0` and ~3% at `k=16` on that day.
  **The stage iteration is Anderson-accelerated** (`forward_solve._stage_solve`,
  shared with `forward_fast`; `_ANDERSON_DEPTH = 2` by default, `0` = the plain
  chord iteration): the factored `M` preconditions a Type II Anderson mix, row by
  row for the sensitivity block. The contraction/divergence tests read the plain
  step `−M⁻¹R(x)` (the preconditioned residual) — accelerated updates are
  non-monotone, and testing them stopped converging solves as "diverged". On the
  BSM2 day: `k=0` ~30% faster, `k=16` ~5%; Newton failures −50% / −67%, but the
  freed step growth becomes error-test rejections. Stages whose plain iteration
  diverges at once are out of its reach. Newton-Krylov (GMRES) stage updates were
  measured and **deleted** — more failures removed, but slower than Anderson at
  both `k` (the note). **The stage test reads the stage-value increment
  `γh·Δk`** (`_stage_weight`: the same in any time unit) at **`_KAPPA = 0.01`**;
  until 2026-11-01 it read the slope increment `Δk` against the state tolerance —
  `1/(γh)` (~2000× at BSM2 steps) stricter and unit-dependent — at `_KAPPA = 0.1`
  tuned in those units. κ is set by an 8-day BSM2 check (worst-layer error 3.3e-2
  of its peak at 0.1, 4.9e-3 at 0.01, the old loop's 5.4e-3 at 0.5-d saves), not by
  a 1-day window: the stage error shows on the Takács blanket layers, whose
  min-flux rule switches branch every few steps and makes fast pulses whose
  timing the stage error shifts. **A stage accepts its first correction on its
  size** (`η = 1`, reporting rate 0): without it a solve started at a steady state
  stalled — the corrections after an already-converged guess are round-off, and
  their ratio read as divergence and fed the growth limiter (BSM1, constant
  influent, 40 days of operating sensitivity from the steady state: over 200 000
  steps, 357 with it; pinned by
  `test_a_solve_started_at_its_steady_state_takes_long_steps`); it rarely fires
  elsewhere. Net, against the 7de1fbe defaults on the BSM2 day: `k=0` 8677 vs
  10695 right-hand-side evaluations (−19%), `k=16` 1967 vs 2135 attempts (−8%).
  `sens_error_control=False` at `k=16` (shipped configuration): 1156 attempts,
  9.3 s against 16.7 s, `S` error 4.1e-4 of a column peak (controlled 2.5e-4);
  not bounded — across earlier settings it ranged 2.8e-4 to 2.4e-3. **On those
  blanket layers the lean loop trails diffrax at equal `rtol`** (pulse-timing
  errors up to 5.6% of a layer's peak at 1e-4, diffrax 0.4%) **because diffrax takes
  twice the steps**: 6852 of its 6857 rejections are failed root finds on the same
  switching, which set its step size whatever its `rtol`; the lean loop (no
  root-find failures) takes the error test's steps, and at `rtol = 1e-5` matches
  diffrax-at-1e-4 there with fewer attempts. The outputs do not see the pulses (the
  note, "Why the lean loop trails diffrax on the settler blanket";
  `scripts/probe_settler_accuracy.py`) — all with the Takács rule; the Godunov
  settler flux, the default since 2026-11-02 (`TakacsClarifier(interface_flux=…)`,
  plant rules), removes the switching: lean attempts 8901 → 3177, diffrax
  17249 → 3653 on that case. Error-test
  rejections are not influent breakpoints; at `k=16` most are driven by the
  sensitivity error estimate jumping (the note, "Where the error-test rejections
  come from"). The scripts (lean-loop settings specs in `_lean_settings.py`):
  `profile_plant_sensitivity.py --variants` (`--reference-rtol` for errors
  against a tight solve), `probe_newton_drift.py`, `probe_step_rejections.py`.
- **`corrector="simultaneous"`** — the diffrax augmented solve
  (`augmented_forward_sensitivity`, `order=3`, decoupled Newton through
  `build_implicit_solver` / `build_step_controller`, the block-arrow
  `SimultaneousCorrector` for `k > 1`). Kept because it is **differentiable** and
  an independent check of the staggered result; much slower (see
  `.claude/notes/plant-sensitivity-log.md`).

**Compiled once per structure.** The jitted solve lives in `plant._jit_cache`
(the `SolveCache.jit` dict, so `invalidate()` drops it), keyed by
`("solve_sensitivity", corrector, wrt+operating indices, op_meta, t_eval shape,
colored?, recycle/flow-map-constant flags, factormax, dtmax, max_steps)`; the
parameter values, `y0`, `t_span` endpoints, `t_eval` values and `rtol`/`atol`/
`sens_rtol` are runtime arguments. Validated: staggered vs closed form on a
one-CSTR plant (kinetic + both operating kinds, both correctors) and the
no-retrace check in `tests/integration/test_staggered_sensitivity.py`; vs
`jacfwd` and vs the simultaneous corrector on BSM1 and a BSM2 window in
`tests/integration/test_dynamic_sensitivity.py` (`heavy`).
**Operating-parameter sensitivity (`operating=`).** Beyond the kinetic `wrt`
parameters, `solve_sensitivity(operating=[...])` differentiates the dynamic output
w.r.t. **operating conditions** — a differentiable multiplicative scale (nominal
`1.0`) on an influent's flow (`{"kind": "influent_flow", "port": p}`) or on a
species' load (`{"kind": "influent_concentration", "port": p, "species": s}`). Each
appends one column to `S` after the `wrt` columns. They thread into the augmented
RHS through the influent `design=` override (the same path the steady-state IFT
uses — extended from an absolute replacement to also accept `Q_scale`/`C_scale`
multipliers on the time-sampled series), so the variational `[y; S]` solve
differentiates them **exactly**: the cached recycle/flow maps are
influent-independent, so (unlike a flow setpoint) no term is dropped. This lets a
forward sensitivity screen include the **boundary-condition drivers** (the influent
load/flow) alongside the kinetic parameters — e.g. resolving that a permit-relevant
effluent *peak* is governed by the influent load and the oxygen system rather than
the kinetic rates, which a kinetic-parameter-only screen cannot show. Matches
finite differences in
`test_dynamic_sensitivity.py::test_solve_sensitivity_operating_influent_matches_fd`.
(Unit-level setpoints such as `kLa` are a follow-up — they are baked into
precomputed aeration vectors with no per-RHS `design` hook.)

**`shared_factor` — dense (Option B) vs simultaneous corrector (Option A).**
The per-step implicit solve has two implementations, selected by
`solve_sensitivity(..., shared_factor=...)`:

- `shared_factor=False` (**Option B**, dense): hand the augmented `[y; S]`
  system to a stock `Kvaerno5`, which factorises the full
  `n(1+p)×n(1+p)` implicit operator each step. Exact and cap-free; the right
  choice for **one** sensitivity parameter and for scalar-loss gradients.
- `shared_factor=True` (**Option A**, CVODES simultaneous corrector): the
  augmented Jacobian is block-lower-triangular "arrow" form — every diagonal
  block is the same `D = I − γ·dt·J`, each `S_j` column couples only to `y`.
  A custom `lineax` solver
  ([`integrate/_simultaneous_corrector.py`](aquakin/integrate/_simultaneous_corrector.py))
  injected into the `Kvaerno5` `VeryChord` root-finder **factorises `D` once
  per step (`O(n³)`) and forward-substitutes across the `S` columns**, instead
  of the dense `O((n(1+p))³)`. The Newton step is identical to the dense
  solve, so results are **bit-equivalent** to Option B (verified to ~5e-14) —
  only the linear-algebra cost differs.

`shared_factor` **defaults to `None`**, which auto-selects Option A for more
than one sensitivity parameter and Option B for a single one (Option A has no
advantage at `p=1`).

**Measured (stiff `wats_sewer_khalil_paper_balanced`, biofilm, jitted, `p=5`):**
Option A beats Option B by **3.8× at ndof=100 and 6.9× at ndof=180** — the win
grows with system size, as the `(1+p)³`→`1` factorisation saving predicts.
Versus the *capped `jacfwd`* workaround the comparison depends on the
integration span: the uncapped augmented solve's adaptive sensitivity control
actually takes **more** steps than a capped primal-only solve over a short
window (the sensitivity transient is what it must resolve), but its step count
**plateaus** (~4300) while the capped step count grows linearly with the span.
So `jacfwd` is faster for short solves and Option A overtakes it for long ones
(measured crossover ≈ 8–10 days: `0.78× → 0.92× → 1.16×` at 2/5/10 d), with a
large Option-A win expected at the multi-week maturation spans the cap was
introduced for. Net guidance: **for a multi-parameter sensitivity of a stiff
model, `shared_factor=True` (the default) is the best forward-sensitivity
option; whether it also beats capped `jacfwd` depends on the span.**

The known cost of the non-invasive design (a custom *solver*, not a custom
diffrax RK stage): the solver only sees the augmented operator, so materialising
`D` and the off-diagonal coupling blocks `L_j` costs `n` probes of the augmented
`M.mv` — each a JVP of the augmented RHS, i.e. second-order in `f` — the `L_j`
blocks that an ideal CVODES with direct access to `f` would not form. The plant
avoids it with the staggered corrector on the lean loop (above); the reactors
(Kvaerno5, diffrax) still pay it — a Kvaerno5 lean loop would be needed to move
them. `calibrate` does not yet expose a
`jacobian="forward_sensitivity"` hook, so the `dtmax` cap is still required for
the reverse-mode `calibrate` gradient until that lands.

**Cap-free *reverse* mode — hand-written discrete adjoint.** The forward
sensitivity above scales with the parameter count, so for a scalar-loss gradient
of many parameters (the calibration case) reverse mode is still wanted — and
that is the mode the `dtmax` cap exists for. `aquakin.implicit_euler_adjoint_solve`
([`integrate/discrete_adjoint.py`](aquakin/integrate/discrete_adjoint.py))
removes the cap there too, by **not differentiating through the solve at all**:
the forward pass is an ordinary robust adaptive diffrax `ImplicitEuler` solve,
and the reverse pass is the **discrete adjoint written out by hand** as a
per-step backward scan over the saved trajectory — each step a single
*transposed* solve through the same well-conditioned `I − dt·J` (a contraction,
so the cotangent stays bounded and nothing overflows). This is the classical
implicit-RK discrete adjoint (Sandu 2006; FATODE, Zhang & Sandu 2014); it is the
*exact* gradient of the discrete solve and is **verified two ways**: against the
closed-form gradient of first-order decay, and against the (correct but capped)
`RecursiveCheckpointAdjoint` gradient of the same implicit-Euler solve
(`rel ≈ 5e-8`, uncapped — see `tests/integration/test_discrete_adjoint.py`).
**Why earlier attempts failed and this works:** the overflow is in reverse-mode
AD forming cotangents of the large stored stage vector-field values `f_i ∼ ‖J‖·y`
(confirmed by reading the diffrax/optimistix source — the per-step Newton solve
is *already* IFT-differentiated by optimistix; the overflow is in the explicit
stage-combination arithmetic on the tape). Writing the per-step adjoint as the
analytic transposed solve never forms those large cotangents. Empirically
checked dead-ends, for the record: a stiffness-aware `dt·‖J‖` step controller
(finite but no faster than the cap), and k-space stage storage (shifts the
threshold out but does not remove the overflow). **Trajectory loss** is
supported: passing `t_eval` returns the states at those times and the backward
scan injects each observation's cotangent at its step; to keep that exact
without differentiating through dense interpolation, the forward is forced to
land steps exactly on `t_eval` (`diffrax.ClipStepSizeController(step_ts=t_eval)`),
so every observation is a step boundary (verified vs a closed-form
multi-observation gradient and vs the capped reference using the same
forced-step forward, `rel ≈ 6e-8`). **Wired into `calibrate`** via `calibrate(..., gradient="stable_adjoint")`. Both
gradient backends compute a discrete adjoint and both use JAX autodiff for the
**model** derivatives (`∂f/∂y` via `jacfwd`, `∂f/∂θ` via `vjp`); they differ only
in how the *integrator's* adjoint is formed — `gradient="jax_adjoint"` (default)
lets JAX/diffrax differentiate the whole solve (`RecursiveCheckpointAdjoint`,
needs the cap for stiff), while `gradient="stable_adjoint"` replaces only the
integrator's adjoint with the explicit per-step transposed solve (cap-free). The
stable backend forces a reverse-mode residual Jacobian under
`optimizer=OptimizerConfig(method="gauss_newton")` (it is a reverse-only `custom_vjp`), and
`stable_adjoint_max_steps` bounds the saved-trajectory buffer the backward scan
walks (set it to a tight upper bound on the step count). Verified end-to-end: a
synthetic Khalil calibration reaches the **same optimum** as the capped-Kvaerno5
`gradient="jax_adjoint"` path — see `test_calibrate_stable_adjoint_matches_jax_adjoint`.
**Also wired into `plant.solve`** via `plant.solve(..., gradient="stable_adjoint")`,
which routes the assembled flat plant RHS through `esdirk_adjoint_solve` so a
reverse-mode gradient flows through the whole monolithic plant solve — across the
ASM↔ADM interface and the recycle loops — with no `dtmax` cap, in the regime where
differentiating *through* the stiff plant solve (`jax_adjoint` /
`RecursiveCheckpointAdjoint`) is non-finite. **`gradient` defaults to `"auto"`**:
a plain forward solve (concrete `params`/`y0`) takes the fast cached `jax_adjoint`
path and a solve under reverse-mode differentiation (the args are JAX tracers)
takes `stable_adjoint`, so a stiff plant gradient is finite by default with no
knob to set; `event=`/`adjoint=`/`dtmax=` pin `jax_adjoint` (so `run_to_steady_state`
is unaffected), and a `jax.jit`-wrapped forward solve looks traced and so routes
to `stable_adjoint` (correct but uncached — pass `gradient="jax_adjoint"` to force
the cached path). It is **exact through a transient
solve**: `plant.solve` passes `time_dependent=True`, so the explicit time
dependence of a time-varying influent is carried in the state
(`esdirk_adjoint_solve(time_dependent=True)`, the classical autonomization that
appends `dτ/dt=1` and reads the time from the state) and the discrete adjoint
captures `∂f/∂t` exactly with no change to the per-step recurrence. Without it the
default autonomous backward evaluates the field at a fixed time and the gradient
of any time-coupled parameter is wrong (zeroed in the worst case). It rejects
`adjoint=`/`dtmax=` (it manages its own integrator and adjoint), and `max_steps`
bounds the saved-trajectory buffer the backward scan walks (the warm-started BSM2
plant takes ~205 forward steps under a constant influent, so a small cap keeps the
reverse pass cheap).
**Validated**: a *water-line* gradient — tank-1 nitrate with respect to the ADM1
acetate-uptake rate `k_m_ac`, flowing back through the digester, the interface and
the reject recycle — is finite and matches central finite differences to
`rel ≈ 4e-5` under a constant influent (the direct digester-biogas gradient to
`rel ≈ 4e-6`), and to `rel ≈ 2e-3` under a *diurnal time-varying* influent (the
`time_dependent` path), where the default reverse adjoint of the stiff plant fails
outright (`tests/integration/test_plant_stable_adjoint.py`). The autonomization is
verified exact against finite differences on a forced ODE, and the autonomous
default is shown to give the wrong gradient for a time-coupled parameter, in
`tests/integration/test_discrete_adjoint.py`.

**Two solvers, low- and high-order.** `implicit_euler_adjoint_solve` (first
order) is the simple, robust baseline. `esdirk_adjoint_solve` is the high-order
version: a general s-stage ESDIRK forward (default **`Kvaerno5`, the same method
the reactors use**) whose discrete adjoint reconstructs the stage values in the
backward pass and applies the transposed-stage recurrence `(I − dt·γ·Jᵢᵀ)⁻¹` per
stage — the FATODE/Sandu construction (verified to reduce to the implicit-Euler
case for s=1). **The stage values are saved by the forward, not recomputed.**
The forward runs `SaveAt(steps=True, dense=True)`; for a Runge–Kutta solver the
dense-output info carries the per-step stage increments `kⱼ` (the **dt-scaled**
stage derivatives `dt·f(Yⱼ)`), so the backward reconstructs each stage exactly by
the Butcher linear combination `Yᵢ = yₙ + Σⱼ A[i,j]·kⱼ` (`A` the full
lower-triangular tableau, dt already folded into `k`) — **no per-step Newton
recompute**, which was the dominant backward cost. (Earlier this re-solved every
stage by a fixed 12-iteration Newton scan, ~72 Jacobian builds + dense solves +
RHS evals per step; the saved-stage path removes all of it.) The saving is threaded
through the shared `_discrete_adjoint_solve` driver as `save_stages=` (ESDIRK sets
it; the s=1 implicit-Euler adjoint reads the post-step state directly and leaves it
off). Do not go back to recomputing the stages in the backward: it was the
dominant backward cost (measurements in `docs/plant_performance.md`, "Saved-stage
discrete-adjoint backward"). Validated: the stage
reconstruction is exact (the discrete-adjoint suite — analytic decay, FD, trajectory
and time-dependent gradients — and the plant BSM1/BSM2 cross-interface, colored,
and flow-setpoint `∂M/∂θ` gradients all match FD / the capped `jax_adjoint` path).
**`calibrate(gradient="stable_adjoint")` uses this Kvaerno5 ESDIRK adjoint**, so
its forward matches the reactor exactly and its gradients agree with the capped
`jax_adjoint` path to the optimiser tolerance (analytic decay `rel ≈ 1e-6`; stiff
model finite-uncapped, matching capped Kvaerno5 to `rel ≈ 2.5e-5`, the residual
being the capped-vs-uncapped *forward* difference, FD-confirmed). **Cost note:** the
backward scan's cost scales with `stable_adjoint_max_steps` (the padded trajectory
length), and with `dense=True` the saved dense-output buffer is ~`n_stages`× the
trajectory, so keep `max_steps` tight; Kvaerno5's high order keeps the step count
low. The autonomous reaction RHS is assumed (the ESDIRK stage times `c` do not
enter). **Low-memory option.** When that ~`n_stages`× dense buffer is the binding
memory constraint (a long, large-state solve), `esdirk_adjoint_solve(low_memory=True)`
drops it: the forward stores only the step states (`dense=False`) and the backward
**recomputes** each step's stages by a fixed Newton scan (`newton_iters`, default
12) before the same transposed-stage sweep — trading the buffer for ~a second
per-step stage solve. The recompute is a contraction through the same
well-conditioned `I − dt·γ·J`, so the reconstructed stages — and the gradient —
match the saved-stage path (machine precision on linear decay; `rel < 1e-5` on the
stiff Khalil model, the residual being the forward root-finder tolerance vs the
machine-precision recompute). It is **guarded to the singly-diagonal ESDIRK shape**
it assumes (explicit first stage, constant implicit γ — Kvaerno3/Kvaerno5); any
other tableau falls back to the saved-stage path with a `RuntimeWarning`. Exposed
to the plant and calibration via `DifferentiationConfig(adjoint_low_memory=True)`,
which threads down to `esdirk_adjoint_solve(low_memory=True)`; the plant folds the
flag into its stable-adjoint compile-cache key so a low-memory compile never
collides with a saved-stage one.
