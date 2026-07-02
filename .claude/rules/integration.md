---
paths:
  - "aquakin/integrate/**"
---

# Rules — `aquakin/integrate/`

ODE integration strategy: solver choice, differentiating stiff models
(`dtmax`, forward/reverse adjoints, the discrete adjoint), operator splitting,
located events, and compiled-solve caching. Loaded automatically when editing
files under `aquakin/integrate/`.

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
`integrate/_common.py`): wrap a freshly computed `jax.grad` in it
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
`H = JᵀJ` instead (see `calibrate(laplace_method="gauss_newton")`).

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
`dgsm(ad_mode="forward")`). A future alternative is a quasi-steady-state (QSS)
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

**`Plant.solve_sensitivity` — the stable forward mode on the plant (the one that
was missing).** The reactors carried the cap-free augmented `[y; S]` solve, but
the *plant* was never wired to it, so a forward sensitivity of a stiff dynamic
plant fell back to `jacfwd` through `plant.solve` — finite per step, but
**non-finite over a long horizon**. That long-horizon failure is **numerical,
not a genuine tangent divergence**: the true sensitivity stays bounded (verified
by watching `‖S‖` *oscillate* with the diurnal load over 17 days — 45→196 — not
grow exponentially), and the break is the reactor-era
`augmented_forward_sensitivity` building a **stock `Kvaerno5`** that lacks the
plant's solver robustness. It was the one implicit mode never brought into the
`build_implicit_solver` / `build_step_controller` consolidation (the forward
`jax_adjoint`, `forward_fast` and `stable_adjoint` paths all build from those
helpers; the forward-*sensitivity* solve built its own). `Plant.solve_sensitivity(params, wrt, *, t_span, t_eval=, y0=, factormax=, dtmax=)`
closes the gap: it integrates the same augmented `[y; S]` system but routes the
solver through those shared helpers, so the `[y; S]` solve inherits the plant's
**decoupled Newton**, **`factormax`** cap, **`Kvaerno3`** base solver and
**cached recycle/flow maps** (its `f_flat` reuses the once-per-solve map instead
of re-resolving the recycle each call), while keeping the block-arrow
`SimultaneousCorrector` for the per-stage linear algebra. That config is exactly
what the stock solver lacked: the 20-day BSM2 forward sensitivity that the stock
augmented solve **and** `jacfwd` both blow up on is **finite** under
`solve_sensitivity`; it matches the dense `Kvaerno5` augmented solve to 5 digits
at 10 days, matches `jacfwd` to ~3e-7 on BSM1, and runs faster than the stock
dense/corrector (the Kvaerno3 + decoupled Newton + cached map all pay off).
Returns `(ts, ys, S)` with `S` the state sensitivity `dy/dθ`, shape
`(n_t, ndof, k)`. The cached map is built from the concrete `params` and closed
over, so the augmented linearisation drops `∂M/∂θ` — **exact for kinetic
parameters** (`M` depends only on the flow setpoints); a flow-setpoint
sensitivity needs the per-call map (not yet wired). The block-arrow corrector is
**specific to the `[y; S]` arrow** (each sensitivity column couples only to `y`,
sharing the one diagonal block `D = I − γ·dt·J`) and does **not** transfer to the
single-state forward/adjoint solves (whose per-step lever is the colored
Jacobian) or the steady-state IFT (which already factors `∂F/∂y` once and reuses
it across parameters). `build_implicit_solver` gained a `linear_solver=` slot to
inject the corrector into the decoupled root finder. Validated in
`tests/integration/test_dynamic_sensitivity.py::test_solve_sensitivity_matches_jacfwd`.
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
`M.mv` (`n·(1+p)` f-JVPs per step) — the `L_j` blocks that an ideal CVODES with
direct access to `f` would not form. A zero-redundancy variant (a custom operator
carrying `f`, needing a `Kvaerno5` stage subclass) is the documented future
optimisation. `calibrate` does not yet expose a
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
`optimizer="gauss_newton"` (it is a reverse-only `custom_vjp`), and
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
off). **Measured (BSM2 `value+grad`, dense backward, 3-day warm-started span): the
backward dropped 10,647 → 617 ms (~17×) and the whole gradient 11,200 → 1,212 ms
(~9×), gradient FD-/jax_adjoint-validated unchanged.** Validated: the stage
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
guard, and BSM1 plant resets).


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
  (`integrate/_common.py`) keyed by `(model identity, solver settings, call
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
  cannot false-hit.) The event path (`run_to_steady_state`) is not cached
  (run-once). The `gradient="stable_adjoint"` path **is** cached for repeat
  *forward* solves (a parameter sweep), keyed the same way but tagged
  `"stable_adjoint"` so it never collides with the forward path, with `t_eval`
  baked into the closure (the discrete adjoint marks it non-differentiable, so it
  cannot be a traced runtime argument) and its values folded into the key. The
  cache is used **only when the inputs are concrete**: under a trace — a gradient
  through the solve, or an enclosing `jax.jit` — the adjoint's `custom_vjp` is
  traced directly into the outer computation rather than routed through an inner
  `jax.jit`, which does not compose with an outer reverse-mode pass. That direct
  path is the one a `gradient="stable_adjoint"` calibration gradient takes, so a
  jitted calibration loss amortizes the (large) plant compile across optimizer
  iterations through the *outer* jit. Jitting that loss is possible because
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

