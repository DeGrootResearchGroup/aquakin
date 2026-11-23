# Differentiation: measurements moved out of the rule

Moved from `.claude/rules/integration-differentiation.md` during the #824 condensing
(2026-11-23). The rule keeps the invariants; this note keeps the measurements and the
investigation findings behind them. Numbers carry the configuration they were measured
under (notably the reactors then defaulted to `Kvaerno5`; the default is now
`Kvaerno3`, `IntegratorConfig(order=3)`): check against the source before quoting one
as current. Plant forward-sensitivity measurements are in
`.claude/notes/plant-sensitivity-log.md`; the saved-stage backward timing is in
`docs/plant_performance.md` ("Saved-stage discrete-adjoint backward").

## Why reverse mode through a stiff solve goes non-finite

- `Kvaerno5` is L-stable: a solve can take steps far larger than the fastest reaction
  timescale and damp the unresolved fast modes, so the primal is fine at any step.
  Forward mode (`jax.jvp` / `jax.jacfwd` via `DirectAdjoint`) stays finite at any step,
  losing only accuracy when the fast modes are unresolved; reverse mode (`jax.grad`,
  the discrete adjoint through diffrax) returns non-finite values above a step-size
  threshold.
- **Not a near-singular per-step solve.** The bare reaction Jacobian of the
  Khalil / extended sewer models is stiff and ill-conditioned (condition number ~1e20
  at t=0, |eig| up to ~1e5 d⁻¹), but that cond(J) is dominated by structurally
  near-zero eigenvalues of depleted / dormant species. The operator the implicit step
  and its adjoint invert, `I − γ·dt·J` (γ ≈ 0.26 for Kvaerno5), regularizes those
  directions and stays well-conditioned across the failing step range (cond ~650 at
  dt = 1e-2, ≤ ~3e4 at dt = 1e-1).
- **It is an overflow in the reverse (backward) accumulation**, controlled by the
  per-step stiffness `γ·dt·‖J‖` (fast-mode timescales per step), not by operator
  conditioning; consistent with the threshold scaling below (the steeper-Jacobian
  half-order variants need a tighter cap). Forward-mode tangent propagation, in the
  same direction as the primal and through the same well-conditioned operator, stays
  finite. Reading the diffrax / optimistix source confirmed where: the per-step Newton
  solve is already IFT-differentiated by optimistix; the overflow is in reverse-mode
  AD forming cotangents of the large stored stage vector-field values `f_i ∼ ‖J‖·y` in
  the explicit stage-combination arithmetic on the tape.
- **Ruled out:** the adjoint flavour (`RecursiveCheckpointAdjoint` and `DirectAdjoint`
  reverse fail identically, so it is not the checkpointing); the positivity limiter;
  the zero-valued initial species; stiffness alone (a 2–3 species stiff toy
  differentiates finitely in reverse even at 1e7 d⁻¹ — the failure needs the full
  coupled, many-species system).
- **The threshold is model-dependent:** ~5e-3 d for the Khalil Monod biofilm, ~10×
  tighter (~5e-4 d) for the half-order variants whose √C kinetics steepen the
  Jacobian; the Khalil study used `dtmax = 1e-4` d (3e-5 d for the stiffer balanced
  base), inside both. With the cap the reverse gradient is finite and matches both
  forward mode and finite differences.
- **Dead ends checked empirically:** a stiffness-aware `dt·‖J‖` step controller
  (finite, but no faster than the cap); k-space stage storage (shifts the threshold
  out, does not remove the overflow).
- A possible future alternative to the cap: a quasi-steady-state (QSS) reduction of
  the near-instantaneous fast reactions, which would remove the stiff modes entirely.
  Not implemented.

## Reactor forward sensitivity (`solve_sensitivity`, `shared_factor`)

- The augmented `[y; S]` solve's `S` matches a tightly-capped `jacfwd` to ~1e-8;
  validated against the closed-form sensitivity of first-order decay and against
  capped `jacfwd` on the stiff Khalil biofilm.
- `shared_factor=True` (the simultaneous corrector, "Option A") against the dense
  augmented solve (`shared_factor=False`, "Option B"): bit-equivalent results
  (verified to ~5e-14). Measured on the stiff `wats_sewer_khalil_paper_balanced`
  biofilm, jitted, `p = 5`: Option A beats Option B by **3.8× at ndof = 100 and 6.9×
  at ndof = 180**; the win grows with system size, as the `(1+p)³ → 1` factorization
  saving predicts. Option A has no advantage at `p = 1`.
- Against the capped-`jacfwd` workaround the comparison depends on the span: the
  uncapped augmented solve's adaptive sensitivity control takes **more** steps than a
  capped primal-only solve over a short window (the sensitivity transient is what it
  must resolve), but its step count **plateaus** (~4300) while the capped step count
  grows linearly with the span. Measured crossover ≈ 8–10 days (Option A / capped
  `jacfwd` speed `0.78× → 0.92× → 1.16×` at 2 / 5 / 10 d); a large Option-A win is
  expected at multi-week maturation spans.

## Discrete adjoint (`implicit_euler_adjoint_solve`, `esdirk_adjoint_solve`)

- Implicit-Euler adjoint vs the (correct but capped) `RecursiveCheckpointAdjoint`
  gradient of the same implicit-Euler solve: `rel ≈ 5e-8`, uncapped. Trajectory loss
  (forced-step forward on `t_eval`) vs a closed-form multi-observation gradient and
  vs the capped reference using the same forced-step forward: `rel ≈ 6e-8`.
- A synthetic Khalil calibration with the stable adjoint reaches the same optimum as
  the capped-Kvaerno5 `through_solve` path. With the Kvaerno5 ESDIRK adjoint the
  gradients agree with the capped path to the optimizer tolerance: analytic decay
  `rel ≈ 1e-6`; the stiff model finite uncapped, matching capped Kvaerno5 to
  `rel ≈ 2.5e-5`, the residual being the capped-vs-uncapped *forward* difference
  (FD-confirmed).
- Plant (`plant.solve` under reverse-mode differentiation): a water-line gradient —
  tank-1 nitrate with respect to the ADM1 acetate-uptake rate `k_m_ac`, flowing back
  through the digester, the ASM↔ADM interface and the reject recycle — is finite and
  matches central finite differences to `rel ≈ 4e-5` under a constant influent (the
  direct digester-biogas gradient to `rel ≈ 4e-6`), and to `rel ≈ 2e-3` under a
  diurnal time-varying influent (the `time_dependent` path), where the default reverse
  adjoint through the stiff plant solve fails outright. The warm-started BSM2 plant
  takes ~205 forward steps under a constant influent, so a small step budget keeps the
  reverse pass cheap.
- The `time_dependent` autonomization is exact against finite differences on a forced
  ODE; the autonomous default gives the wrong gradient for a time-coupled parameter
  (zeroed in the worst case).
- Saved stages vs recompute: the backward used to re-solve every stage by a fixed
  12-iteration Newton scan, ~72 Jacobian builds + dense solves + RHS evaluations per
  step, the dominant backward cost; reading the saved dense-output stage increments
  removes all of it (timings in `docs/plant_performance.md`).
- `low_memory=True` (recompute the stages instead of saving them): matches the
  saved-stage gradient to machine precision on linear decay and to `rel < 1e-5` on the
  stiff Khalil model, the residual being the forward root-finder tolerance against the
  machine-precision recompute; costs about a second per-step stage solve.
