# Investigation log — plant forward sensitivity and the lean loop's stage Newton

> **Lives in `.claude/notes/`, outside the auto-loaded `.claude/rules/` tree, so it
> never auto-loads.** It holds the dated measurements behind the
> `Plant.solve_sensitivity` bullet of `.claude/rules/integration.md` (issue #654) and
> the stage-Newton investigation of `aquakin/integrate/forward_solve.py`. The rules
> file carries the current, load-bearing status; numbers here carry the
> configuration they were measured under — check it against the source before
> quoting one as current. Harnesses: `scripts/profile_plant_sensitivity.py`
> (timings and step counts), `scripts/probe_newton_drift.py` (the drift probe), both
> on the window of `scripts/_bsm2_window.py`.

## Staggered vs simultaneous corrector on a BSM2 window (2026-10-31, #654)

The diffrax augmented `[y; S]` solve (now `corrector="simultaneous"`) had four
costs: it recompiled on every call; its block-arrow root finder probes the augmented
operator on all `n` state tangents, each a second-order JVP of `f`, so the per-step
cost grows with `k`; it has no contraction-rate growth limiter; and many of its
rejected steps are Newton failures (code 7/8), not error-test failures. The default
became the **staggered corrector** on the lean Kvaerno3 loop
(`staggered_sensitivity_solve`, the `forward_fast` loop).

*Configuration of every number in this section:* `build_bsm2` defaults (BSM2-configured ASM1 +
ADM1, 188 states, colored Jacobian 66 colors), started from the constant-influent
steady state (150 d from `bsm2_warm_start`, `rtol=1e-5`, `atol=1e-3`), driven by the
**synthesised** dry-weather influent (`load_bsm2_influent("dry")` — 15-min samples
with ±20% flow noise, far rougher than the IWA 609-day file, so step counts are much
higher than on the JRN-056 case of the issue), `rtol=1e-4`, `atol=1e-3`, the `k`
sensitivity parameters the first `k` `asm1.*` names, daily `t_eval`, default
`factormax` (2 staggered, 10 diffrax), 4-core container, 2026-10-31.

1-day window, each run alone on the machine; "cached" is a second call (before this
change every call cost the first-call time, since the plant recompiled). **Each time
is one sample, and wall times on this container are bimodal (a ~2× slow mode, also
within one process — see "Timing on this container" below)**, so a single sample
may be in either mode; the step counts are deterministic. Interleaved best-of-5 on
2026-11-01 put the staggered cached call at 4.5 s (`k=0`) and 17.0 s (`k=16`) — the
44 s below was a slow-mode sample:

| corrector | k | first call | cached call | attempted | Newton-nonconverged | error-test rejected |
|---|---|---|---|---|---|---|
| simultaneous | 0 | 236 s | 26 s | 2094 | 632 (30%) | 383 |
| simultaneous | 1 | 408 s | 115 s | 2099 | 741 (35%) | 267 |
| simultaneous | 16 | 630 s | 313 s | 2802 | 855 (31%) | 658 |
| staggered | 0 | 158 s | 4.5 s | 1545 | 325 (21%) | 71 |
| staggered | 1 | 128 s | 8.2 s | 1563 | 333 (21%) | 68 |
| staggered | 16 | 155 s | 44 s | 2176 | 327 (15%) | 238 |

5-day window (step counts only; these runs shared the machine, so their times are
void): pre-change code `k=0` 10620 attempted / 3181 nonconverged / 1946 error-test;
staggered `k=0` 7606 / 1582 / 316, `k=1` 7610 / 1591 / 304, `k=16` 9556 / 1665 / 772.
The two correctors agree on the 1-day window to `3.5e-4` of each sensitivity
column's largest entry and `~1e-3` in the state (two valid adaptive solves at
`rtol=1e-4`).

What this shows, and what it does not:
- **Compile once**: a cached call drops the ~150–500 s compile entirely.
- **Per attempt** (staggered, interleaved best-of-5 cached call / attempts): 2.9 ms
  at `k=0`, 7.8 ms at `k=16` — **~2.7× from `k=0` to `k=16`, not flat**. The
  `k`-proportional work is the ~10 Jacobian-vector products of `k` columns per
  accepted step (one per sensitivity-stage iteration; 3.2 iterations per stage on
  average, 78% at the minimum of 2, measured on 0.25 d at `k=16`); a 16-column
  application costs ~1.6 RHS evaluations (a top-level microbenchmark; it predicts
  ~2.3×). The simultaneous per-attempt figures (12 / 55 / 112 ms at `k = 0 / 1 / 16`)
  are single samples of unknown mode, so only "several times slower" is safe.
- **Newton failures fall but do not vanish** (30% → 21% at `k=0`). On this influent
  they are not step growth into divergence: the step-start Jacobian is wrong in one
  or two directions at the failing stages ("Stage-Newton drift" below), and forcing
  a **dense** Jacobian (`forward_root_finder` returning `None`) gave the same ~20% on
  0.5 d, so the colored pattern is not the cause. The growth limiter cannot remove
  that. This run does not separate how much of the 30% → 21% is the limiter and how
  much is `maxfac=2` versus diffrax's 10 and the Hairer convergence test versus
  optimistix's.
- Untried lever for the `k`-proportional cost: accept a sensitivity stage after one
  correction using a remembered contraction rate (CVODES keeps one for its
  sensitivity iteration; the state's rate is measured along a different error
  direction, so it risks leaving a slow sensitivity mode unconverged).
- The convergence-aware limiter is **only** on the lean loop. The diffrax paths
  cannot take it: diffrax hands its step controller only the error estimate (a
  failed root find sets it to `inf`), never the Newton contraction rate.

## Stage-Newton drift on the BSM2 window (2026-11-01)

Each implicit Kvaerno3 stage iterates with `M = LU(I − γhJ_n)`, `J_n` at the step
start. Near the root the plain iteration's error obeys `e_{m+1} = T e_m`,
`T = I − M⁻¹A(x)`, `A = I − γhJ(Y)`. `scripts/probe_newton_drift.py` wraps
`forward_solve._stage_solve` and forms `T` at every state stage's final iterate in the
solver's WRMS weighting. *Configuration:* the window above, 0.25 d, `k=0` (the
`forward_fast` loop), `rtol=1e-4`, `atol=1e-3`, Anderson off; 368 attempted, 74
non-converged, 19 error-test rejected. "Slow directions" = singular values of `T`
above 0.1; iterations are for a 1e-3 error reduction (largest over three random
right-hand sides); GMRES is preconditioned by `M`.

| stage outcome | stage solves | slow directions (median / max) | ρ(T) median | plain iterations (median / p90) | GMRES iterations (median / max) |
|---|---|---|---|---|---|
| converged in 2 | 749 | 0 / 2 | 0.0007 | 1 / 2 | 1 / 3 |
| converged slowly | 171 | 1 / 2 | 0.28 | 4 / 7 | 2 / 3 |
| hit the 12-iteration cap | 125 | 1 / 4 | 0.68 | 12 / 46 | 2 / 5 |
| diverged | 59 | 1 / 4 | 1.46 | >100 | 3 / 6 |

- The drift is **low-rank**: the failing stages are slow along one direction (p90 1–3).
- It sits in the **Takács settler** (the dominant singular directions' largest
  components: settler 669 of 750 in the capped class, then tank 5 and the digester);
  plausibly a layer-flux threshold switching on within the step — not confirmed.
- For a failed stage the final iterate is not the root, so `T` there is the
  linearization where the iteration stopped; the GMRES count is for one linearized
  Newton step.
- The probe is checked on a linear system whose factored matrix is wrong in exactly
  two directions: it reports 2 slow directions, 7 plain and 3 GMRES iterations.

## Anderson acceleration of the stage iteration (2026-11-01)

`forward_solve._stage_solve` takes an Anderson depth (`_ANDERSON_DEPTH`, Type II, Walker &
Ni 2011, row by row for the sensitivity block, least squares in the WRMS weighting
with a `1e-10·trace` Tikhonov term), the factored `M` as the preconditioner. **The
contraction / divergence tests read the plain step `f = −M⁻¹R(x)` at each iterate
(the preconditioned residual), not the applied update**: the first version tested
the applied update, which is non-monotone under acceleration, and stopped a
0.7/0.85-contraction two-direction synthetic after 3 iterations as "diverged"; on
the residual it converges in 5 (plain iteration: the 12 cap). With depth 0 the two
coincide, and the solve is bit-identical to the pre-Anderson loop. **Default 2 since
2026-11-01** (the project owner's decision after the comparison below).

Interleaved A/B (`profile_plant_sensitivity.py --days 1 --k 0 16 --calls 6
--variants plain anderson:2`, one plant per variant, cached calls alternated; best
of 5), the 1-day window above:

| depth | k | best cached call | attempted | Newton-nonconverged | error-test rejected |
|---|---|---|---|---|---|
| 0 | 0 | 4.5 s | 1545 | 325 | 71 |
| 2 | 0 | 3.3 s (−27%) | 1199 | 163 | 196 |
| 0 | 16 | 17.0 s | 2176 | 327 | 238 |
| 2 | 16 | 16.1 s (−5%) | 2135 | 107 | 572 |

- **Newton failures halve (`k=0`) or fall by two-thirds (`k=16`)**, as the drift
  probe predicted. At depth 2 (probe, 0.25 d): the iteration-cap class 125 → 1, slow
  stages converge in a median 4 iterations (was 6); attempts 368 → 292,
  non-converged 74 → 40, error-test 19 → 44.
- **What remains is the diverged class** (59 → 78 stages, ρ(T) median 1.3): the
  plain iteration's first residual grows, so the divergence test stops before
  Anderson has any history (GMRES would need a median 3). Untried: give Anderson
  `depth` iterations before judging divergence.
- **`k=16` gains little** because the freed step growth turns into error-test
  rejections (238 → 572), and at `k=16` each of those has already paid the full
  sensitivity solve, while a Newton failure rejects before it.
- **Accuracy:** against the independent simultaneous solve, `S` (`k=16`) is
  `5.1e-4` of each column's peak at depth 2 (`3.5e-4` at depth 0); the `k=0` state
  `2.9e-4` (`5.8e-4` at depth 0).
- **Dead ends (measured, do not re-try):** depth 3 ≈ depth 2 (`k=0` 1179 / 161 /
  182, `k=16` 2118 / 107 / 566, single samples). Feeding the step-size limiter the
  plain first-iteration rate instead of the accelerated one: `k=0` 3.3 s,
  `k=16` 16.8 s, 1212 / 139 / 144 and 2129 / 99 / 519, `S` `6.9e-4` — no gain;
  deleted.

## Newton-Krylov (GMRES) updates of the stage iteration (2026-11-01, deleted)

**Closed direction — measured, then deleted** in favour of Anderson; the design
below is enough to rebuild it (it lived in `forward_solve._gmres` on the #654
branch). Revisit only for a case where stages whose plain iteration diverges
outright dominate the step count — the one class Anderson cannot reach.


`_stage_solve` (was `_chord`) can instead solve each correction as the exact Newton
step `A(x) dx = −R(x)` by GMRES (`_gmres`): left-preconditioned by the factored `M`
(so the first Krylov vector is the plain step `f`), inner products weighted like the
error norm, modified Gram-Schmidt, the rows of a sensitivity block in lockstep, early
exit at a `1e-2` relative residual or 6 iterations, the least squares by regularized
normal equations. `_KRYLOV_SWITCH_RATE`: `None` off (default); negative = every
update ("always"); `r` = plain steps until the contraction rate first exceeds `r`
("adaptive"). Divergence is judged on Newton-Krylov updates only, so in adaptive
mode a plain step that does not contract switches instead of failing. Exclusive with
Anderson.

Synthetic (`test_staggered_sensitivity.py`, `M` wrong in two directions): at
contractions 0.7/0.85 the plain iteration hits the 12 cap, Anderson takes 5,
Krylov-always 2, Krylov-adaptive 3–4; at 1.5/2.5 (the plain iteration diverges)
plain and Anderson fail, both Krylov forms converge (2 / 3).

BSM2 1-day window (configuration above). `k=0`: one 4-way interleaved run
(`--variants plain anderson:2 krylov:always krylov:0.1`). `k=16`: each Krylov form
paired with plain in its own process — four whole-plant `k=16` programs in one
process exhausted memory (LLVM "Cannot allocate memory" with ~4.5 GB nominally
free), so the harness now drops each `k`'s programs before the next and the `k=16`
Anderson row is the earlier `plain anderson:2` pair. Best of 5 cached calls; "vs" is
the plain call in the same process:

| variant | `k=0` best | `k=0` attempted / nonconv / err | `k=16` best (vs plain) | `k=16` attempted / nonconv / err | `S` vs simultaneous |
|---|---|---|---|---|---|
| plain | 5.1 s | 1545 / 325 / 71 | 18.0 s | 2176 / 327 / 238 | 3.5e-4 |
| anderson:2 | **3.4 s** | 1199 / 163 / 196 | **16.1 s** (17.0) | 2135 / 107 / 572 | 5.1e-4 |
| krylov:always | 5.5 s | 1087 / 64 / 294 | 25.6 s (18.2) | 2119 / 26 / 682 | 7.0e-4 |
| krylov:0.1 | 4.3 s | 1134 / 113 / 237 | 19.7 s (18.0) | 2145 / 77 / 603 | 7.9e-4 |

`k=0` state vs the simultaneous solve: plain 5.8e-4, anderson:2 2.9e-4,
krylov:always 3.1e-4, krylov:0.1 1.0e-3 — all within the solver tolerance band.

- **Newton-Krylov removes the most Newton failures** (−80% / −92% at `k=0` / `k=16`
  for "always"), including the diverged class Anderson cannot reach.
- **It is not the fastest.** Each Newton-Krylov correction pays a linearization at
  the iterate plus GMRES products — `k`-wide on the sensitivity block — so "always"
  is +8% (`k=0`) and +41% (`k=16`) against plain; "adaptive" spends it only on slow
  stages, −16% / +9%. Anderson, at no extra derivative evaluations, is fastest at
  both `k`.
- **Every acceleration turns Newton failures into error-test rejections**
  (`k=16`: 238 → 572 / 603 / 682): once the stage solve stops failing, the step
  controller grows steps until the error test rejects them, and at `k=16` each such
  rejection has paid the full sensitivity solve. The remaining lever is step-size
  control, not the stage solve.

## Where the error-test rejections come from (2026-11-01)

`scripts/probe_step_rejections.py` records every attempt (`t`, `h`, error norm,
Newton rate, convergence, and the state's own error norm, so a rejection splits into
state- and sensitivity-driven). *Configuration:* the window above, 0.25 d,
`rtol=1e-4`, `atol=1e-3`, Anderson depth 2 (the default), controller defaults
(`maxfac=2`, `safety=0.9`, `pi_beta=0.08`).

| | `k=0` | `k=16` |
|---|---|---|
| attempted / nonconverged / error-rejected | 292 / 40 / 44 | 572 / 26 / 153 |
| rejections driven by the state / by `S` | 44 / 0 | 24 / 129 |
| rejected steps straddling an influent sample time (random placement) | 20% (15%) | 4% (6%) |
| rejections right after a grown accepted step | 38 of 44 | 99 of 153 |
| …of which that accepted step itself followed a failure | 54% | 66% |
| growth into the rejection: median (at the 2× cap) | 1.25 (15%) | 1.74 (44%) |
| error jump `en/(en_prev·(h/h_prev)³)` into a rejection: median (p75) | 2.3 (3.5) | 6.6 (73) |
| the same jump between two accepted steps: median (p25–p75) | 0.6 (0.3–1.4) | 0.7 (0.4–1.0) |

- **Not influent breakpoints.** Rejected steps straddle a 15-minute sample time no
  more often than random placement predicts, so clipping steps to the sample times
  (diffrax `jump_ts`) would not help. Rejected steps' Newton rates are ~0: the stage
  solve is not involved.
- **`k=0`: controller overshoot on a noisy estimate.** The `O(h³)` model predicts the
  next error only within a factor ~4 (the accepted→accepted IQR), and over half the
  rejections close a reject → accept → grow → reject cycle.
- **`k=16`: the sensitivity error estimate jumps.** 129 of 153 rejections are driven
  by `S`; the accepted step before one has a small error (median 0.11), the
  controller grows it (44% at the cap), and the estimate then jumps far past the
  `h³` prediction (44% above 10×). Consistent with — not shown to be — the
  sensitivity right-hand side `J(y)S + f_θ` jumping where `f` has a kink in `y`
  (`max`/`min` clips, the Takács settler's thresholds — where the drift probe put the
  Newton failures): the state crosses a kink smoothly, its sensitivity does not.

### Variants (2026-11-01)

Settings, all off by default (`profile_plant_sensitivity.py --variants` names):
**`hold`** — no growth on the attempt accepted right after a failed one
(`StepControl.hold_after_failure`; Hairer's rule after a rejection); **`value`** —
the stage convergence test on the stage-value increment `γh·Δk` rather than the
slope increment `Δk` (`_STAGE_TEST_IN_VALUE_UNITS`, `_stage_weight`); **`noerrcon`**
— the sensitivity left out of the error test (`_SENSITIVITY_IN_ERROR_TEST = False`,
CVODES `errconS = false`, which is CVODES's own default; `S` still converges in
every stage solve). Errors are against a tight solve
(`profile_plant_sensitivity.py --reference-rtol 1e-8`: the default lean loop at
`rtol=1e-8`, `atol=1e-7`), which agrees with diffrax Kvaerno5 at the same
tolerances to `2.4e-5`: `err_y` per state component relative to its peak (floored
at `atol`), `err_S` per sensitivity column relative to its peak, the largest over
the save times. *Configuration:* the 1-day window above, `rtol=1e-4`, `atol=1e-3`,
Anderson depth 2, controller defaults otherwise. Best of the cached calls of an
interleaved process (`k=0`: one process, 5 calls; `k=16`: three processes of three
plants — four `k=16` plants exhaust memory — 4 calls each), with the `default` call
of the same process in brackets (the third process had none). Errors: max over
components / columns, median in brackets (`err_S` median over the 16 columns).

| variant | `k` | best cached call (default) | attempted / nonconv / err-rejected | `err_y` max (median) | `err_S` max (median) |
|---|---|---|---|---|---|
| default | 0 | 4.1 s | 1199 / 163 / 196 | 2.2e-3 (8.4e-6) | — |
| hold | 0 | 4.0 s (4.1) | 1136 / 117 / 139 | 1.4e-3 (6.8e-6) | — |
| value | 0 | 3.5 s (4.1) | 1160 / 125 / 183 | 1.4e-3 (1.7e-5) | — |
| hold + value | 0 | 3.3 s (4.1) | 1110 / 93 / 137 | 1.8e-3 (1.4e-5) | — |
| default | 16 | 18.9 s / 18.5 s | 2135 / 107 / 572 | 4.8e-4 (1.0e-5) | 3.4e-4 (6.0e-5) |
| hold | 16 | 17.6 s (18.9) | 1992 / 78 / 431 | 1.6e-3 (6.6e-6) | 2.1e-4 (1.4e-4) |
| value | 16 | 18.2 s (18.5) | 2023 / 80 / 552 | 2.6e-3 (1.3e-5) | 2.6e-4 (6.8e-5) |
| hold + value | 16 | 16.7 s (18.5) | 1891 / 58 / 406 | 3.1e-3 (1.1e-5) | 2.3e-4 (1.4e-4) |
| noerrcon | 16 | 8.9 s (18.9); 9.6 s | 1199 / 151 / 213 | 2.0e-3 (3.6e-6) | 8.5e-4 (5.0e-4) |
| hold + noerrcon | 16 | 9.1 s | 1141 / 118 / 138 | 1.4e-3 (5.9e-6) | 8.9e-4 (5.4e-4) |
| hold + value + noerrcon | 16 | 9.9 s | 1128 / 82 / 149 | 5.9e-3 (9.0e-6) | 2.4e-3 (1.5e-3) |

- **`err_y` max is one settler state** (`TakacsClarifier` local states 3 and 5 of 80,
  peak ~440) in every row, and moves several-fold with the step sequence alone
  (`hold`, which only shrinks steps, triples it at `k=16`); the median is the
  steadier accuracy figure.
- **`hold`**: fewer attempts at both `k` (−5% / −7%), −7% right-hand-side
  evaluations at `k=0`, accuracy unchanged.
- **`value`**: −19% right-hand-side evaluations and the fastest `k=0` call, little at
  `k=16` (the `k`-column products dominate); the median state error rises 1.3–2×
  (a ~2000× looser stage test at these steps, at `_KAPPA = 0.1`).
- **`noerrcon`**: at `k=16` the steps are the state's (1199 attempts — the `k=0`
  count), so the solve takes about half the time; the price is `S` accuracy, 2.5×
  worse at the max and 8× at the median column (`5e-4` of the column peak).
  **Combined with `value`** the sensitivity stage test is loosened too and nothing
  else bounds `S`: its median column error is 25× the default's — do not combine
  them without a tighter test on the sensitivity block.
- **The earlier "vs simultaneous" accuracy figures are disagreements**, not errors:
  both solves sit at `rtol=1e-4`, and the true global error of the lean loop at
  `k=0` is ~1.4–2.2e-3 of a component's peak (~20× `rtol`, accumulated over ~1000
  steps). No variant moves it outside that band.
- **Deterministic cost at `k=0`** (`probe_step_rejections.py --days 1 --k 0
  --settings …`, which totals the stage iterations; right-hand-side evaluations ≈
  attempts + stage iterations, plus one Jacobian and one LU per attempt):

  | variant | stage iterations (per stage solve) | RHS evaluations |
  |---|---|---|
  | default | 9496 (2.64) | 10695 |
  | hold | 8781 (2.58) | 9917 (−7%) |
  | value | 7458 (2.14) | 8618 (−19%) |
  | hold + value | 7131 (2.14) | 8241 (−23%) |

  The wall-time ranking agrees. The test needs two iterations at least (a rate
  estimate), so `value` leaves stages near that floor; Hairer's codes accept after
  one by reusing the previous step's rate — untried here.
- **The stage test has been in slope units since the lean loop was written**
  (`d = wrms(dk, y)` on `main` before #654): it compares `Δk` (state per day) with
  the state's tolerance, so at `h ~ 1e-3 d` it is ~2000× stricter than Hairer's
  `η‖γhΔk‖ ≤ κ`, and it would change with the time unit. `_KAPPA = 0.1` was tuned in
  those units on the 609-day run.

## Timing on this container (2026-11-01)

Cached-call wall times are **bimodal, ~2×**, across processes *and* within one: the
same depth-0 `k=0` solve took 4.5 s in one process and 10.0 s in three consecutive
calls of another; within one process `k=16` calls ran 17.0, 17.9, 17.2, 38.4, 17.2 s.
CPU steal was 0–2% (`vmstat`). So: compare variants **interleaved in one process**
(`profile_plant_sensitivity.py --variants`) and report the best call; step counts are
deterministic and need no repetition.
