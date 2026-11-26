# Plant solve: measurements moved out of the rule

Moved from `.claude/rules/plant-solve.md` during the #824 condensing. Accuracy
and validation measurements behind the recycle-resolution rules; the dynamic-solve
speedups went to `docs/plant_performance.md`.

## Exact affine recycle solve

- Validated as a fixed point on BSM1, the multi-model BSM2 and a
  temperature-carrying loop: residual ~1e-12.
- A recycle loop whose bare Gauss-Seidel would need thousands of passes (a
  clarifier in a high-capture stateless loop) is solved in one linear solve.

## Fixed-pass mop-up vs adaptive `recycle_tol`

- BSM2: the only iterating stream is the front mixer (influent + reject recycle),
  `rho ≈ 0.0066`, so the fixed 3 passes leave ~1e-6 residual.
- Synthetic tunable-`rho` map, error left by the fixed 3 passes: 13% at
  `rho = 0.5`, 73% at `rho = 0.9`, 97% at `rho = 0.99`. The adaptive solve reaches
  tolerance for `rho` up to 0.99, and its IFT gradient matches central finite
  differences to ~2e-10.
- Real BSM2 reject loop: the adaptive forward fixed point matches a 14-pass deep
  sweep to ~4e-15; its IFT tangent matches the deep-sweep gradient to ~6e-19.
- The adaptive default needs ~3 iterations from the affine seed on BSM (roughly
  neutral cost) and reproduces the published BSM2 / ADM1 / Takács validations
  unchanged.
- With the adaptive default, the `stable_adjoint` discrete adjoint composes with
  the recycle IFT tangent: the cached and probed `dM/dθ` gradients agree to
  ~1e-13 (float rounding, no longer bit-for-bit, since `M` is only the warm-start
  seed), and the cross-interface gradient matches finite differences to the FD
  floor.

## Cached map exactness

- Cached vs probed concentration map `M`: bit-identical RHS; the dynamic
  trajectory drifts ~1e-3 over multi-day runs from floating-point operation order
  (the cached `M` is formed once outside the integration loop), within the solver
  tolerance; validated steady states preserved.
- `stable_adjoint` with the primal / param RHS split: the gradient with respect
  to a kinetic parameter and a flow-setpoint parameter both match the per-call
  probe gradient bit-for-bit (measured before the adaptive `recycle_tol` default).
- Cached flow map `A`: RHS, Jacobian and steady state bit-identical to probing.
  On a sensitive time-varying run (dynamic BSM2) the cached and probed solves
  separate by ~9.5e-3 relative: the probe recomputes
  `A = one_pass(eye) − one_pass(0)` every step and the varying influent perturbs
  that cancellation's rounding by ~1e-16 per step, while the cache holds `A` at its
  constant value. The cached path is the cleaner of the two; the difference cannot
  amplify at a fixed point.

## Coloured vs dense Jacobian, remeasured (2026-11-26)

Behind `Plant._COLORED_BACKWARD_MIN_STATES` (100 → 50). One plant per process,
one process at a time, on a 4-core host at load average 1–2. "Repeat" is the
faster of two repeat calls of the compiled program; "first" includes the
compile and, for coloured, the pattern build and guard. Spans from the warm
start, `rtol=1e-5, atol=1e-3`; BSM1 on the dry-weather influent, A²O and BSM2 on
their constant / dry influents, the CSTR chains (ASM1, no recycle) on a constant
feed. The gradient is the jitted `value_and_grad` of `sum(y(T)²)` w.r.t. the full
parameter vector (the discrete-adjoint `method="stable"` route); the coloured
and dense gradients agree to 10 significant figures on every plant.

| plant | states | colours | grad 1 d, repeat: dense / col (ratio) | grad first call: dense / col |
|---|---|---|---|---|
| 1 ASM1 CSTR | 13 | 8 | 0.07 / 0.07 s (1.00) | 6.9 / 21.0 s |
| 2-CSTR chain | 26 | 10 | 0.15 / 0.14 s (1.07) | 9.6 / 24.7 s |
| 4-CSTR chain | 52 | 11 | 0.39 / 0.35 s (1.11) | 17.6 / 34.9 s |
| BSM1, ideal clarifier | 65 | 14 | 3.43 / 2.25 s (1.52) | 38.4 / 69.9 s |
| 8-CSTR chain | 104 | 13 | 0.95 / 0.78 s (1.22) | 34.9 / 53.7 s |
| A²O (ASM2d), ideal clarifier | 133 | 20 | 5.91 / 3.46 s (1.71) | 125 / 179 s |
| BSM1, Takács | 145 | 15 | 9.40 / 6.82 s (1.38) | 50.0 / 102 s |
| 12-CSTR chain | 156 | 13 | 2.06 / 1.52 s (1.36) | 50.9 / 75.6 s |
| BSM2 (0.25 d span) | 188 | 66 | 7.02 s / not measured | 345 s / not measured |

| plant | fwd 1 d, repeat: dense / col | PTC repeat: dense / col | PTC first call: dense / col | PTC iterations |
|---|---|---|---|---|
| BSM1, ideal clarifier | 1.76 / 1.46 s | 0.04 / 7.31 s | 7.6 / 11.6 s | 45 / 45 |
| A²O | 2.51 / 1.95 s | 0.11 / 35.5 s | 32.5 / 49.5 s | 38 / 38 |
| BSM1, Takács | 4.83 / 2.36 s | 0.18 / 9.67 s | 8.4 / 17.0 s | 111 / 111 |
| BSM2 (fwd 0.25 d) | 3.22 / 1.81 s | 0.72 / 86.8 s (loaded host) | 70.9 / 217 s (loaded host) | 45 / 45 |

- **The backward (the gate's only subject).** Per gradient the coloured backward
  is never slower than dense: even from 26 states, 1.1× at 52 and 1.4–1.7× on
  the recycle plants from 65 states up. What it costs is a one-time 15–60 s on the
  first gradient of a plant (pattern build, guard, a larger compile), repaid after
  roughly 15–25 gradients at BSM1 size, so within any calibration. Below ~50
  states the per-gradient gain is under 10% and the one-time cost dominates,
  hence a gate of 50, below BSM1 with its ideal clarifier, which the old gate of
  100 left dense although the coloured backward is 1.5× faster there.
- **BSM2's coloured gradient was not measured:** the process was killed at the
  host's memory limit twice (16 GB shared with two other agents' JAX runs). BSM2
  is above the gate before and after the change.
- **Forward and PTC are opt-in (`colored_jacobian=True`), not gated.** The
  coloured forward is 1.2–2× faster. Coloured PTC is not in the compiled-solve
  cache (each call recompiles), so it loses on every call, and it reaches the
  same steady state in the same iteration count.
- **Do not measure this on a loaded host.** The same runs at load average 7–10
  (other agents' JAX processes) inflated the timings by up to 20×, unevenly
  between dense and coloured, which reverses or exaggerates the ratios: ideal
  BSM1's dense forward read 31 s per simulated day against 1.4 s coloured, its
  dense gradient 72 s against 33 s, and Takács BSM1's backward 87 s against 85
  s, where the quiet host gives 1.8 / 1.5, 3.4 / 2.3 and 9.4 / 6.8 s.
