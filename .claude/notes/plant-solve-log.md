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
