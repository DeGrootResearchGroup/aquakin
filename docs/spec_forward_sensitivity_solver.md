# Spec: Adaptive forward-sensitivity solver (CVODES-style simultaneous corrector)

Status: proposed. This document is a self-contained implementation spec for a new
`aquakin` feature. It can be implemented in a fresh session with no other context.

---

## 1. Problem this solves

`aquakin` is AD-throughout. To calibrate or do sensitivity analysis we need
`d(output)/d(params)` of a reactor solve. For **stiff** networks (the WATS/Khalil
sewer-biofilm models, `A_V`-scaled biofilm reactions ~1000 d⁻¹, etc.) AD *through*
the ODE solve becomes non-finite above an integrator-step threshold:

- **Reverse mode** (`jax.grad`/`jacrev`, the discrete adjoint) returns non-finite
  values above a `dtmax` threshold.
- **Forward mode** (`jax.jvp`/`jacfwd` via `diffrax.DirectAdjoint`) also returns
  non-finite values above a (looser) threshold.

The current workaround is a **global `dtmax` cap** (see CLAUDE.md
"Differentiating stiff networks (dtmax)"). This is crude: it forces tiny steps
over the *entire* solve even though the tight step is only needed in small,
*local* stiff regions. Measured cost of the cap on the multispecies biofilm
90-day maturation: `dtmax=3e-3` ⇒ ~30,000 steps and ~33 s/solve, vs **~2,800
steps and ~3 s** uncapped — an ~11× penalty for **zero** primal-accuracy gain
(the effluent is bit-identical capped vs uncapped). The cap exists only to keep
the *gradient* finite.

### The fix (validated prototype)

Classical **forward sensitivity analysis**: integrate the sensitivity
`S = ∂y/∂θ` *alongside* the state `y`, and let the **adaptive step controller
bound the sensitivity error too**. Then the controller tightens the step only
where the sensitivity is stiff and runs free elsewhere — no global cap.

A prototype (augmented `[y; S]` system, stock `diffrax.Kvaerno5` +
`PIDController`, no `dtmax`) was validated against the capped `jacfwd`:

| case | capped `jacfwd` (5e-3) | augmented adaptive (no cap) | match |
|---|---|---|---|
| 30 d, 1 param | 35.1 s | **11.6 s (3.0×)** | exact |
| 90 d, 5 params (naive) | 62.4 s | 54.5 s (1.1×) | 1.1e-8 |

The gradients match to machine precision and the cap is gone. **But the naive
multi-param version is only ~even with `jacfwd`** — and understanding *why* is the
whole point of this spec (Section 3).

---

## 2. The math

ODE (per spatial location / compartment, already assembled by the reactor):

```
dy/dt = f(y, θ, t),        y ∈ R^n,  θ ∈ R^p (the free params)
```

Forward sensitivity `S = ∂y/∂θ ∈ R^{n×p}` obeys the **variational equation**:

```
dS/dt = J(y,θ,t) · S + f_θ(y,θ,t),     J = ∂f/∂y (n×n),  f_θ = ∂f/∂θ (n×p)
```

Column `k` of the sensitivity RHS is exactly a JVP of `f`:

```
(dS/dt)[:,k] = J·S[:,k] + f_θ[:,k] = jvp(f, (y, θ), (S[:,k], e_k))[1]
```

where `e_k` is the unit tangent for free-param `k`. So **no explicit Jacobian is
needed** — the whole augmented RHS is `f(y)` plus `p` JVPs of `f`. The primal of
the JVP gives `f(y)` for free (`jax.jvp` returns `(f, df)` together).

Augmented state `z = [y; vec(S)] ∈ R^{n(1+p)}`, augmented RHS:

```
F(z) = [ f(y) ;  J·S[:,0]+f_θ[:,0] ; ... ; J·S[:,p-1]+f_θ[:,p-1] ]
```

### Implicit step structure (the key to efficiency)

A stiff (ESDIRK, e.g. Kvaerno5) step solves, per stage, a Newton system on the
augmented state with operator `M = I − γ·dt·∂F/∂z`. Because `S` does not feed
back into `y`, and `S[:,k]` does not depend on `S[:,j≠k]`, the augmented Jacobian
`∂F/∂z` is **block lower-triangular with identical diagonal blocks**:

```
∂F/∂z = [ J        0     0   ... ]
        [ ∂(JS0+fθ0)/∂y   J     0   ... ]
        [ ∂(JS1+fθ1)/∂y   0     J   ... ]
        [ ...                          ]
```

So `M = I − γ·dt·∂F/∂z` is block lower-triangular with **every diagonal block
equal to `(I − γ·dt·J)`**. Therefore the Newton/linear solve can:

1. **Factorize `(I − γ·dt·J)` once** (an `n×n` factorization), then
2. solve the `y` block, then each `S[:,k]` block by **forward substitution**,
   reusing the *same* factorization (the off-diagonal coupling goes to the RHS).

This is CVODES' **simultaneous corrector**: one `n×n` factorization serves all
`p+1` blocks, instead of factorizing the full `n(1+p) × n(1+p)` system (whose
cost grows like `(1+p)³`).

### Why the naive prototype loses to `jacfwd` on CPU

- `jax.jacfwd(solve)` already shares one `n×n` implicit factorization across all
  `p` tangents (factor once, `p` cheap back-solves per step) — but its step
  controller only sees the **primal** error, so it needs the global `dtmax` cap.
- The naive augmented prototype gets adaptive sensitivity control (fewer steps)
  but **loses the factorization sharing**: either it `vmap`s `p` separate
  `2n`-state solves (and `vmap` on CPU just pads them all to the longest step
  count, no core parallelism), or it builds one dense `n(1+p)`-state system whose
  factorization cost explodes.

**The win requires BOTH at once:** adaptive control of the sensitivity error
*and* the shared `(I − γ·dt·J)` factorization. That is this feature.

---

## 3. What to build

A forward-sensitivity ODE solve that:

1. Integrates `z = [y; S]` with an adaptive stiff solver (Kvaerno5 or similar).
2. The step-acceptance **error norm covers both `y` and `S`** (so the controller
   bounds the sensitivity error → no `dtmax` cap needed).
3. The per-step implicit linear solve **exploits the block-triangular structure**
   to factorize `(I − γ·dt·J)` once and forward-substitute for the `S` blocks
   (the simultaneous corrector).

### Recommended implementation path

This lives in the diffrax/lineax/optimistix ecosystem. The crux is item (3): a
**custom `lineax.AbstractLinearOperator`** representing the block-lower-triangular
`M` with identical diagonal blocks, plus a **custom `lineax` solver** that does
one diagonal-block factorization and forward substitution. Then a stock
`diffrax.Kvaerno5` with its Newton root-finder pointed at that linear solver, and
a `PIDController` whose norm spans `z`.

Two sub-options, in increasing effort/payoff:

- **Option B (stepping stone, ~1 day):** integrate the augmented `[y; S]` system
  with **stock** diffrax (dense `n(1+p)` implicit solve) + a controller whose
  norm covers `S`. This already removes the cap and is adaptive — it *is* the
  validated prototype. Good for single/few params and for scalar-loss gradients;
  it does **not** get the multi-param speedup (no factorization sharing). Ship
  this first as `forward_sensitivity(..., shared_factor=False)` to lock in
  correctness + the API + tests.

- **Option A (the prize, ~several days):** add the custom block-triangular linear
  operator/solver so `(I − γ·dt·J)` is factorized once per step and reused across
  the `S` columns (`shared_factor=True`, the default for `p>1`). This is the
  CVODES simultaneous corrector and delivers the multi-param speedup.

Implementers: study how `diffrax.Kvaerno5` constructs its implicit stage solve
(its `root_finder` is an `optimistix` Newton; that Newton's `linear_solver` is a
`lineax` solver). The custom operator must expose `mv` (matrix-vector, cheap via
the JVPs already in the RHS) and a structured solve. Reuse of the factorization
across stages within a step is a further optional optimization.

### Error-control / tolerances over `S`

- Compute the embedded error estimate over the full `z`. Weight `y` by the user's
  `rtol/atol`. Weight each `S[:,k]` by a **sensitivity tolerance**; the CVODES
  default is `rtol_S = rtol`, `atol_S = atol / |θ_k|` (so `S` is controlled to the
  same *relative* accuracy as `y` after scaling by the param magnitude). Expose
  `sens_rtol`, `sens_atol` (or a `param_scale` vector) with these defaults.
- Make the `PIDController` `norm` cover `z` with these per-block weights. A clean
  way: scale `S` columns by `|θ_k|` inside the integrated state (so a plain RMS
  norm over the scaled `z` gives the right control), then unscale on output.

---

## 4. API

Prefer a thin, composable surface that `calibrate()` and `sensitivity()` can call.

```python
# Reactor method (each reactor: Batch / PFR / Biofilm / Particle):
sol, S = reactor.solve_sensitivity(
    C0, params, t_span, t_eval=None, *,
    sens_params,            # list[str] | (k,) int indices of free params in `params`
    conditions=None,
    sens_rtol=None, sens_atol=None,   # default to rtol/atol scaled by |param|
    shared_factor=True,     # CVODES simultaneous corrector (Option A); False -> Option B
)
# sol : the usual Solution (trajectory of y at t_eval)
# S   : ndarray, shape (n_t, n_obs_or_n_species, n_sens_params) -- dy/dparam
```

- `S` is `∂y/∂θ` at the saved times for the requested params. It must be a real
  JAX array so it composes (the *primal* `sol` carries no cap; the sensitivity is
  exact).
- Optionally also expose a free function
  `aquakin.forward_sensitivity(reactor, ...)` mirroring `aquakin.sensitivity(...)`.

### Integration with `calibrate()`

`calibrate(optimizer="gauss_newton")` currently forms the residual Jacobian via
`jax.jacfwd(residual)` (with `DirectAdjoint` + a `dtmax` cap). Add an option to
form it via `solve_sensitivity` instead:

```python
calibrate(..., jacobian="forward_sensitivity")   # vs "ad" (current jacfwd/jacrev)
```

When selected: assemble the residual Jacobian `∂r/∂θ` from `S` (chain through the
observation map — for observed species at observed times it is just a gather from
`S`). No `dtmax` cap is needed for the Jacobian, so the calibration can run the
*primal* uncapped (fast) and get exact sensitivities. The L-BFGS path (scalar
loss gradient) can use the same machinery (single "param-combination" sensitivity
= contract `S` with `∂loss/∂y`).

Make `"forward_sensitivity"` the recommended Jacobian for stiff reactors and note
in CLAUDE.md that it supersedes the `dtmax`-capped `jacfwd` for those.

---

## 5. Validation / acceptance criteria

1. **Analytic test** (unit, fast). First-order decay `A→B`, `dA/dt=-k A`,
   `A(t)=A0 e^{-kt}`. The sensitivity `∂A/∂k = -t A0 e^{-kt}` is known in closed
   form. `solve_sensitivity` must match it to `~1e-8`. Use
   `tests/fixtures/simple_network.yaml`.
2. **Stiff-network exactness** (integration). On
   `wats_sewer_khalil_paper_balanced_biofilm_multispecies` (the canonical stiff
   case), `solve_sensitivity` (uncapped) must match `jax.jacfwd` (at a tight
   `dtmax`) to `~1e-6`. The validated prototype already shows `1e-8` agreement —
   reuse that as the regression baseline.
3. **No cap.** Runs with `dtmax=None` and returns finite sensitivities where the
   capped `jacfwd` would NaN.
4. **Speed (Option A).** For `p≥3` on the stiff biofilm, the multi-param Jacobian
   must be meaningfully faster than the capped `jacfwd` (the prototype's naive
   multi-param was only 1.1×; Option A's factorization-sharing is what must move
   this to ≥2–3×). Record the number on the 90-day, 5-param biofilm case.
5. **`calibrate` parity.** A calibration using `jacobian="forward_sensitivity"`
   must reach the same optimum (within optimizer tolerance) as the capped-`jacfwd`
   calibration on a stiff test, faster and without a `dtmax` cap.
6. **AD-correctness test** in the affected suites (per the project's checklist:
   "Every integration test suite must include an explicit test that AD flows
   without NaNs").

---

## 6. Validated prototype (reference implementation to start from)

Single-param augmented forward-sensitivity, stock diffrax, no cap. This is Option
B's core and the correctness oracle. (`bio._make_rhs(cond, p)` returns the
reactor RHS `f(t, y, args=p)`; see `aquakin/integrate/biofilm.py`.)

```python
ndof = (n_layers + 1) * net.n_species
cond = conditions.fields
def f_flat(yf, pp):                       # f as a flat (ndof,) -> (ndof,) map
    return bio._make_rhs(cond, pp)(0.0, yf.reshape(n_layers+1, net.n_species), pp).reshape(-1)

e_p = jnp.zeros_like(p).at[param_idx].set(1.0)   # unit tangent for one free param
def aug_rhs(t, z, args):                   # z = [y_flat ; s_flat]
    yf, sf = z[:ndof], z[ndof:]
    dy, ds = jax.jvp(lambda y_, p_: f_flat(y_, p_), (yf, args), (sf, e_p))
    return jnp.concatenate([dy, ds])

z0 = jnp.concatenate([y0.reshape(-1), jnp.zeros(ndof)])
sol = diffrax.diffeqsolve(
    diffrax.ODETerm(aug_rhs), diffrax.Kvaerno5(), t0=0.0, t1=T, dt0=None, y0=z0,
    args=p, saveat=diffrax.SaveAt(t1=True),
    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),   # NO dtmax
    max_steps=2_000_000,
)
S_T = sol.ys[0, ndof:]                      # dy/dparam at t=T (exact, cap-free)
```

Measured on the multispecies biofilm: matches capped `jacfwd` to `1.07e-8`; 3×
faster than capped `jacfwd` for one param over 30 days. Multi-param via
`jax.vmap(sens_one)(param_indices)` works and is exact but only ~1.1× on CPU —
that is the gap Option A (factorization sharing) must close.

---

## 7. References

- Hindmarsh, A.C. et al. (2005). *SUNDIALS: Suite of Nonlinear and
  Differential/Algebraic Equation Solvers.* ACM TOMS 31(3), 363–396. (CVODES
  forward sensitivity; **simultaneous vs staggered corrector** — Section on
  sensitivity analysis.) A copy is in the JRN-055 references
  (`Hindmarsh_2005_SUNDIALS.pdf`).
- Maly, T. & Petzold, L.R. (1996). *Numerical methods and software for
  sensitivity analysis of DAE systems.* (Staggered/simultaneous corrector.)
- Kidger, P. (2021). *On Neural Differential Equations.* (diffrax design.)
- aquakin `CLAUDE.md`, section **"Differentiating stiff networks (dtmax)"** — the
  problem this feature removes; update it once shipped.

---

## 8. Notes / gotchas

- Keep the **primal uncapped** — only the gradient needed the cap, and this
  feature removes that need. The maturation/forward solve already runs fine at
  `dtmax=None` (~3 s for the biofilm).
- The augmented RHS is `1 + p` JVPs of `f`. `jax.jvp` returns the primal `f(y)`
  alongside the first tangent, so the `y`-block is free; compute the `S` columns
  with a `vmap` over the `p` unit param-tangents sharing the same `(y, θ)`
  linearization point.
- `pH`/speciation (`derived_condition_fn`): the multispecies biofilm uses fixed
  pH, but networks with a state-derived pH have an extra implicit dependence
  inside `f`. The JVP handles it automatically (it differentiates through
  `network.rates`, which calls the speciation solver) — no special-casing, but
  add a state-derived-pH network to the exactness test.
- The positivity limiter and density-cap throttle are smooth (`maximum`/`minimum`
  / `clip`) and differentiate fine; they are part of `f`, so the sensitivity sees
  them. No action needed, but include a capped/limited network in the tests.
- Block-triangular linear operator (Option A): the diagonal block `(I − γ·dt·J)`
  is the same one `diffrax` already forms for the un-augmented solve. The
  off-diagonal blocks act only on the RHS during forward substitution and never
  need to be formed densely — they are `∂(J·S_k + f_θ_k)/∂y` applied to the
  already-solved `y`-increment, i.e. another JVP. Avoid materializing any
  `n(1+p) × n(1+p)` matrix.
```
