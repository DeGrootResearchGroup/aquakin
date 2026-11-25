---
paths:
  - "aquakin/core/ph_solver.py"
  - "aquakin/core/speciation.py"
---

# Rules — speciation and the charge-balance pH solver

Split out of `.claude/rules/core.md`.

## Speciation / state-derived pH

Some models couple rates to pH, but pH is not an independent state — it is
fixed by acid/base speciation through electroneutrality. `aquakin` can solve
pH from the instantaneous state and expose it to rate expressions as an
ordinary condition field.

### Charge-balance solver (`core/ph_solver.py`)

`solve_ph(...)` returns pH given the total molar concentrations of the
carbonate, acetate, ammonia, phosphate and sulfide systems, plus strong-anion
charge equivalents, a net fixed cation charge, and temperature. It runs a
**safeguarded Newton-bisection** on the electroneutrality residual in
`u = ln[H+]` space. The residual is strictly monotone with a unique, trivially
bracketable root, so each step takes a Newton step but falls back to bisection
(via the non-strict rtsafe product test) whenever Newton would leave the bracket
— making the iteration **globally convergent**. This matters: a bare Newton step
from the fixed pH-7 start **overshoots to `exp(u) = inf` (NaN), or silently to an
absurd pH that saturates the `pH_switch` rate terms**, when the strong-ion charge
exceeds the buffering (e.g. a weakly buffered transient, or a calibration that
pushes the `S_cat`/`S_an` states) — the bracketed scheme stays finite and correct
there.

**Adaptive iteration + implicit-function-theorem AD (the ideal hot path).** On
the default `activity_model="none"` path the iteration is an **adaptive
`jax.lax.while_loop`** that stops as soon as a Newton step near the root falls
below tolerance (a handful of steps in the buffered regime — measured 5–6 to
machine precision across pH 5.4–8.8) and is **capped at `n_iter`** for the
bisection worst case, wrapped in **`jax.lax.custom_root`** so the **pH
sensitivity is the analytic IFT tangent** (one scalar solve of `df/d[H+]` at the
root). The iteration count therefore never enters the AD graph: `jax.jvp` /
`jacfwd` (forward, the per-step Jacobian-materialization path) and `jax.grad`
(reverse, the calibration-gradient path) through `solve_ph` are **O(1) in the
iteration count** instead of differentiating through every Newton step. The root
is the same one the old fixed scan converged to, so **every steady state is
unchanged** (validated: BSM2 steady state and the ADM1 digester reproduce
bit-for-bit). A Newton step is taken only if it stays in the bracket **and** is at most half the step taken two iterations before (the `rtsafe` progress test); do not drop the second test — without it a step from the wrong side of a 1 M imbalance overshoots to the bracket edge and crawls back one unit of `u` per step, past the cap (#835). An element that has converged is frozen, since a broadcast loop runs until every element converges and a converged element's zero step would otherwise fail the progress test and bisect away from its root. The convergence criterion is `in_bracket AND |Newton step| ≤ tol`
— a *bisection* step can be spuriously small far from the root, so the step size
alone is not a safe criterion, but a tiny in-bracket Newton step `|f/f'|` (with
`|f'| ≥ 1`) means a tiny residual, i.e. the root. *Why this is the lever:* a
Step-0 profile of the BSM2 plant found the pH solve was **~35% of the Jacobian
materialization** purely because `n_iter` was 40 while it converges in ~6; the
adaptive + IFT rewrite drops the pH share to **~8%** (per-tangent pH cost
O(40)→O(1)) and the whole 20-day BSM2 transient by **~20–25%** (default solver),
with bit-identical results. The opt-in **activity-corrected path** solves a
**2-variable coupled root**: its conditional constants couple `[H+]` and the
ionic strength `I`, so it solves the joint fixed point
`(f1 = charge balance, f2 = I − I(h)) = 0` as an **outer fixed-point loop over
`I` around the ideal path's bracketed pH solve**, wrapped in `custom_root` over
the pair `(h, I)`. **Do not update `I` inside the bracketed loop**: a bracket is
built from the signs of one residual, and changing `Kc(I)` between steps lets it
exclude the coupled root, so the solve stalls at the bracket edge at any cap
(#835: ~12% of digester-like inputs, up to 10 pH units off). Each outer pass
re-solves pH at a fixed `I`, warm-started. The sensitivity
is the exact IFT tangent via a **2×2 linear solve** of the joint Jacobian at the
root (so it too is O(1) in the iteration count, forward and reverse). Equilibrium
constants are temperature-corrected via van't Hoff. The chemistry mirrors the
WATS reference pH solver. **The coupled path is also extreme-safe**: a transient
far-overshoot trial `[H+]` makes the water self-ionisation term `½(h + Kw/h)`
explode the ionic strength to ~1e20, where `g = 10^(…)` overflows to `inf` and
the `inf/inf` activity-coefficient ratios become `NaN` — which the bracketing
could then never recover from. The ionic strength fed to the activity models is
clamped to a physical ceiling (`_I_MAX = 10 M`, far above any digester/sewer
~0.01–0.2 M and the models' ~0.5 M validity), so every conditional constant stays
finite and the bracket pulls `[H+]` back to the root, where `I ≪ _I_MAX` makes the
clamp exactly identity (bit-identical converged pH; issue #382). The IFT 2×2
tangent solve floors `|det|` for the same finiteness at a degenerate
(all-zero-totals) input. Validated against an independent bisection root finder;
the weak-buffer / extreme-charge regime that broke the old bare-Newton scheme, the
activity path's extreme-input finiteness and degenerate-point gradient (#382), the
cap-independence of the adaptive loop, and forward-vs-reverse AD agreement (both
the IFT tangent) are regression-tested in `tests/unit/test_ph_solver.py`.

**Ionic-strength activity corrections (`activity_model`).** By default the solver
uses molar **concentrations** directly with thermodynamic equilibrium constants
(all activity coefficients γ = 1) — the published ADM1/BSM2 convention, and the
behaviour every validated BSM2/WATS result is pinned to. Commercial simulators
(SUMO, BioWin, GPS-X) instead apply an ionic-strength activity correction, which
at digester/sewer ionic strengths (I ≈ 0.05–0.2 M) shifts pH by ~0.1–0.3 units.
`solve_ph(..., activity_model=...)` adds this as an **opt-in** static option:
`"none"` (default, the γ = 1 path — a trace-time Python branch, so it is
**bit-identical** to the historic solver), `"davies"` (the Davies equation, valid
to I ≈ 0.5 M), or `"debye_huckel"` (extended Debye–Hückel / Güntelberg, to
I ≈ 0.1 M). With a non-`none` model the dissociation constants become *conditional*
(concentration-basis) constants `Kc = K·γ_acid/(γ_H·γ_base)` at the
**self-consistent** ionic strength — the coupled `I`↔`[H+]` fixed point is solved
by the outer loop over `I` described above, and the returned pH is the **measurable** `−log10(a_H) = −log10(γ_H·[H+])`
(which reduces to `−log10[H+]` at γ = 1). The Debye–Hückel `A(T)` comes from the
water dielectric (`debye_huckel_A`). NH₄⁺/NH₃ is charge-symmetric so its Ka is
unchanged; the strong-ion contribution to `I` (`½Σc z²`) is supplied by the
speciation layer, which holds each strong ion's charge. The whole path stays
`jit`/`vmap`/`grad`-clean (the IFT pH sensitivity flows through it). Decisive
correctness check: pure water in an inert salt stays at the neutral pH for any
`I` (`tests/unit/test_ph_solver.py`). Exposed two ways: the `speciation:` block's
`activity_model:` field (per-model YAML), and a load-time override
`load_model("adm1", activity_model="davies")` (and `load_model_from_file(...,
activity_model=...)`) to run a *shipped* model with activities without editing
its YAML.

### Derived conditions (the wiring)

`CompiledModel` carries an optional `derived_condition_fn(C, params,
condition_arrays, loc_idx) -> {field: scalar}` and a `derived_fields` list.
`CompiledModel.rates()` evaluates it once per RHS call and merges the result
into `condition_arrays` (broadcast across locations) **before** the rate
callables run — so existing `{pH}` / `pH_switch(pKa)` machinery sees the
derived value unchanged, and every reactor (batch/PFR/particle/CFD) gets it
for free. Derived fields are *produced*, so they are added to the AST's valid
condition set but are **not** in `conditions_required` (the user never
supplies them).

### `speciation:` block (`core/speciation.py`, schema in `schema/model_spec.py`)

```yaml
speciation:
  field: pH                 # produced condition field (default "pH")
  temperature_field: T      # condition giving temperature
  temperature_units: celsius # or "kelvin"
  z_cation_eq: 3.28e-3      # net fixed cation-charge OFFSET (eq/L); literal or {condition: name}
  n_iter: 40
  activity_model: none      # ionic-strength correction: none (default) | davies | debye_huckel
  totals:                   # species -> acid/base total (molar_mass converts mg/L -> mol/L)
    carbonate: {species: S_CO2, molar_mass: 12000}
    acetate:   {species: S_VFA, molar_mass: 64000}
    ammonia:   {species: S_NH,  molar_mass: 14000}
    phosphate: {species: S_PO4, molar_mass: 31000}
    sulfide:   {species: sumS,  molar_mass: 32000}
  strong_anions:            # fully dissociated: charge * conc/molar_mass eq/L
    - {species: S_SO4, molar_mass: 32000, charge: 2}
    - {species: S_NO,  molar_mass: 14000, charge: 1}
  strong_cations:           # same form; summed into the net cation charge
    - {species: S_cat, molar_mass: 1.0, charge: 1}
```

`build_ph_derived_fn` (in `core/speciation.py`, no Pydantic) turns the
validated declaration plus `species_index` into the derived-condition
callable. Negative concentrations are clamped to zero before conversion
(mirrors the reference). Valid `totals` keys: `carbonate`, `acetate`,
`propionate`, `butyrate`, `valerate`, `ammonia`, `phosphate`, `sulfide`
(`propionate`/`butyrate`/`valerate` are the additional monoprotic ADM1
volatile fatty acids, treated exactly like `acetate`). The net cation charge
passed to the solver is `z_cation_eq` (a literal/condition *offset*, default
0) **plus** the `strong_cations` sum — so a strong-cation *state* (e.g. ADM1's
`S_cat`, paired with `S_an` in `strong_anions`) drives pH dynamically, the same
way `strong_anions` already does for anions.

---
