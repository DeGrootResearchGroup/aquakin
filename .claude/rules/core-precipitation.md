---
paths:
  - "aquakin/core/precipitation.py"
  - "aquakin/core/precipitation_equilibrium.py"
---

# Rules — mineral precipitation

Split out of `.claude/rules/core.md`. Reuses the pH solver's constants and activities (`.claude/rules/core-speciation.md`).

## Mineral precipitation / state-derived saturation

`aquakin` can model **kinetic mineral precipitation and dissolution** —
struvite, calcite, the calcium phosphates — driven by the aqueous
supersaturation. Like pH, the saturation state is not an independent state: it
is computed from the instantaneous composition and exposed to rate expressions
as ordinary condition fields. The framework is the generalized
precipitation approach of Kazadi Mbamba et al. (2015).

### The rate law (`core/precipitation.py`)

Each mineral precipitates (or dissolves) at

```
R = k_cryst · X_cryst · sign(σ) · |σ|^n
```

driven by the relative supersaturation `σ = (IAP/Ksp)^(1/ν) − 1`
(`SI = log10(IAP/Ksp)` is the saturation index), where
`IAP = Π aᵢ^countᵢ` is the ion-activity product over the mineral's
constituent ions, `Ksp` its solubility product, `ν = Σ countᵢ` the number of
ions, and `aᵢ` the free-ion **activities** at the system pH. A small non-zero
`X_cryst` seed lets precipitation self-nucleate; `sign(σ)` makes the same law
describe dissolution when the phase is undersaturated (`σ < 0`). The kinetic
`power` form requires **`order >= 1`** (validated at load): its rate gradient
`order·|σ|^(order-1)` is infinite at `SI = 0` (saturation) for `order < 1`, so a
sensitivity through the equilibrium would be non-finite; the `bounded` (tanh)
form carries no such restriction.

The aqueous chemistry — the temperature-corrected dissociation constants, the
free-ion fractions of the carbonate/phosphate/ammonia/sulfide acid-base
systems, and the Davies / Debye-Hückel activity coefficients — is **shared with
the charge-balance pH solver** (`core/ph_solver.py`): the module reuses
`equilibrium_constants`, `_log10_gamma`, `debye_huckel_A`. Free Ca/Mg are taken
as the full declared total (ion-pairing not yet subtracted — a documented
simplification of the source aqueous model). AD-clean (no data-dependent control
flow), so it composes inside a Diffrax RHS and `jax.grad` flows through it.

### `precipitation:` block (`core/precipitation.py`, schema in `schema/model_spec.py`)

```yaml
precipitation:
  pH_field: pH               # condition giving pH (a fixed condition, or produced by speciation:)
  temperature_field: T
  temperature_units: kelvin  # or "celsius"
  activity_model: davies     # "none" | "davies" | "debye_huckel"
  ionic_strength_offset: 0.1 # background electrolyte (mol/L); OR ionic_strength_field: I_aq (see below)
  minerals:
    - name: struvite
      pKsp: 13.26            # at the reference temperature (25 C)
      order: 3               # supersaturation exponent n
      dH_sp: 0.0             # enthalpy of dissolution (J/mol); van't Hoff Ksp(T), 0 = T-independent
      solid: X_struvite      # precipitate species -> the reaction is AUTO-DERIVED
      rate_constant: {value: 100.0, units: "1/d", bounds: [1.0, 1000.0], transform: positive_log}
      ions:                  # molar_mass converts the state's units -> mol/L (Ksp basis)
        - {species: S_Mg,  molar_mass: 1000, count: 1, charge: 2}                       # free cation
        - {species: S_NH,  molar_mass: 1000, count: 1, charge: 1, fraction: ammonia}    # NH4+ fraction at pH
        - {species: S_PO4, molar_mass: 1000, count: 1, charge: 3, fraction: phosphate}  # PO4^3- fraction at pH
    - name: calcite
      pKsp: 8.48
      order: 2
      solid: X_calcite
      rate_constant: {value: 10.0, units: "1/d"}
      ions:
        - {species: S_Ca, molar_mass: 1000, count: 1, charge: 2}
        - {species: S_IC, molar_mass: 1000, count: 1, charge: 2, fraction: carbonate}
```

`build_precipitation_derived_fn` (in `core/precipitation.py`, no Pydantic) turns
the validated declaration plus `species_index` into a derived-condition callable
that produces `SI_<name>` and `R_<name>` per mineral.

**Auto-derived reactions.** A mineral declaring `solid` (the precipitate
species) + `rate_constant` **auto-derives its precipitation reaction**: the
schema's `_synthesize_precipitation_reactions` emits `<name>_precipitation` with
rate `k * [solid] * {R_<name>}` and stoichiometry that consumes each ion's
`species` at `-count` and produces `solid` at `+1` (species-less `proton`/
`hydroxide` ions carry no mass term). This removes the old duplication of
re-writing the stoichiometry in a separate `reactions:` block (and the risk of it
drifting from the ion counts — vivianite Fe₃(PO₄)₂ correctly gets `-3`/`-2`). The
rate constant is namespaced `<name>_precipitation.k`. A model whose only
processes are precipitation may omit `reactions:` entirely; omitting `solid`/
`rate_constant` falls back to a hand-written reaction referencing `{R_<name>}`.

**Ksp(T).** `pKsp` is the value at the reference temperature (25 °C); `dH_sp`
(enthalpy of dissolution, J/mol) van't Hoff-corrects it,
`Ksp(T) = Ksp(Tref)·exp(dH_sp/R·(1/Tref − 1/T))` — the same form and reference as
the dissociation constants. `dH_sp = 0` (default) leaves Ksp temperature-
independent (backward compatible).

**Shared ionic strength.** When precipitation composes with a `speciation:`
block, set `ionic_strength_field: I_aq` on **both** blocks: `solve_ph` optionally
returns the self-consistent solution ionic strength (strong ions + weak-acid
speciation + water), speciation exposes it as the `I_aq` condition field, and
precipitation reads it for its Davies/Debye-Hückel coefficients — so the pH and
the saturation indices use the **same** ionic strength. Without it, precipitation
falls back to `ionic_strength_offset` + its own mineral ions (which misses the
bulk electrolyte), the right choice for a standalone fixed-pH model. The mineral
ions are de-duplicated on `(species, fraction)` across every mineral of the
block (`ionic_strength_ions`), so an ion shared by several minerals counts once.
Both modes build `I` the same way: kinetic minerals from the current state
(`mineral_ionic_strength`), equilibrium minerals from the *equilibrium*
dissolved ions inside the Newton residual (self-consistent; a pre-precipitation
`I` can flip a near-threshold phase), and both read `ionic_strength_field` when
set. `speciation:` exports the real `I` even with `activity_model: none` (it is
evaluated at the root but does not feed back into the ideal pH). When the pH
comes from `speciation:`, `CompiledModel.precipitation_equilibrium()` runs the
speciation step first (`_project_after_speciation`).

An ion's `fraction` selects how its free
activity is obtained: an acid/base system key — `carbonate`, `phosphate`,
`ammonia`, `sulfide` (the species total times its de/protonated fraction at pH)
— a pH/water special (`proton`, H⁺ activity `10^-pH`; or `hydroxide`, OH⁻
activity `Kw/[H+]`, both with `species` omitted), or omitted for a fully-free
cation (the species total taken as the free ion). The `hydroxide` fraction is
what lets metal hydroxides (Fe(OH)₃, Al(OH)₃) and hydroxide-bearing minerals
(hydroxylapatite) be declared. `VALID_PRECIP_FRACTIONS` in
`core/precipitation.py` is the single source of truth (with `_PH_SPECIALS` the
`species`-optional subset), imported by the Pydantic schema so an unknown
`fraction` (or an undeclared `species`) is a load-time error.

**Composition with speciation.** `precipitation:` is wired in a `_compile_precipitation`
stage in `core/model.py` *after* `_compile_speciation`, so when both blocks
are present the precipitation reads the **charge-balance pH** the speciation
block produces (the two derived functions compose via `_compose_derived`:
speciation runs, its pH is broadcast into the conditions, then precipitation
runs and both results merge). The shipped
[`precipitation_struvite_calcite.yaml`](aquakin/models/precipitation_struvite_calcite.yaml)
is the worked example (digester supernatant, struvite + calcite, fixed
operating pH); the test suite also exercises the speciation→precipitation
composition.

**Units.** The worked model uses **mol/m³ (= mmol/L)** for the ions and
solids so the precipitation stoichiometry is exact (one mole of mineral consumes
one mole of each constituent ion), with the per-ion `molar_mass: 1000`
converting mol/m³ → the mol/L the IAP/Ksp use. `clip_negative_states: true`
protects the supersaturation term from a transiently-negative ion state.

**Chemical-P removal model + AD limitation at extreme supersaturation.**
[`precipitation_metal_phosphate.yaml`](aquakin/models/precipitation_metal_phosphate.yaml)
is the second worked model: ferric / aluminium dosing precipitates
orthophosphate as the very insoluble phosphates FePO₄ / AlPO₄ (after the
plant-wide P/S/Fe extension of Flores-Alsina et al. 2016), while the same dosed
metal competes to form the hydroxides Fe(OH)₃ / Al(OH)₃ (the `hydroxide`
fraction). The metal hydroxide buffers the free metal, so it sets a
**pH-dependent floor** on the achievable phosphate — chemical-P removal worsens
at higher pH (more OH⁻) and needs a metal dose in excess of stoichiometric. Its
forward behaviour is exact (P removal, the pH trend, machine-precision Fe/Al/P
conservation), but the metal phosphates are so insoluble that a far-from-
equilibrium dose sits at `SI ~ 14`, where the SI-driven rate Jacobian is `~1e13`:
the L-stable `Kvaerno5` damps this so the **forward solve is fine**, but **no
sensitivity method survives the initial transient** (reverse adjoint, even with a
`dtmax` cap; and the cap-free `forward_sensitivity`/`DirectAdjoint` — all return
non-finite). This is the extreme end of the documented stiff-AD spectrum and is
intrinsic to the chemistry (lowering `order` 2→1 cuts the Jacobian ~7 decades and
still fails). So with the **default power-law** kinetics this model is a
forward-simulation demonstration; `precipitation_struvite_calcite` (modest
`SI ~ 1–3`) remains the differentiable / calibratable power-law example. The
`hydroxide` *engine* path is itself AD-clean at moderate supersaturation
(verified on a mild `M(OH)₂` toy in the test suite — `jax.grad` through the solve
is finite).

### Differentiable ultra-insoluble precipitation — algebraic equilibrium + bounded driver (#295)

The `~1e13` power-law Jacobian is **not** the documented `dtmax`/`stable_adjoint`
overflow (where the per-step operator `I − γ·dt·J` stays well-conditioned): here
that operator is genuinely **near-singular** (cond `~1e13` even at tiny `dt`), so
*every* sensitivity path fails — including the hand-written discrete adjoint
(`stable_adjoint`), which NaNs on the multi-mineral model. The lesson (verified
numerically): you must **reduce the primal stiffness**, not swap the adjoint; the
true sensitivity is tame (finite differences on the finite forward solve give an
`O(0.01)` gradient), so the pathology is purely the through-the-transient AD.
Grounded in what geochemistry codes do, two opt-in fixes restore
differentiability while the default power law is **unchanged**:

**(A) Algebraic equilibrium — `mode: equilibrium`**
([`core/precipitation_equilibrium.py`](aquakin/core/precipitation_equilibrium.py)).
Solve the precipitation *equilibrium* algebraically instead of integrating the
stiff kinetics — the geochemistry-standard equilibrium-phase problem: find the
phase amounts and dissolved ions that satisfy mass balance **and** the mineral
complementarity (every precipitated mineral on its solubility `IAP = Ksp`, every
absent one undersaturated, coupled across the shared ions). It is the MINEQL /
PHREEQC structure — unknowns are log-free-ion + phase amounts, the complementarity
written with a smoothed Fischer–Burmeister function — solved by a **fixed-iteration
safeguarded Newton scan** (the `solve_ph` pattern): ε-continuation on the
complementarity smoothing, residual-adaptive Levenberg–Marquardt damping, and a
bounded log-free-ion step make it globally convergent even when a component is
driven to near-complete consumption (free ion `~1e-18` mol/L). The converged
solution is **differentiated via the implicit function theorem** (the forward
scan runs under `stop_gradient`; one Newton step on the converged residual
attaches the exact `−(∂G/∂w)⁻¹∂G/∂θ` sensitivity in a single linear solve — *not*
a backprop through the iterations, which is huge and unnecessary), and the
algebraic Jacobian it inverts is well-conditioned, so the `1e13` stiffness is
gone. A mineral declares `mode: equilibrium` + a `solid:` species; the engine
exposes `Xeq_<name>` (the equilibrium phase amount) and
**`CompiledModel.precipitation_equilibrium(C, conditions)`** returns the
equilibrium-projected state (each equilibrium solid set to `Xeq`, dissolved ions
rebalanced — mass-conserving). Verified on the metal-phosphate: equilibrium P
removal reproduces the kinetic `t→∞` limit, mass closes exactly, the pH trend is
right, and `jax.grad` (w.r.t. dose, pH) is finite. **This is "solve, don't
integrate":** it covers the equilibrium-outcome sensitivity / calibration use
(#295's impact list) but is not an in-ODE reaction (embedding the per-RHS Newton
solve inside the time integration is impractically slow — a documented future
optimization needing solver warm-starting). The solver van't Hoff-corrects each
`Ksp` with temperature (the per-mineral `dH_sp`, same form / reference temperature
as the kinetic engine), so the equilibrium tracks `T` consistently with the
kinetic path. Kinetic minerals in the same block are untouched
(`build_precipitation_derived_fn` skips `mode: equilibrium` minerals; the
equilibrium engine handles them and composes after the speciation pH like the
kinetic one).

**(B) Bounded-driver kinetics — `supersaturation_form: bounded`**
([`core/precipitation.py`](aquakin/core/precipitation.py)). For a differentiable
*dynamic* solve, replace the power-law factor `sign(σ)·|σ|^order` with the
thermodynamically-grounded **bounded driver** `R = tanh(SI/(2ν)·ln10) =
(Ω^{1/ν}−1)/(Ω^{1/ν}+1)` (bounded in `(−1, 1)`, `0` at `SI = 0`, `±1` far from
saturation). The rate Jacobian is then `~k` (non-stiff), so a reverse gradient
through the *time integration* of the ultra-insoluble model is finite, and the
steady state (`R = 0` ⇔ `SI = 0`) is the same equilibrium the projection gives —
verified to agree. The driver is a per-mineral `supersaturation_form: bounded`
flag (default `power`); the reaction expression `k·X·{R}` is unchanged, so it is a
drop-in. The trade-off is a slower precipitation *rate* far from saturation (the
*endpoint* is unchanged); raise `k` to reach equilibrium faster.
[`precipitation_metal_phosphate_equilibrium.yaml`](aquakin/models/precipitation_metal_phosphate_equilibrium.yaml)
(A) and
[`precipitation_metal_phosphate_bounded.yaml`](aquakin/models/precipitation_metal_phosphate_bounded.yaml)
(B) are the worked examples; `tests/integration/test_precipitation_equilibrium.py`
covers both plus the solver complementarity and the schema validation.

---
