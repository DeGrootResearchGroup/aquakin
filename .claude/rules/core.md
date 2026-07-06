---
paths:
  - "aquakin/core/**"
---

# Rules — `aquakin/core/`

AST rate evaluation, parser, vectorized kernel, conditions, parameter
namespacing, the charge-balance pH solver / speciation, and mineral
precipitation. Loaded automatically when editing files under `aquakin/core/`.

## AST Rate Expression Evaluation

Rate expressions in YAML (e.g. `"k1 * [O3] * [Br-]"`) are parsed into an
abstract syntax tree (AST) at model load time. The tree is walked once to
produce a JAX-compatible callable. This callable is used at runtime — the
tree itself is not walked repeatedly.

### Node Types

**Leaf nodes**
- `SpeciesNode(name)` — looks up species concentration by index from `C`
- `ParamNode(namespaced_name)` — looks up rate constant by index from `params`
- `ConditionNode(field_name)` — indexes into `condition_arrays[field][loc_idx]`
- `ConstantNode(value)` — literal numeric constant

**Binary operation nodes**
- `AddNode`, `SubtractNode`, `MultiplyNode`, `DivideNode`, `PowerNode`

**Domain-specific function nodes**
- `ArrheniusNode(A, Ea)` — temperature-dependent rate: `A * exp(-Ea / (R * T))`
- `pHSwitchNode(pKa)` — acid/base speciation fraction: `1 / (1 + 10^(pH - pKa))`
- `pHInhibitNode(pH_LL, pH_UL)` — ADM1 lower-pH Hill inhibition `pHLim^n/(S_H^n+pHLim^n)` (stable sigmoid form); 1 at high pH, 0 at low pH. Needs `pH`.
- `MonodNode(X, K)` — saturation Monod: `X / (K + X)`
- `MonodInhibitionNode(X, K)` — inhibition Monod: `K / (K + X)`
- `MonodRatioNode(A, B, K)` — ratio-saturation Monod: `(A/B) / (K + A/B)`
- `MonodInhibitionRatioNode(A, B, K)` — ratio-inhibition Monod: `K / (K + A/B)`
- `SafeDivideNode(num, denom)` — the `safe_div(num, denom)` function: division
  that returns 0 (with a finite gradient) where `denom == 0`, instead of
  `inf`/`NaN`. For a ratio whose denominator can legitimately reach exactly zero
  — a substrate-competition fraction `[A] / ([A] + [B])` where both deplete to 0
  — so the rate takes its physical limit 0 there without padding the denominator
  with a dimensionless epsilon. Used by ADM1's valerate/butyrate C4 competition
  (`safe_div([S_va], [S_va] + [S_bu])`), replacing the old `+ 1.0e-6` guard.
- `MaxNode(a, b)` — the `max(a, b)` function: elementwise maximum, AD-safe
  (`jnp.maximum`, active-branch subgradient at the kink). For a one-sided clip,
  e.g. ADM1's biogas outflow `k_P * max(0, P_gas - P_atm)` so an
  overpressure-driven flux only stops (never reverses) when the driving
  difference goes negative.

The four Monod nodes and `SafeDivideNode` all evaluate their `num/denom` through
`_safe_ratio` (`core/nodes.py`), a **double-where** guard returning 0 (with a
finite gradient) where the denominator is exactly zero — for the Monod nodes the
full-depletion point where the limiting quantity *and* its saturation constant
are both 0 (`K = X = 0`); for `safe_div` wherever the author's denominator hits
0. The physical limit is 0 (no substrate → no rate); a bare `num/denom` is
`0/0 = NaN` there and the naive single `where` still back-propagates a NaN
through the masked branch, so the denominator is guarded too. Identity for any
nonzero denominator — the only change is exactly at the singularity.

New domain-specific node types are added here as needed. An **operator** node
(everything but the four leaves) subclasses `_OperatorNode` and declares its
arithmetic **once**, as a single `op(operands)` staticmethod plus a `KIND`
string:

```python
@dataclass(frozen=True)
class MonodNode(_OperatorNode):
    X: ASTNode
    K: ASTNode
    KIND = "monod"              # kernel key
    FUNCTION_NAME = "monod"     # optional: the call spelling in a rate expression
    # EXTRA_CONDITIONS = ("T",) # optional: condition fields read beyond the AST children

    @staticmethod
    def op(o):                  # operands = AST children (field order) + EXTRA_CONDITIONS
        x, k = o
        return _safe_ratio(x, k + x)
```

`_OperatorNode.compile` is generic (evaluates the operands and applies `op`), so
there is no per-node `compile`. That one `op` is the single source of truth:
**everything else derives from the node class** —

- the **scalar** rate closure (`_OperatorNode.compile`) calls `op`;
- the **vectorized kernel** table (`vector_kernel._KERNELS`) is built by walking
  the node hierarchy and maps each `KIND` to the node's own `op` — so the two
  paths run the *identical* function (bit-identicality is structural, not two
  hand-aligned copies), and a node added *without* a `KIND` simply raises
  `UnsupportedNode` → scalar fallback;
- the **interner** dispatch (`vector_kernel._Interner.intern`) reads `KIND`,
  `children()` and `EXTRA_CONDITIONS` off the node type — no per-node branch;
- the **parser** built-in registry (`parser._FUNCTIONS`) is discovered from each
  node's `FUNCTION_NAME` (with argument names read off its dataclass fields).

So adding a domain-function node is a *single* edit — define the class — rather
than coordinated edits across `nodes.py`, `parser._FUNCTIONS`,
`vector_kernel._KERNELS` and the interner. `PowerNode` keeps one special case: a
*constant* exponent interns to the `powc` kernel (a kernel-internal variant that
holds the exponent static so its JVP stays finite at base 0); the `powc` kernel
is `PowerNode.op` itself, so even it cannot drift.

A **leaf** node (`ConstantNode`, `SpeciesNode`, `ParamNode`, `ConditionNode`)
carries a literal payload rather than a kernel, so it keeps its own `compile` and
is handled explicitly in the interner (mapping to a pool leaf-block).

### Vectorized rate kernel (`core/vector_kernel.py`)

Evaluating the rates as `jnp.stack([f(C, params, ...) for f in rate_callables])`
traces **one nested closure tree per reaction**, so the jaxpr holds
`O(reactions x ops-per-reaction)` scalar primitives — each leaf a `slice` +
`squeeze`. XLA fuses them so the *runtime* is fine, but its optimization passes
scale with that op count, so the **compile** is dominated by it — and amplified
in the reverse-mode adjoint, where the RHS jaxpr is differentiated ~80x per
step (the multi-minute discrete-adjoint compiles). Compile, not run, is the cost
of a stiff solve (~1.6 s compile vs ~0.02 s run for an ASM1 batch), so this
*compile-time* lever is what hurts the test suite, where each test compiles a
distinct configuration once.

`build_vectorized_rates` builds, from the per-reaction rate ASTs, a single
callable returning the `(n_reactions,)` rate vector by **interning every
distinct subexpression** (node type + operand positions) and evaluating **all
instances of each primitive in one batched elementwise op**, in topological
order — global common-subexpression elimination plus vectorization by node type.
The pool is built **append-only by concatenation** (1 jaxpr op each, vs ~5 for a
scatter incl. index fixup), and operands are read with a **raw `lax.gather`**
(skips the negative-index normalization `slice`/`squeeze`/`select_n` that
`P[idx]` inserts). The traced op count collapses to `~O(node-types x depth)`,
independent of the reaction count.

It is **bit-identical** to the scalar path: each interned instance is a lane of
a batched op doing the identical scalar arithmetic, and IEEE elementwise ops are
deterministic per lane; the interning dedups identical subexpressions (also
bit-identical, since the scalar path recomputes them to the same bits). So it is
*not* the issue's masked-`jnp.prod` assembly (which would change the product
reduction order); the order-preserving batched ops need no revalidation rebasing.
Built once in `CompiledModel.__post_init__` (`_rate_kernel`) and dispatched
from `CompiledModel.rates` after the clip / derived-condition / temperature
preprocessing (which is unchanged); a future AST node type with no batched
kernel raises `UnsupportedNode` and `rates` falls back to the scalar stack, so
the kernel is a safe, transparent overlay. **Measured:** the rate jaxpr is the
dominant term in a differentiated stiff-solve compile (60% asm1 → 85% adm1 →
94% wats_sewer_extended), and the kernel cuts that jaxpr ~2–12x (asm1 2.9x,
adm1 2.2x, asm2d 10x, wats 5.6x), giving an end-to-end differentiated-compile
speedup of ~1.15x (asm1) / 1.5x (adm1) / 3.4x (wats) — larger on the bigger
models, where the suite hurts most. Runtime is unchanged. Regression-guarded
(bit-identicality on randomized states, op-count reduction, AD parity, and the
unsupported-node fallback) in `tests/unit/test_vector_kernel.py`.

### Parser

Recursive descent parser. No external parser dependencies. Operator precedence
(lowest to highest): `+/-`, `*/`, `**`, unary, primary. Species names are
delimited by square brackets: `[O3]`. Rate constant names are namespaced
automatically: `k1` in reaction `O3_Br_direct` becomes `O3_Br_direct.k1`.

### CompileContext

Carries index maps used during AST compilation:

```python
@dataclass
class CompileContext:
    species_index: dict[str, int]
    param_index: dict[str, int]      # namespaced keys -> flat vector indices
    condition_fields: set[str]       # valid condition field names
```

---


## Parameter Namespacing

Rate constants are locally named in the YAML (`k1`, `k2`) but internally
namespaced as `reaction_name.k1`. This allows natural YAML authoring while
maintaining unambiguous global parameter indexing.

Users always interact with namespaced names in the API:

```python
free_params=["O3_Br_direct.k1", "O3_OBr_oxidation.k2"]
```

The flat `params` vector and its index map (`param_index`) are built once at
compile time and stored in `CompiledModel`.

---


## SpatialConditions

Conditions (pH, temperature, UV fluence rate, etc.) are spatially varying by
design. They are passed as full JAX arrays so that indexing occurs inside the
JAX trace — this is required for differentiating with respect to condition
fields.

```python
@dataclass
class SpatialConditions:
    fields: dict[str, jnp.ndarray]   # each array shape (n_locations,)
```

The rate function indexes into these arrays using `loc_idx`. When `loc_idx` is
a traced JAX value (e.g. inside `vmap`), JAX promotes the indexing to a gather
operation automatically.

`SpatialConditions.uniform(n_locations, **kwargs)` is a convenience constructor
for spatially homogeneous conditions.

**`OperatingConditions` — the 0-D alias.** A single stirred tank has no spatial
location, so the spatial array model reads as over-machinery for the most basic
setup. `OperatingConditions(pH=7.5, T=293.15)` (exported as
`aquakin.OperatingConditions`) is a `SpatialConditions` subclass specialised to
one location, built directly from scalar field values — it **is** a
`SpatialConditions`, so it works unchanged in every reactor; use the base
`SpatialConditions` for a spatially varying PFR/CFD case. The 0-D examples and
the README quickstart lead with it.

**`conditions.with_(**overrides)` — edit from defaults.** Returns a copy with the
named fields overridden (or added), scalars broadcast to the object's location
count, the rest carried over, the original untouched. The recommended
edit-from-defaults pattern is `model.default_conditions().with_(T=283.15)`
(start from the YAML-declared defaults, change only what differs). It always
returns the base `SpatialConditions` type.

---


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
`jacfwd` (forward, the per-step Jacobian-materialisation path) and `jax.grad`
(reverse, the calibration-gradient path) through `solve_ph` are **O(1) in the
iteration count** instead of differentiating through every Newton step. The root
is the same one the old fixed scan converged to, so **every steady state is
unchanged** (validated: BSM2 steady state and the ADM1 digester reproduce
bit-for-bit). The convergence criterion is `in_bracket AND |Newton step| ≤ tol`
— a *bisection* step can be spuriously small far from the root, so the step size
alone is not a safe criterion, but a tiny in-bracket Newton step `|f/f'|` (with
`|f'| ≥ 1`) means a tiny residual, i.e. the root. *Why this is the lever:* a
Step-0 profile of the BSM2 plant found the pH solve was **~35% of the Jacobian
materialisation** purely because `n_iter` was 40 while it converges in ~6; the
adaptive + IFT rewrite drops the pH share to **~8%** (per-tangent pH cost
O(40)→O(1)) and the whole 20-day BSM2 transient by **~20–25%** (default solver),
with bit-identical results. The opt-in **activity-corrected path** uses the same
adaptive + IFT scheme lifted to a **2-variable coupled root**: its conditional
constants couple `[H+]` and the ionic strength `I`, so it solves the joint fixed
point `(f1 = charge balance, f2 = I − I(h)) = 0` with an adaptive coupled
`while_loop` wrapped in `custom_root` over the pair `(h, I)`, and the sensitivity
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
*inside* the same bracketed scan by carrying `I` (it converges together with
`[H+]`), and the returned pH is the **measurable** `−log10(a_H) = −log10(γ_H·[H+])`
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


## Mineral precipitation / state-derived saturation

`aquakin` can model **kinetic mineral precipitation and dissolution** —
struvite, calcite, the calcium phosphates — driven by the aqueous
supersaturation. Like pH, the saturation state is not an independent state: it
is computed from the instantaneous composition and exposed to rate expressions
as ordinary condition fields. The framework is the generalised
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
bulk electrolyte), the right choice for a standalone fixed-pH model.

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

