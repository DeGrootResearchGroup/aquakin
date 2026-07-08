---
paths:
  - "aquakin/schema/**"
  - "**/*.yaml"
---

# Rules — YAML model schema (`aquakin/schema/`, model `*.yaml`)

The YAML model format and the Pydantic schema rules. Loaded automatically
when editing the schema package or any `*.yaml` model file. For the catalog
of what each shipped model is, see `docs/model_catalog.md`.

## YAML Model Schema

### Top-Level Structure

```yaml
model:
  name: ozone_bromate
  version: "1.0"
  description: "..."
  references:
    - "Acero & von Gunten (2001), Environ. Sci. Technol. 35(3), 590-599"

species:
  - name: O3
    description: "Ozone"
    units: mol/L
    default_concentration: 1.0e-4

conditions:
  - name: pH
    description: "Solution pH"
    default: 7.5
  - name: T
    description: "Temperature (K)"
    default: 293.15

reactions:
  - name: O3_Br_direct
    description: "Direct ozone oxidation of bromide"
    reference: "Acero & von Gunten (2001)"
    rate: "k1 * [O3] * [Br-]"
    parameters:
      k1:
        value: 160.0
        units: M-1 s-1
        bounds: [10.0, 1000.0]
    stoichiometry:
      O3: -1
      Br-: -1
      HOBr: +1
```

### Schema Rules

- `default_concentration` is a reference value for the species, not an
  experimental initial condition. Users override it at runtime.
- `conditions` block declares all fields the model requires. The loader
  validates that any `SpatialConditions` object passed at runtime provides
  all declared fields. Each condition may carry an optional `units:` string
  (default `""`), advisory metadata used only by `check_units` (e.g. `pH: "-"`).
- `units` on species (default `"mol/L"`), parameters (default `""`), and
  conditions (default `""`) are advisory unit strings. Species/parameter units
  feed result labelling and the opt-in `check_units` dimensional check; they are
  not otherwise used at runtime. `check_units` treats a blank/unparseable unit
  as unknown (skipped).
- Species units are **also** cross-checked against each `speciation:` /
  `precipitation:` `molar_mass` at load time (`_audit_speciation_molar_mass` in
  `schema/model_spec.py`). `molar_mass` converts the species state value to mol/L
  (`mol/L = state / molar_mass`), so it must be a pure unit-conversion factor (a
  power of ten) for an already-molar species and a molecular weight (never a
  clean power of ten, always > 1) for a mass species. A value on the wrong side
  of that split emits an advisory `aquakin.SpeciationUnitsWarning` (a warning,
  not an error — it catches a silent pH / saturation-index shift from a hand-edit
  that changes a species' `units` without updating `molar_mass`). It is
  calibrated to zero false positives on every shipped model; a blank/unparseable
  or non-mass/non-molar unit is skipped.
- `reference` on reactions is optional but strongly encouraged — it makes
  the YAML file a self-documenting scientific artifact.
- `bounds` on parameters are optional, used by `fit()` as box constraints.
- `transform` on parameters is optional (default `"none"`); valid values
  are `"none"`, `"positive_log"` (for `p > 0`), and `"logit"` (for
  `0 < p < 1`). Used by `calibrate()` to optimise in unconstrained space.
- `prior` on parameters is optional — a Gaussian prior in physical space
  declared either as `prior: {mean: m, std: s}` (a measured value with
  reported uncertainty) or `prior: {range: [lo, hi]}` (a literature range,
  converted to a Gaussian centred on the midpoint with `std = (hi-lo)/4`,
  i.e. the range ≈ ±2σ). `calibrate()` adds `0.5·((p-mean)/std)²` to its
  objective for any free parameter carrying a prior (default
  `use_priors=True`; override per-parameter via the `priors=` argument).
  Priors regularise otherwise non-identifiable parameter combinations toward
  literature values; for a proper Bayesian MAP/posterior pair them with
  `loss="nll"` and a measurement `sigma` so the data term is a true negative
  log-likelihood and the prior curvature enters the Laplace covariance.
  The shipped `asm1` and `adm1` models **declare literature-grounded priors**
  on every calibration parameter (the bounded, non-fraction targets), each a
  physical-space Gaussian **centred on the parameter's nominal** with relative
  `std` set to a literature coefficient of variation:
  - **ASM1** — from the activated-sludge parameter database (Hauduc 2011 PhD
    thesis, Table 3-1): the per-parameter reported `V%` (wastewater-specific by
    construction). Most are narrow (`muH` 6%, `bH` 2%); the substrate/nitrate
    half-saturations `KS`/`KNO` and autotroph decay `bA` the widest (50–80%).
  - **ADM1** — **Brun et al. 2002 uncertainty classes around the BSM2 municipal-
    sludge nominal** (kinetics / half-saturations / inhibition / decay = class 3,
    50%; yields = class 1, 5%). It deliberately does **not** use the reported
    ranges of the Mo et al. 2023 ADM1 review (Table 3): those aggregate across
    digester *types* — grass silage, food/agricultural/industrial waste, dry AD —
    so a span like `k_hyd_pr ∈ [0.0014, 10]` or `K_I_h2_c4` down to `5e-8` (grass
    silage, Koch 2010) measures *substrate* variation, not the parameter
    uncertainty of a mesophilic municipal-sludge digester. The within-context
    Brun CV is the right model, and Mo's own "hydrolysis varies considerably with
    the substrate" is the justification for it. (When a municipal-sludge-specific
    ADM1 range source becomes available, replace the Brun CV with it.)

  Priors are far tighter / more defensible than the parameters' mechanical
  ±decade `bounds` (which stay as the hard feasibility limits), and are what a
  global-sensitivity screen or Bayesian calibration should sample — the prior is
  the plausible-uncertainty *belief*, the bounds are the hard *fence*.
  `asm1_ammonia_limitation` inherits them through `extends: asm1`.
- `temperature` on parameters is optional — an Arrhenius-style temperature
  correction `temperature: {theta: t, ref_T: T0, condition: "T"}`. When present,
  the rate constant is multiplied by `theta**(T - ref_T)` during rate
  evaluation, where `T` is read from the named condition field (default `"T"`).
  The parameter `value` is the value **at** `ref_T`, so the correction is unity
  there — a model whose conditions sit at the reference temperature behaves
  exactly as if uncorrected (backward-compatible). `ref_T` is in the condition's
  units (Kelvin for the ASM/ADM models); a difference is used, so Kelvin and
  Celsius give the same `theta`. `theta` is the per-degree factor; from a
  parameter measured `p_hi` at `T_hi` and `p_lo` at `T_lo` it is
  `(p_hi/p_lo)**(1/(T_hi - T_lo))`. The correction is applied to **rate
  constants only** — it is confined to `CompiledModel.rates` (which multiplies
  the corrected param indices by their factor before evaluating the rate
  callables); `compute_stoich` always uses the raw parameters, so stoichiometric
  (yield / composition) coefficients are never temperature-scaled. Stored as
  `CompiledModel.temperature_corrections` (a list of
  `(param_idx, ln_theta, ref_T, condition_field)`); AD-clean. `asm1` ships with
  the six BSM temperature-dependent rate constants corrected (`muH`, `muA`,
  `bH`, `bA`, `ka`, `kh`, `ref_T = 293.15 K`, slopes from the standard BSM
  15 °C/10 °C pairs in `asm1_bsm2.c`), so it slows correctly in the cold
  (nitrification — the most temperature-sensitive — drops to ~36% at 10 °C) while
  staying identical to the old behaviour at the default 20 °C.
- Parameters can live at the model level (a single shared slot used by
  any reaction that references them by bare name) or inside a reaction's
  `parameters:` block (namespaced as `<reaction>.<name>`). Model-level
  parameters and reaction-local parameters with the same name are
  rejected as shadowing.
- Stoichiometric coefficients can be **numeric literals** or **string
  expressions** in model-level / reaction-local parameters. String
  entries may use only constants, parameters, and arithmetic — no
  species, conditions, named expressions, or domain functions. Evaluated
  once per `solve()` call so yield / N-content / fraction parameters can
  be calibrated alongside kinetic constants. See `asm1.yaml` for a worked
  example.
- Optional per-species **`composition:`** block declares the species' content of
  the conserved quantities in its own measure — `composition: {COD: 1.0}` for an
  organic (1 g COD per g COD), `{COD: -1.0}` for dissolved oxygen (an electron
  acceptor), `{COD: -2.86, N: 1.0}` for nitrate-N, `{COD: 2.0, S: 1.0}` for
  sulfide. Quantity names are free-form (`COD` / `N` / `P` / `S` / `Fe` / `charge`
  / …). It is **first-class conservation metadata**: a model carries its own
  table instead of one hand-maintained in a test, read back via
  `model.composition()` and dotted against the stoichiometry by
  `model.check_conservation()` / `model.check_nitrogen()` (the
  advisory/opt-in conservation analogue of `check_units`; it never runs at load
  and never raises on a violation — it returns the list). A model that declares
  no `composition:` falls back to the shipped role-based table
  (`aquakin.composition_table`, the ASM/ADM families) so the check API is uniform;
  a model with neither raises a clear error. Values are literal floats (a
  yield-dependent *derived* coefficient stays in the stoichiometry, not here).
  The WATS sewer family (`wats_sewer`, `wats_sewer_extended` and everything the
  `_make_khalil_*` generators splice from them) ships `composition:` per species,
  so the conservation suite checks each model against its own declared table
  (`tests/integration/test_mass_balance.py`); inherited through `extends:` (a
  derived species keeps the base's composition unless it overrides it).
- A stoichiometric coefficient written **`auto`** (or **`?`**) is left unknown and
  **solved from the declared conservation laws** at compile time, so a
  conservation-determined coefficient *cannot be written wrong* — the failure mode
  behind almost every stoichiometry bug here (a hand-typed electron-acceptor
  demand, an elemental-S reduction donor, a product split). The quantities to
  solve from are the reaction's **`conserved_for: [COD, N, …]`** (or a model-level
  `conserved_for:` default); for each, the stoichiometry-weighted species
  `composition` content must sum to zero, giving one small linear system
  (`core/stoich_resolve.py`, `numpy.linalg.lstsq`) solved before the stoichiometry
  is read. Example: `stoichiometry: {SS: "-1/Y_H", X_BH: 1, SO: auto}` with
  `conserved_for: [COD]` solves the O₂ demand from the COD balance. The compiled
  model then conserves by construction (`check_conservation` is a tautology on
  that reaction — the point). Clear errors for an `auto` with no `conserved_for`,
  an under-determined system (an auto species carrying no content in the conserved
  quantities), or inconsistent balances. Two cases (issue #291 Phases 2–3):
  - **numeric** — every other coefficient is a numeric literal, so the solve is
    purely numeric (`lstsq`, tolerating an over-determined-but-consistent system)
    and the resolved value is a constant baked into the static stoichiometry matrix.
  - **parameter-expression (yield-dependent)** — a neighbour is a string expression
    (e.g. `-1/Y_H`, as in the example above). Conservation is *linear* in the
    coefficients, so each `auto` coefficient is a numeric-weighted linear
    combination of the known coefficient expressions (`x = M⁻¹b`, `M` from the
    composition); the resolver emits that as a *derived* expression string, which
    the normal stoichiometry-expression machinery compiles into a
    parameter-dependent (`stoich_dynamic`) coefficient. So calibrating a yield
    flows through to the derived coefficient — the reaction conserves for **every**
    parameter value, and `jax.grad` flows through it. This path needs a **square**
    system (`#auto == #conserved_for`); an over-determined symbolic system's
    consistency would be parameter-dependent and is rejected.

  `auto` requires declared `composition:` (the resolver does not use the role-based
  fallback). Shipped models keep their published rounded literals; `auto` is
  opt-in.
- Optional `expressions:` block at the model level lets you give a name
  to an intermediate rate expression and reference it from a reaction's
  `rate:` or from another expression. References are inlined into the
  consuming AST at compile time. Cycles among expressions are rejected.
- Optional **`model.extends:`** declares a base model to **inherit** instead
  of copying — a variant that differs by one parameter and a rate is a few lines,
  and a fix to the base reaches every variant. The base is a shipped model
  *name* (or a path relative to the extending file if it contains `/` or ends
  `.yaml`); the derived mapping is **merged onto the (recursively resolved) base
  before Pydantic validation** (so the merged whole is schema-checked).
  ([`schema/inheritance.py`](aquakin/schema/inheritance.py), wired in
  [`schema/loader.py`](aquakin/schema/loader.py).) Merge semantics: the named-list
  blocks (`species`/`conditions`/`reactions`) match by `name` and **field-merge**
  a derived entry onto the base (override just a reaction's `rate`, keep its
  `stoichiometry`; a new name is appended); every other block (`parameters`,
  `expressions`, `speciation`, ...) deep-merges (nested maps key-wise, scalars/
  lists replace). A `remove:` block (`{reactions: [...], parameters: [...],
  species/conditions/expressions: [...]}`) then drops entries; removing a
  not-present name, a cyclic `extends`, an unknown base, or declaring `extends`
  in both `model:` and top-level all error clearly. New parameters append after
  the base's, so the flat parameter *vector order* differs from a hand-written
  full copy (the name→value mapping and the compiled stoichiometry/rates are
  identical). A YAML with no `extends` is byte-for-byte unaffected. Shipped users:
  `asm1_ammonia_limitation` (= `asm1` + the `KNH_H` nutrient switch, ~30 lines vs
  a 200-line copy) and the Khalil `*_halforder` sewer variants (the generator
  [`models/_make_khalil_variants.py`](aquakin/models/_make_khalil_variants.py)
  emits the pure-override `halforder` variant as a thin `extends` file; the
  variants whose stoichiometry is *computed from the base* stay full copies).
- Optional `speciation:` block declares a **state-derived pH** (see below).
- Optional `positivity_limiter:` block (`threshold`, default `1e-3`) throttles
  each species' *net reaction* term as its concentration approaches zero,
  preventing negative states and the stiffness they cause. Reproduces the
  reference WATS S-function scheme
  `R_lim = max(R,0) + min(R,0) * C / max(C, threshold)`. Applied inside
  `CompiledModel.dCdt` to the reaction term only (reactors add transport
  afterwards), so every reactor benefits. Opt-in; off when the block is
  absent. Stored as `CompiledModel.positivity_threshold`.
- Optional `clip_negative_states: true` (default `false`) clamps the
  concentration vector to `>= 0` *when evaluating the reaction rates* (and any
  state-derived condition such as pH), leaving the raw state for the reactor's
  transport term and the unit outputs. This protects nonlinear kinetics (Monod /
  ratio terms) from a transiently-negative state — where they produce
  large/garbage rates and a stiff blow-up — and is exactly the reference
  IWA/BSM S-function convention (`xtemp = max(x, 0)` before the process rates).
  It is **distinct** from `positivity_limiter`: the limiter throttles the *net
  reaction* near zero (output side), while this clamps the rate *inputs* — so it
  prevents the Monod-at-negative-`C` blow-up that the limiter does not. Identity
  at feasible states (does not change the physical solution); leaving transport
  on the raw state keeps the linear washout self-correcting and the inter-unit
  mass balance exact. Enabled on `asm1` (needed once the BSM1 settler recycles
  concentrated solids into a small reactor). Stored as
  `CompiledModel.clip_negative_states`.
- YAML is loaded with `yaml.safe_load()` throughout. Species names that
  could be misread as non-string types (e.g. `NO`) must be quoted in YAML.

---

