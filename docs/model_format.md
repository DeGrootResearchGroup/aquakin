# YAML model format

A model file declares species, conditions, and reactions. The top-level
keys are:

```yaml
model:                # metadata
  name: <str>
  version: "<str>"
  description: <str>
  references: [<str>, ...]

species:                # list of species
  - name: <str>
    description: <str>          # optional
    units: <str>                # default "mol/L"
    default_concentration: <float>  # default 0.0

conditions:             # optional list of condition fields
  - name: <str>
    description: <str>          # optional
    default: <float>

parameters:             # optional: model-level (shared) parameters
  <name>:
    value: <float>
    units: <str>                # optional
    bounds: [<low>, <high>]     # optional
    transform: <str>            # optional: "none" | "positive_log" | "logit"

expressions:            # optional: named intermediate rate expressions
  <name>: "<formula>"
  # formulas may reference species, conditions, parameters, and *other*
  # expressions. Cycles are rejected at load time.

reactions:              # list of reactions
  - name: <str>
    description: <str>          # optional
    reference: <str>            # optional but encouraged
    rate: "<expression>"        # may be inline OR a reference to an
                                # entry in `expressions:`
    parameters:                 # optional reaction-local parameters
      <local_name>:
        value: <float>
        units: <str>            # optional
        bounds: [<low>, <high>] # optional
        transform: <str>        # optional
    stoichiometry:
      <species_name>: <float>
```

## Shared parameters vs reaction-local parameters

Most models of practical interest declare a kinetic constant (e.g.
`muH`) once and reuse it across multiple reactions. The top-level
`parameters:` block is the right home for these — it produces a single
parameter slot that any number of reactions can reference by bare name.
Calibration then tunes that single slot.

Reaction-local `parameters:` blocks remain available for genuinely
reaction-specific values. A name cannot appear in both a model-level
`parameters:` block and a reaction-local `parameters:` block — the
loader rejects shadowing.

## Named rate expressions

The `expressions:` block lets you factor out repeated sub-formulas. A
reaction's `rate:` can then be either an inline formula or a reference to
a named expression by bare name. Expressions can reference other
expressions; the loader topologically resolves them and rejects cycles.

Example:

```yaml
parameters:
  muH: {value: 6.0, transform: positive_log}
  KS:  {value: 20.0}

expressions:
  rho_growth: "muH * [SS] / (KS + [SS]) * [XB_H]"

reactions:
  - name: aerobic_growth
    rate: "rho_growth"            # resolves to the expression's AST inline
    stoichiometry: {SS: -1.49, XB_H: 1.0}
```

## Rate expression grammar

Rate expressions are parsed by a hand-written recursive-descent parser. The
grammar:

```
expr    := term (('+' | '-') term)*
term    := factor (('*' | '/') factor)*
factor  := unary ('**' factor)?
unary   := ('+' | '-') unary | primary
primary := number
         | '[' species_name ']'
         | '{' condition_name '}'
         | identifier ('(' arglist ')')?
         | '(' expr ')'
```

- **Species names** are delimited by square brackets, e.g. `[O3]`, `[Br-]`,
  `[BrO3-]`. Charge suffixes and digits inside the brackets are accepted.
- **Condition references** are delimited by curly braces, e.g. `{pH}`,
  `{fluence_rate}`, `{OH_scavenging}`. The named field must be declared in
  the `conditions:` block.
- **Identifiers without parentheses** are rate-constant references. They are
  resolved against the owning reaction's `parameters:` block and namespaced
  as `<reaction_name>.<name>` internally.
## Stoichiometry coefficients

Each entry of a reaction's `stoichiometry:` block is either a **numeric
literal** or a **string expression** in the model's parameters. Numeric
literals are precomputed; string expressions are evaluated at the start of
every `solve()` call from the current parameter vector, which means
yield / N-content / fraction parameters can be calibrated alongside
kinetic constants.

```yaml
parameters:
  Y_H: {value: 0.67, transform: logit, bounds: [0.4, 0.85]}

reactions:
  - name: aerobic_growth_heterotrophs
    rate: "rho_hetero_aerobic"
    stoichiometry:
      SS:   "-1 / Y_H"            # symbolic — Y_H is calibratable
      XB_H: 1.0                   # numeric literal — fixed
      SO:   "-(1 - Y_H) / Y_H"
```

Stoichiometric expressions may only reference numeric constants, parameters
(reaction-local or model-level), and the arithmetic / negation operators.
**Species references (`[X]`), condition references (`{X}`), named
expressions, and domain functions (`monod`, `arrhenius`, ...) are not
allowed** — stoichiometry must be state-independent so it can be evaluated
once per integration rather than every ODE step.

The shipped `asm1.yaml` is the worked example: yields (`Y_H`, `Y_A`),
nitrogen content (`i_XB`, `i_XP`), and the decay fraction (`f_P`) are all
model-level parameters with symbolic stoichiometry. Calibrating `Y_H`
adjusts every coefficient that depends on it across aerobic and anoxic
heterotrophic growth in one step.

## Rate expression grammar

- **Built-in functions:**
  - `arrhenius(A, Ea)` — `A * exp(-Ea / (R * T))`. Requires a condition field
    named `T` (Kelvin).
  - `pH_switch(pKa)` — `1 / (1 + 10^(pH - pKa))`, the protonated fraction.
    Requires a condition field named `pH`.
  - `monod(X, K)` — `X / (K + X)`. Saturation Monod term; the standard
    substrate-limitation form in microbial kinetics.
  - `monod_inh(X, K)` — `K / (K + X)`. Inhibition Monod; equal to
    `1 - monod(X, K)`. Used as an aerobic/anoxic switch in ASM-family models.
  - `monod_ratio(A, B, K)` — `(A/B) / (K + A/B)`, written numerically as
    `A / (K*B + A)`. The substrate-to-biomass ratio form used in ASM
    hydrolysis kinetics.
  - `monod_inh_ratio(A, B, K)` — `K / (K + A/B)`. The inhibition counterpart
    of `monod_ratio`. Appears in bio-P models as a gate on the
    storage-to-biomass ratio.
  - `safe_div(num, denom)` — `num / denom`, but returns `0` (with a finite
    gradient) where `denom == 0` instead of `inf`/`NaN`. Use it for a ratio
    whose denominator can legitimately reach exactly zero — e.g. a
    substrate-competition fraction `safe_div([A], [A] + [B])` where both
    substrates can deplete to 0 — so the rate takes its physical limit `0`
    there, without padding the denominator with a small dimensionless constant.

Examples mixing all three reference forms:

```yaml
rate: "k_photo * {fluence_rate} * [H2O2]"            # photolysis
rate: "k * [O3] * 10 ** ({pH} - 14)"                  # OH- catalysed
rate: "{OH_scavenging} * [OH]"                        # lumped sink
```

## Notes

- Species names that YAML might otherwise interpret as non-strings (e.g.
  `NO`, `ON`) must be quoted in YAML.
- `default_concentration` is a reference value, not an experimental initial
  condition. Override at runtime via the `C0` argument to `solve()`.
- `references` and `reference` keys are free-form and should cite the source
  literature so that the YAML file is self-documenting.
