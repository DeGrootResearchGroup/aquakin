# YAML model format

A model file is a YAML document that declares the species, the conditions they
react under, the parameters, and the reactions. aquakin validates it when it is
loaded and compiles it into a differentiable rate function, so nothing has to be
rebuilt to change a model. This page is the reference for the format; for a
step-by-step introduction, see [Your first model](adding_models.md).

## A complete minimal model

First-order decay of a species `A` into `B`:

```yaml
# first_order_decay.yaml
model:
  name: first_order_decay
  version: "1.0"
  description: "A decays to B at a first-order rate."
  references:
    - "Textbook first-order kinetics."

species:
  - name: A
    units: mol/L
    default_concentration: 1.0e-3
  - name: B
    units: mol/L

parameters:
  k: {value: 0.01, units: 1/s, bounds: [1.0e-4, 1.0], transform: positive_log}

reactions:
  - name: decay
    reference: "Textbook first-order kinetics."
    rate: "k * [A]"
    stoichiometry: {A: -1, B: 1}
```

Load it from its file and inspect what aquakin read:

```python
import aquakin

model = aquakin.load_model_from_file("first_order_decay.yaml")
print(model.summary())    # species, reactions, parameters, and the time unit
```

`aquakin.load_model("<name>")` loads a model shipped with aquakin by name; see
the [Model catalogue](model_catalog.md).

## Top-level keys

Every key other than `model:`, `species:` and `reactions:` is optional.

```yaml
model:                    # metadata
  name: <str>
  version: "<str>"                # default "1.0"
  description: <str>
  references: [<str>, ...]        # literature citations
  extends: <model>                # optional: inherit a base model (see below)

species:                  # list of species
  - name: <str>
    description: <str>
    units: <str>                  # default "mol/L"
    default_concentration: <float>  # default 0.0
    composition: {<quantity>: <float>, ...}  # e.g. {COD: 1.0, N: 0.086}

conditions:               # list of condition fields (pH, T, light, ...)
  - name: <str>
    description: <str>
    units: <str>                  # default "" (read by check_units only)
    default: <float>              # default 0.0

parameters:               # model-level (shared) parameters
  <name>:
    value: <float>
    units: <str>                  # also sets the model's time unit (see below)
    description: <str>            # optional: what the parameter is
    bounds: [<low>, <high>]
    transform: <str>              # "none" (default) | "positive_log" | "logit"
    prior: {mean: <float>, std: <float>}   # or {range: [<low>, <high>]}
    temperature:                  # rate constant * theta**(T - ref_T)
      theta: <float or name>      # per-degree factor, or a parameter's name
      ref_T: <float>              # temperature at which `value` applies
      condition: <str>            # temperature condition (default "T")

expressions:              # named intermediate rate expressions
  <name>: "<formula>"

conserved_for: [<quantity>, ...]  # default quantities to solve `auto` from

reactions:                # list of reactions
  - name: <str>
    description: <str>
    reference: <str>              # optional but encouraged
    rate: "<expression>"          # inline, or the name of an expression
    parameters:                   # reaction-local parameters (same keys as above)
      <local_name>: {value: <float>, ...}
    stoichiometry:
      <species_name>: <float, "<expression>", or auto>
    conserved_for: [<quantity>, ...]  # overrides the model-level list
    units_acknowledged: <str>     # why the rate's units are unconventional

speciation: {...}         # a pH computed from the state (see below)
precipitation: {...}      # mineral precipitation (see below)
gases: {...}              # which species are dissolved gases (see below)
positivity_limiter: {threshold: <float>}  # see "Keeping states non-negative"
clip_negative_states: <bool>              # default false; every shipped model sets it

remove: {...}             # with `extends:` only (see below)
```

An unknown key is an error, so a misspelt key is reported when the model is
loaded rather than silently ignored.

## Units and the time unit

The solver never converts a value because of its `units:` string: every value
in the file is used exactly as written. The units matter in three ways.

**The parameter units set the model's time unit.** aquakin has no global time
unit. `t_span` and `t_eval` are read in whatever unit the rate constants are
written in, so the same `t_span=(0, 600)` is 600 seconds for a model whose rate
constants are in `1/s` and 600 days for one in `1/d`. aquakin reads that unit
from the parameter units and reports it as `model.time_unit` (`"s"`, `"min"`,
`"h"` or `"d"`); `model.summary()` prints it. It is `None` when no parameter
declares a time unit, or when two rate constants disagree, which is almost
always a mistake. Declare a unit on every rate constant, and write them all on
the same time unit.

**They let `check_units` find dimensional mistakes**, as described in
[Checking units](#checking-units).

**They label and convert results on request.** A solution reports each species
in its declared unit, and readers such as `sol.C_named("SNH", units="mg/L")`
convert to another unit of the same quantity
([Reading results](getting_started.md#reading-results)).

## Shared parameters vs reaction-local parameters

Most models of practical interest declare a kinetic constant (e.g. `muH`) once
and reuse it across several reactions. The top-level `parameters:` block is the
right home for these: it produces a single parameter slot that any number of
reactions can reference by its bare name, and calibration then tunes that one
slot.

A reaction's own `parameters:` block holds values that belong to that reaction
alone. Such a parameter is named `<reaction>.<name>` outside the model file, for
example in the list of parameters to fit,
`aquakin.calibrate(..., free_params=["decay.k"])`
([Calibration](calibration.md)). A name cannot appear in both a
model-level and a reaction-local `parameters:` block; the loader rejects the
shadowing.

## Bounds, transforms and priors

These three keys describe what is known about a parameter, for calibration and
analysis. None of them changes a forward solve.

- `bounds: [low, high]` are hard limits. `fit`, `calibrate`,
  `profile_likelihood` and the design optimizers keep the parameter inside them.
- `transform:` chooses the scale on which calibration and sensitivity analysis
  work. Use `positive_log` for a quantity that must stay positive and can span
  orders of magnitude, such as a rate constant: the optimizer then works on its
  logarithm, so a step is a relative change. Use `logit` for a fraction that
  must stay between 0 and 1, such as a yield. The default `none` works on the
  value itself.
- `prior:` is a Gaussian belief about the value, added to the calibration
  objective. Write a measured value with its uncertainty as
  `prior: {mean: 0.67, std: 0.02}`, or a literature range as
  `prior: {range: [0.6, 0.75]}`, which is read as a Gaussian centred on the
  midpoint with the range spanning ±2 standard deviations. The shipped `asm1`
  and `adm1` models declare priors on their calibration parameters.

## Temperature corrections

A rate constant with a `temperature:` block is multiplied by
`theta**(T - ref_T)` when the rates are evaluated, so its `value` is the
value at `ref_T`. `theta` is usually a number. It can also be the name of
another parameter, which makes the temperature sensitivity something you can
calibrate and analyze like any other parameter:

```yaml
parameters:
  theta_A: {value: 1.07, units: "-", bounds: [1.0, 1.15], prior: {mean: 1.07, std: 0.02}}
  muA: {value: 0.8, units: "1/d", temperature: {theta: theta_A, ref_T: 293.15}}
  bA:  {value: 0.05, units: "1/d", temperature: {theta: theta_A, ref_T: 293.15}}
```

Several rate constants may share one `theta` parameter. A name used in a
reaction's own `parameters:` block is looked up in that reaction first, then
among the model-level parameters. The `theta` parameter must be positive and
cannot carry a temperature correction itself. Without `bounds` it gets
`[1.0, 1.3]` when its value lies in that range, and without `units` it is
labelled `"-"`. Its sensitivity is zero at `T = ref_T`, so estimating it needs
data at a temperature other than `ref_T` (and data at two or more temperatures
to estimate it together with the rate constant it scales).

The correction applies to rate constants only. A parameter used in the
stoichiometry, such as a yield, is never temperature-scaled.

## Named rate expressions

The `expressions:` block lets you factor out repeated sub-formulas. A
reaction's `rate:` can then be either an inline formula or a reference to
a named expression by bare name. Expressions can reference species, conditions,
parameters and other expressions; the loader resolves them in order and rejects
cycles.

Example:

```yaml
parameters:
  muH: {value: 6.0, units: 1/d, transform: positive_log}
  KS:  {value: 20.0, units: g_COD/m3}

expressions:
  rho_growth: "muH * [SS] / (KS + [SS]) * [XB_H]"

reactions:
  - name: aerobic_growth
    rate: "rho_growth"            # resolves to the expression's AST inline
    stoichiometry: {SS: -1.49, XB_H: 1.0}
```

## Rate expression grammar

A rate expression is built from three kinds of name, numbers, arithmetic and
function calls:

- **Species names** are delimited by square brackets, e.g. `[O3]`, `[Br-]`,
  `[BrO3-]`. Charge suffixes and digits inside the brackets are accepted.
- **Condition references** are delimited by curly braces, e.g. `{pH}`,
  `{fluence_rate}`, `{OH_scavenging}`. The named field must be declared in
  the `conditions:` block, or be produced by a `speciation:` or
  `precipitation:` block.
- **Identifiers without parentheses** name a parameter or a named expression.
  A reaction-local parameter is looked up first and is named
  `<reaction>.<name>` outside the model file; a model-level parameter or an
  expression keeps its bare name.
- **Identifiers with parentheses** call one of the
  [functions](#built-in-functions) below.

Examples mixing all three reference forms:

```yaml
rate: "k_photo * {fluence_rate} * [H2O2]"            # photolysis
rate: "k * [O3] * 10 ** ({pH} - 14)"                  # OH- catalyzed
rate: "{OH_scavenging} * [OH]"                        # lumped sink
```

The grammar of a rate expression:

```
expr    := term (('+' | '-') term)*
term    := unary (('*' | '/') unary)*
unary   := ('+' | '-') unary | power
power   := primary ('**' unary)?
primary := number
         | '[' species_name ']'
         | '{' condition_name '}'
         | identifier ('(' arglist ')')?
         | '(' expr ')'
arglist := expr (',' expr)* ','?
```

Precedence follows Python and ordinary mathematical notation, so `**` is
right-associative, a sign binds looser than it (`-k**2` is `-(k**2)`), and an
exponent may carry its own sign (`k**-1`). Bracket the base when you mean to
raise a negative value: `(0 - k)**2`.

The rest of the syntax follows Python too, so an expression generally means what
it looks like it means:

- A number may use Python's underscore separators, `18446_000`, and scientific
  notation, `1.5e-3`.
- A call may carry a trailing comma, `monod([SS], KS,)`, so a long one written
  across lines does not make its last argument special.
- `max` and `min` take any number of arguments, as Python's do: `max(a, b, c)`.
- `[X] ** 0.5` and `sqrt([X])` are the same thing, guard included: a fractional
  power of a species that can reach zero stays differentiable there.

What is deliberately *not* there is anything that cannot be differentiated or
traced: there are no comparisons, no `if`/`else`, and no `%` or `//`. `^` is not
an operator either — it is exclusive-or in Python and a power to most readers,
so rather than quietly meaning one of them it is refused. Write a power as `**`.

## Stoichiometry coefficients

Each entry of a reaction's `stoichiometry:` block is a **numeric literal**, a
**string expression** in the model's parameters, or **`auto`** (see
[Conservation](#conservation-composition-and-auto-coefficients)). Numeric
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

A stoichiometric expression may only use numbers, parameters (reaction-local
or model-level) and arithmetic, including `**`. **Species references (`[X]`),
condition references (`{X}`), named expressions and function calls of any kind
are not allowed**: stoichiometry must be state-independent so it can be
evaluated once per integration rather than every ODE step.

The shipped `asm1.yaml` is the worked example: yields (`Y_H`, `Y_A`),
nitrogen content (`i_XB`, `i_XP`), and the decay fraction (`f_P`) are all
model-level parameters with symbolic stoichiometry. Calibrating `Y_H`
adjusts every coefficient that depends on it across aerobic and anoxic
heterotrophic growth in one step.

## Conservation: composition and `auto` coefficients

A species can declare its content of conserved quantities, in its own measure,
with `composition:`. The quantity names are yours to choose (`COD`, `N`, `P`,
`S`, `charge`, ...): `{COD: 1.0}` for an organic measured as COD, `{COD: -1.0}`
for dissolved oxygen (an electron acceptor), `{COD: -2.86, N: 1.0}` for
nitrate-N. `model.check_conservation()` then multiplies each reaction's
stoichiometry by these contents and reports every reaction that does not
balance; it is a check you run, and loading never fails because of it.

With the contents declared, a coefficient that a balance determines need not be
worked out by hand. Write it as `auto` (or `?`), and name the quantities to
balance in `conserved_for:`, on the reaction or once for the whole model:

```yaml
conserved_for: [COD]

species:
  - {name: SS,   units: g_COD/m3, composition: {COD: 1.0}}
  - {name: XB_H, units: g_COD/m3, composition: {COD: 1.0}}
  - {name: SO,   units: g_O2/m3,  composition: {COD: -1.0}}

parameters:
  Y_H: {value: 0.67, units: g_COD/g_COD, transform: logit}
  muH: {value: 6.0, units: 1/d, transform: positive_log}

reactions:
  - name: aerobic_growth
    rate: "muH * [XB_H]"
    stoichiometry: {SS: "-1 / Y_H", XB_H: 1, SO: auto}
```

aquakin solves the oxygen coefficient from the COD balance when the model is
compiled, here `-(1 - Y_H) / Y_H`. Because it follows `Y_H`, the reaction stays
balanced for every value calibration tries. There must be one conserved
quantity per `auto` coefficient when any other coefficient is an expression,
and every species in the reaction needs the relevant `composition:` entries.
The shipped WATS sewer model declares `composition:` on every species.

## Built-in functions

The following functions may be called in any rate expression.

### Domain functions

- `arrhenius(A, Ea)` — `A * exp(-Ea / (R * T))`, with `R` = 8.314 J/(mol·K), so
  `Ea` is in J/mol. Requires a condition field named `T` (Kelvin). For the
  `theta**(T - ref_T)` form the activated-sludge models use, give the rate
  constant a [`temperature:` block](#temperature-corrections) instead.
- `pH_switch(pKa)` — `1 / (1 + 10**(pH - pKa))`, the protonated fraction.
  Requires a condition field named `pH`.
- `pH_inhibit(pH_LL, pH_UL)` — lower-pH Hill inhibition, `1` at high pH and
  `0` at low pH, spanning roughly the `[pH_LL, pH_UL]` window. Requires a
  condition field named `pH`.
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

### General math functions

For rate laws the domain vocabulary above does not cover.

- `exp(x)` — natural exponential. For an exponential factor that is not an
  Arrhenius temperature correction: a Beer–Lambert attenuation
  `exp(-alpha * {z})`, a thermal-inactivation term, an explicit sigmoid. The
  argument must be dimensionless, which `check_units` enforces.
- `log(x)` — natural logarithm, for a logarithmic driving force such as
  `R*T*log([Q] / K)`. The argument must be dimensionless (take the ratio
  first). For a base-10 logarithm, divide by `log(10)`.
- `sqrt(x)` — square root, the clearer spelling of half-order kinetics than
  `[X] ** 0.5`. Halves the argument's units, so a half-order rate constant
  must carry the matching half-order units.
- `min(a, b, ...)` — elementwise minimum; the ceiling counterpart of `max`. A
  two-sided clip is `min(hi, max(lo, x))`.
- `max(a, b, ...)` — elementwise maximum. Use it for a one-sided clip, e.g.
  `max(0, P_gas - P_atm)` so an overpressure-driven flux only ever stops,
  never reverses.

`log` and `sqrt` are guarded so their **gradient** stays finite where the
underlying function's is not — at a fully depleted species, which is where a
half-order or logarithmic term routinely sits, and where an unguarded
derivative would be `inf` and would poison the whole parameter gradient.
`sqrt(0)` is `0`, and a (transiently) negative argument gives `0` rather than
`NaN`. `log` is different: a logarithm has no finite limit at zero, so the
value there is `-inf` and no stand-in is invented. A model using `log` must
keep its argument positive — set `clip_negative_states: true`, or write the
term so the argument cannot reach zero, e.g. `log(1 + [X]/K)`.

### Rate functions of your own

When your rate law needs a function none of the above provides, register one
instead of folding the term into a precomputed condition field — a coefficient
hidden in a condition field is outside the parameter vector, and so outside
calibration, sensitivity and identifiability.

A rate function is a small class: a frozen dataclass whose fields are its
arguments, carrying the name it is called by in a rate expression
(`FUNCTION_NAME`), an identifier (`KIND`: any string no built-in uses, usually
the same as `FUNCTION_NAME`) and the arithmetic itself (`op`), written with
`jax.numpy` so it differentiates.
It subclasses `aquakin.RateFunction`, the base every rate function shares,
built-in or your own; its arguments are annotated `aquakin.ASTNode`, the type
of a parsed sub-expression.

```python
from dataclasses import dataclass
import aquakin
from aquakin.core.kernels import safe_pow

@dataclass(frozen=True)
class HillNode(aquakin.RateFunction):
    X: aquakin.ASTNode
    K: aquakin.ASTNode
    n: aquakin.ASTNode

    KIND = "hill"            # a unique identifier, usually the function name
    FUNCTION_NAME = "hill"   # how it is spelled in a rate expression

    @staticmethod
    def op(o):               # ONE tuple: the fields in order
        x, k, n = o
        xn = safe_pow(x, n)  # x**n, with a finite derivative at x = 0
        return xn / (safe_pow(k, n) + xn)

aquakin.register_rate_function(HillNode)
```

`register_rate_function` returns the class, so it also works as a decorator
(`@aquakin.register_rate_function` above `@dataclass(frozen=True)`).
`hill(X, K, n)` is now callable in any rate expression, and its constants are
ordinary model parameters:

```yaml
rate: "k * hill([S], K_h, n_h) * [S]"
```

A registered function is not second class. It parses, evaluates, reaches the
same batched rate kernel (bit-identically to its own scalar path), and
differentiates exactly as a built-in does.

Three things to know:

- **Register before loading.** A model compiles its rate expressions when it is
  loaded, so call `register_rate_function` before the `load_model` or
  `load_model_from_file` that loads a model using the function. A model file naming an
  unregistered function fails at load, listing what is available.
- **Names cannot shadow a built-in.** Registering `monod` is refused, because it
  would change what every existing model file using that name means; a `KIND`
  a built-in already uses is refused too. Defining a
  node class is never on its own enough to add a function — only
  `register_rate_function` is.
- **Your `op` gets none of the built-in guards.** The built-in functions keep
  their gradients finite at a depleted species; plain `jax.numpy` arithmetic
  does not. A fractional power at zero has an infinite derivative — written as
  `x**n`, the example's Hill term with `n < 1` would have one at `x = 0`, which
  is where a depleted species sits, and it would poison the whole parameter
  gradient. Use the guarded helpers in `aquakin.core.kernels`: `safe_pow(x, e)`
  for `x**e`, `safe_sqrt(x)` and `safe_log(x)`. Each keeps the value exact and
  the derivative finite at zero (`safe_log`'s value there is still `-inf`, so
  keep its argument positive). None of them gives a negative base with a
  fractional exponent a real value: that is `NaN`, as it should be. Write the
  rest of the arithmetic so it stays finite over the states your model
  reaches.

Optionally, declare how the result's dimension follows from the arguments', so
`check_units` can check expressions using it. Without this the dimension is
simply unknown and the check is skipped for that subexpression:

```python
    @staticmethod
    def DIMENSION(args):          # args: each argument's Dimension, or None
        from aquakin.utils.units import DIMENSIONLESS
        return DIMENSIONLESS      # a Hill term is a saturation fraction
```

`aquakin.registered_rate_functions()` lists what is registered, and
`aquakin.core.registry.unregister_rate_function(name)` removes one — useful
when iterating in a notebook.

## Keeping states non-negative

A stiff solve can step a nearly depleted species slightly below zero, where a
Monod or ratio term gives a large, wrong rate and the solve can blow up. Two
opt-in keys guard against it. Both leave the solution unchanged wherever every
concentration is non-negative.

- `clip_negative_states: true` evaluates the rates (and a state-derived pH)
  with negative concentrations treated as zero. Transport and the reported
  states still use the unclipped values, so mass balances stay exact. Every
  shipped model sets it.
- `positivity_limiter: {threshold: 1.0e-3}` scales down the net consumption of
  a species once its concentration falls below `threshold`, so it approaches
  zero without crossing it. The shipped `asm2d` and WATS sewer models use it.

They act at different points: the clip protects the rate laws from a negative
input, and the limiter keeps the state from going negative in the first place.
A model can use both.

## pH and precipitation

Two blocks add aqueous chemistry that is solved from the state as it changes,
rather than supplied as a fixed condition.

**`speciation:`** computes the pH from a charge balance on every evaluation of
the rates and provides it as a condition field (named `pH` unless `field:` says
otherwise), so `{pH}`, `pH_switch` and `pH_inhibit` follow the chemistry. You
map your species onto the acid/base systems aquakin knows (`carbonate`,
`acetate`, `propionate`, `butyrate`, `valerate`, `ammonia`, `phosphate`,
`sulfide`) and list the strong ions:

```yaml
speciation:
  field: pH
  temperature_field: T
  temperature_units: celsius        # or kelvin
  z_cation_eq: 3.28e-3              # net fixed cation charge (eq/L)
  totals:
    carbonate: {species: S_CO2, molar_mass: 12000}   # g C/m3 -> mol/L
    ammonia:   {species: S_NH,  molar_mass: 14000}   # g N/m3 -> mol/L
  strong_anions:
    - {species: S_SO4, molar_mass: 32000, charge: 2}
```

`molar_mass` converts the species' own units to mol/L: a molecular weight (in
g/mol, times 1000 for g/m³) for a mass-based species, a power of ten for a
molar one. A value that does not fit the species' declared units raises a
`SpeciationUnitsWarning` when the model is loaded. `activity_model:` (`none`,
`davies` or `debye_huckel`) adds activity corrections. The shipped `adm1` and
`wats_sewer` models are complete examples.

**`precipitation:`** declares minerals that precipitate and dissolve at a rate
driven by their saturation index, computed from the free-ion activities at the
current pH. Each mineral gives its solubility product, its ions, and, to have
the precipitation reaction written for you, the precipitate species and a rate
constant:

```yaml
precipitation:
  pH_field: pH
  temperature_field: T
  temperature_units: kelvin
  activity_model: davies
  minerals:
    - name: calcite
      pKsp: 8.48
      order: 2
      solid: X_calcite
      rate_constant: {value: 10.0, units: 1/d, transform: positive_log}
      ions:
        - {species: S_Ca, molar_mass: 1000, count: 1, charge: 2}
        - {species: S_IC, molar_mass: 1000, count: 1, charge: 2, fraction: carbonate}
```

The reaction written for you is `<name>_precipitation`, with rate
`k * [solid] * {R_<name>}`: it is proportional to the solid already present, so
give the solid a small seed `default_concentration` or nothing precipitates.
Each mineral also provides its saturation index `{SI_<name>}` and rate factor
`{R_<name>}` as condition fields, for a reaction you write yourself.
`fraction:` takes the free ion from an acid/base total at the current pH; an
ion without it is taken as fully free. `mode: equilibrium` solves the
precipitate to its solubility limit algebraically instead, and
`supersaturation_form: bounded` keeps a very insoluble mineral's kinetics
well-conditioned. The shipped `precipitation_struvite_calcite`,
`precipitation_metal_phosphate` (with its `_bounded` and `_equilibrium`
variants) and `asm2d_chemp` models show each form.

## Dissolved gases: `gases:`

A model can say which of its species are dissolved gases, and leave the gas
phase to the unit it runs in: the same model then runs in every unit, and the
units that own a gas phase transfer these species (a digester to its
headspace, an aerated tank to the air). This is the alternative to writing the
gas phase into the model as states, as `adm1` does with its headspace
(`S_gas_h2`, `S_gas_ch4`, `S_gas_co2`); no shipped model declares `gases:`
yet.

```yaml
gases:
  temperature_field: T            # the temperature condition
  temperature_units: kelvin       # or celsius
  pH_field: pH                    # needed only for a volatile_fraction
  species:
    SO2:  {gas: O2,  molar_mass: 32000}    # g O2/m3  -> mol/L
    SCH4: {gas: CH4, molar_mass: 64000}    # g COD/m3 -> mol/L (64 g COD/mol)
    SIC:  {gas: CO2, molar_mass: 12000, volatile_fraction: carbonate}
```

Each entry names a gas of the built-in library (`O2`, `N2`, `CO2`, `CH4`, `H2`)
and, as in `speciation:`, the `molar_mass` that converts the species' units to
mol/L. With `volatile_fraction: carbonate` only the un-ionised CO2 (H2CO3*) of
an inorganic-carbon total transfers, at the current pH.

The library holds each gas's Henry's-law solubility at 25 C and its
temperature dependence (the evaluated reviews in Sander, *Atmos. Chem. Phys.*
23, 10901, 2023) and its diffusivity in water (CRC Handbook), from which its
transfer coefficient relative to oxygen's follows, `(D / D_O2) ** 0.5`. An entry
can override any of them: `henry` (mol m-3 Pa-1 at 25 C), `henry_temperature`
(`d ln H / d(1/T)`, K) and `kL_ratio`. The compiled declaration is
`model.gases`.

An aerated tank strips the declared gases with `Aeration(stripped="model")`
(see [Other activated-sludge models](plants.md#other-activated-sludge-models)),
and `aquakin.plant.HeadspaceDigesterUnit` transfers them to its headspace.

## Inheriting a model: `extends:`

A model that differs from another by a few parameters or rates can inherit it
instead of copying it, so a correction to the base reaches every variant.
`model.extends:` names a shipped model, or a file path relative to the
extending file (anything containing `/` or ending in `.yaml`):

```yaml
model:
  name: asm1_cold
  extends: asm1
  description: "ASM1 with a slower autotroph growth rate and no ammonification."

parameters:
  muA: {value: 0.5}          # changes only the value; units, bounds and prior stay

remove:
  reactions: [ammonification]
```

The derived file is merged onto the base before it is validated:

- In `species:`, `conditions:` and `reactions:`, an entry whose `name` matches
  one in the base changes only the fields it gives (override a reaction's
  `rate` and it keeps its `stoichiometry`); a new name is added at the end.
- Every other block merges key by key; a value that is not a mapping, such as a
  list or a number, replaces the base's.
- `remove:` then drops named entries from `species`, `conditions`,
  `reactions`, `parameters` or `expressions`. Removing a name that is not
  there is an error.

New parameters are added after the base's, so the order of the parameter vector
can differ from a hand-written copy; refer to parameters by name. The shipped
`asm1_ammonia_limitation` model is `asm1` plus one extra switch, written this
way.

## Checking units

`model.check_units()` reads the `units:` strings on species, parameters and
conditions to check every rate expression for dimensional consistency. It is
currency-aware: `g_COD/m3` and `g_N/m3` are different units, so it catches a
Monod term comparing two currencies, a sum of unlike terms, a dropped
concentration factor, or rate constants written on different time units. A
blank or unparseable unit is skipped, so declaring units on every parameter is
what makes the check useful.

It also checks each rate against the stoichiometry. A species whose
coefficient is a fixed `+1` or `-1` changes by exactly the rate, so the rate
must be in that species' concentration per time: a rate whose stoichiometry
takes up 1 g P per unit (`SPO4: -1`) must come out in g P/m3/d, which means a
specific uptake rate in g P/(g COD·d), not 1/d. A reaction is flagged only when
its rate matches none of its unit-coefficient species; other coefficients
(yields, conversion factors) legitimately carry mixed units and are not
checked. `check_units(check_stoichiometry=False)` turns this part off.

A literal half-saturation constant written inline, as in
`0.1 / (0.1 + [S_NO])`, is read as an inhibition term (dimensionless) when it
is its own subexpression, for example a named expression.

Some published formulations are knowingly unconventional in their units, for
example a molar mass written as a bare number. Give such a reaction a written
reason:

```yaml
  - name: gas_outflow_h2
    rate: "q_gas_norm * [S_gas_h2]"
    units_acknowledged: "Partial pressures divide by molar masses written as bare numbers."
    stoichiometry: {S_gas_h2: -1.0}
```

`check_units()` then leaves that reaction's findings out of its result;
`check_units(include_acknowledged=True)` still shows them, and
`model.units_acknowledged` lists the reasons.

## Notes

- Species names that YAML might otherwise interpret as non-strings (e.g.
  `NO`, `ON`) must be quoted in YAML.
- A mapping may not repeat a key: a species listed twice in one
  stoichiometry, or a key given twice in any other block, is an error that
  names the key and both lines, rather than the last value silently winning.
- `default_concentration` is a reference value, not an experimental initial
  condition. Override at runtime via the `C0` argument to `solve()`.
- `references` and `reference` keys are free-form and should cite the source
  literature so that the YAML file is self-documenting.
