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
    temperature:                # optional: rate constant * theta**(T - ref_T)
      theta: <float or name>    # per-degree factor, or a parameter's name
      ref_T: <float>            # temperature at which `value` applies
      condition: <str>          # optional: temperature condition (default "T")

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
    units_acknowledged: <str>   # optional: why the rate's units are
                                # knowingly unconventional (see below)
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

## Temperature corrections

A rate constant with a `temperature:` block is multiplied by
`theta**(T - ref_T)` when the rates are evaluated, so its `value` is the
value at `ref_T`. `theta` is usually a number. It can also be the name of
another parameter, which makes the temperature sensitivity something you can
calibrate and analyse like any other parameter:

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

What is deliberately *not* there is anything that cannot be differentiated or
traced: there are no comparisons, no `if`/`else`, and no `%` or `//`. `^` is not
an operator either — it is exclusive-or in Python and a power to most readers,
so rather than quietly meaning one of them it is refused. Write a power as `**`.

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

## Built-in functions

The following functions may be called in any rate expression.

### Domain functions

- `arrhenius(A, Ea)` — `A * exp(-Ea / (R * T))`. Requires a condition field
  named `T` (Kelvin).
- `pH_switch(pKa)` — `1 / (1 + 10^(pH - pKa))`, the protonated fraction.
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
- `min(a, b)` — elementwise minimum; the ceiling counterpart of `max`. A
  two-sided clip is `min(hi, max(lo, x))`.
- `max(a, b)` — elementwise maximum. Use it for a one-sided clip, e.g.
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

Examples mixing all three reference forms:

```yaml
rate: "k_photo * {fluence_rate} * [H2O2]"            # photolysis
rate: "k * [O3] * 10 ** ({pH} - 14)"                  # OH- catalysed
rate: "{OH_scavenging} * [OH]"                        # lumped sink
```

### Rate functions of your own

When your rate law needs a function none of the above provides, register one
instead of folding the term into a precomputed condition field — a coefficient
hidden in a condition field is outside the parameter vector, and so outside
calibration, sensitivity and identifiability.

A rate function is a node class: a frozen dataclass whose fields are its
arguments, carrying the call spelling (`FUNCTION_NAME`), a kernel key (`KIND`)
and the arithmetic itself (`op`), written with `jax.numpy` so it differentiates.

```python
from dataclasses import dataclass
import jax.numpy as jnp
import aquakin
from aquakin.core.nodes import ASTNode, _OperatorNode

@dataclass(frozen=True)
class HillNode(_OperatorNode):
    X: ASTNode
    K: ASTNode
    n: ASTNode

    KIND = "hill"            # the batched kernel's key for this node type
    FUNCTION_NAME = "hill"   # how it is spelled in a rate expression

    @staticmethod
    def op(o):               # ONE tuple: the fields in order
        x, k, n = o
        return x**n / (k**n + x**n)

aquakin.register_rate_function(HillNode)
```

`hill(X, K, n)` is now callable in any rate expression, and its constants are
ordinary model parameters:

```yaml
rate: "k * hill([S], K_h, n_h) * [S]"
```

A registered function is not second class. It parses, evaluates, reaches the
same batched rate kernel (bit-identically to its own scalar path), and
differentiates exactly as a built-in does.

Two things to know:

- **Register before loading.** A model compiles its rate expressions when it is
  built and `load_model` caches the result, so registration has to happen
  before the model that uses the function is loaded. A model file naming an
  unregistered function fails at load, listing what is available.
- **Names cannot shadow a built-in.** Registering `monod` is refused, because it
  would change what every existing model file using that name means. Defining a
  node class is never on its own enough to add a function — only
  `register_rate_function` is.

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

## Checking units

The `units:` strings on species, parameters and conditions are not used to
convert anything, but `model.check_units()` reads them to check every rate
expression for dimensional consistency. It is currency-aware: `g_COD/m3` and
`g_N/m3` are different units, so it catches a Monod term comparing two
currencies, a sum of unlike terms, a dropped concentration factor, or rate
constants written on different time units. A blank or unparseable unit is
skipped, so declaring units on every parameter is what makes the check useful.

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
- `default_concentration` is a reference value, not an experimental initial
  condition. Override at runtime via the `C0` argument to `solve()`.
- `references` and `reference` keys are free-form and should cite the source
  literature so that the YAML file is self-documenting.
