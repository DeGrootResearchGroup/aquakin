---
paths:
  - "aquakin/core/**"
  - "aquakin/utils/**"
---

# Rules — `aquakin/core/`

AST rate evaluation, parser, conditions, parameter
namespacing, the charge-balance pH solver / speciation, and mineral
precipitation. Loaded automatically when editing files under `aquakin/core/`.

**Topic rules split out of this file** (each auto-loads for its own files; read
it on demand when a change elsewhere, e.g. the wiring in `core/model.py` or the
schema in `schema/model_spec.py`, touches the topic):

| Rule | Loads for | Covers |
|---|---|---|
| `core-speciation.md` | `core/ph_solver.py`, `core/speciation.py` | the charge-balance pH solver, derived conditions, the `speciation:` block |
| `core-precipitation.md` | `core/precipitation.py`, `core/precipitation_equilibrium.py` | the precipitation rate law, the `precipitation:` block, the algebraic equilibrium mode |
| `core-rate-functions.md` | `core/nodes.py`, `core/registry.py`, `core/parser.py`, `core/vector_kernel.py` | registering a rate function, authoring a node (one `op`, everything derived), the vectorized rate kernel |

## Layering

`core` and `utils` are one level (they import each other) and must not import
`schema`, `integrate`, `plant` or `study`, not even inside a function body:
`tests/unit/test_layering.py` scans every import and fails on one. A piece both
`core` and a higher layer need lives in `core` and is re-exported upward
(`default_atol` is in `core/tolerance.py`). The test allows exactly two
exceptions, each because a public `CompiledModel` method needs the layer above:
- `CompiledModel.influent` imports `plant.influent`, because `net.influent(...)`
  is how plants build their feed.
- `utils/composition.py` imports `schema.loader.load_model`, because
  `CompiledModel.composition()` falls back to a shipped model's table and
  choosing it compares the model's species with the shipped model's. The
  `extends:` chain is read at call time, so the base cannot be recorded when
  the model is loaded.

Add no third: move the shared piece down instead.

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

**General math function nodes** (plain mathematics, for rate laws the
activated-sludge vocabulary does not cover; `exp`/`log` are the only way to
write such a term *inside* a model file rather than precomputing it and
injecting it as a condition field)

- `ExpNode(x)` — `exp(x)`. Unguarded: finite and smooth everywhere, and an
  overflow to `inf` is an honest report of an ill-scaled model. `check_units`
  requires a dimensionless argument.
- `LogNode(x)` — `log(x)` (natural). Guarded by `safe_log`: the double-where
  keeps the **gradient** finite at and below zero, but the *value* there stays
  `-inf` — a logarithm has no finite limit at zero to adopt, and inventing one
  would silently change the model. So unlike `_safe_ratio` the guard protects
  AD only, and a model using `log` must keep its argument positive
  (`clip_negative_states`, or write `log(1 + [X]/K)`). `check_units` requires a
  dimensionless argument.
- `SqrtNode(x)` — `sqrt(x)`. Guarded by `safe_sqrt`: the value is exact
  (`sqrt(0) == 0`) and the derivative at and below zero is 0 rather than `inf`.
  Same reasoning as the `powc` kernel — half-order kinetics sit at `x == 0`
  whenever the species depletes, which is the common case, not a corner one.
  Dimensionally it halves the argument's units.
- `MinNode(a, b)` — `min(a, b)`, the ceiling counterpart of `MaxNode`; a
  two-sided clip is `min(hi, max(lo, x))`.

`safe_log` / `safe_sqrt` live in **`core/kernels.py`**, not in `nodes.py`:
`sqrt` and `log` are offered by more than one of aquakin's expression
languages, and a function whose derivative is non-finite somewhere needs the
same guard wherever it is evaluated — otherwise the same spelling means two
different things depending on which language it was written in. Put a guarded
numeric there when a second language will evaluate it; `_safe_ratio` stays in
`nodes.py` because the Monod family and `safe_div` exist only in the rate
language.

All three use the same double-where shape. What each returns *off* the domain
differs deliberately and is a modelling decision per function, not a mechanical
one: `_safe_ratio` and `safe_sqrt` return the physical limit 0, while
`safe_log` keeps `-inf`, since a logarithm has no finite limit at zero to
adopt and inventing one would silently change the model. So `log`'s guard
protects the gradient only, and its caller still has to keep the argument
positive.

`safe_pow` guards the **`**` operator** for the same reason, since `[X] ** 0.5`
is the other spelling of `sqrt([X])` and the two must agree: `d/dx x**e` is
infinite at zero whenever `e < 1`. It fires only at an exactly-zero base with
an exponent below one, computes that branch under `stop_gradient` so the value
is exactly what plain `**` gave (signed zeros and all), and therefore changes
**no** value in any shipped model — verified across all 40 at four states.
`PowerNode.op` is shared with the vectorized kernel, so the scalar and batched
paths stay bit-identical by construction. Note the `powc` kernel in
`core/vector_kernel.py` does *not* cover this: it keeps a constant exponent
static to prune the generic `pow` JVP's `log(base)` term, which is a different
NaN (#703).

The second consumer is real: **`plant/time_expr.py` binds `sqrt` and `log` to
these**, not to `jnp`. A clock is the harshest case for them — every solve
starts at `t = 0`, exactly where their derivative is infinite — and the failure
is quiet, because the forward solve stays correct and only the gradient breaks,
surfacing as an exhausted solver step budget that points at the integrator.
`core/lexer.py`, `core/precedence.py` and `core/kernels.py` are now the three
things the expression languages share; a fourth divergence between them is a
bug, not a difference of taste.

The four Monod nodes and `SafeDivideNode` all evaluate their `num/denom` through
`_safe_ratio` (`core/nodes.py`), a **double-where** guard returning 0 (with a
finite gradient) where the denominator is exactly zero — for the Monod nodes the
full-depletion point where the limiting quantity *and* its saturation constant
are both 0 (`K = X = 0`); for `safe_div` wherever the author's denominator hits
0. The physical limit is 0 (no substrate → no rate); a bare `num/denom` is
`0/0 = NaN` there and the naive single `where` still back-propagates a NaN
through the masked branch, so the denominator is guarded too. Identity for any
nonzero denominator — the only change is exactly at the singularity.

### One tokenizer, three languages (`core/lexer.py`)

aquakin reads **three** small expression languages — rate expressions
(`core/parser.py`), expressions of time (`plant/time_expr.py`) and measure
expressions (`plant/measures.py`); control laws (`plant/control_law.py`) are the time
language's parser (`ClosureParser`) with other leaves. Their grammars and leaf rules differ on
purpose; **how text becomes tokens does not**. `core/lexer.py` holds the only
tokenizer: `tokenize(text, error=, offset=)` → `Token(kind, value, pos)`, with
`SYMBOLS` for the symbol kinds and `spell(kind)` for how a kind is written in a
message (`RPAREN` → `')'`, not the internal name).

**A new expression language reuses it.** Writing another tokenizer is the
mistake this file exists to stop: there were three near-identical copies until
#699, and the same subtle rule — an exponent is taken **only when a digit
follows** it, so `1e` is `1` then the name `e` rather than a `float()` crash —
was written and separately tested three times. A precedence bug then had to be
fixed in two of them, by hand, from memory.

Each language keeps its own error class (a message should name the language the
reader is writing in), which is why `tokenize` takes the class to raise rather
than owning one.
**A parse error carries its position**: raise it through `located(error,
message, position)`, as the tokenizer, `PrecedenceParser` and the rate parser
do, so a front end marks `exc.position` (the editor's caret) without reading it
out of the message.

### One precedence, three languages (`core/precedence.py`)

The same applies a level up. `PrecedenceParser` holds the grammar all three
share — `expression := term (('+'|'-') term)*`, `term := unary (('*'|'/')
unary)*`, `unary := ('+'|'-') unary | power`, `power := primary ('**' unary)?`
— and a language subclasses it, supplies `primary()` for its own leaves, and
implements one hook per operation (`add`, `subtract`, `multiply`, `divide`,
`negate`, `power`), each given the operator token so it can name a position.

**A language that does not allow an operation raises in its hook**, rather than
having a grammar of its own: that is how a measure expression stays affine
(`multiply` refuses two measured operands, `power` refuses outright). The base's
`power` already refuses, so a new language is safe before it has thought about
powers.

A sign binds **looser** than a power (`-2**2` is `-(2**2)`), `**` is
right-associative and its exponent may carry a sign. All three had this
backwards until #699 and it had to be fixed in each by hand — which is the whole
reason it is written down once.

`arglist()` is here too, for the same reason: a call's arguments are read the
same way in every language that has calls, **including the trailing comma**
Python allows (`monod([SS], KS,)`). A function that is associative is named in
`ASSOCIATIVE_CALLS` and accepts more arguments than it declares, folding left
— `max(a, b, c)` is `max(max(a, b), c)` — so `max` and `min` behave as Python's
variadic builtins do. `accepts()` and `spell_arity()` keep the check and its
message in step; a language calls both rather than comparing lengths itself.

**The contract is that an expression means what a Python reader thinks it
means**, and every departure is deliberate and written down.
`tests/unit/test_python_consistency.py` is where that line is drawn: it hands
the same text to Python's own `eval` and to the time language and compares,
corpus by corpus, and lists separately the Python-legal text aquakin refuses
with the reason for each (no comparisons, no `if`/`else`, no `%` or `//`, and
`^` refused rather than quietly read as exclusive-or). The one place aquakin is
the *more* permissive is `007`, which Python 3 rejects outright. **Touch the
grammar, add to that corpus** — it is cheaper than discovering a divergence the
way `-2**2` was discovered.

**Changing either shared file is a `full-ci` change.** Guard it the way those
two were: re-parse every expression in every shipped model, fixture and example
and compare the ASTs (845 of them; the throwaway script is in the #699 / lexer
PR description). A refactor that changes no AST changes no model.

### Parser

Recursive descent parser. No external parser dependencies. Operator precedence
(lowest to highest): `+/-`, `*/`, unary, `**`, primary — **Python's**, so a sign
binds *looser* than a power (`-2**2` is `-(2**2)`) and an exponent may carry its
own sign (`2**-1`). It bound a sign tighter until #699; that reading surprised
everyone who met it, and no shipped expression relied on it (all 845 parse
identically). `aquakin/plant/time_expr.py` shares the precedence and must stay
in step. Species names are
delimited by square brackets: `[O3]`. Rate constant names are namespaced
automatically: `k1` in reaction `O3_Br_direct` becomes `O3_Br_direct.k1`.

### CompileContext

Carries index maps used during AST compilation:

```python
@dataclass(frozen=True)
class CompileContext:
    species_index: dict[str, int]
    param_index: dict[str, int]      # namespaced keys -> flat vector indices
    condition_fields: frozenset[str] # valid condition field names
    reaction_name: str = ""          # owner reaction, for param namespacing
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
named fields overridden, scalars broadcast to the object's location
count, the rest carried over, the original untouched. It must refuse a name
the object does not carry (`ValueError` listing the declared fields), because a
misspelt name would add a field no rate reads; a new field is built with
`uniform`. The recommended
edit-from-defaults pattern is `model.default_conditions().with_(T=283.15)`
(start from the YAML-declared defaults, change only what differs). It always
returns the base `SpatialConditions` type.

---
