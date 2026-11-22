# Your first model

This page walks through writing a reaction model of your own, from an empty
file to a model you can solve, check and differentiate. The
[Model file format](model_format.md) is the reference for every key used here.

The example is chlorine decay in a water sample. Free chlorine (HOCl) is used up
in two ways: quickly by a small amount of reactive material R, and slowly by
everything else in the water, which is in such excess that it is lumped into a
first-order rate.

| Reaction | Rate |
|---|---|
| HOCl + R → products | k_fast [HOCl][R] |
| HOCl → products | k_bulk [HOCl] |

## 1. Write the model file

Save this as `chlorine_decay.yaml`:

```yaml
# chlorine_decay.yaml
model:
  name: chlorine_decay
  version: "1.0"
  description: "Free chlorine decay with a fast-reacting fraction and a slow bulk demand."
  references:
    - "Your citation for the rate constants here."

species:
  - name: HOCl
    description: "Free chlorine"
    units: mg/L
    default_concentration: 2.0
  - name: R_fast
    description: "Fast-reacting chlorine demand"
    units: mg/L
    default_concentration: 0.5

conditions:
  - name: T
    description: "Water temperature"
    units: K
    default: 293.15

parameters:
  k_fast:
    value: 0.5
    units: L/(mg*h)
    bounds: [0.01, 10.0]
    transform: positive_log
  k_bulk:
    value: 0.05
    units: 1/h
    bounds: [0.001, 1.0]
    transform: positive_log
    temperature: {theta: 1.05, ref_T: 293.15}

reactions:
  - name: fast_demand
    reference: "Your citation"
    rate: "k_fast * [HOCl] * [R_fast]"
    stoichiometry: {HOCl: -1, R_fast: -1}
  - name: bulk_decay
    reference: "Your citation"
    rate: "k_bulk * [HOCl]"
    stoichiometry: {HOCl: -1}
```

What each block does:

- `species:` lists what the model tracks. `default_concentration` is a
  convenient starting point, not a measurement; you can give other initial
  concentrations when you solve.
- `conditions:` lists the fields the rates may depend on that are not species,
  here the temperature.
- `parameters:` holds the rate constants. They are declared at the top level so
  that a reaction refers to them by name, and calibration has one value to
  adjust. `bounds` and `transform` matter only when you calibrate or analyze
  the model: `positive_log` keeps a rate constant positive and lets it vary
  over orders of magnitude. The `temperature:` block makes `k_bulk` increase by
  5 % per degree above 20 °C (293.15 K).
- `reactions:` gives each reaction a rate, written in the
  [rate-expression language](model_format.md#rate-expression-grammar) with
  species in `[...]`, and a stoichiometry saying how much of each species one
  unit of rate consumes (negative) or produces (positive).

## 2. Load it and read it back

```python
import aquakin

model = aquakin.load_model_from_file("chlorine_decay.yaml")
print(model.summary())
```

The summary lists the species, conditions, reactions and parameters as aquakin
read them, and a line you should always check:

```text
  Time unit: h (t_span / t_eval are in hours)
```

aquakin has no fixed time unit. It reads the unit from your rate constants
(`1/h` here), and every time you pass to a solve is in that unit. If the line
says the time unit cannot be determined, a rate constant is missing its
`units:` or two of them use different time units.

## 3. Check it

```python
print(model.check_units())   # [] when every rate is dimensionally consistent
```

`check_units` reads the `units:` strings and checks each rate: that it is a
concentration per time, that terms added together have the same units, and
that a rate matches the species it changes one for one. An empty list means
no problems were found. It can only check what has units, so give every
species and parameter one.

If your species carry conserved quantities, such as COD, nitrogen or
phosphorus, declare them with `composition:` and
`model.check_conservation()` will report any reaction that does not balance
(see [Conservation](model_format.md#conservation-composition-and-auto-coefficients)).

## 4. Solve it

```python
reactor = aquakin.BatchReactor(model, model.default_conditions())
sol = reactor.solve(
    model.default_concentrations(),
    t_span=(0.0, 24.0),                # hours, the model's time unit
    params=model.default_parameters(),
)
print(sol.final_named(["HOCl"]))       # about 0.44 mg/L after 24 h
```

To solve at another temperature, change only that condition:

```python
cold = aquakin.BatchReactor(model, model.default_conditions().with_(T=283.15))
```

## 5. Differentiate it

Every solve can be differentiated with JAX, which is what sensitivity analysis
and calibration build on:

```python
import jax

def chlorine_after_24_h(params):
    sol = reactor.solve(model.default_concentrations(), t_span=(0.0, 24.0), params=params)
    return sol.C_named("HOCl")[-1]

print(model.parameters)                            # the order of the parameter vector
print(jax.grad(chlorine_after_24_h)(model.default_parameters()))
```

Both entries of the gradient are finite and the second, for `k_bulk`, is large
and negative: a faster bulk decay leaves less chlorine. The
[Sensitivity & calibration](sensitivity_and_calibration.md) guide goes on from
here.

## Common mistakes

- **Mixed time units.** A rate constant in `1/d` beside one in `1/h` makes the
  time unit undetermined and the rates wrong by a factor of 24. `check_units`
  reports it.
- **`^` for a power.** Write `[X] ** 2`; `^` is refused.
- **Unquoted species names YAML misreads.** A species named `NO` or `ON` must be
  quoted (`"NO"`), or YAML reads it as `false`.
- **The same parameter name at both levels.** A name declared in the top-level
  `parameters:` cannot also be declared in a reaction's own `parameters:`.
- **`log` of something that can reach zero.** Write `log(1 + [X]/K)`, or keep
  the argument positive some other way.

## Building on a shipped model

To change a model aquakin ships, such as one rate constant or one rate law,
inherit it with `extends:` instead of copying it; see
[Inheriting a model](model_format.md#inheriting-a-model-extends). The
[Model catalogue](model_catalog.md) lists what is available.

## Contributing a model to aquakin

A model that others would use can be added to the shipped set:

1. Put the YAML file in `aquakin/models/`. It then loads by name with
   `aquakin.load_model("<name>")`.
2. Cite the literature in `references:` and on each reaction's `reference:`.
3. Make sure `model.check_units()` returns no findings.
4. Add tests of its structure (species, key stoichiometry) and, where published
   data exist, a validation test against that data.
5. Describe it in the [Model catalogue](model_catalog.md) and the README.
