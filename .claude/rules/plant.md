---
paths:
  - "aquakin/plant/**"
---

# Rules — `aquakin/plant/`

Plant-wide flowsheet simulation: units, recycle resolution, BSM1/BSM2/A2O
builders, control, dosing, design, evaluation. Loaded automatically when
editing files under `aquakin/plant/`.

> The **dynamic-solve performance** log (the stiffness-bound regime and its
> levers) is in `docs/plant_performance.md`; plant measurements moved out of
> this rule are in `.claude/notes/plant-log.md` (both read on demand).

**Topic rules split out of this file** (each auto-loads for its own files; read
it on demand when a change elsewhere touches the topic):

| Rule | Loads for | Covers |
|---|---|---|
| `plant-solve.md` | `plant.py`, `dynamic_solve.py`, `steady_solve.py`, `recycle.py`, `colored.py`, `solve_cache.py`, `steady.py` | the coloured Jacobian, the parameter-slice and compiled-solve caches, recycle resolution and the cached recycle / flow maps |
| `plant-analysis.md` | `calibrate.py`, `identifiability.py`, `measures.py`, `sensitivity.py` | `Plant.calibrate`, Fisher / profile / sensor ranking, measured channels and measure expressions |
| `plant-audits.md` | `units_check.py`, `balance.py`, `metrics.py` | `plant.check_units()` and `plant.mass_balance()` |
| `plant-ifas.md` | `ifas.py`, `integrate/biofilm*.py` | the IFAS / MBBR biofilm unit |
| `plant-settling.md` | `takacs.py`, `settling.py`, the clarifier / separator modules | the Takács clarifier (soluble holdup, composition mode, interface flux), shared separator helpers |
| `plant-spec.md` | `spec.py`, `catalog.py`, `time_expr.py`, `influent.py`, `aquakin/__main__.py` | time expressions, influent records / patterns, the unit catalogue, the flowsheet spec |

## Private names shared across modules

Each list is a contract, so a rename shows every module it touches. Add a name
with its first use and remove it with its last.

- **Take every private name of `aquakin.integrate` from `integrate/seam.py`**,
  never its defining module, even inside a function body
  (`tests/unit/test_integrate_seam.py`). Public names come from their own modules.
- **List each private `Plant` member another module reaches (`plant._x`,
  `self.plant._x`) under that module in `plant/internals.py`**
  (`tests/unit/test_plant_internals.py`); bound modules' `self._x` are exempt.

## Plant-Wide Simulation

The plant concatenates every unit's state into one flat vector and integrates
it in one monolithic Diffrax solve, so `jax.grad` flows end to end and
`aquakin.calibrate()` works on the plant's flat parameter vector.

**By-name plant parameters.** `Plant.parameter_values`, `parameter_names` and
`parameter_index` key the flat parameter vector by `"<model>.<param>"` and all
read the one `model_param_blocks` layout. Use `parameter_index` (not
`parameter_values`, which materializes concrete values) to differentiate with
respect to one parameter; do not hand-compute block offsets.

### The `Unit` contract (`plant/units.py`)

The `Unit` Protocol's docstring is the contract (methods, state names, the
optional hooks); these are the invariants around it.

- **Every unit has the same fixed signature for each method.** The plant never
  branches its call on a per-unit capability flag: `signals` reaches every
  `rhs` and `compute_outputs`, a `FlowContext` reaches every `flow_outputs`, and
  a unit ignores what it does not use.
- **A new optional capability is a named `runtime_checkable` Protocol in
  `plant/units.py`, detected with `isinstance`** (as `SignalProducer`,
  `LiquidVolumeUnit`, `TemperatureSettable`, `PlantTimeUnitAware`, ...), and is
  listed in the `Unit` docstring. Only hooks read as values with a default
  (`signal_names`, `required_signals`, `flow_param_defaults`, a translator's
  `needs_src_pH` / `needs_dest_pH`) stay plain look-ups.
- **A stateless unit inherits `StatelessUnit`** rather than writing the
  zero-size state members again.
- **`state_names()` is unique within the unit, and stored results key states by
  it**, never by position; a heat-balance temperature is the unit's `"T"` entry.

### Where `Plant` code goes

- **Do not grow `plant.py`.** The dynamic solve is in `plant/dynamic_solve.py`,
  the steady state in `plant/steady_solve.py`, the sensitivity / UQ surface in
  `plant/sensitivity.py`: free functions taking the plant, bound onto `Plant` as
  methods. New solve, steady-state or analysis code goes in those modules (or a
  new one bound the same way). `plant/sensitivity.py` only *consumes* the public
  solve API.
- **Bound functions keep their first argument named `self`**:
  `inspect.signature(Plant.solve)` is a pinned public contract
  (`tests/unit/test_api_signatures.py`).
- **The bound modules import nothing from `plant.py`** (it imports them): a
  helper both need lives in the lower module and `plant.py` imports it; a name
  that is an import path of `aquakin.plant.plant` stays re-exported from it.
- Calibration, identifiability and measured channels: `plant-analysis.md`. The
  coloured Jacobian, the caches and recycle resolution: `plant-solve.md`. The
  unit audit and the mass balance: `plant-audits.md`.

### Wiring, topology and errors

- **Wire with `plant.connect(source, dest)` on `"unit.port"` endpoints** (the
  port may be omitted for a unit with one port in that role); all endpoint
  parsing goes through `Plant._parse_endpoint`. Influents enter only through
  `plant.add_influent(..., to=...)`, never as a `connect` source. Across models
  that are not the same or a pure unit rescaling (`UnitScalingTranslator`),
  `connect` requires an explicit `translator=`.
- **Raise the `plant/errors.py` taxonomy for a name or wiring guard**, not a bare
  `KeyError` / `ValueError`: `UnknownUnitError` / `UnknownPortError` for a name
  that does not exist, `WiringError` for an invalid request, `NoDigesterError`
  for a digester read on a plant without one. Each subclasses the built-in it
  replaced, so existing handlers keep working; detect the no-digester case by
  catching `NoDigesterError`, not `ValueError`.
- **Add order is free; recycles are detected, not declared.**
  `Plant._finalize_topology` (run at every solve) topologically sorts the
  connections into `_unit_order` and cuts the back-edges as recycles,
  deterministically (insertion-order tie-break), so the state and parameter
  layouts are stable and a plant added in a valid order keeps that order and
  recycle set exactly. `_insertion_order` (the raw add order) fixes the
  parameter-block order; do not confuse it with `_unit_order`. How the recycles
  are then resolved is in `plant-solve.md`.
- `plant.check()` is the pre-solve wiring check: an unfed input port is an
  error, an unconsumed output is information (a terminal stream legitimately
  leaves the plant).

### The plant clock (`Plant(name, time_unit=...)`)

- **Every user-supplied quantity with a time in it is on the plant's clock**
  (`t_span`, flows, `kla`, `t_m`, SBR phases, PI integral times) and is never
  converted. What *is* converted happens in `Plant._resolve_time_unit`: each
  `PlantTimeUnitAware` unit is told the unit, and influents are re-expressed.
- **Kinetic units scale the model's rates through `ModelTimeScaling`**; a factor
  of exactly 1 must return the term untouched (bit-identical), and
  `balance._reaction_term` must apply the same factor the unit does.
- **A new unit with a built-in time constant implements `set_time_unit`**
  (converting its constant, as UV, the primary clarifier and Takács do), not
  `assumed_units`.
- A missing `time_unit` is deprecated (inferred with a `FutureWarning`). The
  builders (`build_bsm1`, `build_bsm2`, `build_a2o`) pass `"d"`.

### Tracing, temperature and warm starts

- **Everything `solve` calls must trace**: `monte_carlo`, `compare_scenarios` and
  `dgsm` jit the user's function. No `float()`, `np.asarray` or Python `if` on a
  `jnp`-built value in the solve path (`initial_state`, a `TemperatureModel`'s
  methods); Python scalars only for static layout data.
- `plant.set_temperature` changes baked-in conditions, so it invalidates
  `_solve_cache`; it leaves the heated digester alone (see *Kinetic units*).
- **Seed a plant through `plant.initial_state(overrides=...)` or the shipped
  `bsm1_warm_start` / `bsm2_warm_start`**, never by indexing the private
  `_state_layout`. BSM2 should always be warm-started: the digester's long
  retention makes a cold start slow and stiff (it can hit the step ceiling).

### Reading results back

- **Discover names with the introspection API** (`plant.list_units()`,
  `list_ports()`, `list_species()`, `list_streams()`), which works before a
  solve and gives a "did you mean?" `KeyError`, rather than reading the builder
  source.
- **Species-indexed access is limited to units whose state leads with a
  concentration vector**, decided in one place, `Plant._is_concentration_unit`
  (state size equal to the model's species count, or a `LiquidVolumeUnit`).
  Other units (stateless units, the layered Takács settler) are read as streams,
  with an error that says so, never an `IndexError`.
- **`plant.activated_sludge_reactors()` is the one definition of "the AS
  reactors"** behind the warm-start, design-sizing and evaluation heuristics;
  do not write another.
- **Streams are recomputed from saved states, never stored in the solution.**
  `plant.stream`, `plant.outputs_at`, `plant.derivative` and the RHS all go
  through the one `_resolve_streams`, so a reconstruction matches the integrated
  wiring exactly. The full output sweep is reconstructed once, vectorized over
  the saved times, and cached on the solution (`Plant._cached_streams`), so
  reading several streams costs one reconstruction, not a per-stream or
  per-time loop.
- **Builders register engineering names for their streams**
  (`plant.register_stream`), and `effluent` must follow the builder's
  option-dependent `effluent_endpoint`. A derived output that is not a material
  port (the digester biogas, `plant.digester_gas`) gets its own accessor rather
  than a fake port.

### Shipped units

The unit classes are exported from `aquakin.plant` (their docstrings describe
them); the topic rules above cover IFAS / MBBR and the settlers. Two defaults
are deliberate:

- **`PrimaryClarifier` defaults to `t_m = 0`** (no inflow filter), because a
  standalone unit has no design inflow to seed the filter with; `build_bsm2`
  sets the benchmark `t_m` and `initial_flow`.
- `IdealClarifier` is the fast stateless separator; `TakacsClarifier` the
  layered settler (`plant-settling.md`).

**Parsing anything here goes through `core/lexer.py`.** `plant/time_expr.py` and
`plant/measures.py` are expression languages whose grammars are their own but
whose *tokenizer* is not: both call `aquakin.core.lexer.tokenize`. Do not write
another one — see the "One tokenizer, three languages" note in
`.claude/rules/core.md`, which `aquakin/plant/**` does not auto-load.

**Units in the help.** A setting's numpydoc entry names its unit in
parentheses, flows and times per day as aquakin runs them on the plant's clock
(`Volume (m³).`, `Fixed dose flow (m³/d).`, `Integral time (d).`); the editor
shows that description beside the field. Give each setting its own entry (a
shared `Kp, Ti, Tt : float` cannot name one unit each).

**Kinetic units share one reaction term (`plant/kinetic.py`).** Every unit that
evaluates a model in its `rhs` inherits `KineticUnit` and builds its reaction
term with `_reaction` (the model's `dCdt`, so the clip and positivity limiter
apply, scaled to the plant clock) and its conditions with `_init_conditions` /
`_operating_conditions`. Never write `stoich.T @ rates` in a unit or in
`balance.py`: `_reaction_term` calls `unit._reaction` so the balance integrates
what the unit integrates. A unit needing extra parameter handling overrides
`_reaction` (the digester slaves `V_liq`). A unit whose temperature an operator
sets inherits `AdjustableTemperatureUnit`; the heated digester does not, so
`Plant.set_temperature` leaves it alone. Deliberate exceptions, stated in the
unit: the SBR runs at its static `T`, and IFAS evaluates its reaction per
compartment inside the biofilm RHS (which also calls `model.dCdt`).

**Sizes are checked at construction.** Every unit with a volume calls
`_constants.check_positive(name, "volume", value)` first in `__post_init__`
(a zero, negative or infinite volume makes a steady-state solve never end). A
new unit with a volume, area or other size does the same.

**Co-stripped gases (`Aeration(stripped=...)`).** A gas stripped with oxygen
is folded into the same per-species aeration vectors as oxygen by
`build_aeration_vectors` (open loop: kLa times ratio; closed loop: the same
signal at gain times ratio), so `aeration_transfer` has no gas branch and every
`AerationUnit` (CSTR, IFAS, MBR) gets it. The `C_s(T)` saturation correction
applies to the oxygen entry only; `required_signals` must de-duplicate, since
the gas and oxygen read one signal.

- **Read a tank's aeration through the `AerationUnit` readers** (`oxygen_kla`,
  `kla_vector`, `saturation_vector`, `stripped_gases`), never by a species name:
  models name oxygen differently (`SO` in ASM1, `SO2` in ASM3-based models).
  Closed-loop entries go through the one `_apply_controlled_kla` the RHS uses.
- In a closed-loop plant whose only effluent is a sensed tank, the DO sensor
  tap consumes `tank.out`, so it is not a dangling output: pass it to
  `plant.mass_balance` as `effluent_ports`.

**BSM1 with non-ASM1 models.** `build_bsm1(model, aeration=, settling_species=,
tss_species=)` takes its clarifier solids from `_constants.solids_species`, which
must return the ASM1 constants unchanged for a model whose particulates are
exactly the ASM1 settling species (so ASM1 stays bit-identical). The BSM1
metrics (`derived_TSS`, `evaluate_bsm1`'s EQI) still use the ASM1 species and do
not yet support such models.

**Shared numeric constants (`plant/_constants.py`).** Use `EPS_Q` for every
flow-, volume- or mass-weighted division guard and `SECONDS_PER_DAY` /
`HOURS_PER_DAY` / `MINUTES_PER_DAY` for time conversions, never an inline
literal, so each magnitude is defined once. Do **not** fold a physically
meaningful floor into `EPS_Q` (a minimum settler depth in `settling.py`, the
minimum flow and `log` floor in `PrimaryClarifier._removal_fraction`, a
linearization-probe floor in `ifas` / `takacs`): each is a modelling choice
with its own magnitude.
