---
paths:
  - "aquakin/plant/**"
---

# Rules — `aquakin/plant/`

Plant-wide flowsheet simulation: units, recycle resolution, BSM1/BSM2/A2O
builders, control, dosing, design, evaluation. Loaded automatically when
editing files under `aquakin/plant/`.

> The long **dynamic-solve performance** log (the stiffness-bound regime and its
> levers — colored Jacobian, cached recycle/flow maps, forward_fast, PTC steady
> state, etc.) lives in `docs/plant_performance.md` (read on demand).

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

## Plant-Wide Simulation

`aquakin.plant` composes kinetic reactors with non-reactive unit ops into
a full plant flowsheet. The plant assembles each unit's internal state
into one flat vector and integrates the whole thing under a monolithic
Diffrax solve — so `jax.grad` flows end-to-end across the plant, and
`aquakin.calibrate()` works on plant-level parameter vectors.

**By-name plant parameters.** A `Plant` concatenates its models' parameter
vectors into one flat `default_parameters()`. `Plant.parameter_values(overrides)`
gives that flat vector the same friendly by-name API as
`CompiledModel.parameter_values`, keyed by `"<model>.<param>"` (the model
name plus the model's own namespaced parameter name) — e.g.
`plant.parameter_values({"asm1.muH": 4.0, "adm1.k_hyd_ch": 10.0})` to bump one
rate in a multi-model plant (BSM2's ASM1 water line + ADM1 digester) without
hand-computing the block offset. `parameter_names()` lists the valid keys;
`parameter_index(name)` returns the flat index (the companion for `jax.grad`
w.r.t. one parameter, which can't go through `parameter_values` — that
materialises concrete values). All three reuse the existing
`model_param_blocks` layout. Unknown names raise a `KeyError` with a
close-match hint.

Key types:

- `Stream(Q, C, model)` — the bulk-flow + concentration record passed
  between units.
- `Unit` Protocol — every unit declares `state_size`, `input_ports`,
  `output_ports`, and implements `initial_state()`, `state_names()` (a unique
  name per state entry: species for concentrations, `V` / `R_f` / `integral` /
  `Q_filtered` / `residual` for scalars, `layer<k>.<x>` and `film<k>.<species>`
  for repeated blocks; `Plant.state_names` / `named_state` / `state_from_named`
  build on it, with a heat-balance temperature as the unit's `"T"` entry, and
  stored results key states by these names), `compute_outputs()`,
  `rhs(t, state, inputs, params, signals)`, and `flow_outputs(input_flows,
  params, ctx)`. **Every unit has the *same fixed signature* for each method**
  — the plant never branches its call on a per-unit capability flag. The
  control-signal bus is threaded into every `rhs` as `signals` (an uncontrolled
  unit ignores it), and `flow_outputs` always receives a `FlowContext` carrying
  the unit's own state and the time (a fixed-split unit ignores it). Beyond the
  core contract a unit may implement one or more **optional capability hooks**,
  each with a named `runtime_checkable` Protocol in `plant/units.py` so the plant
  detects it with `isinstance` (not a stringly-typed `hasattr`): `SignalProducer`
  (`signal_outputs`, e.g. `PIController`), `PHOperating` (`operating_pH`),
  `LiquidVolumeUnit` (`liquid_volume`), `ComponentInventoryUnit`
  (`component_inventory`), `CycleEventSource` (`cycle_events`, the SBR), and
  `TemperatureSettable` (`set_temperature`); the `Unit` docstring enumerates them.
  (A few hooks read as *values with a default* — `signal_names` /
  `required_signals` / `flow_param_defaults`, and the translator's
  `needs_src_pH` / `needs_dest_pH` — stay plain look-ups.) A **stateless** unit
  (`state_size == 0`: mixers, splitters, ideal separators) inherits the
  `StatelessUnit` mixin (`plant/units.py`), which supplies the three trivial
  state members (`state_size → 0`, empty `initial_state`, no-op `rhs`), so it
  only writes `compute_outputs` / `flow_outputs`. It is a plain mixin, not part
  of the Protocol, so it composes with the `@dataclass` units.
- `StateTranslator` Protocol — converts streams between models.
  `IdentityTranslator` covers single-model plants (BSM1).
- `Plant` — assembles units and connections, drives the monolithic
  integration. Its **sensitivity / uncertainty-quantification surface** —
  `steady_state_sensitivity` / `steady_state_dgsm` / `solve_sensitivity` /
  `dynamic_sensitivity` / `dynamic_dgsm` and their result dataclasses — lives in
  [`plant/sensitivity.py`](aquakin/plant/sensitivity.py) as free functions taking
  the plant, bound onto `Plant` as methods (the public `plant.steady_state_dgsm(...)`
  API is unchanged); that layer only *consumes* the public solve API.
  **`Plant` is split the same way**: the dynamic solve (`solve`, its four
  paths, the jit builders) is in `plant/dynamic_solve.py` and the steady state
  (`steady_state`, its fallback ladder, `run_to_steady_state`) in
  `plant/steady_solve.py`, bound onto the class as methods. New solve or
  steady-state code goes in those modules (or a new one bound the same way), not
  back into `plant.py`. Their first argument stays named `self`:
  `inspect.signature(Plant.solve)` is a pinned public contract
  (`test_api_signatures.py`) and lists it. The modules import nothing from
  `plant.py` (it imports them), so a helper both need lives in the lower module
  and `plant.py` imports it from there; a name that is an import path of
  `aquakin.plant.plant` stays re-exported from it.
  Calibration, identifiability and measured channels are in
  `.claude/rules/plant-analysis.md`; the coloured Jacobian, the caches and recycle
  resolution in `.claude/rules/plant-solve.md`; the unit audit and the mass
  balance in `.claude/rules/plant-audits.md`.
  - **Wiring API.** `plant.connect(source, dest)` takes two `"unit.port"`
    endpoint strings, read as `source -> dest`. The port may be omitted
    (bare `"unit"`) when the unit has exactly one port for that role — a
    single output (source) or single input (dest) — so only multi-port
    units (mixers/splitters/clarifiers) name a port.
    External influents are wired through
    `plant.add_influent(name, series, to="unit.port")` — they are *not*
    valid `connect` sources (a clear error redirects you). `connect`
    resolves the default `IdentityTranslator` when the two ends share a
    model, a `UnitScalingTranslator` when their models list the same species
    in different units, and requires an explicit `translator=` across any
    other models (e.g. the BSM2 ASM1↔ADM1 digester edges). The endpoint parsing lives in
    `Plant._parse_endpoint`.
  - **Exception taxonomy (`plant/errors.py`).** Plant name/wiring errors use a
    uniform, exported taxonomy so a caller can tell an unknown *name* from an
    invalid *wiring*: `UnknownUnitError` / `UnknownPortError` (both subclass
    `KeyError`) for a unit / port that does not exist, `WiringError` (subclasses
    `ValueError`) for a structurally invalid or unsupported request (an influent
    used as a `connect` source, a bare endpoint that omits a required port, a
    unit that does not support `set_temperature`), and `NoDigesterError`
    (subclasses `ValueError`) for `digester_gas` on a plant with no ADM1
    digester. Each subclasses the built-in it historically raised, so existing
    `except KeyError` / `except ValueError` handlers and message-matching tests
    are unaffected. Use these instead of a bare `KeyError`/`ValueError` when
    adding a name/wiring guard, and catch `NoDigesterError` (not bare
    `ValueError`) when detecting the no-digester case.
  - **Arbitrary add order; topological sort.** Units may be `add_unit`-ed in
    **any order**: `Plant._finalize_topology` (run from `_build_state_layout`,
    so at every solve) topologically sorts the feed-forward connection graph
    into the RHS evaluation order `_unit_order`, and the **recycles are the graph
    back-edges, detected automatically** — you no longer add the downstream unit
    before its upstream consumer or mark recycles by ordering. The sort is Kahn's
    algorithm with an insertion-order tie-break (deterministic, so the
    state/parameter layouts are stable): a connection carrying an explicit
    `initial_value` is a declared recycle and cut first (its seed rides the cut
    edge); any remaining cycle is broken by cutting the earliest-added remaining
    unit's still-active incoming edges, which become auto-detected recycles
    (zero-flow seeded via `_recycle_seeds`). For a plant already added in a valid
    order it **reproduces that order and recycle set exactly** (BSM1/BSM2
    unchanged); any valid feedback-arc cut gives the same converged solve, so the
    result is add-order-independent. `add_unit` records the raw add order in
    `_insertion_order` (the parameter-block order and the tie-break read it);
    `_unit_order` is the computed eval order. Recycles are then resolved by
    iterating the per-RHS stream computation 3 times (sufficient for typical BSM
    topologies). `initial_value=` on `connect` overrides a recycle's zero-flow
    seed with a non-zero warm start (e.g. the BSM2 temperature-carrying seed).
  - **Pre-solve wiring check.** `plant.check()` → `PlantCheck` reports **unfed
    input ports** (`.unfed_ports`, an error — the RHS sweep has no source for
    them) and **unconsumed outputs** (`.dangling_outputs`, info — a terminal
    stream like the final effluent / wasted sludge / disposal cake / biogas
    legitimately leaves the plant), plus the detected `.recycles`; `.ok` is true
    when nothing is unfed and `.summary()` prints it. `check(raise_on_error=True)`
    raises on an unfed port. Exported as `aquakin.PlantCheck`.
  - **Plant clock — `Plant(name, time_unit=...)`.** The declared time unit
    is the plant's clock (`t_span`, `m3/<unit>` flows, every user-supplied unit
    parameter with a time in it: `kla`, `t_m`, SBR phases, PI integral times).
    `_build_state_layout` calls `_resolve_time_unit` (keyed on the declared
    unit + model ids + insertion order + influent ids, so it reruns only when
    those change, and then invalidates `_solve_cache`), which calls
    `set_time_unit(tu)` on every unit implementing the
    `units.PlantTimeUnitAware` hook and replaces each `influents[name]` with
    `_influents_added[name].in_time_unit(tu)`. Kinetic units (CSTR, MBR, SBR,
    IFAS, ADM1 digester) inherit the `units.ModelTimeScaling` mixin:
    `_rate_factor = model_rate_factor(model, tu)` (`conversion_factor("1/<model
    tu>", "1/<tu>")`) multiplies everything the *model* computes per its own
    time unit (`_scale_rates`: the reaction term; IFAS's whole biofilm
    reaction + diffusion term; the digester's dCdt incl. gas transfer/outflow),
    and `balance._reaction_term` applies the same factor. A factor of exactly
    1 returns the term untouched, so a plant on its models' clock is
    bit-identical. UV (`_seconds_per_time`), the primary clarifier
    (`_minutes_per_time`) and Takács (`_v0`/`_vmax`, only while the BSM default
    `settling_params`) convert their built-in day constants in
    `set_time_unit`; with `tu=None` they keep days. **A missing `time_unit` is
    deprecated**: the plant infers the models' shared unit with a one-time
    `FutureWarning`, and `time_unit_errors` raises `UnitsError` at solve only
    when the models disagree. The builders (`build_bsm1/2`, `build_a2o`) pass
    `"d"`. A new unit with a built-in time constant implements
    `set_time_unit`, not `assumed_units`.
  - **Operating temperature.** `plant.set_temperature(celsius)` sets the static
    `T` condition of every temperature-bearing reactor in one call (°C → K),
    leaving a heated fixed-`T` unit like the digester untouched; clears the
    compiled-solve cache and returns `self`. See the seasonal-temperature notes
    below.
  - **Warm-starting.** `plant.initial_state(overrides={"tank1": vec, ...})`
    builds the flat initial-state vector with selected units' states replaced
    by name (each vector must match the unit's `state_size`) — the supported
    way to seed a plant (e.g. a healthy activated-sludge biomass before a slow
    digester settle) instead of reaching into the private `_state_layout`. Pass
    the result as `solve(y0=...)`. For the BSM plants the reference seed is
    shipped — **`bsm2_warm_start(plant)`** / **`bsm1_warm_start(plant)`**
    (`aquakin.plant.bsm`) return a ready flat `y0` with the five AS reactors
    seeded from the reference reactor composition (the dict constants
    `BSM2_WARM_REACTOR_COMPOSITION` / `BSM1_WARM_REACTOR_COMPOSITION`) and every
    other unit at its default. **BSM2 should always be warm-started**: the
    digester's ~19-day retention makes a cold start slow and stiff (the
    near-empty AS basin filling against the recycle loops can crawl or hit the
    step ceiling), and the warm seed removes that transient so only the digester
    has to settle. The reactor set and water-line model are auto-detected from
    the plant, so a single `bsm2_warm_start(plant)` replaces the
    seed-composition dict + tank list + `initial_state(overrides=…)` boilerplate
    the BSM2 scripts used to copy-paste. (The BSM2 composition is the validated
    reference reactor state; the BSM1 one is ~aquakin's BSM1 steady state. Both
    are *seeds* — the solve relaxes them — so the values affect settling speed,
    not the steady state.)
  - **Introspection — discover names instead of reading the builder source.**
    `plant.list_units()` lists the unit names (in add order); `plant.list_ports()`
    lists every `"unit.port"` **output** endpoint — the exact strings
    `plant.stream(sol, …)` accepts (pass `unit=` to scope, `role="input"` for the
    `connect`-destination endpoints); `plant.list_species(unit)` lists a
    concentration-vector unit's species (the valid `C_named` / `to_dataframe`
    columns). All three work **before** solving (plant structure) and raise a
    `KeyError` with a `difflib` "did you mean?" hint for an unknown name.
    `list_species` / `C_named` are restricted to units whose *state leads with a
    concentration vector* via `Plant._is_concentration_unit`: `state_size ==
    model.n_species` (the CSTRs, the digester), or a `LiquidVolumeUnit` whose state
    is `[C…, scalar(s)]` (the filtered primary clarifier, storage, MBR, SBR — so
    free ICs and the colored pattern's source columns cover them too) — a stateless mixer/splitter/ideal-clarifier or the **layered Takács settler**
    (which carries a model but a non-species state) is rejected with a clear
    "read it as a stream with `plant.stream(...)`" message rather than an
    `IndexError`. `PlantSolution.available_streams()` is a convenience alias for
    `plant.list_ports()`, and `solution.C_named(unit, species)` now gives the same
    hinted errors (unknown unit, unknown species, non-concentration unit).
    `plant.activated_sludge_reactors(require_volume=True)` lists the AS reactor
    units (the CSTR/MBR `aeration`-carrying units, digester excluded; in plant
    order) — the single source of truth behind the warm-start / design-sizing /
    evaluation reactor heuristics (`require_volume=False` keeps every mechanically
    mixed reactor, e.g. for the mixing-energy term).
  - **Reading state back by unit.** `plant.states_by_unit(vec)` splits any flat
    plant vector into a `{unit_name: sub-vector}` map — the exact inverse of
    `initial_state(overrides=...)`. It works on a `y0`, a `PlantSolution.final_state`
    (the last save row, shape `(total_state_size,)`, so no opaque `[-1]` on the
    2-D `state` trajectory), or a `derivative` result. For a *trajectory* of one
    unit, `PlantSolution.unit_state(name)` returns `(n_t, unit.state_size)`.
  - **Evaluating the RHS once.** `plant.derivative(state, params=None, *, t=0.0)`
    is the public single evaluation of the assembled flowsheet RHS (`dstate/dt`,
    recycles resolved) — for inspecting the dynamics without a full solve. Same
    layout as `state`; split it with `states_by_unit`. (Wraps the private `_rhs`,
    building the layouts internally.)
  - **Effluent reconstruction (streams are recomputed, not stored).** The plant
    integrates unit *states*, not the inter-unit streams, so a stream such as the
    secondary-clarifier effluent is **recomputed on demand** from the saved
    states — it is *not* in the solution. `plant.stream(solution,
    "clarifier.overflow")` (or the convenience `solution.stream("effluent")`,
    plant carried on the solution) returns a `StreamSeries` (`t`, `Q`, `C` shape
    `(n_t, n_species)`, `model`, with a `C_named(species)` accessor) — feed it
    straight to `effluent_averages`. **The whole output sweep (every `(unit,
    port)`) is reconstructed in one `jax.vmap` pass over the saved times and
    cached on the solution** (`Plant._cached_streams`, keyed by the parameter
    vector via `_concrete_teval_key`; skipped under tracing), so a sequence of
    `stream` calls for different ports — or `evaluate_bsm*` reading ~8 streams —
    costs one reconstruction, not one per stream. The reconstruction is
    **vectorised**: each saved time's `_resolve_streams` sweep (a recycle-flow +
    concentration solve) is batched by `vmap` into a single XLA program rather
    than a Python loop of per-step sweeps — turning a long dynamic run's
    evaluation from minutes into seconds (a 609-day hourly evaluation drops from
    ~20 min to a few seconds). `evaluate_bsm2`'s digester-feed-temperature and
    closed-loop kLa histories are vmapped the same way. `plant.outputs_at(t, state, params=None)`
    is the single-instant primitive (returns `{(unit, port): Stream}`,
    uncached); both reuse the same `_resolve_streams` helper the RHS uses, so the
    reconstruction matches the integrated wiring exactly (including resolved
    recycle flows).
  - **Semantic stream shortcuts.** `plant.stream(sol, …)` also accepts an
    engineering **name** instead of a `"unit.port"` — the builders register a
    `named_streams` map (`plant.register_stream(name, endpoint)`,
    `plant.list_streams()`) so `plant.stream(sol, "effluent")` reads the right
    port without the user knowing it is `"tank5_split.internal_recycle"`. BSM1
    registers `effluent`/`ras`/`wastage`/`internal_recycle`; BSM2 adds
    `primary_effluent`/`primary_sludge`/`thickener_overflow`/`reject`/
    `dewatering_reject`/`disposal_sludge`, with `effluent` tracking the
    option-dependent `effluent_endpoint`. A misspelled name gives a hinted error
    listing `list_streams()`. `plant.effluent_stream(sol)` is the first-class
    shortcut for the most-read one (reads `effluent_endpoint`). The digester
    **biogas** is a *derived* output (computed from the ADM1 headspace state, not
    a material port), so it has its own accessor: `plant.digester_gas(sol)` →
    `DigesterGas` (`t`, `Q` m³/d the biogas flow **normalized to atmospheric
    pressure**, `p_ch4`/`p_co2`/`p_h2` bar, `ch4` kg/d, and
    `.methane_production()` time-averaged kg CH₄/d), reusing the OCI biogas
    formula (`evaluate_bsm2`'s `_methane_production` now delegates to it). Raises
    if the plant has no ADM1 digester.

Shipped units: `CSTRUnit` (kinetics + aeration), `IFASUnit` / `MBBRUnit`
(an IFAS/MBBR tank: a CSTR bulk coupled to a depth-resolved attached biofilm —
see `.claude/rules/plant-ifas.md`), `MBRUnit` (membrane bioreactor: a high-MLSS aerated reactor whose
membrane retains the solids into a near-solids-free permeate, with fouling/TMP —
see `plant/mbr.py`), `MixerUnit`,
`RatioSplitter` / `SetpointSplitter` / `ThresholdSplitter`,
`IdealClarifier` (fast, stateless separator),
`PrimaryClarifier` (BSM2 Otterpohl–Freund: a well-mixed holding tank split by
an HRT-dependent particulate-removal efficiency, fixed underflow `f_PS·Q`; with
`t_m > 0` the HRT reads a first-order filtered inflow, one trailing state seeded
at `initial_flow`, while the flow split stays on the instantaneous inflow —
`build_bsm2` sets the reference `t_m = 0.125 d`, `initial_flow = Q_ref`; the
class default `t_m = 0` keeps no filter, because a standalone unit has no
design inflow to seed it with. On the 609-day BSM2 dynamic influent the missing
filter alone put average effluent S_NH 2.8% above the MATLAB reference, 0.1%
with it — #646),
`IdealThickener` (BSM2 thickener / dewatering — a stateless ideal `%TSS`
separator, concentration-dependent underflow flow), `ADM1DigesterUnit`
(continuously-fed ADM1 CSTR with gas headspace, dilution masked to the liquid
states), `DosingUnit` (chemical dosing: injects a `Reagent` — a fixed
composition, e.g. metal salt / acid-base / external carbon — into a stream at a
fixed or feedback-controlled flow; see `plant/dosing.py`), `UVUnit` /
`ChlorineContactUnit` (disinfection: UV dose-response and chlorine CT /
log-removal — see `plant/disinfection.py`), `SBRUnit`
(sequencing batch reactor: one tank cycling fill/react/settle/decant/idle with
variable volume and a pluggable settling model — see
`plant/sbr.py`), and `TakacsClarifier` (10-layer 1-D Takács 1991 model); see `.claude/rules/plant-settling.md`).

**Parsing anything here goes through `core/lexer.py`.** `plant/time_expr.py` and
`plant/measures.py` are expression languages whose grammars are their own but
whose *tokenizer* is not: both call `aquakin.core.lexer.tokenize`. Do not write
another one — see the "One tokenizer, three languages" note in
`.claude/rules/core.md`, which `aquakin/plant/**` does not auto-load.

**Units in the help.** A setting's numpydoc entry names its unit in
parentheses, flows and times per day as aquakin runs them on the plant's clock
(`Volume (m³).`, `Fixed dose flow (m³/d).`, `Integral time (d).`); the editor
shows that unit beside the field (on the plant's clock). Give each setting its
own entry (a shared `Kp, Ti, Tt : float` cannot name one unit each).

**Kinetic units share one reaction term (`plant/kinetic.py`).** Every unit
that evaluates a model in its `rhs` inherits `KineticUnit` (CSTR, MBR, SBR,
IFAS, ADM1 digester): `_init_conditions(conditions)` checks and builds
`_condition_arrays`, `_reaction(C, params, conditions)` is the reaction term
(`model.dCdt`, so the clip and positivity limiter apply, scaled to the plant
clock), and `_operating_conditions(inputs, signals)` gives the conditions at the
operating temperature (the plant's `OPERATING_T_SIGNAL` under a heat-balance
model, else the flow-weighted inlet, else static) plus the temperature the
aeration correction uses. Never write `stoich.T @ rates` in a unit or in
`balance.py`: `_reaction_term` calls `unit._reaction` so the balance integrates
what the unit integrates. A unit whose `_reaction` needs extra parameter
handling overrides it (the digester slaves `V_liq`). `AdjustableTemperatureUnit`
adds `set_temperature`; the heated digester uses `KineticUnit` alone, so
`Plant.set_temperature` leaves it alone. Deliberate exceptions, stated in the
unit: the SBR runs at its static `T` (intermittent feed, variable volume, not
heat-balance tracked), and IFAS evaluates its reaction per compartment inside
the biofilm RHS (which also calls `model.dCdt`).

**Sizes are checked at construction.** Every unit with a volume calls
`_constants.check_positive(name, "volume", value)` first in `__post_init__`
(zero, negative and infinite volumes are refused: they used to build, and a
steady-state solve then never ended). A new unit with a volume, area or
other size does the same.

**Co-stripped gases (`Aeration(stripped=...)`).** An aerated tank strips other
dissolved gases (N₂O, NO, ...) alongside oxygen inside the solve:
`stripped={"SN2O": ratio}` or `{sp: (ratio, sat)}`, normalized in
`Aeration.__post_init__` to a hashable tuple of `(species, ratio, sat)` triples.
`build_aeration_vectors` folds each gas into the same per-species vectors as
oxygen: open loop `kla_vec[gas] = kla·alpha·ratio`, closed loop an extra
`controlled[gas] = (same signal, gain·ratio)` entry, `sat_vec[gas] = sat` (no
beta/pressure fold). So `aeration_transfer` needs no gas branch, and CSTR, IFAS
and MBR all get it. The open-loop theta factor scales the gas kLa too, but the
`C_s(T)` saturation ratio is applied to the oxygen entry only (`o2_mask`; the
no-gas path keeps the historic whole-vector multiply, bit-identical).
`required_signals` de-duplicates, since the gas and oxygen read one signal.
**Consumers of `_controlled_kla` / `_kla_vec` must not assume only `SO` is
there**: the balance's O₂ term reads the aerated species' entry only, and
`direct_n2o_emission` (`kla_ratio=None` default) takes each reactor's in-solve
ratio and saturation from `unit._av.stripped`, so the post-solve emission equals
the in-solve flux whenever the aeration temperature correction is off (its
`_kla_history` reconstruction ignores that correction). Without in-solve
stripping the function remains a post-solve estimate at ratio 1. The
`plant.mass_balance` content table falls back on a model's declared species
`composition:` when there is no shipped table. Note a closed-loop plant whose
only effluent is a sensed tank: the DO sensor tap consumes `tank.out`, so it is
not a dangling output and must be passed as `effluent_ports`. Covered by
`tests/unit/test_aeration_stripping.py` (spec, vectors, transfer in both modes,
temperature, a toy-model one-tank steady state, mass balance and emission in both
modes, gradient) and `tests/integration/test_bsm1_asmn_n2o.py`.

**BSM1 with non-ASM1 models.** `build_bsm1(model, aeration=, settling_species=,
tss_species=)`: the clarifier solids come from `_constants.solids_species(model)`
(the ASM1 constants unchanged when the model's X-prefixed species are exactly
the ASM1 settling species, so ASM1 stays bit-identical; a model that adds a
particulate to ASM1 is derived, so the extra solid settles; otherwise X-prefixed
species settle and those with COD
content in `model.composition()` are TSS, at `ASM1_TSS_FACTOR`, the rest weight
0). `aeration={"tank3": Aeration(...), ...}` replaces named tanks' aeration
(`None` = unaerated); `tss_species` must be a subset of the settling species
(else `ValueError`). Covered by `tests/unit/test_bsm1_model_solids.py`. The BSM1 metrics (`derived_TSS`, `evaluate_bsm1`'s EQI)
still use the ASM1 species and do not yet support such models.

**Shared numeric constants (`plant/_constants.py`).** Two families of magic
number are consolidated there rather than redefined per module: `EPS_Q` (`1e-12`)
is the one **flow-/mass-weighted division guard** — the small epsilon added to a
near-zero flow/volume/rate denominator so a starved unit gives a finite (~0)
concentration instead of `0/0` (streams, mixer, delay, dosing, separators, the
Takács species-fraction, the design SRT/HRT/F:M ratios, the flow-weighted
time-average). `SECONDS_PER_DAY` / `HOURS_PER_DAY` / `MINUTES_PER_DAY` are the
time-unit factors (aeration, disinfection, clarifier, metrics). Use these instead
of an inline literal. Note the boundary: a *physically meaningful* small-quantity
floor — a minimum settler depth (`settling.py`), a minimum flow that sets a
maximum HRT and a `log`-domain floor (`PrimaryClarifier._removal_fraction`), or a
data-prep floor on a linearisation probe (`ifas`/`takacs`) — is a modelling choice
with its own magnitude and is deliberately **not** folded into `EPS_Q`.
