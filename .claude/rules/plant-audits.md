---
paths:
  - "aquakin/plant/units_check.py"
  - "aquakin/plant/balance.py"
  - "aquakin/plant/metrics.py"
---

# Rules — plant audits (unit audit, mass balance)

`plant.check_units()` and `plant.mass_balance()`. Split out of `.claude/rules/plant.md`.

- **Unit audit — `plant.check_units()` (`plant/units_check.py`).** Stream
  concentrations cross units unconverted (beyond the translators below), so a
  concentration mismatch inside a unit is silent; the audit compares *declared* units on static metadata only (never
  the jitted solve) and returns a `PlantUnitsReport` (`errors` / `warnings`
  of `PlantUnitIssue`). Checks: on a plant without `time_unit`, every model's
  `time_unit` agrees; a unit implementing the `AssumedUnits` hook
  (`assumed_units() -> {species: unit}`) gets them — only *fixed* internal
  concentration-basis assumptions belong there, never a user-supplied number
  in any consistent unit. The thickener and Takács convert a mass-per-volume
  solids unit (below) and list only the species they could not convert;
  each model's own
  `check_units()` is rolled up as warnings. All comparisons go through the
  scale-aware `aquakin.units_match` (`mg/L` = `g/m3` ≠ `kg/m3`; a currency is
  compared only when both sides state one). `plant.check()` carries the
  findings (`PlantCheck.unit_issues` / `.unit_errors`; unit errors make `.ok`
  false and `raise_on_error` raises `UnitsError`).
  `connect` / `add_influent` also reject an explicit translator whose
  `source_model` / `target_model` species list differs from the endpoint's
  (`WiringError`; a different instance with the same species is accepted).
  When a new unit's internals hard-code a concentration basis (a `g/m3`
  constant), prefer converting: fold `_constants.grams_per_m3_factors(model,
  species)` into the species' weights at construction, as `IdealThickener`
  (its `_tss_vec`) and `TakacsClarifier` (its `_part_tss_factors`, only
  while `settling_params` are the BSM defaults and only for species with a
  nonzero TSS factor) do — the per-species flux split and the lumped-TSS
  inventory are ratios, so the species stay in the model's units. A factor
  of exactly 1 is not applied (bit-identical). A unit that is not a mass per
  volume (moles; the schema default `mol/L` of a toy model) gets factor 1 and
  is returned by the unit's `assumed_units` (its `_unconverted_tss`), so the
  audit reports it instead of the unit refusing to build. `assumed_units` is
  for what cannot be converted. **Same species,
  different units:** `_default_translator` (behind `connect` /
  `add_influent`) inserts a `translators.UnitScalingTranslator` (diagonal
  per-species `unit_scaling` factors, `None` when all are 1) when
  `_same_species_other_units` holds (identical species lists, some unit
  string differs), raising `WiringError` on an unconvertible pair; identical
  units on two instances still hit the "different instances" error. It is
  diagonal, so inside a recycle loop the per-model grouping of the affine
  seed is only approximate across models and the adaptive `recycle_tol`
  solve converges it. **Translators convert instead of
  being checked:** `ASM1toADM1` / `ADM1toASM1` (`plant/interfaces.py`) keep
  their equations in fixed *working units* (`_ASM1_UNITS`: g/m3, SALK mol/m3;
  `_ADM1_UNITS`: kg/m3, kmol/m3) and `_unit_scales` derives per-species
  factors from the models' declared `species_units` via
  `utils.units.conversion_factor` at construction (`x_working = f·x_model`;
  applied on the way in, divided out on the way out). It returns `None` when
  every factor is exactly 1 (every shipped model), so `translate` skips the
  multiply and stays bit-identical; an unconvertible unit (mass vs moles, two
  different currencies) raises `ValueError` at construction, an unreadable
  one warns and is taken to be the working unit. Because the factors come
  from the models a translator was *built* with, `connect` also requires
  those models to declare the same units as the endpoints'
  (`_check_translator_endpoints`). A new translator should follow the same
  pattern rather than hard-code a model's scale. **Influents convert too:**
  `InfluentSeries.__post_init__` (`_convert_to_model_units`) converts declared
  `units` to the model's — `t` to `model.time_unit`, `Q` to `m3/<time>`, `T`
  to K (a `utils.units.CELSIUS_SPELLINGS` spelling is offset by 273.15), each
  species to `model.species_units` — via `conversion_factor`, multiplying only
  when a factor differs from 1 (so the BSM loaders, already in ASM1 units, stay
  bit-identical). A different quantity or an unreadable unit raises; a target
  the model leaves unknown (blank/unreadable species unit, no inferable time
  unit) keeps the declared unit and the data unchanged. `units` afterwards
  records the converted units, so `dataclasses.replace` is idempotent. The
  audit therefore has no influent check. The plant then moves each series
  onto its clock with `InfluentSeries.in_time_unit` (built via
  `dataclasses.replace(..., units=None)` so construction does not convert
  back to the model's time unit).
- **Results-level mass-balance closure — `plant.mass_balance(sol, …)` (#150).**
  The first thing an engineer does with a result: *does what went in equal what
  came out + what left as gas + what accumulated?* Returns a `MassBalance`
  (`aquakin.plant.balance`, exported as `aquakin.MassBalance` /
  `aquakin.ComponentBalance` / `aquakin.mass_balance`) with, per component (COD
  / N / P), the **inflow** (influents), **outflow** (terminal/dangling material
  streams — effluent, wasted sludge, disposal cake), **gas** (O₂ transferred in
  by aeration, the digester biogas, denitrification N₂, and any gas co-stripped
  by `Aeration(stripped=...)` (`gas_detail["stripped_<q>"]`, the stripped
  species' transfer dotted with their content) — computed from each unit's
  own `aeration_term`, as its `rhs` adds it, and a reaction-production integral over the reactive units, with
  the digester deliberately excluded from the N gas term since it has no N gas
  phase; plus what a `GasExportUnit` vents through its own gas phase,
  `gas_detail["gas_export_<q>"]`, integrated from its `gas_export`) and
  **accumulation** (ΔInventory across every unit — reactor / clarifier
  / digester liquid+headspace at `V_liq`/`V_gas` / storage / Takács blanket).
  `_unit_inventory` **dispatches on unit-implemented contracts, never on private
  state layout** (#505): a unit whose state layout is non-trivial declares
  **`component_inventory(state, content, params)`** returning its own
  `{component: grams}` — the layered **`TakacsClarifier`** (blanket summed over
  layers, both `composition_mode`s + `soluble_holdup`) and the
  **`ADM1DigesterUnit`** (liquid states at `V_liq`, the three gas-headspace states
  at `V_gas`, via its own `_state_volume_vector(params)` which `_reaction_volume`
  also reuses for the reaction integral). A unit holding a single well-mixed
  liquid volume (`StorageTank` / `MBRUnit` / `SBRUnit`, whose states are
  `[C…, scalar(s)]`) instead declares that volume through **`liquid_volume(state)`**,
  so its inventory is `V·C` (`PrimaryClarifier` too, returning its fixed `volume`,
  since its filter state trails the concentrations); a plain concentration-vector
  unit (`CSTRUnit`) falls through to the generic `volume·C`. This completes the
  inversion `liquid_volume` started — the balance no longer reaches into
  `_part_block_size` / `_n_part` / `param_index["V_gas"]` etc., and a unit with a
  new state representation participates by implementing `component_inventory`
  rather than editing the balance helper. Contract covered by
  `test_mass_balance_plant.py::test_stateful_units_own_their_component_inventory`.
  `imbalance = in − out − gas − accumulation` is the closure; `mb["N"]`,
  `mb.closed(rtol)`, `mb.summary()`, `mb[q].relative_imbalance`. Everything is on
  one canonical g basis (g COD / g N / g P), so the ASM water line (g/m³) and the
  ADM digester (kg/m³, kmol/m³) sum via `aquakin.composition_table` /
  `aquakin.canonical_content` (the shipped per-species COD/N/P content tables;
  `composition_table(net, electron_acceptor_cod=False)` = lab COD, the default
  `True` = the electron-equivalent convention `check_conservation` wants;
  `params=` reads a calibrated/BSM-specific composition such as `i_XB`). Closes
  BSM1 to ~1e-7 and BSM2 (two models, biogas, recycles) to COD ~0.08% / N
  ~0.03% at steady state; the gas integrals are exact at steady state and
  otherwise accurate to the `t_eval` sampling. **Exception: a lumped-TSS
  inventory is approximate in N and P, and the balance must say so.** A
  `TakacsClarifier` with `composition_mode="lumped_tss"` has no blanket
  composition, so `component_inventory` values its N/P at the model-default
  solids composition (COD is exact); off steady state that leaves an N/P
  closure error that does not shrink with `t_eval` (2.2e-3 relative N on BSM1,
  days 1–3 from the warm start; zero at steady state). Do not tighten a
  transient N/P closure test on such a plant; use `per_species` or balance at
  steady state. A unit whose inventory only estimates a component declares
  `units.ApproximateInventoryUnit` (`inventory_caveats() -> {component: note}`)
  and `mass_balance` copies each note into that component's
  `ComponentBalance.notes` (printed by `summary()`); never detect the case by
  probing a unit's class or private state. **The reaction integral must run
  each unit's kinetics at the conditions its `rhs` used** (the inlet or
  heat-balance operating temperature, not the static `T`): `_reaction_terms_at`
  rebuilds the unit's inputs and signals with `Plant._unit_inputs_and_signals`
  and asks the unit for `_reaction_conditions`; evaluating at static conditions
  leaves a spurious imbalance whenever the feed temperature differs. **This is
  the tool that found the ADM1 nitrogen transcription error** (see the `adm1`
  model note).
