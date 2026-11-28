---
paths:
  - "aquakin/schema/**"
  - "aquakin/models/**/*.yaml"
  - "tests/fixtures/*.yaml"
  - "examples/*.yaml"
  - "examples/**/models/*.yaml"
---

# Rules — YAML model schema (`aquakin/schema/`, model `*.yaml`)

What an author of a model YAML, or an editor of the schema code, must not get
wrong. The format itself (every key, its default, the expression grammar, the
`extends:` merge semantics, `composition:` / `auto`, `speciation:` /
`precipitation:`, `check_units`) is documented for users in
`docs/model_format.md`; read it there rather than restating it here. What each
shipped model is: `docs/model_catalog.md`.

## Loading

- Parse a model YAML with `parse_model_yaml` (`schema/loader.py`, a safe loader
  that refuses a repeated mapping key, which `yaml.safe_load` keeps silently as
  the last value); any other YAML with `yaml.safe_load()`. Never a full loader.
- Quote a species name YAML could read as a non-string (`"NO"`, `"ON"`).
- `default_concentration` is a reference value, not an initial condition; do
  not tune it to make a particular experiment work (users pass `C0`).

## Units

- **Every shipped model must pass `check_units()` and declare a determinable
  `time_unit`** (enforced by
  `test_units_check.py::test_shipped_models_are_unit_clean`), because the
  plant unit audit relies on both.
- **Do not reach for `units_acknowledged:` first.** Check the parameter units
  against the model's source: a finding is usually a mislabelled unit, not a
  property of the rate (the ASM2d-TUD findings were: `qPP` / `mAN` are
  g_P/(g_COD·d), the `iCharge_*` factors kmol/g). Acknowledge only a knowingly
  unconventional published form, with a written reason; the acknowledged set
  is pinned in `test_units_check.py::test_acknowledged_reactions_are_pinned`,
  so adding one means updating that pin as a reviewed change.
- **Label a rate constant from the stoichiometry**, not from a source table
  that writes it per unit biomass in its own currency: a species with a fixed
  `+1` / `-1` coefficient changes by exactly the rate, so the rate must match
  at least one such species (`_stoichiometry_issue` in `utils/units.py`). This
  is what catches a specific rate (g X/(g biomass·d)) mislabelled `1/d`.
- **In a SUMO-derived model put `units_acknowledged:` before
  `stoichiometry:`**: `scripts/verify_sumo_asm.py` reads every indented line
  after `stoichiometry:` as a species key.

## `molar_mass` in `speciation:` / `precipitation:`

- `molar_mass` converts the species' state to mol/L
  (`mol/L = state / molar_mass`), so it must be a power of ten for a molar
  species and a molecular weight (> 1, never a clean power of ten) for a mass
  species. **When you change a species' `units`, update every `molar_mass`
  that refers to it**; otherwise pH and saturation indices shift silently.
- The load-time audit (`_audit_speciation_molar_mass` in
  `schema/model_spec.py`) only warns (`SpeciationUnitsWarning`), skips blank,
  unparseable and non-mass / non-molar units, and must keep **zero false
  positives on every shipped model**
  (`test_speciation_units_audit.py::test_shipped_models_do_not_warn`). Do not
  make it an error or widen it without re-checking every shipped model.

## Parameters and priors

- **`bounds` are the hard feasibility fence; a `prior` is the plausible-
  uncertainty belief.** Keep both: priors are far tighter than the ±decade
  bounds, and a global-sensitivity screen or Bayesian calibration must sample
  the prior, not the bounds.
- The shipped `asm1` and `adm1` priors are physical-space Gaussians **centred
  on the parameter's nominal**, with `std` = nominal × a literature
  coefficient of variation, on every bounded, non-fraction calibration
  parameter (`asm1_ammonia_limitation` inherits them through `extends:`):
  - `asm1`: the per-parameter reported CV of the activated-sludge parameter
    database (Hauduc 2011 PhD thesis, Table 3-1).
  - `adm1`: the Brun et al. 2002 uncertainty classes around the BSM2
    municipal-sludge nominal (kinetics / half-saturations / inhibition / decay
    class 3, 50%; yields class 1, 5%).
  - **Do not replace the `adm1` CVs with the ranges of the Mo et al. 2023 ADM1
    review (Table 3)**: they aggregate across digester types and substrates, so
    they measure substrate variation, not the parameter uncertainty of a
    mesophilic municipal-sludge digester. Replace the Brun CVs only with a
    municipal-sludge-specific range source.

## Temperature corrections

- **Temperature-correct rate constants only.** The correction is confined to
  `CompiledModel.rates` (via `_apply_temperature`); `compute_stoich` uses the
  raw parameters, so a yield or composition coefficient is never scaled. Do
  not apply it anywhere else.
- A `temperature:` parameter's `value` is the value at `ref_T` (the shipped
  `asm1` and `asm3_2step*` models use Kelvin, `ref_T = 293.15`), so a model run at
  `ref_T` is unchanged by the correction. Keep it that way when adding one.
- **A `theta` given as a parameter name is resolved at compile time**
  (`_resolve_theta_params` in `core/model.py`). The `ln_theta` slot of its
  `temperature_corrections` entry holds only the log of the theta default; the
  live value comes from the parameter vector through
  `temperature_theta_params`. A new consumer that reads or rewrites
  `temperature_corrections` (as `bsm2_asm1_model` and
  `_warn_if_influent_T_inconsistent` do) must honour
  `temperature_theta_params`, or it silently uses the default theta.

## Stoichiometry, `composition:` and `auto`

- **Stoichiometry must stay state-independent**: a string coefficient may use
  only numbers, parameters and arithmetic (no species, conditions, named
  expressions or functions), because it is evaluated once per `solve()`, not
  per RHS call.
- **`composition:` values are literal floats.** A yield-dependent derived
  coefficient belongs in the stoichiometry, not in `composition:`.
- **The conservation checks are advisory**: `check_conservation()` /
  `check_nitrogen()` never run at load and return the violations rather than
  raising. A model with no `composition:` falls back to the role-based
  `aquakin.composition_table` (the ASM / ADM families) so the API is uniform;
  `auto` does **not** use that fallback and requires declared `composition:`.
- **`auto` is resolved before the stoichiometry is read**
  (`core/stoich_resolve.py`). With all-numeric neighbours it is a numeric
  least-squares solve (an over-determined but consistent system is accepted);
  with an expression neighbour it emits a derived expression, which needs a
  square system (`#auto == #conserved_for`). Keep the expression path
  symbolic: baking a resolved value would break conservation and the gradient
  once a yield is calibrated.
- **Shipped models keep their published rounded literals**; `auto` is opt-in.
  Do not convert a shipped model to `auto` (its numbers would change).
- The WATS sewer model (`wats_sewer`) declares `composition:` on every
  species, and the conservation suite checks it against its own table. A new
  sewer model must declare it too (`tests/integration/test_mass_balance.py`
  fails on a shipped `wats_sewer*` model it does not check).

## `gases:`

- **A model that declares `gases:` carries no gas phase.** The block only says
  which species are dissolved gases; the unit that owns a gas phase (aeration,
  a digester headspace) does the transfer, so one model runs in every unit. Do
  not add headspace states or transfer reactions to such a model. (`adm1` keeps
  its BSM2 headspace in the model and declares no `gases:`.)
- **Gas properties come from `GAS_LIBRARY` in `core/gases.py`**, each value
  with its source in that module's docstring. A model overrides one with a
  reason in a comment; a new gas goes in the library with its source, not in a
  model file.
- `molar_mass` follows the `speciation:` convention (`mol/L = state /
  molar_mass`).

## `extends:`

- **Write a variant as `extends:` rather than a copy** when it differs from a
  shipped model by overrides alone, so a fix to the base reaches it. A variant
  whose stoichiometry is computed from the base stays a full copy
  (a generator emits a pure-override variant as an `extends` file and the
  rest as copies).
- The merge happens **before Pydantic validation**
  (`schema/inheritance.py`, wired in `schema/loader.py`), so the merged whole
  is schema-checked; do not validate the derived fragment on its own.
- New parameters append after the base's, so the flat parameter vector order
  differs from a hand-written copy: refer to parameters by name, never by
  index. A YAML without `extends` must load exactly as before.

## Non-negative states

- `positivity_limiter:` (throttles the *net reaction* output near zero) and
  `clip_negative_states:` (clamps the rate *inputs*, and a state-derived pH, to
  `>= 0`) are distinct and both act inside `CompiledModel.dCdt` on the
  reaction term only. **Never apply either to transport**: transport on the raw
  state keeps washout self-correcting and the inter-unit mass balance exact.
  This is why a unit's reaction term must be `model.dCdt(...)`, not
  `stoich.T @ rates`.
- Both are identity at feasible states; do not change either in a way that
  alters a solution where every concentration is non-negative. The clip is
  what stops the Monod-at-negative-`C` blow-up, which the limiter does not
  (`asm1` needs it once a settler recycles concentrated solids into a small
  reactor).
- **Every shipped model sets `clip_negative_states: true`**, and a new one must
  too: `tests/unit/test_shipped_rate_singularities.py` requires every shipped
  model's rates at a `-1e-9` state to equal those at `0`, because a Monod or
  ratio term evaluated at a negative input runs the reaction backwards. The
  clamp is all-or-nothing per model, which is safe only because no shipped
  state is signed (charge and alkalinity balances are computed from
  non-negative ion pools); a model with a genuinely signed state needs a
  per-species exemption first, which is a model-format decision for the owner.
  Set it in a generator's base YAML (generators copy top-level keys), not
  only in the generated file.
