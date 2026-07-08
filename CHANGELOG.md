# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
Add entries under [Unreleased] as changes are made, grouped by:
Added / Changed / Deprecated / Removed / Fixed / Security (omit empty groups).
At release, rename [Unreleased] to the version with its date and open a fresh
empty [Unreleased] section above it.
-->

## [Unreleased]

The first public release (0.1.0) is being prepared. Entries below accumulate the
notable changes it will contain.

<!--
Release note for 0.1.0: do NOT ship these entries verbatim, and do NOT backfill
pre-release history. A changelog documents changes between releases; there is no
prior release to diff against, so "Changed"/"Fixed" framing (e.g. the
network->model rename, the sludge_metrics fix) is meaningless to a first-time
user — nobody ran the earlier state. At release, collapse this section into a
curated "Initial release" summary: an Added-oriented, high-level description of
what the package does. The entries here are the raw material for that summary,
not the final text. Granular per-change logging (Changed/Fixed/Deprecated/
Removed) begins with the release after 0.1.0, relative to 0.1.0.
-->


### Added

- Plant-wide calibration: `plant.calibrate(...)` fits reaction-model parameters
  (and, optionally, assembled-state initial conditions) against measured stream
  data through the forward-model seam, reusing the reactor calibration
  machinery. Supports multi-stream observables and joint multi-batch fits.
  (#498, #499, #500, #501)
- `aquakin.time_average(values, t)` — the trapezoidal time-average of a solution
  trajectory (with the one-point steady-state convention) is now a single public
  helper. It replaces four private per-module copies, two of which had an
  *inverted* `(t, values)` signature; every plant metric / design / aeration /
  GHG / evaluation path now calls the one `(values, t)` kernel. (#476)
- `CITATION.cff`, so the package can be cited and GitHub shows a "Cite this
  repository" button. (#494)
- `CHANGELOG.md` (this file), following the Keep a Changelog format.
- A uniform plant exception taxonomy (`aquakin.UnknownUnitError`,
  `UnknownPortError`, `WiringError`, `NoDigesterError`) so a caller can tell an
  unknown *name* from an invalid *wiring/usage*. Each subclasses the built-in it
  historically raised (`UnknownUnitError`/`UnknownPortError` are `KeyError`,
  `WiringError`/`NoDigesterError` are `ValueError`), so existing `except`
  clauses keep working. (#466)

### Changed

- `BiofilmReactor` now defaults `atol=None` to the per-component
  `default_atol` noise floor (tiled across its layers), matching every other
  reactor, instead of a fixed `1e-9` scalar that is ~9 orders too tight for
  g/m³ ASM/ADM states (a slow or step-budget-exhausting solve). `IFASUnit` /
  `MBBRUnit`, which build a `BiofilmReactor` internally, inherit the fix. Pass an
  explicit scalar or array `atol=` to override. (#450)
- `CompiledModel.atol()` (the by-name per-species tolerance builder) now defaults
  to the per-component `default_atol` floor rather than a uniform `1e-9`; pass
  `default=` for a uniform scalar floor. (#450)
- `BatchReactor.solve(time_unit=...)` now warns when a `dtmax` cap is set and a
  non-native `time_unit` is used, since `dtmax` stays in the model's native time
  unit while `time_unit` rescales `t_span`/`t_eval` (the cap would otherwise be
  off by the unit ratio). (#450)
- **Renamed "reaction network" to "reaction model" throughout** (a hard rename
  with no back-compatibility aliases, done while pre-release). Public API:
  `CompiledNetwork` → `CompiledModel`, `compile_network` → `compile_model`,
  `NetworkSpec` → `ModelSpec`, `NetworkMeta` → `ModelMeta`,
  `load_network[_from_file]` → `load_model[_from_file]`, `clear_network_cache` →
  `clear_model_cache`, `check_network_units` → `check_model_units`. In YAML model
  files the top-level key `network:` is now `model:`, and the `aquakin.networks`
  package is now `aquakin.models`. (#491)
- Grouped the many tuning arguments of `calibrate`, `profile_likelihood`, and
  `plant.calibrate` into config dataclasses — `OptimizerConfig`, `LaplaceConfig`,
  and `FreeICConfig` (alongside the existing `DifferentiationConfig`) — so each
  signature keeps only its primary arguments. (#504)
- Modernized packaging metadata: adopted PEP 639 SPDX license metadata
  (`license = "MIT"` + `license-files`), single-sourced the package version from
  `aquakin.__version__`, and added `Documentation` and `Changelog` project URLs.
  (#494)
- Harmonized the forward-sensitivity signatures with `solve`: `params` is now
  **keyword-only** on `BatchReactor.solve_sensitivity`,
  `PlugFlowReactor.solve_sensitivity`, `BiofilmReactor.solve_sensitivity`, and the
  free `aquakin.forward_sensitivity`, matching the keyword-only `params` on every
  reactor `solve`. A parameter vector passed positionally (e.g.
  `reactor.solve_sensitivity(C0, params, t_span, ...)`) now raises `TypeError`
  instead of silently landing in the `t_span` slot; pass it as
  `params=...`. (#234)
- Library progress output is now routed through the standard `logging` module
  instead of `print`, so callers can silence or redirect it. The `progress=`
  option of `plant.steady_state_dgsm` / `dynamic_dgsm` emits a `logging.INFO`
  record (on the `aquakin.plant.sensitivity` logger) every N samples rather than
  writing to stdout — enable it with, e.g., `logging.basicConfig(level=
  logging.INFO)`. aquakin attaches a `NullHandler` to its package logger and
  never configures logging on the application's behalf. (#472)
- `Plant.set_temperature` now raises `UnknownUnitError` (a `KeyError` subclass)
  for an unknown unit name, matching every other unknown-unit lookup, instead of
  `ValueError`; a unit that cannot take a temperature now raises `WiringError`
  (still a `ValueError`). Code that caught the unknown-unit case as a bare
  `ValueError` should catch `KeyError`/`UnknownUnitError` (or the message).
  (#466)

### Fixed

- Load-time errors for a rate or stoichiometry expression, a named
  `expressions:` entry, or a `speciation:` / `precipitation:` block that
  references an **undeclared species, parameter, or condition** now append a
  `Did you mean: …?` close-match hint (via the shared `hints.did_you_mean`,
  already used elsewhere) instead of dumping the full sorted list of valid names —
  the graduate-student authoring ergonomics the rest of the API already had,
  applied consistently across the core-compile and Pydantic-schema layers. (#470)
- A forward-mode `dgsm` screen no longer masks unrelated errors. Previously
  *any* exception raised while evaluating the samples in forward mode (a bug in
  the user's `fn`, a bad shape, an OOM) was relabelled as the "use
  `forward_adjoint()`" guidance error; it now converts only JAX's actual
  forward-mode-through-`custom_vjp` rejection and lets every other error
  propagate with its real traceback. (#466)
- The plant mass balance's biogas term no longer swallows genuine failures: it
  now catches only `NoDigesterError` (the "no digester → no biogas" case) from
  `digester_gas`, so a real bug inside `digester_gas` surfaces instead of being
  silently reported as zero biogas. (#466)
- `sludge_metrics(substrate=...)` now validates its argument against
  `{"BOD", "COD"}` and raises on an invalid value, instead of silently falling
  back to COD (which could report F:M and the influent BOD load roughly 2× off).
  (#492)
- `check_conservation` now warns when a species matches no role-based
  composition rule, instead of silently treating it as having zero COD/N/P
  content. (#493)
- Separator/clarifier units (`IdealClarifier`, `IdealThickener`,
  `PrimaryClarifier`, `TakacsClarifier`, and the SBR settling models) now **raise
  a clear error** when a configured settling / particulate / TSS species is not
  in the model, instead of the previous *contradictory* behaviour where some
  units silently dropped it (under-settling / under-counting solids without
  warning) and others raised a bare `KeyError`. A model that does not define a
  unit's default (ASM1) settling species must now name its own `settling_species`
  / `particulate_species` / `tss_species` explicitly. The species-mask
  construction and the Q-weighted feed / capture-split logic these units
  duplicated are now shared helpers. (#464)

[Unreleased]: https://github.com/DeGrootResearchGroup/aquakin/commits/main
