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

### Added

- Plant-wide calibration: `plant.calibrate(...)` fits reaction-model parameters
  (and, optionally, assembled-state initial conditions) against measured stream
  data through the forward-model seam, reusing the reactor calibration
  machinery. Supports multi-stream observables and joint multi-batch fits.
  (#498, #499, #500, #501)
- `CITATION.cff`, so the package can be cited and GitHub shows a "Cite this
  repository" button. (#494)
- `CHANGELOG.md` (this file), following the Keep a Changelog format.

### Changed

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

### Fixed

- `sludge_metrics(substrate=...)` now validates its argument against
  `{"BOD", "COD"}` and raises on an invalid value, instead of silently falling
  back to COD (which could report F:M and the influent BOD load roughly 2× off).
  (#492)
- `check_conservation` now warns when a species matches no role-based
  composition rule, instead of silently treating it as having zero COD/N/P
  content. (#493)

[Unreleased]: https://github.com/DeGrootResearchGroup/aquakin/commits/main
