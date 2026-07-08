# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
Add entries under [Unreleased] as changes are made, grouped by:
Added / Changed / Deprecated / Removed / Fixed / Security (omit empty groups).
At release, rename [Unreleased] to the version with its date and open a fresh
empty [Unreleased] section above it.

0.1.0 is the first release, so its notes are a curated, Added-oriented summary of
what the package does rather than a diff against a prior version (there is none).
Granular per-change logging (Changed / Fixed / Deprecated / Removed, relative to
0.1.0) begins with the next release.
-->

## [Unreleased]

## [0.1.0] - 2026-07-08

The first public release of `aquakin`: a Python library for modelling reactive
scalar transport in aqueous environmental systems. Reaction models are declared
at runtime in YAML and compiled to JAX-native, automatically-differentiable rate
functions integrated with [Diffrax](https://github.com/patrick-kidger/diffrax).

### Added

- **Runtime kinetics engine.** Reaction models are authored in YAML and compiled
  to JAX rate functions — no recompilation to change a model. Rate expressions
  are parsed and evaluated through a custom AST (no `eval()`), so they are safe,
  introspectable, and fully differentiable. Every solve is differentiable
  end-to-end via JAX, and stiff systems are integrated with Diffrax implicit
  solvers (`Kvaerno5` by default). `import aquakin` enables JAX 64-bit mode,
  which the stiff solves require (a documented, one-time-warned side effect).

- **Reactors.** Batch (0-D well-mixed), plug-flow (1-D steady state),
  `BiofilmReactor` (1-D diffusion–reaction over biofilm depth for
  penetration-controlled processes), a Lagrangian particle-track reactor for
  offline OpenFOAM coupling, and a CFD-field reactor. Every reactor shares one
  `solve()` / `solve_sensitivity()` contract.

- **Plant-wide flowsheets (`aquakin.plant`).** Compose reactors with clarifiers,
  mixers/splitters, thickeners, chemical dosing, an ADM1 digester (with gas
  headspace), SBR, MBR, and IFAS/MBBR units into a full plant integrated under
  one monolithic, differentiable solve. Includes the IWA benchmark builders
  BSM1, BSM2, and an A²O plant; exact, gain-independent recycle resolution;
  run-to-steady-state plus a fast pseudo-transient-continuation algebraic
  steady-state solver (~10× faster than settling); per-unit temperature dynamics;
  PI control; dynamic influents; and results-level mass-balance closure.

- **Located events.** Time events and state root-crossings with exact state
  resets / mode switches (on/off pumps, SBR phase transitions, dosing on/off,
  level limits); time-scheduled events keep `jax.grad` finite through the solve.

- **Acid–base chemistry and mineral precipitation.** A charge-balance,
  state-derived pH, and saturation-index-driven mineral precipitation /
  dissolution (struvite + calcite, and iron/aluminium chemical phosphorus
  removal). Two opt-in differentiable formulations handle the stiff, very
  insoluble limit: an algebraic-equilibrium mode solving `IAP = Ksp` directly and
  a bounded-driver kinetic form for differentiable dynamics.

- **Sensitivity and uncertainty quantification.** Cap-free forward sensitivity
  and reverse-mode discrete adjoints through stiff plant solves; derivative-based
  global sensitivity (DGSM) screening; Monte-Carlo propagation; profile
  likelihood; and standardized scenario-comparison KPI tables.

- **Calibration.** MAP calibration of reactor and plant parameters against
  measured data through a shared forward-model seam, with a Laplace posterior,
  multi-stream observables, joint multi-batch fits, free initial-condition
  fitting, and predictive bands.

- **Design, evaluation, and reporting.** BSM effluent-quality (EQI) and
  operational-cost (OCI) indices; aeration/blower and activated-sludge sizing;
  monetised OPEX/CAPEX cost; greenhouse-gas (N₂O / CO₂e) footprinting; and
  influent characterization / SUMO-style fractionation, including per-row
  ingestion of arbitrary lab/SCADA CSVs.

- **Shipped model catalog.** Chemistry — ozonation / bromate formation (after
  Acero & von Gunten 2001) and UV/H₂O₂. Biology — the ASM activated-sludge family
  (ASM1; a two-step nitrification/denitrification variant with explicit nitrite;
  a two-pathway AOB nitrous-oxide model; anammox / deammonification; and a
  comammox complete-nitrifier variant), ADM1 anaerobic digestion in its BSM2 form
  with gas headspace, and the WATS sewer-process models (`wats_sewer_extended`
  and the paper-faithful `wats_sewer_khalil_paper` with structural variants for
  model-structure studies).

- **Utilities and introspection.** Conservation, nitrogen, and dimensional-unit
  consistency checks; LaTeX rendering of rate expressions; residence-time-
  distribution analytics; and by-name access to species, parameters, states, and
  streams. The public API is two-tier: a curated flat `aquakin.*` namespace of
  common entry points, plus complete per-domain surfaces in `aquakin.plant`,
  `aquakin.integrate`, and `aquakin.utils`.

- **Packaging and citation.** MIT-licensed with PEP 639 SPDX metadata, a
  single-sourced version, and a `CITATION.cff` so the package can be cited (and
  GitHub shows a "Cite this repository" button).

- **Documentation.** A Sphinx site on [Read the Docs](https://aquakin.readthedocs.io)
  with a task-oriented guide (getting started, reactors, plant-wide simulation,
  sensitivity & calibration), the YAML model-authoring reference, a catalog of
  every shipped model, and an API reference generated from the public surface.

[Unreleased]: https://github.com/DeGrootResearchGroup/aquakin/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DeGrootResearchGroup/aquakin/releases/tag/v0.1.0
