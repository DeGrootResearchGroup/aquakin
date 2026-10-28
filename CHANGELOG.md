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

## [0.1.1] - unreleased

### Added

- `OptimizerConfig(bounds={name: (lo, hi)})` gives `aquakin.calibrate` (and
  `profile_likelihood`) absolute bounds on the free rate parameters in physical
  units, with `None` for an open side. An entry replaces the parameter's
  model-declared bound. `param_halfwidth` still works and is now an extra box
  around the start, intersected with the absolute bounds.
- `CalibrationResult.at_bound` lists the fitted parameters (and initial pools,
  as `"ic[k].<species>"`) that ended on an optimiser bound, as `"lower"` or
  `"upper"`, and `calibrate` warns about them. Their Laplace standard deviation
  is reported as `nan` rather than a finite value from the curvature at a
  constrained point, they have zero variance in `posterior_cov` (posterior
  draws keep them on the bound), and the other parameters' covariance is
  conditional on them, so their standard deviations can be smaller than
  before, when the bound parameter was treated as free to move either way. `aquakin.identifiability(result)` carries the same flag
  (`IdentifiabilityResult.at_bound`, also accepted as `at_bound=` for a bare
  matrix) and the same `nan` standard deviations.

- `LaplaceConfig(include_ic=True)` gives a joint Laplace posterior over the rate
  parameters and the initial pools fitted with `free_ic`, for both
  `aquakin.calibrate` and `Plant.calibrate`. The joint covariance and Hessian
  (the Fisher information with `loss="nll"` and `method="gauss_newton"`) are
  returned as `CalibrationResult.joint_cov` / `joint_hessian` / `joint_names`,
  and the pool standard deviations as `ic_named_std`. The rate standard
  deviations then account for rate-pool correlation instead of treating the
  pools as known. The default (`include_ic=False`) is unchanged.
- `aquakin.fisher_information` computes the Fisher information of a planned
  set of batch experiments (`aquakin.Experiment`: initial state, sampling
  times, observed species, noise level, optionally its own reactor) at assumed
  parameter values, for the rate parameters and optionally the initial pools.
  No data are needed, so designs can be compared before they are run. The
  returned `FisherInformation` gives the matrix with named rows, its
  physical-space form, and the Cramér-Rao standard deviations. It uses the same
  coordinates as `calibrate`, so at a fit's optimum it equals the Gauss-Newton
  Laplace Hessian.
- `aquakin.identifiability` turns a Fisher information matrix (from
  `fisher_information`, a `CalibrationResult`, or a bare matrix) into an
  identifiability report: eigen-directions with named loadings, flagging the
  weakly identified ones; the correlation matrix; Cramér-Rao
  standard deviations and relative standard errors (infinite for a quantity
  that is not identifiable at all); and the collinearity index of any
  parameter subset, with a search for the largest identifiable subset.
- Optimal experiment design on the Fisher information:
  - `aquakin.design_criterion` scores a design by D (log det), A (trace of the
    inverse), E (smallest eigenvalue) or modified-E (condition number),
    optionally for a subset of parameters with the rest as nuisance. A design
    that cannot determine some combination of the parameters (an eigenvalue at
    most 1e-10 of the largest, the rule `identifiability` uses) gets the worst
    score.
  - `aquakin.rank_experiments` ranks candidate experiments by the
    information they add to a base design.
  - `aquakin.optimize_experiment` optimises sampling times and initial
    concentrations by gradient, with time bounds and a minimum sample spacing.
  - Each can average the criterion over parameter draws (`param_draws`) for a
    design that is robust to uncertainty in the assumed parameters.
- `observation_fn` lets `calibrate`, `profile_likelihood` and `Experiment`
  (for `fisher_information` and experiment design) use a measurement that is a
  function of several states and the parameters, such as a UV absorbance
  that is a weighted sum of organic pools, in place of `observed_species`.
  `observation_fn(C, params)` returns a scalar or a vector per time point, and
  `sigma` may give one value per output.
- `CalibrationResult.posterior_draws(model, n_draw, seed=)` returns full
  parameter vectors drawn from the Laplace posterior, for example as
  `param_draws` for robust experiment design. `predictive_band` uses the same
  draws.

### Changed

- **`aquakin.calibrate` now enforces the `bounds:` declared on model
  parameters.** They are mapped into the space the optimiser works in (log for
  `positive_log`, logit for `logit`), as `aquakin.fit` already did, and are the
  same for every multistart start. Before, `calibrate` ignored them, so a fit
  could leave the declared range. A fit whose optimum lies outside a declared
  bound now stops on the bound, and a starting value outside it is moved to the
  nearest bound with a `RuntimeWarning`. To get the old behaviour, pass
  `OptimizerConfig(use_model_bounds=False)`. `Plant.calibrate` is unchanged.

### Deprecated

- `LaplaceConfig(eig_keep=...)` is deprecated and ignored (see Fixed). Passing
  it emits a `DeprecationWarning`. To flag weakly identified directions, use
  `aquakin.identifiability(..., eig_keep=...)`.

### Fixed

- `Plant.solve` and `Plant.run_to_steady_state` can now be traced by
  `jax.jit` / `jax.grad` on a plant with a `TakacsClarifier`, including the
  validated `build_bsm1(use_takacs=True)` settler. Tracing used to raise a
  `ConcretizationTypeError` while forming the default `atol`, so it had to be
  passed explicitly; the default is now the same value traced or not.
- The Laplace standard deviations from `aquakin.calibrate` and
  `Plant.calibrate` (`params_named_std`, `posterior_cov`, `ic_named_std`) no
  longer understate the uncertainty of weakly identified parameters. The
  covariance used to drop every Hessian direction with an eigenvalue below
  `1e-2` of the largest and give it zero variance, so a parameter on such a
  direction could be reported as known to a fraction of a percent when its true
  spread was around ten percent. The covariance is now the full inverse of the
  Hessian. A parameter on an exactly null direction (one the data do not
  constrain at all) is reported with an infinite standard deviation and a
  `RuntimeWarning`, as `aquakin.identifiability` reports it. Posterior draws
  and `predictive_band` use the same covariance, so they are wider for weakly
  identified fits.
- `aquakin.calibrate` and `Plant.calibrate` no longer abort when the forward
  solve fails part-way through a fit (for example a stiff linear solve going
  non-finite). With `OptimizerConfig(n_starts>1)`, a start whose solve fails is
  skipped with a `RuntimeWarning` and the best remaining start is kept. If every
  start fails, or the solve already fails at the starting guess, you now get a
  `RuntimeError` that says what happened and how to fix it (bound the rates with
  `OptimizerConfig(param_halfwidth=...)`, use a better starting guess, or tighten
  the `free_ic` bounds). This replaces the raw XLA callback traceback.

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

[Unreleased]: https://github.com/DeGrootResearchGroup/aquakin/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/DeGrootResearchGroup/aquakin/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/DeGrootResearchGroup/aquakin/releases/tag/v0.1.0
