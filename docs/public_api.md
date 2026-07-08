# Public API

This page is a task-oriented map of the top-level `aquakin` API: it points you
to the right entry points for each job. Every name listed here is importable as
`aquakin.<name>`, and the [API reference](api.md) has the full signatures and
docstrings for all of them.

```{tip}
The guide pages — [Getting started](getting_started.md),
[Reactors](reactors.md), [Plant-wide simulation](plants.md), and
[Sensitivity & calibration](sensitivity_and_calibration.md) — walk through these
APIs with worked examples. Start there; use this page and the API reference to
look things up.
```

Everything not listed here (the AST node types, the compile context, the schema
models, the raw Diffrax solver objects) is internal and should not be imported
from `aquakin` directly. Reactors are **stateless after construction** —
`solve()` takes every variable input as an argument — which is what lets you
`jax.vmap` a reactor over an ensemble.

## Loading and inspecting a model

| To… | Use |
|---|---|
| Load a shipped model by name | `load_model` |
| Load a model from a YAML file | `load_model_from_file` |
| Reset the model cache | `clear_model_cache` |
| Work with a loaded model | `CompiledModel` (its methods: `summary`, `species`, `default_concentrations`, `default_parameters`, `default_conditions`, `concentrations`, `influent`, `parameter_values`, `units_of`, `check_units`, `check_conservation`, …) |

## Conditions

| To… | Use |
|---|---|
| Set conditions for a single tank (0-D) | `OperatingConditions` |
| Set spatially varying conditions (PFR/CFD) | `SpatialConditions` |

## Reactors and solutions

| To… | Use |
|---|---|
| Integrate a well-mixed batch (0-D) | `BatchReactor` |
| Integrate a plug-flow profile (1-D) | `PlugFlowReactor` |
| Integrate along a Lagrangian particle track | `ParticleTrackReactor`, `Track` |
| Resolve a biofilm over depth (1-D) | `BiofilmReactor` |
| React inside a CFD operator split | `CFDReactor` |
| Handle pumps / dosing / phase switches | `Event`, `solve_with_events` |

Solutions (`BatchSolution`, `PFRSolution`, `TrackSolution`, `BiofilmSolution`)
share by-name accessors: `C_named`, `C_named_many`, `final_named`, `final`,
`to_dataframe`, `to_csv`, `plot`.

## Plant-wide simulation

| To… | Use |
|---|---|
| Assemble a flowsheet | `Plant`, and units `CSTRUnit`, `MixerUnit`, `RatioSplitter` / `SetpointSplitter` / `ThresholdSplitter`, `ChlorineContactUnit`, `UVUnit`, … (see the API reference for the full unit catalog) |
| Move water between units | `Stream`, `InfluentSeries` |
| Read a plant result | `PlantSolution`, `PlantCheck` |
| Select a temperature model | `AlgebraicTemperature`, `HeatBalanceTemperature` |
| Configure aeration | `Aeration`, `AerationSystem` |
| Translate state between models (ASM↔ADM) | `StateTranslator`, `IdentityTranslator` |

## Influent characterisation

| To… | Use |
|---|---|
| Split lab measurements into model states | `characterize_influent`, `fractionate`, `InfluentFractions` |
| Read a time series from CSV | `read_influent_csv` |

## Sensitivity analysis

| To… | Use |
|---|---|
| Gradient of an output w.r.t. parameters/conditions | `sensitivity` → `SensitivityResult` |
| Cap-free forward sensitivity through a stiff solve | `forward_sensitivity` → `ForwardSensitivityResult` |
| Global (Sobol-total-bound) sensitivity | `dgsm` → `DGSMResult` |
| Configure differentiation / integration | `DifferentiationConfig`, `IntegratorConfig`, `forward_adjoint` |
| Guard against silent non-finite gradients | `check_finite_gradient` |

## Calibration and identifiability

| To… | Use |
|---|---|
| Point-estimate fit | `fit` → `FitResult` |
| MAP fit + Laplace posterior | `calibrate` → `CalibrationResult` (→ `predictive_band` → `PredictiveBand`) |
| Profile-likelihood identifiability | `profile_likelihood` → `ProfileResult` |
| Configure the fit | `LaplaceConfig`, `OptimizerConfig`, `FreeICConfig` |

## Uncertainty propagation and design

| To… | Use |
|---|---|
| Propagate input uncertainty | `monte_carlo` → `MonteCarloResult` |
| Compare named scenarios | `compare_scenarios` → `ScenarioComparison` |
| Optimise a constrained design | `optimize_design`, `Constraint` → `OptimizeResult` |
| Run an ensemble of solves | `integrate_ensemble` |

## Model checks: units and conservation

| To… | Use |
|---|---|
| Dimensional consistency of rate expressions | `check_model_units`, `parse_units` |
| Mass / electron balance of stoichiometry | `check_conservation`, `check_nitrogen`, `mass_balance` |
| Composition tables | `composition_table`, `canonical_content` |

## Benchmark evaluation and reporting

| To… | Use |
|---|---|
| Score BSM1 / BSM2 performance | `evaluate_bsm1`, `evaluate_bsm2` |
| Effluent quality / operating-cost indices | `effluent_quality_index`, `operational_cost_index`, `operational_cost_index_bsm2` |
| Sludge and effluent statistics | `sludge_metrics`, `effluent_averages`, `time_average` |
| Greenhouse-gas footprint | `carbon_footprint`, `direct_n2o_emission`, `stripped_n2o` |
| Monetised operating cost | `operating_cost`, `CostFactors` |
| Compare KPIs across scenarios | `kpi_comparison` |

## Design and sizing

| To… | Use |
|---|---|
| Size an activated-sludge process | `size_activated_sludge`, `design_summary` |
| Aeration / blower energy | `aeration_energy`, `required_airflow`, `blower_energy`, `blower_power_kw` |
| Pumping / mixing / heating energy | `pumping_energy`, `mixing_energy`, `heating_energy` |
| Disinfection sizing | `uv_dose`, `uv_log_inactivation`, `ct_value`, `ct_log_removal`, `t10_from_rtd` |

For the exact signatures, return types, and every remaining exported symbol, see
the [API reference](api.md).
