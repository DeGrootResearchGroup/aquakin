# Public API

This page is a task-oriented map of the `aquakin` API: it points you to the
right entry points for each job. A bare name is importable as `aquakin.<name>`
and has its full signature and docstring in the [API reference](api.md). A name
written with its module, such as `aquakin.plant.IFASUnit` or
`aquakin.study.run_study`, lives only in that subpackage; the API reference does
not list it, so read its docstring with `help()`. A name written as
`Plant.<method>` is a method of `aquakin.Plant`.

```{tip}
The guide pages — [Getting started](getting_started.md),
[Reactors](reactors.md), [Plant-wide simulation](plants.md), and
[Sensitivity & calibration](sensitivity_and_calibration.md) — walk through these
APIs with worked examples. Start there; use this page and the API reference to
look things up.
```

Everything not listed here (the AST node types, the compile context, the schema
models, the raw Diffrax solver objects) is internal and should not be imported
from `aquakin` directly. A rate function of your own subclasses the public
`RateFunction` (see
[Rate functions of your own](model_format.md#rate-functions-of-your-own)). Reactors are **stateless after construction** —
`solve()` takes every variable input as an argument — which is what lets you
`jax.vmap` a reactor over an ensemble.

## Loading and inspecting a model

| To… | Use |
|---|---|
| Load a shipped model by name | `load_model` |
| Load a model from a YAML file | `load_model_from_file` |
| Reset the model cache | `clear_model_cache` |
| Work with a loaded model | `CompiledModel` (its attributes and methods: `summary`, `species`, `time_unit`, `default_concentrations`, `default_parameters`, `default_conditions`, `concentrations`, `influent`, `parameter_values`, `units_of`, `check_units`, `check_conservation`, …) |
| Add a rate function aquakin does not ship | subclass `RateFunction`, then `register_rate_function` (before loading the model); `registered_rate_functions`; the guarded `aquakin.core.kernels.safe_pow`, `safe_sqrt` and `safe_log` for its arithmetic |
| Build a model from a spec, or parse one expression | `compile_model`, `parse_rate_expression` |

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
| Handle pumps / dosing / phase switches | `Event`, `solve_with_events` → `EventedResult` |

Solutions (`BatchSolution`, `PFRSolution`, `TrackSolution`, `BiofilmSolution`)
share by-name accessors: `C_named`, `C_named_many`, `final_named`, `final`,
`to_dataframe`, `to_csv`, `plot`.

## Plant-wide simulation

| To… | Use |
|---|---|
| Assemble a flowsheet | `Plant` (with `Plant(name, time_unit="d")` setting the plant's clock), and units `CSTRUnit`, `MixerUnit`, `RatioSplitter` / `SetpointSplitter` / `ThresholdSplitter`, `ChlorineContactUnit`, `UVUnit` |
| More unit operations | `aquakin.plant.ADM1DigesterUnit`, `aquakin.plant.IFASUnit`, `aquakin.plant.MBBRUnit`, `aquakin.plant.MBRUnit`, `aquakin.plant.SBRUnit`, `aquakin.plant.PrimaryClarifier`, `aquakin.plant.IdealThickener`, `aquakin.plant.StorageTank`, `aquakin.plant.DosingUnit`, `aquakin.plant.HydraulicDelayUnit`, `aquakin.plant.PIController` (`aquakin.plant.__all__` lists every unit) |
| Write a unit of your own | `Unit` (the protocol every unit satisfies) |
| Build the IWA benchmark plants | `aquakin.plant.bsm.build_bsm1`, `aquakin.plant.bsm.build_bsm2`, `aquakin.plant.bsm.load_bsm1_influent`, `aquakin.plant.bsm.load_bsm2_influent` |
| Catch wiring mistakes | `WiringError`, `UnknownUnitError`, `UnknownPortError` |
| Move water between units | `Stream`, `InfluentSeries` |
| Read a plant result | `PlantSolution`, `PlantCheck` |
| Check a plant's units before solving | `Plant.check_units` → `PlantUnitsReport` (`PlantUnitIssue`); `UnitsError` |
| Select a temperature model | `AlgebraicTemperature`, `HeatBalanceTemperature` (both `TemperatureModel`s) |
| Configure aeration | `Aeration`, `AerationSystem` |
| Translate state between models (ASM↔ADM) | `StateTranslator`, `IdentityTranslator`, `UnitScalingTranslator`, `aquakin.plant.ASM1toADM1`, `aquakin.plant.ADM1toASM1` |
| Run to steady state | `Plant.steady_state`, `Plant.run_to_steady_state`, `aquakin.plant.solve_steady_state` → `aquakin.plant.SteadyStateResult` |
| Report progress from a long solve, or cancel it | `Progress`, `Cancelled` |

## Flowsheet files

| To… | Use |
|---|---|
| Save a plant as a JSON/YAML file, and build one from it | `Plant.to_spec`, `Plant.from_spec` |
| Read, write and check a flowsheet file | `aquakin.plant.load_spec_file`, `aquakin.plant.write_spec`, `aquakin.plant.check_spec` (`aquakin.plant.SpecError`, `aquakin.plant.SpecWarning`) |
| Tell whether a flowsheet's inputs have changed | `aquakin.plant.spec_fingerprint`, `aquakin.plant.spec_inputs`, `aquakin.plant.spec_changes` |
| Run the same from a terminal | `aquakin catalog` prints the unit catalogue as JSON; `aquakin serve` answers a front end such as the flowsheet editor |

## Influent

| To… | Use |
|---|---|
| Split lab measurements into model states | `characterize_influent`, `fractionate`, `InfluentFractions` |
| Read a time series from CSV | `read_influent_csv` |
| Read or write an influent as a headed table | `aquakin.plant.read_influent_table`, `aquakin.plant.write_influent_table` |
| Describe an influent by expressions of time (`Q="18446 + 4000*sin(2*pi*t)"`) | `aquakin.plant.InfluentExpression`, `aquakin.plant.compile_time_expression` (`aquakin.plant.TimeExpressionError`) |

## Sensitivity analysis

| To… | Use |
|---|---|
| Gradient of an output w.r.t. parameters/conditions | `sensitivity` → `SensitivityResult` |
| Cap-free forward sensitivity through a stiff solve | `forward_sensitivity` → `ForwardSensitivityResult` |
| Global (Sobol-total-bound) sensitivity | `dgsm` → `DGSMResult` |
| Configure differentiation / integration | `DifferentiationConfig`, `IntegratorConfig`, `forward_adjoint` |
| Guard against silent non-finite gradients | `check_finite_gradient` |
| Sensitivities of a plant solve | `Plant.solve_sensitivity`, `Plant.dynamic_sensitivity`, `Plant.steady_state_sensitivity`, `Plant.dynamic_dgsm`, `Plant.steady_state_dgsm` |
| Gradients through a stiff solve by a chosen adjoint | `esdirk_adjoint_solve`, `implicit_euler_adjoint_solve` |

## Calibration and identifiability

| To… | Use |
|---|---|
| Point-estimate fit | `fit` → `FitResult` |
| MAP fit of plant parameters | `Plant.calibrate` (also `aquakin.plant.calibrate_plant`) → `CalibrationResult` |
| MAP fit + Laplace posterior | `calibrate` → `CalibrationResult` (→ `predictive_band` → `PredictiveBand`; `posterior_draws` for parameter samples) |
| Profile-likelihood identifiability | `profile_likelihood` → `ProfileResult` |
| Fisher information of a design (before running it) | `fisher_information` with `Experiment` → `FisherInformation` |
| Fisher information of any JAX observation function, optionally split by candidate measurement | `fisher_information_from_fn` → `FisherInformation` |
| Fisher information and profile likelihood of plant parameters | `aquakin.plant.plant_fisher_information` → `FisherInformation`, `aquakin.plant.plant_profile_likelihood` → `ProfileResult` |
| Choose plant sensors by the information they add | `aquakin.plant.plant_rank_sensor_sets` with `aquakin.plant.SensorSet` → `CandidateScore` |
| Identifiability report from a Fisher matrix | `identifiability` → `IdentifiabilityResult` (`EigenDirection`) |
| Score a design (D / A / E / modified-E) | `design_criterion` |
| Choose between candidate experiments | `rank_experiments` → `CandidateScore` |
| Choose the best set of measurements from a candidate pool under a budget | `design_from_candidates` → `CandidateDesign` |
| Optimize sampling times and initial concentrations | `optimize_experiment` → `ExperimentDesignResult` |
| Configure the fit | `LaplaceConfig`, `OptimizerConfig`, `FreeICConfig` |
| Name what a plant sensor measures | `aquakin.plant.PlantObservable` with species, a measurement keyword, a weighted sum of your own (`"BOD5 = 0.65*[SS] + 0.65*[XS]"`, read by `aquakin.plant.parse_measure`, which raises `aquakin.plant.MeasureSyntaxError`) or `aquakin.plant.Measure` |

## Uncertainty propagation and design

| To… | Use |
|---|---|
| Propagate input uncertainty | `monte_carlo` → `MonteCarloResult` |
| Compare named scenarios | `compare_scenarios` → `ScenarioComparison` |
| Optimize a constrained design | `optimize_design`, `Constraint` → `OptimizeResult` |
| Run an ensemble of solves | `integrate_ensemble` |

## Studies

A study is a folder that keeps a simulation, calibration or analysis of a
flowsheet together with its settings and its results, and knows when those
results have gone stale.

| To… | Use |
|---|---|
| Create, read and change a study | `aquakin.study.create_study`, `aquakin.study.load_study`, `aquakin.study.read_study`, `aquakin.study.update_study`, `aquakin.study.copy_study` |
| Run it and read its results | `aquakin.study.run_study`, `aquakin.study.study_results` |
| Check whether its results are up to date | `aquakin.study.check_study` → `aquakin.study.StudyStatus` (`aquakin.study.StudyError`) |
| Pin it to a snapshot of the flowsheet, or release it | `aquakin.study.freeze_study`, `aquakin.study.unfreeze_study` |
| Read and write result files | `aquakin.study.read_table` / `write_table`, `read_trajectory` / `write_trajectory`, `read_matrix` / `write_matrix`, `read_named_values` / `write_named_values` (all in `aquakin.study`) |
| Run the same from a terminal | `aquakin study run`, `check`, `freeze`, `unfreeze`, `copy` |

## Model checks: units and conservation

| To… | Use |
|---|---|
| Dimensional consistency of rate expressions | `check_model_units`, `parse_units` (findings are `UnitWarning`; a `speciation:` molar mass that disagrees with a species' units gives `SpeciationUnitsWarning`) |
| Compare or convert two unit strings, scale included | `units_match`, `conversion_factor` |
| Mass / electron balance of stoichiometry | `check_conservation`, `check_nitrogen`, `mass_balance` → `MassBalance` (`ComponentBalance`); `Plant.mass_balance` for a whole plant |
| Composition tables | `composition_table`, `canonical_content` |

## Benchmark evaluation and reporting

| To… | Use |
|---|---|
| Score BSM1 / BSM2 performance | `evaluate_bsm1` → `BSM1Evaluation`, `evaluate_bsm2` → `BSM2Evaluation` (`NoDigesterError`) |
| Effluent quality / operating-cost indices | `effluent_quality_index`, `operational_cost_index`, `operational_cost_index_bsm2` |
| Sludge and effluent statistics | `sludge_metrics` → `SludgeMetrics`, `effluent_averages`, `time_average` |
| Lumped measurements from model states | `derived_COD`, `derived_BOD`, `derived_TKN`, `derived_TSS` |
| Greenhouse-gas footprint | `carbon_footprint` → `CarbonFootprint`, `direct_n2o_emission`, `stripped_n2o`, `n2o_n_to_co2e`, `methane_to_co2e`, `co2e_from_energy`, `carbon_mass` |
| Monetized operating cost | `operating_cost` → `OperatingCost`, `CostFactors` |
| Compare KPIs across scenarios | `kpi_comparison` → `KPIComparison` |

## Design and sizing

| To… | Use |
|---|---|
| Size an activated-sludge process | `size_activated_sludge` → `ActivatedSludgeSizing`, `design_summary` |
| Aeration / blower energy | `aeration_energy`, `required_airflow`, `blower_energy`, `blower_power_kw`, `blower_airflow_total`, `AerationDesignPoint` |
| Pumping / mixing / heating energy | `pumping_energy`, `pumping_energy_bsm2`, `mixing_energy`, `heating_energy` |
| Disinfection sizing | `uv_dose`, `uv_log_inactivation`, `ct_value`, `ct_log_removal`, `t10_from_rtd`, `t10_from_baffling` |

For the exact signatures, return types, and every remaining top-level symbol,
see the [API reference](api.md).
