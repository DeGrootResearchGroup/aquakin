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

### Added

- `aquakin.plant.plant_rank_sensor_sets` ranks candidate sensor sets on a plant
  by a design criterion of the Fisher information each adds to the sensors
  already in place. It is the plant version of `rank_experiments`. A candidate
  is a `SensorSet`: its observables, with an absolute or relative `sigma` and
  an optional `mask`. Every candidate is scored from one shared sensitivity
  solve, and `param_draws=` averages the criterion over uncertain parameters.
- `aquakin.plant.plant_fisher_information` gives the Fisher information of a
  plant design, the plant version of `fisher_information`. Measured channels
  are named as for `Plant.calibrate`, including a unit's state such as
  `("digester", ["S_IN", "S_gas_ch4"])`. `sigma` can be absolute or relative
  to the signal, a `mask` drops unmeasured samples, and several runs add their
  information. The sensitivity comes from `Plant.solve_sensitivity` (the
  default, `mode="forward"`) or from the stable adjoint (`mode="reverse"`). At
  a `Plant.calibrate` result with `loss="nll"` and the Gauss-Newton Laplace,
  the reverse-mode matrix equals the Laplace Hessian.
- `aquakin.plant.plant_profile_likelihood` profiles one plant parameter. It
  re-fits the others with `Plant.calibrate` at each grid value and returns a
  `ProfileResult` with the likelihood-ratio interval. The sweep is the same as
  for `profile_likelihood`, and every grid point reuses one compiled
  objective.

### Changed

- `PrimaryClarifier` can compute its retention time from a first-order filtered
  inflow, as the BSM2 reference does: pass `t_m` (the filter time constant, in
  days) and `initial_flow`. The filtered flow becomes one extra state entry
  after the concentrations, and only the particulate removal uses it; the
  underflow is still `f_PS` times the instantaneous inflow. The default
  `t_m=0` keeps the old behaviour and state layout.
- `build_bsm2` now uses the reference filter (`t_m=0.125` d, seeded at
  `Q_ref`). Steady states are unchanged. Dynamic BSM2 results change: on the
  609-day influent the average effluent ammonia was 2.8% above the reference
  without the filter and is 0.1% above it with the filter. The primary's state
  is now one entry longer, so a hand-built `initial_state(overrides=...)` entry
  for `"primary"` needs the filtered flow appended.
- `C_named`, `list_species` and `calibrate(free_ic=...)` now accept any unit
  whose state starts with its concentrations, followed by a volume or other
  scalars. This covers the filtered primary clarifier, `StorageTank`,
  `MBRUnit` and `SBRUnit`.

### Fixed

- Reusing a plant from separate jitted functions with the same closed-over
  parameter vector no longer leaks cached parameter slices between traces.
- Stable-adjoint gradients (`DifferentiationConfig(method="stable")`) on large
  plants such as BSM2 could be wrong while staying finite. The colored backward
  Jacobian, on by default from 100 states, missed couplings: those from the
  settler to the sludge train and the reactors, and the loops through the
  solids-dependent thickener and dewatering flows. Gradients from the activated
  sludge into the digester (for example `asm1.muA` on digester methane) were off
  by up to 2.3×. The sparsity pattern now covers these couplings. The colored
  backward is also checked against the dense Jacobian at several states,
  including the solve's start state, and falls back to the dense Jacobian with
  a warning if anything is missed. Gradients within one domain were unaffected
  or only slightly affected. If you computed cross-domain BSM2 gradients with
  the default settings, recompute them; `IntegratorConfig(colored_jacobian=False)`
  gave correct values before this fix.

## [0.2.0] - 2026-10-29

This release extends parameter estimation into uncertainty quantification and
experiment design (Fisher information, identifiability reports, optimal
experiment design, joint posteriors over rates and initial pools, estimated
noise levels, missing observations), makes `Plant.steady_state` more robust,
and builds the reference BSM2 settler by default. It also fixes a wrong
heterotroph growth rate in the ASM2d family, Laplace standard deviations that
understated the uncertainty of weakly identified parameters, and event and
sampling times under `time_unit=`. Some results change; see **Changed** and
**Fixed**.

### Added

- `observation_fn` lets `calibrate`, `profile_likelihood` and `Experiment`
  (for `fisher_information` and experiment design) use a measurement that is a
  function of several states and the parameters, such as a UV absorbance
  that is a weighted sum of organic pools, in place of `observed_species`.
  `observation_fn(C, params)` returns a scalar or a vector per time point, and
  `sigma` may give one value per output. It may also take the sample time,
  `observation_fn(t, C, params)`, for a sensor drift, a time-varying dilution,
  or a quantity measured only at some times. A function that requires exactly
  three positional arguments is the time-aware form; one that can be called
  with two (such as `f(C, params=None)`) is the `(C, params)` form. `t` is the
  entry of `t_obs` exactly as given, in its unit.
- Missing observations: a `NaN` entry in `calibrate`'s `observations` is left
  out of the loss and the Laplace Hessian (`mse` and `wmse` average over the
  observed entries), so sparse or staggered measurements no longer need a huge
  per-entry `sigma`. A missing entry adds nothing to the loss even where its
  `sigma` or the prediction there is not finite. `Plant.calibrate` does the
  same, and both reject an `inf` observation and a dataset with every entry
  missing. `Experiment(mask=...)` marks which outputs are measured at which
  times for `fisher_information` and the experiment-design functions.
- `aquakin.calibrate(loss="nll", sigma="estimate")` estimates an unknown
  noise level: the concentrated likelihood, with the noise standard deviation
  of each observed output set to its RMS residual over its observed entries.
  One fit now gives a likelihood-scaled Laplace standard deviation and
  profile, where it used to take an `mse` fit and an `nll` refit with `sigma`
  set by hand. The estimate is returned as `CalibrationResult.sigma_estimate`,
  and `profile_likelihood` accepts `sigma="estimate"` too. Both optimisers
  support it, and so does `Plant.calibrate`, with one estimate per observed
  stream channel. The estimate covers every dataset together, so it cannot be
  an entry of a per-dataset `sigma` list.
- `LaplaceConfig(include_ic=True)` gives a joint Laplace posterior over the rate
  parameters and the initial pools fitted with `free_ic`, for both
  `aquakin.calibrate` and `Plant.calibrate`. The joint covariance and Hessian
  (the Fisher information with `loss="nll"` and `method="gauss_newton"`) are
  returned as `CalibrationResult.joint_cov` / `joint_hessian` / `joint_names`,
  and the pool standard deviations as `ic_named_std`. The rate standard
  deviations then account for rate-pool correlation instead of treating the
  pools as known. The default (`include_ic=False`) is unchanged.
- `CalibrationResult.posterior_draws(model, n_draw, seed=)` returns full
  parameter vectors drawn from the Laplace posterior, for example as
  `param_draws` for robust experiment design. `predictive_band` uses the same
  draws.
- `CalibrationResult.at_bound` lists the fitted parameters (and initial pools,
  as `"ic[k].<species>"`) that ended on an optimiser bound, as `"lower"` or
  `"upper"`, and `calibrate` and `Plant.calibrate` warn about them. "On the
  bound" is judged relative to the parameter's own size. Their Laplace standard
  deviation is reported as `nan` rather than a finite value from the curvature
  at a constrained point, and they have zero variance in `posterior_cov`
  (posterior draws keep them on the bound). The other parameters' covariance is
  conditional on them, so their standard deviations can be smaller than if the
  bound parameter were free to move either way. `aquakin.identifiability(result)`
  carries the same flag (`IdentifiabilityResult.at_bound`, also accepted as
  `at_bound=` for a bare matrix) and the same `nan` standard deviations.
- `OptimizerConfig(bounds={name: (lo, hi)})` gives `aquakin.calibrate`,
  `profile_likelihood` and `Plant.calibrate` (keyed by `"<model>.<param>"`)
  absolute bounds on the free rate parameters in physical units, with `None`
  for an open side. An entry replaces the parameter's model-declared bound.
  `param_halfwidth` still works and is now an extra box around the start,
  intersected with the absolute bounds.
- `OptimizerConfig(starts=[...])` runs `aquakin.calibrate` from an explicit list
  of starts, such as a grid, and keeps the best fit. Each start is a full
  parameter vector or a `{name: value}` dict over the free parameters. All
  starts run through one compiled objective, so a grid of starts no longer
  needs one `calibrate` call (and one compilation) per start.
  `profile_likelihood` uses the list for its cold fits, and `Plant.calibrate`
  accepts it too.
- `FreeICConfig(bounds={species: (lo, hi)})` sets bounds per fitted initial
  pool, for pools of very different size; one `(lo, hi)` pair still applies to
  every pool. A pool that ends on its own bound is listed in
  `CalibrationResult.at_bound`. `profile_likelihood` drops the profiled
  species from the dict for its inner fits.
- `FreeICConfig(fixed_total=True)` holds the sum of the fitted pools at its
  value in each dataset's `C0` and fits only the split, for example a known
  total COD divided between two substrate pools. The split is fitted on the
  simplex, so the pools stay positive and sum to the total exactly. The fitted
  fractions are returned as `CalibrationResult.ic_fractions` and, with
  `LaplaceConfig(include_ic=True)`, their standard deviations as
  `ic_fractions_std`. `fisher_information` and `identifiability` use the same
  parameterisation. The split's coordinates are unbounded, so they never
  appear in `at_bound`. `Plant.calibrate` supports both options for its
  `"unit.species"` initial pools (a bounds dict is keyed by those names), and
  `profile_likelihood` cannot profile a pool of a fixed-total split.
- `aquakin.fisher_information` computes the Fisher information of a planned
  set of batch experiments (`aquakin.Experiment`: initial state, sampling
  times, observed species, noise level, optionally its own reactor) at assumed
  parameter values, for the rate parameters, the initial pools, or both. No
  data are needed, so designs can be compared before they are run. The
  returned `FisherInformation` gives the matrix with named rows, its
  physical-space form, and the Cramér-Rao standard deviations. It uses the same
  coordinates as `calibrate`, so at a fit's optimum it equals the Gauss-Newton
  Laplace Hessian.
- `aquakin.identifiability` turns a Fisher information matrix (from
  `fisher_information`, a `CalibrationResult`, or a bare matrix) into an
  identifiability report: eigen-directions with named loadings, flagging the
  weakly identified ones; the correlation matrix; Cramér-Rao standard
  deviations and relative standard errors (infinite for a quantity that is not
  identifiable at all); and the collinearity index of any parameter subset,
  with a search for the largest identifiable subset.
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
- `Plant.run_to_steady_state(criterion="scaled_residual", tol=..., scale_floor=...)`
  stops the forward run on the scaled residual
  `max_i |F_i| / max(|y_i|, scale_floor)` that `Plant.steady_state` converges
  on, with the same default floor (`max(|y0|, 1e-6)` per state), so a forward
  and an algebraic steady state can be held to one tolerance. The default
  `criterion="norm"` keeps the previous stopping test. The norm test can stop
  far from steady state on a plant whose states span many magnitudes: on the
  BSM2 digester started from its published steady state it stops after
  0.03 d at a scaled residual of 0.07.

### Changed

- `aquakin` now requires Python 3.11 or later (`requires-python = ">=3.11"`).
  Its dependency floors (jax/jaxlib 0.10, diffrax 0.7, optimistix 0.1) already
  require 3.11, so 0.1.0's `>=3.10` claimed support that could not be
  installed. The package metadata now lists Python 3.11 and 3.12.
- **`aquakin.calibrate` now enforces the `bounds:` declared on model
  parameters.** They are mapped into the space the optimiser works in (log for
  `positive_log`, logit for `logit`), as `aquakin.fit` already did, and are the
  same for every multistart start. Before, `calibrate` ignored them, so a fit
  could leave the declared range. A fit whose optimum lies outside a declared
  bound now stops on the bound, and a starting value outside it is moved to the
  nearest bound with a `RuntimeWarning`. To get the old behaviour, pass
  `OptimizerConfig(use_model_bounds=False)`. `Plant.calibrate` enforces them
  the same way, and also honours `param_halfwidth`, which it used to ignore.
- **`build_bsm2` now builds the reference BSM2 settler by default:**
  `settler_composition_mode="lumped_tss"` (one TSS state per layer) with
  `settler_soluble_holdup=True` (the solubles carried through the ten layers),
  as in `settler1dv5_bsm2` and as `build_bsm1` already does. The default used
  to be `"per_species"` without holdup, a settler with no reference
  counterpart. Steady states are unchanged. Dynamic results change: the
  settler's liquid volume now damps the soluble effluent signal (the effluent
  ammonia peaks and troughs), as in the reference, and the plant state has a
  different settler layout. To get the old settler, pass
  `build_bsm2(settler_composition_mode="per_species", settler_soluble_holdup=False)`.
- **`Plant.steady_state` (pseudo-transient continuation, PTC) is more robust
  and usually faster.** The operating point it returns is unchanged; only the
  path to it differs. On the default BSM2 with the reference settler (above)
  PTC used to take about 100 iterations, varied erratically with tiny changes
  to the start, and could run past its 400-iteration cap and fall back to
  forward integration. It now converges in about 37 iterations, and with the
  per-species settler in about 25-30 instead of about 50. Three changes in
  `ptc_forward`, the second and third of which can be switched off by keyword:
  - The positivity limit on the step is applied per state instead of as one
    step length for all states, so a near-zero concentration that the step
    drives negative (nitrate in an anoxic tank) no longer shrinks every other
    state's update to nothing.
  - After a healthy step (a full step whose residual did not rise) the
    pseudo-time step grows by at least `growth_floor` (default 2; `1.0` is
    pure switched-evolution relaxation, SER), and the SER ratio is taken on the
    mean scaled residual (`ser_norm="mean"`; `"max"` is the old norm).
    Convergence is still tested on the max-norm.
  - After `stall_patience` iterations (default 30; `0` disables) without a new
    best residual, the iteration restarts from the best iterate at a tenth of
    the pseudo-time step.
  With the reference settler and a flat `scale_floor=1.0`, PTC can still stall
  near the root (the settler's lower layers sit on a kink of the Takács flux
  limiter); `steady_state` then falls back to forward integration as before.
- `SteadyStateResult.residual` is now also set for `Plant.run_to_steady_state`
  and for the forward fallback of `Plant.steady_state` (`"ptc->forward"`, with
  or without `design=`): the scaled residual of the returned state. It used to
  be `None` for forward results.

### Deprecated

- `LaplaceConfig(eig_keep=...)` is deprecated and ignored (see Fixed). Passing
  it emits a `DeprecationWarning`. To flag weakly identified directions, use
  `aquakin.identifiability(..., eig_keep=...)`.

### Fixed

Calibration and uncertainty:

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
  start fails, or the solve already fails at the starting guess, a
  `RuntimeError` says what happened and how to fix it (bound the rates with
  `OptimizerConfig(param_halfwidth=...)`, use a better starting guess, or tighten
  the `free_ic` bounds). This replaces the raw XLA callback traceback.
- A warm-started `profile_likelihood` with `free_ic` now keeps the measured
  `C0` as the centre of the free-pool prior (`FreeICConfig(prior_log_std=...)`)
  at every grid point. Each warm-started fit used to re-centre the prior on its
  neighbour's fitted pools, which made the profile loss too low away from the
  anchor point and disagree with `warm_start=False`. The neighbour's fitted
  pools are now used only as the optimiser's starting point.
- A compiled-objective cache shared between `aquakin.calibrate` calls could
  reuse the objective built for different observations, loss or `sigma`, and
  return the first fit's optimum. The cache key now includes them (and the free
  set, transforms and priors). Only `profile_likelihood` shares the cache, and
  its fits all use the same data, so no released result was affected.

Models:

- The four heterotroph growth processes in `asm2d` (and `asm2d_chemp`, which
  extends it) and `asm2d_tud` all used the rate for aerobic growth on SF. Growth
  on SA now has its own SA saturation term, and anoxic growth on SF and SA now
  has the oxygen inhibition, nitrate saturation and anoxic reduction factor of
  the published model. New parameters: `KA_H` (4 g COD/m3) and `etaNO3_H` (0.8)
  in `asm2d`, and `etaNO_H` (0.8) in `asm2d_tud`, which shares `Kac` with the
  PAOs. Results from these models change, including the A2O plant
  (`build_a2o`): denitrification by heterotrophs is now limited by nitrate and
  inhibited by oxygen, and acetate uptake no longer depends on `KF`.
- Shipped rate expressions no longer return NaN when a substrate-competition
  ratio has a zero denominator. In `asm1` (and `asm1_ammonia_limitation`) the
  hydrolysis rate, in `asm2d`, `asm2d_chemp` and `asm2d_tud` the
  `SF / (SA + SF)` growth fraction and the `asm2d` poly-P / PHA lysis, and in
  the `wats_sewer` family the `S_B / (S_B + S_VFA)` maintenance split were
  `0 / 0` when both species were zero (or clipped from tiny negative values),
  which made every `dC/dt` touching those reactions NaN and could stop a solve.
  These ratios now use `safe_div` and go to 0 there. Values are bit-for-bit
  unchanged wherever the denominator is non-zero.
- Kinetic mineral precipitation counted an ion once per mineral that uses it
  when building the ionic strength, so a shared ion (iron, phosphate or
  aluminium in `precipitation_metal_phosphate`, for example) inflated the
  ionic strength and shifted every saturation index. Each dissolved ion now
  counts once.
- Equilibrium-mode precipitation (`mode: equilibrium`) now reads
  `ionic_strength_field` when one is set, as the kinetic minerals do, and
  otherwise adds the dissolved mineral ions to `ionic_strength_offset`
  (evaluated at the equilibrium composition). It used to apply the fixed
  offset alone. Results of `precipitation_metal_phosphate_equilibrium` change
  slightly.
- A `speciation:` block with `activity_model: none` and an
  `ionic_strength_field` now exports the solution's real ionic strength. It
  used to export 0, which a downstream precipitation block would then use for
  its activity coefficients. The computed pH is unchanged.
- `CompiledModel.precipitation_equilibrium()` no longer raises `KeyError: 'pH'`
  when the pH comes from a `speciation:` block; the speciation step now runs
  before the projection.
- Rate evaluation with an integer concentration vector no longer truncates
  the model's constants and condition values to integers (pH 7.5 became 7, an
  exponent 0.5 became 0). Rates are now computed in floating point.
- `check_nitrogen` now detects the nitrate species (`S_NO`, `SNO`, `SNO3`,
  `SNOX`, ...) instead of assuming `S_NO`, which reported false nitrogen
  violations for the ASM models. Naming a species the model does not declare
  with `nitrate=` now raises a `ValueError`.
- `to_latex` no longer drops needed parentheses: `a - (b + c)`,
  `a - (b - c)` and `-(a + b)` used to render as `a - b + c`, `a - b - c` and
  `-a + b`.

Solvers and events:

- `BatchReactor.solve` and `Plant.solve` with both `time_unit=` and `events=`
  now fire time events at the right time. `Event.at_times` was read in the
  model's native unit while `t_span` was converted, so with `time_unit="min"`
  on a model in seconds an event meant for 1 minute fired at 1 second. Event
  times, the `t` passed to `cond_fn` and `apply`, and the `events_log` times
  are now all in the solve's `time_unit`. `Event` objects are no longer
  modified by a solve.
- An evented solve with `t_eval=None` that is stopped by a terminal event now
  reports the time it stopped, not `t_span[1]`.
- A time event at exactly the start of the solve is now applied to the initial
  state and logged; it used to be dropped silently. An event at exactly the end
  of the solve now reports its pre-reset state whether `t_eval` is `None` or
  includes the end time (previously the two disagreed).
- The fast forward integrator (`Plant.solve(forward_fast=True)`) now raises an
  error when it runs out of steps or its step size collapses. It used to return
  zeros for the unreached output times. It also now honours
  `IntegratorConfig(max_steps=...)`, which it previously ignored.
- `Plant.solve` and `Plant.run_to_steady_state` can now be traced by
  `jax.jit` / `jax.grad` on a plant with a `TakacsClarifier`, including the
  validated `build_bsm1(use_takacs=True)` settler. Tracing used to raise a
  `ConcretizationTypeError` while forming the default `atol`, so it had to be
  passed explicitly; the default is now the same value traced or not.
- `BiofilmReactor` can be built inside `jax.jit`. With the default
  `fixed_mask` its reactive-particulate check called `bool()` on a traced
  value and raised `TracerBoolConversionError`, and a traced scalar `atol`
  raised `ConcretizationTypeError`. The check now runs on concrete values (it
  is skipped when `soluble_mask` itself is traced), and a scalar `atol` may be
  traced.
- `BiofilmReactor` no longer reuses a solve compiled with a different adjoint.
  Forward-mode `aquakin.sensitivity` on a biofilm reactor that had already been
  solved could fail, because the copy it makes shared the compiled solve. Copies
  made for sensitivity, calibration and experimental design now start with
  their own compile caches, and the biofilm cache is keyed on the solver
  settings.

Plants:

- `Plant.steady_state(design=...)` no longer drops the design overrides when
  PTC does not converge. The forward fallback used to integrate the recorded
  influent and report `converged=True` for the wrong operating point; it now
  integrates the design right-hand side, accepts `run_to_steady_state`'s
  `criterion`, `tol` and `scale_floor` through `fallback_kwargs`, and stops on
  the same test as the fallback without `design`. The continuation and
  arclength fallbacks work in the parameter vector alone and cannot carry
  design overrides, so they are skipped when `design` is given; a
  `UserWarning` now names them when `continuation_from` was requested.
- After `Plant.solve(..., time_unit=...)` in a unit other than the plant's own
  (days for the ASM/BSM plants), streams rebuilt with `plant.stream`, the BSM
  evaluation indices, `sludge_metrics` / `plant.sludge_age` and
  `plant.mass_balance` sampled the influent and control signals at the wrong
  times, and the mass balance integrated per-day flows over hours (so it did
  not close). They now use the solve's native times; the new
  `PlantSolution.t_native` gives them. `solution.t`, stream `t` and the
  balance `window` stay in the requested unit, flows stay per native time unit,
  and the balance terms are grams whatever the unit.
- `sludge_metrics` and `plant.sludge_age` take the SRT's solids losses from
  the right streams. The effluent is the plant's recorded final effluent
  (`plant.effluent_endpoint`), except that a bypass-combined final effluent,
  whose untreated bypass solids never entered the activated-sludge system,
  gives way to the secondary-clarifier overflow. The wastage is the registered
  `"wastage"` stream. Unit names (`settler`, `clarifier`, ...) are matched only
  when those are missing, so a custom plant whose primary clarifier is named
  `settler` no longer gets its SRT from the primary overflow. On BSM2 the SRT
  is now about 20.9 d at the constant-influent steady state, up from about
  17.5 d: the wastage used to be read from the dewatering cake, which also
  carries primary and digested solids, and is now the activated-sludge wastage
  `underflow_split.waste`. BSM1 and A²O results are unchanged. Pass
  `waste_port=` / `effluent_port=` to choose other streams.
- `aquakin.utils.rtd.percentile_time` and `morrill_index` gave wrong values
  for a tracer response with a dead time: the early percentiles were
  interpolated back across the leading zero stretch (a plug-flow pulse at
  t = 5 gave a Morrill index near 1.8 instead of 1). They now interpolate from
  the last sample before the tracer arrives.
- The BSM docstrings now say which benchmark convention the indices follow.
  `evaluate_bsm1` / `BSM1Evaluation` give the **updated** BSM1 EQI and OCI
  (Alex et al. 2008: EQI nitrogen weights 30 / 10, AE = S_sat·V·kLa / 1800,
  per-stream pumping factors), not the original Copp (2002) ones (weights
  20 / 20, a quadratic kLa correlation, one pumping factor), and
  `evaluate_bsm2` uses the same EQI weights with the full BSM2 OCI. The numbers
  are unchanged. The `TakacsClarifier` / `build_bsm1` / `build_bsm2` docstrings
  now agree on the settler attribution: the soluble holdup is the ten-layer
  soluble model of the reference settlers (`settler1dv4` for BSM1,
  `settler1dv5_bsm2` for BSM2), and both reference settlers are
  `composition_mode="lumped_tss"`. The earlier statement that `"per_species"`
  matches the BSM2 settler was wrong: neither reference has per-layer
  composition memory.

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

[Unreleased]: https://github.com/DeGrootResearchGroup/aquakin/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DeGrootResearchGroup/aquakin/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DeGrootResearchGroup/aquakin/releases/tag/v0.1.0
