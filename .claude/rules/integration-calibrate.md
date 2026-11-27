---
paths:
  - "aquakin/integrate/calibrate.py"
  - "aquakin/integrate/calibration/**"
  - "aquakin/integrate/seam.py"
  - "aquakin/integrate/_linalg.py"
  - "aquakin/integrate/fit.py"
  - "aquakin/integrate/fisher.py"
  - "aquakin/integrate/profile.py"
  - "aquakin/integrate/identifiability.py"
  - "aquakin/integrate/experiment_design.py"
  - "aquakin/integrate/candidate_design.py"
  - "aquakin/integrate/_transforms.py"
  - "aquakin/plant/calibrate.py"
  - "aquakin/plant/identifiability.py"
---

# Rules — calibration internals (`aquakin/integrate/calibrate.py`, `calibration/`, and their consumers)

### Module layout

- `integrate/calibrate.py` holds only `calibrate()`, and its `__all__` lists
  only what it provides (`calibrate`, the configs and the results). **It is
  not a re-export facade, for private names or for public ones defined
  elsewhere:** import `DifferentiationConfig` and friends from `aquakin`, and a
  helper from the `integrate/calibration/` module that defines it (or, from
  `aquakin.plant`, from `integrate/seam.py`), never through `calibrate.py`, so
  no name is kept alive there for an importer. **New calibration code
  goes in the `integrate/calibration/` module for its concern** (`config`,
  `observation`, `loss`, `coords`, `result`, `problem`, `objective`,
  `multistart`, `laplace`, `driver`, `criteria`, `saved`), never back into
  `calibrate.py`. A public name defined there (`criteria`'s `aic`, ...) is
  imported by `aquakin` and `aquakin.integrate` from that module.
- `calibration/*` modules must not import `calibrate.py` (it imports them), and
  `calibration/__init__.py` stays a docstring with no imports, so `fisher.py`
  can import the pieces without a cycle. Cross-module references that are only
  type annotations go under `TYPE_CHECKING`.
- `calibrate()` reads `_ReactorForwardModel` from `calibrate.py`'s namespace,
  where `test_calibrate.py` monkeypatches it; `_IRLS_MAX_ITER` lives with
  `_run_multistart`, where a unit test patches it. Keep both where the patch
  lands when moving code.
- `aquakin.plant` takes the calibration family's private names, like every
  private `integrate` name, from `integrate/seam.py` (the rule is in
  `plant.md`); the names are unchanged, so the plant tests' stubs on
  `aquakin.plant.calibrate` still apply.
- **A saved fit** (`saved`): `CalibrationResult.save`, `load_calibration` and
  the study's calibrate runner write `parameters.yaml` / `fit.yaml` /
  `covariance.csv` through the one `write_fit_files` (a study's `fit.yaml` is
  `fit_summary` alone). A new `CalibrationResult` field is saved and read there,
  or `test_calibration_save.py` fails; a layout change raises `FORMAT`. `save`
  names vectors by `problem.model`'s `param_index` / `species_index` / `models`,
  which `_PlantParamNamespace` must offer as a `CompiledModel` does.

### The optimizer's objective scale

- SciPy's stopping tests are absolute once the objective is below 1, so an
  `"mse"` fit on mol/L data (loss ~1e-13) used to stop at its start.
  **The optimizer sees the objective divided by `_ObjectiveBundle.loss_scale`**,
  set by `_scale_objective` in the multistart runner from the loss at each start via
  `_objective_scale` (`min(1, start loss)`, floored at `_SCALE_FLOOR` times
  `_CalibrationProblem.data_mean_square`; `"mse"` only, `data_mean_square` is
  `None` for the other losses). Both resolvers set `data_mean_square` with
  `_data_mean_square`. **`fit()` reuses the same two helpers** (its floor is
  the data's sum of squares, its loss being a sum); do not write another scale.
- **Only the optimizer sees the scaled objective.** The reported `loss`
  (multiplied back in `_fit_result`), the Laplace Hessian and the profile use
  the unscaled loss: do not fold the scale into `_data_term`,
  `_residual_parts` or the compiled programs (it is applied outside them, so
  it is not in `struct_key`). **Each start is scaled by its own starting loss,
  and starts are compared on the unscaled loss** (`fun * loss_scale`): a start
  then runs exactly as it would alone, so a start list returns the best of
  separate fits from its starts, whatever their order.

### Information criteria (`calibration/criteria.py`)

- `_fit_statistics` (in `_fit_result`, so reactor and plant share it) takes the
  log-likelihood as `loss - prior penalty - (N/2) log(2 pi)`, the penalty from
  `_add_prior_term`, the function the objective adds: **a new prior term goes
  there**, or the log-likelihood silently includes it.
- Both resolvers set `_CalibrationProblem.is_likelihood`, `n_observations`
  (non-NaN entries) and `observations_key` (observations and times only, not
  `sigma` or the loss, so different models' fits of one dataset compare).
- `k` = free rates + fitted IC coordinates + one noise level per output under
  `sigma="estimate"`. Only `"nll"` is a likelihood: no AIC for `"mse"`/`"wmse"`.

### `calibrate()` internal structure

`calibrate`'s tuning knobs are grouped into three public config dataclasses
(alongside `DifferentiationConfig`), so the top-level signature stays short:
`optimizer=OptimizerConfig(...)` (backend + multistart, either jittered
`n_starts` or an explicit `starts=[...]` list resolved by `_resolve_starts`,
each start moved inside the bounds by `_clip_start`, and run by
`_run_multistart(..., starts=)` through the one compiled objective; plus rate
bounds: the model-declared `bounds:` by default (`use_model_bounds=True`),
explicit absolute `bounds={name: (lo, hi)}`, and the optional `param_halfwidth`
box around `initial_params`, intersected with them; `_resolve_rate_bounds` maps
them through each transform into `_CalibrationProblem.rate_bounds`, and
`_optimizer_bounds` does the intersection. The bounds reach only SciPy, not the
compiled objective, so they are not in `struct_key`),
`laplace=` (a `bool` **or** a `LaplaceConfig(...)`), and
`free_ic=FreeICConfig(species=..., bounds=..., prior_log_std=..., fixed_total=...)`
(`bounds` one pair or a per-species dict, normalized by `_per_species_ic_bounds`
into per-pool arrays in `_CalibrationProblem.ic_bounds`; `fixed_total` fits
`m_ic - 1` log-ratio coordinates per dataset, `problem.m_ic_theta`, mapped to
pools by `problem.ic_pools` / `_simplex_pools` with the total read from the
runtime `C0`, and `_FisherSetup` mirrors it; `_optimizer_bounds` gives the
per-pool log bounds to the IC dims, or leaves them open for a split, so
`_at_bound_sides` flags a pool on its own bound and never a log-ratio; the
coordinate setup is `_resolve_ic_coords`, shared with the plant resolver, so
`Plant.calibrate` supports both; `profile_likelihood` drops the profiled species from a
bounds dict and refuses to profile a pool of a split). **`calibrate` and
`Plant.calibrate` share one driver; do not copy any of it into either entry
point.** Validation that a config can do on its own belongs in its
`__post_init__` (`OptimizerConfig` rejects an unknown `method` there, so no
entry point can skip it); the multistart checks are `_check_start_args`;
`_FitConfig.from_configs(optimizer, laplace, gradient=, ad_mode=, check_finite=,
compiled_cache=, ...)` is the only place the configs become a `_FitConfig`; and
`_run_fit(problem, fm, cfg, optimizer, ic_start, progress)` is the whole run
once the problem and forward model exist (build the objective, place the start,
check the start gradient, multistart, `_fit_result`). An entry point resolves
its own problem and forward model and nothing else. `Plant.calibrate` passes
`build_objective=` / `run_multistart=` as the names `plant/calibrate.py`
imported, because its tests stub those module names. The other
argument checks and result assembly that do not depend on the forward model are
shared helpers that `Plant.calibrate` calls too: `_check_sigma_arg` /
`_sigma_per_dataset` / `_coerce_sigma` / `_observed_counts` /
`_n_obs_per_output` (the `sigma` and missing-data checks, and the concentrated
likelihood's per-output counts), `_resolve_rate_bounds` (fed by
`_PlantParamNamespace.parameter_bounds` for a plant), `_initial_point` (bounds,
start clipping, the explicit start list) and `_fit_result` (fitted pools,
`at_bound` + warning, `sigma_estimate`, the Laplace posterior with
`at_bound_mask`, the `CalibrationResult`). Change them, not a copy, so the two
entry points cannot drift. The
same three configs serve `profile_likelihood`. Below that, the body is decomposed
([`integrate/calibration/`](aquakin/integrate/calibration/)). The argument
handling is resolved **once** into a `_CalibrationProblem` dataclass (datasets,
`free_indices`, resolved transforms, priors, free-IC blocks, `p0_full`) by
`_resolve_problem`, a short assembly of `_resolve_rate_params`,
`_resolve_datasets`, `_resolve_priors`, `_resolve_free_ic` and
`_resolve_rate_bounds`; the plant resolver shares `_check_start_in_domain`,
`_is_multi_dataset` and `_resolve_priors` with it rather than copying them; the fit run is then three free functions taking that
`problem`: `_build_objective` (the jit-cached objective / residual / Jacobian,
data threaded as *arguments* so a `profile_likelihood` sweep reuses one compiled
program via the shared `_compiled_cache`; everything the closures bake in -- the
loss, observations, `sigma`, observation times (fingerprinted into
`_CalibrationProblem.data_key`), the free set, transforms and priors -- is in
`struct_key`, so a cache shared across different data rebuilds instead of
mis-hitting; the reactor is not keyed, so never share a cache across reactors),
`_run_multistart`, and `_laplace_posterior`, whose `_LaplaceOut` fields map
one-to-one onto the `CalibrationResult` posterior fields (both call sites splat
`**laplace_out._asdict()`). Its Hessian is over the rates (pools at the MAP); with
`LaplaceConfig(include_ic=True)` it is over rates + log-space pools jointly
(`joint_cov` / `joint_hessian` / `joint_names`), and `posterior_cov` becomes the
rate block of the joint covariance (marginal over the pools), while
`hessian_unconstrained` stays the rate-rate block in both modes. The FD Hessian
reuses the fit's `_ObjectiveBundle.objective_value_and_grad` unless `laplace_dtmax`
changes the forward model; never write it a second objective. The standalone
`fisher_information` ([`integrate/fisher.py`](aquakin/integrate/fisher.py)) is
deliberately *not* built on `_CalibrationProblem`: a design has no observations
and each `Experiment` may carry its own reactor and observed species, which the
shared-observed-species problem cannot express. It uses the same coordinates
(rate transforms, log pools, `ic[e].<species>` names) and forward-mode AD
through `reactor.solve`; tests pin it and the joint Gauss-Newton Hessian to
one closed-form Fisher information — keep the two in step if either changes. `identifiability`
([`integrate/identifiability.py`](aquakin/integrate/identifiability.py)) is pure
NumPy on the matrix (no solves); it reads `joint_hessian` / `joint_names` from a
`CalibrationResult` when present, else the rate `hessian_unconstrained`, and its
`eig_keep` only *flags* weak directions. `calibrate`'s Laplace covariance
(`_laplace_covariance`) is the full inverse of the Hessian and shares
identifiability's null convention (`_NULL_RTOL`, `_NULL_LOADING_TOL`, which both
import from `integrate/_linalg.py`): an exactly null direction is left out of
`posterior_cov` and its involved parameters get an infinite std plus a
`RuntimeWarning`. `FisherInformation.covariance` (behind `std()`) uses the
same `_NULL_RTOL` to raise `LinAlgError` on a singular matrix; do not rely on
`np.linalg.inv` raising, which it does only on an exactly zero LU pivot and not
on every LAPACK build. `LaplaceConfig.eig_keep` is deprecated and ignored. A quantity counts as
"involved" in an exactly-null direction (rel. eigenvalue ≤ 1e-10) only if its
loading there exceeds `null_loading_tol` (default 1e-2): a null eigenvector is
only accurate to the sensitivities' round-off, and a 1e-6 cut marked
well-determined pools infinite.
`fisher.py`'s `_FisherSetup` holds the static part (validation, reactors,
observed columns, noise, priors) and exposes `jacobian` / `information` as pure
JAX functions of `(p, tobs_list, C0_list)`; pools are linearized at each
`C0`. The sigma/mask resolution (`_observation_noise`), the active
priors (`_active_priors`) and `_resolve_transforms` are free helpers that
`plant/identifiability.py`'s `plant_fisher_information` shares, so the reactor
and plant matrices cannot drift. Likewise `profile.py`'s sweep is factored into
`_check_profile_args` / `_cold_starts` / `_point_optimizer` / `_guarded` /
`_profile_sweep` / `_profile_result`, which `plant_profile_likelihood` reuses
with `Plant.calibrate` as the inner fit; change them, not a copy. `optimize_experiment`
fixes `t_end` per optimized experiment so sampling times can be traced, and runs
SLSQP on `(value, jacfwd)` of the criterion over a [0, 1]-scaled design vector
(`param_draws` `vmap`ped). That program is held in `cached_jitted_solver` under
`_FisherSetup.structure_key()`: anything new `_FisherSetup.information` bakes
in must join that key, or a repeat call reuses a stale program.
`fisher_information_from_fn` (in `fisher.py`) is the matrix-only path for a
forward model that is not a reactor: AD Jacobian of `fn(theta)`, the same
transforms and `FisherInformation` result, and `groups=` returning
per-candidate matrices. `candidate_design.py`'s `design_from_candidates` takes
such a pool and scores designs with `experiment_design.py`'s `_criterion` (its
`rtol=0` mode for the ridge-regularized selection used when the base is
singular) and `_subset_indices`, so its values agree with `design_criterion`.
Its relaxed solver is projected gradient in NumPy with an analytic gradient
(D, Ds, A only) and reports the Frank-Wolfe gap; `auto` enumerates the
feasible designs when there are at most `exhaustive_limit`, since exchange
alone stops at local optima on about one random small pool in ten.
`observation_units` (calibrate / profile, and `Plant.calibrate`) is applied
before the problem is resolved, by the shared `_observation_unit_factors(columns,
units)` (one factor per observed column from `conversion_factor(declared, model
unit)`, `None` when all are 1) and `_convert_observations(observations, sigma,
factors)` (multi-batch lists, a 1-D single column, and every `sigma` form; a
scalar `sigma` becomes per-column when the factors differ; `"estimate"` /
`None` pass through). Everything downstream, including `sigma_estimate` and the
posterior, is therefore in model units; it is refused with `observation_fn`.
The plant builds its columns from each observable's stream model.
`observation_fn` (calibrate / profile / `Experiment`) goes through the shared
`observe(ys, p, species_idx, fn, t=)` helper in `calibration/observation.py`, used by both
`_predict` and `_FisherSetup.jacobian`; a three-argument `fn(t, C, p)`
(`observation_takes_time`, by signature) gets the sample times in `t_obs`'s
unit exactly as given (`_CalibrationProblem.obs_times`; `_FisherSetup.tobs`,
converted to native time by `_FisherSetup.time_factor` only at the solve), never
native times divided back. `NaN` observations are masked in `_build_loss` /
`_build_residual` (`_build_masked_*`, only when a `NaN` is present, so the
unmasked path is unchanged); `Experiment(mask=)` sets the unmeasured entries'
sigma to `inf` in `_FisherSetup` and is part of the design `_structure_key`; it is a defaulted field on
`_CalibrationProblem` (and part of `struct_key`), so plant calibration, which
builds its own problem, is unaffected. `CalibrationResult.posterior_draws(model, ...)`
is the single Laplace sampler; `predictive_band` propagates its rows (or the given
`draws=`). MCMC (`sample_posterior`) is in `integration-sampling.md`. **The
forward-solve is a seam**: `_ReactorForwardModel` is the *only* piece that touches the fitted object
— it turns `(params, C0)` into an observed-species trajectory (via the reactor's
diffrax `solve`, or the cap-free ESDIRK adjoint on `gradient="stable_adjoint"`),
and exposes `solve_trajectory` + `with_dtmax`. Everything else is generic over
that contract. A `Plant` has a different solve contract (flat assembled state,
`plant.solve(t_span, …)`, no `.model`/`.conditions`), so **plant calibration is a
*new* forward model, not a change to the generic layer**: `plant.calibrate(...)`
([`plant/calibrate.py`](aquakin/plant/calibrate.py)) supplies a
`_PlantForwardModel` (`plant.solve` + stream reconstruction) and a
`_PlantParamNamespace` adapter, and reuses `_build_objective` / `_run_multistart`
/ `_laplace_posterior` unchanged — the payoff of the seam. **Solve failures
mid-fit:** `_run_multistart` treats a start whose forward solve raises at runtime
as a failed start. `_is_solve_failure` recognizes these: equinox's runtime error,
or a JAX runtime / `ValueError` carrying the host-callback failure that
`equinox.error_if` raises from compiled code. Such a start is skipped with one
`RuntimeWarning` and the best remaining start is kept. If no start survives, a
`RuntimeError` with the `param_halfwidth` / initial-guess / `free_ic`-bounds
remedies is raised, chained to the first failure. Any other exception still
propagates unchanged. `_check_start_gradient` turns a solve failure at the
starting guess into a similar error. Both reach `Plant.calibrate` through the
shared functions.
