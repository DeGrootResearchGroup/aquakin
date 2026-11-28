---
paths:
  - "aquakin/integrate/calibration/sampling.py"
  - "aquakin/plant/posterior.py"
---

# Rules — posterior sampling (`calibration/sampling.py`, `plant/posterior.py`)

`aquakin.sample_posterior` and `Plant.sample_posterior`: a random-walk
Metropolis on the posterior a calibration maximized. The calibration internals
it builds on are in `.claude/rules/integration-calibrate.md`.

### The density is the fit's; the Jacobian is the `prior_space` choice

- **The log density is `-_objective_fn(problem, fm)(theta, *problem.data)`,
  plus `_log_jacobian(...)` only with `prior_space="physical"`** (built in
  `_log_density`). Do not write a second data term, prior or transform for
  the sampler: a change to the objective must reach the fit and the sampler
  together, which only one function guarantees.
- **The default `prior_space="transformed"` adds no Jacobian**, so the
  sampled density is the one the MAP and the Laplace Hessian describe and
  MCMC, Laplace and the profile likelihood are comparable. The default is the
  owner's decision: do not change it or add the Jacobian back to it.
- **The Jacobian of the rate transforms belongs to the sampled density only.**
  Never add it to `_objective_fn`, `_add_prior_term` or `_data_term`: the MAP,
  the reported `loss`, `log_likelihood` and the Laplace Hessian are defined
  without it.
- `prior_space` is checked by `_check_prior_space` inside `_run_sampler`, so
  both entry points take it and refuse a bad value the same way.
- The fitted pools carry no Jacobian: their prior (`ic_prior_log_std`, or flat)
  is defined on their log / log-ratio coordinates, which is what is sampled.
- `loss="nll"` only (`problem.is_likelihood`). The `sigma="estimate"`
  concentrated likelihood is sampled as is: it is the posterior with each
  output's noise integrated out under `p(sigma) ∝ 1/sigma`; do not "fix" it by
  sampling `sigma`.

### The problem comes from the fit, not from the arguments again

- **`_fit_result` stores `_FitRecord(problem, forward_model, theta_map)` on
  `CalibrationResult._fit_record`** (`repr=False, compare=False`); the sampler
  reads the problem, the MAP and the support from it and never re-resolves the
  `calibrate` arguments, so it cannot sample other data than were fitted.
- The support is the fit's own `_optimizer_bounds`: a proposal outside it is
  rejected without a solve.
- The reactor entry point rebuilds only the forward model from the reactor it
  is given (forward solves, no adjoint); the plant entry point uses the
  recorded `_PlantForwardModel`. Both call `_check_sampling_result` and
  `_run_sampler`, which the plant takes from `integrate/seam.py`; dispatch is on
  the recorded forward model's type. Put new behaviour in those shared
  functions, not in an entry point.

### Evaluation

- **Each chain's point is one call of the jitted scalar density, not a `vmap`
  over chains**: a batched adaptive solve steps every chain as finely as the
  stiffest, which measured slower than the loop on CPU, and a per-point call
  lets one failed solve reject one proposal.
- A failed solve is a rejection only through `_is_solve_failure`; any other
  exception propagates. A non-finite density is a rejection.
- The sampler takes no gradient, so it works with every forward model and
  every `diff` the fit used.
- `progress=` reaches `_metropolis` through `_run_sampler` (both entry points),
  reported once per step of all chains, outside the jitted density. The draws
  must not depend on it: a study run reports through it and a cancel raises
  from it.

### Output and diagnostics

- `PosteriorSamples.draws` has the layout of `posterior_draws` (full parameter
  vectors, fixed entries at the calibrated values), so `predictive_band(draws=)`
  and `param_draws` take either.
- R-hat and bulk ESS follow Vehtari et al. (2021): rank-normalized split
  chains, the folded R-hat taking the median of the unsplit draws, and Geyer's
  initial monotone sequence. Keep them NumPy-only and free of a sampler
  dependency.
