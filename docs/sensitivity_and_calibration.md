# Sensitivity & calibration

Every solve in `aquakin` — a single reactor or a whole plant — is
**differentiable**. Because rate constants are always passed in as arguments
(never baked into the rate functions), you can take exact gradients of any
output with respect to parameters, conditions, or initial states, and feed them
to sensitivity analysis, parameter estimation, and design optimisation.

A plain forward simulation never needs any of this; reach for it when you want
to know *how* an output depends on inputs, or to fit a model to data.

## Differentiating a solve

The simplest case is an ordinary JAX gradient through a reactor solve:

```python
def loss(params):
    sol = reactor.solve(C0, t_span=(0.0, 600.0), t_eval=t_eval, params=params)
    return sol.C_named("BrO3-")[-1]

g = jax.grad(loss)(params)      # d[BrO3-]_final / d params
```

```{warning}
**Silent non-finite reverse gradients on stiff models.** A reverse-mode
gradient (`jax.grad`) taken *directly* through a stiff model's solve (ASM, ADM,
WATS) can return silent `NaN`/`Inf` when the integrator step is uncapped — no
exception is raised, so the bad gradient flows into your optimizer.

`aquakin.calibrate` and `aquakin.sensitivity` guard against this for you. If you
write your own loss and optimizer over a stiff model, either cap the step with
`integrator=aquakin.IntegratorConfig(dtmax=...)`, use the cap-free forward
sensitivity below, or wrap the result with `aquakin.check_finite_gradient(...)`
to get an actionable error instead of silent `NaN`.
```

## Local sensitivity

`aquakin.sensitivity` computes the gradient of a scalar output with respect to
every parameter and condition field in one call, and ranks them:

```python
sens = aquakin.sensitivity(
    reactor, C0,
    output_fn=lambda sol: sol.C_named("BrO3-")[-1],
    t_span=(0.0, 600.0), t_eval=t_eval,
)
sens.doutput_dparams                 # (n_params,) gradient w.r.t. each parameter
sens.doutput_dconditions["pH"]       # sensitivity to a condition field
sens.ranked_params()                 # [(name, value), ...] most influential first
```

## Forward sensitivity (cap-free stiff gradients)

Differentiating *through* a stiff reaction-model solve with ordinary AD goes
non-finite above an integrator-step threshold, and the usual workaround — a
global `dtmax` cap — forces tiny steps over the whole solve. `solve_sensitivity`
avoids both: it integrates the sensitivity `S = dC/dθ` **alongside** the state
and lets the adaptive controller bound the sensitivity error too, so the step
tightens only where the sensitivity is stiff and the result is exact with no cap.

```python
model = aquakin.load_model("uv_h2o2")
reactor = aquakin.BatchReactor(model, model.default_conditions())

sol, S = reactor.solve_sensitivity(
    model.default_concentrations(), t_span=(0.0, 5.0),
    t_eval=jnp.linspace(0.0, 5.0, 6), params=model.default_parameters(),
    sens_params=["H2O2_photolysis.k_photo", "OH_target.k_OH_target"],
)
# sol : the usual solution; S : dC/dθ, shape (n_t, n_species, n_sens_params)

# A wrapper with by-name accessors:
res = aquakin.forward_sensitivity(
    reactor, model.default_concentrations(), params=model.default_parameters(),
    t_span=(0.0, 5.0), t_eval=jnp.linspace(0.0, 5.0, 6),
    sens_params=["H2O2_photolysis.k_photo"])
res.dC_dparam("target", "H2O2_photolysis.k_photo")   # (n_t,)
```

`solve_sensitivity` is available on `BatchReactor`, `PlugFlowReactor`, and
`BiofilmReactor`, and the plant exposes `Plant.solve_sensitivity(...)`. For a
scalar-loss gradient over many parameters (the calibration case) reverse mode is
cheaper; `Plant.solve` uses a cap-free stable discrete adjoint automatically, so
`jax.grad` through a stiff plant is finite by default with nothing to tune.

## Global sensitivity (DGSM)

`aquakin.dgsm` is a derivative-based global sensitivity measure: it averages the
squared partial derivative of an output over quasi-random samples of the input
ranges, bounding the Sobol total-order index for each input. You supply a
function that maps a named input vector to a scalar (or vector) output:

```python
res = aquakin.dgsm(fn, ranges, input_names=names, n_samples=64, seed=0)
res.sobol_total_bound     # (d,) upper bound on the Sobol total index per input
res.ranked()              # [(name, bound), ...] most influential first
```

The `ad_mode=` argument selects forward or reverse AD for the per-sample
derivatives (identical results — purely a performance choice); forward mode is
faster when there are many outputs or the reverse adjoint is stiff-inflated.

## Fitting parameters to data

Two entry points fit a model to observations:

- `aquakin.fit` — a point estimate by box-constrained least squares.
- `aquakin.calibrate` — a MAP fit with parameter transforms, priors, deterministic
  multistart, and an optional Laplace posterior for uncertainty.

```python
result = aquakin.calibrate(
    reactor, C0, observations, t_obs,
    free_params=["O3_Br_direct.k1", "O3_decay.k2"],
    loss="nll", sigma=sigma,          # measurement noise -> a proper posterior
    laplace=True,                     # add a Laplace posterior approximation
    optimizer=aquakin.OptimizerConfig(n_starts=24, seed=0),  # multistart
)
result.params_named       # MAP estimate, in physical space, by name
result.params_named_std   # marginal standard deviations
result.converged
```

`OptimizerConfig(n_starts=...)` adds jittered restarts around one start. To run
a fixed set of starts instead, such as a grid, pass them as `starts`, each a
full parameter vector or a `{name: value}` dict over the free parameters (the
others start from `initial_params`). Every start runs through one compiled
objective and the best fit is kept:

```python
starts = [{"r1.k1": k1, "r2.k2": k2} for k1 in (0.01, 0.1, 1.0) for k2 in (0.1, 1.0)]
result = aquakin.calibrate(reactor, C0, observations, t_obs, free_params,
                           optimizer=aquakin.OptimizerConfig(starts=starts))
```

`calibrate` keeps each free parameter inside the `bounds:` declared on it in
the model, as `fit` does. The optimiser works in each parameter's transformed
space, so a declared bound on a `positive_log` parameter becomes a bound on its
logarithm, and every multistart start searches the same region. Pass explicit
absolute bounds, which replace the declared ones, or an extra box around the
starting value:

```python
optimizer = aquakin.OptimizerConfig(
    bounds={"hetero.muH": (0.2, 10.0), "hetero.KS": (1.0, None)},  # None: open side
    param_halfwidth=np.log(100.0),    # optional: within 100x of the start as well
)
```

`OptimizerConfig(use_model_bounds=False)` ignores the declared bounds. A
starting value outside the bounds, including each entry of a `starts` list, is
moved to the nearest bound, with a warning.

A parameter that ends on a bound is listed in `result.at_bound` (for example
`{"hetero.muH": "lower"}`), with a warning. The Laplace approximation is not
valid on an active bound, so its `params_named_std` is `nan`, it has zero
variance in `posterior_cov` (posterior draws keep it on the bound), and the
other parameters' standard deviations are conditional on it staying there.
`aquakin.identifiability(result)` carries the same flag. A profile likelihood
gives the one-sided interval for such a parameter. Fitted initial pools on
their `FreeICConfig` bounds are flagged the same way, as `"ic[k].<species>"`.

`observations` is shape `(n_t,)` for one species or `(n_t, n_observed)` for
several; `t_obs` is in the model's native time unit unless you pass `time_unit=`.
Pass **lists** of `C0`/`observations`/`t_obs` for a joint multi-batch fit, and
`free_ic=aquakin.FreeICConfig([...])` to fit unmeasured initial pools alongside
the rate constants.

When a measurement is not a single species but a function of several (a UV
absorbance from several organic pools, a probe-compound ratio), pass
`observation_fn` instead of `observed_species`. It maps one time point's
concentration vector and the parameter vector to the measured outputs, and must
be written with `jax.numpy` so it can be differentiated:

```python
uv254 = lambda C, p: 0.02 * C[i_OMf] + 0.05 * C[i_OMs]
result = aquakin.calibrate(reactor, C0, uv_data, t_obs, free_params,
                           observation_fn=uv254, loss="nll", sigma=0.005)
```

The columns of `observations` follow the function's outputs, and `sigma` may
give one value per output. `profile_likelihood` and `aquakin.Experiment` accept
`observation_fn` the same way.

A function of three arguments, `observation_fn(t, C, p)`, also receives the
sample time, in the unit of `t_obs`. Use it for a measurement whose mapping
changes with time, such as a drifting sensor:

```python
drifting = lambda t, C, p: C[i_B] * (1.0 + 0.05 * t)
```

A missing observation is a `NaN` in `observations`. It is left out of the fit,
so outputs sampled at different times can share one `t_obs` grid. For example,
a total measured only at the start is a second output that is `NaN` except at
`t = 0`:

```python
obs = np.stack([b_series, np.where(t_obs == 0.0, total0, np.nan)], axis=1)
fn = lambda C, p: jnp.stack([C[i_B], C[i_A] + C[i_B]])
result = aquakin.calibrate(reactor, C0, obs, t_obs, free_params,
                           observation_fn=fn, loss="nll", sigma=0.01)
```

With `"mse"` or `"wmse"` the mean is over the observed entries. For a planned
experiment, `aquakin.Experiment(mask=...)` (`True` where an output is measured)
marks the same thing for `fisher_information` and experiment design.

### Unknown noise level

A Laplace standard deviation or a profile-likelihood interval needs
`loss="nll"` with the measurement noise `sigma`. When the noise level is not
known, as with most real data, let `calibrate` estimate it:

```python
result = aquakin.calibrate(reactor, C0, observations, t_obs, free_params,
                           loss="nll", sigma="estimate")
result.sigma_estimate     # estimated noise std of each observed output
result.params_named_std   # Laplace std at the estimated noise level
```

This is the concentrated likelihood: the noise std of each output is replaced
by its best estimate, the RMS residual `sqrt(SSE_j / N_j)` over its observed
entries (pooled over the datasets of a multi-batch fit; a `NaN` is not counted), and the objective becomes the `"nll"` loss at
that estimate, `sum_j (N_j / 2) log(SSE_j / N_j) + N / 2`. It is one fit, in
place of fitting with `"mse"` and refitting with `sigma` set to the residual
RMS.

The profile-likelihood threshold reads the same way in both cases: a 95 %
interval is where the profile rises by 1.92 (`0.5 * chi2_{1, 0.95}`). With
`sigma` known, the profile is the `"nll"` loss itself. With
`sigma="estimate"` and one output, the rise is `(N / 2) log(SSE / SSE_min)`,
so the interval is where the sum of squares grows by a factor
`exp(2 * 1.92 / N)`.

### Uncertainty: posterior-predictive bands

With `laplace=True`, sample the posterior and propagate it through a solve to get
per-timepoint predictive envelopes (the `C0` may be a held-out validation batch):

```python
band = result.predictive_band(reactor, C0, t_eval, n_draw=200,
                              percentiles=(2.5, 97.5))
band.median, band.lo, band.hi        # (n_t, n_species) envelopes
```

### Uncertainty when initial pools are fitted

By default the Laplace covariance covers the rate parameters only, with any
fitted initial pools held at their best-fit values. When a rate constant and a
pool trade off against each other (for example `k` and `[X]0` when only their
product shapes the data), that understates the rate uncertainty. Ask for the
joint covariance instead:

```python
result = aquakin.calibrate(
    reactor, C0, observations, t_obs, free_params,
    loss="nll", sigma=sigma,
    free_ic=aquakin.FreeICConfig(["X"]),
    laplace=aquakin.LaplaceConfig(method="gauss_newton", include_ic=True),
)
result.joint_names        # ["<rate names>", "ic[0].X", ...]
result.joint_cov          # rates + log-space pools, unconstrained space
result.joint_hessian      # with loss="nll" and gauss_newton: the Fisher information
result.params_named_std   # rate std devs, now marginal over the pools
result.ic_named_std       # pool std devs in physical units, per dataset
```

Strong correlation between a rate and a pool in `joint_cov` is a sign that the
experiment cannot separate them.

Pools of very different size can take their own bounds,
`FreeICConfig(["SS", "XS"], bounds={"SS": (1, 300), "XS": (1, 500)})`; a
species left out keeps the default `(1e-3, 1e4)`. A pool that ends on its own
bound is listed in `result.at_bound` as `"ic[k].<species>"`, with a `nan`
standard deviation.

When the total of some pools is known and only its split is uncertain (a
measured total COD divided between readily and slowly biodegradable
substrate, say), hold the total fixed. Put the total into `C0`, with any split
as the starting guess, and fit only the split:

```python
C0 = C0.at[i_SS].set(0.5 * cod).at[i_XS].set(0.5 * cod)   # sum = the known total
result = aquakin.calibrate(
    reactor, C0, observations, t_obs, free_params,
    loss="nll", sigma=sigma,
    free_ic=aquakin.FreeICConfig(["SS", "XS"], fixed_total=True),
    laplace=aquakin.LaplaceConfig(method="gauss_newton", include_ic=True),
)
result.ic_named           # [{"SS": ..., "XS": ...}], summing to the total
result.ic_fractions       # [{"SS": f, "XS": 1 - f}]
result.ic_fractions_std   # std devs of the fractions
```

The split is fitted through the log-ratios `log(c_j / c_0)` of the pools to the
first listed one, which keep every pool positive and the sum exact; these are
the `"ic[k].XS:SS"` rows of `joint_cov`. `fisher_information` takes the same
`FreeICConfig`. The log-ratios are unbounded, so a split never appears in
`result.at_bound`. `profile_likelihood` cannot profile a pool of a fixed-total
split.

### Fisher information of a design

`aquakin.fisher_information` evaluates the Fisher information of a set of
experiments at assumed parameter values, with no data. Use it to check, before
running an experiment, whether it can estimate the parameters you care about,
and to compare candidate designs:

```python
design = [
    aquakin.Experiment(C0_low, t_obs=jnp.linspace(1, 60, 20), sigma=0.02,
                       observed_species=["O3"]),
    aquakin.Experiment(C0_high, t_obs=jnp.linspace(1, 30, 20), sigma=0.02,
                       observed_species=["O3"], reactor=reactor_ph8),
]
fim = aquakin.fisher_information(
    reactor, design, free_params,
    params=assumed_params,
    free_ic=aquakin.FreeICConfig(["IOD"]),   # also estimate a pool per experiment
)
fim.names        # rates, then "ic[e].<species>" per experiment
fim.matrix       # in the coordinates calibrate fits in (log for positive_log, pools)
fim.physical()   # with respect to the physical values
fim.std()        # Cramér-Rao lower bound on each standard deviation
```

The information of independent experiments adds, so an experiment that sharpens
a poorly determined parameter shows up as a large drop in its `std()`. A
singular matrix (`fim.std()` raises) means some combination of the quantities
cannot be determined from the design at all. Each experiment may carry its own
`reactor` to represent different operating conditions. At a calibration's
optimum, with `loss="nll"`, the matrix equals the Gauss-Newton Laplace Hessian
(`joint_hessian` with `include_ic=True`).

### Identifiability report

`aquakin.identifiability` analyses a Fisher information matrix and reports
which parameter combinations the design can and cannot determine. It accepts
the output of `fisher_information`, a `CalibrationResult` fitted with
`laplace=` (use `loss="nll"` and `method="gauss_newton"` so its Hessian is the
Fisher information), or a bare matrix with `names=`:

```python
rep = aquakin.identifiability(fim)
for d in rep.unidentifiable:          # directions the data barely constrain
    print(d.relative_eigenvalue, d.loadings)
rep.correlation                       # correlation matrix, ordered as rep.names
rep.relative_std                      # Cramér-Rao relative standard errors
rep.collinearity_index(["r1.k", "ic[0].X"])
rep.largest_identifiable_subset(threshold=15.0)
```

Each eigen-direction carries its eigenvector loadings by name. In the log
coordinates used for positive parameters and initial pools, equal and opposite
loadings on a rate constant and a pool (for example `+0.71` on `k` and `-0.71`
on `ic[0].X`) mean the data see only their product `k [X]0`. Such a pair should
be lumped into one parameter or separated by a new experiment. A direction is
flagged when its eigenvalue is below `eig_keep` (default `1e-2`) times the
largest. Flagging is a diagnostic only. The standard deviations here, like
`params_named_std` from `calibrate`, come from the full inverse of the
information, so a weakly identified quantity has a large standard deviation,
and a quantity on an exactly null direction has an infinite one.

The collinearity index of a subset is `1 / sqrt(λ_min)` of the information
normalised to unit diagonal. It is 1 when the subset's sensitivities are
orthogonal and grows without bound as they become linearly dependent,
regardless of how sensitive the output is to each parameter.
`identifiable_subsets` lists every subset below a threshold (default 15),
largest first.

### Identifiability: profile likelihood

The Laplace covariance is a local, quadratic approximation. For an exact
identifiability analysis — which distinguishes a well-constrained parameter from
one the data cannot pin — use `aquakin.profile_likelihood`. It fixes one quantity
on a grid, re-optimises everything else at each point, and reports the confidence
interval where the profile rises by the likelihood-ratio threshold:

```python
prof = aquakin.profile_likelihood(
    reactor, C0, observations, t_obs, free_params,
    grid=grid, profile_param="k_s0_anox_f",
    loss="nll", sigma=sigma, warm_start=True)
prof.mle    # grid value at the profile minimum
prof.ci     # (lo, hi); None on a side means the parameter is unidentified there
```

### Designing experiments

For a complete study using these tools, see
[Worked example: designing an experiment](worked_example_design.md).

Once you can compute the information of a design, you can choose the design
that gives the most. A design is scored by a criterion of its Fisher
information `F`:

| Criterion | Value | Better |
|---|---|---|
| `"D"` | `log det F` | larger |
| `"A"` | `trace(F⁻¹)`, the summed variance bound | smaller |
| `"E"` | smallest eigenvalue of `F` | larger |
| `"modified_E"` | condition number of `F` | smaller |

D is the usual default: it shrinks the joint confidence region and does not
depend on the units of the parameters. Pass `subset=[...]` to score only the
information about some quantities, treating the others (for example nuisance
initial pools) as unknown.

```python
aquakin.design_criterion(fim, "D", subset=["r1.k1"])

# Choose the most useful next experiment from a list of candidates.
ranked = aquakin.rank_experiments(reactor, candidates, free_params,
                                  base=experiments_already_run)
ranked[0].index, ranked[0].improvement

# Or optimise the sampling times and initial concentrations directly.
res = aquakin.optimize_experiment(
    reactor,
    aquakin.Experiment(C0, t_obs=jnp.linspace(1, 30, 6), sigma=0.02,
                       observed_species=["O3"]),
    free_params,
    time_bounds=(0.5, 60.0), min_spacing=2.0,   # in the experiment's time unit
    initial_bounds={"O3": (0.5, 5.0)},          # dose range to search
)
res.experiments[0].t_obs, res.experiments[0].C0
```

`optimize_experiment` keeps the number of samples fixed and moves them within
the bounds by gradient, keeping consecutive samples at least `min_spacing`
apart.

The optimal design depends on the parameter values you assume, which are
uncertain before the experiment. To make the design robust to that, pass
`param_draws`, an array of parameter vectors drawn from a prior or a previous
calibration's posterior, and the criterion is averaged over them. A calibration
fitted with `laplace=` provides posterior draws directly:

```python
draws = result.posterior_draws(reactor.model, n_draw=50, seed=0)  # (50, n_params)
res = aquakin.optimize_experiment(reactor, experiment, free_params,
                                  param_draws=draws, time_bounds=(0.5, 60.0))
```

## Calibrating a plant

`Plant.calibrate` is the plant analogue of `aquakin.calibrate`: it fits by-name
plant parameters against one or more measured effluent streams, reusing the same
machinery (transforms, priors, multistart, Laplace) behind the plant's cap-free
adjoint — so the gradient is finite for a stiff plant with no `dtmax` to tune.

```python
result = plant.calibrate(
    observations,               # (n_t, n_channels) measured data
    t_obs,
    ["asm1.muH", "asm1.bH"],    # plant parameters to fit (see plant.parameter_names())
    target="effluent",          # a registered stream, or "unit.port"
    observed_channels=["SNH", "SNO"],
    y0=bsm2_warm_start(plant),  # warm start recommended for a stiff plant
)
result.params_named
```

Fit against several streams at once with `observables=[PlantObservable(...), ...]`,
fit assembled-state initial conditions with `free_ic=FreeICConfig([...])`, and run
a joint multi-batch fit by passing list-valued `observations`/`t_obs`/`y0`.

Everything above about bounds, initial pools and noise applies to a plant fit
too. The free parameters stay inside the `bounds:` their models declare, and
`OptimizerConfig(bounds=...)` keys are plant parameter names such as
`"asm1.muH"`. A fit that ends on a bound lists it in `result.at_bound`, with a
`nan` Laplace standard deviation. The fitted initial pools are named
`"unit.species"` (also the keys of a per-pool `FreeICConfig(bounds={...})`), and
`fixed_total=True` fits their split of the total in `y0`. With `loss="nll"`,
`sigma="estimate"` estimates the noise level of each observed channel.

## Uncertainty propagation and design

Three higher-level tools share the same `fn(inputs) -> output` contract as `dgsm`:

- `aquakin.monte_carlo` — propagate uncertain inputs (uniform / normal /
  lognormal marginals) through the model to an output ensemble and percentiles.
- `aquakin.compare_scenarios` — run several named input sets side by side and
  tabulate the KPIs.
- `aquakin.optimize_design` — minimise a cost/energy objective over bounded design
  variables subject to effluent constraints, using AD gradients (constrained NLP).

```python
opt = aquakin.optimize_design(
    objective=lambda x: x[0],
    bounds=[(0.5, 2.0)], input_names=["muAOB"],
    constraints=[aquakin.Constraint(fn=eff_nh4, upper=6.5, name="eff_NH4")],
    x0=[1.5])
opt.x_named; opt.objective; opt.feasible
```

## Choosing the integrator

The integrator and step machinery are configured with one value object,
`aquakin.IntegratorConfig`, passed as `integrator=` to a reactor or `plant.solve`:

```python
aquakin.IntegratorConfig(order=3, factormax=3.0, colored_jacobian="auto",
                         dtmax=None, max_steps=100_000, solver=None)
```

- `order` — the ESDIRK method: `3` → `Kvaerno3` (the fast default), `5` →
  `Kvaerno5` (higher order, more robust).
- `factormax` — caps the PID step-size growth.
- `dtmax` — caps the step; set it **only** for a reverse-mode gradient taken
  *through* a stiff solve. It is always in the model's native time unit.
- `colored_jacobian` — sparse coloured-AD Jacobian materialisation (`"auto"` by
  default), which speeds up large stiff plants such as BSM2.

`rtol`/`atol` stay separate arguments (the accuracy contract, not the machinery).
The default `IntegratorConfig(order=3, factormax=3.0)` is the fast stack and is a
good starting point; raise `order` to 5 if a solve struggles to converge.

If a solve never needs gradients, `plant.solve(..., forward_fast=True)` runs a
lean forward-only integrator that skips the adjoint machinery for a markedly
faster compile and run, at the same accuracy.
