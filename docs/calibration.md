# Calibration

Fit a model's parameters to measured data, check how well the data determine
them, and compare model structures. Reactor fits are on this page; a whole
plant is fitted the same way ([Calibrating a plant](plant_calibration.md)).

The calibration examples share the setup of
[Sensitivity analysis](sensitivity_and_calibration.md#the-examples-on-this-page):
the shipped `ozone_bromate` model in a batch reactor, with synthetic
observations of ozone and bromate made from the default parameters plus 3 %
noise. Run this block first; every block below then runs as written, in order.

```python
import aquakin
import jax
import jax.numpy as jnp
import numpy as np

model = aquakin.load_model("ozone_bromate")
reactor = aquakin.BatchReactor(model, model.default_conditions())
C0 = model.default_concentrations()
params = model.default_parameters()
i = model.species_index                          # species name -> column

free_params = ["O3_Br_direct.k1", "O3_decay_OH.k_dec"]
i_free = jnp.array([model.param_index[n] for n in free_params])
observed = ["O3", "BrO3-"]

t_obs = jnp.linspace(30.0, 600.0, 12)            # seconds
truth = reactor.solve(C0, t_span=(0.0, 600.0), t_eval=t_obs, params=params)
clean = jnp.stack([truth.C_named(s) for s in observed], axis=1)   # (12, 2)
sigma = 0.03 * clean.max(axis=0)                 # one noise level per species
rng = np.random.default_rng(0)
observations = clean + sigma * rng.standard_normal(clean.shape)
```

## Fitting parameters to data

Two entry points fit a model to observations:

- `aquakin.fit`: a point estimate by box-constrained least squares.
- `aquakin.calibrate`: a maximum a posteriori (MAP) fit with parameter
  transforms, priors, deterministic multistart and a Laplace posterior for the
  uncertainty (`laplace=True` by default).

```python
result = aquakin.calibrate(
    reactor, C0, observations, t_obs, free_params,
    observed_species=observed,
    loss="nll", sigma=sigma,          # Gaussian likelihood with known noise
    optimizer=aquakin.OptimizerConfig(n_starts=4, seed=0),   # multistart
)
result.params_named       # MAP estimate, in physical units, by name
result.params_named_std   # Laplace standard deviations
result.converged          # True
```

`observations` has shape `(n_t,)` for one observed output or `(n_t,
n_observed)` for several. `t_obs` is in the model's time unit unless you pass
`time_unit=`. Pass **lists** of `C0`, `observations` and `t_obs` for a joint fit
of several batches.

### Loss and measurement noise

`loss=` chooses what is minimized:

- `"mse"` (the default): the mean squared error.
- `"wmse"`: the mean squared error weighted by `1 / sigma²`; needs `sigma`.
- `"nll"`: the Gaussian negative log-likelihood with noise `sigma` (one value,
  or one per observed output). This is the one to use: with it, the Laplace
  standard deviations and profile-likelihood intervals have their proper scale.

```{note}
`"mse"` is in the square of the data's unit, so on concentrations near
1e-6 mol/L it is of order 1e-12. The optimizer's stopping tolerance
(`OptimizerConfig(tol=)`) is therefore applied relative to the misfit at the
start, which gives the same fit whatever unit the data are in; the reported
`loss` and the Laplace posterior are of the `"mse"` itself. `aquakin.fit`
scales its sum of squares the same way. That posterior
assumes a unit noise level on the data's own scale, so its standard deviations
are only meaningful with `"nll"` (or `"wmse"`) and a `sigma`, or
`sigma="estimate"` below.
```

When the noise level is not known, as with most real data, let `calibrate`
estimate it:

```python
result_est = aquakin.calibrate(reactor, C0, observations, t_obs, free_params,
                               observed_species=observed,
                               loss="nll", sigma="estimate")
result_est.sigma_estimate     # (2,) estimated noise std of each observed output
result_est.params_named_std   # Laplace std at the estimated noise level
```

This is the concentrated likelihood: the noise std of each output is replaced
by its best estimate, the RMS residual over its observed entries (pooled over
the datasets of a multi-batch fit). It takes one fit, in place of fitting with
`"mse"` and refitting with `sigma` set to the residual RMS.

### Parameter transforms, bounds and starting values

`calibrate` works on each free parameter in a transformed space, set by the
parameter's `transform:` in the model file
([Model file format](model_format.md)) or by `transforms={name: ...}`:

- `"none"` (the default): the physical value.
- `"positive_log"`: the logarithm, for a parameter that must stay positive.
- `"logit"`: for a fraction between 0 and 1.

The fit keeps each free parameter inside the `bounds:` declared on it in the
model, as `fit` does. Pass explicit absolute bounds, which replace the declared
ones, or an extra box around the starting value:

```python
optimizer = aquakin.OptimizerConfig(
    bounds={"O3_Br_direct.k1": (20.0, 800.0), "O3_decay_OH.k_dec": (5.0, None)},  # None: open
    param_halfwidth=50.0,    # also within ±50 of the start (k1, k_dec are untransformed)
)
result_box = aquakin.calibrate(reactor, C0, observations, t_obs, free_params,
                               observed_species=observed, loss="nll", sigma=sigma,
                               optimizer=optimizer)
```

`param_halfwidth` is a half-width in the transformed space. For a
`positive_log` parameter, `np.log(100.0)` keeps it within a factor of 100 of
its start; for an untransformed parameter, as here, it is an absolute width in
the parameter's own units. `OptimizerConfig(use_model_bounds=False)` ignores
the declared bounds. A starting value outside the bounds is moved to the
nearest bound, with a warning.

`OptimizerConfig(n_starts=...)` adds jittered restarts around one start. To
run a fixed set of starts instead, such as a grid, pass them as `starts`, each
a full parameter vector or a `{name: value}` dict over the free parameters (the
others start from `initial_params`). Every start runs through one compiled
objective and the best fit is kept:

```python
starts = [{"O3_Br_direct.k1": k1, "O3_decay_OH.k_dec": kd}
          for k1 in (50.0, 150.0, 450.0) for kd in (20.0, 100.0)]
result_grid = aquakin.calibrate(reactor, C0, observations, t_obs, free_params,
                                observed_species=observed, loss="nll", sigma=sigma,
                                optimizer=aquakin.OptimizerConfig(starts=starts))
```

### What you observe

Observations are taken to be in the units the model declares for each species.
For data measured in another unit, say which with `observation_units=`:

```python
data_mmol = 1e3 * observations                 # the same data, in mmol/L
result_units = aquakin.calibrate(reactor, C0, data_mmol, t_obs, free_params,
                                 observed_species=observed,
                                 observation_units={"O3": "mmol/L", "BrO3-": "mmol/L"},
                                 loss="nll", sigma=1e3 * sigma)
```

The named columns, and an absolute `sigma` given in the same units, are
converted to the model's units before the fit, so the loss, `sigma_estimate`
and the posterior are in the model's units. Only the scale changes: a unit
that measures something else (mg/L where the model has mol/L) raises an error.
`profile_likelihood` and `plant.calibrate` take `observation_units` too.

When a measurement is not a single species but a function of several (a UV
absorbance from several organic pools, a probe-compound ratio), pass
`observation_fn` instead of `observed_species`. It maps one time point's
concentration vector and the parameter vector to the measured outputs, and
must be written with `jax.numpy` so it can be differentiated. Its outputs have
no declared unit, so convert them yourself rather than with
`observation_units`.

```python
oxidized_br = lambda C, p: C[i["HOBr"]] + C[i["BrO2-"]] + C[i["BrO3-"]]
o3_and_ox = lambda C, p: jnp.stack([C[i["O3"]], oxidized_br(C, p)])

# Synthetic oxidized-bromine data, with 3 % noise
ox_clean = oxidized_br(truth.C.T, params)                  # (12,)
ox_data = ox_clean + 0.03 * ox_clean.max() * rng.standard_normal(12)
data_fn = jnp.stack([observations[:, 0], ox_data], axis=1)  # ozone, oxidized Br

result_fn = aquakin.calibrate(reactor, C0, data_fn, t_obs, free_params,
                              observation_fn=o3_and_ox, loss="nll",
                              sigma=jnp.array([sigma[0], 0.03 * ox_clean.max()]))
```

The columns of `observations` follow the function's outputs, and `sigma` may
give one value per output, as here. `profile_likelihood` and `aquakin.Experiment` accept
`observation_fn` the same way.

A function of three arguments, `observation_fn(t, C, p)`, also receives the
sample time, in the unit of `t_obs`. Use it for a measurement whose mapping
changes with time, such as a drifting sensor:

```python
drifting = lambda t, C, p: C[i["O3"]] * (1.0 + 1e-4 * t)
```

A missing observation is a `NaN` in `observations`. It is left out of the fit,
so outputs sampled at different times can share one `t_obs` grid. For example,
a total measured only at the first sample is a second output that is `NaN`
everywhere else:

```python
total_br = truth.C[0, i["Br-"]] + ox_clean[0]
obs = np.stack([np.asarray(observations[:, 1]),
                np.where(np.arange(12) == 0, total_br, np.nan)], axis=1)
fn = lambda C, p: jnp.stack([C[i["BrO3-"]], C[i["Br-"]] + oxidized_br(C, p)])
result_nan = aquakin.calibrate(reactor, C0, obs, t_obs, free_params,
                               observation_fn=fn, loss="nll",
                               sigma=jnp.array([sigma[1], 0.03 * total_br]))
```

With `"mse"` or `"wmse"` the mean is over the observed entries. For a planned
experiment, `aquakin.Experiment(mask=...)` (`True` where an output is
measured) marks the same thing for `fisher_information` and experiment design.

### Fitting initial pools

An initial concentration that was not measured can be fitted alongside the
rate constants with `free_ic=aquakin.FreeICConfig([...])`, one value per
dataset, in log space. Its `bounds` are in the model's units and default to
`(1e-3, 1e4)`, which suits mg/L-scale models; give bounds that bracket the
pool for anything else, a single pair or a `{species: (lo, hi)}` dict.

By default the Laplace covariance covers the rate parameters only, with the
fitted pools held at their best values. When a rate constant and a pool trade
off against each other (for example `k` and `[X]0` when only their product
shapes the data), that understates the rate uncertainty. Ask for the joint
covariance instead. Here the rates are also fitted in log space, so that every
coordinate of the joint covariance is relative and the
[identifiability report](#identifiability-report) below compares like with
like:

```python
joint = aquakin.calibrate(
    reactor, C0, observations, t_obs, free_params,
    observed_species=observed, loss="nll", sigma=sigma,
    transforms={name: "positive_log" for name in free_params},
    free_ic=aquakin.FreeICConfig(["Br-"], bounds=(1e-8, 1e-4)),   # mol/L
    laplace=aquakin.LaplaceConfig(method="gauss_newton", include_ic=True),
)
joint.joint_names        # ['O3_Br_direct.k1', 'O3_decay_OH.k_dec', 'ic[0].Br-']
joint.joint_cov          # rates and log-space pools, in the fit's coordinates
joint.joint_hessian      # with loss="nll" and gauss_newton: the Fisher information
joint.params_named_std   # rate std devs, now marginal over the pools
joint.ic_named_std       # [{'Br-': ...}] pool std devs in mol/L, per dataset
```

Strong correlation between a rate and a pool in `joint_cov` is a sign that the
experiment cannot separate them; the [identifiability report](#identifiability-report)
says which combination the data do determine.

When the total of some pools is known and only its split is uncertain (a
measured total COD divided between readily and slowly biodegradable substrate,
say), hold the total fixed. Put the total into `C0`, with any split as the
starting guess, and fit only the split. Here the bromine present at the start
is known, but not how much of it was already oxidized to `HOBr`; the synthetic
batch has 20 % as `HOBr`, and `HOBr` is measured too:

```python
total = C0[i["Br-"]]
C0_true = C0.at[i["Br-"]].set(0.8 * total).at[i["HOBr"]].set(0.2 * total)
batch = reactor.solve(C0_true, t_span=(0.0, 600.0), t_eval=t_obs, params=params)
measured = ["O3", "HOBr", "BrO3-"]
clean_b = jnp.stack([batch.C_named(s) for s in measured], axis=1)
sigma_b = 0.03 * clean_b.max(axis=0)
obs_b = clean_b + sigma_b * rng.standard_normal(clean_b.shape)

C0_split = C0.at[i["Br-"]].set(0.5 * total).at[i["HOBr"]].set(0.5 * total)  # guess
split = aquakin.calibrate(
    reactor, C0_split, obs_b, t_obs, free_params,
    observed_species=measured, loss="nll", sigma=sigma_b,
    free_ic=aquakin.FreeICConfig(["Br-", "HOBr"], fixed_total=True),
    laplace=aquakin.LaplaceConfig(method="gauss_newton", include_ic=True),
)
split.ic_named           # [{'Br-': ..., 'HOBr': ...}], summing to the total
split.ic_fractions       # [{'Br-': 0.83, 'HOBr': 0.17}] (to two decimals)
split.ic_fractions_std   # [{'Br-': 0.07, 'HOBr': 0.07}]
```

The split is fitted through the log-ratios `log(c_j / c_0)` of the pools to the
first listed one, which keep every pool positive and the sum exact; these are
the `"ic[k].HOBr:Br-"` rows of `joint_cov`. `fisher_information` takes the same
`FreeICConfig`. The log-ratios are unbounded, so a split never appears in
`result.at_bound`, and `profile_likelihood` cannot profile a pool of a
fixed-total split.

### What can be fitted

Anything in the model's parameter vector can be fitted, and is also reached by
`sensitivity`, `fisher_information`, `identifiability`, `dgsm` and
`monte_carlo`. Two model-file features put more into that vector (both in
[Model file format](model_format.md)):

- A temperature correction can name a model parameter as its `theta`, so the
  temperature dependence itself can be estimated.
- `aquakin.register_rate_function` adds a rate function aquakin does not ship.
  Its coefficients are written as ordinary model parameters in the rate
  expression, so they are fitted and differentiated like any other.

## How certain is the fit?

### The Laplace posterior

With `laplace=True` (the default), `calibrate` approximates the posterior by a
Gaussian at the optimum. `result.params_named_std` holds the marginal
standard deviations and `result.posterior_cov` the covariance. Read them as
uncertainties only with `loss="nll"` and a calibrated `sigma` (or
`sigma="estimate"`): with `"mse"` or `"wmse"` they have the right shape but
not the right scale. `LaplaceConfig(method="gauss_newton")` forms the Hessian
from the residual Jacobian, which with `loss="nll"` is the Fisher information.

A parameter that ends on a bound, its own or one of its initial pools
(`"ic[k].<species>"`), is listed in `result.at_bound` (for example
`{"O3_Br_direct.k1": "lower"}`), with a warning. The Laplace approximation is
not valid on an active bound, so its standard deviation is `nan`, it has zero
variance in `posterior_cov` (posterior draws keep it on the bound), and the
other parameters' standard deviations are conditional on it staying there.
`aquakin.identifiability(result)` carries the same flag. A
[profile likelihood](#profile-likelihood) gives the one-sided interval for
such a parameter.

Sample the posterior and propagate it through a solve to get per-time-point
predictive envelopes (the `C0` may be a held-out validation batch):

```python
band = result.predictive_band(reactor, C0, t_obs, n_draw=200,
                              percentiles=(2.5, 97.5))
band.median, band.lo, band.hi        # each (12, 6): (n_t, n_species)

draws = result.posterior_draws(model, n_draw=50, seed=0)   # (50, 7) parameter vectors
```

### Sampling the posterior

The Laplace posterior is a Gaussian at the MAP, built from the local
curvature. It describes the posterior well when the data determine the
parameters precisely enough that the model is close to linear over their
uncertainty, in the coordinates of their transforms. It misdescribes a
posterior that is curved (a ridge where only a product or a sum of
parameters is seen), skewed, or cut by a bound, and those errors carry into
everything built on its draws: predictive bands and robust experiment
designs.

`aquakin.sample_posterior` draws from the posterior itself by Markov chain
Monte Carlo, starting from a fit:

```python
samples = aquakin.sample_posterior(
    result, reactor,
    config=aquakin.SamplerConfig(n_chains=2, n_draws=200, n_warmup=100, seed=0),
)
samples.params_named["O3_Br_direct.k1"]   # (400,) draws, physical units
samples.draws                             # (400, 7) parameter vectors, as posterior_draws
samples.r_hat, samples.ess_bulk           # convergence diagnostics, by coordinate
samples.acceptance_rate                   # per chain
```

The run above is short to keep this page quick. For a real analysis keep the
default `SamplerConfig()` (4 chains of 1000 draws after 500 warm-up steps) and,
before using the draws, check that every `r_hat` is below about 1.01 and that
`ess_bulk` is large enough for what you report: the Monte Carlo error of a
posterior mean is about its standard deviation over `sqrt(ess_bulk)`. A larger
`r_hat` means the chains disagree; run them longer.

The sampled density is the one `calibrate` maximized, so the likelihood, the
priors and the transforms are the same. It needs `loss="nll"` (the other
losses are not likelihoods) and a Laplace posterior, which supplies the
starting points and the proposal. Fitted initial pools are sampled together
with the rates, which needs `LaplaceConfig(include_ic=True)`. With
`sigma="estimate"` the density is the posterior with each output's noise level
integrated out under the prior `p(sigma) ∝ 1/sigma`, so its intervals are
wider than the Laplace ones by the extra uncertainty of an estimated noise
level. The draws stay inside the bounds the fit used.

The sampler is a random-walk Metropolis. Each chain starts from a draw of the
Laplace posterior and proposes Gaussian steps shaped by the Laplace
covariance, adapting their size during the warm-up. Each step costs one
forward solve per chain and no gradient, so it suits a stiff model, but a few
thousand solves add up: a plant, or a model whose posterior is far from the
Laplace shape, needs patience. A plant fit is sampled with
`plant.sample_posterior(result)`, through the plant's own solves.

The draws plug into what takes Laplace draws:

```python
band = result.predictive_band(reactor, C0, t_obs, draws=samples.draws)
robust_draws = samples.draws[::40]      # e.g. param_draws for optimize_experiment
```

When the Laplace intervals are enough, read them in the coordinates the fit
ran in. For a parameter with `transform: positive_log`, the interval
`exp(log(p) ± 1.96 sd)`, with `sd` the square root of the diagonal of
`result.posterior_cov`, follows a skewed posterior far better than `p ± 1.96 *
params_named_std`, which is the same uncertainty mapped linearly to physical
units.

### Profile likelihood

The Laplace covariance is a local, quadratic approximation. For an exact
analysis, which tells a well-constrained parameter from one the data cannot
pin down, use `aquakin.profile_likelihood`. It fixes one quantity at each
point of a grid, re-optimizes everything else, and reports the confidence
interval where the profile rises by the likelihood-ratio threshold:

```python
prof = aquakin.profile_likelihood(
    reactor, C0, observations, t_obs, free_params,
    grid=jnp.linspace(80.0, 280.0, 9), profile_param="O3_Br_direct.k1",
    observed_species=observed, loss="nll", sigma=sigma)
prof.mle    # grid value at the profile minimum
prof.ci     # (lo, hi); None on a side means the parameter is unidentified there
```

A 95 % interval is where the profile rises by 1.92 (`0.5 * chi2_{1, 0.95}`, the
default `delta`). With `sigma` known, the profile is the `"nll"` loss itself.
With `sigma="estimate"` and one output, the rise is `(N / 2) log(SSE /
SSE_min)`, so the interval is where the sum of squares grows by a factor
`exp(2 * 1.92 / N)`.

### Identifiability report

`aquakin.identifiability` analyzes a Fisher information matrix and reports
which parameter combinations the data can and cannot determine. It accepts a
`CalibrationResult` fitted with `loss="nll"` and
`LaplaceConfig(method="gauss_newton")`, the output of `fisher_information`
(below), or a bare matrix with `names=`:

```python
rep = aquakin.identifiability(joint)
for d in rep.unidentifiable:          # directions the data barely constrain
    print(d.relative_eigenvalue, d.loadings)
rep.correlation                       # correlation matrix, ordered as rep.names
rep.relative_std                      # Cramér-Rao relative standard errors
rep.collinearity_index(["O3_Br_direct.k1", "ic[0].Br-"])
rep.largest_identifiable_subset(threshold=15.0)
```

In this fit the report flags two weak directions, and `rep.correlation` shows
`k_dec` and the initial bromide strongly anticorrelated (about -0.94): ozone
and bromate alone barely tell faster ozone decay with less bromide from slower
decay with more.

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
normalized to unit diagonal. It is 1 when the subset's sensitivities are
orthogonal and grows without bound as they become linearly dependent,
regardless of how sensitive the output is to each parameter.
`rep.identifiable_subsets()` lists every subset below a threshold (default
15), largest first.

## Comparing model structures

When several model structures could explain the same data (a different rate
law, an extra pathway, a parameter fitted or held at a literature value), an
information criterion says which the data support, charging each structure for
the quantities it estimates. A fit with `loss="nll"` carries what they need:

- `result.log_likelihood`: the Gaussian log-likelihood of the observations at
  the fit, complete with its `-(N/2) log(2π)` constant and without any prior.
  It is not `-result.loss`, which leaves out the constant and includes the
  prior penalty.
- `result.n_observations`: the observed entries over every dataset (a `NaN` is
  not counted).
- `result.n_estimated`: the free rate parameters, plus the fitted initial
  pools of every dataset, plus one noise level per observed output with
  `sigma="estimate"`.

`aquakin.aic(result)`, `aquakin.aicc(result)` and `aquakin.bic(result)` give
the criteria, and `aquakin.compare_models` ranks several fits of the same
observations. Here the question is whether the data call for fitting
`k_dec`, or whether its default value will do:

```python
k1_only = aquakin.calibrate(reactor, C0, observations, t_obs, ["O3_Br_direct.k1"],
                            observed_species=observed, loss="nll", sigma=sigma)
table = aquakin.compare_models({"k1 and k_dec": result, "k1 only": k1_only},
                               criterion="aicc")
table.best       # 'k1 only'
table.delta      # how far each is behind the best
table.weights    # {'k1 only': 0.59, 'k1 and k_dec': 0.41} (to two decimals)
```

% invisible-code-block: python
%     assert table.best == "k1 only"
%     assert round(table.weights["k1 only"], 2) == 0.59
%     assert round(table.weights["k1 and k_dec"], 2) == 0.41
%     assert (result.n_estimated, k1_only.n_estimated, result.n_observations) == (2, 1, 24)

Lower is better, and only differences between fits of the same observations
mean anything, so `compare_models` refuses fits whose observations or
observation times differ. Fitting `k_dec` as well raises the likelihood too
little to pay for the extra parameter, though a weight of 0.41 says the data do
not rule it out either. As a rule of thumb, a difference below 2 is weak
evidence and one above 10 is decisive. AICc corrects AIC for small samples and
is the safer default unless there are many more observations than estimated
quantities (more than about 40 per quantity); BIC charges more for each
parameter and favours the simpler structure more strongly as the data grow.

Some fits cannot be compared this way:

- `"mse"` and `"wmse"` are not likelihoods, so the criteria refuse them. Refit
  with `loss="nll"` and `sigma="estimate"`, which is the least-squares form of
  the criteria with one noise level estimated per output.
- A prior (a model-declared one with `use_priors=True`, an explicit `priors=`,
  or `FreeICConfig(prior_log_std=)`) moves the fit off the maximum-likelihood
  point, so the criteria warn. Refit with `use_priors=False` and no priors to
  compare structures.
- `aicc` needs more observations than estimated quantities plus one.

`plant.calibrate` results work the same way.
