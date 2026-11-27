# Uncertainty and model comparison

A fit returns parameter values; this page is about how far to trust them: the
uncertainty of each value, which parameter combinations the data determine,
how to report the result, and which of several model structures the data
support. The ideas behind these tools (the Laplace approximation, the profile
likelihood, identifiability) are explained in
[Key ideas: information, uncertainty and identifiability](analysis_concepts.md).

The examples continue the fits of [Calibration](calibration.md): the shipped
`ozone_bromate` model in a batch reactor, with synthetic observations of ozone
and bromate made from the default parameters plus 3 % noise, a fit of two rate
constants, and a fit of the same rates in log space together with the initial
bromide. Run this block first; every block below then runs as written, in
order.

```python
import aquakin
import jax.numpy as jnp
import numpy as np

model = aquakin.load_model("ozone_bromate")
reactor = aquakin.BatchReactor(model, model.default_conditions())
C0 = model.default_concentrations()
params = model.default_parameters()

free_params = ["O3_Br_direct.k1", "O3_decay_OH.k_dec"]
observed = ["O3", "BrO3-"]

t_obs = jnp.linspace(30.0, 600.0, 12)            # seconds
truth = reactor.solve(C0, t_span=(0.0, 600.0), t_eval=t_obs, params=params)
clean = jnp.stack([truth.C_named(s) for s in observed], axis=1)   # (12, 2)
sigma = 0.03 * clean.max(axis=0)                 # one noise level per species
rng = np.random.default_rng(0)
observations = clean + sigma * rng.standard_normal(clean.shape)

result = aquakin.calibrate(
    reactor, C0, observations, t_obs, free_params,
    observed_species=observed, loss="nll", sigma=sigma,
    optimizer=aquakin.OptimizerConfig(n_starts=4, seed=0),
)
joint = aquakin.calibrate(        # rates in log space, with the initial bromide
    reactor, C0, observations, t_obs, free_params,
    observed_species=observed, loss="nll", sigma=sigma,
    transforms={name: "positive_log" for name in free_params},
    free_ic=aquakin.FreeICConfig(["Br-"], bounds=(1e-8, 1e-4)),   # mol/L
    laplace=aquakin.LaplaceConfig(method="gauss_newton", include_ic=True),
)
```

## The Laplace posterior

With `laplace=True` (the default), `calibrate` approximates the posterior by a
Gaussian at the optimum, its
[Laplace approximation](analysis_concepts.md#fitting-likelihood-posterior-and-the-laplace-approximation).
`result.params_named_std` holds the marginal standard deviations in physical
units. `result.posterior_cov` is the covariance **in the coordinates the fit
ran in**, the unconstrained ones of each parameter's transform: the logarithm
for a `positive_log` parameter, the physical value for an untransformed one.
Its rows follow `result.parameter_names`. Read these as uncertainties only
with `loss="nll"` and a calibrated `sigma` (or `sigma="estimate"`): with
`"mse"` or `"wmse"` they have the right shape but not the right scale.
`LaplaceConfig(method="gauss_newton")` forms the Hessian from the residual
Jacobian, which with `loss="nll"` and no prior is the Fisher information of the
experiment at the fitted values.

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

## Sampling the posterior

When the posterior is curved, skewed or cut by a bound, the Laplace Gaussian
misdescribes it, and the error carries into everything built on its draws:
predictive bands and robust experiment designs.

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
priors and the transforms are the same. For a rate fitted through a transform
(`positive_log` or `logit`) there is one more choice, the prior the data are
combined with, set by `prior_space`:

- `prior_space="transformed"` (the default) takes the prior to be flat in the
  coordinate the fit ran in: in `log k` for a `positive_log` rate, so every
  factor of ten in `k` is equally likely beforehand. This is the density whose
  peak `calibrate` found and which the Laplace posterior approximates, so the
  MCMC, Laplace and profile-likelihood intervals answer the same question.
  Where the posterior is close to Gaussian in `log k` the three agree; where
  they differ, the posterior really is curved or skewed.
- `prior_space="physical"` takes the prior to be flat in `k` itself, so every
  equal step in `k` is equally likely beforehand. That puts more weight on
  large values, and moves a poorly determined rate's draws up, on the log
  scale by about the variance of `log k`. Use it when a uniform prior on `k`
  is what you mean; it is what aquakin 0.3.0 sampled.

A prior you give with `priors=` is a Gaussian on the physical value in both
modes, and multiplies the flat one. Rates without a transform and fitted
initial pools are sampled the same way in both. It needs `loss="nll"` (the other
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
`plant.sample_posterior(result)`, through the plant's own solves. Both take
`progress=` to follow a long run, as `calibrate` does. To keep the draws with
the project, and know when they are out of date, sample a calibrate study's
fit with a [posterior study](plant_files.md#studies).

The draws plug into what takes Laplace draws:

```python
band = result.predictive_band(reactor, C0, t_obs, draws=samples.draws)
robust_draws = samples.draws[::40]      # e.g. param_draws for optimize_experiment
```

## Profile likelihood

The Laplace covariance is a local, quadratic approximation. A
[profile likelihood](analysis_concepts.md#profile-likelihood) does not assume
the objective is quadratic, so it follows a skewed or one-sided uncertainty and
tells a well-constrained parameter from one the data cannot pin down.
`aquakin.profile_likelihood` fixes one quantity at each point of a grid,
re-optimizes everything else, and reports the confidence interval where the
profile rises by the likelihood-ratio threshold. It takes a single batch:
`C0`, `observations` and `t_obs` are arrays, not lists.

```python
prof = aquakin.profile_likelihood(
    reactor, C0, observations, t_obs, free_params,
    grid=jnp.linspace(80.0, 280.0, 9), profile_param="O3_Br_direct.k1",
    observed_species=observed, loss="nll", sigma=sigma)
prof.mle    # grid value at the profile minimum
prof.ci     # (lo, hi): where the profile crosses the threshold
```

% invisible-code-block: python
%     assert None not in prof.ci and prof.ci[0] < result.params_named["O3_Br_direct.k1"] < prof.ci[1]

`None` on a side of `prof.ci` means the profile did not reach the threshold
**within the grid** on that side. That is what an unidentified parameter
gives, but also a grid that is too narrow, so extend the grid on that side and
run again before concluding that the data do not bound it (a failed inner fit
on that side gives a warning instead). A grid centred on the estimate and
reaching about four Laplace standard deviations each side, or a factor of 10
for a parameter fitted in log space, usually finds both crossings. The bounds
are interpolated between grid points, and `prof.mle` is the best grid point,
so use enough points to resolve them; each costs one fit.

The default `delta`, 1.92, gives a 95 % interval. With `sigma` known, the
profile is the `"nll"` loss itself.
With `sigma="estimate"` and one output, the rise is `(N / 2) log(SSE /
SSE_min)`, so the interval is where the sum of squares grows by a factor
`exp(2 * 1.92 / N)`.

## Reporting the estimates

A table of estimates with their standard deviations and 95 % intervals comes
straight from a result. For a parameter fitted in log space, build the
interval in that space, from the diagonal of `posterior_cov`: it follows a
skewed posterior far better than `p ± 1.96 * params_named_std`, the same
uncertainty mapped linearly to physical units, and it never reaches zero.

```python
sd_fit = np.sqrt(np.diag(joint.posterior_cov))       # in the fit's coordinates
for name, transform, sd in zip(joint.parameter_names, joint.transforms, sd_fit):
    p = joint.params_named[name]
    if transform == "positive_log":
        lo, hi = p * np.exp(-1.96 * sd), p * np.exp(1.96 * sd)
    else:                                             # "none"
        lo, hi = p - 1.96 * sd, p + 1.96 * sd
    print(f"{name:18s} {p:9.4g} ± {joint.params_named_std[name]:.2g}"
          f"   95 %: {lo:.4g} to {hi:.4g}")
```

% invisible-code-block: python
%     assert joint.transforms == ["positive_log", "positive_log"]
%     k1 = joint.params_named["O3_Br_direct.k1"]
%     assert k1 * np.exp(-1.96 * sd_fit[0]) < k1 < k1 * np.exp(1.96 * sd_fit[0])

The profile interval is `prof.ci`, and the MCMC interval the percentiles of the
draws, `np.percentile(samples.params_named[name], [2.5, 97.5])`.
[Which interval to report](analysis_concepts.md#which-interval-to-report) says
when each is the right one; say which you used, with the noise model and any
priors.

## Saving a fit

`result.save(folder)` writes a fit to a folder of plain files, and
`aquakin.load_calibration(folder)` reads it back, every number exactly as it
was:

```python
joint.save("ozone_fit")
saved = aquakin.load_calibration("ozone_fit")
saved.params_named == joint.params_named     # True
```

% invisible-code-block: python
%     assert saved.params_named == joint.params_named
%     assert saved.ic_named == joint.ic_named
%     assert np.array_equal(saved.joint_cov, joint.joint_cov)

The folder holds `fit.yaml` (the fitted values and their standard deviations,
the loss and the optimizer's status, the transforms and priors, the fitted
initial pools, the counts the information criteria use, a fingerprint of the
model and the aquakin version), `parameters.yaml` (the full parameter vector
by name) and, when the fit has them, the covariance and Hessian matrices and
the fitted initial states as CSV tables with named rows and columns. A
calibration study writes its `fit.yaml`, `parameters.yaml` and
`covariance.csv` the same way, and a
[study of a batch reactor](plant_files.md#a-study-of-a-batch-reactor) saves its
whole fit with `save`. `save` refuses a folder that already holds a saved fit.

A loaded fit works wherever only its numbers are needed: the information
criteria and `compare_models` (it still knows which observations it was fitted
to), `posterior_draws(model)` and `predictive_band(reactor, ...)`. The model
the fit ran on is not saved, so `aquakin.sample_posterior` needs the result
`calibrate` returned.

## Identifiability report

`aquakin.identifiability` analyzes a Fisher information matrix and reports
which parameter combinations the data can and cannot determine
([eigen-directions and identifiability](analysis_concepts.md#eigen-directions-and-identifiability)). It accepts a
`CalibrationResult` fitted with `loss="nll"` and
`LaplaceConfig(method="gauss_newton")`, the output of `fisher_information`
(below), or a bare matrix with `names=`:

```python
rep = aquakin.identifiability(joint)
for d in rep.unidentifiable:          # directions the data barely constrain
    print(d.relative_eigenvalue, d.loadings)
rep.correlation                       # correlation matrix, ordered as rep.names
rep.relative_std                      # Cramér-Rao relative standard deviations
rep.collinearity_index(["O3_Br_direct.k1", "ic[0].Br-"])
rep.largest_identifiable_subset(threshold=15.0)
```

In this fit the report flags two weak directions, and `rep.correlation` shows
`k_dec` and the initial bromide strongly anticorrelated (about -0.94): ozone
and bromate alone barely tell faster ozone decay with less bromide from slower
decay with more.

Each eigen-direction carries its eigenvector loadings by name;
[Reading a direction](analysis_concepts.md#reading-a-direction-products-and-ratios)
explains how loadings name a product or a ratio. A direction is flagged when
its eigenvalue is below `eig_keep` (default `1e-2`) times the largest. Flagging is a diagnostic only. The standard deviations here, like
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

`plant.calibrate` results work the same way
([Calibrating a plant](plant_calibration.md)).
