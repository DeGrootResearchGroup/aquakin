# Calibration

Fit a model's parameters to measured data. Reactor fits are on this page; a
whole plant is fitted the same way ([Calibrating a plant](plant_calibration.md)).
How well the data determine the fitted values, and which model structure they
support, is in [Uncertainty and model comparison](fit_uncertainty.md). The
statistical terms used here (likelihood, prior, posterior, MAP, Laplace
approximation) are explained in
[Key ideas](analysis_concepts.md#fitting-likelihood-posterior-and-the-laplace-approximation).

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
  transforms, [priors](#priors), deterministic multistart and a Laplace
  posterior for the uncertainty (`laplace=True` by default).

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
n_observed)` for several, its columns in the order of `observed_species`.
`t_obs` is in the model's time unit unless you pass `time_unit=`. Pass
**lists** of `C0`, `observations` and `t_obs` for a joint fit of several
batches. `result.converged` is not checked for you: see
[When a fit does not converge](#when-a-fit-does-not-converge).

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

This is the
[concentrated likelihood](analysis_concepts.md#fitting-likelihood-posterior-and-the-laplace-approximation):
the noise std of each output is replaced by the RMS residual over its observed
entries (pooled over the datasets of a multi-batch fit). It takes one fit, in
place of fitting with `"mse"` and refitting with `sigma` set to the residual
RMS.

### Using your own measurements

Measured data usually arrive as a table: one row per sample time, one column
per measured quantity, in the laboratory's units, with gaps where a sample was
lost. The block below writes such a file from the synthetic data, so that the
page runs; with your own data, start from the `np.genfromtxt` line.

```python
# A stand-in for a laboratory file: minutes, mmol/L, one lost bromate sample.
with open("batch_1.csv", "w") as f:
    f.write("time_min,O3_mmol,BrO3_mmol\n")
    for k, (t, (o3, bro3)) in enumerate(zip(t_obs / 60.0, 1e3 * observations)):
        f.write(f"{t:.1f},{o3:.6f},{'' if k == 4 else f'{bro3:.8f}'}\n")

data = np.genfromtxt("batch_1.csv", delimiter=",", names=True)   # an empty cell is NaN
t_lab = data["time_min"]                                         # minutes
obs_lab = np.column_stack([data["O3_mmol"], data["BrO3_mmol"]])  # the order of observed

C0_lab = model.concentrations({"O3": 1.0e-4, "Br-": 1.0e-6})     # measured at t = 0, mol/L
reactor_lab = aquakin.BatchReactor(                              # the batch's conditions
    model, aquakin.OperatingConditions(pH=7.5, OH_scavenging=5.0e4))
fit_lab = aquakin.calibrate(
    reactor_lab, C0_lab, obs_lab, t_lab, free_params,
    observed_species=["O3", "BrO3-"],           # one per column, in column order
    observation_units={"O3": "mmol/L", "BrO3-": "mmol/L"},
    time_unit="min", loss="nll", sigma="estimate",
)
fit_lab.n_observations    # 23: the lost sample is left out
```

% invisible-code-block: python
%     assert fit_lab.n_observations == 23
%     assert fit_lab.converged

- **Columns** follow `observed_species`, whatever the file calls them. A
  quantity that is not one species is an `observation_fn`
  ([What you observe](#what-you-observe)).
- **Times** may be in any unit `time_unit=` names (`"s"`, `"min"`, `"h"`,
  `"d"`); they are converted to the model's unit before the solve.
- **A missing sample** is a `NaN`, which `np.genfromtxt` gives for an empty
  cell; it is left out of the fit.
- **Units** of the measurements go in `observation_units=`
  ([What you observe](#what-you-observe)).
- **Initial values** are what you measured at the start, set by name with
  `model.concentrations({...})`; species you do not name keep the model's
  defaults. An initial pool you did not measure can be
  [fitted](#fitting-initial-pools).
- **Conditions** (pH, temperature, scavenging, ...) are those of the batch,
  given to its reactor ([Reactors](reactors.md)). A joint fit of several
  batches shares one reactor, so it suits batches that differ only in their
  initial state.

With pandas installed (the `dataframe` extra),
`pd.read_csv("batch_1.csv")[["O3_mmol", "BrO3_mmol"]].to_numpy()` reads the same
array; an empty cell is again `NaN`.

### Priors

A prior is a Gaussian belief about a parameter, `(mean, std)` in physical
units, added to the objective as `0.5 * ((p - mean) / std) ** 2`. Pass priors
by name with `priors=`; only free parameters use them:

```python
result_prior = aquakin.calibrate(reactor, C0, observations, t_obs, free_params,
                                 observed_species=observed, loss="nll", sigma=sigma,
                                 priors={"O3_Br_direct.k1": (160.0, 40.0)})
result_prior.priors_applied     # {'O3_Br_direct.k1': (160.0, 40.0)}
```

% invisible-code-block: python
%     assert result_prior.priors_applied == {"O3_Br_direct.k1": (160.0, 40.0)}

A model can also declare priors on its parameters with a `prior:` block
([Model file format](model_format.md#bounds-transforms-and-priors)); the
shipped `asm1`, `adm1` and `wats_*` models do. **These are applied by
default** (`use_priors=True`) to every free parameter that declares one, so a
fit of those models is pulled toward the literature values unless you say
otherwise. An entry in `priors=` replaces the declared prior of that
parameter, and `use_priors=False` ignores the declared ones. Check
`result.priors_applied` to see which priors a fit used. A fitted initial pool
takes a prior in log space with `FreeICConfig(prior_log_std=)`.

A prior is weighed against the data term, so use it with `loss="nll"` and a
`sigma`: the objective is then a log posterior. With `"mse"` the data term is
in the square of the data's units (about 1e-12 for concentrations in mol/L),
so the same prior can swamp the data or count for nothing depending on the
units the data happen to be in. A fit that carries a prior is not a
maximum-likelihood fit, which matters when
[comparing model structures](fit_uncertainty.md#comparing-model-structures).

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
coordinate of the joint covariance is relative and an
[identifiability report](fit_uncertainty.md#identifiability-report) of it
compares like with like:

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
joint.joint_hessian      # the Hessian it inverts
joint.params_named_std   # rate std devs, now marginal over the pools
joint.ic_named_std       # [{'Br-': ...}] pool std devs in mol/L, per dataset
```

Strong correlation between a rate and a pool in `joint_cov` is a sign that the
experiment cannot separate them; the
[identifiability report](fit_uncertainty.md#identifiability-report) says which
combination the data do determine.

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

## When a fit does not converge

`calibrate` does not warn when the optimizer stops without converging: check
`result.converged` before using a fit. When it is `False`:

1. **Read `result.message`**, the optimizer's reason for stopping. If
   `result.n_iter` reached `OptimizerConfig(max_iter=)` (default 500), raise
   `max_iter`.
2. **Start from more places.** `OptimizerConfig(n_starts=8)` adds jittered
   restarts, and `starts=[...]` runs a grid; the best fit is kept. Starts that
   end far apart with similar losses point to a parameter the data do not
   determine ([Identifiability report](fit_uncertainty.md#identifiability-report)).
3. **Try the least-squares optimizer**, `OptimizerConfig(method="gauss_newton")`,
   which works on the residual vector rather than the scalar loss and is often
   more robust on a stiff reaction model.
4. **Fit positive parameters in log space** with
   `transforms={name: "positive_log"}` (or `transform:` in the model file),
   so that a rate constant of 1e9 and one of 0.1 take steps of the same size.
5. **Keep a stiff model's solve feasible.** A start whose solve fails is
   skipped with a warning. If the optimizer keeps stepping into such a region,
   bound the search with `OptimizerConfig(param_halfwidth=np.log(100.0))` (two
   decades either side of the start, for `positive_log` parameters) or
   `OptimizerConfig(bounds={...})`. A non-finite gradient raises an error
   naming its remedies: the cap-free adjoint
   `diff=aquakin.DifferentiationConfig(method="stable")` (batch reactors), or a
   step cap on the reactor,
   `settings=aquakin.SolverSettings(integrator=aquakin.IntegratorConfig(dtmax=...))`.
6. **Look at `result.at_bound`.** A parameter that ends on a bound is held
   there, which can stop the others short. Widen the bound if it is not
   physical, or fix the parameter.
7. **Check the noise level.** With `loss="nll"`, a `sigma` far from the real
   noise makes the objective very flat or very steep; `sigma="estimate"`
   avoids guessing it.
