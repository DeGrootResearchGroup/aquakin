# Sensitivity & calibration

Every solve in `aquakin`, of a single reactor or a whole plant, is
**differentiable**. Rate constants are always passed in as arguments (never
baked into the rate functions), so you can take exact gradients of any output
with respect to parameters, conditions or initial states, and use them for
sensitivity analysis, parameter estimation, experiment design and design
optimization.

A plain forward simulation never needs any of this. Reach for it when you want
to know *how* an output depends on its inputs, or to fit a model to data.

## Which tool do I need?

| Question | Reactor | Plant |
|---|---|---|
| How does an output change with each parameter, here? | `aquakin.sensitivity`, `reactor.solve_sensitivity` | `plant.steady_state_sensitivity`, `plant.dynamic_sensitivity`, `plant.solve_sensitivity` |
| Which inputs matter over a whole range? | `aquakin.dgsm` | `plant.steady_state_dgsm`, `plant.dynamic_dgsm` |
| Which parameter values fit my data? | `aquakin.calibrate` (`aquakin.fit` for a point estimate only) | `plant.calibrate` |
| How well do the data determine them? | the Laplace posterior, `aquakin.profile_likelihood`, `aquakin.identifiability` | the same, with `plant_profile_likelihood` |
| What should I measure next? | `aquakin.fisher_information`, `rank_experiments`, `optimize_experiment`, `design_from_candidates` | `plant_fisher_information`, `plant_rank_sensor_sets` |
| How uncertain is a prediction? Which design is best? | `aquakin.monte_carlo`, `compare_scenarios`, `optimize_design` | the same |

For a complete study that uses most of these tools in turn, see
[Worked example: designing an experiment](worked_example_design.md).

### The examples on this page

The reactor examples share one setup: the shipped `ozone_bromate` model in a
batch reactor ([Reactors](reactors.md) explains the reactor options), with
synthetic observations of ozone and bromate made from the default parameters
plus 3 % noise. The model's concentrations are in mol/L and its time unit is
the second. Run this block first; every reactor block below then runs as
written, in order.

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

## Differentiating a solve

The simplest case is an ordinary JAX gradient through a reactor solve:

```python
def final_bromate(p):
    sol = reactor.solve(C0, t_span=(0.0, 600.0), t_eval=t_obs, params=p)
    return sol.C_named("BrO3-")[-1]

g = jax.grad(final_bromate)(params)   # d[BrO3-]_final / d params, shape (7,)
```

A reactor solve is differentiated in reverse mode (`jax.grad`, `jax.jacrev`).
`jax.jacfwd` through `reactor.solve` raises an error; the tools below that work
in forward mode build a forward-capable solve for you.

```{warning}
**Silent non-finite reverse gradients on stiff models.** A reverse-mode
gradient taken *directly* through the solve of a stiff model (ASM, ADM, WATS)
can come back as `NaN` or `Inf` when the integrator step is uncapped. No
exception is raised, so the bad gradient flows into your optimizer.

`aquakin.calibrate` and `aquakin.sensitivity` check the gradient they compute
and raise an error naming the remedy instead of returning `NaN`. They do not
make it finite by themselves. To get a finite gradient through a stiff reactor:

- cap the step with `integrator=aquakin.IntegratorConfig(dtmax=...)`;
- use forward-mode sensitivities (`solve_sensitivity`, or
  `diff=aquakin.DifferentiationConfig(mode="forward")`), which need no cap; or
- in `calibrate`, use the cap-free adjoint,
  `diff=aquakin.DifferentiationConfig(method="stable")` (batch reactors).

In your own loss, `aquakin.check_finite_gradient(g, what="my loss gradient",
remedy="...")` raises that kind of error for you.
```

### How a gradient is formed

`aquakin.DifferentiationConfig`, passed as `diff=`, says how a gradient flows
through a solve. Its fields are:

- `mode`: `"reverse"` (default; `jax.grad`, one pass for many parameters) or
  `"forward"` (`jax.jacfwd`, or the variational solve; finite through stiff
  models and suited to few parameters).
- `method`: `"stable"` (default) or `"through_solve"`. `"stable"` replaces
  differentiation through the integrator with a cap-free discrete adjoint
  (reverse) or the variational solve (forward); `"through_solve"`
  differentiates the integrator's own steps, which on a stiff model needs a
  `dtmax` cap in reverse mode.
- `check_finite` (default `True`): raise an error instead of returning a
  non-finite gradient.
- `adjoint_max_steps` (default 100 000) and `adjoint_low_memory` (default
  `False`): the step buffer of the stable reverse adjoint, and whether it
  recomputes each step in the backward pass instead of storing it, trading
  compute for memory on a long, large plant solve.

What each entry point does with it:

| Entry point | Default | Notes |
|---|---|---|
| `reactor.solve` under `jax.grad` | through the solve | a reactor has no cap-free adjoint for a bare solve; cap `dtmax` for a stiff model |
| `aquakin.calibrate` | `method="through_solve"` | `method="stable"` is the cap-free adjoint (batch reactors); `mode="forward"` needs `method="through_solve"` and suits `OptimizerConfig(method="gauss_newton")` |
| `aquakin.sensitivity`, `aquakin.dgsm` | reverse | `mode="forward"` builds a forward-capable solve, finite through a stiff model |
| `plant.solve`, `plant.calibrate` | `method="stable"` | the plant's cap-free discrete adjoint: `jax.grad` through a stiff plant is finite with nothing to tune |

## Sensitivity analysis

### Local sensitivity

`aquakin.sensitivity` computes the gradient of a scalar output with respect to
every parameter and every condition field in one call, and ranks them:

```python
sens = aquakin.sensitivity(
    reactor, C0,
    output_fn=lambda sol: sol.C_named("BrO3-")[-1],
    t_span=(0.0, 600.0), t_eval=t_obs,
)
sens.doutput_dparams              # (7,) gradient with respect to each parameter
sens.doutput_dconditions["pH"]    # (1,) sensitivity to a condition field
sens.ranked_params()              # [(name, |gradient|), ...] most influential first
```

### Forward sensitivity (cap-free stiff gradients)

Differentiating *through* a stiff solve with ordinary AD goes non-finite
above a step-size threshold, and the usual workaround, a global `dtmax` cap,
forces tiny steps over the whole solve. `solve_sensitivity` avoids both: it
integrates the sensitivity `S = dC/dθ` alongside the state and lets the
adaptive step controller bound the error of the sensitivity too, so the step
shrinks only where the sensitivity needs it and the result is exact with no
cap.

```python
uv = aquakin.load_model("uv_h2o2")
uv_reactor = aquakin.BatchReactor(uv, uv.default_conditions())

uv_sol, S = uv_reactor.solve_sensitivity(
    uv.default_concentrations(), t_span=(0.0, 5.0),
    t_eval=jnp.linspace(0.0, 5.0, 6), params=uv.default_parameters(),
    sens_params=["H2O2_photolysis.k_photo", "OH_target.k_OH_target"],
)
S.shape    # (6, 4, 2): (n_t, n_species, n_sens_params)

# The same with by-name accessors:
res = aquakin.forward_sensitivity(
    uv_reactor, uv.default_concentrations(), params=uv.default_parameters(),
    t_span=(0.0, 5.0), t_eval=jnp.linspace(0.0, 5.0, 6),
    sens_params=["H2O2_photolysis.k_photo"])
res.dC_dparam("target", "H2O2_photolysis.k_photo").shape   # (6,)
```

`solve_sensitivity` is available on `BatchReactor`, `PlugFlowReactor` and
`BiofilmReactor`. The plant version is
`plant.solve_sensitivity(params, wrt, t_span=..., t_eval=..., y0=...)`, which
returns `(ts, ys, S)` with `S` of shape `(n_t, n_states, k)` for the `k`
parameters named in `wrt`. Repeated calls reuse one compiled solve. Its
options:

- `sens_error_control=False` chooses the steps on the state's error alone. The
  sensitivity is still solved at every step, but its error is no longer
  bounded: use it for a screening run, or one that would otherwise be very
  long.
- `corrector="simultaneous"` (instead of the default `"staggered"`) solves the
  state and sensitivity together in a form you can differentiate again. It is
  much slower, and `sens_error_control=False` needs the default corrector.
- `return_stats=True` also returns the step counts of the solve.

For a scalar loss of many parameters, as in calibration, reverse mode is
cheaper; `plant.solve` uses its cap-free adjoint for that automatically.

### Global sensitivity (DGSM)

`aquakin.dgsm` is a derivative-based global sensitivity measure: it averages
the squared derivative of an output over quasi-random samples of the input
ranges, which bounds the Sobol total-order index of each input. You supply a
function from an input vector to a scalar (or vector) output and a `(low,
high)` range per input:

```python
def bromate_umol(theta):            # theta = [k1, k_dec] -> final bromate, µmol/L
    sol = reactor.solve(C0, t_span=(0.0, 600.0), t_eval=t_obs,
                        params=params.at[i_free].set(theta))
    return 1e6 * sol.C_named("BrO3-")[-1]

res = aquakin.dgsm(bromate_umol, [(80.0, 320.0), (35.0, 140.0)],
                   input_names=["k1", "k_dec"], n_samples=32, seed=0)
res.sobol_total_bound     # (2,) upper bound on the Sobol total index per input
res.ranked()              # [(name, bound), ...] most influential first
```

`diff=aquakin.DifferentiationConfig(mode="forward")` forms the per-sample
derivatives in forward mode. The result is the same; forward mode is faster
when there are many outputs, or when the reverse pass through a stiff solve is
expensive.

### Plant sensitivity

A plant has its own sensitivity methods, all taking parameters by name
(`"asm1.muA"`, see `plant.parameter_names()`) through `wrt=`. The example
uses the BSM1 plant of
[Plant-wide simulation](plants.md#the-benchmark-plants), with its constant
average-load influent:

```python
from aquakin.plant.bsm import build_bsm1

asm1 = aquakin.load_model("asm1")
plant = build_bsm1(asm1)
plant.add_influent("feed", asm1.influent(
    {"SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
     "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0}, Q=18446.0))
si = asm1.species_index

def effluent_n(y):                   # flat plant state -> outputs
    tank5 = plant.states_by_unit(y)["tank5"]
    return jnp.array([tank5[si["SNH"]], tank5[si["SNO"]]])

# Exact steady-state sensitivity, from one steady-state solve: (2 outputs, 2 parameters)
S_ss = plant.steady_state_sensitivity(output_fn=effluent_n,
                                      wrt=["asm1.muA", "asm1.KNH_A"])

# Sensitivity of a transient output, here the tank-5 nitrate trajectory
y_ss = plant.steady_state().state
S_dyn = plant.dynamic_sensitivity(
    output_fn=lambda sol: sol.C_named("tank5", "SNO"),
    t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 25),
    wrt=["asm1.muA", "asm1.KNH_A"], mode="forward", y0=y_ss)
```

- `steady_state_sensitivity` uses the implicit function theorem: no time
  integration, and one factorization serves every output and parameter. Pass
  `state=` to reuse a steady state you already have, and `elasticity=True` for
  the dimensionless `(dg/dθ)(θ/g)`.
- `dynamic_sensitivity` differentiates a solve. `mode="reverse"` (default)
  uses the plant's cap-free adjoint and suits many parameters;
  `mode="forward"` uses `plant.solve_sensitivity`, whose memory does not grow
  with the horizon, and passes its options (such as `sens_error_control`)
  through.
- `plant.steady_state_dgsm(ranges, output_fn=..., wrt=...)` and
  `plant.dynamic_dgsm(ranges, output_fn=..., wrt=..., t_span=...)` are the
  plant versions of `dgsm`, with `ranges` of shape `(k, 2)` aligned with
  `wrt`. A dynamic screen costs one differentiated solve per sample, so keep
  `n_samples` and the parameter count modest.

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

## Designing experiments

Before running an experiment you can ask whether it will determine the
parameters you care about, and choose the design that determines them best.
The tools work on the Fisher information of a design at assumed parameter
values; no data are needed.
[Worked example: designing an experiment](worked_example_design.md) explains
the ideas behind them step by step.

### Fisher information of a design

`aquakin.fisher_information` evaluates the Fisher information of a set of
experiments, each an `aquakin.Experiment` (initial state, sampling times,
noise, what is measured, and optionally its own `reactor` for different
operating conditions):

```python
reactor_ph8 = aquakin.BatchReactor(
    model, aquakin.SpatialConditions.uniform(1, pH=8.0, T=293.15, OH_scavenging=5e4))
design = [
    aquakin.Experiment(C0, t_obs=t_obs, sigma=sigma, observed_species=observed),
    aquakin.Experiment(C0.at[i["Br-"]].set(2e-6), t_obs=t_obs, sigma=sigma,
                       observed_species=observed, reactor=reactor_ph8),
]
fim = aquakin.fisher_information(
    reactor, design, free_params, params=params,
    free_ic=aquakin.FreeICConfig(["Br-"]),   # also estimate a pool per experiment
)
fim.names        # ['O3_Br_direct.k1', 'O3_decay_OH.k_dec', 'ic[0].Br-', 'ic[1].Br-']
fim.matrix       # in the coordinates calibrate fits in (log for positive_log, pools)
fim.physical()   # with respect to the physical values
fim.std()        # Cramér-Rao lower bound on each standard deviation, physical units
```

The information of independent experiments adds, so an experiment that
sharpens a poorly determined parameter shows up as a large drop in its
`std()`. A singular matrix (`fim.std()` raises) means some combination of the
quantities cannot be determined from the design at all. At a calibration's
optimum, with `loss="nll"`, the matrix equals the Gauss-Newton Laplace Hessian
(`joint_hessian` with `include_ic=True`).

### Fisher information of any forward model

`aquakin.fisher_information_from_fn` gives the Fisher information of a forward
model that is not an aquakin reactor: any function `fn(theta) -> y` written in
`jax.numpy`, such as a transport operator or a reactor wrapped in your own
code. The Jacobian of the observations comes from AD (`mode="auto"` picks
forward mode when there are more observations than parameters), and the result
is a `FisherInformation`, so `identifiability`, `design_criterion` and `std()`
work on it unchanged:

```python
t_toy = jnp.linspace(0.5, 10.0, 12)

def decay(theta):
    k, a0 = theta
    return a0 * jnp.exp(-k * t_toy)

fim_toy = aquakin.fisher_information_from_fn(
    decay, jnp.array([0.25, 1.5]),
    names=["k", "A0"],
    sigma=0.05, noise="lognormal",           # observe log y; sigma ~ relative error
    transforms=["positive_log", "positive_log"],
)
```

`sigma` broadcasts to the shape of `y`, and `inf` marks an entry as unmeasured.
With `groups=`, a label per observation (or per row), the information of each
group is returned too. Each group is one candidate measurement, for example
one sampling site, and the groups' information sums to the total:

```python
def decay_sites(theta):               # the same decay seen at three dilutions
    return jnp.stack([decay(theta) * d for d in (1.0, 0.5, 0.2)])   # (3, 12)

total, per_site = aquakin.fisher_information_from_fn(
    decay_sites, jnp.array([0.25, 1.5]), names=["k", "A0"], sigma=0.1,
    groups=["inlet", "zone A", "zone B"],
)
per_site["zone A"].matrix     # what sampling zone A alone contributes
```

If `fn` calls `reactor.solve`, pass `mode="reverse"`: a reactor solve is
differentiated in reverse mode.

### Scoring and optimizing a design

A design is scored by a criterion of its Fisher information `F`:

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
aquakin.design_criterion(fim, "D", subset=["O3_Br_direct.k1"])

# Choose the most useful next experiment from a list of candidates.
candidates = [
    aquakin.Experiment(C0, t_obs=t_obs, sigma=sigma[1], observed_species=["BrO3-"]),
    aquakin.Experiment(C0.at[i["O3"]].set(2e-4), t_obs=t_obs, sigma=sigma,
                       observed_species=observed),
]
ranked = aquakin.rank_experiments(reactor, candidates, free_params, base=design[:1])
ranked[0].index, ranked[0].improvement

# Or optimize the sampling times and initial concentrations directly.
res = aquakin.optimize_experiment(
    reactor,
    aquakin.Experiment(C0, t_obs=jnp.linspace(60.0, 360.0, 6), sigma=sigma,
                       observed_species=observed),
    free_params,
    time_bounds=(30.0, 600.0), min_spacing=20.0,   # in the experiment's time unit
    initial_bounds={"O3": (5e-5, 2e-4)},           # dose range to search, mol/L
)
res.experiments[0].t_obs, res.experiments[0].C0
```

`optimize_experiment` keeps the number of samples fixed and moves them within
the bounds by gradient, keeping consecutive samples at least `min_spacing`
apart.

The optimal design depends on the parameter values you assume, which are
uncertain before the experiment. To make the design robust to that, pass
`param_draws`, an array of parameter vectors drawn from a prior or a previous
calibration's posterior, and the criterion is averaged over them. A
calibration fitted with `laplace=` provides posterior draws directly:

```python
draws = result.posterior_draws(model, n_draw=20, seed=0)   # (20, 7)
robust = aquakin.optimize_experiment(reactor, design[0], free_params,
                                     param_draws=draws, time_bounds=(30.0, 600.0),
                                     min_spacing=20.0)
```

### Choosing measurements from a candidate pool

`rank_experiments` scores one candidate at a time. When you can afford several
measurements from a fixed list (sampling sites, sampling times, species to
analyze, experiments to run), `aquakin.design_from_candidates` chooses the
best set under a budget. Each candidate carries its own Fisher information
`F_c`, and the information of a design that takes candidate `c` `n_c` times is
`base + sum_c n_c F_c`.

The per-candidate information comes most easily from
`fisher_information_from_fn` with `groups=`. Here each candidate is one
(time, species) sample of the ozonation batch, 30 in all, for three rate
constants, and we can afford six:

```python
names = ["O3_Br_direct.k1", "O3_decay_OH.k_dec", "O3_OBr_oxidation.k2"]
idx = jnp.array([model.param_index[n] for n in names])
t_cand = jnp.linspace(60.0, 600.0, 10)
species = ["O3", "HOBr", "BrO3-"]
cols = jnp.array([i[s] for s in species])

def samples_of(theta):
    sol = reactor.solve(C0, params=params.at[idx].set(theta),
                        t_span=(0.0, 600.0), t_eval=t_cand)
    return sol.C[:, cols]                    # (10 times, 3 species)

labels = [[f"{s} @ {float(t):.0f} s" for s in species] for t in t_cand]
_, pool = aquakin.fisher_information_from_fn(
    samples_of, params[idx], names=names,
    sigma=0.03 * samples_of(params[idx]).max(axis=0),
    transforms=["positive_log"] * 3, groups=labels, mode="reverse",
)
choice = aquakin.design_from_candidates(pool, budget=6)
choice.selected      # the six samples to take
choice.efficiency    # 0.9997 of the bound from the relaxed problem
```

The candidates can also be a stack of `(p, p)` matrices or a list of
`FisherInformation` objects. The main options are:

- `budget=` is the number of candidates to take or, with `costs=` (one per
  candidate), the total cost.
- `required=[...]` lists candidates that must be in the design.
- `max_repeats=` allows replicates (default 1; `None` for no cap).
- `base=` is information already in hand, such as a prior or experiments
  already run.
- `criterion=` and `subset=` work as for `design_criterion`. D on a subset
  (Ds-optimality) chooses for some parameters while treating the rest as
  nuisance.
- `param_draws_axis=` names an axis of parameter draws in a stack of
  candidates, for a design robust to the assumed values; the criterion is
  averaged over the draws.

`method=` selects the search:

| Method | What it does |
|---|---|
| `"relaxed"` | Solves the continuous problem (real weights that spend the budget), then rounds it. D and A only. |
| `"greedy"` | Adds the best affordable candidate one at a time. |
| `"exchange"` | Improves the greedy design by single swaps until none helps. |
| `"auto"` (default) | Scores every design when there are few enough (`exhaustive_limit`), otherwise runs the rounded relaxed and greedy designs through exchange and keeps the better. |

For D and A the relaxed problem is concave, so its optimum is a bound that no
integer design can beat. The result reports it as `bound` and the chosen
design's `efficiency` against it. For D with no base, no required candidates
and no binding cap, `equivalence` gives the check from the Kiefer-Wolfowitz
equivalence theorem: at the relaxed optimum the largest standardized variance
equals the number of scored parameters. `gains` shows how much one more of
each candidate would add, which is useful when the budget grows.

A design with fewer informative candidates than parameters has singular
information, so without a positive-definite `base` the search compares
candidates with a tiny ridge added (`regularized` is then `True`; the reported
criterion has no ridge). If the whole pool cannot determine the parameters,
the function raises an error and asks for a `base` or a `subset`.

The forward model does not have to be an aquakin reactor. In this toy source
localization problem, three sources on a line spread a tracer to 41 candidate
sensor nodes, and we choose four sensors to estimate the third source's
strength, treating the other two as nuisance. Node 0 is always sampled:

```python
nodes = jnp.linspace(0.0, 10.0, 41)
sources = jnp.array([2.0, 5.0, 8.0])

def conc(q):
    return jnp.exp(-jnp.abs(nodes[:, None] - sources[None, :]) / 1.5) @ q

_, sensors = aquakin.fisher_information_from_fn(
    conc, jnp.array([1.0, 2.0, 0.5]), names=["q1", "q2", "q3"], sigma=0.1,
    groups=[f"node {n}" for n in range(41)],
)
placement = aquakin.design_from_candidates(
    sensors, budget=4, subset=["q3"], required=["node 0"],
)
```

## Calibrating a plant

`plant.calibrate` is the plant version of `aquakin.calibrate`: it fits plant
parameters, by name, against measured channels of one or more streams or
units. It uses the same transforms, priors, multistart and Laplace posterior,
behind the plant's cap-free adjoint, so the gradient is finite for a stiff
plant with no `dtmax` to tune. Two defaults differ from `aquakin.calibrate`:
the loss is `"mse"` and `laplace=False`, so pass `loss="nll"`, `sigma=` and
`laplace=True` for uncertainties.

The plant examples below use BSM2, built with its influent wired as in
[Plant-wide simulation](plants.md#the-benchmark-plants), and need a measured
`observations` array. A BSM2 solve takes minutes, and a fit many solves.

% skip: next "a BSM2 plant calibration that needs measured data; its API is checked by the plant calibration tests"

```python
from aquakin.plant.bsm import build_bsm2, bsm2_warm_start
# plant = build_bsm2(...) with its influent, as in Plant-wide simulation

result = plant.calibrate(
    observations,               # (n_t, n_channels) measured data
    t_obs,
    ["asm1.muH", "asm1.bH"],    # plant parameters to fit (see plant.parameter_names())
    target="effluent",          # a registered stream, or "unit.port"
    observed_channels=["SNH", "SNO"],
    loss="nll", sigma=jnp.array([0.2, 0.5]), laplace=True,
    y0=bsm2_warm_start(plant),  # warm start recommended for a stiff plant
)
result.params_named
```

Fit against several streams at once with
`observables=[PlantObservable(...), ...]`, fit initial states of the assembled
plant with `free_ic=FreeICConfig([...])`, and run a joint multi-batch fit by
passing list-valued `observations`, `t_obs` and `y0`.

### Measured channels

A channel need not be a species. The channels correspond to what a laboratory
or an online sensor measures:

| Channel | Measures |
|---|---|
| `"Q"`, `"T"` | the stream's flow and temperature (a stream carries a temperature when the influents upstream of it have one) |
| `"COD"` | total COD as a COD test sees it: dissolved oxygen and nitrate count none, nor do dissolved hydrogen and methane |
| `"TKN"` | total Kjeldahl nitrogen: ammonium and organic nitrogen |
| `"NOx"` | nitrate plus nitrite nitrogen |
| `"TN"` | `TKN + NOx`; dissolved N₂, N₂O and NO are in no nitrogen channel |
| `"TP"` | total phosphorus |
| `"TSS"` | the model's own TSS species, or 0.75 × the particulate COD for the ASM1 family |
| `Measure(name, weights, keywords, flow, temperature)` | any other weighted sum |
| `"BOD5 = 0.65*[SS] + 0.65*[XS]"` | the same weighted sum, written out |

COD, TKN, NOx, TN and TP come from the model's composition table, so their
weights follow the composition parameters of the run (`i_XB`, `iN_BM`, ...): a
fit or a Fisher information with `"asm1.i_XB"` free sees total nitrogen move
with it. A digester's headspace (`S_gas_*`) is in no channel. Channels work
everywhere an observable does: calibration, Fisher information, profile
likelihood and sensor ranking.

```python
from aquakin.plant import Measure, PlantObservable, SensorSet

SensorSet([PlantObservable("effluent", ("TKN", "NOx", "TSS", "Q"))], sigma=[1.0, 0.5, 2.0, 50.0])
```

### A channel of your own

A channel the table does not cover is written as a weighted sum: `name =` and
then the sum, with the species in brackets and the measurements above named as
they are:

```python
PlantObservable("effluent", ("SNH", "BOD5 = 0.65*[SS] + 0.65*[XS]", "sCOD = COD - [XS] - [XI]"))
```

It is the same thing as `Measure("BOD5", {"SS": 0.65, "XS": 0.65})`, in a form
that also fits in a study file and in the flowsheet editor, where it is typed
straight into the channels field of the study dialog. Each is reported under
its name, so the one above gives the columns `effluent.BOD5` and
`effluent.sCOD`.

A channel built on `COD`, `TN`, ... carries their dependence on the composition
parameters with it, so `"half_TN = 0.5*TN"` moves with a fitted `i_XB` exactly
as `TN` does.

The sum has to stay **linear** in the stream: `*` needs a plain number on one
side, `/` a plain number on the right, and there is no constant term (so a
temperature in °C is not a channel). This keeps every channel a column of one
measurement matrix, which lets a sensor ranking score many candidate designs
from a single solve. An expression that breaks the rule, or cannot be read at
all, raises `MeasureSyntaxError` (a `ValueError`) when a study using it is
written, or in Python when a fit or analysis first uses it.

```text
MeasureSyntaxError: a measure is a weighted sum, so it cannot multiply two
measured quantities (at position 8); one side of a '*' must be a number
```

`aquakin.plant.parse_measure` reads one on its own, if you want to check an
expression or build the `Measure` it stands for.

### Bounds, initial states and noise in a plant fit

Everything in [Fitting parameters to data](#fitting-parameters-to-data) about
bounds, initial pools and noise applies to a plant fit too:

- The free parameters stay inside the `bounds:` their models declare, and
  `OptimizerConfig(bounds=...)` keys are plant parameter names such as
  `"asm1.muH"`. A fit that ends on a bound lists it in `result.at_bound`, with
  a `nan` Laplace standard deviation.
- The fitted initial states are named `"unit.species"` (also the keys of a
  per-pool `FreeICConfig(bounds={...})`), and `fixed_total=True` fits their
  split of the total in `y0`.
- With `loss="nll"`, `sigma="estimate"` estimates the noise level of each
  observed channel.
- `observation_units={"SNH": "mg/L"}` declares channels measured in other
  units, converted to each stream's model units as for a reactor fit.

### Progress and cancelling

`plant.calibrate`, `plant_fisher_information`, `plant_profile_likelihood` and
`plant_rank_sensor_sets` report their progress to a `progress=` callable,
which receives an `aquakin.Progress` (`stage`, `done`, `total`, `message`) at
each objective evaluation, profile point or run. Raising `aquakin.Cancelled`
from it stops the analysis.

### Studies

A plant calibration, profile likelihood, Fisher information or sensor ranking
can also be saved as a *study*, run from the command line or the flowsheet
editor, with its results kept beside it and checked for staleness. Calibration
and profile studies always fit the Gaussian likelihood, so they need each
channel's `sigma` or `estimate_sigma: true`. See
[Studies](plants.md#studies).

### Identifiability of plant parameters

`aquakin.plant.plant_fisher_information` and
`aquakin.plant.plant_profile_likelihood` are the plant versions of
`fisher_information` and `profile_likelihood`. They name the measured channels
the way `plant.calibrate` does, and work in its coordinates, so their results
feed `aquakin.identifiability` and `aquakin.design_criterion` directly.

% skip: next "a BSM2 plant calibration that needs measured data; its API is checked by the plant calibration tests"

```python
from aquakin.plant import plant_fisher_information, plant_profile_likelihood

fim_plant = plant_fisher_information(
    plant,
    [("effluent", ["SNH", "SNO"]), ("digester", ["S_IN", "S_gas_ch4"])],
    t_obs,
    ["asm1.muA", "asm1.KNH_A", "adm1.k_m_ac"],
    sigma=[0.2, 0.5, 0.01, 0.005],   # one per channel, in observable order
    y0=bsm2_warm_start(plant),
)
rep_plant = aquakin.identifiability(fim_plant)
fim_plant.std()                      # Cramér-Rao standard deviations
```

A unit name reads that unit's outlet, so a reactor's or the digester's state
can be measured directly, as above. `sigma` is absolute by default. With
`relative_sigma=True` it is a fraction of the predicted signal, with
`sigma_floor` as a lower bound. `mask=` leaves out samples that are not taken.
A list of `t_obs` arrays describes several runs whose information adds. `y0`,
`t_span`, `sigma`, `mask` and the plant itself can then be lists too, one per
run, for runs at different initial states or influents.

The sensitivity comes from `plant.solve_sensitivity` by default
(`mode="forward"`), whose memory does not grow with the number of
observations, so it suits long horizons. `mode="reverse"` differentiates
`plant.solve` with the cap-free adjoint, one pass per observation, which suits
a few observations of many parameters. At a `plant.calibrate` result with
`loss="nll"`, no priors and `LaplaceConfig(method="gauss_newton")`, the
reverse-mode matrix equals the fit's `hessian_unconstrained`.

To choose what to measure, `aquakin.plant.plant_rank_sensor_sets` scores
candidate `SensorSet`s (observables plus their noise) by a design criterion of
the information each adds to the sensors already in place (`base=`). One
sensitivity solve is shared by every candidate, so each extra candidate costs
only linear algebra. It returns the same `CandidateScore` list as
`rank_experiments`, best first, and takes `criterion=`, `subset=` and
`param_draws=` as that does:

% skip: next "a BSM2 plant calibration that needs measured data; its API is checked by the plant calibration tests"

```python
from aquakin.plant import SensorSet, plant_rank_sensor_sets

online = SensorSet([("effluent", ["SNH"])], sigma=0.2)
sensor_candidates = [
    SensorSet([("effluent", ["SNO"])], sigma=0.5),
    SensorSet([("tank3", ["SO"])], sigma=0.1),
    SensorSet([("digester", ["S_IN"])], sigma=0.05, relative_sigma=True),
]
ranked_sensors = plant_rank_sensor_sets(plant, sensor_candidates, t_obs,
                                        ["asm1.muA", "asm1.KNH_A"], base=online,
                                        y0=bsm2_warm_start(plant))
ranked_sensors[0].index, ranked_sensors[0].improvement
```

`plant_profile_likelihood` re-fits the other parameters with `plant.calibrate`
at each grid value of the profiled one, with the same warm-started sweep as
`profile_likelihood`, and every grid point reuses one compiled objective.
Other keywords go to `plant.calibrate`:

% skip: next "a BSM2 plant calibration that needs measured data; its API is checked by the plant calibration tests"

```python
prof_plant = plant_profile_likelihood(
    plant, observations, t_obs, ["asm1.muA", "asm1.KNH_A"],
    grid=np.linspace(0.4, 1.2, 9), profile_param="asm1.muA",
    observables=[("effluent", ["SNH"])], sigma=jnp.asarray([0.2]),
    y0=bsm2_warm_start(plant))
prof_plant.ci
```

## Uncertainty propagation and design

Three higher-level tools take a function of an input vector, as `dgsm` does.
Here they reuse `bromate_umol` from the [DGSM example](#global-sensitivity-dgsm),
the final bromate as a function of `[k1, k_dec]`:

- `aquakin.monte_carlo` propagates uncertain inputs (uniform, normal or
  lognormal marginals) through the model to an output ensemble.
- `aquakin.compare_scenarios` runs several named input sets side by side and
  tabulates the outputs.
- `aquakin.optimize_design` minimizes (or maximizes) an objective over bounded
  design variables subject to constraints, using AD gradients.

```python
mc = aquakin.monte_carlo(
    bromate_umol,
    {"k1": {"dist": "lognormal", "mean": 160.0, "std": 40.0},   # physical mean, std
     "k_dec": (35.0, 140.0)},                                   # (low, high): uniform
    n_samples=64, seed=0)
mc.percentiles((2.5, 50.0, 97.5))     # (3, 1): per output
print(mc.summary())

scen = aquakin.compare_scenarios(
    bromate_umol, {"slow": [80.0, 70.0], "fast": [320.0, 70.0]},
    input_names=["k1", "k_dec"])
scen.output_named("output")           # one value per scenario
```

`optimize_design` takes the objective and each constraint as functions of the
design variables. Here the design variable is the ozone dose in µmol/L, and
the largest dose is sought that keeps bromate at 600 s below 0.1 µmol/L:

```python
def bromate_at_dose(x):              # x = [ozone dose, µmol/L]
    sol = reactor.solve(C0.at[i["O3"]].set(1e-6 * x[0]), t_span=(0.0, 600.0),
                        t_eval=t_obs, params=params)
    return 1e6 * sol.C_named("BrO3-")[-1]

opt = aquakin.optimize_design(
    objective=lambda x: x[0], maximize=True,
    bounds=[(20.0, 200.0)], input_names=["O3_dose"],
    constraints=[aquakin.Constraint(fn=bromate_at_dose, upper=0.1, name="BrO3-")],
    x0=[50.0])
opt.x_named, opt.feasible     # about 42.7 µmol/L, True
```

## Choosing the integrator

The integrator is configured with one value object, `aquakin.IntegratorConfig`,
passed as `integrator=` to a reactor or to `plant.solve`. Its defaults are:

```python
aquakin.IntegratorConfig(order=3, factormax=3.0, colored_jacobian="auto",
                         dtmax=None, max_steps=100_000, solver=None)
```

- `order`: the ESDIRK method, `3` (`Kvaerno3`, the default) or `5` (`Kvaerno5`,
  higher order and more robust). Raise it to 5 if a solve struggles to
  converge.
- `factormax`: caps how fast the step size may grow from one step to the next.
- `dtmax`: caps the step. Set it **only** for a reverse-mode gradient taken
  *through* a stiff solve (`method="through_solve"`); it is in the model's
  time unit. The cap-free methods above do not need it.
- `colored_jacobian`: builds the implicit-step Jacobian sparsely. It applies to
  plants only; reactors ignore it.
- `max_steps`: the step budget; `solver` overrides `order` with an explicit
  Diffrax solver.

`rtol` and `atol` stay separate arguments of the reactor and of `plant.solve`.
For faster plant solves, including `forward_fast`, see
[Plant-wide simulation](plants.md#faster-dynamic-solves).
