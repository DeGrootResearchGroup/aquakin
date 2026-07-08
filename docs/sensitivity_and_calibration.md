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
    n_starts=24, seed=0,              # multistart escapes local minima
)
result.params_named       # MAP estimate, in physical space, by name
result.params_named_std   # marginal standard deviations
result.converged
```

`observations` is shape `(n_t,)` for one species or `(n_t, n_observed)` for
several; `t_obs` is in the model's native time unit unless you pass `time_unit=`.
Pass **lists** of `C0`/`observations`/`t_obs` for a joint multi-batch fit, and
`free_ic=[...]` to fit unmeasured initial pools alongside the rate constants.

### Uncertainty: posterior-predictive bands

With `laplace=True`, sample the posterior and propagate it through a solve to get
per-timepoint predictive envelopes (the `C0` may be a held-out validation batch):

```python
band = result.predictive_band(reactor, C0, t_eval, n_draw=200,
                              percentiles=(2.5, 97.5))
band.median, band.lo, band.hi        # (n_t, n_species) envelopes
```

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
