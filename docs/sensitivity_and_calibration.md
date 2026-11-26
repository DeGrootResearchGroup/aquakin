# Sensitivity analysis

Every solve in `aquakin`, of a single reactor or a whole plant, is
**differentiable**. Rate constants are always passed in as arguments (never
baked into the rate functions), so you can take exact gradients of any output
with respect to parameters, conditions or initial states, and use them for
sensitivity analysis, parameter estimation, experiment design and design
optimization.

A plain forward simulation never needs any of this. Reach for it when you want
to know *how* an output depends on its inputs, or to fit a model to data. This
page covers gradients, sensitivity analysis and uncertainty propagation; the
other questions have pages of their own.

## Which tool do I need?

| Question | Reactor | Plant | Page |
|---|---|---|---|
| How does an output change with each parameter, here? | `aquakin.sensitivity`, `reactor.solve_sensitivity` | `plant.steady_state_sensitivity`, `plant.dynamic_sensitivity`, `plant.solve_sensitivity` | [Sensitivity analysis](#sensitivity-analysis) |
| Which inputs matter over a whole range? | `aquakin.dgsm`; `aquakin.sobol_indices` for first-order and interaction shares | `plant.steady_state_dgsm`, `plant.dynamic_dgsm` | [Global sensitivity](#global-sensitivity-dgsm) |
| How uncertain is a prediction? Which design is best? | `aquakin.monte_carlo`, `compare_scenarios`, `optimize_design` | the same | [Uncertainty propagation](#uncertainty-propagation-and-design) |
| Which parameter values fit my data? | `aquakin.calibrate` (`aquakin.fit` for a point estimate only) | `plant.calibrate` | [Calibration](calibration.md), [Calibrating a plant](plant_calibration.md) |
| How well do the data determine them? | the Laplace posterior, `aquakin.sample_posterior`, `aquakin.profile_likelihood`, `aquakin.identifiability` | the same, with `plant.sample_posterior` and `plant_profile_likelihood` | [How certain is the fit?](calibration.md#how-certain-is-the-fit) |
| Which model structure do the data support? | `aquakin.aic`, `aicc`, `bic`, `compare_models` | the same | [Comparing model structures](calibration.md#comparing-model-structures) |
| What should I measure next? | `aquakin.fisher_information`, `rank_experiments`, `optimize_experiment`, `design_from_candidates` | `plant_fisher_information`, `plant_rank_sensor_sets` | [Experiment design](experiment_design.md), [Identifiability of plant parameters](plant_calibration.md#identifiability-of-plant-parameters) |

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
sens.ranked_params()              # [(name, |elasticity|), ...] most influential first
```

`ranked_params()` ranks by the **elasticity** `(dy/dp)·p/y`, the percentage
change in the output per percentage change in a parameter
(`sens.elasticities()` returns them). The raw gradients in `doutput_dparams`
depend on each parameter's units: this model's rate constants span from about
10 to 10⁹, so a large gradient can belong to a parameter that barely matters.
`ranked_params(elasticity=False)` ranks by the raw gradient instead.

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

### Variance-based global sensitivity (Sobol indices)

DGSM gives an upper bound on each input's total effect. It does not say what
fraction of the output variance an input explains on its own, or which inputs
interact. `aquakin.sobol_indices` estimates both, for each input `z_j`:

- the **first-order index** `S_j`, the fraction of the output variance that
  `z_j` explains on its own (Saltelli et al. 2010 estimator);
- the **total-order index** `S_j^tot`, the same fraction including every
  interaction with other inputs (Jansen 1999 estimator);
- their difference, `res.interaction`, the share that `z_j` explains only
  through interactions.

It takes the same function as `dgsm`, and its inputs are described as for
`monte_carlo`: a `(low, high)` range, or a uniform, normal or lognormal
distribution. The function does not need to be differentiable. As with
`monte_carlo`, it can also wrap a dynamic plant solve
([Uncertainty and scenarios](custom_flowsheets.md#uncertainty-and-scenarios)).

The decomposition costs much more than DGSM. For `d` inputs and a base sample
of `N` points it runs `N (d + 2)` forward solves, where DGSM runs about `N`
differentiated ones. At `N = 1024` and 17 inputs that is about 19 000 solves.
Check the count with `sobol_indices_cost` before you start:

| Method | Gives | Solves for `d` inputs, base sample `N` |
|---|---|---|
| `aquakin.dgsm` | an upper bound on each `S_j^tot`; a ranking | about `N` differentiated solves |
| `aquakin.sobol_indices` | `S_j`, `S_j^tot`, interactions, confidence intervals | `N (d + 2)` forward solves |

Use DGSM to screen many inputs. Then use Sobol indices on the few that matter,
or on a cheap model.

```python
aquakin.sobol_indices_cost(2, 64)     # 256 solves for the two rate constants

sob = aquakin.sobol_indices(bromate_umol, {"k1": (80.0, 320.0), "k_dec": (35.0, 140.0)},
                            n_samples=64, seed=0,
                            bootstrap=aquakin.BootstrapConfig(confidence=0.9))
sob.first_order          # (2,) share of Var(bromate) each rate explains alone
sob.total_order          # (2,) the same including interactions
sob.interaction          # (2,) total minus first order
sob.total_order_ci       # (2, 2) bootstrap 90 % intervals, from 1000 resamples
```

% invisible-code-block: python
%     assert aquakin.sobol_indices_cost(2, 64) == 256 == sob.n_evaluations
%     assert (sob.confidence, sob.n_bootstrap) == (0.9, 1000)
%     assert np.all(sob.total_order <= np.asarray(res.sobol_total_bound))

The confidence intervals come from a bootstrap over the base sample. The
bootstrap costs no extra solves. `bootstrap=aquakin.BootstrapConfig(...)` sets
the number of resamples (`n_resamples`, default 1000, at least 100) and the
interval level (`confidence`, default 0.95); raise `n_resamples` for a high
level such as 0.99, whose tails hold few resamples. The bootstrap treats the
sample as independent random draws, so on the scrambled-Sobol design used here the intervals are
conservative: they are usually wider than the actual error. Increase
`n_samples` until the intervals are narrow enough to separate the inputs you
want to compare. With a small sample an index near zero can come out slightly
negative, which is sampling noise. The indices assume that the inputs are
independent.

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
