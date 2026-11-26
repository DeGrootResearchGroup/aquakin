# Experiment design

Before running an experiment you can ask whether it will determine the
parameters you care about, and choose the design that determines them best.
The tools work on the Fisher information of a design at assumed parameter
values; no data are needed.
[Worked example: designing an experiment](worked_example_design.md) explains
the ideas behind them step by step.

The experiment-design examples share the setup of
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

result = aquakin.calibrate(      # a fit with its Laplace posterior, used below
    reactor, C0, observations, t_obs, free_params, observed_species=observed,
    loss="nll", sigma=sigma, optimizer=aquakin.OptimizerConfig(n_starts=4, seed=0))
```

## Fisher information of a design

`aquakin.fisher_information` evaluates the Fisher information of a set of
experiments, each an `aquakin.Experiment` (initial state, sampling times,
noise, what is measured, and optionally its own `reactor` for different
operating conditions):

```python
reactor_ph8 = aquakin.BatchReactor(
    model, aquakin.SpatialConditions.uniform(1, pH=8.0, OH_scavenging=5e4))
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

## Fisher information of any forward model

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

## Scoring and optimizing a design

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

## Choosing measurements from a candidate pool

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
