# Calibrating a plant

`plant.calibrate` is the plant version of `aquakin.calibrate`
([Calibration](calibration.md)): it fits plant parameters, by name, against
measured channels of one or more streams or units. It uses the same
transforms, [priors](calibration.md#priors), multistart and Laplace posterior,
behind the plant's cap-free adjoint, so the gradient is finite for a stiff
plant with no `dtmax` to tune. Two defaults differ from `aquakin.calibrate`:
the loss is `"mse"` and `laplace=False`, so pass `loss="nll"`, `sigma=` and
`laplace=True` for uncertainties.

```{warning}
The shipped `asm1` and `adm1` models declare priors on their calibration
parameters, and `plant.calibrate` applies them by default (`use_priors=True`).
With the default `loss="mse"` the prior is added to a mean squared error in the
square of the data's units, so how strongly it pulls the fit depends on the
units the data happen to be in. Fit with `loss="nll"` and each channel's
`sigma` (or `sigma="estimate"`) so the prior is weighed against a likelihood,
or pass `use_priors=False`.
```

## The data of a plant fit

- **Times are on the plant's clock.** `t_obs` is in the plant's time unit
  (`Plant(name, time_unit=...)`; days for the BSM plants) unless you pass
  `time_unit=`.
- **The solve runs over `t_span`**, which defaults to `(t_obs[0], t_obs[-1])`.
  Starting it earlier, `t_span=(t_start, t_obs[-1])` with `t_start` before the
  first sample, gives a spin-up window in which a mismatch between `y0` and the
  real plant decays before the first observation is compared.
- **`y0` is the state at `t_span[0]`.** Without it the fit starts from
  `plant.initial_state()`, the units' own default states, which need not
  resemble the plant's operating point. Pass a steady state
  (`plant.steady_state().state`) or a warm start (`bsm1_warm_start(plant)`,
  `bsm2_warm_start(plant)`), ideally under the influent that preceded the
  data; a BSM2 plant should always be warm-started.
- **What is measured** is one stream with `target=` and `observed_channels=`,
  or several with `observables=`, a list of observables. Each observable is a
  stream (a registered name such as `"effluent"`, a `"unit.port"`, or a unit
  name for that unit's outlet) with its channels, written as
  `PlantObservable("effluent", ["SNH", "SNO"])` or the pair
  `("effluent", ["SNH", "SNO"])`. **The columns of `observations` run in
  observable order, and in channel order within each**: two observables with
  two and one channels give three columns. A channel can be a species or one
  of the [measured channels](#measured-channels) below; `NaN` marks a missing
  sample, as for a reactor.

The examples below use BSM2, built with its influent wired as in
[Plant-wide simulation](plants.md#the-benchmark-plants), and need a measured
`observations` array. A BSM2 solve takes minutes, and a fit many solves, so
they are not run on this page.

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
passing list-valued `observations`, `t_obs` and `y0`. How well the fitted
values are determined, and how to report them, works as for a reactor
([Uncertainty and model comparison](fit_uncertainty.md)); the plant tools for
it are below.

## Measured channels

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
from aquakin.plant import Measure, PlantObservable

PlantObservable("effluent", ("TKN", "NOx", "TSS", "Q"))
```

## A channel of your own

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

## Bounds, initial states and noise in a plant fit

Everything in [Fitting parameters to data](calibration.md#fitting-parameters-to-data) about
bounds, initial pools and noise applies to a plant fit too:

- The free parameters stay inside the `bounds:` their models declare, and
  `OptimizerConfig(bounds=...)` keys are plant parameter names such as
  `"asm1.muH"`. A fit that ends on a bound lists it in `result.at_bound`, with
  a `nan` Laplace standard deviation.
- The fitted initial states are named `"unit.species"` (also the keys of a
  per-pool `FreeICConfig(bounds={...})`), and `fixed_total=True` fits their
  split of the total in `y0`.
- With `loss="nll"`, `sigma` gives one noise level per observed column, in
  column order, and `sigma="estimate"` estimates the noise level of each.
- `observation_units={"SNH": "mg/L"}` declares channels measured in other
  units, converted to each stream's model units as for a reactor fit.

## Progress and cancelling

`plant.calibrate`, `plant_fisher_information`, `plant_profile_likelihood` and
`plant_rank_sensor_sets` report their progress to a `progress=` callable,
which receives an `aquakin.Progress` (`stage`, `done`, `total`, `message`) at
each objective evaluation, profile point or run. Raising `aquakin.Cancelled`
from it stops the analysis.

## Studies

A plant calibration, profile likelihood, Fisher information or sensor ranking
can also be saved as a *study*, run from the command line or the flowsheet
editor, with its results kept beside it and checked for staleness. Calibration
and profile studies always fit the Gaussian likelihood, so they need each
channel's `sigma` or `estimate_sigma: true`. See
[Studies](plant_files.md#studies).

## Identifiability of plant parameters

`aquakin.plant.plant_fisher_information` and
`aquakin.plant.plant_profile_likelihood` are the plant versions of
`fisher_information` ([Experiment design](experiment_design.md)) and
`profile_likelihood` ([Profile likelihood](fit_uncertainty.md#profile-likelihood));
the ideas behind them are in [Key ideas](analysis_concepts.md). They name the measured channels
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
reverse-mode matrix equals the fit's `hessian_unconstrained` (both are the
[Fisher information](analysis_concepts.md#fitting-likelihood-posterior-and-the-laplace-approximation)).

To choose what to measure, `aquakin.plant.plant_rank_sensor_sets` scores
candidate sensor sets by a design criterion of the information each adds to
the sensors already in place (`base=`). A `SensorSet` is a list of
observables, in either form, with the noise of their channels: `sigma` is one
value for every channel or one per channel in column order, absolute unless
`relative_sigma=True`. One
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
