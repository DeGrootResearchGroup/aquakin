# Worked example: designing an experiment

This page works through an identifiability study and an experiment design for
a small ozone-demand model, from "can these parameters be estimated at all?" to
a calibrated model with checked uncertainties. It uses the tools described in
[Sensitivity & calibration](sensitivity_and_calibration.md): the Fisher
information, the identifiability report, design criteria and optimisation, and
calibration with a Laplace posterior and profile likelihood.

The complete script is
[`examples/ozone_demand_design.py`](https://github.com/DeGrootResearchGroup/aquakin/blob/main/examples/ozone_demand_design.py),
and the model is
[`examples/ozone_demand.yaml`](https://github.com/DeGrootResearchGroup/aquakin/blob/main/examples/ozone_demand.yaml).
Every number on this page is printed by that script, which runs in about a
minute.

## The model

Ozone dosed into a water sample is consumed in two ways. A small amount of
fast-reacting material D (the "instantaneous ozone demand") is used up within
seconds. Organic matter M reacts slowly and is present in large excess over
ozone.

| Reaction | Rate | True value |
|---|---|---|
| O₃ + D → products | k₁ [O₃][D] | k₁ = 0.01 L µmol⁻¹ s⁻¹ |
| O₃ + M → products | k₂ [O₃][M] | k₂ = 2×10⁻⁶ L µmol⁻¹ s⁻¹ |

Concentrations are in µmol/L and time in seconds. The ozone dose is known
(50 µmol/L to start with). There are four unknowns: the two rate constants and
the two initial pools, [D]₀ = 15 µmol/L and [M]₀ = 1000 µmol/L. The data are
synthetic, generated from these true values, so every result can be checked.

```python
model = aquakin.load_model_from_file("examples/ozone_demand.yaml")
reactor = aquakin.BatchReactor(model, model.default_conditions(),
                               rtol=1e-8, atol=1e-8)

free_params = ["fast.k1", "slow.k2"]
free_ic = aquakin.FreeICConfig(["D", "M"], bounds=(1e-2, 1e5))
```

Both rate constants are declared with `transform: positive_log`, so they are
analysed on a log scale, like the initial pools.

## Step 1: diagnose a simple design

A natural first experiment doses 50 µmol/L of ozone and measures it five times
between 60 and 600 s, with a measurement standard deviation of 1 µmol/L:

```python
base = aquakin.Experiment(C0, t_obs=jnp.linspace(60, 600, 5), sigma=1.0,
                          observed_species=["O3"])
fim = aquakin.fisher_information(reactor, base, free_params, free_ic=free_ic)
rep = aquakin.identifiability(fim)
rep.unidentifiable
```

The report flags two directions:

| Relative eigenvalue | Largest loadings |
|---|---|
| ≈ 0 (−7×10⁻¹⁷) | `fast.k1`: 1.00 |
| 8×10⁻⁷ | `slow.k2`: −0.71, `ic[0].M`: +0.70 |

Each line is a parameter combination the design cannot determine.

- **k₁ is invisible.** D is used up within about 20 s, before the first
  sample. The samples show how much ozone D removed, which is [D]₀, but not how
  fast. The information about k₁ is exactly zero.
- **Only k₂[M]₀ is seen.** Equal and opposite loadings on log k₂ and log [M]₀
  mean that raising one and lowering the other by the same factor leaves the
  ozone curve unchanged. M is in such excess that it is barely consumed (about
  3 % over 30 minutes), so ozone decays at the apparent first-order rate
  k₂[M]₀, and only that product can be estimated.

Because the information matrix is singular, every quantity's standard
deviation is unbounded (`rep.relative_std` reports `inf`) and the D-criterion
is −∞. Fitting this data set would return *some* values of k₁, k₂ and [M]₀,
but they would mean nothing.

## Step 2: compare ways to improve the run

Each fix must add information along the flagged directions. The script
evaluates four changes to the same run, and reports each quantity's relative
standard deviation (the Cramér-Rao lower bound, `rep.relative_std`):

| Design | k₁ | k₂ | [D]₀ | [M]₀ | D-criterion |
|---|---|---|---|---|---|
| base: 5 ozone samples, 60–600 s | ∞ | ∞ | ∞ | ∞ | −∞ |
| + 5 late samples, 700–1800 s | ∞ | ∞ | ∞ | ∞ | −21.5 |
| + 5 early samples, 2–20 s | 21 % | 1545 % | 4.6 % | 1526 % | 10.9 |
| + UV254 at the same times | ∞ | 6.6 % | ∞ | ∞ | −6.7 |
| + early samples and UV254 | 20 % | 5.0 % | 3.7 % | 0.16 % | 29.3 |

More ozone samples, however late, do not help: every ozone sample sees the
same product k₂[M]₀. Two different measurements are needed, one for each
flagged direction:

- **Early ozone samples** (2–20 s) catch D being consumed, which gives k₁.
- **UV254**, the UV absorbance at 254 nm per centimetre of path length,
  measures organic matter directly. Absorbance (not transmittance) is
  proportional to concentration, so A₂₅₄ = ε[M]. Assuming the specific
  absorbance ε is known (here 2×10⁻⁴ cm⁻¹ per µmol/L, with a measurement noise
  of 0.001 cm⁻¹), the absorbance gives [M]₀, and the ozone decay then gives k₂.

UV254 is a function of the state rather than a species, so it is passed as an
observation function returning both measured quantities, with one standard
deviation for each:

```python
eps = 2e-4
i_o3, i_m = model.species_index["O3"], model.species_index["M"]
ozone_and_uv = lambda C, p: jnp.stack([C[i_o3], eps * C[i_m]])

run = aquakin.Experiment(C0, t_obs, sigma=jnp.asarray([1.0, 0.001]),
                         observation_fn=ozone_and_uv)
```

Only the combination of both changes makes all four quantities identifiable.
When the candidates are whole alternative designs like these, the same ranking
is available directly from `aquakin.rank_experiments`.

## Step 3: optimise the sampling times and dose

With ozone and UV254 measured, the remaining choices are continuous: when to
take the samples and how much ozone to dose. `optimize_experiment` moves them
to maximise the D-criterion, starting from eight evenly spaced samples
between 5 and 900 s:

```python
res = aquakin.optimize_experiment(
    reactor, start, free_params, free_ic=free_ic,
    time_bounds=(1.0, 1800.0), min_spacing=5.0,   # seconds
    initial_bounds={"O3": (20.0, 150.0)},         # dose range, umol/L
)
res.experiments[0].t_obs, res.experiments[0].C0
```

The optimiser converges with the D-criterion rising from 26.7 to 31.3. It
places three samples as early as allowed (1, 6 and 11 s) and five together
around 520 s (517–537 s), and raises the dose to its upper limit of
150 µmol/L.

| Quantity | Evenly spaced, 50 µmol/L | Optimised |
|---|---|---|
| k₁ | 43 % | 23 % |
| k₂ | 5.9 % | 1.1 % |
| [D]₀ | 9.6 % | 4.8 % |
| [M]₀ | 0.22 % | 0.18 % |

```{figure} images/ozone_demand_design.png
:alt: Ozone and UV254 curves with the base and optimised sampling times.

True ozone and UV254 curves at the starting dose (solid) and the optimised
dose (dashed), with the base sampling times (blue, bottom) and the optimised
ones (red, top).
```

Two features of the result are typical of D-optimal designs.

- **Samples cluster.** Rather than spreading out, the samples sit where the
  measurements are most sensitive to the parameters: the early cluster resolves
  the fast demand, and the cluster near 520 s is where ozone is most sensitive
  to k₂. The minimum spacing (`min_spacing`) is what keeps them apart. In
  practice, also check that the model is right between the clusters, for
  example with a few extra samples or a validation run.
- **The dose goes to its limit.** A larger dose consumes more organic matter,
  so UV254 changes more, and the slow phase lasts longer. The upper bound
  should reflect what is practical in the lab.

## Step 4: calibrate and check the uncertainty

To check the design, the script simulates noisy measurements of the optimised
run and calibrates the model, starting away from the truth (k₁ = 0.003,
k₂ = 5×10⁻⁶, [D]₀ = 5, [M]₀ = 500):

```python
result = aquakin.calibrate(
    reactor, C0_guess, data, t_obs, free_params,
    initial_params=guess, observation_fn=ozone_and_uv,
    loss="nll", sigma=jnp.asarray([1.0, 0.001]), free_ic=free_ic,
    laplace=aquakin.LaplaceConfig(method="gauss_newton", include_ic=True,
                                  eig_keep=1e-10),
    optimizer=aquakin.OptimizerConfig(method="gauss_newton", n_starts=4),
)
```

| Quantity | Estimate ± standard deviation | True value |
|---|---|---|
| k₁ | 0.0099 ± 0.0023 (23 %) | 0.01 |
| k₂ | (2.033 ± 0.022)×10⁻⁶ (1.1 %) | 2×10⁻⁶ |
| [D]₀ | 14.86 ± 0.72 (4.8 %) | 15 |
| [M]₀ | 1000.0 ± 1.8 (0.18 %) | 1000 |

Every estimate is within its uncertainty of the true value, and the standard
deviations match those predicted at the design stage (step 3).

```{note}
`include_ic=True` matters here. It makes the Laplace posterior cover the
initial pools too, so the rate-constant uncertainties allow for the trade-off
between each rate and its pool. Without it, the pools are treated as known.

So does `eig_keep`. By default the Laplace posterior drops directions whose
information is less than 1 % of the largest. Here the information spans about
four orders of magnitude ([M]₀ is fixed precisely by UV254, k₁ only by three
early samples), so the default would drop the k₁ direction and report a
standard deviation for k₁ that is far too small. Use `identifiability` on the
result to see how far the eigenvalues spread, and lower `eig_keep` when every
quantity is identifiable but some are much better determined than others.
```

The Laplace posterior is a local approximation. A profile likelihood checks it
without that assumption: it fixes k₂ on a grid, refits the other quantities at
each point, and finds where the fit becomes significantly worse.

```python
prof = aquakin.profile_likelihood(
    reactor, result.C0_fitted[0], data, t_obs, free_params,
    grid=grid, profile_param="slow.k2", observation_fn=ozone_and_uv,
    loss="nll", sigma=jnp.asarray([1.0, 0.001]), free_ic=free_ic,
)
prof.ci
```

```{figure} images/ozone_demand_profile.png
:alt: Profile likelihood of k2 with the Laplace interval and the true value.
:width: 65%

Profile likelihood of k₂. The shaded band is the Laplace 95 % interval and the
vertical line is the true value.
```

The profile gives a 95 % interval for k₂ of (1.992–2.076)×10⁻⁶, which agrees
with the Laplace interval and contains the true value.

## Step 5: design the next run

The design in step 3 was optimised for the true parameter values, which are
not known in practice. After a first run, a second one can be designed for the
values the data now support, averaged over their remaining uncertainty:

```python
draws = result.posterior_draws(model, n_draw=10, seed=1)
res2 = aquakin.optimize_experiment(
    reactor, start, free_params, free_ic=free_ic,
    time_bounds=(1.0, 1800.0), min_spacing=5.0,
    initial_bounds={"O3": (20.0, 150.0)}, param_draws=draws,
)
```

The robust design is almost the same: early samples at about 1, 6 and 11 s, a
late cluster from 509 to 529 s, and the maximum dose. That is expected when the
parameters are already well determined. When the first-round estimates are
uncertain, averaging over the posterior protects the second round from being
designed around the wrong values.

## Summary

- Compute the Fisher information of a planned design before running it.
  `identifiability` names the parameter combinations it cannot determine.
- More samples of the same quantity rarely fix a structural problem: here no
  number of ozone samples separates k₂ from [M]₀. A different measurement, or
  samples in a different time window, does.
- Use `rank_experiments` to compare discrete options and `optimize_experiment`
  for sampling times and doses.
- After calibrating, compare the estimated uncertainties with the design-stage
  predictions, and confirm key parameters with a profile likelihood.
