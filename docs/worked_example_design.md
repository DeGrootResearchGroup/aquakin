# Worked example: designing an experiment

Before fitting a kinetic model to data, it is worth asking two questions:

1. **Can the data determine the parameters at all?** A model can fit a data set
   well while some of its parameters are almost arbitrary, because changing
   one can be compensated by changing another.
2. **Which experiment would determine them best?** Sampling times, doses, and
   the choice of what to measure all change how much a run can tell us.

This page answers both for a small ozone-demand model, step by step. It starts
from a first design that can determine only one of the four unknowns and ends
with a calibrated model whose uncertainties are checked against the truth.

All the code on this page comes from the script
[`examples/ozone_demand_design.py`](https://github.com/DeGrootResearchGroup/aquakin/blob/main/examples/ozone_demand_design.py),
and every output block is what that script prints. Each code block continues
from the previous one, so the page can be followed as one program. The script
takes a few minutes to run.

## The model

Ozone dosed into a water sample is consumed in two ways. A small amount of
fast-reacting material D (the "instantaneous ozone demand") is used up within
seconds. Organic matter M reacts slowly and is present in large excess over
ozone.

| Reaction | Rate | True value |
|---|---|---|
| O₃ + D → products | k₁ [O₃][D] | k₁ = 0.01 L µmol⁻¹ s⁻¹ |
| O₃ + M → products | k₂ [O₃][M] | k₂ = 2×10⁻⁶ L µmol⁻¹ s⁻¹ |

The model file is
[`examples/ozone_demand.yaml`](https://github.com/DeGrootResearchGroup/aquakin/blob/main/examples/ozone_demand.yaml).
Concentrations are in µmol/L and time in seconds. The ozone dose is chosen by
the experimenter (50 µmol/L to start with). There are four unknowns: the two
rate constants and the initial concentrations of D and M, [D]₀ = 15 µmol/L and
[M]₀ = 1000 µmol/L. These initial concentrations are not measured directly, so
they are estimated along with the rate constants. The page calls them the
initial *pools*.

Two quantities can be measured:

- **Dissolved ozone**, with a standard deviation of 1 µmol/L.
- **UV254**, the UV absorbance at 254 nm per centimetre of path length.
  Absorbance is proportional to concentration, so the example takes
  A₂₅₄ = ε[M] with a known specific absorbance ε = 2×10⁻⁴ cm⁻¹ per µmol/L,
  and a standard deviation of 0.001 cm⁻¹.

The data on this page are synthetic, generated from the true values above, so
every result can be checked against the truth.

The script starts with its imports. `aquakin` is imported before JAX because
importing it switches JAX to 64-bit floating point, which the stiff solves
need; imported after JAX, it still does so but warns that it is overriding
JAX's default.

```{literalinclude} ../examples/ozone_demand_design.py
:language: python
:start-after: "# [imports]"
:end-before: "# [end imports]"
```

UV254 is not a single species, so it is passed to aquakin as an *observation
function*, a function that maps the concentration vector at one time, and the
parameter vector, to the measured values. aquakin always calls it with those
two arguments, `f(C, p)`. Anything else it needs is bound in when it is
created, here by a small factory that takes the model and ε:

```{literalinclude} ../examples/ozone_demand_design.py
:language: python
:start-after: "# [observation]"
:end-before: "# [end observation]"
```

The setup code loads the model and defines the true composition, the unknowns,
and the measurements:

```{literalinclude} ../examples/ozone_demand_design.py
:language: python
:start-after: "# [setup]"
:end-before: "# [end setup]"
:dedent: 4
```

Rate constants declared with `transform: positive_log` in the model file, and
the initial pools, are analyzed on a log scale, so their standard deviations
read as relative ones.

The steps below use the Fisher information, the Cramér-Rao bound, the
eigen-directions of the information and the criteria that score a design.
[Key ideas: information, uncertainty and identifiability](analysis_concepts.md)
explains each of them; read it first if they are new to you.

## Step 1: diagnose a simple design

A natural first experiment doses 50 µmol/L of ozone and measures it five times
between 60 and 600 s. Its Fisher information and identifiability report are:

```{literalinclude} ../examples/ozone_demand_design.py
:language: python
:start-after: "# [step1]"
:end-before: "# [end step1]"
:dedent: 4
```

```{literalinclude} images/ozone_demand_output.txt
:language: text
:start-after: "== Step 1 =="
:end-before: "== Step 2 =="
```

In the output, `ic[0].D` and `ic[0].M` are the initial pools [D]₀ and [M]₀ of
experiment 0, the only experiment here. Two of the four eigenvalues are far
below the others, so two combinations of the unknowns are flagged. The loadings
say which, and the standard deviations show that they are
[different kinds of problem](analysis_concepts.md#structural-and-practical-identifiability):

- **k₁ on its own (structural).** Its eigenvalue is zero (the script prints
  round-off below 10⁻¹² as 0), and its standard deviation is infinite. D is
  used up within about 20 s, before
  the first sample. The samples show how much ozone D removed, which is [D]₀,
  but not how fast it did so.
- **log k₂ − log [M]₀ (practical).** The loadings are
  [equal and opposite](analysis_concepts.md#reading-a-direction-products-and-ratios),
  so the data determine mainly the product k₂[M]₀. M is in such excess that it is
  barely consumed (about 3 % over 30 minutes), so ozone decays at almost the
  apparent first-order rate k₂[M]₀. Doubling k₂ and halving [M]₀ gives nearly
  the same ozone curve. The small consumption of M separates them in
  principle, which is why the eigenvalue is not exactly zero, but only to a
  relative standard deviation of about 2000 %.

Fitting data from this design would return *some* values of k₁, k₂, and [M]₀,
and the fit could look good, but those values would mean almost nothing. Only
[D]₀ is reasonably determined. The Fisher information shows this before any
experiment is run.

## Step 2: compare ways to improve the run

A change to the design helps only if it adds information along the
undetermined directions, that is, if the new measurements are sensitive to
those combinations. The script compares four changes to the same run:

```{literalinclude} ../examples/ozone_demand_design.py
:language: python
:start-after: "# [step2]"
:end-before: "# [end step2]"
:dedent: 4
```

The table shows the relative standard deviation of each unknown
(the Cramér-Rao bound, as a percentage):

```{literalinclude} images/ozone_demand_output.txt
:language: text
:start-after: "== Step 2 =="
:end-before: "== Step 3 =="
```

- **More late samples help very little.** Every ozone sample after D is gone
  is sensitive almost only to the product k₂[M]₀, so their sensitivities nearly
  all point the same way. Five more of them, extending the run to 30 minutes,
  let a little more M be consumed. That halves the uncertainty of k₂ and [M]₀,
  but it is still far above 100 %, and k₁ stays undetermined.
- **Early samples** (2–20 s) catch D being consumed, which determines k₁ and
  improves [D]₀. They barely help k₂ and [M]₀.
- **UV254** measures M directly. Its sensitivities point along [M]₀, a
  direction the ozone samples hardly reach, so it determines [M]₀ and, through
  the product k₂[M]₀, also k₂. It leaves k₁ unknown.
- **Both together** make all four unknowns identifiable.

The last lines rank the same designs with `aquakin.rank_experiments`, which
scores each one by the D-criterion
([Scoring a design](analysis_concepts.md#scoring-a-design), larger is better). This is the quicker tool when there are many candidates. Early samples
plus UV254 come first. The three designs that leave k₁ structurally
undetermined all score −∞, however well they determine the other unknowns. The
D-criterion cannot order such designs among themselves, so compare them by
their standard deviations instead.

The general lesson is that more data of the same kind rarely fixes an
identifiability problem
([Combining measurements and experiments](analysis_concepts.md#combining-measurements-and-experiments)).
A different measurement, or samples in a different time window, does.

## Step 3: optimize the sampling times and dose

Once every unknown is identifiable, the design can be improved further by
moving the samples and changing the dose. `aquakin.optimize_experiment`
maximizes a [design criterion](analysis_concepts.md#scoring-a-design) (D by default) over the design. It
differentiates the criterion with respect to the sampling times and the
initial concentrations, and moves them with a gradient-based optimizer, within
bounds and keeping consecutive samples at least `min_spacing` apart. The
number of samples stays fixed. Here the run measures ozone and UV254 at eight
times, starting from evenly spaced samples between 5 and 900 s:

```{literalinclude} ../examples/ozone_demand_design.py
:language: python
:start-after: "# [step3]"
:end-before: "# [end step3]"
:dedent: 4
```

```{literalinclude} images/ozone_demand_output.txt
:language: text
:start-after: "== Step 3 =="
:end-before: "== Step 4 =="
```

```{figure} images/ozone_demand_design.png
:alt: Ozone and UV254 curves with the base and designed sampling times.

True ozone and UV254 curves at the starting dose (solid) and the designed dose
(dashed). Ticks mark the sampling times of the base design from step 1 (blue,
bottom) and of the designed run (red, top).
```

The D-criterion rises by 4.65. So det F grows by a factor of $e^{4.65} \approx 100$,
and the volume of the joint confidence region shrinks by a factor of
$e^{4.65/2} \approx 10$. The designed run roughly halves the uncertainty of k₁ and
[D]₀ and reduces that of k₂ by a factor of five. Two features of the result are typical of
D-optimal designs:

- **Samples cluster.** Rather than spreading out, the samples sit where the
  measurements are most sensitive to the unknowns. The early cluster resolves
  the fast demand, and the cluster near 520 s is where ozone is most sensitive
  to k₂. Only the minimum spacing keeps them apart. Because the design trusts
  the model completely, in practice it is wise to add a few samples between the
  clusters, or a validation run, to check the model there.
- **The dose goes to its limit.** A larger dose consumes more organic matter,
  so UV254 changes more and the slow phase lasts longer. The upper bound should
  reflect what is practical in the laboratory.

## Step 4: calibrate and check the uncertainty

To check the design, the script simulates noisy measurements of the designed
run from the true values, then fits the model to them as if the truth were
unknown.

```{literalinclude} ../examples/ozone_demand_design.py
:language: python
:start-after: "# [step4-data]"
:end-before: "# [end step4-data]"
:dedent: 4
```

### Fitting and the Laplace approximation

The fit minimizes the negative log-likelihood (`loss="nll"`) and reports the
uncertainty of its estimates by the
[Laplace approximation](analysis_concepts.md#fitting-likelihood-posterior-and-the-laplace-approximation).
With `method="gauss_newton"` the curvature it inverts is the Fisher
information at the fitted values, so the uncertainty after the fit can be
compared directly with the prediction made before the experiment.

```{literalinclude} ../examples/ozone_demand_design.py
:language: python
:start-after: "# [step4-fit]"
:end-before: "# [end step4-fit]"
:dedent: 4
```

```{literalinclude} images/ozone_demand_output.txt
:language: text
:start-after: "== Step 4 =="
:end-before: "Profile likelihood:"
```

The fit starts a factor of 2–3 away from the true rate constants and still
recovers every unknown within its uncertainty of the true value. The relative
standard deviations match those predicted for the designed run in step 3.

Three settings in the call matter:

- **`param_halfwidth=np.log(100.0)`** keeps each rate constant within a factor
  of 100 of its starting guess (the bound is on the log scale). Without it, an
  early optimizer step can jump by many orders of magnitude. The model is
  *stiff*, because it combines a very fast reaction (D) with a very slow one
  (M), and an extreme value of k₁ makes it so stiff that the solve fails.
  Bounds a factor of 100 either side of a reasonable guess rarely constrain the
  answer but make the fit much more reliable.
- **`n_starts=4`** runs the fit from the initial guess and from three randomly
  perturbed copies of it, and keeps the best. This guards against stopping in a
  local minimum. A start whose solve fails is skipped.
- **`include_ic=True`** makes the uncertainty cover the initial pools as well
  as the rate constants. The rate uncertainties then allow for the trade-off
  between each rate and its pool. Without it, the pools would be treated as
  known and the rate uncertainties would be too small.

### Checking with a profile likelihood

The Laplace approximation assumes the objective is quadratic near its minimum.
A [profile likelihood](analysis_concepts.md#profile-likelihood) checks this
without that assumption, by refitting the other unknowns at each of a series of
fixed values of k₂. Its 95 % interval is where the profile rises less than
1.92 above its minimum.

```{literalinclude} ../examples/ozone_demand_design.py
:language: python
:start-after: "# [step4-profile]"
:end-before: "# [end step4-profile]"
:dedent: 4
```

```{literalinclude} images/ozone_demand_output.txt
:language: text
:start-after: "Profile likelihood:"
:end-before: "== Step 5 =="
```

```{figure} images/ozone_demand_profile.png
:alt: Profile likelihood of k2 with the Laplace interval and the true value.
:width: 65%

Profile likelihood of k₂. The dashed line is the 95 % threshold, the shaded band
is the Laplace 95 % interval and the vertical line is the true value.
```

The profile is close to a parabola, and the two intervals agree and contain the
true value, so the Laplace uncertainty can be trusted here. The minimum is not
at the true value because the fit sees one noisy data set. Its estimate,
2.033×10⁻⁶, is 1.5 standard deviations from the truth, an ordinary miss. A
different draw of the noise would move the minimum but leave the width of the
profile almost unchanged.

## Step 5: design the next run

The design in step 3 was optimized for the true parameter values, which are
never known in practice. A design depends on the values it is computed for. The
best sampling times for a fast reaction differ from those for a slow one. A
practical sequence is to run a first experiment, fit it, and design the next
run for the values the data now support.

Since those values are still uncertain, the second design should work well
across their plausible range, not just at the best estimate: a
[robust design](analysis_concepts.md#scoring-a-design). Passing `param_draws`,
parameter vectors drawn from the calibration's posterior, makes
`optimize_experiment` maximize the criterion averaged over them. The script
uses ten draws and the default D-criterion.

```{literalinclude} ../examples/ozone_demand_design.py
:language: python
:start-after: "# [step5]"
:end-before: "# [end step5]"
:dedent: 4
```

```{literalinclude} images/ozone_demand_output.txt
:language: text
:start-after: "== Step 5 =="
:end-before: "== End =="
```

The robust design is almost the same as the one in step 3, as expected when
the parameters are already well determined. When the first-round estimates are
uncertain, averaging over the posterior keeps the second round from being
designed around the wrong values.

## Summary

- Compute the Fisher information of a planned design before running it.
  `identifiability` names the parameter combinations it cannot determine. Equal
  and opposite loadings in log coordinates point to a product.
- More samples of the same quantity rarely fix an identifiability problem.
  Here 30 minutes of ozone samples still leave k₂ and [M]₀ uncertain by about
  1000 %, and k₁ undetermined. A different measurement, or samples in a
  different time window, does.
- Use `rank_experiments` to compare discrete options and `optimize_experiment`
  for sampling times and doses. D-optimal designs cluster samples, so add a few
  in between to check the model.
- After calibrating, compare the estimated uncertainties with the design-stage
  predictions, and confirm key parameters with a profile likelihood.
- Design the next experiment from the fitted values, averaged over their
  uncertainty.
