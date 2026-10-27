# Worked example: designing an experiment

Before fitting a kinetic model to data, it is worth asking two questions:

1. **Can the data determine the parameters at all?** A model can fit a data set
   well while some of its parameters are almost arbitrary, because changing
   one can be compensated by changing another.
2. **Which experiment would determine them best?** Sampling times, doses, and
   the choice of what to measure all change how much a run can tell us.

This page answers both for a small ozone-demand model, step by step, from a
first design that cannot estimate anything to a calibrated model whose
uncertainties are checked against the truth.

All the code on this page comes from the script
[`examples/ozone_demand_design.py`](https://github.com/DeGrootResearchGroup/aquakin/blob/main/examples/ozone_demand_design.py),
and every output block is what that script prints. Each code block continues
from the previous one, so the page can be followed as one program. The script
runs in about a minute.

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
rate constants and the two initial pools, [D]₀ = 15 µmol/L and
[M]₀ = 1000 µmol/L.

Two quantities can be measured:

- **Dissolved ozone**, with a standard deviation of 1 µmol/L.
- **UV254**, the UV absorbance at 254 nm per centimetre of path length.
  Absorbance is proportional to concentration, so the example takes
  A₂₅₄ = ε[M] with a known specific absorbance ε = 2×10⁻⁴ cm⁻¹ per µmol/L,
  and a standard deviation of 0.001 cm⁻¹.

The data on this page are synthetic, generated from the true values above, so
every result can be checked against the truth.

The script starts with its imports:

```{literalinclude} ../examples/ozone_demand_design.py
:language: python
:start-after: "# [imports]"
:end-before: "# [end imports]"
```

UV254 is not a single species, so it is passed to aquakin as an *observation
function*: a function that maps the concentration vector at one time, and the
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
the initial pools, are analysed on a log scale; why that matters is explained
below.

## Background: how much can an experiment tell us?

### Sensitivities

Write θ for the vector of unknowns and y for the vector of all predicted
measurements in a design (every sample of every measured quantity). The
**sensitivity matrix** S has one row per measurement and one column per
unknown:

$$
S_{ij} = \frac{\partial y_i}{\partial \theta_j}.
$$

A large entry means that measurement *i* changes a lot when unknown *j*
changes, so the measurement carries information about it. aquakin computes S
exactly by automatic differentiation through the ODE solve.

The unknowns are analysed as θ = log p, the logarithms of the physical values
p. A sensitivity to log p is the change in y for a given *relative* change in
p, which lets a rate constant of 10⁻⁶ and a pool of 1000 be compared directly.
A standard deviation in log p is approximately a relative standard deviation
of p, as long as it is small (below about 0.1–0.2): 0.05 in log k₂ means about
±5 % in k₂. For larger values the uncertainty in p becomes lopsided: 0.5 in
log p means +65 % / −39 %.

### The Fisher information

Each measurement has noise with standard deviation σᵢ, collected in the
diagonal covariance matrix Σ = diag(σᵢ²). The **Fisher information** of the
design is

$$
F = S^\top \Sigma^{-1} S .
$$

It weights each sensitivity by how precise its measurement is: a precise
measurement (small σᵢ) contributes more.

### The Cramér-Rao bound

The Fisher information turns the precision of the measurements into the best
precision the parameters can have. For a given design (what is measured, when,
and with what noise σᵢ), no unbiased estimate of the unknowns from that data
can have a smaller covariance than the inverse of F:

$$
\operatorname{Cov}(\hat\theta) \succeq F^{-1}.
$$

This is the **Cramér-Rao bound**. The symbol ⪰ means that the difference
Cov − F⁻¹ is positive semidefinite. In practice:

- The standard deviation of unknown *j* is at least $\sqrt{(F^{-1})_{jj}}$,
  the square root of the matching diagonal entry of F⁻¹.
- The same holds for any combination of the unknowns. This is where the
  1/√λₖ standard deviations of the eigen-directions below come from.

The bound is a floor. No fitting method can beat it; only a more informative
experiment can, through better measurements, better timing, or less noise. A
good fit, such as maximum likelihood (the `nll` loss used in step 4), comes
close to it when the noise is small and the model is nearly linear over the
range of uncertainty. With sparse or very noisy data, the real uncertainties
can be larger than the bound.

Because F needs only the model, assumed parameter values, and the design, the
bound can be computed before any data are collected;
`aquakin.fisher_information` computes F. The calculation can also be turned
around to ask how precise the measurements must be. The noise enters F as
1/σᵢ², so if every σᵢ is multiplied by a factor c, every bound on a parameter's
standard deviation is multiplied by c too. If a design gives k₂ to ±5 % but
±2 % is needed, the measurements must be about 2.5 times more precise, or the
design about 2.5² ≈ 6 times more informative.

The bound holds only if the model and the noise levels are right, and it is
computed at the assumed parameter values. A wrong model, an underestimated σ,
or parameter values far from the truth all make the real uncertainty larger
than the bound suggests.

### Combining measurements and experiments

Because the measurement errors are independent (Σ is diagonal), each
measurement contributes its own term to F, and the terms add. A second
experiment is just more measurements, so the information of a combined design
is the sum of the information of its experiments:

$$
F_{\text{1 and 2}} = F_{1} + F_{2} .
$$

This is what makes design possible: the value of a new experiment can be
judged by what it adds to the information already in hand. Two consequences
are used below:

- Repeating a measurement n times multiplies its contribution by n, so the
  standard deviations it controls shrink by √n.
- A measurement adds information only along the direction of its
  sensitivities (its row of S). Measurements whose sensitivities all point the
  same way add up along that one direction and leave the others unchanged,
  however many there are.

The sum assumes the errors of different measurements are independent; errors
that are correlated, for example through a shared calibration, add less. When
each experiment has its own initial pools, those pools are separate unknowns
that appear only in their own experiment's terms, while the shared rate
constants collect information from every experiment.

### Eigen-directions and identifiability

F is a symmetric matrix, so it can be written in terms of its eigenvalues λₖ
and unit eigenvectors vₖ:

$$
F = \sum_k \lambda_k\, v_k v_k^\top .
$$

Each eigenvector is a direction in parameter space: a combination of the
unknowns. The data determine the combination vₖᵀθ with a standard deviation of
1/√λₖ. A large eigenvalue means a well-determined combination; a small one
means a combination the data hardly constrain; a zero eigenvalue means a
combination the design cannot determine at all. The entries of an eigenvector,
its **loadings**, name the unknowns involved.

A useful pattern to recognise: in log coordinates, an eigenvector with equal
and opposite loadings on two unknowns, (+0.71, −0.71), is the direction
log a − log b. If its eigenvalue is near zero, the data determine only the
*product* a·b: raising a and lowering b by the same factor leaves every
prediction unchanged.

`aquakin.identifiability` reports the eigen-directions, flags those with an
eigenvalue below a fraction `eig_keep` (default 1 %) of the largest, and gives
the Cramér-Rao standard deviations. A parameter involved in a zero-eigenvalue
direction has an infinite standard deviation: it is **not identifiable** from
that design.

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

Two of the four eigenvalues are effectively zero, so two combinations of the
unknowns cannot be determined, and every standard deviation is infinite. The
loadings say which:

- **k₁ on its own.** D is used up within about 20 s, before the first sample.
  The samples show how much ozone D removed, which is [D]₀, but not how fast it
  did so.
- **log k₂ − log [M]₀.** The loadings are equal and opposite, so only the
  product k₂[M]₀ is determined. M is in such excess that it is barely consumed
  (about 3 % over 30 minutes), so ozone decays at the apparent first-order rate
  k₂[M]₀. Doubling k₂ and halving [M]₀ gives the same ozone curve.

Fitting this data set would return *some* values of k₁, k₂, and [M]₀, and the
fit could look good, but those values would mean nothing. The Fisher
information shows this before any experiment is run.

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

- **More late samples do not help.** Every ozone sample after D is gone is
  sensitive only to the product k₂[M]₀, so their sensitivities all point the
  same way and add nothing along the undetermined direction, however many
  there are.
- **Early samples** (2–20 s) catch D being consumed, which determines k₁ and
  [D]₀. k₂ and [M]₀ become identifiable only in principle, through the small
  consumption of M, and with huge uncertainties.
- **UV254** measures M directly, so it determines k₂ but leaves k₁ unknown.
- **Both together** make all four unknowns identifiable.

The general lesson is that more data of the same kind rarely fixes a
structural problem. A different measurement, or samples in a different time
window, does.

## Step 3: optimise the sampling times and dose

### Design criteria

Once every unknown is identifiable, the design can be improved further by
moving the samples and changing the dose. To compare designs we need a single
number that summarises F. The common choices are:

| Criterion | Value | Better | Meaning |
|---|---|---|---|
| D | log det F = Σ log λₖ | larger | The volume of the joint confidence region scales as (det F)^(−1/2). |
| A | trace F⁻¹ = Σ 1/λₖ | smaller | The sum of the variance bounds. |
| E | min λₖ | larger | The information along the worst-determined direction. |
| modified E | max λₖ / min λₖ | smaller | How unevenly the directions are determined. |

D-optimality is the usual default: it shrinks the uncertainty of all unknowns
together, and it does not depend on the units of the parameters.

`aquakin.optimize_experiment` maximises the criterion over the design. It
differentiates the criterion with respect to the sampling times and the
initial concentrations, and moves them with a gradient-based optimiser, within
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

The designed run roughly halves the uncertainty of k₁ and [D]₀ and reduces
that of k₂ by a factor of five. Two features of the result are typical of
D-optimal designs:

- **Samples cluster.** Rather than spreading out, the samples sit where the
  measurements are most sensitive to the unknowns: the early cluster resolves
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

With Gaussian noise of known standard deviation, the natural fitting objective
is the negative log-likelihood (`loss="nll"`): half the sum of the squared,
noise-scaled residuals, plus a constant. Its minimum is the best fit.

Near the minimum the objective is close to a quadratic bowl, and its curvature
(the Hessian) measures how sharply the fit gets worse when the unknowns move.
The **Laplace approximation** treats the uncertainty of the estimates as a
Gaussian whose covariance is the inverse of that curvature. With
`method="gauss_newton"`, the curvature is computed as SᵀΣ⁻¹S at the fitted
values, which is exactly the Fisher information. So the uncertainty reported
after the fit can be compared directly with the prediction made before the
experiment.

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

- **`param_halfwidth=np.log(100.0)`** keeps each rate constant within a
  factor of 100 of its starting guess (the bound is on the log scale). Without
  it, an early optimiser step can jump by many orders of magnitude, into a
  region where the stiff solve fails. Bounds a factor of 100 either side of a
  reasonable guess rarely constrain the answer but make the fit much more
  reliable.
- **`include_ic=True`** makes the uncertainty cover the initial pools as well
  as the rate constants. The rate uncertainties then allow for the trade-off
  between each rate and its pool. Without it, the pools would be treated as
  known and the rate uncertainties would be too small.
- **`eig_keep=1e-10`.** By default the Laplace step drops directions with less
  than 1 % of the largest eigenvalue, treating them as not identifiable. Here
  every unknown is identifiable, but the information spans about four orders of
  magnitude: [M]₀ is fixed precisely by UV254, while k₁ rests on three early
  samples. With the default, the k₁ direction would be dropped and its standard
  deviation reported as far too small. Check the spread of the eigenvalues with
  `aquakin.identifiability` before relying on a reported uncertainty.

### Checking with a profile likelihood

The Laplace approximation assumes the objective is quadratic near its minimum.
A **profile likelihood** checks this without that assumption. It fixes one
unknown at a series of values and, at each, refits all the others. The result
is the best objective attainable at each fixed value. The 95 % confidence
interval is the range where the profile rises less than 1.92 above its minimum
(half the 95 % point of a χ² distribution with one degree of freedom, because
the objective is half a sum of squares).

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
true value, so the Laplace uncertainty can be trusted here. A profile that is
flat on one side, or much wider than the Laplace interval, would show that the
local approximation is not.

## Step 5: design the next run

The design in step 3 was optimised for the true parameter values, which are
never known in practice. A design depends on the values it is computed for: the
best sampling times for a fast reaction differ from those for a slow one. A
practical sequence is to run a first experiment, fit it, and design the next
run for the values the data now support.

Since those values are still uncertain, the second design should work well
across their plausible range, not just at the best estimate. Passing
`param_draws`, a set of parameter vectors drawn from the calibration's
posterior, makes `optimize_experiment` maximise the criterion averaged over
the draws. This is called robust, or pseudo-Bayesian, design.

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
  `identifiability` names the parameter combinations it cannot determine;
  equal and opposite loadings in log coordinates point to a product.
- More samples of the same quantity rarely fix a structural problem: here no
  number of ozone samples separates k₂ from [M]₀. A different measurement, or
  samples in a different time window, does.
- Use `rank_experiments` to compare discrete options and `optimize_experiment`
  for sampling times and doses. D-optimal designs cluster samples, so add a few
  in between to check the model.
- After calibrating, compare the estimated uncertainties with the design-stage
  predictions, check the eigenvalue spread before trusting a Laplace
  uncertainty, and confirm key parameters with a profile likelihood.
- Design the next experiment from the fitted values, averaged over their
  uncertainty.
