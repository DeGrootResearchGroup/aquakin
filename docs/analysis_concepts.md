# Key ideas: information, uncertainty and identifiability

The analysis pages use a handful of statistical ideas: the Fisher information
of an experiment, the bound it puts on the uncertainty of the parameters, the
posterior of a fit and its Laplace approximation, the profile likelihood, and
identifiability. This page explains each one as far as you need it to use the
tools and read their outputs. The other pages link here at first use, and
[Worked example: designing an experiment](worked_example_design.md) shows them
at work on one study.

## Sensitivities

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

A positive unknown, such as a rate constant declared with
`transform: positive_log` or an initial pool, is analyzed as θ = log p, the
natural logarithm of its physical value p. A sensitivity to log p is the
change in y for a given *relative* change in p, which lets a rate constant of
10⁻⁶ and a pool of 1000 be compared directly. A standard deviation in log p is
approximately a relative standard deviation of p, as long as it is small
(below about 0.1–0.2). For example, 0.05 in log k means about ±5 % in k. For
larger values the uncertainty in p becomes lopsided: a standard deviation of
0.5 in log p means +65 % / −39 %.

## The Fisher information

Each measurement has noise with standard deviation σᵢ, collected in the
diagonal covariance matrix Σ = diag(σᵢ²). The **Fisher information** of the
design is

$$
F = S^\top \Sigma^{-1} S .
$$

It weights each sensitivity by how precise its measurement is. A precise
measurement (small σᵢ) contributes more. F needs only the model, assumed
parameter values and the design, so it can be computed before any data are
collected.

### The Cramér-Rao bound

The Fisher information turns the precision of the measurements into the best
precision the parameters can have. For a given design (what is measured, when,
and with what noise σᵢ), no unbiased estimate of the unknowns from that data
can have a smaller covariance than the inverse of F. (An estimate is unbiased
if, averaged over many repeats of the experiment, it would equal the true
value.)

$$
\operatorname{Cov}(\hat\theta) \succeq F^{-1}.
$$

This is the **Cramér-Rao bound**. The symbol ⪰ compares two matrices. It means
the left side is at least as large as the right side for every combination of
the unknowns, in the following sense:

- The standard deviation of unknown *j* is at least $\sqrt{(F^{-1})_{jj}}$,
  the square root of the matching diagonal entry of F⁻¹.
- The same holds for any combination of the unknowns. This is where the
  1/√λₖ standard deviations of the
  [eigen-directions](#eigen-directions-and-identifiability) come from.

The bound is a floor. No fitting method can beat it. Only a more informative
experiment can, through better measurements, better timing, or less noise. A
maximum-likelihood fit (`loss="nll"`) comes close to it when the noise is
small and the model is nearly linear over the range of uncertainty. With
sparse or very noisy data, the real uncertainties can be larger than the
bound.

The calculation can also be turned around to ask how precise the measurements
must be. The noise enters F as 1/σᵢ², so if every σᵢ is multiplied by a factor
c, every bound on a parameter's standard deviation is multiplied by c too. If a
design gives a rate constant to ±5 % but ±2 % is needed, the measurements must
be about 2.5 times more precise, or the design about 2.5² ≈ 6 times more
informative.

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

This is what makes design possible: the value of a new experiment can be judged
by what it adds to the information already in hand. Two consequences follow:

- Repeating a measurement n times multiplies its contribution by n, so the
  standard deviations it controls shrink by √n.
- A measurement adds information only along the direction of its
  sensitivities (its row of S). Measurements whose sensitivities all point the
  same way add up along that one direction and leave the others unchanged,
  however many there are. More data of the same kind therefore rarely fixes an
  identifiability problem; a different measurement, or samples in a different
  time window, does.

The sum assumes the errors of different measurements are independent. Errors
that are correlated, for example through a shared calibration, add less. When
each experiment has its own initial pools, those pools are separate unknowns
that appear only in their own experiment's terms, while the shared rate
constants collect information from every experiment.

## Eigen-directions and identifiability

F is a symmetric matrix, so it can be written in terms of its eigenvalues λₖ
and unit eigenvectors vₖ:

$$
F = \sum_k \lambda_k\, v_k v_k^\top .
$$

Each eigenvector is a direction in parameter space, a combination of the
unknowns. The data determine the combination vₖᵀθ with a standard deviation of
1/√λₖ. A large eigenvalue means a well-determined combination. A small one
means a combination the data hardly constrain. A zero eigenvalue means a
combination the design cannot determine at all. The entries of an eigenvector,
its **loadings**, name the unknowns involved. A parameter is **identifiable**
from a design when every combination it takes part in is determined.

### Reading a direction: products and ratios

Suppose two unknowns a and b, analyzed as log a and log b, and an eigenvector
with loadings (+0.71, −0.71) on them. The value 0.71 is 1/√2, so the vector has
unit length and gives equal weight to the two unknowns.

Moving the parameters a distance t along this eigenvector changes them by

$$
\log a \to \log a + \frac{t}{\sqrt 2}, \qquad
\log b \to \log b - \frac{t}{\sqrt 2} .
$$

So a is multiplied by a factor $e^{t/\sqrt 2}$ and b is divided by the same
factor. The ratio a/b changes, but the product a·b stays exactly the same,
because log(a·b) = log a + log b does not change.

A zero eigenvalue means the predictions do not change along that direction, so
every point on it fits the measurements equally well. If moving along the
direction changes a/b but not a·b, the data cannot tell different values of
a/b apart. They determine only the product a·b. The perpendicular direction,
with loadings (+0.71, +0.71), is the one that changes the product, and its
eigenvalue measures how well the product is known. A rate constant k and an
initial pool [X]₀ that load this way are seen only as the product k[X]₀: lump
them into one parameter, or separate them with a new experiment.

The same reasoning covers the other cases:

- Loadings of the **same sign**, (+0.71, +0.71), on a zero-eigenvalue direction
  leave a/b unchanged, so only the ratio a/b is determined.
- **Unequal** loadings point to a power-law combination. A zero-eigenvalue
  direction with loadings $(v_a, v_b)$ leaves $a^{v_b}\, b^{-v_a}$ unchanged,
  so that combination is what the data determine.
- A direction loaded on **one unknown** alone, (1.00, 0), means that unknown
  has no effect on the predictions at all.

The same reading applies when the eigenvalue is small rather than zero. The
combination that changes along the direction is then poorly determined instead
of not determined at all, and the perpendicular combination is still the
well-determined one.

### Structural and practical identifiability

Two different situations give a small eigenvalue:

- **Structurally non-identifiable.** The eigenvalue is exactly zero (to
  round-off). The predictions do not depend on that combination at all, so no
  amount of the same kind of data can determine it. A parameter involved in
  such a direction has an infinite standard deviation.
- **Practically non-identifiable.** The eigenvalue is small but not zero. The
  combination does affect the predictions, but so weakly that its standard
  deviation is enormous. More or better data of the same kind would determine
  it eventually, but often not within a realistic experiment.

`aquakin.identifiability` flags every direction whose eigenvalue is below a
fraction `eig_keep` (default 1 %) of the largest, which covers both
situations. Its standard deviations show which one applies: an infinite
standard deviation means a structural problem, and a very large finite one a
practical problem.

## Scoring a design

To compare designs, or to optimize one, a single number summarizes F. The
common choices are:

| Criterion | Value | Better | Meaning |
|---|---|---|---|
| D | log det F = Σ log λₖ | larger | The volume of the joint confidence region scales as (det F)^(−1/2). |
| A | trace F⁻¹ = Σ 1/λₖ | smaller | The sum of the variance bounds. |
| E | min λₖ | larger | The information along the worst-determined direction. |
| modified E | max λₖ / min λₖ | smaller | How unevenly the directions are determined. |

Choosing the design that maximizes D is called **D-optimal design**. It is the
usual default, because it shrinks the uncertainty of all unknowns together, and
it does not depend on the units of the parameters. A design that leaves some
combination structurally undetermined has a zero eigenvalue, so its D-criterion
is −∞, the worst possible score; compare such designs by their standard
deviations instead. D computed for only some of the unknowns, with the others
treated as unknown nuisance quantities, is called Ds-optimality.

D-optimal designs tend to cluster their samples where the measurements are
most sensitive. Because the design trusts the model completely, add a few
samples between the clusters, or a validation run, to check the model there.

When a design spreads a budget over a list of candidate measurements, the
Kiefer-Wolfowitz equivalence theorem gives a check that the D-optimal
allocation has been found. The **standardized variance** of a candidate
measures how poorly the design predicts what that candidate would measure. At
the D-optimal allocation the largest standardized variance over the candidates
equals the number of scored unknowns; for any other allocation it is larger.

A design depends on the parameter values it is computed for, which are never
known exactly. A **robust** (pseudo-Bayesian) design maximizes the criterion
averaged over parameter draws from a prior or a previous fit's posterior, so
it works across their plausible range rather than only at one estimate.

## Fitting: likelihood, posterior and the Laplace approximation

With Gaussian measurement noise of known standard deviation, the natural
fitting objective is the **negative log-likelihood** (`loss="nll"`): half the
sum of the squared, noise-scaled residuals, plus a constant. Its minimum is the
maximum-likelihood estimate.

A **prior** is a Gaussian belief about a parameter from earlier work, added to
the objective as a penalty. The distribution of the unknowns given both the
data and the prior is the **posterior**, and the fit that minimizes the
combined objective is its peak, the maximum a posteriori (**MAP**) estimate.
With no prior, the MAP is the maximum-likelihood estimate.

Near the minimum the objective is close to a quadratic bowl. Its curvature, the
matrix of second derivatives (the Hessian), measures how sharply the fit gets
worse when the unknowns move. The **Laplace approximation** treats the
posterior as a Gaussian whose covariance is the inverse of that curvature. The
**Gauss–Newton** form of the curvature uses first derivatives only, SᵀΣ⁻¹S at
the fitted values, dropping the terms that involve the residuals. With the
`"nll"` loss and no prior, that is exactly the Fisher information, so the
uncertainty reported after a fit can be compared directly with the prediction
made before the experiment.

The Laplace approximation is good when the data determine the parameters
precisely enough that the model is close to linear over their uncertainty. It
misdescribes a posterior that is curved (a ridge where only a product or a sum
of parameters is seen), skewed, or cut by a bound. Drawing from the posterior
itself by Markov chain Monte Carlo (MCMC) avoids that approximation, at the
cost of thousands of solves.

When the noise level is not known, it can be estimated with the parameters.
The **concentrated likelihood** replaces each output's noise standard deviation
by its best estimate at the current parameters, the root-mean-square residual
of that output, which leaves a likelihood of the parameters alone.

## Profile likelihood

A **profile likelihood** checks one unknown without assuming the objective is
quadratic. It fixes the unknown at a series of values and, at each, refits all
the others. The result is the best objective attainable at each fixed value.
The 95 % confidence interval is the range where the profile rises less than
1.92 above its minimum.

The 1.92 comes from the likelihood-ratio test. When one unknown is fixed at its
true value and the others are refitted, twice the rise in the negative
log-likelihood approximately follows a χ² distribution with one degree of
freedom. The 95 % point of that distribution is 3.84, and the objective is the
negative log-likelihood itself, so the threshold on the profile is half of
3.84, 1.92. The χ² result is asymptotic (it holds as the data grow), so the
interval is an approximation too, but one that follows a skewed or one-sided
uncertainty where the Laplace interval cannot.

A profile close to a parabola, with an interval close to the Laplace one, says
the Laplace uncertainty can be trusted. A profile that is flat on one side, or
much wider than the Laplace interval, shows that the local approximation is
not. A profile that stays below 1.92 out to the edge of every reasonable value
shows a parameter the data do not bound on that side.

## Which interval to report

| Interval | Cost | Use it when |
|---|---|---|
| Laplace (estimate ± 1.96 sd, in the fit's coordinates) | free with the fit | the parameters are well determined and the profile, where checked, is close to a parabola |
| Profile likelihood | one refit per grid point, per parameter | a key parameter, one near a bound, or one whose Laplace interval is wide or reaches zero |
| MCMC (percentiles of the draws) | thousands of solves | the posterior is curved or skewed, or you need intervals of predictions or of functions of several parameters |

State which one you report, with the noise model (known σ or estimated) and
any priors, since all three depend on them.
