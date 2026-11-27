---
paths:
  - "aquakin/integrate/finite_volume.py"
  - "aquakin/integrate/dispersed_flow.py"
---

# Rules — 1-D transport on the finite-volume core

## One finite-volume core

`integrate/finite_volume.py` is the only 1-D finite-volume implementation:
the grid (`Grid1D`), the face fluxes (diffusive, first-order upwind, limited
second-order upwind with its Green–Gauss gradient and van Albada slope) and
the divergence. A new 1-D transport operator composes these and builds only
its own boundary face values; do not write another flux, gradient or limiter
elsewhere, because two copies of a scheme drift. The biofilm operators
(`biofilm.py`, `biofilm_growth.py`) are built on it and must stay
bit-identical when it changes: keep the arithmetic order of `upwind_flux`,
`diffusive_flux` and `divergence`.

## The reaction term is `model.dCdt`

`DispersedFlowReactor` and `TanksInSeries` add transport to `model.dCdt(...)`
per cell; never `stoich.T @ rates`, which drops the negative-state clip and
the positivity limiter. A steady state differentiated through the implicit
function theorem must let `dCdt` evaluate the stoichiometry from the traced
parameters (no hoisted `stoich`), or a parameter-dependent coefficient loses
its derivative.

## The scheme

The advective face value is reconstructed from the upwind cell with the
Green–Gauss gradient and the van Albada slope with its constant
(`a = 2 g_C - b`, `b` the gradient toward the downwind node, `ε = atol / Δx`
per species), so the scheme is second order on smooth, monotone profiles and
does not oscillate at fronts; dispersion is the central difference. A slope
limiter used with the implicit solver must be differentiable at a = b = 0; one
with no constant is not, and Newton stalls on flat regions. `steady_state` and
`solve` share the fluxes, so they use the same `ε`. The inlet is Danckwerts (total flux `u C_in`) and the outlet zero
gradient. `TanksInSeries` is the same core with first-order upwind and no
dispersion, which is exactly `N` tanks in series: do not give it its own
right-hand side. A change to the scheme keeps the second-order convergence
test and the closed-form tests in `tests/integration/test_dispersed_flow.py`
passing unchanged.
