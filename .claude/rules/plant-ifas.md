---
paths:
  - "aquakin/plant/ifas.py"
  - "aquakin/integrate/biofilm.py"
  - "aquakin/integrate/biofilm_growth.py"
---

# Rules — the IFAS / MBBR biofilm unit

Split out of `.claude/rules/plant.md`.

**IFAS / MBBR biofilm (`plant/ifas.py`).** Three things about this unit are
load-bearing and easy to get wrong.

- **Layer count.** The depth discretization is **first order**, and its error is
  set by the penetration ratio `L_f/delta`, `delta = sqrt(D_eff*K_s/(k*X_f))` —
  not by thickness alone. At `L_f/delta = 4`, four layers put the substrate flux
  **26%** high; sixteen (the default since the BM1 validation) give ~6% there and
  ~2% at `L_f/delta = 2`. `IFASUnit.layer_resolution(state)` reads the achieved
  resolution off a solved state (largest fractional soluble drop across one
  layer); a few per cent is fine, near 1 means the substrate dies inside one
  cell. The error table is in the class docstring and pinned by
  `tests/validation/test_biofilm_analytical.py`.
- **Steady state.** `run_to_steady_state`'s default `criterion="norm"` is
  **unreliable for this unit**: the held-fixed attached biomass adds
  `n_layers * density` to `||y||` and nothing to `||dy/dt||`, inflating the
  tolerance every other state is held to. On BM1 it stopped with the bulk moving
  at -22 g/(m3 d) and reported convergence, 6% wrong in flux. It now warns when
  the result's componentwise scaled residual exceeds
  `NORM_RESIDUAL_WARN_FACTOR * ss_rtol` (100x; BSM1 sits at ~4x with either
  clarifier). Use `criterion="scaled_residual"` for an IFAS plant.
- **Growing the film (`dynamic_thickness=True`).** Thickness becomes the final
  state entry (`L_f`) instead of a fixed parameter, implementing Wanner & Gujer
  (1986) via `integrate/biofilm_growth.py`: incompressibility fixes the
  displacement velocity `u(z)` from the production integrated below `z`, species
  advect on a grid that stretches with the film, and `dL_f/dt = u(L_f) - u_det`.
  Two things the arbitrary Lagrangian–Eulerian form makes exact and must stay
  that way: the wall face is impermeable and the surface face carries exactly the
  detachment velocity. **The grid-dilation term `-X*(dL_f/dt)/L_f` is not
  optional** — the fluxes govern the mass in a cell, so recovering a
  concentration from them must account for the cell's own changing volume;
  without it a uniform film drifts off its own density (measured 2%) and the
  incompressibility the velocity was derived from is violated. A growing film
  **freezes nothing** (`biofilm_fixed_mask` is refused), since its inventory is
  what the balance solves for. The thickness row/column of the coupling pattern
  is declared **dense**, because those couplings run through saturated Monod
  terms an AD probe cannot see. Validated against `L_f = mu_net/lambda` for a
  fully penetrated film and against BM1 cases 2 and 5, where the benchmark lets
  thickness be an output of the growth/lysis balance (`detachment_rate=0`, so the
  film settles where net production integrates to zero).
  **The thickness is either specified or predicted, chosen by `detachment_law`.**
  `"balance"` pins it: detachment is set to the velocity arriving at the surface
  (`balanced_detachment_velocity`), so `dL_f/dt` is **exactly** zero — not zero to
  a tolerance, which matters because a specified thickness must not drift over a
  long solve — while the species still advect, stratify and shed at the surface.
  It takes no `detachment_rate` and refuses one, since the velocity is not a free
  parameter under that law. The power laws (`"quadratic"`, `"linear"`,
  `"constant"`, via `DETACHMENT_POWERS`) instead let the thickness follow from
  the physics. A benchmark that fixes the film at a stated thickness (BM3, and
  BM1 cases 1, 3, 4) wants `"balance"`; one that lets lysis thin it (BM1 cases 2
  and 5) wants a rate-driven law with `detachment_rate=0`.
