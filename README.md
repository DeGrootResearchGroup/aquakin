# aquakin

[![CI](https://github.com/DeGrootResearchGroup/aquakin/actions/workflows/ci.yml/badge.svg)](https://github.com/DeGrootResearchGroup/aquakin/actions/workflows/ci.yml)

`aquakin` is a Python library for modelling reactive scalar transport in
aqueous environmental systems. Reaction networks are declared at runtime in
YAML and compiled to JAX-native, automatic-differentiable rate functions
integrated with [Diffrax](https://github.com/patrick-kidger/diffrax).

Shipped networks span chemistry (ozonation/bromate after Acero & von Gunten,
2001; UV/H₂O₂) and biology (the ASM activated-sludge family, including a
two-step nitrification/denitrification variant with explicit nitrite, a
two-pathway AOB nitrous-oxide (N₂O) model, an anammox / deammonification
variant, and a comammox complete-nitrifier variant; ADM1 anaerobic
digestion in its BSM2 form, with gas headspace; the WATS sewer-process models
`wats_sewer_extended` and the paper-faithful `wats_sewer_khalil_paper`, the
latter with structural variants for model-structure studies) as well as
chemistry coupled to acid-base speciation (a charge-balance state-derived pH,
and SI-driven mineral precipitation/dissolution — struvite + calcite, and
iron/aluminium chemical-phosphorus removal — after Kazadi Mbamba et al. 2015 and
Flores-Alsina et al. 2016. A very insoluble mineral's stiff kinetics defeat every
sensitivity method, so two opt-in differentiable variants are provided: an
**algebraic equilibrium** mode that solves `IAP = Ksp` directly
(`network.precipitation_equilibrium(...)`) and a **bounded-driver** kinetic form
for differentiable dynamics). The network
YAML files live under `aquakin/networks/`; see `CLAUDE.md` for the full list.
Future networks include UV/TiO₂ and chlorine decay.

## Features

- Reaction networks declared in YAML — no recompilation required.
- Full automatic differentiation through `solve()` via JAX.
- JAX-native stiff ODE integration via Diffrax (`Kvaerno5` by default).
- Safe rate expression evaluation via a custom AST (no `eval()`).
- Decoupled transport / reaction operator splitting at all scales (0D, 1D, 3D).
- Reactors for batch (0D), plug flow (1D), Lagrangian particle tracks, and a
  layered biofilm (`BiofilmReactor`: 1-D diffusion-reaction over biofilm depth,
  for penetration-controlled processes).
- Plant-wide flowsheets (`aquakin.plant`): the IWA benchmark plants BSM1 and
  BSM2 — reactors, clarifiers, mixers/splitters and an ADM1 digester integrated
  under one monolithic solve, with run-to-steady-state, a fast differentiable
  algebraic steady-state solver (`plant.steady_state`, pseudo-transient
  continuation — ~10× faster than integrating to settle), dynamic influents,
  EQI/OCI performance metrics, and GHG (N₂O / CO₂e) + monetised-cost reporting
  with standardized scenario-comparison KPI tables. Includes an `IFASUnit` / 
  `MBBRUnit` (a CSTR bulk coupled to a depth-resolved attached biofilm) for 
  modelling MBBR/IFAS intensification retrofits.
- Located events / discontinuities (`Event` + `solve(events=...)`): time events
  and state root-crossings with exact state resets / mode switches (on/off pumps,
  SBR phases, dosing on/off, level limits) — time-scheduled events keep
  `jax.grad` finite.
- Full automatic differentiation everywhere, including cap-free forward
  sensitivity and reverse-mode gradients through stiff plant solves (see
  [Advanced: differentiation & sensitivity](#advanced-differentiation--sensitivity)).

## Installation

```bash
pip install -e ".[test]"
```

> **⚠️ `import aquakin` enables JAX 64-bit (x64) mode process-wide.** The stiff
> implicit ODE solves need double precision, so at import aquakin runs
> `jax.config.update("jax_enable_x64", True)` — which is **global JAX state**.
> Any other JAX code in the same process will use float64 afterward (more memory,
> different numerics). This is required, not optional. If you are co-running JAX
> code that needs float32, run aquakin in a separate process. aquakin emits a
> one-time warning if it overrides an explicit float32 preference (JAX already
> imported, or `JAX_ENABLE_X64` set off), so the side effect is never silent.

## Quickstart

```python
import jax.numpy as jnp
import aquakin

network = aquakin.load_network("asm1")   # Activated Sludge Model No. 1 (IAWQ)

# 0-D (a single well-mixed tank): start from the network's declared condition
# defaults (ASM1 runs at a temperature T) and change only what differs.
conditions = network.default_conditions()                   # YAML defaults (T = 20 C)
# conditions = network.default_conditions().with_(T=283.15)   # ...or run it at 10 C
# (OperatingConditions(T=293.15) is the scalar 0-D shorthand; use SpatialConditions
#  for a spatially varying PFR/CFD case.)

reactor = aquakin.BatchReactor(network, conditions)

# Build the initial state by name -- no .at[species_index[...]].set() chains.
# A simple aerobic batch: activated-sludge biomass + substrate + ammonia.
# (A dict, since some ASM species names aren't valid kwargs; rest = YAML defaults.)
C0 = network.concentrations({
    "SS": 60.0, "SNH": 25.0, "XB_H": 500.0, "XB_A": 80.0, "SO": 2.0})

# For a FEED composition use base="zero" (or network.influent): unlisted species
# are absent, not silently left at their YAML reference value.
feed = network.concentrations({"SS": 60.0, "SNH": 25.0}, base="zero")
influent = network.influent({"SS": 60.0, "SNH": 25.0}, Q=18446.0)   # InfluentSeries

# Or characterize an influent from lab measurements (total COD, TKN, ammonia,
# alkalinity, ...): the SUMO-style fractionation splits them into the ASM1 states.
influent = aquakin.characterize_influent(network, flow=24000.0, total_cod=420.0,
                                         tkn=34.4, ammonia=24.0, alkalinity=330.0)
# A lab/SCADA CSV with arbitrary headers maps + fractionates per row, no renaming:
#   aquakin.read_influent_csv("plant_log.csv", network,
#       column_map={"t": "day", "Q": "flow_m3d", "total_cod": "COD",
#                   "tkn": "TKN", "ammonia": "NH4-N", "alkalinity": "Alk"})

# There is NO global time unit: t_span / t_eval are in whatever unit the
# network's rate constants use, and it differs by network -- ozone/UV are in
# SECONDS (M-1 s-1), the biological models (ASM/ADM/WATS) in DAYS (1/d). Check
# it before choosing a span:
network.time_unit                    # "d" for asm1 (the ozone/UV networks are in "s")

# ...or pass time_unit= to work in a unit of your choice: the input times are
# converted to the network's native unit for the solve and solution.t comes back
# in the unit you asked for (solution.time_unit reports it). Works the same on
# BatchReactor / BiofilmReactor / Plant.solve. e.g. an ASM run in hours:
#   sol = reactor.solve(C0, t_span=(0.0, 48.0), t_eval=..., time_unit="h")

# params is optional and defaults to network.default_parameters().
t_eval = jnp.linspace(0.0, 1.0, 121)        # one day, in days (asm1's native unit)
solution = reactor.solve(C0, t_span=(0.0, 1.0), t_eval=t_eval)

print("[SNH] at t=1 d:", float(solution.C_named("SNH")[-1]))   # final effluent ammonia

# Reporting last-point values without the per-species [-1] slice:
solution.final_named(["SS", "SNH", "SNO"])  # {name: float} at the final time (None = all)
solution.final                          # == final_named(): every species' last value
solution.C_named_many(["SNH", "SNO"])   # several full trajectories -> {name: array}

# Species units and descriptions are carried from the YAML to results, so you
# never have to re-derive units by string-matching names.
network.units_of("SNH")              # e.g. "g_N/m³"
network.description_of("SNH")
solution.units_named("SNH")          # same, for axis/column labels
network.summary()                    # tabulates every species with its units

# Dimensional ("unit") consistency check of the rate expressions. Currency-aware:
# g_COD/m3 and g_N/m3 are different dimensions, so it catches a dropped
# concentration factor, a wrong rate-constant exponent, or a Monod term mixing
# two currencies -- bugs a plain SI dimension check misses. Opt-in and advisory
# (never raises; unknown/unparseable units are skipped).
for w in network.check_units():      # -> list of (reaction, location, detail)
    print(w)

# Conservation (mass / electron balance) check. The companion to check_units:
# each species declares its content of the conserved quantities in the YAML
# (`composition: {COD: 1.0}` for an organic, `{COD: -1.0}` for oxygen, `{COD:
# -2.86, N: 1.0}` for nitrate-N, ...), and check_conservation dots that table
# against the stoichiometry -- so a wrong electron-acceptor demand breaks COD and
# a wrong product split breaks an elemental (S/N/P/Fe) balance. Opt-in and
# advisory (never raises). The ASM/ADM families fall back to a shipped table.
network.composition()                # -> {species: {quantity: content}}
for r, q, residual in network.check_conservation(quantities=["COD"]):
    print(r, q, residual)            # reactions whose COD content does not balance
# (For ASM1 this lists only `anoxic_growth_heterotrophs`: denitrification's
# electrons leave as N2 gas, which ASM1 does not track -- a known, intentional
# exception. The WATS sewer networks declare composition in their YAML and close
# COD/S/N/Fe exactly.)

# Better than checking after the fact: write a conservation-determined coefficient
# as `auto` and let it be SOLVED from the declared balances at load -- so it can
# never be typed wrong. With composition declared on each species:
#   reactions:
#     - name: growth
#       conserved_for: [COD]                 # or a network-level `conserved_for:`
#       rate: "mu * [SS] * [XBH]"
#       stoichiometry: {SS: -2.0, XBH: 1.0, SO: auto}   # O2 demand solved from COD

# Export results to a table instead of float()-casting one species at a time.
# Requires the optional `pandas` extra: pip install aquakin[dataframe]
df = solution.to_dataframe()         # time-indexed, one column per species
df.attrs["units"]                    # {species: unit} (units kept off the labels)
solution.to_csv("run.csv")           # units embedded in the CSV header

# Plot a species (or several) without matplotlib boilerplate -- the x-axis is
# labelled with the network's time unit, the y-axis with the species' units.
# Returns a matplotlib Axes. Requires the optional `plot` extra: aquakin[plot]
ax = solution.plot("SNH")            # one line; y-axis "SNH [g_N/m³]"
solution.plot(["SNH", "SNO"])        # several, legended; pass ax= to overlay
```

Discontinuous operations -- on/off pumps, SBR phases, dosing, level limits --
are handled with **located events** (`solve(events=...)`, on reactors and the
plant). A time event fires at a known time (AD-safe), a state event when a
`cond_fn` crosses zero; each can reset the state or terminate the solve:

```python
# An anoxic (denitrification) batch: dose external carbon partway through to
# drive denitrification, and stop once the nitrate has been removed.
i_ss = network.species_index["SS"]
i_sno = network.species_index["SNO"]
anoxic = network.concentrations(
    {"SS": 70.0, "SNO": 12.0, "SNH": 20.0, "XB_H": 150.0, "SO": 0.0})
events = [
    aquakin.Event(at_times=[0.1],                           # dose carbon at t = 0.1 d
                  apply=lambda t, C, p: C.at[i_ss].add(60.0)),
    aquakin.Event(cond_fn=lambda t, C, p: C[i_sno] - 0.5,   # stop when nitrate is gone
                  direction=-1, terminal=True, name="denitrified"),
]
sol = reactor.solve(anoxic, t_span=(0.0, 0.5),
                    t_eval=jnp.linspace(0.0, 0.5, 101), events=events)
sol.events_log                       # [(0.1, 'event0'), (~0.13, 'denitrified')] -- the audit trail
```

## Plant-wide simulation

Beyond a single reactor, `aquakin.plant` assembles full treatment-plant
flowsheets. The IWA benchmark plants ship ready to build: load BSM1, drive it to
steady state, and read the effluent — no autodiff or solver tuning required.

```python
import jax.numpy as jnp
import aquakin
from aquakin.plant.bsm import build_bsm1, load_bsm1_influent, evaluate_bsm1

network = aquakin.load_network("asm1")
plant = build_bsm1(network)            # 5 reactors + secondary clarifier + recycles

# A constant average-load influent. add_influent wires it to the plant's
# canonical front -- no "unit.port" string to hard-code.
plant.add_influent("feed", network.influent(
    {"SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
     "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0}, Q=18446.0))

# Integrate until the plant settles (a self-terminating steady-state event --
# no horizon to guess). Sensible solver defaults; nothing to tune.
ss = plant.run_to_steady_state()
print("converged:", ss.converged, "after", round(ss.time), "days")

# ...or snap straight to steady state algebraically (pseudo-transient
# continuation: ~10x faster, robust on stiff topologies, and differentiable --
# jax.grad of a loss on ss.state flows to the plant parameters for design sweeps).
ss = plant.steady_state()
print("converged:", ss.converged, "in", int(ss.iterations), "iterations")

# Reconstruct the clarified effluent and read its quality.
eff = plant.stream(ss.solution, plant.effluent_endpoint)
print("effluent SNH:", round(float(eff.C_named("SNH")[-1]), 2), "g N / m³")   # ~0.5
```

For a dynamic run, drive a fresh plant with a diurnal dry-weather influent,
warm-started from the steady state, and score the headline performance indices:

```python
dyn = build_bsm1(network)
dyn.add_influent("feed", load_bsm1_influent("dry", network))   # a 14-day diurnal load
sol = dyn.solve(t_span=(0.0, 14.0), t_eval=jnp.linspace(0.0, 14.0, 15), y0=ss.state)

ev = evaluate_bsm1(dyn, sol)           # Effluent Quality / Operational Cost indices
print(f"EQI = {ev.eqi:.0f} kg/d   OCI = {ev.oci:.0f}")
```

`build_bsm2(...)` assembles the full BSM2 sludge train (primary clarifier,
thickener, ADM1 digester with the ASM1↔ADM1 interfaces, dewatering, reject
recycle) the same way; see `examples/` and `CLAUDE.md` for the BSM2 steady state,
dynamic/seasonal runs, DO control, and the SRT/HRT/F:M design helpers.

Reactor temperature is a selectable `TemperatureModel`: the default
`AlgebraicTemperature` takes each reactor's temperature to be its instantaneous
flow-weighted inlet temperature, while `HeatBalanceTemperature` gives every
finite-volume unit a dynamic temperature *state* governed by the first-order heat
balance `V dT/dt = Q_in (T_in − T)` (the heated digester stays fixed), so the
reactor temperature lags and damps the influent. Select it with
`plant.set_temperature_model(aquakin.HeatBalanceTemperature())` or
`build_bsm2(temperature_model=aquakin.HeatBalanceTemperature())`.

Aeration energy can be scored from **blower/diffuser physics** instead of the
Copp-2002 correlation. An `AerationSystem` (diffuser submergence, SOTE, fouling,
blower efficiency) turns the `kLa` a solve produced into the required air flow
(via the standard oxygen transfer rate `kLa·C_s·V` and the diffuser SOTE) and the
adiabatic blower power against the submergence head — keeping the `kLa` kinetic
interface unchanged. Pass it to size a tank
(`aquakin.design_summary(kla, volume, system)`) or to the evaluators, where it
replaces the aeration-energy term and reports the air flow:

```python
syst = aquakin.AerationSystem(depth=5.0, sote=0.20)      # 5 m diffusers, 20 % SOTE
ev   = evaluate_bsm2(plant, sol, params, aeration_system=syst)
print(ev.aeration_energy, "kWh/d   air:", ev.air_flow, "m3/d")   # mechanistic AE
```

**Disinfection** unit ops reduce a pathogen indicator at the end of the train:
a `UVUnit` (dose = intensity × exposure × UVT-factor → log-linear inactivation)
and a `ChlorineContactUnit` (a chlorine residual that decays first-order; the CT
credit `residual × T10` → log-removal, with `T10` from a baffling factor or a
residence-time distribution). Both pass the process stream through and reduce the
indicator-organism density carried on the stream (`Stream.org`, the disinfection
analogue of the temperature scalar), so the reconstructed effluent reports it:

```python
p.add_unit(aquakin.ChlorineContactUnit("cl", net, volume=500.0, dose=5.0,
                                       ct_per_log=8.0, decay_rate=2.0,
                                       inlet_density=1e6))   # CFU/100 mL in
sol = p.solve(t_span=(0.0, 2.0), t_eval=t)
print(p.stream(sol, "cl.out").org[-1])                      # effluent indicator
```

### GHG, cost and scenario reporting

On top of the EQI / OCI evaluation, `aquakin` reports a **carbon footprint**
(CO₂e/d) and a monetised **operating cost** (currency/d), and tabulates
scenarios side by side:

```python
ev  = evaluate_bsm2(plant, sol, params)          # EQI / OCI + physical flows
n2o = aquakin.direct_n2o_emission(plant, sol)    # stripped N₂O (0 unless the AS
                                                 # network resolves an SN2O state)

fp = aquakin.carbon_footprint(                   # kg CO₂e/d, with breakdown
    ev.total_energy(), grid_factor=0.4, n2o_emission=n2o,
    methane_production=ev.methane_production, ch4_fugitive_fraction=0.015)

oc = aquakin.operating_cost(                     # currency/d OPEX (+ optional CAPEX)
    energy_kwh_per_d=ev.total_energy(), carbon_kg_cod_per_d=ev.carbon_mass,
    sludge_kg_tss_per_d=ev.sludge_production,
    methane_kg_per_d=ev.methane_production,
    factors=aquakin.CostFactors(energy_price=0.12), co2e_per_d=fp.total_co2e)

print(fp)            # labeled CO₂e breakdown
print(oc)            # labeled cost breakdown
print(aquakin.kpi_comparison({"baseline": ev, "low-DO": ev_b}).table())
```

`carbon_footprint` weights direct N₂O (GWP ~273), grid-energy CO₂e and fugitive
biogas methane (GWP ~27), crediting recovered biogas energy; `operating_cost`
prices energy / carbon / sludge / biogas (and an optional CAPEX + carbon charge);
`kpi_comparison` puts any report objects (`BSM2Evaluation`, `CarbonFootprint`,
`OperatingCost`) into one standardized KPI table. The scenario-orchestration
primitives `monte_carlo`, `compare_scenarios` and `optimize_design` propagate
input uncertainty and size designs to a permit at minimum cost. See
`examples/bsm2_ghg_cost_report.py`.

## Advanced: differentiation & sensitivity

Every solve — reactor or whole plant — is differentiable. The machinery below is
for parameter estimation and sensitivity analysis; a plain forward simulation
(above) never needs it.

> **Heads-up — silent non-finite reverse gradients.** A reverse-mode gradient
> (`jax.grad` / `jax.jacrev`) taken *directly* through a stiff network's `solve`
> (ASM / ADM / WATS) returns silent `NaN`/`Inf` when the reactor's `dtmax` is
> uncapped — no exception, so the garbage gradient flows into your optimizer and
> the fit never converges. `aquakin.calibrate` and `aquakin.sensitivity` guard
> this for you; if you roll your own loss + optimizer, either cap `dtmax`, use
> forward mode (`jax.jacfwd` with `adjoint=aquakin.forward_adjoint()`), or wrap
> your gradient in `reactor.check_gradient_finite(jax.grad(loss)(p))`
> (equivalently the free `aquakin.check_finite_gradient`) to get an actionable
> error instead of silent `NaN`.

### Forward sensitivity (cap-free stiff gradients)

Differentiating *through* a stiff reaction-network solve with ordinary AD goes
non-finite above an integrator-step threshold, and the usual workaround — a
global `dtmax` cap — forces tiny steps over the whole solve. `solve_sensitivity`
avoids both: it integrates the sensitivity `S = dC/dθ` *alongside* the state and
lets the adaptive step controller bound the sensitivity error too, so the step
tightens only where the sensitivity is stiff and the result is exact with no cap.

```python
import jax.numpy as jnp
import aquakin

network = aquakin.load_network("uv_h2o2")
conditions = network.default_conditions(1)
reactor = aquakin.BatchReactor(network, conditions)

C0 = network.default_concentrations()
params = network.default_parameters()
t_eval = jnp.linspace(0.0, 5.0, 6)

sol, S = reactor.solve_sensitivity(
    C0, params, t_span=(0.0, 5.0), t_eval=t_eval,
    sens_params=["H2O2_photolysis.k_photo", "OH_target.k_OH_target"],
)
# sol : the usual solution; S : dC/dθ, shape (n_t, n_species, n_sens_params)

# A richer wrapper with by-name accessors:
res = aquakin.forward_sensitivity(
    reactor, C0, params, t_span=(0.0, 5.0), t_eval=t_eval,
    sens_params=["H2O2_photolysis.k_photo"],
)
res.dC_dparam("target", "H2O2_photolysis.k_photo")   # (n_t,)
```

`solve_sensitivity` is available on `BatchReactor`, `PlugFlowReactor` and
`BiofilmReactor`. For more than one parameter it defaults to a CVODES-style
*simultaneous corrector* (`shared_factor=True`): the augmented Jacobian is
block-lower-triangular with one shared diagonal block, so that block is
factorised once per step and reused across the sensitivity columns instead of
factorising the full augmented system. This is several times faster than the
dense augmented solve on large stiff systems (e.g. the layered biofilm) and
gives bit-identical results.

### Cap-free reverse-mode gradients (stable adjoint)

`solve_sensitivity` scales with the parameter count, so for a scalar-loss
gradient over many parameters — the calibration case — reverse mode is wanted,
and that is the mode the `dtmax` cap exists for. The cap-free alternative there is
a hand-written discrete adjoint: the forward is an ordinary robust adaptive ESDIRK
(Kvaerno5) solve and the reverse is a per-step transposed solve over the saved
trajectory, finite at any step size with no cap. **`Plant.solve` uses it
automatically:** `gradient` defaults to `"auto"`, which keeps a plain forward solve
on the fast cached path but routes a solve under `jax.grad` to the cap-free
stable adjoint — so a stiff plant gradient is finite by default with nothing to
tune:

```python
sol = plant.solve(t_span=(0.0, T), t_eval=t_eval, params=params, y0=y0)
g = jax.grad(loss)(params)   # finite through the stiff, coupled BSM2 plant — no dtmax
```

This is what lets a reverse-mode gradient flow through the whole monolithic BSM2
solve — across the ASM↔ADM interface and the recycle loops — where differentiating
*through* the stiff solve is non-finite. It is exact through a transient solve:
`plant.solve` carries the integration time in the state, so the explicit time
dependence of a time-varying influent is captured exactly in the gradient.

### Choosing the integrator (`solver=`, `factormax=`)

A long dynamic plant run — the multi-hundred-day dynamic BSM2 simulation — is
**stiffness-bound**: the step count barely depends on the tolerance, and the
per-step cost is dominated by the implicit Jacobian factorisation of the whole
167-state plant. The forward solve defaults to `Kvaerno5` (a 7-stage L-stable
ESDIRK) **with a decoupled root finder** — the per-stage Newton tolerance is
loosened 10× from the step tolerance, so each step ends in fewer iterations and
is ~15–20% cheaper at preserved accuracy (the step controller still enforces the
solution accuracy). That speedup is automatic; nothing to pass.

Two knobs go further on the long stiff run:

```python
import diffrax
sol = plant.solve(t_span=(0.0, 609.0), t_eval=t_eval, params=params, y0=y0,
                  solver=diffrax.Kvaerno3(),   # 4 stages, less linear algebra/step
                  factormax=3.0)               # cap step growth (damps reject churn)
```

`Kvaerno3` takes somewhat more, but cheaper, steps; with `factormax` the two
stack to **~40% faster** than the old default, matching `Kvaerno5` to ~6e-5 on
the final state. Passing a `solver` opts out of the default Newton decoupling
(the solver's own root finder is used) — to keep it with a different order, pass
it explicitly: `diffrax.Kvaerno3(root_finder=diffrax.VeryChord(rtol=10*rtol,
atol=10*atol))`. Both knobs apply to the forward solve only (rejected alongside
`gradient="stable_adjoint"` or `events=`, which manage their own integrator).

A third knob, `colored_jacobian=True`, forms the per-step implicit Jacobian by
**sparse column compression** instead of densely:

```python
sol = plant.solve(t_span=(0.0, 609.0), t_eval=t_eval, params=params, y0=y0,
                  colored_jacobian=True)   # ~1.4x on dynamic BSM2, numerically identical
```

The plant Jacobian is sparse (dense per-unit kinetic blocks + sparse inter-unit
coupling), so it is built in a handful of colored Jacobian-vector products (~45
for BSM2) rather than one per state (167) — the dominant per-step linear-algebra
cost. The reconstructed matrix equals the dense Jacobian, so the trajectory and
gradient are unchanged to integration tolerance; only the cost of forming it
drops (**~1.4× on dynamic BSM2**, and it stacks with `Kvaerno3`/`factormax`). It
is built and guarded against the dense Jacobian once per plant, falling back to
the dense solver if the guard fails. Most worthwhile on a large stiff plant
(BSM2); on the small BSM1 the materialisation is not the bottleneck. Forward
solve only, like the knobs above.

### A non-AD fast lane (`forward_fast=True`)

If a solve never needs gradients (`jax.grad` / `calibrate` / `sensitivity` of the
result), `forward_fast=True` runs a lean integrator that skips the diffrax
adjoint / optimistix / lineax machinery entirely — a plain `lax.while_loop`
adaptive ESDIRK with a simplified Newton and the colored Jacobian:

```python
sol = plant.solve(t_span=(0.0, 609.0), t_eval=t_eval, params=params, y0=y0,
                  forward_fast=True)   # ~3x compile, ~1.3-1.9x run (dynamic BSM2)
```

That machinery exists to make the *whole solve* differentiable, and tracing it
dominates compile time — so dropping it gives **~3× faster compile** (a big deal
for the multi-minute full-BSM2 compile, which file-caching can't help because the
cost is Python-level tracing) **and ~1.3–1.9× faster run** on the dynamic BSM2
(the run gain narrows over a long run with dense `t_eval`, the compile win is the
robust benefit), at the same accuracy. The
per-step Jacobian is *still* colored forward-mode AD (the exact same matrix), so
the trajectory matches a valid adaptive solution to the same `rtol`; only
end-to-end differentiability is given up. It is opt-in and forward-only: it needs
concrete `params`/`y0` and raises a clear error under `jax.grad`/`jax.jit` or with
`events=`, and falls back to the diffrax path if the colored-Jacobian guard fails.

For reactor-level fits, the adjoint plumbing is hidden too: `aquakin.calibrate`
and `aquakin.sensitivity` take `ad_mode="forward"|"reverse"` and build the right
adjoint internally (no `diffrax` import), and `calibrate(check_finite=True)` (the
default) raises a friendly error with the remedy instead of returning silent
`NaN` gradients on a stiff network.

## Testing

```bash
pytest -m "not validation"   # unit + integration (fast)
pytest -m validation          # scientific validation against published data
pytest                        # everything
```

## License

MIT.
