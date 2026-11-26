# Plant-wide simulation

Beyond a single reactor, `aquakin.plant` assembles full treatment-plant
flowsheets — reactors, clarifiers, mixers, splitters, digesters, and their
recycle loops — and integrates the whole plant under one solve. The IWA
Benchmark Simulation Models BSM1 (an activated-sludge line) and BSM2 (the
activated-sludge line plus primary treatment and an anaerobic sludge train) ship
ready to build.

This page runs the benchmark plants: their steady state, a dynamic run and its
performance indices, design and reporting, and checking a plant's units and mass
balance. The other plant pages build on it:

| Page | Covers |
|---|---|
| [Building a flowsheet](custom_flowsheets.md) | a plant of your own: units, wiring, the time unit, events, and every unit operation |
| [Influents](influents.md) | feeds that follow a pattern, scheduled pumps, laboratory and SCADA records, influent units |
| [Flowsheet files, studies and the editor](plant_files.md) | saving a plant as a file, reproducible studies, and the browser flowsheet editor |

If you have not used aquakin before, read [Getting started](getting_started.md)
first; single reactors, integrator settings and events are covered in
[Reactors](reactors.md). The shipped kinetic models are listed in the
[Model catalogue](model_catalog.md).

The Python blocks on this page continue from one another, so the page can be
followed as one program. Most run in seconds to a minute; the BSM1 dynamic run
takes several minutes, and the BSM2 blocks are marked because they take longer.

## The benchmark plants

### BSM1

Load the biological model, build the plant, wire an influent, and drive it to
steady state:

```python
import aquakin
import jax.numpy as jnp
from aquakin.plant.bsm import build_bsm1, load_bsm1_influent, evaluate_bsm1

model = aquakin.load_model("asm1")
bsm1 = build_bsm1(model)      # 5 anoxic/aerated reactors + secondary clarifier + recycles

# A constant average-load influent, wired to the plant's inlet.
bsm1_feed = {"SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
             "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0}
bsm1.add_influent("feed", model.influent(bsm1_feed, Q=18446.0))

# Integrate until the plant settles: the solve stops itself once the plant is at
# steady state, so there is no horizon to guess.
ss = bsm1.run_to_steady_state()
print("converged:", ss.converged, "after", round(ss.time), "days")
# converged: True after 33 days
```

Reconstruct the clarified effluent from that run and read its quality:

```python
eff = bsm1.stream(ss.solution, bsm1.effluent_endpoint)
print("effluent SNH:", round(float(eff.C_named("SNH")[-1]), 2), "g N/m³")
# effluent SNH: 0.46 g N/m³
```

The plant integrates the state of each unit, not the streams between them, so
`plant.stream(solution, endpoint)` recomputes a stream (flow and concentrations
over time) from a solution. An endpoint is a named stream such as
`"effluent"` (`plant.list_streams()`) or a `"unit.port"` (`plant.list_ports()`).

### BSM2

`build_bsm2()` assembles the full BSM2 plant: primary clarifier, the activated
sludge line, thickener, an ADM1 digester with the ASM1↔ADM1 interfaces,
dewatering, and the reject-water recycle. Three things differ from BSM1:

- BSM2 runs ASM1 with its temperature corrections referenced to 15 °C. Use
  `bsm2_asm1_model()` for the plant and its influent; `build_bsm2()` uses it by
  default, while passing the plain `load_model("asm1")` runs the 20 °C reference.
- The benchmark parameter set is `bsm2_parameters(asm1, adm1)`; pass it to every
  solve and evaluation.
- Warm-start with `bsm2_warm_start(plant)`: the digester's retention time of
  about 19 days makes a cold start slow and stiff.

This block takes several minutes:

% skip: next "a BSM2 solve that takes several minutes; its API is checked by the BSM2 tests"

```python
from aquakin.plant.bsm import (bsm2_asm1_model, bsm2_constant_influent, bsm2_parameters,
                               bsm2_warm_start, build_bsm2, evaluate_bsm2)

asm1_bsm2 = bsm2_asm1_model()
adm1 = aquakin.load_model("adm1")
bsm2_params = bsm2_parameters(asm1_bsm2, adm1)

def run_bsm2(do_control):
    plant = build_bsm2(asm1_bsm2, adm1, do_control=do_control)
    plant.add_influent("feed", bsm2_constant_influent(asm1_bsm2))
    sol = plant.solve(t_span=(0.0, 30.0), t_eval=jnp.linspace(0.0, 30.0, 31),
                      params=bsm2_params, y0=bsm2_warm_start(plant),
                      rtol=1e-5, atol=1e-3,
                      integrator=aquakin.IntegratorConfig(max_steps=500_000))
    return plant, sol, evaluate_bsm2(plant, sol, bsm2_params)

bsm2, bsm2_sol, bsm2_eval = run_bsm2(do_control=False)
print(bsm2_eval)
```

`do_control=True` closes the dissolved-oxygen (DO) loop: a PI controller
senses the DO in reactor 4 and sets the aeration of reactors 3 to 5. The
benchmark's dynamic load files are read with `load_bsm2_influent(profile,
asm1_bsm2)`, with `profile` one of `"dry"`, `"rain"` or `"storm"`.

### The secondary clarifier and the benchmark

The BSM plants use a ten-layer secondary clarifier (`TakacsClarifier`). By
default it computes the settling flux from one layer to the next with the
Godunov flux of the settling conservation law (Bürger, Diehl & Nopens 2011)
rather than the benchmark's Takács rule. Steady states are the same, the mean
effluent concentrations of a dynamic BSM2 run differ by under 0.1%, and a
dynamic solve is several times cheaper. To reproduce the benchmark settler
exactly, pass `settler_interface_flux="takacs"` to `build_bsm1` / `build_bsm2`
(or `interface_flux="takacs"` to `TakacsClarifier`).

```{note}
Why the two differ: the Takács rule takes the smaller of the two layers'
settling fluxes. Where neighbouring layers of the sludge blanket sit at nearly
equal concentration, as under a dynamic BSM2 influent, it switches between
their fluxes each time their order flips, which makes the blanket pulse and
forces the integrator into small steps. The Godunov flux agrees with it wherever
no layer is denser than the one below, and has no such switch.
```

### Other activated-sludge models

`build_bsm1` runs any activated-sludge model that keeps the ASM1 names for the
components it shares with ASM1, so the BSM1 influent maps onto it by name. The
clarifier settles every species whose name starts with `X` and counts those
carrying COD as suspended solids; pass `settling_species=` and `tss_species=` to
choose them yourself. The `aeration=` argument replaces the aeration of named
tanks with your own `Aeration` objects, for example to put them under DO
control (`Aeration(do_setpoint=2.0)`) or, for a model that calls oxygen
something other than `SO`, to name it with `Aeration(species=...)`.

Aeration also strips dissolved gases other than oxygen, inside the solve, in
proportion to the oxygen mass-transfer coefficient kLa. List
them in `Aeration(stripped={"SN2O": 1.0, "SNO": 1.0})` with the ratio of each
gas's transfer coefficient to the oxygen one (close to 1 for N₂O and NO); each
then leaves the liquid at `ratio × kLa_O2 × (sat − C)`, with the saturation 0
unless you give a `(ratio, saturation)` pair. Under DO control the gas follows
the controller's kLa. Because the stripping is part of the plant equations,
`jax.grad` and the sensitivity tools see how the emission depends on the
kinetic parameters and the DO setpoint, and `plant.mass_balance` counts the
stripped gases in its gas term. The shipped nitrous-oxide model
`asm3_2step_n2o` is described in the [Model catalogue](model_catalog.md). It
names dissolved oxygen `SO2`, so its aerated tanks name that species too:

```python
from aquakin.plant import Aeration

n2o_model = aquakin.load_model("asm3_2step_n2o")
n2o_aeration = Aeration(do_setpoint=2.0, species="SO2", stripped={"SN2O": 1.0})
bsm1_n2o = build_bsm1(n2o_model, aeration={f"tank{i}": n2o_aeration for i in (3, 4, 5)})
```

A model that declares its dissolved gases in a `gases:` block (see [Model
format](model_format.md)) need not list them: `Aeration(stripped="model")`
strips every declared gas other than the aerated one, each at its transfer
ratio from the gas library, toward the concentration in equilibrium with its
share of the air at the tank's temperature (inorganic carbon as its dissolved
CO₂ at the tank's pH). The air is dry air at one atmosphere times
`pressure_factor`; give `air={"CO2": 0.02}` to change a gas's mole fraction,
for example in a covered tank. Oxygen still uses `do_sat`. An unaerated tank
with some surface exchange is `Aeration(kla=<surface kLa>, stripped="model")`;
without an `Aeration` a tank exchanges no gas.

Wire an influent in that model's species (`bsm1_n2o.add_influent(...)`) and
solve as for ASM1. `direct_n2o_emission(bsm1_n2o, sol)` then reports the N₂O-N
the solve stripped (kg N/d), time-averaged over the solution: for a tank whose
`Aeration` strips the gas it is the transfer the solve applied, temperature
correction included. For a tank that does not strip it, or when you pass
`kla_ratio=`, it is an estimate from that tank's oxygen kLa (read from the
species its `Aeration` names, so it works whatever the model calls oxygen).

## Steady state, faster

`run_to_steady_state()` integrates forward in time until the plant settles.
For design sweeps you can instead find the steady state **algebraically**, by
pseudo-transient continuation (PTC: a damped Newton iteration on
`dy/dt = 0`). It is typically several times faster, and it is
**differentiable**: `jax.grad` of a loss on the steady state flows to the plant
parameters.

```python
root = bsm1.steady_state()
print("converged:", root.converged, "by", root.method, "in", root.iterations, "iterations")
# converged: True by ptc in 34 iterations
```

Use `steady_state()` for sweeps and gradients, and `run_to_steady_state()` when
you also want the path to steady state (`ss.solution`). `root.method` says how
the state was found: `"ptc"`, or `"ptc->forward"` when PTC did not converge and
`steady_state()` fell back to integrating forward. The fallback stops on the
same scaled-residual test, with the `tol` and `scale_floor` you passed, and
reports `converged` only when `root.residual` meets that `tol`; options for
the forward run go in `fallback_kwargs`.

Both methods report `residual`, the scaled residual
`max_i |dy_i/dt| / max(|y_i|, scale_floor)` of the state they return. By
default `run_to_steady_state()` stops on a looser test of the norm of `dy/dt`,
which the largest states dominate. To hold a forward run to the same criterion
and tolerance as `steady_state()`, for example to cross-check the algebraic
root, pass `criterion="scaled_residual"`:

```python
fwd = bsm1.run_to_steady_state(criterion="scaled_residual", tol=1e-7)
root = bsm1.steady_state(tol=1e-7)
print(root.method, fwd.residual <= 1e-7, root.residual <= 1e-7)
# ptc True True
```

## Dynamic runs and performance indices

For a dynamic simulation, drive a fresh plant with the benchmark's 14-day dry
weather influent, warm-started from the steady state. The benchmark protocol
uses the first 7 days to settle the plant and scores the last 7 on the Effluent
Quality Index (EQI, kg of pollution units per day) and the Operational Cost
Index (OCI). Save output every 15 minutes, since the indices are time
integrals over the saved points. This block takes several minutes (see
[Faster dynamic solves](#faster-dynamic-solves) for a quicker forward run):

```python
dyn = build_bsm1(model)
dyn.add_influent("feed", load_bsm1_influent("dry", model))
first_week = dyn.solve(t_span=(0.0, 7.0), y0=root.state)
dyn_sol = dyn.solve(t_span=(7.0, 14.0), t_eval=jnp.linspace(7.0, 14.0, 673),
                    y0=first_week.state[-1])

ev = evaluate_bsm1(dyn, dyn_sol)
print(f"EQI = {ev.eqi:.0f} kg/d   OCI = {ev.oci:.0f}")
# EQI = 5051 kg/d   OCI = 14454
```

`print(ev)` gives the full breakdown. `evaluate_bsm2(plant, sol, params)` does
the same for BSM2 and also reports the sludge-train flows (methane production,
heating energy, external carbon) used by the reporting tools below.

```{note}
The benchmark literature gives two versions of the BSM1 indices. `evaluate_bsm1`
follows the **updated** evaluation (Alex et al. 2008): EQI weights 2 (TSS),
1 (COD), 2 (BOD₅), 30 (TKN) and 10 (nitrate), and OCI = AE + PE + ME +
5·sludge, with the aeration energy AE = S_sat·V·kLa / 1800 and per-stream
pumping factors. The original evaluation (Copp 2002) weights TKN and nitrate at
20 each and uses different aeration and pumping formulas, so compare against the
updated-convention reference values. `evaluate_bsm2` uses the same EQI weights
and the full BSM2 OCI (Gernaey et al. 2014).
```

### Faster dynamic solves

A plain forward run that you will not differentiate can use the lean forward
integrator, which compiles and runs several times faster:

```python
fast = dyn.solve(t_span=(7.0, 14.0), t_eval=jnp.linspace(7.0, 14.0, 673),
                 y0=first_week.state[-1], forward_fast=True)
print(f"EQI = {evaluate_bsm1(dyn, fast).eqi:.0f} kg/d")
# EQI = 5051 kg/d
```

`forward_fast=True` needs concrete `params` and `y0` (not traced values), and
its result is **not differentiable**. It honours only `rtol`, `atol` and
`IntegratorConfig(max_steps=...)`; any other `IntegratorConfig` setting raises
rather than being ignored, and it cannot be combined with `events=`.

Two other settings tune the ordinary solve, through
`integrator=aquakin.IntegratorConfig(...)`:

- `order=5` selects the fifth-order Kvaerno5 method in place of the default
  third-order Kvaerno3, which can take fewer steps at tight tolerances.
- `colored_jacobian` controls whether the Jacobian of each implicit step is
  built sparsely by graph colouring, which pays on large plants. The default
  `"auto"` uses it for the reverse-mode gradient of a plant with many states
  (BSM2 yes, BSM1 no; `plant.colored_jacobian_decision()` reports the choice);
  `True` forces it on, `False` off.

The other integrator settings are described in
[Choosing the integrator](reactors.md#choosing-the-integrator), and gradient
settings in
[Differentiating a solve](sensitivity_and_calibration.md#differentiating-a-solve).

## Design and reporting

### Aeration design

`Aeration(kla=...)` sets the clean-water oxygen transfer coefficient a tank
receives. `alpha` (fouling of transfer in process water, typically 0.4 to 0.7
in activated sludge), and with `temperature_correction=True` the `kla_theta`
factor, scale it into the kLa the mass balance applies; they apply the same way
to a kLa set by a DO controller or by `kla_signal`. The aeration energy and the
blower air flow are computed on the clean-water kLa, because that is what the
blowers produce: a fouled tank needs more of it for the same oxygen.
`AerationSystem` adds the diffusers and blower behind it (submergence depth,
standard oxygen-transfer efficiency SOTE, fouling, blower efficiency) and turns
a kLa into an air flow and a blower power:

```python
from aquakin.plant import AerationSystem, design_summary

print(design_summary(kla=240.0, volume=1333.0, system=AerationSystem(depth=4.5)).report())

ev_blower = evaluate_bsm1(dyn, dyn_sol, aeration_system=AerationSystem(depth=4.0))
print(round(ev_blower.aeration_energy), "kWh/d,", round(ev_blower.air_flow), "m³ air/d")
```

Passed as `aeration_system=` to `evaluate_bsm1` or `evaluate_bsm2`, it replaces
the benchmark's fixed aeration-energy formula in the OCI and adds the air flow
to the report.

### Sizing and achieved sludge age

`size_activated_sludge` turns design targets for the solids retention time
(SRT, the sludge age) and hydraulic retention time (HRT) into a basin volume
and a wastage flow. `plant.sludge_age(sol)` reports what a solved plant
actually achieved: SRT, HRT and the food-to-microorganism ratio (F:M). Because
the SRT follows from the wastage flow, comparing the two is how you adjust the
wastage flow to a target SRT.

```python
from aquakin.plant import size_activated_sludge

print(size_activated_sludge(SRT=10.0, Q=18446.0, HRT_h=8.0, n_tanks=5, ras_ratio=1.0))
print(dyn.sludge_age(dyn_sol).summary())
```

### Greenhouse gases, cost and scenarios

On top of the EQI/OCI evaluation, `aquakin` reports a **greenhouse-gas
footprint** (kg CO₂-equivalent per day) and a monetized **operating cost**
(currency per day), and tabulates scenarios side by side. This block uses the
BSM2 runs from above and takes several minutes:

% skip: next "a BSM2 solve that takes several minutes; its API is checked by the BSM2 tests"

```python
_, _, bsm2_do_eval = run_bsm2(do_control=True)

fp = aquakin.carbon_footprint(                     # kg CO2e/d, with a breakdown
    bsm2_eval.total_energy(), grid_factor=0.4,
    methane_production=bsm2_eval.methane_production, ch4_fugitive_fraction=0.015)

oc = aquakin.operating_cost(                       # currency/d
    energy_kwh_per_d=bsm2_eval.total_energy(), carbon_kg_cod_per_d=bsm2_eval.carbon_mass,
    sludge_kg_tss_per_d=bsm2_eval.sludge_production,
    methane_kg_per_d=bsm2_eval.methane_production,
    factors=aquakin.CostFactors(energy_price=0.12), co2e_per_d=fp.total_co2e)

print(fp)   # labelled CO2e breakdown
print(oc)   # labelled cost breakdown
print(aquakin.kpi_comparison({"open loop": bsm2_eval, "DO control": bsm2_do_eval}).table())
```

`carbon_footprint` weights grid-energy CO₂e, direct N₂O (`n2o_emission=`, for
a model that resolves it) and fugitive biogas methane, crediting recovered
biogas energy; `operating_cost` prices energy, carbon, sludge and biogas; and
`kpi_comparison` collects any of these report objects into one standardized KPI
table.

## Checking a plant

### Units

The plant converts what it can: time units, influent data, concentrations on
connections between models that declare the same species in different units,
and the g/m³ constants built into the solids units. `plant.check_units()`
reports what is left, before you solve:

```python
print(bsm1.check_units().summary())
# Plant units: OK (time unit: d)
```

It checks that, on a plant created without `time_unit`, the kinetic models
agree on their time unit; that a unit with a fixed concentration basis gets it
from its model; and that each model passes its own `model.check_units()`.
Comparisons account for scale (`mg/L` matches `g/m3`; `kg/m3` does not).
`plant.check()` also reports wiring problems such as an input port with nothing
connected, and includes this report.

In brief, the conversions are:

- The thickener and the Takács settler put each solids species on the g/m³
  basis of their built-in constants, so a model with its particulates in
  `kgCOD/m3` thickens and settles as one in `g/m3`. A solids species declared in
  moles cannot be converted without a molar mass; it is used as if in g/m³, and
  `check_units()` reports it. Settling parameters you supply are used as given.
- Two models with the same species in the same order but some in different
  units are joined directly: `plant.connect` and `plant.add_influent` insert a
  `UnitScalingTranslator`. Models with different species need an explicit
  translator.
- The ASM1↔ADM1 translators (`ASM1toADM1`, `ADM1toASM1`) read the units the two
  models declare, so an ADM1 variant declared in g/m³ works unchanged.
- A pair of units that cannot be converted (moles against grams, or an
  unreadable unit) raises `WiringError` when connected.

### Mass balance

`plant.mass_balance(sol)` closes the balance of COD, nitrogen and phosphorus
over a solved window: what came in, what left in the streams, what left as gas
(oxygen transferred by aeration, digester biogas, nitrogen gas from
denitrification, stripped gases) and what accumulated, all on one gram basis
across models. The imbalance is computed independently, so it is a real check:

```python
mb = dyn.mass_balance(dyn_sol)
print(mb.summary())
print(mb.closed(rtol=1e-2), round(mb["N"].relative_imbalance, 4))
# True 0.0008
```

The gas terms are integrated over the saved points, so save them finely enough
to resolve the dynamics (the 15-minute output above is ample).

One closure is approximate however finely you save: a `TakacsClarifier` with
`composition_mode="lumped_tss"` (the BSM2 default, and BSM1's with
`use_takacs=True`) tracks only the total solids in each layer, so it values the
nitrogen and phosphorus held in its sludge blanket at the model's default
solids composition. COD is exact. While the blanket is filling or emptying,
nitrogen misses by about 0.2% (on BSM1 over days 1 to 3 from the warm start);
at steady state it closes. The balance then lists the reason in
`mb["N"].notes` and `mb["P"].notes`, and `summary()` prints it. Use
`composition_mode="per_species"` when you need the transient nitrogen and
phosphorus balance exact.
