# Plant-wide simulation

Beyond a single reactor, `aquakin.plant` assembles full treatment-plant
flowsheets — reactors, clarifiers, mixers, splitters, digesters, and their
recycle loops — and integrates the whole plant under one solve. The IWA
benchmark simulation models BSM1 and BSM2 ship ready to build.

## The benchmark plants

Load the biological model, build the plant, wire an influent, and drive it to
steady state:

```python
import jax.numpy as jnp
import aquakin
from aquakin.plant.bsm import build_bsm1, load_bsm1_influent, evaluate_bsm1

model = aquakin.load_model("asm1")
plant = build_bsm1(model)     # 5 aerated/anoxic reactors + secondary clarifier + recycles

# A constant average-load influent, wired to the plant's canonical inlet.
plant.add_influent("feed", model.influent(
    {"SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
     "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0}, Q=18446.0))

# Integrate until the plant settles — a self-terminating steady-state event,
# so there is no horizon to guess.
ss = plant.run_to_steady_state()
print("converged:", ss.converged, "after", round(ss.time), "days")
```

Reconstruct the clarified effluent from the steady state and read its quality:

```python
eff = plant.stream(ss.solution, plant.effluent_endpoint)
print("effluent SNH:", round(float(eff.C_named("SNH")[-1]), 2), "g N / m³")
```

`build_bsm2(model)` assembles the full BSM2 sludge train — primary clarifier,
thickener, an ADM1 digester with the ASM1↔ADM1 interfaces, dewatering, and the
reject-water recycle — the same way.

## Steady state, faster

`run_to_steady_state()` integrates forward until the plant settles. For design
sweeps you can instead snap straight to the steady state **algebraically**, via
pseudo-transient continuation — typically about 10× faster, robust on stiff
topologies, and **differentiable**, so `jax.grad` of a loss on the steady state
flows to the plant parameters:

```python
ss = plant.steady_state()
print("converged:", ss.converged, "in", int(ss.iterations), "iterations")
```

## Dynamic runs and performance indices

For a dynamic simulation, drive a fresh plant with a time-varying influent,
warm-started from the steady state, and score the benchmark performance indices:

```python
dyn = build_bsm1(model)
dyn.add_influent("feed", load_bsm1_influent("dry", model))   # 14-day diurnal load
sol = dyn.solve(t_span=(0.0, 14.0), t_eval=jnp.linspace(0.0, 14.0, 15), y0=ss.state)

ev = evaluate_bsm1(dyn, sol)          # Effluent Quality & Operational Cost indices
print(f"EQI = {ev.eqi:.0f} kg/d   OCI = {ev.oci:.0f}")
```

`evaluate_bsm2(plant, sol, params)` does the same for BSM2 and additionally
reports the physical flows (aeration energy, pumping, sludge and methane
production) that feed the reporting layer below.

## Reporting: GHG, cost, and scenarios

On top of the EQI/OCI evaluation, `aquakin` reports a **greenhouse-gas
footprint** (CO₂e/d) and a monetised **operating cost** (currency/d), and
tabulates scenarios side by side:

```python
ev  = aquakin.evaluate_bsm2(plant, sol, params)
n2o = aquakin.direct_n2o_emission(plant, sol)      # 0 unless the model resolves N2O

fp = aquakin.carbon_footprint(                     # kg CO2e/d, with a breakdown
    ev.total_energy(), grid_factor=0.4, n2o_emission=n2o,
    methane_production=ev.methane_production, ch4_fugitive_fraction=0.015)

oc = aquakin.operating_cost(                       # currency/d OPEX
    energy_kwh_per_d=ev.total_energy(), carbon_kg_cod_per_d=ev.carbon_mass,
    sludge_kg_tss_per_d=ev.sludge_production, methane_kg_per_d=ev.methane_production,
    factors=aquakin.CostFactors(energy_price=0.12), co2e_per_d=fp.total_co2e)

print(fp)   # labelled CO2e breakdown
print(oc)   # labelled cost breakdown
print(aquakin.kpi_comparison({"baseline": ev, "low-DO": ev_low_do}).table())
```

`carbon_footprint` weights direct N₂O, grid-energy CO₂e, and fugitive biogas
methane, crediting recovered biogas energy; `operating_cost` prices energy,
carbon, sludge, and biogas; and `kpi_comparison` collects any of these report
objects into one standardised KPI table.

## Building a custom flowsheet

The benchmark builders are convenience wrappers over the general `Plant` API.
To assemble your own flowsheet, construct the units, add them (downstream-first
if there is a recycle), wire influents and connections, and solve:

```python
plant = aquakin.Plant("my_plant")
tank = aquakin.CSTRUnit("aerobic", model, volume=1333.0,
                        input_port_names=["in"], conditions={"T": 293.15},
                        aeration=aquakin.Aeration(kla=240.0))
plant.add_unit(tank)
plant.add_influent("feed",
                  model.influent({"SS": 60.0, "SNH": 25.0}, Q=18446.0),
                  to="aerobic.in")            # wire the feed to the tank's inlet
sol = plant.solve(t_span=(0.0, 10.0), t_eval=t_eval)
```

Units are wired to each other with `plant.connect("aerobic.out", "clarifier.in")`;
`add_influent(..., to=...)` attaches an external feed to a unit port.

The plant library also provides secondary clarifiers, an ADM1 digester,
sequencing-batch and membrane-bioreactor units, IFAS/MBBR units, disinfection
(UV and chlorine contact), dosing, aeration-system physics, and temperature
models. See the [API reference](api.md) for the full unit catalog, and
`Plant.calibrate` in [Sensitivity & calibration](sensitivity_and_calibration.md#calibrating-a-plant)
for fitting plant parameters to measured streams.
