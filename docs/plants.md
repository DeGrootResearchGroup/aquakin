# Plant-wide simulation

Beyond a single reactor, `aquakin.plant` assembles full treatment-plant
flowsheets — reactors, clarifiers, mixers, splitters, digesters, and their
recycle loops — and integrates the whole plant under one solve. The IWA
Benchmark Simulation Models BSM1 (an activated-sludge line) and BSM2 (the
activated-sludge line plus primary treatment and an anaerobic sludge train) ship
ready to build.

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
`asm3_2step_n2o` is described in the [Model catalogue](model_catalog.md).
`direct_n2o_emission(plant, sol)` reports the N₂O a solve stripped, for a model
that names dissolved oxygen `SO` as ASM1 does.

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
`steady_state()` fell back to integrating forward. After a fallback, check
`root.residual`: the fallback stops on the forward run's own test, so the
residual can be larger than the `tol` you asked for.

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

Gradient settings are described in
[Sensitivity & calibration](sensitivity_and_calibration.md#choosing-the-integrator).

## Building a custom flowsheet

The benchmark builders are convenience wrappers over the general `Plant` API.
To assemble your own flowsheet, create a plant with its time unit, add the
units (in any order: recycles are found automatically), wire the influents and
connections, and solve. The flowsheet is built in a function here because
later sections extend copies of it:

```python
def two_tanks(name):
    plant = aquakin.Plant(name, time_unit="d")
    plant.add_unit(aquakin.CSTRUnit("anoxic", model, volume=1000.0,
                                    input_port_names=["in"], conditions={"T": 293.15}))
    plant.add_unit(aquakin.CSTRUnit("aerobic", model, volume=1333.0,
                                    input_port_names=["in"], conditions={"T": 293.15},
                                    aeration=aquakin.Aeration(kla=240.0)))
    plant.add_influent("feed", model.influent(bsm1_feed, Q=18446.0, T=288.15),
                       to="anoxic.in")           # wire the feed to the first tank's inlet
    plant.connect("anoxic.out", "aerobic.in")    # unit to unit, as "unit.port"
    return plant

plant = two_tanks("my_plant")
t_eval = jnp.linspace(0.0, 10.0, 241)
sol = plant.solve(t_span=(0.0, 10.0), t_eval=t_eval)
print("effluent SNH:", round(float(plant.stream(sol, "aerobic.out").C_named("SNH")[-1]), 1))
# effluent SNH: 32.4
```

With no clarifier to hold the sludge, the biomass washes out of these two
tanks, so there is no nitrification and the effluent ammonia ends up above the
feed's (organic nitrogen is still released). The benchmark builders show a
full activated-sludge line. A plant can still be extended after it has been
solved: units, connections and influents added later are used by the next
solve and by `plant.stream`. A solution from before the change keeps the old
layout, so solve again before reading streams from the extended plant.

### The plant's time unit

`time_unit=` sets the plant's clock. `t_span` and `t_eval` are in it, flows are
in m³ per that unit, and so is every unit setting that contains a time: an
aeration `kla`, a primary clarifier's filter time constant `t_m`, an SBR's phase
lengths, a controller's integral time. Everything that comes with its own time
unit is converted to the clock before the solve: each kinetic model's rates (a
model whose rate constants are per second can run beside one that is per day,
and the model's parameters stay in their own units), each influent's times and
flows, and the constants built into the unit operations (the UV reactor's
exposure time in seconds, the primary clarifier's removal correlation in
minutes, the default BSM settling velocities in m/d).

The BSM1, BSM2 and A²O builders use days. A plant created without `time_unit`
still works: it takes the time unit its models share with a `FutureWarning`,
and raises `UnitsError` if the models disagree. `time_unit` will be required in
a future release.

### Pumps, doses and switches: events

Discontinuous operations are **events**, exactly as on a single reactor (see
[Reactors](reactors.md#events-pumps-dosing-phase-switches)): `at_times=` fires
at known times and keeps `jax.grad` finite, `cond_fn=` fires when a condition
crosses zero. On a plant the state passed to `cond_fn` and `apply` is the flat
plant state; find an entry from `plant.state_names()`, which lists each unit's
entries in state order. SBR phase switches are added automatically.

```python
names = plant.state_names()
flat = [(unit, entry) for unit, entries in names.items() for entry in entries]
i_ss = flat.index(("anoxic", "SS"))

slug = aquakin.Event(at_times=[5.0], apply=lambda t, y, p: y.at[i_ss].add(50.0),
                     name="slug")
sol_ev = plant.solve(t_span=(0.0, 10.0), t_eval=t_eval, events=[slug])
print(sol_ev.events_log)
# [(5.0, 'slug')]
```

This flattening holds while every unit's states are contiguous; under the
heat-balance temperature model (below) the temperatures sit at the end of the
flat vector instead.

## Unit operations

The plant library provides these units, all importable from `aquakin.plant`
except `TakacsClarifier` (`aquakin.plant.takacs`). Every one documents its
settings in the [API reference](api.md).

| Unit | What it models |
|---|---|
| `CSTRUnit` | a completely mixed reactor running a kinetic model, with optional `Aeration` |
| `IFASUnit` / `MBBRUnit` | a mixed tank with a depth-resolved biofilm on carrier media |
| `SBRUnit` | a sequencing batch reactor with timed phases (`SBRPhase`) |
| `MBRUnit` | a membrane bioreactor |
| `ADM1DigesterUnit` | a continuously fed, mesophilic ADM1 digester with a gas headspace |
| `TakacsClarifier` | a layered secondary clarifier |
| `PrimaryClarifier` | a primary clarifier with a flow-dependent removal correlation |
| `IdealThickener` | an ideal solids separator: thickener or dewatering |
| `MixerUnit`, `RatioSplitter`, `SetpointSplitter`, `ThresholdSplitter` | joining and splitting streams |
| `StorageTank`, `HydraulicDelayUnit` | variable-volume storage with an overflow bypass, and a first-order hydraulic lag |
| `DosingUnit` | inline chemical dosing of a `Reagent` (fixed or feedback-controlled) |
| `UVUnit`, `ChlorineContactUnit` | disinfection |
| `PIController` | a proportional-integral controller with anti-windup |
| `ASM1toADM1`, `ADM1toASM1` | translators between the activated-sludge and digester models |

`build_a2o` builds an anaerobic–anoxic–oxic (A²O) biological nutrient removal
plant on ASM2d in the same way as the BSM builders.

### IFAS and MBBR tanks

An integrated fixed-film activated sludge (IFAS) or moving-bed biofilm reactor
(MBBR) tank is a mixed bulk coupled to a biofilm on carrier media, resolved over
`n_layers` layers between the bulk and the carrier wall. `MBBRUnit` is the same
class as `IFASUnit`. By default the attached biomass is a mature film held at
its initial inventory, while solubles diffuse and react through it.

```python
from aquakin.plant import MBBRUnit

mbbr = MBBRUnit("mbbr", model, volume=500.0, input_port_names=["in"],
                specific_surface_area=500.0, fill_fraction=0.5,   # biofilm area 250 m²/m³
                biofilm_thickness=2e-4, n_layers=8,
                conditions={"T": 293.15}, aeration=aquakin.Aeration(kla=240.0))
mbbr_plant = aquakin.Plant("mbbr_demo", time_unit="d")
mbbr_plant.add_unit(mbbr)
mbbr_plant.add_influent("feed", model.influent(
    {"SS": 60.0, "SNH": 25.0, "SO": 2.0, "SALK": 7.0}, Q=2000.0), to="mbbr.in")

mbbr_ss = mbbr_plant.run_to_steady_state(criterion="scaled_residual")
worst = mbbr.layer_resolution(mbbr_plant.states_by_unit(mbbr_ss.state)["mbbr"])
print(round(max(worst.values()), 3))
# 0.058
```

`layer_resolution` reports, for each soluble species, the largest fractional
drop in concentration across one layer. A few per cent is well resolved; a
value near 1 means the substrate is used up inside a single layer and
`n_layers` should be raised. Use `criterion="scaled_residual"` for the steady
state of a plant with these tanks: the fixed biofilm adds a large constant block
to the state, which can make the default norm test stop while the bulk is still
changing.

To let the film grow, pass `dynamic_thickness=True` with `biofilm_density=`
(the film's particulate density, e.g. `1e4` g COD/m³). The thickness `L_f` then
becomes the last state entry, and the species stratify by growth rate.
`detachment_law=` sets what happens at the surface: `"quadratic"` (the
default), `"linear"` or `"constant"`, with `detachment_rate=`, let the thickness
follow from the balance of growth and detachment; `"balance"` holds the total
thickness at `biofilm_thickness` while the species still grow and stratify.

### Disinfection

`UVUnit` and `ChlorineContactUnit` pass the process stream through unchanged and
reduce an indicator-organism density carried on the stream (read back as
`StreamSeries.org`). The UV dose is the fluence rate times the exposure time
`V/Q` in seconds. The chlorine credit is CT, the residual times the contact time
T10 (the time by which 10% of the water has passed through,
`T10 = baffling_factor × V/Q`), with the residual a state that decays.

```python
from aquakin.plant import UVUnit, ChlorineContactUnit

dis_plant = two_tanks("disinfection")
dis_plant.add_unit(UVUnit("uv", model, volume=1.0, intensity=10.0, d10=5.0,
                          max_log=4.0, inlet_density=1e6))   # mW/cm², mJ/cm² per log
dis_plant.add_unit(ChlorineContactUnit("cct", model, volume=400.0, dose=5.0,
                                       ct_per_log=4.0 / 1440.0,   # 4 mg·min/L, in mg·d/L
                                       decay_rate=2.0, max_log=4.0))
dis_plant.connect("aerobic.out", "uv.in")
dis_plant.connect("uv.out", "cct.in")

dis_sol = dis_plant.solve(t_span=(0.0, 10.0), t_eval=t_eval)
removal = jnp.log10(1e6 / dis_plant.stream(dis_sol, "cct.out").org[-1])
print(f"{float(removal):.1f} log")
# 8.0 log
```

T10 and CT are in the **plant's** time unit, so convert a CT table given in
mg·min/L (divide by 1440 for a plant in days). `uv_dose`, `uv_log_inactivation`,
`ct_value`, `ct_log_removal`, `t10_from_baffling` and `t10_from_rtd` compute
the same credits standalone, for sizing a contactor.

### Aeration design

`Aeration(kla=...)` sets the oxygen transfer coefficient a tank receives.
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

### Plant temperature

By default a unit's temperature is the flow-weighted mix of its inlets, with no
thermal storage (`AlgebraicTemperature`). `HeatBalanceTemperature` instead gives
every tank with a volume (except the heated digester) a temperature state,
`V dT/dt = Q_in (T_in − T)`, so tanks lag and damp a diurnal or seasonal
influent temperature:

```python
from aquakin.plant import HeatBalanceTemperature

thermal = two_tanks("thermal")
thermal.set_temperature_model(HeatBalanceTemperature())
sol_T = thermal.solve(t_span=(0.0, 10.0), t_eval=t_eval)
T_out = thermal.stream(sol_T, "aerobic.out").T
print(float(T_out[0]), "->", round(float(T_out[-1]), 2), "K")
# 293.15 -> 288.15 K
```

The model changes the length of the plant state, so build a new `y0` after
setting it. It needs an influent that carries a temperature, and
`build_bsm2(temperature_model=...)` accepts it too. A flowsheet file cannot hold
it yet.

## Influents

`model.influent({...}, Q=..., T=...)` gives a constant feed; the BSM loaders
read the benchmark load files. Three more ways to describe a feed follow, and
`InfluentSeries` holds any measured record.

### A feed that follows a pattern

A load that follows a *shape* rather than a measured record needs no table file.
`InfluentExpression` takes the flow, the temperature and any species as
expressions of the plant's clock:

```python
from aquakin.plant import InfluentExpression

diurnal = build_bsm1(model)
diurnal.add_influent("feed", InfluentExpression(
    model,
    Q="18446 + 4000*sin(2*pi*t)",                 # a diurnal flow
    composition={"SNH": "31.6 + 8*sin(2*pi*t - 0.5)", "SS": 69.5},
))
```

Each value may equally be a plain number, so only the parts that vary need an
expression; a species that is not named is absent from the feed. The expressions
are ordinary arithmetic with `t`, the constants `pi` and `e`, and the functions
in `aquakin.plant.time_expr.FUNCTIONS` (`sin`, `cos`, `tan`, `exp`, `log`,
`log10`, `sqrt`, `abs`, `floor`, `min`, `max`, `mod`, `clip`, `step`). They are
evaluated inside the solve, so a run driven by one still differentiates.

There is no `if`: a value that switches uses `step` or `clip`, which stay
differentiable. A wastage pump that steps up for half the year is
`"300 + 150*step(t - 182)"`.

`t` is the plant's clock, and the expressions keep the clock they were written
on, so one written for a plant running on days still means days if the plant is
later put on hours. Because the whole feed is text, it goes into a flowsheet
file as it stands, with no table file beside it:

```yaml
feed:
  kind: influent
  model: asm1
  Q: 18446 + 4000*sin(2*pi*t)
  composition: {SNH: 31.6 + 8*sin(2*pi*t - 0.5), SS: 69.5}
```

### A feed from laboratory measurements

`characterize_influent` splits the aggregate measurements a laboratory reports
(total COD, TKN, ammonia, alkalinity, and optionally filtered COD) into the
ASM1 states, with the fractions in `InfluentFractions`. `read_influent_csv`
reads a time series, and its `column_map=` names your file's own columns, so a
laboratory or SCADA export is read as it stands; aggregate columns are
fractionated row by row.

```python
from pathlib import Path
from aquakin.plant import characterize_influent, read_influent_csv

lab_feed = characterize_influent(model, flow=18446.0, total_cod=420.0, tkn=40.0,
                                 ammonia=28.0, alkalinity=7.0, T=288.15)

Path("scada.csv").write_text(
    "hours,flow_m3h,COD_mgL,TKN_mgL,NH4_mgL,temp_C\n"
    "0,700,420,40,28,15\n6,900,480,44,31,15.5\n12,800,450,42,30,16\n24,700,420,40,28,15\n")
scada_feed = read_influent_csv(
    "scada.csv", model,
    column_map={"t": "hours", "Q": "flow_m3h", "T": "temp_C",
                "total_cod": "COD_mgL", "tkn": "TKN_mgL", "ammonia": "NH4_mgL"},
    units={"t": "h", "Q": "m3/h", "T": "degC"})
print(scada_feed.t, scada_feed.Q)
# [0.   0.25 0.5  1.  ] [16800. 21600. 19200. 16800.]
```

### Influent units

Influent data in other units are converted. Declare the units with `units=` on
`model.influent(...)`, `InfluentSeries` or `read_influent_csv`:

```python
feed = model.influent({"SS": 0.06}, Q=760.0, T=15.0,
                      units={"Q": "m3/h", "T": "degC", "SS": "kgCOD/m3"})
print(feed.Q[0], feed.T[0], feed.C[0, model.species_index["SS"]])
# 18240.0 288.15 60.0
```

Time goes to the model's time unit, flow to m³ per that unit, temperature to
kelvin, and each species to the unit the model declares for it; `feed.units`
lists the converted units. Only the scale changes, so a unit that measures
something else (an amount in moles where the model has a mass, or g COD where it
has g N) raises, as does a unit that cannot be read. Where the model leaves a
unit undeclared the data are used as given. The BSM influent loaders declare
their units for you.

## Checking a plant

### Units

The plant converts what it can: time units, influent data, concentrations on
connections between models that declare the same species in different units,
and the g/m³ constants built into the solids units. `plant.check_units()`
reports what is left, before you solve:

```python
print(plant.check_units().summary())
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

## Uncertainty and scenarios

`aquakin.monte_carlo` and `aquakin.compare_scenarios` run a function of a few
inputs many times; on a plant, the function sets the parameters and solves.
Here the input is the autotroph growth rate `asm1.muA` of the MBBR tank above,
whose biofilm nitrifies, and the output its effluent ammonia after two days
from the steady state:

```python
p0 = mbbr_plant.default_parameters()
i_mu = mbbr_plant.parameter_index("asm1.muA")

def effluent_snh(x):
    run = mbbr_plant.solve(t_span=(0.0, 2.0), params=p0.at[i_mu].set(x[0]),
                           y0=mbbr_ss.state)
    return mbbr_plant.stream(run, "mbbr.out").C_named("SNH")[-1]

mc = aquakin.monte_carlo(effluent_snh, {"muA": {"dist": "normal", "mean": 0.8, "std": 0.1}},
                         n_samples=8, batched=False)
print(mc.summary())

cs = aquakin.compare_scenarios(effluent_snh, {"base": {}, "slow nitrifiers": {"muA": 0.5}},
                               input_names=["muA"], baseline=[0.8], batched=False)
print(cs.table())
```

`batched=False` solves one sample at a time. `aquakin.optimize_design` uses the
same kind of function to minimize a cost subject to effluent limits; see
[Uncertainty propagation and design](sensitivity_and_calibration.md#uncertainty-propagation-and-design).
Fitting plant parameters to measured streams (`Plant.calibrate`) is in
[Calibrating a plant](sensitivity_and_calibration.md#calibrating-a-plant), and
the Fisher information and sensor ranking of a plant in
[Identifiability of plant parameters](sensitivity_and_calibration.md#identifiability-of-plant-parameters).

## Reporting: GHG, cost, and scenarios

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

## Saving a plant as a file

A plant can be written as a YAML (or JSON) file and rebuilt from it:

```python
bsm1.to_spec("flowsheet.yaml")
rebuilt = aquakin.Plant.from_spec("flowsheet.yaml")
```

The file lists every unit by its name, with an optional display `label` and all
of its settings (defaults included, so the file means the same plant whatever
the defaults of a later aquakin), the connections as `unit.port -> unit.port`,
the influents, named streams and recycle options, and the version of aquakin
that wrote it:

```yaml
format: 2
aquakin: 0.2.0
name: BSM1
models:
  asm1: {source: shipped, sha256: ...}
units:
  feed: {kind: influent, model: asm1, Q: 18446.0, composition: {SI: 30.0, SS: 69.5, ...}}
  tank3: {kind: cstr, label: Aerobic zone 1, model: asm1, volume: 1333.0, ...}
connections:
  - tank2.out -> tank3.inlet
```

Every model a unit uses is listed under `models:`, either a shipped model or a
model YAML file beside the flowsheet. A model file can `extends:` a shipped
model, so a variant is a few lines: `examples/projects/bsm2/` holds BSM2 this
way, with its 15 °C ASM1 parameter set as `models/asm1_bsm2.yaml`. Each model
carries a fingerprint of its content, and reading a flowsheet whose model has
changed since it was written warns.

A time-varying influent is kept as a table file beside the flowsheet (columns
`t`, `Q`, optionally `T`, and species; see `aquakin.plant.read_influent_table`).
`to_spec(path)` writes it to `data/<influent>.csv` unless it was already read
from a table inside the flowsheet's folder.

Reading a file checks it strictly: a misspelled setting, a value of the wrong
type or a model that is not listed is an error that names where it is. Anything
a file cannot express yet (array-valued settings such as initial
concentrations, SBR phases, a heat-balance temperature model) raises a
`SpecError` that names it rather than being left out.

`aquakin.plant.spec_fingerprint(spec)` gives a fingerprint of the plant a file
builds: its settings, wiring and the content of its models and influent tables,
but not its layout, labels or where its files sit. Record it beside a result to
tell later whether the result still matches the flowsheet.

## Studies

A *study* is one analysis of a flowsheet saved in the project with its results.
Each study is a folder under `studies/` with a `study.yaml` saying what to run:

```yaml
# studies/effluent/study.yaml
study:
  kind: simulate
  parameters: {asm1.muH: 5.0}          # changed from the flowsheet's
  initial_state: steady                # or default, or a named-state file
  time: {t_span: [0, 14], t_eval: {start: 0, stop: 14, num: 337}}
  streams: [effluent]                  # saved whole: flow, temperature, species
  observables:
    - stream: effluent
      channels: [COD, TKN, NOx, TN, TSS]
```

```bash
aquakin study run studies/effluent      # writes results/ and run.yaml
aquakin study check studies/effluent    # fresh, stale (and why), damaged or failed
```

or `aquakin.study.run_study(path)` / `check_study(path)` from Python. Every
result is keyed by name (parameters by `model.param`, states by unit and state
name, trajectories as CSV columns such as `effluent.SNH`), so results stay
readable after the flowsheet is reorganized. A channel is a species, a
laboratory measurement (`COD`, `TKN`, `NOx`, `TN`, `TP`, `TSS`, `Q`, `T`) or a
weighted sum of your own; see
[A channel of your own](sensitivity_and_calibration.md#a-channel-of-your-own).

Besides `simulate`, a study can be a `fisher` information of a sensor design, a
`calibrate` fit to observations, a `profile` likelihood of one parameter, or a
`rank` of candidate sensor sets (the methods are described in
[Designing experiments](sensitivity_and_calibration.md#designing-experiments)).
A fit reads its observations from a CSV table in the study folder, a `t` column
and one `<stream>.<channel>` column per measured channel. Because it fits the
likelihood of the data, it needs each channel's measurement error: `sigma`, one
value for all channels or one per channel, or `estimate_sigma: true` to
estimate it from the fit:

```yaml
study:
  kind: calibrate
  free: [asm1.muH, asm1.KS]
  observables:
    - {stream: effluent, channels: [SNH, TSS]}
  data: data.csv          # columns t, effluent.SNH, effluent.TSS
  sigma: [0.2, 2.0]       # each channel's measurement error
  laplace: true
```

A study follows the flowsheet: once a setting, a connection, a model or an
influent table it was run on changes, `check` reports it **stale** and names
what changed. A newer aquakin is noted but does not make a result stale; read
the changelog to decide whether to run it again. Running a study that has
results replaces them only when asked (`--replace`); `aquakin study copy` keeps
a copy first. The new results take the old ones' place only when the run
succeeds: a run that fails or is cancelled leaves the earlier results as they
were, and `check` notes the attempt.

`aquakin study freeze` instead pins a study to a snapshot of its plant, taken
inside the study folder with every model flattened (shipped models too) and
every influent table copied, so neither an edit to the project nor an aquakin
upgrade can change what it runs. `aquakin study unfreeze --discard-results`
makes it follow the flowsheet again and deletes its results, which were
computed on the snapshot.

From Python, `aquakin.study.create_study(project, name, settings)` and
`update_study(path, settings)` write a `study.yaml`, checking the settings as a
run would read them; `plant_choices(spec)` lists what a study of that flowsheet
may name, and `study_results(path)` reads a study's results back.

## The flowsheet editor

```bash
pip install "aquakin[ui]"
aquakin-ui examples/projects/bsm2/flowsheet.yaml
```

opens a flowsheet editor in your browser, served on this machine only. Drag
units from the palette, wire their ports, and edit each unit in a form generated
from its own documentation; aquakin builds each unit and then the whole plant as
you edit and lists any errors.

- **Open** and **Save** read and write the files described above. Save writes
  exactly what `plant.to_spec(path)` would, copying a model (with every file it
  `extends:`) or influent table from outside the project folder into it. Save
  refuses to overwrite a flowsheet that changed on disk since the window opened
  it, so two windows cannot silently undo each other's work.
- Opening another file while the flowsheet has unsaved edits asks first (save
  them, or discard them). **Undo** takes back the last edits: a removed unit or
  wire, a move, a new unit or wire, or an applied form. Delete removes the
  selected units and wires (Backspace does not).
- Edits typed into a unit's form are applied when you select another unit, and
  a value that cannot be read is named under its field.

**Simulate** saves the flowsheet, creates a study called `simulation` the first
time (`simulation_<file name>` for a file not named `flowsheet.yaml`, so each
flowsheet in a folder has its own), runs the plant to steady state, and shows a
table of every stream that leaves the plant: its flow, COD, TKN, NOx, TN and TSS,
with every species under *Flow and species* (temperatures in °C). To run in time
instead, or to save other streams, **Edit** the `simulation` study; a study that
runs in time shows a chart per channel of the stream you pick. From Python,
`aquakin.study.simulation_study(spec)` gives those settings.

The **Studies** tab lists the project's studies and their status (fresh, stale
and why, not run, failed, cancelled). From it you can run a study and watch its
progress or cancel it; freeze, unfreeze or copy it; and open **Results**: a
simulation's streams, a Fisher study's standard deviations, design criteria and
information matrix, a ranking of sensor sets, a calibration's fitted parameters
with the data against the fit, or a profile's curve and interval. Running a
study first saves the flowsheet's unsaved edits, so it runs what you see (a
frozen study runs its snapshot). Replacing results or unfreezing asks first and
offers to keep a copy. Tick **Don't ask again** to replace results without the
question from then on; the **Ask before replacing results** switch at the top of
the tab brings it back (the choice is kept in `aquakin/ui.json` in your
configuration folder). Unfreezing always asks.

**New study** sets one up in a dialog: pick its kind, and the fields that kind
uses appear, with the plant's own streams, parameters and measurable channels to
choose from; **Edit** opens a study's settings in the same dialog. The dialog
writes the same `study.yaml` you would write by hand, and a data table or saved
state chosen from elsewhere is copied into the study folder. A study set up in
the editor follows the flowsheet file that is open.
