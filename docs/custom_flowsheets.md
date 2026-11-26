# Building a flowsheet

The benchmark builders on [Plant-wide simulation](plants.md) are convenience
wrappers over the general `Plant` API. This page assembles plants of your own:
the units, their wiring and time unit, events, and each of the unit operations
the plant library provides.

The Python blocks on this page continue from one another. They start from ASM1
and a constant feed of average BSM1 composition:

```python
import aquakin
import jax.numpy as jnp

model = aquakin.load_model("asm1")
bsm1_feed = {"SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
             "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0}
```

## A first flowsheet

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

The plant library provides these units, all importable from `aquakin.plant`.
Every one documents its settings in the [API reference](api.md).

| Unit | What it models |
|---|---|
| `CSTRUnit` | a completely mixed reactor running a kinetic model, with optional `Aeration` |
| `IFASUnit` / `MBBRUnit` | a mixed tank with a depth-resolved biofilm on carrier media |
| `SBRUnit` | a sequencing batch reactor with timed phases (`SBRPhase`) |
| `MBRUnit` | a membrane bioreactor |
| `ADM1DigesterUnit` | a continuously fed, mesophilic ADM1 digester with a gas headspace |
| `HeadspaceDigesterUnit` | a digester whose gas headspace belongs to the unit, for any model that declares its dissolved gases (`gases:`) |
| `TakacsClarifier` | a layered secondary clarifier |
| `IdealClarifier` | a stateless ideal clarifier that sends a fixed fraction of the particulates to the underflow |
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

### Sequencing batch reactors: how the feed arrives

An `SBRUnit` cycles one tank through timed phases (`SBRPhase`: fill, react,
settle, decant, idle). A real SBR only takes its feed during the fill, so what
happens to wastewater that arrives at other times depends on the plant. The
SBR's `feed` setting says which plant it is, and it has no default:

| `feed=` | A plant like | What happens to the inflow |
|---|---|---|
| `"continuous"` | a continuous-inflow SBR (ICEAS or CFID type) | The tank takes the inflow whenever it arrives, in every phase; its volume follows the inflow minus the decant. `feed_flow` is left unset. |
| `"storage"` | an equalization (holding) tank ahead of the SBR | A holding volume inside the unit, `storage_volume` m³, receives the inflow; the fill pump draws `feed_flow` from it during fill phases. What it cannot hold leaves through the `overflow` port; when it runs empty, the draw falls to what arrives. |
| `"batch"` | a laboratory or reservoir-fed SBR | The inlet must deliver `feed_flow` during fill phases and nothing in any other phase. Each solve checks the influent feeding the SBR over the solve span and refuses one that does not, or an SBR fed by another unit, whose flow is not known before the solve. |

Plants with several SBRs in parallel, the influent routed to whichever is
filling, are not modelled yet.

Here the influent arrives continuously and a holding volume buffers it for the
fill pump:

```python
from aquakin.plant import InterfaceSettling, SBRPhase, SBRUnit, storage_report

phases = [SBRPhase("fill", 0.1, feed=True), SBRPhase("react", 0.3, kla=240.0),
          SBRPhase("settle", 0.1, settle=True), SBRPhase("decant", 0.1, decant=True),
          SBRPhase("idle", 0.05)]                       # one cycle every 0.65 d
sbr = SBRUnit("sbr", model, phases, full_volume=1000.0, feed_flow=5000.0,
              decant_flow=5000.0, settling=InterfaceSettling(v_settle=400.0, area=200.0),
              particulate_species=["XI", "XS", "XB_H", "XB_A", "XP", "XND"],
              initial_fraction=0.5, conditions={"T": 293.15},
              feed="storage", storage_volume=600.0)
sbr_plant = aquakin.Plant("sbr_demo", time_unit="d")
sbr_plant.add_unit(sbr)
sbr_plant.add_influent("feed", model.influent(bsm1_feed, Q=769.0), to="sbr.feed")
print(sbr.output_ports)
# ['effluent', 'overflow']

sbr_sol = sbr_plant.solve(t_span=(0.0, 1.3), t_eval=jnp.linspace(0.0, 1.3, 261))
report = storage_report(sbr_plant, sbr_sol, "sbr")
print(report.summary())
```

% invisible-code-block: python
%     assert sbr.output_ports == ["effluent", "overflow"], "the ports stated above changed"
%     assert report.level.max() <= 600.0 * (1 + 1e-6) and report.level.min() > 0.0
%     assert abs(sbr_plant.mass_balance(sbr_sol)["COD"].relative_imbalance) < 2e-2

`storage_report` gives the holding volume's level and its inflow, draw,
overflow and shortfall over the saved times, with the overflow and the
shortfall integrated. The overflow is also a stream (`sbr.overflow`) that can be
wired onward, and `plant.mass_balance` counts it as an outflow when it is left
unconnected, like the effluent. The holding volume is part of the SBR's state
(`storage.<species>`, `storage.V`) and of its inventory in the mass balance.
Its pump cuts out over the bottom 1 % of `storage_volume` and the overflow takes
over in the top 1 %, so the level stays between them without a located event,
and `jax.grad` flows through both.

For `feed="batch"`, write a fill-only influent as an expression of time (see
[A feed that follows a pattern](influents.md#a-feed-that-follows-a-pattern)), for the
cycle above `Q="5000*(1 - step(mod(t, 0.65) - 0.1))"`, or as a table whose flow
steps at the phase boundaries.

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
[Calibrating a plant](plant_calibration.md), and
the Fisher information and sensor ranking of a plant in
[Identifiability of plant parameters](plant_calibration.md#identifiability-of-plant-parameters).
