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

### The secondary clarifier and the benchmark

The BSM plants' layered secondary clarifier (`TakacsClarifier`) follows the
benchmark settlers in everything but one choice: how solids settle from one layer
into the next. The benchmark uses the Takács rule, which takes the smaller of the
two layers' settling fluxes. aquakin uses by default the Godunov flux of the
settling conservation law (Bürger, Diehl & Nopens 2011). The two agree wherever no
layer is denser than the one below it. Where neighbouring layers in the sludge
blanket sit at nearly equal concentration, as under a dynamic BSM2 influent, the
Takács rule switches between their fluxes each time their order flips, which makes
the blanket pulse and forces the integrator into small steps; the Godunov flux has
no such switch. On BSM2 the steady state is the same, the effluent's mean
concentrations under the dynamic influent differ by under 0.1%, and a dynamic solve
is several times cheaper. To reproduce the benchmark settler exactly, pass
`settler_interface_flux="takacs"` to `build_bsm1` / `build_bsm2` (or
`interface_flux="takacs"` to `TakacsClarifier`).

### Other activated-sludge models and N₂O

`build_bsm1` runs any activated-sludge model that keeps the ASM1 names for the
components it shares with ASM1, so the BSM1 influent maps onto it by name. The
clarifier settles every species whose name starts with `X` and counts those
carrying COD as suspended solids. Pass `settling_species=` and `tss_species=` to
choose them yourself. The `aeration=` argument replaces the aeration of named
tanks, for example to put them under dissolved-oxygen control.

Aeration also strips dissolved gases other than oxygen. List them in
`Aeration(stripped=...)` with the ratio of their transfer coefficient to the
oxygen one. Each gas then leaves the liquid at `ratio × kLa_O2 × (sat − C)`
inside the solve, with the saturation 0 unless you give a `(ratio, saturation)`
pair. Under DO control the gas follows the controller's kLa. For N₂O and NO the
ratio is close to 1.

```python
from aquakin.plant import Aeration
from aquakin.plant.bsm import build_bsm1, direct_n2o_emission

model = aquakin.load_model("asmn_n2o")
gases = {"SN2O": 1.0, "SNO": 1.0}
plant = build_bsm1(
    model,
    aeration={f"tank{i}": Aeration(do_setpoint=2.0, stripped=gases) for i in (3, 4, 5)},
)
plant.add_influent("feed", model.influent(
    {"SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
     "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0}, Q=18446.0))
sol = plant.solve(t_span=(0.0, 10.0), t_eval=jnp.linspace(0.0, 10.0, 41))

print(direct_n2o_emission(plant, sol), "kg N2O-N / d")
print(plant.mass_balance(sol).summary())
```

`direct_n2o_emission` uses each tank's own stripping ratio, so it reports the
N₂O the solve removed, and `plant.mass_balance` counts the stripped gases in its
gas term. Because the stripping is part of the plant equations, `jax.grad` and
the sensitivity tools see how N₂O emission depends on the kinetic parameters and
the DO setpoint.

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

Both methods report `ss.residual`, the scaled residual
`max_i |dy_i/dt| / max(|y_i|, scale_floor)` of the state they return. By
default `run_to_steady_state()` stops on a looser norm test, which large states
dominate. To hold a forward run to the same criterion and tolerance as
`steady_state()`, for example to cross-check the algebraic root, pass
`criterion="scaled_residual"`:

```python
fwd = plant.run_to_steady_state(criterion="scaled_residual", tol=1e-8, scale_floor=1e-6)
root = plant.steady_state(tol=1e-8, scale_floor=1e-6)
print(fwd.residual, root.residual)   # both <= 1e-8
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

```{note}
The benchmark literature gives two versions of the BSM1 indices. `evaluate_bsm1`
follows the **updated** evaluation (Alex et al. 2008): EQI weights 2 (TSS),
1 (COD), 2 (BOD₅), 30 (TKN) and 10 (nitrate), and OCI = AE + PE + ME +
5·sludge, with AE = S_sat·V·kLa / 1800 and per-stream pumping factors. The
original evaluation (Copp 2002) weights TKN and nitrate at 20 each and uses
different aeration and pumping formulas, so compare against the
updated-convention reference values. `evaluate_bsm2` uses the same EQI weights
and the full BSM2 OCI (Gernaey et al. 2014).
```

## Reporting: GHG, cost, and scenarios

On top of the EQI/OCI evaluation, `aquakin` reports a **greenhouse-gas
footprint** (CO₂e/d) and a monetised **operating cost** (currency/d), and
tabulates scenarios side by side:

```python
ev  = aquakin.evaluate_bsm2(plant, sol, params)
n2o = aquakin.direct_n2o_emission(plant, sol)      # 0 unless the model resolves N2O (see above)

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
plant = aquakin.Plant("my_plant", time_unit="d")
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

### The plant's time unit

`time_unit=` sets the plant's clock. `t_span` and `t_eval` are in it, flows are
in m3 per that unit, and so is every unit parameter that contains a time: an
aeration `kla`, a primary clarifier's filter time constant `t_m`, an SBR's phase
lengths, a controller's integral time. Everything that comes with its own time
unit is converted to the clock before the solve:

- each kinetic model's rates, so a model whose rate constants are per second
  can run beside one that is per day. The model's parameters stay in their own
  units;
- each influent's times and flows, which `model.influent(...)` and the loaders
  give in the model's time unit;
- the constants built into the unit operations: the UV reactor's exposure time
  in seconds, the primary clarifier's removal correlation in minutes, and the
  Takács settler's default (BSM) settling velocities in m/d.

The BSM1, BSM2 and A²O builders use days. A plant created without `time_unit`
still works: it takes the time unit its models share and gives a
`FutureWarning`, and raises `UnitsError` if the models disagree. `time_unit`
will be required in a future release.

### Checking units before solving

The plant converts time units but passes stream concentrations between units
unconverted, so every part must use the same concentration units.
`plant.check_units()` compares the units the parts declare and reports any
disagreement before you solve:

```python
report = plant.check_units()
print(report.summary())
# Plant units: OK (time unit: d)
```

It checks that:

- units whose internals fix a concentration basis get it. The thickener needs
  its TSS species in `g/m3`, and so does the Takács settler while it uses its
  default (BSM) settling parameters, which are per `g/m3` of solids. Settling
  parameters you supply are taken to be in your plant's units;
- on a plant created without `time_unit`, the kinetic models agree on their
  time unit;
- each model passes its own `model.check_units()`; its findings are listed as
  warnings.

Comparisons account for scale (`mg/L` matches `g/m3`; `kg/m3` does not), and a
unit that is not declared is not checked. `plant.check()` includes the report,
and `plant.connect` also rejects a translator built for models other than the
ones it connects.

Influent data in other units are converted rather than checked. Declare the
units with `units=` on `model.influent(...)`, `InfluentSeries` or
`read_influent_csv`:

```python
feed = model.influent({"SS": 0.06}, Q=760.0, T=15.0,
                      units={"Q": "m3/h", "T": "degC", "SS": "kgCOD/m3"})
```

For a model whose rate constants are per day this feed becomes `Q = 18240`
m3/d, `T = 288.15` K and `SS = 60` g COD/m3. Time goes to the model's time
unit, flow to m3 per that unit, temperature to Kelvin, and each species to the
unit the model declares for it; `feed.units` then lists the converted units.
Only the scale changes, so a unit that measures something else (an amount in
moles where the model has a mass, or g COD where it has g N) raises, as does a
unit that cannot be read. Where the model leaves a unit undeclared there is
nothing to convert to, and the data are used as given. The BSM influent loaders
declare their units for you. A plant whose clock differs from the model's then
moves the times and flows onto its own clock when the influent is added.

The ASM1 ↔ ADM1 translators (`ASM1toADM1`, `ADM1toASM1`) convert units
themselves: they read the species units the two models they are built with
declare and convert them into the units their equations are written in (`g/m3`
for ASM1; `kg/m3` and `kmol/m3` for ADM1). An ADM1 variant declared in `g/m3`
therefore works unchanged. A unit they cannot convert, such as inorganic
nitrogen given as a mass where the interface expects moles, raises when the
translator is built; a unit they cannot read gives a warning and is taken to be
the working unit. Because the conversion follows the models the translator was
built with, `plant.connect` rejects a translator whose models declare different
units from the units it connects.

The plant library also provides secondary clarifiers, an ADM1 digester,
sequencing-batch and membrane-bioreactor units, IFAS/MBBR units, disinfection
(UV and chlorine contact), dosing, aeration-system physics, and temperature
models. See the [API reference](api.md) for the full unit catalog, and
`Plant.calibrate` in [Sensitivity & calibration](sensitivity_and_calibration.md#calibrating-a-plant)
for fitting plant parameters to measured streams.

## Saving a plant as a file

A plant can be written as a YAML (or JSON) file and rebuilt from it:

```python
plant.to_spec("flowsheet.yaml")
plant = aquakin.Plant.from_spec("flowsheet.yaml")
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
  influent: {kind: influent, model: asm1, Q: 18446.0, composition: {SS: 69.5, ...}}
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
concentrations, SBR phases) raises a `SpecError` that names it rather than being
left out.

`aquakin.plant.spec.spec_fingerprint(spec)` gives a fingerprint of the plant a
file builds: its settings, wiring and the content of its models and influent
tables, but not its layout, labels or where its files sit. Record it beside a
result to tell later whether the result still matches the flowsheet.

## Studies

A *study* is one analysis of a flowsheet saved in the project with its results:
a simulation (a steady state, or a dynamic run) or the Fisher information of a
sensor design. Each study is a folder under `studies/` with a `study.yaml`
saying what to run:

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

or `aquakin.study.run_study(path)` / `check_study(path)` from Python. Besides
`simulate`, a study can be a `fisher` information of a sensor design, a
`calibrate` fit to observations, a `profile` likelihood of one parameter, or a
`rank` of candidate sensor sets. A fit reads its observations from a CSV table
in the study folder, a `t` column and one `<stream>.<channel>` column per
measured channel:

```yaml
study:
  kind: calibrate
  free: [asm1.muH, asm1.KS]
  observables:
    - {stream: effluent, channels: [SNH, TSS]}
  data: data.csv          # columns t, effluent.SNH, effluent.TSS
  loss: nll
  sigma: [0.2, 2.0]
  laplace: true
```

Every
result is keyed by name (parameters by `model.param`, states by unit and state
name, trajectories as CSV columns such as `effluent.SNH`), so results stay
readable after the flowsheet is reorganized.

A study follows the flowsheet: once a setting, a connection, a model or an
influent table it was run on changes, `check` reports it **stale** and names
what changed. A newer aquakin is noted but does not make a result stale; read
the changelog to decide whether to run it again. Running a study that has
results replaces them only when asked (`--replace`); `aquakin study copy` keeps
a copy first.

`aquakin study freeze` instead pins a study to a snapshot of its plant, taken
inside the study folder with every model flattened (shipped models too) and
every influent table copied, so neither an edit to the project nor an aquakin
upgrade can change what it runs. `aquakin study unfreeze --discard-results`
makes it follow the flowsheet again and deletes its results, which were
computed on the snapshot.

## The flowsheet editor

```bash
pip install "aquakin[ui]"
aquakin-ui examples/projects/bsm2/flowsheet.yaml
```

opens a flowsheet editor in your browser, served on this machine only. Drag
units from the palette, wire their ports, and edit each unit in a form generated
from its own documentation; aquakin builds each unit and then the whole plant as
you edit and lists any errors. **Open** and **Save** read and write the files
described above; Save writes exactly what `plant.to_spec(path)` would, copying a
model (with every file it `extends:`) or influent table from outside the project
folder into it. Save refuses to overwrite a flowsheet that changed on disk since
the window opened it, so two windows cannot silently undo each other's work.

The **Studies** tab lists the project's studies and where each stands (fresh,
stale and why, not run, failed, cancelled). Run one to watch its progress and
cancel it; freeze, unfreeze or copy it. Running a study that has results, or
unfreezing one, asks first and offers to keep a copy.

