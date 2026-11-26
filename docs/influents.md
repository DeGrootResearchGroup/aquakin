# Influents

A plant is driven by its influents: the flow, temperature and composition of
each feed over time. This page covers the ways to describe one, and the flow
setpoints inside the plant that can follow a schedule in the same way.

The examples use ASM1:

```python
import aquakin
from aquakin.plant.bsm import build_bsm1

model = aquakin.load_model("asm1")
```

`model.influent({...}, Q=..., T=...)` gives a constant feed, and the BSM
loaders read the benchmark load files (see
[Plant-wide simulation](plants.md)). Three more ways to describe a feed follow, and
`InfluentSeries` holds any measured record.

## A feed that follows a pattern

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

## Pumps and splits that follow a schedule

The same expressions drive the plant's own flow setpoints: a clarifier's
`underflow_Q` or `overflow_Q`, a `SetpointSplitter`'s pump flows, a
`ThresholdSplitter`'s `threshold` and a primary clarifier's sludge fraction
`f_PS`. Write the expression where the number would go:

```python
from aquakin.plant import PrimaryClarifier, SetpointSplitter

sludge = SetpointSplitter("sludge_split", model,
                          {"waste": "385 + 75*step(t - 30)"},   # wastage up on day 30
                          remainder_port="ras")
primary = PrimaryClarifier("primary", model, volume=900.0,
                           f_PS="0.007*(1 + 0.2*sin(2*pi*t))")  # a diurnal sludge draw
print(sludge.scheduled_setpoints(), primary.scheduled_setpoints())
# ('waste',) ('f_PS',)
```

% invisible-code-block: python
%     assert sludge.scheduled_setpoints() == ("waste",)
%     assert primary.scheduled_setpoints() == ("f_PS",)

Like every unit setting, a setpoint is on the plant's clock: `t` is in the
plant's time unit and a flow in m³ per that unit. Its value is clipped as it is
evaluated, a flow to zero or more and a fraction to between 0 and 1, because no
check made when the unit is built can tell what an expression does over a whole
run; a constant that is out of range is still refused. In a flowsheet file the
expression is the setting's value:

```yaml
sludge_split:
  kind: setpoint_splitter
  model: asm1
  output_port_flows: {waste: 385 + 75*step(t - 30)}
  remainder_port: ras
```

A run driven by such a setpoint differentiates with respect to the model
parameters and the other setpoints; the numbers inside an expression are not
parameters, so to differentiate with respect to a pump flow, keep that flow a
number. A plant with a setpoint that varies in time works out its recycle
streams afresh at every step instead of reusing a map computed once per run, so
it solves somewhat more slowly than the same plant with constant setpoints.

## A feed from laboratory measurements

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

## Influent units

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
