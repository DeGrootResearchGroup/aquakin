# The plant-wide core on BSM1

`bsm1_comparison.py` runs the plant-wide core (`../models/plantwide_core.yaml`)
and the shipped `asm2d` on BSM1 (`aquakin.plant.bsm.build_bsm1`, open loop,
the ten-layer Takács settler) with BSM1's constant dry-weather influent, and
prints tank 5 at each steady state.

## The benchmark

BSM1's published open-loop steady state (Alex et al. 2008), ASM1 at 15 C,
tank 5, g/m3: NH4-N 1.733, NO3-N 10.415, O2 0.491, heterotrophs 2559.3,
autotrophs 149.8, SS 0.889. `build_bsm1` with ASM1 at BSM1's parameter values
reproduces all of them to the printed figures. The plant-wide models are
compared at 15 C (the benchmark) and at 20 C, their parameters' reference
temperature:

| tank 5 (g/m3) | NH4-N | NO2-N | NO3-N | O2 | nitrifiers |
|---|---|---|---|---|---|
| BSM1 benchmark (ASM1), 15 C | 1.73 | - | 10.42 | 0.49 | 149.8 |
| core, 15 C | 3.86 | - | 10.65 | 0.38 | 125.6 |
| nitrite level, 15 C | 9.85 | 5.68 | 0.00 | 0.82 | 98.7 AOB, NOB washed out |
| core, 20 C | 1.11 | - | 11.02 | | 107.7 |
| nitrite level, 20 C | 1.77 | 5.40 | 2.34 | 0.50 | 108.4 AOB + 13.3 NOB |

The core's autotrophs are ASM2d's (growth 1.0 1/d and decay 0.15 1/d at
20 C, 0.59 and 0.087 at 15 C), against BSM1's ASM1 values at 15 C (0.5 and
0.05). At 15 C, with ASM1's decay alone the core leaves 1.25 g N/m3 of
ammonium; with ASM1's growth and decay together, 3.58: the autotrophs' rates
do not account for the gap on their own (the core's tank 5 also holds less
oxygen, 0.38 against 0.49). The nitrite level's NOB
(Hiatt & Grady 2008: growth 0.78 1/d at 20 C, 0.46 at 15 C, oxygen
half-saturation 1.2) cannot hold on at BSM1's sludge age and aeration at
15 C.
 Run it as a script
(about 2 minutes). `tests/validation/test_plantwide_core_bsm1.py` checks the
results.

## The influent

BSM1's ASM1 influent (Copp 2002) in each model's species: `SS` as fermentable
`SF`, inert products with the particulate inerts, total nitrogen kept by putting
what the model's organic N contents leave in ammonium, alkalinity as inorganic
carbon for the core (12 g C per mol). ASM2d gets phosphate and alkalinity in
excess, since its growth is limited by both and the core's by neither.

The aerated tanks strip the gases the core declares (its hydrogen, methane,
nitrogen and CO2); ASM2d declares none.

## Against ASM2d (20 C, tank 5, g/m3)

| | NH4-N | NO3-N | XH | XS | propionate | H2 |
|---|---|---|---|---|---|---|
| core, aligned | 1.153 | 10.603 | 1957.3 | 64.76 | 0 | 0 |
| ASM2d, aligned, without its positivity limiter | 1.153 | 10.603 | 1957.3 | 64.76 | - | - |
| ASM2d, aligned, with its limiter | 1.151 | 10.612 | 1957.3 | 64.75 | - | - |
| core, as shipped | 1.108 | 11.019 | 1986.3 | 63.42 | 0.004 | 0.004 |
| ASM2d, as shipped | 1.119 | 10.425 | 1957.4 | 63.35 | - | - |

*Aligned* switches off the processes the models do not share: the core's
fermentation and slower anaerobic decay, ASM2d's fermentation, with ASM2d's
anaerobic hydrolysis factor set to the core's. The two plants then agree to
six figures once ASM2d's positivity limiter is off. The limiter throttles
oxygen uptake below 1e-3 g/m3, where the anoxic tanks sit, and moves ASM2d's
steady state by 0.2%.

As shipped, the core leaves 0.6 g N/m3 more nitrate. Each of its own choices,
set to ASM2d's alone, lowers it from 11.02: fermentation off (its hydrogen,
about 17% of the fermented COD, is stripped by the air rather than used for
denitrification) to 10.78; anaerobic decay as aerobic to 10.86; `etafe` at
ASM2d's 0.4 to 10.81; all three together to 10.35.

## The nitrite level

`plantwide_nitrite` runs on the same plant, with ASM1's nitrifiers carried over
as AOB and NOB in the ratio of their yields and phosphate at a non-limiting
10 g P/m3 (BSM1's influent has none). Its nitrifiers have the kinetics of Hiatt
& Grady (2008). Tank 5 at its steady state (g/m3):

| | NH4-N | NO2-N | NO3-N | nitrifiers |
|---|---|---|---|---|
| core (one-step) | 1.11 | - | 11.02 | 107.7 |
| nitrite level | 1.77 | 5.40 | 2.34 | 108.4 AOB + 13.3 NOB |
| nitrite level, NOB oxygen half-saturation 0.6 | 2.74 | 1.14 | 9.14 | 33.3 NOB |

Its AOB grow more slowly than the core's nitrifiers (0.78 against 1.0 1/d, and
on free ammonia, whose half-saturation is about 1.9 g NH4-N/m3 at pH 7 and
20 C), so more ammonium is left. Nitrite builds up because tank 5 runs at
0.5 g O2/m3 and the NOB's oxygen half-saturation (1.2) is twice the AOB's: the
NOB only just outgrow their washout. At the AOB's 0.6 most of the nitrite is
oxidized, and the NOB, competing for oxygen, leave more ammonium. From the
steady state the plant's N, P and COD balances close over 10 days, the stripped
N2 and the oxygen taken up counted
(`tests/validation/test_plantwide_nitrite_bsm1.py`).
