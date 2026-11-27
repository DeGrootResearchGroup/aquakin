# The plant-wide core on BSM1

`bsm1_comparison.py` runs the plant-wide core (`../models/plantwide_core.yaml`)
and the shipped `asm2d` on BSM1 (`aquakin.plant.bsm.build_bsm1`, open loop,
the ten-layer Takács settler) with BSM1's constant dry-weather influent, and
prints tank 5 at each steady state.

## The benchmark

BSM1's published open-loop steady state (Alex et al. 2008), ASM1 at 15 C,
tank 5, g/m3: NH4-N 1.733, NO3-N 10.415, O2 0.491, heterotrophs 2559.3,
autotrophs 149.8, SS 0.889. `build_bsm1` with ASM1 at BSM1's parameter values
reproduces all of them to the printed figures. These are ASM1's results with
BSM1's agreed parameters, a reference point between models, not measurements:
a difference from them says the models differ, not which is right. The
plant-wide models are
compared at 15 C (the benchmark) and at 20 C, their parameters' reference
temperature:

| tank 5 (g/m3) | NH4-N | NO2-N | NO3-N | O2 | nitrifiers |
|---|---|---|---|---|---|
| BSM1 benchmark (ASM1), 15 C | 1.73 | - | 10.42 | 0.49 | 149.8 |
| core, 15 C | 3.86 | - | 10.65 | 0.38 | 125.6 |
| nitrite level, 15 C | 7.85 | 0.28 | 9.34 | 0.42 | 104.7 AOB + 37.3 NOB |
| core, 20 C | 1.11 | - | 11.02 | | 107.7 |
| nitrite level, 20 C | 1.57 | 0.10 | 13.96 | 0.22 | 109.8 AOB + 42.8 NOB |

The core's autotrophs are ASM2d's (growth 1.0 1/d and decay 0.15 1/d at
20 C, 0.59 and 0.087 at 15 C), against BSM1's ASM1 values at 15 C (0.5 and
0.05). At 15 C, with ASM1's decay alone the core leaves 1.25 g N/m3 of
ammonium; with ASM1's growth and decay together, 3.58: the autotrophs' rates
do not account for the gap on their own (the core's tank 5 also holds less
oxygen, 0.38 against 0.49).

The nitrite level keeps its NOB and holds nitrite below 0.3 g N/m3 at both
temperatures, as plants at low DO do. Its ammonium at 15 C reflects AOB growth
and decay values that are still provisional.

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
10 g P/m3 (BSM1's influent has none). Its results at 15 and 20 C are in the
benchmark table above. The NOB's oxygen half-saturation decides whether nitrite
accumulates: with Hiatt & Grady's (2008) 1.2 g O2/m3 the NOB wash out at 15 C
and nitrite reaches 5.4 g N/m3 at 20 C; with the apparent floc value measured on
activated sludge, 0.23 (Daebel et al. 2007), they persist. From the steady state
the plant's N, P and COD balances close over 10 days, the stripped N2 and the
oxygen taken up counted (`tests/validation/test_plantwide_nitrite_bsm1.py`).
