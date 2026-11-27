# The plant-wide core on BSM1

`bsm1_comparison.py` runs the plant-wide core (`../models/plantwide_core.yaml`)
and the shipped `asm2d` on BSM1 (`aquakin.plant.bsm.build_bsm1`, open loop,
ideal clarifier) with BSM1's constant dry-weather influent, and prints tank 5
(whose solubles are the effluent's) at each steady state. Run it as a script
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

## Results (20 C, tank 5, g/m3)

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
10 g P/m3 (BSM1's influent has none). Tank 5 at its steady state (g/m3):

| | NH4-N | NO2-N | NO3-N | nitrifiers |
|---|---|---|---|---|
| core (one-step) | 1.11 | - | 11.02 | 107.7 |
| nitrite level | 0.22 | 0.15 | 9.35 | 73.7 AOB + 38.8 NOB |

Its AOB grow faster at low ammonium (half-saturation 0.7 against the core
nitrifiers' 1.0), and its heterotrophs denitrify nitrite as well as nitrate.
From the steady state the plant's N, P and COD balances close to 1e-13 over
10 days, the stripped N2 and the oxygen taken up counted
(`tests/validation/test_plantwide_nitrite_bsm1.py`).
