# The plant-wide nitrite level in a SHARON reactor

`sharon.py` runs the nitrite level (`../models/plantwide_nitrite.yaml`) in the
reactor of Hellinga et al. (1998): a stirred tank without sludge retention,
1.5 d residence time, fed digester centrate (1000 g NH4-N/m3). Run it as a
script (about 30 s). `tests/validation/test_plantwide_nitrite_sharon.py`
checks the results.

Without sludge retention a group stays only if its net growth exceeds the
dilution rate, 1 / (1.5 d). Hellinga et al.'s reactor at 35 C kept the AOB and
lost the NOB, so ammonium was oxidized to nitrite only; at typical plant
temperatures neither group grows that fast.

Against their lab reactor this one is aerated continuously, gets no methanol
(so no denitrification) and runs at a fixed pH of 7.4, their predicted
average.

## Results (reactor, after 60 days, g/m3)

| | NH4-N | NO2-N | NO3-N | AOB | NOB |
|---|---|---|---|---|---|
| 35 C | 400.8 | 569.0 | 0 | 82.2 | 0 |
| 20 C | 979.6 | 18.1 | 0 | 2.7 (washing out) | 0 |

The NOB lose at 35 C to free ammonia (about 10 g NH3-N/m3 here) as well as to
temperature. The AOB stay only because they are not inhibited by free
ammonia, as Hellinga et al. measured (no inhibition up to 6000 g NH4-N/m3 at
pH 7 and 40 C).
