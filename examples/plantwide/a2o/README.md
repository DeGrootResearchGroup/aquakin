# The plant-wide bio-P level on an A2O

`a2o_comparison.py` runs the plant-wide model's bio-P level
(`../models/plantwide_bio_p.yaml`) and the shipped `asm2d` on the same A2O
(`aquakin.plant.build_a2o`) with ASM2d's default influent, and prints, at each
steady state, the end of the anaerobic zone (fermentable substrate, VFA,
phosphate), the PAOs and the effluent phosphate. Run it as a script (about 10
minutes). `tests/validation/test_plantwide_bio_p_a2o.py` checks the results.

## The plant-wide influent

ASM2d's influent, species by species, with:

- its one VFA split into acetate and propionate, 75% / 25% by default (sewage
  VFA carries a propionate share; all acetate is a sensitivity case);
- alkalinity as inorganic carbon (1 mol per mol, 12 g C).

The reactors start from ASM2d's healthy mixed liquor, with PAO glycogen added
and both GAO groups at 50 g COD/m3, so the PAO / GAO competition starts from
both sides.

## Reading a steady state of this model

A steady state with no GAOs is an equilibrium whether or not GAOs can invade
it, and they invade slowly (a fraction of a percent per day), slower than the
pseudo-transient solver's convergence test can see. So each plant-wide steady
state is followed by a stability run: both GAO groups seeded at 1 g COD/m3 and
150 days forward; a negative growth rate means the state stands.

On an all-acetate VFA at 20 C and pH 7 the PAO / GAO competition is close to
neutral, and the GAOs' anoxic processes favour Competibacter; a propionate
share, a lower temperature or a higher pH favour the PAOs, as López-Vázquez et
al. (2009) predict. The heterotrophs' fermentation also makes propionate, so
both influents below give a stable PAO state.

## Results (20 C, pH 7)

| | SF end of anaerobic zone (g COD/m3) | PAO (g COD/m3) | Effluent PO4 (g P/m3) |
|---|---|---|---|
| ASM2d | 0.93 | 784 | 5.13 |
| plant-wide, VFA 75/25 | 0.36 | 1057 | 4.41 |
| plant-wide, all acetate | 0.36 | 1027 | 4.56 |

Seeded GAOs die out in both plant-wide cases (Competibacter -0.028 and -0.013
1/d, Alphaproteobacteria -0.027 and -0.034 1/d).
