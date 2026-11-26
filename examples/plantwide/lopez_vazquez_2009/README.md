# Reference implementation: López-Vázquez et al. (2009)

A plain numpy/scipy implementation of the PAO–GAO competition model of
López-Vázquez et al. (2009, *Water Res.* 43(2), 450–462,
doi:10.1016/j.watres.2008.10.032): *Accumulibacter* (PAO), *Competibacter* and
*Alphaproteobacteria*-GAO, competing for acetate and propionate in an
anaerobic/aerobic sequencing batch reactor. It is the yardstick for the
plant-wide model's bio-P level (`plantwide_bio_p`), which simplifies the
source to fit the model format; it is not part of the package.

Units are C-mmol/L, P-mmol/L and hours, as in the source.

## Files

| File | What it does |
|---|---|
| `stoich.py` | Aerobic yields (Appendices 3 and 4) and the electron and phosphorus balances of each aerobic process, which settle the signs of the storage processes. `python stoich.py` prints the balances. |
| `model.py` | The model as published, with PHA tracked by type (PHB, PHV, PH2MV) per population and the aerobic yields recomputed from the current PHA composition; and the reconstructed SBR. |
| `grid.py` | Runs the source's 36 conditions (10/20/30 °C × pH 6.0/7.0/7.5 × four acetate:propionate ratios, 50 days) and compares the dominant organism with the source's Fig. 6. Writes `grid.json`. |
| `aquakin_grid.py` | Runs the generated `plantwide_bio_p` model (through its own `dCdt`, in g COD and g P) in the same SBR over the grid, against `grid.json` and Fig. 6. About 25 minutes. |
| `origin.py` | The simplified form the model format allows: two PHA pools per population by origin (from acetate, from propionate), each with a fixed composition at pH 7, and every pH-dependent amount carried in a rate. Compares it with `grid.json`. |

Run from this directory: `python grid.py`, then `python origin.py` (about
3 minutes each).

## Corrections to the source

The source's appendices are applied with these corrections, each checked
against the cited primary source or by a conservation balance:

1. PAO phosphate release per acetate is `1/4 + α`, not `1/2 + α`
   (Smolders et al. 1994, Eq. 5; Filipe et al. 2001, Eq. 1); α(pH) =
   0.16 pH − 0.7985 (Filipe et al. 2001, Eq. 8).
2. PAO phosphate release per propionate is `2/9 + α`, not `2/9 − α` (it then
   joins the pH > 7.5 branch exactly, and matches the source's Fig. 2d).
3. Glycogen and polyphosphate formation reduce biomass (`−Y_SX/Y_S,store`):
   only that sign conserves electrons.
4. Aerobic maintenance uses the electron-balanced oxygen demand,
   `Y_SX/Y_OX − redox_PHA/4`, not `Y_SX/Y_OX − 1`.
5. GAO anaerobic maintenance converts glycogen to 1/6 PHB + 5/12 PHV +
   1/4 PH2MV + 1/6 CO2 at 4.7e-3 C-mmol/(C-mmol h) (Zeng et al. 2003a,
   Eq. 21 and its measured rate); the source leaves these undefined.
6. The PAO anaerobic maintenance's factor 1.096^(T−20) is not capped above
   20 °C; the PAO glycogen rate on propionate is 0.008 (Table 4).

The source does not report its SBR operation; this one uses a 6 h cycle
(2 h anaerobic, 3.5 h aerobic, 0.5 h settle), 50% exchange, an 8-day sludge
age and the source's feed (5 C-mmol/L VFA, 0.40 P-mmol/L).

## Results

- Faithful model against the source's Fig. 6: the same dominant organism in
  33 of 36 conditions; the three others are at 30 °C, with PAO 7–12 points
  higher than the source at the 60% dominance threshold.
- Simplified form against the faithful model: worst population-share
  deviation 3.1 points; the same outcome in 35 of 36 conditions. Fixing the
  GAO acetate PHA yield at pH 7 instead of carrying it in the rate costs up
  to 22 points at pH 6 and 7.5.
- The generated model, `plantwide_bio_p`, against the faithful model: worst
  deviation 13.9 points, every condition within 5 points except propionate
  at pH 7.5 (14 at 30 °C, 6 at 20 °C), where the model's PAO propionate
  stoichiometry above pH 7 is COD-balanced and the source's is not; the same
  outcome in 34 of 36 conditions, and as the source's Fig. 6 in 32.
