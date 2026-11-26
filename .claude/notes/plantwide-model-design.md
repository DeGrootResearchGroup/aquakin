# A unified plant-wide model: design proposal (2026-11-23)

Status: **in progress.** The owner's decisions so far are under *Decisions
made*; the core level is drafted as
`examples/plantwide/models/plantwide_core.yaml` (tested in
`tests/integration/test_plantwide_core.py`). The other levels are proposals, and
the questions under *Open decisions* are the owner's.

## Purpose

aquakin currently simulates a plant with a separate literature model per unit
(ASM1 / ASM2d / ASM3 variants in the water line, ADM1 in the digester) and
converts the state at each boundary with an interface (`plant/interfaces.py`,
after Nopens et al., 2009). The interfaces work, but each one makes modelling
choices (how COD is partitioned, how nitrogen is drawn from a priority list of
pools, the pH the charge balance is evaluated at) that are not part of either
model. Everything that crosses a boundary passes through those choices.

Many plant-wide simulators instead use **one model in every unit**: a single
state set and process matrix, with the unit supplying only its conditions
(aeration, volume, temperature, solids separation). Biomass, nutrients and
minerals then carry through the whole plant without conversion, and COD, N, P
and charge balance across it by construction. That matters most for questions
that cross unit boundaries: phosphorus release and recovery in the sludge line,
struvite and calcite scaling, reject-water treatment, chemical P removal with
iron that ends up in the digester, and plant-wide pH and alkalinity.

This note proposes adding such a model **alongside** the existing ones. The
literature models and their interfaces stay: they carry the published
benchmarks (BSM1, BSM2) and remain the reference the unified model is checked
against.

## Principles

1. **Built from the primary literature.** Every process, rate expression,
   stoichiometric coefficient and default value is taken from a cited paper, or
   derived here and documented as our own choice. Each model file carries its
   sources in `references:`; each parameter whose default is our own choice says
   so in its `description` and gets a wide prior. The general scope of
   plant-wide simulators (which process groups such a model usually covers)
   informs which blocks to include; it is not a source of structure, names or
   values.
2. **One state basis.** Every species is in g/m³ on a single basis (COD for
   organics and biomass, N / P / inorganic C as the element, minerals as mass),
   including in the digester. No unit converts the state.
3. **Continuity by construction.** Every species declares `composition:` (COD,
   N, P, and charge from the chemistry level), and coefficients a balance
   determines are written as `auto`, so `check_conservation()` passes on every
   reaction and a change to a nutrient content reaches every balance.
4. **Each level is a complete, runnable model** that can be tested on its own
   against its source papers before the next level is added.
5. **Start small.** Processes whose effect is secondary (methanol utilizers,
   colloidal fractions, algae, dewatering water content, temperature as a
   state) are out of the first version, and are added later as further levels
   if a use case needs them.
6. **No unit-type switches.** Which organism groups are active in a unit
   follows from its conditions (oxygen, nitrate, temperature) alone; no process
   or parameter depends on whether the unit is a digester. BNRM1 (Seco et al.,
   2004) is the published precedent. The UCT plant-wide model (Ikumi, 2011)
   instead runs only the activated-sludge processes in activated-sludge units
   and only the digestion processes in the digester, and commercial models
   switch some parameter values in digesters.

## Decisions made (owner, 2026-11-23)

1. **Keep propionate** (`SPRO`, propionate oxidizers `XPRO`): propionate and
   acetate degraders set digester stability (Siegrist et al., 2002), and all
   three digestion sources keep it.
2. **Methanogen and propionate-oxidizer kinetics from ADM1** (BSM2 values at
   35 C, converted from kg to g), with the temperature dependence of Siegrist
   et al. (2002), so the BSM2 digester comparison stays clean.
3. **A separate group of fermenting organisms** (acidogens, `XFE`), not the
   heterotrophs with a reduced anaerobic decay.
4. **The BNRM1 structure for fermentation and hydrolysis** (Seco et al., 2004;
   Barat et al., 2013): acidogens do all the fermentation, in every unit, and
   ASM2d's heterotrophic fermentation is removed; heterotrophs hydrolyze under
   every condition (ASM2d), and without oxygen and nitrate the acidogens
   hydrolyze as well (Contois, Sötemann et al., 2005). Two findings led here.
   Hydrolysis catalyzed by heterotrophs and acidogens together, with ASM2d's
   half-saturation, ran about ten times too fast in a seeded batch digester.
   And keeping ASM2d's heterotrophic fermentation (half-saturation 4 g/m³)
   beside acidogens (500 g/m³) starved the acidogens out of a continuous
   digester (20 d, 35 C), taking the hydrogen and propionate pathways with
   them. With the BNRM1 structure every anaerobic group persists and 83% of the
   biodegradable COD fed leaves as methane.
5. **Branching:** model work continues on the session branch after the design
   note's PR merged.

## Decisions made (owner, 2026-11-25), from the BSM2 digester comparison

6. **Heterotroph anaerobic hydrolysis factor `etafe` = 0.15** (was ASM2d's 0.4),
   from the comparison below; within the full-scale fits (Insel et al., 2022;
   Ozyildiz et al., 2023).
7. **Acidogen kinetics deferred to `plantwide_bio_p`:** the digester cannot
   tell the candidates apart, so ADM1's sugar degraders stay until an
   anaerobic-zone VFA comparison against `asm2d` on an A2O flowsheet.

## Decisions made (owner, 2026-11-26)

8. **Bio-P populations: metabolic PAOs and separate GAOs.** The PAO is the
   TU Delft metabolic form (Meijer et al., 2001; as in the shipped
   `asm2d_tud`), with polyphosphate, PHA and glycogen; the GAO is that of
   López-Vázquez et al. (2009), which extends the same metabolic model, so
   the two share one framework. Each population keeps its own PHA and
   glycogen. Their competition for acetate and propionate uses the core's
   separate `SA` and `SPRO`, so the A2O comparison that settles the acidogen
   kinetics (decision 7) also tests the PAO / GAO balance. Considered and
   left for later: fermentative PAOs (sugar and amino-acid uptake; recent
   metabolic models, parameters not yet settled), a metabolic shift of PAOs
   to glycogen-based metabolism when polyphosphate runs out, and a separate
   denitrifying-PAO group.

### The BSM2 digester comparison (2026-11-25)

The BSM2 open-loop steady state's digester feed (aquakin's BSM2, the ASM1
`sludge_mix` output, 178.37 m3/d; through BSM2's interface it reproduces the
published ADM1 input to 0.2%) fed to the core in a `HeadspaceDigesterUnit` with
BSM2's dimensions (3400 m3 liquid, 300 m3 gas, 35 C, pH 7.26), solved 900 d to
steady state. ADM1 is the published BSM2 digester steady state. The feed maps
ASM1 to core species directly (XI + XP to XI, alkalinity to SIC), with total N
kept equal by putting the organic-N difference in ammonium.

| | ADM1 (BSM2) | etafe 0.4 | 0.3 | **0.15** | 0.1 |
|---|---|---|---|---|---|
| Methane (kg CH4/d) | 1065.3 | 1104.3 | 1088.9 | **1063.6** | 1054.6 |
| Particulate COD destroyed | 54.2% | 55.8% | 55.0% | **53.8%** | 53.3% |

- Acidogen half-saturation `K_S_su` 500 against 50 g/m3: methane +0.04%,
  destruction unchanged; only the residual SF moves (12.4 to 1.2 g/m3).
- The totals agree by different routes: the core treats 90% of decaying
  biomass as biodegradable (`fXI` 0.1) where BSM2's interface treats 68%
  (`frxs_adm`), so it makes 2.4 kg/m3 less inert COD (XI out 14.9 against
  ADM1's 17.2), and leaves 2.1 kg/m3 of XS unhydrolyzed (ADM1: 0.26 of
  composites and their fractions). At 0.15 the acidogens do about 75% of the
  hydrolysis.
- Propionate 17.6 g/m3 in both; acetate 130 against ADM1's 89; methane partial
  pressure 0.670 against 0.662 bar.
- Ammonium released (effluent minus feed) 1198 g N/m3 against ADM1's 1295
  (-7.5%). Effluent ammonium itself is not comparable: the core's inerts carry
  0.02 g N/g COD (ASM2d) against BSM2's 0.06, so the difference enters the feed
  as ammonium.
- Pinned by `tests/validation/test_plantwide_core_digester.py`.

9. **Two GAO groups**, as López-Vázquez et al. (2009): *Competibacter*
   (acetate; propionate uptake negligible) and *Alphaproteobacteria*-GAO
   (propionate preferred, acetate at half the rate). The core's separate `SA`
   and `SPRO` are what tell them apart, and the paper's Fig. 4 benchmark needs
   both.
10. **PHA pools by origin, constant stoichiometry.** The source's yields depend
    on each population's stored-PHA composition (aerobic: through the
    acetyl- and propionyl-CoA fractions λ, β) and on pH (anaerobic), but the
    engine evaluates stoichiometry once per solve. Each population therefore
    keeps two PHA pools, PHA made from acetate and PHA made from propionate,
    each with its Table 1 composition, so its aerobic yields are constants.
    pH effects linear in the transport-energy term α(pH) (P release, glycogen
    use) are exact, carried in the rates; the nonlinear GAO acetate-PHA yield
    is fixed at a reference pH. The error of both against a faithful
    implementation of the source is measured before the level is committed.
    *Measured (2026-11-26)* with a reference implementation of the source
    (scratch, numpy/scipy; the corrections below applied; PHA by type, yields
    from the current composition) on the source's own 36-condition grid
    (10/20/30 C, pH 6.0/7.0/7.5, four acetate:propionate ratios; reconstructed
    SBR: 6 h cycle, 2 h anaerobic, 3.5 h aerobic, 50% exchange, SRT 8 d, 5
    C-mmol/L VFA, 0.40 P-mmol/L, 50 d):
    - the reference reproduces the source's Fig. 6 outcome in 33 of 36
      conditions; the three others are at 30 C, a 7-12 point PAO excess at the
      60% dominance threshold, plausibly the unreported SBR operation; at 20 C
      the shares agree with Fig. 4 within a few points;
    - pools by origin with every pH-dependent amount in the rates (P release,
      glycogen use and PHA made per acid, as a rate times the coefficient,
      stoichiometry +1) and only the pool compositions fixed at pH 7: worst
      population-share deviation from the reference 3.1 points, outcome
      matching the reference in 35/36 and the paper in 34/36;
    - fixing the GAO acetate PHA yield at pH 7 instead costs up to 22 points
      at pH 6 and 7.5, so pH-dependent amounts go in the rates, not in fixed
      stoichiometry.
    *The generated model (`plantwide_bio_p`, 2026-11-26)* on the same grid,
    through its own `dCdt`: worst deviation from the reference 13.9 points,
    every condition within 5 points except propionate at pH 7.5 (14 at 30 C,
    6 at 20 C), outcome matching the reference in 34/36 and the paper in
    32/36. The pH 7.5 propionate gap is the COD-balanced PAO propionate
    stoichiometry above pH 7 (the source's PHA and glycogen lines there do not
    balance electrons, 0.56 e-/C-mol per pH unit; the model keeps the glycogen
    line and balances COD). Its PAO equals `asm2d_tud` to 1% in growth
    yields, except the anaerobic P release per acetate (see decision 13).
11. **Anoxic processes and decay come from elsewhere**: the source has neither.
    PAO anoxic processes from the TU Delft model (`asm2d_tud`); the GAOs get
    the same anoxic reduction factor, an assumption stated as such; PAOs and
    GAOs decay by the core's death-regeneration, releasing their storage
    pools.

12. **PAO and GAO endogenous loss: the source's maintenance, plus decay only
    without oxygen and nitrate.** López-Vázquez et al. (2009) and the TU Delft
    model use maintenance alone, and the source's yields were calibrated with
    it; the core decays every group everywhere. Decay switched off under O2
    and NO3 lets bio-P sludge die and digest in the digester, without counting
    aerobic endogenous loss twice. Rate revised (owner, 2026-11-26) from
    ASM2d's b_PAO 0.2 1/d to 0.02 1/d, the core's anaerobic-group rate: at 0.2
    it also ran in the SBR's anaerobic phase, where the source has no decay,
    and moved the PAO-GAO outcome by up to 48 points at 30 C (to within about
    1 point with no decay, about 3 at 0.02). On death the pools are released: polyphosphate to
    phosphate and PHA to acetate (ASM2d's lysis), glycogen to `XS`. (This
    replaces decision 11's "decay by the core's death-regeneration".)

13. **PAO P release per acetate: the source's value** (owner, 2026-11-26),
    `1/4 + 0.16 pH - 0.7985` P-mol/C-mol (0.57 at pH 7, 0.55 g P/g COD), so
    the PAO-GAO competition stays as the source calibrated it; the intercept
    is a parameter with bounds and a prior that reach the full-scale TU Delft
    value (0.36, as in `asm2d_tud`), to be calibrated per plant.

## What the bio-P and acidogen sources give (read 2026-11-26)

- **López-Vázquez et al. (2009)** models three populations, not two:
  *Accumulibacter* (PAO), *Competibacter* (GAO, acetate only: propionate
  uptake 0.01 C-mmol/(C-mmol h)) and *Alphaproteobacteria*-GAO (propionate at
  0.20, acetate at 0.10). Each stores its own PHA, split into PHB, PHV and
  PH2MV by carbon source (Table 1), and glycogen; the PAO also polyphosphate.
  Its aerobic yields follow from δ = 1.85 ATP/NADH (the same for all) and the
  stored PHA composition, as in Murnleitner et al. (1997). Units C-mmol; 19
  processes, 21 components. The main text gives the anaerobic uptake rates
  and PHA splits (Table 1), temperature coefficients (Table 2), the calibrated
  aerobic rates per carbon source (Table 4) and the shared half-saturations;
  the online supplement (Appendices 1-7, read) gives the 14-component,
  19-process matrix, the anaerobic stoichiometry as functions of pH and of
  the PHA type made (App. 2), the aerobic yields from δ, K1, K2 and the PHA
  composition (App. 3, 4; biomass CH1.84O0.5N0.19P0.015), the temperature-
  and pH-dependent rate constants (App. 5: piecewise in T, constant above
  20 C for uptake; GAO uptake times 10^-pH/(1.2e-8 + 10^-pH)), the rate
  expressions with Murnleitner storage-fraction switches (App. 6) and the
  half-saturations and maximum storage fractions (App. 7). It has no anoxic
  processes and no decay (maintenance only), and works in C-mmol / P-mmol. Its Fig. 4 (population shares after 50 d for 36 combinations of
  10/20/30 C, pH 6.0/7.0/7.5 and four acetate:propionate ratios) is a
  benchmark a reimplementation should reproduce.
- **Two misprints in López-Vázquez et al. (2009), Appendix 2 (PAO P release):**
  - per acetate it prints `1/2 + α`; Smolders et al. (1994a), the source it
    cites, derives `1/4 + α` for the glycogen model it uses (Eq. 5; `1/2 + α`
    is that paper's rejected TCA-cycle model) and fits `0.19 pH - 0.85`
    P-mol/C-mol (Eq. 8; 0.48 at pH 7, 0.50 measured). The paper's own Fig. 2b
    model curve releases about 0.4 per acetate, not the 0.82 the misprint
    gives. Use `1/4 + α`, with α(pH) = `0.16 pH - 0.7985` as the paper
    takes it from Filipe et al. (2001d, Eq. 8; their Eq. 1 confirms the
    `1/4 + α` form), so the release is `0.16 pH - 0.55` (0.57 at pH 7). The
    literature spread is wide and unexplained: Smolders et al. (1994a)
    `0.19 pH - 0.85` (0.48), Smolders et al. (1995a) 0.36, and Filipe's own
    SBR 0.55 against 0.64 predicted at pH 7.4. Keep the intercept a
    parameter (prior over 0.36-0.57 at pH 7) so a calibration can move it;
  - per propionate it prints `2/9 - α` (0.04); `2/9 + α` (0.40) joins its
    own pH > 7.5 branch `0.20 pH - 1.1` exactly at 7.5, and the paper's Fig.
    2d model curve releases about 0.42 per propionate. Use `2/9 + α`.
- **Two more in its aerobic stoichiometry (Appendix 1, checked by electron and
  P balances with O2 and CO2 implicit):**
  - the storage rows (glycogen and poly-P formation, every population) print
    the biomass entry as `-i` with `i = -Y_SX / Y_S,store`, a net `+`; only
    the negative sign conserves electrons (residual < 1e-15 for PHB and for
    propionate-derived PHA, against about 7 e-/C-mmol for glycogen as
    printed). Storage diverts PHA from growth, so biomass falls: use
    `-Y_SX / Y_S,store`;
  - the aerobic maintenance rows' O2 entry `Y_SX / Y_OX - 1` balances only
    for a substrate of 4 e-/C-mol (glycogen); for PHA (4.5 to 5) it is short
    by `redox_PHA / 4 - 1` O2 per C-mmol, 0.5 to 0.9 e-/C-mmol of
    maintenance. Use the balanced `Y_SX / Y_OX - redox_PHA / 4`.
- **Two parameter details**: the PAO anaerobic maintenance's temperature
  factor 1.096^(T-20) is not capped above 20 C (the uptake and aerobic rates
  are; capping it gave a PAO excess at 30 C); the PAO glycogen rate on
  propionate is 0.008 (Table 4), where Appendix 5 prints 0.077 under a
  mislabelled symbol.
- **GAO anaerobic maintenance**, undefined in López-Vázquez's appendices
  (its matrix names `Y_PHA,mANA`, `Y_GLY,mANA` without values), comes from
  Zeng et al. (2003a), Eq. 21: 1 C-mol glycogen -> 1/6 PHB + 5/12 PHV +
  1/4 PH2MV + 1/6 CO2 (carbon and electrons close), at a glycogen
  consumption of 4.7e-3 C-mol/(C-mol h) measured = 2 m_ATP with 0.5 ATP per
  C-mol glycogen and m_ATP = 2.35e-3. López-Vázquez writes the rate as
  `0.5 m_ATP`, a quarter of that; use the measured value.
- Zeng, R.J., van Loosdrecht, M.C.M., Yuan, Z. & Keller, J. (2003a).
  Metabolic model for glycogen-accumulating organisms in anaerobic/aerobic
  activated sludge systems. *Biotechnol. Bioeng.* 81(1), 92–105.
  https://doi.org/10.1002/bit.10455 : confirms the aerobic yield expressions
  (its Eqs. 8-10, as López-Vázquez's Appendix 3), maintenance consuming PHA
  directly (Eq. 7), and the GAO acetate stoichiometry at pH 7 (Eq. I).
- Smolders, G.J.F., van der Meij, J., van Loosdrecht, M.C.M. & Heijnen, J.J.
  (1994a). Model of the anaerobic metabolism of the biological phosphorus
  removal process: stoichiometry and pH influence. *Biotechnol. Bioeng.*
  43(6), 461–470: glycogen model per C-mol acetate, 1/2 C-mol glycogen used,
  4/3 C-mol PHB made, 1/6 C-mol CO2, `1/4 + α` P released.
- Filipe, C.D.M., Daigger, G.T. & Grady, C.P.L. Jr. (2001d). Stoichiometry
  and kinetics of acetate uptake under anaerobic conditions by an enriched
  culture of phosphorus-accumulating organisms at different pHs.
  *Biotechnol. Bioeng.* 76(1), 32–43: α_PAO = 0.16 pH - 0.7985 (pH 6.5-8.0),
  maximum acetate uptake 0.185 C-mmol/(C-mmol h) independent of pH,
  anaerobic maintenance 2.0e-3 mmol ATP/(C-mmol h), PHV 11% of PHA.
- **Meijer et al. (2001)** prints no matrix; it uses the TUDP model of van
  Veldhuizen et al. (1999), as `asm2d_tud` does, with no propionate uptake.
- **Varga et al. (2018)**: separate PAOs (storing PHA only) and GAOs
  (glycogen only), extending Barker-Dold; ORP estimated from DO, NOx and
  methane switches GAO storage off and PAO fermentation on at low ORP; no
  acetate / propionate distinction; equations only in a commercial
  simulator's file. A different family from López-Vázquez, not used.
- **Ribes et al. (2002)**, primary-settler fermentation at 20 C, fits a
  separate acidogen group (`X_ACID`) in activated sludge: growth 2.14 1/d,
  half-saturation 15 g COD/m3 on fermentable substrate, yield 0.15, decay
  0.33 1/d, acetate inhibition (6500 g COD/m3), ammonium and phosphate
  limitation and a pH switch; products 0.58 acetate, 0.25 propionate, 0.17
  hydrogen (as the core). It is the candidate for decision 7 to compare
  against ADM1's sugar degraders; its decay is 16 times ADM1's, so adopting it
  means re-running the BSM2 digester comparison.
- **Solon et al. (2017)**: sludge-line P on a BSM2-type plant (single PAO
  group, ADM1 extended for P, S and Fe, minerals by saturation index): 129 kg
  P/d into the digester, 22% returning as soluble phosphate, effluent TP 6.4
  g/m3 open loop; model outputs, usable as a later plausibility check.

## Who ferments in activated sludge (A2O comparison, 2026-11-26)

**Finding.** On `build_a2o` (anaerobic zone 23% of the reactor volume, ASM2d's
default influent with all VFA as acetate, 20 C), the bio-P level's acidogens
wash out under every literature parameter set. The invasion rate (largest
eigenvalue of the plant Jacobian's `XFE` block at the acidogen-free steady
state) is negative for ADM1's sugar degraders (-0.052 1/d), ADM1 with
`K_S_su` 50 (-0.019), Ribes et al. (2002) (-0.194), and even ADM1 with no
decay (-0.045). Only Ribes with no decay (+0.136) or with decay scaled to the
anaerobic share (+0.060) invades. Without fermentation the anaerobic zone
ends at 14.9 g/m3 `SF` (ASM2d 0.93); effluent phosphate is 7.1 g P/m3 against
ASM2d's 5.1, PAOs 522 against 774 g COD/m3.

**What the literature does.**
- Fermentation is OHO (heterotroph) activity in ASM2d (a transformation,
  `qfe` 3/d, `KF` 4), Barker & Dold (1997; anaerobic growth, 4 1/d, `K` 2,
  yield 0.1, off with O2 or NO3), UCTPHO+, and Ikumi (2011; activated sludge
  only, with separate acidogens in the digester behind a model interface).
  Hauduc et al. (2013) review seven models: "fermentation is a growth process
  under anaerobic conditions for OHOs", and anaerobic growth "may not be
  neglected" for large anaerobic zones.
- The commercial plant-wide model the owner supplied (the Varga et al. 2018
  lineage) has OHOs ferment by anaerobic growth (0.3 1/d, yield 0.1,
  half-saturation 5 g/m3 in activated sludge and 350 in digesters, chosen by
  unit type), hydrogen yield switching from 0.35 to 0.10 as VFA rises past
  50 g/m3, OHO decay 0.62 1/d aerobic, x0.5 anoxic, x0.25 anaerobic; and
  PAOs ferment too, switched by ORP (Tetrasphaera-type).
- BNRM1 (Seco et al., 2004) is the one with separate acidogens in activated
  sludge: fermentation inhibited by O2 only (it runs in anoxic zones),
  lysis unswitched; and in practice the acidogens are seeded with the influent
  (Ruano et al., 2012, p. 1011: "Because of the presence of acidogenic
  bacteria in the sewerage system, the simulations were carried out by
  considering these bacteria in the influent").
- Ruel et al. (2002) fit "heterotrophic acidogenic biomass" in 13 raw
  wastewaters at 20 C: 3 to 20 g COD/m3; growth 0.70 1/d, yield 0.10,
  half-saturation 8.5 g/m3, decay 0.20 1/d; products 0.45 to 0.60 acetate,
  0.27 to 0.49 propionate.
- Nielsen et al. (2012): glucose fermenters are 39% of the biovolume in ten
  full-scale EBPR plants, mostly Tetrasphaera (33%); "most seem to be
  facultative fermenters", and some take up glucose aerobically as well.

**Measured on the A2O** (steady states by PTC; "not conv." did not converge
and is indicative only):

| Variant | `SF` end of anaerobic zone | PAO | Competibacter | Effluent PO4 |
|---|---|---|---|---|
| ASM2d | 0.93 | 774 | - | 5.13 |
| acidogens, ADM1 kinetics (current) | 14.9 | 522 | 0 | 7.08 |
| heterotrophs ferment, 3 1/d substrate, `K` 5 (not conv.) | 0.48 | 1022 | 0 | 4.67 |
| BNRM1-type: fermentation off by O2 only, Ruel kinetics, 10 g/m3 acidogens in the influent | 5.4 | 806 | 0 | 5.82 |
| the same without influent acidogens | 13.9 | 0 | 552 | 9.69 |

**The PAO / GAO columns above are not steady states.** Two 400-day forward
solves of the current model, from warm starts differing only in the seeded
acidogens (50 and 0 g/m3), follow the same path within 2 g/m3: PAOs fall from
500 to 465 g COD/m3 between days 100 and 400 while Competibacter grows from 29
to 67 (about 0.3% per day). A PTC solve from the day-400 state converges
(residual 8e-7) to PAOs 0, Competibacter 560, effluent phosphate 9.7 g P/m3,
and every eigenvalue of the plant Jacobian there is negative (largest -0.049
1/d): Competibacter excludes the PAOs. The PAO-only states the earlier PTC
solves stopped at are equilibria (zero GAOs is invariant) that GAOs invade too
slowly for the convergence test to see. So on this level a PTC steady state
needs a stability check (the Jacobian's eigenvalues, or a long forward solve)
before its PAO / GAO split is read, and the acidogen variants above need the
same check. A full GAO takeover at 20 C on acetate at pH 7 is not what the
source predicts for its SBR (coexistence, 53% PAO and 46% Competibacter, the
validation case), so the cause in the A2O (the anoxic zone and the GAOs'
anoxic processes, carbon limitation, the recycles) is to be found before the
fermentation structure is chosen.

## Structure: a linear `extends:` chain

```
plantwide_core       heterotrophs, one-step nitrification, hydrolysis,
                     fermentation, methanogenesis; COD and N
 └ plantwide_bio_p   + phosphate, polyphosphate-accumulating organisms, P contents
    └ plantwide_nitrite  + nitrite as an intermediate (AOB / NOB), two-step
                         denitrification, anammox
       └ plantwide_chem  + pH by speciation, cations, mineral precipitation,
                           iron-based chemical P removal, gas stripping
```

Model names are placeholders. The chain uses only what `schema/inheritance.py`
already supports: new species and reactions are appended, an existing reaction
changes only the fields a level gives, and `remove:` drops what a level
replaces (`plantwide_nitrite` removes the one-step nitrifier and `SNO3`'s role
as the only oxidized nitrogen).

**Why not several bases, or a new composition mechanism.** The blocks are not
additive: adding phosphate puts a P coefficient and a P-limitation term into
every existing growth process; adding nitrite adds new denitrification
reactions and changes the existing ones; adding pH puts an inhibition term on
every organism group. A module would have to edit the others, so merging
several bases would make the result depend on merge order and need a resolver
for shared species. A single chain keeps each level's change reviewable as one
concept and needs no format change.

### Cross-cutting terms as named expressions

The core defines the factors that later levels widen, once, in `expressions:`,
and every rate refers to them by name. A level then overrides one expression
instead of editing every rate. Proposed names:

| Expression | Core definition | Widened by |
|---|---|---|
| `nutrient_limit` | ammonium saturation | `plantwide_bio_p`: × phosphate saturation |
| `aerobic`, `anoxic`, `anaerobic` | O₂ / NO₃ switching | `plantwide_nitrite`: NO₂ + NO₃ |
| `ph_inhibition_<group>` | `1` | `plantwide_chem`: pH inhibition from the speciated pH |
| `nh3_inhibition_<group>` | from total ammonium | `plantwide_chem`: from free NH₃ |

### Closure species (deferred format decision)

Even with `auto`, adding phosphate means writing `SPO4: auto` into every
existing reaction's stoichiometry. A model-level declaration of which species
closes each balance (`closure: {N: SNH4, P: SPO4}`) would make that one line. It
is a model-format change, so it is **not** proposed yet: write
`plantwide_bio_p` by hand first, count how many of its edits are this
bookkeeping alone, and decide from that.

*Counted (2026-11-26):* `plantwide_bio_p` overrides all 16 core reactions
only to add `SPO4: auto` (10 of them also extend `conserved_for`) and 10 core
species only to add a P content; its 67 new reactions carry 201 `auto`
entries. A generator writes all of it from the core file, so the bookkeeping
costs nothing at this level. Recommendation: keep deferring the closure
declaration until a hand-written level (`plantwide_nitrite`, which adds
`SNO2` to every N-carrying reaction) shows the cost, and decide then.

## Proposed state set

Names follow ASM2d where ASM2d has the concept, so the models read naturally to
anyone who knows the IWA family.

**Core**

| Group | Species |
|---|---|
| Organics | `SF` fermentable substrate, `SA` acetate, `SPRO` propionate, `SI` soluble inert, `XS` slowly biodegradable, `XI` particulate inert |
| Biomass | `XH` heterotrophs, `XAUT` nitrifiers, `XFE` acidogens, `XPRO` propionate oxidizers, `XAC` acetoclastic methanogens, `XH2M` hydrogenotrophic methanogens |
| Dissolved | `SO2`, `SNH4`, `SNO3`, `SN2`, `SH2`, `SCH4`, `SIC` (total inorganic carbon, g C/m³) |
| Not yet | inorganic suspended solids (for TSS) and the gas phase (see *Engine work*) |

**`plantwide_bio_p` adds** `SPO4`, `XPAO`, `XPP`, `XPHA`, `XGLY` (the PAO's
storage pools) and `XGAO`, `XPHA_GAO`, `XGLY_GAO` (decision 8).

**`plantwide_nitrite` replaces** `XAUT` with `XAOB`, `XNOB` and adds `SNO2`,
`XAMX`.

**`plantwide_chem` adds** `SCa`, `SMg`, `SK`, lumped strong cations / anions,
`SFe3` / `SFe2` and the iron and mineral solids (calcite, struvite, a calcium
phosphate, iron phosphate / hydroxide), plus `G_NH3`.

## Processes and sources by level

### `plantwide_core`

| Processes | Primary source |
|---|---|
| Hydrolysis of `XS` by heterotrophs (aerobic, anoxic, anaerobic), heterotroph growth on `SF` and `SA` (O₂, NO₃), nitrification | ASM2d (Henze et al., 1999) |
| Decay of every group by death-regeneration, at the same rate in every zone (heterotrophs and nitrifiers with ASM2d's `fXI` = 0.1) | ASM2d; Barker & Dold (1997) use the same form with 0.08, the value Ikumi et al. (2014) found consistent with digestion |
| Fermentation by acidogens only, in every unit; hydrolysis by acidogens without O₂ and NO₃ | BNRM1 structure (Seco et al., 2004; Barat et al., 2013); Contois hydrolysis constants of Sötemann et al. (2005) |
| Fermentation products: acetate, propionate, hydrogen | split of Siegrist et al. (2002), Table 2 |
| Propionate oxidation, acetoclastic and hydrogenotrophic methanogenesis, their H₂ and NH₃ inhibition | ADM1 (Batstone et al., 2002; BSM2 values), temperature dependence of Siegrist et al. (2002) |
| Carbon contents of the organic components | Siegrist et al. (2002), Table 2 |

**Reconciliation choices this level makes:**
- ADM1's carbohydrate, protein and lipid pools and its sugar, amino acid, LCFA,
  butyrate and valerate degraders collapse into `XS` → `SF` →
  `SA` + `SPRO` + `SH2`; butyrate and valerate are dropped as minor (Siegrist et
  al., 2002; Sötemann et al., 2005).
- Anaerobic growth stops under oxygen and nitrate through inhibition terms with
  ASM2d's anaerobic-switch constants. No source gives these for this kind of
  model, so the parameters carry wide priors and say so.
- Without ASM2d's heterotrophic fermentation, anaerobic zones differ from
  `asm2d`; the equivalence with `asm2d` holds under aerobic and anoxic
  conditions (and against `asm2d` with its fermentation switched off).
- The acidogen kinetics are ADM1's sugar degraders (half-saturation 500 g/m³)
  for now, provisional until a source tested in activated sludge is found;
  the digester cannot tell the candidates apart (decision 7).
- Particulate organics the activated-sludge models treat as unbiodegradable
  stay largely unbiodegradable in a digester (Ikumi et al., 2014), so `XI` is
  inert in every unit and the first version has no anaerobic conversion of
  inert material.

### `plantwide_bio_p`

| Processes | Primary source |
|---|---|
| PAO: anaerobic VFA uptake into PHA with P release and glycogen use, polyphosphate and glycogen storage, growth on PHA (aerobic, anoxic), maintenance, decay of the cell and its pools | TU Delft metabolic model (Meijer et al., 2001), as in the shipped `asm2d_tud` |
| GAO: anaerobic VFA uptake into PHA from glycogen, glycogen storage and growth on PHA (aerobic, anoxic), maintenance, decay; acetate / propionate, temperature and pH effects on the competition | López-Vázquez et al. (2009) |
| P release in the sludge line | checked against Solon et al. (2017) |

### `plantwide_nitrite`

| Processes | Primary source |
|---|---|
| AOB and NOB growth, two-step denitrification on NO₂ and NO₃ | Kaelin et al. (2009); Hiatt & Grady (2008) |
| Anammox growth and decay | Strous et al. (1998, 1999) |

The repository's `asm3_2step` and `asm3_2step_anammox` models already carry
these sources and kinetics; the level reuses their parameter values where the
process is the same.

### `plantwide_chem`

| Processes | Primary source |
|---|---|
| pH and ion speciation, activity correction | Batstone et al. (2012); Flores-Alsina et al. (2015); the existing `speciation:` block |
| Mineral precipitation (calcite, struvite, calcium phosphate) | Kazadi Mbamba et al. (2015a, 2015b); the existing `precipitation:` block |
| Iron-based chemical P removal | either the saturation-index approach already in `asm2d_chemp`, or surface complexation (Smith et al., 2008; Hauduc et al., 2015) — open decision |
| CO₂ and NH₃ stripping | standard two-film gas transfer |

## Gas phase (owner, 2026-11-24)

The model carries no gas phase; the unit owns it (the method of Lizarralde et
al., 2015: the biochemical model is unchanged in the aqueous phase, and each
gas phase has its own balance in the unit).

- **The model declares its dissolved gases** in a `gases:` block, mapping each
  species onto a gas of a built-in library (`core/gases.py`) with the
  `molar_mass` that converts it to mol/L, and, for inorganic carbon, the
  `volatile_fraction` that transfers (H2CO3*). Any library value can be
  overridden per entry.
- **Library values.** Henry's-law solubility at 25 C and its temperature
  dependence: the first evaluated-review entries of Sander (2023), Burkholder
  et al. (2019) for O2, N2, CO2, CH4 and Fernandez-Prini et al. (2003) for H2.
  For CH4, CO2 and H2 these agree with BSM2's within a few percent. Diffusivity
  in water at 25 C: CRC Handbook, "Diffusion of gases in water". Transfer
  coefficient relative to oxygen: `(D / D_O2) ** 0.5`; CO2 gives 0.888, within
  the 0.88-0.92 Lizarralde et al. measured. NH3 is left for the chemistry
  level: its transfer is partly gas-film limited.
- **Units supply the gas side.** An aerated tank transfers every declared gas at
  its ratio times kLa_O2 toward its saturation in air at the tank temperature;
  an unaerated tank optionally by a surface kLa; the digester owns one
  headspace state per declared gas, held as a concentration in the gas volume
  as in BSM2, with one kLa for all gases and the BSM2 pressure-driven outflow.
- **Order:** the `gases:` block and library (done), the digester headspace
  (`HeadspaceDigesterUnit`, done: equal to ADM1's BSM2 headspace when given
  BSM2's constants; the plant-wide core's 20 d digester vents 64% CH4 / 36%
  CO2 and closes its COD and N balances; `plant.digester_gas` reports it,
  through `BiogasHeadspaceUnit`), then aeration
  (`Aeration(stripped="model")`, done: dry air at one atmosphere, `air=` to
  change it; oxygen keeps `do_sat`; the surface kLa of an unaerated tank is
  an `Aeration(kla=...)`, zero unless given).

## Engine and plant work the model needs

1. **Gas-phase species.** The digester unit defaults to ADM1's headspace
   species (`ADM1_GAS_SPECIES` in `plant/digester.py`) but takes them as its
   `gas_species` argument, so a model with its own headspace states can run in
   it. Whether a model should declare its gas-phase species itself is a
   model-format question (owner's decision). The core level has no gas phase:
   methane, hydrogen and CO₂ stay dissolved.
2. **Gas transfer in aerated units** for species other than O₂ (CO₂ and N₂
   stripping, NH₃ at the chemistry level): to check what the aeration units
   support today.
3. **The digester unit's assumptions about units and species names** (ADM1's
   kg/m³ basis): to check, so a g/m³ model runs in it unchanged.
4. **Performance.** The unified model is larger in every unit than the model it
   replaces there. Measure the right-hand side, the coloured Jacobian and a
   BSM2-sized dynamic and steady solve at each level before optimizing
   anything.

## Validation plan

- **Per reaction:** `check_conservation()` for COD, N and P (and charge at the
  chemistry level) passes on every reaction, as a unit test.
- **Per level, against its sources:**
  - core aerobic / anoxic reaction terms against `asm2d` with fermentation
    switched off (done: equal to 1e-6), then on a BSM1-type flowsheet;
  - core digestion against `adm1` on the BSM2 digester feed: methane
    production, VS destruction, effluent ammonium;
  - bio-P against `asm2d` / `asm2d_tud` on an A2O flowsheet;
  - nitrite and anammox against `asm3_2step` / `asm3_2step_anammox` batch
    cases;
  - pH and precipitation against the existing speciation and precipitation
    models' validation cases.
- **Plant-wide study:** BSM2 simulated twice — ASM1 / ADM1 with interfaces, and
  the unified model — comparing effluent quality, sludge production, biogas,
  the COD / N / P error the interfaces introduce, and the computational cost.
  The Fisher and profile tools then show which of the unified model's
  parameters plant data can identify.

## Open decisions (the owner's)

1. **Bio-P populations.** Settled: metabolic PAOs and separate GAOs
   (decision 8).
2. **Acidogen kinetics** (deferred to `plantwide_bio_p`, decision 7). BNRM2's parameter sources (Barat et al., 2013,
   appendix p. x) name none for the anaerobic groups; Ribes et al. (2002, primary
   settler fermentation) is the likeliest source for acidogens tested in
   activated sludge. Candidates meanwhile: ADM1's sugar degraders (in use) or
   Siegrist et al.'s fermenters (half-saturation 50 g/m³). Both sustain the
   continuous digester; they differ in the VFA an anaerobic zone produces,
   which matters at the bio-P level.
3. **Heterotroph anaerobic hydrolysis factor.** Settled: 0.15 (decision 6).
4. **Iron chemistry.** Saturation-index precipitation (already in the
   repository) or surface complexation. Recommendation: start with the
   existing approach; add surface complexation as its own level if aged-iron
   and P-desorption behaviour is needed.
5. **Gas-phase species in the model format** (*Engine work* item 1).
6. **The closure declaration**, after `plantwide_bio_p` is written.
7. **Shipped or example.** Recommendation: keep the levels out of the shipped
   catalogue until the plant-wide study is done, then ship them with a catalogue
   entry.

## References

DOIs were found from publisher, PubMed and repository records; three entries
have no DOI that could be confirmed and are marked. Check each against the
paper itself before citing it in a model file.

- Barat, R. et al. (2013). Biological Nutrient Removal Model No. 2 (BNRM2): a
  general model for wastewater treatment plants. *Water Sci. Technol.* 67(7).
  https://doi.org/10.2166/wst.2013.004
- Barker, P.S. & Dold, P.L. (1997). General model for biological nutrient
  removal activated-sludge systems: model presentation. *Water Environ. Res.*
  69(5), 969–984. DOI not confirmed; the companion paper (model application,
  985–991) is https://doi.org/10.2175/106143097X125678.
- Batstone, D.J. et al. (2002). *Anaerobic Digestion Model No. 1 (ADM1).* IWA
  Scientific and Technical Report No. 13. ISBN 1 900222 78 7; no DOI
  confirmed.
- Batstone, D.J. et al. (2012). Towards a generalized physicochemical framework.
  *Water Sci. Technol.* 66(6), 1147–1161.
  https://doi.org/10.2166/wst.2012.300
- Flores-Alsina, X. et al. (2015). A plant-wide aqueous phase chemistry module
  describing pH variations and ion speciation/pairing in wastewater treatment
  process models. *Water Res.* 85, 255–265.
  https://doi.org/10.1016/j.watres.2015.07.014
- Hauduc, H. et al. (2015). A dynamic physicochemical model for chemical
  phosphorus removal. *Water Res.* 73, 157–170.
  https://doi.org/10.1016/j.watres.2014.12.053
- Henze, M. et al. (1999). Activated Sludge Model No. 2d, ASM2d. *Water Sci.
  Technol.* 39(1), 165–182. https://doi.org/10.1016/S0273-1223(98)00829-4
- Hiatt, W.C. & Grady, C.P.L. (2008). An updated process model for carbon
  oxidation, nitrification, and denitrification. *Water Environ. Res.* 80(11),
  2145–2156. https://doi.org/10.2175/106143008X304776
- Ikumi, D.S. (2011). *The development of a three phase plant-wide
  mathematical model for sewage treatment.* PhD thesis, University of Cape
  Town.
- Ikumi, D.S., Harding, T.H. & Ekama, G.A. (2014). Biodegradability of
  wastewater and activated sludge organics in anaerobic digestion. *Water Res.*
  56, 267–279. https://doi.org/10.1016/j.watres.2014.02.008
- Insel, G. et al. (2022). A comprehensive evaluation of process kinetics: a
  plant-wide approach for nutrient removal and biogas production. *Water Res.*
  118410. https://doi.org/10.1016/j.watres.2022.118410
- Kaelin, D. et al. (2009). Extension of ASM3 for two-step nitrification and
  denitrification and its calibration and validation with batch tests and
  pilot scale data. *Water Res.* 43(6), 1680–1692.
  https://doi.org/10.1016/j.watres.2008.12.039
- Kazadi Mbamba, C., Batstone, D.J., Flores-Alsina, X. & Tait, S. (2015a). A
  `generalised` chemical precipitation modelling approach in wastewater treatment
  applied to calcite. *Water Res.* 68, 342–353.
  https://doi.org/10.1016/j.watres.2014.10.011
- Kazadi Mbamba, C., Tait, S., Flores-Alsina, X. & Batstone, D.J. (2015b). A
  systematic study of multiple minerals precipitation modelling in wastewater
  treatment. *Water Res.* 85, 359–370.
  https://doi.org/10.1016/j.watres.2015.08.041
- López-Vázquez, C.M. et al. (2009). Modeling the PAO–GAO competition: effects
  of carbon source, pH and temperature. *Water Res.* 43(2), 450–462.
  https://doi.org/10.1016/j.watres.2008.10.032
- Lizarralde, I. et al. (2015). A new general methodology for incorporating
  physico-chemical transformations into multi-phase wastewater treatment
  process models. *Water Res.* 74, 239–256.
  https://doi.org/10.1016/j.watres.2015.01.031
- Meijer, S.C.F., van Loosdrecht, M.C.M. & Heijnen, J.J. (2001). Metabolic
  modelling of full-scale biological nitrogen and phosphorus removing WWTP's.
  *Water Res.* 35(11), 2711–2723.
  https://doi.org/10.1016/S0043-1354(00)00567-4
- Nopens, I. et al. (2009). An ASM/ADM model interface for dynamic plant-wide
  simulation. *Water Res.* 43(7), 1913–1923.
  https://doi.org/10.1016/j.watres.2009.01.012
- Ozyildiz, G. et al. (2023). Restructuring anaerobic hydrolysis kinetics in
  plant-wide models for accurate prediction of biogas production. *Water Res.*
  120620. https://doi.org/10.1016/j.watres.2023.120620
- Sander, R. (2023). Compilation of Henry's law constants (version 5.0.0) for
  water as solvent. *Atmos. Chem. Phys.* 23, 10901–12440.
  https://doi.org/10.5194/acp-23-10901-2023
- Seco, A., Ribes, J., Serralta, J. & Ferrer, J. (2004). Biological nutrient
  removal model No.1 (BNRM1). *Water Sci. Technol.* 50(6), 69–78.
  https://doi.org/10.2166/wst.2004.0361
- Siegrist, H. et al. (2002). Mathematical model for meso- and thermophilic
  anaerobic sewage sludge digestion. *Environ. Sci. Technol.* 36(5), 1113–1123.
  https://doi.org/10.1021/es010139p
- Smith, S. et al. (2008). Phosphate complexation model and its implications
  for chemical phosphorus removal. *Water Environ. Res.* 80(5), 428–438.
  https://doi.org/10.1002/j.1554-7531.2008.tb00349.x
- Solon, K. et al. (2017). Plant-wide modelling of phosphorus transformations in
  wastewater treatment systems: impacts of control and operational strategies.
  *Water Res.* 113, 97–110. https://doi.org/10.1016/j.watres.2017.02.007
- Sötemann, S.W. et al. (2005). Integrated chemical, physical and biological
  processes modelling. Part 2: Anaerobic digestion of sewage sludges. *Water SA*
  31(4), 545–568. No DOI confirmed; open access at
  https://ajol.info/index.php/wsa/article/view/5145
- Strous, M., Heijnen, J.J., Kuenen, J.G. & Jetten, M.S.M. (1998). The
  sequencing batch reactor as a powerful tool for the study of slowly growing
  anaerobic ammonium-oxidizing microorganisms. *Appl. Microbiol. Biotechnol.*
  50(5), 589–596. https://doi.org/10.1007/s002530051340
- Strous, M., Kuenen, J.G. & Jetten, M.S.M. (1999). Key physiology of anaerobic
  ammonium oxidation. *Appl. Environ. Microbiol.* 65(7), 3248–3250.
  https://doi.org/10.1128/aem.65.7.3248-3250.1999
- Varga, E., Hauduc, H. et al. (2018). Recent advances in bio-P modelling – a
  new approach verified by full-scale observations. *Water Sci. Technol.*
  78(10), 2119–2130. Separate PAO and GAO populations (see *What the bio-P
  and acidogen sources give*), not one switching population as this note
  said earlier.
- Ribes, J., Ferrer, J., Bouzas, A. & Seco, A. (2002). Modelling of an
  activated primary settling tank including the fermentation process and VFA
  elutriation. *Environ. Technol.* 23(10), 1147–1156. No DOI confirmed.
- van Veldhuizen, H.M., van Loosdrecht, M.C.M. & Heijnen, J.J. (1999).
  Modelling biological phosphorus and nitrogen removal in a full scale
  activated sludge process. *Water Res.* 33(16), 3459–3468.
  https://doi.org/10.1016/S0043-1354(99)00064-0
  https://doi.org/10.2166/wst.2018.490 (also presented at WRRmod 2018).
