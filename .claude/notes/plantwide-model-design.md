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
    through its own `dCdt`: worst deviation from the reference 13.8 points,
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
14. **Storage switches on fractions of the biomass** (owner, 2026-11-26): the
    depletion switches on glycogen, poly-P and PHA act on the stored fractions
    (`K_fGLY`, `K_fPP`, `K_fPHA_store`, each the source's 0.01 C-mmol/L at its
    30 C-mmol/L starting biomass), as ASM2d writes `XPP/XPAO`, not on absolute
    concentrations as the source does. A population's rates then scale with
    its size, so the PAO / GAO outcome in a plant does not depend on its
    history (see *Why GAOs win on the A2O*). The SBR validation is unchanged
    within its tolerance.
15. **The A2O comparison's base influent VFA is 75% acetate, 25% propionate**
    (owner, 2026-11-26), typical of sewage and clear of the PAO / GAO
    boundary; all-acetate (ASM2d's default influent) is a sensitivity case.
16. **The heterotrophs ferment** (owner, 2026-11-26; replaces decisions 3, 4
    and 7): fermentation is heterotroph growth on `SF` without oxygen or
    nitrate, in every unit (Barker and Dold 1997; Hauduc et al. 2013), with
    the acetate / propionate / hydrogen split kept, uptake 3 1/d at 20 C
    (theta 1.04), half-saturation 5 g/m3, yield 0.1 (provisional: the owner
    keeps these values until a paper source is found, 2026-11-27); the acidogens `XFE` and their Contois
    hydrolysis are removed, so hydrolysis is the heterotrophs' alone; and
    heterotroph decay without oxygen or nitrate is 0.25 of the aerobic rate
    (provisional likewise), which lets them persist in a digester on fermentation
    growth. Decision 4's reason (heterotroph fermentation starving the
    acidogens out of the digester) does not apply without acidogens. `etafe`
    is refitted to the BSM2 digester: 0.144 (methane +0.1%, particulate COD
    destroyed 53.9% against ADM1's 54.2%). The digester's hydrolysis is then
    saturated in `XS/XH` (2545 against 2038 g/m3, with `KX` 0.1), a capacity
    proportional to `etafe`, so methane is steep in it: 0.12 gives -15.5%,
    0.135 -4.7%, 0.155 +3.9%, 0.2 +8.0%.

17. **`plantwide_nitrite`'s nitrifier and anammox kinetics come from papers**
    (owner, 2026-11-27, revised the same day after the NOB literature):
    AOB and NOB grow on total ammonium and total nitrite, with free ammonia
    and free nitrous acid as inhibitors only (Hiatt & Grady 2008's constants,
    Anthonisen et al. 1976's equilibria); oxygen half-saturations are apparent
    floc values, AOB 0.65 and NOB 0.23 (Daebel et al. 2007), since a model
    without diffusion takes apparent values (Picioreanu et al. 2016); growth
    theta 1.11 for NOB (Mehrani et al. 2021). AOB growth and decay are the
    WERF pair, 0.9 and 0.17 1/d with thetas 1.072 and 1.029 (Jones et al.
    2005; Friedrich et al. 2017), ammonium half-saturation 0.6 (Mannucci et
    al. 2020); NOB decay 0.089 1/d, theta 1.036 (Friedrich et al.). Anammox: Hao et
    al. (2002) with Arrhenius at 70 kJ/mol (Strous et al. 1999) and Strous et
    al.'s (1998) stoichiometry. No NOB mixotrophy.
18. **Heterotrophs denitrify in two steps at ASMN's step capacities** (owner,
    2026-11-27): NO3 to NO2 and NO2 to N2 run side by side, neither inhibiting
    the other, each on its own acceptor. ASMN's anoxic growth factors are 0.28
    (NO3 to NO2) and 0.16, 0.35, 0.35 (NO2 to NO to N2O to N2, one electron
    each); NO2 to N2 is lumped at the slowest step's capacity, 3 x 0.16 =
    0.48. Each inherited process keeps its own anoxic factor (ASM2d's 0.8 for
    the heterotrophs) and takes 0.28 / 0.8 and 0.48 / 0.8 of it. The
    four-step form is recorded in *The nitrite level* for the N2O level.
19. **PAOs and both GAO groups denitrify in two steps too** (owner,
    2026-11-27), with the same fractions of their own anoxic factors.
20. **The level is generated** (owner, 2026-11-27):
    `_make_plantwide_nitrite.py` merges the core and the bio-P level and
    writes `plantwide_nitrite.yaml`; the closure declaration stays deferred.

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
ASM2d's 5.1, PAOs 522 against 784 g COD/m3.

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
| ASM2d | 0.93 | 784 | - | 5.13 |
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

**Why GAOs win on the A2O (2026-11-26).** Two separate effects.

1. *The source's storage switches are on absolute concentrations* (as the
   level first had them; decision 14 replaced them): anaerobic uptake carried
   a Monod switch on the glycogen concentration (and the PAOs' on poly-P),
   aerobic glycogen storage one on PHA, with K = 0.31 to 0.36 g/m3 (the
   source's 0.01 C-mmol/L,
   the same form in `lopez_vazquez_2009/model.py`). Near 1 for a population of
   hundreds of g/m3, they throttle a rare one, so the outcome depends on
   history: a Competibacter seed of 1 g/m3 (glycogen 0.3) in the PAO-only
   steady state dies out (-0.077 1/d over days 60 to 120), a seed of 30 grows
   (+0.0023). With the three K at 1e-3 both seeds grow at +0.0045. The
   linearized invasion rate cannot see this (the storage ratios are 0/0 at
   zero biomass); a seeded forward solve can. In the source's SBR every
   population starts at about 1000 g COD/m3, so there the switches are inert.
   ASM2d writes its poly-P switch on the ratio `XPP/XPAO`, which has no such
   effect. With the switches on fractions (decision 14) the seeds grow at
   +0.0052 (1 g/m3) and +0.0048 (30 g/m3).
2. *The A2O's conditions sit on the PAO / GAO boundary*, tipped to
   Competibacter by the anoxic zone. In the source's SBR at 20 C, acetate,
   pH 7, the PAOs win slowly (53% at day 50, 65% at day 400; the generated
   model 54% and 67%), so the source's "coexistence" there is a 50-day
   snapshot. On the A2O (K at 1e-3, seed 1 g/m3, Competibacter growth rate
   days 60 to 120, 1/d): base +0.0045; no anoxic bio-P activity
   (`etaNO3_BIOP` 1e-6) +0.0002; no internal recycle +0.0034; sludge age 8 d
   instead of 19 +0.0030; anaerobic zone doubled +0.0030; no bio-P decay
   +0.0046; 15 C -0.037; pH 7.5 -0.013; feed VFA 75% acetate / 25%
   propionate -0.028. So the GAOs' edge is small and comes mostly from their
   anoxic processes (decision 11); temperature, pH and a propionate share,
   all as the source predicts, turn it to the PAOs.

Consequences: an A2O comparison at 20 C on all-acetate VFA tests the knife
edge, not the plant; a feed with a propionate share (typical of sewage) is
clear of it. The absolute storage switches are a choice for the owner
(faithful to the source and history-dependent, or written on ratios to the
biomass, as ASM2d does, which leaves the SBR validation's large populations
unchanged).

**Fermentation variants on the decided base (decisions 14 and 15).** 75%
acetate / 25% propionate VFA, 20 C, pH 7, storage switches on fractions.
Steady state by PTC, then both GAO groups seeded at 1 g/m3 (glycogen 0.3) and
150 days forward: in every variant both die out (-0.025 to -0.029 1/d over
days 75 to 150) and nothing else moves, so these are stable steady states
(g/m3; ASM2d, with one VFA, from its all-acetate run):

| Variant | `SF` end of anaerobic zone | PAO | Acidogens | Effluent PO4 |
|---|---|---|---|---|
| ASM2d | 0.93 | 784 | - | 5.13 |
| acidogens, ADM1 kinetics (current) | 14.5 | 571 | 0 (washed out) | 6.86 |
| heterotrophs ferment (3 1/d substrate, `K` 5, yield 0.1) | 0.47 | 1061 | - | 4.52 |
| BNRM1-type (off by O2 only, Ruel kinetics, 10 g/m3 acidogens in the influent) | 5.35 | 846 | 139 | 5.65 |
| as adopted (decision 16: also slower anaerobic decay, `etafe` 0.144) | 0.36 | 1057 | - | 4.41 |

## The core on BSM1 (2026-11-26)

`examples/plantwide/bsm1` runs the core and `asm2d` on `build_bsm1` (open loop,
Takács settler, 20 C) with BSM1's constant influent carried into each model's
species (`tests/validation/test_plantwide_core_bsm1.py`).

- *Aligned* (the core's fermentation off and anaerobic decay as aerobic;
  ASM2d's fermentation off and its `etafe` set to the core's): the steady
  states agree to six figures once ASM2d's positivity limiter (threshold
  1e-3) is off. With it on they differ by 0.2% (effluent ammonium 1.151
  against 1.153 g N/m3): oxygen in the anoxic tanks sits below the threshold,
  where the limiter throttles its uptake.
- *A defect it found:* the core split ASM2d's `SA` (all fermentation products)
  into acetate and propionate but gave the heterotrophs only acetate, so
  1.3 g COD/m3 of propionate (and 0.9 of hydrogen, with the aeration not
  stripping the declared gases) left the plant. The heterotrophs now grow on
  propionate with oxygen and nitrate, on ASM2d's VFA kinetics applied to
  acetate plus propionate (exactly ASM2d without propionate); the BSM1
  comparison strips the core's declared gases in the aerated tanks (gas
  transfer only, no biological uptake of hydrogen). Left: 0.004
  g COD/m3 of each.
- *As shipped*, the core leaves 11.0 g N/m3 of nitrate against ASM2d's 10.4,
  ammonium 1.11 against 1.12: its fermentation in the anoxic tanks turns some
  substrate into hydrogen, which is stripped rather than used for
  denitrification; the lower `etafe` and the slower anaerobic decay add to it.
- `build_a2o`'s aeration strips nothing, so on the A2O the core's hydrogen
  leaves in the effluent (a small COD loss); giving `build_a2o` the
  `aeration` argument `build_bsm1` has would fix that.

## The nitrite level (`plantwide_nitrite`, design 2026-11-27)

Extends `plantwide_bio_p`; decisions 17 to 20. Sources: Hiatt & Grady (2008)
for denitrification and the nitrifiers' inhibition, Daebel et al. (2007) and
Mehrani et al. (2021) for their oxygen affinities and temperature, Hao et al.
(2002) and Strous et al. (1998) for anammox.

**States.** Adds `SNO2` (g N/m3), `XAOB`, `XNOB` and `XAMX` (biomass, the
core's composition); removes `XAUT` and the one-step nitrification.

**Switches.** With `I(x) = monod_inh(x, K)` and `M = 1 - I`: the reduction
steps use `anoxic_NO3 = I(O2) M(NO3; KNO3_H)` and `anoxic_NO2 = I(O2) M(NO2;
KNO2_H)`, each on its own acceptor as in ASMN. The processes that do not
reduce nitrogen (hydrolysis, decay, fermentation, anaerobic storage) use
`anoxic = I(O2) M(NO3 + NO2)` and `anaerobic = I(O2) I(NO3 + NO2)`, so
`aerobic + anoxic + anaerobic = 1` and without nitrite every switch is the
core's. The methanogens' `anaerobic_only` takes NO3 + NO2.

**Two-step reduction.** Every inherited process that consumes nitrate under
`anoxic` (heterotroph growth on SF, acetate, propionate; every PAO and GAO
anoxic process) becomes two: `f_NO3_NO2 * anoxic_NO3` with its electrons going
NO3 to NO2 (2 e per N: the nitrate coefficient times 5/2), and `f_NO2_N2 *
anoxic_NO2` going NO2 to N2 (3 e per N: the nitrate coefficient times 5/3, as
nitrite). Yields and the other coefficients are unchanged. `f_NO3_NO2` 0.35
and `f_NO2_N2` 0.6 are ASMN's 0.28 and 3 x 0.16 over ASM2d's 0.8; `KNO2_H` is
ASMN's 0.2. In balance (nitrite made as fast as used, `r2 = 1.5 r1`) the
nitrate step limits, and the pair runs at 0.35 + 0.525 = 0.875 of the one
process's rate: ASMN's chain at 0.70 against ASM1's 0.8.

**Nitrifiers** (20 C). Growth is Monod in total ammonium (AOB) and total
nitrite (NOB), in oxygen, and non-competitively inhibited by free ammonia and
free nitrous acid:

| | AOB | NOB | source |
|---|---|---|---|
| max growth (1/d) | 0.9 | 0.78 | Jones et al. 2005 (WERF); Hiatt & Grady 2008 |
| growth theta | 1.072 | 1.11 | Jones et al. 2005; Mehrani et al. 2021 |
| decay (1/d), theta | 0.17, 1.029 | 0.089, 1.036 | WERF via Friedrich et al. 2017; Friedrich et al. (NOB activity loss) |
| substrate half-saturation (g N/m3) | NH4 0.6 | NO2 0.1 | Mannucci et al. 2020 (batch); Mehrani et al. 2021 (measured 0.06-0.14) |
| oxygen half-saturation (g/m3), apparent | 0.65 | 0.23 | Daebel et al. 2007, conventional sludge, 20 C |
| FA / FNA inhibition (g N/m3) | 1.0 / 0.1 | 0.2 / 0.04 | Hiatt & Grady 2008 |
| yield (g COD/g N) | 0.18 | 0.06 | Hiatt & Grady 2008 |

`FA = SNH4 10^pH / (exp(6344/T) + 10^pH)`, `FNA = SNO2 / (exp(-2300/T)
10^pH)`, as N. The NOB keep the core's ammonium nutrient term, and both the
phosphate one. ASMN's NOB mixotrophy is left out.

**Why these values (NOB literature, read 2026-11-27).** ASMN's NOB oxygen
half-saturation (1.2) and Hao et al.'s (2.2) wash the NOB out of BSM1 at 15 C,
against plant experience: Law et al. (2019) find Nitrospira and 0.3-0.7 g N/m3
of effluent nitrite at full scale at DO 0.8-1.2 and 4-6 d sludge age (30 C);
Liu et al. (2019) no nitrite at DO 0.3 (22 C, 20 d); Mehrani et al. (2021)
little accumulation in batch below DO 0.7. Measured values:

- intrinsic (dispersed cells): Nitrospira 0.54 (Blackburne et al. 2007), 0.33
  (Park et al. 2017), 0.13 (Manser 2005, cited by Blackburne); Picioreanu et
  al. use 0.03 (AOB) and 0.06 (NOB);
- apparent, on flocs: Daebel et al. 2007, conventional sludge (d50 100-110
  um, 20 C, 14 tests) AOB 0.65 (0.51-0.77), NOB 0.23 (0.19-0.29); MBR sludge
  0.42 and 0.28; Mehrani et al. 2021 AOB 0.30-0.39, NOB 0.20-0.36 (12, 20 C);
  Law et al. 2019 AOB 0.30, NOB 0.09 (30 C); Blackburne's Nitrobacter 0.43 to
  1.69 as floc median size goes 44 to 126 um.

Picioreanu et al.'s 3-D model explains the floc reversal (apparent AOB > NOB):
AOB colonies are larger and take more oxygen per cell, so their colonies have
steeper gradients; a 1-D homogeneous floc model never reverses. They state
that a model without diffusion takes apparent values, AOB's above NOB's.
Daebel et al. find K_O varies over weeks; they suggest treating it as
time-variable, a case-specific finding.

**AOB growth and decay (read 2026-11-27).** Growth and decay are taken as a
pair, since they are correlated in every fit (Mehrani et al. 2021; Friedrich
et al. 2017 for heterotrophs). For nitrifiers the decay rate is the biomass
loss rate under both death-regeneration and endogenous respiration (ASM2d and
ASM3 both use 0.15), so the conventions do not change the number.

| Source | AOB growth (1/d, 20 C) | AOB decay (1/d, 20 C) | Basis |
|---|---|---|---|
| Jones et al. 2005 (WERF) | 0.80-1.0 | 0.17 (basis) | several plants, real wastewater, three methods; growth theta 1.072, their test 1.07-1.1; one-step, ammonia oxidation limiting |
| Mannucci et al. 2020 | 0.72 (set B), 0.85 (set A) | 0.25, 0.35 | full-scale, 70% textile wastewater, sludge age 26 d; reference temperature not stated; K_NH 0.4 fitted to effluent, 0.6 from batch |
| Munz et al. 2011 | 1.05-1.4 | 0.28 aerobic, 0.07-0.16 anoxic | bench SBRs, synthetic feed, aerobic sludge age 2-5 d (ASM3 form) |
| Friedrich et al. 2017 (WERF 2016) | - | 0.136 (theta 1.023); NOB 0.089 (1.036) | decay only, loss of activity, 15-27 C; WERF basis 0.170 (1.029); ASM2/3 0.150 (1.116) |
| Hiatt & Grady 2008 | 0.78 | 0.096 | defaults |

Measured nitrifier decay depends weakly on temperature (theta 1.02-1.04,
Friedrich et al.); ASM2d's 1.116 overstates it. Munz et al.'s aerobically
grown sludge decays anoxically at 0.57 (AOB) and 0.45 (NOB) of the aerobic
rate, which supports the 0.5 anoxic factor; their alternating reactor gave
0.15-0.25. Growth rises at short sludge ages (Munz et al.'s literature summary:
AOB 1.0 at 10 d, 0.5 at 20 d), so Munz et al.'s high values reflect their 2-5 d.
On BSM1 at 15 C the AOB growth theta decides ammonium: 1.072 gives 7.84 g N/m3,
1.11 gives 18.98; 1.072 is used, consistent with the 0.9 it normalized.

Temperature (NOB): Blackburne et al.'s Nitrospira activity 0.44 e^(0.055 (T - 15))
(theta 1.057, 15-30 C, enriched culture, short-term) against Mehrani et al.'s
1.11 for both groups on full-scale sludge between 12 and 20 C; the latter is
the range that matters here. Park et al.'s growth rate (0.69 1/d) is inferred
from respirometry through a yield (0.14) that their own SI shows is 5-7 times
the recalculated pure-culture yields, so it is not used. Mehrani et al.'s
calibrated growth rates (AOB 0.57, NOB 0.11, apparently at 12 C, with
comammox in the model and decay fixed) are not used either.

**Anammox** (Hao et al. 2002, Table AIII, 30 C, from Strous et al. 1998): max
growth 0.072 1/d, decay 0.003 1/d, both with theta 1.0994 (70 kJ/mol between
20 and 30 C; 0.028 1/d at 20 C, Hao et al. quote 0.03), ammonium 0.07 and
nitrite 0.05 g N/m3, oxygen inhibition 0.01 g/m3. Stoichiometry: Strous et
al.'s 1.32 NO2 and 0.26 NO3 per NH4, the ammonium and N2 from the COD and N
balances. Autotroph decay is x0.5 with nitrite or nitrate (Hao et al., after
Koch et al. 2000), x`eta_b_AN` without either.

**ASMN's four-step denitrification**, for the N2O level. Each step is
heterotroph growth `muH etag_i S/(KS_i + S) KOH/(KOH + SO) ...` (KOH 0.1,
KS 20, KS5 40 g COD/m3), all at anoxic yield `etaY YH` (0.9 x 0.6):

| step | etag | acceptor term | NO inhibition |
|---|---|---|---|
| R2 NO3 to NO2 | 0.28 | `SNO3/(0.2 + SNO3)` | none |
| R3 NO2 to NO | 0.16 | `SNO2/(0.2 + SNO2)` | `0.5/(0.5 + SNO)` |
| R4 NO to N2O | 0.35 | `SNO/(0.05 + SNO + SNO^2/0.3)` | self (Haldane, 0.3) |
| R5 N2O to N2 | 0.35 | `SN2O/(0.05 + SN2O)` | `0.075/(0.075 + SNO)` |

Nitrite does not inhibit any step (ASMN departs from von Schulthess et al.
here); NO inhibits its own and the neighbouring reductions. Alkalinity is
made in the nitrite-to-NO step. ASMN also has assimilative nitrate reduction
(R6, R7), not needed while ammonium is present.

**Built (2026-11-27).** `_make_plantwide_nitrite.py` writes 35 species, 111
reactions: 23 inherited nitrate-reducing processes (3 heterotroph, 20 PAO /
GAO) each become a pair; COD, N, C and P close in every reaction and the units
check is clean. Checked (`tests/integration/test_plantwide_nitrite.py`,
`tests/validation/test_plantwide_nitrite_bsm1.py`):

- without nitrite every shared process runs at the bio-P level's rate, each
  pair's first step at 0.35 of it, the second steps idle;
- the AOB and NOB rates equal the forms above written out by hand, at 20 C /
  pH 7 and 15 C / pH 7.8; anammox equals Hao et al.'s at 30 and 20 C;
- anammox: 1.32 NO2 and 0.26 NO3 per NH4, N2 1.02 (Strous);
- on BSM1, tank 5 (g N/m3): at 20 C ammonium 2.58, nitrite 0.10, nitrate
  13.21; at 15 C 7.84, 0.54 and 8.40; 22.4% of the influent N leaves as N2 at
  20 C. With ASMN's NOB oxygen half-saturation (1.2) instead, nitrite 5.40 at
  20 C and NOB washout at 15 C. The published BSM1 steady state (ASM1, 15 C)
  is 1.73 ammonium and 10.42 nitrate; it is ASM1's output with agreed
  parameters, a reference between models, not data, and its nitrifier decay
  (0.05 1/d at 15 C) is a third of the measured rates. N, P and COD close over
  the plant with the stripped gases counted.

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
| Biomass | `XH` heterotrophs (also the fermenters), `XAUT` nitrifiers, `XPRO` propionate oxidizers, `XAC` acetoclastic methanogens, `XH2M` hydrogenotrophic methanogens |
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
| Decay of every group by death-regeneration (heterotrophs and nitrifiers with ASM2d's `fXI` = 0.1); heterotrophs at 0.25 of the rate without O₂ and NO₃ | ASM2d; Barker & Dold (1997) use the same form with 0.08, the value Ikumi et al. (2014) found consistent with digestion |
| Fermentation as heterotroph growth without O₂ and NO₃, in every unit (decision 16) | Barker & Dold (1997); Hauduc et al. (2013); kinetics of the Varga et al. (2018) model |
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
| AOB and NOB growth on ammonium / nitrite, FA and FNA inhibition, decay | Jones et al. (2005); Friedrich et al. (2017); Mannucci et al. (2020); Hiatt & Grady (2008); Anthonisen et al. (1976); Daebel et al. (2007); Mehrani et al. (2021) |
| Two-step denitrification | Hiatt & Grady (2008) |
| Anammox growth and decay | Hao et al. (2002); Strous et al. (1998, 1999) |

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

1. **Gas-phase species** (done): a model declares its dissolved gases in a
   `gases:` block and the unit owns the gas phase (*Gas phase* above).
2. **Gas transfer in aerated units** (done): `Aeration(stripped="model")`
   strips every gas the model declares.
3. **The digester unit** (done): `HeadspaceDigesterUnit` runs the g/m³ core.
4. **Performance.** The unified model is larger in every unit than the model it
   replaces there. Measure the right-hand side, the coloured Jacobian and a
   BSM2-sized dynamic and steady solve at each level before optimizing
   anything.

## Validation plan

- **Per reaction:** `check_conservation()` for COD, N and P (and charge at the
  chemistry level) passes on every reaction, as a unit test.
- **Per level, against its sources:**
  - core aerobic / anoxic reaction terms against `asm2d` with fermentation
    switched off (done: equal to 1e-6), then on a BSM1-type flowsheet (done,
    2026-11-26: the steady states agree to six figures with the processes the
    models do not share switched off and ASM2d's positivity limiter off; see
    *The core on BSM1*);
  - core digestion against `adm1` on the BSM2 digester feed: methane
    production, VS destruction, effluent ammonium;
  - bio-P against `asm2d` / `asm2d_tud` on an A2O flowsheet (`asm2d` done,
    `examples/plantwide/a2o`; `asm2d_tud` not yet);
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
2. **Who ferments.** Settled: the heterotrophs (decision 16).
3. **Heterotroph anaerobic hydrolysis factor.** Settled: 0.144 (decision 16,
   refitting decision 6's 0.15).
4. **Iron chemistry.** Saturation-index precipitation (already in the
   repository) or surface complexation. Recommendation: start with the
   existing approach; add surface complexation as its own level if aged-iron
   and P-desorption behaviour is needed.
5. **Gas-phase species in the model format.** Settled: the `gases:` block.
6. **The closure declaration**, after `plantwide_bio_p` is written.
7. **Shipped or example.** Recommendation: keep the levels out of the shipped
   catalogue until the plant-wide study is done, then ship them with a catalogue
   entry.

## References

DOIs were found from publisher, PubMed and repository records; three entries
have no DOI that could be confirmed and are marked. Check each against the
paper itself before citing it in a model file.

- Anthonisen, A.C., Loehr, R.C., Prakasam, T.B.S. & Srinath, E.G. (1976).
  Inhibition of nitrification by ammonia and nitrous acid. *J. Water Pollut.
  Control Fed.* 48(5), 835–852 (no DOI).
- Blackburne, R., Vadivelu, V.M., Yuan, Z. & Keller, J. (2007). Kinetic
  characterisation of an enriched Nitrospira culture with comparison to
  Nitrobacter. *Water Res.* 41(14), 3033–3042. doi:10.1016/j.watres.2007.01.043
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
- Daebel, H., Manser, R. & Gujer, W. (2007). Exploring temporal variations of
  oxygen saturation constants of nitrifying bacteria. *Water Res.* 41(5),
  1094–1102. doi:10.1016/j.watres.2006.11.011
- Flores-Alsina, X. et al. (2015). A plant-wide aqueous phase chemistry module
  describing pH variations and ion speciation/pairing in wastewater treatment
  process models. *Water Res.* 85, 255–265.
  https://doi.org/10.1016/j.watres.2015.07.014
- Friedrich, M., Jimenez, J., Pruden, A., Miller, J.H., Metch, J. & Takács, I.
  (2017). Rethinking growth and decay kinetics in activated sludge – towards a
  new adaptive kinetics approach. *Water Sci. Technol.* 75(3), 501–506.
  doi:10.2166/wst.2016.439
- Hao, X., Heijnen, J.J. & van Loosdrecht, M.C.M. (2002). Sensitivity analysis
  of a biofilm model describing a one-stage completely autotrophic nitrogen
  removal (CANON) process. *Biotechnol. Bioeng.* 77(3), 266–277.
  doi:10.1002/bit.10105
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
- Jones, R.M., Bye, C.M. & Dold, P.L. (2005). Nitrification parameter
  measurement for plant design: experience and experimental issues with new
  methods. *Water Sci. Technol.* 52(10–11), 461–468. doi:10.2166/wst.2005.0724
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
- Law, Y. et al. (2019). High dissolved oxygen selection against Nitrospira
  sublineage I in full-scale activated sludge. *Environ. Sci. Technol.* 53(14),
  8157–8166. doi:10.1021/acs.est.9b00955
- Liu, W., Chen, W., Yang, D. & Shen, Y. (2019). Functional and compositional
  characteristics of nitrifiers reveal the failure of achieving mainstream
  nitritation under limited oxygen or ammonia conditions. *Bioresour.
  Technol.* 275, 272–279. doi:10.1016/j.biortech.2018.12.065
- López-Vázquez, C.M. et al. (2009). Modeling the PAO–GAO competition: effects
  of carbon source, pH and temperature. *Water Res.* 43(2), 450–462.
  https://doi.org/10.1016/j.watres.2008.10.032
- Lizarralde, I. et al. (2015). A new general methodology for incorporating
  physico-chemical transformations into multi-phase wastewater treatment
  process models. *Water Res.* 74, 239–256.
  https://doi.org/10.1016/j.watres.2015.01.031
- Mehrani, M.-J., Lu, X., Kowal, P., Sobotka, D. & Makinia, J. (2021).
  Incorporation of the complete ammonia oxidation (comammox) process for
  modeling nitrification in suspended growth wastewater treatment systems.
  *J. Environ. Manage.* 297, 113223. doi:10.1016/j.jenvman.2021.113223
- Mannucci, A. et al. (2020). Ammonia oxidizing bacteria (AOB) kinetic
  parameters estimated using the actual maximum ammonia oxidation rate. *Int.
  Biodeterior. Biodegrad.* 147, 104876. doi:10.1016/j.ibiod.2019.104876
- Meijer, S.C.F., van Loosdrecht, M.C.M. & Heijnen, J.J. (2001). Metabolic
  modelling of full-scale biological nitrogen and phosphorus removing WWTP's.
  *Water Res.* 35(11), 2711–2723.
  https://doi.org/10.1016/S0043-1354(00)00567-4
- Munz, G., Lubello, C. & Oleszkiewicz, J.A. (2011). Factors affecting the
  growth rates of ammonium and nitrite oxidizing bacteria. *Chemosphere* 83(5),
  720–725. doi:10.1016/j.chemosphere.2011.01.058
- Nopens, I. et al. (2009). An ASM/ADM model interface for dynamic plant-wide
  simulation. *Water Res.* 43(7), 1913–1923.
  https://doi.org/10.1016/j.watres.2009.01.012
- Ozyildiz, G. et al. (2023). Restructuring anaerobic hydrolysis kinetics in
  plant-wide models for accurate prediction of biogas production. *Water Res.*
  120620. https://doi.org/10.1016/j.watres.2023.120620
- Sander, R. (2023). Compilation of Henry's law constants (version 5.0.0) for
  water as solvent. *Atmos. Chem. Phys.* 23, 10901–12440.
  https://doi.org/10.5194/acp-23-10901-2023
- Picioreanu, C., Pérez, J. & van Loosdrecht, M.C.M. (2016). Impact of cell
  cluster size on apparent half-saturation coefficients for oxygen in
  nitrifying sludge and biofilms. *Water Res.* 106, 371–382.
  doi:10.1016/j.watres.2016.10.017
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
- Park, M.-R., Park, H. & Chandran, K. (2017). Molecular and kinetic
  characterization of planktonic Nitrospira spp. selectively enriched from
  activated sludge. *Environ. Sci. Technol.* 51(5), 2720–2728.
  doi:10.1021/acs.est.6b05184
- Ribes, J., Ferrer, J., Bouzas, A. & Seco, A. (2002). Modelling of an
  activated primary settling tank including the fermentation process and VFA
  elutriation. *Environ. Technol.* 23(10), 1147–1156. No DOI confirmed.
- van Veldhuizen, H.M., van Loosdrecht, M.C.M. & Heijnen, J.J. (1999).
  Modelling biological phosphorus and nitrogen removal in a full scale
  activated sludge process. *Water Res.* 33(16), 3459–3468.
  https://doi.org/10.1016/S0043-1354(99)00064-0
  https://doi.org/10.2166/wst.2018.490 (also presented at WRRmod 2018).
