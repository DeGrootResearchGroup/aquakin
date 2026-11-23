# A unified plant-wide model: design proposal (2026-11-23)

Status: **draft for the owner's review.** Nothing here is implemented. The
decisions listed under *Open decisions* are the owner's; everything else is a
proposal that stays reversible until the first level is merged.

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

## Proposed state set

Names follow ASM2d where ASM2d has the concept, so the models read naturally to
anyone who knows the IWA family.

**Core**

| Group | Species |
|---|---|
| Organics | `SF` fermentable substrate, `SA` fermentation products (as acetate), `SI` soluble inert, `XS` slowly biodegradable, `XI` particulate inert |
| Biomass | `XH` heterotrophs (including fermentation), `XAUT` nitrifiers, `XAC` acetoclastic methanogens, `XH2M` hydrogenotrophic methanogens |
| Dissolved | `SO2`, `SNH4`, `SNO3`, `SN2`, `SH2`, `SCH4`, `SIC` (total inorganic carbon) |
| Solids | `XISS` inorganic suspended solids (for TSS) |
| Gas phase | `G_CH4`, `G_CO2`, `G_H2` (headspace; see *Engine work*) |

**`plantwide_bio_p` adds** `SPO4`, `XPAO`, `XPP`, `XPHA` (and `XGAO`, `XGLY` if
GAOs are included, see *Open decisions*).

**`plantwide_nitrite` replaces** `XAUT` with `XAOB`, `XNOB` and adds `SNO2`,
`XAMX`.

**`plantwide_chem` adds** `SCa`, `SMg`, `SK`, lumped strong cations / anions,
`SFe3` / `SFe2` and the iron and mineral solids (calcite, struvite, a calcium
phosphate, iron phosphate / hydroxide), plus `G_NH3`.

## Processes and sources by level

### `plantwide_core`

| Processes | Primary source |
|---|---|
| Hydrolysis of `XS` (aerobic, anoxic, anaerobic), heterotroph growth on `SF` and `SA` (O₂, NO₃), fermentation of `SF` to `SA`, nitrification, decay | ASM2d (Henze et al., 1999) |
| Decay as endogenous respiration with reduced anoxic / anaerobic rates (instead of lysis) | Barker & Dold (1997) |
| Acetoclastic and hydrogenotrophic methanogenesis, their NH₃ and H₂ inhibition | ADM1 (Batstone et al., 2002) |
| Hydrogen as a fermentation product | ADM1 acidogenesis product yields, lumped (our derivation, documented) |
| Simplified digestion on an activated-sludge state set | Siegrist et al. (2002); Sötemann et al. (2005) |

**Reconciliation choices this level makes, each to be documented:**
- ADM1's separate carbohydrate, protein and lipid pools and its sugar, amino
  acid, LCFA, propionate and butyrate degraders collapse into
  `XS` → `SF` → `SA` + `SH2`. Whether to keep propionate is an open decision:
  it is the intermediate that accumulates first when a digester is overloaded.
- Methanogen inhibition by O₂ and nitrate is not in ADM1. The form and its
  constants need a source, or are stated as our choice.
- Particulate organics the activated-sludge models treat as unbiodegradable
  stay largely unbiodegradable in a digester (Ikumi et al., 2014), so `XI` is
  inert in every unit and the first version has no anaerobic conversion of
  inert material.

### `plantwide_bio_p`

| Processes | Primary source |
|---|---|
| PHA storage with P release, polyphosphate storage, PAO growth on PHA (aerobic, anoxic), lysis of PAO, PP and PHA | ASM2d (Henze et al., 1999); metabolic form as in Meijer et al. (2001) |
| GAO glycogen storage and competition for VFA (if included) | López-Vázquez et al. (2009) |
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

## Engine and plant work the model needs

1. **Gas-phase species declared by the model.** The digester unit hard-codes
   ADM1's headspace species (`ADM1_GAS_SPECIES` in `plant/digester.py`). A
   model should declare which of its species are gas phase, so the digester
   works with any model. This is a model-format change (owner's decision); it is
   worth doing independently of this model.
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
  - core aerobic / anoxic behaviour against `asm2d` with bio-P inactive, on a
    BSM1-type flowsheet;
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

1. **Bio-P populations.** PAOs only (ASM2d), PAOs and GAOs as separate
   populations (López-Vázquez et al., 2009), or a single carbon-storing
   population whose metabolism switches with conditions (Varga et al., 2018).
   Recommendation: separate PAOs and GAOs, the more established and more easily
   benchmarked form; read Varga et al. before deciding.
2. **Digestion detail.** Methanogens on `SA` and `SH2` only, or also carry
   propionate. Recommendation: include propionate if the core's digester
   comparison against ADM1 shows it is needed for overload behaviour.
3. **Iron chemistry.** Saturation-index precipitation (already in the
   repository) or surface complexation. Recommendation: start with the
   existing approach; add surface complexation as its own level if aged-iron
   and P-desorption behaviour is needed.
4. **Gas-phase species in the model format** (*Engine work* item 1).
5. **The closure declaration**, after `plantwide_bio_p` is written.
6. **Shipped or example.** Recommendation: keep the levels out of the shipped
   catalogue until the plant-wide study is done, then ship them with a catalogue
   entry.

## References

DOIs were found from publisher, PubMed and repository records; three entries
have no DOI that could be confirmed and are marked. Check each against the
paper itself before citing it in a model file.

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
- Ikumi, D.S., Harding, T.H. & Ekama, G.A. (2014). Biodegradability of
  wastewater and activated sludge organics in anaerobic digestion. *Water Res.*
  56, 267–279. https://doi.org/10.1016/j.watres.2014.02.008
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
- Meijer, S.C.F., van Loosdrecht, M.C.M. & Heijnen, J.J. (2001). Metabolic
  modelling of full-scale biological nitrogen and phosphorus removing WWTP's.
  *Water Res.* 35(11), 2711–2723.
  https://doi.org/10.1016/S0043-1354(00)00567-4
- Nopens, I. et al. (2009). An ASM/ADM model interface for dynamic plant-wide
  simulation. *Water Res.* 43(7), 1913–1923.
  https://doi.org/10.1016/j.watres.2009.01.012
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
  https://doi.org/10.2166/wst.2018.490 (also presented at WRRmod 2018).
