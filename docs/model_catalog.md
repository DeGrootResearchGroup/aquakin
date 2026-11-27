# Model catalogue

`aquakin` ships a library of ready-to-use reaction models spanning oxidation
chemistry, biological wastewater treatment, anaerobic digestion, sewer
processes, and mineral precipitation. Load any of them by name:

```python
import aquakin

model = aquakin.load_model("asm1")
print(model.summary())          # species, reactions, parameters, and references
print(model.references)         # the literature the model is built from
```

Every model carries its own literature `references`, per-species units and
descriptions, and default parameter and concentration vectors — see
[Getting started](getting_started.md) for the loading and inspection API, and
the [YAML model format](model_format.md) if you want to write your own.

Each entry below lists the model's size as *(species, reactions)*, as reported
by the compiled model itself; `model.references` lists the full citations.

A model's rate constants set the unit its time axis is read in: the
oxidation-chemistry models integrate in **seconds**, and the biological and
mineral-precipitation models in **days**. `model.time_unit` reports it, and `model.summary()` prints it.

## Oxidation chemistry

`ozone_bromate`
: Bromate (BrO₃⁻) formation during the ozonation of bromide-containing
  drinking water, with explicit hydroxyl-radical chemistry — the direct
  (molecular ozone) and indirect (·OH) oxidation pathways that together set the
  bromate yield. *(6 species, 7 reactions.)* After Acero & von Gunten (2001)
  and Pinkernell & von Gunten (2001).

`uv_h2o2`
: UV/H₂O₂ advanced oxidation of a generic target micropollutant: hydrogen
  peroxide photolysis generates hydroxyl radicals that oxidize the target, in
  competition with background scavenging. *(4 species, 4 reactions.)* After
  Glaze et al. (1987) with rate constants from Buxton et al. (1988).

## Activated sludge (ASM family)

The IWA Activated Sludge Models are the standard framework for biological
carbon and nutrient removal in wastewater treatment. `aquakin` ships the
reference models and several literature extensions. Concentrations are in
`g/m³` of COD, N or P, suspended solids in g TSS/m³, and alkalinity in mol/m³
for `asm1` (and `asm1_ammonia_limitation`) but kmol/m³ for the ASM2d and ASM3
models. They integrate in **days**. Each model's default concentrations are a
typical aerated mixed liquor (an established bio-P sludge for the models with
phosphorus-accumulating organisms), a starting point for trying the model.

`asm1`
: Activated Sludge Model No. 1, the reference model for carbon oxidation,
  nitrification, and denitrification (Henze et al. 1987). This is the
  textbook Gujer matrix as used in the IWA benchmark simulation plants, so it
  reproduces canonical ASM1/BSM results directly. *(13 species, 8 reactions.)*

`asm1_ammonia_limitation`
: `asm1` plus a nutrient-availability switch: both heterotrophic growth rates
  carry an extra `[SNH]/(KNH_H + [SNH])` factor that shuts growth down as
  ammonia is exhausted. A recognized vendor-style extension (not part of the
  reference matrix); it is inert in ammonia-rich influent — matching `asm1`
  there — and only acts where ammonia is driven low. *(13 species, 8 reactions.)*

`asm2d`
: Activated Sludge Model No. 2d: ASM1 extended with biological phosphorus
  removal by polyphosphate-accumulating organisms (PAOs), denitrifying PAOs,
  and simple chemical phosphorus precipitation (Henze et al. 1999).
  *(19 species, 21 reactions.)*

`asm2d_tud`
: The Delft (TU Delft) variant of ASM2d, with a metabolic description of the
  PAO storage/growth cycle in place of ASM2d's black-box bio-P kinetics
  (Meijer et al. 2001; Murnleitner et al. 1997). As published, the poly-P
  storage rate is inversely proportional to the stored fraction
  f_PP = XPP/XPAO, which makes it infinite when XPP is exactly 0. As a
  numerical guard that is not part of the published model, aquakin floors
  that 1/f_PP factor at f_min = 0.001 g P/g COD, so the rate is capped and
  finite down to XPP = 0. The floor is inactive whenever f_PP is above
  f_min, where the rate is exactly the published one.
  *(18 species, 22 reactions.)*

`asm2d_chemp`
: ASM2d with **saturation-index-driven** chemical phosphorus precipitation
  replacing the simple empirical metal model. Dosed ferric precipitates
  orthophosphate as strengite (FePO₄) and competes to form ferrihydrite
  (Fe(OH)₃), with the rate driven by the mineral saturation index — so the
  achievable effluent phosphate carries a pH-dependent floor. The bounded rate
  form keeps a dynamic solve differentiable. The precipitates add their molar
  mass to the suspended solids (`XTSS`), so the settler and the TSS readings
  count them. The worked example of the precipitation engine composed with a
  full biological model. *(20 species, 21 reactions.)* Precipitation framework after Kazadi Mbamba et al. (2015).

`asm3`
: Activated Sludge Model No. 3: a revision of ASM1 in which stored internal
  products, rather than direct substrate uptake, mediate heterotrophic growth,
  giving a cleaner separation of storage and growth (Gujer et al. 1999).
  *(13 species, 12 reactions.)*

`asm3_biop`
: ASM3 extended with the EAWAG Bio-P module (Rieger et al. 2001):
  phosphorus-accumulating organisms store PHA, take up and release
  polyphosphate, and grow aerobically and anoxically alongside ASM3's
  heterotrophs and nitrifiers. *(17 species, 23 reactions.)*

### Two-step nitrogen variants

These build on ASM3 to resolve nitrogen conversions the lumped models cannot —
nitrite as an explicit intermediate, nitrous-oxide emission, anammox, and
comammox. Each closes COD, nitrogen, and charge balances to machine precision.

`asm3_2step`
: ASM3 with **two-step nitrification and denitrification** (Kaelin et al.
  2009): nitrite (NO₂) is carried explicitly, the single autotroph splits into
  ammonia-oxidizing (AOB) and nitrite-oxidizing (NOB) organisms, and each
  denitrification step is resolved separately. Resolves nitrite peaks and the
  nitrite shunt, and is the basis for the N₂O, anammox, and comammox variants.
  *(15 species, 19 reactions.)*

`asm3_2step_n2o`
: `asm3_2step` extended with the two-pathway AOB **nitrous-oxide (N₂O)** model
  of Pocquet et al. (2016), resolving the AOB electron-transport intermediates
  (hydroxylamine, nitric oxide) and both the NN and ND N₂O production pathways.
  Reproduces the observed rise of N₂O with nitrite and its peak at intermediate
  dissolved oxygen. *(18 species, 23 reactions.)*

`asm3_2step_anammox`
: `asm3_2step` extended with **anammox** (anaerobic ammonium-oxidizing)
  bacteria (Strous et al. 1998, 1999), which oxidize ammonium with nitrite
  directly to dinitrogen. With AOB, NOB, and anammox all present the model
  supports partial-nitritation/anammox (PN/A) deammonification for sidestream
  autotrophic nitrogen removal. *(16 species, 22 reactions.)*

`asm3_2step_comammox`
: `asm3_2step` extended with a **complete-ammonia-oxidizing (comammox)**
  organism parameterized from Kits et al. (2017). Comammox performs complete
  nitrification (NH₄ → NO₃) in a single organism with a very high ammonia
  affinity, so it out-competes canonical AOB at low ammonium — the documented
  niche differentiation. *(16 species, 22 reactions.)*

## Anaerobic digestion

`adm1`
: Anaerobic Digestion Model No. 1 (Batstone et al. 2002) in its BSM2
  implementation form (Rosen & Jeppsson 2006): disintegration and hydrolysis,
  the acidogenic/acetogenic/methanogenic uptake reactions with pH, hydrogen,
  and free-ammonia inhibition, biomass decay, and a gas headspace with
  liquid–gas transfer and biogas outflow. pH is state-derived through the
  charge-balance speciation solver. Validated against the published BSM2
  open-loop steady-state digester. *(29 states — 26 liquid + 3 gas — 25
  reactions.)*

## Sewer processes (WATS)

The WATS (Wastewater Aerobic/anaerobic Transformations in Sewers) framework
models the carbon and sulfur transformations that drive sulfide generation and
odour/corrosion in sewers. `aquakin` ships the reference model. It integrates in
**days**.

`wats_sewer`
: The reference WATS model (Hvitved-Jacobsen, Vollertsen & Nielsen 2013):
  aerobic, anoxic, and anaerobic heterotrophic carbon turnover (growth,
  maintenance, hydrolysis, fermentation) coupled to the sulfur cycle —
  sulfate reduction to sulfide and chemical plus biological sulfide oxidation.
  pH is state-derived by charge balance. *(15 species, 34 reactions.)*

## Mineral precipitation

These use the generalized saturation-index precipitation framework of Kazadi
Mbamba et al. (2015): each mineral declares its constituent ions, solubility
product, and supersaturation order, and the engine drives precipitation or
dissolution from the free-ion activities at the operating pH.

`precipitation_struvite_calcite`
: Precipitation and dissolution of struvite (MgNH₄PO₄) and calcite (CaCO₃) from
  an anaerobic-digester supernatant — the worked example of the precipitation
  framework. *(7 species, 2 reactions.)*

`precipitation_metal_phosphate`
: Chemical phosphorus removal by ferric or aluminium dosing: the metal
  precipitates orthophosphate as the very insoluble FePO₄/AlPO₄ while competing
  to form the hydroxide, giving a pH-dependent floor on the achievable
  phosphate. *(7 species, 4 reactions.)* Because these minerals are so
  insoluble their kinetics are extremely stiff, which defeats gradient-based
  sensitivity analysis; two differentiable variants are provided:

  - `precipitation_metal_phosphate_equilibrium` — solves the precipitation
    **equilibrium** algebraically (`IAP = Ksp` with mass balance) via
    `model.precipitation_equilibrium(...)`, exact and `jax.grad`-clean.
  - `precipitation_metal_phosphate_bounded` — uses a bounded kinetic driver so
    the rate Jacobian stays well-conditioned and a **dynamic** solve is
    differentiable, relaxing to the same equilibrium.
