# CLAUDE.md — aquakin Project Briefing

This file is the authoritative reference for developing the `aquakin` library.
Read it in full before writing any code. After every code change, consult the
**Post-Change Checklist** at the bottom of this file.

---

## Project Overview

`aquakin` is an open source Python library for modelling reactive scalar
transport in aqueous environmental systems. It provides a modular, runtime-
configurable kinetics engine that can be coupled to any flow solver.

Both **chemistry** (ozonation, advanced oxidation, chlorine decay, ...) and
**biology** (activated sludge models, anaerobic digestion, ...) are in scope.
The shipped networks currently are:

- `ozone_bromate` — bromate formation during ozonation, with explicit OH
  radical chemistry (after Acero & von Gunten 2001; Pinkernell & von Gunten 2001).
- `uv_h2o2` — UV/H₂O₂ advanced oxidation of a generic target micropollutant.
- `asm1` — Activated Sludge Model No. 1, the IAWQ reference biological
  wastewater treatment model (Henze et al. 1987). The shipped network is the
  textbook / BSM-faithful Gujer matrix — heterotroph growth has **no** ammonia
  (nitrogen-source) availability switch — so `load_network("asm1")` reproduces
  canonical ASM1/BSM results out of the box.
- `asm1_ammonia_limitation` — ASM1 plus a BioWin/SUMO-style nutrient-availability
  switch: both heterotroph growth rates carry an extra `[SNH] / (KNH_H + [SNH])`
  factor (default `KNH_H = 0.05 g_N/m³`) that shuts growth down when ammonia is
  exhausted. A recognized vendor extension, **not** in the reference Gujer matrix;
  inert in N-rich influent (so it matches `asm1` there) and only bites where
  ammonia is driven low. Identical to `asm1` apart from that factor and the
  `KNH_H` parameter.
- `asm2d` — ASM2D, ASM1 extended with biological phosphorus removal
  and denitrifying polyphosphate-accumulating organisms. *(Several SUMO-import
  errors in the original YAML were corrected against both the SUMO `ASM2D.xlsm`
  source and Henze et al. 2000 STR No. 9 — they survived because each conserves
  COD/N/P, so the continuity suite passed while nitrification and bio-P were
  broken; running the network to a viable steady state in the A²O plug below is
  what surfaced them: (1) **heterotroph lysis** `Lysis_1` (rate `bH·[XH]`)
  decremented `XAUT` instead of `XH`, applying the large heterotroph decay to the
  nitrifiers and washing them out; (2) the **aerobic poly-P storage** inhibition
  dropped the maximum-ratio term — restored to the Henze/SUMO
  `(K_MAX − XPP/XPAO)/(K_iPP + K_MAX − XPP/XPAO)` with `K_MAX = 0.34`, so stored
  poly-P reaches ~`K_MAX·XPAO` instead of being capped ~17× low; (3) the
  **autotroph and PP-uptake half-saturations** had been collapsed onto the
  heterotroph/hydrolysis values by an import that deduplicated Monod terms by
  species — `KO2_AUT` (0.5, was 0.2), `KNH4_AUT` (1.0, was 0.05 — a 20× error that
  let nitrifiers over-compete for ammonia and N-starve PAO growth), `KALK_AUT`
  (0.0005), `KPS` (0.2, the PO₄-uptake constant for poly-P storage). Also added
  the standard IWA `clip_negative_states` clamp and a `positivity_limiter` (asm2d
  had neither); and (4) the **chemical-P precipitation** reactions had dropped
  their metal coefficients — `Precipitation`/`Redissolution` changed `SPO4` and
  the lumped `XTSS` but not `XMeOH`/`XMeP`, so the metal hydroxide was never
  consumed (an inexhaustible precipitant) and metal phosphate never formed;
  restored to the SUMO/Henze Table 3.6 `XMeOH = fMeOH_PO4_MW (−3.45)`,
  `XMeP = fMeP_PO4_MW (+4.87)` (and the `KALK_PRE` redissolution alkalinity
  constant). With these the network nitrifies, removes phosphorus biologically,
  and the dosed-metal chemical-P works (the A²O plant's ferric dosing). The same
  import errors were found and fixed in `asm2d_tud`, `asm3` and `asm3_biop` — the
  whole bio-P/nitrification family is corrected. **The entire family is now
  verified term-by-term, value-by-value against the SUMO source spreadsheets**
  (every parameter value, every rate-term Monod half-saturation constant, every
  participating-species set, and every numeric stoichiometric coefficient
  including the charge-balance `SALK`/`SHCO` terms match SUMO — 482 coefficients,
  0 mismatches across the four models; the corrected constants are pinned per
  network in `tests/integration/test_asm_family.py::
  test_biop_autotroph_and_polyp_constants`). **The root cause was UPSTREAM, in the
  external `wastewaterad.tools.sumo_import` JSON dump the networks were originally
  imported from** — not in any aquakin-side transcription. The collapse (per-group
  Monod constants deduplicated by species), the lost `K_max` poly-P-storage
  override (the SUMO `MRinh`/`MRsat` *generic* function form was exported instead
  of the instantiated "Calculated variables" form), and the dropped precipitation
  metal coefficients (collapsed into their `XTSS` sum) all already existed in that
  JSON. The four ASM networks are now **maintained directly as YAML** (the
  one-shot SUMO import converter has been retired, so there is no regeneration step
  to reintroduce the upstream bugs); they remain audited against the SUMO `.xlsm`
  source by [`scripts/verify_sumo_asm.py`](scripts/verify_sumo_asm.py), which
  re-checks every parameter, rate-term constant, participating species and numeric
  coefficient and exits non-zero on any discrepancy — run it after any hand-edit
  to a SUMO-derived network.)*
- `asm2d_chemp` — ASM2D with **saturation-index-driven chemical-P precipitation**
  (ferric), replacing ASM2d's simple empirical `kPRE·[SPO4]·[XMeOH]`
  Precipitation/Redissolution with the generalised precipitation framework
  (Kazadi Mbamba et al. 2015). An **`extends: asm2d`** file that `remove:`s the
  simple metal model (the `XMeOH`/`XMeP` species, `kPRE`/`kRED`, the
  `fMeOH`/`fMeP` mass factors) and adds a `precipitation:` block: dosed ferric
  `S_Fe3` precipitates orthophosphate as strengite `FePO4` and competes to form
  ferrihydrite `Fe(OH)3`, with the rate driven by the saturation index from the
  free-ion activities at the operating pH (a fixed condition) — so the achievable
  effluent phosphate has a **pH-dependent floor** (the hydroxide buffers the free
  metal, so chemical-P worsens at higher pH; verified — the FeOH3 vs FePO4
  saturation gap widens pH 6.5→8.0). The minerals use the **bounded** driver
  (`R = tanh(SI/(2ν)·ln10)`), so the rate Jacobian is `~k` (non-stiff) and a
  dynamic reactor/plant solve is differentiable — `jax.grad` of the effluent
  phosphate w.r.t. the ferric dose flows through the time integration. The metal
  and solids are mol/m³; ASM2d's `SPO4` stays g_P/m³, so the precipitation ions
  carry `molar_mass: 31000` (g_P/m³→mol/L for the activity product) and the
  reactions carry the P molar mass `P_MW` to convert between the bases — phosphorus
  is conserved exactly (the SPO4 lowered equals 31 g/mol × the FePO4 precipitated;
  `tests/integration/test_asm2d_chemp.py`). This is the rigorous,
  thermodynamically-grounded counterpart of the A²O plant's native-ASM2d ferric
  dosing (`FerricDose` on the simple kPRE model), and the worked example of the
  precipitation engine composing with a full biological network.
- `asm2d_tud` — Delft TUD variant of ASM2D with revised bio-P stoichiometry.
  *(Carried the same import errors as `asm2d`, now fixed against SUMO
  `ASM2D_TUD.xlsm` + Henze STR No. 9: the `Lysis_1` heterotroph→autotroph biomass
  swap; the autotroph half-saturations (`KNH_A = 1.0`, `KO_A = 0.5`,
  `KHCO_A = 0.0005`) collapsed onto the heterotroph values; and the metabolic
  poly-P storage limiter, which used a plain ratio Monod where the model limits
  storage to a maximum ratio `(fPP_max − XPP/XPAO)/(KfPP + fPP_max − XPP/XPAO)`,
  `fPP_max = 0.35` — without it stored poly-P is unbounded.)*
- `asm3` — ASM3, ASM1 with internal storage products replacing hydrolysis.
  *(Carried the collapsed-autotroph-constant import error: nitrifier growth and
  aerobic respiration used the heterotroph O₂/ammonia/alkalinity half-saturations
  instead of `KA_O2 = 0.5`, `KA_NH4 = 1.0`, `KA_ALK = 0.0005` — fixed and verified
  against SUMO `ASM3.xlsm`.)*
- `asm3_2step` — ASM3 extended for **two-step nitrification and two-step
  denitrification** (Kaelin et al. 2009), carrying nitrite (NO2) as an explicit
  intermediate. The lumped `SNOX` splits into `SNO3` + `SNO2` and the single
  autotroph `XA` into ammonia-oxidising `XAOB` (NH4→NO2, nitritation) and
  nitrite-oxidising `XNOB` (NO2→NO3, nitratation); ASM3's 8 processes become 19
  (autotroph processes doubled; every heterotroph anoxic process split into a
  NO3→NO2 and a NO2→N2 step). The two denitrification steps share the
  electron budget of full denitrification (NO3→NO2 is the 2-electron
  `iCOD_NO3NO2 = 8/7` gCOD/gN step, NO2→N2 the 3-electron `iCOD_NO2N2 = 12/7`,
  summing to the 5-electron `iNO3_N2 = 40/14`; nitritation/nitratation O2 demands
  `48/14` + `16/14` sum to the full `64/14`). Resolves nitrite peaks, the
  nitrite-shunt (NOB out → N stalls at nitrite), and is the substrate for the
  N2O extension (#269 PR2). Default parameters are the Kaelin (2009) 20 °C set
  with the Table-5 temperature dependencies; **standalone YAML** (not an
  `extends: asm3` inheritance file) because the two state-variable splits rewrite
  most of the process matrix — the full Petersen matrix stays visible in one
  place for the COD/N/charge continuity auditing. COD, N and charge close to
  machine precision (`tests/integration/test_asm_continuity.py`), and the
  two-step nitrification/denitrification signatures are checked in
  `tests/integration/test_asm3_2step.py`.
- `asm3_2step_n2o` — `asm3_2step` extended with the **two-pathway AOB nitrous-
  oxide (N₂O) model** of Pocquet et al. (2016). An **`extends: asm3_2step`
  inheritance file** (this layer *is* a clean delta, unlike the base two-step
  refactor): it `remove:`s the lumped AOB nitritation and replaces it with the
  AOB electron-transport chain through explicit intermediates — NH₄ → `SNH2OH`
  (hydroxylamine, AMO) → `SNO` (nitric oxide, HAO) → `SNO2` — plus dissolved
  `SN2O`. N₂O is produced by two competing pathways: **NN** (Nor reduces NO to
  N₂O coupled to NH₂OH oxidation) and **ND** (free nitrous acid `[SNO2]·pH_switch(pKa)`
  reduced to N₂O, with the source study's Haldane DO term — production rises as
  DO falls to a maximum, then falls toward zero DO). Adds a **fixed operating-pH
  condition** (for the free-nitrous-acid fraction; the source study held pH
  constant). 18 compounds, 23 processes. The N-oxide intermediates carry their
  NH₄-referenced electron COD (NH₂OH 16/14, NO 40/14, N₂O 32/14 gO₂/gN), so COD,
  N and charge close to machine precision (continuity suite); NO/N₂O are kept as
  dissolved states (gas stripping is a reactor concern, not modelled). Reproduces
  Pocquet's headline trends — N₂O rises with nitrite and peaks at intermediate DO
  (`tests/integration/test_asm3_2step_n2o.py`). NOTE: `SNO` is **nitric oxide**
  here, which collides with the ASM1 family's `SNO` (= nitrate); the composition
  table disambiguates by the presence of `SNO3`.
- `asm3_2step_anammox` — `asm3_2step` extended with **anammox** (anaerobic
  ammonium-oxidising) bacteria, after Strous et al. (1998, 1999). An additive
  **`extends: asm3_2step`** file (a new biomass `XAMX` + its 3 processes; the
  inheritance sweet spot): anammox oxidises ammonium with nitrite as the electron
  acceptor straight to N₂ (no organic carbon), oxidising a little extra nitrite to
  nitrate to fix CO₂ for growth — the canonical Strous stoichiometry **NH₄ : NO₂ :
  NO₃ ≈ 1 : 1.32 : 0.26** (1.02 N₂ per NH₄). The stoichiometry is written
  symbolically in the yield `Y_AMX` and the nitrate-production ratio `f_NO3_AMX`,
  **derived to conserve COD/N/charge exactly while reproducing the Strous ratios**
  (verified to machine precision; the matrix is COD-exact, unlike the raw
  measured coefficients which carry a ~1% imbalance). Anammox is very slow
  (μ ≈ 0.08 d⁻¹), high-affinity (`K_NH4`/`K_NO2` ≈ 0.05–0.07 gN/m³), reversibly
  **O₂-inhibited** (a `monod_inh` O₂ term), with the source 70 kJ/mol activation
  energy (θ ≈ 1.10). 16 compounds, 22 processes. With AOB + NOB + anammox all
  present the network supports **partial-nitritation/anammox (PN/A)
  deammonification** (autotrophic N removal — the sidestream process); the
  anammox half is validated as anaerobic deammonification (NH₄ + NO₂ → N₂,
  `tests/integration/test_asm3_2step_anammox.py`, `examples/anammox_deammonification.py`).
  The full single-stage **PN/A sidestream flowsheet** is a continuously-fed
  low-DO `CSTRUnit` (`examples/pna_sidestream.py`, regression-tested in
  `tests/integration/test_pna_sidestream.py`): a high-ammonium reject stream with
  no organic carbon reaches an autotrophic-N-removal steady state (~86% TIN
  removal) where anammox is retained, **NOB are out-competed for nitrite and wash
  out**, and the effluent nitrate is the anammox autotrophic byproduct (~26% of N
  removed) rather than NOB activity. Three couplings control it: a long HRT
  (~30 d) retains the slow anammox (in a plain CSTR SRT = HRT); a low kLa holds
  DO ~0.05 gO₂/m³ (anammox is strongly O₂-inhibited — `K_O2_AMX_inh` is small, so
  the viable DO window is narrow); and a limited feed alkalinity caps nitritation
  at ~half the ammonium, supplying anammox its ~1:1.3 NH₄:NO₂ feed. (A
  `BatchReactor` cannot sustain low DO, so PN/A needs the continuous reactor.)
- `asm3_2step_comammox` — `asm3_2step` extended with a **complete-ammonia-oxidising
  (comammox) organism** `XCMX`, parameterised from Kits et al. (2017) for
  *Nitrospira inopinata*. An additive **`extends: asm3_2step`** file. Comammox is
  a single organism doing **complete nitrification** (NH₄ → NO₃, the original
  one-step ASM3 autotroph stoichiometry) rather than the AOB/NOB division of
  labour; its defining trait is an **oligotrophic lifestyle** — a very high
  ammonia affinity (`K_NH4` ≈ 0.012 gN/m³, ~10× the AOB value) with a low maximum
  rate. The competition this creates is the point: comammox **out-competes the
  AOB at low ammonium** (its affinity dominates) while the canonical AOB win at
  high ammonium (their higher max rate dominates) — the documented niche
  differentiation, reproduced in `tests/integration/test_asm3_2step_comammox.py`
  (comammox/AOB per-biomass rate ratio 2.3 at NH₄ = 0.02, < 1 above ~0.1 gN/m³).
  The complete-nitrification O₂ demand is the inherited `iO2_AOB + iO2_NOB`
  (nitritation + nitratation), so COD/N/charge close to machine precision.
  16 compounds, 22 processes. **Caveat:** there is no settled canonical
  activated-sludge comammox Gujer matrix in the literature — this network is a
  defensible *synthesis* (single-organism complete nitrification + Kits 2017
  kinetics), not a faithful reproduction of one published matrix; nitrite leakage
  (comammox's poor nitrite affinity) is omitted (modelled as complete
  nitrification), a possible two-internal-step refinement.
- `asm3_biop` — ASM3 + bio-P extension. *(Carried the same import errors as
  `asm2d`, now fixed against SUMO `ASM3_BioP.xlsm` + Henze STR No. 9: the
  autotroph growth/respiration used the heterotroph O₂/ammonia/alkalinity
  half-saturations instead of the nitrifier-specific `KO_A = 0.5`,
  `KNH_A = 1.0`, `KHCO_A = 0.0005`; the poly-P storage dropped its maximum-ratio
  term, restored to `(Kmax_PAO − XPP/XPAO)/(KiPP_PAO + Kmax_PAO − XPP/XPAO)` with
  `Kmax_PAO = 0.2`; and the PP-uptake PO₄ term used the nutrient `KPO4_H` rather
  than `KPO4_PP = 0.2`. Added the standard `clip_negative_states` clamp.)*
- `adm1` — Anaerobic Digestion Model No. 1 (Batstone et al. 2002), BSM2
  implementation form (Rosen & Jeppsson 2006). 29 states (26 liquid + 3 gas
  headspace), 25 processes: disintegration, the three hydrolyses, seven
  substrate-uptake reactions (with pH / hydrogen / free-ammonia /
  inorganic-N inhibition, the lower-pH inhibition via the new `pHInhibitNode`),
  seven biomass decays, and the gas headspace (kLa transfer of H₂/CH₄/CO₂ plus
  overpressure-driven biogas outflow). The gas-transfer headspace-gain factor is
  the symbolic `V_liq / V_gas` (the liquid lost to transfer raises the headspace
  concentration by that ratio), not a baked-in constant; `ADM1DigesterUnit`
  slaves the `V_liq` parameter to its own liquid volume, so the gas transfer is
  correct for any digester size (the network default is the BSM2 3400/300 = 11⅓).
  Inorganic-carbon and -nitrogen
  stoichiometry are symbolic parameter-expressions (the ADM1 elemental
  balances), so a calibrated yield/composition flows through them. The full
  biochemical stoichiometry (all 19 reactions, every coefficient including the
  inorganic C/N balances) is **verified identical to the official BSM2
  `adm1_ODE_bsm2.c`** and the kinetic / equilibrium constants to
  `adm1init_bsm2.m`. *(Fixed: the disintegration and biomass-decay
  inorganic-nitrogen coefficients had dropped their protein term `f_pr_xc·N_aa`
  and their `N_bac − N_xc` release respectively — a transcription error in the
  original YAML, surfaced by the results-level mass balance below and corrected
  against the official `reac11`; the digester now conserves nitrogen exactly.
  Also corrected the carbonate Ka1 van't Hoff enthalpy in the shared pH solver,
  5200 → 7646 J/mol, to the BSM2 / literature value — its absence had biased the
  inhibition-sensitive acetate methanogens.)* pH is
  **state-derived** through the charge-balance `speciation:` solver (extended to
  the four ADM1 volatile fatty acids), with the strong-ion difference carried by
  explicit conservative `S_cat`/`S_an` ion states (via the solver's
  `strong_cations`/`strong_anions` terms); the free-ammonia and dissolved-CO₂
  pH-switch fractions therefore track the instantaneous state. This is the
  complete ADM1 in BSM2 form, **validated** against the published BSM2
  open-loop steady-state digester: run as the benchmark CSTR (3400 m³ liquid /
  300 m³ headspace, fed at ~178 m³/d, HRT ~19 d) **with the exact published feed
  and reference** (the "ADM1 influent (post ASM2ADM interface)" and "ADM1
  effluent" tables of the official `Results/BSM2_steady_state.pdf`) it reproduces
  the reference steady state to within **~1.5% on every state** (the ~1% residual
  is the difference between the charge-balance pH solver and the reference DAE).
  Methane (the defining output) matches to <1%; the charge-balance pH (~7.26)
  matches the reference electroneutrality relation
  (`tests/validation/test_adm1_bsm2_steadystate.py`). *(The earlier standalone
  test used a slightly mis-transcribed feed — `S_aa`/`X_pr` ~6%/2% high — which
  alone accounted for a ~5% steady-state offset; the model itself matches a
  faithful port of `adm1_ODE_bsm2.c` to <1%.)*
  Note: the BSM2 init file
  lists hydrolysis `k_hyd = 0.3` d⁻¹, but that is inconsistent with its own
  steady state (which needs ~10, the canonical value, for the observed ~99.5%
  particulate conversion at HRT 19 d); the network ships `k_hyd = 10`.
- `wats_sewer` — the **original reference-book WATS model** (Hvitved-Jacobsen,
  Vollertsen & Nielsen 2013, process matrices Tables 9.1–9.4): aerobic/anoxic/
  anaerobic heterotrophic carbon turnover (growth bulk+biofilm, endogenous
  maintenance, fast/slow hydrolysis in all three redox regimes, fermentation)
  plus the sulfur cycle — sulfate reduction (H₂S formation, Eq 6.24) and chemical
  + biological sulfide oxidation to sulfate (bulk Eqs 5.13–5.15; biofilm Eq 5.16).
  State-derived (charge-balance) pH. 34 reactions, 15 species. This is the base
  model **without** any nitrate-dosing / methane / elemental-sulfur extensions;
  `wats_sewer_extended` and the Khalil paper/thesis models build on it. (The two
  authored sulfur-cycle rate constants `a_h2s`, `k_sox_f` are flagged PROVISIONAL
  pending confirmation against the book parameter tables.)
- `wats_sewer_extended` — extended WATS sewer-process model (carbon/sulfur/nitrogen
  turnover with a two-step sulfide→S⁰→sulfate cycle and nitrate-driven sulfide
  control). The full reaction structure (46 reactions) covers heterotrophic
  growth, hydrolysis, endogenous maintenance, fermentation, nitrification,
  methanogenesis, sulfate reduction, elemental-S cycling, and pH-dependent
  aerobic + nitrate-driven sulfide oxidation. **Forward integration is hardened
  and stable** (positivity limiter; see *Positivity limiter* below) and
  reproduces the nitrate→sulfide-reduction effect. **Differentiation through
  the full stiff solve** (`A_V≈57` makes the biofilm reactions ~1000 d⁻¹)
  requires capping the integrator step (`dtmax`; see *Differentiating stiff
  networks* below) — with that, `jax.grad`/`jax.jvp` are finite and match
  finite differences. **Validated** against the published batch nitrate-dosing
  experiments: after parameterizing the saturation constants the published study
  calibrated (the reference C hardcoded them) and AD-calibrating the influential
  rates, sulfide/sulfate/nitrate match the measured data well; VFA is the weak
  point (as in the source study), and the calibration/validation batches favour
  slightly different sulfur kinetics. The batch measurement data and the
  figure/analysis scripts for that study live in a **separate
  paper-reproduction repository** (it imports this library and pins an exact
  commit); they are not shipped here. This library provides only the reusable
  pieces: the network + structural variants, and `calibrate` / `dgsm`. First
  network to use a **state-derived pH** (see *Speciation / state-derived pH*
  below). **Documented deviation from the original published model:** anoxic
  (denitrifying) heterotrophic growth uses the proper lower WATS anoxic yield
  `Y_H,NO3 = 0.30–0.40` (parameter `y_h_anox`, default 0.35) instead of reusing
  the aerobic yield 0.55; `Y_H` is a stoichiometric COD/electron-balance
  coefficient and is fixed from the literature, not calibrated. Calibration
  frees only the **identifiable** rate constants under a **relative** loss; the
  fixed-vs-free split is justified by the Laplace posterior correlation (e.g.
  `k_12_no`/`K_NO_f` are a near-degenerate ratio, so `K_NO_f` is fixed) rather
  than by manual trial — the AD analogue of the source study's pairwise
  calibration.
- `wats_sewer_khalil_paper` — paper-faithful re-implementation of the *published*
  Khalil et al. (2025) sewer nitrate-dosing model. **It is the FULL WATS model
  (Hvitved-Jacobsen et al. 2013, Tables 9.1–9.4: carbon backbone + complete
  sulfur cycle) plus the paper's stated additions and modifications**, not a
  hand-trimmed subset — the paper says the model is "based on the WATS model"
  with extensions. This network (27 reactions, 18 species) includes aerobic +
  anoxic heterotrophic growth (bulk + biofilm), aerobic + anoxic maintenance,
  anaerobic fast/slow hydrolysis, fermentation, methanogenesis, VFA-driven
  sulfate and elemental-sulfur reduction, the two-step nitrate-driven sulfur
  oxidation (paper additions), **and the base-WATS aerobic sulfur cycle**:
  pH-dependent chemical and biological *bulk* sulfide oxidation (Table 9.4) and
  biological *biofilm* sulfide / elemental-S oxidation. The **aerobic backbone
  and the aerobic sulfur-oxidation pathway are O2-gated and therefore dormant**
  under the air-sealed anoxic batch (`S_O ~ 0`) — they carry zero rate and do
  not change the batch trajectories, but make the network the structurally
  complete WATS model. Paper modifications: half-order WATS biofilm terms →
  Monod (so the aerobic biofilm oxidation is Monod-ized); no temperature
  correction (batch at 20 °C); pH supplied as a **fixed operating condition**
  (not the charge-balance solver); removes chemical oxidation of sulfide *by
  nitrate* (anoxic oxidation is biological only). Carries the full WATS species
  vector; the N/P/inorganic-carbon/inert/autotroph pools are present but largely
  inert in the batch. The paper-active core is augmented with the dormant
  full-WATS aerobic pieces by
  [`networks/_make_khalil_paper.py`](aquakin/networks/_make_khalil_paper.py)
  (comment-preserving ruamel splice from `wats_sewer_extended.yaml`). Ships with
  structural variants (`_halforder`, `_directsulfate`, `_srbsubstrate`,
  `_combined`, and the standalone falsification variant `_stopatS0` — the
  nitrate-driven oxidation stops at elemental sulfur, the S⁰→sulfate step
  removed, which is mutually exclusive with `_directsulfate` and so is NOT folded
  into `_combined`) generated reproducibly by
  [`networks/_make_khalil_variants.py`](aquakin/networks/_make_khalil_variants.py).
  That generator now produces the same structural variants for **both** the
  faithful base and the `_balanced` base (`wats_sewer_khalil_paper_balanced_{halforder,
  directsulfate,combined}` — the balanced base already makes the `srbsubstrate`
  change by design, so it is omitted there). The `directsulfate` one-step nitrate
  demand is computed from the base's own two-step coefficients (so it stays
  electron-balanced on either base, e.g. −0.70 on the corrected base, not the
  old hard-coded −0.75), and the `srbsubstrate` elemental-S reduction donor
  coefficient is −0.5 gCOD/gS (X_S0 at COD 1.5 → sulfide at COD 2); both fixed
  pre-existing variant-stoichiometry COD imbalances, now guarded by
  `tests/integration/test_mass_balance.py` (all variants are in `_MODELS`). The
  balanced variants conserve COD/S/Fe/N; the faithful variants carry the base's
  VFA throttle (excluded from the COD check).
  The half-order variants' square-root kinetics need a tighter `dtmax` (≈1e-4)
  for the reverse-mode adjoint to stay finite. Built for the JRN-055
  identifiability study, which finds the elemental-sulfur oxidation rate and the
  unmeasured initial carbon pools to be non-identifiable (they trade off through
  the shared dosed-nitrate budget). This network is the **faithful** reproduction
  of the published model: it carries the reconstruction-bug fixes (electron-
  balanced sulfur stoichiometry; the Table-2 `X_Sn/X_HW` hydrolysis form) but
  **not** the extensions the published model omits — so, like the paper, it does
  not track nitrogen and has no iron chemistry.
- `wats_sewer_khalil_paper_balanced` — the mass- and electron-balanced
  "improved" counterpart of `wats_sewer_khalil_paper`, for running side by side
  to quantify what the corrections buy. Same kinetics, plus two
  conservation-restoring extensions the published model omits: (1) **FeS
  precipitation** (ferrous iron precipitates dissolved sulfide as solid FeS;
  Nielsen et al. 2005 / SeweX-SUMO chemistry — Khalil's own iron chemistry is
  commented out, leaving the measured effluent sulfide below the reduced-sulfate
  level), and (2) **full nitrogen/phosphorus tracking** (ammonia uptake on
  growth, release on hydrolysis, ammonification on biomass decay). It is the only
  Khalil network that conserves all of COD, S, Fe and N. Generated by
  [`networks/_make_khalil_balanced.py`](aquakin/networks/_make_khalil_balanced.py)
  from the faithful model.
- `wats_sewer_khalil_thesis` — the same complete WATS model + nitrate-dosing
  additions as specified in Khalil's *thesis* (the base WATS model of thesis
  Ch. 3 plus the Table 4-1 additions): the full WATS process matrix
  (Hvitved-Jacobsen Tables 9.1–9.4 — aerobic/anoxic/anaerobic carbon turnover
  plus the sulfur cycle, including the chemical/biological **aerobic** sulfide
  oxidation with **half-order** biofilm kinetics) extended with methanogenesis,
  elemental-sulfur reduction and the two-step nitrate-driven sulfur oxidation.
  Generated by [`networks/_make_khalil_thesis.py`](aquakin/networks/_make_khalil_thesis.py)
  from `wats_sewer_extended.yaml` by dropping the charge-balance pH solver (→ a fixed
  operating-pH condition, since the thesis uses a fixed pH) and
  nitrification/autotrophs, and reverting parameters to thesis/paper values
  (single yield 0.55; faster hydrolysis `k_h1=12`, `k_h2=5`; `q_ferm=2`). Used to
  test whether the published-fidelity model reproduces the batch nitrate-dosing
  data: it matches sulfide well but not VFA/sulfate, consistent with the
  published curves coming from the continuous 4-CSTR runs rather than an
  isolated batch. Now that `wats_sewer_khalil_paper` is also the full WATS model
  (same 18-species state, same aerobic sulfur-oxidation branch, fixed pH), the
  two differ chiefly in **half-order (thesis) vs Monod-ized (paper) biofilm
  kinetics**, the temperature correction (thesis keeps `theta^(T-20)`; paper
  omits it), and the parameter values — i.e. the paper model is essentially the
  thesis model with the `_halforder`→Monod structural choice and paper Table-3
  parameters.
- `precipitation_struvite_calcite` — mineral **precipitation/dissolution** from
  an anaerobic-digester supernatant: struvite (MgNH₄PO₄) and calcite (CaCO₃),
  the worked example of the `precipitation:` block (generalised precipitation
  framework, Kazadi Mbamba et al. 2015). Each mineral declares its constituent
  ions, pKsp and supersaturation order; the engine computes the saturation index
  `SI_<name>` and the SI-driven rate factor `R_<name> = sign(σ)·|σ|^n` from the
  free-ion activities at the operating pH, and a precipitation reaction reads
  `{R_<name>}` to consume the ions / form the solid (or dissolve it when
  undersaturated). State is in mol/m³ so the stoichiometry is exact; pH is a
  fixed condition here, but a `speciation:` block can instead supply a
  charge-balance pH the precipitation reads (the two derived functions compose).
  See *Mineral precipitation / state-derived saturation* below.
- `precipitation_metal_phosphate` — **chemical phosphorus removal** by ferric /
  aluminium dosing: the metals precipitate orthophosphate as the very insoluble
  FePO₄ / AlPO₄ while competing to form the hydroxides Fe(OH)₃ / Al(OH)₃ (the
  `hydroxide` ion fraction, OH⁻ = Kw/[H⁺]). The hydroxide buffers the free metal,
  giving a pH-dependent floor on the achievable phosphate (removal worsens at
  higher pH). With the **default power-law** kinetics this is a
  **forward-simulation** demonstration: ferric/aluminium phosphates are so
  insoluble (`SI ~ 14`) that the dose transient, while finite to solve, is not
  differentiable by any sensitivity method (the `~1e13` rate Jacobian). Two opt-in
  variants restore differentiability (issue #295) — see *Mineral precipitation /
  state-derived saturation* below:
- `precipitation_metal_phosphate_equilibrium` — the same chemistry with every
  mineral `mode: equilibrium`: the precipitation **equilibrium** is solved
  *algebraically* (`IAP = Ksp` with complementarity, mass-balanced across the
  shared metal) and exposed via `network.precipitation_equilibrium(...)`, the
  differentiable equilibrium-projected state. The principled
  "solve-don't-integrate" reduction (what geochemistry codes do) — exact, fast,
  and `jax.grad`-clean, so `calibrate` / `sensitivity` flow through the
  equilibrium outcome. The equilibrium reproduces the kinetic `t→∞` limit (ferric
  ripens from FePO₄ to the more stable Fe(OH)₃).
- `precipitation_metal_phosphate_bounded` — the same chemistry with every mineral
  `supersaturation_form: bounded`: the kinetic driver is the bounded
  `R = tanh(SI/(2ν)·ln10)` instead of the power law, so the rate Jacobian is `~k`
  (non-stiff) and a **dynamic** reactor/plant solve is differentiable, relaxing to
  the same equilibrium. The reaction expressions are unchanged (`k·X·{R}`).

### Khalil model-improvement sequence (JRN-055 reproduction log)

This is the chronological record of structural corrections applied to the
faithful published model on the way to `wats_sewer_khalil_paper_balanced`, so
the provenance is not lost (and so the paper can narrate it). Each entry is a
*structural* change and what it bought, with the diagnostic that motivated it.
The faithful model (`wats_sewer_khalil_paper`) is frozen as the published
baseline; every correction below lives only in the `_balanced` model and its
generator [`networks/_make_khalil_balanced.py`](aquakin/networks/_make_khalil_balanced.py).

1. **Hydrolysis saturation form** (faithful + balanced). The first
   reconstruction used the substrate *fraction* `X_S1/(X_S1+X_S2)`; the WATS
   book (Eq 5.4) and Khalil Table 2 use the substrate/biomass *ratio*
   `X_Sn/X_HW` → `monod_ratio([X_Sn],[X_BH],k_xn)`. Fixed in all WATS models.
2. **Electron-balanced sulfur stoichiometry** (faithful + balanced). The
   COD-conservation check (`tests/integration/test_mass_balance.py`) flagged the
   aerobic sulfide-oxidation O2 demand (−0.8/−1.62/−0.42/−1.22 → −2.0/−2.0/
   −0.5/−1.5), elemental-S reduction (−0.75 → −0.5), and nitrate demand
   (−0.225 → −0.175 gCOD/gN·N). These are reconstruction-bug fixes carried by
   *both* models — they make the published stoichiometry self-consistent
   without changing its structure.
3. **FeS precipitation** (balanced only). Khalil's own iron chemistry is
   commented out in his code, leaving measured effluent sulfide *below* the
   reduced-sulfate level — an open sulfur balance. Added `S_Fe2 + sumS → X_FeS`
   (Nielsen et al. 2005 / SeweX-SUMO; Fe 56/88, S 32/88). Closes the S balance.
4. **Full nitrogen / phosphorus tracking** (balanced only). The published model
   omits ammonia uptake on growth, release on hydrolysis, and ammonification on
   decay, so it does not conserve N. Restoring the WATS `i_n_bio`/`i_p_bio`/
   `i_n_xb` terms makes the balanced model the only Khalil network conserving
   all of COD, S, Fe and N.
5. **SRB consume fermentable substrate `S_B`, not VFA** (balanced only). The
   peer review of the original submission (Reviewer 1 and co-author Batstone)
   called Khalil's VFA-only sulfate/elemental-S reduction "simply wrong":
   sulfate-reducers grow on fermentable substrate, and VFA (acetate)
   *accumulates* as a fermentation byproduct, matching the observed VFA plateau.
   The balanced model rewires `*_VFA_biofilm` → `*_SB_biofilm` (rate and
   stoichiometry on `[S_B]`). Raises continuous-cascade VFA from ~15 toward the
   measured ~35. **Note:** the published *revision* did not adopt the reviewer
   comments (the paper was rejected on that round and accepted elsewhere without
   the change), so this correction is ours, not Khalil's.
6. **Reverted Khalil's undocumented anoxic-VFA-uptake throttle** (balanced
   only). His *code* (not the paper) scales anoxic VFA uptake by `0.01/Y`
   instead of `1/Y`, letting denitrification consume nitrate without the
   matching VFA — which does *not* conserve COD. The faithful model reproduces
   this throttle (and is excluded from the COD check on those two reactions, with
   `test_faithful_reproduces_khalil_nonconserving_throttle` documenting the
   contrast); the balanced model reverts to the standard `1/Y` and conserves COD.
7. **Biofilm-stored substrate as the missing COD-conserving donor**
   (diagnostic, not yet a shipped structural change). Without the throttle, the
   no-cheat balanced batch is ~30 mgCOD short of donor: nitrate stalls (~8.7
   mgN residual) because the cascade's bulk particulate pool cannot supply the
   electrons the throttle previously faked. Freeing the batch initial `X_S2`
   (the slow-hydrolysis reservoir, representing substrate stored in the biofilm
   at the start of the pump-off batch) **closes the nitrate balance
   COD-conservingly** — nitrate goes 30 → 0, sulfide RMSE 1.18 → 0.96 — so the
   donor-deficit diagnosis is correct and the throttle was standing in for a
   real biofilm reservoir. **But** the reservoir is non-identifiable from this
   batch alone (the fit rails `X_S2` to its 500 mg upper bound), and it does
   *not* fix the two remaining misfits: VFA still crashes to ~0 (measured
   plateaus ~13) and sulfate still plateaus at ~13 (measured rises to ~22).
   The residual misfit has therefore *moved* from the donor budget to the
   **nitrate-driven sulfur-oxidation pathway**.
8. **Biofilm-stored elemental sulfur `X_S0` as the sulfate-rise reservoir**
   (diagnostic, parallel to 7). The nitrate-driven oxidation runs
   `sumS → X_S0 → S_SO4` at 0.70 mgN per mgS fully oxidized; with only the 9
   mgS of initial *bulk* sulfide the model can raise sulfate at most to
   4 + 9 = 13 (exactly where it plateaued), while the data rises to ~22. Sewer
   biofilms store reduced sulfur, so — exactly as for the organic substrate —
   freeing the batch initial `X_S0` supplies it. Unlike `X_S2`, this reservoir
   is **identifiable** (the fit settles at ~18 mg, it does not rail), and it
   confirms the sulfur-cycle diagnosis: sulfate now tracks the measured rise
   (it actually slightly overshoots to ~30, flipping the error from undershoot
   to overshoot — the *shape* is right), sulfide is excellent including the
   post-nitrate rebound (SRB resume once nitrate clears; model 0→3.6 vs
   measured 2.8), and the joint fit loss drops from ~80 to ~53. **The two
   biofilm reservoirs (organic `X_S2`, sulfur `X_S0`) are the unifying
   structural insight: the pump-off batch starts from a biofilm loaded with
   stored substrate and stored sulfur, both well above the bulk
   concentrations.**
9. **The residual VFA misfit is a structural donor-budget constraint, not a
   tunable/preference problem** (investigated; two candidate fixes tested and
   *rejected*). After 7–8 the only badly-fit target is VFA: `anox_growth_VFA`
   draws acetate to zero during dosing, while the measured VFA *plateaus* at ~13
   (even rising 12.4 → 13.9 as nitrate depletes) — the real denitrifiers stop
   consuming acetate well before it is exhausted. Two mechanisms were tested:
   - **(C) retune the existing kinetics.** Freeing the *shared* anoxic-growth
     knobs (`eta_no`, `k_sw`, `k_sf`) cannot spare acetate: the optimizer drives
     `eta_no` to a physically impossible ~242 (a dimensionless anoxic factor
     that should be ≤1) and VFA gets *worse*, because the published structure
     treats `S_B` and acetate as **interchangeable** electron donors, so faster
     denitrification consumes acetate faster, not slower.
   - **(B) add a principled `S_B`-preference term** `monod_inh([S_B], k_pref)`
     on the acetate-uptake rates (denitrifiers prefer fermentable substrate,
     use acetate only as `S_B` depletes). The fit **robustly rejects it**:
     `k_pref` rails to its no-preference limit (~403) from *both* a weak-term
     and a strong-term start, with identical loss and unchanged VFA. (The term
     was implemented in `_make_khalil_balanced.py`, tested, and then reverted —
     it does not ship.)

   The reason both fail is the same and is the real finding: **the model
   *requires* acetate as an electron donor to consume the dosed nitrate at the
   observed rate.** Sparing acetate stalls nitrate (a larger relative-loss
   penalty than the VFA gain), and a preference switch is inert anyway because
   `S_B` is itself depleted during dosing. The real system reaches nitrate → 0
   *without* exhausting acetate, so it must have a faster **non-acetate
   electron-donor flux** than this structure supplies — i.e. faster `S_B`
   regeneration (hydrolysis/fermentation of the biofilm reservoir) and/or a
   larger sustained reduced-sulfur cycle. That donor-flux question — not acetate
   kinetics — is the open lever, and the inability to fit VFA without breaking
   nitrate is itself a clean structural-identifiability result for the study.
10. **Resolution — a model that fits all four targets: rapidly-mobilized
    biofilm storage (organic + elemental sulfur).** The donor-flux question of 9
    is resolved by recognizing the batch electron donor is **biofilm-stored
    substrate released rapidly at pump-off**, not bulk slow hydrolysis. The
    decisive move is to *release the bulk-literature hydrolysis prior* on `k_h2`:
    pinned near literature (~1, capped at ~18 by the prior at a 5.6σ penalty) the
    model cannot build up `S_B` and acetate crashes; freed, `k_h2` jumps to
    ~10²–10³ d⁻¹, which dumps the `X_S2` reservoir (~100 mg) into `S_B` (spikes
    to ~70–100 mg). Now denitrifiers consume *abundant* `S_B` instead of acetate,
    so **VFA holds a plateau** (RMSE ~2.2, better than the paper's 3.06) — exactly
    the regeneration mechanism 9 predicted. With fast `S_B`, however, the sulfur
    pathway is out-competed for nitrate and sulfate undershoots **unless** the
    stored elemental-sulfur reservoir `X_S0` is present; the combined fit
    (fast hydrolysis + fixed `X_S0` ≈ 12 mg + freed nitrate-driven sulfur-ox
    rates) matches **all four** series: sulfide 0.45, sulfate 3.96, VFA 2.15,
    nitrate 2.65 (paper: 0.52 / 2.35 / 3.06 / 1.51) — comparable to or better
    than the published curves. **Key identifiability findings (the paper's
    point):** (a) the batch's organic donor is *biofilm-mobilized stored
    substrate*, requiring a hydrolysis/mobilization rate ~20× the bulk WATS value
    — bulk hydrolysis kinetics do not transfer to the pump-off batch; (b) the
    stored elemental-sulfur pool `X_S0(0)` is **practically identifiable** — a
    profile likelihood (re-optimizing all other free params at each fixed
    `X_S0(0)`) is a clean bowl with a minimum at ~7 mgS and a 95% CI ≈ [~1, ~18]
    mgS; at the optimum sulfate RMSE is ~2.0 (better than the published 2.35).
    The earlier observation that a *gradient* fit "trades `X_S0` away" to ~1 mg
    was a **local-minimum artifact** (the joint landscape is multimodal — the
    nitrate competition makes nudging `X_S0` up locally worse before it helps);
    the profile escapes it and shows the data *do* locate the biofilm sulfur
    inventory, so fixing `X_S0(0)` near the optimum is recovering a data-preferred
    value, not a convenient hack. **But the identifiability is conditional on a
    supra-measured oxidation rate.** A second profile with the sulfur-oxidation
    rate *pinned at its measured value* (k_sII=17.1, k_s0=2.2) is **monotonic, not
    a bowl**: the loss is minimized at `X_S0(0)`≈0 and rises as the pool grows,
    and sulfate cannot be matched for any pool value (RMSE ≥ 4.5 vs ~2.0 at the
    free rate). So at the measured rate the stored pool does *not* rescue sulfate —
    the **oxidation rate is the binding lever**, and `X_S0(0)` helps only in
    concert with a supra-measured rate. This **confirms and strengthens the
    paper's existing thesis** (sulfate needs a non-identifiable supra-measured
    rate): even the most plausible extra physical lever cannot fix sulfate at the
    measured rate. What genuinely improves is **VFA** (the other holdout),
    resolved by the biofilm-mobilization mechanism. JRN-055 framing decision:
    *refine* the existing thesis (keep the cautionary spine; add VFA-resolution +
    the rate-coupled stored-S pool). (c) VFA and sulfate are **coupled through the shared dosed-nitrate budget** —
    matching both requires the sulfur pathway to claim its ~13 mgN early (fast,
    high-affinity nitrate-driven oxidation of the stored sulfide+`X_S0`) before
    organics consume it. Net: the published WATS structure + bulk kinetics cannot
    reproduce the batch (items 7–9), but adding two rapidly-mobilized biofilm
    storage pools — organic `X_S2` and elemental sulfur `X_S0` — is necessary and
    sufficient, with `X_S0` constrained rather than fit. (This is a calibration /
    initial-condition configuration on the existing balanced network, explored in
    the JRN-055 batch-fit scripts; it is *not* a new shipped reaction.)

The four ASM networks `asm2d` / `asm2d_tud` / `asm3` / `asm3_biop` were
**originally derived** from WastewaterAD's SUMO model export
(`wastewaterad.tools.sumo_import`) and are now **maintained directly as YAML**
(the one-shot import converter has been retired). Stoichiometric coefficients
that depend on yield / N-content / fraction parameters are **live symbolic string
expressions**, re-evaluated from the current parameter vector every `solve()`
(via `CompiledNetwork.compute_stoich`), so calibrating `Y_H` / `i_XB` / `f_P` and
the like propagates to every dependent coefficient — matching GPS-X/BioWin/SUMO,
which keep the Gujer matrix live-coupled. Only cells with no parameter leaf are
numeric literals. Each YAML carries real units that follow the network's own
structure — half-saturation constants the currency of the species their Monod
term limits (so `check_units` always matches), rate constants `1/d` (the
second-order precipitation constant `kPRE` is `m3/(g_TSS*d)`), species the
parseable `g_<currency>/m3` form (oxygen → `g_O2/m3` per the `asm1` convention),
and the stoichiometry-only yields / composition / charge constants best-effort
advisory units. All four resolve `network.time_unit` to `"d"` and are
`check_units`-clean **except** the two ASM2d-TUD biomass-normalised PP-storage
rates (`qPP·XPAO²/XPP·…`), whose root is irreducibly cross-currency (`COD²/P`) —
a documented property of that model's rate form, surfaced (not fixed) by the root
check.

**COD/N/P continuity (Gujer-matrix conservation) on the ASM family** is checked
by `tests/integration/test_asm_continuity.py` (issue #210): for each of asm1,
asm2d, asm2d_tud, asm3, asm3_biop, every reaction conserves COD, N and (where
modelled) P against a per-species composition vector, evaluated through the live
`compute_stoich` (so the symbolic coefficients are exercised, and a yield
calibration is shown to keep COD closed). Nitrate carries the NH4-referenced COD
`iCOD_NO3 = -4.571` g COD/g N; the N2-tracking models give `SN2` the COD
`iCOD_NO3 + iNO3_N2` so denitrification closes, while ASM1 (no N2 state) excludes
its single denitrification reaction from the COD check and the N balance credits
the gas via `check_nitrogen`. **This test caught a real coefficient error:** the
asm2d / asm2d_tud heterotroph anoxic-growth nitrate/N2 coefficient was
`(1-YH)/iNO3_N2*YH` — which evaluates as `(1-YH)·YH/iNO3_N2` — instead of the
published `(1-YH)/(iNO3_N2·YH)` (Henze et al. 2000, STR No. 9: `(1-YH)/(2.86·YH)`),
breaking COD continuity by ~0.37/unit-rate. Root cause: in the original SUMO
import a denominator product `a/(b·c)` had been written `a/b·c` (a missing
parenthesis on the right operand of a left-associative `/` or `-` at equal
precedence); corrected in the two YAMLs. The legitimate `(-1/YH)·iN_SF`-style
coefficients are genuine `(a/b)·c` and are unchanged.

Future networks include UV/TiO₂ and chlorine decay.

**The library is a standalone scientific contribution.** It is not tied to any
specific application project.

---

## Design Goals

- Reaction networks defined at runtime via YAML — no recompilation required
- Full automatic differentiation (AD) throughout via JAX
- JAX-native stiff ODE integration via Diffrax
- Clean separation between flow solver and kinetic system
- Safe, introspectable rate expression evaluation via AST (no `eval()`)
- Graduate-student-friendly authoring experience for network files

---

## Technology Stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python | Flexibility, ecosystem |
| Numerical backend | JAX | AD for free, jit compilation |
| ODE integration | Diffrax | JAX-native stiff solvers, adjoint support |
| Schema validation | Pydantic | Clear load-time errors for malformed YAML |
| Runtime data model | Python dataclasses | JAX-friendly, no Pydantic overhead at runtime |
| Network format | YAML | Human-readable, supports inline comments for literature citations |
| Rate expression evaluation | Custom AST + recursive descent parser | Safe, differentiable, introspectable |

---

## Architecture

### Two-Layer Data Model

**Layer 1 — Pydantic schema (load time)**
Parses and validates YAML network files. Produces clean Python objects with
clear error messages for malformed input. Pydantic is used only during loading
— it never appears in the runtime hot path.

**Layer 2 — Compiled runtime (CompiledNetwork dataclass)**
JAX-friendly dataclasses built from the validated schema via a `compile()`
step. This is what the integrators and rate functions operate on. No Pydantic
dependency at runtime.

### Core Data Flow

```
YAML file
   ↓  loader.py (Pydantic validation)
NetworkSpec
   ↓  compile()
CompiledNetwork  ←→  SpatialConditions
   ↓
rates(C, params, condition_arrays, loc_idx)
   ↓
Diffrax (BatchReactor / PlugFlowReactor)
   ↓
Solution object
```

### Rate Function Signature

All compiled rate functions share this signature:

```python
rates(
    C: jnp.ndarray,                          # (n_species,) concentration vector
    params: jnp.ndarray,                     # (n_params,) flat parameter vector
    condition_arrays: dict[str, jnp.ndarray],# (n_locations,) per field
    loc_idx: int | jnp.ndarray               # spatial location index
) -> jnp.ndarray                             # (n_reactions,) rate vector
```

The ODE right-hand side is then:

```python
dC_dt = stoich_matrix.T @ rates(C, params, condition_arrays, loc_idx)
```

Rate constants are **always external** to the rate functions (passed via
`params`), never baked in. This is required for AD-based parameter estimation
and sensitivity analysis.

### Stoichiometry Matrix

Shape `(n_reactions, n_species)`. Entry `[i, j]` is the stoichiometric
coefficient of species j in reaction i. Built once at compile time and stored
in `CompiledNetwork`.

---

## AST Rate Expression Evaluation

Rate expressions in YAML (e.g. `"k1 * [O3] * [Br-]"`) are parsed into an
abstract syntax tree (AST) at network load time. The tree is walked once to
produce a JAX-compatible callable. This callable is used at runtime — the
tree itself is not walked repeatedly.

### Node Types

**Leaf nodes**
- `SpeciesNode(name)` — looks up species concentration by index from `C`
- `ParamNode(namespaced_name)` — looks up rate constant by index from `params`
- `ConditionNode(field_name)` — indexes into `condition_arrays[field][loc_idx]`
- `ConstantNode(value)` — literal numeric constant

**Binary operation nodes**
- `AddNode`, `SubtractNode`, `MultiplyNode`, `DivideNode`, `PowerNode`

**Domain-specific function nodes**
- `ArrheniusNode(A, Ea)` — temperature-dependent rate: `A * exp(-Ea / (R * T))`
- `pHSwitchNode(pKa)` — acid/base speciation fraction: `1 / (1 + 10^(pH - pKa))`
- `pHInhibitNode(pH_LL, pH_UL)` — ADM1 lower-pH Hill inhibition `pHLim^n/(S_H^n+pHLim^n)` (stable sigmoid form); 1 at high pH, 0 at low pH. Needs `pH`.
- `MonodNode(X, K)` — saturation Monod: `X / (K + X)`
- `MonodInhibitionNode(X, K)` — inhibition Monod: `K / (K + X)`
- `MonodRatioNode(A, B, K)` — ratio-saturation Monod: `(A/B) / (K + A/B)`
- `MonodInhibitionRatioNode(A, B, K)` — ratio-inhibition Monod: `K / (K + A/B)`
- `SafeDivideNode(num, denom)` — the `safe_div(num, denom)` function: division
  that returns 0 (with a finite gradient) where `denom == 0`, instead of
  `inf`/`NaN`. For a ratio whose denominator can legitimately reach exactly zero
  — a substrate-competition fraction `[A] / ([A] + [B])` where both deplete to 0
  — so the rate takes its physical limit 0 there without padding the denominator
  with a dimensionless epsilon. Used by ADM1's valerate/butyrate C4 competition
  (`safe_div([S_va], [S_va] + [S_bu])`), replacing the old `+ 1.0e-6` guard.
- `MaxNode(a, b)` — the `max(a, b)` function: elementwise maximum, AD-safe
  (`jnp.maximum`, active-branch subgradient at the kink). For a one-sided clip,
  e.g. ADM1's biogas outflow `k_P * max(0, P_gas - P_atm)` so an
  overpressure-driven flux only stops (never reverses) when the driving
  difference goes negative.

The four Monod nodes and `SafeDivideNode` all evaluate their `num/denom` through
`_safe_ratio` (`core/nodes.py`), a **double-where** guard returning 0 (with a
finite gradient) where the denominator is exactly zero — for the Monod nodes the
full-depletion point where the limiting quantity *and* its saturation constant
are both 0 (`K = X = 0`); for `safe_div` wherever the author's denominator hits
0. The physical limit is 0 (no substrate → no rate); a bare `num/denom` is
`0/0 = NaN` there and the naive single `where` still back-propagates a NaN
through the masked branch, so the denominator is guarded too. Identity for any
nonzero denominator — the only change is exactly at the singularity.

New domain-specific node types are added here as needed. Each node implements:

```python
def compile(self, ctx: CompileContext) -> Callable:
    """Returns a JAX-compatible callable: (C, params, condition_arrays, loc_idx) -> scalar"""
```

### Vectorized rate kernel (`core/vector_kernel.py`)

Evaluating the rates as `jnp.stack([f(C, params, ...) for f in rate_callables])`
traces **one nested closure tree per reaction**, so the jaxpr holds
`O(reactions x ops-per-reaction)` scalar primitives — each leaf a `slice` +
`squeeze`. XLA fuses them so the *runtime* is fine, but its optimization passes
scale with that op count, so the **compile** is dominated by it — and amplified
in the reverse-mode adjoint, where the RHS jaxpr is differentiated ~80x per
step (the multi-minute discrete-adjoint compiles). Compile, not run, is the cost
of a stiff solve (~1.6 s compile vs ~0.02 s run for an ASM1 batch), so this
*compile-time* lever is what hurts the test suite, where each test compiles a
distinct configuration once.

`build_vectorized_rates` builds, from the per-reaction rate ASTs, a single
callable returning the `(n_reactions,)` rate vector by **interning every
distinct subexpression** (node type + operand positions) and evaluating **all
instances of each primitive in one batched elementwise op**, in topological
order — global common-subexpression elimination plus vectorization by node type.
The pool is built **append-only by concatenation** (1 jaxpr op each, vs ~5 for a
scatter incl. index fixup), and operands are read with a **raw `lax.gather`**
(skips the negative-index normalization `slice`/`squeeze`/`select_n` that
`P[idx]` inserts). The traced op count collapses to `~O(node-types x depth)`,
independent of the reaction count.

It is **bit-identical** to the scalar path: each interned instance is a lane of
a batched op doing the identical scalar arithmetic, and IEEE elementwise ops are
deterministic per lane; the interning dedups identical subexpressions (also
bit-identical, since the scalar path recomputes them to the same bits). So it is
*not* the issue's masked-`jnp.prod` assembly (which would change the product
reduction order); the order-preserving batched ops need no revalidation rebasing.
Built once in `CompiledNetwork.__post_init__` (`_rate_kernel`) and dispatched
from `CompiledNetwork.rates` after the clip / derived-condition / temperature
preprocessing (which is unchanged); a future AST node type with no batched
kernel raises `UnsupportedNode` and `rates` falls back to the scalar stack, so
the kernel is a safe, transparent overlay. **Measured:** the rate jaxpr is the
dominant term in a differentiated stiff-solve compile (60% asm1 → 85% adm1 →
94% wats_sewer_extended), and the kernel cuts that jaxpr ~2–12x (asm1 2.9x,
adm1 2.2x, asm2d 10x, wats 5.6x), giving an end-to-end differentiated-compile
speedup of ~1.15x (asm1) / 1.5x (adm1) / 3.4x (wats) — larger on the bigger
networks, where the suite hurts most. Runtime is unchanged. Regression-guarded
(bit-identicality on randomized states, op-count reduction, AD parity, and the
unsupported-node fallback) in `tests/unit/test_vector_kernel.py`.

### Parser

Recursive descent parser. No external parser dependencies. Operator precedence
(lowest to highest): `+/-`, `*/`, `**`, unary, primary. Species names are
delimited by square brackets: `[O3]`. Rate constant names are namespaced
automatically: `k1` in reaction `O3_Br_direct` becomes `O3_Br_direct.k1`.

### CompileContext

Carries index maps used during AST compilation:

```python
@dataclass
class CompileContext:
    species_index: dict[str, int]
    param_index: dict[str, int]      # namespaced keys -> flat vector indices
    condition_fields: set[str]       # valid condition field names
```

---

## Parameter Namespacing

Rate constants are locally named in the YAML (`k1`, `k2`) but internally
namespaced as `reaction_name.k1`. This allows natural YAML authoring while
maintaining unambiguous global parameter indexing.

Users always interact with namespaced names in the API:

```python
free_params=["O3_Br_direct.k1", "O3_OBr_oxidation.k2"]
```

The flat `params` vector and its index map (`param_index`) are built once at
compile time and stored in `CompiledNetwork`.

---

## SpatialConditions

Conditions (pH, temperature, UV fluence rate, etc.) are spatially varying by
design. They are passed as full JAX arrays so that indexing occurs inside the
JAX trace — this is required for differentiating with respect to condition
fields.

```python
@dataclass
class SpatialConditions:
    fields: dict[str, jnp.ndarray]   # each array shape (n_locations,)
```

The rate function indexes into these arrays using `loc_idx`. When `loc_idx` is
a traced JAX value (e.g. inside `vmap`), JAX promotes the indexing to a gather
operation automatically.

`SpatialConditions.uniform(n_locations, **kwargs)` is a convenience constructor
for spatially homogeneous conditions.

**`OperatingConditions` — the 0-D alias.** A single stirred tank has no spatial
location, so the spatial array model reads as over-machinery for the most basic
setup. `OperatingConditions(pH=7.5, T=293.15)` (exported as
`aquakin.OperatingConditions`) is a `SpatialConditions` subclass specialised to
one location, built directly from scalar field values — it **is** a
`SpatialConditions`, so it works unchanged in every reactor; use the base
`SpatialConditions` for a spatially varying PFR/CFD case. The 0-D examples and
the README quickstart lead with it.

**`conditions.with_(**overrides)` — edit from defaults.** Returns a copy with the
named fields overridden (or added), scalars broadcast to the object's location
count, the rest carried over, the original untouched. The recommended
edit-from-defaults pattern is `network.default_conditions().with_(T=283.15)`
(start from the YAML-declared defaults, change only what differs). It always
returns the base `SpatialConditions` type.

---

## ODE Integration Strategy

### Solver

**Diffrax** is the JAX-native ODE library used throughout. Default solver for
stiff systems: `Kvaerno5`. Adjoint sensitivity (`diffrax.RecursiveCheckpointAdjoint`)
is the default for parameter estimation — memory-efficient for long integration
times typical of reactor contact times.

**Adjoint choice and forward-mode AD.** Every reactor takes an `adjoint=`
argument. The default `RecursiveCheckpointAdjoint` is **reverse-mode only**: it
registers a `custom_vjp`, so `jax.grad`/`jacrev` work but `jax.jvp`/`jacfwd`
(forward mode) are rejected with *"can't apply forward-mode autodiff (jvp) to a
custom_vjp function"*. When you need forward-mode AD through the solve — e.g. a
forward-mode sensitivity Jacobian or a Gauss–Newton/Fisher matrix — construct the
reactor with `adjoint=diffrax.DirectAdjoint()` (or the dependency-free alias
`aquakin.forward_adjoint()`), which is plainly differentiable
in both modes. Its drawback is *usually* memory (it stores/unrolls the whole
solve, cost growing with step count), so keep `RecursiveCheckpointAdjoint` as the
default and switch to `DirectAdjoint` only when forward-mode is actually required.
**The plumbing is now hidden for the common consumers:** `calibrate` and
`sensitivity` take `ad_mode="forward"|"reverse"` and build the right reactor
adjoint internally (no `diffrax` / `adjoint=` in user code); `calibrate` also has
`check_finite=True` to turn a non-finite stiff gradient into a friendly error with
the remedy. **For a user who rolls their own loss + optimizer through
`reactor.solve` (outside `calibrate`/`sensitivity`),** the same silent-non-finite
reverse-gradient footgun is exposed and nothing raises. Every reactor therefore
carries `check_gradient_finite(grad_value, what=...)` (the `GradientCheckMixin` in
`integrate/_common.py`): wrap a freshly computed `jax.grad` in it
(`g = reactor.check_gradient_finite(jax.grad(loss)(p))`) to convert a silent
`NaN`/`Inf` into an actionable error whose remedy is tailored to whether the
reactor already caps `dtmax`. The underlying free checker `check_finite_gradient`
is exported at the package level too (`aquakin.check_finite_gradient`) for guarding
a gradient computed without a reactor handle. `dgsm` takes `ad_mode=` too but cannot set the adjoint for you (your
`fn` constructs the reactor), so its `fn` still needs
`adjoint=aquakin.forward_adjoint()`. And `Plant.solve(gradient="auto")` (the
default) auto-routes a differentiated stiff plant to the cap-free `stable_adjoint`
while keeping a plain forward solve on the fast cached path — so a plant gradient
is finite by default with no `dtmax` to tune.
`dgsm(..., ad_mode="forward")` is the first-class consumer of this (see the DGSM API
below): for a **multi-output sensitivity screen of a stiff network** forward mode
can be *faster and lighter* than reverse — the reverse adjoint is paid once per
output and is inflated by the `dtmax` step cap, whereas forward pushes all `d`
tangents through one solve, independent of the output count. (Benchmarked at
~2× faster, less memory, for the 4-output/17-input Khalil batch screen; for a
single scalar output reverse still wins. Forward and reverse agree to machine
precision — the choice is purely performance, and the right one depends on the
output/input counts and the adjoint stiffness, so both are exposed.) The
**full** second-order AD Hessian (`jax.hessian`) is best avoided entirely: with
the default adjoint it hits the `custom_vjp` wall, and even with `DirectAdjoint`
the second derivatives through the stiff implicit solve are unreliable
(they disagree with finite differences). Use a first-order Gauss–Newton
`H = JᵀJ` instead (see `calibrate(laplace_method="gauss_newton")`).

### Differentiating stiff networks (`dtmax`)

Every reactor takes an optional `dtmax` (maximum integrator step), threaded
into the `PIDController`. Default `None` (uncapped) — fastest for plain forward
solves.

**Set `dtmax` when taking a *reverse-mode* gradient of a very stiff network.**
`Kvaerno5` is L-stable, so a solve can take steps far larger than the fastest
reaction timescale and simply damp the unresolved fast modes — the primal is
fine at any step. Differentiation splits by mode: **forward mode**
(`jax.jvp`/`jax.jacfwd` via `DirectAdjoint`) stays **finite at any step**,
losing only accuracy when the fast modes are unresolved; **reverse mode**
(`jax.grad`, the discrete adjoint) returns **non-finite** values above a
step-size threshold. The reverse failure is **not** a near-singular
per-step solve, despite the bare reaction Jacobian being stiff *and*
ill-conditioned (condition number ~1e20 at t=0 for the Khalil/extended sewer
models, |eig| up to ~1e5 d⁻¹). That cond(J) is dominated by structurally
near-zero eigenvalues of depleted/dormant species; the operator the implicit
step and its adjoint actually invert, `I − γ·dt·J` (γ≈0.26 for Kvaerno5),
regularizes those directions and stays **well-conditioned** across the failing
step range (cond ~650 at dt=1e-2, ≤~3e4 at dt=1e-1). The failure is instead an
**overflow in the reverse (backward) accumulation**, controlled by the per-step
stiffness `γ·dt·‖J‖` (fast-mode timescales per step), not by operator
conditioning — consistent with the threshold scaling below (the steeper-Jacobian
half-order variants need a tighter cap). Forward-mode tangent propagation, in the
same direction as the primal and through the same well-conditioned operator,
stays finite. Capping `dtmax` to a small multiple of the fastest reaction
timescale bounds `γ·dt·‖J‖`; the resulting reverse gradient is finite and matches
both forward mode and finite differences. This is **reverse-mode-specific and
independent of the adjoint flavour** — `RecursiveCheckpointAdjoint` and
`DirectAdjoint` reverse both fail identically, so it is not the checkpointing.
(It is also *not* the positivity limiter, *not* the zero-valued initial
species, and *not* stiffness alone: a 2–3 species stiff toy differentiates
finitely in reverse even at 1e7 d⁻¹ — the failure needs the full coupled,
many-species system.) The threshold is model-dependent: ~5e-3 d for the
Khalil Monod biofilm, ~10× tighter (~5e-4 d) for the half-order variants whose
√C kinetics steepen the Jacobian; the study uses `dtmax = 1e-4` d (3e-5 d for
the stiffer balanced base) — inside both. Because calibration needs the reverse
adjoint (one pass for the whole parameter gradient), the cap matters there; a
forward-mode sensitivity screen is unaffected, and is also faster (see
`dgsm(ad_mode="forward")`). A future alternative is a quasi-steady-state (QSS)
reduction of the near-instantaneous fast reactions, which would remove the
stiff modes entirely and avoid needing the cap.

**Cap-free alternative — forward-sensitivity solve.** For the sensitivity
`dC/dθ` itself there is now a way that needs **no cap at all**:
`reactor.solve_sensitivity(...)` (and the free function
`aquakin.forward_sensitivity(...)`). Instead of differentiating *through* the
stiff solve, it integrates the variational equation `dS/dt = J·S + f_θ`
*alongside* the state — one augmented `[y; S]` system, stock `Kvaerno5` +
`PIDController` whose error norm now also bounds `S` — so the adaptive
controller tightens the step only where the *sensitivity* is stiff and runs
free elsewhere. The primal stays uncapped and the returned `S` is finite and
exact (it matches a tightly-capped `jacfwd` to ~1e-8; validated against the
closed-form sensitivity of first-order decay and against capped `jacfwd` on the
stiff Khalil biofilm in `tests/integration/test_forward_sensitivity.py`). The
augmented RHS is `f(y)` plus one JVP of `f` per sensitivity parameter (the
JVP's primal gives `f(y)` for free; the JVP also differentiates through the
state-derived-pH speciation solver, the positivity limiter and the density-cap
throttle, so no special-casing). Implemented in
[`integrate/forward_sensitivity.py`](aquakin/integrate/forward_sensitivity.py)
on `BatchReactor`, `PlugFlowReactor` and `BiofilmReactor`.

**`Plant.solve_sensitivity` — the stable forward mode on the plant (the one that
was missing).** The reactors carried the cap-free augmented `[y; S]` solve, but
the *plant* was never wired to it, so a forward sensitivity of a stiff dynamic
plant fell back to `jacfwd` through `plant.solve` — finite per step, but
**non-finite over a long horizon**. That long-horizon failure is **numerical,
not a genuine tangent divergence**: the true sensitivity stays bounded (verified
by watching `‖S‖` *oscillate* with the diurnal load over 17 days — 45→196 — not
grow exponentially), and the break is the reactor-era
`augmented_forward_sensitivity` building a **stock `Kvaerno5`** that lacks the
plant's solver robustness. It was the one implicit mode never brought into the
`build_implicit_solver` / `build_step_controller` consolidation (the forward
`jax_adjoint`, `forward_fast` and `stable_adjoint` paths all build from those
helpers; the forward-*sensitivity* solve built its own). `Plant.solve_sensitivity(params, wrt, *, t_span, t_eval=, y0=, factormax=, dtmax=)`
closes the gap: it integrates the same augmented `[y; S]` system but routes the
solver through those shared helpers, so the `[y; S]` solve inherits the plant's
**decoupled Newton**, **`factormax`** cap, **`Kvaerno3`** base solver and
**cached recycle/flow maps** (its `f_flat` reuses the once-per-solve map instead
of re-resolving the recycle each call), while keeping the block-arrow
`SimultaneousCorrector` for the per-stage linear algebra. That config is exactly
what the stock solver lacked: the 20-day BSM2 forward sensitivity that the stock
augmented solve **and** `jacfwd` both blow up on is **finite** under
`solve_sensitivity`; it matches the dense `Kvaerno5` augmented solve to 5 digits
at 10 days, matches `jacfwd` to ~3e-7 on BSM1, and runs faster than the stock
dense/corrector (the Kvaerno3 + decoupled Newton + cached map all pay off).
Returns `(ts, ys, S)` with `S` the state sensitivity `dy/dθ`, shape
`(n_t, ndof, k)`. The cached map is built from the concrete `params` and closed
over, so the augmented linearisation drops `∂M/∂θ` — **exact for kinetic
parameters** (`M` depends only on the flow setpoints); a flow-setpoint
sensitivity needs the per-call map (not yet wired). The block-arrow corrector is
**specific to the `[y; S]` arrow** (each sensitivity column couples only to `y`,
sharing the one diagonal block `D = I − γ·dt·J`) and does **not** transfer to the
single-state forward/adjoint solves (whose per-step lever is the colored
Jacobian) or the steady-state IFT (which already factors `∂F/∂y` once and reuses
it across parameters). `build_implicit_solver` gained a `linear_solver=` slot to
inject the corrector into the decoupled root finder. Validated in
`tests/integration/test_dynamic_sensitivity.py::test_solve_sensitivity_matches_jacfwd`.

**`shared_factor` — dense (Option B) vs simultaneous corrector (Option A).**
The per-step implicit solve has two implementations, selected by
`solve_sensitivity(..., shared_factor=...)`:

- `shared_factor=False` (**Option B**, dense): hand the augmented `[y; S]`
  system to a stock `Kvaerno5`, which factorises the full
  `n(1+p)×n(1+p)` implicit operator each step. Exact and cap-free; the right
  choice for **one** sensitivity parameter and for scalar-loss gradients.
- `shared_factor=True` (**Option A**, CVODES simultaneous corrector): the
  augmented Jacobian is block-lower-triangular "arrow" form — every diagonal
  block is the same `D = I − γ·dt·J`, each `S_j` column couples only to `y`.
  A custom `lineax` solver
  ([`integrate/_simultaneous_corrector.py`](aquakin/integrate/_simultaneous_corrector.py))
  injected into the `Kvaerno5` `VeryChord` root-finder **factorises `D` once
  per step (`O(n³)`) and forward-substitutes across the `S` columns**, instead
  of the dense `O((n(1+p))³)`. The Newton step is identical to the dense
  solve, so results are **bit-equivalent** to Option B (verified to ~5e-14) —
  only the linear-algebra cost differs.

`shared_factor` **defaults to `None`**, which auto-selects Option A for more
than one sensitivity parameter and Option B for a single one (Option A has no
advantage at `p=1`).

**Measured (stiff `wats_sewer_khalil_paper_balanced`, biofilm, jitted, `p=5`):**
Option A beats Option B by **3.8× at ndof=100 and 6.9× at ndof=180** — the win
grows with system size, as the `(1+p)³`→`1` factorisation saving predicts.
Versus the *capped `jacfwd`* workaround the comparison depends on the
integration span: the uncapped augmented solve's adaptive sensitivity control
actually takes **more** steps than a capped primal-only solve over a short
window (the sensitivity transient is what it must resolve), but its step count
**plateaus** (~4300) while the capped step count grows linearly with the span.
So `jacfwd` is faster for short solves and Option A overtakes it for long ones
(measured crossover ≈ 8–10 days: `0.78× → 0.92× → 1.16×` at 2/5/10 d), with a
large Option-A win expected at the multi-week maturation spans the cap was
introduced for. Net guidance: **for a multi-parameter sensitivity of a stiff
network, `shared_factor=True` (the default) is the best forward-sensitivity
option; whether it also beats capped `jacfwd` depends on the span.**

The known cost of the non-invasive design (a custom *solver*, not a custom
diffrax RK stage): the solver only sees the augmented operator, so materialising
`D` and the off-diagonal coupling blocks `L_j` costs `n` probes of the augmented
`M.mv` (`n·(1+p)` f-JVPs per step) — the `L_j` blocks that an ideal CVODES with
direct access to `f` would not form. A zero-redundancy variant (a custom operator
carrying `f`, needing a `Kvaerno5` stage subclass) is the documented future
optimisation. `calibrate` does not yet expose a
`jacobian="forward_sensitivity"` hook, so the `dtmax` cap is still required for
the reverse-mode `calibrate` gradient until that lands.

**Cap-free *reverse* mode — hand-written discrete adjoint.** The forward
sensitivity above scales with the parameter count, so for a scalar-loss gradient
of many parameters (the calibration case) reverse mode is still wanted — and
that is the mode the `dtmax` cap exists for. `aquakin.implicit_euler_adjoint_solve`
([`integrate/discrete_adjoint.py`](aquakin/integrate/discrete_adjoint.py))
removes the cap there too, by **not differentiating through the solve at all**:
the forward pass is an ordinary robust adaptive diffrax `ImplicitEuler` solve,
and the reverse pass is the **discrete adjoint written out by hand** as a
per-step backward scan over the saved trajectory — each step a single
*transposed* solve through the same well-conditioned `I − dt·J` (a contraction,
so the cotangent stays bounded and nothing overflows). This is the classical
implicit-RK discrete adjoint (Sandu 2006; FATODE, Zhang & Sandu 2014); it is the
*exact* gradient of the discrete solve and is **verified two ways**: against the
closed-form gradient of first-order decay, and against the (correct but capped)
`RecursiveCheckpointAdjoint` gradient of the same implicit-Euler solve
(`rel ≈ 5e-8`, uncapped — see `tests/integration/test_discrete_adjoint.py`).
**Why earlier attempts failed and this works:** the overflow is in reverse-mode
AD forming cotangents of the large stored stage vector-field values `f_i ∼ ‖J‖·y`
(confirmed by reading the diffrax/optimistix source — the per-step Newton solve
is *already* IFT-differentiated by optimistix; the overflow is in the explicit
stage-combination arithmetic on the tape). Writing the per-step adjoint as the
analytic transposed solve never forms those large cotangents. Empirically
checked dead-ends, for the record: a stiffness-aware `dt·‖J‖` step controller
(finite but no faster than the cap), and k-space stage storage (shifts the
threshold out but does not remove the overflow). **Trajectory loss** is
supported: passing `t_eval` returns the states at those times and the backward
scan injects each observation's cotangent at its step; to keep that exact
without differentiating through dense interpolation, the forward is forced to
land steps exactly on `t_eval` (`diffrax.ClipStepSizeController(step_ts=t_eval)`),
so every observation is a step boundary (verified vs a closed-form
multi-observation gradient and vs the capped reference using the same
forced-step forward, `rel ≈ 6e-8`). **Wired into `calibrate`** via `calibrate(..., gradient="stable_adjoint")`. Both
gradient backends compute a discrete adjoint and both use JAX autodiff for the
**model** derivatives (`∂f/∂y` via `jacfwd`, `∂f/∂θ` via `vjp`); they differ only
in how the *integrator's* adjoint is formed — `gradient="jax_adjoint"` (default)
lets JAX/diffrax differentiate the whole solve (`RecursiveCheckpointAdjoint`,
needs the cap for stiff), while `gradient="stable_adjoint"` replaces only the
integrator's adjoint with the explicit per-step transposed solve (cap-free). The
stable backend forces a reverse-mode residual Jacobian under
`optimizer="gauss_newton"` (it is a reverse-only `custom_vjp`), and
`stable_adjoint_max_steps` bounds the saved-trajectory buffer the backward scan
walks (set it to a tight upper bound on the step count). Verified end-to-end: a
synthetic Khalil calibration reaches the **same optimum** as the capped-Kvaerno5
`gradient="jax_adjoint"` path — see `test_calibrate_stable_adjoint_matches_jax_adjoint`.
**Also wired into `plant.solve`** via `plant.solve(..., gradient="stable_adjoint")`,
which routes the assembled flat plant RHS through `esdirk_adjoint_solve` so a
reverse-mode gradient flows through the whole monolithic plant solve — across the
ASM↔ADM interface and the recycle loops — with no `dtmax` cap, in the regime where
differentiating *through* the stiff plant solve (`jax_adjoint` /
`RecursiveCheckpointAdjoint`) is non-finite. **`gradient` defaults to `"auto"`**:
a plain forward solve (concrete `params`/`y0`) takes the fast cached `jax_adjoint`
path and a solve under reverse-mode differentiation (the args are JAX tracers)
takes `stable_adjoint`, so a stiff plant gradient is finite by default with no
knob to set; `event=`/`adjoint=`/`dtmax=` pin `jax_adjoint` (so `run_to_steady_state`
is unaffected), and a `jax.jit`-wrapped forward solve looks traced and so routes
to `stable_adjoint` (correct but uncached — pass `gradient="jax_adjoint"` to force
the cached path). It is **exact through a transient
solve**: `plant.solve` passes `time_dependent=True`, so the explicit time
dependence of a time-varying influent is carried in the state
(`esdirk_adjoint_solve(time_dependent=True)`, the classical autonomization that
appends `dτ/dt=1` and reads the time from the state) and the discrete adjoint
captures `∂f/∂t` exactly with no change to the per-step recurrence. Without it the
default autonomous backward evaluates the field at a fixed time and the gradient
of any time-coupled parameter is wrong (zeroed in the worst case). It rejects
`adjoint=`/`dtmax=` (it manages its own integrator and adjoint), and `max_steps`
bounds the saved-trajectory buffer the backward scan walks (the warm-started BSM2
plant takes ~205 forward steps under a constant influent, so a small cap keeps the
reverse pass cheap).
**Validated**: a *water-line* gradient — tank-1 nitrate with respect to the ADM1
acetate-uptake rate `k_m_ac`, flowing back through the digester, the interface and
the reject recycle — is finite and matches central finite differences to
`rel ≈ 4e-5` under a constant influent (the direct digester-biogas gradient to
`rel ≈ 4e-6`), and to `rel ≈ 2e-3` under a *diurnal time-varying* influent (the
`time_dependent` path), where the default reverse adjoint of the stiff plant fails
outright (`tests/integration/test_plant_stable_adjoint.py`). The autonomization is
verified exact against finite differences on a forced ODE, and the autonomous
default is shown to give the wrong gradient for a time-coupled parameter, in
`tests/integration/test_discrete_adjoint.py`.

**Two solvers, low- and high-order.** `implicit_euler_adjoint_solve` (first
order) is the simple, robust baseline. `esdirk_adjoint_solve` is the high-order
version: a general s-stage ESDIRK forward (default **`Kvaerno5`, the same method
the reactors use**) whose discrete adjoint reconstructs the stage values in the
backward pass and applies the transposed-stage recurrence `(I − dt·γ·Jᵢᵀ)⁻¹` per
stage — the FATODE/Sandu construction (verified to reduce to the implicit-Euler
case for s=1). **The stage values are saved by the forward, not recomputed.**
The forward runs `SaveAt(steps=True, dense=True)`; for a Runge–Kutta solver the
dense-output info carries the per-step stage increments `kⱼ` (the **dt-scaled**
stage derivatives `dt·f(Yⱼ)`), so the backward reconstructs each stage exactly by
the Butcher linear combination `Yᵢ = yₙ + Σⱼ A[i,j]·kⱼ` (`A` the full
lower-triangular tableau, dt already folded into `k`) — **no per-step Newton
recompute**, which was the dominant backward cost. (Earlier this re-solved every
stage by a fixed 12-iteration Newton scan, ~72 Jacobian builds + dense solves +
RHS evals per step; the saved-stage path removes all of it.) The saving is threaded
through the shared `_discrete_adjoint_solve` driver as `save_stages=` (ESDIRK sets
it; the s=1 implicit-Euler adjoint reads the post-step state directly and leaves it
off). **Measured (BSM2 `value+grad`, dense backward, 3-day warm-started span): the
backward dropped 10,647 → 617 ms (~17×) and the whole gradient 11,200 → 1,212 ms
(~9×), gradient FD-/jax_adjoint-validated unchanged.** Validated: the stage
reconstruction is exact (the discrete-adjoint suite — analytic decay, FD, trajectory
and time-dependent gradients — and the plant BSM1/BSM2 cross-interface, colored,
and flow-setpoint `∂M/∂θ` gradients all match FD / the capped `jax_adjoint` path).
**`calibrate(gradient="stable_adjoint")` uses this Kvaerno5 ESDIRK adjoint**, so
its forward matches the reactor exactly and its gradients agree with the capped
`jax_adjoint` path to the optimiser tolerance (analytic decay `rel ≈ 1e-6`; stiff
network finite-uncapped, matching capped Kvaerno5 to `rel ≈ 2.5e-5`, the residual
being the capped-vs-uncapped *forward* difference, FD-confirmed). **Cost note:** the
backward scan's cost scales with `stable_adjoint_max_steps` (the padded trajectory
length), and with `dense=True` the saved dense-output buffer is ~`n_stages`× the
trajectory, so keep `max_steps` tight; Kvaerno5's high order keeps the step count
low. The autonomous reaction RHS is assumed (the ESDIRK stage times `c` do not
enter).

### Operator Splitting

Transport and reaction are decoupled at all scales:

| Scale | Transport | Reaction |
|---|---|---|
| 0D batch | n/a | Diffrax directly |
| 1D PFR | advection/diffusion step | Diffrax reaction sub-step |
| 3D CFD | OpenFOAM transport step | Diffrax (or C++ stiff solver) reaction sub-step |

The ODE integrator only ever sees the reaction sub-problem — a pure chemistry
integration over one transport timestep at a fixed spatial location.

### Located events / discontinuities ([`integrate/events.py`](aquakin/integrate/events.py))

A plain solve is continuous; on/off pumps, SBR fill/react/settle/decant phase
switches, relay/saturating control, dosing on/off and tank-level limits are
**discontinuous**. `aquakin.Event` + `solve_with_events` locate the switch
exactly and apply a **state reset / mode switch** there, then continue —
instead of smoothing it or grid-snapping with `searchsorted`. Exposed as an
`events=` argument on `BatchReactor.solve` and `Plant.solve`; both build their
RHS and hand it to the shared driver, which returns the trajectory on the
requested `t_eval` grid plus a `solution.events_log` of `(time, name)` firings.
**No drift from the plain solve:** the event path reuses the *same* two pieces
the plain solve uses — the reaction RHS comes from the shared
`make_chemistry_rhs` factory (batch) or `self._rhs` (plant), and each segment is
integrated by the canonical `_run_diffeqsolve` (Kvaerno5 + `PIDController` +
adjoint), so the per-step integration and the RHS cannot diverge between
`solve()` and `solve_with_events`. A parity test pins this: an identity reset (or
a never-firing state event) reproduces the plain `solve()` trajectory, so any
future change to the RHS/kernel that reaches only one path fails the test
(`tests/integration/test_events.py`).

An `Event` carries exactly one trigger — `at_times=[...]` (a **time event**) or
`cond_fn(t, y, args)` (a **state event**, located by an optimistix root find on
the zero crossing, filtered by `direction` ±1) — plus an optional
`apply(t, y, args) -> y` reset and a `terminal` flag. The driver splits the
solve into segments at the firings; the boundary convention is that a `t_eval`
point coinciding with a firing reports its **pre-reset** value (it belongs to
the segment ending at the event), so the reset defines the next segment's
initial condition.

Two paths, one driver (`_drive`), chosen by whether any state event is present:
- **Time events only** — the segment boundaries are static Python constants and
  no branch depends on traced state, so the whole solve is a fixed sequence of
  differentiable diffrax sub-solves: **`jax.grad` flows through it** (the SBR /
  scheduled-dosing / AD-safe case). It still needs the `dtmax` cap for a stiff
  reverse-mode gradient, exactly like the plain solve.
- **Any state event** — the firing time/count is discovered at runtime (located
  via a terminating `diffrax.Event` whose `event_mask` says which fired), so the
  loop is an **eager forward simulation**, not differentiable through the switch
  (use a smoothed `cond_fn` where a gradient through the threshold is needed). A
  `max_segments` guard raises a clear error if a reset fails to clear the
  threshold and the event re-fires without advancing.

This is distinct from the low-level `Plant.solve(event=<diffrax.Event>)` used
internally by `run_to_steady_state` (a single terminating event); the
user-facing API is `events=[Event(...)]`. `events=` is rejected with
`gradient="stable_adjoint"` (it runs its own segmented solve). It is the
prerequisite for the SBR unit (#273) and relay/on-off control studies.
Demonstrated in `examples/event_handling.py` (scheduled ozone re-dosing + a
bromate-limit terminal cut-off); tested in `tests/integration/test_events.py`
(reset/terminal/direction/multi-event, AD through a time event, the runaway
guard, and BSM1 plant resets).

### JAX x64 Mode

Stiff ODE integration requires 64-bit floats. `aquakin` enables x64 mode
automatically at import time. This is **global, process-wide** JAX state, so it
is a documented side effect of `import aquakin` (README install section). To keep
it from being *silent*, `aquakin/__init__.py` warns (once) when it overrides what
looks like an explicit float32 preference — JAX already imported when aquakin is
imported, or `JAX_ENABLE_X64` set to a false value — but stays silent on a plain
fresh import (the common case). It always enables x64 regardless; the warning is
only a signal. See `tests/unit/test_x64_import.py` (subprocess-per-scenario, since
the effect is process-global). Do not remove the x64 enablement.

---

## YAML Network Schema

### Top-Level Structure

```yaml
network:
  name: ozone_bromate
  version: "1.0"
  description: "..."
  references:
    - "Acero & von Gunten (2001), Environ. Sci. Technol. 35(3), 590-599"

species:
  - name: O3
    description: "Ozone"
    units: mol/L
    default_concentration: 1.0e-4

conditions:
  - name: pH
    description: "Solution pH"
    default: 7.5
  - name: T
    description: "Temperature (K)"
    default: 293.15

reactions:
  - name: O3_Br_direct
    description: "Direct ozone oxidation of bromide"
    reference: "Acero & von Gunten (2001)"
    rate: "k1 * [O3] * [Br-]"
    parameters:
      k1:
        value: 160.0
        units: M-1 s-1
        bounds: [10.0, 1000.0]
    stoichiometry:
      O3: -1
      Br-: -1
      HOBr: +1
```

### Schema Rules

- `default_concentration` is a reference value for the species, not an
  experimental initial condition. Users override it at runtime.
- `conditions` block declares all fields the network requires. The loader
  validates that any `SpatialConditions` object passed at runtime provides
  all declared fields. Each condition may carry an optional `units:` string
  (default `""`), advisory metadata used only by `check_units` (e.g. `pH: "-"`).
- `units` on species (default `"mol/L"`), parameters (default `""`), and
  conditions (default `""`) are advisory unit strings. Species/parameter units
  feed result labelling and the opt-in `check_units` dimensional check; they are
  not otherwise used at runtime. `check_units` treats a blank/unparseable unit
  as unknown (skipped).
- `reference` on reactions is optional but strongly encouraged — it makes
  the YAML file a self-documenting scientific artifact.
- `bounds` on parameters are optional, used by `fit()` as box constraints.
- `transform` on parameters is optional (default `"none"`); valid values
  are `"none"`, `"positive_log"` (for `p > 0`), and `"logit"` (for
  `0 < p < 1`). Used by `calibrate()` to optimise in unconstrained space.
- `prior` on parameters is optional — a Gaussian prior in physical space
  declared either as `prior: {mean: m, std: s}` (a measured value with
  reported uncertainty) or `prior: {range: [lo, hi]}` (a literature range,
  converted to a Gaussian centred on the midpoint with `std = (hi-lo)/4`,
  i.e. the range ≈ ±2σ). `calibrate()` adds `0.5·((p-mean)/std)²` to its
  objective for any free parameter carrying a prior (default
  `use_priors=True`; override per-parameter via the `priors=` argument).
  Priors regularise otherwise non-identifiable parameter combinations toward
  literature values; for a proper Bayesian MAP/posterior pair them with
  `loss="nll"` and a measurement `sigma` so the data term is a true negative
  log-likelihood and the prior curvature enters the Laplace covariance.
- `temperature` on parameters is optional — an Arrhenius-style temperature
  correction `temperature: {theta: t, ref_T: T0, condition: "T"}`. When present,
  the rate constant is multiplied by `theta**(T - ref_T)` during rate
  evaluation, where `T` is read from the named condition field (default `"T"`).
  The parameter `value` is the value **at** `ref_T`, so the correction is unity
  there — a network whose conditions sit at the reference temperature behaves
  exactly as if uncorrected (backward-compatible). `ref_T` is in the condition's
  units (Kelvin for the ASM/ADM networks); a difference is used, so Kelvin and
  Celsius give the same `theta`. `theta` is the per-degree factor; from a
  parameter measured `p_hi` at `T_hi` and `p_lo` at `T_lo` it is
  `(p_hi/p_lo)**(1/(T_hi - T_lo))`. The correction is applied to **rate
  constants only** — it is confined to `CompiledNetwork.rates` (which multiplies
  the corrected param indices by their factor before evaluating the rate
  callables); `compute_stoich` always uses the raw parameters, so stoichiometric
  (yield / composition) coefficients are never temperature-scaled. Stored as
  `CompiledNetwork.temperature_corrections` (a list of
  `(param_idx, ln_theta, ref_T, condition_field)`); AD-clean. `asm1` ships with
  the six BSM temperature-dependent rate constants corrected (`muH`, `muA`,
  `bH`, `bA`, `ka`, `kh`, `ref_T = 293.15 K`, slopes from the standard BSM
  15 °C/10 °C pairs in `asm1_bsm2.c`), so it slows correctly in the cold
  (nitrification — the most temperature-sensitive — drops to ~36% at 10 °C) while
  staying identical to the old behaviour at the default 20 °C.
- Parameters can live at the network level (a single shared slot used by
  any reaction that references them by bare name) or inside a reaction's
  `parameters:` block (namespaced as `<reaction>.<name>`). Network-level
  parameters and reaction-local parameters with the same name are
  rejected as shadowing.
- Stoichiometric coefficients can be **numeric literals** or **string
  expressions** in network-level / reaction-local parameters. String
  entries may use only constants, parameters, and arithmetic — no
  species, conditions, named expressions, or domain functions. Evaluated
  once per `solve()` call so yield / N-content / fraction parameters can
  be calibrated alongside kinetic constants. See `asm1.yaml` for a worked
  example.
- Optional per-species **`composition:`** block declares the species' content of
  the conserved quantities in its own measure — `composition: {COD: 1.0}` for an
  organic (1 g COD per g COD), `{COD: -1.0}` for dissolved oxygen (an electron
  acceptor), `{COD: -2.86, N: 1.0}` for nitrate-N, `{COD: 2.0, S: 1.0}` for
  sulfide. Quantity names are free-form (`COD` / `N` / `P` / `S` / `Fe` / `charge`
  / …). It is **first-class conservation metadata**: a network carries its own
  table instead of one hand-maintained in a test, read back via
  `network.composition()` and dotted against the stoichiometry by
  `network.check_conservation()` / `network.check_nitrogen()` (the
  advisory/opt-in conservation analogue of `check_units`; it never runs at load
  and never raises on a violation — it returns the list). A network that declares
  no `composition:` falls back to the shipped role-based table
  (`aquakin.composition_table`, the ASM/ADM families) so the check API is uniform;
  a network with neither raises a clear error. Values are literal floats (a
  yield-dependent *derived* coefficient stays in the stoichiometry, not here).
  The WATS sewer family (`wats_sewer`, `wats_sewer_extended` and everything the
  `_make_khalil_*` generators splice from them) ships `composition:` per species,
  so the conservation suite checks each network against its own declared table
  (`tests/integration/test_mass_balance.py`); inherited through `extends:` (a
  derived species keeps the base's composition unless it overrides it).
- A stoichiometric coefficient written **`auto`** (or **`?`**) is left unknown and
  **solved from the declared conservation laws** at compile time, so a
  conservation-determined coefficient *cannot be written wrong* — the failure mode
  behind almost every stoichiometry bug here (a hand-typed electron-acceptor
  demand, an elemental-S reduction donor, a product split). The quantities to
  solve from are the reaction's **`conserved_for: [COD, N, …]`** (or a network-level
  `conserved_for:` default); for each, the stoichiometry-weighted species
  `composition` content must sum to zero, giving one small linear system
  (`core/stoich_resolve.py`, `numpy.linalg.lstsq`) solved before the stoichiometry
  is read. Example: `stoichiometry: {SS: "-1/Y_H", X_BH: 1, SO: auto}` with
  `conserved_for: [COD]` solves the O₂ demand from the COD balance. The compiled
  network then conserves by construction (`check_conservation` is a tautology on
  that reaction — the point). Clear errors for an `auto` with no `conserved_for`,
  an under-determined system (an auto species carrying no content in the conserved
  quantities), or inconsistent balances. Two cases (issue #291 Phases 2–3):
  - **numeric** — every other coefficient is a numeric literal, so the solve is
    purely numeric (`lstsq`, tolerating an over-determined-but-consistent system)
    and the resolved value is a constant baked into the static stoichiometry matrix.
  - **parameter-expression (yield-dependent)** — a neighbour is a string expression
    (e.g. `-1/Y_H`, as in the example above). Conservation is *linear* in the
    coefficients, so each `auto` coefficient is a numeric-weighted linear
    combination of the known coefficient expressions (`x = M⁻¹b`, `M` from the
    composition); the resolver emits that as a *derived* expression string, which
    the normal stoichiometry-expression machinery compiles into a
    parameter-dependent (`stoich_dynamic`) coefficient. So calibrating a yield
    flows through to the derived coefficient — the reaction conserves for **every**
    parameter value, and `jax.grad` flows through it. This path needs a **square**
    system (`#auto == #conserved_for`); an over-determined symbolic system's
    consistency would be parameter-dependent and is rejected.

  `auto` requires declared `composition:` (the resolver does not use the role-based
  fallback). Shipped networks keep their published rounded literals; `auto` is
  opt-in.
- Optional `expressions:` block at the network level lets you give a name
  to an intermediate rate expression and reference it from a reaction's
  `rate:` or from another expression. References are inlined into the
  consuming AST at compile time. Cycles among expressions are rejected.
- Optional **`network.extends:`** declares a base network to **inherit** instead
  of copying — a variant that differs by one parameter and a rate is a few lines,
  and a fix to the base reaches every variant. The base is a shipped network
  *name* (or a path relative to the extending file if it contains `/` or ends
  `.yaml`); the derived mapping is **merged onto the (recursively resolved) base
  before Pydantic validation** (so the merged whole is schema-checked).
  ([`schema/inheritance.py`](aquakin/schema/inheritance.py), wired in
  [`schema/loader.py`](aquakin/schema/loader.py).) Merge semantics: the named-list
  blocks (`species`/`conditions`/`reactions`) match by `name` and **field-merge**
  a derived entry onto the base (override just a reaction's `rate`, keep its
  `stoichiometry`; a new name is appended); every other block (`parameters`,
  `expressions`, `speciation`, ...) deep-merges (nested maps key-wise, scalars/
  lists replace). A `remove:` block (`{reactions: [...], parameters: [...],
  species/conditions/expressions: [...]}`) then drops entries; removing a
  not-present name, a cyclic `extends`, an unknown base, or declaring `extends`
  in both `network:` and top-level all error clearly. New parameters append after
  the base's, so the flat parameter *vector order* differs from a hand-written
  full copy (the name→value mapping and the compiled stoichiometry/rates are
  identical). A YAML with no `extends` is byte-for-byte unaffected. Shipped users:
  `asm1_ammonia_limitation` (= `asm1` + the `KNH_H` nutrient switch, ~30 lines vs
  a 200-line copy) and the Khalil `*_halforder` sewer variants (the generator
  [`networks/_make_khalil_variants.py`](aquakin/networks/_make_khalil_variants.py)
  emits the pure-override `halforder` variant as a thin `extends` file; the
  variants whose stoichiometry is *computed from the base* stay full copies).
- Optional `speciation:` block declares a **state-derived pH** (see below).
- Optional `positivity_limiter:` block (`threshold`, default `1e-3`) throttles
  each species' *net reaction* term as its concentration approaches zero,
  preventing negative states and the stiffness they cause. Reproduces the
  reference WATS S-function scheme
  `R_lim = max(R,0) + min(R,0) * C / max(C, threshold)`. Applied inside
  `CompiledNetwork.dCdt` to the reaction term only (reactors add transport
  afterwards), so every reactor benefits. Opt-in; off when the block is
  absent. Stored as `CompiledNetwork.positivity_threshold`.
- Optional `clip_negative_states: true` (default `false`) clamps the
  concentration vector to `>= 0` *when evaluating the reaction rates* (and any
  state-derived condition such as pH), leaving the raw state for the reactor's
  transport term and the unit outputs. This protects nonlinear kinetics (Monod /
  ratio terms) from a transiently-negative state — where they produce
  large/garbage rates and a stiff blow-up — and is exactly the reference
  IWA/BSM S-function convention (`xtemp = max(x, 0)` before the process rates).
  It is **distinct** from `positivity_limiter`: the limiter throttles the *net
  reaction* near zero (output side), while this clamps the rate *inputs* — so it
  prevents the Monod-at-negative-`C` blow-up that the limiter does not. Identity
  at feasible states (does not change the physical solution); leaving transport
  on the raw state keeps the linear washout self-correcting and the inter-unit
  mass balance exact. Enabled on `asm1` (needed once the BSM1 settler recycles
  concentrated solids into a small reactor). Stored as
  `CompiledNetwork.clip_negative_states`.
- YAML is loaded with `yaml.safe_load()` throughout. Species names that
  could be misread as non-string types (e.g. `NO`) must be quoted in YAML.

---

## Speciation / state-derived pH

Some models couple rates to pH, but pH is not an independent state — it is
fixed by acid/base speciation through electroneutrality. `aquakin` can solve
pH from the instantaneous state and expose it to rate expressions as an
ordinary condition field.

### Charge-balance solver (`core/ph_solver.py`)

`solve_ph(...)` returns pH given the total molar concentrations of the
carbonate, acetate, ammonia, phosphate and sulfide systems, plus strong-anion
charge equivalents, a net fixed cation charge, and temperature. It runs a
**safeguarded Newton-bisection** on the electroneutrality residual in
`u = ln[H+]` space. The residual is strictly monotone with a unique, trivially
bracketable root, so each step takes a Newton step but falls back to bisection
(via the non-strict rtsafe product test) whenever Newton would leave the bracket
— making the iteration **globally convergent**. This matters: a bare Newton step
from the fixed pH-7 start **overshoots to `exp(u) = inf` (NaN), or silently to an
absurd pH that saturates the `pH_switch` rate terms**, when the strong-ion charge
exceeds the buffering (e.g. a weakly buffered transient, or a calibration that
pushes the `S_cat`/`S_an` states) — the bracketed scheme stays finite and correct
there.

**Adaptive iteration + implicit-function-theorem AD (the ideal hot path).** On
the default `activity_model="none"` path the iteration is an **adaptive
`jax.lax.while_loop`** that stops as soon as a Newton step near the root falls
below tolerance (a handful of steps in the buffered regime — measured 5–6 to
machine precision across pH 5.4–8.8) and is **capped at `n_iter`** for the
bisection worst case, wrapped in **`jax.lax.custom_root`** so the **pH
sensitivity is the analytic IFT tangent** (one scalar solve of `df/d[H+]` at the
root). The iteration count therefore never enters the AD graph: `jax.jvp` /
`jacfwd` (forward, the per-step Jacobian-materialisation path) and `jax.grad`
(reverse, the calibration-gradient path) through `solve_ph` are **O(1) in the
iteration count** instead of differentiating through every Newton step. The root
is the same one the old fixed scan converged to, so **every steady state is
unchanged** (validated: BSM2 steady state and the ADM1 digester reproduce
bit-for-bit). The convergence criterion is `in_bracket AND |Newton step| ≤ tol`
— a *bisection* step can be spuriously small far from the root, so the step size
alone is not a safe criterion, but a tiny in-bracket Newton step `|f/f'|` (with
`|f'| ≥ 1`) means a tiny residual, i.e. the root. *Why this is the lever:* a
Step-0 profile of the BSM2 plant found the pH solve was **~35% of the Jacobian
materialisation** purely because `n_iter` was 40 while it converges in ~6; the
adaptive + IFT rewrite drops the pH share to **~8%** (per-tangent pH cost
O(40)→O(1)) and the whole 20-day BSM2 transient by **~20–25%** (default solver),
with bit-identical results. The opt-in **activity-corrected path** uses the same
adaptive + IFT scheme lifted to a **2-variable coupled root**: its conditional
constants couple `[H+]` and the ionic strength `I`, so it solves the joint fixed
point `(f1 = charge balance, f2 = I − I(h)) = 0` with an adaptive coupled
`while_loop` wrapped in `custom_root` over the pair `(h, I)`, and the sensitivity
is the exact IFT tangent via a **2×2 linear solve** of the joint Jacobian at the
root (so it too is O(1) in the iteration count, forward and reverse). Equilibrium
constants are temperature-corrected via van't Hoff. The chemistry mirrors the
WATS reference pH solver. **The coupled path is also extreme-safe**: a transient
far-overshoot trial `[H+]` makes the water self-ionisation term `½(h + Kw/h)`
explode the ionic strength to ~1e20, where `g = 10^(…)` overflows to `inf` and
the `inf/inf` activity-coefficient ratios become `NaN` — which the bracketing
could then never recover from. The ionic strength fed to the activity models is
clamped to a physical ceiling (`_I_MAX = 10 M`, far above any digester/sewer
~0.01–0.2 M and the models' ~0.5 M validity), so every conditional constant stays
finite and the bracket pulls `[H+]` back to the root, where `I ≪ _I_MAX` makes the
clamp exactly identity (bit-identical converged pH; issue #382). The IFT 2×2
tangent solve floors `|det|` for the same finiteness at a degenerate
(all-zero-totals) input. Validated against an independent bisection root finder;
the weak-buffer / extreme-charge regime that broke the old bare-Newton scheme, the
activity path's extreme-input finiteness and degenerate-point gradient (#382), the
cap-independence of the adaptive loop, and forward-vs-reverse AD agreement (both
the IFT tangent) are regression-tested in `tests/unit/test_ph_solver.py`.

**Ionic-strength activity corrections (`activity_model`).** By default the solver
uses molar **concentrations** directly with thermodynamic equilibrium constants
(all activity coefficients γ = 1) — the published ADM1/BSM2 convention, and the
behaviour every validated BSM2/WATS result is pinned to. Commercial simulators
(SUMO, BioWin, GPS-X) instead apply an ionic-strength activity correction, which
at digester/sewer ionic strengths (I ≈ 0.05–0.2 M) shifts pH by ~0.1–0.3 units.
`solve_ph(..., activity_model=...)` adds this as an **opt-in** static option:
`"none"` (default, the γ = 1 path — a trace-time Python branch, so it is
**bit-identical** to the historic solver), `"davies"` (the Davies equation, valid
to I ≈ 0.5 M), or `"debye_huckel"` (extended Debye–Hückel / Güntelberg, to
I ≈ 0.1 M). With a non-`none` model the dissociation constants become *conditional*
(concentration-basis) constants `Kc = K·γ_acid/(γ_H·γ_base)` at the
**self-consistent** ionic strength — the coupled `I`↔`[H+]` fixed point is solved
*inside* the same bracketed scan by carrying `I` (it converges together with
`[H+]`), and the returned pH is the **measurable** `−log10(a_H) = −log10(γ_H·[H+])`
(which reduces to `−log10[H+]` at γ = 1). The Debye–Hückel `A(T)` comes from the
water dielectric (`debye_huckel_A`). NH₄⁺/NH₃ is charge-symmetric so its Ka is
unchanged; the strong-ion contribution to `I` (`½Σc z²`) is supplied by the
speciation layer, which holds each strong ion's charge. The whole path stays
`jit`/`vmap`/`grad`-clean (the IFT pH sensitivity flows through it). Decisive
correctness check: pure water in an inert salt stays at the neutral pH for any
`I` (`tests/unit/test_ph_solver.py`). Exposed two ways: the `speciation:` block's
`activity_model:` field (per-network YAML), and a load-time override
`load_network("adm1", activity_model="davies")` (and `load_network_from_file(...,
activity_model=...)`) to run a *shipped* network with activities without editing
its YAML.

### Derived conditions (the wiring)

`CompiledNetwork` carries an optional `derived_condition_fn(C, params,
condition_arrays, loc_idx) -> {field: scalar}` and a `derived_fields` list.
`CompiledNetwork.rates()` evaluates it once per RHS call and merges the result
into `condition_arrays` (broadcast across locations) **before** the rate
callables run — so existing `{pH}` / `pH_switch(pKa)` machinery sees the
derived value unchanged, and every reactor (batch/PFR/particle/CFD) gets it
for free. Derived fields are *produced*, so they are added to the AST's valid
condition set but are **not** in `conditions_required` (the user never
supplies them).

### `speciation:` block (`core/speciation.py`, schema in `schema/network_spec.py`)

```yaml
speciation:
  field: pH                 # produced condition field (default "pH")
  temperature_field: T      # condition giving temperature
  temperature_units: celsius # or "kelvin"
  z_cation_eq: 3.28e-3      # net fixed cation-charge OFFSET (eq/L); literal or {condition: name}
  n_iter: 40
  activity_model: none      # ionic-strength correction: none (default) | davies | debye_huckel
  totals:                   # species -> acid/base total (molar_mass converts mg/L -> mol/L)
    carbonate: {species: S_CO2, molar_mass: 12000}
    acetate:   {species: S_VFA, molar_mass: 64000}
    ammonia:   {species: S_NH,  molar_mass: 14000}
    phosphate: {species: S_PO4, molar_mass: 31000}
    sulfide:   {species: sumS,  molar_mass: 32000}
  strong_anions:            # fully dissociated: charge * conc/molar_mass eq/L
    - {species: S_SO4, molar_mass: 32000, charge: 2}
    - {species: S_NO,  molar_mass: 14000, charge: 1}
  strong_cations:           # same form; summed into the net cation charge
    - {species: S_cat, molar_mass: 1.0, charge: 1}
```

`build_ph_derived_fn` (in `core/speciation.py`, no Pydantic) turns the
validated declaration plus `species_index` into the derived-condition
callable. Negative concentrations are clamped to zero before conversion
(mirrors the reference). Valid `totals` keys: `carbonate`, `acetate`,
`propionate`, `butyrate`, `valerate`, `ammonia`, `phosphate`, `sulfide`
(`propionate`/`butyrate`/`valerate` are the additional monoprotic ADM1
volatile fatty acids, treated exactly like `acetate`). The net cation charge
passed to the solver is `z_cation_eq` (a literal/condition *offset*, default
0) **plus** the `strong_cations` sum — so a strong-cation *state* (e.g. ADM1's
`S_cat`, paired with `S_an` in `strong_anions`) drives pH dynamically, the same
way `strong_anions` already does for anions.

---

## Mineral precipitation / state-derived saturation

`aquakin` can model **kinetic mineral precipitation and dissolution** —
struvite, calcite, the calcium phosphates — driven by the aqueous
supersaturation. Like pH, the saturation state is not an independent state: it
is computed from the instantaneous composition and exposed to rate expressions
as ordinary condition fields. The framework is the generalised
precipitation approach of Kazadi Mbamba et al. (2015).

### The rate law (`core/precipitation.py`)

Each mineral precipitates (or dissolves) at

```
R = k_cryst · X_cryst · sign(σ) · |σ|^n
```

driven by the relative supersaturation `σ = (IAP/Ksp)^(1/ν) − 1`
(`SI = log10(IAP/Ksp)` is the saturation index), where
`IAP = Π aᵢ^countᵢ` is the ion-activity product over the mineral's
constituent ions, `Ksp` its solubility product, `ν = Σ countᵢ` the number of
ions, and `aᵢ` the free-ion **activities** at the system pH. A small non-zero
`X_cryst` seed lets precipitation self-nucleate; `sign(σ)` makes the same law
describe dissolution when the phase is undersaturated (`σ < 0`). The kinetic
`power` form requires **`order >= 1`** (validated at load): its rate gradient
`order·|σ|^(order-1)` is infinite at `SI = 0` (saturation) for `order < 1`, so a
sensitivity through the equilibrium would be non-finite; the `bounded` (tanh)
form carries no such restriction.

The aqueous chemistry — the temperature-corrected dissociation constants, the
free-ion fractions of the carbonate/phosphate/ammonia/sulfide acid-base
systems, and the Davies / Debye-Hückel activity coefficients — is **shared with
the charge-balance pH solver** (`core/ph_solver.py`): the module reuses
`equilibrium_constants`, `_log10_gamma`, `debye_huckel_A`. Free Ca/Mg are taken
as the full declared total (ion-pairing not yet subtracted — a documented
simplification of the source aqueous model). AD-clean (no data-dependent control
flow), so it composes inside a Diffrax RHS and `jax.grad` flows through it.

### `precipitation:` block (`core/precipitation.py`, schema in `schema/network_spec.py`)

```yaml
precipitation:
  pH_field: pH               # condition giving pH (a fixed condition, or produced by speciation:)
  temperature_field: T
  temperature_units: kelvin  # or "celsius"
  activity_model: davies     # "none" | "davies" | "debye_huckel"
  ionic_strength_offset: 0.1 # background electrolyte (mol/L); OR ionic_strength_field: I_aq (see below)
  minerals:
    - name: struvite
      pKsp: 13.26            # at the reference temperature (25 C)
      order: 3               # supersaturation exponent n
      dH_sp: 0.0             # enthalpy of dissolution (J/mol); van't Hoff Ksp(T), 0 = T-independent
      solid: X_struvite      # precipitate species -> the reaction is AUTO-DERIVED
      rate_constant: {value: 100.0, units: "1/d", bounds: [1.0, 1000.0], transform: positive_log}
      ions:                  # molar_mass converts the state's units -> mol/L (Ksp basis)
        - {species: S_Mg,  molar_mass: 1000, count: 1, charge: 2}                       # free cation
        - {species: S_NH,  molar_mass: 1000, count: 1, charge: 1, fraction: ammonia}    # NH4+ fraction at pH
        - {species: S_PO4, molar_mass: 1000, count: 1, charge: 3, fraction: phosphate}  # PO4^3- fraction at pH
    - name: calcite
      pKsp: 8.48
      order: 2
      solid: X_calcite
      rate_constant: {value: 10.0, units: "1/d"}
      ions:
        - {species: S_Ca, molar_mass: 1000, count: 1, charge: 2}
        - {species: S_IC, molar_mass: 1000, count: 1, charge: 2, fraction: carbonate}
```

`build_precipitation_derived_fn` (in `core/precipitation.py`, no Pydantic) turns
the validated declaration plus `species_index` into a derived-condition callable
that produces `SI_<name>` and `R_<name>` per mineral.

**Auto-derived reactions.** A mineral declaring `solid` (the precipitate
species) + `rate_constant` **auto-derives its precipitation reaction**: the
schema's `_synthesize_precipitation_reactions` emits `<name>_precipitation` with
rate `k * [solid] * {R_<name>}` and stoichiometry that consumes each ion's
`species` at `-count` and produces `solid` at `+1` (species-less `proton`/
`hydroxide` ions carry no mass term). This removes the old duplication of
re-writing the stoichiometry in a separate `reactions:` block (and the risk of it
drifting from the ion counts — vivianite Fe₃(PO₄)₂ correctly gets `-3`/`-2`). The
rate constant is namespaced `<name>_precipitation.k`. A network whose only
processes are precipitation may omit `reactions:` entirely; omitting `solid`/
`rate_constant` falls back to a hand-written reaction referencing `{R_<name>}`.

**Ksp(T).** `pKsp` is the value at the reference temperature (25 °C); `dH_sp`
(enthalpy of dissolution, J/mol) van't Hoff-corrects it,
`Ksp(T) = Ksp(Tref)·exp(dH_sp/R·(1/Tref − 1/T))` — the same form and reference as
the dissociation constants. `dH_sp = 0` (default) leaves Ksp temperature-
independent (backward compatible).

**Shared ionic strength.** When precipitation composes with a `speciation:`
block, set `ionic_strength_field: I_aq` on **both** blocks: `solve_ph` optionally
returns the self-consistent solution ionic strength (strong ions + weak-acid
speciation + water), speciation exposes it as the `I_aq` condition field, and
precipitation reads it for its Davies/Debye-Hückel coefficients — so the pH and
the saturation indices use the **same** ionic strength. Without it, precipitation
falls back to `ionic_strength_offset` + its own mineral ions (which misses the
bulk electrolyte), the right choice for a standalone fixed-pH network.

An ion's `fraction` selects how its free
activity is obtained: an acid/base system key — `carbonate`, `phosphate`,
`ammonia`, `sulfide` (the species total times its de/protonated fraction at pH)
— a pH/water special (`proton`, H⁺ activity `10^-pH`; or `hydroxide`, OH⁻
activity `Kw/[H+]`, both with `species` omitted), or omitted for a fully-free
cation (the species total taken as the free ion). The `hydroxide` fraction is
what lets metal hydroxides (Fe(OH)₃, Al(OH)₃) and hydroxide-bearing minerals
(hydroxylapatite) be declared. `VALID_PRECIP_FRACTIONS` in
`core/precipitation.py` is the single source of truth (with `_PH_SPECIALS` the
`species`-optional subset), imported by the Pydantic schema so an unknown
`fraction` (or an undeclared `species`) is a load-time error.

**Composition with speciation.** `precipitation:` is wired in a `_compile_precipitation`
stage in `core/network.py` *after* `_compile_speciation`, so when both blocks
are present the precipitation reads the **charge-balance pH** the speciation
block produces (the two derived functions compose via `_compose_derived`:
speciation runs, its pH is broadcast into the conditions, then precipitation
runs and both results merge). The shipped
[`precipitation_struvite_calcite.yaml`](aquakin/networks/precipitation_struvite_calcite.yaml)
is the worked example (digester supernatant, struvite + calcite, fixed
operating pH); the test suite also exercises the speciation→precipitation
composition.

**Units.** The worked network uses **mol/m³ (= mmol/L)** for the ions and
solids so the precipitation stoichiometry is exact (one mole of mineral consumes
one mole of each constituent ion), with the per-ion `molar_mass: 1000`
converting mol/m³ → the mol/L the IAP/Ksp use. `clip_negative_states: true`
protects the supersaturation term from a transiently-negative ion state.

**Chemical-P removal network + AD limitation at extreme supersaturation.**
[`precipitation_metal_phosphate.yaml`](aquakin/networks/precipitation_metal_phosphate.yaml)
is the second worked network: ferric / aluminium dosing precipitates
orthophosphate as the very insoluble phosphates FePO₄ / AlPO₄ (after the
plant-wide P/S/Fe extension of Flores-Alsina et al. 2016), while the same dosed
metal competes to form the hydroxides Fe(OH)₃ / Al(OH)₃ (the `hydroxide`
fraction). The metal hydroxide buffers the free metal, so it sets a
**pH-dependent floor** on the achievable phosphate — chemical-P removal worsens
at higher pH (more OH⁻) and needs a metal dose in excess of stoichiometric. Its
forward behaviour is exact (P removal, the pH trend, machine-precision Fe/Al/P
conservation), but the metal phosphates are so insoluble that a far-from-
equilibrium dose sits at `SI ~ 14`, where the SI-driven rate Jacobian is `~1e13`:
the L-stable `Kvaerno5` damps this so the **forward solve is fine**, but **no
sensitivity method survives the initial transient** (reverse adjoint, even with a
`dtmax` cap; and the cap-free `forward_sensitivity`/`DirectAdjoint` — all return
non-finite). This is the extreme end of the documented stiff-AD spectrum and is
intrinsic to the chemistry (lowering `order` 2→1 cuts the Jacobian ~7 decades and
still fails). So with the **default power-law** kinetics this network is a
forward-simulation demonstration; `precipitation_struvite_calcite` (modest
`SI ~ 1–3`) remains the differentiable / calibratable power-law example. The
`hydroxide` *engine* path is itself AD-clean at moderate supersaturation
(verified on a mild `M(OH)₂` toy in the test suite — `jax.grad` through the solve
is finite).

### Differentiable ultra-insoluble precipitation — algebraic equilibrium + bounded driver (#295)

The `~1e13` power-law Jacobian is **not** the documented `dtmax`/`stable_adjoint`
overflow (where the per-step operator `I − γ·dt·J` stays well-conditioned): here
that operator is genuinely **near-singular** (cond `~1e13` even at tiny `dt`), so
*every* sensitivity path fails — including the hand-written discrete adjoint
(`stable_adjoint`), which NaNs on the multi-mineral network. The lesson (verified
numerically): you must **reduce the primal stiffness**, not swap the adjoint; the
true sensitivity is tame (finite differences on the finite forward solve give an
`O(0.01)` gradient), so the pathology is purely the through-the-transient AD.
Grounded in what geochemistry codes do, two opt-in fixes restore
differentiability while the default power law is **unchanged**:

**(A) Algebraic equilibrium — `mode: equilibrium`**
([`core/precipitation_equilibrium.py`](aquakin/core/precipitation_equilibrium.py)).
Solve the precipitation *equilibrium* algebraically instead of integrating the
stiff kinetics — the geochemistry-standard equilibrium-phase problem: find the
phase amounts and dissolved ions that satisfy mass balance **and** the mineral
complementarity (every precipitated mineral on its solubility `IAP = Ksp`, every
absent one undersaturated, coupled across the shared ions). It is the MINEQL /
PHREEQC structure — unknowns are log-free-ion + phase amounts, the complementarity
written with a smoothed Fischer–Burmeister function — solved by a **fixed-iteration
safeguarded Newton scan** (the `solve_ph` pattern): ε-continuation on the
complementarity smoothing, residual-adaptive Levenberg–Marquardt damping, and a
bounded log-free-ion step make it globally convergent even when a component is
driven to near-complete consumption (free ion `~1e-18` mol/L). The converged
solution is **differentiated via the implicit function theorem** (the forward
scan runs under `stop_gradient`; one Newton step on the converged residual
attaches the exact `−(∂G/∂w)⁻¹∂G/∂θ` sensitivity in a single linear solve — *not*
a backprop through the iterations, which is huge and unnecessary), and the
algebraic Jacobian it inverts is well-conditioned, so the `1e13` stiffness is
gone. A mineral declares `mode: equilibrium` + a `solid:` species; the engine
exposes `Xeq_<name>` (the equilibrium phase amount) and
**`CompiledNetwork.precipitation_equilibrium(C, conditions)`** returns the
equilibrium-projected state (each equilibrium solid set to `Xeq`, dissolved ions
rebalanced — mass-conserving). Verified on the metal-phosphate: equilibrium P
removal reproduces the kinetic `t→∞` limit, mass closes exactly, the pH trend is
right, and `jax.grad` (w.r.t. dose, pH) is finite. **This is "solve, don't
integrate":** it covers the equilibrium-outcome sensitivity / calibration use
(#295's impact list) but is not an in-ODE reaction (embedding the per-RHS Newton
solve inside the time integration is impractically slow — a documented future
optimization needing solver warm-starting). The solver van't Hoff-corrects each
`Ksp` with temperature (the per-mineral `dH_sp`, same form / reference temperature
as the kinetic engine), so the equilibrium tracks `T` consistently with the
kinetic path. Kinetic minerals in the same block are untouched
(`build_precipitation_derived_fn` skips `mode: equilibrium` minerals; the
equilibrium engine handles them and composes after the speciation pH like the
kinetic one).

**(B) Bounded-driver kinetics — `supersaturation_form: bounded`**
([`core/precipitation.py`](aquakin/core/precipitation.py)). For a differentiable
*dynamic* solve, replace the power-law factor `sign(σ)·|σ|^order` with the
thermodynamically-grounded **bounded driver** `R = tanh(SI/(2ν)·ln10) =
(Ω^{1/ν}−1)/(Ω^{1/ν}+1)` (bounded in `(−1, 1)`, `0` at `SI = 0`, `±1` far from
saturation). The rate Jacobian is then `~k` (non-stiff), so a reverse gradient
through the *time integration* of the ultra-insoluble network is finite, and the
steady state (`R = 0` ⇔ `SI = 0`) is the same equilibrium the projection gives —
verified to agree. The driver is a per-mineral `supersaturation_form: bounded`
flag (default `power`); the reaction expression `k·X·{R}` is unchanged, so it is a
drop-in. The trade-off is a slower precipitation *rate* far from saturation (the
*endpoint* is unchanged); raise `k` to reach equilibrium faster.
[`precipitation_metal_phosphate_equilibrium.yaml`](aquakin/networks/precipitation_metal_phosphate_equilibrium.yaml)
(A) and
[`precipitation_metal_phosphate_bounded.yaml`](aquakin/networks/precipitation_metal_phosphate_bounded.yaml)
(B) are the worked examples; `tests/integration/test_precipitation_equilibrium.py`
covers both plus the solver complementarity and the schema validation.

---

## Package Structure

```
aquakin/
│
├── aquakin/
│   ├── __init__.py                  # public API + jax x64 config
│   │
│   ├── core/
│   │   ├── nodes.py                 # ASTNode base class + all node types
│   │   ├── parser.py                # recursive descent parser -> AST
│   │   ├── vector_kernel.py         # vectorized rate kernel: intern subexprs +
│   │   │                            #   batch each primitive (bit-identical to the
│   │   │                            #   scalar stack, smaller jaxpr -> faster compile)
│   │   ├── network.py               # CompiledNetwork dataclass + compile()
│   │   ├── stoich_resolve.py        # `auto`/`?` coefficient resolver: solve a
│   │   │                            #   conservation-determined coefficient from the
│   │   │                            #   composition table + conserved_for (numeric, or
│   │   │                            #   a derived param-expression for yield-dependent)
│   │   ├── conditions.py            # SpatialConditions dataclass
│   │   ├── context.py               # CompileContext dataclass
│   │   ├── ph_solver.py             # differentiable charge-balance pH solver
│   │   │                            #   (safeguarded Newton-bisection: globally convergent, no NaN)
│   │   ├── speciation.py            # speciation block -> derived pH condition fn
│   │   ├── precipitation.py         # precipitation block -> derived SI_/R_ condition fn
│   │   │                            #   (Kazadi Mbamba 2015; kinetic power-law OR bounded driver;
│   │   │                            #   reuses ph_solver constants/activities)
│   │   ├── precipitation_equilibrium.py # mode: equilibrium -> algebraic equilibrium solve
│   │   │                            #   (MINEQL/PHREEQC: log-free-ion + phase amounts, smoothed-FB
│   │   │                            #   complementarity; IFT-differentiable; -> Xeq_/projection)
│   │   └── units.py                 # prettify_units: plain-ASCII unit exponents -> Unicode superscripts
│   │
│   ├── schema/
│   │   ├── network_spec.py          # Pydantic models
│   │   ├── inheritance.py           # network.extends: merge a base + add/modify/remove
│   │   └── loader.py                # YAML -> (resolve extends) -> Pydantic -> CompiledNetwork
│   │
│   ├── integrate/
│   │   ├── _common.py               # shared helpers: atol coercion, _run_diffeqsolve,
│   │   │                            #   solve_chemistry (the one stoich-hoist + RHS + solve
│   │   │                            #   factory the Batch/PFR/Particle/CFD reactors all call,
│   │   │                            #   parameterised by cond_fn / rate_scale / saveat),
│   │   │                            #   validate_t_eval; init_solver_settings /
│   │   │                            #   resolve_state_atol / validate_C0_params (shared reactor
│   │   │                            #   construction + validation); Reactor & ConditionedReactor
│   │   │                            #   Protocols (the latter adds `conditions`; sensitivity
│   │   │                            #   requires it. CFDReactor has step() not solve() -> not a
│   │   │                            #   Reactor); _HasNamedSpecies mixin (C_named/units_named/
│   │   │                            #   to_dataframe/to_csv) + build_dataframe/require_pandas
│   │   ├── batch.py                 # BatchReactor, BatchSolution
│   │   ├── biofilm.py               # BiofilmReactor (layered 1-D diffusion-reaction)
│   │   ├── pfr.py                   # PlugFlowReactor, PFRSolution
│   │   ├── particle.py              # Track, ParticleTrackReactor, integrate_ensemble
│   │   ├── cfd.py                   # CFDReactor (Option C runtime coupling)
│   │   ├── sensitivity.py           # sensitivity(), fit(), dgsm()
│   │   ├── experiments.py           # compare_scenarios(), monte_carlo(),
│   │   │                            #   optimize_design(): scenario comparison +
│   │   │                            #   Monte-Carlo uncertainty + constrained design
│   │   │                            #   optimization on the fn(x)->output contract
│   │   │                            #   (reuses dgsm's Sobol QMC; AD-gradient NLP)
│   │   ├── forward_sensitivity.py   # solve_sensitivity / forward_sensitivity:
│   │   │                            #   augmented [y; S] variational solve giving
│   │   │                            #   cap-free exact stiff sensitivities
│   │   ├── _simultaneous_corrector.py # CVODES simultaneous-corrector lineax
│   │   │                            #   solver (shared_factor=True, Option A):
│   │   │                            #   factorise the shared diagonal block once
│   │   ├── discrete_adjoint.py      # implicit_euler_adjoint_solve /
│   │   │                            #   esdirk_adjoint_solve (Kvaerno5): cap-free
│   │   │                            #   REVERSE-mode gradient via a hand-written
│   │   │                            #   discrete adjoint (no autodiff through the solve)
│   │   ├── colored_jacobian.py      # ColoredVeryChord: sparse (column-compressed
│   │   │                            #   colored-AD) per-step Jacobian for the implicit
│   │   │                            #   stage solve; Plant.solve(colored_jacobian=True)
│   │   ├── forward_solve.py         # forward_solve: lean non-AD adaptive ESDIRK
│   │   │                            #   (lax.while_loop, no diffrax adjoint/optimistix/
│   │   │                            #   lineax); Plant.solve(forward_fast=True) -- ~3x
│   │   │                            #   faster compile, ~1.3-1.9x run, forward-only
│   │   ├── calibrate.py             # calibrate(): transforms, priors, Laplace posterior,
│   │   │                            #   multistart, free initial conditions, Gauss-Newton
│   │   │                            #   optimizer, posterior-predictive bands
│   │   ├── events.py                # Event + solve_with_events: located events
│   │   │                            #   (time / state root-crossing) + AD-safe state
│   │   │                            #   resets / mode switches, via a segmented solve
│   │   └── profile.py               # profile_likelihood(): parameter / initial-condition
│   │                                #   profile-likelihood identifiability analysis
│   │
│   ├── transport/
│   │   └── openfoam/
│   │       ├── bridge.py            # SpatialConditions <-> OpenFOAM interface
│   │       └── README.md            # coupling contract documentation
│   │
│   ├── networks/
│   │   ├── ozone_bromate.yaml       # with explicit OH radical chemistry
│   │   ├── uv_h2o2.yaml             # UV/H2O2 AOP
│   │   ├── asm1.yaml                # Activated Sludge Model No. 1
│   │   ├── asm2d.yaml               # ASM2D (bio-P + denitrification)  [SUMO-derived]
│   │   ├── asm2d_chemp.yaml         # ASM2D + saturation-driven chemical-P (ferric); extends: asm2d
│   │   ├── asm2d_tud.yaml           # Delft TUD variant of ASM2D       [SUMO-derived]
│   │   ├── asm3.yaml                # ASM3 (storage products replace hydrolysis)  [SUMO-derived]
│   │   ├── asm3_2step.yaml          # ASM3 + two-step nitrification/denitrification (explicit NO2; Kaelin 2009)
│   │   ├── asm3_2step_n2o.yaml      # asm3_2step + two-pathway AOB N2O (NH2OH/NO/N2O; Pocquet 2016); extends: asm3_2step
│   │   ├── asm3_2step_anammox.yaml  # asm3_2step + anammox (NH4+NO2->N2; Strous 1998/1999); extends: asm3_2step
│   │   ├── asm3_2step_comammox.yaml # asm3_2step + comammox complete nitrifier (Kits 2017); extends: asm3_2step
│   │   ├── asm3_biop.yaml           # ASM3 + bio-P extension           [SUMO-derived]
│   │   ├── adm1.yaml                # ADM1 anaerobic digestion (BSM2 form, Rosen-Jeppsson
│   │   │                            #   2006); complete: liquid + gas headspace, state-derived
│   │   │                            #   pH with explicit S_cat/S_an strong-ion states
│   │   ├── wats_sewer.yaml          # original reference-book WATS (Tables 9.1-9.4)
│   │   ├── wats_sewer_extended.yaml  # extended WATS (+ nitrate/methane/elemental-S, state-derived pH)
│   │   ├── wats_sewer_extended_*.yaml # extended-model structural variants + v0
│   │   ├── wats_sewer_khalil_paper*.yaml # paper-faithful Khalil (2025) model + variants
│   │   ├── wats_sewer_khalil_thesis.yaml # thesis-faithful Khalil model
│   │   ├── wats_sewer_khalil_paper_balanced_biofilm.yaml  # layered-biofilm variant ({A_V} areal)
│   │   ├── wats_sewer_khalil_paper_balanced_biofilm_biomass.yaml  # per-layer-biomass biofilm (heterotroph)
│   │   ├── wats_sewer_khalil_paper_balanced_biofilm_multispecies.yaml  # + X_SRB/X_MA/X_SOB groups
│   │   ├── precipitation_struvite_calcite.yaml  # mineral precipitation (Kazadi Mbamba 2015): struvite + calcite
│   │   ├── precipitation_metal_phosphate.yaml   # iron/Al chemical-P removal (FePO4/AlPO4 + Fe(OH)3/Al(OH)3 hydroxide fraction); kinetic power-law (AD-limited)
│   │   ├── precipitation_metal_phosphate_equilibrium.yaml  # mode: equilibrium (algebraic projection, differentiable)
│   │   └── precipitation_metal_phosphate_bounded.yaml      # supersaturation_form: bounded (differentiable dynamics)
│   │
│   │   # wats_sewer_khalil_paper (paper) is the paper-active core augmented with the
│   │   #   dormant full-WATS aerobic pieces by networks/_make_khalil_paper.py;
│   │   # wats_sewer_khalil_thesis is generated from wats_sewer_extended.yaml by
│   │   #   networks/_make_khalil_thesis.py; the structural variants by
│   │   #   networks/_make_khalil_variants.py;
│   │   # wats_sewer_khalil_paper_balanced_biofilm is generated by
│   │   #   networks/_make_khalil_balanced_biofilm.py -- it splits the 3 composite
│   │   #   bulk+biofilm reactions (fermentation, fast/slow hydrolysis) into
│   │   #   _bulk ([X_BH]) and _biofilm (eps*{X_BF}*{A_V}) halves so the depth-
│   │   #   resolved BiofilmReactor can run bulk reactions in the bulk and biofilm
│   │   #   reactions in the layers. Same chemistry; the lumped balanced model is
│   │   #   its well-mixed limit. Depth-resolved, nitrate is consumed in the outer
│   │   #   layers and never reaches the deep methanogens (Sun et al. 2014), so
│   │   #   methane accumulates toward the wall and acetate is diffusion-limited --
│   │   #   the stratification the lumped model cannot represent. This variant keeps
│   │   #   the areal {A_V} device: biofilm activity is spatially UNIFORM, so it
│   │   #   cannot represent a biomass GRADIENT (the next variant does).
│   │   # wats_sewer_khalil_paper_balanced_biofilm_biomass is generated by
│   │   #   networks/_make_khalil_balanced_biofilm_biomass.py -- biomass is an
│   │   #   explicit per-layer growing/decaying STATE: every biofilm process is
│   │   #   driven by the LOCAL volumetric [X_BH] (no {A_V}/{X_BF}), run in every
│   │   #   compartment (no phase split; the biomass concentration -- low in bulk,
│   │   #   high in the layers -- carries the bulk/biofilm distinction). Run in
│   │   #   BiofilmReactor with biofilm_reactions=None, a stratified C0, and
│   │   #   fixed_mask holding only the inert solids; the biomass gradient then
│   │   #   evolves. INCREMENT 1 (heterotroph): the sulfur/methane processes are
│   │   #   interim-coupled to [X_BH] -- a stand-in pending their own
│   │   #   functional-group biomass (X_SRB, methanogens, S-oxidizers), which in
│   │   #   reality stratify DIFFERENTLY from heterotrophs (Sun 2014: SRB outer,
│   │   #   methanogens inner). So NO sulfur/sulfate conclusions may be drawn from
│   │   #   this increment -- only the heterotroph/VFA result. FINDING (JRN-055,
│   │   #   increment 1, reviewer-checked): with a real per-layer biomass gradient,
│   │   #   depth resolution still does NOT reproduce the measured bulk VFA plateau
│   │   #   (flat ~13 mgCOD/L held while nitrate->0), across biofilm thickness
│   │   #   0.8-3 mm. There is a hard VFA-vs-nitrate trade-off: every configuration
│   │   #   that consumes the dosed nitrate (as the data require) crashes bulk VFA
│   │   #   to ~0, because VFA is consumed wherever nitrate persists
│   │   #   (denitrification) and, in any nitrate-free deep zone, by methanogenesis
│   │   #   (nitrate-inhibited elsewhere). [CORRECTION: an earlier note here claimed
│   │   #   the deep VFA is "trapped behind the denitrifying zone and cannot export
│   │   #   to the bulk (bulk VFA <=0.2), reproducing Jiang Fig 8". That was
│   │   #   OVERSTATED -- a transient of the donor-limited regime: in the CLOSED
│   │   #   batch the dosed 30 mgN is not globally cleared within 5 h, so the outer
│   │   #   denitrifying zone persists and consumes exported VFA; it is not a steady
│   │   #   export barrier, and the closed batch does not reproduce Jiang's
│   │   #   continuous-flow Fig 8 (deep S_A peak + bulk S_A~0) as a SIMULTANEOUS
│   │   #   state.] The robust conclusion is the negative empirical one: sparing
│   │   #   VFA enough to match the plateau requires producing enough donor that
│   │   #   nitrate clears early -- i.e. the same supra-literature hydrolysis the
│   │   #   lumped model needs -- so the plateau is consistent with a bulk-phase
│   │   #   mobilization effect, not with biofilm depth structure. A genuinely
│   │   #   decisive test (deferred) is to run the model continuous-feed / CSTR-
│   │   #   coupled (sustained nitrate, as in Jiang/Sun and the real sewer) rather
│   │   #   than as the closed pump-off batch.
│   │   # wats_sewer_khalil_paper_balanced_biofilm_multispecies is generated by
│   │   #   networks/_make_khalil_balanced_biofilm_multispecies.py from the
│   │   #   increment-1 _biofilm_biomass model. It resolves the interim-coupling
│   │   #   confounder: the sulfur/methane processes now grow their OWN per-layer
│   │   #   functional-group biomass instead of riding on [X_BH] -- X_SRB (sulfate
│   │   #   + elemental-S reducers on S_B), X_MA (acetoclastic + hydrogenotrophic
│   │   #   methanogens), X_SOB (nitrate-driven + aerobic sulfide/S0 oxidisers).
│   │   #   Each process keeps its Monod form but is driven by [X_group] and
│   │   #   produces biomass at a literature YIELD with COD/S/N-conserving
│   │   #   stoichiometry (-> the original electron balance as Y->0); each group
│   │   #   decays first-order to inert X_I. The old areal rate constants
│   │   #   (k_h2s_acid, k_sII_anox_f, ...) are superseded by growth rates mu_* and
│   │   #   auto-pruned. Yields/electron-stoichiometry are fixed from the
│   │   #   literature (Jiang 2009 Table 2 for SRB; ADM1/Sun 2014 for X_MA;
│   │   #   Mohanakrishnan 2009 / Nielsen 2005 for X_SOB); the mu_* are
│   │   #   literature-range placeholders (mu and biofilm biomass density are
│   │   #   confounded -- only their product is grounded) and the variant is
│   │   #   meant to be re-calibrated. Run in BiofilmReactor with
│   │   #   biofilm_reactions=None, a stratified C0 (X_BH/X_SRB/X_MA/X_SOB high in
│   │   #   the layers), fixed_mask holding only X_I. Conserves COD/S/Fe/P (N lost
│   │   #   only via denitrification N2). Built so the calibration is PHYSICAL:
│   │   #   the optimizer can no longer abuse an [X_BH]-coupled sulfur term to fit
│   │   #   sulfide/sulfate, since each process has its own biomass that grows,
│   │   #   decays and stratifies by its own kinetics (SRB/methanogens stratify
│   │   #   differently from heterotrophs -- Sun 2014).
│   │
│   └── utils/
│       ├── latex.py                 # AST -> LaTeX rate expressions
│       ├── balance.py               # mass / electron (COD) conservation checks
│       ├── units.py                 # currency-aware dimensional check of rate
│       │                            #   expressions (network.check_units); distinct
│       │                            #   from core/units.py, which only formats units
│       └── rtd.py                   # RTD analysis (E-curve, Morrill index)
│
├── tests/
│   ├── unit/
│   │   ├── test_parser.py
│   │   ├── test_nodes.py
│   │   ├── test_loader.py
│   │   └── test_network.py
│   ├── integration/
│   │   ├── test_batch_simple.py     # validates against analytical solution
│   │   └── test_pfr_simple.py
│   ├── validation/
│   │   ├── test_bromate_vongunten.py# validates against published data
│   │   ├── test_adm1_bsm2_steadystate.py # ADM1 vs published BSM2 AD steady state
│   │   └── test_takacs_vs_bsm1_reference.py # Takács settler vs published BSM1 settler derivative
│   └── fixtures/
│       └── simple_network.yaml      # minimal 2-species toy network for unit tests
│
├── examples/
│   ├── batch_bromate.py
│   ├── lagrangian_demo.py
│   ├── sensitivity_demo.py
│   ├── bsm1_dry_weather.py            # BSM1 open-loop steady state
│   ├── bsm1_target_srt.py             # hit a target sludge age (SRT) by solving for Qw
│   ├── bsm1_dynamic_influent.py       # BSM1 dry-vs-rain dynamic influent (warm-started)
│   ├── bsm2_steady_state.py           # BSM2 two-network open-loop steady state
│   ├── bsm2_seasonal_temperature.py   # BSM2 cold->warm nitrification effect
│   ├── dgsm_sensitivity_screen.py     # DGSM global sensitivity, forward==reverse
│   ├── wats_nitrate_dosing_calibration.py  # synthetic sewer rate recovery (calibrate + Laplace)
│   ├── bsm2_ghg_cost_report.py     # GHG (N2O/CO2e) + cost reporting + scenario KPI table
│   ├── event_handling.py           # located events: scheduled re-dosing + terminal cut-off
│   └── adjoint_speed_benchmark.py  # stable_adjoint vs capped jax_adjoint timing
│   # NOTE: the wats_sewer_extended batch-fitting / calibration / sensitivity scripts and
│   # their measurement data live in the separate paper-reproduction repository,
│   # not here (this repo ships only the reusable library + networks).
│   # wats_nitrate_dosing_calibration.py is a self-contained *synthetic* demo of
│   # the calibration API, not the paper reproduction.
│
├── docs/
│   ├── index.md
│   ├── network_format.md
│   └── adding_networks.md
│
├── pyproject.toml
├── README.md
├── CLAUDE.md                        # this file
└── LICENSE
```

**Key structural rules:**
- `core/` has no Pydantic dependency — only dataclasses and JAX
- `schema/` is the only module that imports Pydantic
- `networks/` ships with the package, accessible via `importlib.resources`
- Unit tests use only `tests/fixtures/simple_network.yaml`, never `ozone_bromate.yaml`

---

## Public API

```python
import aquakin

# Loading
network = aquakin.load_network("ozone_bromate")
network = aquakin.load_network_from_file("path/to/network.yaml")

# Inspection
network.name
network.species
network.parameters
network.conditions_required
network.species_units                # {species: units} carried from the YAML
network.species_descriptions         # {species: description}
network.units_of("SNH")              # "g_N/m³" (YAML "g_N/m3" prettified; raises KeyError on unknown)
network.description_of("SNH")        # "Ammonia + ammonium nitrogen"
network.time_unit                    # "d" | "s" | "h" | "min" | None — integration
                                     #   time unit, inferred from the rate-constant
                                     #   units (the inverse-time token they share).
                                     #   t_span / t_eval are in THIS unit; it differs
                                     #   by network (ozone/UV "s", ASM/ADM/WATS "d"),
                                     #   so there is no global time unit. None only
                                     #   when it can't be inferred — a network that
                                     #   declares no rate-constant time unit, or whose
                                     #   rate constants disagree on one. (All shipped
                                     #   networks, including the SUMO-derived ASM ones,
                                     #   resolve to "d"/"s".)
network.default_concentrations()     # jnp.array (all YAML defaults)
network.default_parameters()         # jnp.array
network.summary()                    # human-readable table (species listed with units)
network.to_latex()                   # LaTeX rate expressions
# Project a composition onto its mineral precipitation EQUILIBRIUM (only for a
#   precipitation network with `mode: equilibrium` minerals): solve IAP=Ksp with
#   complementarity, mass-balanced, and return the equilibrium-projected state.
#   Differentiable via the implicit function theorem -- the non-stiff alternative
#   to integrating an ultra-insoluble mineral's ~1e13 kinetics (issue #295).
network.precipitation_equilibrium(C, conditions)   # -> equilibrium state (n_species,)
# Solutions carry the labels too: solution.units_named("SNH") for axis/columns,
#   and solution.time_unit for the time axis (delegates to network.time_unit).

# Dimensional ('unit') consistency check of the rate expressions (issue #161).
# Currency-AWARE: units are a free abelian group over currency tokens
# {g, mol, m, L, d, s, COD, TSS, N, O2, P, S, C, ...} where COD/N/O2/TSS are
# DISTINCT base dimensions, so g_COD/m3 vs g_N/m3 are different (a plain SI check waves them
# through). Walks each rate AST: +/- operands must match, monod/monod_ratio
# saturation args must share a currency (-> dimensionless), and the root must
# resolve to currency/volume/time (e.g. g_COD/m3/d, mol/L/s). Catches a dropped
# concentration factor, a wrong rate-constant exponent, a Monod mixing
# currencies. It also runs ONE cross-reaction rule: every rate constant drives
# dC/dt against the same integration time, so all rates must share one
# inverse-time unit -- a network mixing 1/d and 1/s rates is malformed (its RHS
# sums terms on inconsistent time bases) yet each rate passes the per-rate root
# check on its own, so the disagreement is flagged once at network scope
# (reaction "(network)", location "time unit"). The shipped networks all share
# one time unit, so this never fires on them. ADVISORY + opt-in: never run at load, never raises; a blank or
# unparseable unit is treated as unknown and skipped (no false alarm), so an
# empty result means "no inconsistency among the declared, parseable units", not
# a proof. Stoichiometry (deliberately cross-currency yields) is OUT of scope --
# that is conservation, via check_conservation / utils/balance.py.
network.check_units()                # -> list[UnitWarning] (reaction, location, detail)
network.check_units(check_root=False)  # local rules only (skip currency/vol/time root)
aquakin.parse_units("g_COD/m3")      # -> Dimension (or None if unknown); aquakin.UnitWarning

# Conservation (mass / electron balance). The currency-aware companion to
# check_units: dots the per-species composition table against the stoichiometry,
# so a wrong electron-acceptor (O2/NO3) demand breaks COD and a wrong product
# split breaks an elemental (S/N/P/Fe) balance. ADVISORY + opt-in like check_units
# (never run at load, never raises on a violation -- it returns the list).
network.composition()                # -> {species: {quantity: content}}; declared
                                     #   `composition:` metadata, else the shipped
                                     #   role-based table (composition_table) for
                                     #   ASM/ADM, else {}.
network.check_conservation()         # -> [(reaction, quantity, residual)] above tol
network.check_conservation(tol=1e-2, quantities=["COD"], params=p)  # restrict / calibrated
network.check_nitrogen()             # -> [(reaction, residual)]; credits nitrate -> N2 gas
# Raises ValueError if no composition is available (declare a `composition:` per
# species or pass composition=...). Quantity content lives in the network YAML
# (the WATS family) or the shipped composition_table (ASM/ADM); both feed one API.
# The shipped ASM1/2d/3, ozone, UV and WATS networks are unit-clean (0
# warnings); ADM1 is clean on its dissolved/biological reactions but the check
# DOES flag the three gas_outflow reactions -- the BSM2 gas headspace carries
# H2/CH4 in COD (kgCOD/m3) and CO2 in carbon (kmolC/m3) and the /16, /64 molar
# masses are bare numbers, so the partial-pressure sum mixes currencies. This is
# a documented inherent characteristic of the BSM2 gas phase, not a model error
# (homogenising it would need molar-mass parameters, which would change ADM1's
# parameter vector). Pressure (`bar`) and temperature (`K`) are recognised unit
# tokens so the gas units parse, but they are outside the canonical
# currency/volume/time root form. (A bare dimensionless constant added to a
# concentration would be a ConstantNode and treated as dimension-neutral, so NOT
# flagged; ADM1 no longer relies on such a guard -- its valerate/butyrate
# competition uses `safe_div` instead of a `+ 1.0e-6` denominator.) Conditions
# carry advisory units too (`pH: "-"`, `T: "K"`). All regression-guarded in
# tests/unit/test_units_check.py.

# By-name vector builders (avoid .at[species_index[...]].set() chains). The
# dict form is primary -- many species/param names are not valid Python
# identifiers ("Br-", the namespaced "O3_Br_direct.k1"); kwargs are a
# convenience for identifier-safe names. Unknown names raise with a
# difflib "did you mean?" hint.
network.concentrations({"O3": 1e-4, "Br-": 1e-5})   # YAML defaults + overrides
network.concentrations({"SS": 60.0}, base="zero")   # FEED: unlisted species = 0,
                                                    #   not at their reference value
network.influent({"SS": 60.0, "SNH": 25.0}, Q=18446.0, T=288.15)  # zero-based,
                                                    #   constant-in-time InfluentSeries
                                                    #   (== InfluentSeries.constant(net, ...))
network.parameter_values({"O3_Br_direct.k1": 175.0})
network.atol({"OH": 1e-20}, default=1e-12)          # per-species tolerance vector

# Conditions  (n_locations defaults to 1, for the 0-D batch case)
conditions = aquakin.OperatingConditions(pH=7.5, T=293.15)   # 0-D alias (1 location)
conditions = network.default_conditions().with_(T=283.15)    # edit from YAML defaults
conditions = aquakin.SpatialConditions.uniform(pH=7.5, T=293.15)
conditions = aquakin.SpatialConditions(fields={"pH": jnp.array([...]), ...})  # PFR/CFD

# Batch reactor  (params defaults to network.default_parameters())
reactor = aquakin.BatchReactor(network, conditions)
solution = reactor.solve(C0, t_span=(0.0, 600.0), t_eval=t_eval)
solution = reactor.solve(C0, t_span, t_eval, params=params)   # t_span is the 2nd
#   positional arg; params is KEYWORD-ONLY, so a positional t_span tuple can never
#   land in it -- reactor.solve(C0, (0.0, 600.0)) just works (no shape-error footgun).
# t_span / t_eval are in the network's native time unit (network.time_unit); pass
# time_unit= to work in another unit. The input times are converted into the
# native unit for the solve (rate constants unchanged) and solution.t is reported
# back in the requested unit (solution.time_unit is set to it). Raises if the
# network's own time unit is undeclared (network.time_unit is None). Wired on
# BatchReactor / BiofilmReactor / Plant.solve (PFR is space-indexed -> N/A; the
# AD/fitting paths -- solve_sensitivity / calibrate / sensitivity -- stay native).
solution = reactor.solve(C0, t_span=(0.0, 24.0), t_eval=t_eval, time_unit="h")
solution.t                           # (n_t,)
solution.C                           # (n_t, n_species)
solution.C_named("BrO3-")           # one species' trajectory (hinted KeyError on a typo)
solution.C_named_many(["O3", "BrO3-"])  # several at once -> {name: trajectory}
solution.final_named(["O3", "BrO3-"])   # last-point values -> {name: float} (reporting;
                                    #   None = every species). Use C_named(sp)[-1] for a
                                    #   *differentiable* last value -- final_named returns floats.
solution.final                       # == final_named(): every species' last-point value
# These four come from the shared _HasNamedSpecies mixin, so every single-vector
# solution has them (Batch/PFR(x-indexed: "final" = outlet)/Track/Biofilm) AND the
# reconstructed StreamSeries. PlantSolution mirrors them per unit:
#   plantsol.C_named_many("tank5", ["SNH","SNO"]) / plantsol.final_named("tank5"[, [...]])
#   (and plantsol.final_state for the whole flat vector -> states_by_unit).
solution.to_dataframe()              # time-indexed pandas DataFrame, species columns
solution.to_csv("run.csv")           # delegates to to_dataframe().to_csv(...)
# to_dataframe(units_in_columns=False): bare species columns + df.attrs["units"];
#   True -> "SNH [g_N/m³]" labels. to_csv defaults units_in_columns=True (a CSV
#   can't carry attrs). pandas is the optional `dataframe` extra. Every solution
#   has it (Batch/PFR(x-indexed)/Track/Biofilm); BiofilmSolution.to_dataframe(
#   profile=True) gives the depth-resolved (t, compartment) MultiIndex + depth
#   column; StreamSeries adds a Q column; PlantSolution.to_dataframe(unit="tank5").
solution.plot("SNH")                 # matplotlib Axes: one species over the
solution.plot(["SNH", "SNO"], ax=ax) #   independent axis, no boilerplate
# plot(species=None|str|iterable, ax=None, **plot_kwargs) -> matplotlib.axes.Axes.
#   The x-axis is labelled with the network's time unit (PFR: "axial position
#   [m]"); a single species labels the y-axis with its units, several get a
#   legend; None plots every species. matplotlib is the optional `plot` extra
#   (also in the `test` extra). Same mixin as to_dataframe, so every single-vector
#   solution has it (Batch/PFR/Track/Biofilm) AND StreamSeries; PlantSolution is
#   per unit -- PlantSolution.plot(unit, species=None, ax=None). Unknown species /
#   non-concentration unit raise the same hinted errors as C_named.

# Plug flow reactor
reactor = aquakin.PlugFlowReactor(network, conditions, n_points, length, velocity)
solution = reactor.solve(C0, params=params)            # params keyword-only
solution.x                           # (n_points,)
solution.C                           # (n_points, n_species)

# Layered biofilm reactor (1-D diffusion-reaction over biofilm depth)
# Resolves the biofilm into n_layers between a well-mixed bulk and the (no-flux)
# wall, so penetration-controlled processes are captured (an acceptor consumed in
# the outer layers never reaches deep organisms; deep uptake is diffusion-limited)
# -- the lumped area-to-volume reactor cannot represent this. Solubles diffuse
# (Fick, D_eff) + exchange with the bulk across a boundary layer. The boundary
# layer is liquid, so its mass-transfer coefficient uses the free-water
# diffusivity (boundary_diffusivity=D_w); leaving it None reuses the reduced
# in-biofilm D_eff and understates bulk<->film exchange. Two species roles are
# DECOUPLED: diffusion (soluble_mask: S*/sumS diffuse, X* do not) is separate
# from being held fixed (fixed_mask: the "mature biofilm" sustained, non-depleting
# source/sink). The default fixed_mask = ~soluble_mask (every particulate fixed)
# is right for INERT particulates (biomass, inert solids) but WRONG for REACTIVE
# particulates that do not diffuse yet must still react -- elemental sulfur X_S0,
# precipitated FeS -- whose inventory genuinely drains/fills. Freezing such a
# species turns it into an unbounded source/sink and silently breaks mass balance
# (e.g. a frozen X_S0 makes the nitrate-driven X_S0->SO4 oxidation a non-depleting
# sulfate source). For those networks pass fixed_mask holding only the inert
# biomass/solids fixed. The same CompiledNetwork runs in every compartment, so
# identical chemistry behaves differently once depth is resolved (Wanner & Gujer
# 1986; Jiang et al. 2009; Sun et al. 2014). In the well-mixed limit it reduces to
# BatchReactor *under the fixed-particulate assumption* (exact only for species
# that are fixed on both sides; particulates that evolve in a plain BatchReactor
# but are held fixed here diverge over finite time).
# A WATS-style network has two phases: bulk-suspended reactions (carry [X_BH]) and
# biofilm reactions (carry the {A_V} area factor). biofilm_reactions=[names...]
# runs those reactions in the LAYERS only and the rest in the BULK only -- an
# explicit per-reaction phase split (no reliance on a zeroed biomass state). A
# composite term like bio_hf=[X_BH]+eps*{X_BF}*{A_V} is handled by splitting the
# reaction into _bulk ([X_BH]) and _biofilm (eps*{X_BF}*{A_V}) halves in the
# network YAML; biofilm rate constants are areal (per m^2), so set A_V=1/thickness
# per layer (the lumped model is then the well-mixed limit, conserving mass).
#
# BIOFILM-GROWTH / MATURATION features (all off by default; used to mature a
# multispecies biofilm to its operating state before a downstream experiment):
#   - max_density (per-species rho_i^f, gCOD/m^3) + packing_fraction: the
#     Jiang 2009 Eqs 8-10 density cap. Biomass GROWTH (the whole reaction, so
#     mass-conserving) is throttled by the remaining space (1 - sum X_i/rho_i /
#     packing). A physical UPPER BOUND only -- on its own it gives NO reachable
#     steady state (biomass drifts to the cap over many months), so it is not the
#     closure.
#   - k_att / attach_mask (Eq 1): bulk particulates attach to the surface layer
#     (k_att*X_bulk), seeding the groups. k_det / detach_mask (Eqs 2-3, lumped to
#     first order): biofilm particulates erode back to the bulk (k_det*X), where
#     they wash out with the feed. DETACHMENT, not the cap, is the steady-state
#     closure: growth = decay + detachment is a chemostat-like fixed point, sets
#     the weeks-to-months maturation timescale, and (as a -k_det Jacobian-diagonal
#     term) conditions the steady state. With no sewer shear data k_det is a
#     calibration knob (low shear -> low k_det -> thicker, denser biofilm).
#   - feed (influent vector) + dilution_rate (Q/V, 1/d): a CSTR feed on the bulk
#     (d_bulk += dilution*(feed-bulk)); the steady bulk is the predicted effluent.
#   - clamp_bulk: hold the bulk as a fixed reservoir (Dirichlet) instead.
#   - steady_state(C0, params, warmup=...): pseudo-transient continuation (PTC)
#     root-find on RHS=0 (aquakin.plant.steady.solve_steady_state), with
#     implicit-function-theorem AD. PTC's per-state pseudo-time damping is robust
#     where the old Newton/Levenberg-Marquardt root-find stalled; for a VERY
#     stiff/slow biofilm whose asymptotic fixed point is hundreds of days out
#     (the multispecies maturation), raise newton_steps or integrate forward to
#     the physical maturation time (~90 d for the Khalil rig) and use that profile
#     as the IC instead.
reactor = aquakin.BiofilmReactor(
    network, conditions, n_layers=6, thickness=8e-4, area_per_volume=50.0,
    diffusivity=1e-4, boundary_layer=1e-4,
    biofilm_reactions=[...])             # names of the {A_V} reactions (run in layers only)
solution = reactor.solve(C0, t_span, t_eval, params=params)  # C0 (n_species,) or (n_layers+1, n_species); params keyword-only
solution.C                           # (n_t, n_species) -- BULK (measurable) trajectory
solution.profile                     # (n_t, n_layers+1, n_species) -- depth-resolved (0=bulk)
solution.depth                       # (n_layers,) layer mid-depths from the surface
solution.profile_named("S_NO")       # (n_t, n_layers+1) depth profile over time

# Sensitivity and fitting  (params defaults to network defaults; t_span/t_eval
# can be passed directly instead of via solve_kwargs)
sens = aquakin.sensitivity(reactor, C0, output_fn=out, t_span=(0.0, 600.0), t_eval=t_obs)
sens.doutput_dparams                 # (n_params,)
sens.doutput_dconditions["pH"]       # (n_locations,) — dict access
sens.ranked_params()

# Forward (variational) sensitivity — integrate S = dC/dθ ALONGSIDE the state,
# with the adaptive controller bounding S too, so the sensitivity is exact and
# finite WITHOUT a dtmax cap (the cap-free alternative for stiff networks; see
# "Differentiating stiff networks" above). Each reactor exposes solve_sensitivity:
sol, S = reactor.solve_sensitivity(
    C0, params, t_span, t_eval,
    sens_params=["mu_h", "q_m"],     # names or int indices of the free params
    sens_rtol=None, sens_atol=None,  # default: rtol_S=rtol, atol_S=atol/|θ_k| (CVODES)
    param_scale=None,                # override the |θ_k| error-control scale
    shared_factor=False,             # True (CVODES simultaneous corrector) not yet implemented
)
# sol : the usual Solution (uncapped primal); S : dC/dθ at the saved times,
#       shape (n_t, n_species, n_sens_params). For a BiofilmReactor S is the
#       BULK (measurable) sensitivity, aligned with sol.C.
res = aquakin.forward_sensitivity(reactor, C0, params, sens_params=[...], t_span=..., t_eval=...)
res.S_named("S_SO4")                 # (n_t, n_sens_params)
res.dC_dparam("S_SO4", "mu_h")       # (n_t,)

# Derivative-based global sensitivity (DGSM) — AD Sobol-total-index analogue.
# fn maps an uncertain-input vector to a scalar OR vector output (it builds
# params / C0 and calls reactor.solve internally). Scrambled-Sobol QMC; seed
# makes it exactly reproducible; bounds the Sobol total-order index per input.
res = aquakin.dgsm(fn, ranges, input_names=names, n_samples=64, seed=0)
res.sobol_total_bound                # (d,) upper bound on Sobol S_j^tot
res.std_error                        # (d,) MC standard error (convergence)
res.ranked()                         # [(name, bound), ...] sorted

# ad_mode= selects the AD direction used to form the per-sample sensitivities
# (identical results to machine precision — purely a performance choice):
#   "reverse" (default) — m reverse passes (one per output), each d-independent.
#                         Best for few outputs and a cheap adjoint.
#   "forward"           — d forward-mode tangents through one solve, m-independent.
#                         Best for many outputs, or when the reverse adjoint is
#                         stiff-inflated (dtmax-capped). REQUIRES the reactor in
#                         fn to use adjoint=aquakin.forward_adjoint() (== diffrax
#                         DirectAdjoint, no diffrax import) — dgsm cannot set the
#                         adjoint for you because fn constructs the reactor.
# (mode= is a deprecated alias for ad_mode=, kept with a DeprecationWarning.)
# If fn returns a vector of m outputs, dgsm returns a list[DGSMResult], one per
# output (each carrying .output_name) — screen all outputs in a single call.
outs = aquakin.dgsm(fn_vec, ranges, output_names=[...], ad_mode="forward")
# Benchmark (tests/ + the JRN-055 reproduction): for a 4-output, 17-input stiff
# batch screen, forward mode is ~2x faster (and lighter on memory) than reverse,
# because reverse pays the stiff adjoint once per output while forward pushes all
# d tangents through one solve. For a single scalar output, reverse is cheaper.

# Scenario comparison and Monte-Carlo uncertainty (integrate/experiments.py).
# Same fn(x)->output contract as dgsm: fn maps a named input VECTOR to a scalar
# or vector output (it builds params/C0 and runs the solve itself). These turn
# the per-solve primitives into the two engineering deliverables.
#
# compare_scenarios -- run several named input sets side by side, tabulate KPIs.
# Scenarios are {name: {input_name: value}} overrides on a baseline vector (a
# scenario states only what it changes), or {name: full_vector}.
sc = aquakin.compare_scenarios(fn, {"base": {}, "fast_AOB": {"muAOB": 1.2}},
                               input_names=["muAOB", "KAOBNH4"],
                               baseline=[0.9, 0.14], output_names=["NH4", "NO3"])
sc.table()                           # KPI table, one row per scenario (str)
sc.output_named("NH4")               # (n_scenarios,) one output across scenarios
sc.best("NH4", minimize=True)        # scenario name with the lowest NH4
#
# monte_carlo -- propagate uncertain inputs through fn -> output ensemble +
# percentiles. Each input has a distribution: a (low, high) tuple (uniform) or
# {"dist": "uniform"|"normal"|"lognormal", ...} (mean/std in physical space).
# Reuses dgsm's scrambled-Sobol QMC; sampler="sobol" (default) / "lhs" /
# "random"; low-discrepancy unit points are mapped through each input's inverse
# CDF so non-uniform marginals still get a good design. Non-finite outputs (a
# failed/clipped solve) are dropped; the seed makes it reproducible.
mc = aquakin.monte_carlo(fn,
        {"muAOB": {"dist": "normal", "mean": 0.9, "std": 0.15},
         "KAOBNH4": {"dist": "lognormal", "mean": 0.14, "std": 0.05}},
        output_names=["NH4", "NO3"], n_samples=256, sampler="sobol", seed=0)
mc.percentiles((2.5, 50, 97.5))      # (3, m) per-output percentiles
mc.mean(); mc.std(); mc.summary()    # (m,), (m,), human-readable table
mc.output_named("NH4")               # (n_valid,) ensemble of one output
#
# optimize_design -- minimise (or maximise) an objective over BOUNDED design
# variables subject to inequality constraints, using AD gradients (a constrained
# NLP via SciPy SLSQP/trust-constr). The canonical use is "size a design to a
# permit at minimum cost": objective is a cost/energy metric, each Constraint is
# an effluent ceiling. objective/constraint fns share the fn(x)->scalar contract
# and must be JAX-differentiable (gradients taken by autodiff). n_starts does
# quasi-random (Sobol) multistart and returns the best feasible optimum.
opt = aquakin.optimize_design(
        objective=lambda x: x[0],                    # e.g. minimise OCI
        bounds=[(0.5, 2.0)], input_names=["muAOB"],
        constraints=[aquakin.Constraint(fn=eff_nh4, upper=6.5, name="eff_NH4")],
        x0=[1.5], n_starts=1)
opt.x; opt.x_named; opt.objective    # optimal design + objective value
opt.constraint_values; opt.feasible  # {name: fn(x)} at the optimum; permit met?
opt.report()                         # human-readable summary (str)

# Point-estimate fit (SciPy box-constrained least squares)
result = aquakin.fit(reactor, C0, observations, t_obs, free_params, method="adjoint")
result.params
result.params_named

# MAP fit with parameter transforms + Laplace posterior approximation
calib = aquakin.calibrate(
    reactor, C0, observations, t_obs, free_params,
    transforms={"O3_Br_direct.k1": "positive_log", ...},  # or omit to use schema defaults
    loss="nll", sigma=sigma,                              # for proper posterior interpretation
    laplace=True,
    laplace_method="gauss_newton",   # AD Fisher H=JᵀJ (exact, PSD); or "fd" (default)
    optimizer="gauss_newton",        # robust trust-region least-squares; or "lbfgsb" (default)
    ad_mode="forward",               # AD direction for the Jacobian: "forward" builds a
                                     #   forward-capable adjoint INTERNALLY (no diffrax in
                                     #   user code), finite through a stiff solve — pair with
                                     #   optimizer="gauss_newton"; "reverse"/"auto" (default,
                                     #   legacy: forward iff reactor already DirectAdjoint).
                                     #   forward is mutually exclusive with gradient="stable_adjoint".
    check_finite=True,               # default: raise a friendly error (with the remedy) if the
                                     #   start-point gradient is non-finite, vs silent NaNs
    n_starts=24, jitter=0.5, seed=0, # deterministic multistart (escapes local minima); default n_starts=1
    free_ic=["X_S2"],                # fit unmeasured initial pools (per batch) alongside rates
    ic_bounds=(1e-3, 1e4), ic_prior_log_std=0.7,   # bounds + optional weak log-prior for free ICs
)
calib.params_named                   # MAP estimate in physical space
calib.params_named_std               # marginal std devs (delta-method projected)
calib.posterior_cov                  # (d, d) covariance in unconstrained space (rates only when free_ic used)
calib.C0_fitted                      # per-batch fitted initial states (when free_ic used)
calib.ic_named                       # per-batch fitted free pools by species name
result.converged
# Posterior-predictive curve bands: a first-class method that samples the Laplace
# posterior (= `posterior_cov`), propagates each draw through a solve, and returns
# per-timepoint percentiles. The C0 passed in may differ from calibration (e.g. a
# held-out validation batch). The non-identifiable directions are dropped ONCE, at
# calibrate time, by a single eigen-truncated covariance (`calibrate(laplace_eig_keep
# =...)`, default 1e-2) built by `_laplace_covariance`; `posterior_cov`,
# `params_named_std` and `predictive_band` all read it, so the reported marginal
# std devs and the band regularise identically (a well-identified fit keeps every
# direction, so the covariance equals inv(H+ridge)). `predictive_band(eig_keep=...)`
# is deprecated/ignored.
band = calib.predictive_band(reactor, C0, t_eval, n_draw=200, percentiles=(2.5, 97.5))
band.median, band.lo, band.hi        # (n_t, n_species) envelopes -> PredictiveBand

# optimizer="gauss_newton" minimises the residual vector with scipy.least_squares
# (trf), forming the Jacobian by forward-mode AD when the reactor uses
# adjoint=diffrax.DirectAdjoint() (finite at any step, for very stiff networks
# whose reverse-mode adjoint is non-finite), else reverse-mode. It is markedly
# more robust than L-BFGS-B on the multimodal landscapes of stiff network fits.

# Profile-likelihood identifiability analysis (the exact companion to the local
# Laplace covariance). Fix one quantity -- a parameter OR an initial condition --
# at each value on a grid, re-optimise all the OTHER free quantities (each grid
# point is a calibrate() fit, so multistart / Gauss-Newton / free_ic flow
# through), and trace the best attainable objective. The 95% interval is where
# that profile rises by the one-DOF likelihood-ratio threshold (delta=1.92).
prof = aquakin.profile_likelihood(
    reactor, C0, observations, t_obs, free_params,
    grid=grid, profile_param="k_s0_anox_f",   # or profile_ic="X_S0"
    loss="nll", sigma=sigma, n_starts=8,
    warm_start=True,   # continuation sweep keeps consecutive points in one basin
    polish=True,       # re-fit any point a better-fitting neighbour can improve
)
prof.mle                             # grid value at the profile minimum
prof.ci                              # (lo, hi); None on a side => open/unidentified
prof.delta_loss                      # profile relative to its minimum (vs delta)
prof.fits                            # the re-optimised CalibrationResult per grid point
# Unlike Laplace, the profile is exact for non-quadratic / non-identifiable
# parameters: a parameter the data cannot pin gives a flat profile and an open
# (None) interval -- a diagnosis the quadratic approximation cannot give.
#
# COMPILED-OBJECTIVE REUSE ACROSS GRID POINTS. Each grid point is a calibrate()
# fit, and the points of a sweep differ ONLY in the pinned value / warm start --
# the reactor, observations, free set, transforms and loss are identical. So
# calibrate threads its per-call-varying data (p0_full, the per-dataset initial
# states, the ic-prior centre) into the compiled objective + Jacobian as runtime
# ARGUMENTS rather than baking them into the closure, and profile_likelihood
# passes one shared compiled-objective cache (calibrate's private
# `_compiled_cache`) across every point. The stiff objective + Jacobian then
# compile ONCE for the whole sweep instead of per point -- ~4x faster on a
# 9-point ASM1 sweep, growing with the grid size -- and the result is
# bit-identical to recompiling per point (the cache reuses only the compiled
# program). The key carries the structural shape, so a cache accidentally shared
# across differently-shaped fits rebuilds rather than mis-hitting; a plain
# calibrate() call (no `_compiled_cache`) is byte-for-byte unchanged. Verified in
# tests/integration/test_profile.py::test_profile_compiled_cache_matches_uncached.
```

Internal implementation details (`ASTNode` subclasses, `CompileContext`,
Pydantic models, Diffrax solver objects) are not part of the public API and
should not be imported from `aquakin` directly. They are accessible via
submodules for advanced users.

Reactors are **stateless after construction** — `solve()` takes all variable
inputs as arguments. This enables `jax.vmap` over initial conditions or
parameter ensembles.

### Compiled-solve caching

Compiling a stiff solve (JAX trace + lower + XLA) dominates its cost — the run
itself is comparatively free (measured ~1.6 s compile vs ~0.02 s run for an
ASM1 batch solve; ~34 s vs ~4 s for the full BSM2 plant). So the cost of code
that solves repeatedly is *recompilation*, and the integrators cache the
compiled solve to avoid it:

- **Networks** (`load_network`) are cached by name, so repeated
  `load_network("asm1")` returns the **same** object (and skips re-parsing the
  YAML). A `CompiledNetwork` is immutable in use; `clear_network_cache()` resets
  the cache. The stable identity is what lets the solver caches key on the
  network across calls.
- **Reactor solves** are cached **across instances** in a module-level cache
  (`integrate/_common.py`) keyed by `(network identity, solver settings, call
  signature)`. Two *fresh* reactors for the same network + settings + signature
  reuse one compiled solve — so building many short-lived reactors (ensembles,
  library code that constructs reactors internally) no longer recompiles each
  time. (The `_build_jitted_solve` closure captures only the network and the
  scalar settings, so the key is complete; argument shapes/dtypes are handled by
  JAX's own per-function cache.) `BatchReactor`, `PlugFlowReactor` and
  `ParticleTrackReactor` all route through this cache: the batch key carries the
  `(t0, t1, t_eval shape)` call signature, the PFR key the fixed geometry
  (`velocity`/`length`/`n_points`/`n_locations`), and the particle key only the
  `(network, settings)` — the particle reactor passes the track's sample times
  and condition fields as **runtime arguments** (not baked into the closure), so
  an `integrate_ensemble` over same-shape tracks compiles **once** and JAX's
  per-shape cache covers tracks of differing length. (`BiofilmReactor` keeps a
  per-instance cache — its multi-compartment geometry makes a complete
  cross-instance key less clear-cut.)
- **Plant solves** are cached **per instance** (`Plant._jit_cache`), keyed by
  signature + settings. The plant RHS closes over the (static) unit graph, so
  the first solve compiles and every later solve of that plant reuses it —
  e.g. a parameter sweep / Monte Carlo that builds the plant once and solves
  many times, or a warm-started steady-state-then-dynamic run. (Cross-*instance*
  plant caching is deliberately **not** done: a fresh plant's compiled RHS
  depends on the entire unit-config + connection graph, and a structural key
  complete enough to never false-hit would be fragile — a miss there would
  silently return a solve compiled for a *different* plant. Per-instance keying
  cannot false-hit.) The event path (`run_to_steady_state`) is not cached
  (run-once). The `gradient="stable_adjoint"` path **is** cached for repeat
  *forward* solves (a parameter sweep), keyed the same way but tagged
  `"stable_adjoint"` so it never collides with the forward path, with `t_eval`
  baked into the closure (the discrete adjoint marks it non-differentiable, so it
  cannot be a traced runtime argument) and its values folded into the key. The
  cache is used **only when the inputs are concrete**: under a trace — a gradient
  through the solve, or an enclosing `jax.jit` — the adjoint's `custom_vjp` is
  traced directly into the outer computation rather than routed through an inner
  `jax.jit`, which does not compose with an outer reverse-mode pass. That direct
  path is the one a `gradient="stable_adjoint"` calibration gradient takes, so a
  jitted calibration loss amortizes the (large) plant compile across optimizer
  iterations through the *outer* jit. Jitting that loss is possible because
  `_coerce_atol` returns the 0-d `atol` array unchanged under tracing instead of
  forcing it to a Python `float` (a `float()` on a tracer raises a
  concretization error).

**Correctness guarantees.** A cache key never omits anything that changes the
compiled result, so a hit always returns a solver compiled for the exact same
computation. The key materialises `atol` values, which is impossible **under
tracing** (a calibration loss differentiating through `solve`); in that case the
key is `None` and the cache is bypassed (the solve is traced into the outer
computation, which JAX compiles as a whole, so caching gives nothing there
anyway). Both caches assume the network / plant is not structurally mutated
after the first solve — the same assumption reactors already make about their
fixed network and conditions.

**What this is and isn't.** It removes *duplicate* compiles; it does not remove
the first compile of each distinct `(network/plant, settings, signature)`, and
the JAX **persistent** (cross-process / cross-run) compilation cache does *not*
help these Diffrax solves (verified: no reuse across processes for either an
ASM1 reactor or the BSM2 plant — it caches only the XLA step, not tracing, and
Diffrax programs miss it across processes). So this speeds repeated solving
within a process; it does not by itself shrink a cold test suite where each test
compiles a distinct configuration once.

---

## Testing Architecture

### Three Layers

**Unit** — test individual components in isolation, fast, run on every change.
**Integration** — test full YAML → solution pipeline against analytical solutions.
**Validation** — test scientific correctness against published experimental data,
marked `@pytest.mark.validation`, run separately.

### pytest Configuration

```toml
[tool.pytest.ini_options]
markers = [
    "validation: scientific validation tests against published data (slow)",
    "slow: multi-minute stiff/plant integration tests; excluded from the fast PR gate, run in full on merge to main",
]
testpaths = ["tests"]
```

```bash
pytest -m "not validation and not slow"   # fast gate (the PR merge gate)
pytest -m slow                             # the multi-minute stiff/plant solves
pytest -m validation                       # validation suite (published data)
pytest                                     # everything
```

### Canonical Integration Test

First-order decay `A → B` with rate `k * [A]` has the analytical solution
`[A](t) = [A]₀ * exp(-k*t)`. This is the primary integration test and must
always pass. It lives in `tests/fixtures/simple_network.yaml` and
`tests/integration/test_batch_simple.py`.

### AD Correctness Test

Every integration test suite must include an explicit test that `jax.grad`
flows through `reactor.solve()` without error and without producing NaNs.

### x64 Test

```python
def test_64bit_precision_enabled():
    import jax
    assert jax.config.x64_enabled
```

---

## Comment Convention

Code comments and docstrings must stand on their own for a reader of *this*
repository. Do **not** reference external artifacts that are not in the repo —
private source files (e.g. a `.c` we ported from), internal model names, paper
filenames, or "the reference" / "the original". Such notes are meaningless to
anyone but the original authors and rot immediately. Explain the code on its
own terms (what the math/logic is and why), not by pointing at something the
reader cannot see. Genuine scientific provenance belongs in a network YAML's
`references:` block as a proper literature citation, not scattered through code
comments.

## Docstring Convention

All functions, classes, and methods use **NumPy docstring format**:

```python
def solve(self, C0, params, t_span, t_eval=None):
    """
    Integrate the reaction network over a time span.

    Parameters
    ----------
    C0 : jnp.ndarray
        Initial concentration vector, shape (n_species,).
    params : jnp.ndarray
        Rate constant vector, shape (n_params,). Use
        ``network.default_parameters()`` as a starting point.
    t_span : tuple of float
        (t_start, t_end) integration interval in seconds.
    t_eval : jnp.ndarray, optional
        Time points at which to record solution. If None, solver
        chooses output times.

    Returns
    -------
    BatchSolution
        Solution object with attributes ``t`` (n_t,) and ``C``
        (n_t, n_species).

    Raises
    ------
    ValueError
        If ``C0`` length does not match number of declared species.

    Examples
    --------
    >>> reactor = aquakin.BatchReactor(network, conditions)
    >>> sol = reactor.solve(network.default_concentrations(),
    ...                     network.default_parameters(),
    ...                     t_span=(0.0, 600.0))
    >>> sol.C_named("BrO3-")
    """
```

---

## Plant-Wide Simulation

`aquakin.plant` composes kinetic reactors with non-reactive unit ops into
a full plant flowsheet. The plant assembles each unit's internal state
into one flat vector and integrates the whole thing under a monolithic
Diffrax solve — so `jax.grad` flows end-to-end across the plant, and
`aquakin.calibrate()` works on plant-level parameter vectors.

**By-name plant parameters.** A `Plant` concatenates its networks' parameter
vectors into one flat `default_parameters()`. `Plant.parameter_values(overrides)`
gives that flat vector the same friendly by-name API as
`CompiledNetwork.parameter_values`, keyed by `"<network>.<param>"` (the network
name plus the network's own namespaced parameter name) — e.g.
`plant.parameter_values({"asm1.muH": 4.0, "adm1.k_hyd_ch": 10.0})` to bump one
rate in a multi-network plant (BSM2's ASM1 water line + ADM1 digester) without
hand-computing the block offset. `parameter_names()` lists the valid keys;
`parameter_index(name)` returns the flat index (the companion for `jax.grad`
w.r.t. one parameter, which can't go through `parameter_values` — that
materialises concrete values). All three reuse the existing
`network_param_blocks` layout. Unknown names raise a `KeyError` with a
close-match hint.

Key types:

- `Stream(Q, C, network)` — the bulk-flow + concentration record passed
  between units.
- `Unit` Protocol — every unit declares `state_size`, `input_ports`,
  `output_ports`, and implements `initial_state()`, `compute_outputs()`,
  `rhs(t, state, inputs, params, signals)`, and `flow_outputs(input_flows,
  params, ctx)`. **Every unit has the *same fixed signature* for each method**
  — the plant never branches its call on a per-unit capability flag. The
  control-signal bus is threaded into every `rhs` as `signals` (an uncontrolled
  unit ignores it), and `flow_outputs` always receives a `FlowContext` carrying
  the unit's own state and the time (a fixed-split unit ignores it). The one
  optional, duck-typed hook is `signal_outputs(...)`, implemented only by units
  that *produce* control signals (e.g. `PIController`). A **stateless** unit
  (`state_size == 0`: mixers, splitters, ideal separators) inherits the
  `StatelessUnit` mixin (`plant/units.py`), which supplies the three trivial
  state members (`state_size → 0`, empty `initial_state`, no-op `rhs`), so it
  only writes `compute_outputs` / `flow_outputs`. It is a plain mixin, not part
  of the Protocol, so it composes with the `@dataclass` units.
- `StateTranslator` Protocol — converts streams between networks.
  `IdentityTranslator` covers single-network plants (BSM1).
- `Plant` — assembles units and connections, drives the monolithic
  integration. Recycles are resolved **exactly and gain-independently** per RHS,
  in two decoupled steps that both use the same affine-probe + linear-solve trick
  (no iterate-to-tolerance — the RHS is jitted/differentiated):
  - **Flows** — `_resolve_flows` probes the (affine) recycle-flow map and solves
    `(I − A)x = b` for the back-edge flows.
  - **Concentrations** — `_resolve_recycle_concentrations` does the same for the
    recycle-edge *concentrations*. One forward output sweep at fixed flows is an
    affine map `c → M·c + d` (mixers/splitters/clarifiers are linear in
    concentration; stateful units output their state, a constant), so it probes
    `M`/`d` (one pass at `c=0`, one per recycle edge set to a unit concentration)
    and solves `(I − M)c = d`. The map is **species-decoupled** (the only
    species-coupling unit, an ASM↔ADM translator, is fed by a digester *state* so
    never enters the cyclic map), so one probe per edge yields its whole column
    across all species — `n_recycle_edges + 1` cheap passes, like the flow probe.
    Edges of **different networks** don't couple (the translator that would couple
    them is broken by the digester state), so the solve is grouped by network;
    **temperature**, when an influent carries it, is one more decoupled scalar
    channel. Exact and gain-independent: a recycle loop whose bare Gauss-Seidel
    would need thousands of passes (a clarifier in a high-capture stateless loop)
    is solved in one linear solve. Validated as a fixed point on BSM1, the
    multi-network BSM2, and a temperature-carrying loop (residual ~1e-12).
  The exact concentration solve **seeds** the `recycle_passes` Gauss-Seidel
  mop-up (default 3), which therefore does no work for any linear topology (every
  shipped plant) and only refines a genuinely *non-affine* in-cycle unit (a
  translator inside a pure-stateless loop — not constructible from the shipped
  units). A one-time `_check_recycle_convergence` diagnostic (concrete-only,
  skipped under tracing, skipped without recycle edges) warns if even that has
  not converged — the backstop for the non-affine case.
  - **Adaptive AD-safe recycle convergence (`recycle_tol`).** The fixed
    `recycle_passes` count is a *diagnostic-backed default*, not a general
    convergence guarantee: the mop-up converges geometrically in
    `log(tol)/log(rho)` passes, where `rho` is the spectral radius of the
    nonlinear flow↔concentration coupling Jacobian (the reject loop). For BSM2 the
    only iterating stream is the front mixer (influent + reject recycle) and
    `rho ≈ 0.0066`, so 3 passes leaves ~1e-6 residual — but `rho` is
    topology-dependent and **not bounded below 1**: a recycle-heavy plant with a
    strong concentration-dependent in-loop flow (e.g. a high-capture
    thickener/dewatering `%TSS` underflow on a tight reject loop) can have `rho`
    near 1, where the fixed 3 passes leaves residual `rho³` — a **silently wrong
    steady state** (measured on a synthetic map: 13% error at `rho=0.5`, 73% at
    `rho=0.9`, 97% at `rho=0.99`). `Plant(..., recycle_tol=...)` (**on by
    default, `1e-8`**) replaces the fixed mop-up with an **adaptive solve to that
    relative tolerance**, mirroring the charge-balance pH solver
    ([`core/ph_solver.py`](aquakin/core/ph_solver.py)): the recycle back-edge
    **streams** — flow `Q`, concentration `C`, and (when carried) temperature `T`
    — are the fixed point `x = G(x)` of one forward output sweep
    (`Plant._recycle_context`'s `forward_full`, which lets `Q` vary so the true
    `Q↔C` reject-loop coupling is captured; iterating `C` alone solves the wrong
    problem), warm-started from the exact affine seed and iterated by a
    `jax.lax.while_loop` that **stops once the actual residual clears** (capped at
    `recycle_max_passes`, default 100), wrapped in `jax.lax.custom_root` so the
    sensitivity is the exact **implicit-function-theorem tangent** — a small dense
    solve of the linearised recycle-edge operator (the recycle edges are few ×
    ~tens of channels), the vector generalisation of the pH solver's scalar
    `y / g(1)`. AD (forward and reverse) is therefore **O(1) in the iteration
    count** rather than differentiating through every sweep
    (`Plant._adaptive_recycle_refine`). It converges for any `rho < 1`, stops
    early on a low-gain plant, and is **on by default** at `1e-8` (well below the
    typical solver `rtol`, a strict improvement on the old fixed-3-pass ~1e-6 at
    ~neutral cost — ~3 iterations from the affine seed for BSM); `recycle_tol=None`
    falls back to the fixed `recycle_passes` mop-up (the bit-identical historic
    behaviour). **The validated BSM steady states are reproduced** with the
    default on (the published BSM2 / ADM1 / Takács validations pass unchanged) —
    the adaptive path converges to the *same* recycle fixed point the fixed-pass
    mop-up approximates, only tighter. **Verified** (`tests/integration/
    test_adaptive_recycle.py`): on a synthetic tunable-`rho` map the adaptive
    solve reaches tolerance for `rho` up to 0.99 where the fixed 3-pass leaves
    13–97%, and its IFT gradient matches central finite differences to ~2e-10; on
    the real BSM2 reject loop the adaptive forward fixed point matches a 14-pass
    deep sweep to ~4e-15, its IFT tangent matches the deep-sweep gradient to
    ~6e-19, and a short BSM2 solve matches the fixed-pass trajectory to ~3e-8
    (within the solver tolerance — the adaptive path is the more-converged of the
    two). `recycle_tol` is read inside `_resolve_recycle_concentrations`, so it
    reaches every solve path automatically (no per-path threading); the cached
    affine `recycle_map` still supplies the warm-start seed. Because the default
    routes every plant solve through `jax.lax.custom_root`, the
    `gradient="stable_adjoint"` discrete adjoint composes with the recycle IFT
    tangent (verified exact: the cached/probed `dM/dθ` gradient agrees to ~1e-13,
    and the cross-interface gradient matches finite differences to the FD floor);
    with the adaptive default `M` is only the warm-start seed (the fixed point is
    M-independent), so the `#366` cached/probed-map gradient distinction now agrees
    to float rounding rather than bit-for-bit.
  - **Cached recycle map (per-RHS speedup).** The concentration map `M` is fixed
    by the recycle flows + topology, so for a **fixed-pump** plant (every BSM
    plant — the recycle pumps are constant) it is **invariant to the state and
    time**; only `d = forward(0)` varies. The `n_recycle_edges` per-species
    `M`-probe sweeps are therefore recomputing a constant on every one of the
    ~17 RHS calls per implicit step. `_compute_recycle_map` precomputes `M`
    **once per solve** (from the runtime `params`, so the gradient still flows
    and a parameter sweep stays correct) and `_build_jitted_solve` threads it into
    every RHS as `recycle_map=`; the per-RHS recycle resolution then computes only
    `d` (one sweep) + the cached `(I−M)` solve. Profiling located the per-RHS cost
    as ~88% the recycle resolution and the per-step cost as ~RHS-evaluation-bound,
    so this is a real dynamic-solve win. The **temperature** map `MT` is cached
    too **when it is state-invariant**, which depends on the temperature model:
    in **heat-balance** mode the reactor temperature is a *state* (it breaks the
    loop coupling at reactors, exactly as concentration does) so `MT` is constant
    and cached → full win; in **algebraic** mode temperature *passes through*
    reactors (no thermal mass) so `MT` rides on the concentration-dependent
    recycle flows and is **not** constant → it is re-probed every RHS (a cheap
    scalar T-only sweep, its per-species part CSE-shared with `d`), while `M`
    stays cached. Net measured speedup on the dynamic BSM2 (algebraic):
    ~recycle-resolution 2.4× → ~1.18× wall; larger (full ~5.6× recycle) for
    heat-balance / no-temperature plants. **Exactness:** the cached and probed
    paths produce a **bit-identical RHS** (the cached `M` *is* the probed `M`);
    the dynamic trajectory shows only ~1e-3 floating-point operation-order drift
    over multi-day runs (the cached `M` is formed once outside the integration
    loop vs per-call inside), within the solver tolerance, and the validated
    steady states are preserved. A one-time concrete guard
    (`_check_recycle_map_constant`, set per instance) compares each map at two
    states and **falls back to per-RHS probing** for any topology whose `M` is
    genuinely state-coupled — so the optimization is safe for arbitrary plants.
    The cached map is built once from `params` by `_maybe_recycle_map` (the
    shared helper) and reused by **four** paths: the forward `jax_adjoint` solve,
    the located-event segmented solve (`events=`, reused across every segment),
    the single-instant `outputs_at`, and the whole-trajectory stream
    reconstruction (`_cached_streams` / `plant.stream`, the `evaluate_bsm*`
    evaluation path — measured ~1.16×, bit-identical). The events path needed the
    one-time constancy check hoisted *above* the events branch in `solve` so an
    events-only plant (SBR / control study) still gets the cached map (and the
    affinity/convergence diagnostics). The reconstruction win confirms its
    per-time cost was also recycle-resolution-dominated. **`gradient=
    "stable_adjoint"` also uses the cached map (#366)**, via a *primal/param RHS
    split* of the discrete-adjoint kernel. `esdirk_adjoint_solve` forms `∂f/∂θ` by
    differentiating the per-call `rhs(t, y, params)`, so a precomputed `M` closed
    over as a constant would be invisible to that vjp and a gradient w.r.t. a
    **flow-setpoint param** (RAS/`Qw`/`f_PS`, the only params `M` depends on) would
    silently drop its `∂M/∂θ` term. The kernel therefore takes an optional
    `primal_rhs=`: the forward solve and the backward **`∂f/∂y`** stage Jacobians
    use the cached-`M` `primal_rhs` (the recycle probe hoisted out of the hot
    loop), while the **`∂f/∂θ`** vjp keeps the map-recomputing `rhs` — so `∂M/∂θ`
    is captured exactly. Because the discrete adjoint draws its *entire* parameter
    gradient from that vjp and uses the stages/Jacobians only to propagate the
    *state* cotangent, and the cached `M` *is* the probed `M`, the result is
    **bit-identical** to probing every call, just faster (the gradient w.r.t. a
    kinetic *and* a flow-setpoint param both match the per-call-probe gradient
    bit-for-bit). The cached map must be `stop_gradient`'d before being closed over
    (it is a params-derived value inside the `custom_vjp`; its parameter dependence
    is the vjp's job). Both the cached jitted forward and the under-trace
    calibration-gradient path go through the shared `Plant._esdirk_stable_adjoint`.
    Falls back to per-call probing when `M` is not state-invariant
    (`_recycle_map_constant` not True). **Measured (clean serial min-of-8 timing):
    a modest reverse-gradient win where the recycle probe is non-trivial — BSM2
    `value+grad` ~1.15× (14.7→12.8 s, the ASM↔ADM-interface probe hoisted out of
    the backward) — and neutral on BSM1** (whose probe is cheap, so its gradient is
    unchanged). The one-time map build makes the *forward* marginally slower (BSM2
    ~0.95→1.09 s), but `stable_adjoint` exists for gradients, where the net is
    positive. *(Historical note: when this was measured the backward's dominant
    cost was the per-step `n≈167` **dense stage Jacobian builds** — ~82% of the
    backward — because it recomputed every ESDIRK stage by Newton; the saved-stage
    backward later removed that recompute, dropping the builds to ~7/step (~24%),
    so the cached-map win and the colored-build win are both proportionally
    different now — see the stage-saving and colored-backward bullets for the
    re-measured numbers.)*
    Covered by
    `tests/integration/test_recycle_cached_map.py` and the bit-identical
    flow-setpoint `∂M/∂θ` guard
    `test_plant_stable_adjoint.py::test_stable_adjoint_flow_setpoint_gradient_preserves_dM_dtheta`.
  - **Cached recycle *flow* map (the analogue, #397).** The recycle *flow* solve
    `_resolve_flows` is the same `(I−A)x=b` affine structure as the concentration
    solve: the `n×n` back-edge flow-response `A` is fixed by the recycle flows +
    topology (so constant for a fixed-pump plant), while only `b` (the
    influent-driven constant) varies per RHS. The per-RHS flow probe therefore
    re-derives a constant `A` (the `n` per-back-edge `one_pass(eye[i])` column
    passes) on every RHS. `_compute_flow_map` precomputes `A` **once per solve**
    (from `params`, gradient-preserving — `A` depends on the flow-setpoint block,
    so it is coerced in and `stop_gradient`'d on the `stable_adjoint` primal like
    `M`), threaded into every RHS as `flow_map=` by `_maybe_flow_map`; the flow
    resolution then computes only `b` (one pass) + the cached `(I−A)` solve,
    skipping the `n` column probes. State-invariance is detected once by
    `_check_flow_map_constant` (compares `A` at two states; falls back to per-RHS
    probing for a state-coupled split like a level-gated storage bypass), wired
    into the same four paths as `M` (`make_rhs` forward, events, `outputs_at`,
    `_cached_streams`) plus the `forward_fast`, steady-state-PTC, and
    `stable_adjoint` builders. **Correctness:** the cached-`A` RHS *and Jacobian*
    are **bit-identical** to the probe (`A` is state-invariant, so `dA/dy=0` in
    both), and every steady state (constant influent → convergent dynamics) is
    bit-identical — so all validations are preserved. On a **sensitive
    time-varying** run the cached and probe solves can separate (measured ~9.5e-3
    rel over a dynamic BSM2), because the probe recomputes
    `A = one_pass(eye)−one_pass(0)` each step and the varying influent perturbs
    that cancellation's rounding by ~1e-16 each step, while the cache holds `A` at
    its exact constant value — the cached path is the **cleaner** of the two valid
    solves, and the divergence cannot amplify at a fixed point (hence the
    bit-identical steady state). **Measured 1.076× (7.6%)** on the 60-day dynamic
    BSM2 forward solve. Covered by the flow-map tests in
    `tests/integration/test_recycle_cached_map.py` (constancy detection,
    bit-identical RHS + Jacobian, bit-identical steady state, the no-recycle case,
    and a gradient through the cached path).
  - **Wiring API.** `plant.connect(source, dest)` takes two `"unit.port"`
    endpoint strings, read as `source -> dest`. The port may be omitted
    (bare `"unit"`) when the unit has exactly one port for that role — a
    single output (source) or single input (dest) — so only multi-port
    units (mixers/splitters/clarifiers) name a port.
    External influents are wired through
    `plant.add_influent(name, series, to="unit.port")` — they are *not*
    valid `connect` sources (a clear error redirects you). `connect`
    resolves the default `IdentityTranslator` when the two ends share a
    network and requires an explicit `translator=` across networks (e.g.
    the BSM2 ASM1↔ADM1 digester edges). The endpoint parsing lives in
    `Plant._parse_endpoint`.
  - **Arbitrary add order; topological sort.** Units may be `add_unit`-ed in
    **any order**: `Plant._finalize_topology` (run from `_build_state_layout`,
    so at every solve) topologically sorts the feed-forward connection graph
    into the RHS evaluation order `_unit_order`, and the **recycles are the graph
    back-edges, detected automatically** — you no longer add the downstream unit
    before its upstream consumer or mark recycles by ordering. The sort is Kahn's
    algorithm with an insertion-order tie-break (deterministic, so the
    state/parameter layouts are stable): a connection carrying an explicit
    `initial_value` is a declared recycle and cut first (its seed rides the cut
    edge); any remaining cycle is broken by cutting the earliest-added remaining
    unit's still-active incoming edges, which become auto-detected recycles
    (zero-flow seeded via `_recycle_seeds`). For a plant already added in a valid
    order it **reproduces that order and recycle set exactly** (BSM1/BSM2
    unchanged); any valid feedback-arc cut gives the same converged solve, so the
    result is add-order-independent. `add_unit` records the raw add order in
    `_insertion_order` (the parameter-block order and the tie-break read it);
    `_unit_order` is the computed eval order. Recycles are then resolved by
    iterating the per-RHS stream computation 3 times (sufficient for typical BSM
    topologies). `initial_value=` on `connect` overrides a recycle's zero-flow
    seed with a non-zero warm start (e.g. the BSM2 temperature-carrying seed).
  - **Pre-solve wiring check.** `plant.check()` → `PlantCheck` reports **unfed
    input ports** (`.unfed_ports`, an error — the RHS sweep has no source for
    them) and **unconsumed outputs** (`.dangling_outputs`, info — a terminal
    stream like the final effluent / wasted sludge / disposal cake / biogas
    legitimately leaves the plant), plus the detected `.recycles`; `.ok` is true
    when nothing is unfed and `.summary()` prints it. `check(raise_on_error=True)`
    raises on an unfed port. Exported as `aquakin.PlantCheck`.
  - **Operating temperature.** `plant.set_temperature(celsius)` sets the static
    `T` condition of every temperature-bearing reactor in one call (°C → K),
    leaving a heated fixed-`T` unit like the digester untouched; clears the
    compiled-solve cache and returns `self`. See the seasonal-temperature notes
    below.
  - **Warm-starting.** `plant.initial_state(overrides={"tank1": vec, ...})`
    builds the flat initial-state vector with selected units' states replaced
    by name (each vector must match the unit's `state_size`) — the supported
    way to seed a plant (e.g. a healthy activated-sludge biomass before a slow
    digester settle) instead of reaching into the private `_state_layout`. Pass
    the result as `solve(y0=...)`. For the BSM plants the reference seed is
    shipped — **`bsm2_warm_start(plant)`** / **`bsm1_warm_start(plant)`**
    (`aquakin.plant.bsm`) return a ready flat `y0` with the five AS reactors
    seeded from the reference reactor composition (the dict constants
    `BSM2_WARM_REACTOR_COMPOSITION` / `BSM1_WARM_REACTOR_COMPOSITION`) and every
    other unit at its default. **BSM2 should always be warm-started**: the
    digester's ~19-day retention makes a cold start slow and stiff (the
    near-empty AS basin filling against the recycle loops can crawl or hit the
    step ceiling), and the warm seed removes that transient so only the digester
    has to settle. The reactor set and water-line network are auto-detected from
    the plant, so a single `bsm2_warm_start(plant)` replaces the
    seed-composition dict + tank list + `initial_state(overrides=…)` boilerplate
    the BSM2 scripts used to copy-paste. (The BSM2 composition is the validated
    reference reactor state; the BSM1 one is ~aquakin's BSM1 steady state. Both
    are *seeds* — the solve relaxes them — so the values affect settling speed,
    not the steady state.)
  - **Introspection — discover names instead of reading the builder source.**
    `plant.list_units()` lists the unit names (in add order); `plant.list_ports()`
    lists every `"unit.port"` **output** endpoint — the exact strings
    `plant.stream(sol, …)` accepts (pass `unit=` to scope, `role="input"` for the
    `connect`-destination endpoints); `plant.list_species(unit)` lists a
    concentration-vector unit's species (the valid `C_named` / `to_dataframe`
    columns). All three work **before** solving (plant structure) and raise a
    `KeyError` with a `difflib` "did you mean?" hint for an unknown name.
    `list_species` / `C_named` are restricted to units whose *state is a
    concentration vector* (`state_size == network.n_species`: the CSTRs, the
    primary clarifier holding tank, the digester) via `Plant._is_concentration_unit`
    — a stateless mixer/splitter/ideal-clarifier or the **layered Takács settler**
    (which carries a network but a non-species state) is rejected with a clear
    "read it as a stream with `plant.stream(...)`" message rather than an
    `IndexError`. `PlantSolution.available_streams()` is a convenience alias for
    `plant.list_ports()`, and `solution.C_named(unit, species)` now gives the same
    hinted errors (unknown unit, unknown species, non-concentration unit).
    `plant.activated_sludge_reactors(require_volume=True)` lists the AS reactor
    units (the CSTR/MBR `aeration`-carrying units, digester excluded; in plant
    order) — the single source of truth behind the warm-start / design-sizing /
    evaluation reactor heuristics (`require_volume=False` keeps every mechanically
    mixed reactor, e.g. for the mixing-energy term).
  - **Reading state back by unit.** `plant.states_by_unit(vec)` splits any flat
    plant vector into a `{unit_name: sub-vector}` map — the exact inverse of
    `initial_state(overrides=...)`. It works on a `y0`, a `PlantSolution.final_state`
    (the last save row, shape `(total_state_size,)`, so no opaque `[-1]` on the
    2-D `state` trajectory), or a `derivative` result. For a *trajectory* of one
    unit, `PlantSolution.unit_state(name)` returns `(n_t, unit.state_size)`.
  - **Evaluating the RHS once.** `plant.derivative(state, params=None, *, t=0.0)`
    is the public single evaluation of the assembled flowsheet RHS (`dstate/dt`,
    recycles resolved) — for inspecting the dynamics without a full solve. Same
    layout as `state`; split it with `states_by_unit`. (Wraps the private `_rhs`,
    building the layouts internally.)
  - **Effluent reconstruction (streams are recomputed, not stored).** The plant
    integrates unit *states*, not the inter-unit streams, so a stream such as the
    secondary-clarifier effluent is **recomputed on demand** from the saved
    states — it is *not* in the solution. `plant.stream(solution,
    "clarifier.overflow")` (or the convenience `solution.stream("effluent")`,
    plant carried on the solution) returns a `StreamSeries` (`t`, `Q`, `C` shape
    `(n_t, n_species)`, `network`, with a `C_named(species)` accessor) — feed it
    straight to `effluent_averages`. **The whole output sweep (every `(unit,
    port)`) is reconstructed in one `jax.vmap` pass over the saved times and
    cached on the solution** (`Plant._cached_streams`, keyed by the parameter
    vector via `_concrete_teval_key`; skipped under tracing), so a sequence of
    `stream` calls for different ports — or `evaluate_bsm*` reading ~8 streams —
    costs one reconstruction, not one per stream. The reconstruction is
    **vectorised**: each saved time's `_resolve_streams` sweep (a recycle-flow +
    concentration solve) is batched by `vmap` into a single XLA program rather
    than a Python loop of per-step sweeps — turning a long dynamic run's
    evaluation from minutes into seconds (a 609-day hourly evaluation drops from
    ~20 min to a few seconds). `evaluate_bsm2`'s digester-feed-temperature and
    closed-loop kLa histories are vmapped the same way. `plant.outputs_at(t, state, params=None)`
    is the single-instant primitive (returns `{(unit, port): Stream}`,
    uncached); both reuse the same `_resolve_streams` helper the RHS uses, so the
    reconstruction matches the integrated wiring exactly (including resolved
    recycle flows).
  - **Semantic stream shortcuts.** `plant.stream(sol, …)` also accepts an
    engineering **name** instead of a `"unit.port"` — the builders register a
    `named_streams` map (`plant.register_stream(name, endpoint)`,
    `plant.list_streams()`) so `plant.stream(sol, "effluent")` reads the right
    port without the user knowing it is `"tank5_split.internal_recycle"`. BSM1
    registers `effluent`/`ras`/`wastage`/`internal_recycle`; BSM2 adds
    `primary_effluent`/`primary_sludge`/`thickener_overflow`/`reject`/
    `dewatering_reject`/`disposal_sludge`, with `effluent` tracking the
    option-dependent `effluent_endpoint`. A misspelled name gives a hinted error
    listing `list_streams()`. `plant.effluent_stream(sol)` is the first-class
    shortcut for the most-read one (reads `effluent_endpoint`). The digester
    **biogas** is a *derived* output (computed from the ADM1 headspace state, not
    a material port), so it has its own accessor: `plant.digester_gas(sol)` →
    `DigesterGas` (`t`, `Q` m³/d, `p_ch4`/`p_co2`/`p_h2` bar, `ch4` kg/d, and
    `.methane_production()` time-averaged kg CH₄/d), reusing the OCI biogas
    formula (`evaluate_bsm2`'s `_methane_production` now delegates to it). Raises
    if the plant has no ADM1 digester.
  - **Results-level mass-balance closure — `plant.mass_balance(sol, …)` (#150).**
    The first thing an engineer does with a result: *does what went in equal what
    came out + what left as gas + what accumulated?* Returns a `MassBalance`
    (`aquakin.plant.balance`, exported as `aquakin.MassBalance` /
    `aquakin.ComponentBalance` / `aquakin.mass_balance`) with, per component (COD
    / N / P), the **inflow** (influents), **outflow** (terminal/dangling material
    streams — effluent, wasted sludge, disposal cake), **gas** (O₂ transferred in
    by aeration, the digester biogas, denitrification N₂ — computed from the
    aeration term and a reaction-production integral over the reactive units, with
    the digester deliberately excluded from the N gas term since it has no N gas
    phase) and **accumulation** (ΔInventory across every unit — reactor / clarifier
    / digester liquid+headspace at `V_liq`/`V_gas` / storage / Takács blanket).
    A unit holding a single well-mixed liquid volume (`StorageTank` / `MBRUnit` /
    `SBRUnit`, whose states are `[C…, scalar(s)]`) declares that volume through an
    explicit **`liquid_volume(state)`** contract, so its inventory is `V·C` — the
    `_unit_inventory` dispatch reads that method instead of the former fragile
    `hasattr`/state-size guessing (whose MBR-before-storage ordering existed only
    because both states were `[C.., scalar]`); a future such unit just implements
    the contract.
    `imbalance = in − out − gas − accumulation` is the closure; `mb["N"]`,
    `mb.closed(rtol)`, `mb.summary()`, `mb[q].relative_imbalance`. Everything is on
    one canonical g basis (g COD / g N / g P), so the ASM water line (g/m³) and the
    ADM digester (kg/m³, kmol/m³) sum via `aquakin.composition_table` /
    `aquakin.canonical_content` (the shipped per-species COD/N/P content tables;
    `composition_table(net, electron_acceptor_cod=False)` = lab COD, the default
    `True` = the electron-equivalent convention `check_conservation` wants;
    `params=` reads a calibrated/BSM-specific composition such as `i_XB`). Closes
    BSM1 to ~1e-7 and BSM2 (two networks, biogas, recycles) to COD ~0.08% / N
    ~0.03% at steady state; the gas integrals are exact at steady state and
    otherwise accurate to the `t_eval` sampling. **This is the tool that found the
    ADM1 nitrogen transcription error** (see the `adm1` network note).

Shipped units: `CSTRUnit` (kinetics + aeration), `IFASUnit` / `MBBRUnit`
(an IFAS/MBBR tank: a CSTR bulk coupled to a depth-resolved attached biofilm —
see below), `MBRUnit` (membrane bioreactor: a high-MLSS aerated reactor whose
membrane retains the solids into a near-solids-free permeate, with fouling/TMP —
see *Membrane bioreactor* below), `MixerUnit`,
`SplitterUnit`, `IdealClarifier` (fast, stateless separator),
`PrimaryClarifier` (BSM2 Otterpohl–Freund: a well-mixed holding tank split by
an HRT-dependent particulate-removal efficiency, fixed underflow `f_PS·Q`),
`IdealThickener` (BSM2 thickener / dewatering — a stateless ideal `%TSS`
separator, concentration-dependent underflow flow), `ADM1DigesterUnit`
(continuously-fed ADM1 CSTR with gas headspace, dilution masked to the liquid
states), `DosingUnit` (chemical dosing: injects a `Reagent` — a fixed
composition, e.g. metal salt / acid-base / external carbon — into a stream at a
fixed or feedback-controlled flow; see *Chemical dosing* below), `UVUnit` /
`ChlorineContactUnit` (disinfection: UV dose-response and chlorine CT /
log-removal — see *Disinfection* below), `SBRUnit`
(sequencing batch reactor: one tank cycling fill/react/settle/decant/idle with
variable volume and a pluggable settling model — see *Sequencing batch reactor*
below), and `TakacsClarifier` (10-layer 1-D Takács 1991 model). Its settling physics
are correct and verified in isolation at BSM1 solids loading: the
clarification-zone flux limiting (above the feed, the downward flux is
limited by the layer below only when that layer exceeds `X_threshold`) and
the per-species flux apportioning (each species settles at the bulk
velocity, `flux_tss · X_k/TSS`, conserving total settleable solids) produce
a monotone sludge blanket, a strongly clarified effluent, a thickened
underflow, and tight solids mass balance (verified to machine precision
against an independent port of the reference BSM1 settler derivative in
`tests/validation/test_takacs_vs_bsm1_reference.py`). `build_bsm1(use_takacs=
True)` selects it in the full plant (both clarifiers expose the same ports),
and `Plant.solve` takes `max_steps`. By default the **soluble** species are not
held in the settler — they pass straight through (overflow = underflow = feed,
no holdup), the common simplification. The opt-in **`soluble_holdup=True`** makes
each soluble a per-layer well-mixed state advected by the bulk flow (convection
only, no settling), so the clarifier's liquid volume (~`area·height`) damps the
soluble effluent signal — the BSM2 `settler1dv5` behaviour, which carries
`SNH_1..SNH_10` etc. per layer. The soluble holdup is a tail block of shape
`(n_layers, n_soluble)` appended to the state (so the particulate layout /
`state_size` are unchanged when off), orthogonal to `composition_mode`. **It
leaves every steady state unchanged** — a non-reacting soluble's only transport
is convection, whose fixed point is the uniform feed concentration (overflow =
underflow = feed), verified in `tests/integration/test_takacs.py` — so it only
matters under a dynamic influent, where it smooths the effluent ammonia
peaks/troughs. This is the structural cause of aquakin's wider dynamic-BSM2
effluent-NH4 distribution vs the reference: the reactors agree to corr 0.99 but
the pass-through settler does not damp the soluble signal the way BSM2's
soluble-carrying settler does (the JRN-056 dynamic validation). `build_bsm2(
settler_soluble_holdup=True)` enables it plant-wide.

### Dynamic-solve performance — the stiffness-bound regime and its levers

Profiling the long dynamic BSM2 run (the JRN-056 609-day simulation) established
that it is **stiffness-bound, not wasted-work-bound** — a finding worth not
re-discovering. The signature: ~750–1000 accepted steps/day, **step count nearly
invariant to `rtol`** (1e-4 vs 1e-3 → <2%), **~50% step rejection** with the
default solver, and per-step cost dominated by the **implicit Jacobian
factorisation of the 167-state plant** across the solver's stages (the raw RHS is
~4% of per-step cost). It extrapolates to ~38 min run-only for 609 days.

**Levers that do NOT help (measured — do not re-try):** `jump_ts` at the 15-min
influent kinks (the kinks aren't the bottleneck; a state-triggered clamp kink is
— see below); looser `rtol` (step count is tolerance-independent); PI-controller
tuning (`pcoeff`/`icoeff`) and `factormax` *alone* (they cut the rejection *rate*
but trade rejections for accepted steps → wall flat — reducing the rejection rate
is **not** itself the win); `recycle_passes` 3→1 (changes the answer ~2% via the
concentration-dependent reject loop — unsafe). The wall-time wins come from
**cheaper steps**, **fewer stages**, and **less stiffness**, not from chasing the
rejection rate.

**Verified speedups (20-day proxy, final-state agreement ≤ 6e-5 vs the old
default):** decoupled Newton tolerance **~18%**; `Kvaerno3` **~16%**; the two plus
`factormax=3` together **~42%**. These are exposed as two `Plant.solve` knobs and
one new default:

- **Default decoupled root finder (no opt-in).** `_run_diffeqsolve` now builds
  the default `Kvaerno5` with `root_finder=VeryChord(rtol=10·rtol, atol=10·atol)`
  — the per-stage **Newton** tolerance loosened 10× from the step tolerance.
  diffrax's stock Kvaerno root finder *copies* the controller tolerances, driving
  each stage solve to the full step accuracy (more Newton iterations than the
  embedded error estimate needs); the step controller still enforces the solution
  accuracy through `rtol`/`atol`, so this only ends each stage solve sooner —
  ~15–20% faster everywhere at preserved accuracy. Applies to **every** reactor
  and the forward `jax_adjoint` plant path (the shared `_run_diffeqsolve`); a
  user-supplied `solver=` is honoured verbatim (opts out of the loosening). The
  10× scale is off the *actual* `rtol`/`atol`, so it is correct for any network
  scale (mol/L ozone as well as g/m³ ASM/ADM); validated steady states are
  unchanged within their tolerances.
- **`Plant.solve(solver=...)`** overrides the integrator (`None` keeps the
  decoupled `Kvaerno5`). `diffrax.Kvaerno3` (4 stages vs 7) does less linear
  algebra per step. To keep the Newton decoupling with a custom order, pass it on
  the solver: `Kvaerno3(root_finder=VeryChord(rtol=10*rtol, atol=10*atol))`.
- **`Plant.solve(factormax=...)`** caps the `PIDController` per-step growth factor
  (diffrax default 10). On `Kvaerno3` the levers **stack** (unlike on `Kvaerno5`,
  where `factormax` cancels the Newton saving): `solver=Kvaerno3(...)` +
  `factormax=3` is the **~42%** config.

Both are threaded `Plant.solve` → `_build_jitted_solve` → `_run_diffeqsolve` and
keyed into the per-instance compiled-solve cache (`solver` **by class** — a fresh
stock instance shares the entry, a different class keys separately, a
custom-*configured* instance of an otherwise-default class shares the default's
entry; `factormax` by value). `events=` (the segmented solve) rejects them. They
are **also supported on `gradient="stable_adjoint"`**: the discrete adjoint builds
its backward from the forward solver's Butcher tableau generically, so a cheaper
4-stage `Kvaerno3` forward (with the matching backward recurrence) and a
`factormax` cap apply there too — the same optimized configuration the
`forward_fast` path uses, keyed into the stable-adjoint cache by solver class +
`factormax`. Like `dtmax=`, they do not change the `gradient="auto"` routing.
Covered by `tests/integration/test_plant_solver_option.py` and (the stable-adjoint
path) `tests/integration/test_plant_stable_adjoint.py`.

**Single source of truth for the integrator config — no drift between modes.**
The forward solve and the stable-adjoint *forward* pass used to construct their
solver + step controller independently, so the forward path's accumulated per-step
optimizations (the decoupled-Newton root finder, the colored Jacobian, the
`Kvaerno3`/`factormax` knobs) silently failed to reach the adjoint's forward pass —
it kept paying dense, full-Newton, 7-stage costs. Both now build from one pair of
helpers in [`integrate/_common.py`](aquakin/integrate/_common.py):
`build_implicit_solver(rtol, atol, order=, solver=, colored_root_finder=, linear_solver=, force_root_finder=)`
(the decoupled-Newton `Kvaerno` of the requested `order` — `5` default, `3` for the
lean / forward-sensitivity paths — built from the `_CANONICAL_SOLVERS` table, with
the colored `ColoredVeryChord` or the block-arrow `SimultaneousCorrector`
`linear_solver` injected when given) and
`build_step_controller(rtol, atol, factormax=, dtmax=)` (the PID core the forward
uses directly and the adjoint wraps in a `ClipStepSizeController`). So a future
per-step optimization lands in one place and reaches **both** modes. Specifically
the stable-adjoint forward pass now gets the **decoupled Newton** — previously only
the forward `jax_adjoint`/`forward_fast` paths had it. (The helpers also carry a
`colored_root_finder` so the adjoint *forward* chord could color its per-step
Jacobian too, and `esdirk_adjoint_solve` accepts it — but it is **not auto-enabled**:
the colored *backward* feeds J straight into the transposed solve and is exact on a
superset pattern, whereas the colored *forward* feeds J into an iterative chord
whose decoupled-Newton convergence point depends on the J approximation, so a
colored-vs-dense difference shifts the forward trajectory at the
~Newton-tolerance level (~1e-4) — which would break the bit-identical
`colored_jacobian=True` == dense invariant. It awaits the structural pattern being
reconciled so the colored and dense forward chords converge identically.) The
forward path's construction is unchanged (the helper reproduces it). A regression
guard,
`test_plant_stable_adjoint.py::test_forward_paths_agree_no_config_drift`, asserts
the `jax_adjoint`, `forward_fast`, and `stable_adjoint`-forward integrators realize
the **same primal trajectory** — so a future divergence in any one path's
configuration fails loudly.

**Every diffrax solve funnels through these helpers, structurally enforced.** The
forward-sensitivity reactor path and the plant's colored-solver method were the
last two paths still constructing a `Kvaerno5` / `PIDController` *directly* (the
former with a *tight*, controller-tied Newton — a real divergence from the
decoupled Newton everything else used; the latter hand-injecting the colored root
finder into a bare `Kvaerno5`). Both now build through `build_implicit_solver` /
`build_step_controller`: the augmented `[y; S]` forward-sensitivity solve passes
`order=` (5 reactors / 3 plant) + the `SimultaneousCorrector` `linear_solver`, and
the colored method passes `colored_root_finder=`. So the diffrax ESDIRK object is
constructed in exactly **one** place — the `_CANONICAL_SOLVERS` table inside
`build_implicit_solver` — the only legitimate variation being the explicit axes the
helper exposes (order, colored, `linear_solver`, factormax, dtmax) plus the one
escape hatch (`Plant.solve(solver=...)`, a user object honoured verbatim). The lone
*conceptual* exception is `forward_solve.py` (the `forward_fast` lean
`lax.while_loop`), which builds no diffrax solver at all — its Kvaerno3 tableau is
hand-rolled and must track diffrax's by hand. **A drift guard
([`tests/unit/test_solver_config_single_source.py`](tests/unit/test_solver_config_single_source.py))
AST-scans the package for any direct `Kvaerno*` / `PIDController` / `VeryChord` /
`with_stepsize_controller_tols` *call* outside an explicit allowlist (just
`_common.py`) and fails on a new one**, so a future solve path cannot silently
re-introduce the drift — it must route through the helpers or add an audited
allowlist entry (Python has no real access control, so this static AST lint is the
enforcement, backed by the runtime trajectory-agreement guard above). The
unification also switched the reactor forward-sensitivity from its tight Newton to
the decoupled Newton; the reactor forward-sensitivity suite passes unchanged at its
~1e-8 `jacfwd` tolerance, and the forward / discrete-adjoint paths are bit-identical
(`order=5` reproduces the previous `diffrax.Kvaerno5(root_finder=rf)`).

**`Plant.solve(colored_jacobian=True)` — sparse (colored-AD) Jacobian
materialisation ([`integrate/colored_jacobian.py`](aquakin/integrate/colored_jacobian.py)).**
Profiling the per-step *linear algebra* (after the decoupled-Newton + cached-map
wins) found the dominant cost is **forming the implicit Jacobian, not factorising
it**: diffrax's `VeryChord` materialises + factorises `I − γ·dt·J` **once per
step** (reused across stages/iterations), and for the 167-state plant the dense
materialisation (`jacfwd`, ~33% of a step-attempt) dwarfs the LU factor (~4%,
`cond ~10³` — well-conditioned, fast). But the plant Jacobian is **5–15%
nonzero** — dense per-unit kinetic blocks (the network stoichiometry × rate
dependencies) plus sparse inter-unit flow coupling — so it can be formed by
**column compression** (Curtis–Powell–Reid 1974): group structurally-orthogonal
columns (sharing no nonzero row) into *colors*, push one seed per color through a
single forward linearisation (`jax.linearize` once + `vmap` the tangent — **not**
`jax.jvp` per color, which redoes the expensive nonlinear primal — the recycle +
pH solves — every color), and scatter each color's JVP back to its columns via
the pattern. For BSM2 that is **~45 colors vs 167 columns**, set by the widest
dense block (the digester). The reconstructed matrix **equals the dense Jacobian**
when the pattern is a superset of the real nonzeros, so the chord iteration — the
step sequence, the trajectory, the gradient — is numerically unchanged; only the
formation cost drops. **Measured ~1.43× on the 14-day BSM2 solve** (trajectory
within integration tolerance, ~5e-3, the within-tolerance `LU`-vs-`AutoLinearSolver`
step-path drift; gradient finite and matching the dense path to ~1e-8). It
**stacks** with `Kvaerno3`/`factormax` (it helps any ESDIRK) and the cached map.
- `ColoredVeryChord(VeryChord)` overrides only `init` (materialise via colored
  forward AD into an `lx.MatrixLinearOperator` with an explicit `lx.LU()`,
  avoiding the `AutoLinearSolver(well_posed=None)` least-squares fallback a bare
  matrix would trigger); `step`/`terminate` are inherited, so the chord is
  identical.
- **Sparsity pattern** (`jacobian_sparsity_pattern`): the union of `|J|>tol` over
  **strictly-positive** probe states drawn at **two scales**, plus `y0` itself and
  the full diagonal. Two failure modes must both be covered and a single scale
  covers only one: (1) a *depleted* (zero-at-`y0`) component zeroes the entries
  that couple through it, so the probe lifts every component to `|y0|+1` and
  jitters to reveal those couplings (the *lifted* scale; missing it made an early
  prototype 6× *slower* via an 8× step explosion); (2) a *small-natural-scale*
  component — the ADM1 dissolved hydrogen `S_h2` sits at ~`1e-7` at its inhibition
  knee, where its Jacobian column is enormous — is pushed by that same `|y0|+1`
  lift into a **saturated**, flat regime where the steep column collapses below
  the relative threshold (set by the large biomass/settling entries) and is
  dropped, so the probe also jitters each component around its **own** magnitude
  (the *own* scale) to keep it in its physical regime. Including the Jacobian at
  `y0` makes the start-state guard pass by construction. *(This two-scale probe
  fixed a real fall-back: the BSM2 settler `soluble_holdup` states settle the
  digester to its operating point, surfacing the steep `S_h2` column that the
  lifted-only probe dropped — the colored path then fell back to dense; it now
  stays colored, ~2.6× over dense, matching to round-off.)*
- **Correctness model:** a pattern *miss* does **not** corrupt the result — the
  chord still converges to the stage residual's root — it only degrades
  convergence (costs steps, not accuracy). The pattern is therefore conservative,
  and **guarded once per plant** (`colored_jacobian_max_error`): the colored and
  dense Jacobians are compared at the start state and the solve **falls back to
  the dense solver with a warning** on any mismatch. Built concretely once and
  reused; a first solve under reverse-mode tracing also falls back (the probe
  needs concrete arrays — run one concrete solve to build it, then differentiate).
- Wired like `solver=`/`factormax=` on the forward `jax_adjoint` path (rejected
  with `events=`), keyed into the compiled-solve cache by a `colored_active` flag
  so it never collides with a plain solve. **Most worthwhile for a large stiff
  plant (BSM2); on small BSM1 the materialisation is not the bottleneck (≈1×, but
  still numerically matches to ~3e-13).** Covered by
  `tests/integration/test_colored_jacobian.py` (coloring/reconstruction math,
  positive-probe superset over the trajectory, colored==dense J, full-solve
  trajectory + gradient match, the guard/fallback on a truncated pattern, BSM2).
- **The IC probe goes STALE on a wide dynamic run — fixed by per-component
  structural couplings (issue #388, [`plant/coupling.py`](aquakin/plant/coupling.py)).**
  `jacobian_sparsity_pattern` probes `|J|` at the *start state*, so on a long
  dynamic BSM2 run it drops every coupling that is numerically tiny at the
  warm-start operating point but switches on once the influent drives the plant
  off it — saturated Monod kinetics, the Takacs settling velocity, the ASM<->ADM
  interface's nitrogen-budget branches. Those are the **stiff** couplings, so the
  stale pattern collapses the chord-Newton convergence: colored ran **~6×
  *slower* than no-colored** on the validated 244-state JRN-056 dynamic BSM2 (a
  convergence-rejection explosion — 17× the step attempts, 100% Newton-failure
  rejections). The fix is to build the pattern from the **equations, not a probe**:
  every stateful unit emits its structural Jacobian sparsity via the
  **`CouplingAware`** ABC's **`coupling_pattern()`** — a `self` block
  (`d rhs / d own state`) and an `inlet` block (`d rhs / d inlet concentration`).
  Reactors (`CSTRUnit`, `ADM1DigesterUnit`) derive `self` from the rate AST
  (`structural_sparsity_pattern`; a saturated Monod term is numerically invisible
  to a probe, so the *syntactic* dependency is needed) and `inlet` from the
  dilution diagonal; the `TakacsClarifier` derives both by AD over diverse solids
  profiles (`ad_union` — the settling law is a smooth nonlinearity whose branches
  a sample exercises, unlike Monod saturation); cross-network translators emit
  their own `coupling_pattern()` (`translator_coupling_pattern`, AD over the
  interface branches); stateless units are empty (the `StatelessUnit` default).
  `Plant._structural_plant_pattern` assembles these — `self` blocks on the
  diagonal, each `inlet` block composed with the feeding stream's translator
  coupling on the off-diagonal — **unioned with the IC probe**, which supplies the
  linear, always-on couplings and the recycle's real block structure (so the
  off-diagonal placement is restricted to genuinely-coupled unit pairs, keeping
  the coloring tight). The result is a structural superset that **cannot go stale
  for any influent**, with **no trajectory sampling** (only the single
  always-available IC probe). On the JRN-056 dynamic BSM2 it turns colored from
  ~6× slower into **1.71× faster than no-colored** (49.8 s vs 85.3 s, 37 colors —
  tighter than even a trajectory-sampled pattern), with 0 within-unit couplings
  missing and the residual misses all `|J| ≲ 0.3` (negligible for convergence).
  Covered by `tests/integration/test_coupling_pattern.py` (the contract shapes,
  ABC enforcement, the settler/`ad_union` superset, and the assembled BSM1 plant
  pattern leaving no within-unit coupling missing along a trajectory). The
  reactive units present only in non-BSM plants — `MBRUnit`, `SBRUnit` and
  `IFASUnit`/`MBBRUnit` — **also emit `coupling_pattern()`** (issues #390–#392):
  the MBR is the CSTR's AST kinetics block plus a decoupled fouling-resistance
  diagonal; the SBR unions the AD-derived state couplings (the `1/V` convection
  and the settling-clarity dynamics) over its phases with the rate AST for the
  kinetics; the IFAS unions the (linear) soluble inter-layer diffusion +
  bulk-convection couplings from AD with the per-compartment rate AST (frozen
  layer-biomass rows dropped in the layers). So colored is staleness-free on
  those flowsheets too; the single-unit assembled-pattern superset is checked per
  unit in the same test.
- **Also colors the `gradient="stable_adjoint"` BACKWARD pass.** Since the
  saved-stage backward (above) reconstructs the stages instead of re-solving them
  by a 12-iteration Newton scan, the per-step `df/dy` builds dropped from ~79 to
  the **~7 stage `Js` only**. So the Jacobian builds — once **~82%** of the
  backward (with the dense solves ~17% and the parameter vjp ~1%, when the Newton
  recompute dominated) — are now only **~24%** (BSM2, estimated from the
  single-build colored ratio below); the rest is the transposed solves + parameter
  vjps + the stage reconstruction. **The bottleneck has shifted off the builds**,
  so coloring them — which still cuts each build's cost — now moves the total far
  less (the measured numbers below). `colored_jacobian=True` passes a colored
  builder into
  `esdirk_adjoint_solve` (`jacobian_builder=`), which builds each stage Jacobian
  in one JVP per *color* instead of per state. The coloring is derived once,
  concretely, for the **augmented** (`n+1`, time-carrying) primal rhs the discrete
  adjoint differentiates — `Plant._colored_adjoint_jacobian_builder`, the backward
  analogue of `_colored_jacobian_solver`, guarded by `colored_jacobian_max_error`
  with a dense fallback, cached in `_colored_adjoint_builder`. **Its sparsity
  pattern is the per-component structural pattern (issue #381).** The backward
  feeds `J` directly into `I − dt·γ·Jᵀ` and the transposed solve, so a missed
  coupling does **not** cost steps (as it does for the self-correcting forward
  chord) — it **silently corrupts the gradient**, undetected mid-transient by the
  start-state guard, and a start-state-only or trajectory-*sampled* pattern can
  miss a coupling that only activates at an unvisited correlated operating point.
  So the builder unions the IC probe with **`_structural_plant_pattern`** (each
  unit's equation-derived `coupling_pattern()` — the same complete assembly the
  forward path uses), embedded in the augmented `[y; τ]` layout's `df/dy` block
  (the probe supplies the always-on `τ` time-dependence column). A *complete*
  structural superset closes the silent-corruption risk a sampled pattern only
  reduces. The PTC steady-state builder (`_colored_steady_jacobian_builder`) uses
  the same structural pattern (it marches in a narrow neighbourhood so the probe
  usually suffices, but the superset is complete regardless). Validated: the
  colored backward gradient w.r.t. a kinetic param **and** a flow-setpoint param
  (`underflow_split.ras`, where `dM/dθ ≠ 0`) matches the dense-Jacobian gradient,
  and the backward / PTC guards fall back to dense on a truncated pattern
  (`tests/integration/test_plant_stable_adjoint.py`, `test_colored_jacobian.py`).
  The dense default
  (`jacobian_builder=None`) is a trace-time branch, so it is bit-identical to the
  historic backward; the colored gradient equals the dense one (exact on the
  superset pattern — only float summation order differs: ~1e-15 on BSM1, ~6e-7 on
  BSM2 through the ADM1 pH-solver linearization, well inside the FD/`jax_adjoint`
  match envelope). **Re-measured on the saved-stage backward (BSM2/BSM1
  `value+grad`, 3-day warm-started span, dense vs colored backward): BSM2 backward
  `615 → 558 ms` (colored `0.91×`, ~9% faster — down from the ~1.95× of the
  Newton-recompute era); BSM1 backward `1105 → 1328 ms` (colored `1.20×`, now
  *slower* — its build overhead exceeds the saving at `n_colors=14` vs 65). With
  the builds no longer dominant, the colored win is marginal on BSM2 and negative
  on BSM1 — which is exactly what the `"auto"` decision (below) picks.**
- **`colored_jacobian="auto"` is the default — it measures whether the backward
  coloring pays and turns it on only then.** The GO/NO-GO reduces to "is the
  colored `df/dy` build cheaper than dense?" (the backward rebuilds it ~7×/step —
  ~24% of the backward — so the *sign* of the build-time difference is still the
  sign of the overall speedup, just a smaller magnitude than in the
  Newton-recompute era; the heuristic still picks correctly, enabling colored only
  when it helps). The
  decision **must** use *jitted* build times — eager timing is misleading because
  XLA fuses the colored build's scatter away (eager shows colored slower for both
  plants; jitted shows BSM1 `ratio=0.50`, BSM2 `ratio=1.65`). So on the first
  concrete `stable_adjoint` solve `_colored_build_speedup` jit-compiles the dense
  and colored builds once (a few-seconds one-time cost, cached, amortized over a
  calibration) and stores `ratio = t_dense/t_colored`; `"auto"` enables coloring
  iff `ratio > _COLORED_BACKWARD_MARGIN` (1.05). Validated: BSM1 → `("dense",
  0.50)`, BSM2 → `("colored", 1.65)`, each matching the measured outcome (BSM2
  colored is ~9% faster, BSM1 colored is slower), with the auto gradient equal to
  the dense gradient. `plant.colored_jacobian_decision()`
  returns `("colored"|"dense", ratio)`. **`"auto"` governs only the
  `stable_adjoint` backward** — it leaves the **forward** `jax_adjoint` solve dense
  (forward coloring swaps the implicit linear solver and so is not guaranteed
  bit-identical; making it the all-solves default is a separate, full-suite-
  validated change). `colored_jacobian=True` forces coloring on **both** paths
  (skipping the measurement); `False` disables it. Covered by
  `test_plant_stable_adjoint.py` (`test_stable_adjoint_colored_jacobian_matches_dense`,
  the forced path; `test_stable_adjoint_colored_jacobian_auto_off_for_small_plant`,
  the auto decision).

**`Plant.solve(forward_fast=True)` — lean non-AD forward integrator
([`integrate/forward_solve.py`](aquakin/integrate/forward_solve.py)).** A stiff
diffrax solve carries machinery whose purpose is to make the *whole solve*
differentiable — an optimistix root finder, a lineax linear-solve abstraction, and
a checkpointing reverse-mode adjoint (`custom_vjp`). **Tracing all of that
dominates compile time** (the implicit scaffolding traces ~10× slower than the
bare ODE loop — an explicit-solver solve of the same plant RHS traces in ~2 s vs
~30 s for the diffrax implicit solve; the RHS itself is ~0.2 s). A **forward-only**
plant solve — one that never needs `jax.grad`/`calibrate`/`sensitivity` of the
result — can skip all of it: `forward_solve` is a plain `lax.while_loop` running
the Kvaerno3 ESDIRK stages with a simplified Newton (Hairer–Wanner contraction
test) + a direct dense `lu_factor`/`lu_solve`, an embedded-error PI controller, and
a **convergence-aware step-growth limiter** (the Newton contraction rate caps the
step growth, so it rarely grows into nonlinear-divergence). Output at `t_eval` is
*exact* — the step is clipped to land on each save time (no dense-output
interpolation; for a dense `t_eval` this adds ~clip-per-save-interval steps, the
one cost vs diffrax's interpolation — a future dense-output refinement).
- **The per-step Jacobian `J = df/dy` is STILL colored forward-mode AD** (the same
  exact matrix the differentiable path forms, so the step behaviour matches). What
  is dropped is only the *adjoint over the whole solve*: the result is **not**
  differentiable w.r.t. parameters / initial conditions. So `J` uses AD locally;
  the solve is just not wrapped to be differentiable globally.
- **Opt-in, forward-only, concrete-only.** Rejected with `events=` and
  `gradient="stable_adjoint"`, and **requires concrete `params`/`y0`** (a `jax.grad`
  / `jax.jit` of a `forward_fast` solve raises a clear error — the `lax.while_loop`
  is not reverse-mode differentiable and the colored pattern needs concrete arrays
  to build). It needs the colored-Jacobian pattern; it builds + guards it like
  `colored_jacobian=True` and **falls back to the diffrax forward path (with a
  warning) if the guard fails**. Composes with the cached recycle map; its compiled
  solve is cached per instance (so a parameter sweep at fixed signature reuses it).
- **Measured: ~3× faster compile** (the implicit-machinery tracing collapses — the
  part that *cannot* be file-cached, since tracing is Python; standalone 14 s vs
  42–48 s) — the robust win, and the main reason to use it for long one-off dynamic
  runs (the 7-min full-BSM2 compile). **Run is ~1.3–1.9× faster** on the
  validated 244-state JRN-056 dynamic BSM2 (real 609-day influent, `HeatBalance` +
  `settler_soluble_holdup`): a 60-day window is 1.91× and the full 609-day run
  1.26× — the run gain narrows over a long run because the `t_eval` step-clipping
  adds a boundary per save point (8737 over days 245–609) where diffrax interpolates
  (a future dense-output refinement would recover it). Same accuracy (a valid
  solution to the same `rtol` — it differs from the diffrax trajectory only by the
  step-sequence variation between two valid adaptive solves, ~2e-2 on the dynamic
  BSM2). Covered by
  `tests/integration/test_forward_fast.py` (analytic decay + order, exact `t_eval`,
  BSM1/BSM2 agreement with diffrax, the guards). NOTE the lean integrator (Kvaerno3,
  3rd order) vs the diffrax default (Kvaerno5, 5th order) — both controlled to
  `rtol`, so equally `rtol`-accurate, K3 just takes more (cheaper) steps; a Kvaerno5
  forward_fast is a future option if a tighter match to the validated steady states
  is wanted.

**`S_h2` quasi-steady-state — TESTED AND REJECTED for our solver (issue #361).**
Every production WWTP simulator (BSM2 reference, GPS-X, WEST) makes the two
fastest ADM1 states — pH **and dissolved hydrogen `S_h2`** — algebraic, reporting
~18–28× (Rosen et al. 2006, Table 4.5). **But that win is an *explicit*-solver
(ODE45) benefit and does NOT transfer to our L-stable implicit `Kvaerno5`.** A
proof-of-concept confirmed it: the QSS equation is sound (monotone residual,
unique root, algebraic `S_h2` = 2.508e-7 vs the reference 2.506e-7, slow states
reproduced to 1e-10), but freezing `S_h2` at its exact QSS value via a smooth
Newton solver left the digester step count **unchanged** (801 vs 812 steps; 384
vs 389 rejections). The reason is fundamental: an L-stable implicit method
*already* performs the QSS implicitly — it damps the `S_h2` fast mode (eigenvalue
~1.4e6 d⁻¹) to its quasi-steady value at any step size, so removing it by hand is
redundant. (pH is different: it is not a fast *mode* but a state-derived
algebraic condition, which is why we solve it directly.) **Do not build the
`S_h2` DAE machinery** — it is multi-day, fragile, and zero-benefit here.

**The `clip_negative_states` `max(x,0)` kink — ALSO tested and rejected (issue
#361).** The hypothesis was that the hard clamp is a *state-triggered moving
derivative kink* near depleted species (DO in anoxic zones, depleted substrates)
that the embedded error estimator rejects at (and that, being state-triggered,
can't be a `jump_ts` breakpoint — why `jump_ts` did nothing). A direct test
replaced it with a smooth clamp `½(x+√(x²+ε²))` and swept `ε` over three orders
of magnitude on the BSM2 dynamic solve: the rejection rate stayed **pinned at
~50.7%** at every `ε` (even `ε=0.5`, which rewrites the entire sub-0.5 region of
the RHS), while the smooth clamp *biased* depleted species toward `ε/2`. So the
clip kink is **not** the rejection source either, and the smooth clamp is a net
negative (no speedup, worse accuracy).

**Diagnosis of the ~50% rejection — a controller property, and why it is not
worth chasing.** Across all experiments the rejection rate moved *only* for
*step-size-controller* changes — `factormax=3` (→39%) and PI coefficients (→31%)
— and never for RHS changes (tolerance, `jump_ts`, `S_h2` QSS, the clip kink).
It is the classic deadbeat-I-controller overshoot→reject→shrink oscillation on a
stiff forced system: diffrax's `PIDController` is a pure Söderlind error filter
with **no Gustafsson iteration-count-aware predictive control** (the standard
cure in RADAU/SDIRK codes, which diffrax lacks). Crucially, **lowering the
rejection rate this way does not lower wall time** — it trades rejected steps for
accepted ones. The wall-time wins are therefore *cheaper steps* (the decoupled
Newton default) and *fewer stages* (`Kvaerno3`), banked in the dynamic-solve
knobs above; the residual rejection rate is left as-is. A genuinely different
integrator (a Gustafsson-predictive controller, or an exponential/QSS-reduced
non-stiff formulation) is the only thing that would cut it further, and is a
large effort not currently justified by the already-shipped ~42%. See issue #361
for the full experiment log.

**IFAS / MBBR unit ([`plant/ifas.py`](aquakin/plant/ifas.py)).** `IFASUnit`
(alias `MBBRUnit`) places carrier-media biofilm in the flowsheet by **wiring the
existing depth-resolved `BiofilmReactor`** (1-D diffusion–reaction over biofilm
depth) into a plant unit, alongside the suspended (CSTR) fraction — the
intensification retrofit the BSM palette lacked. Its state is the bulk
concentration **plus** the biofilm layer profile (`(n_layers+1)·n_species`); its
`rhs` is `BiofilmReactor._make_rhs` (finite-volume bulk↔surface↔…↔wall soluble
diffusion + per-compartment reaction) with the **plant's bulk convection +
aeration added on the bulk row**, replacing the biofilm reactor's own
stand-alone CSTR feed (built with `feed=None`). Carrier geometry is the
designer's `specific_surface_area` (media SSA, m²/m³) × `fill_fraction` →
`area_per_volume`; oxygen enters the **bulk** and reaches the biofilm only by
diffusion (so deep layers can be O₂-limited — the reason for depth resolution).
The effluent is the well-mixed bulk; the biofilm stays on the carrier. Aeration
reuses the **same `Aeration` spec as `CSTRUnit`** (open- or closed-loop; the
plant's generic `_materialize_aeration` auto-wires a DO controller from the
spec), via aeration helpers (`build_aeration_vectors` / `aeration_transfer`)
factored out of `CSTRUnit` and shared by both (CSTR behaviour is bit-unchanged).
The biofilm is a **mature, fixed attached-biomass** model: the layers' biomass +
inert structure is held as a sustained reservoir while the substrate pools and
solubles react and diffuse and the suspended bulk fraction evolves fully. The
default freeze mask is **stoichiometry-derived** (`_default_biofilm_fixed_mask`):
freeze every particulate **except** a hydrolysis substrate (one consumed while a
soluble is produced, `XS→SS`) — freezing such a pool would make it a
non-depleting soluble source (the biofilm footgun), whereas biomass/inerts are
the intended structure. For ASM1 that freezes `XI`/`XB_H`/`XB_A`/`XP` and leaves
`XS`/`XND` dynamic. **Validated:** at equal volume + aeration an IFAS tank
removes more soluble COD (effluent SS 1.8 vs 2.7) and nitrifies markedly more
(SNH 0.4 vs 2.6) than a plain CSTR, converges to steady state, and `jax.grad`
flows end-to-end through the biofilm core (`tests/integration/test_ifas.py`). A
fully dynamic biofilm (growth with attachment/detachment/a density cap) is the
underlying `BiofilmReactor`'s domain and a follow-up for the unit.

**Coupled BSM1 — steady state now works.** The *coupled* BSM1 plant reaches the
correct steady state for **both** clarifiers (Takács and Ideal agree: tank-5
XB_H ≈ 1.7e3, SNH ≈ 0.5, healthy nitrification) in ~10 s. Getting there took
three fixes, all diagnosed against the official BSM1 reference code:
- **Decoupled recycle-flow resolution** (`Plant._resolve_flows`): the recycle
  *flow* network is linear and concentration-independent, but BSM1's loop gain
  is ≈0.99 (3× internal + 1× RAS), so the old 3-pass Gauss–Seidel left the
  flows at ~40% of steady → starved underflow → washout. Each unit now exposes a
  `flow_outputs` rule; `Plant` solves the small flow fixed point **exactly**
  (probe the affine map, one `lineax`/`jnp.linalg.solve`), then runs the
  concentration sweep on the fixed flows. This was the keystone. **Affinity is
  checked:** the probe is exact only if every `flow_outputs` is affine in the
  recycle flows, which a threshold-mode `SplitterUnit` / `StorageTank` bypass is
  *not* (piecewise-linear, a kink). On the first non-traced `solve` the plant
  re-evaluates the forward pass at the solved recycle flows and `warnings.warn`s
  if it does not reproduce them (`Plant._warn_if_flow_nonaffine`) — the residual
  is exactly the affine-violation indicator, so it fires only when a
  recycle-dependent inlet actually crosses a kink (no false positives: the shipped
  BSM2 bypass/storage plants, whose such units are fed by the influent / a fixed
  pump, never warn). It is a warning, not an error: a kink the flow never crosses
  in operation is still correct.
- **Non-negative flow split** (the remainder outflow clamped into `[0, Q_in]` in
  both clarifiers): guards against a negative underflow when the feed dips below
  the design split — closes issue #17; inactive at steady state.
- **`clip_negative_states`** on ASM1 (the reference `xtemp = max(x,0)` clamp).

`Plant.solve` takes an optional `y0=` for warm-starting (e.g. a dynamic run
from a precomputed steady state).

**Reaching steady state — `plant.run_to_steady_state(...)`.** A single continuous
adaptive solve that **self-terminates** at steady state via diffrax's
`steady_state_event` (halts when `||dstate/dt|| <= ss_atol + ss_rtol*||state||`,
the standard march-to-steady-state criterion) — no fixed horizon to guess and no
chunked re-integration; `max_time` is only a safety cap (reached ⇒
`converged=False`). Returns a `SteadyStateResult(state, converged, time,
solution)`. Implemented by threading an `event=` argument through
`Plant.solve` → `_run_diffeqsolve` → `diffeqsolve` (forward `jax_adjoint` path
only; rejected under `stable_adjoint`). Warm-started BSM2 settles in ~51 d /
~25 s reproducing the validated steady state.

**Algebraic steady state — `plant.steady_state(...)` (pseudo-transient continuation).**
A fast, robust, *differentiable* alternative to the forward solve: it finds the
root of the plant RHS `F(y)=dy/dt=0` directly by **pseudo-transient continuation
(PTC)** rather than integrating until the dynamics die out. The core lives in
[`plant/steady.py`](aquakin/plant/steady.py) (`solve_steady_state` / `ptc_forward`)
and is reusable on any `rhs(y, params)` — `BiofilmReactor.steady_state` is also
routed through it (replacing the Levenberg–Marquardt root-find that stalled). PTC takes damped-Newton steps
`(V/δ − J)·Δy = F(y)` with the exact AD Jacobian `J = ∂F/∂y` (forward-mode) and a
**per-state** pseudo-time `V/δ`, `V = diag(max(|y|, floor))`: at small `δ` the
step is a stable backward-Euler move along the physical transient (globally
convergent, like time-stepping — the regime where a plain Newton root-find
stalls), and as the Switched-Evolution-Relaxation ramp `δ ← δ·min(cap,
‖F_old‖/‖F_new‖)` grows `δ` the term vanishes and it becomes Newton (quadratic
terminal convergence). The per-state `V` is essential — plant states span orders
of magnitude (DO ~2, heterotrophs ~2000, gas ~1e-3) and a scalar `I/δ` thrashes.
This is the standard method for "forward integration converges but Newton stalls"
stiff systems (Kelley–Keyes 1998; the flowsheet form is Pattison–Baldea 2014) and
is what production simulators use to snap to steady state on any topology.
- **Validated:** BSM1 (75 iters) and BSM2 (the 167-state plant with the long-SRT
  digester — the stiff case a plain root-find stalls on; 85 iters) both reach the
  forward-integration steady state to within ~1–3% on every key state, **~10×
  faster** than `run_to_steady_state`, to a tighter residual.
- **Differentiable in BOTH AD directions** for design sweeps and sensitivity: the
  returned `state` carries the **implicit-function-theorem** parameter
  sensitivity (the iteration — a `while_loop` — is gradient-blocked; the
  sensitivity is re-attached by a **`custom_jvp`** that gives the forward tangent
  `dy = −J⁻¹(∂F/∂params)·dθ`). Because that map is *linear in the tangent*, JAX
  transposes it automatically to the reverse gradient `−(∂F/∂params)ᵀJ⁻ᵀḡ`, so the
  one rule serves **forward** (`jax.jvp`/`jacfwd` — the many-output
  sensitivity-screen direction) and **reverse** (`jax.grad`/`jacrev` — the
  calibration-gradient direction) alike. (It was a reverse-only `custom_vjp`;
  the `custom_jvp` is what unblocks forward-mode AD and `dgsm(ad_mode="forward")`
  through `plant.steady_state`.) `J = ∂F/∂y` is full rank for the shipped networks
  at their operating point, where the `jnp.linalg.solve` is exact; a rank-deficient
  `J` (a fully dormant species) leaves the IFT sensitivity undefined along that
  null direction (the old `lstsq` returned an arbitrary min-norm cotangent there
  rather than the exact gradient, so it is not used). Verified: forward == reverse
  to machine precision and both match finite differences (`tests/integration/test_steady_state.py`).
- **`plant.steady_state_sensitivity(params, *, output_fn=, wrt=, mode=, elasticity=)`** —
  the exact steady-state output sensitivity `d(output)/dθ` from the IFT, **far
  cheaper than `jacfwd`/`jacrev` through `steady_state`** (which re-solves per
  call): it solves the steady state once and reuses a single `∂F/∂y` factorisation
  for every output and parameter. `output_fn` maps the flat plant state to a
  length-`m` output vector (default: the full state, giving `dy*/dθ`). `wrt`
  selects the parameters to differentiate (flat indices or `"<network>.<param>"`
  names; default all) — restricting to `k` parameters makes forward mode cost `k`
  solves rather than `n_params`. `mode` selects the AD direction — `"forward"`
  (one solve per parameter, all outputs follow; efficient when outputs outnumber
  parameters), `"reverse"` (one transposed solve + VJP per output, all parameters
  follow; efficient when parameters outnumber outputs), or `"auto"` (forward iff
  `k ≤ m`). Both give the same exact sensitivity; `elasticity=True` returns the
  dimensionless `(dg/dθ)(θ/g)`. This is the general form of the plant-scale
  sensitivity screen.
- **`plant.steady_state_dgsm(ranges, *, output_fn=, wrt=, mode=, n_samples=, seed=, cond_factor=)`**
  — **global** sensitivity (DGSM) of the steady state: samples the screened
  parameters over their ranges (scrambled-Sobol QMC), solves the steady state at
  each sample, and reads each output's sensitivity through
  `steady_state_sensitivity` — reusing **one** `∂F/∂y` factorisation per sample, so
  it is far cheaper than the generic `aquakin.dgsm` over `steady_state` (whose
  `jacfwd`/`jacrev` recompute the steady-state structure per input tangent /
  output). Aggregates to the Sobol total-index upper bound
  `S_ij^tot ≤ ν_ij(b_j−a_j)²/(π²Var(g_i))`, `ν_ij = E[(∂g_i/∂z_j)²]`, returning a
  `SteadyStateDGSMResult` (`sobol_total_bound`/`std_error` shape `(m, k)`,
  `.ranked(output)`). Non-finite samples are dropped per output exactly as
  `aquakin.dgsm` does, so with `cond_factor=None` (default) the bounds are
  **bit-identical to `aquakin.dgsm`** (same Sobol seed → same points → same
  formula), just computed more cheaply. **`cond_factor`** adds the
  heavy-tail robustification a stiff plant needs: a steady-state sensitivity
  `−J⁻¹(…)` is only well-defined at a hyperbolic operating point, but over a wide
  parameter screen many Sobol samples land near plant bifurcations (washout,
  nitrification collapse) where `∂F/∂y` is near-singular and the sensitivity blows
  up (finite but huge), giving the DGSM a heavy tail the Monte-Carlo mean cannot
  resolve (it spikes and fails to reach `1/√N`); `cond_factor` drops any sample
  whose Jacobian condition number exceeds `cond_factor ×` the sample median — a
  near-singular operating point — restoring finite variance and clean `1/√N`
  convergence (the condition number is recorded per sample in `result.cond`).
  It **retains the per-sample data**, so `result.convergence()` returns the running
  bound + MC standard error versus sample count — the **sample-size convergence
  study** with no re-solving — and `result.with_cond_factor(c)` re-applies a
  different threshold (re-aggregating from the retained data, no re-solve).
  (`tests/integration/test_steady_state.py`.)
- **Dynamic (transient) sensitivity — `plant.dynamic_sensitivity(params, *, output_fn=, t_span=, t_eval=, wrt=, mode=, elasticity=)` and `plant.dynamic_dgsm(ranges, *, output_fn=, t_span=, t_eval=, wrt=, mode=, n_samples=, seed=)`.**
  The dynamic counterparts of the steady-state pair, for an output that depends on
  the *trajectory* (an effluent time series, a window average, a peak) rather than
  the operating point. There is no implicit-function-theorem shortcut here, so the
  cost is one stiff solve per direction (sensitivity) or per sample (DGSM), far
  heavier than the steady-state IFT. The wrapper's value is **using the stable
  method for each AD direction**, the easy thing to get wrong by hand — a naive
  differentiation of `plant.solve` is non-finite on a stiff plant. `mode="reverse"`
  differentiates the solve through the cap-free `gradient="stable_adjoint"` (one
  `jax.vjp`). `mode="forward"` integrates the augmented `[y; S]` variational system
  (`plant.solve_sensitivity`), whose step controller bounds `S` so it stays
  **finite over long horizons** where forward-mode `jacfwd` through the stiff solve
  goes non-finite (a numerical, not genuine, blow-up — the true sensitivity stays
  bounded), then chains the full-state sensitivity through `output_fn` with one
  `jax.linearize` of the output map over the saved trajectory (no extra solve).
  `output_fn` maps the `PlantSolution` to a length-`m` output vector. The reverse
  solve is differentiated directly (no enclosing jit — the primal must run with
  concrete params, since some plant setup, e.g. a unit `initial_state`,
  concretizes; an outer jit makes the BSM2 dynamic plant fail with a
  `ConcretizationTypeError`); the solve's own compiled-solve cache still reuses the
  integrator compile. **Forward is the memory-light direction over a long horizon**
  — `solve_sensitivity` carries the parameter tangents in lockstep with the state,
  so memory is independent of the integration length, whereas reverse stores the
  whole trajectory to replay it (prohibitive over a 609-day horizon). Both
  directions go through the shared `Plant._dynamic_value_jac` helper, so
  `dynamic_dgsm`'s per-sample screen inherits the same stable forward/reverse.
  `dynamic_dgsm` reuses the per-sample sensitivity into a Sobol total-index screen
  returning a `DynamicDGSMResult`
  (mirroring `SteadyStateDGSMResult`: `.ranked()`, `.convergence()`); verified
  forward == reverse, the reverse sensitivity matches a manual `stable_adjoint`
  gradient to machine precision, `dynamic_dgsm` matches `aquakin.dgsm` over the
  same transient solve, and `solve_sensitivity` matches `jacfwd` where both are
  finite (`tests/integration/test_dynamic_sensitivity.py`). The steady-state pair
  stays the cheap, both-directions-free path (the IFT); the dynamic pair is the
  convenience layer over the stable differentiation of `plant.solve`.
- **Design variables** (`steady_state(..., design=...)`): because the IFT
  differentiates w.r.t. *whatever pytree the residual consumes*, the steady state
  is differentiable w.r.t. design variables, not only kinetic parameters, by
  folding them into `θ = (params, design)`. **Influent load** is wired:
  `design={"influent": {port: {"Q": ..., "C": ..., "T": ...}}}` (plain arrays —
  a `Stream` can't be a θ leaf, it carries the non-JAX `network`) overrides the
  recorded influent at `influent_time` inside `_resolve_streams`/`_resolve_flows`,
  so `jax.grad` of a steady-state output w.r.t. the influent composition/flow
  works (BSM1 `d(effluent NH)/d(influent NH)` matches FD).
- **Flow setpoints as first-class parameters (the SRT / recycle knobs).** A flow
  setpoint — a recycle / wastage pump flow, a clarifier underflow, the primary
  sludge fraction — is consumed in **two** code paths (`_resolve_flows` →
  `flow_outputs` and `_sweep_outputs` → `compute_outputs`, which recompute the
  split). [`plant/flow_setpoint.py`](aquakin/plant/flow_setpoint.py)'s
  `FlowSetpoint` is the single source of truth: both paths call
  `resolve(flow_params)` on the same object, so they cannot desync, and the value
  is read from the unit's slice of the **parameter vector** (which both
  `flow_outputs` and `compute_outputs` already receive as `params_unit`) — making
  it differentiable everywhere (steady-state IFT *and* dynamic solves) with no
  Protocol change. `_build_parameter_layout` **appends** a per-unit flow-setpoint
  block after the kinetic network blocks (so kinetic indices are unchanged); the
  setpoints are addressed by name `"<unit>.<setpoint>"` (e.g.
  `"underflow_split.ras"`, `"clarifier.underflow_Q"`, `"primary.f_PS"`). The
  `FlowParameterized` mixin (on `SplitterUnit`, `IdealClarifier`,
  `TakacsClarifier`, `PrimaryClarifier`) provides the resolution; a unit used
  standalone (no plant) resolves the default, so it is unchanged. **Backward
  compatible:** a kinetic-only parameter vector (the pre-flow convention, e.g.
  `bsm2_parameters`) is padded with the default flow setpoints by
  `Plant._coerce_params`. Validated: BSM1 `d(effluent NH)/d(RAS flow)` matches FD
  (and is negative — more recycle retains biomass, lowering effluent ammonia).
  *Not* a flow setpoint: the thickener/dewatering underflow is
  concentration-derived (`%TSS` target), so `IdealThickener` is left as-is.
- Returns the same `SteadyStateResult` (now `method="ptc"`, with `iterations`
  and the scaled `residual`; `time`/`solution` are `None`). Eager calls get
  concrete diagnostics and, if PTC fails to converge within `max_iter`, an
  automatic **fallback** to `run_to_steady_state` (`method="ptc->forward"`);
  under a `jit`/`grad` trace the diagnostics are traced values and the fallback
  is skipped (only the differentiable `state` is used there). Constant influent
  is assumed (the residual samples the influent at `influent_time`, default 0).
- **Step-acceptance guard (the robustness lever, `divergence_factor`).** PTC is
  legitimately **non-monotone** — a healthy step can spike the scaled residual
  (~20–30× on the BSM plants) and recover, so the ramp must accept those. But a
  Newton step from far off the solution (a cold start) can overshoot into a bad
  region where the residual blows up by orders of magnitude and then goes
  non-finite — the accept-always iteration runs to **NaN**. `ptc_forward` now
  **rejects** a non-finite or grossly-diverging step (scaled-residual growth past
  the generous `divergence_factor`, default `1000`): it **holds the iterate and
  hard-shrinks `dt`** (×0.1, floored) so the retry is a stabler backward-Euler
  step. The threshold sits in the **wide gap between benign (~30×) and
  catastrophic (~1e5×) growth**, so a converging run **never rejects and is
  bit-identical** to the unguarded iteration (BSM1/BSM2 warm: same 23/38
  iterations), while a divergent one is pulled back. **Measured:** BSM2 from a
  *cold* `initial_state` went to NaN before; it now stays finite and **converges**
  (~450 PTC iterations on the dev machine — the cold-start *count* is numerically
  platform-sensitive, so it is not asserted in CI). The growth guard is what
  rescues it, not merely catching NaN: a non-finite-only guard
  (`divergence_factor=inf`) accepts the finite blow-ups and stalls. The guard is
  in the core `ptc_forward`, so it also hardens `BiofilmReactor.steady_state` and
  direct callers, identity on the happy path. Regressions (both fast +
  deterministic — no brittle convergence count):
  `test_steady_state.py::test_ptc_step_guard_keeps_overshoot_finite` (an overshoot
  to a non-finite region is rescued to convergence) and
  `::test_ptc_step_guard_rejects_finite_blowup` (a large *finite* residual blow-up
  is rejected with the default `divergence_factor` but accepted with `inf`,
  checking the acceptance logic directly).
- **Per-state pseudo-time / residual scaling (the iteration-count lever).** PTC's
  step damping `V` and convergence criterion both use `max(|y|, scale_floor)`. A
  flat scalar floor (the old default `1.0`) **over-damps the small-magnitude
  states** (gas fractions ~1e-3, dissolved hydrogen ~1e-7): their relative rate is
  throttled, which throttles the SER `dt`-ramp and roughly **doubles** the
  iteration count. `plant.steady_state` now defaults `scale_floor` to a
  **per-state** floor `max(|y0|, 1e-6)` — each state scaled by its own warm-start
  magnitude — so every state has a magnitude-consistent pseudo-time. Measured on
  **BSM2: 80 → 38 PTC iterations (run-only 39.4 → 18.9 ms, ~2.1×), same root
  (rel ≤ 7e-6)**; neutral on BSM1 (24 → 23). The small `1e-6` absolute floor
  anchors near-zero states — a *pure* `|y|` relative scale (no floor) is faster
  still on BSM2 (~44 it) but **destabilises BSM1** (~280 it), so the `|y0|`-anchored
  floor is the robust choice. The win is in the **run/amortized** time (a jitted
  design sweep, calibration); the un-jitted one-shot `steady_state` wall is
  compile-bound, so its time is roughly unchanged. The change is confined to
  `plant.steady_state`'s default — `ptc_forward` / `solve_steady_state` keep the
  scalar `scale_floor=1.0` default (so `BiofilmReactor.steady_state` and direct
  callers are unchanged), and an explicit `scale_floor` (scalar or per-state
  array) is always honoured. `scale_floor` only affects the path and the
  convergence criterion, never the root. Regression: `test_steady_state.py::
  test_bsm2_steady_state_per_state_scaling_cuts_iterations`.
- **Compiled-solve cache — the single-run-compile lever (`Plant._steady_jit_cache`).**
  A one-shot `steady_state` is **~99% compilation** (BSM2: ~12 s compile of the
  plant-RHS `jacfwd` inside the PTC `while_loop`, vs ~40 ms of actual solving),
  and the eager `jax.lax.while_loop` in `ptc_forward` **re-traces and recompiles
  on every call** — so before this, a *repeated* `steady_state` (a temperature /
  SRT sweep, multistart, regenerating a figure) paid the full ~12–17 s each time.
  `plant.steady_state` now **persists a jitted forward solver** keyed by the PTC
  settings (`dt0`/`dt_max`/`growth_cap`/`max_iter`/`tol`/`nonneg`/`influent_time`)
  and reuses it, so JAX skips the recompile: **BSM2 call 1 ≈ 15.6 s, call 2 ≈
  0.02 s (~780×), and a swept-`params` call is also ~0.02 s** (the `rhs` reads
  `params` as a jit *argument* and recomputes the recycle map inside, so one
  compiled solver is correct for any params — and `y0`-derived `scale_floor` is an
  argument too, so a varying warm start does not recompile). Cached **only on the
  dense, design-free, concrete path**: the colored primal bakes a params-derived
  recycle map, the `design=` path differentiates a pytree, and a traced (gradient)
  call needs the IFT `custom_vjp` — those keep `solve_steady_state` (the gradient
  is amortized by the caller's own `jit`). The cached path returns the converged
  state directly (no IFT wrapper — a concrete call takes no gradient) and still
  honours the non-convergence fallback to `run_to_steady_state`. The cache is
  cleared by `set_temperature` / `set_temperature_model` (they change the RHS /
  state size). **This does NOT speed up the *first* (one-shot) call — that compile
  is irreducible here — only repeated calls.** Regression: `test_steady_state.py::
  test_bsm1_steady_state_solve_is_cached` (one entry, reused across params,
  bit-identical re-call, gradient path bypasses it and stays finite).
- **`steady_state(..., colored_jacobian=True)` — sparse (colored-AD) PTC
  Jacobian.** PTC forms the full plant `dF/dy` every Newton step (~tens of times
  for BSM2), the *same* block-sparse object the integrator's implicit-stage
  (`Plant.solve(colored_jacobian=True)`) and the `stable_adjoint` backward color.
  This flag materializes it by column compression (one Jacobian-vector product
  per color — BSM2 46 colors vs 167 states) instead of dense `jax.jacfwd`,
  reconstructing the same matrix on the sparsity-pattern support: **bit-identical
  to dense on a single-network plant** (BSM1, the recycle reconstruction is
  exact) and **identical to PTC tolerance (~1e-7) on a multi-network plant**
  (BSM2 — the colored `linearize`+vmap materialization orders the recycle
  linear-solve arithmetic differently from dense `jacfwd`, a round-off difference
  well inside the 1e-6 convergence tolerance; same 83 iterations). The injection
  point is `ptc_forward`/`solve_steady_state`'s new `jac_fn=(F, y) -> dF/dy`
  argument; `Plant.steady_state` builds the colored materializer once concretely
  (`_colored_steady_jacobian_builder`, reusing the `colored_jacobian` module's
  pattern/coloring) and **guards** it against the dense Jacobian at the warm
  start, **falling back to dense** on a mismatch or under a `jit`/`grad` trace
  (the probe needs concrete arrays). To stay leak-free it builds the pattern from
  a **cached-recycle-map** forward rhs (the per-call recycle probing in
  `_rhs(recycle_map=None)` leaks a traced intermediate under the pattern-probe
  `jit` on a multi-network plant); this cached-map rhs (`primal_rhs`) is also used
  for the **forward iteration** (identical result, faster), while the one-shot
  implicit-function-theorem *gradient* keeps the **map-recomputing** rhs so a
  flow-setpoint parameter retains its `d(map)/d(param)` term (the #366 split).
  **PTC is a better fit for coloring than the dynamic solve**: it marches to a
  single operating point in a narrow neighbourhood, so the start-state sparsity
  pattern stays valid throughout — unlike the 609-day dynamic run's wide load
  excursion. **Measured (BSM2, 167 states / 46 colors):** the per-iteration
  Jacobian build is **2.4× cheaper** (0.62 → 0.26 ms) and the whole PTC solve,
  **run-only under `jit`, is 1.87× faster** (58 → 31 ms). **But the un-jitted
  one-shot `steady_state` call is compile/trace-bound, not Jacobian-build-bound**
  (the `while_loop` re-traces per call), so the run-phase saving is invisible
  there and the one-time pattern build (~49 dense probes) makes a single
  `steady_state(colored_jacobian=True)` call *slower* (~0.8×). The win therefore
  materializes only when the solve is run **repeatedly under `jit`** (differentiable
  design sweeps / optimization loops, where compilation is amortized) or for a
  much larger plant. The implicit-function-theorem *gradient* Jacobian stays
  dense (a single evaluation). **Default off; opt in for the jitted/amortized
  regime, not for a one-shot steady state.**
- **Carrying the RHS across PTC iterations — TESTED AND REJECTED (no benefit).**
  The PTC `step` (`plant/steady.py`) evaluates the RHS twice per iteration —
  `Fy = F(y)` at the top (for the linear solve) and `F(y_new)` for the residual —
  and the next iteration's `F(y)` *is* the previous iteration's `F(y_new)`. The
  obvious optimization is to carry the `F` vector in the `while_loop` carry so the
  RHS is evaluated once per iteration (each eval includes the recycle and pH
  solves, so on paper the saving looks real). **It does not help:** the result is
  bit-identical but the BSM2 run-only (jitted) time is **neutral-to-slightly-worse**
  (~43 → ~44 ms, measured min-of-8). The reason is that `jac = jax.jacfwd(F)`
  computes `F(y)` as its forward-mode **primal** on the same `y`, so **XLA already
  CSE-eliminates the redundant top-level `F(y)`** against the Jacobian's primal
  pass; removing it by hand saves nothing and threading the extra `n`-vector
  through the loop carry adds a hair of overhead. **Lesson:** redundant RHS
  *evaluations* in the PTC loop are the compiler's job — it fuses them. Real PTC
  speedups must cut *distinct* work the compiler cannot share across iterations:
  the per-iteration Jacobian materialization (the `colored_jacobian` builder, or
  freezing/reusing `J` for several steps) or the **iteration count** itself
  (better pseudo-time / residual scaling, a line search). Do not re-attempt the
  carry-`F` micro-optimization.

**Default `atol` is now per-component, scaled to the state magnitudes.** When
`atol` is omitted, **every single-concentration-vector reactor**
(`BatchReactor`/`PlugFlowReactor`/`ParticleTrackReactor`/`CFDReactor`, via the
shared `integrate/_common.resolve_state_atol`) and `Plant.solve` build a
per-species noise floor
`atol_i = atol_factor·max(|operating_i|, |reference_i|, floor_frac·char)`
(`atol_factor=floor_frac=1e-6`) via `integrate/_common.default_atol` — the
SUNDIALS "vector atol" / Hairer "atol ∝ typical value" rule. The reactors scale
off the network's `default_concentrations` (at construction); the plant scales
off `y0` (at solve time). **`default_atol` `stop_gradient`s its result** — the
tolerance is a solver noise floor, never a differentiated quantity. This matters
because the plant scales off `y0`: under a gradient **with respect to `y0`** the
floor would otherwise be a traced array, get baked into the integrator's step
controller, and — inside the discrete-adjoint (`gradient="stable_adjoint"`)
custom-VJP forward (which re-runs `diffrax.diffeqsolve`) — escape that inner solve
as a leaked tracer (`UnexpectedTracerError`, issue #420). Detaching it is the
identity for the value (so every steady state is unchanged) and lets the
**initial-state gradient** flow through `stable_adjoint`, the one direction the
standard `jax_adjoint` already handled. (The leak was specific to the
state-derived tolerance; the param-gradient direction was always fine, since the
tolerance does not depend on the parameters.) When **every** magnitude is zero (an all-zero
`scale_like` with no reference) the relative floor `floor_frac·char` would
itself be 0, so `char` falls back to unit scale — keeping every `atol_i`
strictly positive rather than 0 (the very invariant this floor upholds). This
fallback is identity for any input with a nonzero magnitude (the common path). `BiofilmReactor` is the exception — its multi-
compartment `(n_layers+1, n_species)` state does not match the per-species
vector, so it keeps an explicit scalar `atol` (default `1e-9`). This replaces
the old fixed `atol=1e-9`, which was ~9 orders too tight for g/m³ ASM/ADM states
and forced the integrator step ceiling — so a warm-started BSM2 now solves with
**nothing passed** (no `atol=1e-3, max_steps=500_000` magic). An explicit scalar
or `(n_species,)` array still overrides it verbatim (e.g. the ozone `OH→1e-20`
per-species atol), so existing calls are unchanged. Verified to reproduce every validated steady
state (691 non-validation + 23 validation tests). Any solve that hits the
integrator step budget -- `Plant.solve` **and every reactor**
(`BatchReactor`/`PlugFlowReactor`/`BiofilmReactor`/`ParticleTrackReactor`) --
re-raises the Diffrax/Equinox failure as a domain `RuntimeError` naming the
remedies (warm-start via `run_to_steady_state`, loosen `rtol`, raise
`max_steps`), with the noisy equinox exception chain suppressed (`from None`).
This is the shared `integrate/_common.friendly_solve_errors(max_steps, what=...)`
context manager wrapped around each solve's *execution* (the call to the jitted
solve / `diffeqsolve`, where the runtime error surfaces -- not the traced
`_run_diffeqsolve`). The jitted reactors emit one extra equinox *stderr* line
about `filter_jit` that the exception machinery cannot suppress; the raised
exception itself is clean. The **same** context manager also catches the other
opaque solve-time failure — a `jax.jacfwd`/`jax.jvp` (forward-mode AD) through
the default reverse-only adjoint, which JAX rejects with "can't apply
forward-mode autodiff (jvp) to a custom_vjp function". That is re-raised as a
`RuntimeError` naming the cure, `aquakin.forward_adjoint()` (build the reactor
with that adjoint, or take a reverse-mode gradient). `sensitivity`/`dgsm` with
`ad_mode="forward"` set the forward-capable adjoint for you and so never hit it.

**Dynamic influent now works too — flow-controlled recycle pumps (issue #30).**
The dynamic (time-varying-influent) run *used* to hit the step ceiling, which
was attributed to diurnal-forcing stiffness. The real cause was a **flow-model
bug**: the recycle streams (internal recycle `Qa`, RAS `Qr`, wastage `Qw`) were
modelled as fixed-*fraction* `SplitterUnit`s and the clarifier effluent as a
fixed flow — constants calibrated only at the design influent `Q_avg`. The
recycle-flow algebra then has a near-singular gain
(`tank5_throughput = (Q_fresh − 17693)/0.00816`), so a ±10% influent swing whips
the throughput from 5× to ~23× `Q_in`; that violently amplified, fast-varying
flow field is what made the monolithic solve crawl. `_resolve_flows` was exact —
it was faithfully resolving the *wrong* flow model — which is why it stayed
hidden at steady state (sitting exactly at `Q_avg`, where the fractions are
correct). The BSM1/BSM2 reference (`asm1init_bsm2.m`) settles it: `Qa = 3·Qin0`,
`Qr = Qin0`, `Qw = 300` are **constant pumped flows** off a fixed reference flow,
and the settler computes the effluent as the *free remainder*
(`Q_e = Q_f − (Q_r + Q_w)`). The fix mirrors this: `SplitterUnit` gains a
fixed-setpoint *flow mode* (`output_port_flows` + `remainder_port`); the
clarifiers gain a fixed `underflow_Q` (= `Qr + Qw`) with the overflow as the
remainder; `build_bsm1` wires the recycles as constant pumps. Throughput now
holds ~5× `Q_in` under any influent, and the 14-day dry run integrates in ~5k
steps (Ideal, ~10 s) / ~18k steps (Takács, ~30 s) to a healthy state —
**~1000× fewer steps**, steady state unchanged. Regression-guarded by
`test_bsm1_dry_weather_runs` and `test_bsm1_takacs_dry_weather_runs`.

The first plant-wide demonstration target is **BSM1** (Copp 2002 / Alex
2008) — built by `aquakin.plant.bsm.build_bsm1()`. Three synthesised
influent CSVs (dry / rain / storm) ship under
`aquakin/plant/bsm/data/` and load via `load_bsm1_influent()`. The
synthesised files match BSM1's *statistical* profile but are not the
canonical IWA files; for quantitative comparison to Alex 2008's
published EQI / OCI values, users should replace them with the
official files.

**A²O biological-nutrient-removal plant (`aquakin.plant.build_a2o`,
[`plant/a2o.py`](aquakin/plant/a2o.py)).** The first phosphorus-capable
flowsheet — the BSM plants run the P-free ASM1, so they cannot host bio-P or the
chemical-P (metal-salt dosing) demonstration. `build_a2o` is the canonical
**Anaerobic–Anoxic–Oxic** layout on the shipped `asm2d` network: an anaerobic
selector (where PAOs release phosphate and store fermentation products) → anoxic
denitrification → aerated nitrification + luxury P uptake → secondary clarifier,
with the mixed-liquor internal recycle (aerobic→anoxic) and RAS
(underflow→anaerobic) closing the loop, so it removes carbon, nitrogen **and**
phosphorus in one plant. `a2o_influent(net)` is a matching constant municipal
(VFA-bearing) influent and `a2o_warm_start(plant)` seeds the AS reactors with an
established EBPR mixed liquor (a large PAO population + stored poly-P), so a solve
starts from healthy bio-P sludge rather than the slow, seed-sensitive cold-start
PAO establishment. The default config reaches a feasible steady state with
**complete biological P removal** (effluent SPO4 ≈ 0) and ~80% N removal (full
nitrification of the influent ammonia + denitrification), with no recirculating
negative soluble pools (it relies on the `asm2d` `positivity_limiter`, now
honoured inside `CSTRUnit`). It is **not** a standardised benchmark — the sizing
is a representative municipal design, not a published reference set, so it is a
worked nutrient-removal flowsheet, not a validation target. Building it is what
surfaced the `asm2d` process-matrix import errors (see the `asm2d` network note);
the A²O viability test (`tests/integration/test_a2o.py`) is the regression guard
the COD/N/P continuity suite could not be (each broken coefficient still
conserves mass). It is the substrate for the chemical-P (ferric/alum dosing)
demonstration.

**BSM2 — open-loop plant (Gernaey et al. 2014 / Jeppsson et al. 2007).**
`aquakin.plant.bsm.build_bsm2()` wraps the BSM1 activated-sludge core with the
full sludge train: a **primary clarifier** ahead of the reactors, and
downstream a **thickener**, an **ADM1 anaerobic digester** (35 °C, 3400 m³
liquid + headspace) with the **ASM1↔ADM1 interfaces**, and a **dewatering**
unit, with the two reject-water streams (thickener overflow + dewatering reject)
recycled to the plant front. This is a genuinely **two-network** plant (ASM1
water line + ADM1 digester); the interfaces ride on the cross-network
connections as `StateTranslator`s, so the whole thing still integrates under one
monolithic Diffrax solve with `jax.grad` flowing end to end. All controlled
flows (internal recycle `Qintr=3·Q_ref`, RAS `Qr=Q_ref`, wastage `Qw=300`,
primary sludge `f_PS·Q`) are fixed-flow pumps (the BSM1 flow-control fix carries
over); the thickener/dewatering underflows are concentration-dependent but sit
on the low-gain reject loop, which the concentration sweep resolves (their
`flow_outputs` seed the linear pre-solve with a nominal fraction). A constant
**external carbon dose** (2 m³/d of readily-biodegradable COD to reactor 1,
`carbon_flow`/`carbon_conc`, BSM2 default on) feeds denitrification in the
anoxic tanks. **It reaches a healthy open-loop steady state in ~20 s** —
nitrifying AS, biomass sustained, and a methanogenic digester.

**Optional features are configured with option objects, and the entry/exit
endpoints are exposed (so callers never hard-code a port).** The optional BSM2
features used to be a dozen cross-coupled boolean/float flags; they are now small
**frozen option objects** (`aquakin.plant.bsm`), one per feature, passed to
`build_bsm2` — present ⇒ enabled, `None` ⇒ off: `ExternalCarbon(flow, conc)`
(`carbon=`, default-on; `carbon=None` disables), `RejectStorage(volume,
output_flow, control)` (`reject=`, with `control=True` for the closed-loop level
controller), `InfluentBypass(threshold)` (`bypass=`), `HydraulicDelay(tau)`
(`hydraulic_delay=`). `do_control: bool` and `wastage_schedule` stay as they were
(a lone toggle / an already-an-object schedule). Some features move the front
ports (the bypass relocates the entry to `bypass_split.in` and the effluent to
`effluent_mix.out`; the hydraulic delay relocates the entry to
`influent_delay.in`) — so `build_bsm2`/`build_bsm1` record the canonical ports on
the generic **`Plant.influent_endpoint`** / **`Plant.effluent_endpoint`**
attributes. `plant.add_influent("feed", series)` defaults its `to=` to
`plant.influent_endpoint`, and `evaluate_bsm2` / `sludge_metrics` default their
effluent port to `plant.effluent_endpoint`, so feature flags can no longer
silently mis-wire the influent or score the wrong effluent. (The endpoints are
plain optional attributes on every `Plant`, `None` unless a builder sets them.)

**Quantitatively validated** against the published BSM2 open-loop steady state
(`tests/validation/test_bsm2_steadystate.py`): run with the published constant
influent (`bsm2_constant_influent`) and the BSM2 (15 °C) ASM1 parameter set
(`bsm2_parameters`), the whole multi-network plant — the 5 AS reactors, the
secondary settler, the primary clarifier, both ASM1↔ADM1 interfaces, the
digester, and all recycle loops including the reject water — reproduces the
reference reactor states (`asm1init_bsm2` `XINIT`: XB_H ≈ 2245, XB_A ≈ 167,
XP ≈ 967, XI ≈ 1532, the SNH/SNO/SO profiles) **to round-off (≤0.06% on every AS
state — the level at which the reference ring-test simulators agree with one
another)** and the digester (`DIGESTERINIT`: headspace methane to ~0.2%) **to
within ~1.3% (worst: headspace CO₂, the charge-balance-pH vs algebraic-pH
difference in the gas phase)**. **Reaching the round-off AS match needs the
benchmark operating temperature, not just the 15 °C parameters.** The ASM1 rates
are defined at 15 °C and Arrhenius-corrected to each reactor's (flow-weighted)
*inlet* temperature; the BSM2 constant influent enters at **14.858 °C**
(`BSM2_CONSTANT_INFLUENT_T`, the `constinfluent` T column = the annual mean), so
the AS line operates 0.14 °C below the reference and every rate is slowed ~1.4%.
Omitting this (running the line at the bare 15 °C reference) over-predicts
nitrification by ~1.4% — the entire otherwise-residual deviation (SNH/SNO drift
~1–1.5%). `bsm2_constant_influent` therefore takes a `T=` argument: pass
`T=BSM2_CONSTANT_INFLUENT_T` **together with `bsm2_asm1_network()`** (the 15 °C-
referenced corrections) for the faithful match. The default `T=None` keeps the
historic temperature-agnostic behaviour (reactors fall back to their static 15 °C
condition); do **not** pass `T` with the plain 20 °C `load_network("asm1")` — a
14.858 °C inlet on a 20 °C-referenced network applies a large spurious slowdown
(~40% on nitrification). aquakin carries the reactor temperature *algebraically*
(the flow-weighted inlet each RHS, resolved with the recycle solve), not as a
BSM2-style heat-balance state `dT/dt=(Q/V)(T_in−T)`; the two agree at steady
state (both give T=T_in) and differ only by the (sub-hour) thermal lag in
transient. Two parameter
reconciliations were needed: the BSM2 ASM1 values are the 15 °C set
(`muH=4, KS=10, muA=0.5, bH=0.3, KX=0.1, etah=0.8`). (The shipped `asm1` is the
textbook Gujer matrix with no heterotroph ammonia-limitation term, so — unlike
earlier versions — no neutralising override is needed; for the BioWin/SUMO
nutrient switch use the `asm1_ammonia_limitation` network, where that term
suppresses tank-5 growth ~24% and roughly halves XB_H.) ASM1 has no Arrhenius T-dependence
(the `T` condition is declared but unused), so only the parameter *values*
matter, not the 15 °C operating temperature.

**Dynamic influent runs too.** Synthesised BSM2 dry / rain / storm influent
files (`scripts/generate_bsm2_influent.py` → `aquakin/plant/bsm/data/BSM2_*.csv`,
loaded by `load_bsm2_influent()`) drive the plant under diurnal + wet-weather
forcing. The fixed-flow-pump fix carries straight over to BSM2 scale: warm-started
from steady state, the 167-state two-network plant integrates a 14-day dynamic
run **efficiently** (~140 steps/day, not a step-ceiling blow-up) to a finite,
healthy trajectory, and a rain event doubling the influent stays bounded because
the recycle pumps hold throughput at `Q_in + Qintr + Qr`
(`tests/integration/test_bsm2_dynamic.py`). The shipped influent CSVs are
**synthesised**, not the canonical 609-day IWA series, so the dynamic tests
assert qualitative stability, not published dynamic metrics.

**Temperature handling is a selectable `TemperatureModel`
([`plant/temperature.py`](aquakin/plant/temperature.py)).** Two strategies, set on
the plant (`plant.set_temperature_model(...)`, or `build_bsm2(temperature_model=
...)`); exported at the top level (`aquakin.TemperatureModel` /
`AlgebraicTemperature` / `HeatBalanceTemperature`):
- **`AlgebraicTemperature`** (default) — temperature is *instantaneous*: each unit
  flow-weights its inlet `T` (a heat balance) and passes it through, so a reactor
  runs its kinetics at its flow-weighted inlet temperature, with **no thermal
  storage**. Carries **zero** extra state and is a pure no-op (every existing
  plant and validated steady state is byte-for-byte unchanged). This is the
  historic behaviour, described in the rest of this section.
- **`HeatBalanceTemperature`** — every finite-volume liquid unit (one exposing a
  positive `volume`) that is not temperature-fixed carries its temperature as a
  **dynamic state** with the completely-mixed first-order balance
  `V dT/dt = Q_in (T_in − T)`; the heated digester sets `temperature_fixed = True`
  and stays pinned (the BSM2-protocol treatment, Jeppsson et al. 2007). The
  reactor then runs at this **lagged tank temperature**, so it damps/lags the
  influent (important because recycles trap heat — the effective AS time constant
  `V_total/Q_fresh` is hours, comparable to diurnal forcing — which the algebraic
  model cannot represent). For BSM2 it tracks the 5 reactors + primary clarifier +
  settler (the `TakacsClarifier` exposes a `volume = area·height` for this). The
  temperature states are appended as one block at the **tail** of the flat plant
  state vector (the `FlowSetpoint` tail-append pattern, but for state), so every
  per-unit state slice keeps its index (warm-starts / `states_by_unit` unaffected);
  `Plant._split_state` exposes the block under a reserved key, `_sweep_outputs`
  overrides each tracked unit's outlet `T` with its state (so the lag propagates
  through the exact recycle-temperature solve), and the reactor reads its operating
  temperature from a reserved control-signal key (`OPERATING_T_SIGNAL`), falling
  back to the flow-weighted inlet T when absent. At a constant influent temperature
  the heat-balance fixed point IS the influent temperature, so it reproduces the
  algebraic steady state. Tested in
  `tests/integration/test_temperature_model.py` (tracked set, the first-order
  balance + `V/Q` time constant, the constant-influent fixed point, AD through the
  state). *(Motivation: investigating the ~16% effluent-S_NH gap in the dynamic
  BSM2 vs the ring-test consensus — the algebraic and heat-balance reactor
  temperatures are equal to ≤0.1 °C across the AS line because the lag averages
  out over a seasonal window, so this is for transient-temperature fidelity, not a
  fix for that gap.)*

The default-model behaviour: temperature is carried *algebraically* through the
flowsheet: `Stream` and `InfluentSeries` have an optional `T` (Kelvin); mixers
flow-weight it (a heat balance) and every other unit passes it through, so a
reactor reads its (flow-weighted) inlet temperature and feeds it to the ASM1
temperature corrections. `T=None` is the default and a static structural
property — a temperature-agnostic influent leaves every stream `T=None` and the
reactors fall back to their static condition, so existing plants are unchanged.
The single heat-balance rule every multi-inlet unit (mixer, CSTR, clarifier,
digester) uses is `streams.mixed_temperature(inputs, names)`: it flow-weights
only the inlets that carry a temperature and *ignores* a `T=None` inlet rather
than letting one collapse the whole mix to `None`. This is what lets a
temperature-carrying influent propagate around a recycle loop whose back-edge is
auto-seeded with a zero-flow, temperature-agnostic stream (the seed contributes
nothing and is ignored); earlier the `all(inlet.T is not None)` gate meant one
agnostic seed disabled temperature around the loop, so `build_bsm2` had to
hand-seed its recycles with a nominal `T` (now redundant — kept only as an
explicit warm start). The helper is also zero-flow-safe: if every
temperature-carrying inlet is momentarily at zero flow it returns their mean
rather than dividing by the flow epsilon (which would drive the result toward
0 K and feed a garbage value into the Arrhenius correction). For BSM2 the AS
reactors run at 15 °C:
`bsm2_asm1_network()` re-references the ASM1 temperature corrections from 20 °C
to 15 °C (keeping the BSM2 slopes), so with `bsm2_parameters` (the 15 °C values)
the correction is unity at 15 °C — a constant-15 °C run reproduces the validated
steady state exactly — and a temperature-carrying influent drives it away:
colder water nitrifies more slowly (higher residual ammonia), warmer faster
(`tests/integration/test_bsm2_seasonal.py`). `build_bsm2()` now **defaults** its
ASM1 network to `bsm2_asm1_network()` (the 15 °C reference), so the out-of-the-box
plant is the BSM2 calibration; pass the plain `load_network("asm1")` explicitly to
get the 20 °C reference. When you build the influent yourself, reuse the **same
network instance** for both `build_bsm2` and the influent so their identities match
(a clear error fires otherwise). The
synthesised BSM2 influent CSVs carry a time-varying temperature column (`T`, in
°C; a shoulder-season ~12→18 °C ramp + diurnal ripple), which
`load_bsm2_influent` returns as `InfluentSeries.T` in **Kelvin** — so a dynamic
run on `load_bsm2_influent(...)` is seasonally temperature-driven out of the box.
(The generic `read_influent_csv` / `_influent_from_text` capture a `T` column
when present, in the file's own units; only the BSM2 loader converts °C→K.)

**Temperature-dependent oxygen transfer (issue #206).** By default the aeration
term is `kLa·(C_sat − C)` with `C_sat` a fixed constant (8.0 gO₂/m³) and `kLa`
constant — the literal IWA benchmark definition. That left a seasonal-run
inconsistency: a warm influent already speeds the (Arrhenius) biology while the
oxygen driving force stayed pinned. `Aeration` now carries **opt-in** transfer
corrections, all identity by default so the benchmark stays bit-faithful:
`temperature_correction=True` scales the saturation by the clean-water ratio
`C_s(T)/C_s(ref_T)` (the Benson–Krause `aquakin.plant.oxygen_saturation`,
~9.09 mg/L at 20 °C → ~7.56 at 30 °C) and the **open-loop** `kLa` by
`kla_theta**(T−ref_T)` (default θ=1.024), using the same flow-weighted inlet `T`
the kinetics use (falling back to the static `T` condition); a closed-loop
controlled `kLa` is **not** θ-scaled (the controller already manipulates it) but
its driving-force saturation still gets the `C_s(T)` correction. Constant factors
`alpha` (kLa transfer fouling), `beta` (salinity) and `pressure_factor`
(elevation) fold into the precomputed vectors at construction (defaults 1.0). All
AD-clean (the correction is a smooth function of `T` inside the monolithic plant
solve). `build_bsm2(do_temperature_correction=True)` turns it on plant-wide with
`ref_T` = the reactors' static temperature (so it is unity at the benchmark
operating point and only a temperature-carrying influent drives it); default off
reproduces the validated steady state exactly. The saturation curve used for the
`C_s(T)/C_s(ref_T)` ratio is selectable via `Aeration(saturation_model=...)`:
`"benson_krause"` (default, the APHA `oxygen_saturation`) or `"bsm2"` (the IWA
benchmark van't Hoff `oxygen_saturation_bsm2`, normalised to 8.0 mg/L at 15 °C);
the two differ by ~0.5 % in shape. `build_bsm2(do_temperature_correction=True)`
uses `"bsm2"` so the seasonal oxygen driving force matches the benchmark exactly.

**Diffuser / blower aeration-design physics (issue #279,
[`plant/aeration_system.py`](aquakin/plant/aeration_system.py)).** The kinetic
model aerates through a per-species `kLa`, and the Copp-2002 OCI scores aeration
*energy* with the fixed correlation `AE ∝ Σ V_i·kLa_i`. `AerationSystem` is the
blower/diffuser physics behind that `kLa` — how much **air** must be blown and the
**power** to compress it — kept **standalone** (it does **not** change the `kLa`
interface). From the `kLa` a solve produced it computes the standard oxygen
transfer rate `SOTR = kLa·C_s,std·V` (the clean-water transfer the airflow must
deliver — a given `kLa` needs a given airflow, independent of the operating DO
deficit), the **air flow** `Q_air = SOTR/(SOTE·o2_per_air)` from the diffuser's
standard transfer efficiency `SOTE` (rising with submergence, default `6 %/m`,
reduced by a fouling factor `F`), the blower **discharge pressure**
`p_atm + ρ_w·g·depth + headloss`, and the blower **power** by adiabatic
compression `P = (Q·p1/η)·(γ/(γ−1))·[(p2/p1)^((γ−1)/γ) − 1]`. Because the power is
linear in airflow and airflow is linear in `kLa`, `blower_energy(t, kla_history,
volumes, system)` has the same form as the Copp kernel but with a mechanistic
coefficient (SOTE/depth/blower curve) in place of the fixed one, and stays
`jit`/`grad`-clean (the differentiable primitives are `required_airflow` /
`blower_power_kw`; the float-returning `blower_energy` is the reporting kernel, the
drop-in for `aeration_energy`). The α/β/temperature *field* corrections stay on
`Aeration` (they shape the `kLa` and driving force in the solve); `AerationSystem`
adds the diffuser-fouling `F` and the blower curve. **Wired into the evaluators:**
`evaluate_bsm1(..., aeration_system=AerationSystem(...))` and `evaluate_bsm2(...,
aeration_system=...)` **replace** the correlation AE with the mechanistic blower
energy (flowing into the OCI and, via `total_energy()`, the GHG/cost report) and
expose `air_flow` (m³/d) on the evaluation; `aeration_system=None` (default) keeps
the validated Copp AE, so the benchmark numbers are unchanged. `design_summary(kla,
volume, system)` is the standalone sizing entry point → an `AerationDesignPoint`
(SOTE / SOTR / airflow / discharge pressure / power) with a labeled `report()`.
Covered by `tests/integration/test_aeration_system.py` (physics vs closed form,
SOTE/depth/fouling, validation, AD) and the evaluator wiring in
`tests/integration/test_bsm2_evaluation.py`.

**Influent characterization + CSV `column_map` (issue #136).** Real influent is
measured as aggregates (total COD, TKN, ammonia, alkalinity, optionally
filtered/flocculated COD), not as the 13 ASM1 states. `aquakin/plant/characterize.py`
maps them: `fractionate(total_cod=, tkn=, ...) -> {ASM1 state: value}` follows the
**SUMO Sumo1 raw-influent fractionation reduced to ASM1** — COD split by
filtration (soluble/colloidal/particulate) then biodegradability, reduced to ASM1
by lumping colloidal-biodegradable into `XS` and colloidal/soluble-inert into
`XI`/`SI` (`SI=SU, SS=SB, XI=CU+XU, XS=CB+XB, XB_H=XOHO, XP=XE, XB_A=0`); N gives
`SNH` (ammonia or `f_snh·TKN`), `SND` (soluble-biodeg N), `XND` (TKN-balance
remainder using ASM1's `i_XB`/`i_XP`); alkalinity mg CaCO₃/L → `SALK` mol/m³ via
`/50`. A measured `filtered_cod`/`flocculated_filtered_cod`/`soluble_inert_cod`
drives its split; absent, the SUMO default fraction (`InfluentFractions`, the
Sumo1 tool's municipal values) is used. The reduction **conserves total COD**
(`Σ COD states = total_cod`) and closes the ASM1 TKN balance. `fractionate` is
plain arithmetic, so it runs element-wise on scalars **or arrays** — the per-row
path. `characterize_influent(network, flow=, total_cod=, ...)` wraps it into a
constant `InfluentSeries`. `read_influent_csv(..., column_map={role: header})`
loads an **arbitrary-header** CSV (a lab/SCADA export — no renaming): roles are
`t`/`Q`/`T`, any ASM species (mapped directly), and the aggregate names; mapped
aggregates are fractionated **per row** (a directly-mapped species overrides its
fractionated value; unmapped species default to zero). Validated against the
spreadsheet's worked example (`tests/integration/test_characterize.py`). Exported
as `aquakin.characterize_influent` / `fractionate` / `InfluentFractions` /
`read_influent_csv`.

**`Plant.set_temperature(celsius)` — one knob for the operating temperature.**
Setting a plant's temperature used to mean writing the static `T` condition of
every reactor by hand (in Kelvin, at the correction `ref_T`). `set_temperature`
takes **°C**, converts to Kelvin, and writes the static `T` of every
temperature-bearing reactor — so a re-solve runs the Arrhenius
`temperature_corrections` at that temperature (`build_bsm2(...)` then
`plant.set_temperature(15)` is the BSM2 15 °C operating point; `set_temperature(10)`
drives nitrification down — verified in
`tests/integration/test_plant_temperature.py`). It targets the activated-sludge
reactors (`CSTRUnit`s exposing `set_temperature` with a `T` condition) and
**leaves the heated anaerobic digester untouched** (a fixed-`T` ADM1 unit without
the method); pass `units=[...]` to target a specific set. It clears the plant's
compiled-solve cache (`_jit_cache`) so the next solve recompiles at the new
temperature, and returns `self` for chaining after `build_*`. The per-unit
mechanic is `CSTRUnit.set_temperature(temperature_K)` (updates `conditions["T"]`
and its precomputed condition array).

**Clear error on an influent/plant network-instance mismatch.** The seasonal
footgun was using *different instances* of the same ASM1 model for the plant and
the influent (e.g. calling `bsm2_asm1_network()` twice): their temperature
corrections / parameters then silently disagree. `Plant._default_translator` now
distinguishes this from a genuine cross-network edge — when the two networks have
the same `name` and `species` but are different objects, the error says to *build
the network once and pass that same object to both* (rather than the old, here
misleading, "supply an explicit translator"); a truly different model still gets
the translator message.

**Closed-loop DO/kLa control (`build_bsm2(do_control=True)`).** The first
closed-loop element is the BSM2 dissolved-oxygen controller: a PI loop senses
`SO` in reactor 4 and manipulates its aeration `kLa` (reactors 3 and 5 scale off
the same signal at gains 1.0/0.5), driving the oxygen to the `SO=2` gO₂/m³
setpoint instead of the fixed open-loop `kLa`. Tuning is the reference DO loop
(`Kp=25`, integral time `Ti=0.002` d, anti-windup tracking `Tt=0.001` d, `kLa`
offset 120 d⁻¹, bounded `[0, 360]`). It is built on a small, general
**control-signal bus** layered on the material flowsheet (so the loop closes
inside the one monolithic Diffrax solve and `jax.grad` still flows end to end):
- `PIController` ([`plant/control.py`](aquakin/plant/control.py)) is a Unit with
  one integral state. It reads its measured variable from a *sensed input
  stream* (wired like any other connection, `tank4 → do_control.measured`, but it
  produces no material output), and publishes a named scalar **signal**
  `u_sat = clip(offset + Kp·e + x_i, out_min, out_max)` via `signal_outputs(...)`;
  its `rhs` integrates `dx_i/dt = (Kp/Ti)·e + (1/Tt)·(u_sat − u)` (back-calculation
  anti-windup). `x_i` is the integral *contribution to the output* (already
  scaled), so the tracking term has consistent units.
- `Plant._rhs` evaluates `signal_outputs` on every controller each RHS call,
  gathers the results into a `signals` dict, and threads it into **every** unit's
  `compute_outputs` *and* `rhs` as the trailing `signals` argument (a unit that
  reads no signals simply ignores it). Producing signals is the one optional,
  class-level/duck-typed hook (`hasattr(unit, "signal_outputs")`), so the branch
  is static and jit/AD-safe. **The bus is computed from the reactor states
  *before* the stream sweep** (`_compute_signals`): a controller senses a
  reactor's concentration, which *is* that unit's state, so the sensed value is
  read directly from `states` (the controller's sensed `inputs` stream is
  reconstructed as `C = states[sensor]`, so the sensor must be a reactor whose
  output concentration is its state, i.e. a `CSTRUnit`). Computing signals first
  is what lets a unit whose *output stream* depends on a signal — a
  feedback-`DosingUnit` — read it in `compute_outputs` (the sweep), which runs
  before any post-sweep quantity. For a DO controller sensing a CSTR the value is
  identical to the old post-sweep read (state == output `C`), so aeration is
  unchanged.
- **`Aeration` on `CSTRUnit` (issue #137).** A tank's aeration is set with one
  `aeration=Aeration(...)` object, not raw per-species `kla`/`C_sat`/`controlled_kla`
  dicts (those fields are gone — pre-release, single interface). `Aeration` has two
  modes: open loop `Aeration(kla=120, do_sat=8)` (a fixed mass-transfer
  coefficient; `do_sat` defaults to 8.0), and closed loop `Aeration(do_setpoint=2.0)`
  (a DO target). `CSTRUnit.__post_init__` translates the spec into the internal
  `_kla_vec`/`_sat_vec`/`_controlled_kla` the `rhs` uses, so the aeration term
  `kLa·(do_sat - C)` is unchanged. For closed loop the species' `kLa` is taken from
  `signals[name]·gain` each step (overriding the fixed `kLa`); the signal name is
  derived deterministically from the controller id.
- **Auto-wired DO controllers (`Plant._materialize_aeration`).** A closed-loop
  `Aeration` consumes a kLa signal but does not itself add the controller. The
  plant materialises it: at topology setup (once, before the state layout) it
  groups the closed-loop tanks by their aeration `controller` id — the shared-
  controller case (BSM2: one sensor on `tank4`, per-tank `gain`s) — or, when no id
  is given, gives each tank its own controller (per-tank DO control, `sensor`
  defaults to the tank). One `PIController` per group is added (named after the
  shared id, or `<tank>_aeration`), sensing the group's `sensor`, with the
  setpoint/PI tuning/bounds from the `Aeration` (defaults are the BSM2 DO loop).
  Tanks sharing a controller must agree on its setpoint/sensor/tuning; only `gain`
  differs. `build_bsm2(do_control=True)` now expresses the loop purely as
  `Aeration(do_setpoint=2.0, controller="do_control", sensor="tank4", gain=...)`
  on the reactors — the controller and its sensor tap are auto-wired (the manual
  `PIController` + `connect` are gone). `build_bsm1`/`build_bsm2` open-loop tanks
  use `Aeration(kla=..., do_sat=8)`.
- **Assembly-time signal validation.** A unit declares the bus names it reads via
  `required_signals` (`CSTRUnit` derives it from its closed-loop aeration) and the
  names it publishes via `signal_names` (`PIController` -> its `signal_name`).
  `Plant._validate_control_signals` (run from `_build_state_layout`, before the
  RHS is traced) checks every consumed name is published, so a forgotten/mistyped
  controller signal raises a clear `ValueError` naming the unit and the available
  signals -- not a bare `KeyError` from deep in the first jitted solve. It is
  conservative: if any producer (a unit exposing `signal_outputs`) does not
  declare `signal_names`, the published set is unknown and validation is skipped.
Covered by `tests/integration/test_bsm2_control.py` (controller-unit behaviour:
signal sign, saturation, integral direction, anti-windup; closed-loop setpoint
tracking; closed-vs-open contrast; `jax.grad` through the closed loop). The
digester is additionally validated at the unit level in
`tests/validation/test_bsm2_digester_unit.py`.

**Chemical dosing (`DosingUnit` / `Reagent`, issue #278).** A general inline
dosing unit (`aquakin/plant/dosing.py`): `in` stream → `out` = inlet + reagent
dose, flow-mixing the compositions. A `Reagent` is a value object — a fixed
composition vector built by name (`Reagent.from_species(asm1, SS=4e5,
label="methanol")`, base zero, so the neat reagent contains only what you name) —
covering metal salts, acid/base, and external carbon. The dose flow is either
**fixed** (`DosingUnit(name, reagent, flow=2.0)`) or **feedback-controlled**
(`DosingUnit(name, reagent, setpoint=1.0, measured_species="SNO", sensor="anoxic",
flow_max=...)`): a feedback dose declares a sensed reactor + species + setpoint,
and the plant auto-wires a `PIController` (`Plant._materialize_dosing`, the dosing
analogue of `_materialize_aeration`, reusing the same controller) that
manipulates the dose flow to hold the setpoint, publishing a `_dose_<id>_flow`
signal the unit consumes. The unit is **stateless** — a fixed dose needs no
state, a feedback dose's PI integral lives in the controller — and reads its
dose-flow signal in `compute_outputs` (the dose changes the *output stream*),
which is why the signal bus is computed before the sweep (above); `flow_outputs`
seeds the recycle-flow solve with the nominal `flow_offset` for a feedback dose
(the exact, concentration-dependent flow is applied in `compute_outputs`, the
same convention the separators' concentration-dependent flows use). `build_bsm2`
now expresses its external-carbon feed as a fixed `DosingUnit` on the
`as_mix → tank1` line (the former hard-coded carbon influent is gone); the
validated steady state is unchanged (same carbon mass, same tank-1 inlet). The
dose only adds the reagent's *mass*; the **reactive** response — an acid/base's
pH shift, metal-phosphate precipitation, the added COD's oxygen demand — is the
downstream reactor's chemistry (the precipitation/pH engine, issue #271), not
this unit's job. Covered by `tests/integration/test_dosing.py`.

**Disinfection unit ops (`UVUnit` / `ChlorineContactUnit`, issue #280,
[`plant/disinfection.py`](aquakin/plant/disinfection.py)).** The `uv_h2o2` /
`ozone_bromate` *networks* model the oxidation chemistry, but neither is a
disinfection *unit op* that reduces a pathogen indicator in the flowsheet. These
two add that, matching the commercial simulators (GPS-X / SUMO track an indicator
organism + the disinfectant residual and apply a dose/CT log-removal). Both
**pass the process (ASM) stream through unchanged** (disinfection does not
materially change COD/N/P at this fidelity) and reduce an **indicator-organism
density carried on the stream** — a new optional `Stream.org` scalar, the
disinfection analogue of the temperature `T` scalar: mixers flow-weight it (the
shared `streams._flow_weighted_scalar` behind both `mixed_temperature` and the
new `mixed_organism`) and pass-through units propagate it, and a disinfection unit
applies `N = N0·10^(−log)`. When the inlet carries no indicator (`org is None`)
the unit falls back to its design `inlet_density`, so a terminal disinfection
train works without wiring an indicator influent. The reconstructed effluent
surfaces it: `Plant.stream(...)` returns a `StreamSeries` with an `org` trajectory
(reconstructed on demand via `Plant._reconstruct_stream_org`; `None` for an
indicator-agnostic stream, so every BSM stream is unaffected and does no extra
work). `UVUnit` is **stateless**: the dose is `intensity · exposure · UVT-factor`
with the exposure the baffling-scaled residence `V/Q` converted to seconds
(fluence rate mW/cm², dose mJ/cm²), and a log-linear dose-response
`log = dose/d10` (optional `max_log` tailing). `ChlorineContactUnit` carries a
**one-state chlorine residual** (a completely-mixed tank with first-order decay,
`dCl/dt = (Q/V)(dose − Cl) − k_decay·Cl`); the CT credit `residual · T10` drives
`log = CT/ct_per_log`, with `T10 = baffling·V/Q` (or `t10_from_rtd`, the
non-ideal-contactor 10th-percentile of a residence-time distribution, reusing
`utils/rtd.percentile_time`); `dechlorinate=True` reports the discharged residual
as zero. The credit physics is exposed as pure, AD-clean functions (`uv_dose`,
`uv_log_inactivation`, `ct_value`, `ct_log_removal`, `t10_from_baffling`,
`t10_from_rtd`) for standalone sizing/credit, and `jax.grad` flows through both the
credit and a plant solve (design optimisation). Covered by
`tests/integration/test_disinfection.py`. **Scope/fidelity notes:** the UVT
correction is a first-order linear `uvt/uvt_ref` (a full UVDGM dose-distribution is
future work); chlorine is modelled as a residual-decay + CT credit (no breakpoint
/ chloramine speciation); the indicator transports through mixers + the
disinfection units (the biological train does not yet copy `org`, which is right
for a terminal disinfection step).

**Sequencing batch reactor (`SBRUnit`, issue #273).** A single tank that treats
in batches, cycling through timed phases (fill → react → settle → decant → idle)
defined by a list of `SBRPhase(name, duration, feed=, decant=, kla=, settle=,
mixed=)`. Variable-volume state `[C, V]` (volume rises at `feed_flow` during fill,
falls at `decant_flow` during decant; the `StorageTank` `dV/dt = Q_in − Q_out`
pattern) plus the internal state of a pluggable `SettlingModel`. The biology reacts
every phase; aeration is the per-phase `kla` on the oxygen species; the settle phase
clarifies the supernatant the decant draws as the treated effluent. **Phase transitions are
located events:** `SBRUnit.cycle_events(t0, t1)` returns the phase-boundary times
as a time `Event`, and `Plant.solve` **auto-collects** every unit's `cycle_events`
(merged with any user `events=`) so the integrator lands exactly on each switch —
the flow/aeration discontinuities are resolved at the boundary, not stepped across,
while within a phase the ODE is smooth and differentiable (`jax.grad` flows through
a cycle). Feed is drawn at the unit's own `feed_flow` (a fill pump) taking the
connected stream's composition; a standalone SBR plant is just the `SBRUnit` + an
influent on `sbr.feed`. **Modular settling** ([`plant/settling.py`](aquakin/plant/settling.py)):
a `SettlingModel` strategy reports, each step, how its internal clarity state
evolves and a per-species multiplier the decant draw is scaled by (1 for solubles,
< 1 for settled particulates); mass is conserved by the SBR (a clarified decant
concentrates the retained solids). Two ship: `InterfaceSettling` (one state — a
clarified fraction growing at a settling velocity while the tank settles) and
`LayeredSettling` (a Takács-style vertical profile of the particulate distribution;
the decant draws the top layer). New models slot in by implementing `SettlingModel`.
**Clarity is driven by three regimes**, so the decant actually draws a clarified
effluent: the model's clarity *grows* while settling, *relaxes* back to mixed only
while the tank is **actively mixed** (fed or aerated), and is **held** during a
quiescent phase (decant/idle) — relaxing whenever `settle=False` would otherwise
wash the clarity out during the decant draw itself. A phase's mixing is derived as
`feed or kla>0` unless `SBRPhase(..., mixed=)` is set explicitly (e.g. an unaerated
but mechanically mixed anoxic react). (Settling is well-mixed for the biology — the
bulk `C` is the average; the model affects only the decant clarity.) `plant.mass_balance`
reads the SBR's `[C, V, settling]` inventory (volume at index `n_species`, the
settling state massless); the reaction/aeration *gas* term does not yet cover the
SBR's variable volume and per-phase aeration, so end-to-end closure of an aerated
SBR plant is a follow-up. Note: sludge wasting is not yet a phase, so over many
cycles solids concentrate (a clarified decant retains them); add a waste draw for a
closed long-run solids balance. The
located-event machinery also gained a fix here: a `t_eval` point landing exactly on
an event boundary now emits the segment-endpoint state rather than a dense-output
edge evaluation, which could return NaN for a stiff segment
([`integrate/events.py`](aquakin/integrate/events.py)). Covered by
`tests/integration/test_sbr.py`.

**Membrane bioreactor (`MBRUnit`, issue #274).** A high-MLSS aerated reactor
([`plant/mbr.py`](aquakin/plant/mbr.py)) whose membrane retains the solids,
replacing the secondary clarifier. Fixed-volume reactor state `[C, R_f]` (the bulk
concentrations + a membrane-fouling resistance), reusing the CSTR kinetics and the
`Aeration` machinery — so it takes an open-loop `kla` or a `do_setpoint` the plant
**auto-wires** a DO controller for, exactly like a CSTR. Two outlets: `permeate`
(the filtrate — solubles pass, particulates carried at `(1 − rejection)`, so the
effluent is near solids-free) and `waste` (mixed liquor at the full reactor MLSS,
drawn at the `waste_flow` setpoint). The volume is held constant
(`Q_permeate = Q_in − Q_waste`), and because solids leave **only** via the waste
draw the MLSS concentrates: at a 1-day HRT the biomass is retained where a
clarifier-less CSTR would wash out, and the **SRT = V / Q_waste decouples from the
HRT** (the defining MBR behaviour). A simple membrane-fouling state grows with the
permeate flux and relaxes (`dR_f/dt = fouling_rate·J − fouling_relax·R_f`,
`J = Q_permeate / membrane_area`), reaching a quasi-steady fouled state;
`MBRUnit.tmp(R_f, Q_permeate)` reports the trans-membrane pressure
`tmp_viscosity·J·(R_m + R_f)`. The permeate particulate split uses a per-species
mask built from `particulate_species` (solubles pass unhindered). Scour-air energy
couples to the aeration/blower accounting (the membrane needs continuous coarse-
bubble aeration); for now the biological aeration is the modelled term. The
`_materialize_aeration` sensor tap now names the sensor's first output port
explicitly (a bare endpoint is ambiguous for a multi-output unit like the MBR);
the controller reads the sensed value from the sensor's *state*, so any output
port carries it (unchanged for single-output CSTRs). Modelling choices for the
MVP — fixed volume with the permeate following the feed (vs a flux-controlled
variable-volume membrane) and reversible-fouling TMP (vs explicit backwash/cleaning
events) — are the natural simple forms; both are extension points. Like the CSTR,
the MBR carries the **flow-weighted inlet temperature** onto its outlet streams and
into the Arrhenius kinetics/aeration (via `streams.mixed_temperature`), so a
seasonal influent drives it; and `plant.mass_balance` treats it as a first-class
reactive aerated unit — its `[C, R_f]` inventory reads the fixed reactor volume (the
fouling resistance `R_f` is massless), and it exposes the CSTR `_kla_vec`/`_sat_vec`/
`_controlled_kla` accessors so its aeration-O2 and reaction-gas terms are counted (a
single-MBR plant closes COD/N to a few %). Covered by
`tests/integration/test_mbr.py`.

**Reject storage tank (`build_bsm2(reject=RejectStorage())`).** A variable-volume
equalisation tank on the reject-recycle line: a completely-mixed CSTR with **no
reactions** (`StorageTank`, [`plant/storage.py`](aquakin/plant/storage.py))
whose liquid volume `V` is a state (`dV/dt = Q_in_stored − Q_out`,
`dC_i/dt = Q_in_stored/V·(C_in,i − C_i)`). It releases at a controlled rate
`storage_output_flow` (default 0) with a **level-gated automatic bypass**: full
and filling → divert the whole inflow (don't overfill); full and draining →
release normally; empty → stop releasing and just fill. The two outlets
(`out`, the released stream at tank concentration; `bypass`, the diverted inflow
at inlet concentration) recombine at the front mixer. With the default zero
release the open-loop tank fills to its upper limit (`0.9·Vmax`) and bypasses
**all** reject, so it is a faithful pass-through and the steady state is
unchanged from the no-storage plant (verified: tank5 XB_H identical).
*Architecture note:* the bypass split is gated by the tank's own volume state,
which `StorageTank.flow_outputs` reads from the `FlowContext` `Plant._resolve_flows`
passes into every unit's `flow_outputs`. The exact affine flow solve stays valid
because the tank's *inlet* comes from the fixed-pump sludge line (the wastage
`Qw` is a constant pump), so at fixed volume its outputs are constant in the
recycle flows — the state-dependence does not couple to the recycle variables.
(In this benchmark the reject flow is nearly constant, so a *fixed* release just
fills or drains the tank; genuine equalisation needs the level-based release
controller below.) Wired into `build_bsm2` behind `reject_storage`; demonstrated
in `examples/bsm2_reject_storage.py` (level-gated behaviour by release rate) and
tested in `tests/integration/test_bsm2_storage.py` (the four regimes + flow/
volume conservation, no-solve; wired plant fills-and-bypasses, steady state
healthy).

**Scheduled (timed) wastage (`build_bsm2(wastage_schedule=...)`).** The
waste-sludge pump can follow a time schedule instead of the constant `Qw=300`:
the BSM2 strategy steps the wastage between a low (300) and a high (450) rate at
~182-day half-year blocks over the 609-day evaluation, managing the sludge
inventory (wasting more sludge shortens the solids retention time and draws the
reactor biomass down — verified: tank5 XB_H falls after the step). Built on a
reusable **`PiecewiseConstantSchedule`**
([`plant/schedule.py`](aquakin/plant/schedule.py)): `values[i]` holds on
`[t_breaks[i-1], t_breaks[i])`, evaluated by a `jit`/AD-safe `searchsorted`
gather. `bsm2_wastage_schedule()` returns the BSM2 `Qw(t)`; `build_bsm2` makes
the secondary-clarifier underflow the schedule `Qr + Qw(t)` (via
`schedule.shifted(Qr)`), so the `underflow_split` sends `Qr` to RAS and the
scheduled remainder to wastage. **Time-threaded flow solve:** the settler's
underflow is now time-dependent, so `TakacsClarifier.flow_outputs` reads the time
from the `FlowContext` `Plant._resolve_flows` passes into every unit's
`flow_outputs` (a scheduled setpoint uses `ctx.t`; a constant setpoint ignores
it). The schedule value is a constant at a given `t`, so the affine recycle-flow
probe stays exact; constant-setpoint clarifiers are unaffected.
`split_controlled_flows` drops its `float()` cast so a traced (scheduled)
setpoint flows through. Demonstrated in `examples/bsm2_wastage_schedule.py` and
tested in `tests/integration/test_bsm2_wastage.py` (the schedule's step/validation/
shift/jit behaviour, no-solve; wired plant steps the waste flow on schedule with
RAS held fixed, and higher wastage lowers the biomass).

**Closed-loop reject control (`build_bsm2(reject=RejectStorage(control=True))`).** The storage
tank's release runs a **proportional level controller** instead of a fixed
`Q_out`: `Q_out = clip(bias + gain·(V − V_set), 0, Q_max)` (BSM2: setpoint
`0.5·Vmax`, gain 30 m³/d per m³, pump cap `Q_max = 1500` m³/d = the reference
`Qstorage_max`). The release rises with the level, so the tank self-regulates to
a steady mid-level and releases the reject *smoothly through the controlled pump
with no overflow bypass* — a functioning equalisation tank, versus the open-loop
fill-and-bypass. The net reject returned is the same, so the activated-sludge
steady state is unchanged (XB_H ≈ 2224); only the path differs (controlled
release vs bypass spill), and under a varying reject load the controlled tank
buffers (a level step up → smoothly higher release, no bypass, no chatter — the
proportional law is continuous, so unlike a fixed release > inflow it does *not*
chatter at the empty limit). **Architecture:** the controller lives *inside*
`StorageTank` (`level_setpoint`/`level_gain`/`output_flow_bias`/`output_flow_max`),
not on the signal bus, because the release feeds back into the flow network and
must be resolved *during* the flow solve — but the signal bus is computed
*after* it (`Plant._compute_signals` follows the stream sweep). Since the
release is a pure function of the volume *state*, the in-tank law resolves
exactly via the state the `flow_outputs` `FlowContext` carries. (The signal bus
remains the
right home for a non-flow actuator like the DO `kLa`, which senses a
concentration the sweep must produce first.) Demonstrated in
`examples/bsm2_reject_control.py` (open-loop bypass vs closed-loop control) and
tested in `tests/integration/test_bsm2_reject_control.py` (the release law +
flow/volume conservation, no-solve; wired plant holds a mid-level and releases
the reject with zero bypass).

**Influent hydraulic delay (`build_bsm2(hydraulic_delay=HydraulicDelay())`).** A first-order
lag on the raw influent's flow and load, modelling the transport delay of the
sewer/channel ahead of the works. `HydraulicDelayUnit`
([`plant/delay.py`](aquakin/plant/delay.py)) carries the **load** (`Q·C`) and
the **flow** `Q` as state, each relaxing to the inlet with time constant `tau`
(`d(Q·Cᵢ)/dt = (Q_in·C_in,i − Q·Cᵢ)/tau`, `dQ/dt = (Q_in − Q)/tau`); the outlet
concentration is the lagged load over the lagged flow. This is the BSM2
`hyddelay` structure (a fixed-`tau` lag on load, *not* a fixed-volume tank whose
residence time varies with flow). A flow/load pulse emerges delayed and rounded
(first-order, ~63% of a step after one `tau`); at steady state `Q→Q_in`,
`C→C_in` (a pass-through, so the operating point is unchanged). The **outlet
flow is the held-flow state**, which `flow_outputs` reads from the `FlowContext`
the plant passes into every unit (the same mechanism the storage tank uses).
Wired front-most: `build_bsm2(hydraulic_delay=HydraulicDelay())` puts it on the
influent (entry point becomes `"influent_delay.in"`, read off
`plant.influent_endpoint`), composing with the bypass
(influent → delay → bypass_split → front). **Faithfulness note:** the BSM2
reference `tau≈1e-4` d is a near-instantaneous lag whose role is to break
algebraic loops in a sequential-modular solver — aquakin resolves recycles
directly in one monolithic solve and does not need it, so the unit is here to
model a *physical* delay (`hydraulic_delay_tau`, default ~0.02 d) and to complete
the BSM2 element set. Demonstrated in `examples/bsm2_hydraulic_delay.py` (a flow
pulse emerges lagged) and tested in
`tests/integration/test_bsm2_hydraulic_delay.py` (the lag's fixed-point /
load-over-flow / first-order-response behaviour, no-solve; wired plant builds
front-most, steady state unchanged).

**Hydraulic influent bypass (`build_bsm2(bypass=InfluentBypass())`).** The BSM2
wet-weather bypass: raw influent flow above `bypass_threshold` (default 60000
m³/d) is diverted around the whole treatment train (primary, AS, secondary
clarifier) and rejoined with the clarified effluent — protecting the plant
hydraulics at the cost of releasing untreated wastewater. Built on a new
`SplitterUnit` **threshold mode** (`threshold` + `threshold_port` +
`remainder_port`): `above = max(Q_in − threshold, 0)` to the threshold port,
`min(Q_in, threshold)` to the remainder. The split is on the **raw influent**
flow (an external input), so it stays a constant within the exact recycle-flow
solve (`_resolve_flows`) and doesn't break its affine assumption — important
because the split is piecewise-linear (a kink at the threshold) and would
otherwise be non-affine in the recycle flows. The diverted flow skips the
clarifier too (matching the reference `Qbypassplant=1`: it bypasses the *plant*,
not just the AS) and joins the final effluent through a new `effluent_mix`
combiner, so the final effluent is `effluent_mix.out` (treated + bypassed) —
`evaluate_bsm2` auto-detects it. When a bypass is present `evaluate_bsm2` also
applies the BSM2 **split BOD weighting** — the benchmark's `0.65` raw-sewage
BOD₅/BODu coefficient on the *bypassed* (untreated) BOD vs `0.25` on the treated
effluent — both to the reported BOD *average* (a load-weighted average over the two
source streams `settler.overflow` + `bypass_split.bypass`) **and to the scored
`effluent_quality_index`** (the flat-weight EQI runs on the combined effluent, so
the extra `0.65 − 0.25` weight on the bypass BOD load is added back); the no-bypass
path keeps the flat `0.25` and is untouched. **This changes the influent entry point**: with
the bypass, the influent entry moves to `bypass_split.in` and the effluent to
`effluent_mix.out` -- both reported on `plant.influent_endpoint` /
`plant.effluent_endpoint`, so example/user code reads those instead of a literal.
Default `influent_bypass=False` leaves the plant and its entry point unchanged.
Demonstrated in `examples/bsm2_influent_bypass.py` (storm flow degrades the
effluent) and tested in `tests/integration/test_bsm2_bypass.py` (threshold-mode
flow split + validation; wired-plant flow balance, effluent = treated + bypass,
bypass degrades effluent, evaluation auto-detects the combined effluent).

**BSM2 performance evaluation — EQI / full OCI (`evaluate_bsm2`).** The generic
metric kernels (`aquakin/plant/metrics.py`) are wired to a concrete BSM2
flowsheet by `aquakin.plant.bsm.evaluate_bsm2(plant, solution, params)`,
returning a `BSM2Evaluation` with the EQI, the **full BSM2 OCI** and every
component term. The OCI is the Gernaey et al. 2014 index:
`AE + PE + ME + 3·sludge + 3·carbon − 6·methane + max(0, HE − 7·methane)`:
- **AE** aeration + **ME** mixing energy from the actual kLa over the run
  (`aeration_energy`, `mixing_energy`). Mixing counts the *unaerated* reactors
  (anoxic tanks need mechanical mixing; an aerated tank is mixed by its aeration)
  plus the always-mixed digester, so it spans **all** AS reactors, not just the
  aerated subset.
- **PE** pumping over the full BSM2 pump set (`pumping_energy_bsm2`): AS internal
  recycle / RAS / wastage + the primary / thickener / dewatering underflows, each
  with its own per-m³ factor.
- **sludge** disposal TSS mass flow (factor 3, not the BSM1 5); **carbon** the
  external dose `Q·conc` (`carbon_mass`, kg COD/d).
- **methane** the digester biogas credit — reconstructed from the ADM1 headspace
  gas state and parameters (`_methane_production`: `Q_gas = k_P·(P_gas−P_atm)`,
  `CH4 = (p_ch4/P_gas)·P_atm·16/R_T · Q_gas`); ~1010 kg CH₄/d at the BSM2 steady
  state (reference ≈ 1065).
- **HE** sludge-heating energy (`heating_energy`): raise the digester feed from
  its temperature (the carried stream T, else a 15 °C default) to 35 °C. At the
  BSM2 operating point methane more than covers it, so `max(0, HE − 7·methane)`
  contributes **0** — the biogas self-sufficiency the index rewards.

The aeration kLa **reads the actual value over the run** — a fixed `kla`
open-loop, or under closed-loop DO control the controller's manipulated signal
recovered per saved state by **`Plant.signals_at(t, state, params)`** (the
signal-bus analogue of `outputs_at`; `_rhs`'s signal step is the shared
`Plant._compute_signals`). All output streams are reconstructed in **one
`outputs_at` pass per saved time** (`_reconstruct`), since the indices need ~8
streams. The BSM1-form kernels (`operational_cost_index`, `pumping_energy`) are
kept for BSM1, wrapped by **`evaluate_bsm1(plant, solution, params)`** →
`BSM1Evaluation` (the BSM1 analogue of `evaluate_bsm2`): EQI + the BSM1 OCI
`AE + PE + 5·sludge`, with sludge the wastage TSS mass flow and PE over the
internal-recycle / RAS / wastage pumps. Demonstrated in
`examples/bsm1_dry_weather.py`.

**Labeled report (`str(eval)` / `eval.report()`).** Both `BSM1Evaluation` and
`BSM2Evaluation` render a units-annotated breakdown when printed: the EQI
(`kg poll.-units/d`) and OCI, then each OCI term with its physical value, units
(`kWh/d`, `kg TSS/d`, `kg COD/d`, `kg CH4/d`) and **signed OCI contribution**
(so the methane credit shows as `−6·CH4` and the BSM2 heating enters via
`max(0, HE − 7·methane)`), the effluent averages with currency-specific units,
the aerated reactors counted, and the `oci_note` caveat (always shown, wrapped).
The raw float fields stay available for programmatic use; `str` delegates to
`report()`. So the headline indices are not bare floats to misread against
published Alex 2008 / Gernaey 2014 values (issue #153).

**Top-level exports + `StreamSeries`-friendly kernels.** The metric kernels
(`effluent_quality_index`, `effluent_averages`, `derived_TSS`/`COD`/`BOD`/`TKN`,
`aeration_energy`, `pumping_energy`, `mixing_energy`, `carbon_mass`,
`heating_energy`, `operational_cost_index`, `operational_cost_index_bsm2`,
`pumping_energy_bsm2`), both evaluators (`evaluate_bsm1`/`evaluate_bsm2`) and
`check_conservation` are exported at the top level (`aquakin.…`), not only via the
deep `aquakin.plant.metrics` path. The effluent kernels and the `derived_*`
functions accept a **`StreamSeries` directly** — `effluent_quality_index(eff)` /
`derived_TSS(eff)` (network taken from the stream) — as well as the original
explicit `(t, C, Q, network)` / `(C, network)` forms; a `StreamSeries` is
duck-typed (`.t`/`.C`/`.network`), so a plain concentration array is unaffected.
Demonstrated in `examples/bsm2_evaluation.py` (open- vs
closed-loop table with the full term breakdown) and tested in
`tests/integration/test_bsm2_evaluation.py` (plant terms finite/positive,
aerated-tank detection, AE/ME/carbon match their closed forms, OCI equals the
full-formula sum; plus fast no-solve kernel tests). Note the shipped influent is
synthesised, so these are method-validated numbers, not the published EQI/OCI over
the canonical days-245–609 window (that needs the official IWA influent file).

**Single-point (steady-state) evaluation.** Every time-averaged kernel runs
through one `metrics._time_average(integrand, t)` (and the evaluator's own
`_time_average`): over a multi-point window it is the trapezoidal mean, but for a
**single saved point** — exactly what `plant.run_to_steady_state()` returns (the
terminal state only) — the average of a constant is that sample, so it returns
the **instantaneous steady-state value** instead of dividing by a zero-width
window. So the natural "run to steady state, then `evaluate_bsm1(plant,
ss.solution)`" flow returns finite, meaningful indices rather than raising
`ZeroDivisionError` (the old `aeration_energy` divided by the bare window) or a
spurious zero (the other kernels' `+1e-12` guard). Multi-point results are
unchanged.

**GHG / cost reporting + standardized scenario KPI tables.** On top of the
EQI/OCI evaluation, two presentation layers turn the physical flows a
`BSM2Evaluation`/`BSM1Evaluation` already carries into the carbon-footprint and
cost-OPEX deliverables, plus a standardized side-by-side KPI table:
- **Carbon footprint** ([`aquakin/plant/ghg.py`](aquakin/plant/ghg.py)):
  generic CO₂e kernels (`co2e_from_energy`, `n2o_n_to_co2e` — N₂O-N → N₂O via
  44/28 then ×GWP, `methane_to_co2e`) plus `stripped_n2o` (the aeration-rate
  stripping `Σ kLa_N2O·(S_N2O−S*)·V`, so only aerated tanks emit), assembled by
  `carbon_footprint(energy_kwh, *, grid_factor, n2o_emission, methane_production,
  ch4_fugitive_fraction, biogas_recovered_kwh, ...)` into a `CarbonFootprint`
  (direct N₂O + grid-energy CO₂e + fugitive CH₄ − biogas-energy credit). IPCC
  AR6 100-yr GWP defaults (N₂O 273, biogenic CH₄ 27) and a representative grid
  factor, all overridable. The plant-coupled `direct_n2o_emission(plant, solution,
  params)` (in [`bsm/evaluation.py`](aquakin/plant/bsm/evaluation.py)) reconstructs
  the stripped N₂O from a solved plant (reusing the control-aware `_kla_history`
  and reading the dissolved `SN2O` per reactor); it returns **0** when the AS
  network has no `SN2O` state (the standard ASM1 BSM2 plant — only an N₂O-capable
  network such as `asm3_2step_n2o` gives a non-zero direct term).
- **Operating cost** ([`aquakin/plant/cost.py`](aquakin/plant/cost.py)):
  `operating_cost(*, energy_kwh_per_d, carbon_kg_cod_per_d, sludge_kg_tss_per_d,
  methane_kg_per_d, factors, co2e_per_d)` prices energy / external carbon /
  sludge disposal / biogas credit (`CostFactors`, currency/d) with an optional
  annualised CAPEX and a CO₂e carbon charge → `OperatingCost` (per-day +
  annual).
- **Standardized KPI comparison** (`kpi_comparison` in
  [`integrate/experiments.py`](aquakin/integrate/experiments.py)): tabulates
  heterogeneous report objects (`BSM2Evaluation`, `CarbonFootprint`,
  `OperatingCost` — anything exposing `.kpis()`, or a plain dict) side by side
  into a `KPIComparison` (union of KPI columns, `.best(kpi, minimize=)`). The
  report-object companion to `compare_scenarios` (which runs a model and
  tabulates a fixed output vector). The four evaluation/report dataclasses each
  expose `.kpis()`, and the evaluators a `total_energy()` (AE+PE[+ME]) — the
  energy basis for the GHG/cost layers.
  Demonstrated in `examples/bsm2_ghg_cost_report.py`; kernels + KPI logic tested
  fast in `tests/unit/test_ghg_cost.py`, the plant-coupled path on the shared
  BSM2 solve in `tests/integration/test_bsm2_evaluation.py`.

**Activated-sludge design layer — SRT / HRT / F:M (`aquakin/plant/design.py`).**
Plants are specified in the quantities the solver integrates (tank `volume`,
fixed pump flows, per-species `kLa`), but engineers design in the quantities
those derive *from*: the solids retention time (SRT / sludge age), the hydraulic
retention time (HRT) and the food-to-microorganism ratio (F:M). The design layer
bridges both directions, exported at top level (`aquakin.size_activated_sludge`,
`aquakin.sludge_metrics`, `ActivatedSludgeSizing`, `SludgeMetrics`):
- **Forward sizing** — `size_activated_sludge(SRT=…, HRT_h=…, Q=…, …)` →
  `ActivatedSludgeSizing`. `V = Q·HRT`; the wastage `Qw` from the SRT under a
  stated wasting model: `wastage_from="mixed_liquor"` (hydraulic/Garrett control,
  `Qw = V/SRT`, concentration-independent) or `"underflow"`
  (`Qw = V/(SRT·thickening_ratio)`). Optional `n_tanks`/`volume_fractions` split
  the basin into a CSTR cascade; `internal_recycle_ratio`/`ras_ratio` report the
  pump flows.
- **Achieved metrics (closing the loop)** — `sludge_metrics(plant, solution, …)`,
  also reachable as **`plant.sludge_age(solution)`** (a thin `Plant` method with a
  lazy import to avoid the plant↔design circular). SRT is an *emergent* property of
  `Qw`, so rather than guessing it this reports what the solved model achieved,
  time-averaged over the window: **SRT** = system solids inventory / (wastage +
  effluent solids loss); **HRT** = total reactor volume / influent flow; **F:M** =
  influent BOD load / reactor TSS mass. Reactors are auto-detected (the
  `aeration`-carrying `CSTRUnit`s, so the ADM1 digester is excluded); the
  effluent/wastage ports auto-detect for BSM1/BSM2; the secondary-clarifier sludge
  blanket is included via a new **`TakacsClarifier.solids_mass(state)`** accessor
  (the stateless `IdealClarifier` holds ~0), so the Takács plant correctly reports
  a larger system SRT than the ideal clarifier at the same `Qw`.
- `build_bsm1` gains a `wastage_flow=` argument so `Qw` can be varied without
  reaching into the wiring. Worked end-to-end in
  `examples/bsm1_target_srt.py` — a secant iteration on `achieved_SRT(Qw) − target`
  lands the wastage flow that hits a target sludge age (the by-hand iteration the
  layer replaces; it converges to `Qw ≈ 269` m³/d for a 10-day SRT and shows the
  mixed-liquor forward guess `Qw ≈ 599` differs because BSM1 wastes from the
  thickened underflow). Tested in `tests/integration/test_design.py` (fast
  sizing-relation + validation + `solids_mass` tests in the PR gate; slow
  plant-solve tests for the achieved metrics: SRT/HRT/F:M sensible, HRT = V/Q,
  `plant.sludge_age` delegation, SRT monotone-decreasing in `Qw`, Takács inventory
  raising SRT).

The **ASM1↔ADM1 interfaces** (`aquakin/plant/interfaces.py`, `ASM1toADM1` /
`ADM1toASM1`) are the continuity-based BSM2 interfaces (Nopens et al. 2009 /
Rosen & Jeppsson 2006). `asm2adm` removes the COD demand of O₂/NO₃, then
partitions the remaining ASM COD into ADM substrates under a nitrogen budget
drawn greedily from a priority-ordered list of N pools, with inorganic carbon
and the strong-ion difference (`S_cat`/`S_an`) from a charge balance at the
digester pH; `adm2asm` maps biomass→XS+XP, solubles→SS (H₂/CH₄ stripped),
inerts→XI/SI and inorganic N→SNH. The reference's deeply nested `if/else`
nitrogen cascades are written here **branch-free with `jnp.minimum`** (the
unrolled conditionals are mathematically a greedy allocation), so the maps are
AD-clean. Both **conserve total COD** (`asm2adm` minus the electron-acceptor
demand; `adm2asm` minus the stripped `S_h2`+`S_ch4`) **and total nitrogen** —
verified to `rel 1e-6` in `tests/integration/test_interfaces.py`. The COD
conservation holds whenever the `asm2adm` electron-acceptor (O₂+NO₃) demand does
not exceed the degradable COD it draws from (`SS+XS+XB_H+XB_A`) — always true for
a real near-anoxic digester feed; in the pathological case (recycled nitrate far
exceeding the substrate) the surplus demand is dropped and COD is over-conserved,
mirroring the reference. Construct `ASM1toADM1(strict=True)` to instead raise
(jit/AD-safe, via `eqx.error_if`) when the demand is not fully absorbed, asserting
the feed stays in the intended regime (default `False` is bit-faithful to the
reference). Only the BSM2 `fdegrade = 0` case is implemented (other values raise
`NotImplementedError`).
The charge balances (inorganic carbon + `S_cat`/`S_an` in `asm2adm`, alkalinity
`SALK` in `adm2asm`) are evaluated at the **digester pH**, fed back from the
digester's own state-derived (charge-balance speciation) pH each RHS — as in the
benchmark, where the interface pH is the digester's. The plumbing: each interface
declares `needs_dest_pH` (`asm2adm`, whose destination is the digester) or
`needs_src_pH` (`adm2asm`, whose source is the digester), and
`Plant._collect_inputs` reads that unit's `operating_pH(state, params)` and passes
it as `translate(..., digester_pH=...)`. The `pH_adm` parameter (default 7.0) is
only the fallback for a standalone `translate` call with no plant to supply it.
This is the **only pH-dependent part of the maps** (the inorganic-carbon and
alkalinity charge balances); the COD/N partition is pH-independent, so feeding the
real digester pH (~7.27) instead of the fixed 7.0 leaves every substrate pool
unchanged and only corrects the charge-balance pools: the post-interface digester
**`S_IC`** matches the published BSM2 feed to **0.13%** (was 7.6% at the fixed 7.0)
and the strong-ion `S_an` to 0.015% (was 1.17%), eliminating what had been the
digester's largest steady-state residual. The validated BSM2 reactor steady state
is unchanged (≤0.06%); the digester's remaining ~1.3% is the headspace CO₂
(the charge-balance-pH vs reference-algebraic-pH difference, not the interface).

---

## OpenFOAM Coupling (Future)

The `transport/openfoam/` submodule is the Python side of the OpenFOAM
coupling seam. The C++ plugin lives in a separate repository.

**Option A (current target):** Offline coupling — run OpenFOAM for flow/RTD,
export Lagrangian particle tracks, integrate kinetics along tracks in Python.
No runtime coupling. Suitable for steady-state reactors.

**Option C (future):** pybind11 shared library — C++ fvOptions plugin loads
the Python rate evaluator via pybind11. The `bridge.py` interface documents
the coupling contract that the C++ plugin must satisfy.

The `SpatialConditions` interface is the seam between the two systems. The
C++ plugin is responsible for populating it from OpenFOAM cell field data.

---

## Development workflow

Code review goes through GitHub PRs. The Claude Code sandbox doesn't
have an SSH key for the repo's `git@github.com:...` remote and `gh`
isn't installed, so Claude can't push branches or open PRs directly.
The convention is:

1. Claude commits changes on a sensibly-named local branch (e.g.
   `fix-<thing>`, `add-<feature>`, `<area>-<change>`) — not on
   `main`, not on the `claude/<worktree-name>` scratch branch.
2. The user pushes that branch from their own checkout and opens
   the PR. From a Claude worktree the branch can be picked up with
   `git fetch <worktree-path> <branch>:<branch>` and then
   `git push -u origin <branch>` from the main checkout, or by
   `cd`-ing into the worktree and pushing if the user's shell has
   the SSH key.

Don't try to `git push` from inside the sandbox; it fails with
"Permission denied (publickey)" and wastes a turn.

### Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) splits the suite into a **fast PR
gate** and a **full merge-to-main suite**, because the integration tests are
real stiff-ODE solves with AD (~irreducible seconds each) and the cost is spread
across hundreds of tests — neither a JAX compile-cache (verified: cold ≈ warm)
nor concentrating it in a few files makes the whole suite fast.

- The **`test`** job (`pytest -m "not validation and not slow"`, Python
  3.11/3.12) runs on **every PR and every push** — unit + fast integration. It is
  the merge gate, and is **`pytest-split`-sharded** (`--splits 4 --group i`, `-n
  auto` within each shard) for the *same* reason the slow/validation suites are:
  the fast suite grew to ~1300 tests and a single long-lived run accumulated XLA
  cache + live JAX buffers until an xdist worker was OOM-killed (a worker *crash*
  on a random lightweight test) and/or it hit the time ceiling — intermittent red
  on the required gate with no real regression. Four fresh shards × 2 Python
  versions bound each shard's footprint to ~1/4 (no OOM) and run in parallel (well
  under the ceiling). The sharded per-version jobs are **not** the required check —
  the **`fast-gate`** aggregator (job name **`fast gate`**) is: it `needs:` the
  shard matrix and is green iff every shard succeeded (or was skipped — a bare
  `labeled` event skips the gate, which must not block). It is a *stable* required-
  check name that survives shard-count changes, so **branch protection requires
  `fast gate`**, not the per-shard `fast tests (...)` jobs.
- The **`slow`** job (`pytest -m "slow and not validation and not heavy"`,
  3.11/3.12) and **`validation`** job (`pytest -m "validation and not heavy"`,
  3.12) run **only on push to `main`** (`if: github.event_name == 'push'`). They carry the multi-minute
  stiff/plant solves (the `slow` marker on `test_bsm2_dynamic`, `test_bsm1`,
  `test_biofilm`, `test_forward_sensitivity`, the two `test_wats_sewer_*` files)
  and the published-data checks. A regression a PR's fast gate cannot catch
  therefore surfaces within minutes of merging — revert from there. **Both are
  `pytest-split`-sharded** (`--splits N --group i`, `-n 1` within each shard):
  even serially, a single long-lived process accumulating the whole set's XLA
  compilation cache + live JAX buffers exhausts the 16 GB hosted runner and is
  OOM-reclaimed mid-suite (~62 min in, SIGTERM / exit 143, before the timeout) —
  first the validation set, then the slow set once enough whole-plant tests
  landed. Sharding across N fresh processes bounds each process's footprint to
  ~1/N (slow: 8 shards × 2 Python versions; validation: 4 shards, 3.12). The
  partition is complete and disjoint, so coverage is unchanged. (pytest-split
  balances by `.test_durations` where recorded — now including the heaviest slow
  tests — else evenly by count; duration-balancing evens the wall time.) **But
  sharding alone is loose memory control: more shards or better balancing only
  *relocate* an overweight shard** (a new whole-plant-AD test once walked the OOM
  4/6 → 5/8 across re-shard / re-balance attempts), because the accumulation is
  per-shard, not per-test. A per-test cache-clear fixture (`jax.clear_caches()` +
  `gc.collect()` after every `slow` test, optionally with a `malloc_trim`
  follow-up) was tried and **rejected** — do not re-add it: profiling showed
  `clear_caches()` itself *spikes* RSS and only partially reclaims (the freed
  compiled programs are not returned to the OS, so the process RSS still creeps up
  test-by-test), and `malloc_trim` had nothing to reclaim because the programs stay
  live until GC. The accumulation is intrinsic to the **whole-plant
  stable-adjoint gradient tests**: each compiles a multi-GB plant program, and a
  shard's worth piles up faster than any between-test clear can reclaim,
  OOM-killing the runner regardless of shard count (a single such test runs fine
  alone — the failure is cumulative). So those tests carry the **`heavy`** marker
  and are kept off the free-runner jobs — every slow / validation / smoke /
  durations job runs `... and not heavy`. They run instead on the **`heavy` job**:
  a GitHub-hosted **larger runner** (16-core / 64 GB, Team plan; `runs-on:
  aquakin-heavy`) whose RAM fits every heavy test in one shared process, gated to
  **push-to-main** like slow/validation (a fork PR cannot reach a hosted runner, so
  there is no self-hosted security concern; per-minute billing is why it stays off
  the PR path). It runs `pytest -m heavy`, covering both the BSM2 validation-heavy
  tests and the BSM1 slow-heavy dynamic-sensitivity tests; locally they run the
  same way. (If the `aquakin-heavy` runner does not exist the job stays queued and
  blocks nothing.) Sharding still earns its keep on **wall time** (parallel
  shards), not memory.
- The **`heavy`** job (`pytest -m heavy`, 3.12) runs the whole-plant
  stable-adjoint gradient tests that no free runner fits, on a GitHub-hosted
  **larger runner** (`runs-on: aquakin-heavy`, 16-core/64 GB, Team plan), **only on
  push to `main`** (`if: github.event_name == 'push'`). One shared process (the
  64 GB RAM fits the accumulated compiles), with the top-level single-thread `env:`
  overridden to use the runner's cores. See the memory discussion above for why
  these are off the free-runner jobs.
- The **`smoke`** job (`pytest -m "slow and not validation and not heavy" --splits
  18 --group <rotating>`, 3.12) runs on **every PR** as an early-warning slice of the
  merge-only `slow` set: it *executes* a bounded ~1/18 shard (~8 tests) so
  shared-fixture breaks, whole-plant call-site regressions and memory creep show
  up before merge, not after. It is deliberately probabilistic — a single PR
  runs only ~8 of the ~141 slow tests — but the shard **rotates by run number**,
  so consecutive runs cover the whole set, and a fixture break (which hits most
  slow tests) is caught by any shard. Scope is `slow and not validation`: with no
  recorded durations those tests split evenly by count (no empty shard, unlike a
  duration-skewed slow+validation split) and the heaviest ~4-min published-data
  tests are excluded, keeping the slice time predictable. It complements the
  fast-gate signature-contract test (`tests/unit/test_api_signatures.py`), which
  catches the *signature* sub-case deterministically; the smoke adds breadth.
- **The `full-ci` label** runs the full merge suite (the `slow` **and**
  `validation` jobs) on a PR *before* merge. Apply it to a PR (the workflow
  listens for the `labeled` event, so no fresh push is needed) and the slow
  whole-plant solves and published-data validation run before it lands. Without
  the label those jobs run only **after** merge to `main`, so a regression they
  catch surfaces post-merge and is reverted from there; the label moves that
  signal earlier at the cost of the runtime. The bare `labeled` event
  deliberately does **not** re-run the fast gate / smoke (they already ran on the
  latest commit) and does not cancel an in-progress run, so labelling never
  disturbs the required checks.
  - **Convention — apply `full-ci` to any PR that touches convergence-sensitive
    code:** the integrators / adjoints, the PTC steady-state solver
    (`plant/steady.py`), the plant assembly / recycle resolution, a network's
    stoichiometry, the pH / precipitation solvers, or the metric / mass-balance
    kernels. Those are exactly the changes whose regressions live in the `slow` /
    `validation` suites, which **do not run on the PR fast gate** — so the
    fast-gate-plus-rotating-smoke coverage can pass while a whole-plant solve or a
    published-data check is broken. (Concretely: a brittle slow test added in
    #394 — a BSM2 PTC cold-start convergence with a platform-sensitive iteration
    count — passed the fast gate and broke `main` only on the post-merge slow run;
    `full-ci` before merge would have caught it. The fix was to make the test
    deterministic, but the label is the process guard.) Skipping the label is fine
    only for changes that cannot reach the slow/validation paths (docs, a new
    isolated unit, a fast-gated network add).

**Branch protection:** the required status check must be the **`fast gate`**
aggregator job — **not** the per-shard `fast tests (py3.x shard i/4)` jobs (their
names/count change when the shard count is tuned) and **not** `slow`/`validation`
(which do not run on PRs and would otherwise block every PR forever). The
`fast gate` job is green only when every fast shard passes, so requiring it gives
the same gate with a stable name. (When the fast gate was sharded, the old
required checks `fast tests (py3.11)` / `(py3.12)` ceased to exist; branch
protection must be repointed to `fast gate` or merges block forever waiting on
checks that never report.)

A green fast gate is the merge gate. Python 3.10 stays
install-compatible (`requires-python >= 3.10`) but is **not** CI-tested:
its heavier jaxlib 0.6.2 build ran close to the hosted runner's
resource/time limits and the job was intermittently killed mid-run, while
3.11/3.12 stayed green and the suite passed locally on 3.10. When adding a runtime
dependency, add it to `pyproject.toml` `dependencies` so the CI install
(`pip install -e ".[test]"`) picks it up — the `_make_*` network
generators' `ruamel` need is intentionally *not* a runtime dep (they are
run manually, never imported by the package or tests). `pandas` is an
**optional** dependency (the `dataframe` extra, for the `to_dataframe()` /
`to_csv()` result exporters), but it is also in the `test` extra so the fast
gate exercises those exporters; the library imports it lazily inside the
exporters (`require_pandas`), so solving never needs it.

---

## Post-Change Checklist

After **every code change**, before considering the task complete, review and
act on the following:

1. **Tests** — Are new tests needed for the changed or added functionality?
   - New node type → unit test in `test_nodes.py`
   - New public API function → integration test
   - New built-in network → validation test against literature data
   - Bug fix → regression test

2. **CLAUDE.md** — Does this file need updating?
   - New architecture decision made during implementation
   - Public API surface changed
   - New node types added
   - Package structure changed
   - New dependencies added

3. **README.md** — Does the user-facing documentation need updating?
   - New public API functions
   - New built-in networks
   - Installation or dependency changes
   - New examples

If the answer to any of the above is yes, make those updates as part of the
same task before marking it complete.

---

## Key Literature

- Acero, J.L. & von Gunten, U. (2001). Characterization of oxidation processes:
  ozonation and the AOP O₃/H₂O₂. *J. AWWA*, 93(10), 90–100.
- von Gunten, U. & Hoigné, J. (1994). Bromate formation during ozonation of
  bromide-containing waters. *Environ. Sci. Technol.*, 28(7), 1234–1242.
- Kidger, P. (2021). On neural differential equations. *PhD Thesis, University
  of Oxford*. (Diffrax reference)
