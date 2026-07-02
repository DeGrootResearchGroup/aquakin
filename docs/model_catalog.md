# Model catalog

The full catalog of shipped `aquakin` reaction models: what each is, its
literature provenance, and the documented corrections. Extracted verbatim from
the project briefing. The Khalil model-improvement sequence is in
`khalil_reproduction_log.md`.

The shipped models currently are:

- `ozone_bromate` — bromate formation during ozonation, with explicit OH
  radical chemistry (after Acero & von Gunten 2001; Pinkernell & von Gunten 2001).
- `uv_h2o2` — UV/H₂O₂ advanced oxidation of a generic target micropollutant.
- `asm1` — Activated Sludge Model No. 1, the IAWQ reference biological
  wastewater treatment model (Henze et al. 1987). The shipped model is the
  textbook / BSM-faithful Gujer matrix — heterotroph growth has **no** ammonia
  (nitrogen-source) availability switch — so `load_model("asm1")` reproduces
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
  broken; running the model to a viable steady state in the A²O plug below is
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
  constant). With these the model nitrifies, removes phosphorus biologically,
  and the dosed-metal chemical-P works (the A²O plant's ferric dosing). The same
  import errors were found and fixed in `asm2d_tud`, `asm3` and `asm3_biop` — the
  whole bio-P/nitrification family is corrected. **The entire family is now
  verified term-by-term, value-by-value against the SUMO source spreadsheets**
  (every parameter value, every rate-term Monod half-saturation constant, every
  participating-species set, and every numeric stoichiometric coefficient
  including the charge-balance `SALK`/`SHCO` terms match SUMO — 482 coefficients,
  0 mismatches across the four models; the corrected constants are pinned per
  model in `tests/integration/test_asm_family.py::
  test_biop_autotroph_and_polyp_constants`). **The root cause was UPSTREAM, in the
  external `wastewaterad.tools.sumo_import` JSON dump the models were originally
  imported from** — not in any aquakin-side transcription. The collapse (per-group
  Monod constants deduplicated by species), the lost `K_max` poly-P-storage
  override (the SUMO `MRinh`/`MRsat` *generic* function form was exported instead
  of the instantiated "Calculated variables" form), and the dropped precipitation
  metal coefficients (collapsed into their `XTSS` sum) all already existed in that
  JSON. The four ASM models are now **maintained directly as YAML** (the
  one-shot SUMO import converter has been retired, so there is no regeneration step
  to reintroduce the upstream bugs); they remain audited against the SUMO `.xlsm`
  source by [`scripts/verify_sumo_asm.py`](scripts/verify_sumo_asm.py), which
  re-checks every parameter, rate-term constant, participating species and numeric
  coefficient and exits non-zero on any discrepancy — run it after any hand-edit
  to a SUMO-derived model.)*
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
  precipitation engine composing with a full biological model.
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
  present the model supports **partial-nitritation/anammox (PN/A)
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
  activated-sludge comammox Gujer matrix in the literature — this model is a
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
  correct for any digester size (the model default is the BSM2 3400/300 = 11⅓).
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
  particulate conversion at HRT 19 d); the model ships `k_hyd = 10`.
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
  models* below) — with that, `jax.grad`/`jax.jvp` are finite and match
  finite differences. **Validated** against the published batch nitrate-dosing
  experiments: after parameterizing the saturation constants the published study
  calibrated (the reference C hardcoded them) and AD-calibrating the influential
  rates, sulfide/sulfate/nitrate match the measured data well; VFA is the weak
  point (as in the source study), and the calibration/validation batches favour
  slightly different sulfur kinetics. The batch measurement data and the
  figure/analysis scripts for that study live in a **separate
  paper-reproduction repository** (it imports this library and pins an exact
  commit); they are not shipped here. This library provides only the reusable
  pieces: the model + structural variants, and `calibrate` / `dgsm`. First
  model to use a **state-derived pH** (see *Speciation / state-derived pH*
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
  with extensions. This model (27 reactions, 18 species) includes aerobic +
  anoxic heterotrophic growth (bulk + biofilm), aerobic + anoxic maintenance,
  anaerobic fast/slow hydrolysis, fermentation, methanogenesis, VFA-driven
  sulfate and elemental-sulfur reduction, the two-step nitrate-driven sulfur
  oxidation (paper additions), **and the base-WATS aerobic sulfur cycle**:
  pH-dependent chemical and biological *bulk* sulfide oxidation (Table 9.4) and
  biological *biofilm* sulfide / elemental-S oxidation. The **aerobic backbone
  and the aerobic sulfur-oxidation pathway are O2-gated and therefore dormant**
  under the air-sealed anoxic batch (`S_O ~ 0`) — they carry zero rate and do
  not change the batch trajectories, but make the model the structurally
  complete WATS model. Paper modifications: half-order WATS biofilm terms →
  Monod (so the aerobic biofilm oxidation is Monod-ized); no temperature
  correction (batch at 20 °C); pH supplied as a **fixed operating condition**
  (not the charge-balance solver); removes chemical oxidation of sulfide *by
  nitrate* (anoxic oxidation is biological only). Carries the full WATS species
  vector; the N/P/inorganic-carbon/inert/autotroph pools are present but largely
  inert in the batch. The paper-active core is augmented with the dormant
  full-WATS aerobic pieces by
  [`models/_make_khalil_paper.py`](aquakin/models/_make_khalil_paper.py)
  (comment-preserving ruamel splice from `wats_sewer_extended.yaml`). Ships with
  structural variants (`_halforder`, `_directsulfate`, `_srbsubstrate`,
  `_combined`, and the standalone falsification variant `_stopatS0` — the
  nitrate-driven oxidation stops at elemental sulfur, the S⁰→sulfate step
  removed, which is mutually exclusive with `_directsulfate` and so is NOT folded
  into `_combined`) generated reproducibly by
  [`models/_make_khalil_variants.py`](aquakin/models/_make_khalil_variants.py).
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
  the shared dosed-nitrate budget). This model is the **faithful** reproduction
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
  Khalil model that conserves all of COD, S, Fe and N. Generated by
  [`models/_make_khalil_balanced.py`](aquakin/models/_make_khalil_balanced.py)
  from the faithful model.
- `wats_sewer_khalil_thesis` — the same complete WATS model + nitrate-dosing
  additions as specified in Khalil's *thesis* (the base WATS model of thesis
  Ch. 3 plus the Table 4-1 additions): the full WATS process matrix
  (Hvitved-Jacobsen Tables 9.1–9.4 — aerobic/anoxic/anaerobic carbon turnover
  plus the sulfur cycle, including the chemical/biological **aerobic** sulfide
  oxidation with **half-order** biofilm kinetics) extended with methanogenesis,
  elemental-sulfur reduction and the two-step nitrate-driven sulfur oxidation.
  Generated by [`models/_make_khalil_thesis.py`](aquakin/models/_make_khalil_thesis.py)
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
  shared metal) and exposed via `model.precipitation_equilibrium(...)`, the
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

