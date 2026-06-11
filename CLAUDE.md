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
  wastewater treatment model (Henze et al. 1987).
- `asm2d` — ASM2D, ASM1 extended with biological phosphorus removal
  and denitrifying polyphosphate-accumulating organisms.
- `asm2d_tud` — Delft TUD variant of ASM2D with revised bio-P stoichiometry.
- `asm3` — ASM3, ASM1 with internal storage products replacing hydrolysis.
- `asm3_biop` — ASM3 + bio-P extension.
- `adm1` — Anaerobic Digestion Model No. 1 (Batstone et al. 2002), BSM2
  implementation form (Rosen & Jeppsson 2006). 29 states (26 liquid + 3 gas
  headspace), 25 processes: disintegration, the three hydrolyses, seven
  substrate-uptake reactions (with pH / hydrogen / free-ammonia /
  inorganic-N inhibition, the lower-pH inhibition via the new `pHInhibitNode`),
  seven biomass decays, and the gas headspace (kLa transfer of H₂/CH₄/CO₂ plus
  overpressure-driven biogas outflow). Inorganic-carbon and -nitrogen
  stoichiometry are symbolic parameter-expressions (the ADM1 elemental
  balances), so a calibrated yield/composition flows through them. pH is
  **state-derived** through the charge-balance `speciation:` solver (extended to
  the four ADM1 volatile fatty acids), with the strong-ion difference carried by
  explicit conservative `S_cat`/`S_an` ion states (via the solver's
  `strong_cations`/`strong_anions` terms); the free-ammonia and dissolved-CO₂
  pH-switch fractions therefore track the instantaneous state. This is the
  complete ADM1 in BSM2 form, **validated** against the published BSM2
  open-loop steady-state digester: run as the benchmark CSTR (3400 m³ liquid /
  300 m³ headspace, fed at ~178 m³/d, HRT ~19 d) it reproduces the reference
  steady state to ~1–3% on substrates/biomass/biogas, with a charge-balance pH
  (~7.27) matching the reference electroneutrality relation
  (`tests/validation/test_adm1_bsm2_steadystate.py`). Note: the BSM2 init file
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

The four SUMO-derived ASM networks were generated by
[`scripts/sumo_to_aquakin.py`](scripts/sumo_to_aquakin.py) from JSON dumps
produced by WastewaterAD's `wastewaterad.tools.sumo_import`. Stoichiometric
coefficients that depend on yield / N-content / fraction parameters are
precomputed at literature defaults; kinetic parameters remain free.

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

New domain-specific node types are added here as needed. Each node implements:

```python
def compile(self, ctx: CompileContext) -> Callable:
    """Returns a JAX-compatible callable: (C, params, condition_arrays, loc_idx) -> scalar"""
```

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
reactor with `adjoint=diffrax.DirectAdjoint()`, which is plainly differentiable
in both modes. Its drawback is *usually* memory (it stores/unrolls the whole
solve, cost growing with step count), so keep `RecursiveCheckpointAdjoint` as the
default and switch to `DirectAdjoint` only when forward-mode is actually required.
`dgsm(..., mode="forward")` is the first-class consumer of this (see the DGSM API
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
`dgsm(mode="forward")`). A future alternative is a quasi-steady-state (QSS)
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

**Two solvers, low- and high-order.** `implicit_euler_adjoint_solve` (first
order) is the simple, robust baseline. `esdirk_adjoint_solve` is the high-order
version: a general s-stage ESDIRK forward (default **`Kvaerno5`, the same method
the reactors use**) whose discrete adjoint recomputes the stage values in the
backward pass (diffrax saves only step states) and applies the transposed-stage
recurrence `(I − dt·γ·Jᵢᵀ)⁻¹` per stage — the FATODE/Sandu construction (verified
to reduce to the implicit-Euler case for s=1). **`calibrate(gradient=
"stable_adjoint")` now uses the Kvaerno5 ESDIRK adjoint**, so its forward matches
the reactor exactly and its gradients agree with the capped `jax_adjoint` path to
the optimiser tolerance (analytic decay `rel ≈ 1e-6`; stiff network
finite-uncapped, matching capped Kvaerno5 to `rel ≈ 2.5e-5`, the residual being
the capped-vs-uncapped *forward* difference, FD-confirmed). **Cost note:** the
backward scan's cost scales with `stable_adjoint_max_steps` (the padded
trajectory length), and the ESDIRK backward recomputes the 7 stages per step
(Newton + transposed solves), so keep `max_steps` tight; Kvaerno5's high order
keeps the step count low. The autonomous reaction RHS is assumed (the ESDIRK
stage times `c` do not enter).

### Operator Splitting

Transport and reaction are decoupled at all scales:

| Scale | Transport | Reaction |
|---|---|---|
| 0D batch | n/a | Diffrax directly |
| 1D PFR | advection/diffusion step | Diffrax reaction sub-step |
| 3D CFD | OpenFOAM transport step | Diffrax (or C++ stiff solver) reaction sub-step |

The ODE integrator only ever sees the reaction sub-problem — a pure chemistry
integration over one transport timestep at a fixed spatial location.

### JAX x64 Mode

Stiff ODE integration requires 64-bit floats. `aquakin` enables x64 mode
automatically at import time:

```python
# aquakin/__init__.py
import jax
jax.config.update("jax_enable_x64", True)
```

This is documented in the README. Do not remove this.

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
  all declared fields.
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
- Optional `expressions:` block at the network level lets you give a name
  to an intermediate rate expression and reference it from a reaction's
  `rate:` or from another expression. References are inlined into the
  consuming AST at compile time. Cycles among expressions are rejected.
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
**fixed number of Newton iterations** on the electroneutrality residual in
`u = ln[H+]` space (`jax.lax.scan`, no data-dependent control flow), so it is
`jit` / `vmap` / `grad` friendly and composes inside a Diffrax RHS.
Equilibrium constants are temperature-corrected via van't Hoff. The chemistry
mirrors the WATS reference pH solver. Validated against an independent
bisection root finder in `tests/unit/test_ph_solver.py`.

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
│   │   ├── network.py               # CompiledNetwork dataclass + compile()
│   │   ├── conditions.py            # SpatialConditions dataclass
│   │   ├── context.py               # CompileContext dataclass
│   │   ├── ph_solver.py             # differentiable charge-balance pH solver
│   │   └── speciation.py            # speciation block -> derived pH condition fn
│   │
│   ├── schema/
│   │   ├── network_spec.py          # Pydantic models
│   │   └── loader.py                # YAML -> Pydantic -> CompiledNetwork
│   │
│   ├── integrate/
│   │   ├── _common.py               # shared helpers: atol coercion, _run_diffeqsolve,
│   │   │                            #   solve_chemistry (the one stoich-hoist + RHS + solve
│   │   │                            #   factory the Batch/PFR/Particle/CFD reactors all call,
│   │   │                            #   parameterised by cond_fn / rate_scale / saveat),
│   │   │                            #   validate_t_eval, Reactor Protocol
│   │   ├── batch.py                 # BatchReactor, BatchSolution
│   │   ├── biofilm.py               # BiofilmReactor (layered 1-D diffusion-reaction)
│   │   ├── pfr.py                   # PlugFlowReactor, PFRSolution
│   │   ├── particle.py              # Track, ParticleTrackReactor, integrate_ensemble
│   │   ├── cfd.py                   # CFDReactor (Option C runtime coupling)
│   │   ├── sensitivity.py           # sensitivity(), fit()
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
│   │   ├── calibrate.py             # calibrate(): transforms, priors, Laplace posterior,
│   │   │                            #   multistart, free initial conditions, Gauss-Newton
│   │   │                            #   optimizer, posterior-predictive bands
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
│   │   ├── asm2d.yaml               # ASM2D (bio-P + denitrification)
│   │   ├── asm2d_tud.yaml           # Delft TUD variant of ASM2D
│   │   ├── asm3.yaml                # ASM3 (storage products replace hydrolysis)
│   │   ├── asm3_biop.yaml           # ASM3 + bio-P extension
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
│   │   └── wats_sewer_khalil_paper_balanced_biofilm_multispecies.yaml  # + X_SRB/X_MA/X_SOB groups
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
│   ├── bsm1_dry_weather.py
│   └── adjoint_speed_benchmark.py  # stable_adjoint vs capped jax_adjoint timing
│   # NOTE: the wats_sewer_extended batch-fitting / calibration / sensitivity scripts and
│   # their measurement data live in the separate paper-reproduction repository,
│   # not here (this repo ships only the reusable library + networks).
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
network.default_concentrations()     # jnp.array
network.default_parameters()         # jnp.array
network.summary()                    # prints human-readable table
network.to_latex()                   # LaTeX rate expressions

# Conditions
conditions = aquakin.SpatialConditions.uniform(n_locations=1, pH=7.5, T=293.15)
conditions = aquakin.SpatialConditions(fields={"pH": jnp.array([...]), ...})

# Batch reactor
reactor = aquakin.BatchReactor(network, conditions)
solution = reactor.solve(C0, params, t_span, t_eval)
solution.t                           # (n_t,)
solution.C                           # (n_t, n_species)
solution.C_named("BrO3-")           # convenience accessor

# Plug flow reactor
reactor = aquakin.PlugFlowReactor(network, conditions, n_points, length, velocity)
solution = reactor.solve(C0, params)
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
#   - steady_state(C0, params, warmup=...): Newton/Levenberg-Marquardt root-find
#     on RHS=0 with implicit-diff for AD (optimistix). Works for well-conditioned
#     steady states; for a VERY stiff/slow biofilm (the multispecies maturation,
#     whose asymptotic fixed point is hundreds of days out) it stalls -- there,
#     integrate forward to the physical maturation time (~90 d for the Khalil rig)
#     and use that profile as the IC instead.
reactor = aquakin.BiofilmReactor(
    network, conditions, n_layers=6, thickness=8e-4, area_per_volume=50.0,
    diffusivity=1e-4, boundary_layer=1e-4,
    biofilm_reactions=[...])             # names of the {A_V} reactions (run in layers only)
solution = reactor.solve(C0, params, t_span, t_eval)  # C0 (n_species,) or (n_layers+1, n_species)
solution.C                           # (n_t, n_species) -- BULK (measurable) trajectory
solution.profile                     # (n_t, n_layers+1, n_species) -- depth-resolved (0=bulk)
solution.depth                       # (n_layers,) layer mid-depths from the surface
solution.profile_named("S_NO")       # (n_t, n_layers+1) depth profile over time

# Sensitivity and fitting
sens = aquakin.sensitivity(reactor, C0, params, output_fn)
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

# mode= selects the AD direction used to form the per-sample sensitivities
# (identical results to machine precision — purely a performance choice):
#   "reverse" (default) — m reverse passes (one per output), each d-independent.
#                         Best for few outputs and a cheap adjoint.
#   "forward"           — d forward-mode tangents through one solve, m-independent.
#                         Best for many outputs, or when the reverse adjoint is
#                         stiff-inflated (dtmax-capped). REQUIRES the reactor in
#                         fn to use adjoint=diffrax.DirectAdjoint().
# If fn returns a vector of m outputs, dgsm returns a list[DGSMResult], one per
# output (each carrying .output_name) — screen all outputs in a single call.
outs = aquakin.dgsm(fn_vec, ranges, output_names=[...], mode="forward")
# Benchmark (tests/ + the JRN-055 reproduction): for a 4-output, 17-input stiff
# batch screen, forward mode is ~2x faster (and lighter on memory) than reverse,
# because reverse pays the stiff adjoint once per output while forward pushes all
# d tangents through one solve. For a single scalar output, reverse is cheaper.

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
```

Internal implementation details (`ASTNode` subclasses, `CompileContext`,
Pydantic models, Diffrax solver objects) are not part of the public API and
should not be imported from `aquakin` directly. They are accessible via
submodules for advanced users.

Reactors are **stateless after construction** — `solve()` takes all variable
inputs as arguments. This enables `jax.vmap` over initial conditions or
parameter ensembles.

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
]
testpaths = ["tests"]
```

```bash
pytest -m "not validation"   # development
pytest -m validation          # validation suite
pytest                        # everything
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

Key types:

- `Stream(Q, C, network)` — the bulk-flow + concentration record passed
  between units.
- `Unit` Protocol — every unit declares `state_size`, `input_ports`,
  `output_ports`, and implements `initial_state()`, `compute_outputs()`,
  and `rhs()`.
- `StateTranslator` Protocol — converts streams between networks.
  `IdentityTranslator` covers single-network plants (BSM1).
- `Plant` — assembles units and connections, drives the monolithic
  integration. Recycles are resolved by iterating the per-RHS stream
  computation 3 times (sufficient for typical BSM topologies).

Shipped units: `CSTRUnit` (kinetics + aeration), `MixerUnit`,
`SplitterUnit`, `IdealClarifier` (fast, stateless separator),
`PrimaryClarifier` (BSM2 Otterpohl–Freund: a well-mixed holding tank split by
an HRT-dependent particulate-removal efficiency, fixed underflow `f_PS·Q`),
`IdealThickener` (BSM2 thickener / dewatering — a stateless ideal `%TSS`
separator, concentration-dependent underflow flow), `ADM1DigesterUnit`
(continuously-fed ADM1 CSTR with gas headspace, dilution masked to the liquid
states), and `TakacsClarifier` (10-layer 1-D Takács 1991 model). Its settling physics
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
and `Plant.solve` takes `max_steps`.

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
  concentration sweep on the fixed flows. This was the keystone.
- **Non-negative flow split** (the remainder outflow clamped into `[0, Q_in]` in
  both clarifiers): guards against a negative underflow when the feed dips below
  the design split — closes issue #17; inactive at steady state.
- **`clip_negative_states`** on ASM1 (the reference `xtemp = max(x,0)` clamp).

`Plant.solve` takes an optional `y0=` for warm-starting (e.g. a dynamic run
from a precomputed steady state).

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

**Quantitatively validated** against the published BSM2 open-loop steady state
(`tests/validation/test_bsm2_steadystate.py`): run with the published constant
influent (`bsm2_constant_influent`) and the BSM2 (15 °C) ASM1 parameter set
(`bsm2_parameters`), the whole multi-network plant — the 5 AS reactors, the
secondary settler, the primary clarifier, both ASM1↔ADM1 interfaces, the
digester, and all recycle loops including the reject water — reproduces the
reference reactor states (`asm1init_bsm2` `XINIT`: XB_H ≈ 2245, XB_A ≈ 167,
XP ≈ 967, XI ≈ 1532, the SNH/SNO/SO profiles) and the digester (`DIGESTERINIT`:
headspace methane to ~0.2%) **to within ~3% on every key state**. Two parameter
reconciliations were needed: the BSM2 ASM1 values are the 15 °C set
(`muH=4, KS=10, muA=0.5, bH=0.3, KX=0.1, etah=0.8`), and aquakin's ASM1 adds a
heterotroph ammonia-limitation term (`KNH_H`) that the BSM/IWA ASM1 lacks, so
`bsm2_parameters` disables it (`KNH_H → 0`); without it tank-5 growth is
suppressed ~24% and XB_H comes out ~half. ASM1 has no Arrhenius T-dependence
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
(`tests/integration/test_bsm2_dynamic.py`). The influent files share the BSM1
column layout (TSS and the time-varying influent temperature are omitted —
aquakin's ASM1 is temperature-independent and the digester is fixed-T) and are
**synthesised**, not the canonical 609-day IWA series, so the dynamic tests
assert qualitative stability, not published dynamic metrics.

The **storage tank, hydraulic delay, influent bypass, sensors and controllers
are still omitted** (the closed-loop additions are the remaining BSM2 phase).
The digester is additionally validated at the unit level in
`tests/validation/test_bsm2_digester_unit.py`.

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
verified to `rel 1e-6` in `tests/integration/test_interfaces.py`. Only the BSM2
`fdegrade = 0` case is implemented (other values raise `NotImplementedError`).
The digester pH used in the charge balance is a fixed parameter (default 7.0);
the digester's own charge-balance speciation solver sets the actual pH from the
state, so a representative fixed value is sufficient for the steady state.

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

GitHub Actions runs the test suite on every push to `main` and every
pull request (`.github/workflows/ci.yml`). The `test` job runs
`pytest -m "not validation"` across Python 3.11/3.12; a separate
`validation` job runs `pytest -m validation` (the slower published-data
checks) on 3.12. A green CI is the merge gate. Python 3.10 stays
install-compatible (`requires-python >= 3.10`) but is **not** CI-tested:
its heavier jaxlib 0.6.2 build ran close to the hosted runner's
resource/time limits and the job was intermittently killed mid-run, while
3.11/3.12 stayed green and the suite passed locally on 3.10. When adding a runtime
dependency, add it to `pyproject.toml` `dependencies` so the CI install
(`pip install -e ".[test]"`) picks it up — the `_make_*` network
generators' `ruamel` need is intentionally *not* a runtime dep (they are
run manually, never imported by the package or tests).

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
