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

Future networks include UV/TiO₂, chlorine decay, and ADM1 (anaerobic
digestion).

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
  z_cation_eq: 3.28e-3      # net fixed cation charge (eq/L); literal or {condition: name}
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
```

`build_ph_derived_fn` (in `core/speciation.py`, no Pydantic) turns the
validated declaration plus `species_index` into the derived-condition
callable. Negative concentrations are clamped to zero before conversion
(mirrors the reference). Valid `totals` keys: `carbonate`, `acetate`,
`ammonia`, `phosphate`, `sulfide`.

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
│   │   ├── _common.py               # shared helpers (atol coercion, jit'd solve, Reactor Protocol)
│   │   ├── batch.py                 # BatchReactor, BatchSolution
│   │   ├── pfr.py                   # PlugFlowReactor, PFRSolution
│   │   ├── particle.py              # Track, ParticleTrackReactor, integrate_ensemble
│   │   ├── cfd.py                   # CFDReactor (Option C runtime coupling)
│   │   ├── sensitivity.py           # sensitivity(), fit()
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
│   │   ├── wats_sewer.yaml          # original reference-book WATS (Tables 9.1-9.4)
│   │   ├── wats_sewer_extended.yaml  # extended WATS (+ nitrate/methane/elemental-S, state-derived pH)
│   │   ├── wats_sewer_extended_*.yaml # extended-model structural variants + v0
│   │   ├── wats_sewer_khalil_paper*.yaml # paper-faithful Khalil (2025) model + variants
│   │   └── wats_sewer_khalil_thesis.yaml # thesis-faithful Khalil model
│   │
│   │   # wats_sewer_khalil_paper (paper) is the paper-active core augmented with the
│   │   #   dormant full-WATS aerobic pieces by networks/_make_khalil_paper.py;
│   │   # wats_sewer_khalil_thesis is generated from wats_sewer_extended.yaml by
│   │   #   networks/_make_khalil_thesis.py; the structural variants by
│   │   #   networks/_make_khalil_variants.py
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
│   │   └── test_bromate_vongunten.py# validates against published data
│   └── fixtures/
│       └── simple_network.yaml      # minimal 2-species toy network for unit tests
│
├── examples/
│   ├── batch_bromate.py
│   ├── lagrangian_demo.py
│   └── sensitivity_demo.py
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

# Sensitivity and fitting
sens = aquakin.sensitivity(reactor, C0, params, output_fn)
sens.doutput_dparams                 # (n_params,)
sens.doutput_dconditions["pH"]       # (n_locations,) — dict access
sens.ranked_params()

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
# posterior, drops near-null (non-identifiable) eigen-directions so draws stay
# finite, propagates each through a solve, and returns per-timepoint percentiles.
# The C0 passed in may differ from calibration (e.g. a held-out validation batch).
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
`TakacsClarifier` (10-layer 1-D Takács 1991 model). Its settling physics
are correct and verified in isolation at BSM1 solids loading: the
clarification-zone flux limiting (above the feed, the downward flux is
limited by the layer below only when that layer exceeds `X_threshold`) and
the per-species flux apportioning (each species settles at the bulk
velocity, `flux_tss · X_k/TSS`, conserving total settleable solids) produce
a monotone sludge blanket, a strongly clarified effluent, a thickened
underflow, and tight solids mass balance. `build_bsm1(use_takacs=True)`
selects it in the full plant (both clarifiers expose the same ports), and
`Plant.solve` takes `max_steps`. **Known limitation:** the *coupled* BSM1
plant transient is severely stiff — the non-smooth Takács flux (the `min` /
`where` / `clip` kinks) plus the strong recycle feedback drive the adaptive
stepper into tiny steps, so the full plant does not yet integrate
efficiently to steady state (the default uses `IdealClarifier`). Making the
full plant tractable needs further numerical hardening (smoothing the flux
kinks and/or warm-starting from a steady-state initialization), tracked
separately.

The first plant-wide demonstration target is **BSM1** (Copp 2002 / Alex
2008) — built by `aquakin.plant.bsm.build_bsm1()`. Three synthesised
influent CSVs (dry / rain / storm) ship under
`aquakin/plant/bsm/data/` and load via `load_bsm1_influent()`. The
synthesised files match BSM1's *statistical* profile but are not the
canonical IWA files; for quantitative comparison to Alex 2008's
published EQI / OCI values, users should replace them with the
official files.

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
`pytest -m "not validation"` across Python 3.10/3.11/3.12; a separate
`validation` job runs `pytest -m validation` (the slower published-data
checks) on 3.12. A green CI is the merge gate. When adding a runtime
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
