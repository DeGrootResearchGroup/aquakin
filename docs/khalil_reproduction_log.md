# Khalil model-improvement sequence (JRN-055 reproduction log)


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

