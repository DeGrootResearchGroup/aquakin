---
paths:
  - "aquakin/plant/takacs.py"
  - "aquakin/plant/settling.py"
  - "aquakin/plant/clarifier.py"
  - "aquakin/plant/primary_clarifier.py"
  - "aquakin/plant/separators.py"
---

# Rules — settling and separation units

The Takács clarifier and the shared separator helpers. Split out of `.claude/rules/plant.md`.

`TakacsClarifier` (10-layer 1-D Takács 1991 model). With
`interface_flux="takacs"` its settling physics are the benchmark's, verified in
isolation at BSM1 solids loading: the clarification-zone flux limiting (above the feed, the downward flux is
limited by the layer below only when that layer exceeds `X_threshold`) and
the per-species flux apportioning (each species settles at the bulk
velocity, `flux_tss · X_k/TSS`, conserving total settleable solids) produce
a monotone sludge blanket, a strongly clarified effluent, a thickened
underflow, and tight solids mass balance (verified to machine precision
against an independent port of the reference BSM1 settler derivative in
`tests/validation/test_takacs_vs_bsm1_reference.py`). `build_bsm1` and `build_a2o` select it
by default (`use_takacs=False` gives the `IdealClarifier`; both expose the same ports),
and `Plant.solve` takes `max_steps`. By default a bare `TakacsClarifier` does not
hold the **soluble** species in the settler — they pass straight through (overflow = underflow = feed,
no holdup), the common simplification. The opt-in **`soluble_holdup=True`** makes
each soluble a per-layer well-mixed state advected by the bulk flow (convection
only, no settling), so the clarifier's liquid volume (~`area·height`) damps the
soluble effluent signal — the ten-layer soluble model of the BSM1 reference
`settler1dv4` (`MODELTYPE=0`, verified against its source: each soluble is
advected layer to layer, no settling), which the BSM2 `settler1dv5_bsm2` has
too (`SNH_1..SNH_10` etc. per layer, also `MODELTYPE=0`). It is **independent**
of `composition_mode`. **Both reference settlers are `"lumped_tss"`**: 10 TSS
states, outlet particulates = feed composition × `gamma = x_boundary/TSS_feed`
(checked in both `settler1dv4.c` and `settler1dv5_bsm2.c`, R2019b; the latter
has 120 states = 10 TSS + 10×10 solubles + 10 temperature). `"per_species"`
(per-layer composition memory) has no reference counterpart. `build_bsm2` and
`build_bsm1` both default to the reference (`"lumped_tss"` with
`soluble_holdup=True`) **except for the interface flux** (Godunov, below);
`build_bsm2` defaulted to `"per_species"` without holdup before v0.2.0. Earlier docs said `per_species` "matches settler1dv5"; it
does not. The soluble holdup is a tail block of shape
`(n_layers, n_soluble)` appended to the state (so the particulate layout /
`state_size` are unchanged when off), orthogonal to `composition_mode`. **It
leaves every steady state unchanged** — a non-reacting soluble's only transport
is convection, whose fixed point is the uniform feed concentration (overflow =
underflow = feed), verified in `tests/integration/test_takacs.py` — so it only
matters under a dynamic influent, where it smooths the effluent ammonia
peaks/troughs. This is the structural cause of aquakin's wider dynamic-BSM2
effluent-NH4 distribution vs the reference: the reactors agree to corr 0.99 but
the pass-through settler does not damp the soluble signal the way BSM2's
soluble-carrying settler does (the JRN-056 dynamic validation). `build_bsm2`
enables it by default (`settler_soluble_holdup=False` turns it off).
**`interface_flux`** (`"godunov"` default | `"takacs"`; builders
`settler_interface_flux`, BSM1 and BSM2) picks the settling flux between layers
(`TakacsClarifier._interface_flux`). **The Godunov default is a deliberate,
documented departure from the benchmark settlers** (project owner's decision,
2026-11-02; stated in the `TakacsClarifier` / `build_bsm1` / `build_bsm2`
docstrings, `docs/plants.md` "The secondary clarifier and the benchmark", the
README and the CHANGELOG); `"takacs"` reproduces them, and the reference-port
validation `test_takacs_vs_bsm1_reference.py` pins it. `"godunov"` is the Godunov flux of the settling
conservation law for the unimodal `F(X) = v_s(X)·X` (Bürger, Diehl & Nopens 2011),
at every interface (no `X_threshold`): equal to the Takács minimum when the upper
layer is the more dilute, the maximum of `F` between the two concentrations when it
is the denser (`F` at its peak, located by `_flux_peak`'s ternary search, when they
straddle it). In BSM2's blanket, layers 2–5 sit within ~1% of each other on the
rising branch of `F` and swap order hundreds of times a day; the Takács rule
switches flux at every swap (kinks, pulses ±10–15%, solver failures), the Godunov
flux is the upper layer's either way. That is why `"godunov"` is the default:
it cuts the solver's attempts several-fold on dynamic BSM2 while the effluent
means stay within 0.07% of Takács and the constant-influent steady state is
identical (layers 2–5 are equal there, where the rules agree). Measurements:
`.claude/notes/plant-sensitivity-log.md` and `scripts/probe_settler_accuracy.py`.

**Shared separator/clarifier helpers.** The separator/clarifier family resolves
its settling / particulate / TSS species lists through the shared
[`species_mask` / `species_indices`](aquakin/plant/_constants.py) helpers, which
enforce **one** missing-species policy: a species named in a unit's config but
not in the model **raises** a clear `ValueError` at construction (it does *not*
silently drop it — the old, contradictory behaviour that under-settled without
warning). So a non-ASM1 model must name its own `settling_species` /
`particulate_species` / `tss_species` rather than rely on the ASM1 defaults.
**A model that tracks its TSS as a state (`XTSS` / `XSS`, the ASM2d and ASM3
families) takes its settler `tss_factors` from `_constants.lumped_tss_weights`**
(solved from the stoichiometry: each particulate at its `iTSS` factor, the TSS
state at 0), as `build_a2o`, `build_bsm1` (unless given `tss_species`) and
`design.sludge_metrics` do; the Takács
`tss_factors.get(sp, 1.0)` fallback otherwise adds COD, g P and the TSS state
together into the TSS that sets the settling velocity. A shipped composition
for such a model sets the TSS state with `with_lumped_tss`, never by hand. TSS
from a weight vector is `tss_concentration(C, tss_vec)`; the Q-weighted multi-inlet
feed is `mixed_feed(inputs, names)` and the total inflow `total_flow(...)` (the
flow siblings of `mixed_scalars`, in `plant/streams.py`); and the ideal secondary
clarifier's fixed-capture-fraction partition is `split_by_capture(...)`. The
`TakacsClarifier` keeps its own per-layer gather (a hot loop) and the
`PrimaryClarifier` its state-partition CSTR formulation — they share the species
policy but not the split kernel, whose physics differs. When adding a separating
unit, reuse these rather than re-deriving a mask/split.
