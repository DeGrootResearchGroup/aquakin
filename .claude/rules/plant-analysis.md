---
paths:
  - "aquakin/plant/calibrate.py"
  - "aquakin/plant/identifiability.py"
  - "aquakin/plant/measures.py"
  - "aquakin/plant/sensitivity.py"
---

# Rules — plant calibration, identifiability and measured channels

`Plant.calibrate`, `plant_fisher_information` / `plant_profile_likelihood` /
`plant_rank_sensor_sets`, and measured channels (`plant/measures.py`). Split out
of `.claude/rules/plant.md`. The reactor machinery these reuse is described in
`.claude/rules/integration-calibrate.md`.

**Calibration** — `plant.calibrate(observations, t_obs, free_params, target=,
observed_channels=, y0=, ...)` ([`plant/calibrate.py`](aquakin/plant/calibrate.py),
bound as a method) MAP-fits by-name plant parameters (`"<model>.<param>"`)
against an output stream. It **reuses the reactor-calibration machinery**
(`integrate/calibration/`, imported through `integrate/seam.py`:
`_CalibrationProblem`, `_FitConfig.from_configs`, and `_run_fit` — the same driver `aquakin.calibrate` runs, so it resolves only its
problem and forward model and must not grow its own fit loop) through the
forward-model seam — the
plant supplies a `_PlantForwardModel` (`plant.solve` + stream reconstruction; the
cap-free stable adjoint by default, so the reverse gradient is finite) and a
`_PlantParamNamespace` (adapts `plant.parameter_index` / per-model transforms &
priors to the `model` interface the generic layer expects). Fits kinetic
parameters against **one or more output streams** — `observables=[PlantObservable(
stream, channels), ...]` (exported `aquakin.plant.PlantObservable`; the
`_PlantForwardModel` reconstructs each stream and concatenates the observed
channels, so the observation columns run in observable order), or the single
`target=`/`observed_channels=` sugar. **Assembled-state initial conditions** can
be fit alongside the parameters via `free_ic=FreeICConfig(species=["unit.species",
...])` (species also accept `(unit, species)` pairs) — each names a slot of `y0`
on a concentration unit
(CSTR / digester), resolved to a flat-state index (`_state_layout[unit][0] +
species_index`) and fit in log space through the *same* reactor free-IC machinery
(`m_ic`/`ic_species_idx`/`ic_center_full` on `_CalibrationProblem`); the fitted
state comes back as `result.C0_fitted[0]` / `result.ic_named[0]`, and the
`_PlantForwardModel` uses the per-dataset `C0` as `y0` when `use_c0_as_y0` (free
ICs *or* multi-batch). A **joint multi-batch fit** — several plant runs from
different initial states sharing the parameters, list-valued `observations` /
`t_obs` / `y0`, data terms summed — is the plant analogue of the reactor
multi-batch and reuses the generic multi-dataset machinery (n_datasets > 1,
per-dataset `C0_base`); `free_ic` + multi-batch are not yet combinable, and
`predictive_band` remains reactor-only. The rate bounds (model-declared via
`_PlantParamNamespace.parameter_bounds`, `OptimizerConfig(bounds=,
param_halfwidth=)`), per-pool free-IC bounds and `fixed_total`, `sigma=
"estimate"` and `at_bound` all go through the same shared helpers as
`aquakin.calibrate` (`_resolve_rate_bounds`, `_resolve_ic_coords`,
`_initial_point`, `_fit_result`, ...), so they behave identically. Covered by
`tests/integration/test_plant_calibrate.py` (single- and multi-stream synthetic
recovery, a joint rate + initial-condition recovery, a multi-batch shared-rate
recovery, a full **BSM1** muH recovery through the stiff recycled plant, an
`at_bound` + `sigma="estimate"` fit and a `fixed_total` split recovery (slow),
stubbed-runner checks of the bounds / free-IC / sigma wiring, and the
finite-through-the-plant gradient in the fast gate).
**Identifiability** — `plant_fisher_information` / `plant_profile_likelihood`
([`plant/identifiability.py`](aquakin/plant/identifiability.py), exported from
`aquakin.plant`) are the plant versions of the reactor tools. Observables are
resolved by `plant/calibrate.py`'s `_resolve_observable_columns` (shared with
`_resolve_plant_problem`), free names by `_check_free_params`. **Measured
channels** (`plant/measures.py`): a channel is a species, a keyword (`Q`, `T`,
`COD`, `TKN`, `NOx`, `TN`, `TP`, `TSS`), a `Measure` (species weights,
`keywords=` weights on those measurements, `flow`, `temperature`) or a
**measure expression** — the same weighted sum as text, `"BOD5 = 0.65*[SS] +
0.65*[XS]"`, so a custom channel fits in a study file and in the editor's
study dialog (typed into the channels combobox, which takes free text).
`parse_measure` reads one (`MeasureSyntaxError`, a `ValueError`, with the
position in the text the user wrote, not in the body after the name);
`measure_map` calls `as_measure` on every channel **once**, up front, so every
consumer below — and the `dW` probe, which re-weighs the same channels — works
on resolved `Measure`s. `channel_name` is the results column (the name, never
the defining text); use it anywhere channels are named, not `isinstance(c,
Measure)`. The grammar is deliberately **affine**: `*` needs a number on one
side, `/` a number on the right, and a constant term is refused — the
ranking's `y @ M` and `_union_columns` are only valid because a channel is
linear in the stream, so **do not relax this**. A `Measure`'s keywords resolve
through `_keyword` (not `_channel`, so a keyword is never shadowed by a
same-named species) and therefore ride the existing `theta0`/`dW`
central-difference probe — a channel built on `TN` follows `i_XB` exactly as
`TN` does (`_follows_composition` is what arms the probe; extend it alongside
any new parameter-dependent keyword). Syntax is validated in the study schema
(`_Observable._readable_measures`) when the study is **written**; species are
resolved against the stream's model at **run** time, as an unknown plain
channel already is (a `KeyError` naming the stream). `measure_map(model,
channels)` gives a `MeasureMap` (`W`, `q`, `t`, and the composition parameters
the weights depend on with `theta0`/`dW`: a central-difference probe, exact for
the affine content fractions, linearized otherwise, e.g. ASM2d's
`fMeP_PO4_MW`). `_resolve_observable_columns` turns it into `(endpoint,
_Measurement)` with the parameters addressed in the plant flat vector, and
every consumer reads `measured(C, Q, T, measurement, p) = C @ W(p) + Q q + T t`
(`W(p)` follows the run's parameters, so a free `i_XB` is seen through TN's
weights as well as the state). Temperature comes from the stream's `T` scalar
(`StreamSeries.T`, filled by `_stream_sweep` in the same pass as `(Q, C)`); a
`T` channel on a stream with none raises `ValueError`. The ranking's
`_union_columns` observes each stream whole (species, flow, temperature when
any set reads it) with per-set maps `(M, idx, theta0, dM)`; `_set_channels`
adds `y_all @ dM` for a free composition parameter. Classification:
COD clips negative (electron acceptor) entries and drops `S_h2`/`S_ch4`;
`SN2`, `SN2O` and `SNO` when `SNO3` exists (nitric oxide) are in no nitrogen
channel; `S_gas_*` is in nothing. TSS is a model's `XTSS`/`XSS` species or
the ASM1 `0.75 × ASM1_TSS_SPECIES`; a model with neither raises. A species
named like a keyword raises (use a `Measure`). `composition_table` follows a
file model's `extends:` chain (`composition_base`, via `_aquakin_source`) to a
shipped table when the species match, including its unit factors (ADM1's
kg-based contents). Unknown-channel errors keep the historic "Unknown observed
channel" wording. The Fisher
forward mode takes `S` from `Plant.solve_sensitivity` and maps it through the
stream reconstruction (`outputs_at`, vmapped) with one `jax.linearize`, so it
includes the reconstruction's direct `dg/dθ`; the reverse mode is `jacrev` of
`_PlantForwardModel.solve_trajectory` — the solve `Plant.calibrate`
differentiates, which is why it equals the Gauss-Newton Laplace Hessian at a
`loss="nll"` fit (pinned to 1e-8). Multi-run designs follow the calibrate
multi-batch convention (list-valued `t_obs` selects it); `plant` may also be a
list, one per run, for runs at different influents. The profile wraps
`calibrate_plant` per grid point through the private `_compiled_cache` /
`_ic_start` hooks (now on `calibrate_plant` as on `calibrate`); for that the
plant problem's `data_key` fingerprints the loss, sigma, observations and
times. `plant_rank_sensor_sets` (with `SensorSet`) is the plant
`rank_experiments`: `_union_columns` resolves every candidate's streams to one
union, observed whole (all species, then the flow), keyed by the parsed
`(unit, port)` so two spellings share columns; one `_PlantDesign.sensitivity`
call per run per parameter draw then serves every candidate through its own
linear map `M` (`y @ M`, `einsum("tok,os->tsk", dy, M)`), exact because every
measured channel is linear in the stream. It reuses
`experiment_design.py`'s `_criterion` / `_subset_indices` /
`_LARGER_IS_BETTER` / `CandidateScore`. `_PlantDesign` (runs, free parameters,
transforms, priors, the forward/reverse dispatch) and `_scaled_rows` (noise,
mask, relative sigma, transform chain) are shared with
`plant_fisher_information`, so a candidate's score is exactly the criterion
of its own `plant_fisher_information`. Covered by
`tests/integration/test_plant_identifiability.py` (closed-form CSTR
information in both modes, the hand-built `solve_sensitivity` Jacobian on the
CSTR and on BSM1 (slow), forward = reverse, the Laplace-Hessian equality,
identified / flat profiles (slow), the profile's per-point wiring into
`calibrate_plant` with it stubbed, and each ranking score equal to its own
set's `plant_fisher_information` criterion from one counted solve) plus `tests/unit/test_identifiability_helpers.py`
(the shared sweep / polish / noise helpers on stub fits).
