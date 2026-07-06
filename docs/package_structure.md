# Package structure

Annotated file tree of the `aquakin` package. Reference; read on demand.


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
│   │   ├── model.py               # CompiledModel dataclass + compile()
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
│   │   ├── model_spec.py          # Pydantic models
│   │   ├── inheritance.py           # model.extends: merge a base + add/modify/remove
│   │   └── loader.py                # YAML -> (resolve extends) -> Pydantic -> CompiledModel
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
│   │   ├── _qmc.py                   # shared unit sampler + fn(x)->output eval
│   │   │                            #   helpers (reuses dgsm's Sobol QMC)
│   │   ├── monte_carlo.py           # monte_carlo(): uncertainty propagation
│   │   │                            #   (distribution samplers -> inverse-CDF)
│   │   ├── scenarios.py             # compare_scenarios() + kpi_comparison():
│   │   │                            #   scenario / standardized-report KPI tables
│   │   ├── design.py                # optimize_design() + Constraint: AD-gradient
│   │   │                            #   constrained design NLP
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
│   ├── models/
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
│   │   #   dormant full-WATS aerobic pieces by models/_make_khalil_paper.py;
│   │   # wats_sewer_khalil_thesis is generated from wats_sewer_extended.yaml by
│   │   #   models/_make_khalil_thesis.py; the structural variants by
│   │   #   models/_make_khalil_variants.py;
│   │   # wats_sewer_khalil_paper_balanced_biofilm is generated by
│   │   #   models/_make_khalil_balanced_biofilm.py -- it splits the 3 composite
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
│   │   #   models/_make_khalil_balanced_biofilm_biomass.py -- biomass is an
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
│   │   #   models/_make_khalil_balanced_biofilm_multispecies.py from the
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
│       │                            #   expressions (model.check_units); distinct
│       │                            #   from core/units.py, which only formats units
│       └── rtd.py                   # RTD analysis (E-curve, Morrill index)
│
├── tests/
│   ├── unit/
│   │   ├── test_parser.py
│   │   ├── test_nodes.py
│   │   ├── test_loader.py
│   │   └── test_model.py
│   ├── integration/
│   │   ├── test_batch_simple.py     # validates against analytical solution
│   │   └── test_pfr_simple.py
│   ├── validation/
│   │   ├── test_bromate_vongunten.py# validates against published data
│   │   ├── test_adm1_bsm2_steadystate.py # ADM1 vs published BSM2 AD steady state
│   │   └── test_takacs_vs_bsm1_reference.py # Takács settler vs published BSM1 settler derivative
│   └── fixtures/
│       └── simple_model.yaml      # minimal 2-species toy model for unit tests
│
├── examples/
│   ├── batch_bromate.py
│   ├── lagrangian_demo.py
│   ├── sensitivity_demo.py
│   ├── bsm1_dry_weather.py            # BSM1 open-loop steady state
│   ├── bsm1_target_srt.py             # hit a target sludge age (SRT) by solving for Qw
│   ├── bsm1_dynamic_influent.py       # BSM1 dry-vs-rain dynamic influent (warm-started)
│   ├── bsm2_steady_state.py           # BSM2 two-model open-loop steady state
│   ├── bsm2_seasonal_temperature.py   # BSM2 cold->warm nitrification effect
│   ├── dgsm_sensitivity_screen.py     # DGSM global sensitivity, forward==reverse
│   ├── wats_nitrate_dosing_calibration.py  # synthetic sewer rate recovery (calibrate + Laplace)
│   ├── bsm2_ghg_cost_report.py     # GHG (N2O/CO2e) + cost reporting + scenario KPI table
│   ├── event_handling.py           # located events: scheduled re-dosing + terminal cut-off
│   └── adjoint_speed_benchmark.py  # stable_adjoint vs capped jax_adjoint timing
│   # NOTE: the wats_sewer_extended batch-fitting / calibration / sensitivity scripts and
│   # their measurement data live in the separate paper-reproduction repository,
│   # not here (this repo ships only the reusable library + models).
│   # wats_nitrate_dosing_calibration.py is a self-contained *synthetic* demo of
│   # the calibration API, not the paper reproduction.
│
├── docs/
│   ├── index.md
│   ├── model_format.md
│   └── adding_models.md
│
├── pyproject.toml
├── README.md
├── CLAUDE.md                        # this file
└── LICENSE
```

**Key structural rules:**
- `core/` has no Pydantic dependency — only dataclasses and JAX
- `schema/` is the only module that imports Pydantic
- `models/` ships with the package, accessible via `importlib.resources`
- Unit tests use only `tests/fixtures/simple_model.yaml`, never `ozone_bromate.yaml`

---

