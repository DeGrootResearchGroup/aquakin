# Public API

The full public API surface with worked usage. Reference; read on demand.
The compiled-solve caching section is in `.claude/rules/integration.md`.


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

