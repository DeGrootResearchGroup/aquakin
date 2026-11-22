# CLAUDE.md — aquakin Project Briefing

This is the slim root briefing for the `aquakin` library: orientation,
conventions, the **Engineering Standards**, testing, the development workflow,
and the **Post-Change Checklist**. Read it before writing any code.

Detailed subsystem guidance is **split out of this file** so it loads only when
relevant rather than every session:

- **Path-scoped rules** under `.claude/rules/*.md` carry `paths:` frontmatter and
  auto-load when you read or edit files matching their glob.
- **On-demand docs** under `docs/*.md` are reference / investigation logs; read
  them when a task needs the detail.
- **Claude-facing notes** under `.claude/notes/*.md` are dated investigation logs
  (measurements, probes, A/B results) kept out of `docs/`; they never auto-load.

When you change a subsystem, the matching rule below is the authoritative
guidance for it — keep it updated as part of the change (this file's Post-Change
Checklist still governs). How to keep these files useful is in **Maintaining the
Claude-facing files** at the end.

> **A path-scoped rule only loads for its own paths.** Writing
> `aquakin/plant/measures.py` loads `plant.md` and *not* `core.md`, so guidance
> that is really cross-cutting can sit unread in a neighbouring file. Before
> adding machinery that feels generic — a parser, a tokenizer, a units
> conversion, a cache, a numeric guard — **search for an existing one first**
> (`grep` the package, not just the directory you are in) and read the rule
> that owns it. Three copies of one tokenizer got written this way (#699); the
> cure is `core/lexer.py` plus the pointer now in `plant.md`.

## Index — where the detail lives

**Path-scoped rules (auto-load on matching files):**

| Rule | Loads for | Covers |
|---|---|---|
| `.claude/rules/tests.md` | `tests/**` | what makes a test able to fail, gradient tests, shared fixtures, the tiers and what CI actually runs |
| `.claude/rules/core.md` | `aquakin/core/**`, `aquakin/utils/**` | AST rate evaluation, parser, **the one expression tokenizer (`core/lexer.py`) every expression language shares**, vectorized kernel, `SpatialConditions`, parameter namespacing |
| `.claude/rules/core-speciation.md` | `core/ph_solver.py`, `core/speciation.py` | charge-balance pH solver, derived conditions, the `speciation:` block |
| `.claude/rules/core-precipitation.md` | `core/precipitation*.py` | mineral precipitation: rate law, the `precipitation:` block, the algebraic equilibrium mode |
| `.claude/rules/integration.md` | `aquakin/integrate/**` | solver choice, operator splitting, located events, compiled-solve caching |
| `.claude/rules/integration-calibrate.md` | the calibration / Fisher / profile / experiment-design modules, `plant/calibrate.py`, `plant/identifiability.py` | `calibrate()`'s internal structure and the helpers `Plant.calibrate` shares |
| `.claude/rules/integration-differentiation.md` | `integrate/_common.py`, the adjoint / forward-sensitivity modules, `plant/sensitivity.py`, `plant/colored.py` | differentiating stiff models (`dtmax`, adjoints, discrete adjoint, forward sensitivities) |
| `.claude/rules/schema.md` | `aquakin/schema/**`, model `*.yaml` (shipped, fixtures, examples) | YAML model format + Pydantic schema rules (stoichiometry expressions, `extends:`, `auto`, `composition:`, `speciation:`, `precipitation:`) |
| `.claude/rules/plant.md` | `aquakin/plant/**` | plant-wide simulation: the `Unit` contract, wiring, the plant clock, introspection, shipped units, shared helpers and constants |
| `.claude/rules/plant-solve.md` | `plant/plant.py`, `recycle.py`, `colored.py`, `solve_cache.py`, `steady.py`, `integrate/steady.py` | coloured Jacobian, solve caches, recycle resolution and the cached recycle / flow maps |
| `.claude/rules/plant-analysis.md` | `plant/calibrate.py`, `identifiability.py`, `measures.py`, `sensitivity.py` | plant calibration, Fisher / profile / sensor ranking, measured channels |
| `.claude/rules/plant-audits.md` | `plant/units_check.py`, `balance.py`, `metrics.py` | `plant.check_units()`, `plant.mass_balance()` |
| `.claude/rules/plant-ifas.md` | `plant/ifas.py`, `integrate/biofilm*.py` | the IFAS / MBBR biofilm unit |
| `.claude/rules/plant-settling.md` | `plant/takacs.py`, `settling.py`, clarifier / separator modules | the Takács clarifier, shared separator helpers |
| `.claude/rules/plant-spec.md` | `plant/spec.py`, `catalog.py`, `time_expr.py`, `influent.py`, `aquakin/__main__.py` | flowsheet spec, unit catalogue, time expressions, influent records / patterns |
| `.claude/rules/models.md` | `aquakin/models/**` | model authoring: pointers to the catalog, the schema rules, and the `_make_*` generators |
| `.claude/rules/study.md` | `aquakin/study/**` | studies: the study folder layout, results keyed by name, staleness, freezing / unfreezing, the `aquakin study` command |
| `.claude/rules/ui.md` | `aquakin_ui/**`, `tests/ui/**`, `tests/browser/**`, `ui-widget/**` | the browser flowsheet editor `aquakin-ui`: the independence rule (never imports aquakin or JAX), the `aquakin serve` worker, the document / forms / page split, the Vue Flow widget and its sync rules |

**On-demand docs (read when needed):**

| Doc | Contents |
|---|---|
| `docs/khalil_reproduction_log.md` | the JRN-055 Khalil model-improvement sequence |
| `docs/package_structure.md` | the annotated package file tree |
| `docs/plant_performance.md` | the dynamic-solve performance log (the stiffness-bound regime and its levers) |
| `docs/ci.md` | the CI architecture (sharding, gates, heavy runner, labels, the OOM memory watchdog) |

**Claude-facing notes (`.claude/notes/`, never auto-load — dated investigation logs, read when a
task needs the detail):**

| Note | Contents |
|---|---|
| `.claude/notes/studies-design.md` | the studies design and the owner's decisions (live / frozen, replace with a warning, version notes, unfreezing discards results) |
| `.claude/notes/plant-sensitivity-log.md` | `Plant.solve_sensitivity` corrector measurements (#654), the lean loop's stage-Newton drift diagnostic and its Anderson acceleration |

**Published documentation site** (Sphinx/MyST → Read the Docs; built from
`docs/conf.py`, gated `-W` in CI). These are **user-facing** — keep dev-internal
detail (bug history, issue numbers, SUMO spreadsheets, test/script paths) out of
them:

| Page | Contents |
|---|---|
| `docs/index.md` | landing page + toctree (Guide / Authoring / Reference) |
| `docs/getting_started.md`, `docs/reactors.md`, `docs/plants.md`, `docs/sensitivity_and_calibration.md` | the task-oriented user guide |
| `docs/model_format.md`, `docs/adding_models.md` | YAML model-authoring reference |
| `docs/model_catalog.md` | user-facing catalog of every shipped model (what it is, provenance, size, how to load) |
| `docs/public_api.md` | task-grouped map of the public API, linking into the generated `api.md` |

---

## Project Overview

`aquakin` is an open source Python library for modelling reactive scalar
transport in aqueous environmental systems. It provides a modular, runtime-
configurable kinetics engine that can be coupled to any flow solver.

Both **chemistry** (ozonation, advanced oxidation, chlorine decay, ...) and
**biology** (activated sludge models, anaerobic digestion, ...) are in scope.
> The user-facing per-model catalog (what each shipped model is, its provenance,
> size, and how to load it) is the published **`docs/model_catalog.md`**; the
> dev-internal model-improvement history (documented corrections) lives in
> **`docs/khalil_reproduction_log.md`** and the git log, not the published catalog.

---

## Design Goals

- Reaction models defined at runtime via YAML — no recompilation required
- Full automatic differentiation (AD) throughout via JAX
- JAX-native stiff ODE integration via Diffrax
- Clean separation between flow solver and kinetic system
- Safe, introspectable rate expression evaluation via AST (no `eval()`)
- Graduate-student-friendly authoring experience for model files

---


## Technology Stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python | Flexibility, ecosystem |
| Numerical backend | JAX | AD for free, jit compilation |
| ODE integration | Diffrax | JAX-native stiff solvers, adjoint support |
| Schema validation | Pydantic | Clear load-time errors for malformed YAML |
| Runtime data model | Python dataclasses | JAX-friendly, no Pydantic overhead at runtime |
| Model format | YAML | Human-readable, supports inline comments for literature citations |
| Rate expression evaluation | Custom AST + recursive descent parser | Safe, differentiable, introspectable |

---


## Architecture

### Two-Layer Data Model

**Layer 1 — Pydantic schema (load time)**
Parses and validates YAML model files. Produces clean Python objects with
clear error messages for malformed input. Pydantic is used only during loading
— it never appears in the runtime hot path.

**Layer 2 — Compiled runtime (CompiledModel dataclass)**
JAX-friendly dataclasses built from the validated schema via a `compile_model()`
step. This is what the integrators and rate functions operate on. No Pydantic
dependency at runtime.

### Core Data Flow

```
YAML file
   ↓  loader.py (Pydantic validation)
ModelSpec
   ↓  compile_model()
CompiledModel  ←→  SpatialConditions
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
in `CompiledModel`.

---


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


## Testing Architecture

### Three Layers

**Unit** — test individual components in isolation, fast, run on every change.
**Integration** — test full YAML → solution pipeline against analytical solutions.
**Validation** — test scientific correctness against published experimental data,
marked `@pytest.mark.validation`, run separately.

How to write a test that can actually fail, what each kind of change needs, and
which tier a test lands in are in **`.claude/rules/tests.md`** (auto-loads under
`tests/`; read it before adding tests from elsewhere).

### Running

The markers are `validation`, `slow`, `heavy` and `xheavy` (definitions in
`pyproject.toml`; the CI jobs that run each are in `docs/ci.md`).

```bash
pytest tests/unit tests/integration tests/ui -m "not validation and not slow"   # the PR merge gate
pytest -m slow                             # the multi-minute stiff/plant solves
pytest -m validation                       # validation suite (published data)
pytest                                     # everything
pytest tests/browser                       # the editor in Chromium (`browser` extra; non-required CI job)
```

A test under `tests/validation/` must carry `@pytest.mark.validation`, or no CI
job selects it.

### Canonical Integration Test

First-order decay `A → B` with rate `k * [A]` has the analytical solution
`[A](t) = [A]₀ * exp(-k*t)`. This is the primary integration test and must
always pass. It lives in `tests/fixtures/simple_model.yaml` and
`tests/integration/test_batch_simple.py`.

### AD Correctness Test

Every integration test suite must include an explicit test that `jax.grad`
flows through `reactor.solve()` and gives **the right value**: checked against
the closed form where one exists, otherwise a central finite difference or a
second AD route. Finite and NaN-free is necessary, not sufficient — a gradient
that is finite and wrong passes a finiteness check.

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
reader cannot see. Genuine scientific provenance belongs in a model YAML's
`references:` block as a proper literature citation, not scattered through code
comments.


## Spelling Convention

Write Canadian English in all prose: documentation pages, comments, docstrings,
error messages, and commit and PR text. Use **-ize** and **-yze** endings
(*optimize*, *optimizer*, *summarize*, *recognize*, *analyze*), with
Canadian/British forms elsewhere (*colour*, *behaviour*, *centre*, *metre*,
*litre*, *modelling*). Do not change identifiers, file names, or quoted text to
match. The public API already uses -ize (`optimize_experiment`,
`optimize_design`). Existing -ise spellings in older files are converted as
they are edited.


## Docstring Convention

All functions, classes, and methods use **NumPy docstring format**:

```python
def solve(self, C0, params, t_span, t_eval=None):
    """
    Integrate the reaction model over a time span.

    Parameters
    ----------
    C0 : jnp.ndarray
        Initial concentration vector, shape (n_species,).
    params : jnp.ndarray
        Rate constant vector, shape (n_params,). Use
        ``model.default_parameters()`` as a starting point.
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
    >>> reactor = aquakin.BatchReactor(model, conditions)
    >>> sol = reactor.solve(model.default_concentrations(),
    ...                     model.default_parameters(),
    ...                     t_span=(0.0, 600.0))
    >>> sol.C_named("BrO3-")
    """
```

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


## Engineering Standards

These hold for every change. They exist because each one has already cost this
codebase something; the examples are real.

### One implementation of each thing

- **Search before you write.** Before adding a helper, a validation, a parser,
  a fixture or a constant, `grep` for it. The shared places are
  `integrate/_common.py` (solver / controller builders, atol, validation, the
  solution mixins), `integrate/_transforms.py`, `plant/_constants.py`,
  `plant/_builder_support.py`, `plant/units.py` (the `Unit` contract and
  capability Protocols), `core/lexer.py` and `core/precedence.py` (the one
  tokenizer and precedence table for every expression language),
  `core/kernels.py` (guarded numerics) and `tests/conftest.py`.
- **A second caller means a shared function, in the same change.** Do not copy
  a block and adapt it. If the two callers live in different packages, put the
  shared piece in the lower one (`core` < `schema` < `integrate` < `plant` <
  `study`) and give it a name without a leading underscore.
- **Two paths that must agree call one function**, they are not "kept in step"
  by a comment. Reactor and plant calibration, the scalar and vectorized rate
  evaluators, the rate-expression and time-expression languages, a spec reader
  and its writer: when you change one, find the other. (`plant/time_expr.py`
  once bound `sqrt` / `log` to the bare `jnp` functions while the rate language
  used the guarded ones in `core/kernels.py`, and three tokenizers were written
  before `core/lexer.py` replaced them.)
- **Units compose the canonical term rather than re-deriving it.** A reactor
  unit's reaction term is `model.dCdt(...)`, which applies the negative-state
  clip and the positivity limiter; writing `stoich.T @ rates` in a unit drops
  both.
- **Imports point downward.** `core` does not import `integrate` or `plant`;
  `integrate` does not import `plant`; `aquakin_ui` imports neither `aquakin`
  nor `jax`. A function-level import that exists to dodge a cycle is a sign the
  shared piece is in the wrong module: move it instead of adding another.
- **Dispatch on a declared interface.** A new per-unit capability is a
  `runtime_checkable` Protocol in `plant/units.py`, detected with `isinstance`;
  not a `hasattr` / `getattr(unit, "x", None)` probe, and never a check for a
  private attribute of one unit class.

### No shortcuts

- **Fix the cause.** Do not special-case the failing input, clamp the symptom,
  or wrap a call in `try` / `except Exception` so the failure disappears. Catch
  the specific exception you expect and let the rest propagate.
- **Never change a test to make it pass** (loosen a tolerance, delete an
  assertion, add a skip or `xfail`) unless the test itself was wrong, and then
  say so explicitly. See `.claude/rules/tests.md`.
- **No suppressions without a reason on the line**: `# noqa`, a new ruff
  `ignore`, `# pragma: no cover`, `warnings.simplefilter("ignore")`.
- **No placeholders**: no `TODO` / `FIXME`, stub bodies, unused parameters kept
  "for later", or backward-compatibility shims nobody asked for. If part of the
  task is not done, say so in the hand-off instead of leaving a marker.
- **Finish the change.** A new option reaches every entry point that should
  take it (reactor *and* plant, forward *and* reverse mode, the spec, the
  catalogue, the study form), or the hand-off lists which ones it does not.
  Delete what the change makes dead.
- **Say what you verified.** State which tests you ran and their result. If
  something could not be run (no JAX in the environment, a `slow` solve), say
  that; do not describe unrun code as working.

### Keep modules and functions small enough to read

- Do not grow the large modules (`plant/plant.py`, `integrate/calibrate.py`,
  `aquakin_ui/app.py`, `integrate/_common.py`, `core/model.py`). New behaviour
  goes in its own module as free functions taking the object (the pattern
  `plant/sensitivity.py`, `plant/calibrate.py` and `plant/spec.py` already use,
  bound onto `Plant` as methods), not as more methods on the class.
- A function past about 100 lines of body, or with more than about eight
  parameters, gets split or takes a config dataclass before it gets longer.
- A behaviour-preserving refactor is its own commit (ideally its own PR) and
  changes no test.

---

## Development workflow

Code review goes through GitHub PRs. The GitHub remote is HTTPS; on a
workstation the `gh` CLI is installed and authenticated, so branches can be
pushed and PRs opened directly with `git push` / `gh pr create`. (A Claude Code
cloud session has no `gh`; it uses its GitHub tools and its assigned branch
instead.) Commit or push only when the user asks. The convention is:

1. Commit changes on a sensibly-named branch (e.g. `fix-<thing>`,
   `add-<feature>`, `<area>-<change>`) — never on `main`, and not on the
   `claude/<worktree-name>` scratch branch. The main checkout is shared and its
   branch can switch underfoot, so do branch work in a worktree and commit early.
2. Push with `git push -u <remote> <branch>` and open the PR with
   `gh pr create --base main`. Work done in a worktree is already on the branch
   — just push it.

Run ruff before committing or pushing (Post-Change Checklist item 1). The
committed pre-push hook (`.githooks/pre-push`) runs the same gate; enable it
once per clone with `git config core.hooksPath .githooks`.

> The full CI architecture — pytest-split sharding, the fast PR gate vs the
> merge-to-main `slow`/`validation`/`heavy` suites, the memory/heavy-runner
> rationale, and the `full-ci` / `skip-heavy` labels — is in **`docs/ci.md`**.
> A green **`fast gate`** is the merge gate; apply the **`full-ci`** label to any
> PR touching convergence-sensitive code (integrators/adjoints, the PTC steady
> solver, plant assembly/recycle, model stoichiometry, the pH/precipitation
> solvers, or the metric/mass-balance kernels).

---

## Post-Change Checklist

After **every code change**, before considering the task complete, review and
act on the following:

1. **Lint & format** — Run ruff locally before committing and pushing any code
   change. The CI `lint (ruff)` job runs
   `ruff check aquakin aquakin_ui tests scripts examples` **and**
   `ruff format --check aquakin aquakin_ui` on every PR and is a **required merge
   gate**, so an
   unformatted file (or a lint error) fails CI — a common way a refactor that
   only shortens lines trips the format gate. Run both, from the repo root:
   - `ruff check aquakin aquakin_ui tests scripts examples` — must report no
     errors. In a test, an unused import or variable often means an assertion
     that was never written: look at it before deleting it.
   - `ruff format aquakin aquakin_ui` — auto-applies formatting in place (CI runs
     it with `--check`, which only verifies; running it without `--check` fixes
     the files). Re-run `ruff format --check aquakin aquakin_ui` to confirm it is
     clean. `tests/`, `scripts/` and `examples/` are linted but not yet
     formatted: do not run `ruff format` over them as part of an unrelated change.

   Ruff is pinned via the `lint` extra (`pip install -e ".[lint]"`); see the
   `[tool.ruff]` config in `pyproject.toml` for the rule set and per-file
   ignores. Not needed for docs/config-only changes that touch no `.py` files.

   **Type check** (a change under `aquakin_ui/`): run `pyright` from the repo
   root (`pip install -e ".[ui,typecheck]"`). The CI `typecheck (pyright)` job
   runs it in basic mode over `aquakin_ui` (`[tool.pyright]` in
   `pyproject.toml`) and it must report 0 errors. Fix a type error where the
   value comes from: narrow a value that can be `None` on that path, or correct
   a wrong annotation. A `# type: ignore` needs a reason on the line.

2. **Tests** — Every behaviour change comes with a test that would fail without
   it (`.claude/rules/tests.md` has the full table and the quality bar):
   - New node type → unit test in `test_nodes.py`
   - New public API function → integration test of its values and its errors
   - New built-in model → validation test against literature data
   - Bug fix → regression test, **seen to fail on the unfixed code**
   - Run the tests you touched, and the fast gate for a cross-cutting change;
     report what ran and what did not.
   - New and changed lines are expected to be covered: Codecov's patch status
     targets 100% (a signal, not a required check). When it is red, add the
     test, or state in the PR which lines are uncovered and why (a defensive
     branch no input can reach; code only `slow` tests exercise, on a PR
     without `full-ci`, since only the fast tier uploads on a normal PR). Never
     add `# pragma: no cover`, or a test that only executes lines, to make it
     green. Details in `docs/ci.md`.

3. **Duplication and leftovers** — Re-read the diff as a reviewer:
   - Did you copy a block that should be one shared function?
   - Did a change to one of two parallel paths (reactor / plant, scalar /
     vectorized, reader / writer) reach the other?
   - Is anything now dead, or left as a shim, a `TODO`, a broad `except`, or an
     unexplained suppression?

4. **Claude-facing files** — Does a rule under `.claude/rules/`, this file, or
   `docs/package_structure.md` now say something untrue or lack a rule the
   change created? Update the matching rule (see *Maintaining the Claude-facing
   files*). Triggers:
   - New architecture decision or invariant made during implementation
   - Public API surface changed
   - New node types added
   - Package structure changed (a new module is listed in
     `docs/package_structure.md`)
   - New dependencies added

5. **README.md** — Does the user-facing documentation need updating?
   - New public API functions
   - New built-in models
   - Installation or dependency changes
   - New examples

6. **CHANGELOG.md** — Does this change warrant a changelog entry? Add it under
   the `[Unreleased]` section, in the appropriate Keep a Changelog category
   (`Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`).
   Entries are curated for users deciding whether to upgrade — write for a human,
   not one line per commit. Document, for example:
   - New or changed public API, built-in models, or model-file format
   - Behavioural changes, bug fixes, and deprecations/removals users would notice
   - Installation or dependency changes
   Do **not** log changes with no user-visible effect (internal refactors, test-
   only or CI changes, docs, tracking issues) unless they are otherwise notable.

If the answer to any of the above is yes, make those updates as part of the
same task before marking it complete.

---


## Maintaining the Claude-facing files

Every line of a rule is read on every edit of the files it matches, so a rule
earns its place only if it changes what gets written.

- **A rule states an invariant and its reason**: what must stay true, what must
  be reused, what must not be done, and the one-sentence why. Write it as
  "do / do not / must", under a heading someone can find.
- **Not in a rule** (they go where they are read only when needed):
  measurements, timings and A/B results → `.claude/notes/` or
  `docs/plant_performance.md`; the history of a bug and issue numbers → the
  commit and the changelog; a description of what a function does → its
  docstring; an inventory of which tests cover a feature → nowhere (the tests
  are greppable, and the list rots).
- **Scope a rule to the files it governs.** A topic that concerns a few modules
  gets its own file with those paths (the `plant-*.md`, `core-*.md` and
  `integration-*.md` files), so editing one unit does not load the recycle
  solver's rules. Keep a rule file under about 250 lines; past that, split it
  or move narrative out.
- **Change a rule in the same commit as the code it describes**, and delete the
  rule text for anything removed. A rule that names a function that no longer
  exists is worse than no rule.
- **A new rule file gets a row in the Index above.**

The topic files were split out of `plant.md`, `integration.md` and `core.md`
verbatim, so most of their text still predates these guidelines: it is accurate
but reads as a narrative log. Condense a section to its invariants when you next
change the code it covers, moving any measurement worth keeping to
`.claude/notes/`.

---

## Key Literature

- Acero, J.L. & von Gunten, U. (2001). Characterization of oxidation processes:
  ozonation and the AOP O₃/H₂O₂. *J. AWWA*, 93(10), 90–100.
- von Gunten, U. & Hoigné, J. (1994). Bromate formation during ozonation of
  bromide-containing waters. *Environ. Sci. Technol.*, 28(7), 1234–1242.
- Kidger, P. (2021). On neural differential equations. *PhD Thesis, University
  of Oxford*. (Diffrax reference)
