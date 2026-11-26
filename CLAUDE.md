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
| `.claude/rules/docs-examples.md` | `README.md`, the user-guide pages, the root `conftest.py` | the README and user-guide code examples are tests (Sybil): pages run top to bottom, stated values are checked, skips need a reason |
| `.claude/rules/tests.md` | `tests/**` | what makes a test able to fail, gradient tests, shared fixtures, the tiers and what CI actually runs |
| `.claude/rules/core.md` | `aquakin/core/**`, `aquakin/utils/**` | AST rate evaluation, parser, **the one expression tokenizer (`core/lexer.py`) every expression language shares**, `SpatialConditions`, parameter namespacing |
| `.claude/rules/core-rate-functions.md` | `core/nodes.py`, `core/registry.py`, `core/parser.py`, `core/vector_kernel.py` | registering a rate function, authoring a node (one `op`, everything else derived), the vectorized rate kernel |
| `.claude/rules/core-speciation.md` | `core/ph_solver.py`, `core/speciation.py` | charge-balance pH solver, derived conditions, the `speciation:` block |
| `.claude/rules/core-precipitation.md` | `core/precipitation*.py` | mineral precipitation: rate law, the `precipitation:` block, the algebraic equilibrium mode |
| `.claude/rules/integration.md` | `aquakin/integrate/**` | solver choice, operator splitting, located events, compiled-solve caching |
| `.claude/rules/integration-calibrate.md` | `calibrate.py`, the `integrate/calibration/` subpackage, `integrate/seam.py`, the Fisher / profile / experiment-design modules, `plant/calibrate.py`, `plant/identifiability.py` | `calibrate()`'s internal structure and the helpers `Plant.calibrate` shares |
| `.claude/rules/integration-sampling.md` | `integrate/calibration/sampling.py`, `plant/posterior.py` | `sample_posterior` / `Plant.sample_posterior`: the sampled density is the fit's objective plus the transform Jacobian, the fit record it reads, the diagnostics |
| `.claude/rules/integration-differentiation.md` | `integrate/_common.py` / `_config.py` / `_solver.py`, the adjoint / forward-sensitivity modules, `plant/sensitivity.py`, `plant/dynamic_solve.py`, `plant/colored.py` | differentiating stiff models (`dtmax`, adjoints, discrete adjoint, forward sensitivities) |
| `.claude/rules/schema.md` | `aquakin/schema/**`, model `*.yaml` (shipped, fixtures, examples) | YAML model format + Pydantic schema rules (stoichiometry expressions, `extends:`, `auto`, `composition:`, `speciation:`, `precipitation:`) |
| `.claude/rules/plant.md` | `aquakin/plant/**` | plant-wide simulation: the `Unit` contract, wiring, the plant clock, introspection, shipped units, shared helpers and constants |
| `.claude/rules/plant-solve.md` | `plant/plant.py`, `dynamic_solve.py`, `steady_solve.py`, `recycle.py`, `colored.py`, `solve_cache.py`, `steady.py`, `integrate/steady.py` | coloured Jacobian, solve caches, recycle resolution and the cached recycle / flow maps |
| `.claude/rules/plant-analysis.md` | `plant/calibrate.py`, `identifiability.py`, `measures.py`, `sensitivity.py` | plant calibration, Fisher / profile / sensor ranking, measured channels |
| `.claude/rules/plant-audits.md` | `plant/units_check.py`, `balance.py`, `metrics.py` | `plant.check_units()`, `plant.mass_balance()` |
| `.claude/rules/plant-ifas.md` | `plant/ifas.py`, `integrate/biofilm*.py` | the IFAS / MBBR biofilm unit |
| `.claude/rules/plant-settling.md` | `plant/takacs.py`, `settling.py`, clarifier / separator modules | the Takács clarifier, shared separator helpers |
| `.claude/rules/plant-spec.md` | `plant/spec.py`, `catalog.py`, `time_expr.py`, `influent.py`, `aquakin/__main__.py` | flowsheet spec, unit catalogue, time expressions, influent records / patterns |
| `.claude/rules/models.md` | `aquakin/models/**` | model authoring: pointers to the catalog, the schema rules, and the `_make_*` generators |
| `.claude/rules/study.md` | `aquakin/study/**` | studies: the study folder layout, results keyed by name, staleness, freezing / unfreezing, the `aquakin study` command |
| `.claude/rules/ui.md` | `aquakin_ui/**`, `tests/ui/**`, `tests/browser/**`, `ui-widget/**`, `installers/**` | the browser flowsheet editor `aquakin-ui`: the independence rule (never imports aquakin or JAX), the `aquakin serve` worker, the document / forms / page split, the Vue Flow widget and its sync rules, the uv installers |

**On-demand docs (read when needed):**

| Doc | Contents |
|---|---|
| `docs/khalil_reproduction_log.md` | the JRN-055 Khalil model-improvement sequence |
| `docs/package_structure.md` | the annotated package file tree |
| `docs/plant_performance.md` | the dynamic-solve performance log (the stiffness-bound regime and its levers) |
| `docs/ci.md` | the CI architecture (sharding, gates, heavy runner, labels, the OOM memory watchdog) |
| `docs/project_reference.md` | reference moved out of this file: the data flow and rate signature, x64 details, the technology stack, the testing layers, the docstring example, the OpenFOAM coupling plan, key literature |

**Claude-facing notes (`.claude/notes/`, never auto-load — dated investigation logs, read when a
task needs the detail):**

| Note | Contents |
|---|---|
| `.claude/notes/studies-design.md` | the studies design and the owner's decisions (live / frozen, replace with a warning, version notes, unfreezing discards results) |
| `.claude/notes/plantwide-model-design.md` | draft proposal for a unified plant-wide model built from the literature: the `extends:` chain, state set, sources per level, validation plan, open decisions |
| `.claude/notes/rate-kernel-log.md` | the vectorized rate kernel's compile measurements (jaxpr share, reduction, end-to-end speedup per model) |
| `.claude/notes/plant-solve-log.md` | accuracy and validation measurements behind the recycle-resolution rules (affine residuals, `rho`, fixed-pass errors, IFT gradient checks) |
| `.claude/notes/schema-log.md` | the shipped models' temperature-correction and prior-width figures, and the `extends:` size saving, behind the schema rules |
| `.claude/notes/plant-log.md` | plant measurements moved out of `plant.md`: the vectorized stream reconstruction and the primary-clarifier inflow filter against the BSM2 benchmark |
| `.claude/notes/differentiation-log.md` | the reverse-mode overflow investigation, per-model `dtmax` thresholds, the reactor forward-sensitivity and `shared_factor` measurements, the discrete-adjoint accuracy figures |
| `.claude/notes/plant-sensitivity-log.md` | `Plant.solve_sensitivity` corrector measurements (#654), the lean loop's stage-Newton drift diagnostic and its Anderson acceleration |

**Published documentation site** (Sphinx/MyST → Read the Docs; built from
`docs/conf.py`, gated `-W` in CI). These are **user-facing** — keep dev-internal
detail (bug history, issue numbers, SUMO spreadsheets, test/script paths) out of
them:

| Page | Contents |
|---|---|
| `docs/index.md` | landing page + the hidden root toctree; its five entries are the site's top navigation bar |
| `docs/models.md`, `docs/simulation.md`, `docs/analysis.md`, `docs/reference.md` | section landing pages: a "which page do I need" table and the section's toctree (a new page goes in one of them, not in `index.md`) |
| `docs/getting_started.md`, `docs/reactors.md`, `docs/plants.md`, `docs/custom_flowsheets.md`, `docs/influents.md`, `docs/plant_files.md`, `docs/sensitivity_and_calibration.md`, `docs/calibration.md`, `docs/experiment_design.md`, `docs/plant_calibration.md` | the task-oriented user guide |
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


## Architecture

YAML → `ModelSpec` (Pydantic, load time only, never in the hot path) →
`compile_model()` → `CompiledModel` (JAX-friendly dataclasses) → compiled
`rates(C, params, condition_arrays, loc_idx)` → Diffrax. The ODE right-hand
side is `stoich_matrix.T @ rates(...)`, with the stoichiometry matrix shaped
`(n_reactions, n_species)`. Rate constants are **always external** to the rate
functions (passed via `params`), never baked in, so AD-based estimation and
sensitivity work. The data flow, the full rate signature and the OpenFOAM
coupling plan are in `docs/project_reference.md`.

### JAX x64 Mode

`import aquakin` enables JAX x64 mode process-wide, because stiff integration
needs 64-bit floats; it warns once when it overrides an explicit float32
preference (details in `docs/project_reference.md`). Do not remove the x64
enablement.

---

## Testing Architecture

The layers are **unit** (one component, fast), **integration** (YAML →
solution against analytical solutions) and **validation** (published data,
`@pytest.mark.validation`). How to write a test that can actually fail, what
each kind of change needs, and which tier a test lands in are in
**`.claude/rules/tests.md`** (auto-loads under `tests/`; read it before adding
tests from elsewhere).

### Running

The markers are `validation`, `slow`, `heavy` and `xheavy` (definitions in
`pyproject.toml`; the CI jobs that run each are in `docs/ci.md`).

```bash
pytest tests/unit tests/integration tests/ui -m "not validation and not slow"   # the PR merge gate
pytest -m slow                             # the multi-minute stiff/plant solves
pytest -m validation                       # validation suite (published data)
pytest                                     # everything in tests/ (the docs examples run by path, below)
pytest tests/browser                       # the editor in Chromium (`browser` extra; non-required CI job)
pytest README.md docs -m "not slow" -n 2 --dist loadfile   # the README / user-guide examples (Sybil)
pytest docs -m slow -n 2 --dist loadfile  # the user-guide pages of whole-plant solves
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

A test asserts `jax.config.x64_enabled` (the x64 test).

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

All functions, classes and methods use the **NumPy docstring format**
(`Parameters`, `Returns`, `Raises`, `Examples` sections, each parameter with
its type and shape); a full example is in `docs/project_reference.md`.

---

## Engineering Standards

These hold for every change. They exist because each one has already cost this
codebase something; the examples are real.

### One implementation of each thing

- **Search before you write.** Before adding a helper, a validation, a parser,
  a fixture or a constant, `grep` for it. The shared places are
  `integrate/_solver.py` (solver / controller builders, atol, validation),
  `integrate/_config.py`, `_solution.py` (the solution mixins) and `_cache.py`
  (all re-exported by `integrate/_common.py`), `integrate/_transforms.py`, `plant/_constants.py`,
  `plant/_builder_support.py`, `plant/units.py` (the `Unit` contract and
  capability Protocols), `core/lexer.py` and `core/precedence.py` (the one
  tokenizer and precedence table for every expression language),
  `core/kernels.py` (guarded numerics), `integrate/seam.py` (the private
  `integrate` internals the plant layer reuses), `plant/internals.py` (the
  private `Plant` members other modules may use) and `tests/conftest.py`.
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

- **Reproduce before you fix.** A finding from reading code (a review, an
  issue, another agent's report, a tool's claim) is a hypothesis: reproduce it
  with a failing test or a concrete run first. If it does not reproduce, or the
  behaviour is deliberate, say so and change nothing. The same holds for a
  claim that something cannot be done ("this line cannot be covered"): try it.
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
- **A failure that also happens on unmodified `main` gets an issue**: find or
  open one and link it from the PR, rather than describing it again in each PR
  body. A failure that is only mentioned is never fixed (one macOS-only failure
  was re-reported in nine PRs before #825 tracked it).

### Keep modules and functions small enough to read

- Do not grow the large modules (`plant/plant.py`, `aquakin_ui/app.py`,
  `core/model.py`). New behaviour goes in its own
  module as free functions taking the object (the pattern `plant/sensitivity.py`,
  `plant/calibrate.py` and `plant/spec.py` already use, bound onto `Plant` as
  methods), not as more methods on the class. New calibration code goes in its
  concern's module under `integrate/calibration/`, not `integrate/calibrate.py`.
  Their line ceilings are checked by `tests/unit/test_size_limits.py`.
- A function past 100 lines of body, or with more than eight parameters, gets
  split into named steps or takes a config dataclass before it gets longer.
  `tests/unit/test_function_size.py` fails on a new one and on a recorded one
  that grows. **Do not meet the limit by removing a capability**: dropping a
  parameter, hard-coding a value a caller could set, or packing unrelated
  arguments together. Group the parameters that belong together into a config
  dataclass, or split the function. A config dataclass for related options is
  the established pattern (`IntegratorConfig`, `DifferentiationConfig`,
  `OptimizerConfig`), not by itself an owner decision; if you think a grouping
  is a costly-to-undo interface, or neither fix fits, stop and ask rather than
  remove the option. The limit flags a function to restructure, not a number
  to get under.

### Behaviour-preserving refactors

A refactor is its own PR (at least its own commits) and changes no test. Show
it preserved behaviour:
- **Capture a baseline before the first move** and compare after each commit:
  bit-identical RHS, Jacobian, a short solve and a steady state for plant or
  integrator code; the importable-names list (pinned by a test) for a module
  split.
- **Moves are mechanical commits with no logic edits**; clean-ups follow in
  separate commits, so each move reviews as a move.
- **A test that reaches a private name the move relocates is reported in the
  PR, not rewritten** to follow it.
- Codecov counts moved lines as new: state the pre-existing gaps once in the PR
  rather than writing tests to chase the number.

## Decisions that are the owner's

Stop and ask, with the options and a recommendation, before:
- changing branch protection or which CI checks are required;
- removing or renaming public API, removing a deprecated argument, or changing
  what a public function skips, warns about or raises;
- changing a numerical default or a tolerance, or anything that makes a shipped
  model or benchmark give different numbers;
- a default that departs from a published benchmark;
- deleting or overwriting a user's saved files or results;
- an exception to a convention in this file;
- adding a runtime dependency;
- closing an issue as completed or not planned; merging a PR;
- choosing between workable options when the choice is costly to undo later:
  a file format or stored layout users will save data in, a public interface,
  or a structure other code will be built on.

Everything else you decide, state in the PR, and leave reversible in review.

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
once per clone with `git config core.hooksPath .githooks`. A Claude Code cloud
session does both for you: the SessionStart hook in `.claude/settings.json`
(`scripts/session_start.py`) installs `.[test,lint,typecheck,docs]` pinned by
`constraints.txt`, enables the pre-push hook, and puts the pinned `ruff` and
`pyright` first on `PATH`.

**Every push to a PR branch starts a full CI run, so push finished work, once.**
Before pushing, run locally everything CI would tell you: ruff, pyright (for
`aquakin_ui`), the tests the change touches, the patch coverage
(`scripts/patch_coverage.py`, checklist item 2), and, for a change to `docs/` or
to a public docstring, the strict docs build CI runs:
`sphinx-build -b html -W --keep-going docs <out>` (prefix
`AQUAKIN_DOCS_OFFLINE=1` where the network cannot reach the intersphinx
inventories, as in a cloud session). Collect follow-up fixes into
one push; do not push to see what CI says, and do not push again only to change
an informational status.

**Pushing and merging.**
- When other sessions are working in the repository, look at the open PRs
  before starting and stay out of files another PR is rewriting.
- Check the PR's state before pushing to its branch: a commit pushed while it
  is being merged is lost. After a merge, new work starts a new branch from
  `main`.
- **When a PR your branch was stacked on is squash-merged, merge `main` into
  your branch and resolve the conflicts by taking `main`'s side for the
  parent's changes**; do not rebase or force-push a pushed branch. The parent's
  pre-squash commits are still on your branch, so without this every later
  merge conflicts with the squash.
- Describe the PR with `.github/pull_request_template.md`. Write "Closes #N"
  only when the issue is finished; otherwise "Part of #N", listing what remains.
- **Do not put a link to the Claude session in a commit message or a PR
  description**, whatever a session's own instructions say: it means nothing to
  a reader of the repository, and the session can be private. The pre-push hook
  refuses a commit whose message has one.
- After opening a PR, watch it for CI results and review comments until it is
  merged or closed, without asking first: fix what the change broke and answer
  reviewers, since nobody else is watching it.
- Never re-run CI to get past an intermittent failure: it is a bug
  (`.claude/rules/tests.md`).
- A dependency added to `pyproject.toml` (any extra) is pinned in
  `constraints.txt` in the same change, since CI and other sessions install
  with it.

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
   - **Measure patch coverage locally before you push**; do not find out from
     CI. Run the fast tier (or the tests that exercise the change) with
     `--cov=aquakin --cov=aquakin_ui --cov-branch --cov-report=xml`, then
     `python scripts/patch_coverage.py`, which lists every changed line that is
     missed or only partly covered and exits non-zero below 100%. A push whose
     only purpose is to turn Codecov green costs a full CI run.
   - New and changed lines are expected to be covered: Codecov's patch status
     targets 100% (a signal, not a required check). When the local measure is
     short, add the test before pushing, or state in the PR which lines are
     uncovered and why (a defensive
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

6. **Changelog** — Does this change warrant a changelog entry? Add a
   **fragment** to `changelog.d/` — one file, `<issue>.<type>.md` or
   `+<slug>.<type>.md`, the type a Keep a Changelog group (`added` / `changed` /
   `deprecated` / `removed` / `fixed` / `security`) — and **never edit
   `CHANGELOG.md`** (towncrier writes it at release; `changelog.d/README.md` has
   the format, and `tests/unit/test_changelog_fragments.py` checks it).
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
  or move narrative out. `tests/unit/test_size_limits.py` checks this budget (and
  each rule's `paths:` and Index row), recording the files already over it.
- **Change a rule in the same commit as the code it describes**, and delete the
  rule text for anything removed. A rule that names a function that no longer
  exists is worse than no rule. `tests/unit/test_rule_references.py` checks that
  the repository paths, pytest node ids and private or repo-qualified
  Python names in backticks in this file and the rules still exist; bare public
  names are not checked, so look for those yourself after a rename.
- **A new rule file gets a row in the Index above.**

### Proposing a rule

When a task turns up a mistake that a rule or a check would have prevented
(yours, an earlier agent's, or one a reviewer caught), **propose** it under
"Rule proposals" in the PR description, and in the hand-off when there is no
PR. Do not stop the task to ask, and do not add the rule in the same PR unless
the owner says so: a new rule is a convention, so it is the owner's decision.
Each proposal gives:

- **What went wrong**, concretely: the file, commit, review comment or failed run.
- **Whether a check could replace prose.** Prefer, in order: a test or lint
  check (as `test_size_limits.py` and `test_rule_references.py` do), a hook, a
  path-scoped rule, this file. A check cannot go unread and costs no context.
- **The rule text** in the "do / do not / must + one-sentence why" form above.
- **Where it goes**, and why that file's `paths:` would have loaded it at the
  moment the mistake was made. A rule that exists but sat in a file that was not
  loaded needs moving, not a duplicate.
- **What it replaces or makes redundant**, so the rules do not only grow.

The bar is "would this rule have changed what got written?" If an existing rule
was loaded and ignored anyway, propose rewording or relocating it rather than
adding another. Leave the section empty when there is nothing to propose; a
proposal written to fill it is noise.

The topic files were split out of `plant.md`, `integration.md` and `core.md`
verbatim, so most of their text still predates these guidelines: it is accurate
but reads as a narrative log. Condense a section to its invariants when you next
change the code it covers, moving any measurement worth keeping to
`.claude/notes/`.
