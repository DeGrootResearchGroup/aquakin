# CLAUDE.md — aquakin Project Briefing

This is the slim root briefing for the `aquakin` library: orientation,
conventions, testing, the development workflow, and the **Post-Change
Checklist**. Read it before writing any code.

Detailed subsystem guidance is **split out of this file** so it loads only when
relevant rather than every session:

- **Path-scoped rules** under `.claude/rules/*.md` carry `paths:` frontmatter and
  auto-load when you read or edit files matching their glob.
- **On-demand docs** under `docs/*.md` are reference / investigation logs; read
  them when a task needs the detail.

When you change a subsystem, the matching rule below is the authoritative
guidance for it — keep it updated as part of the change (this file's Post-Change
Checklist still governs).

## Index — where the detail lives

**Path-scoped rules (auto-load on matching files):**

| Rule | Loads for | Covers |
|---|---|---|
| `.claude/rules/core.md` | `aquakin/core/**` | AST rate evaluation, parser, vectorized kernel, `SpatialConditions`, parameter namespacing, charge-balance pH / `speciation:`, mineral `precipitation:` |
| `.claude/rules/integration.md` | `aquakin/integrate/**` | solver choice, differentiating stiff models (`dtmax`, adjoints, discrete adjoint), operator splitting, located events, compiled-solve caching |
| `.claude/rules/schema.md` | `aquakin/schema/**`, `**/*.yaml` | YAML model format + Pydantic schema rules (stoichiometry expressions, `extends:`, `auto`, `composition:`, `speciation:`, `precipitation:`) |
| `.claude/rules/plant.md` | `aquakin/plant/**` | plant-wide simulation: units, recycle resolution, BSM1/BSM2/A²O builders, control, dosing, design, evaluation |
| `.claude/rules/models.md` | `aquakin/models/**` | model authoring: pointers to the catalog, the schema rules, and the `_make_*` generators |

**On-demand docs (read when needed):**

| Doc | Contents |
|---|---|
| `docs/model_catalog.md` | the full per-model catalog (every shipped model, provenance, documented corrections) |
| `docs/khalil_reproduction_log.md` | the JRN-055 Khalil model-improvement sequence |
| `docs/public_api.md` | the full public API reference with worked usage |
| `docs/package_structure.md` | the annotated package file tree |
| `docs/plant_performance.md` | the dynamic-solve performance log (the stiffness-bound regime and its levers) |
| `docs/ci.md` | the CI architecture (sharding, gates, heavy runner, labels) |
| `docs/model_format.md`, `docs/adding_models.md`, `docs/index.md` | existing reference docs |

---

## Project Overview

`aquakin` is an open source Python library for modelling reactive scalar
transport in aqueous environmental systems. It provides a modular, runtime-
configurable kinetics engine that can be coupled to any flow solver.

Both **chemistry** (ozonation, advanced oxidation, chlorine decay, ...) and
**biology** (activated sludge models, anaerobic digestion, ...) are in scope.
> The full per-model catalog (what each shipped model is, its provenance,
> and the documented corrections) is in **`docs/model_catalog.md`**; the
> Khalil model-improvement log is in **`docs/khalil_reproduction_log.md`**.

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
JAX-friendly dataclasses built from the validated schema via a `compile()`
step. This is what the integrators and rate functions operate on. No Pydantic
dependency at runtime.

### Core Data Flow

```
YAML file
   ↓  loader.py (Pydantic validation)
ModelSpec
   ↓  compile()
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

### pytest Configuration

```toml
[tool.pytest.ini_options]
markers = [
    "validation: scientific validation tests against published data (slow)",
    "slow: multi-minute stiff/plant integration tests; excluded from the fast PR gate, run in full on merge to main",
]
testpaths = ["tests"]
```

```bash
pytest -m "not validation and not slow"   # fast gate (the PR merge gate)
pytest -m slow                             # the multi-minute stiff/plant solves
pytest -m validation                       # validation suite (published data)
pytest                                     # everything
```

### Canonical Integration Test

First-order decay `A → B` with rate `k * [A]` has the analytical solution
`[A](t) = [A]₀ * exp(-k*t)`. This is the primary integration test and must
always pass. It lives in `tests/fixtures/simple_model.yaml` and
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
reader cannot see. Genuine scientific provenance belongs in a model YAML's
`references:` block as a proper literature citation, not scattered through code
comments.


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

Before committing or pushing any code change, run the linter locally so the
required `lint (ruff)` gate stays green — `ruff check aquakin` and `ruff format
aquakin` (see Post-Change Checklist item 1 for details).

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
   change. The CI `lint (ruff)` job runs `ruff check aquakin` **and** `ruff
   format --check aquakin` on every PR and is a **required merge gate**, so an
   unformatted file (or a lint error) fails CI — a common way a refactor that
   only shortens lines trips the format gate. Run both, from the repo root:
   - `ruff check aquakin` — must report no errors.
   - `ruff format aquakin` — auto-applies formatting in place (CI runs it with
     `--check`, which only verifies; running it without `--check` fixes the
     files). Re-run `ruff format --check aquakin` to confirm it is clean.

   Ruff is pinned via the `lint` extra (`pip install -e ".[lint]"`); see the
   `[tool.ruff]` config in `pyproject.toml` for the rule set and per-file
   ignores. Not needed for docs/config-only changes that touch no `.py` files.

2. **Tests** — Are new tests needed for the changed or added functionality?
   - New node type → unit test in `test_nodes.py`
   - New public API function → integration test
   - New built-in model → validation test against literature data
   - Bug fix → regression test

3. **CLAUDE.md** — Does this file need updating?
   - New architecture decision made during implementation
   - Public API surface changed
   - New node types added
   - Package structure changed
   - New dependencies added

4. **README.md** — Does the user-facing documentation need updating?
   - New public API functions
   - New built-in models
   - Installation or dependency changes
   - New examples

5. **CHANGELOG.md** — Does this change warrant a changelog entry? Add it under
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


## Key Literature

- Acero, J.L. & von Gunten, U. (2001). Characterization of oxidation processes:
  ozonation and the AOP O₃/H₂O₂. *J. AWWA*, 93(10), 90–100.
- von Gunten, U. & Hoigné, J. (1994). Bromate formation during ozonation of
  bromide-containing waters. *Environ. Sci. Technol.*, 28(7), 1234–1242.
- Kidger, P. (2021). On neural differential equations. *PhD Thesis, University
  of Oxford*. (Diffrax reference)
