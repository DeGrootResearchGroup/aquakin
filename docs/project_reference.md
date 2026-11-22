# Project reference

Reference material moved out of the root `CLAUDE.md` so it does not load in
every session: the runtime data flow and rate-function contract, the x64
details, the technology stack, the testing layers, the full docstring example,
the OpenFOAM coupling plan and the key literature. The rules
that govern this material stay in `CLAUDE.md`; read this page when a task needs
the detail.

## Architecture

### Two-layer data model

**Layer 1 — Pydantic schema (load time)**
Parses and validates YAML model files. Produces clean Python objects with
clear error messages for malformed input. Pydantic is used only during loading
— it never appears in the runtime hot path.

**Layer 2 — Compiled runtime (CompiledModel dataclass)**
JAX-friendly dataclasses built from the validated schema via a `compile_model()`
step. This is what the integrators and rate functions operate on. No Pydantic
dependency at runtime.

### Core data flow

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

### Rate function signature

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

### Stoichiometry matrix

Shape `(n_reactions, n_species)`. Entry `[i, j]` is the stoichiometric
coefficient of species j in reaction i. Built once at compile time and stored
in `CompiledModel`.

### JAX x64 mode

Stiff ODE integration requires 64-bit floats. `aquakin` enables x64 mode
automatically at import time. This is **global, process-wide** JAX state, so it
is a documented side effect of `import aquakin` (README install section). To keep
it from being *silent*, `aquakin/__init__.py` warns (once) when it overrides what
looks like an explicit float32 preference — JAX already imported when aquakin is
imported, or `JAX_ENABLE_X64` set to a false value — but stays silent on a plain
fresh import (the common case). It always enables x64 regardless; the warning is
only a signal. See `tests/unit/test_x64_import.py` (subprocess-per-scenario, since
the effect is process-global). Do not remove the x64 enablement.

## Technology stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python | Flexibility, ecosystem |
| Numerical backend | JAX | AD for free, jit compilation |
| ODE integration | Diffrax | JAX-native stiff solvers, adjoint support |
| Schema validation | Pydantic | Clear load-time errors for malformed YAML |
| Runtime data model | Python dataclasses | JAX-friendly, no Pydantic overhead at runtime |
| Model format | YAML | Human-readable, supports inline comments for literature citations |
| Rate expression evaluation | Custom AST + recursive descent parser | Safe, differentiable, introspectable |

## Testing layers

**Unit** — test individual components in isolation, fast, run on every change.
**Integration** — test full YAML → solution pipeline against analytical solutions.
**Validation** — test scientific correctness against published experimental data,
marked `@pytest.mark.validation`, run separately.

The x64 test:

```python
def test_64bit_precision_enabled():
    import jax
    assert jax.config.x64_enabled
```

## Docstring example

Every function, class and method uses the NumPy docstring format (`CLAUDE.md`,
*Docstring Convention*). A complete example:

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

## OpenFOAM coupling (future)

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

## Key literature

- Acero, J.L. & von Gunten, U. (2001). Characterization of oxidation processes:
  ozonation and the AOP O₃/H₂O₂. *J. AWWA*, 93(10), 90–100.
- von Gunten, U. & Hoigné, J. (1994). Bromate formation during ozonation of
  bromide-containing waters. *Environ. Sci. Technol.*, 28(7), 1234–1242.
- Kidger, P. (2021). On neural differential equations. *PhD Thesis, University
  of Oxford*. (Diffrax reference)
